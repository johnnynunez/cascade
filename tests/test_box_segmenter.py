"""Box-prompted SAM: one full-frame mask per brain box.

The brain returns labelled boxes; `BoxSegmenter.masks` turns them into the
same-frame bool masks that the observed-finger gate, belief identity and grasp
clouds need. These tests inject a fake ultralytics model (no weights, no torch,
no GPU) and pin the contract the detector relies on: one model call per frame
for every box, one entry per box in prompt order, `None` where SAM has nothing.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.perception import segmenter as segmenter_mod
from cascade.perception.segmenter import BoxSegmenter, PointSegmenter
from cascade.types import Frame, SkillError

H, W = 96, 128


def _frame() -> Frame:
    return Frame(rgb=np.zeros((H, W, 3), dtype=np.uint8), depth_m=None, K=np.eye(3))


def _box_mask(box, shape=(H, W), scale=1) -> np.ndarray:
    """The region a box covers, at 1/scale of the frame resolution."""
    x1, y1, x2, y2 = (int(v) // scale for v in box)
    m = np.zeros((shape[0] // scale, shape[1] // scale), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


class _Tensor:
    """The slice of a torch tensor the segmenters read, over numpy."""

    def __init__(self, array, device="cpu"):
        self._a = np.asarray(array)
        self.device = SimpleNamespace(type=device)

    def __len__(self):
        return len(self._a)

    def __getitem__(self, i):
        return _Tensor(self._a[i], self.device.type)

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _FakeSam:
    """Stands in for `ultralytics.SAM`: records every call, returns one mask
    per prompt covering that prompt's box (or point neighbourhood).

    `drop` removes prompts the way ultralytics' confidence filter does (the
    survivors keep their prompt index in `boxes.cls`); `empty` returns an
    all-False mask; `scale` returns masks at a lower working resolution.
    """

    def __init__(self, *, drop=(), empty=(), scale=1, device="cpu", nothing=False):
        self.calls = []
        self.drop, self.empty, self.scale = set(drop), set(empty), scale
        self.device, self.nothing = device, nothing

    def __call__(self, source, **kw):
        self.calls.append(kw)
        if self.nothing:
            return [SimpleNamespace(masks=None, boxes=None)]
        if "bboxes" in kw:
            prompts = kw["bboxes"]
        else:
            prompts = [[u - 6, v - 6, u + 6, v + 6] for u, v in kw["points"]]
        kept = [i for i in range(len(prompts)) if i not in self.drop]
        masks = [np.zeros((H // self.scale, W // self.scale), dtype=bool)
                 if i in self.empty else _box_mask(prompts[i], scale=self.scale)
                 for i in kept]
        return [SimpleNamespace(
            masks=SimpleNamespace(data=_Tensor(np.stack(masks), self.device)),
            boxes=SimpleNamespace(cls=np.asarray(kept, dtype=np.float32)),
        )]


@pytest.fixture(autouse=True)
def _no_torch_probe(monkeypatch):
    """Resolve devices without importing torch; record what was asked."""
    asked = []

    def resolve(spec, *, what="model"):
        asked.append((spec, what))
        return "cpu"

    monkeypatch.setattr("cascade.device.resolve_device", resolve)
    monkeypatch.delenv("CASCADE_REQUIRE_CUDA", raising=False)
    return asked


@pytest.fixture(autouse=True)
def _fresh_model_cache(monkeypatch):
    """Each test starts with no shared SAM loaded in this process."""
    monkeypatch.setattr(segmenter_mod, "_SHARED", {})


class _SlowSam(_FakeSam):
    """A `_FakeSam` that stays inside each call for a while and records the
    most callers ever inside at once."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self._guard = threading.Lock()
        self.inside = self.max_inside = 0

    def __call__(self, source, **kw):
        with self._guard:
            self.inside += 1
            self.max_inside = max(self.max_inside, self.inside)
        try:
            time.sleep(0.05)
            return super().__call__(source, **kw)
        finally:
            with self._guard:
                self.inside -= 1


@pytest.fixture
def ultralytics(monkeypatch):
    """Fake `ultralytics` module; `loaded` lists every `SAM(path)` built."""
    mod = types.ModuleType("ultralytics")
    mod.loaded = []

    def SAM(path):
        time.sleep(0.02)  # a slow load widens any first-use race
        model = _SlowSam()
        mod.loaded.append((path, model))
        return model

    mod.SAM = SAM
    monkeypatch.setitem(sys.modules, "ultralytics", mod)
    return mod


def _together(*fns):
    """Run each fn in its own thread, all released at once; re-raise errors."""
    barrier = threading.Barrier(len(fns))
    errors = []

    def run(fn):
        barrier.wait()
        try:
            fn()
        except BaseException as e:  # noqa: BLE001 - surfaced below
            errors.append(e)

    threads = [threading.Thread(target=run, args=(fn,)) for fn in fns]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not any(t.is_alive() for t in threads), "a segmenter call deadlocked"
    if errors:
        raise errors[0]


def _segmenter(fake, **kw) -> BoxSegmenter:
    seg = BoxSegmenter(**kw)
    seg._model = fake
    return seg


BOXES = np.array([[10.0, 20.0, 40.0, 60.0],
                  [64.0, 8.0, 120.0, 50.0]])


# ── the call ──────────────────────────────────────────────────────────────

def test_defaults_to_the_large_checkpoint_and_auto_device():
    seg = BoxSegmenter()
    assert seg._model_path == "sam2.1_l.pt"
    assert seg._device_spec == "auto"


def test_one_model_call_per_frame_prompts_every_box(_no_torch_probe):
    fake = _FakeSam()
    out = _segmenter(fake).masks(_frame(), BOXES)

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["bboxes"] == BOXES.tolist()
    assert call["verbose"] is False
    assert call["device"] == "cpu"
    assert _no_torch_probe == [("auto", "segmenter")]
    assert len(out) == 2
    for mask, box in zip(out, BOXES):
        assert mask.shape == (H, W) and mask.dtype == bool
        assert np.array_equal(mask, _box_mask(box))


def test_no_boxes_returns_empty_without_calling_the_model():
    fake = _FakeSam()
    assert _segmenter(fake).masks(_frame(), np.zeros((0, 4))) == []
    assert fake.calls == []


def test_no_boxes_never_loads_the_model(ultralytics):
    seg = BoxSegmenter()
    assert seg.masks(_frame(), np.zeros((0, 4))) == []
    assert ultralytics.loaded == []
    assert seg._model is None


# ── one entry per box, in prompt order ────────────────────────────────────

def test_an_all_false_mask_maps_to_none():
    out = _segmenter(_FakeSam(empty={1})).masks(_frame(), BOXES)
    assert np.array_equal(out[0], _box_mask(BOXES[0]))
    assert out[1] is None


def test_no_masks_at_all_maps_every_box_to_none():
    out = _segmenter(_FakeSam(nothing=True)).masks(_frame(), BOXES)
    assert out == [None, None]


def test_a_box_sam_dropped_is_none_and_the_rest_keep_their_box():
    """ultralytics drops low-score masks; the survivors must not shift onto
    the wrong box (their prompt index rides in `boxes.cls`)."""
    boxes = np.vstack([BOXES, [[50.0, 70.0, 90.0, 94.0]]])
    out = _segmenter(_FakeSam(drop={1})).masks(_frame(), boxes)

    assert len(out) == 3
    assert np.array_equal(out[0], _box_mask(boxes[0]))
    assert out[1] is None
    assert np.array_equal(out[2], _box_mask(boxes[2]))


def test_fewer_masks_without_prompt_indices_leave_the_tail_none():
    data = _Tensor(np.stack([_box_mask(BOXES[0])]))

    def fake(source, **kw):
        return [SimpleNamespace(masks=SimpleNamespace(data=data))]

    out = _segmenter(fake).masks(_frame(), BOXES)
    assert len(out) == 2
    assert np.array_equal(out[0], _box_mask(BOXES[0]))
    assert out[1] is None


def test_masks_at_working_resolution_are_resized_to_the_frame():
    out = _segmenter(_FakeSam(scale=2)).masks(_frame(), BOXES)
    for mask, box in zip(out, BOXES):
        assert mask.shape == (H, W) and mask.dtype == bool
        assert np.array_equal(mask, _box_mask(box))


# ── strict GPU mode (CASCADE_REQUIRE_CUDA=1) ──────────────────────────────

def test_strict_gpu_mode_refuses_cpu_tensors(monkeypatch):
    monkeypatch.setenv("CASCADE_REQUIRE_CUDA", "1")
    with pytest.raises(RuntimeError, match="CUDA tensors"):
        _segmenter(_FakeSam(device="cpu")).masks(_frame(), BOXES)


def test_strict_gpu_mode_keeps_masks_on_the_cuda_path(monkeypatch):
    from cascade.perception import cuda_math

    seen = []

    def segment_mask(mask, shape):
        seen.append(shape)
        return mask.numpy().astype(bool)

    monkeypatch.setattr(cuda_math, "segment_mask", segment_mask)
    monkeypatch.setenv("CASCADE_REQUIRE_CUDA", "1")
    out = _segmenter(_FakeSam(device="cuda", empty={1})).masks(_frame(), BOXES)

    assert seen == [(H, W), (H, W)]
    assert np.array_equal(out[0], _box_mask(BOXES[0]))
    assert out[1] is None


# ── PointSegmenter keeps its behaviour on the shared helpers ──────────────

def test_point_segmenter_returns_the_mask_under_the_pixel():
    fake = _FakeSam(scale=2)
    seg = PointSegmenter()
    seg._model = fake
    mask = seg.mask_at(_frame(), 30, 40)

    assert fake.calls[0]["points"] == [[30, 40]]
    assert fake.calls[0]["labels"] == [1]
    assert mask.shape == (H, W) and mask.dtype == bool
    assert np.array_equal(mask, _box_mask([24, 34, 36, 46]))


def test_point_segmenter_raises_when_sam_returns_nothing():
    seg = PointSegmenter()
    seg._model = _FakeSam(nothing=True)
    with pytest.raises(SkillError, match="no mask"):
        seg.mask_at(_frame(), 30, 40)


def test_point_segmenter_strict_gpu_mode_refuses_cpu_tensors(monkeypatch):
    monkeypatch.setenv("CASCADE_REQUIRE_CUDA", "1")
    seg = PointSegmenter()
    seg._model = _FakeSam(device="cpu")
    with pytest.raises(RuntimeError, match="CUDA tensors"):
        seg.mask_at(_frame(), 30, 40)


# ── one model and one lock per checkpoint ─────────────────────────────────

def test_point_and_box_segmenters_on_one_path_share_one_model(ultralytics):
    point, box = PointSegmenter("shared.pt"), BoxSegmenter("shared.pt")
    assert point._load() is box._load()
    assert [path for path, _ in ultralytics.loaded] == ["shared.pt"]

    assert BoxSegmenter("other.pt")._load() is not box._load()
    assert [path for path, _ in ultralytics.loaded] == ["shared.pt", "other.pt"]


def test_first_use_from_two_threads_loads_the_model_once(ultralytics):
    a, b = BoxSegmenter("shared.pt"), BoxSegmenter("shared.pt")
    _together(lambda: a.masks(_frame(), BOXES), lambda: b.masks(_frame(), BOXES))
    assert len(ultralytics.loaded) == 1


def test_two_threads_calling_masks_never_overlap_inside_the_model():
    """ultralytics predict is not re-entrant (prompts are predictor state),
    and the brain detector calls masks() from the watcher and the skill
    runtime at once."""
    fake = _SlowSam()
    seg = _segmenter(fake)

    def three_frames():
        for _ in range(3):
            seg.masks(_frame(), BOXES)

    _together(three_frames, three_frames)
    assert len(fake.calls) == 6
    assert fake.max_inside == 1


def test_point_and_box_calls_on_one_model_never_overlap(ultralytics):
    point, box = PointSegmenter("shared.pt"), BoxSegmenter("shared.pt")

    def clicks():
        for _ in range(3):
            point.mask_at(_frame(), 30, 40)

    def frames():
        for _ in range(3):
            box.masks(_frame(), BOXES)

    _together(clicks, frames)
    [(_, model)] = ultralytics.loaded
    assert len(model.calls) == 6
    assert model.max_inside == 1
