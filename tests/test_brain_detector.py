"""BrainDetector (perception/brain_detector.py): transport, refer, inventory cache.

Everything runs against fakes: a client object with `chat.completions.create`
returning canned replies (optionally blocking on an Event), a segmenter that
records its box prompts, and an injected clock. No network, no GPU, no openai.
"""

from __future__ import annotations

import base64
import json
import logging
import sys
import threading
import time
import types
from types import SimpleNamespace
from typing import ClassVar

import cv2
import numpy as np
import pytest

from cascade.perception.brain_detector import (
    INVENTORY_PROMPT,
    REFER_PROMPT,
    BrainDetector,
)

W, H = 320, 180  # thumbnail 64x36 is an exact 5x area reduction


class APITimeoutError(Exception):
    """Stand-in for openai.APITimeoutError (CI installs no openai)."""


def _reply(*items) -> str:
    return json.dumps([{"label": label, "bbox_2d": list(box)} for label, box in items])


def _px(box) -> list[float]:
    x1, y1, x2, y2 = box
    return [x1 * W / 1000, y1 * H / 1000, x2 * W / 1000, y2 * H / 1000]


CUP = ("paper cup", (100, 200, 300, 600))
BALL = ("red ball", (500, 100, 700, 400))
SPOOL = ("green filament spool", (600, 500, 900, 900))


class Clock:
    def __init__(self, now: float = 0.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakeClient:
    """chat.completions.create -> canned reply per prompt; inventory can block on `gate`."""

    def __init__(self):
        self.requests: list[dict] = []
        self.inventory: str | None = "[]"
        self.refer: dict[str, str] = {}
        self.error: BaseException | None = None
        self.refresh_error: BaseException | None = None
        self.gate: threading.Event | None = None
        # Every request on a background refresh thread (inventory or refer)
        # waits for one release: the test answers in-flight requests one by one.
        self.refresh_slot: threading.Semaphore | None = None
        self.finish_reason = "stop"
        self.entered = threading.Event()
        self.blocked = 0
        self.threads: list[str] = []  # name of the thread that sent each request
        self._lock = threading.Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        prompt = kwargs["messages"][0]["content"][1]["text"]
        with self._lock:
            self.requests.append(kwargs)
            self.threads.append(threading.current_thread().name)
        slot = self.refresh_slot
        if slot is not None and threading.current_thread().name.startswith("brain-refresh"):
            self.entered.set()
            slot.acquire(timeout=10)
        if self.error is not None:
            raise self.error
        if prompt == INVENTORY_PROMPT:
            gate = self.gate
            if gate is not None:
                with self._lock:
                    self.blocked += 1
                self.entered.set()
                gate.wait(10)
                with self._lock:
                    self.blocked -= 1
                if self.refresh_error is not None:
                    raise self.refresh_error
            reply = self.inventory
        else:
            reply = next((r for q, r in self.refer.items() if prompt == REFER_PROMPT.format(q=q)), "[]")
        msg = SimpleNamespace(content=reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=self.finish_reason)])

    def prompts(self) -> list[str]:
        with self._lock:
            return [r["messages"][0]["content"][1]["text"] for r in self.requests]

    def inventory_calls(self) -> int:
        return sum(p == INVENTORY_PROMPT for p in self.prompts())


class FakeSegmenter:
    """Records (frame, boxes) per call; one box-shaped mask per box, None for `drop`."""

    def __init__(self, drop=()):
        self.calls: list[tuple[object, np.ndarray]] = []
        self.returned: list[list] = []
        self.drop = set(drop)
        self.error: BaseException | None = None

    def masks(self, frame, boxes):
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        self.calls.append((frame, boxes.copy()))
        if self.error is not None:
            raise self.error
        out = []
        for i, (x0, y0, x1, y1) in enumerate(boxes.astype(int)):
            if i in self.drop:
                out.append(None)
                continue
            mask = np.zeros(frame.rgb.shape[:2], dtype=bool)
            mask[y0:y1, x0:x1] = True
            out.append(mask)
        self.returned.append(out)
        return out


def _frame(level: int = 0, fx: float = 500.0):
    from cascade.types import Frame

    rgb = np.full((H, W, 3), level, dtype=np.uint8)
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]])
    return Frame(rgb=rgb, depth_m=None, K=K)


def _brain_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith("brain-refresh") and t.is_alive()]


@pytest.fixture
def make():
    """Build detectors with fakes; teardown releases every gate and closes them."""
    built: list[tuple[BrainDetector, FakeClient]] = []

    def _make(*, segmenter="fake", client=None, clock=None, **kw):
        client = client or FakeClient()
        seg = FakeSegmenter() if segmenter == "fake" else segmenter
        det = BrainDetector(base_url="http://brain/v1", model="qwen3.8-27b", client=client,
                            segmenter=seg, clock=clock or Clock(), **kw)
        built.append((det, client))
        return det, client, seg

    yield _make
    for det, client in built:
        if client.gate is not None:
            client.gate.set()
        if client.refresh_slot is not None:
            client.refresh_slot.release(1000)
        det.close()
    assert _brain_threads() == [], "a refresh thread outlived close()"


def test_concurrent_safe_flag():
    # world.LockedDetector skips its global RLock for detectors with this flag.
    assert BrainDetector.concurrent_safe is True


# --- refer -----------------------------------------------------------------


def test_refer_sends_the_refer_prompt_with_the_jpeg(make):
    det, client, seg = make()
    client.refer["paper cup"] = _reply(("cup", CUP[1]))
    frame = _frame()
    frame.rgb[60:120, 100:200] = (0, 0, 255)  # red in BGR

    dets = det.detect(frame, ["paper cup"])

    (req,) = client.requests
    assert req["model"] == "qwen3.8-27b"
    assert req["temperature"] == 0
    assert req["max_tokens"] == 1500
    assert req["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    (msg,) = req["messages"]
    assert msg["role"] == "user"
    image, text = msg["content"]
    assert text == {"type": "text", "text": REFER_PROMPT.format(q="paper cup")}
    assert image["type"] == "image_url"
    url = image["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8),
                           cv2.IMREAD_COLOR)
    assert decoded.shape == (H, W, 3)
    b, g, r = (int(v) for v in decoded[90, 150])
    assert r > 200 and g < 60 and b < 60, "colours swapped: the frame is BGR already"

    (d,) = dets
    assert d.label == "paper cup"  # the agent's words, not the brain's label
    assert d.conf == 1.0
    assert d.bbox.dtype == np.float32
    np.testing.assert_allclose(d.bbox, _px(CUP[1]), rtol=1e-6)
    (seg_frame, boxes), = seg.calls
    assert seg_frame is frame
    np.testing.assert_allclose(boxes, [_px(CUP[1])], rtol=1e-6)
    assert d.mask is seg.returned[0][0]


def test_max_tokens_comes_from_the_constructor(make):
    det, client, _ = make(max_tokens=777)
    det.detect(_frame(), ["paper cup"])
    assert client.requests[0]["max_tokens"] == 777


def test_refer_several_classes_one_request_each(make):
    det, client, _ = make()
    client.refer["paper cup"] = _reply(CUP)
    client.refer["red ball"] = _reply(("ball", BALL[1]), ("ball", SPOOL[1]))
    dets = det.detect(_frame(), ["paper cup", "red ball"])
    assert client.prompts() == [REFER_PROMPT.format(q="paper cup"), REFER_PROMPT.format(q="red ball")]
    assert [d.label for d in dets] == ["paper cup", "red ball", "red ball"]


def test_refer_keeps_maskless_boxes(make):
    det, client, _ = make(segmenter=FakeSegmenter(drop={0}))
    client.refer["paper cup"] = _reply(CUP)
    (d,) = det.detect(_frame(), ["paper cup"])
    assert d.mask is None  # today's bbox fallback


def test_refer_bypasses_the_inventory_cache(make):
    det, client, _ = make()
    client.refer["paper cup"] = _reply(CUP)
    client.inventory = _reply(BALL)
    det.detect(_frame(), ["paper cup"])
    # A refer neither reads nor fills the cache: the first inventory is cold.
    assert [d.label for d in det.detect(_frame(), None)] == ["red ball"]
    assert client.inventory_calls() == 1
    det.detect(_frame(), ["paper cup"])
    assert client.prompts()[-1] == REFER_PROMPT.format(q="paper cup")
    det.detect(_frame(), None)
    assert client.inventory_calls() == 1


@pytest.mark.parametrize("open_request", [None, []])
def test_a_refer_never_narrows_the_next_open_world_call(make, open_request):
    # Moved from the removed open-vocabulary detector's query-scope test: a
    # targeted search must not hide unrelated objects from later open looks.
    det, client, _ = make()
    client.inventory = _reply(CUP, BALL, SPOOL)
    client.refer["red ball"] = _reply(BALL)
    client.refer["paper cup"] = _reply(CUP)
    everything = ["paper cup", "red ball", "green filament spool"]
    assert [d.label for d in det.detect(_frame(), None)] == everything
    assert [d.label for d in det.detect(_frame(), ["red ball"])] == ["red ball"]
    assert [d.label for d in det.detect(_frame(), open_request)] == everything
    assert [d.label for d in det.detect(_frame(), ["paper cup"])] == ["paper cup"]
    assert [d.label for d in det.detect(_frame(), open_request)] == everything


# --- inventory cache -------------------------------------------------------


def test_inventory_cold_calls_the_brain_once_synchronously(make):
    det, client, seg = make()
    client.inventory = _reply(CUP, BALL)
    frame = _frame()
    dets = det.detect(frame, None)
    assert client.prompts() == [INVENTORY_PROMPT]
    assert [d.label for d in dets] == ["paper cup", "red ball"]
    assert all(d.conf == 1.0 for d in dets)
    assert [d.mask is m for d, m in zip(dets, seg.returned[0])] == [True, True]
    assert _brain_threads() == []


def test_cache_hit_segments_the_cached_boxes_on_the_new_frame(make):
    clock = Clock()
    det, client, seg = make(clock=clock)
    client.inventory = _reply(CUP, BALL)
    det.detect(_frame(), None)
    clock.now = 3.9
    frame = _frame()
    dets = det.detect(frame, None)
    assert client.inventory_calls() == 1
    seg_frame, boxes = seg.calls[-1]
    assert seg_frame is frame
    np.testing.assert_allclose(boxes, [_px(CUP[1]), _px(BALL[1])], rtol=1e-6)
    assert [d.label for d in dets] == ["paper cup", "red ball"]
    assert _brain_threads() == []


def test_empty_classes_behave_as_none(make):
    det, client, _ = make()
    client.inventory = _reply(CUP)
    assert [d.label for d in det.detect(_frame(), [])] == ["paper cup"]
    assert client.prompts() == [INVENTORY_PROMPT]
    det.set_classes(["red ball"])  # holds no vocabulary: None stays open world
    assert [d.label for d in det.detect(_frame(), None)] == ["paper cup"]
    assert client.prompts() == [INVENTORY_PROMPT]


def test_scene_change_threshold(make):
    det, client, _ = make(refresh_s=100.0)
    client.inventory = _reply(CUP)
    det.detect(_frame(level=0), None)
    det.detect(_frame(level=6), None)  # mean abs diff 6.0: same scene
    assert client.inventory_calls() == 1
    client.inventory = _reply(BALL)
    dets = det.detect(_frame(level=7), None)  # 7.0 > 6.0: synchronous brain call
    assert client.inventory_calls() == 2
    assert [d.label for d in dets] == ["red ball"]
    assert _brain_threads() == []


def test_entry_older_than_max_age_is_refetched_synchronously(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    det.detect(_frame(), None)
    clock.now = 30.5
    client.inventory = _reply(BALL)
    assert [d.label for d in det.detect(_frame(), None)] == ["red ball"]
    assert client.inventory_calls() == 2
    assert _brain_threads() == []


def test_aged_entry_ten_calls_start_one_background_refresh(make):
    clock = Clock()
    det, client, seg = make(clock=clock)
    client.inventory = _reply(CUP, BALL)
    det.detect(_frame(), None)
    client.gate = threading.Event()
    clock.now = 5.0
    for i in range(10):
        dets = det.detect(_frame(), None)
        assert [d.label for d in dets] == ["paper cup", "red ball"], i
        np.testing.assert_allclose(seg.calls[-1][1], [_px(CUP[1]), _px(BALL[1])], rtol=1e-6)
    assert client.entered.wait(5), "no background refresh started"
    assert client.inventory_calls() == 2
    assert client.blocked == 1 and len(_brain_threads()) == 1
    client.gate.set()
    det.close()
    assert client.inventory_calls() == 2


def test_refresh_replaces_the_entry_on_success(make):
    clock = Clock()
    det, client, seg = make(clock=clock)
    client.inventory = _reply(CUP)
    det.detect(_frame(), None)
    client.inventory = _reply(BALL)
    clock.now = 5.0
    assert [d.label for d in det.detect(_frame(), None)] == ["paper cup"]
    det.close()  # joins the refresh
    clock.now = 6.0  # the new entry dates from 5.0: fresh
    assert [d.label for d in det.detect(_frame(), None)] == ["red ball"]
    np.testing.assert_allclose(seg.calls[-1][1], [_px(BALL[1])], rtol=1e-6)
    assert client.inventory_calls() == 2


def test_failed_refresh_keeps_the_old_entry(make, caplog):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    det.detect(_frame(), None)
    client.gate = threading.Event()
    client.gate.set()
    client.refresh_error = APITimeoutError("brain timed out")
    clock.now = 5.0
    with caplog.at_level(logging.WARNING, logger="cascade.perception.brain_detector"):
        det.detect(_frame(), None)
        det.close()
    assert client.inventory_calls() == 2
    assert len(caplog.records) == 1
    clock.now = 6.0
    assert [d.label for d in det.detect(_frame(), None)] == ["paper cup"]
    clock.now = 30.5  # it still ages out
    client.refresh_error = None
    client.inventory = _reply(BALL)
    assert [d.label for d in det.detect(_frame(), None)] == ["red ball"]


def test_slow_refresh_never_overwrites_a_newer_entry(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    det.detect(_frame(level=0), None)
    gate = client.gate = threading.Event()
    clock.now = 5.0
    det.detect(_frame(level=0), None)  # refresh of the t=5 frame hangs
    assert client.entered.wait(5)
    client.gate = None
    client.inventory = _reply(BALL)
    clock.now = 6.0
    assert [d.label for d in det.detect(_frame(level=20), None)] == ["red ball"]  # scene changed
    client.inventory = _reply(SPOOL)
    gate.set()  # the t=5 refresh now answers, older than the t=6 entry
    det.close()
    clock.now = 7.0
    assert [d.label for d in det.detect(_frame(level=20), None)] == ["red ball"]
    assert client.inventory_calls() == 3


def test_close_stops_later_refreshes(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    det.detect(_frame(), None)
    det.close()
    clock.now = 5.0
    assert [d.label for d in det.detect(_frame(), None)] == ["paper cup"]
    assert client.inventory_calls() == 1
    assert _brain_threads() == []


def _blocked_warm_up(det: BrainDetector):
    """A SAM warm-up (apps.demo starts it, the detector owns it) blocked on an Event."""
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def warm_up():
        started.set()
        release.wait(10)
        finished.set()

    det.warm_up_thread = threading.Thread(target=warm_up, name="sam-warm-up", daemon=True)
    det.warm_up_thread.start()
    assert started.wait(5)
    return release, finished


def test_a_detector_starts_without_a_warm_up_thread(make):
    det, _, _ = make()
    assert det.warm_up_thread is None


def test_close_waits_for_the_sam_warm_up(make):
    det, _, _ = make()
    release, finished = _blocked_warm_up(det)
    thread = det.warm_up_thread
    done = threading.Event()
    at_return = {}

    def close():
        det.close()
        at_return.update(finished=finished.is_set(), alive=thread.is_alive())
        done.set()

    threading.Thread(target=close, daemon=True, name="test-close").start()
    try:
        early = done.wait(0.3)
    finally:
        release.set()
    assert not early, "close() returned while the SAM warm-up was still running"
    assert done.wait(5)
    assert at_return == {"finished": True, "alive": False}


def test_close_gives_up_on_a_warm_up_past_its_bound_and_keeps_the_handle(make, monkeypatch, capsys):
    from cascade.perception import brain_detector

    monkeypatch.setattr(brain_detector, "_WARM_UP_JOIN_S", 0.2)
    det, _, _ = make()
    release, _ = _blocked_warm_up(det)
    thread = det.warm_up_thread
    try:
        det.close()  # returns after the bound; the thread is still in its warm-up
        assert thread.is_alive() and det.warm_up_thread is thread
        assert "thread 'sam-warm-up' still running after" in capsys.readouterr().err
    finally:
        release.set()
        thread.join(5)


def _join_refreshes(timeout_s: float = 5.0) -> None:
    """Wait for the refresh threads WITHOUT close(): they must clear their own bookkeeping."""
    for thread in _brain_threads():
        thread.join(timeout_s)
    assert _brain_threads() == []


def test_a_finished_refresh_lets_a_later_aged_call_start_a_new_one(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    det.detect(_frame(), None)
    clock.now = 5.0
    det.detect(_frame(), None)  # aged: refresh #1, entry now dates from 5.0
    _join_refreshes()
    assert client.inventory_calls() == 2
    client.inventory = _reply(BALL)
    clock.now = 10.0  # aged again
    assert [d.label for d in det.detect(_frame(), None)] == ["paper cup"]
    _join_refreshes()
    assert client.inventory_calls() == 3, "the finished refresh left its camera marked in flight"
    clock.now = 11.0
    assert [d.label for d in det.detect(_frame(), None)] == ["red ball"]


def test_refer_returns_while_a_refresh_is_in_flight(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    client.refer["red ball"] = _reply(BALL)
    det.detect(_frame(), None)
    client.gate = threading.Event()
    clock.now = 5.0
    det.detect(_frame(), None)
    assert client.entered.wait(5)
    assert [d.label for d in det.detect(_frame(), ["red ball"])] == ["red ball"]
    assert client.blocked == 1, "the refer waited for the background inventory"


def test_cache_lock_is_not_held_across_a_brain_call(make):
    det, client, _ = make()
    client.inventory = _reply(SPOOL)
    det.detect(_frame(fx=600.0), None)  # wrist camera primed
    client.gate = threading.Event()
    cold = threading.Thread(target=det.detect, args=(_frame(fx=500.0), None))
    cold.start()
    try:
        assert client.entered.wait(5)
        dets = det.detect(_frame(fx=600.0), None)  # track step while the scene call hangs
        assert [d.label for d in dets] == ["green filament spool"]
        assert client.blocked == 1
    finally:
        client.gate.set()
        cold.join(5)


def test_cameras_with_different_intrinsics_keep_separate_entries(make):
    det, client, _ = make()
    client.inventory = _reply(CUP)
    det.detect(_frame(fx=500.0), None)
    client.inventory = _reply(BALL)
    det.detect(_frame(fx=600.0), None)
    assert client.inventory_calls() == 2
    assert [d.label for d in det.detect(_frame(fx=500.0), None)] == ["paper cup"]
    assert [d.label for d in det.detect(_frame(fx=600.0), None)] == ["red ball"]
    assert client.inventory_calls() == 2


# --- masks -----------------------------------------------------------------


def test_track_step_drops_boxes_without_a_mask(make):
    clock = Clock()
    seg = FakeSegmenter(drop={1})
    det, client, _ = make(segmenter=seg, clock=clock)
    client.inventory = _reply(CUP, BALL)
    cold = det.detect(_frame(), None)
    assert [(d.label, d.mask is None) for d in cold] == [("paper cup", False), ("red ball", True)]
    clock.now = 1.0
    assert [d.label for d in det.detect(_frame(), None)] == ["paper cup"]


def test_without_a_segmenter_boxes_keep_no_mask(make):
    clock = Clock()
    det, client, _ = make(segmenter=None, clock=clock)
    client.inventory = _reply(CUP, BALL)
    det.detect(_frame(), None)
    clock.now = 1.0
    dets = det.detect(_frame(), None)
    assert [(d.label, d.mask) for d in dets] == [("paper cup", None), ("red ball", None)]


def test_segmenter_failure_keeps_boxes_without_masks(make, caplog):
    clock = Clock()
    seg = FakeSegmenter()
    seg.error = RuntimeError("SAM weights missing")
    det, client, _ = make(segmenter=seg, clock=clock)
    client.inventory = _reply(CUP)
    with caplog.at_level(logging.WARNING, logger="cascade.perception.brain_detector"):
        cold = det.detect(_frame(), None)
        clock.now = 1.0
        track = det.detect(_frame(), None)
    assert [(d.label, d.mask) for d in cold] == [("paper cup", None)]
    assert [(d.label, d.mask) for d in track] == [("paper cup", None)]
    assert len(caplog.records) == 1


# --- failures --------------------------------------------------------------


def test_brain_failure_is_empty_and_warns_at_most_every_30s(make, caplog):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.error = APITimeoutError("Request timed out.")
    with caplog.at_level(logging.WARNING, logger="cascade.perception.brain_detector"):
        for t in (0.0, 10.0, 29.9):
            clock.now = t
            assert det.detect(_frame(), None) == []
        assert len(caplog.records) == 1
        clock.now = 30.0  # the other camera: this one backs off since its 29.9 failure
        assert det.detect(_frame(fx=600.0), None) == []
        assert len(caplog.records) == 2
        client.error = RuntimeError("connection refused")
        clock.now = 31.0
        assert det.detect(_frame(), ["paper cup"]) == []
        assert len(caplog.records) == 2
    assert all(r.levelno == logging.WARNING for r in caplog.records)


def test_garbled_or_missing_reply_is_empty_with_a_warning(make, caplog):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = "I cannot see a table here."
    with caplog.at_level(logging.WARNING, logger="cascade.perception.brain_detector"):
        assert det.detect(_frame(), None) == []
        assert len(caplog.records) == 1
        clock.now = 30.0
        client.refer["paper cup"] = None  # message.content may be None
        assert det.detect(_frame(), ["paper cup"]) == []
        assert len(caplog.records) == 2
        assert all("garbled reply" in r.getMessage() for r in caplog.records)
        # A garbled reply is a failed call: nothing was cached.
        client.inventory = "[]"
        clock.now = 60.0
        assert det.detect(_frame(), None) == []
        assert client.inventory_calls() == 2
        # "[]" is a real answer (an empty table), cached without a warning.
        clock.now = 61.0
        assert det.detect(_frame(), None) == []
        assert client.inventory_calls() == 2
        assert len(caplog.records) == 2


def test_truncated_reply_is_garbled_and_not_cached(make, caplog):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.finish_reason = "length"  # cut at max_tokens
    client.inventory = _reply(CUP, BALL)[:-30]
    with caplog.at_level(logging.WARNING, logger="cascade.perception.brain_detector"):
        assert det.detect(_frame(), None) == []
        # A list that closed before the cut is still a cut answer: the brain had more to list.
        client.inventory = _reply(CUP) + "\nAlso a red ball at [5"
        assert det.detect(_frame(fx=600.0), None) == []
    (record,) = caplog.records  # one per 30 s
    assert "truncated" in record.getMessage()
    # Nothing was cached: once the backoff is over both cameras ask again.
    client.finish_reason = "stop"
    client.inventory = _reply(CUP, BALL)
    clock.now = 4.5
    assert [d.label for d in det.detect(_frame(), None)] == ["paper cup", "red ball"]
    assert [d.label for d in det.detect(_frame(fx=600.0), None)] == ["paper cup", "red ball"]
    assert client.inventory_calls() == 4


def test_prose_reply_is_garbled_and_not_cached(make, caplog):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = 'I can see [a paper cup] next to a ball.'
    with caplog.at_level(logging.WARNING, logger="cascade.perception.brain_detector"):
        assert det.detect(_frame(), None) == []
        client.inventory = '["paper cup", "red ball"]'  # names, no boxes
        assert det.detect(_frame(fx=600.0), None) == []
    (record,) = caplog.records
    assert "garbled reply" in record.getMessage()
    client.inventory = _reply(CUP)
    clock.now = 4.5
    assert [d.label for d in det.detect(_frame(), None)] == ["paper cup"]
    assert [d.label for d in det.detect(_frame(fx=600.0), None)] == ["paper cup"]
    assert client.inventory_calls() == 4


def test_bare_empty_list_is_a_cached_empty_table(make, caplog):
    clock = Clock()
    det, client, _ = make(clock=clock)
    replies = {500.0: "[]", 600.0: "```json\n[ ]\n```", 700.0: "<think>nothing there</think>\n[]"}
    with caplog.at_level(logging.WARNING, logger="cascade.perception.brain_detector"):
        for fx, reply in replies.items():
            client.inventory = reply
            assert det.detect(_frame(fx=fx), None) == []
        client.inventory = _reply(CUP)
        clock.now = 1.0
        for fx in replies:
            assert det.detect(_frame(fx=fx), None) == []  # the cached empty table, no request
    assert client.inventory_calls() == 3
    assert caplog.records == []


# Qwen's trailing-[] habit, and an empty answer followed by a remark.
EMPTY_WITH_TRAILING_TEXT = ("[]\n[]", "[]\n\nNo manipulable objects are visible.")


@pytest.mark.parametrize("reply", EMPTY_WITH_TRAILING_TEXT)
def test_empty_list_with_trailing_text_is_a_cached_empty_table(make, caplog, reply):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    det.detect(_frame(), None)
    client.inventory = reply
    with caplog.at_level(logging.WARNING, logger="cascade.perception.brain_detector"):
        clock.now = 31.0  # the cup entry aged out: a synchronous call, the table is empty now
        assert det.detect(_frame(), None) == []
        assert client.inventory_calls() == 2
        clock.now = 32.0  # the empty table replaced the cup: no request, the cup is gone
        assert det.detect(_frame(), None) == []
        assert client.inventory_calls() == 2
        # Not a failure, so no backoff: a changed scene asks at once.
        client.inventory = _reply(BALL)
        assert [d.label for d in det.detect(_frame(level=20), None)] == ["red ball"]
        assert client.inventory_calls() == 3
    assert caplog.records == []


@pytest.mark.parametrize("reply", EMPTY_WITH_TRAILING_TEXT)
def test_a_refresh_answering_an_empty_list_with_trailing_text_clears_the_table(make, caplog, reply):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    det.detect(_frame(), None)
    client.inventory = reply
    with caplog.at_level(logging.WARNING, logger="cascade.perception.brain_detector"):
        clock.now = 5.0
        assert [d.label for d in det.detect(_frame(), None)] == ["paper cup"]  # track; refresh starts
        _join_refreshes()
        clock.now = 6.0  # the refreshed entry (from 5.0) is the empty table
        assert det.detect(_frame(), None) == []
    assert client.inventory_calls() == 2
    assert caplog.records == []


def test_truncated_reply_opening_with_an_empty_list_is_still_garbled(make, caplog):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    det.detect(_frame(), None)
    client.finish_reason = "length"
    client.inventory = '[]\n[{"label": "red ball", "bbox_2d": [5'
    with caplog.at_level(logging.WARNING, logger="cascade.perception.brain_detector"):
        clock.now = 31.0  # a failed call: nothing cached, the outage fallback tracks the cup
        assert [d.label for d in det.detect(_frame(), None)] == ["paper cup"]
        clock.now = 32.0  # and the camera backs off
        assert [d.label for d in det.detect(_frame(), None)] == ["paper cup"]
    assert client.inventory_calls() == 2
    (record,) = caplog.records
    assert "truncated" in record.getMessage()


# --- brain outage: keep tracking what is known, back off -------------------


def test_outage_keeps_tracking_known_objects_past_max_age_with_backoff(make, caplog):
    clock = Clock()
    det, client, seg = make(clock=clock)
    client.inventory = _reply(CUP, BALL)
    det.detect(_frame(), None)
    client.error = APITimeoutError("Request timed out.")
    with caplog.at_level(logging.WARNING, logger="cascade.perception.brain_detector"):
        for t in (31.0, 32.0, 34.9):  # the entry is older than max_cache_age_s; same scene
            clock.now = t
            frame = _frame()
            assert [d.label for d in det.detect(frame, None)] == ["paper cup", "red ball"], t
            seg_frame, boxes = seg.calls[-1]
            assert seg_frame is frame  # a track step on THIS frame
            np.testing.assert_allclose(boxes, [_px(CUP[1]), _px(BALL[1])], rtol=1e-6)
        assert client.inventory_calls() == 2, "no backoff: every call asked the down brain"
        clock.now = 35.0  # refresh_s after the failure: ask again
        assert [d.label for d in det.detect(_frame(), None)] == ["paper cup", "red ball"]
        assert client.inventory_calls() == 3
    assert len(caplog.records) == 1
    assert _brain_threads() == []


def test_outage_on_a_changed_scene_is_empty_and_backs_off(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    det.detect(_frame(level=0), None)
    client.error = RuntimeError("connection refused")
    clock.now = 1.0
    # Old boxes on a changed table would put old labels on new things.
    assert det.detect(_frame(level=20), None) == []
    clock.now = 2.0
    assert det.detect(_frame(level=20), None) == []
    clock.now = 3.0  # back on the cached scene: an ordinary track step
    assert [d.label for d in det.detect(_frame(level=0), None)] == ["paper cup"]
    clock.now = 4.9
    assert det.detect(_frame(level=20), None) == []
    assert client.inventory_calls() == 2
    clock.now = 5.0
    assert det.detect(_frame(level=20), None) == []
    assert client.inventory_calls() == 3


def test_outage_backoff_is_per_camera(make):
    det, client, _ = make()
    client.error = RuntimeError("connection refused")
    assert det.detect(_frame(fx=500.0), None) == []
    assert det.detect(_frame(fx=500.0), None) == []
    assert client.inventory_calls() == 1
    assert det.detect(_frame(fx=600.0), None) == []  # the wrist camera asks for itself
    assert client.inventory_calls() == 2


def test_after_the_backoff_a_successful_call_replaces_the_entry(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    det.detect(_frame(), None)
    client.error = RuntimeError("connection refused")
    clock.now = 31.0
    assert [d.label for d in det.detect(_frame(), None)] == ["paper cup"]
    client.error = None  # the brain is back
    client.inventory = _reply(BALL)
    clock.now = 33.0  # still backing off
    assert [d.label for d in det.detect(_frame(), None)] == ["paper cup"]
    assert client.inventory_calls() == 2
    clock.now = 35.0
    assert [d.label for d in det.detect(_frame(), None)] == ["red ball"]
    assert client.inventory_calls() == 3
    clock.now = 36.0  # the new entry is fresh: a track step, no request
    assert [d.label for d in det.detect(_frame(), None)] == ["red ball"]
    assert client.inventory_calls() == 3


def test_backoff_counts_from_the_failure_not_the_request(make):
    # A hung brain times out after timeout_s: counted from the request, the
    # backoff would be over before the call returned and every tick would block.
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.error = APITimeoutError("Request timed out.")
    create = client.chat.completions.create

    def slow_create(**kwargs):
        clock.now += 20.0  # timeout_s passes inside the call
        return create(**kwargs)

    client.chat.completions.create = slow_create
    assert det.detect(_frame(), None) == []  # asked at 0, failed at 20
    clock.now = 22.0
    assert det.detect(_frame(), None) == []
    assert client.inventory_calls() == 1


def test_a_success_clears_the_backoff(make):
    # Two synchronous calls for one camera overlap: the later one fails, then
    # the earlier one answers. The brain is up, so a scene change asks at once.
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    client.gate = threading.Event()
    slow = threading.Thread(target=det.detect, args=(_frame(level=0), None))
    slow.start()
    try:
        assert client.entered.wait(5)
        client.error = RuntimeError("connection reset")
        clock.now = 1.0
        assert det.detect(_frame(level=0), None) == []  # fails: backoff from t=1
        client.error = None
    finally:
        client.gate.set()
        slow.join(5)
    client.gate = None
    client.inventory = _reply(BALL)
    clock.now = 2.0
    assert [d.label for d in det.detect(_frame(level=40), None)] == ["red ball"]
    assert client.inventory_calls() == 3


def test_a_successful_refresh_clears_the_backoff(make):
    # One backoff per key, shared by synchronous and background calls: a
    # refresh already running when a synchronous call failed ends it by answering.
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    det.detect(_frame(level=0), None)
    client.gate = threading.Event()
    client.inventory = _reply(BALL)
    clock.now = 5.0  # the cup entry is aged: a track step starts a refresh, which hangs
    assert [d.label for d in det.detect(_frame(level=0), None)] == ["paper cup"]
    assert client.entered.wait(5)
    client.error = RuntimeError("connection refused")
    clock.now = 5.5
    assert det.detect(_frame(level=20), None) == []  # scene changed, call failed: backoff until 9.5
    client.error = None  # the brain is back
    client.gate.set()
    _join_refreshes()  # the refresh answers
    client.gate = None
    assert client.inventory_calls() == 3
    client.inventory = _reply(SPOOL)
    clock.now = 6.0  # the refresh answered: a changed scene asks at once
    assert [d.label for d in det.detect(_frame(level=40), None)] == ["green filament spool"]
    assert client.inventory_calls() == 4


def test_failure_warning_says_what_the_fallback_returns(make, caplog, tmp_path):
    key = "sk-fallback-wording-0123456789"
    (tmp_path / "k").write_text(key)
    clock = Clock()
    det, client, _ = make(clock=clock, api_key_file=str(tmp_path / "k"))
    client.inventory = _reply(CUP)
    det.detect(_frame(fx=600.0), None)  # the wrist camera knows a cup
    client.error = RuntimeError(f"connection refused (key {key})")
    with caplog.at_level(logging.WARNING, logger="cascade.perception.brain_detector"):
        assert det.detect(_frame(fx=500.0), None) == []  # nothing known for this camera
        clock.now = 30.0
        assert det.detect(_frame(fx=600.0, level=20), None) == []  # the scene changed
        clock.now = 60.0
        assert [d.label for d in det.detect(_frame(fx=600.0), None)] == ["paper cup"]  # same scene
        clock.now = 61.0  # one limiter for both wordings
        assert det.detect(_frame(fx=500.0), None) == []
    assert client.inventory_calls() == 5
    empty_no_entry, empty_changed, tracking = (r.getMessage() for r in caplog.records)
    for message in (empty_no_entry, empty_changed):
        assert "tracking" not in message
        assert "returning no detections" in message
    assert "tracking the known objects" in tracking
    assert "connection refused" in tracking
    assert all(key not in r.getMessage() for r in caplog.records)


def test_unparseable_reply_never_raises(make):
    det, client, _ = make()
    client.refer["paper cup"] = '[{"label": "cup", "bbox_2d": [1' + "0" * 309 + ', 0, 500, 500]}]'
    client.inventory = "[" * 100000 + "]" * 100000
    det.detect(_frame(), ["paper cup"])
    assert det.detect(_frame(), None) == []


def test_pathologically_nested_reply_is_a_garbled_reply(make, caplog):
    # Nesting beyond the decoder's recursion is garbage like any other: the
    # garbled-reply failure, not a RecursionError escaping the reply check.
    det, client, _ = make()
    client.inventory = "[" * 100000 + "]" * 100000
    with caplog.at_level(logging.WARNING, logger="cascade.perception.brain_detector"):
        assert det.detect(_frame(), None) == []
    (record,) = caplog.records
    assert "garbled reply" in record.getMessage()


# --- detect_nowait: the watcher's call never waits for the brain -------------
#
# The watcher sends the motion watchdog's heartbeat once per tick (5 s
# watchdog), so its detector call must never block on a ~5 s inventory or a
# hung brain: detect_nowait answers from the cache and asks the brain only on
# a refresh thread.


def _on_refresh_threads(client: FakeClient) -> bool:
    return bool(client.threads) and all(name.startswith("brain-refresh") for name in client.threads)


def test_nowait_without_an_entry_is_empty_and_asks_once_in_the_background(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP, BALL)
    client.gate = threading.Event()
    for i in range(10):
        assert det.detect_nowait(_frame()) == [], i  # returns while the brain is still busy
    assert client.entered.wait(5), "no background request started"
    assert client.inventory_calls() == 1 and client.blocked == 1
    assert _on_refresh_threads(client), f"the brain was asked on {client.threads}"
    client.gate.set()
    _join_refreshes()
    clock.now = 1.0
    assert [d.label for d in det.detect_nowait(_frame())] == ["paper cup", "red ball"]
    assert client.inventory_calls() == 1


def test_nowait_same_scene_is_a_track_step_without_a_brain_call(make):
    clock = Clock()
    det, client, seg = make(clock=clock)
    client.inventory = _reply(CUP, BALL)
    det.detect(_frame(), None)  # a synchronous inventory fills the same (camera, None) entry
    clock.now = 4.0  # exactly refresh_s old: not aged yet
    frame = _frame(level=6)  # mean abs diff 6.0: the same scene
    dets = det.detect_nowait(frame)
    assert [d.label for d in dets] == ["paper cup", "red ball"]
    seg_frame, boxes = seg.calls[-1]
    assert seg_frame is frame  # a track step on THIS frame
    np.testing.assert_allclose(boxes, [_px(CUP[1]), _px(BALL[1])], rtol=1e-6)
    assert [d.label for d in det.detect_nowait(frame, [])] == ["paper cup", "red ball"]  # [] is None
    assert client.inventory_calls() == 1
    assert _brain_threads() == []


def test_nowait_tracks_an_aged_entry_at_any_age_and_refreshes_once(make):
    clock = Clock()
    det, client, seg = make(clock=clock)
    client.inventory = _reply(CUP)
    det.detect(_frame(), None)
    client.gate = threading.Event()
    client.inventory = _reply(BALL)
    clock.now = 100.0  # past max_cache_age_s too: detect() would ask synchronously here
    for i in range(10):
        frame = _frame()
        assert [d.label for d in det.detect_nowait(frame)] == ["paper cup"], i
        assert seg.calls[-1][0] is frame
    assert client.entered.wait(5), "no background refresh started"
    assert client.inventory_calls() == 2 and client.blocked == 1 and len(_brain_threads()) == 1
    assert client.threads[1].startswith("brain-refresh")
    client.gate.set()
    _join_refreshes()
    clock.now = 101.0  # the refreshed entry dates from 100.0
    assert [d.label for d in det.detect_nowait(_frame())] == ["red ball"]
    assert client.inventory_calls() == 2


def test_nowait_on_a_changed_scene_is_empty_and_refreshes_in_the_background(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    det.detect(_frame(level=0), None)
    client.gate = threading.Event()
    client.inventory = _reply(BALL)
    clock.now = 1.0
    for i in range(5):
        assert det.detect_nowait(_frame(level=7)) == [], i  # 7.0 > scene_change: old boxes do not apply
    assert client.entered.wait(5), "no background refresh started"
    assert client.inventory_calls() == 2
    assert client.threads[1].startswith("brain-refresh")
    # The cached scene still tracks while its replacement is being asked for.
    assert [d.label for d in det.detect_nowait(_frame(level=0))] == ["paper cup"]
    client.gate.set()
    _join_refreshes()
    clock.now = 2.0
    assert [d.label for d in det.detect_nowait(_frame(level=7))] == ["red ball"]
    assert client.inventory_calls() == 2


def test_nowait_refer_asks_in_the_background_under_its_own_key(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(SPOOL)
    client.refer["paper cup"] = _reply(("cup", CUP[1]))
    client.refer["red ball"] = _reply(("ball", BALL[1]), ("ball", SPOOL[1]))
    det.detect(_frame(), None)  # the inventory entry is not the refer's
    queries = ["paper cup", "red ball"]
    assert det.detect_nowait(_frame(), queries) == []
    _join_refreshes()
    assert client.prompts() == [INVENTORY_PROMPT, REFER_PROMPT.format(q="paper cup"),
                                REFER_PROMPT.format(q="red ball")]
    assert all(name.startswith("brain-refresh") for name in client.threads[1:]), client.threads
    clock.now = 1.0
    dets = det.detect_nowait(_frame(), queries)
    assert [d.label for d in dets] == ["paper cup", "red ball", "red ball"]  # the queries, not the brain's labels
    np.testing.assert_allclose([d.bbox for d in dets], [_px(CUP[1]), _px(BALL[1]), _px(SPOOL[1])], rtol=1e-6)
    assert len(client.requests) == 3
    assert [d.label for d in det.detect_nowait(_frame())] == ["green filament spool"]  # inventory untouched
    assert det.detect_nowait(_frame(), ["red ball"]) == []  # another query list is another key
    _join_refreshes()
    assert client.prompts()[3:] == [REFER_PROMPT.format(q="red ball")]
    # The synchronous refer stays uncached: it asks the brain every time.
    assert [d.label for d in det.detect(_frame(), queries)] == ["paper cup", "red ball", "red ball"]
    assert len(client.requests) == 6


def test_nowait_refreshes_an_aged_refer_entry_with_refer_prompts(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.refer["paper cup"] = _reply(CUP)
    det.detect_nowait(_frame(), ["paper cup"])
    _join_refreshes()
    client.refer["paper cup"] = _reply(BALL)  # the cup moved
    clock.now = 30.5  # older than idle_refresh_s: the watcher's refresh of an unchanged scene
    assert [d.label for d in det.detect_nowait(_frame(), ["paper cup"])] == ["paper cup"]
    _join_refreshes()
    clock.now = 31.0
    (d,) = det.detect_nowait(_frame(), ["paper cup"])
    np.testing.assert_allclose(d.bbox, _px(BALL[1]), rtol=1e-6)
    assert client.prompts() == [REFER_PROMPT.format(q="paper cup")] * 2
    assert _on_refresh_threads(client)


def test_a_nowait_refresh_ends_the_inventory_backoff(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    client.gate = threading.Event()
    assert det.detect_nowait(_frame(level=0)) == []  # a background inventory, hanging
    assert client.entered.wait(5)
    client.error = RuntimeError("connection refused")
    clock.now = 0.5
    assert det.detect(_frame(level=0), None) == []  # a failed synchronous call: backoff until 4.5
    client.error = None
    client.gate.set()
    _join_refreshes()  # the background inventory answers
    client.gate = None
    client.inventory = _reply(BALL)
    clock.now = 1.0  # the brain answered: a changed scene asks at once
    assert [d.label for d in det.detect(_frame(level=40), None)] == ["red ball"]
    assert client.inventory_calls() == 3


def test_nowait_after_close_asks_nothing(make):
    det, client, _ = make()
    det.close()
    assert det.detect_nowait(_frame()) == []
    assert det.detect_nowait(_frame(), ["paper cup"]) == []
    assert client.requests == [] and _brain_threads() == []


def test_nowait_never_raises(make, caplog):
    from cascade.types import Frame

    det, client, _ = make()
    gray = Frame(rgb=np.zeros((H, W), dtype=np.uint8), depth_m=None, K=_frame().K)  # BGR2GRAY fails
    with caplog.at_level(logging.WARNING, logger="cascade.perception.brain_detector"):
        assert det.detect_nowait(gray) == []
    (record,) = caplog.records
    assert "detection failed" in record.getMessage()
    assert client.requests == []


# --- one failure backoff per cache key, synchronous or background ------------
#
# Every failed brain call for a key (a synchronous inventory or a refresh
# thread) starts that key's backoff: no new brain call for the key, from either
# path, for refresh_s. Without it detect_nowait starts a request per watcher
# tick (~3/s per camera) against a refusing brain. Any answer for the key ends it.


def _fail(client: FakeClient, how: str) -> None:
    if how == "refused":
        client.error = RuntimeError("connection refused")  # fails at once, like a closed port
    else:
        client.inventory = "I cannot see a table here."  # garbled: a failed call too


@pytest.mark.parametrize("how", ["refused", "garbled"])
@pytest.mark.parametrize("caller", ["nowait-cold", "nowait-aged", "detect-aged"])
def test_a_failing_brain_gets_one_background_request_per_refresh_s(make, caller, how):
    clock = Clock()
    # idle_refresh_s = refresh_s: an aged entry refreshes on both paths, so the backoff decides.
    det, client, _ = make(clock=clock, idle_refresh_s=4.0)
    if caller == "nowait-cold":
        t0, expected = 0.0, []
    else:
        client.inventory = _reply(CUP)
        det.detect(_frame(), None)  # the entry dates from t=0
        t0, expected = 5.0, ["paper cup"]  # aged (> refresh_s), younger than max_cache_age_s
    call = det.detect_nowait if caller.startswith("nowait") else (lambda frame: det.detect(frame, None))
    asked = client.inventory_calls()
    _fail(client, how)
    for i in range(9):  # watcher ticks within refresh_s of the first failure
        clock.now = t0 + 0.4 * i
        assert [d.label for d in call(_frame())] == expected, i
        _join_refreshes()  # a fast-failing brain: each refresh is over before the next tick
    assert client.inventory_calls() == asked + 1, "a new background request per tick against a failing brain"
    clock.now = t0 + 4.0  # refresh_s after the failure: one more request
    assert [d.label for d in call(_frame())] == expected
    assert [d.label for d in call(_frame())] == expected
    _join_refreshes()
    assert client.inventory_calls() == asked + 2
    assert all(name.startswith("brain-refresh") for name in client.threads[asked:]), client.threads


def test_a_background_backoff_counts_from_the_failure_not_the_request(make):
    # A hung brain fails after timeout_s: counted from the request, the backoff
    # would already be over and the next tick would ask the hung brain at once.
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.error = APITimeoutError("Request timed out.")
    create = client.chat.completions.create

    def slow_create(**kwargs):
        clock.now += 20.0  # timeout_s passes inside the call, on the refresh thread
        return create(**kwargs)

    client.chat.completions.create = slow_create
    assert det.detect_nowait(_frame()) == []  # asked at 0, failed at 20
    _join_refreshes()
    clock.now = 22.0
    assert det.detect_nowait(_frame()) == []
    _join_refreshes()
    assert client.inventory_calls() == 1
    clock.now = 24.0  # refresh_s after the failure
    assert det.detect_nowait(_frame()) == []
    _join_refreshes()
    assert client.inventory_calls() == 2


@pytest.mark.parametrize("tick", ["aged", "changed"])
def test_a_success_ends_the_shared_backoff_for_the_next_nowait_tick(make, tick):
    clock = Clock()
    det, client, _ = make(clock=clock, idle_refresh_s=4.0)  # the watcher refreshes an aged entry
    client.inventory = _reply(CUP)
    det.detect(_frame(level=0), None)
    client.gate = threading.Event()
    client.inventory = _reply(BALL)
    clock.now = 5.0  # aged: the refresh of the t=5 frame hangs
    assert [d.label for d in det.detect_nowait(_frame(level=0))] == ["paper cup"]
    assert client.entered.wait(5)
    client.error = RuntimeError("connection reset")
    clock.now = 9.0
    assert det.detect(_frame(level=20), None) == []  # a failed call for the key: backoff until 13.0
    client.error = None
    client.gate.set()
    _join_refreshes()  # the t=5 refresh answers: the brain is up, the backoff ends
    client.gate = None
    assert client.inventory_calls() == 3
    client.inventory = _reply(SPOOL)
    clock.now = 9.5
    if tick == "aged":  # the refreshed entry dates from 5.0
        assert [d.label for d in det.detect_nowait(_frame(level=0))] == ["red ball"]
    else:
        assert det.detect_nowait(_frame(level=40)) == []
    _join_refreshes()
    assert client.inventory_calls() == 4, "the backoff outlived a successful answer"
    clock.now = 10.0
    frame = _frame(level=0 if tick == "aged" else 40)
    assert [d.label for d in det.detect_nowait(frame)] == ["green filament spool"]


def test_a_failed_refer_refresh_backs_off_only_its_own_key(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.refer["paper cup"] = "I cannot see a paper cup."  # garbled: the refer refresh fails
    client.inventory = _reply(BALL)
    assert det.detect_nowait(_frame(), ["paper cup"]) == []
    _join_refreshes()
    clock.now = 1.0
    assert det.detect_nowait(_frame(), ["paper cup"]) == []  # its key backs off: no request
    _join_refreshes()
    assert client.prompts() == [REFER_PROMPT.format(q="paper cup")]
    # The same camera's inventory key is not backed off, on either path.
    assert [d.label for d in det.detect(_frame(), None)] == ["red ball"]
    assert det.detect_nowait(_frame(level=40)) == []  # changed scene: a background refresh at once
    _join_refreshes()
    assert client.prompts()[1:] == [INVENTORY_PROMPT, INVENTORY_PROMPT]
    clock.now = 4.0  # refresh_s after the refer failure
    assert det.detect_nowait(_frame(), ["paper cup"]) == []
    _join_refreshes()
    assert client.prompts()[3:] == [REFER_PROMPT.format(q="paper cup")]


@pytest.mark.parametrize("scene", ["cold", "known", "changed"])
def test_a_failed_background_refresh_backs_off_the_synchronous_inventory(make, scene):
    clock = Clock()
    det, client, seg = make(clock=clock)
    if scene != "cold":
        client.inventory = _reply(CUP)
        det.detect(_frame(level=0), None)
    frame = _frame(level=20 if scene == "changed" else 0)
    expected = ["paper cup"] if scene == "known" else []
    client.error = RuntimeError("connection refused")
    clock.now = 31.0  # past max_cache_age_s: detect() would ask the brain synchronously
    assert [d.label for d in det.detect_nowait(frame)] == expected  # its refresh fails
    _join_refreshes()
    asked = client.inventory_calls()
    for t in (32.0, 34.9):
        clock.now = t
        assert [d.label for d in det.detect(frame, None)] == expected, t  # the outage fallback
        if expected:
            assert seg.calls[-1][0] is frame  # a track step on THIS frame
    assert client.inventory_calls() == asked, "a synchronous brain call during the key's backoff"
    assert _brain_threads() == []
    clock.now = 35.0  # refresh_s after the background failure: detect asks again
    assert [d.label for d in det.detect(frame, None)] == expected
    assert client.inventory_calls() == asked + 1


def test_a_failed_synchronous_inventory_also_holds_back_background_refreshes(make):
    clock = Clock()
    det, client, _ = make(clock=clock, idle_refresh_s=4.0)  # the watcher refreshes an aged entry
    client.inventory = _reply(CUP)
    det.detect(_frame(level=0), None)
    client.error = RuntimeError("connection refused")
    clock.now = 1.0
    assert det.detect(_frame(level=20), None) == []  # scene changed, the call failed: backoff until 5.0
    client.error = None  # the brain is back, but the key is backing off
    for t in (4.5, 4.9):  # the cup entry is aged: both paths would refresh it
        clock.now = t
        assert [d.label for d in det.detect(_frame(level=0), None)] == ["paper cup"]
        assert [d.label for d in det.detect_nowait(_frame(level=0))] == ["paper cup"]
        assert det.detect_nowait(_frame(level=20)) == []
        assert _brain_threads() == []
    assert client.inventory_calls() == 2
    clock.now = 5.0
    assert [d.label for d in det.detect_nowait(_frame(level=0))] == ["paper cup"]
    _join_refreshes()
    assert client.inventory_calls() == 3


# --- transport and key hygiene ---------------------------------------------


class _StubOpenAI:
    built: ClassVar[list[dict]] = []
    client: ClassVar[FakeClient | None] = None

    def __new__(cls, **kwargs):
        cls.built.append(kwargs)
        return cls.client


@pytest.fixture
def stub_openai(monkeypatch):
    monkeypatch.delitem(sys.modules, "openai", raising=False)
    mod = types.ModuleType("openai")
    mod.OpenAI = _StubOpenAI
    _StubOpenAI.built = []
    _StubOpenAI.client = FakeClient()

    def install():
        monkeypatch.setitem(sys.modules, "openai", mod)
        return _StubOpenAI

    return install


def test_key_from_file_never_logged_nor_in_repr(tmp_path, stub_openai, caplog):
    key = "sk-brain-5ecret-77"
    (tmp_path / "k").write_text(key + "\n")
    with caplog.at_level(logging.DEBUG, logger="cascade.perception.brain_detector"):
        det = BrainDetector(base_url="http://brain/v1", model="qwen3.8-27b",
                            api_key_file=str(tmp_path / "k"), timeout_s=7.5, clock=Clock())
        assert "openai" not in sys.modules, "openai imported before the first brain call"
        stub = stub_openai()
        stub.client.error = RuntimeError(f"401 Unauthorized: bad key {key}")
        assert det.detect(_frame(), None) == []
        assert det.detect(_frame(), ["paper cup"]) == []
    det.close()
    # max_retries=0: the next watcher tick is the retry; openai's default 2
    # would block a synchronous detect for ~3x timeout_s on a hung brain.
    assert stub.built == [{"base_url": "http://brain/v1", "api_key": key, "timeout": 7.5, "max_retries": 0}]
    assert caplog.records, "the failure was not logged"
    for record in caplog.records:
        assert key not in record.getMessage()
    assert key not in repr(det) and key not in str(det)
    assert "qwen3.8-27b" in repr(det)


def test_missing_key_file_sends_EMPTY_with_one_warning(tmp_path, stub_openai, caplog):
    with caplog.at_level(logging.WARNING, logger="cascade.perception.brain_detector"):
        det = BrainDetector(base_url="http://brain/v1", model="m",
                            api_key_file=str(tmp_path / "missing"), clock=Clock())
    assert len(caplog.records) == 1
    stub = stub_openai()
    stub.client.inventory = _reply(CUP)
    assert [d.label for d in det.detect(_frame(), None)] == ["paper cup"]
    det.close()
    assert [b["api_key"] for b in stub.built] == ["EMPTY"]


def test_injected_client_never_imports_openai(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)  # any import would raise
    client = FakeClient()
    client.inventory = _reply(CUP)
    det = BrainDetector(base_url="http://brain/v1", model="m", client=client, clock=Clock())
    assert [d.label for d in det.detect(_frame(), None)] == ["paper cup"]
    det.close()


# --- the configured default vocabulary: cached like the inventory (Ruling 30) ---
#
# The runtime's own vocabulary passes (describe_scene, _reobserve, the
# localization vocabulary round) call detect_vocabulary(frame, detect_classes):
# the (camera, tuple(classes)) entry the watcher's detect_nowait keeps warm,
# under the inventory's rules. An agent's own query stays detect(frame, [q]):
# a fresh, synchronous refer every time.

VOCAB = ["paper cup", "red ball"]


def _vocab_prompts(vocab=VOCAB) -> list[str]:
    return [REFER_PROMPT.format(q=q) for q in vocab]


def _vocab_client(client: FakeClient) -> FakeClient:
    client.refer["paper cup"] = _reply(("cup", CUP[1]))
    client.refer["red ball"] = _reply(("ball", BALL[1]))
    return client


def test_vocabulary_cold_asks_every_class_synchronously_then_tracks(make):
    clock = Clock()
    det, client, seg = make(clock=clock)
    _vocab_client(client)
    dets = det.detect_vocabulary(_frame(), VOCAB)
    assert [d.label for d in dets] == VOCAB  # the configured words, not the brain's labels
    assert client.prompts() == _vocab_prompts()
    assert not any(name.startswith("brain-refresh") for name in client.threads), client.threads
    clock.now = 3.9  # younger than refresh_s, the same scene: a track step, no request
    frame = _frame(level=6)
    dets = det.detect_vocabulary(frame, VOCAB)
    assert [d.label for d in dets] == VOCAB
    seg_frame, boxes = seg.calls[-1]
    assert seg_frame is frame
    np.testing.assert_allclose(boxes, [_px(CUP[1]), _px(BALL[1])], rtol=1e-6)
    assert len(client.requests) == 2
    assert _brain_threads() == []


def test_vocabulary_reads_the_entry_the_watcher_keeps_warm(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    _vocab_client(client)
    assert det.detect_nowait(_frame(), VOCAB) == []  # the watcher's tick: a background refer
    _join_refreshes()
    assert client.prompts() == _vocab_prompts()
    clock.now = 1.0
    assert [d.label for d in det.detect_vocabulary(_frame(), VOCAB)] == VOCAB
    assert len(client.requests) == 2, "the synchronous vocabulary pass asked the brain again"
    # And the other way round: the synchronous pass fills the watcher's entry.
    other = ["red ball"]
    assert [d.label for d in det.detect_vocabulary(_frame(), other)] == ["red ball"]
    assert [d.label for d in det.detect_nowait(_frame(), other)] == ["red ball"]
    _join_refreshes()
    assert len(client.requests) == 3


def test_an_aged_vocabulary_entry_tracks_and_refreshes_in_the_background(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    _vocab_client(client)
    det.detect_vocabulary(_frame(), VOCAB)
    client.refer["paper cup"] = _reply(("cup", SPOOL[1]))  # the cup moved
    clock.now = 5.0  # older than refresh_s: the cached boxes now, the brain in the background
    (cup, ball) = det.detect_vocabulary(_frame(), VOCAB)
    np.testing.assert_allclose(cup.bbox, _px(CUP[1]), rtol=1e-6)
    _join_refreshes()
    assert client.prompts() == _vocab_prompts() * 2
    assert all(name.startswith("brain-refresh") for name in client.threads[2:]), client.threads
    clock.now = 6.0
    (cup, ball) = det.detect_vocabulary(_frame(), VOCAB)
    np.testing.assert_allclose(cup.bbox, _px(SPOOL[1]), rtol=1e-6)
    assert len(client.requests) == 4


def test_vocabulary_on_a_changed_scene_or_past_max_age_asks_synchronously(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    _vocab_client(client)
    det.detect_vocabulary(_frame(level=0), VOCAB)
    clock.now = 1.0
    assert [d.label for d in det.detect_vocabulary(_frame(level=7), VOCAB)] == VOCAB
    assert len(client.requests) == 4
    clock.now = 31.5  # the t=1 entry is older than max_cache_age_s
    assert [d.label for d in det.detect_vocabulary(_frame(level=7), VOCAB)] == VOCAB
    assert len(client.requests) == 6
    assert not any(name.startswith("brain-refresh") for name in client.threads), client.threads
    assert _brain_threads() == []


def test_a_failed_vocabulary_pass_tracks_known_objects_and_shares_the_backoff(make, caplog):
    clock = Clock()
    det, client, seg = make(clock=clock)
    _vocab_client(client)
    det.detect_vocabulary(_frame(), VOCAB)
    client.error = RuntimeError("connection refused")
    with caplog.at_level(logging.WARNING, logger="cascade.perception.brain_detector"):
        clock.now = 31.0  # past max_cache_age_s: a synchronous pass, which fails
        frame = _frame()
        assert [d.label for d in det.detect_vocabulary(frame, VOCAB)] == VOCAB
        assert seg.calls[-1][0] is frame  # the outage fallback: a track step on THIS frame
        assert len(client.requests) == 3  # the first class's request failed, the pass stopped
        for t in (32.0, 34.9):  # backing off on both paths
            clock.now = t
            assert [d.label for d in det.detect_vocabulary(_frame(), VOCAB)] == VOCAB
            assert [d.label for d in det.detect_nowait(_frame(), VOCAB)] == VOCAB
            assert det.detect_vocabulary(_frame(level=20), VOCAB) == []  # changed scene: nothing known
            assert _brain_threads() == []
        assert len(client.requests) == 3
    (record,) = caplog.records
    assert "vocabulary" in record.getMessage() and "tracking the known objects" in record.getMessage()
    client.error = None
    clock.now = 35.0  # refresh_s after the failure: the synchronous pass asks again
    assert [d.label for d in det.detect_vocabulary(_frame(level=20), VOCAB)] == VOCAB
    assert len(client.requests) == 5


def test_a_failed_background_vocabulary_refresh_backs_off_the_synchronous_pass(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    _vocab_client(client)
    client.error = RuntimeError("connection refused")
    assert det.detect_nowait(_frame(), VOCAB) == []  # its refresh fails
    _join_refreshes()
    asked = len(client.requests)
    clock.now = 1.0
    assert det.detect_vocabulary(_frame(), VOCAB) == []  # backoff: no synchronous call
    assert len(client.requests) == asked


def test_a_single_query_refer_stays_uncached_next_to_a_vocabulary_entry(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    _vocab_client(client)
    det.detect_vocabulary(_frame(), ["paper cup"])  # a one-word vocabulary has an entry
    for t in (0.5, 1.0):
        clock.now = t
        assert [d.label for d in det.detect(_frame(), ["paper cup"])] == ["paper cup"]
    assert client.prompts() == [REFER_PROMPT.format(q="paper cup")] * 3, "an agent's query was served from a cache"


@pytest.mark.parametrize("vocab", [None, []])
def test_no_vocabulary_is_the_inventory(make, vocab):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    assert [d.label for d in det.detect_vocabulary(_frame(), vocab)] == ["paper cup"]
    clock.now = 1.0
    assert [d.label for d in det.detect(_frame(), None)] == ["paper cup"]
    assert [d.label for d in det.detect_vocabulary(_frame(), vocab)] == ["paper cup"]
    assert client.prompts() == [INVENTORY_PROMPT]


def test_vocabulary_never_raises(make, caplog):
    from cascade.types import Frame

    det, client, _ = make()
    gray = Frame(rgb=np.zeros((H, W), dtype=np.uint8), depth_m=None, K=_frame().K)  # BGR2GRAY fails
    with caplog.at_level(logging.WARNING, logger="cascade.perception.brain_detector"):
        assert det.detect_vocabulary(gray, VOCAB) == []
    (record,) = caplog.records
    assert "detection failed" in record.getMessage()
    assert client.requests == []


# --- the watcher's cadence: idle refresh and one request in flight (Ruling 31) ---
#
# On an unchanged scene detect_nowait refreshes only after idle_refresh_s
# (default 30 s): the watcher no longer asks the brain every refresh_s for a
# table nobody touched. A changed scene (or no entry) asks at once. At most one
# background request is in flight per detector, across cameras and keys.


def test_idle_refresh_s_defaults_to_30_s(make):
    det, _, _ = make()
    assert det._idle_refresh_s == 30.0


@pytest.mark.parametrize("idle", [None, 10.0])
def test_nowait_on_an_unchanged_scene_refreshes_only_after_idle_refresh_s(make, idle):
    clock = Clock()
    det, client, _ = make(clock=clock, **({} if idle is None else {"idle_refresh_s": idle}))
    idle = 30.0 if idle is None else idle
    client.inventory = _reply(CUP)
    det.detect(_frame(), None)  # the entry dates from t=0
    for t in (4.5, idle / 2, idle):  # older than refresh_s, not older than idle_refresh_s
        clock.now = t
        assert [d.label for d in det.detect_nowait(_frame(level=3))] == ["paper cup"], t
        _join_refreshes()
    assert client.inventory_calls() == 1, "the watcher refreshed an unchanged scene before idle_refresh_s"
    clock.now = idle + 0.1
    assert [d.label for d in det.detect_nowait(_frame(level=3))] == ["paper cup"]
    _join_refreshes()
    assert client.inventory_calls() == 2
    assert client.threads[1].startswith("brain-refresh")


def test_nowait_on_a_changed_scene_refreshes_at_once(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    det.detect(_frame(level=0), None)
    client.inventory = _reply(BALL)
    clock.now = 0.5  # far younger than refresh_s and idle_refresh_s
    assert det.detect_nowait(_frame(level=20)) == []
    _join_refreshes()
    assert client.inventory_calls() == 2
    clock.now = 1.0
    assert [d.label for d in det.detect_nowait(_frame(level=20))] == ["red ball"]


def test_the_synchronous_path_still_refreshes_after_refresh_s(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    det.detect(_frame(), None)
    clock.now = 4.5  # older than refresh_s, younger than idle_refresh_s
    assert [d.label for d in det.detect(_frame(), None)] == ["paper cup"]
    _join_refreshes()
    assert client.inventory_calls() == 2


def test_one_background_request_in_flight_across_cameras_and_keys(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    client.refer["paper cup"] = _reply(CUP)
    client.gate = threading.Event()
    assert det.detect_nowait(_frame(fx=500.0)) == []  # camera A: its refresh starts and hangs
    assert client.entered.wait(5)
    for i in range(5):  # camera B and a vocabulary key need one too: no second request
        assert det.detect_nowait(_frame(fx=600.0)) == [], i
        assert det.detect_nowait(_frame(fx=500.0), ["paper cup"]) == [], i
    assert len(client.requests) == 1 and len(_brain_threads()) == 1
    client.gate.set()
    _join_refreshes()
    client.gate = None
    assert det.detect_nowait(_frame(fx=600.0)) == []  # the next tick: camera B's turn
    _join_refreshes()
    assert client.inventory_calls() == 2
    clock.now = 1.0
    assert [d.label for d in det.detect_nowait(_frame(fx=600.0))] == ["paper cup"]
    assert [d.label for d in det.detect_nowait(_frame(fx=500.0))] == ["paper cup"]
    assert all(name.startswith("brain-refresh") for name in client.threads), client.threads


def test_an_aged_synchronous_track_step_waits_for_the_refresh_in_flight(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    det.detect(_frame(fx=600.0), None)  # the wrist camera's entry dates from t=0
    client.gate = threading.Event()
    assert det.detect_nowait(_frame(fx=500.0)) == []  # the scene camera's refresh hangs
    assert client.entered.wait(5)
    clock.now = 5.0  # the wrist entry is aged: its track step would refresh, but one is in flight
    assert [d.label for d in det.detect(_frame(fx=600.0), None)] == ["paper cup"]
    assert client.inventory_calls() == 2 and len(_brain_threads()) == 1
    client.gate.set()
    _join_refreshes()


# --- the refresh slot is served oldest-first (Ruling 34) ---
#
# One background request at a time must not let one camera keep the slot. The
# watcher ticks its cameras in order: a first camera whose scene changes every
# tick took the free slot again on its very next tick, and a later camera never
# got a refresh. Every key that asks while the slot is busy now waits, from its
# first ask; a free slot goes to the longest waiter (ties: first asked), with
# that waiter's latest frame, whichever caller finds it free.


def _background(client: FakeClient) -> list[tuple[int, int]]:
    """(camera fx, frame level) of each background request: the camera from its
    refresh thread's name, the level from the image the brain was sent."""
    with client._lock:
        sent = list(zip(client.threads, client.requests))
    out = []
    for name, request in sent:
        if not name.startswith("brain-refresh"):
            continue
        fx = next(fx for fx in (500, 600, 700, 800) if f", {fx}, " in name)
        url = request["messages"][0]["content"][0]["image_url"]["url"]
        jpeg = np.frombuffer(base64.b64decode(url.split(",", 1)[1]), dtype=np.uint8)
        out.append((fx, round(float(cv2.imdecode(jpeg, cv2.IMREAD_GRAYSCALE).mean()))))
    return out


def _sent(client: FakeClient, n: int, timeout_s: float = 5.0) -> None:
    """Wait until `n` requests reached the fake brain: a refresh thread just
    started may not have sent its request yet."""
    deadline = time.monotonic() + timeout_s
    while len(client.requests) < n and time.monotonic() < deadline:
        time.sleep(0.005)
    assert len(client.requests) == n, client.threads


def _foreground(client: FakeClient) -> list[str]:
    """Requests sent on the caller's thread (not a background refresh)."""
    with client._lock:
        return [name for name in client.threads if not name.startswith("brain-refresh")]


def test_a_camera_whose_scene_changes_every_tick_cannot_starve_the_next_one(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    det.detect(_frame(level=0, fx=500.0), None)  # both cameras' entries date from t=0
    det.detect(_frame(level=0, fx=600.0), None)
    client.refresh_slot = threading.Semaphore(0)
    level_a = 0
    for _cycle in range(4):  # the request in flight answers after three watcher ticks
        for _tick in range(3):  # the watcher's order: camera A (fx 500), then camera B (fx 600)
            clock.now += 0.3
            level_a += 20
            assert det.detect_nowait(_frame(level=level_a, fx=500.0)) == []  # A: a new scene every tick
            det.detect_nowait(_frame(level=100, fx=600.0))  # B: its scene changed once, then still
            assert len(_brain_threads()) == 1
        client.refresh_slot.release()
        _join_refreshes()
    cameras = [fx for fx, _ in _background(client)]
    # A's request in flight answers, then it is B's turn.
    assert cameras[:2] == [500, 600], f"camera B starved behind camera A: {cameras}"
    assert cameras.count(600) == 1 and _background(client)[1] == (600, 100)  # B's own latest frame
    clock.now += 0.3
    assert [d.label for d in det.detect_nowait(_frame(level=100, fx=600.0))] == ["paper cup"]


def test_a_free_slot_goes_to_the_longest_waiter_with_its_latest_frame(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    client.refresh_slot = threading.Semaphore(0)
    det.detect_nowait(_frame(level=10, fx=500.0))  # camera 500 takes the slot
    clock.now = 1.0
    det.detect_nowait(_frame(level=40, fx=600.0))  # 600 and 700 wait from t=1: a tie, 600 asked first
    det.detect_nowait(_frame(level=80, fx=700.0))
    clock.now = 2.0
    det.detect_nowait(_frame(level=60, fx=600.0))  # 600 again: still waiting since t=1, with this frame
    client.refresh_slot.release()
    _join_refreshes()
    clock.now = 3.0
    det.detect_nowait(_frame(level=90, fx=800.0))  # 800 finds the slot free: 600 has waited longest
    _sent(client, 2)
    assert [fx for fx, _ in _background(client)] == [500, 600]
    client.refresh_slot.release()
    _join_refreshes()
    clock.now = 3.5
    # 600's entry comes from its t=2 frame: a fresh track step, no request.
    assert [d.label for d in det.detect(_frame(level=60, fx=600.0), None)] == ["paper cup"]
    det.detect_nowait(_frame(level=90, fx=800.0))  # 800 asks again: 700 (waiting since t=1) goes first
    client.refresh_slot.release()
    _join_refreshes()
    det.detect_nowait(_frame(level=90, fx=800.0))  # then 800
    client.refresh_slot.release()
    _join_refreshes()
    assert _background(client) == [(500, 10), (600, 60), (700, 80), (800, 90)]
    assert len(client.requests) == 4


def test_a_waiting_camera_that_gets_a_fresh_entry_stops_waiting(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    client.refresh_slot = threading.Semaphore(0)
    det.detect_nowait(_frame(fx=500.0))  # camera 500 takes the slot
    assert det.detect_nowait(_frame(fx=600.0)) == []  # camera 600 waits
    assert [d.label for d in det.detect(_frame(fx=600.0), None)] == ["paper cup"]  # a synchronous pass stores it
    client.refresh_slot.release()
    _join_refreshes()
    clock.now = 1.0
    det.detect_nowait(_frame(fx=700.0))  # the free slot: 600 needs no refresh any more
    _sent(client, 3)
    assert [fx for fx, _ in _background(client)] == [500, 700]


def test_a_waiting_key_that_backs_off_keeps_its_turn_for_after_the_backoff(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    client.refresh_slot = threading.Semaphore(0)
    det.detect_nowait(_frame(fx=500.0))  # camera 500 takes the slot
    det.detect_nowait(_frame(fx=600.0))  # camera 600 waits from t=0
    client.error = RuntimeError("connection refused")
    assert det.detect(_frame(fx=600.0), None) == []  # its synchronous pass fails: 600 backs off until t=4
    client.error = None
    client.refresh_slot.release()
    _join_refreshes()
    clock.now = 1.0
    det.detect_nowait(_frame(fx=700.0))  # a free slot during 600's backoff: no call for 600, 700 goes
    client.refresh_slot.release()
    _join_refreshes()
    clock.now = 4.0  # refresh_s after the failure: 600 (waiting since t=0) goes before 800
    det.detect_nowait(_frame(fx=800.0))
    client.refresh_slot.release()
    _join_refreshes()
    assert [fx for fx, _ in _background(client)] == [500, 700, 600]


def test_a_key_asking_again_while_its_own_refresh_runs_is_not_queued_twice(make):
    # Its request is the one being served: after a failed refresh the key asks
    # again on its own next tick, behind the keys that waited meanwhile.
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.refresh_slot = threading.Semaphore(0)
    client.error = RuntimeError("connection refused")
    det.detect_nowait(_frame(fx=500.0))  # 500's refresh is in flight; it will fail
    det.detect_nowait(_frame(fx=500.0))  # 500's next tick: nothing new is queued
    clock.now = 1.0
    det.detect_nowait(_frame(fx=600.0))  # 600 waits from t=1
    client.refresh_slot.release()
    _join_refreshes()  # 500 fails at t=1: it backs off until t=5
    client.error = None
    clock.now = 5.0
    det.detect_nowait(_frame(fx=700.0))  # a free slot: 600 has waited longest
    client.refresh_slot.release()
    _join_refreshes()
    assert [fx for fx, _ in _background(client)] == [500, 600]


def test_close_drops_the_waiting_requests(make):
    det, client, _ = make()
    client.refresh_slot = threading.Semaphore(0)
    det.detect_nowait(_frame(fx=500.0))
    det.detect_nowait(_frame(fx=600.0))  # waits, holding its frame
    assert list(det._waiting) == [((H, W, 600, W // 2), None)]
    client.refresh_slot.release()
    det.close()
    assert det._waiting == {}


# --- an entry past max_cache_age_s serves while its refresh runs (Ruling 35) ---
#
# idle_refresh_s = max_cache_age_s = 30 s: a describe_scene landing while the
# watcher's idle refresh runs found the entry too old and asked the brain
# synchronously. While the scene is unchanged and that key's background refresh
# is in flight or waiting for the slot, the synchronous path tracks the cached
# entry instead. max_cache_age_s stays 30.

VOCAB7 = ["red cube", "banana", "disc", "box", "bowl", "cup", "green cube"]


@pytest.mark.parametrize("pending", ["in flight", "waiting for the slot"])
def test_an_inventory_past_max_age_tracks_while_its_refresh_is_pending(make, pending):
    clock = Clock()
    det, client, seg = make(clock=clock)
    client.inventory = _reply(CUP)
    det.detect(_frame(fx=500.0), None)  # the entry dates from t=0
    client.refresh_slot = threading.Semaphore(0)
    clock.now = 50.0  # past max_cache_age_s
    if pending == "waiting for the slot":
        det.detect_nowait(_frame(level=40, fx=600.0))  # another camera's refresh holds the slot
    det.detect_nowait(_frame(level=3, fx=500.0))  # the watcher's idle refresh for this key
    assert client.entered.wait(5)
    frame = _frame(level=3, fx=500.0)
    assert [d.label for d in det.detect(frame, None)] == ["paper cup"]
    assert seg.calls[-1][0] is frame  # a track step on THIS frame
    assert len(_foreground(client)) == 1, "a synchronous brain call while the key's refresh was pending"
    assert len(_brain_threads()) == 1


@pytest.mark.parametrize("case", ["nothing in flight", "another camera in flight", "changed scene"])
def test_an_inventory_past_max_age_asks_synchronously_without_a_pending_refresh(make, case):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    det.detect(_frame(fx=500.0), None)
    client.refresh_slot = threading.Semaphore(0)
    clock.now = 50.0
    if case == "another camera in flight":
        det.detect_nowait(_frame(level=40, fx=600.0))
    elif case == "changed scene":
        det.detect_nowait(_frame(level=3, fx=500.0))  # this key's refresh is in flight, for the old scene
    client.inventory = _reply(BALL)
    level = 40 if case == "changed scene" else 3
    assert [d.label for d in det.detect(_frame(level=level, fx=500.0), None)] == ["red ball"]
    assert len(_foreground(client)) == 2


def test_a_pending_track_step_hands_a_free_slot_to_its_waiting_refresh(make):
    # No watcher tick comes (perception loop off): the synchronous caller's own
    # track step must not leave its waiting refresh queued forever.
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    det.detect(_frame(fx=500.0), None)
    client.refresh_slot = threading.Semaphore(0)
    clock.now = 50.0
    det.detect_nowait(_frame(level=40, fx=600.0))  # 600 holds the slot
    det.detect_nowait(_frame(level=3, fx=500.0))  # 500's refresh waits
    client.refresh_slot.release()
    _join_refreshes()  # the slot is free; nobody ticks
    clock.now = 51.0
    assert [d.label for d in det.detect(_frame(level=3, fx=500.0), None)] == ["paper cup"]
    _sent(client, 3)
    assert [fx for fx, _ in _background(client)] == [600, 500]
    assert len(_foreground(client)) == 1


@pytest.mark.parametrize("in_flight", [True, False])
def test_a_vocabulary_past_max_age_tracks_while_its_refresh_is_in_flight(make, in_flight):
    clock = Clock()
    det, client, _ = make(clock=clock)
    for q in VOCAB7:
        client.refer[q] = _reply((q, CUP[1]))
    det.detect_vocabulary(_frame(), VOCAB7)  # 7 synchronous refers at t=0
    client.refresh_slot = threading.Semaphore(0)
    clock.now = 61.0  # past max_cache_age_s
    if in_flight:
        det.detect_nowait(_frame(level=3), VOCAB7)  # the watcher's idle refresh: its first refer hangs
        assert client.entered.wait(5)
    assert [d.label for d in det.detect_vocabulary(_frame(level=3), VOCAB7)] == VOCAB7
    # In flight: no synchronous request. Without one: a synchronous pass, as before.
    assert len(_foreground(client)) == (7 if in_flight else 14)
