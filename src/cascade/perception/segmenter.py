"""Learned segmentation behind the pixel-addressing interface.

`pixel_target.segment_at_pixel` grows a region by depth connectivity, which is
free and needs no weights, but it cannot separate an object from the surface it
rests on: the depth step across the contact line is smaller than any usable
threshold. Measured on a LIBERO agentview frame, probing a bowl on a table
filled 10000 px (the cap), produced a 102 cm extent, and put the centre 34 cm
from the true body pose.

A learned segmenter separates them by appearance, which is exactly the
information depth alone lacks. Same frame, same pixels, same scoring:

    object                    depth fill      SAM2.1 point prompt
    akita_black_bowl_2_main   FAIL (34 cm)    3.0 cm   (612 px,  11.2 cm extent)
    plate_1_main              FAIL (22 cm)    0.9 cm   (1356 px, 13.4 cm extent)

This is the pattern ASPIRE, VoLo and RPent all use: a segmenter invoked ON
DEMAND at a point the agent chose, rather than a detector vocabulary that must
already contain the object's class. RPent's `segment` tool degrades to image
inspection when its service is absent; the same principle applies here, so a
missing or broken segmenter falls back to depth connectivity rather than
failing the grasp.

On model choice, measured rather than assumed. Once access to `facebook/sam3`
was granted, both were run through this same interface on the same LIBERO
frame, same pixels, same scoring:

    model              object                    error     extent    latency
    SAM2.1-t (74 MB)   akita_black_bowl_2_main   3.0 cm    11.2 cm     60 ms
    SAM2.1-t (74 MB)   plate_1_main              0.9 cm    13.4 cm     60 ms
    SAM3 (3.29 GB)     akita_black_bowl_2_main   3.0 cm    11.0 cm    982 ms
    SAM3 (3.29 GB)     plate_1_main              1.1 cm    13.1 cm    982 ms

SAM3 is not more accurate here (3.0 vs 3.0 cm on the bowl, 1.1 vs 0.9 cm on
the plate, both far inside the 6 cm grasp tolerance) and it is 16x slower and
45x larger. On a booth where a visitor waits for the arm to move, a second of
segmentation per grasp is the whole interaction budget.

That is a result about THIS task, not about the models: point-prompted
segmentation of a well-separated tabletop object is easy, and SAM3's advantage
is concept prompting and video tracking, which nothing here exercises yet. If
text-prompted segmentation replaces the detector vocabulary later, SAM3 earns
its size; for point prompts it does not. Both are selectable via
`segmenter.model`.

SAM3 is licensed under Meta's SAM License, which is redistributable and has no
commercial restriction, but section 1.b.ii requires acknowledging SAM in any
published research that uses it. Any paper using this path must cite it.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field

import numpy as np

from ..types import Frame, ObjectFix, SkillError
from .pixel_target import MIN_REGION_PX, fix_from_pixel, fix_from_mask

logger = logging.getLogger(__name__)

#: Default checkpoint. `sam2.1_t` is the tiny variant (74 MB): the smallest
#: that still separates tabletop objects, chosen because this runs on the
#: interactive path where latency is visible to a booth visitor.
DEFAULT_MODEL = "sam2.1_t.pt"

#: Box-prompted default: SAM 2.1-L (Apache-2.0, no gated download). The demo
#: passes its `segmenter.model` to both segmenters so they share one copy.
BOX_MODEL = "sam2.1_l.pt"


def configured_segmenter(scfg) -> tuple[str, str]:
    """(model path, device) of a `segmenter:` config block (dict or Cfg).

    Every segmenter built from the block -- the click `PointSegmenter` and
    the brain detector's `BoxSegmenter` -- takes exactly these strings: the
    shared model is keyed by the path string, and a second device string
    would make ultralytics set the shared model up again.
    """
    return str(scfg.get("model") or DEFAULT_MODEL), str(scfg.get("device") or "auto")


@dataclass
class _SharedSam:
    """One loaded SAM and the lock that serializes every call into it:
    ultralytics predict is not re-entrant (prompts are predictor state)."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    model: object | None = None


#: Process-wide, keyed by checkpoint path: a point and a box segmenter on the
#: same checkpoint share one GPU copy and one lock.
_SHARED: dict[str, _SharedSam] = {}
_SHARED_GUARD = threading.Lock()


def _shared(model_path: str) -> _SharedSam:
    with _SHARED_GUARD:
        entry = _SHARED.get(model_path)
        if entry is None:
            entry = _SHARED[model_path] = _SharedSam()
        return entry


class _SamSegmenter:
    """Lazy ultralytics SAM model and device, shared by the prompt styles."""

    def __init__(self, model_path: str, device: str = "auto"):
        self._model_path = model_path
        self._device_spec = device
        self._device: str | None = None
        self._model = None
        self._shared = _shared(model_path)

    @property
    def device(self) -> str:
        """Resolved device. Deferred like the weights: probing torch costs
        seconds, and a booth that never uses pixel addressing should not pay
        for it at startup."""
        if self._device is None:
            from ..device import resolve_device

            self._device = resolve_device(self._device_spec, what="segmenter")
        return self._device

    def _load(self):
        if self._model is None:
            with self._shared.lock:
                if self._shared.model is None:
                    from ultralytics import SAM  # lazy heavy import

                    self._shared.model = SAM(self._model_path)
            self._model = self._shared.model
        return self._model

    def _predict(self, frame: Frame, **prompts):
        """One model call, serialized with every other user of this model."""
        model = self._load()
        with self._shared.lock:
            return model(frame.rgb, device=self.device, verbose=False, **prompts)


def _frame_mask(data, shape: tuple[int, int]):
    """One SAM mask tensor as a full-frame bool mask.

    Strict GPU mode keeps it a CUDA tensor; otherwise it becomes a numpy
    array at the frame's resolution.
    """
    if os.environ.get("CASCADE_REQUIRE_CUDA", "0") == "1":
        if data.device.type != "cuda":
            raise RuntimeError("GPU segmenter prediction did not return CUDA tensors; CPU fallback is forbidden")
        from .cuda_math import segment_mask
        return segment_mask(data, shape)

    mask = data.cpu().numpy().astype(bool)
    h, w = shape
    if mask.shape != (h, w):
        # Ultralytics may return the mask at the model's working
        # resolution; nearest-neighbour keeps it boolean.
        import cv2

        mask = cv2.resize(mask.astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask


class PointSegmenter(_SamSegmenter):
    """Segment the object under a pixel, with a learned model.

    Lazily loads on first use: a booth that never invokes pixel addressing
    should not pay for the weights.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL, device: str = "auto"):
        super().__init__(model_path, device)

    def mask_at(self, frame: Frame, u: int, v: int) -> np.ndarray:
        """Boolean mask of the object at (u, v), full frame resolution."""
        res = self._predict(frame, points=[[int(u), int(v)]], labels=[1])
        if not res or res[0].masks is None or len(res[0].masks.data) == 0:
            raise SkillError(f"segmenter returned no mask at ({u}, {v})")
        return _frame_mask(res[0].masks.data[0], frame.rgb.shape[:2])


class BoxSegmenter(_SamSegmenter):
    """One full-frame mask per box, from a single box-prompted SAM pass.

    The brain detector returns labelled boxes; the observed-finger gate,
    belief identity and grasp clouds need same-frame masks, so every box of a
    frame is segmented in one model call.
    """

    def __init__(self, model_path: str = BOX_MODEL, device: str = "auto"):
        super().__init__(model_path, device)

    def masks(self, frame: Frame, boxes: np.ndarray) -> list[np.ndarray | None]:
        """Masks for (N, 4) xyxy pixel `boxes`, in box order.

        An entry is None where SAM returned no mask for that box or an empty
        one. No boxes means no model call (and no model load).
        """
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        n = len(boxes)
        if n == 0:
            return []
        res = self._predict(frame, bboxes=boxes.tolist())
        out: list[np.ndarray | None] = [None] * n
        if not res or res[0].masks is None:
            return out
        data = res[0].masks.data
        # ultralytics drops low-score masks; each survivor's prompt index
        # rides in `boxes.cls`, so a dropped box cannot shift the rest.
        cls = getattr(getattr(res[0], "boxes", None), "cls", None)
        prompts = [int(i) for i in cls.tolist()] if cls is not None else range(len(data))
        shape = frame.rgb.shape[:2]
        for k, box_index in zip(range(len(data)), prompts):
            if 0 <= box_index < n:
                mask = _frame_mask(data[k], shape)
                out[box_index] = mask if bool(mask.any()) else None
        return out


def fix_at_pixel(
    frame: Frame,
    T_cam2base: np.ndarray,
    u: int,
    v: int,
    segmenter: PointSegmenter | None = None,
    label: str = "object at pixel",
) -> ObjectFix:
    """Lift a pixel to an ObjectFix, preferring the learned segmenter.

    Falls back to depth connectivity when the segmenter is unavailable or
    returns nothing usable. RPent's `segment` tool takes the same stance: a
    missing service degrades to a cheaper path instead of failing the task.

    The fallback is genuinely worse (it cannot separate touching objects) and
    `fix_from_pixel` raises rather than returning a wrong pose when its fill
    escapes, so a degraded run fails honestly instead of grasping into a table.
    """
    import os
    require_cuda = os.environ.get("CASCADE_REQUIRE_CUDA", "0") == "1"
    if segmenter is None and require_cuda:
        raise SkillError("GPU pixel perception requires its CUDA segmenter; depth fallback is forbidden")
    if segmenter is not None:
        try:
            mask = segmenter.mask_at(frame, u, v)
            if require_cuda:
                # The shared CUDA geometry path checks finite depth and the
                # unchanged minimum region size without copying a mask to CPU.
                return fix_from_mask(frame, T_cam2base, mask, label=label)
            valid = int((mask & np.isfinite(frame.depth_m)
                         & (frame.depth_m > 0)).sum()) if frame.depth_m is not None else 0
            if valid >= MIN_REGION_PX:
                return fix_from_mask(frame, T_cam2base, mask, label=label)
            if require_cuda:
                raise SkillError(f"GPU segmenter mask has only {valid} valid-depth pixels; depth fallback is forbidden")
            logger.warning(
                "segmenter mask at (%s, %s) has only %d valid-depth pixels; "
                "falling back to depth connectivity", u, v, valid,
            )
        except SkillError:
            raise
        except Exception as e:  # noqa: BLE001 - never let a model break a grasp
            if require_cuda:
                raise
            logger.warning("segmenter unavailable (%s); falling back to depth", e)

    return fix_from_pixel(frame, T_cam2base, u, v, label=label)
