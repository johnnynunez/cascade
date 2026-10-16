"""Brain-driven perception: the local vision LLM (the brain) boxes objects, SAM masks them.

This module holds the two prompts measured in the perception bake-off, the parser
that turns the brain's text reply into labelled pixel boxes, and `BrainDetector`,
which asks the brain for those boxes and segments them with a box-prompted SAM.

Wire format: the brain answers a JSON list `[{"label": ..., "bbox_2d": [x1, y1, x2, y2]}]`
with coordinates normalized to 0-1000 of the full image.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import re
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import cv2
import numpy as np

from ..types import Detection, Frame
from .detector import Detector
from .grounding import host_crop as _host_crop
from .grounding import mask_box as _mask_box
from .thread_join import join_thread

if TYPE_CHECKING:
    from .segmenter import BoxSegmenter

logger = logging.getLogger(__name__)

_EXCLUDE = ("Exclude the robot arm and its gripper, people and anything in their hands, screens, monitors, "
            "laptops, keyboards, furniture, walls and the table itself.")

# Open-world inventory: every manipulable object on the table, one entry per instance.
INVENTORY_PROMPT = (
    "You are the perception module of a tabletop robot arm. List every manipulable object resting on the "
    "table or inside a container on the table, one entry per instance. " + _EXCLUDE + " Give each a short "
    "specific noun phrase (e.g. 'red plush lobster', 'green filament spool', 'paper cup', 'gray plastic "
    "basket') and a tight bounding box. Answer ONLY with JSON: [{\"label\": \"...\", \"bbox_2d\": [x1, y1, "
    "x2, y2]}] with coordinates normalized to 0-1000. Answer [] if there is none.")

# Referring query for one named object; format with q=<the agent's words>.
REFER_PROMPT = (
    "Locate every instance of \"{q}\" in the image. Answer ONLY with JSON: [{{\"label\": \"{q}\", "
    "\"bbox_2d\": [x1, y1, x2, y2]}}] with coordinates normalized to 0-1000, or [] if it is not visible.")

_THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?")


def _answer_list(text: str) -> list:
    """The FIRST JSON list in `text` that holds at least one dict with `bbox_2d`.

    The brain sometimes appends a second `[]` after its answer, and lists without
    boxes (an empty answer, a list of names) are not the answer, so keep scanning.
    """
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\[", text):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except ValueError:
            # JSONDecodeError, or an int literal past Python's 4300-digit
            # conversion limit (a plain ValueError): not a list here.
            continue
        except RecursionError:
            # Nested deeper than the decoder recurses: garbage, and every later
            # '[' of the same run would fail the same way.
            return []
        if isinstance(value, list) and any(isinstance(o, dict) and "bbox_2d" in o for o in value):
            return value
    return []


def _is_coord(v) -> bool:
    # JSON ints are always finite; math.isfinite would overflow on huge ones.
    if isinstance(v, bool):
        return False
    return isinstance(v, int) or (isinstance(v, float) and math.isfinite(v))


def _reply_body(text: str) -> str:
    """The reply without thinking blocks (closed, unterminated, or only closed
    because the template opened them) and code fences."""
    text = _THINK_BLOCK.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    # raw_decode starts at '[' anyway; fences only matter to the opening-[] check.
    return _FENCE.sub("", text)


def _opens_with_empty_list(body: str) -> bool:
    """The reply's first JSON value is an empty list, whatever text follows it
    (Qwen's trailing "[]", or a remark such as "No objects are visible.")."""
    try:
        value, _ = json.JSONDecoder().raw_decode(body.strip())
    except (ValueError, RecursionError):  # JSONDecodeError, or a >4300-digit int
        return False
    return isinstance(value, list) and not value


def _is_answer(text: str) -> bool:
    """A reply holding a box list (the first one wins), or one that opens with
    an empty list: the two shapes both prompts ask for. Anything else (prose,
    a list of names) is garbled."""
    body = _reply_body(text)
    return bool(_answer_list(body)) or _opens_with_empty_list(body)


def parse_boxes(text: str, width: int, height: int) -> list[tuple[str, np.ndarray]]:
    """Brain reply -> [(label, float32 xyxy pixel box)].

    Thinking blocks and code fences are stripped first. Coordinates are clamped to
    0..1000, corners ordered, then scaled to pixels; entries without a label or
    with a malformed or zero-area box are dropped. Never raises on any string.
    """
    out: list[tuple[str, np.ndarray]] = []
    for item in _answer_list(_reply_body(text)):
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        box = item.get("bbox_2d")
        if not isinstance(label, str) or not label.strip():
            continue
        if not isinstance(box, list) or len(box) != 4 or not all(_is_coord(v) for v in box):
            continue
        # Clamp before float(): an int beyond float range compares fine but cannot convert.
        x1, y1, x2, y2 = (float(min(max(v, 0), 1000)) for v in box)
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 == x1 or y2 == y1:
            continue
        sx, sy = width / 1000.0, height / 1000.0
        out.append((label.strip(), np.array([x1 * sx, y1 * sy, x2 * sx, y2 * sy], dtype=np.float32)))
    return out


#: JPEG quality of the image sent to the brain (same as the old VLM detector).
_JPEG_QUALITY = 85
#: Scene thumbnail (width, height): grayscale, compared by mean absolute difference.
_THUMB_SIZE = (64, 36)
#: At most one failure warning per detector in this many seconds.
_WARN_EVERY_S = 30.0
#: close() waits this long beyond the request timeout for a refresh to finish.
_CLOSE_MARGIN_S = 5.0
#: close() waits this long for the SAM warm-up (weights load + the first,
#: cold CUDA predict): the same minute thread_join.EXIT_JOIN_S allows the
#: slowest cold tick measured. Past it, shutdown names the thread instead.
_WARM_UP_JOIN_S = 60.0
#: reanchor(): the local image check compares the brain's box grown by this
#: fraction of its width and height (half on each side).
_REANCHOR_BOX_GROWTH = 0.10
#: reanchor(): SAM's prompt on the fresh frame (and on the named frame for
#: the mask it is compared with: reanchor_reference) is the brain's box grown
#: by this fraction (half on each side), at least _REANCHOR_PROMPT_MIN_PX on
#: each side, and a fresh mask must not newly reach its border. A mask clipped
#: by a stale prompt always touches it; the margin (8 px on an 80 px box)
#: keeps a still object, SAM jitter and a box a pixel tight off it.
_REANCHOR_PROMPT_GROWTH = 0.20
#: ...the floor of that margin: on a 24x18 px object 20 % is 2 px, which a
#: still object's mask reaches with 2 px of SAM jitter or a brain box 2 px
#: tight. Neighbours the wider prompt takes in are dropped by components.
_REANCHOR_PROMPT_MIN_PX = 6.0
#: reanchor(): the new mask's bounding box must overlap the brain's box by at
#: least this IoU, or SAM masked something else than the brain's box.
REANCHOR_MIN_IOU = 0.5
#: reanchor(): of the new mask, only the connected components that come
#: within this many pixels of the old mask (the old mask dilated by it) are
#: the object; the rest (a neighbour inside SAM's grown prompt) is dropped
#: before the border, agreement and geometry see the mask.
_REANCHOR_COMPONENT_REACH_PX = 2.0
#: reanchor(): the two masks agree when at least `reanchor_mask_agreement` of
#: each one's pixels lie within this tolerance of the other: max(the floor,
#: the fraction x the diagonal of the old mask's box) pixels. SAM's 1-2 px
#: edge jitter and a brain box a few px tight stay inside it on any size.
_REANCHOR_TOL_MIN_PX = 2.0
_REANCHOR_TOL_FRAC = 0.03


class _Entry(NamedTuple):
    """One cache entry (a camera's inventory, or its configured vocabulary
    filled by refer calls): the brain's boxes and the frame they came from."""

    labels: tuple[str, ...]
    boxes: np.ndarray  # (N, 4) float32 xyxy pixels, read-only
    thumb: np.ndarray  # (36, 64) uint8 grayscale thumbnail
    t: float  # clock time of the frame the brain saw


class _Request(NamedTuple):
    """A background refresh waiting for the detector's single slot."""

    since: float  # clock time of the key's first ask while it waits: its place in the queue
    frame: Frame  # the key's latest frame: the refresh asks the brain about this one
    thumb: np.ndarray
    t: float  # clock time of that latest ask


class ReanchorResult(NamedTuple):
    """What `BrainDetector.reanchor` found on the newer frame.

    `detection` is the detection to measure on the newer frame, or None:
    then `scene_changed` says whether the image changed around it (the
    brain's box no longer describes this frame) or no usable mask was found,
    and `reason` says which. `path` says how it was found: "still" (the
    region's colours did not change: the named detection itself, its mask
    and box), "sam" (SAM's mask on the newer frame and that mask's bounding
    box; with no detection and no refusal when the caller asked for no SAM
    call: segment=False), "" (refused before either). The two diffs are
    always measured (mean absolute grey levels of the thumbnails; the
    largest B/G/R mean absolute difference in the grown box). Evidence once
    known (SAM path): `prompt_box`, the grown box SAM was prompted with (xyxy
    pixels); `mask_iou`, the pixel IoU of the newer mask with the
    same-prompt mask of the image the object was named on (its re-mask,
    `ReanchorReference`); `mask_agreement`, the smaller of the two shares of
    each mask's pixels within the size-aware tolerance of the other;
    `dropped_components`, how many separate parts of SAM's newer mask
    (neighbours) were dropped before any of that."""

    detection: Detection | None
    scene_changed: bool
    reason: str
    scene_diff_global: float
    scene_diff_local: float
    prompt_box: tuple[int, int, int, int] | None = None
    mask_iou: float = math.nan
    mask_agreement: float = math.nan
    dropped_components: int = 0
    path: str = ""


class ReanchorReference(NamedTuple):
    """What `BrainDetector.reanchor_reference` made of the image the object
    was named on: `detection` is the named detection with SAM's mask for the
    grown prompt reanchor() gives the fresh image (only its parts that
    overlap the naming mask, and of those only the pixels within the
    tolerance of it) and that mask's box, or None with `reason` (a failed
    re-anchor). `prompt_box` is that prompt (xyxy pixels); `coverage` the
    share of the naming mask's pixels within the tolerance of the re-mask."""

    detection: Detection | None
    reason: str
    prompt_box: tuple[int, int, int, int] | None = None
    coverage: float = math.nan


def _camera_key(frame: Frame) -> tuple[int, int, int, int]:
    """Frame carries no camera name; image size + intrinsics tell scene from wrist."""
    h, w = frame.rgb.shape[:2]
    return h, w, round(float(frame.K[0, 0])), round(float(frame.K[0, 2]))


def _cache_key(frame: Frame, queries: tuple[str, ...] | None):
    """(camera key, queries): None is the camera's open-world inventory; a
    tuple of refer queries is a vocabulary entry, which detect_nowait and
    detect_vocabulary share (an agent's detect(frame, [q]) never reads it)."""
    return _camera_key(frame), queries


def _thumbnail(frame: Frame) -> np.ndarray:
    gray = cv2.cvtColor(frame.rgb, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, _THUMB_SIZE, interpolation=cv2.INTER_AREA)


def _scene_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def _grown_box(box, shape, growth: float = _REANCHOR_BOX_GROWTH, min_px: float = 0.0) -> tuple[int, int, int, int]:
    """`box` (xyxy pixels) grown by `growth` of its width and height (half on
    each side, at least `min_px` on each side), outward to whole pixels,
    clipped to an image of `shape` (h, w, ...): integer xyxy, the end
    exclusive."""
    h, w = shape[:2]
    x0, y0, x1, y1 = (float(v) for v in np.asarray(box, dtype=np.float64).reshape(-1)[:4])
    gx, gy = max((x1 - x0) * growth / 2, min_px), max((y1 - y0) * growth / 2, min_px)
    return (max(int(math.floor(x0 - gx)), 0), max(int(math.floor(y0 - gy)), 0),
            min(int(math.ceil(x1 + gx)), w), min(int(math.ceil(y1 + gy)), h))


def _reanchor_prompt(box, shape) -> tuple[int, int, int, int]:
    """SAM's prompt for a re-anchor, on the fresh frame and on the named one
    alike: `box` grown by _REANCHOR_PROMPT_GROWTH, at least
    _REANCHOR_PROMPT_MIN_PX a side."""
    return _grown_box(box, shape, _REANCHOR_PROMPT_GROWTH, _REANCHOR_PROMPT_MIN_PX)


def _local_diff(a: Frame, b: Frame, box) -> float:
    """The largest of the B, G and R mean absolute differences of two
    same-size frames inside `box` (xyxy pixels) grown by
    _REANCHOR_BOX_GROWTH; inf when nothing of it is inside the image.
    Colour, not grey: an object as bright as the table moves unseen in grey."""
    xa, ya, xb, yb = _grown_box(box, a.rgb.shape)
    if xb <= xa or yb <= ya:
        return math.inf
    diff = np.abs(a.rgb[ya:yb, xa:xb].astype(np.float32) - b.rgb[ya:yb, xa:xb].astype(np.float32))
    return float(diff.reshape(-1, diff.shape[2] if diff.ndim == 3 else 1).mean(axis=0).max())


class _MaskMatch(NamedTuple):
    """`_match_masks`: the newer mask's kept components (`kept`, host bools on
    the crop at `origin`, None when no component is near the old mask) and
    how the kept mask compares with the old one."""

    kept: np.ndarray | None
    origin: tuple[int, int]
    dropped: int
    agreement: float
    iou: float
    tol_px: float
    new_near: float = math.nan  # share of the kept mask's pixels within tol_px of `old`
    old_near: float = math.nan  # share of `old`'s pixels within tol_px of the kept mask
    trimmed: int = 0  # pixels of the kept components cut off farther than tol_px from `old` (clip=True)
    whole_box: np.ndarray | None = None  # the kept components' box before that cut (frame xyxy)


def _match_masks(old, old_box, new, new_box, *, reach_px: float = _REANCHOR_COMPONENT_REACH_PX,
                 clip: bool = False) -> _MaskMatch:
    """Compare an older mask `old` with a newer mask `new` (same size,
    opaque, non-empty, with their boxes) on the crop covering both, at full
    resolution: one host copy per mask, sliced on the device for a CUDA
    tensor.

    First the component filter: of `new`, only the 8-connected components
    with a pixel within `reach_px` of `old` are kept (0: overlapping it; by
    default _REANCHOR_COMPONENT_REACH_PX: a still neighbour inside SAM's
    grown prompt is a separate component; a neighbour that moved into
    contact is part of the object's). With `clip`, the kept mask also loses
    every pixel farther than `tol_px` from `old` (`old` dilated by the
    tolerance: a neighbour merged with the object never reaches past it).
    Then, for the kept mask: `new_near`, the share of its pixels within
    `tol_px` of `old`; `old_near`, the share of `old`'s pixels within
    `tol_px` of it; `agreement`, the smaller of the two, with tol_px =
    max(_REANCHOR_TOL_MIN_PX, _REANCHOR_TOL_FRAC x the diagonal of `old`'s
    box); `iou`, their raw pixel IoU (evidence)."""
    x0, y0 = int(min(old_box[0], new_box[0])), int(min(old_box[1], new_box[1]))
    x1, y1 = int(max(old_box[2], new_box[2])), int(max(old_box[3], new_box[3]))
    a, b = _host_crop(old, x0, y0, x1, y1), _host_crop(new, x0, y0, x1, y1)
    tol_px = max(_REANCHOR_TOL_MIN_PX, _REANCHOR_TOL_FRAC * math.hypot(old_box[2] - old_box[0],
                                                                        old_box[3] - old_box[1]))
    # Euclidean distance of every crop pixel to the nearest pixel of `old`.
    to_old = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    count, labels = cv2.connectedComponents(b.astype(np.uint8), connectivity=8)
    near = np.unique(labels[b & (to_old <= reach_px)])
    near = near[near != 0]
    dropped = count - 1 - near.size
    if near.size == 0:
        return _MaskMatch(None, (x0, y0), dropped, math.nan, math.nan, tol_px)
    kept = np.isin(labels, near)
    trimmed = 0
    whole_box = _mask_box(kept) + np.array([x0, y0, x0, y0], dtype=np.float64)
    if clip:
        far = kept & (to_old > tol_px)
        trimmed = int(np.count_nonzero(far))
        kept &= ~far
        if not kept.any():  # unreachable with reach_px <= tol_px: a kept component comes within it
            return _MaskMatch(None, (x0, y0), dropped + near.size, math.nan, math.nan, tol_px, trimmed=trimmed,
                              whole_box=whole_box)
    to_kept = cv2.distanceTransform((~kept).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    new_near = np.count_nonzero(to_old[kept] <= tol_px) / np.count_nonzero(kept)
    old_near = np.count_nonzero(to_kept[a] <= tol_px) / np.count_nonzero(a)
    iou = np.count_nonzero(a & kept) / np.count_nonzero(a | kept)
    return _MaskMatch(kept, (x0, y0), dropped, float(min(new_near, old_near)), float(iou), tol_px,
                      float(new_near), float(old_near), trimmed, whole_box)


def _kept_mask(new, match: _MaskMatch):
    """The newer mask with only the kept components (and pixels): `new`
    itself (still opaque) when nothing was dropped or trimmed, else a host
    bool mask of its size."""
    if not match.dropped and not match.trimmed:
        return new
    out = np.zeros(tuple(new.shape), dtype=bool)
    x0, y0 = match.origin
    h, w = match.kept.shape
    out[y0:y0 + h, x0:x0 + w] = match.kept
    return out


def _touched_sides(mask_box, grown) -> set[str]:
    """The sides of the integer box `grown` that a mask with box `mask_box`
    reaches (or crosses)."""
    xa, ya, xb, yb = grown
    sides = set()
    if mask_box[0] <= xa:
        sides.add("left")
    if mask_box[1] <= ya:
        sides.add("top")
    if mask_box[2] >= xb:
        sides.add("right")
    if mask_box[3] >= yb:
        sides.add("bottom")
    return sides


def _box_iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = (float(v) for v in np.asarray(a, dtype=np.float64).reshape(-1)[:4])
    bx0, by0, bx1, by1 = (float(v) for v in np.asarray(b, dtype=np.float64).reshape(-1)[:4])
    iw, ih = min(ax1, bx1) - max(ax0, bx0), min(ay1, by1) - max(ay0, by0)
    inter = max(iw, 0.0) * max(ih, 0.0)
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def _jpeg_data_url(frame: Frame) -> str:
    # Frame.rgb is BGR (OpenCV) despite the name and imencode expects BGR:
    # no channel swap, or the brain sees wrong colours.
    ok, buf = cv2.imencode(".jpg", frame.rgb, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding of the frame failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


class BrainDetector(Detector):
    """Detector backed by the brain (an OpenAI-compatible vision LLM) plus SAM masks.

    `detect(frame, [q, ...])` (refer: an agent's own query) asks the brain
    for each query synchronously and never uses a cache; labels are the
    queries. `detect(frame, None)` (inventory) and
    `detect_vocabulary(frame, classes)` (the configured default vocabulary:
    one refer per class, labels = the classes) keep one cache entry per
    (camera, None | tuple(classes)): while the entry is younger than
    `max_cache_age_s` and the scene thumbnail moved by at most `scene_change`,
    they return a track step (SAM on the cached boxes in THIS frame) and, once
    the entry is older than `refresh_s`, start a background refresh.
    Otherwise (no entry, scene changed, too old) they ask the brain
    synchronously and store the answer. One exception (Ruling 35): an entry
    past `max_cache_age_s` whose scene is unchanged keeps being tracked while
    that key's background refresh is in flight or waiting for the slot.

    Two clocks:
    - `refresh_s` (default 4 s): the synchronous path's freshness threshold
      (an entry older than this gets a background refresh behind its track
      step), and the failure backoff window of every cache key.
    - `idle_refresh_s` (default 30 s): the watcher's cadence. `detect_nowait`
      refreshes an UNCHANGED scene only once its entry is older than this; a
      changed scene or a missing entry starts a refresh at once.

    At most ONE background brain request is in flight per detector, across
    every camera and key (synchronous calls are not capped). A key that asks
    while it runs waits for the slot, from its first ask, holding its latest
    frame; a free slot goes to the key that has waited longest (ties: first
    asked), whichever caller finds it free (Ruling 34). So a camera whose
    scene changes on every watcher tick cannot starve the cameras after it.

    Booth rule: a down/slow/garbled brain yields a rate-limited warning, never
    an exception. A failed refer yields []. A failed synchronous inventory or
    vocabulary pass keeps tracking the known objects: the cached entry's track
    step at any age while this frame still shows its scene, [] once the scene
    changed. After any failed brain call for a cache key, synchronous or
    background, that key makes no brain call on either path for `refresh_s`
    (backoff, same fallback meanwhile); any successful answer for the key,
    synchronous or background, ends it.

    `detect_nowait` (the watcher's call: its tick also sends the 5 s motion
    watchdog's heartbeat) never asks the brain on the caller's thread: it
    answers from the cache and refreshes in the background.

    `reanchor` (two-phase localization) re-finds one detection the brain
    named on an older frame in a newer frame of the same camera, without a
    brain call. Refused as a scene change when the image changed around it
    (`scene_change` over the whole view, `reanchor_local_diff` in colour
    inside the box grown by 10 %). A region whose colours did not change
    (at most `reanchor_still_diff`) keeps the named detection, mask and all
    (the still path: no SAM call). Otherwise SAM masks the brain's box grown
    by 20 % (at least 6 px a side); parts of the new mask not connected to
    the old mask's place (a neighbour) are dropped; refused as a scene
    change when the new mask reaches a border of SAM's prompt box the old
    mask did not, or when under `reanchor_mask_agreement` of either mask's
    pixels lie within a few pixels (size-aware) of the other; refused as a
    failed re-anchor when the mask is missing, empty or off the box.

    Safe to call from several threads at once: the cache lock is never held
    across the brain or SAM call, and the segmenter serializes its own model
    calls.
    """

    concurrent_safe = True

    def __init__(self, *, base_url: str, model: str, api_key_file: str | None = None,
                 timeout_s: float = 20.0, max_tokens: int = 1500, segmenter: BoxSegmenter | None = None,
                 refresh_s: float = 4.0, max_cache_age_s: float = 30.0, scene_change: float = 6.0,
                 idle_refresh_s: float = 30.0, reanchor_local_diff: float = 8.0,
                 reanchor_mask_agreement: float = 0.95, reanchor_still_diff: float = 2.0, client=None,
                 clock=time.monotonic):
        self._base_url = str(base_url)
        self._model = str(model)
        self._timeout_s = float(timeout_s)
        self._max_tokens = int(max_tokens)
        self._segmenter = segmenter
        self._refresh_s = float(refresh_s)
        self._idle_refresh_s = float(idle_refresh_s)
        self._max_cache_age_s = float(max_cache_age_s)
        self._scene_change = float(scene_change)
        self._reanchor_local_diff = float(reanchor_local_diff)
        self._reanchor_mask_agreement = float(reanchor_mask_agreement)
        self._reanchor_still_diff = float(reanchor_still_diff)
        self._clock = clock
        self._api_key = self._read_key(api_key_file)
        self._client = client
        self._client_lock = threading.Lock()
        # Guards the cache, the refresh bookkeeping and the warning clock only.
        # Keyed by _cache_key: (camera key, None) is the camera's inventory.
        self._lock = threading.Lock()
        self._cache: dict[tuple, _Entry] = {}
        # The keys with a background refresh in flight: at most one at a time.
        self._refreshing: set[tuple] = set()
        # Refreshes waiting for that slot, in ask order: served oldest-first.
        self._waiting: dict[tuple, _Request] = {}
        # Clock time of the last failed brain call per cache key, synchronous or
        # background: the backoff both paths obey. A success for the key clears it.
        self._failed_at: dict[tuple, float] = {}
        self._threads: set[threading.Thread] = set()
        self._closed = False
        self._last_warn: float | None = None
        # The SAM warm-up on this detector's segmenter. apps.demo starts it
        # and hands it over here: close() joins it and shutdown_runtime names
        # it while it runs (SAM load + a CUDA predict are native code that
        # must not still be running at interpreter exit).
        self.warm_up_thread: threading.Thread | None = None

    def __repr__(self) -> str:
        seg = type(self._segmenter).__name__ if self._segmenter is not None else None
        return f"{type(self).__name__}(base_url={self._base_url!r}, model={self._model!r}, segmenter={seg})"

    @staticmethod
    def _read_key(api_key_file: str | None) -> str:
        if api_key_file is None:
            return "EMPTY"
        path = Path(api_key_file).expanduser()
        try:
            key = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("brain detector: api key file %s is unreadable (%s); sending no key",
                           path, type(e).__name__)
            return "EMPTY"
        if not key:
            logger.warning("brain detector: api key file %s is empty; sending no key", path)
            return "EMPTY"
        return key

    def set_classes(self, classes: list[str] | None) -> None:
        # The brain holds no vocabulary: each detect() call names its own
        # queries, and classes=None is always open-world inventory.
        return None

    def detect(self, frame: Frame, classes: list[str] | None = None) -> list[Detection]:
        try:
            if classes:
                return self._refer(frame, [str(q) for q in classes])
            return self._cached(frame, None)
        except Exception as e:  # noqa: BLE001 - booth rule: degrade, never crash the caller
            self._warn("detection failed; returning no detections", e)
            return []

    def detect_vocabulary(self, frame: Frame, classes: list[str] | None = None) -> list[Detection]:
        """detect() for the configured default vocabulary (`detect_classes`),
        served from the cache entry detect_nowait keeps for the same
        (camera, tuple(classes)), under the inventory's rules: a track step
        while the scene is unchanged (plus a background refresh once the
        entry is older than `refresh_s`; past `max_cache_age_s` only while
        that refresh is pending), else one synchronous refer per class,
        stored. None/[] is the inventory, as in detect(). Never for an
        agent's own query: that stays a fresh detect(frame, [q])."""
        try:
            return self._cached(frame, tuple(str(q) for q in classes) if classes else None)
        except Exception as e:  # noqa: BLE001 - booth rule: degrade, never crash the caller
            self._warn("detection failed; returning no detections", e)
            return []

    def detect_nowait(self, frame: Frame, classes: list[str] | None = None) -> list[Detection]:
        """detect() without ever waiting for the brain, for the watcher thread.

        One cache entry per (camera, classes): None/[] is the inventory entry
        detect(frame, None) shares; a list of queries has its own entry,
        filled by refer calls (labels = the queries) and shared with
        detect_vocabulary. While this frame shows the entry's scene, its track
        step at any age, plus one background refresh once the entry is older
        than `idle_refresh_s`. No entry or a changed scene: [] now, and a
        background refresh asks the brain at once. No refresh starts while
        the key backs off after a failed call; while another refresh runs,
        the request waits for the slot (served oldest-first).
        """
        try:
            queries = tuple(str(q) for q in classes) if classes else None
            key = _cache_key(frame, queries)
            thumb = _thumbnail(frame)
            now = self._clock()
            known = self._known_entry(key, thumb)
            if known is None or now - known.t > self._idle_refresh_s:
                self._start_refresh(key, frame, thumb, now)
            return self._track_known(frame, known)
        except Exception as e:  # noqa: BLE001 - booth rule: degrade, never crash the caller
            self._warn("detection failed; returning no detections", e)
            return []

    def can_reanchor(self) -> bool:
        """reanchor() can mask a box on a new frame: a segmenter is configured."""
        return self._segmenter is not None

    def reanchor_reference(self, frame: Frame, det: Detection) -> ReanchorReference:
        """The mask reanchor() compares a fresh image's mask with: SAM
        re-masks `det` on `frame`, the image the brain named it on, with the
        SAME prompt reanchor() gives the fresh image (the brain's box grown
        20 %, at least _REANCHOR_PROMPT_MIN_PX a side). SAM keeps a mask inside
        its prompt, so a brain box that cuts the object clips the naming mask
        but neither of these two: the checks compare like with like. Call it
        BEFORE taking the fresh image: its SAM call is no part of that image's
        age.

        Of the re-mask only the connected components that overlap the naming
        mask are kept (a neighbour the wider prompt takes in is not the
        object), and of those only the pixels within the tolerance (as
        reanchor's) of the naming mask: a neighbour in contact, which SAM
        merges with the object under the grown prompt, never enters the
        reference, so a fresh mask that still merges it disagrees with it.
        Refused, as a failed re-anchor (one image: nothing moved),
        when no segmenter is configured, `det` has no mask, SAM gives no mask
        or an empty one, no part of it overlaps the naming mask, the kept
        mask's box overlaps the brain's box by IoU < REANCHOR_MIN_IOU, or under
        `reanchor_mask_agreement` of the naming mask's pixels lie within the
        tolerance (as reanchor's) of the kept mask: SAM masked only part of
        the object, and two such masks would agree with each other and
        measure that part. The re-mask may reach past the naming mask by up
        to the tolerance: that ring is what a tight box cut off."""
        where = "on the image the object was named on"
        if self._segmenter is None:
            return ReanchorReference(None, "no segmenter is configured")
        named_box = _mask_box(det.mask) if det.mask is not None else None
        if named_box is None:
            return ReanchorReference(None, f"the object has no mask {where}")
        prompt_box = _reanchor_prompt(det.bbox, frame.rgb.shape)
        prompt = np.asarray(prompt_box, dtype=np.float32).reshape(1, 4)
        found = self._detections(frame, [det.label], prompt, track=True)
        mask = found[0].mask if found else None
        if mask is None:
            return ReanchorReference(None, f"SAM returned no mask for the grown box {where}", prompt_box)
        mask_box = _mask_box(mask)
        if mask_box is None:
            return ReanchorReference(None, f"SAM returned an empty mask for the grown box {where}", prompt_box)
        match = _match_masks(det.mask, named_box, mask, mask_box, reach_px=0.0, clip=True)
        if match.kept is None:
            return ReanchorReference(None, f"SAM's mask lies off the box: no part of it overlaps the object's mask "
                                           f"{where}", prompt_box)
        kept_box = _mask_box(match.kept) + np.array([*match.origin, *match.origin], dtype=np.float64)
        # SAM's own mask (its parts on the object, before the cut to the
        # tolerance) must lie on the brain's box: one leaking far onto the
        # table is no mask of the object, whatever the cut leaves of it.
        iou = _box_iou(match.whole_box, det.bbox)
        if not iou >= REANCHOR_MIN_IOU:
            return ReanchorReference(None, f"SAM's mask lies off the box (IoU {iou:.2f} < {REANCHOR_MIN_IOU:g}) "
                                           f"{where}", prompt_box)
        if not match.old_near >= self._reanchor_mask_agreement:
            return ReanchorReference(None, f"SAM's mask for the grown box covers only {match.old_near:.2f} of the "
                                           f"object's mask {where} (< {self._reanchor_mask_agreement:g} of its "
                                           f"pixels within {match.tol_px:.1f} px)", prompt_box, match.old_near)
        detection = Detection(label=det.label, conf=det.conf, bbox=kept_box.astype(np.float32),
                              mask=_kept_mask(mask, match))
        return ReanchorReference(detection, "", prompt_box, match.old_near)

    def reanchor(self, frame: Frame, det: Detection, *, since: Frame,
                 reference: ReanchorReference | None = None, segment: bool = True) -> ReanchorResult:
        """Re-find `det`, which the brain named on `since` (with SAM's mask
        there), in the newer `frame` of the same camera, without asking the
        brain. `reference` is `reanchor_reference(since, det)`, taken before
        `frame` was (None: made here, its SAM call inside this frame's age).

        Refused as a scene change, with no SAM call, when the thumbnails of the
        two frames differ by more than `scene_change` or, inside the brain's
        box grown by 10 %, the largest B/G/R mean difference exceeds
        `reanchor_local_diff`. Refused as a failed re-anchor without a
        segmenter or when `det` has no mask.

        The still path (path "still"): when that colour difference is at
        most `reanchor_still_diff`, the object's pixels did not change, so
        `det` itself (its naming mask: main's single-frame mask, and its box)
        describes the newer frame; no SAM call, no reference. The caller
        measures it on the newer frame's depth.

        Otherwise the SAM path (path "sam"); with `segment` False it stops
        here, with no detection and no refusal, for the caller to re-mask
        `since` and call again on a frame taken after that. Refused as a
        failed re-anchor when the reference was refused. Otherwise SAM masks
        the brain's box grown by 20 %, at least _REANCHOR_PROMPT_MIN_PX a side,
        on `frame` (the track step): the reference's prompt, so both masks
        are made alike and compared with each other. Of the new mask only the
        connected components within 2 px of the reference mask are kept (a
        still neighbour inside the prompt is dropped before any check and the
        geometry; one that moved into contact is part of the object's
        component). Refused as a failed re-anchor when SAM gives no mask, an
        empty one, none of it near the reference mask, or a kept mask whose
        bounding box overlaps the brain's box by IoU < REANCHOR_MIN_IOU.
        Refused as a scene change when the kept mask reaches a side of the
        prompt box the reference mask did not reach (the object extends past
        the box it was named in; a mask clipped by a stale prompt touches it;
        a neighbour merged with it, which the reference never holds), or when
        under `reanchor_mask_agreement` of its pixels lie within the
        tolerance (max(2 px, 3 % of the reference mask box's diagonal)) of the
        reference mask, or of the reference mask's pixels within it of the
        kept one. The detection returned carries the kept mask and ITS
        bounding box. Never falls back to the box: a box cloud includes table
        pixels. Masks stay opaque (numpy or CUDA tensors): only projections
        and one crop per mask are copied to the host; a mask with dropped
        components comes back as a host bool mask."""
        if frame.rgb.shape != since.rgb.shape:
            return ReanchorResult(None, True, "the image size changed", math.inf, math.inf)
        diff_global = _scene_diff(_thumbnail(since), _thumbnail(frame))
        diff_local = _local_diff(since, frame, det.bbox)
        prompt_box = None
        match = None
        path = ""

        def refused(scene_changed: bool, reason: str) -> ReanchorResult:
            evidence = (match.iou, match.agreement, match.dropped) if match is not None else (math.nan, math.nan, 0)
            return ReanchorResult(None, scene_changed, reason, diff_global, diff_local, prompt_box, *evidence, path)

        if not diff_global <= self._scene_change:
            return refused(True, f"the view changed (mean grey difference {diff_global:.1f} > "
                                 f"{self._scene_change:g})")
        if not diff_local <= self._reanchor_local_diff:
            return refused(True, f"the image around the object changed (mean colour difference "
                                 f"{diff_local:.1f} > {self._reanchor_local_diff:g})")
        if self._segmenter is None:
            return refused(False, "no segmenter is configured")
        if det.mask is None or _mask_box(det.mask) is None:
            return refused(False, "the object has no mask on the image it was named on")
        if diff_local <= self._reanchor_still_diff:
            # The still path: nothing around the object changed colour, so its
            # naming mask (main's single-frame mask) still covers it.
            still = Detection(label=det.label, conf=det.conf, bbox=np.array(det.bbox, dtype=np.float32),
                              mask=det.mask, obb=det.obb)
            return ReanchorResult(still, False, "", diff_global, diff_local, path="still")
        path = "sam"
        if not segment:
            return refused(False, f"the image around the object changed a little (mean colour difference "
                                  f"{diff_local:.1f} > {self._reanchor_still_diff:g}): SAM re-anchors it")
        if reference is None:
            reference = self.reanchor_reference(since, det)
        ref = reference.detection
        if ref is None or ref.mask is None:
            return refused(False, reference.reason or "the object was not re-masked on the image it was named on")
        ref_box = np.asarray(ref.bbox, dtype=np.float64).reshape(-1)[:4]
        prompt_box = _reanchor_prompt(det.bbox, frame.rgb.shape)
        prompt = np.asarray(prompt_box, dtype=np.float32).reshape(1, 4)
        found = self._detections(frame, [det.label], prompt, track=True)
        mask = found[0].mask if found else None
        if mask is None:
            return refused(False, "SAM returned no mask for the box")
        mask_box = _mask_box(mask)
        if mask_box is None:
            return refused(False, "SAM returned an empty mask for the box")
        match = _match_masks(ref.mask, ref_box, mask, mask_box)
        if match.kept is None:
            return refused(False, f"SAM's mask lies off the box: no part of it comes within "
                                  f"{_REANCHOR_COMPONENT_REACH_PX:g} px of the object's mask on the image it "
                                  "was named on")
        mask = _kept_mask(mask, match)
        mask_box = _mask_box(match.kept) + np.array([*match.origin, *match.origin], dtype=np.float64)
        iou = _box_iou(mask_box, det.bbox)
        if not iou >= REANCHOR_MIN_IOU:
            return refused(False, f"SAM's mask lies off the box (IoU {iou:.2f} < {REANCHOR_MIN_IOU:g})")
        crossed = _touched_sides(mask_box, prompt_box) - _touched_sides(ref_box, prompt_box)
        if crossed:
            return refused(True, f"the object's mask now reaches the {'/'.join(sorted(crossed))} border of the "
                                 "box it was named in (grown 20 %, at least "
                                 f"{_REANCHOR_PROMPT_MIN_PX:g} px a side): it extends past that box")
        if not match.agreement >= self._reanchor_mask_agreement:
            return refused(True, f"the object's mask changed (mask agreement {match.agreement:.2f} < "
                                 f"{self._reanchor_mask_agreement:g}: the smaller share of either mask's "
                                 f"pixels within {match.tol_px:.1f} px of the other, with its same-prompt mask on "
                                 f"the image it was named on; pixel IoU {match.iou:.2f})")
        detection = Detection(label=det.label, conf=det.conf, bbox=mask_box.astype(np.float32), mask=mask)
        return ReanchorResult(detection, False, "", diff_global, diff_local, prompt_box, match.iou,
                              match.agreement, match.dropped, path)

    def close(self) -> None:
        """Join the refresh threads in flight and the SAM warm-up, each with
        a bound; later refreshes do not start. A thread past its bound keeps
        running and is reported on stderr; shutdown_runtime lists a warm-up
        still alive in its pending threads."""
        with self._lock:
            self._closed = True
            self._waiting.clear()  # no later refresh; drops their frames
            threads = list(self._threads)
        for thread in threads:
            join_thread(thread, what="brain detector", timeout_s=self._timeout_s + _CLOSE_MARGIN_S)
        join_thread(self.warm_up_thread, what="brain detector", timeout_s=_WARM_UP_JOIN_S)

    # -- brain transport ----------------------------------------------------

    def _get_client(self):
        with self._client_lock:
            if self._client is None:
                from openai import OpenAI  # lazy: CI's minimal install has no openai

                # max_retries=0: the next detect is the retry. openai's default 2
                # would hold a synchronous detect ~3x timeout_s on a hung brain
                # and outlive close()'s join.
                self._client = OpenAI(base_url=self._base_url, api_key=self._api_key,
                                      timeout=self._timeout_s, max_retries=0)
            return self._client

    def _ask(self, image_url: str, prompt: str) -> str:
        resp = self._get_client().chat.completions.create(
            model=self._model,
            temperature=0,
            max_tokens=self._max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            }],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        # A garbled reply is a failed call, never an empty table: cut at
        # max_tokens (even if a list closed before the cut), or neither a box
        # list nor a reply opening with "[]" (what both prompts ask for when
        # nothing is there).
        if getattr(choice, "finish_reason", None) == "length":
            raise ValueError(f"garbled reply, truncated at max_tokens={self._max_tokens}")
        if not _is_answer(text):
            raise ValueError(f"garbled reply, no box list: {text[:80]!r}")
        return text

    # -- refer and inventory ------------------------------------------------

    def _refer(self, frame: Frame, queries: list[str]) -> list[Detection]:
        labels, boxes = self._ask_refer(frame, queries)
        return self._detections(frame, labels, boxes, track=False)

    def _ask_refer(self, frame: Frame, queries) -> tuple[list[str], np.ndarray]:
        """One request per query, concatenated; labels are the queries."""
        h, w = frame.rgb.shape[:2]
        image_url = _jpeg_data_url(frame)
        labels: list[str] = []
        boxes: list[np.ndarray] = []
        for q in queries:
            for _, box in parse_boxes(self._ask(image_url, REFER_PROMPT.format(q=q)), w, h):
                labels.append(q)
                boxes.append(box)
        return labels, np.array(boxes, dtype=np.float32).reshape(-1, 4)

    def _ask_inventory(self, frame: Frame, thumb: np.ndarray, t: float) -> _Entry:
        h, w = frame.rgb.shape[:2]
        found = parse_boxes(self._ask(_jpeg_data_url(frame), INVENTORY_PROMPT), w, h)
        boxes = np.array([box for _, box in found], dtype=np.float32).reshape(-1, 4)
        boxes.setflags(write=False)
        return _Entry(tuple(label for label, _ in found), boxes, thumb, t)

    def _ask_entry(self, key, frame: Frame, thumb: np.ndarray, t: float) -> _Entry:
        """The brain call behind a cache key: the inventory prompt for
        (camera, None), the refer loop for (camera, queries)."""
        queries = key[1]
        if queries is None:
            return self._ask_inventory(frame, thumb, t)
        labels, boxes = self._ask_refer(frame, queries)
        boxes.setflags(write=False)
        return _Entry(tuple(labels), boxes, thumb, t)

    def _store(self, key, entry: _Entry) -> None:
        """A successful answer for a key (a synchronous pass or a refresh):
        the brain is up, so that key's backoff ends; the entry is stored
        unless the cached one comes from a newer frame."""
        with self._lock:
            self._failed_at.pop(key, None)
            # The key no longer needs the refresh it may be waiting for.
            self._waiting.pop(key, None)
            old = self._cache.get(key)
            # A slow refresh must not overwrite an entry from a newer frame.
            if old is None or entry.t >= old.t:
                self._cache[key] = entry

    def _cached(self, frame: Frame, queries: tuple[str, ...] | None) -> list[Detection]:
        """The synchronous cached path: the inventory (queries None) or the
        configured vocabulary (a tuple of refer queries)."""
        key = _cache_key(frame, queries)
        thumb = _thumbnail(frame)
        now = self._clock()
        with self._lock:
            entry = self._cache.get(key)
            failed_at = self._failed_at.get(key)
            pending = key in self._refreshing or key in self._waiting
        same_scene = entry is not None and _scene_diff(entry.thumb, thumb) <= self._scene_change
        if same_scene and now - entry.t <= self._max_cache_age_s:
            if now - entry.t > self._refresh_s:
                self._start_refresh(key, frame, thumb, now)
            return self._detections(frame, entry.labels, entry.boxes, track=True)
        if failed_at is not None and now - failed_at < self._refresh_s:
            return self._track_known(frame, self._known_entry(key, thumb))  # backoff: no brain call
        if same_scene and pending:
            # Too old, but this key's background refresh is in flight or
            # waiting for the slot (Ruling 35): track the cached boxes instead
            # of a second, synchronous brain call. _start_refresh also hands a
            # free slot to the oldest waiting request, so this key's turn
            # comes even without watcher ticks.
            self._start_refresh(key, frame, thumb, now)
            return self._detections(frame, entry.labels, entry.boxes, track=True)
        try:
            entry = self._ask_entry(key, frame, thumb, now)
        except Exception as e:  # noqa: BLE001 - booth rule: keep tracking, never raise
            with self._lock:
                self._failed_at[key] = self._clock()
            what = "inventory" if queries is None else "vocabulary refer"
            known = self._known_entry(key, thumb)
            if known is None:
                self._warn(f"{what} failed; nothing known for this scene, returning no detections", e)
            else:
                self._warn(f"{what} failed; tracking the known objects", e)
            return self._track_known(frame, known)
        self._store(key, entry)
        return self._detections(frame, entry.labels, entry.boxes, track=False)

    def _known_entry(self, key, thumb: np.ndarray) -> _Entry | None:
        """The cached entry at ANY age while this frame still shows its scene;
        None once the scene changed (old boxes would put old labels on new
        things; an agent's own refer bypasses the cache, so picking by name is
        unaffected). The outage fallback, and detect_nowait's answer."""
        with self._lock:
            entry = self._cache.get(key)
        if entry is None or _scene_diff(entry.thumb, thumb) > self._scene_change:
            return None
        return entry

    def _track_known(self, frame: Frame, known: _Entry | None) -> list[Detection]:
        if known is None:
            return []
        return self._detections(frame, known.labels, known.boxes, track=True)

    def _start_refresh(self, key, frame: Frame, thumb: np.ndarray, t: float) -> None:
        """Ask for one background refresh of `key` with this frame; `t` is the
        caller's clock reading. Nothing happens if the detector is closed, the
        key's own refresh is in flight, or the key backs off (`t` within
        `refresh_s` of its last failed brain call, shared with the
        synchronous path).

        One background brain request per detector, across cameras and keys,
        served oldest-first: the request waits in `_waiting`, which keeps the
        key's first-ask time and its latest frame. When the slot is free, it
        goes to the key that has waited longest (ties: first asked) and is
        not backing off, with that key's own latest frame, whichever caller
        found the slot free. A waiting key leaves the queue when its refresh
        starts, when an answer for it is stored, or on close().
        """
        with self._lock:
            if self._closed or key in self._refreshing or self._backs_off(key, t):
                return
            waiting = self._waiting.get(key)
            # Asking again keeps the key's place (dict order: first ask) and
            # updates its frame.
            self._waiting[key] = _Request(t if waiting is None else waiting.since, frame, thumb, t)
            if self._refreshing:
                return  # the slot is busy: wait for a turn
            # The caller's own key qualifies, so there is always a choice;
            # min() keeps the first of equal times, i.e. the first asked.
            chosen, request = min(((k, r) for k, r in self._waiting.items() if not self._backs_off(k, t)),
                                  key=lambda item: item[1].since)
            del self._waiting[chosen]
            thread = threading.Thread(target=self._refresh,
                                      args=(chosen, request.frame, request.thumb, request.t),
                                      name=f"brain-refresh-{chosen}", daemon=True)
            # Started under the lock: close() never misses it, and the thread
            # cannot reach its own bookkeeping before this block has recorded it.
            thread.start()
            self._refreshing.add(chosen)
            self._threads.add(thread)

    def _backs_off(self, key, t: float) -> bool:
        """`key` had a failed brain call less than `refresh_s` before `t`.
        The caller holds the lock."""
        failed_at = self._failed_at.get(key)
        return failed_at is not None and t - failed_at < self._refresh_s

    def _refresh(self, key, frame: Frame, thumb: np.ndarray, t: float) -> None:
        try:
            # Replaces the entry only on success; a failure keeps the old one,
            # which still ages out after max_cache_age_s.
            self._store(key, self._ask_entry(key, frame, thumb, t))
        except Exception as e:  # noqa: BLE001 - background thread: log, never raise
            # The key backs off from the failure (not the request), as after a
            # failed synchronous inventory. Recorded before the in-flight mark
            # is cleared below, so no refresh slips in between.
            with self._lock:
                self._failed_at[key] = self._clock()
            what = "inventory" if key[1] is None else "vocabulary refer"
            self._warn(f"{what} refresh failed; keeping the cached boxes", e)
        finally:
            with self._lock:
                self._refreshing.discard(key)
                self._threads.discard(threading.current_thread())

    # -- masks ----------------------------------------------------------------

    def _detections(self, frame: Frame, labels, boxes: np.ndarray, *, track: bool) -> list[Detection]:
        """Boxes -> Detections with SAM masks on this frame.

        Masks are opaque (np.ndarray, or a CUDA tensor in strict GPU mode):
        only `is None` is ever asked of them. A box without a mask keeps the
        bbox fallback, except in a track step with a working segmenter: there
        the cached box no longer covers an object in this frame.
        """
        if len(labels) == 0:
            return []
        masks = None
        if self._segmenter is not None:
            try:
                masks = self._segmenter.masks(frame, boxes)
            except Exception as e:  # noqa: BLE001 - SAM unavailable: keep boxes
                self._warn("segmenter failed; keeping boxes without masks", e)
        out: list[Detection] = []
        for i, label in enumerate(labels):
            mask = masks[i] if masks is not None else None
            if track and masks is not None and mask is None:
                continue
            out.append(Detection(label=label, conf=1.0, bbox=np.array(boxes[i], dtype=np.float32), mask=mask))
        return out

    # -- logging --------------------------------------------------------------

    def _warn(self, what: str, error: BaseException) -> None:
        now = self._clock()
        with self._lock:
            if self._last_warn is not None and now - self._last_warn < _WARN_EVERY_S:
                return
            self._last_warn = now
        reason = f"{type(error).__name__}: {error}"
        if self._api_key != "EMPTY":
            reason = reason.replace(self._api_key, "<api key>")
        logger.warning("brain detector (%s): %s (%s)", self._model, what, reason)
