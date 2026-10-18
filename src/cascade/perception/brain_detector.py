"""Brain-driven perception: the local vision LLM (the brain) boxes objects, SAM masks them.

This module holds the two prompts measured in the perception bake-off, the parser
that turns the brain's text reply into labelled pixel boxes, and `BrainDetector`,
which asks the brain for those boxes and segments them with a box-prompted SAM.

Wire format: the brain answers a JSON list `[{"label": ..., "bbox_2d": [x1, y1, x2, y2]}]`
with coordinates normalized to 0-1000 of the full image.
"""

from __future__ import annotations

import base64
import collections
import contextlib
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
    return _pixel_boxes(_answer_list(_reply_body(text)), width, height)


#: Served ids of brains whose box replies are guarded (`reply_guards`).
#: Nemotron 3.5 Bolt, greedy, repeated one label with its box sliding a few
#: units per entry until the token cap on 6 of the 47 bake-off frames, and the
#: cut reply was rejected, so those frames showed nothing
#: (docs/evidence/bolt-default-fixes-2026-10-19). It also lists categories the
#: inventory prompt excludes (keyboards, the robot arm) and the rest of that
#: computer (mice, computer towers). Guarded requests ask for BOX_LIST_SCHEMA
#: within GUARDED_MAX_TOKENS and their replies go through parse_guarded_boxes.
#: Every other brain (Qwen3.8 never looped) keeps its request and parse
#: unchanged.
GUARDED_REPLY_MODELS = frozenset({"nemotron-3.5-bolt"})
#: A guarded reply lists at most this many boxes (the schema's maxItems): the
#: busiest bake-off frame has 9 identifiable objects, and no object Bolt found
#: sat past the 12th entry of its reply (it listed at most 15 when it did not
#: loop). A loop starts within the first 8 entries and ends here, after ~430
#: tokens instead of the ~720 a bound of 20 let it spend.
GUARDED_MAX_BOXES = 12
#: ...and decodes at most this many tokens (Bolt spends ~36 per box), so a
#: loop ends at the schema's bound well before this cap.
GUARDED_MAX_TOKENS = 1024
#: The answer both prompts ask for, as a JSON schema the server turns into a
#: grammar (llama.cpp; vLLM's structured outputs): a list of at most
#: GUARDED_MAX_BOXES {label, bbox_2d} objects, 4 integer coordinates 0..1000.
BOX_LIST_SCHEMA = {
    "type": "array", "maxItems": GUARDED_MAX_BOXES,
    "items": {"type": "object", "additionalProperties": False, "required": ["label", "bbox_2d"],
              "properties": {"label": {"type": "string", "minLength": 1, "maxLength": 64},
                             "bbox_2d": {"type": "array", "minItems": 4, "maxItems": 4,
                                         "items": {"type": "integer", "minimum": 0, "maximum": 1000}}}}}
#: In a guarded reply that hit a bound, a label with this many or more entries
#: is a repetition loop: only its first entry is kept.
_LOOP_ENTRIES = 4
#: A label whose head noun (its last words) is a category _EXCLUDE names, or
#: the rest of the computer whose screen and keyboard it names (a computer
#: mouse, a mouse pad, a computer tower or PC case: 11 of Bolt's 41 false
#: positives on the bake-off frames): the inventory prompt asks the brain not
#: to list these, and a guarded reply's entries that do are dropped. Head noun
#: only: "keyboard" goes, "table tennis ball", "hand cream", "plush mouse" and
#: "pencil case" stay.
_EXCLUDED_HEAD = re.compile(
    r"(?:\brobot(?:ic)? arm|\bgrippers?|\bperson|\bpeople|\bhands?|\bscreens?|\bmonitors?|\blaptops?|"
    r"\bkeyboards?|\bfurniture|\bwalls?|\btables?|"
    r"\bcomputer (?:mouse|mice)|\bmouse ?pads?|\b(?:computer|pc) (?:towers?|cases?))$")


def _complete_items(body: str) -> list:
    """The complete JSON values of the first list in `body` that was cut
    before its closing ']' (a reply stopped by max_tokens), in order."""
    decoder = json.JSONDecoder()
    start = body.find("[")
    items: list = []
    if start < 0:
        return items
    i, n = start + 1, len(body)
    while True:
        while i < n and body[i] in " \t\r\n,":
            i += 1
        if i >= n or body[i] != "{":
            return items
        try:
            value, i = decoder.raw_decode(body, i)
        except (ValueError, RecursionError):  # the entry the cut went through
            return items
        items.append(value)


def _label_key(item) -> str | None:
    label = item.get("label") if isinstance(item, dict) else None
    return " ".join(label.lower().split()) if isinstance(label, str) else None


def _without_loops(items: list) -> list:
    """Each label with _LOOP_ENTRIES or more entries reduced to its first
    entry. A loop's later boxes are not objects: they slide off the object,
    often off the image, and the loop can alternate with a second label."""
    counts = collections.Counter(_label_key(item) for item in items)
    seen: set = set()
    out: list = []
    for item in items:
        key = _label_key(item)
        if key is not None and counts[key] >= _LOOP_ENTRIES:
            if key in seen:
                continue
            seen.add(key)
        out.append(item)
    return out


def parse_guarded_boxes(text: str, width: int, height: int, *, cut: bool,
                        inventory: bool = True) -> list[tuple[str, np.ndarray]]:
    """A guarded reply (GUARDED_REPLY_MODELS) -> [(label, pixel box)].

    `cut`: the server stopped the reply at max_tokens. Its complete entries
    before the cut are kept (parse_boxes would find no closed list). A reply
    that hit a bound (cut, or GUARDED_MAX_BOXES entries) ends in a repetition
    loop: each label with _LOOP_ENTRIES or more entries keeps only its first.
    An entry repeating an earlier one exactly (label and box) is dropped, and
    in an `inventory` reply one naming an excluded category (_EXCLUDED_HEAD):
    a refer's entries carry the agent's own words. Then as parse_boxes.
    Never raises.
    """
    body = _reply_body(text)
    items = _answer_list(body)
    if not items and cut:
        items = _complete_items(body)
    if cut or len(items) >= GUARDED_MAX_BOXES:
        items = _without_loops(items)
    seen: set = set()
    kept = []
    for item in items:
        label = _label_key(item)
        key = (label, json.dumps(item.get("bbox_2d")) if isinstance(item, dict) else None)
        if label is not None and (key in seen or (inventory and _EXCLUDED_HEAD.search(label))):
            continue
        seen.add(key)
        kept.append(item)
    return _pixel_boxes(kept, width, height)


def _pixel_boxes(items: list, width: int, height: int) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    for item in items:
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
#: close() waits this long beyond the background request timeout for a
#: refresh to finish (after cancelling it, a stream ends at its next chunk).
_CLOSE_MARGIN_S = 5.0
#: The default max_cache_age_s is idle_refresh_s plus this: the watcher's
#: tick that starts the idle refresh, so an unchanged scene's entry ages out
#: only once that refresh is pending (never a synchronous inventory).
_IDLE_REFRESH_MARGIN_S = 5.0
#: Every foreground() scope entry waits at most this long for a cancelled
#: background request to close its stream before the foreground work starts
#: (BEFORE the skill takes its frame, so it never ages that frame). A request
#: still waiting for its first chunk (prefill) may outlast it.
_CANCEL_WAIT_S = 0.5
#: close() waits this long for the SAM warm-up (weights load + the first,
#: cold CUDA predict): the same minute thread_join.EXIT_JOIN_S allows the
#: slowest cold tick measured. Past it, shutdown names the thread instead.
_WARM_UP_JOIN_S = 60.0
#: reanchor(): the local image check compares the brain's box grown by this
#: fraction of its width and height (half on each side).
_REANCHOR_BOX_GROWTH = 0.10
#: reanchor(): a real camera's auto-exposure / white balance steps between
#: two frames seconds apart (a x1.06 step read a colour local diff of 8.45 on
#: a rig keyframe). The B, G and R gains of the fresh frame over the named one
#: are the medians of their pixel ratios OUTSIDE the local check's box (on a
#: grid of every _GAIN_STRIDE-th pixel), over pixels below _GAIN_SATURATED on
#: both frames (a saturated pixel says nothing about the gain) and at least
#: _GAIN_DARK on the named one (a dark pixel's ratio is noise); a channel with
#: fewer than _GAIN_MIN_SAMPLES such pixels keeps gain 1. The local check then
#: compares the fresh frame with the named one scaled by them (a named-image
#: pixel at _GAIN_SATURATED or more as a lower bound of its true value).
_GAIN_STRIDE = 4
_GAIN_SATURATED = 250
_GAIN_DARK = 16
_GAIN_MIN_SAMPLES = 256
#: ...a gain outside this range is not an exposure step but a real change of
#: the scene (lights, a shadow over the table): refused as a scene change,
#: never normalised away. The edges get _GAIN_RANGE_EPS of slack (the gain is
#: a median of 8-bit ratios), and a refusal prints the gain to 3 decimals: a
#: refused gain never reads as the edge itself.
REANCHOR_GAIN_RANGE = (0.7, 1.4)
_GAIN_RANGE_EPS = 5e-4
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
#: reanchor_reference(), evidence only (perception_timing): a naming-mask
#: pixel outside the brain's box is of a FOREIGN colour when one of its B, G,
#: R values lies more than this many levels from the median of the mask's
#: pixels inside the box.
_FOREIGN_COLOUR_DIFF = 40
#: reanchor(): of the new mask, only the connected components that come
#: within this many pixels of the old mask (the old mask dilated by it) are
#: the object; the rest (a neighbour inside SAM's grown prompt) is dropped
#: before the border, agreement and geometry see the mask.
_REANCHOR_COMPONENT_REACH_PX = 2.0
#: reanchor(): the two masks agree when at least `reanchor_mask_agreement` of
#: each one's pixels lie within this tolerance of the other: max(the floor,
#: the fraction x the diagonal of the old mask's box) pixels. SAM's 1-2 px
#: edge jitter and a brain box a few px tight stay inside it on any size.
#: The same tolerance bounds the geometry: what is measured is SAM's mask
#: without what both images' same-prompt masks add farther than it from the
#: naming mask (a merged neighbour, the ring a tight brain box cut off).
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

    `detection` is the brain's detection with a SAM mask on the newer frame
    and that mask's bounding box, or None: then `scene_changed` says whether
    the image changed around it (the brain's box no longer describes this
    frame) or SAM found no usable mask in it, and `reason` says which. The
    two diffs are always measured (mean absolute grey levels of the
    thumbnails, evidence only; the largest B/G/R mean absolute difference in
    the grown box after the exposure gain). Evidence once known: `prompt_box`, the grown box SAM was prompted
    with (xyxy pixels); `mask_iou`, the pixel IoU of the newer mask with the
    same-prompt mask of the image the object was named on (its re-mask,
    `ReanchorReference`); `mask_agreement`, the smaller of the two shares of
    each mask's pixels within the size-aware tolerance of the other; `dropped_components`, how many separate parts of SAM's newer mask
    (neighbours) were dropped before any of that; `clipped_px`, how many
    pixels of the kept mask are not in the detection's mask (the geometry):
    those farther than that tolerance from the naming mask that the
    reference mask also has (a neighbour SAM merged under the grown prompt
    on both images, or the ring a tight box cut off beyond the tolerance).
    `gain`: the newer image's exposure gain over the older one per B, G, R
    channel (`_exposure_gain`), which the local diff divides out; the
    whole-view (thumbnail) diff is evidence only."""

    detection: Detection | None
    scene_changed: bool
    reason: str
    scene_diff_global: float
    scene_diff_local: float
    prompt_box: tuple[int, int, int, int] | None = None
    mask_iou: float = math.nan
    mask_agreement: float = math.nan
    dropped_components: int = 0
    clipped_px: int = 0
    gain: tuple[float, ...] | None = None


class ReanchorReference(NamedTuple):
    """What `BrainDetector.reanchor_reference` made of the image the object
    was named on: `detection` is the named detection with SAM's mask for the
    grown prompt reanchor() gives the fresh image (only its parts that
    overlap the naming mask) and that mask's box, or None with `reason` (a
    failed re-anchor). `prompt_box` is that prompt (xyxy pixels); `coverage`
    the share of the naming mask's pixels within the tolerance of the
    re-mask. `detection` is what reanchor() COMPARES the fresh mask with;
    `geometry` is what the runtime MEASURES on this image (the 2 cm move
    check's frame0 side): the same re-mask without its pixels farther than
    the agreement tolerance from the naming mask, as reanchor() clips the
    fresh mask.

    Evidence only, refused or not (once the naming mask has a box; never a
    refusal): `naming_overhang`, how far the naming mask's box passes each
    side of the brain's box (left, top, right, bottom), as a fraction of
    that box's width or height, negative where it stops short of the side;
    `naming_foreign_share`, the share of the naming mask's pixels that lie
    outside the brain's box with a colour foreign to the object
    (_FOREIGN_COLOUR_DIFF), NaN with no mask pixel inside the box. The rig's
    data decides between a refusal on them and clipping (follow-ups M5)."""

    detection: Detection | None
    reason: str
    prompt_box: tuple[int, int, int, int] | None = None
    coverage: float = math.nan
    geometry: Detection | None = None
    naming_overhang: tuple[float, float, float, float] | None = None
    naming_foreign_share: float = math.nan


class _Cancelled(Exception):
    """A background refresh closed by a foreground scope or close(): not a
    failure (no backoff, no warning); the scope's exit requeues it."""


def _camera_key(frame: Frame) -> tuple:
    """Image size + intrinsics tell scene from wrist; a frame whose capture
    names its camera (Isaac bridge, OVRTX) also keys by (source, camera): two
    simulated cameras with the same size and K (the kitchen's cam0 and side)
    would otherwise share one cache entry and invalidate each other's
    inventory on every tick. Frames without that identity (RealSense, mock)
    keep the size + K key."""
    h, w = frame.rgb.shape[:2]
    key = (h, w, round(float(frame.K[0, 0])), round(float(frame.K[0, 2])))
    capture = getattr(frame, "capture", None)
    if isinstance(capture, dict) and isinstance(capture.get("camera"), str):
        # str(): the bridge's source is a (host, port) tuple; any other shape
        # must not make the key unhashable.
        return key + (str(capture.get("source")), capture["camera"])
    return key


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


def _exposure_gain(a: Frame, b: Frame, box) -> tuple[float, ...]:
    """`b`'s exposure gain over `a` (same size), per channel (B, G, R): the
    median ratio b / a over the pixels OUTSIDE `box` (xyxy pixels) grown by
    _REANCHOR_BOX_GROWTH, the local check's region, on a grid of every
    _GAIN_STRIDE-th pixel. Pixels saturated on either frame or too dark on
    `a` are skipped; 1.0 for a channel with fewer than _GAIN_MIN_SAMPLES
    left (nothing to measure: no correction)."""
    xa, ya, xb, yb = _grown_box(box, a.rgb.shape)
    outside = np.ones(a.rgb.shape[:2], dtype=bool)
    outside[ya:yb, xa:xb] = False
    outside = outside[::_GAIN_STRIDE, ::_GAIN_STRIDE]
    channels = a.rgb.shape[2] if a.rgb.ndim == 3 else 1
    pa = a.rgb[::_GAIN_STRIDE, ::_GAIN_STRIDE][outside].astype(np.float32).reshape(-1, channels)
    pb = b.rgb[::_GAIN_STRIDE, ::_GAIN_STRIDE][outside].astype(np.float32).reshape(-1, channels)
    gains = []
    for c in range(channels):
        usable = (pa[:, c] >= _GAIN_DARK) & (pa[:, c] < _GAIN_SATURATED) & (pb[:, c] < _GAIN_SATURATED)
        if np.count_nonzero(usable) < _GAIN_MIN_SAMPLES:
            gains.append(1.0)
        else:
            gains.append(float(np.median(pb[usable, c] / pa[usable, c])))
    return tuple(gains)


def _local_diff(a: Frame, b: Frame, box, gain=None) -> float:
    """The largest of the B, G and R mean absolute differences of two
    same-size frames inside `box` (xyxy pixels) grown by
    _REANCHOR_BOX_GROWTH; inf when nothing of it is inside the image.
    Colour, not grey: an object as bright as the table moves unseen in grey.
    With `gain` (`_exposure_gain(a, b, box)`), `b` is compared with `a`
    scaled by it and clipped to 255 as the camera clips it, in `a`'s units:
    an exposure step alone reads no difference on an unclipped pixel. A pixel
    of `a` at _GAIN_SATURATED or more is only a LOWER bound of its true value
    (the sensor clipped it): there `b` at or above the prediction is no
    difference (a white object brighter than the sensor stays white under a
    darkening step) and only a shortfall counts. Such pixels are never
    dropped: the strip a sliding white object leaves behind reads the table
    under the prediction.
    """
    xa, ya, xb, yb = _grown_box(box, a.rgb.shape)
    if xb <= xa or yb <= ya:
        return math.inf
    pa, pb = a.rgb[ya:yb, xa:xb].astype(np.float32), b.rgb[ya:yb, xa:xb].astype(np.float32)
    if gain is None:
        diff = np.abs(pa - pb)
    else:
        g = np.asarray(gain, dtype=np.float32).reshape(-1)
        g = g if pa.ndim == 3 else g[:1]
        predicted = np.minimum(pa * g, 255.0)
        diff = np.where(pa >= _GAIN_SATURATED, np.maximum(predicted - pb, 0.0), np.abs(predicted - pb)) / g
    return float(diff.reshape(-1, diff.shape[2] if diff.ndim == 3 else 1).mean(axis=0).max())


def _tolerance_px(box) -> float:
    """The agreement tolerance of a mask with box `box` (xyxy pixels):
    max(_REANCHOR_TOL_MIN_PX, _REANCHOR_TOL_FRAC x its diagonal)."""
    return max(_REANCHOR_TOL_MIN_PX, _REANCHOR_TOL_FRAC * math.hypot(box[2] - box[0], box[3] - box[1]))


class _Clip(NamedTuple):
    """`_clip_to_naming`: the clipped mask (`mask`: the input itself, still
    opaque, when nothing was cut; else a host bool mask of its size), its
    box (frame xyxy; None when nothing is left) and the pixels cut."""

    mask: object
    box: np.ndarray | None
    removed: int


def _clip_to_naming(mask, mask_box, naming, naming_box, tol_px: float, reference=None,
                    reference_box=None) -> _Clip:
    """`mask` without its pixels farther than `tol_px` (Euclidean) from the
    naming mask `naming` (the naming mask dilated by the tolerance) that
    `reference`, the other image's same-prompt mask, ALSO has within the
    tolerance: what SAM's grown prompt adds beyond the naming mask on both
    images alike (a neighbour merged with the object, the ring a tight brain
    box cut off), which main's single-frame mask lacks too. Pixels new on
    this image (farther than the tolerance from `reference`: the object
    where it moved to, within the agreement's slack) stay: cutting them
    would measure a slid object where it was. Without `reference`, every
    pixel farther than the tolerance from the naming mask is cut. On the
    crop covering the boxes: one host copy per mask, sliced on the device
    for a CUDA tensor."""
    boxes = [mask_box, naming_box] + ([reference_box] if reference is not None else [])
    x0, y0 = int(min(b[0] for b in boxes)), int(min(b[1] for b in boxes))
    x1, y1 = int(max(b[2] for b in boxes)), int(max(b[3] for b in boxes))
    named, crop = _host_crop(naming, x0, y0, x1, y1), _host_crop(mask, x0, y0, x1, y1)
    near = cv2.distanceTransform((~named).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) <= tol_px
    if reference is not None:
        ref = _host_crop(reference, x0, y0, x1, y1)
        near |= cv2.distanceTransform((~ref).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) > tol_px
    removed = int(np.count_nonzero(crop & ~near))
    if not removed:
        return _Clip(mask, np.asarray(mask_box, dtype=np.float64), 0)
    kept = crop & near
    box = _mask_box(kept)
    if box is None:
        return _Clip(None, None, removed)
    out = np.zeros(tuple(mask.shape), dtype=bool)
    out[y0:y1, x0:x1] = kept
    return _Clip(out, box + np.array([x0, y0, x0, y0], dtype=np.float64), removed)


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


def _match_masks(old, old_box, new, new_box, *, reach_px: float = _REANCHOR_COMPONENT_REACH_PX) -> _MaskMatch:
    """Compare an older mask `old` with a newer mask `new` (same size,
    opaque, non-empty, with their boxes) on the crop covering both, at full
    resolution: one host copy per mask, sliced on the device for a CUDA
    tensor.

    First the component filter: of `new`, only the 8-connected components
    with a pixel within `reach_px` of `old` are kept (0: overlapping it; by
    default _REANCHOR_COMPONENT_REACH_PX: a still neighbour inside SAM's
    grown prompt is a separate component; a neighbour that moved into
    contact is part of the object's). Then, for the kept mask: `new_near`,
    the share of its pixels within `tol_px` of `old`; `old_near`, the share
    of `old`'s pixels within `tol_px` of it; `agreement`, the smaller of the
    two, with tol_px = max(_REANCHOR_TOL_MIN_PX, _REANCHOR_TOL_FRAC x the
    diagonal of `old`'s box); `iou`, their raw pixel IoU (evidence)."""
    x0, y0 = int(min(old_box[0], new_box[0])), int(min(old_box[1], new_box[1]))
    x1, y1 = int(max(old_box[2], new_box[2])), int(max(old_box[3], new_box[3]))
    a, b = _host_crop(old, x0, y0, x1, y1), _host_crop(new, x0, y0, x1, y1)
    tol_px = _tolerance_px(old_box)
    # Euclidean distance of every crop pixel to the nearest pixel of `old`.
    to_old = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    count, labels = cv2.connectedComponents(b.astype(np.uint8), connectivity=8)
    near = np.unique(labels[b & (to_old <= reach_px)])
    near = near[near != 0]
    dropped = count - 1 - near.size
    if near.size == 0:
        return _MaskMatch(None, (x0, y0), dropped, math.nan, math.nan, tol_px)
    kept = np.isin(labels, near)
    to_kept = cv2.distanceTransform((~kept).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    new_near = np.count_nonzero(to_old[kept] <= tol_px) / np.count_nonzero(kept)
    old_near = np.count_nonzero(to_kept[a] <= tol_px) / np.count_nonzero(a)
    iou = np.count_nonzero(a & kept) / np.count_nonzero(a | kept)
    return _MaskMatch(kept, (x0, y0), dropped, float(min(new_near, old_near)), float(iou), tol_px,
                      float(new_near), float(old_near))


def _kept_mask(new, match: _MaskMatch):
    """frame1's mask with only the kept components: `new` itself (still
    opaque) when nothing was dropped, else a host bool mask of its size."""
    if not match.dropped:
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


def _naming_evidence(frame: Frame, mask, mask_box, box) -> tuple[tuple[float, float, float, float] | None, float]:
    """`ReanchorReference.naming_overhang` and `.naming_foreign_share` of a
    naming mask (opaque, its box `mask_box`) for the brain's box `box` (xyxy
    pixels) on `frame`: evidence only. On the mask's own box crop: one host
    copy. (None, NaN) when it cannot be measured; never raises."""
    try:
        bx0, by0, bx1, by1 = (float(v) for v in np.asarray(box, dtype=np.float64).reshape(-1)[:4])
        mx0, my0, mx1, my1 = (int(v) for v in mask_box)
        bw, bh = bx1 - bx0, by1 - by0
        if not (bw > 0 and bh > 0):
            return None, math.nan
        overhang = ((bx0 - mx0) / bw, (by0 - my0) / bh, (mx1 - bx1) / bw, (my1 - by1) / bh)
        crop = _host_crop(mask, mx0, my0, mx1, my1)
        pixels = int(np.count_nonzero(crop))
        if not pixels or frame.rgb.ndim != 3:
            return overhang, math.nan
        # Pixel centres inside the brain's box (a 0.5 px border either way).
        xs = np.arange(mx0, mx1) + 0.5
        ys = np.arange(my0, my1) + 0.5
        inside = ((ys >= by0) & (ys < by1))[:, None] & ((xs >= bx0) & (xs < bx1))[None, :]
        colours = frame.rgb[my0:my1, mx0:mx1].astype(np.int16)
        own = crop & inside
        if not own.any():
            return overhang, math.nan
        ref = np.median(colours[own], axis=0)
        foreign = crop & ~inside & (np.abs(colours - ref).max(axis=-1) > _FOREIGN_COLOUR_DIFF)
        return overhang, float(np.count_nonzero(foreign) / pixels)
    except Exception:  # noqa: BLE001 -- evidence never fails a re-anchor
        return None, math.nan


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
    synchronously, under `inventory_timeout_s` (the same tokens as a
    background inventory), and store the answer. One exception (Ruling 35):
    an entry past `max_cache_age_s` whose scene is unchanged keeps being
    tracked while that key's background refresh is in flight or waiting for
    the slot. Keep `max_cache_age_s >= idle_refresh_s + 5 s` (its default,
    None, is exactly that; every shipped profile keeps it; the 5 s cover the watcher's tick that starts the idle
    refresh): an unchanged scene's entry then ages out only once the
    watcher's idle refresh is pending, so it never costs a synchronous
    inventory.

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
    named on an older frame in a newer frame of the same camera: SAM on the
    brain's box grown by 20 % (at least 6 px a side), no brain call; parts of
    the new mask not connected to the old mask's place (a neighbour) are
    dropped. Refused as a scene change when the image changed around it
    (`reanchor_local_diff` in colour inside the box grown by 10 %, after
    dividing out the camera's exposure gain measured outside it; a gain
    outside REANCHOR_GAIN_RANGE is itself a change; the whole-view diff is
    evidence only here), when the new mask reaches a border of
    SAM's prompt box the old mask did not, or when under
    `reanchor_mask_agreement` of either mask's pixels lie within a few pixels
    (size-aware) of the other; refused as a failed re-anchor when the mask is
    missing, empty or off the box.

    Foreground scope (`foreground()`, re-entrant; the runtime holds it for
    motion skills, preview_grasp and localize_object): background work never
    competes with a foreground localization for the brain or SAM. While it
    is held, no background refresh starts (requests still queue, keeping
    their latest frame), the one in flight is cancelled (background requests
    stream with `inventory_timeout_s`; a cancelled stream is closed, which
    aborts the request on the server, and goes back to the queue with the
    key's newest frame: no backoff, no warning, whatever error ends it), and
    `detect_nowait` returns [] without a SAM call. When the outermost scope
    ends, the oldest waiting request starts. An agent's refers stay single
    non-streaming requests (`timeout_s`).

    Guarded replies (`reply_guards`, default: by served id, see
    GUARDED_REPLY_MODELS: Nemotron 3.5 Bolt): every request asks for
    BOX_LIST_SCHEMA within GUARDED_MAX_TOKENS, a reply cut at the token cap
    keeps its complete boxes instead of failing, a reply that hit a bound
    loses its repetition loop, and entries naming a category the prompt
    excludes are dropped (parse_guarded_boxes). Other brains' requests and
    replies are handled exactly as before.

    Safe to call from several threads at once: the cache lock is never held
    across the brain or SAM call, and the segmenter serializes its own model
    calls.
    """

    concurrent_safe = True
    #: world.WorldWatcher asks a non-fusing camera (fuse=False: the sim's proof
    #: camera, the rig's wrist) nothing: no background inventory, no SAM step.
    #: The primary camera is exempt (its entry serves describe_scene and
    #: localization).
    background_fusing_cameras_only = True

    def __init__(self, *, base_url: str, model: str, api_key_file: str | None = None,
                 timeout_s: float = 20.0, max_tokens: int = 1500, segmenter: BoxSegmenter | None = None,
                 refresh_s: float = 4.0, max_cache_age_s: float | None = None, scene_change: float = 6.0,
                 idle_refresh_s: float = 30.0, inventory_timeout_s: float | None = None,
                 reanchor_local_diff: float = 8.0, reanchor_mask_agreement: float = 0.95,
                 reply_guards: bool | None = None, client=None, clock=time.monotonic):
        self._base_url = str(base_url)
        self._model = str(model)
        self._timeout_s = float(timeout_s)
        # Background (streamed) requests: the whole request, first byte to last
        # chunk. Defaults to timeout_s; a slow brain's inventory needs more.
        self._inventory_timeout_s = float(timeout_s if inventory_timeout_s is None else inventory_timeout_s)
        self._max_tokens = int(max_tokens)
        # None: by served id (GUARDED_REPLY_MODELS), the default every profile uses.
        self._guarded = self._model in GUARDED_REPLY_MODELS if reply_guards is None else bool(reply_guards)
        self._segmenter = segmenter
        self._refresh_s = float(refresh_s)
        self._idle_refresh_s = float(idle_refresh_s)
        # None: idle_refresh_s + _IDLE_REFRESH_MARGIN_S, the invariant every
        # shipped profile keeps (see the class docstring).
        self._max_cache_age_s = float(self._idle_refresh_s + _IDLE_REFRESH_MARGIN_S if max_cache_age_s is None
                                      else max_cache_age_s)
        self._scene_change = float(scene_change)
        self._reanchor_local_diff = float(reanchor_local_diff)
        self._reanchor_mask_agreement = float(reanchor_mask_agreement)
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
        # The cancel Event of each background request in flight (by key).
        self._cancels: dict[tuple, threading.Event] = {}
        # Depth of foreground() scopes held right now (re-entrant, any thread).
        self._foreground = 0
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

        Inside a foreground() scope: [] without any SAM call; a refresh the
        frame needs is queued (with this frame) and starts after the scope.
        """
        try:
            queries = tuple(str(q) for q in classes) if classes else None
            key = _cache_key(frame, queries)
            thumb = _thumbnail(frame)
            now = self._clock()
            known = self._known_entry(key, thumb)
            if known is None or now - known.t > self._idle_refresh_s:
                self._start_refresh(key, frame, thumb, now)
            with self._lock:
                if self._foreground:
                    return []  # the GPU belongs to the foreground localization
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
        object). Refused, as a failed re-anchor (one image: nothing moved),
        when no segmenter is configured, `det` has no mask, the naming mask's
        box overlaps the brain's box by IoU < REANCHOR_MIN_IOU (SAM's mask
        and the brain's box disagree: a neighbour merged into the mask, only
        part of the object masked, an offset; checked before the re-mask's
        SAM call), SAM gives no mask
        or an empty one, no part of it overlaps the naming mask, the kept
        mask's box overlaps the brain's box by IoU < REANCHOR_MIN_IOU, or under
        `reanchor_mask_agreement` of the naming mask's pixels lie within the
        tolerance (as reanchor's) of the kept mask: SAM masked only part of
        the object, and two such masks would agree with each other and
        measure that part. The re-mask may reach past the naming mask: that
        ring is what a tight box cut off. The reference's `geometry` is the
        kept re-mask clipped as reanchor() clips the fresh mask (to the naming
        mask dilated by the agreement tolerance of the kept re-mask's box):
        the frame0 side of the runtime's move check. Every reference made
        once the naming mask has a box, refused or not, carries the naming
        mask's overhang and foreign-colour share (evidence only)."""
        where = "on the image the object was named on"
        if self._segmenter is None:
            return ReanchorReference(None, "no segmenter is configured")
        named_box = _mask_box(det.mask) if det.mask is not None else None
        if named_box is None:
            return ReanchorReference(None, f"the object has no mask {where}")
        evidence = _naming_evidence(frame, det.mask, named_box, det.bbox)

        def out(detection, reason, prompt_box=None, coverage=math.nan, geometry=None) -> ReanchorReference:
            return ReanchorReference(detection, reason, prompt_box, coverage, geometry, *evidence)

        # M5: the naming mask's box and the brain's box disagree (SAM merged
        # a neighbour, masked part of the object, or the mask is offset): a
        # coarse fail-closed guard, refused before any re-mask. The overhang
        # and foreign share above are logged for a finer rule (owner ruling).
        named_iou = _box_iou(named_box, det.bbox)
        if not named_iou >= REANCHOR_MIN_IOU:
            return out(None, f"SAM's mask and the brain's box disagree (box IoU {named_iou:.2f} < "
                             f"{REANCHOR_MIN_IOU:g}) {where}")
        prompt_box = _reanchor_prompt(det.bbox, frame.rgb.shape)
        prompt = np.asarray(prompt_box, dtype=np.float32).reshape(1, 4)
        found = self._detections(frame, [det.label], prompt, track=True)
        mask = found[0].mask if found else None
        if mask is None:
            return out(None, f"SAM returned no mask for the grown box {where}", prompt_box)
        mask_box = _mask_box(mask)
        if mask_box is None:
            return out(None, f"SAM returned an empty mask for the grown box {where}", prompt_box)
        match = _match_masks(det.mask, named_box, mask, mask_box, reach_px=0.0)
        if match.kept is None:
            return out(None, f"SAM's mask lies off the box: no part of it overlaps the object's mask {where}",
                       prompt_box)
        kept_box = _mask_box(match.kept) + np.array([*match.origin, *match.origin], dtype=np.float64)
        iou = _box_iou(kept_box, det.bbox)
        if not iou >= REANCHOR_MIN_IOU:
            return out(None, f"SAM's mask lies off the box (IoU {iou:.2f} < {REANCHOR_MIN_IOU:g}) {where}",
                       prompt_box)
        if not match.old_near >= self._reanchor_mask_agreement:
            return out(None, f"SAM's mask for the grown box covers only {match.old_near:.2f} of the object's mask "
                             f"{where} (< {self._reanchor_mask_agreement:g} of its pixels within "
                             f"{match.tol_px:.1f} px)", prompt_box, match.old_near)
        kept = _kept_mask(mask, match)
        detection = Detection(label=det.label, conf=det.conf, bbox=kept_box.astype(np.float32), mask=kept)
        clip = _clip_to_naming(kept, kept_box, det.mask, named_box, _tolerance_px(kept_box))
        if clip.box is None:  # unreachable while the kept mask overlaps the naming mask; fail closed
            return out(None, f"no part of SAM's mask for the grown box lies near the object's mask {where}",
                       prompt_box, match.old_near)
        geometry = Detection(label=det.label, conf=det.conf, bbox=clip.box.astype(np.float32), mask=clip.mask)
        return out(detection, "", prompt_box, match.old_near, geometry)

    def reanchor(self, frame: Frame, det: Detection, *, since: Frame,
                 reference: ReanchorReference | None = None) -> ReanchorResult:
        """Re-find `det`, which the brain named on `since` (with SAM's mask
        there), in the newer `frame` of the same camera, without asking the
        brain. `reference` is `reanchor_reference(since, det)`, taken before
        `frame` was (None: made here, its SAM call inside this frame's age).

        Refused as a scene change, with no SAM call, when the fresh frame's
        exposure gain over `since` (B, G, R medians of their pixel ratios
        outside the brain's box grown by 10 %, `_exposure_gain`) lies outside
        REANCHOR_GAIN_RANGE (the light changed, not the exposure) or when,
        inside that box, the largest B/G/R mean difference of the fresh frame
        and `since` scaled by that gain exceeds `reanchor_local_diff`. The
        thumbnails' whole-view difference is measured as evidence only: a
        visitor walking behind the table changes it, the object's box does
        not, and the checks below guard the object. Refused as a failed
        re-anchor when the
        reference was refused. Otherwise SAM masks the brain's box grown by
        20 %, at least _REANCHOR_PROMPT_MIN_PX a side, on `frame` (the track
        step): the reference's prompt, so both masks are made alike and
        compared with each other. Of the new mask only the connected
        components within 2 px of the reference mask are kept (a still
        neighbour inside the prompt is dropped before any check and the
        geometry; one that moved into contact is part of the object's
        component). Refused as a failed re-anchor when SAM gives no mask, an
        empty one, none of it near the reference mask, or a kept mask whose
        bounding box overlaps the brain's box by IoU < REANCHOR_MIN_IOU.
        Refused as a scene change when the kept mask reaches a side of the
        prompt box the reference mask did not reach (the object extends past
        the box it was named in; a mask clipped by a stale prompt touches it),
        or when under `reanchor_mask_agreement` of its pixels lie within the
        tolerance (max(2 px, 3 % of the reference mask box's diagonal)) of the
        reference mask, or of the reference mask's pixels within it of the
        kept one. All of these compare SAM's own (unclipped) masks. The
        detection returned is the GEOMETRY: the kept mask without its pixels
        that lie farther than that tolerance from `det`'s naming mask (the
        naming mask dilated by it) and within it of the reference mask, and
        that mask's bounding box. A neighbour SAM merged with the object
        under the grown prompt on both images agrees with itself and is cut
        off here, and a tight brain box gives main's single-frame geometry;
        pixels new on `frame` (an object that slid within the agreement's
        slack: up to 5 % of a long object's pixels past the tolerance) stay,
        so a slid object is measured where it is. Refused as a failed
        re-anchor when `det` has no naming mask or nothing of the kept mask
        is left. Never falls back
        to the box: a box cloud includes table pixels. Masks stay opaque
        (numpy or CUDA tensors): only projections and one crop per mask are
        copied to the host; a mask with dropped components or clipped pixels
        comes back as a host bool mask."""
        if frame.rgb.shape != since.rgb.shape:
            return ReanchorResult(None, True, "the image size changed", math.inf, math.inf)
        # Evidence only (I1b): the cache's scene check keeps it; here the
        # local diff, components, border, agreement and the move check
        # guard the object.
        diff_global = _scene_diff(_thumbnail(since), _thumbnail(frame))
        gain = _exposure_gain(since, frame, det.bbox)
        lo, hi = REANCHOR_GAIN_RANGE
        exposure = all(lo - _GAIN_RANGE_EPS <= g <= hi + _GAIN_RANGE_EPS for g in gain)
        diff_local = _local_diff(since, frame, det.bbox, gain if exposure else None)
        prompt_box = None
        match = None

        def refused(scene_changed: bool, reason: str) -> ReanchorResult:
            evidence = (match.iou, match.agreement, match.dropped) if match is not None else (math.nan, math.nan, 0)
            return ReanchorResult(None, scene_changed, reason, diff_global, diff_local, prompt_box, *evidence,
                                  gain=gain)

        if not exposure:
            return refused(True, "the brightness of the view changed (gain x"
                                 f"{'/'.join(f'{g:.3f}' for g in gain)} outside {lo:g}-{hi:g}): not an exposure step")
        if not diff_local <= self._reanchor_local_diff:
            return refused(True, f"the image around the object changed (mean colour difference "
                                 f"{diff_local:.1f} > {self._reanchor_local_diff:g})")
        if self._segmenter is None:
            return refused(False, "no segmenter is configured")
        if reference is None:
            reference = self.reanchor_reference(since, det)
        ref = reference.detection
        if ref is None or ref.mask is None:
            return refused(False, reference.reason or "the object was not re-masked on the image it was named on")
        named_box = _mask_box(det.mask) if det.mask is not None else None
        if named_box is None:
            return refused(False, "the object has no mask on the image it was named on")
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
        clip = _clip_to_naming(mask, mask_box, det.mask, named_box, match.tol_px, ref.mask, ref_box)
        if clip.box is None:
            return refused(False, f"no part of SAM's mask lies within {match.tol_px:.1f} px of the object's mask on "
                                  "the image it was named on")
        detection = Detection(label=det.label, conf=det.conf, bbox=clip.box.astype(np.float32), mask=clip.mask)
        return ReanchorResult(detection, False, "", diff_global, diff_local, prompt_box, match.iou,
                              match.agreement, match.dropped, clip.removed, gain)

    @contextlib.contextmanager
    def foreground(self):
        """Hold off background work while a foreground localization runs
        (re-entrant, from any thread; see the class docstring). Every entry
        cancels the background request in flight and waits up to
        `_CANCEL_WAIT_S` for a cancelled stream still open (one in prefill
        outlives the outermost scope's wait; a nested or another thread's
        scope must not overlap its last chunk either). Leaving the outermost
        scope (also by an exception) starts the oldest waiting refresh."""
        entered = False
        try:
            with self._lock:
                # Counted inside the try: an asynchronous KeyboardInterrupt
                # after this line still ends the scope in the finally.
                self._foreground += 1
                entered = True
                for cancel in self._cancels.values():
                    cancel.set()
                threads = list(self._threads)
            if threads:
                deadline = time.monotonic() + _CANCEL_WAIT_S
                for thread in threads:
                    thread.join(max(0.0, deadline - time.monotonic()))
            yield
        finally:
            if entered:
                with self._lock:
                    self._foreground -= 1
                    self._start_next(self._clock())

    @staticmethod
    def camera_key(frame: Frame) -> tuple:
        """The camera part of this frame's cache key (apps.demo warns when
        two watched cameras share one)."""
        return _camera_key(frame)

    def refresh_threads(self) -> list[threading.Thread]:
        """The background refresh threads in flight (apps.demo.owned_threads)."""
        with self._lock:
            return list(self._threads)

    def close(self) -> None:
        """Cancel the background request in flight (its stream closes at
        the next chunk), then join the refresh threads and the SAM warm-up,
        each with a bound; later refreshes do not start. A thread past its
        bound keeps running and is reported on stderr; shutdown_runtime lists
        one still alive in its pending threads."""
        with self._lock:
            self._closed = True
            self._waiting.clear()  # no later refresh; drops their frames
            for cancel in self._cancels.values():
                cancel.set()
            threads = list(self._threads)
        for thread in threads:
            join_thread(thread, what="brain detector", timeout_s=self._inventory_timeout_s + _CLOSE_MARGIN_S)
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

    def _ask(self, image_url: str, prompt: str, width: int, height: int, cancel: threading.Event | None = None,
             timeout: float | None = None, *, inventory: bool = False) -> list[tuple[str, np.ndarray]]:
        """One brain request -> its (label, pixel box) list. Foreground
        (`cancel` None): a single non-streaming completion under `timeout`
        (None: the client's `timeout_s`, an agent's refer). Background (a
        refresh's cancel Event): streamed under `inventory_timeout_s`, see
        `_ask_streamed`. A guarded brain (GUARDED_REPLY_MODELS) is asked for
        BOX_LIST_SCHEMA within GUARDED_MAX_TOKENS and its reply goes through
        parse_guarded_boxes (`inventory`: INVENTORY_PROMPT's reply); any
        other brain's request and parse are as they always were."""
        max_tokens = min(self._max_tokens, GUARDED_MAX_TOKENS) if self._guarded else self._max_tokens
        extra_body: dict = {"chat_template_kwargs": {"enable_thinking": False}}
        if self._guarded:
            extra_body["response_format"] = {"type": "json_schema",
                                             "json_schema": {"name": "box_list", "schema": BOX_LIST_SCHEMA}}
        request = dict(
            model=self._model,
            temperature=0,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            }],
            extra_body=extra_body,
        )
        if cancel is None:
            per_request = {} if timeout is None else {"timeout": timeout}
            choice = self._get_client().chat.completions.create(**request, **per_request).choices[0]
            text, finish_reason = choice.message.content or "", getattr(choice, "finish_reason", None)
        else:
            text, finish_reason = self._ask_streamed(request, cancel)
        # A garbled reply is a failed call, never an empty table: cut at
        # max_tokens (even if a list closed before the cut; a guarded brain's
        # cut reply only when no complete box precedes the cut), or neither a
        # box list nor a reply opening with "[]" (what both prompts ask for
        # when nothing is there).
        cut = finish_reason == "length"
        if self._guarded:
            found = parse_guarded_boxes(text, width, height, cut=cut, inventory=inventory)
            if cut and not found:
                raise ValueError(f"garbled reply, truncated at max_tokens={max_tokens} without a usable box")
            if cut:
                logger.info("brain detector (%s): reply cut at max_tokens=%d; kept its %d complete boxes",
                            self._model, max_tokens, len(found))
                return found
        elif cut:
            raise ValueError(f"garbled reply, truncated at max_tokens={self._max_tokens}")
        if not _is_answer(text):
            raise ValueError(f"garbled reply, no box list: {text[:80]!r}")
        return found if self._guarded else parse_boxes(text, width, height)

    def _ask_streamed(self, request: dict, cancel: threading.Event) -> tuple[str, str | None]:
        """A background request: `stream=True` with the per-request timeout
        `inventory_timeout_s` (openai applies it per read), plus the same
        bound on the whole stream, checked between chunks. `cancel` is
        checked before the request and between chunks: once set, the stream
        is closed (the server aborts the request) and _Cancelled raised. A
        stream that already ended is kept: the server's work is done.
        -> (reply text, finish reason)."""
        if cancel.is_set():
            raise _Cancelled
        deadline = time.monotonic() + self._inventory_timeout_s
        stream = self._get_client().chat.completions.create(
            **request, stream=True, timeout=self._inventory_timeout_s)
        parts: list[str] = []
        finish_reason = None
        try:
            for chunk in stream:
                if cancel.is_set():
                    raise _Cancelled
                if time.monotonic() > deadline:
                    raise TimeoutError(f"background request exceeded {self._inventory_timeout_s:g} s")
                for choice in getattr(chunk, "choices", None) or ():
                    content = getattr(getattr(choice, "delta", None), "content", None)
                    if content:
                        parts.append(content)
                    finish_reason = getattr(choice, "finish_reason", None) or finish_reason
        finally:
            stream.close()
        return "".join(parts), finish_reason

    # -- refer and inventory ------------------------------------------------

    def _refer(self, frame: Frame, queries: list[str]) -> list[Detection]:
        labels, boxes = self._ask_refer(frame, queries)
        return self._detections(frame, labels, boxes, track=False)

    def _ask_refer(self, frame: Frame, queries, cancel: threading.Event | None = None,
                   timeout: float | None = None) -> tuple[list[str], np.ndarray]:
        """One request per query, concatenated; labels are the queries."""
        h, w = frame.rgb.shape[:2]
        image_url = _jpeg_data_url(frame)
        labels: list[str] = []
        boxes: list[np.ndarray] = []
        for q in queries:
            for _, box in self._ask(image_url, REFER_PROMPT.format(q=q), w, h, cancel, timeout):
                labels.append(q)
                boxes.append(box)
        return labels, np.array(boxes, dtype=np.float32).reshape(-1, 4)

    def _ask_inventory(self, frame: Frame, thumb: np.ndarray, t: float,
                       cancel: threading.Event | None = None, timeout: float | None = None) -> _Entry:
        h, w = frame.rgb.shape[:2]
        found = self._ask(_jpeg_data_url(frame), INVENTORY_PROMPT, w, h, cancel, timeout, inventory=True)
        boxes = np.array([box for _, box in found], dtype=np.float32).reshape(-1, 4)
        boxes.setflags(write=False)
        return _Entry(tuple(label for label, _ in found), boxes, thumb, t)

    def _ask_entry(self, key, frame: Frame, thumb: np.ndarray, t: float,
                   cancel: threading.Event | None = None, timeout: float | None = None) -> _Entry:
        """The brain call behind a cache key: the inventory prompt for
        (camera, None), the refer loop for (camera, queries). `cancel`: a
        background refresh (streamed, cancellable); `timeout`: a synchronous
        pass's per-request timeout."""
        queries = key[1]
        if queries is None:
            return self._ask_inventory(frame, thumb, t, cancel, timeout)
        labels, boxes = self._ask_refer(frame, queries, cancel, timeout)
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
            # A synchronous inventory decodes as many tokens as a background
            # one: it gets inventory_timeout_s, not an agent refer's timeout_s
            # (the kitchen's 25 s inventory never fits timeout_s 20).
            entry = self._ask_entry(key, frame, thumb, now, timeout=self._inventory_timeout_s)
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
        starts, when an answer for it is stored, or on close(). Inside a
        foreground() scope the request only queues (see `_start_next`).
        A key whose own refresh was cancelled but is still closing (a request
        in prefill sees the cancel only at its first chunk) queues this newer
        frame: that refresh is requeued with it (`_refresh`).
        """
        with self._lock:
            if self._closed or self._backs_off(key, t):
                return
            cancel = self._cancels.get(key)
            if key in self._refreshing and (cancel is None or not cancel.is_set()):
                return  # its own refresh is in flight and will answer
            waiting = self._waiting.get(key)
            # Asking again keeps the key's place (dict order: first ask) and
            # updates its frame.
            self._waiting[key] = _Request(t if waiting is None else waiting.since, frame, thumb, t)
            self._start_next(t)

    def _start_next(self, t: float) -> None:
        """Start the oldest waiting refresh that does not back off at `t`
        (ties: first asked), if the slot is free, no foreground() scope is
        held and the detector is open. The caller holds the lock."""
        if self._closed or self._foreground or self._refreshing:
            return  # closed, the GPU belongs to the foreground, or the slot is busy
        # min() keeps the first of equal times, i.e. the first asked.
        chosen, request = min(((k, r) for k, r in self._waiting.items() if not self._backs_off(k, t)),
                              key=lambda item: item[1].since, default=(None, None))
        if request is None:
            return
        del self._waiting[chosen]
        cancel = threading.Event()
        thread = threading.Thread(target=self._refresh, args=(chosen, request, cancel),
                                  name=f"brain-refresh-{chosen}", daemon=True)
        # Started under the lock: close() and foreground() never miss it, and
        # the thread cannot reach its own bookkeeping before this block has
        # recorded it.
        thread.start()
        self._refreshing.add(chosen)
        self._threads.add(thread)
        self._cancels[chosen] = cancel

    def _backs_off(self, key, t: float) -> bool:
        """`key` had a failed brain call less than `refresh_s` before `t`.
        The caller holds the lock."""
        failed_at = self._failed_at.get(key)
        return failed_at is not None and t - failed_at < self._refresh_s

    def _refresh(self, key, request: _Request, cancel: threading.Event) -> None:
        cancelled = False
        try:
            # Replaces the entry only on success; a failure keeps the old one,
            # which still ages out after max_cache_age_s.
            self._store(key, self._ask_entry(key, request.frame, request.thumb, request.t, cancel))
        except Exception as e:  # noqa: BLE001 - background thread: log, never raise
            # A refresh whose cancel was set is never a failure, whatever
            # ended it (_Cancelled, or a request cancelled in prefill that then
            # timed out or broke): a foreground scope (or close) took the
            # brain. A hung brain is caught by the next uncancelled refresh.
            cancelled = isinstance(e, _Cancelled) or cancel.is_set()
            with self._lock:
                if cancelled:
                    # Back to the queue with its place and the newest frame (a
                    # scene change seen while it was closing queued that one,
                    # see _start_refresh). close() drops it instead.
                    if not self._closed:
                        newer = self._waiting.get(key)
                        self._waiting[key] = request if newer is None else newer._replace(
                            since=min(newer.since, request.since))
                else:
                    # The key backs off from the failure (not the request), as
                    # after a failed synchronous inventory. Recorded before the
                    # in-flight mark is cleared below, so no refresh slips in.
                    self._failed_at[key] = self._clock()
            if not cancelled:
                what = "inventory" if key[1] is None else "vocabulary refer"
                self._warn(f"{what} refresh failed; keeping the cached boxes", e)
        finally:
            with self._lock:
                self._refreshing.discard(key)
                self._threads.discard(threading.current_thread())
                self._cancels.pop(key, None)
                if cancelled:
                    # The scope may have ended while this stream was closing.
                    self._start_next(self._clock())

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
