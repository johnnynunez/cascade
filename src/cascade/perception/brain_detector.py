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
        except json.JSONDecodeError:
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
    except (json.JSONDecodeError, RecursionError):
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


class _Entry(NamedTuple):
    """One camera's inventory: the brain's boxes and the frame they came from."""

    labels: tuple[str, ...]
    boxes: np.ndarray  # (N, 4) float32 xyxy pixels, read-only
    thumb: np.ndarray  # (36, 64) uint8 grayscale thumbnail
    t: float  # clock time of the frame the brain saw


def _camera_key(frame: Frame) -> tuple[int, int, int, int]:
    """Frame carries no camera name; image size + intrinsics tell scene from wrist."""
    h, w = frame.rgb.shape[:2]
    return h, w, round(float(frame.K[0, 0])), round(float(frame.K[0, 2]))


def _thumbnail(frame: Frame) -> np.ndarray:
    gray = cv2.cvtColor(frame.rgb, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, _THUMB_SIZE, interpolation=cv2.INTER_AREA)


def _scene_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def _jpeg_data_url(frame: Frame) -> str:
    # Frame.rgb is BGR (OpenCV) despite the name and imencode expects BGR:
    # no channel swap, or the brain sees wrong colours.
    ok, buf = cv2.imencode(".jpg", frame.rgb, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding of the frame failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


class BrainDetector(Detector):
    """Detector backed by the brain (an OpenAI-compatible vision LLM) plus SAM masks.

    `detect(frame, [q, ...])` (refer) asks the brain for each query
    synchronously; labels are the queries. `detect(frame, None)` (inventory)
    keeps one cache entry per camera: while the entry is younger than
    `max_cache_age_s` and the scene thumbnail moved by at most `scene_change`,
    it returns a track step (SAM on the cached boxes in THIS frame) and, once
    the entry is older than `refresh_s`, starts one background refresh.
    Otherwise it asks the brain synchronously.

    Booth rule: a down/slow/garbled brain yields a rate-limited warning, never
    an exception. A failed refer yields []. A failed synchronous inventory
    keeps tracking the known objects: the cached entry's track step at any age
    while this frame still shows its scene, [] once the scene changed. After
    such a failure that camera makes no synchronous brain call for `refresh_s`
    (backoff, same fallback meanwhile); any successful inventory answer, a
    synchronous one or a background refresh, ends it.

    Safe to call from several threads at once: the cache lock is never held
    across the brain or SAM call, and the segmenter serializes its own model
    calls.
    """

    concurrent_safe = True

    def __init__(self, *, base_url: str, model: str, api_key_file: str | None = None,
                 timeout_s: float = 20.0, max_tokens: int = 1500, segmenter: BoxSegmenter | None = None,
                 refresh_s: float = 4.0, max_cache_age_s: float = 30.0, scene_change: float = 6.0,
                 client=None, clock=time.monotonic):
        self._base_url = str(base_url)
        self._model = str(model)
        self._timeout_s = float(timeout_s)
        self._max_tokens = int(max_tokens)
        self._segmenter = segmenter
        self._refresh_s = float(refresh_s)
        self._max_cache_age_s = float(max_cache_age_s)
        self._scene_change = float(scene_change)
        self._clock = clock
        self._api_key = self._read_key(api_key_file)
        self._client = client
        self._client_lock = threading.Lock()
        # Guards the cache, the refresh bookkeeping and the warning clock only.
        self._lock = threading.Lock()
        self._cache: dict[tuple[int, int, int, int], _Entry] = {}
        self._refreshing: set[tuple[int, int, int, int]] = set()
        # Clock time of the last failed synchronous inventory per camera (backoff).
        self._failed_at: dict[tuple[int, int, int, int], float] = {}
        self._threads: set[threading.Thread] = set()
        self._closed = False
        self._last_warn: float | None = None

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
            return self._inventory(frame)
        except Exception as e:  # noqa: BLE001 - booth rule: degrade, never crash the caller
            self._warn("detection failed; returning no detections", e)
            return []

    def close(self) -> None:
        """Join the refresh threads in flight; later refreshes do not start."""
        with self._lock:
            self._closed = True
            threads = list(self._threads)
        for thread in threads:
            join_thread(thread, what="brain detector", timeout_s=self._timeout_s + _CLOSE_MARGIN_S)

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
        h, w = frame.rgb.shape[:2]
        image_url = _jpeg_data_url(frame)
        labels: list[str] = []
        boxes: list[np.ndarray] = []
        for q in queries:
            for _, box in parse_boxes(self._ask(image_url, REFER_PROMPT.format(q=q)), w, h):
                labels.append(q)
                boxes.append(box)
        return self._detections(frame, labels, np.array(boxes, dtype=np.float32).reshape(-1, 4), track=False)

    def _ask_inventory(self, frame: Frame, thumb: np.ndarray, t: float) -> _Entry:
        h, w = frame.rgb.shape[:2]
        found = parse_boxes(self._ask(_jpeg_data_url(frame), INVENTORY_PROMPT), w, h)
        boxes = np.array([box for _, box in found], dtype=np.float32).reshape(-1, 4)
        boxes.setflags(write=False)
        return _Entry(tuple(label for label, _ in found), boxes, thumb, t)

    def _store(self, key, entry: _Entry) -> None:
        """A successful inventory answer (synchronous or refresh): the brain is
        up, so the camera's backoff ends; the entry is stored unless the cached
        one comes from a newer frame."""
        with self._lock:
            self._failed_at.pop(key, None)
            old = self._cache.get(key)
            # A slow refresh must not overwrite an entry from a newer frame.
            if old is None or entry.t >= old.t:
                self._cache[key] = entry

    def _inventory(self, frame: Frame) -> list[Detection]:
        key = _camera_key(frame)
        thumb = _thumbnail(frame)
        now = self._clock()
        with self._lock:
            entry = self._cache.get(key)
            failed_at = self._failed_at.get(key)
        if (entry is not None and now - entry.t <= self._max_cache_age_s
                and _scene_diff(entry.thumb, thumb) <= self._scene_change):
            if now - entry.t > self._refresh_s:
                self._start_refresh(key, frame, thumb, now)
            return self._detections(frame, entry.labels, entry.boxes, track=True)
        if failed_at is not None and now - failed_at < self._refresh_s:
            return self._track_known(frame, self._known_entry(key, thumb))  # backoff: no brain call
        try:
            entry = self._ask_inventory(frame, thumb, now)
        except Exception as e:  # noqa: BLE001 - booth rule: keep tracking, never raise
            with self._lock:
                self._failed_at[key] = self._clock()
            known = self._known_entry(key, thumb)
            if known is None:
                self._warn("inventory failed; nothing known for this scene, returning no detections", e)
            else:
                self._warn("inventory failed; tracking the known objects", e)
            return self._track_known(frame, known)
        self._store(key, entry)
        return self._detections(frame, entry.labels, entry.boxes, track=False)

    def _known_entry(self, key, thumb: np.ndarray) -> _Entry | None:
        """Outage fallback: the cached entry at ANY age while this frame still
        shows its scene; None once the scene changed (old boxes would put old
        labels on new things; refer bypasses the cache, so picking is unaffected)."""
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
        with self._lock:
            if self._closed or key in self._refreshing:
                return
            thread = threading.Thread(target=self._refresh, args=(key, frame, thumb, t),
                                      name=f"brain-refresh-{key}", daemon=True)
            # Started under the lock: close() never misses it, and the thread
            # cannot reach its own bookkeeping before this block has recorded it.
            thread.start()
            self._refreshing.add(key)
            self._threads.add(thread)

    def _refresh(self, key, frame: Frame, thumb: np.ndarray, t: float) -> None:
        try:
            # Replaces the entry only on success; a failure keeps the old one,
            # which still ages out after max_cache_age_s.
            self._store(key, self._ask_inventory(frame, thumb, t))
        except Exception as e:  # noqa: BLE001 - background thread: log, never raise
            self._warn("inventory refresh failed; keeping the cached boxes", e)
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
