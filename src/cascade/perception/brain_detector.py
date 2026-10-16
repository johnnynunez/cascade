"""Brain-driven perception: the local vision LLM (the brain) boxes objects, SAM masks them.

This module holds the two prompts measured in the perception bake-off, the parser
that turns the brain's text reply into labelled pixel boxes, and `BrainDetector`,
which asks the brain for those boxes and segments them with a box-prompted SAM.

Wire format: the brain answers a JSON list `[{"label": ..., "bbox_2d": [x1, y1, x2, y2]}]`
with coordinates normalized to 0-1000 of the full image.
"""

from __future__ import annotations

import base64
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
#: close() waits this long beyond the background request timeout for a
#: refresh to finish (after cancelling it, a stream ends at its next chunk).
_CLOSE_MARGIN_S = 5.0
#: The outermost foreground() scope waits at most this long for the cancelled
#: background request to close its stream before the foreground work starts
#: (BEFORE the skill takes its frame, so it never ages that frame). A request
#: still waiting for its first chunk (prefill) may outlast it.
_CANCEL_WAIT_S = 0.5
#: close() waits this long for the SAM warm-up (weights load + the first,
#: cold CUDA predict): the same minute thread_join.EXIT_JOIN_S allows the
#: slowest cold tick measured. Past it, shutdown names the thread instead.
_WARM_UP_JOIN_S = 60.0


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

    Foreground scope (`foreground()`, re-entrant; the runtime holds it for
    motion skills, preview_grasp and localize_object): background work never
    competes with a foreground localization for the brain or SAM. While it
    is held, no background refresh starts (requests still queue, keeping
    their latest frame), the one in flight is cancelled (background requests
    stream with `inventory_timeout_s`; a cancelled stream is closed, which
    aborts the request on the server, and goes back to the queue: no
    backoff, no warning), and `detect_nowait` returns [] without a SAM call.
    When the outermost scope ends, the oldest waiting request starts.
    Foreground refers stay single non-streaming requests (`timeout_s`).

    Safe to call from several threads at once: the cache lock is never held
    across the brain or SAM call, and the segmenter serializes its own model
    calls.
    """

    concurrent_safe = True
    #: world.WorldWatcher asks a non-fusing camera (fuse=False: the sim's proof
    #: camera, the rig's wrist) nothing: no background inventory, no SAM step.
    background_fusing_cameras_only = True

    def __init__(self, *, base_url: str, model: str, api_key_file: str | None = None,
                 timeout_s: float = 20.0, max_tokens: int = 1500, segmenter: BoxSegmenter | None = None,
                 refresh_s: float = 4.0, max_cache_age_s: float = 30.0, scene_change: float = 6.0,
                 idle_refresh_s: float = 30.0, inventory_timeout_s: float | None = None,
                 client=None, clock=time.monotonic):
        self._base_url = str(base_url)
        self._model = str(model)
        self._timeout_s = float(timeout_s)
        # Background (streamed) requests: the whole request, first byte to last
        # chunk. Defaults to timeout_s; a slow brain's inventory needs more.
        self._inventory_timeout_s = float(timeout_s if inventory_timeout_s is None else inventory_timeout_s)
        self._max_tokens = int(max_tokens)
        self._segmenter = segmenter
        self._refresh_s = float(refresh_s)
        self._idle_refresh_s = float(idle_refresh_s)
        self._max_cache_age_s = float(max_cache_age_s)
        self._scene_change = float(scene_change)
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

    @contextlib.contextmanager
    def foreground(self):
        """Hold off background work while a foreground localization runs
        (re-entrant, from any thread; see the class docstring). Entering
        cancels the background request in flight and, in the outermost
        scope, waits up to `_CANCEL_WAIT_S` for its stream to close. Leaving
        the outermost scope (also by an exception) starts the oldest waiting
        refresh."""
        with self._lock:
            self._foreground += 1
            outermost = self._foreground == 1
            for cancel in self._cancels.values():
                cancel.set()
            threads = list(self._threads)
        try:
            if outermost and threads:
                deadline = time.monotonic() + _CANCEL_WAIT_S
                for thread in threads:
                    thread.join(max(0.0, deadline - time.monotonic()))
            yield
        finally:
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

    def _ask(self, image_url: str, prompt: str, cancel: threading.Event | None = None) -> str:
        """One brain request. Foreground (`cancel` None): a single
        non-streaming completion under `timeout_s`. Background (a refresh's
        cancel Event): streamed under `inventory_timeout_s`, see
        `_ask_streamed`."""
        request = dict(
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
        if cancel is None:
            choice = self._get_client().chat.completions.create(**request).choices[0]
            text, finish_reason = choice.message.content or "", getattr(choice, "finish_reason", None)
        else:
            text, finish_reason = self._ask_streamed(request, cancel)
        # A garbled reply is a failed call, never an empty table: cut at
        # max_tokens (even if a list closed before the cut), or neither a box
        # list nor a reply opening with "[]" (what both prompts ask for when
        # nothing is there).
        if finish_reason == "length":
            raise ValueError(f"garbled reply, truncated at max_tokens={self._max_tokens}")
        if not _is_answer(text):
            raise ValueError(f"garbled reply, no box list: {text[:80]!r}")
        return text

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

    def _ask_refer(self, frame: Frame, queries, cancel: threading.Event | None = None) -> tuple[list[str], np.ndarray]:
        """One request per query, concatenated; labels are the queries."""
        h, w = frame.rgb.shape[:2]
        image_url = _jpeg_data_url(frame)
        labels: list[str] = []
        boxes: list[np.ndarray] = []
        for q in queries:
            for _, box in parse_boxes(self._ask(image_url, REFER_PROMPT.format(q=q), cancel), w, h):
                labels.append(q)
                boxes.append(box)
        return labels, np.array(boxes, dtype=np.float32).reshape(-1, 4)

    def _ask_inventory(self, frame: Frame, thumb: np.ndarray, t: float,
                       cancel: threading.Event | None = None) -> _Entry:
        h, w = frame.rgb.shape[:2]
        found = parse_boxes(self._ask(_jpeg_data_url(frame), INVENTORY_PROMPT, cancel), w, h)
        boxes = np.array([box for _, box in found], dtype=np.float32).reshape(-1, 4)
        boxes.setflags(write=False)
        return _Entry(tuple(label for label, _ in found), boxes, thumb, t)

    def _ask_entry(self, key, frame: Frame, thumb: np.ndarray, t: float,
                   cancel: threading.Event | None = None) -> _Entry:
        """The brain call behind a cache key: the inventory prompt for
        (camera, None), the refer loop for (camera, queries). `cancel`: a
        background refresh (streamed, cancellable)."""
        queries = key[1]
        if queries is None:
            return self._ask_inventory(frame, thumb, t, cancel)
        labels, boxes = self._ask_refer(frame, queries, cancel)
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
        starts, when an answer for it is stored, or on close(). Inside a
        foreground() scope the request only queues (see `_start_next`).
        """
        with self._lock:
            if self._closed or key in self._refreshing or self._backs_off(key, t):
                return
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
        except _Cancelled:
            # A foreground scope (or close) took the brain: not a failure. The
            # request goes back to the queue with its place and frame; a later
            # ask for the key updates the frame. close() drops it instead.
            cancelled = True
            with self._lock:
                if not self._closed:
                    self._waiting.setdefault(key, request)
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
