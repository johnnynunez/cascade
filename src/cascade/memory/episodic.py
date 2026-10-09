"""Short-horizon episodic memory (the 10-15 s window the agent can consult).

A time-pruned ring of events: keyframes (as JPEG thumbnails), detections,
actions and outcomes, plus an optional TurboQuant-compressed embedding per
event for similarity recall. Two views feed the agent each turn:

- ``digest()`` -- compact, chronological, plain text;
- ``memory_frames(k)`` -- the Vesta memory harness (arXiv:2606.20905 §2.4):
  up to K PAST frames, each captioned with its step index, age, the action
  taken and the independent verdict on it. Vesta's ablation (Table 5) is the
  reason both exist: text-only history scored 49.7 on their planner suite,
  image-only 63.1, image+text 75.9 -- a text-only planner "learns to be
  overly reliant on the history text shortcuts" and keeps predicting
  "continue the current task". The first frame is always retained (initial
  state); the rest are sampled uniformly. Vesta found uniform and
  recency-biased sampling on par, so the simple one is used.

Thread-safe: the skill thread writes while dashboard /state handlers and the
MCP world_state tool read concurrently (deque iteration during popleft
raises RuntimeError without the lock -- reproduced in review 2026-07-18).

Visual recall (ROADMAP #7, opt-in): with ``embedder=`` every frame-carrying
event and every object crop handed to ``add(crops=...)`` is embedded into the
TurboQuant index, and ``recall_visual`` answers "which remembered frame or
object looks like this" (an image, always) or "... like X" (a text, only when
the embedder has a JOINT image-text space). Index entries live in their own
task-scale ring (``frame_horizon_s``, at most ``max_visual`` entries) and are
dropped from the index with it -- before this, explicitly embedded events were
never removed from the index at all. A recall hit is a remembered appearance,
not a current observation: it never confirms an outcome and never aims motion.
Without an embedder nothing is embedded and every pre-existing path is
unchanged.

Visual recall v2 (B43, both opt-in, both need an embedder):

- ``add_detection_crop`` / :class:`DetectionCropRecorder` -- the WorldWatcher
  also indexes what each COMMITTED detection looked like (fusion is paused
  while the arm moves, so a motion frame never yields one), deduplicated per
  belief and rate-limited, in a ``detection`` ring of its own
  (``max_detections``) so it can never evict a motion frame or a localized
  crop. Its events stay out of the text and frame rings.
- ``save_visual`` / ``load_visual`` -- the index survives a restart, modelled
  on ``BeliefStore.save/load``: wall-clock timestamps, entries past the max
  age dropped BEFORE a minimum apparent age (``LOADED_MIN_AGE_S``) is applied,
  atomic temp + ``os.replace`` writes, and a file from another embedder is
  refused (vectors of two models are not comparable). Restored entries keep
  their own ring, pruned at the load's max age rather than the task horizon,
  and every restored hit carries ``restored: True, state: "remembered"``.
"""

from __future__ import annotations

import base64
import json
import math
import os
import threading
import time
import weakref
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .vector_index import QuantizedIndex


def _unavailable():
    from .embedder import EmbedderUnavailable

    return EmbedderUnavailable(
        "no memory embedder configured (memory.embedder.backend: none); "
        "visual recall needs memory.embedder.backend: hash | siglip | clip")


@dataclass
class MemoryEvent:
    t: float
    kind: str  # "observation" | "action" | "outcome" | "note"
    text: str  # one-line human/agent readable summary
    data: dict[str, Any] = field(default_factory=dict)
    thumb_jpeg: bytes | None = None


@dataclass
class VisualEntry:
    """One vector in the episodic index: a whole frame, an object crop, or an
    embedding the caller supplied, and the event it belongs to."""

    t: float
    kind: str  # "frame" | "object" | "event" | "detection"
    label: str | None
    event: MemoryEvent
    #: Loaded from an earlier session (B43): remembered, never current.
    restored: bool = False


def crop_bbox(rgb: np.ndarray, bbox, min_px: int = 2) -> np.ndarray | None:
    """A copy of the pixels inside `bbox` (x0, y0, x1, y1), clipped to the
    image; None without a box or when fewer than `min_px` remain on a side."""
    if bbox is None:
        return None
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = (int(round(float(v))) for v in np.asarray(bbox).reshape(-1)[:4])
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < min_px or y1 - y0 < min_px:
        return None
    return rgb[y0:y1, x0:x1].copy()


class DetectionCropRecorder:
    """Feeds the visual index from the WorldWatcher (B43, opt-in
    ``memory.visual_recall_detections``).

    The watcher calls :meth:`offer` with the observations of a frame it has
    just COMMITTED to the belief store and the belief each one was fused into
    (``BeliefStore.update_frame``'s return). Per call: one crop per belief
    (a second name over the same pixels is the same object), none for a
    belief cropped less than `interval_s` ago, at most `max_per_tick` --
    so the embedding cost per watcher tick is bounded and a static table
    is not re-embedded at 3 Hz. Beliefs are tracked by identity through a
    weak reference: a belief that died never shadows a new one that reuses
    its id, and the bookkeeping never keeps a belief alive.
    """

    def __init__(self, memory, *, interval_s: float = 30.0, max_per_tick: int = 2):
        if getattr(memory, "embedder", None) is None:
            raise ValueError("detection crops need a memory embedder "
                             "(memory.embedder.backend: hash | siglip | clip)")
        interval = float(interval_s)
        if not math.isfinite(interval) or interval < 0.0:
            raise ValueError(f"visual_recall_interval_s must be finite and >= 0, got {interval_s!r}")
        if isinstance(max_per_tick, bool) or int(max_per_tick) != max_per_tick or max_per_tick < 1:
            raise ValueError(f"visual_recall_max_per_tick must be an integer >= 1, got {max_per_tick!r}")
        self.memory = memory
        self.interval_s = interval
        self.max_per_tick = int(max_per_tick)
        #: id(belief) -> (weakref to it, when it was last cropped)
        self._last: dict[int, tuple[weakref.ref, float]] = {}

    @staticmethod
    def describe(obs, source: str | None = None) -> str:
        """The recall line of a watcher crop: what, where, which camera."""
        label = str(obs.label)
        color = getattr(obs, "color", None)
        name = label if not color or color in label.split() else f"{color} {label}"
        x, y, z = (float(v) for v in np.asarray(obs.position, dtype=float).reshape(3))
        text = f"watcher saw {name} at ({x:.2f}, {y:.2f}, {z:.2f}) m"
        return text + (f" [{source}]" if source else "")

    def offer(self, rgb: np.ndarray, observations, beliefs, *, source: str | None = None) -> int:
        """Index the crops that are due; returns how many were indexed."""
        now = self.memory._clock()
        # Forget dead beliefs and expired intervals: bounded by live beliefs.
        self._last = {k: (ref, t) for k, (ref, t) in self._last.items()
                      if ref() is not None and now - t < self.interval_s}
        picked = []
        seen: set[int] = set()
        for obs, belief in zip(observations, beliefs):
            if belief is None or id(belief) in seen or id(belief) in self._last:
                continue
            crop = crop_bbox(rgb, getattr(obs, "bbox", None))
            if crop is None:
                continue
            seen.add(id(belief))
            picked.append((obs, belief, crop))
            if len(picked) >= self.max_per_tick:
                break
        indexed = 0
        for obs, belief, crop in picked:
            self._last[id(belief)] = (weakref.ref(belief), now)
            indexed += bool(self.memory.add_detection_crop(
                str(obs.label), crop, text=self.describe(obs, source), t=now))
        return indexed


class EpisodicMemory:
    #: Persisted visual entries older than this (wall clock) are dropped on
    #: load (B43) -- the same default as ``BeliefStore.DEFAULT_MAX_AGE_S``.
    DEFAULT_MAX_AGE_S = 6 * 3600.0
    #: Floor on the apparent age of anything restored from disk, the
    #: ``BeliefStore.LOADED_MIN_AGE_S`` rule: nothing restored reads as
    #: just seen.
    LOADED_MIN_AGE_S = 2.0
    #: Entry kinds whose vectors the embedder produced: only these persist (a
    #: caller-supplied ``embedding=`` lives in a space nobody declared).
    PERSISTED_KINDS = ("frame", "object", "detection")

    def __init__(
        self,
        horizon_s: float = 15.0,
        max_events: int = 400,
        embed_dim: int | None = None,
        embed_bits: int = 4,
        thumb_width: int = 320,
        clock=time.monotonic,
        frame_horizon_s: float = 600.0,
        max_frames: int = 64,
        embedder=None,
        max_visual: int = 512,
        max_detections: int = 128,
    ):
        self.horizon_s = horizon_s
        self._events: deque[MemoryEvent] = deque(maxlen=max_events)
        #: Frame-carrying events live in their own ring with a TASK-scale
        #: horizon. The text ring is a 15 s situational window; a memory
        #: harness pruned at 15 s would forget the initial state before the
        #: first pick finished (a pick takes ~20 s here) and could never show
        #: "which drawers did I already open" ten steps later. The
        #: orchestrator resets it per episode (Vesta: plan.ResetSession()).
        self.frame_horizon_s = frame_horizon_s
        self._frames: deque[MemoryEvent] = deque(maxlen=max_frames)
        self._thumb_width = thumb_width
        #: Optional image/text embedder (memory/embedder.py). None = the
        #: pre-ROADMAP-#7 memory exactly: nothing is embedded implicitly.
        self.embedder = embedder
        if embedder is not None:
            dim = int(embedder.dim)
            if embed_dim is not None and int(embed_dim) != dim:
                raise ValueError(
                    f"embed_dim={embed_dim} disagrees with the embedder's dim {dim} ({embedder.name})")
            embed_dim = dim
        self._index = QuantizedIndex(embed_dim, bits=embed_bits) if embed_dim else None
        #: Index entries in index order, pruned with the frame horizon and the
        #: `max_visual` cap; the index itself is pruned in lockstep.
        self.max_visual = int(max_visual)
        self._visual: deque[VisualEntry] = deque()
        #: B43 rings, both empty unless their opt-in feature feeds them:
        #: watcher-detection crops (``add_detection_crop``, at most
        #: `max_detections`, task-scale horizon) and entries restored from an
        #: earlier session (``load_visual``, at most `max_visual`, pruned at
        #: the load's max age). Each has its own index, created on first use.
        self.max_detections = int(max_detections)
        self._embed_bits = embed_bits
        self._detections: deque[VisualEntry] = deque()
        self._det_index: QuantizedIndex | None = None
        self._restored: deque[VisualEntry] = deque()
        self._restored_index: QuantizedIndex | None = None
        self._restored_max_age_s = self.DEFAULT_MAX_AGE_S
        self.embed_errors = 0
        self._clock = clock
        self._lock = threading.RLock()

    # ── recording ────────────────────────────────────────────────────────

    def add(
        self,
        kind: str,
        text: str,
        data: dict[str, Any] | None = None,
        rgb: np.ndarray | None = None,
        embedding: np.ndarray | None = None,
        t: float | None = None,
        thumb_jpeg: bytes | None = None,
        crops: dict[str, np.ndarray] | None = None,
    ) -> MemoryEvent:
        """Record one event. `rgb` is downscaled to a thumbnail; `thumb_jpeg`
        attaches an already-encoded one (the runtime reuses the AFTER
        keyframe it just wrote for the trace, so a frame is encoded once).

        With an embedder, `rgb` is also embedded as a ``frame`` entry and each
        of `crops` ({label: BGR crop}) as an ``object`` entry. An embedder
        fault is counted (``visual_stats``), never raised: memory bookkeeping
        must not fail the skill that is reporting into it."""
        now = self._clock() if t is None else t
        thumb = thumb_jpeg
        if rgb is not None and thumb is None:
            h, w = rgb.shape[:2]
            scale = self._thumb_width / w
            small = cv2.resize(rgb, (self._thumb_width, max(1, int(h * scale))))
            ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
            thumb = buf.tobytes() if ok else None
        # Embed OUTSIDE the lock: a model forward pass must not block readers.
        vectors: list[tuple[str, str | None, np.ndarray]] = []
        errors = 0
        if self.embedder is not None:
            if rgb is not None:
                try:
                    vectors.append(("frame", None, self.embedder.embed_image(rgb)))
                except Exception:  # noqa: BLE001
                    errors += 1
            for label, crop in (crops or {}).items():
                try:
                    vectors.append(("object", str(label), self.embedder.embed_image(crop)))
                except Exception:  # noqa: BLE001
                    errors += 1
        if embedding is not None:
            vectors.append(("event", None, embedding))
        ev = MemoryEvent(t=now, kind=kind, text=text, data=data or {}, thumb_jpeg=thumb)
        with self._lock:
            self._events.append(ev)
            if thumb is not None:
                self._frames.append(ev)
            self.embed_errors += errors
            if self._index is not None:
                for vkind, label, vec in vectors:
                    entry = VisualEntry(t=now, kind=vkind, label=label, event=ev)
                    self._index.add(vec, meta=entry)
                    self._visual.append(entry)
            self.prune(now)
        return ev

    def _new_index(self) -> QuantizedIndex:
        """A fresh index in the embedder's space (same quantizer settings, so
        scores from every ring are comparable)."""
        return QuantizedIndex(int(self.embedder.dim), bits=self._embed_bits)

    def add_detection_crop(self, label: str, crop: np.ndarray, *, text: str,
                           t: float | None = None) -> bool:
        """Index one WATCHER detection crop as a ``detection`` entry (B43,
        opt-in; fed by :class:`DetectionCropRecorder`).

        Its event (`text`: what was seen, where, by which camera) belongs to
        the visual index only: the text digest and the memory frames are
        unchanged. The ring is capped at `max_detections` and pruned with the
        task-scale horizon, apart from the session ring, so the always-on
        watcher cannot evict a motion frame or a localized crop. Returns
        whether a vector was indexed; an embedder fault is counted
        (``visual_stats``), never raised."""
        if self.embedder is None:
            raise _unavailable()
        now = self._clock() if t is None else t
        try:  # outside the lock, like add(): a forward pass must not block readers
            vec = self.embedder.embed_image(crop)
        except Exception:  # noqa: BLE001
            with self._lock:
                self.embed_errors += 1
            return False
        entry = VisualEntry(t=now, kind="detection", label=str(label),
                            event=MemoryEvent(t=now, kind="observation", text=str(text)))
        with self._lock:
            if self._det_index is None:
                self._det_index = self._new_index()
            self._det_index.add(vec, meta=entry)
            self._detections.append(entry)
            self.prune(now)
        return True

    @staticmethod
    def _prune_ring(ring: deque, index: QuantizedIndex | None, now: float,
                    horizon_s: float, cap: int) -> None:
        """Expire a time-ordered ring and its index in lockstep: entries past
        `horizon_s` are a prefix, and so is the overflow past `cap`."""
        drop = 0
        n = len(ring)
        while drop < n and now - ring[drop].t > horizon_s:
            drop += 1
        drop = max(drop, n - cap)
        if drop > 0 and index is not None:
            index.remove_ids(set(range(drop)))
            for _ in range(drop):
                ring.popleft()

    def prune(self, now: float | None = None) -> None:
        now = self._clock() if now is None else now
        with self._lock:
            while self._events and now - self._events[0].t > self.horizon_s:
                self._events.popleft()
            while self._frames and now - self._frames[0].t > self.frame_horizon_s:
                self._frames.popleft()
            # Visual entries are appended in time order, so expiry is a prefix
            # of the deque -- and of the index, which is pruned in lockstep.
            self._prune_ring(self._visual, self._index, now, self.frame_horizon_s, self.max_visual)
            self._prune_ring(self._detections, self._det_index, now, self.frame_horizon_s,
                             self.max_detections)
            # Restored entries: their own horizon (the load's max age, never
            # below the restored-age floor), not the task-scale one -- they
            # are older than any task by construction.
            self._prune_ring(self._restored, self._restored_index, now,
                             max(self._restored_max_age_s, self.LOADED_MIN_AGE_S), self.max_visual)

    def reset_frames(self) -> None:
        """New episode: drop the visual history. The text ring is untouched
        (it is a rolling situational window, not per-task state)."""
        with self._lock:
            self._frames.clear()

    # ── recall ───────────────────────────────────────────────────────────

    def events(self, kinds: tuple[str, ...] | None = None) -> list[MemoryEvent]:
        with self._lock:
            self.prune()
            evs = list(self._events)
        if kinds:
            evs = [e for e in evs if e.kind in kinds]
        return evs

    def recall_similar(self, embedding: np.ndarray, k: int = 3) -> list[MemoryEvent]:
        if self._index is None:
            return []
        with self._lock:
            hits = self._index.search(embedding, k=k)
            live = {id(e) for e in self._events}
        return [meta.event for _, meta in hits if id(meta.event) in live]

    def recall_visual(self, query, k: int = 3, *, min_sim: float | None = None,
                      kinds: tuple[str, ...] | None = None, now: float | None = None) -> list[dict]:
        """Remembered frames/objects that look like `query`, best first.

        `query` is a BGR image (always allowed), a text (only with a JOINT
        image-text embedder; its default floor is the embedder's
        ``text_image_floor``) or a ready vector. Each hit is
        ``{"score", "kind", "label", "age_s", "text", "verdict", "step"}``: the
        cosine, ``frame``/``object``/``event``, the object label for crops, how
        long ago, the recorded action line and verdict of the event it came
        from, and its memory-frame step while that frame is still in the
        per-task ring (else None). Visual entries keep their own task-scale
        horizon and survive the per-task ``reset_frames``. Advisory: a hit
        says what something LOOKED like, never where it is now or whether an
        action worked.
        """
        if isinstance(query, str):
            if self.embedder is None:
                raise _unavailable()
            if not getattr(self.embedder, "joint_space", False):
                raise ValueError(
                    f"recall by description needs a joint image-text embedder (siglip/clip); "
                    f"{self.embedder.name!r} embeds text and images in different spaces")
            vec = self.embedder.embed_text(query)
            if min_sim is None:
                min_sim = getattr(self.embedder, "text_image_floor", None)
        else:
            arr = np.asarray(query)
            if arr.ndim >= 2:
                if self.embedder is None:
                    raise _unavailable()
                vec = self.embedder.embed_image(arr)
            else:
                vec = arr
        if self._index is None:
            raise _unavailable()
        now = self._clock() if now is None else now
        with self._lock:
            self.prune(now)
            hits = self._index.search(vec, k=len(self._index))
            # B43 rings (watcher crops, restored entries) ranked together with
            # the session ring; a stable sort keeps the session's own order,
            # and without them this is exactly the session search.
            for index in (self._det_index, self._restored_index):
                if index is not None:
                    hits = sorted(hits + index.search(vec, k=len(index)), key=lambda h: -h[0])
            frame_steps = {id(ev): n + 1 for n, ev in enumerate(self._frames)}
        out: list[dict] = []
        for score, entry in hits:
            if kinds and entry.kind not in kinds:
                continue
            if min_sim is not None and score < float(min_sim):
                break  # hits are sorted: everything after is lower
            ev = entry.event
            hit = {
                "score": round(float(score), 3),
                "kind": entry.kind,
                "label": entry.label,
                "age_s": round(now - entry.t, 1),
                "text": ev.text,
                "verdict": str((ev.data or {}).get("verdict") or ""),
                "step": frame_steps.get(id(ev)),
            }
            if entry.restored:
                # From an earlier session: remembered, never seen in this one
                # (its age is at least LOADED_MIN_AGE_S by construction).
                hit["restored"] = True
                hit["state"] = "remembered"
            out.append(hit)
            if len(out) >= k:
                break
        return out

    def visual_stats(self) -> dict:
        """What the visual index holds right now (for status pages and tests)."""
        with self._lock:
            return {
                "embedder": getattr(self.embedder, "name", None),
                "joint_space": bool(getattr(self.embedder, "joint_space", False)),
                "entries": len(self._visual) + len(self._detections) + len(self._restored),
                "embed_errors": int(self.embed_errors),
            }

    def visual_breakdown(self) -> dict:
        """Entries per ring: this session's frames/crops, watcher crops, and
        entries restored from an earlier session (B43)."""
        with self._lock:
            return {"session": len(self._visual), "detections": len(self._detections),
                    "restored": len(self._restored)}

    # ── persistence (B43, opt-in memory.persist_episodic) ────────────────

    def save_visual(self, path, *, now_wall: float | None = None) -> int:
        """Persist the visual index; returns how many entries were written.

        Modelled on ``BeliefStore.save``: monotonic timestamps become WALL
        CLOCK (the monotonic origin resets with the process), and the write is
        atomic (temp file + ``os.replace``) because the watcher may be adding
        crops meanwhile. Each entry stores its kind, label, recall line,
        verdict and the vector the index holds, decoded to the embedder's own
        space (float16) so the file does not depend on the quantizer's
        rotation. A caller's explicit ``embedding=`` is not persisted."""
        if self.embedder is None or self._index is None:
            raise _unavailable()
        path = Path(path)
        now_wall = time.time() if now_wall is None else float(now_wall)
        records = []
        with self._lock:
            now = self._clock()
            self.prune(now)
            for ring, index in ((self._restored, self._restored_index), (self._visual, self._index),
                                (self._detections, self._det_index)):
                if index is None:
                    continue
                for entry, vec in zip(ring, index.decoded()):
                    if entry.kind not in self.PERSISTED_KINDS:
                        continue
                    records.append({
                        "kind": entry.kind,
                        "label": entry.label,
                        "text": entry.event.text,
                        "verdict": str((entry.event.data or {}).get("verdict") or ""),
                        "wall": now_wall - (now - entry.t),
                        "vec": base64.b64encode(np.asarray(vec, np.float16).tobytes()).decode("ascii"),
                    })
        records.sort(key=lambda r: r["wall"])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps({"version": 1, "embedder": self.embedder.name,
                                   "dim": int(self.embedder.dim), "saved_wall": now_wall,
                                   "entries": records}))
        try:
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return len(records)

    def load_visual(self, path, max_age_s: float | None = None, *,
                    now_wall: float | None = None) -> int:
        """Restore entries saved by :meth:`save_visual`; returns how many.

        Modelled on ``BeliefStore.load``: ages are wall clock relative to now,
        entries older than `max_age_s` (default ``DEFAULT_MAX_AGE_S``) are
        dropped, and only THEN is the apparent age floored at
        ``LOADED_MIN_AGE_S`` (a clock that went backwards reads as "just
        saved"), so the floor cannot smuggle in an expired entry and nothing
        restored reads as current. A load replaces the restored ring (newest
        `max_visual` kept). A missing or corrupt file, or a bad record, is
        skipped -- a broken memory must not stop the robot. A file written by
        ANOTHER embedder (name or dimension) raises ValueError: its vectors
        live in a different space and would answer recall with noise."""
        if self.embedder is None or self._index is None:
            raise _unavailable()
        path = Path(path)
        try:
            blob = json.loads(path.read_text())
            records = list(blob["entries"])
            name, dim = blob.get("embedder"), int(blob.get("dim"))
        except Exception:  # noqa: BLE001 - missing or corrupt: never block startup
            return 0
        if name != self.embedder.name:
            raise ValueError(f"{path} was embedded by {name!r}, this memory uses "
                             f"{self.embedder.name!r}: not comparable, not loaded")
        if dim != int(self.embedder.dim):
            raise ValueError(f"{path} holds dim {dim} vectors, {self.embedder.name!r} embeds "
                             f"dim {self.embedder.dim}: not loaded")
        max_age_s = self.DEFAULT_MAX_AGE_S if max_age_s is None else float(max_age_s)
        now_wall = time.time() if now_wall is None else float(now_wall)
        now = self._clock()
        loaded: list[tuple[VisualEntry, np.ndarray]] = []
        for r in records:
            try:
                kind = str(r["kind"])
                vec = np.frombuffer(base64.b64decode(r["vec"]), dtype=np.float16).astype(np.float32)
                if kind not in self.PERSISTED_KINDS or vec.shape[0] != dim:
                    continue
                age = now_wall - float(r["wall"])
                if age > max_age_s:
                    continue
                # AFTER the max-age test; it also turns the negative age of a
                # clock that went backwards into "just saved", never the future
                age = max(age, self.LOADED_MIN_AGE_S)
                verdict = str(r.get("verdict") or "")
                event = MemoryEvent(t=now - age, kind="observation", text=str(r.get("text") or ""),
                                    data={"verdict": verdict} if verdict else {})
                label = r.get("label")
                loaded.append((VisualEntry(t=now - age, kind=kind,
                                           label=None if label is None else str(label),
                                           event=event, restored=True), vec))
            except Exception:  # noqa: BLE001 - skip a bad record, keep the rest
                continue
        loaded.sort(key=lambda pair: pair[0].t)
        loaded = loaded[max(0, len(loaded) - self.max_visual):]
        index = self._new_index()
        for entry, vec in loaded:
            index.add(vec, meta=entry)
        with self._lock:
            self._restored = deque(entry for entry, _ in loaded)
            self._restored_index = index
            self._restored_max_age_s = max_age_s
            self.prune(now)
        return len(loaded)

    def memory_frames(self, k: int = 4, now: float | None = None) -> list[dict]:
        """Vesta-style visual history: up to `k` past events that carry a
        thumbnail, oldest first, each as
        ``{"step", "age_s", "kind", "text", "verdict", "jpeg"}``.

        Sampling (arXiv:2606.20905 §2.4): the FIRST frame is always kept --
        it is the initial state the task is measured against -- and the
        remaining k-1 slots are spread uniformly over the rest, so the newest
        frame is always the last entry. `step` is the event's ordinal among
        frame-carrying events, the "step index i" of Vesta's memory tuple.
        `verdict` is the independent postcondition status recorded with the
        action (confirmed / refuted / unverified) or "" when there was none.
        """
        now = self._clock() if now is None else now
        with self._lock:
            self.prune(now)
            framed = list(enumerate(self._frames))
        if not framed or k <= 0:
            return []
        if len(framed) > k:
            if k == 1:
                picked = [framed[-1]]
            else:
                rest = framed[1:]
                # k-1 slots over `rest`, always ending on the newest
                idx = [round(j * (len(rest) - 1) / (k - 2)) for j in range(k - 1)] if k > 2 else [len(rest) - 1]
                picked = [framed[0]] + [rest[i] for i in sorted(set(idx))]
        else:
            picked = framed
        steps = {id(ev): n + 1 for n, (_, ev) in enumerate(framed)}
        return [
            {
                "step": steps[id(ev)],
                "age_s": round(now - ev.t, 1),
                "kind": ev.kind,
                "text": ev.text,
                "verdict": str((ev.data or {}).get("verdict") or ""),
                "jpeg": ev.thumb_jpeg,
            }
            for _, ev in picked
        ]

    @staticmethod
    def frame_caption(fr: dict) -> str:
        """One line the planner reads next to a memory frame."""
        v = f" [{fr['verdict'].upper()}]" if fr.get("verdict") else ""
        return f"memory frame {fr['step']} ({fr['age_s']:.1f}s ago): {fr['text']}{v}"

    def last_frame_jpeg(self) -> bytes | None:
        with self._lock:
            if self._frames:
                return self._frames[-1].thumb_jpeg
        return None

    def digest(self, max_lines: int = 20, now: float | None = None) -> str:
        """Chronological plain-text view for the agent prompt."""
        now = self._clock() if now is None else now
        with self._lock:
            self.prune(now)
            recent = list(self._events)[-max_lines:]
        lines = [f"[{now - ev.t:5.1f}s ago] ({ev.kind}) {ev.text}" for ev in recent]
        if not lines:
            return "(memory empty)"
        return "\n".join(lines)
