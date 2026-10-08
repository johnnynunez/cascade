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
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
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
    kind: str  # "frame" | "object" | "event"
    label: str | None
    event: MemoryEvent


class EpisodicMemory:
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

    def prune(self, now: float | None = None) -> None:
        now = self._clock() if now is None else now
        with self._lock:
            while self._events and now - self._events[0].t > self.horizon_s:
                self._events.popleft()
            while self._frames and now - self._frames[0].t > self.frame_horizon_s:
                self._frames.popleft()
            # Visual entries are appended in time order, so expiry is a prefix
            # of the deque -- and of the index, which is pruned in lockstep.
            drop = 0
            n = len(self._visual)
            while drop < n and now - self._visual[drop].t > self.frame_horizon_s:
                drop += 1
            drop = max(drop, n - self.max_visual)
            if drop > 0 and self._index is not None:
                self._index.remove_ids(set(range(drop)))
                for _ in range(drop):
                    self._visual.popleft()

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
            frame_steps = {id(ev): n + 1 for n, ev in enumerate(self._frames)}
        out: list[dict] = []
        for score, entry in hits:
            if kinds and entry.kind not in kinds:
                continue
            if min_sim is not None and score < float(min_sim):
                break  # hits are sorted: everything after is lower
            ev = entry.event
            out.append({
                "score": round(float(score), 3),
                "kind": entry.kind,
                "label": entry.label,
                "age_s": round(now - entry.t, 1),
                "text": ev.text,
                "verdict": str((ev.data or {}).get("verdict") or ""),
                "step": frame_steps.get(id(ev)),
            })
            if len(out) >= k:
                break
        return out

    def visual_stats(self) -> dict:
        """What the visual index holds right now (for status pages and tests)."""
        with self._lock:
            return {
                "embedder": getattr(self.embedder, "name", None),
                "joint_space": bool(getattr(self.embedder, "joint_space", False)),
                "entries": len(self._visual),
                "embed_errors": int(self.embed_errors),
            }

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
