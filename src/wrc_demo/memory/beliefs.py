"""Object-permanence belief store.

Tracks the last known base-frame position of every object the system has
seen, so the agent can act on things that scrolled out of view or got
occluded ("the mug is where it was 8 seconds ago"). Beliefs decay to
`remembered` state after not being re-observed; they are only dropped after
`forget_after_s` (default: never during a demo run).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ObjectBelief:
    label: str
    position: np.ndarray  # (3,) base frame
    extent: np.ndarray | None = None  # (3,) OBB extents (desc-sorted, NOT axis-aligned)
    top_z: float | None = None  # highest observed point (base frame) - use this
    # for stacking/placing, never extent[2] (extents are eigenvalue-ordered).
    conf: float = 0.5
    last_seen_t: float = field(default_factory=time.monotonic)
    first_seen_t: float = field(default_factory=time.monotonic)
    observations: int = 1

    def state(self, now: float, visible_horizon_s: float = 1.5) -> str:
        return "visible" if (now - self.last_seen_t) <= visible_horizon_s else "remembered"


class BeliefStore:
    def __init__(
        self,
        match_radius_m: float = 0.08,
        pos_alpha: float = 0.4,
        conf_alpha: float = 0.3,
        forget_after_s: float | None = None,
    ):
        self._beliefs: list[ObjectBelief] = []
        self._match_radius = match_radius_m
        self._pos_alpha = pos_alpha
        self._conf_alpha = conf_alpha
        self._forget_after = forget_after_s

    def update(
        self,
        label: str,
        position: np.ndarray,
        conf: float,
        extent: np.ndarray | None = None,
        top_z: float | None = None,
        t: float | None = None,
    ) -> ObjectBelief:
        """Fuse one 3D observation; matches same-label beliefs by proximity."""
        now = time.monotonic() if t is None else t
        position = np.asarray(position, dtype=float).reshape(3)
        best, best_d = None, self._match_radius
        for b in self._beliefs:
            if b.label != label:
                continue
            d = float(np.linalg.norm(b.position - position))
            if d < best_d:
                best, best_d = b, d
        if best is None:
            best = ObjectBelief(
                label=label, position=position, extent=extent, top_z=top_z,
                conf=conf, last_seen_t=now, first_seen_t=now,
            )
            self._beliefs.append(best)
            return best
        a = self._pos_alpha
        best.position = (1 - a) * best.position + a * position
        best.conf = (1 - self._conf_alpha) * best.conf + self._conf_alpha * conf
        if extent is not None:
            best.extent = extent
        if top_z is not None:
            best.top_z = top_z
        best.last_seen_t = now
        best.observations += 1
        return best

    def mark_removed(self, label: str, near: np.ndarray | None = None) -> bool:
        """Drop a belief after the robot itself moved the object away."""
        cands = [b for b in self._beliefs if b.label == label]
        if near is not None and cands:
            near = np.asarray(near, dtype=float).reshape(3)
            cands.sort(key=lambda b: float(np.linalg.norm(b.position - near)))
        if not cands:
            return False
        self._beliefs.remove(cands[0])
        return True

    def all(self, now: float | None = None) -> list[ObjectBelief]:
        now = time.monotonic() if now is None else now
        if self._forget_after is not None:
            self._beliefs = [
                b for b in self._beliefs if now - b.last_seen_t <= self._forget_after
            ]
        return list(self._beliefs)

    def find(self, label: str) -> ObjectBelief | None:
        matches = [b for b in self._beliefs if b.label == label]
        if not matches:
            # loose contains-match ("mug" vs "red mug")
            matches = [
                b for b in self._beliefs
                if label.lower() in b.label.lower() or b.label.lower() in label.lower()
            ]
        if not matches:
            return None
        return max(matches, key=lambda b: b.last_seen_t)

    def summary(self, now: float | None = None) -> list[dict]:
        now = time.monotonic() if now is None else now
        out = []
        for b in sorted(self.all(now), key=lambda b: -b.last_seen_t):
            out.append(
                {
                    "label": b.label,
                    "position": [round(float(x), 3) for x in b.position],
                    "state": b.state(now),
                    "age_s": round(now - b.last_seen_t, 1),
                    "conf": round(b.conf, 2),
                }
            )
        return out
