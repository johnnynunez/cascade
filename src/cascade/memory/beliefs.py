"""Object-permanence belief store.

Tracks the last known base-frame position of every object the system has
seen, so the agent can act on things that scrolled out of view or got
occluded ("the mug is where it was 8 seconds ago"). Beliefs decay to
`remembered` state after not being re-observed; they are only dropped after
`forget_after_s` (default: never during a demo run).

Beliefs also carry a named color (median mask HSV, see perception.colors) so
color-word queries like "pink object" resolve against the live world model
even when the detector vocabulary has no such class. All methods are
thread-safe: the WorldWatcher fuses observations from N camera streams while
the skill runtime reads.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from ..perception.colors import parse_color_query


@dataclass
class ObjectBelief:
    label: str
    position: np.ndarray  # (3,) base frame
    extent: np.ndarray | None = None  # (3,) OBB extents (desc-sorted, NOT axis-aligned)
    top_z: float | None = None  # highest observed point (base frame) - use this
    # for stacking/placing, never extent[2] (extents are eigenvalue-ordered).
    conf: float = 0.5
    color: str | None = None  # named color (perception.colors palette)
    points: np.ndarray | None = None  # (<=384,3) last REAL-mask cloud, base
    # frame -- lets grasp-from-memory use the object's true shape instead of
    # a box approximation (a bbox-inflated box slab reads as ungraspable).
    aliases: set[str] = field(default_factory=set)  # other names the detector
    # has used for this same object; open-vocabulary detectors rename things
    # between frames, and a query for any alias must still resolve.
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
        label_agnostic: bool = True,
        extent_frac: float = 0.5,
    ):
        self._beliefs: list[ObjectBelief] = []
        self._match_radius = match_radius_m
        self._extent_frac = extent_frac
        self._pos_alpha = pos_alpha
        self._conf_alpha = conf_alpha
        self._forget_after = forget_after_s
        self._label_agnostic = label_agnostic
        self._lock = threading.RLock()

    def update(
        self,
        label: str,
        position: np.ndarray,
        conf: float,
        extent: np.ndarray | None = None,
        top_z: float | None = None,
        t: float | None = None,
        color: str | None = None,
        points: np.ndarray | None = None,
    ) -> ObjectBelief:
        """Fuse one 3D observation into the world model.

        Matching is by PROXIMITY, and by default ignores the label entirely.
        That matters for open-vocabulary perception: a ~4.5k-concept detector
        names the same physical object differently between frames (a bin came
        back as both "storage box" and "building block", a cube as "cube" and
        "hassock"). Matching on label alone would register each alias as a
        separate object sitting at the same place, so a table with 3 things on
        it reports 10.

        Two objects genuinely within `match_radius_m` of each other cannot be
        told apart this way, but at 8 cm on a tabletop they are touching, and
        merging them is a better failure than inventing duplicates. Pass
        `label_agnostic=False` for the old same-label-only behaviour.

        `points` should only be passed for REAL segmentation masks (never
        bbox-rectangle fallbacks -- those sweep in table/neighbor pixels and
        poison remembered geometry).
        """
        now = time.monotonic() if t is None else t
        position = np.asarray(position, dtype=float).reshape(3)
        if points is not None:
            pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
            if pts.shape[0] > 384:
                idx = np.random.default_rng(0).choice(pts.shape[0], 384, replace=False)
                pts = pts[idx]
            points = pts.copy()
        with self._lock:
            best, best_d = None, None
            for b in self._beliefs:
                if not self._label_agnostic and b.label != label:
                    continue
                # A big object's centre estimate wanders further between
                # frames than a small one's: two views of a 30 cm bin can
                # disagree by 8 cm while two 5 cm cubes that far apart are
                # genuinely different objects. Scaling the gate by the
                # object's own measured size keeps this honest instead of
                # tuning one radius to whatever is on the table today.
                radius = self._match_radius
                for e in (b.extent, extent):
                    if e is not None:
                        radius = max(radius, self._extent_frac * float(np.max(e)))
                d = float(np.linalg.norm(b.position - position))
                if d < radius and (best_d is None or d < best_d):
                    best, best_d = b, d
            if best is None:
                best = ObjectBelief(
                    label=label, position=position, extent=extent, top_z=top_z,
                    conf=conf, color=color, points=points,
                    last_seen_t=now, first_seen_t=now,
                )
                self._beliefs.append(best)
                return best
            a = self._pos_alpha
            best.position = (1 - a) * best.position + a * position
            # Decide the name BEFORE conf is smoothed, so the comparison is
            # this observation against the belief as it stood.
            if label != best.label:
                if conf > best.conf:
                    best.aliases.add(best.label)
                    best.label = label
                else:
                    best.aliases.add(label)
                best.aliases.discard(best.label)
            best.conf = (1 - self._conf_alpha) * best.conf + self._conf_alpha * conf
            if extent is not None:
                best.extent = extent
            if top_z is not None:
                best.top_z = top_z
            if color is not None:
                best.color = color
            if points is not None:
                best.points = points
            best.last_seen_t = now
            best.observations += 1
            return best

    def mark_removed(self, label: str, near: np.ndarray | None = None) -> bool:
        """Drop a belief after the robot itself moved the object away."""
        with self._lock:
            cands = [b for b in self._beliefs if b.label == label]
            if not cands:  # the label may be a query ("pink object")
                q = self.find(label)
                cands = [q] if q is not None else []
            if near is not None and cands:
                near = np.asarray(near, dtype=float).reshape(3)
                cands.sort(key=lambda b: float(np.linalg.norm(b.position - near)))
            if not cands:
                return False
            # Remove by IDENTITY: list.remove falls back to dataclass __eq__,
            # which compares numpy fields elementwise and raises "truth value
            # of an array is ambiguous" unless the match is the first element.
            victim = cands[0]
            self._beliefs = [b for b in self._beliefs if b is not victim]
            return True

    def all(self, now: float | None = None) -> list[ObjectBelief]:
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._forget_after is not None:
                self._beliefs = [
                    b for b in self._beliefs if now - b.last_seen_t <= self._forget_after
                ]
            return list(self._beliefs)

    def find(self, query: str) -> ObjectBelief | None:
        """Resolve a label OR a color query ("pink object", "red mug").

        Aliases count as names: an open-vocabulary detector may have called
        this object "storage box" on the frame the user was looking at and
        "building block" on the one that won the confidence vote.
        """
        with self._lock:
            matches = [
                b for b in self._beliefs
                if b.label == query or query in b.aliases
            ]
            if not matches:
                color, noun = parse_color_query(query)
                cands = list(self._beliefs)
                if noun:
                    exact = [
                        b for b in cands
                        if b.label.lower() == noun
                        or any(a.lower() == noun for a in b.aliases)
                    ]
                    loose = exact or [
                        b for b in cands
                        if noun in b.label.lower() or b.label.lower() in noun
                        or any(noun in a.lower() or a.lower() in noun for a in b.aliases)
                    ]
                    cands = loose
                if color:
                    # Prefer the exact palette band, then perceptual
                    # neighbors (red<->pink boundary objects), and only then
                    # untagged beliefs -- never a wrong-colored object.
                    from ..perception.colors import color_matches

                    exact = [b for b in cands if b.color == color]
                    near = [
                        b for b in cands
                        if b.color is not None and b.color != color
                        and color_matches(color, b.color)
                    ]
                    cands = exact or near or [b for b in cands if b.color is None]
                # A bare color word needs the color to actually constrain;
                # a bare noun already did. No constraint at all -> no match.
                if color is None and noun is None:
                    cands = []
                matches = cands
            if not matches:
                return None
            return max(matches, key=lambda b: (b.last_seen_t, b.conf))

    def summary(self, now: float | None = None) -> list[dict]:
        now = time.monotonic() if now is None else now
        out = []
        for b in sorted(self.all(now), key=lambda b: -b.last_seen_t):
            out.append(
                {
                    "label": b.label,
                    "color": b.color,
                    "position": [round(float(x), 3) for x in b.position],
                    "state": b.state(now),
                    "age_s": round(now - b.last_seen_t, 1),
                    "conf": round(b.conf, 2),
                }
            )
        return out
