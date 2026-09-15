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

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

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
            if os.environ.get("CASCADE_REQUIRE_CUDA", "0") == "1":
                from ..perception.cuda_math import remember_cloud
                points = remember_cloud(points)
            else:
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
                # Two DIFFERENT confirmed colours are two objects, however
                # close. Proximity matching exists for label aliases of ONE
                # object ("cube" vs "hassock"); it must not fuse two small
                # props that sit inside the 8 cm gate. Measured on the
                # two-cube MuJoCo scene (3.5 cm cubes, 5.8 cm apart): the
                # blue cube was absorbed into the red belief, count_objects
                # said 1, and the merged position sat one cube-width off
                # physics truth. Colour comes from the HSV classifier on the
                # mask, so it is a measurement, not a label.
                if color is not None and b.color is not None and b.color != color:
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

    def clear(self) -> int:
        """Forget every object (scene reset). Returns how many were dropped."""
        with self._lock:
            n = len(self._beliefs)
            self._beliefs = []
            return n

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

    # ── persistence across restarts ─────────────────────────────────────
    #
    # In-session object permanence already works (visible -> remembered, EMA
    # fusion, grasp-from-memory). What did not survive was the PROCESS: every
    # restart began with an empty table, so the robot re-discovered a world it
    # had already mapped, and any question about "the mug from before" was
    # unanswerable. This is the ROADMAP's "persistent spatial memory" item.
    #
    # THE TRAP, and the reason this is not a plain json.dump of the dataclass:
    # every timestamp in a belief is `time.monotonic()`, which counts from an
    # arbitrary origin that RESETS on reboot. Writing those numbers and reading
    # them back yields ages like "-4210 s" or "seen 3 hours in the future", and
    # `state()` then reports a stale belief as freshly visible -- worse than no
    # memory at all, because it looks authoritative. So timestamps are
    # converted to WALL CLOCK on save and back to this process's monotonic
    # origin on load, and everything loaded is deliberately aged past the
    # visible horizon: it is `remembered`, never `visible`, because nothing has
    # actually been observed yet in this session.

    #: Beliefs older than this (wall clock) are dropped on load. A day-old
    #: tabletop is not evidence; the default keeps a session-to-session demo
    #: warm without resurrecting last week's objects.
    DEFAULT_MAX_AGE_S = 6 * 3600.0

    #: Floor on the apparent age of anything loaded from disk. Without it, a
    #: file saved seconds ago restores with age ~0 and `state()` reports
    #: `visible` -- the robot claiming to SEE an object it has not looked at
    #: yet this session. That is the failure mode this whole format exists to
    #: avoid, so the floor is enforced rather than left to the caller.
    LOADED_MIN_AGE_S = 2.0

    def save(self, path) -> int:
        """Persist every belief with wall-clock timestamps. Returns the count.

        Written atomically (temp file + replace) because the WorldWatcher may
        be fusing observations while this runs, and a half-written JSON is a
        corrupt world model on the next boot.
        """
        path = Path(path)
        now_mono = time.monotonic()
        now_wall = time.time()
        with self._lock:
            records = []
            for b in self._beliefs:
                records.append({
                    "label": b.label,
                    "position": [float(x) for x in b.position],
                    "extent": None if b.extent is None else [float(x) for x in b.extent],
                    "top_z": None if b.top_z is None else float(b.top_z),
                    "conf": float(b.conf),
                    "color": b.color,
                    # Points are the biggest field by far and are only used by
                    # grasp-from-memory, which re-observes anyway. Store a
                    # decimated copy so the file stays small enough to write
                    # every episode.
                    "points": (
                        None if b.points is None
                        else b.points[::4].tolist()
                    ),
                    "aliases": sorted(b.aliases),
                    "observations": int(b.observations),
                    # monotonic -> wall clock, the whole point of this format
                    "last_seen_wall": now_wall - (now_mono - b.last_seen_t),
                    "first_seen_wall": now_wall - (now_mono - b.first_seen_t),
                })
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps({"version": 1, "saved_wall": now_wall,
                                   "beliefs": records}, indent=1))
        os.replace(tmp, path)
        return len(records)

    def load(self, path, max_age_s: float | None = None) -> int:
        """Restore beliefs saved by `save()`. Returns how many were loaded.

        Everything loaded is marked as last seen `visible_horizon_s` ago at the
        newest, so `state()` reports `remembered` -- the agent is told what was
        there, never that it can see it. A corrupt or unreadable file is
        ignored: a broken memory must not stop the robot from starting.
        """
        path = Path(path)
        if not path.exists():
            return 0
        try:
            blob = json.loads(path.read_text())
            records = blob["beliefs"] if isinstance(blob, dict) else blob
        except Exception:  # noqa: BLE001 - never block startup on a bad file
            return 0
        if max_age_s is None:
            max_age_s = self.DEFAULT_MAX_AGE_S
        now_mono = time.monotonic()
        now_wall = time.time()
        loaded = []
        for r in records:
            try:
                age = now_wall - float(r["last_seen_wall"])
                # A file written on a machine whose clock later moved backwards
                # yields a negative age; treat it as "just saved" rather than
                # trusting it or discarding a good world model.
                age = max(0.0, age)
                if age > max_age_s:
                    continue
                # Nothing restored from disk may read as `visible` (see
                # LOADED_MIN_AGE_S). Applied AFTER the max_age test so the
                # floor cannot smuggle in an expired belief.
                age = max(age, self.LOADED_MIN_AGE_S)
                first_age = max(age, now_wall - float(
                    r.get("first_seen_wall", r["last_seen_wall"])))
                pts = r.get("points")
                loaded.append(ObjectBelief(
                    label=str(r["label"]),
                    position=np.asarray(r["position"], dtype=float).reshape(3),
                    extent=(None if r.get("extent") is None
                            else np.asarray(r["extent"], dtype=float)),
                    top_z=r.get("top_z"),
                    conf=float(r.get("conf", 0.5)),
                    color=r.get("color"),
                    points=(None if pts is None
                            else np.asarray(pts, dtype=np.float32).reshape(-1, 3)),
                    aliases=set(r.get("aliases") or []),
                    observations=int(r.get("observations", 1)),
                    # Age is preserved RELATIVE to now, so "seen 40 minutes
                    # ago" still reads as 40 minutes ago after a restart.
                    last_seen_t=now_mono - age,
                    first_seen_t=now_mono - first_age,
                ))
            except Exception:  # noqa: BLE001 - skip a bad record, keep the rest
                continue
        if not loaded:
            return 0
        with self._lock:
            self._beliefs.extend(loaded)
        return len(loaded)
