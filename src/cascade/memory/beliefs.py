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

A whole camera frame is fused with `update_frame()` (instance-level
association, 2026-10-08): the frame's detections are grouped into instances
by shared image support, and instances and beliefs are matched ONE-TO-ONE by
a min-cost assignment inside the unchanged gates. `update()` stays the
single-observation writer (place_at, push, localize).
"""

from __future__ import annotations

import copy
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


@dataclass
class FrameObservation:
    """One detection of ONE camera frame, lifted to the base frame: the input
    of `BeliefStore.update_frame`.

    `bbox` (x0, y0, x1, y1 pixels) and `mask` (bool HxW, numpy or torch;
    None = the bbox rectangle) are the detection's IMAGE support, i.e. the
    pixels that were lifted. They are the only evidence that two detections
    of the same frame are one object (an open-vocabulary second name, a part
    inside its whole) rather than two objects side by side, and they are used
    for that decision only: nothing image-side is stored. No `bbox` = no
    image evidence = the observation is an instance of its own.
    """

    label: str
    position: np.ndarray
    conf: float
    extent: np.ndarray | None = None
    top_z: float | None = None
    color: str | None = None
    points: np.ndarray | None = None
    bbox: np.ndarray | None = None
    mask: Any = None


#: Two detections of ONE frame are the same instance only when they share at
#: least this fraction of the SMALLER one's image support (and pass the
#: store's own fusion gate). An open-vocabulary detector's second name for an
#: object (class-aware NMS keeps both, near-identical masks) and a part inside
#: its whole (a handle inside the mug's mask) score ~1.0; two props side by
#: side share no pixels and score 0.
SAME_INSTANCE_OVERLAP = 0.5

#: Cost of an assignment the gate forbids; any value above the sum of all
#: birth costs keeps the solver off it (birth costs are gate radii, < 1 m).
_FORBIDDEN = 1.0e6


def _min_cost_assignment(cost) -> list[int]:
    """Rectangular min-cost assignment: one distinct column per row (rows <=
    columns), minimising the summed cost. Returns the column of each row.

    Hungarian algorithm with row/column potentials (Kuhn-Munkres in the
    O(n^2 m) shortest-augmenting-path form), the column scan vectorised with
    numpy. Written here because scipy (`linear_sum_assignment`) is not a base
    dependency; `tests/test_same_colour_beliefs.py` checks it against brute
    force. Ties go to the lowest column index, so the result is deterministic.
    """
    c = np.asarray(cost, dtype=float)
    n, m = c.shape
    if n == 0:
        return []
    if n > m:
        raise ValueError(f"assignment needs rows <= columns, got {n} x {m}")
    a = np.zeros((n + 1, m + 1))
    a[1:, 1:] = c
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)  # p[j]: row holding column j (0 = free)
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            cur = a[i0] - u[i0] - v
            better = ~used & (cur < minv)
            minv[better] = cur[better]
            way[better] = j0
            open_minv = np.where(used, np.inf, minv)
            j1 = int(np.argmin(open_minv))
            delta = open_minv[j1]
            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    cols = [0] * n
    for j in range(1, m + 1):
        if p[j]:
            cols[p[j] - 1] = j - 1
    return cols


def _support_area(o: FrameObservation) -> float:
    if o.mask is not None:
        return float(o.mask.sum())
    x0, y0, x1, y1 = (float(v) for v in o.bbox)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _support_overlap(a: FrameObservation, b: FrameObservation,
                     area_a: float, area_b: float) -> float:
    """Shared image support over the SMALLER support (1.0 = one inside the
    other, 0.0 = disjoint). Masks are compared inside the intersection of
    the two boxes only, so the cost is the overlap window, not the image."""
    ax0, ay0, ax1, ay1 = (int(v) for v in a.bbox)
    bx0, by0, bx1, by1 = (int(v) for v in b.bbox)
    x0, y0 = max(ax0, bx0, 0), max(ay0, by0, 0)
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    small = min(area_a, area_b)
    if x1 <= x0 or y1 <= y0 or small <= 0:
        return 0.0
    ma, mb = a.mask, b.mask
    if ma is not None and mb is not None and tuple(ma.shape) == tuple(mb.shape):
        shared = float((ma[y0:y1, x0:x1] & mb[y0:y1, x0:x1]).sum())
    elif ma is not None:
        shared = float(ma[y0:y1, x0:x1].sum())
    elif mb is not None:
        shared = float(mb[y0:y1, x0:x1].sum())
    else:
        shared = float((x1 - x0) * (y1 - y0))
    return shared / small


@dataclass
class SceneSnapshot:
    """A named, ADVISORY record of where the confirmed objects were -- Pigey's
    "Original positions (memorize)" list (`real/agent-system.md`), kept in
    the world model instead of in the planner's reasoning text.

    Entries are plain dicts (label, aliases, colour, centroid, extent, top_z,
    conf, observations) copied at snapshot time, so later belief fusion
    cannot rewrite the memorized layout. A position here is a restore
    TARGET: nothing in a snapshot asserts where an object is now, and
    `restore_scene` only claims a restored layout through its postcondition.
    """

    name: str
    t: float                      # monotonic, when it was taken
    entries: list[dict]
    wall: float = field(default_factory=time.time)

    def age_s(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        return max(0.0, now - self.t)

    def as_dict(self, now: float | None = None) -> dict:
        return {
            "name": self.name,
            "age_s": round(self.age_s(now), 1),
            "count": len(self.entries),
            "objects": [
                {
                    "label": e["label"],
                    "color": e.get("color"),
                    "position": [round(float(v), 3) for v in e["position"]],
                    "extent_m": (None if e.get("extent") is None
                                 else [round(float(v), 3) for v in e["extent"]]),
                }
                for e in self.entries
            ],
        }


class BeliefStore:
    def __init__(
        self,
        match_radius_m: float = 0.08,
        pos_alpha: float = 0.4,
        conf_alpha: float = 0.3,
        forget_after_s: float | None = None,
        label_agnostic: bool = True,
        extent_frac: float = 0.5,
        instance_association: bool = True,
    ):
        self._beliefs: list[ObjectBelief] = []
        self._match_radius = match_radius_m
        self._extent_frac = extent_frac
        self._pos_alpha = pos_alpha
        self._conf_alpha = conf_alpha
        self._forget_after = forget_after_s
        self._label_agnostic = label_agnostic
        #: `update_frame` associates a frame's instances with beliefs
        #: one-to-one; False = per-detection `update()` (pre-2026-10-08,
        #: `memory.instance_association: false`, the live A/B baseline)
        self._instance_association = bool(instance_association)
        self._lock = threading.RLock()
        #: named advisory layouts (SceneSnapshot), see `snapshot()`
        self._snapshots: dict[str, SceneSnapshot] = {}

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
        (That is the SINGLE-observation rule. A whole camera frame goes
        through `update_frame`, which keeps two detections of one frame apart
        when the image shows them side by side.)

        `points` should only be passed for REAL segmentation masks (never
        bbox-rectangle fallbacks -- those sweep in table/neighbor pixels and
        poison remembered geometry).
        """
        now = time.monotonic() if t is None else t
        position = np.asarray(position, dtype=float).reshape(3)
        points = self._remembered_cloud(points)
        with self._lock:
            best, best_d = None, None
            for b in self._beliefs:
                if not self._may_fuse(b.label, b.color, label, color):
                    continue
                radius = self._fusion_radius(b.extent, extent)
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
            self._fuse(best, label, position, conf, extent, top_z, color, points, now)
            return best

    # ── the fusion gate and the fusion step, shared by update/update_frame ──

    @staticmethod
    def _remembered_cloud(points):
        """The <=384-point copy of a REAL-mask cloud that a belief keeps."""
        if points is None:
            return None
        if os.environ.get("CASCADE_REQUIRE_CUDA", "0") == "1":
            from ..perception.cuda_math import remember_cloud
            return remember_cloud(points)
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if pts.shape[0] > 384:
            idx = np.random.default_rng(0).choice(pts.shape[0], 384, replace=False)
            pts = pts[idx]
        return pts.copy()

    def _may_fuse(self, label_a: str, color_a: str | None,
                  label_b: str, color_b: str | None) -> bool:
        """Identity gate: could these two be the same object at all?"""
        if not self._label_agnostic and label_a != label_b:
            return False
        # Two DIFFERENT confirmed colours are two objects, however
        # close. Proximity matching exists for label aliases of ONE
        # object ("cube" vs "hassock"); it must not fuse two small
        # props that sit inside the 8 cm gate. Measured on the
        # two-cube MuJoCo scene (3.5 cm cubes, 5.8 cm apart): the
        # blue cube was absorbed into the red belief, count_objects
        # said 1, and the merged position sat one cube-width off
        # physics truth. Colour comes from the HSV classifier on the
        # mask, so it is a measurement, not a label.
        return not (color_a is not None and color_b is not None and color_a != color_b)

    def _fusion_radius(self, extent_a, extent_b) -> float:
        # A big object's centre estimate wanders further between
        # frames than a small one's: two views of a 30 cm bin can
        # disagree by 8 cm while two 5 cm cubes that far apart are
        # genuinely different objects. Scaling the gate by the
        # object's own measured size keeps this honest instead of
        # tuning one radius to whatever is on the table today.
        radius = self._match_radius
        for e in (extent_a, extent_b):
            if e is not None:
                radius = max(radius, self._extent_frac * float(np.max(e)))
        return radius

    def _fuse(self, best: ObjectBelief, label, position, conf, extent, top_z,
              color, points, now, *, aliases=(), reanchor: bool = False) -> None:
        a = 1.0 if reanchor else self._pos_alpha
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
        for name in aliases:  # the frame's other names for this instance
            if name != best.label:
                best.aliases.add(name)
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

    # ── one camera frame at a time (instance-level association) ─────────

    def update_frame(self, observations, t: float | None = None) -> list[ObjectBelief]:
        """Fuse ALL detections of ONE camera frame; returns, per observation
        and in input order, the belief it was fused into.

        `update()` per detection let the SECOND detection of a frame match
        the belief the FIRST had just created or moved: two identical red
        cubes 5 cm apart became one belief at an EMA blend of both, inside
        the 8 cm gate that exists for label aliases of ONE object (backlog
        B31, ARCHITECTURE "Known limitations"). A frame is now associated as
        a whole:

        1. its detections are grouped into INSTANCES: two detections are one
           object only if they share image support (`SAME_INSTANCE_OVERLAP`:
           an open-vocabulary second name, a part inside its whole) AND pass
           the store's own gate (label rule, colour rule, size-scaled
           radius); an instance never joins two measured colours;
        2. instances and beliefs are matched ONE-TO-ONE by a min-cost
           assignment on 3D distance inside the same gates, a new object
           costing the instance's gate radius. Two instances of one frame
           never claim one belief, twins that move together do not swap
           (greedy nearest-first would), and a lone detection still goes to
           the nearest belief inside the gate, exactly as in `update()`;
        3. an unmatched instance becomes a new belief; an unmatched belief is
           not touched (occlusion / object permanence unchanged). A matched
           belief that a NEW instance of the same frame could also have
           taken was carrying two objects (or is being joined by one): it is
           re-anchored on its own instance rather than EMA-blended with a
           position that may be a midpoint.

        A multi-detection instance takes its geometry from its largest image
        support (the whole, not the part), its name by the `update()` rule
        (the frame's most confident name competes with the belief's; every
        other name becomes an alias), and counts as ONE observation.

        Not fixable here: a detector that returns two objects as ONE
        detection (the mock detector's one blob per colour, a box drawn
        around both) hands the store one instance. `instance_association=
        False` restores per-detection `update()` for a live A/B.
        """
        now = time.monotonic() if t is None else t
        obs = list(observations)
        if not self._instance_association:
            return [
                self.update(o.label, o.position, o.conf, extent=o.extent, top_z=o.top_z,
                            t=now, color=o.color, points=o.points)
                for o in obs
            ]
        if not obs:
            return []
        pos = [np.asarray(o.position, dtype=float).reshape(3) for o in obs]
        groups = self._frame_instances(obs, pos)
        named = []
        for g in groups:
            head = max(g, key=lambda q: float(obs[q].conf))  # first max: anchor wins ties
            colour = next((obs[q].color for q in g if obs[q].color is not None), None)
            cloud = next((obs[q].points for q in g if obs[q].points is not None), None)
            named.append((obs[head].label, float(obs[head].conf), colour,
                          self._remembered_cloud(cloud), [obs[q].label for q in g]))
        out: list = [None] * len(obs)
        with self._lock:
            beliefs = list(self._beliefs)
            feasible: list[list[tuple[int, float, float]]] = []
            for g, (label, _, colour, _, _) in zip(groups, named):
                k = g[0]
                row = []
                for j, b in enumerate(beliefs):
                    if not self._may_fuse(b.label, b.color, label, colour):
                        continue
                    radius = self._fusion_radius(b.extent, obs[k].extent)
                    d = float(np.linalg.norm(b.position - pos[k]))
                    if d < radius:
                        row.append((j, d, radius))
                feasible.append(row)
            cols = sorted({j for row in feasible for j, _, _ in row})
            col_of = {j: c for c, j in enumerate(cols)}
            n, m = len(groups), len(cols)
            cost = np.full((n, m + n), _FORBIDDEN)
            for i, row in enumerate(feasible):
                cost[i, m:] = max((r for _, _, r in row), default=self._match_radius)
                for j, d, _ in row:
                    cost[i, col_of[j]] = d
            assign = _min_cost_assignment(cost)
            matched = {i: cols[c] for i, c in enumerate(assign) if c < m}
            contested = {j for i, row in enumerate(feasible) if i not in matched
                         for j, _, _ in row}
            born = []
            for i, g in enumerate(groups):
                k = g[0]
                label, conf, colour, cloud, names = named[i]
                if i in matched:
                    b = beliefs[matched[i]]
                    self._fuse(b, label, pos[k], conf, obs[k].extent, obs[k].top_z,
                               colour, cloud, now, aliases=names,
                               reanchor=matched[i] in contested)
                else:
                    b = ObjectBelief(
                        label=label, position=pos[k], extent=obs[k].extent,
                        top_z=obs[k].top_z, conf=conf, color=colour, points=cloud,
                        aliases={x for x in names if x != label},
                        last_seen_t=now, first_seen_t=now,
                    )
                    born.append(b)
                for q in g:
                    out[q] = b
            self._beliefs.extend(born)
        return out

    def _frame_instances(self, obs: list[FrameObservation], pos) -> list[list[int]]:
        """Group one frame's observations into instances (see update_frame).

        Union-find over pairs that share image support and pass the gate,
        strongest overlap first; a merge that would join two measured colours
        is refused. Each group lists its anchor (largest support, then most
        confident) first; groups come in input order.
        """
        n = len(obs)
        area = [(_support_area(o) if o.bbox is not None else 0.0) for o in obs]
        edges = []
        for i in range(n):
            if area[i] <= 0:
                continue
            for j in range(i + 1, n):
                if area[j] <= 0:
                    continue
                if not self._may_fuse(obs[i].label, obs[i].color, obs[j].label, obs[j].color):
                    continue
                if float(np.linalg.norm(pos[i] - pos[j])) >= self._fusion_radius(
                        obs[i].extent, obs[j].extent):
                    continue
                overlap = _support_overlap(obs[i], obs[j], area[i], area[j])
                if overlap >= SAME_INSTANCE_OVERLAP:
                    edges.append((-overlap, i, j))
        root = list(range(n))
        colours = [{o.color} - {None} for o in obs]

        def find(x: int) -> int:
            while root[x] != x:
                root[x] = root[root[x]]
                x = root[x]
            return x

        for _, i, j in sorted(edges):
            ri, rj = find(i), find(j)
            if ri == rj or len(colours[ri] | colours[rj]) > 1:
                continue
            root[rj] = ri
            colours[ri] |= colours[rj]
        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        out = [sorted(g, key=lambda q: (-area[q], -float(obs[q].conf), q))
               for g in groups.values()]
        out.sort(key=min)
        return out

    def clear(self) -> int:
        """Forget every object (scene reset). Returns how many were dropped.

        Named snapshots go too: a layout memorized for one visitor must not
        become the next visitor's restore target.
        """
        with self._lock:
            n = len(self._beliefs)
            self._beliefs = []
            self._snapshots = {}
            return n

    # ── named scene snapshots (Pigey memorize/restore) ──────────────────

    def snapshot(
        self,
        name: str = "scene",
        now: float | None = None,
        visible_horizon_s: float = 1.5,
        exclude_label: str | None = None,
    ) -> SceneSnapshot:
        """Record the CONFIRMED objects -- those currently `visible`, i.e.
        re-observed within `visible_horizon_s` -- as the named layout.

        Remembered-only beliefs are left out on purpose: a position nobody
        has confirmed lately is a poor restore target, and a stale one would
        send the arm to put an object where it may never have been. The
        held object (`exclude_label`, its detector label) is not on the
        table, so it has no layout position either. Taking a snapshot under
        an existing name replaces it.
        """
        now = time.monotonic() if now is None else now
        with self._lock:
            entries = []
            for b in self.all(now):
                if b.state(now, visible_horizon_s) != "visible":
                    continue
                if exclude_label is not None and b.label == exclude_label:
                    continue
                entries.append({
                    "label": b.label,
                    "aliases": sorted(b.aliases),
                    "color": b.color,
                    "position": [float(v) for v in b.position],
                    "extent": None if b.extent is None else [float(v) for v in b.extent],
                    "top_z": None if b.top_z is None else float(b.top_z),
                    "conf": float(b.conf),
                    "observations": int(b.observations),
                })
            snap = SceneSnapshot(name=str(name), t=now, entries=entries)
            self._snapshots[str(name)] = snap
            return snap

    def get_snapshot(self, name: str) -> SceneSnapshot | None:
        with self._lock:
            return self._snapshots.get(str(name))

    def snapshots(self) -> list[str]:
        with self._lock:
            return sorted(self._snapshots)

    def drop_snapshot(self, name: str) -> bool:
        with self._lock:
            return self._snapshots.pop(str(name), None) is not None

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
            # Named layouts travel with the world model. They are advisory
            # data (restore targets), so the only clock work they need is the
            # same monotonic -> wall conversion; nothing about them can read
            # as "visible" after a load.
            snapshots = [
                {
                    "name": s.name,
                    "taken_wall": now_wall - (now_mono - s.t),
                    "entries": copy.deepcopy(s.entries),
                }
                for s in self._snapshots.values()
            ]
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps({"version": 1, "saved_wall": now_wall,
                                   "beliefs": records, "snapshots": snapshots}, indent=1))
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
            saved_snapshots = blob.get("snapshots") or [] if isinstance(blob, dict) else []
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
        restored_snapshots: dict[str, SceneSnapshot] = {}
        for s in saved_snapshots:
            try:
                entries = [dict(e) for e in s["entries"]]
                for e in entries:
                    e["position"] = [float(v) for v in e["position"]]
                    if e.get("extent") is not None:
                        e["extent"] = [float(v) for v in e["extent"]]
                taken_age = max(0.0, now_wall - float(s.get("taken_wall", now_wall)))
                restored_snapshots[str(s["name"])] = SceneSnapshot(
                    name=str(s["name"]), t=now_mono - taken_age, entries=entries,
                    wall=float(s.get("taken_wall", now_wall)))
            except Exception:  # noqa: BLE001 - a bad layout must not block the beliefs
                continue
        if not loaded and not restored_snapshots:
            return 0
        with self._lock:
            self._beliefs.extend(loaded)
            self._snapshots.update(restored_snapshots)
        return len(loaded)
