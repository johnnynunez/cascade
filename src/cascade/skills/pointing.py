"""point_at's pointing pose when no top-down pose clears, and its route.

All geometry is in the arm's base frame.

The object column (`ObjectColumn`) is the fix's oriented box extruded down to
the table, made a vertical prism over the convex xy hull of the box's corners
and the fix's observed points (the box is centred on `fix.position`, which
perception may recentre away from the points). Its height is the object top
point_at computed (the observed top, raised by the map's capped column top and
by an object stacked over it), never a PCA box corner: a box tilted by noise
would invent height. An object stacked over the target adds its own prism.

The tool samples (`tool_samples`) are the TCP and the open-jaw sweep box
(`grasp.graspgenx.sweep`, `safety.tool_volume.jaw_box`) on a grid at most
TOOL_SPACING_M apart, faces and interior included: once the tool is tilted,
the point nearest the object is in general not a box corner.

A pose is admissible (`PoseClearance.admissible`) when, on its solved FK,
every tool sample keeps `margin` from the column and stays `margin` above the
table, every joint origin the harness checks keeps `margin` from the column,
and the approach ray passes within AIM_TOLERANCE_M of the aim point.

The lean-in family (`lean_in`) aims at the centre of the object's top, P:
approach a = cos(phi)(-z) + sin(phi) u, u the horizontal direction from the
arm's first (pan) joint axis to P, TCP = P - L a, so the TCP sits between the
base and the object. phi in LEAN_IN_TILTS_DEG, then L in
LEAN_IN_DISTANCES_M, smallest tilt first; the first admissible pose wins. The
solve leaves the jaw roll free (`Kinematics.ik_approach`): a 5-joint arm such
as the SO-101 holds a tilted approach only in its own working plane, which
passes through the pan axis; a 6-joint arm holds it in any plane.

The route (`plan_route`) is checked by FK samples along the straight joint
segment the streamer interpolates (min-jerk in time, a straight line in
joint space), at most ROUTE_STEP_RAD apart, against the same column and
table: tool samples and joint origins. An arm that already starts inside the
margin (the SO-101's home holds its jaw box over the mock_small cube) may only
move away until it is out, never closer. A failing direct segment gets one
rise-first waypoint: over the start TCP (or backed off from the object), at
the start's height + 3 or 6 cm or at the higher of the start and the goal
(+0/3/6 cm), lowest first, at the start's rotation or tilted in the arm's
plane. A route with no clear waypoint is refused before any command.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: lean-in tilt from straight down, tried in this order (smallest first)
LEAN_IN_TILTS_DEG = (15.0, 25.0, 35.0, 45.0)
#: TCP distance from the aim point along the approach, per tilt, in order
LEAN_IN_DISTANCES_M = (0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10)
#: largest spacing of the jaw-box samples (faces and interior)
TOOL_SPACING_M = 0.005
#: the approach ray must pass this close to the aim point
AIM_TOLERANCE_M = 0.01
#: largest joint step between two route samples (5 mm at 0.5 m of lever)
ROUTE_STEP_RAD = 0.01
#: rise-first waypoint heights: (above which TCP, how much). "start": the
#: arm's own TCP (a short top-down rise stays under a low top-down ceiling
#: such as the SO-101's); "higher": the higher of the start and the goal TCP.
#: Tried lowest first.
RISE_WAYPOINTS = (("start", 0.03), ("start", 0.06), ("higher", 0.0), ("higher", 0.03), ("higher", 0.06))
#: ... each first straight over the start TCP, then this far further from the
#: aim point horizontally (backing off as it rises: a joint-space rise is not
#: a vertical line, and from beside the object it would first close in)
RISE_BACK_OFF_M = (0.0, 0.03)
#: float noise allowed on a clearance that must not shrink (the harness's
#: escape floor, `safety.harness.CLEARANCE_ESCAPE_TOL_M`)
ESCAPE_TOL_M = 1e-9


class RouteRefused(Exception):
    """No route to the pointing pose keeps the margin; nothing was commanded."""


# ── the object column ──────────────────────────────────────────────────────


def _hull_xy(xy: np.ndarray) -> np.ndarray:
    """Convex hull (counter-clockwise, no repeated vertex) of 2-D points;
    fewer than 3 distinct points come back as they are."""
    pts = np.unique(np.asarray(xy, dtype=float).reshape(-1, 2), axis=0)
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _footprint_distance(xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Horizontal distance from points to a convex polygon: positive
    outside, minus the distance to the nearest edge inside."""
    v0 = poly
    edge = np.roll(poly, -1, axis=0) - v0
    length2 = (edge * edge).sum(axis=1)
    rel = xy[:, None, :] - v0[None, :, :]
    t = np.clip((rel * edge[None]).sum(axis=-1) / np.where(length2 > 0, length2, 1.0)[None], 0.0, 1.0)
    to_edge = np.linalg.norm(rel - t[..., None] * edge[None], axis=-1).min(axis=1)
    if len(poly) < 3:
        return to_edge
    inside = ((edge[None, :, 0] * rel[..., 1] - edge[None, :, 1] * rel[..., 0]) >= 0.0).all(axis=1)
    return np.where(inside, -to_edge, to_edge)


@dataclass(frozen=True)
class ObjectColumn:
    """Vertical prisms from the table (`z_lo`) to each `tops` entry over the
    convex `footprints`, and the aim point (the centre of the object's top)."""

    footprints: tuple
    tops: tuple
    z_lo: float
    aim: np.ndarray

    @property
    def top(self) -> float:
        return max(self.tops)

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance from each point to the column: positive outside,
        minus the depth to the nearest side or the top inside (the table is
        no way out), so going deeper shows; NaN for a non-finite point, which
        every `>= margin` test refuses."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        out = np.full(len(points), np.inf)
        for poly, top in zip(self.footprints, self.tops):
            dxy = _footprint_distance(points[:, :2], poly)
            above = points[:, 2] - top
            dz = np.maximum(0.0, np.maximum(above, self.z_lo - points[:, 2]))
            outside = np.hypot(np.maximum(dxy, 0.0), dz)
            inside = (dxy <= 0.0) & (dz <= 0.0)
            out = np.minimum(out, np.where(inside, np.maximum(dxy, above), outside))
        out[~np.isfinite(points).all(axis=1)] = np.nan
        return out


def object_column(fix, table_z: float, top: float, occluders=()) -> ObjectColumn:
    """The fix's oriented box and observed points extruded to the table, up
    to `top`, plus one prism per `occluders` entry ((xy polygon, top)).
    Raises ValueError for a non-finite box, points or top."""
    position = np.asarray(fix.position, dtype=float).reshape(3)
    axes = np.asarray(fix.axes, dtype=float).reshape(3, 3)
    half = np.asarray(fix.extent, dtype=float).reshape(3) / 2.0
    points = np.asarray(fix.points, dtype=float).reshape(-1, 3)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    corners = position + (signs * half) @ axes.T
    xy = np.vstack([corners[:, :2], points[:, :2]])
    if not (np.isfinite(xy).all() and math.isfinite(top) and math.isfinite(table_z)):
        raise ValueError("the object's box or points are not finite")
    footprints, tops = [_hull_xy(xy)], [float(top)]
    for poly, occ_top in occluders:
        footprints.append(_hull_xy(poly))
        tops.append(float(occ_top))
    return ObjectColumn(footprints=tuple(footprints), tops=tuple(tops), z_lo=float(table_z),
                        aim=np.array([position[0], position[1], max(tops)]))


# ── the tool and the acceptance rule ───────────────────────────────────────


def tool_samples(sweep, axis_order: str, spacing: float = TOOL_SPACING_M) -> np.ndarray:
    """(N, 3) TCP-frame samples: the TCP, then the open-jaw sweep box on a
    grid at most `spacing` apart. No sweep box: the TCP alone. A malformed
    box raises ValueError (`jaw_box`)."""
    from ..safety.tool_volume import box_grid, jaw_box

    if sweep is None or sweep.get("extents_open") is None:
        return np.zeros((1, 3))
    return np.vstack([np.zeros(3), box_grid(*jaw_box(sweep, axis_order), spacing)])


@dataclass(frozen=True)
class PoseClearance:
    """What one solved pose keeps (m): tool samples from the column, the
    lowest tool sample above the table, the joint origins from the column,
    and the aim point's distance from the approach ray."""

    tool_m: float
    table_m: float
    links_m: float
    aim_m: float

    def admissible(self, margin: float) -> bool:
        # written so that NaN refuses
        return bool(self.tool_m >= margin and self.table_m >= margin and self.links_m >= margin
                    and self.aim_m <= AIM_TOLERANCE_M)

    def worst(self, margin: float) -> str:
        """The first failed condition, in words."""
        if not self.tool_m >= margin:
            return f"the jaws {self.tool_m:.3f} m from it"
        if not self.table_m >= margin:
            return f"the jaws {self.table_m:.3f} m above the table"
        if not self.links_m >= margin:
            return f"a joint {self.links_m:.3f} m from it"
        return f"the aim {self.aim_m:.3f} m off its top's centre"


def _link_points(kin, q) -> np.ndarray:
    """The joint origins the harness checks (`link_positions(q)[1:]`)."""
    return np.asarray(kin.link_positions(q), dtype=float)[1:]


def pose_clearance(kin, q, column: ObjectColumn, tool: np.ndarray, approach_col: int,
                   table_z: float) -> PoseClearance:
    T = np.asarray(kin.fk(q), dtype=float)
    world = tool @ T[:3, :3].T + T[:3, 3]
    links = _link_points(kin, q)
    a = T[:3, approach_col]
    v = column.aim - T[:3, 3]
    along = max(float(v @ a), 0.0)
    return PoseClearance(tool_m=float(column.distance(world).min()),
                         table_m=float(world[:, 2].min()) - float(table_z),
                         links_m=float(column.distance(links).min()) if len(links) else math.inf,
                         aim_m=float(np.linalg.norm(v - along * a)))


# ── the lean-in family ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class LeanIn:
    """A found pose (q, tilt, distance, its clearance), or none (q is None)
    with the best rejected candidate's clearance for the refusal text."""

    q: np.ndarray | None
    tilt_deg: float = math.nan
    distance_m: float = math.nan
    clearance: PoseClearance | None = None
    solved: int = 0


def lean_in(kin, column: ObjectColumn, tool: np.ndarray, approach_col: int, q_seed, *, margin: float,
            table_z: float, vet=None) -> LeanIn:
    """The first admissible lean-in pose (see the module docstring), or
    LeanIn(q=None, ...). `vet(q)` (the harness's static `vet_pose`) may veto
    a candidate with a reason; the next candidate is tried."""
    aim = np.asarray(column.aim, dtype=float)
    pan = np.asarray(kin.link_positions(q_seed), dtype=float)[0]
    radial = (aim - pan)[:2]
    if not float(np.linalg.norm(radial)) > 1e-6:
        return LeanIn(q=None)
    u = np.array([*(radial / np.linalg.norm(radial)), 0.0])
    axis = np.eye(3)[:, approach_col]
    best, solved = None, 0
    for tilt in LEAN_IN_TILTS_DEG:
        phi = math.radians(tilt)
        approach = math.cos(phi) * np.array([0.0, 0.0, -1.0]) + math.sin(phi) * u
        for dist in LEAN_IN_DISTANCES_M:
            res = kin.ik_approach(aim - dist * approach, approach, np.asarray(q_seed, dtype=float), axis)
            if not res.success:
                continue
            solved += 1
            clr = pose_clearance(kin, res.q, column, tool, approach_col, table_z)
            if clr.admissible(margin) and (vet is None or vet(res.q) is None):
                return LeanIn(q=np.asarray(res.q, dtype=float), tilt_deg=tilt, distance_m=dist,
                              clearance=clr, solved=solved)
            if best is None or _score(clr) > _score(best):
                best = clr
    return LeanIn(q=None, clearance=best, solved=solved)


def _score(c: PoseClearance) -> float:
    vals = [c.tool_m, c.table_m, c.links_m]
    return min(vals) if all(math.isfinite(v) for v in vals) else -math.inf


# ── the route ──────────────────────────────────────────────────────────────


_WORDS = ("the jaws {} m from it", "the jaws {} m above the table", "a joint {} m from it")


def _pair_clearances(kin, q, column: ObjectColumn, tool: np.ndarray, table_z: float):
    """The clearances at q, one per pair that must not close in, and which
    kind each is (an index into _WORDS): the jaws (every tool sample) from
    the column (signed: negative inside), the jaws above the table, then each
    joint origin from the column (signed)."""
    T = np.asarray(kin.fk(q), dtype=float)
    world = tool @ T[:3, :3].T + T[:3, 3]
    links = _link_points(kin, q)
    values = np.concatenate([[column.distance(world).min(), world[:, 2].min() - float(table_z)],
                             column.distance(links) if len(links) else np.empty(0)]).astype(float)
    kinds = np.concatenate([[0, 1], np.full(len(links), 2)]).astype(int)
    values[~np.isfinite(values)] = np.nan
    return values, kinds


def _sample_clearance(kin, q, column: ObjectColumn, tool: np.ndarray, table_z: float) -> tuple[float, str]:
    """The smallest clearance at q (the jaws from the column and above the
    table, the joint origins from the column) and its words."""
    values, kinds = _pair_clearances(kin, q, column, tool, table_z)
    if np.isnan(values).any():
        return math.nan, _WORDS[int(kinds[np.flatnonzero(np.isnan(values))[0]])].format("nan")
    i = int(np.argmin(values))
    return float(values[i]), _WORDS[int(kinds[i])].format(f"{values[i]:.3f}")


def segment_violation(kin, q_from, q_to, column: ObjectColumn, tool: np.ndarray, margin: float,
                      table_z: float, step_rad: float = ROUTE_STEP_RAD, from_inside: bool = False) -> str | None:
    """Why the straight joint segment q_from -> q_to does not keep the margin
    (the first failing FK sample), or None. Samples at most `step_rad`
    apart, both ends included.

    `from_inside`: q_from is where the arm IS. Pair by pair (the jaws and
    the object, the jaws and the table, each joint and the object): a pair
    already inside the margin (the SO-101's home holds its jaw box 5 mm into
    the mock_small cube's top) may only separate, never close in (beyond
    float noise, ESCAPE_TOL_M), until it is out; a pair out of the margin
    never comes in. The object distance is signed, so deeper is closer, and
    one pair's slack never covers another closing in. Every other segment
    keeps the margin for every pair at every sample. MEASURED: an escape
    can still dip between two samples (densely resampled: 0.25 mm below its
    best so far on the Piper's escape from home, whose jaw box starts
    11.5 mm inside the mock box's hull)."""
    q_from = np.asarray(q_from, dtype=float)
    q_to = np.asarray(q_to, dtype=float)
    span = float(np.max(np.abs(q_to - q_from))) if q_from.size else 0.0
    if not math.isfinite(span):
        return "the joint targets are not finite"
    n = max(1, int(math.ceil(span / step_rad)))
    previous = None
    for i in range(n + 1):
        s = i / n
        values, kinds = _pair_clearances(kin, q_from + s * (q_to - q_from), column, tool, table_z)
        if from_inside and previous is None:
            previous = values                # where the arm is: judged by where it goes
            continue
        was_in = np.zeros(len(values), dtype=bool) if previous is None else ~(previous >= margin)
        # written so that NaN refuses; `previous` is only kept from_inside
        floor = np.where(was_in, (previous if previous is not None else values) - ESCAPE_TOL_M, margin)
        bad = ~(values >= floor)
        if bad.any():
            j = int(np.flatnonzero(bad)[np.argmin(np.nan_to_num(values[bad], nan=-np.inf))])
            text = "nan" if np.isnan(values[j]) else f"{values[j]:.3f}"
            where = "where it starts" if i == 0 else f"{100.0 * s:.0f}% of the way"
            return f"{_WORDS[int(kinds[j])].format(text)} {where}" + (
                ", closing in from inside the margin" if was_in[j] else "")
        previous = values if from_inside else None
    return None


def plan_route(kin, q_start, q_goal, column: ObjectColumn, tool: np.ndarray, approach_col: int, *,
               margin: float, table_z: float, vet=None, preflight=None, preflight_direct: bool = False) -> list:
    """The joint targets to command, in order, to reach `q_goal`: [q_goal]
    when the straight segment keeps the margin, else [rise waypoint, q_goal].
    An arm that starts inside the margin may only move away until it is out
    (`segment_violation(from_inside=True)`). Raises RouteRefused (nothing is
    commanded) when no rise-first waypoint clears it.

    `vet(q)` (the harness's static `vet_pose`) vetoes a waypoint;
    `preflight(q_from, q_to)` (the harness's static route vet,
    `safety.trajectory.vet_segment`) vetoes a leg the harness would stop
    mid-way (a TCP leaving the workspace box between two vetted ends). Every
    rise leg is preflighted; the direct leg only with `preflight_direct` (a
    new pose: the lean-in), so a top-down route that main streamed is
    streamed exactly as before."""
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)
    direct = segment_violation(kin, q_start, q_goal, column, tool, margin, table_z, from_inside=True)
    if direct is None and preflight_direct and preflight is not None:
        harness = preflight(q_start, q_goal)
        direct = None if harness is None else f"the harness would stop it ({harness})"
    if direct is None:
        return [q_goal]
    T0 = np.asarray(kin.fk(q_start), dtype=float)
    T1 = np.asarray(kin.fk(q_goal), dtype=float)
    base = {"start": float(T0[2, 3]), "higher": max(float(T0[2, 3]), float(T1[2, 3]))}
    heights = sorted({round(base[which] + extra, 9) for which, extra in RISE_WAYPOINTS})
    away = T0[:2, 3] - np.asarray(column.aim, dtype=float)[:2]
    away = away / np.linalg.norm(away) if float(np.linalg.norm(away)) > 1e-6 else np.zeros(2)
    axis = np.eye(3)[:, approach_col]
    for z in heights:
        for back in RISE_BACK_OFF_M:
            if back and not away.any():
                continue
            position = np.array([*(T0[:2, 3] + back * away), z])
            for waypoint in _rise_waypoints(kin, q_start, T0, T1, position, approach_col, axis):
                if vet is not None and vet(waypoint) is not None:
                    continue
                if (segment_violation(kin, q_start, waypoint, column, tool, margin, table_z,
                                      from_inside=True) is None
                        and segment_violation(kin, waypoint, q_goal, column, tool, margin, table_z) is None
                        and (preflight is None or (preflight(q_start, waypoint) is None
                                                   and preflight(waypoint, q_goal) is None))):
                    return [waypoint, q_goal]
    raise RouteRefused(f"going straight, {direct}, and no rise-first waypoint clears it")


def _rise_waypoints(kin, q_start, T0, T1, position, approach_col: int, axis: np.ndarray):
    """Solved joint poses with the TCP at `position`: the start's own
    rotation, then the start's and the goal's tilt aimed radially from the
    first (pan) joint (the plane a 5-joint arm can tilt in)."""
    target = T0.copy()
    target[:3, 3] = position
    res = kin.ik(target, q_start)
    if res.success:
        yield np.asarray(res.q, dtype=float)
    pan = np.asarray(kin.link_positions(q_start), dtype=float)[0]
    radial = np.array([position[0] - pan[0], position[1] - pan[1], 0.0])
    if not float(np.linalg.norm(radial)) > 1e-6:
        return
    radial /= np.linalg.norm(radial)
    for T in (T0, T1):
        down = -float(T[2, approach_col])
        tilt = math.acos(min(1.0, max(-1.0, down)))
        approach = math.cos(tilt) * np.array([0.0, 0.0, -1.0]) + math.sin(tilt) * radial
        res = kin.ik_approach(position, approach, q_start, axis)
        if res.success:
            yield np.asarray(res.q, dtype=float)
