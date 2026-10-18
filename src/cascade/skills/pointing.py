"""point_at's pointing pose when no top-down pose clears, and its route.

All geometry is in the arm's base frame.

The object column (`ObjectColumn`) is the fix's oriented box extruded down to
the table, made a vertical prism over the convex xy hull of the box's corners
and the fix's observed points (the box is centred on `fix.position`, which
perception may recentre away from the points). Its height is the object top
point_at computed (the observed top, raised by the map's capped column top and
by an object stacked over it), never a PCA box corner: a box tilted by noise
would invent height. An object stacked over the target adds its own prism.
Every OTHER known object (a remembered belief) is a neighbour: its own prism
(`neighbour_column`), kept the same margin as the target by every motion this
module plans that main never made (the lean-in pose, its route, every
rise-first leg) and by the jaws' close.

The jaws (`Jaws`, `jaw_samples`) are TCP-frame samples on a grid at most
TOOL_SPACING_M apart: the TCP and the open-jaw sweep box
(`grasp.graspgenx.sweep`, `safety.tool_volume.jaw_box`), plus, for a hinged
jaw such as the SO-101's beak (it opens 62 mm to ONE side of a sweep box
centred on the TCP), the profile's measured jaw geometry
(`gripper.jaw_volume`: the fixed jaw, and the moving jaw turned about its
hinge): `closed` once they closed, `now` as the gripper reads before the
close, `sweep` every pose a close passes through from there. Without
`jaw_volume` (parallel jaws, which close inward inside the open-jaw box)
all three are the sweep box. Once the tool is tilted the point nearest the
object is in general not a box corner, hence the grid.

A pose is admissible (`PoseClearance.admissible`) when, on its solved FK,
every closed-jaw sample keeps `margin` from the column and from every
neighbour and stays `margin` above the table, every joint origin the harness
checks keeps `margin` from the column and every neighbour, and the approach
ray passes within AIM_TOLERANCE_M of the aim point.

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
joint space), at most ROUTE_STEP_RAD apart, against the same columns and the
table: jaw samples and joint origins. An arm that already starts inside the
margin (the SO-101's home holds its jaw box over the mock_small cube) may
only move away until it is out, never closer, and between two samples it
never dips more than DIP_TOL_M below where it started (`_dip`, a bound from
the joints' lever arms, not more samples). A failing direct segment gets one
rise-first waypoint: over the start TCP (or backed off from the object), at
the start's height + 3 or 6 cm or at the higher of the start and the goal
(+0/3/6 cm), lowest first, then backed off 6 or 9 cm at the start's height
+ 0-2 cm (`BACK_OFF_WAYPOINTS`), at the start's rotation or tilted in the
arm's plane; on the rise leg the jaws never come closer to the target than
they were (sampled, and bounded between samples the same way).

The jaws close (`Route.close_at`) at the first pose of the route where their
`sweep` samples keep `margin` from the target, every neighbour and the table
(`close_violation`): where the arm starts, else at the rise-first waypoint or
the pointing pose. Every leg travelled before the close is checked with the
jaws as they are (`now`), every leg after it with the closed ones. A route
with no such close, or no clear waypoint, is refused before any command.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: lean-in tilt from straight down, tried in this order (smallest first)
LEAN_IN_TILTS_DEG = (15.0, 25.0, 35.0, 45.0)
#: TCP distance from the aim point along the approach, per tilt, in order
LEAN_IN_DISTANCES_M = (0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10)
#: largest spacing of the jaw samples (faces and interior)
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
#: when none of those clears: (how far further back, how far above the start
#: TCP), tried in order. An open hinged jaw beside the object (the SO-101's
#: beak reaches 6 cm sideways) leaves it sideways before it rises.
BACK_OFF_WAYPOINTS = ((0.06, 0.01), (0.06, 0.02), (0.06, 0.0), (0.09, 0.01), (0.09, 0.02), (0.09, 0.0))
#: float noise allowed on a clearance that must not shrink (the harness's
#: escape floor, `safety.harness.CLEARANCE_ESCAPE_TOL_M`)
ESCAPE_TOL_M = 1e-9
#: a clearance that must not shrink (an escape's, a rise's jaws) never dips
#: more than this below where the leg started, between two samples either
DIP_TOL_M = 5e-5
#: the dip bound halves an interval at most this often, and evaluates at most
#: DIP_MAX_EVALS extra poses per leg; past either the leg is refused
DIP_MAX_DEPTH = 24
DIP_MAX_EVALS = 4000


class RouteRefused(Exception):
    """No route to the pointing pose keeps the margin; nothing was commanded."""


# ── the object columns ─────────────────────────────────────────────────────


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
    convex `footprints`, the aim point (the centre of the object's top) and
    how a refusal names it."""

    footprints: tuple
    tops: tuple
    z_lo: float
    aim: np.ndarray
    name: str = "it"

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

    def box(self) -> tuple[np.ndarray, np.ndarray]:
        """The column's axis-aligned bounds (lo, hi)."""
        xy = np.vstack(self.footprints)
        return (np.array([*xy.min(axis=0), self.z_lo]), np.array([*xy.max(axis=0), max(self.tops)]))


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
        poly = np.asarray(poly, dtype=float).reshape(-1, 2)
        if not (np.isfinite(poly).all() and math.isfinite(float(occ_top))):
            raise ValueError("an object standing over it has a non-finite box")
        footprints.append(_hull_xy(poly))
        tops.append(float(occ_top))
    return ObjectColumn(footprints=tuple(footprints), tops=tuple(tops), z_lo=float(table_z),
                        aim=np.array([position[0], position[1], max(tops)]))


def neighbour_column(footprint, top: float, table_z: float, name: str) -> ObjectColumn:
    """A known neighbour: its xy `footprint` (points, hulled) extruded from
    the table to `top`. Raises ValueError when any of it is not finite."""
    poly = np.asarray(footprint, dtype=float).reshape(-1, 2)
    if not (len(poly) and np.isfinite(poly).all() and math.isfinite(float(top)) and math.isfinite(table_z)):
        raise ValueError(f"{name}'s box is not finite")
    hull = _hull_xy(poly)
    centre = hull.mean(axis=0)
    return ObjectColumn(footprints=(hull,), tops=(float(top),), z_lo=float(table_z),
                        aim=np.array([centre[0], centre[1], float(top)]), name=name)


def _columns(columns) -> tuple:
    return (columns,) if isinstance(columns, ObjectColumn) else tuple(columns)


# ── the jaws and the acceptance rule ───────────────────────────────────────


def tool_samples(sweep, axis_order: str, spacing: float = TOOL_SPACING_M) -> np.ndarray:
    """(N, 3) TCP-frame samples: the TCP, then the open-jaw sweep box on a
    grid at most `spacing` apart. No sweep box: the TCP alone. A malformed
    box raises ValueError (`jaw_box`)."""
    from ..safety.tool_volume import box_grid, jaw_box

    if sweep is None or sweep.get("extents_open") is None:
        return np.zeros((1, 3))
    return np.vstack([np.zeros(3), box_grid(*jaw_box(sweep, axis_order), spacing)])


@dataclass(frozen=True)
class Jaws:
    """TCP-frame samples of the jaws: `closed` once they closed, `now` as
    they are before the close (the gripper's reading), `sweep` every pose a
    close sweeps from there to closed. Each is a tuple of sample groups (the
    sweep box, then a hinged jaw's measured geometry), and each group is its
    own pair in the route rule: one group's slack never covers another's
    closing in."""

    closed: tuple
    now: tuple
    sweep: tuple


def _box_surface(center, axes, half, spacing: float) -> np.ndarray:
    """The faces of an oriented box on a grid at most `spacing` apart (the
    columns stand on the table, so a column reaching into the box crosses a
    face: the faces see it)."""
    from ..safety.tool_volume import box_grid

    grid = box_grid(np.zeros(3), np.eye(3), half, spacing)
    on_face = np.isclose(np.abs(grid), np.asarray(half, dtype=float), rtol=0.0, atol=1e-12).any(axis=1)
    return np.asarray(center, dtype=float) + grid[on_face] @ np.asarray(axes, dtype=float).T


def _volume_boxes(volume, which: str, axis_order: str, spacing: float) -> np.ndarray:
    from ..safety.tool_volume import jaw_box

    boxes = volume.get(which)
    if not isinstance(boxes, (list, tuple)) or not boxes:
        raise ValueError(f"gripper.jaw_volume.{which} must be a non-empty list of boxes, got {boxes!r}")
    parts = []
    for i, b in enumerate(boxes):
        try:
            box = jaw_box({"extents_open": b.get("extents"), "offset_open": b.get("offset", [0.0, 0.0, 0.0])},
                          axis_order)
        except (AttributeError, ValueError):
            raise ValueError(f"gripper.jaw_volume.{which}[{i}] must have finite non-negative extents and a "
                             f"finite offset, got {b!r}") from None
        parts.append(_box_surface(*box, spacing))
    return np.vstack(parts)


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * k @ k


#: the gripper's reading is trusted to this (rad): the jaws as they are span
#: the reading +- this, and a close sweeps from the reading + this
JAW_READING_TOL_RAD = 0.05


def jaw_samples(sweep, gripper, axis_order: str, opening=None, spacing: float = TOOL_SPACING_M) -> Jaws:
    """The jaws (see the module docstring). `tool_samples`, plus, for a
    profile that declares `gripper.jaw_volume` (a hinged jaw), its measured
    geometry: `fixed` boxes (the fixed jaw and the gripper body) and
    `moving` boxes (the moving jaw at the gripper angle `stop_rad`), each
    `{extents: [x, y, z], offset: [x, y, z]}` in the sweep frame (+X the
    jaws' opening axis, +Y third, +Z approach, origin at the TCP), and the
    `hinge` ({point, axis}, same frame; a larger angle turns the moving jaw
    about it by the right-hand rule). The moving jaw is turned to every
    angle at most `spacing` of its tip apart: `closed` from `stop_rad` to
    the profile's `closed_pos`, `now` around `opening` (the gripper's reading
    in the same units; None or not finite: `open_pos`, the widest point_at's
    profile opens them), `sweep` from `stop_rad` to the wider of the two.
    Without `jaw_volume` (parallel jaws, which close inward inside the
    open-jaw box) all three are the sweep box. Raises ValueError for a
    malformed volume."""
    base = tool_samples(sweep, axis_order, spacing)
    volume = None if gripper is None else gripper.get("jaw_volume")
    if volume is None:
        return Jaws(closed=(base,), now=(base,), sweep=(base,))
    from ..safety.tool_volume import jaw_box

    if not hasattr(volume, "get"):
        raise ValueError(f"gripper.jaw_volume must be a mapping, got {volume!r}")
    fixed = _volume_boxes(volume, "fixed", axis_order, spacing)
    moving = _volume_boxes(volume, "moving", axis_order, spacing)
    hinge = volume.get("hinge") or {}
    try:
        _, axes, _ = jaw_box({"extents_open": [0.0, 0.0, 0.0]}, axis_order)
        point = axes @ np.asarray(hinge.get("point"), dtype=float).reshape(3)
        axis = axes @ np.asarray(hinge.get("axis"), dtype=float).reshape(3)
        stop = float(volume.get("stop_rad"))
        closed_pos, open_pos = float(gripper.get("closed_pos")), float(gripper.get("open_pos"))
    except (AttributeError, TypeError, ValueError):
        point = axis = np.full(3, np.nan)
        stop = closed_pos = open_pos = math.nan
    if not (np.isfinite(point).all() and np.isfinite(axis).all() and float(np.linalg.norm(axis)) > 1e-9
            and all(math.isfinite(v) for v in (stop, closed_pos, open_pos))):
        raise ValueError(f"gripper.jaw_volume needs a finite hinge {{point, axis}}, stop_rad and the gripper's "
                         f"closed_pos / open_pos, got {volume!r}")
    axis = axis / np.linalg.norm(axis)
    rel = moving - point
    radius = float(np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1).max())
    reading = math.nan if opening is None else float(opening)
    reading = max(reading if math.isfinite(reading) else open_pos, stop)

    def turned(a0: float, a1: float) -> np.ndarray:
        steps = max(1, int(math.ceil((a1 - a0) * radius / spacing)))
        return np.vstack([rel @ _rotation(axis, a - stop).T + point for a in np.linspace(a0, a1, steps + 1)])

    tol = JAW_READING_TOL_RAD
    return Jaws(closed=(base, np.vstack([fixed, turned(stop, max(closed_pos, stop) + tol)])),
                now=(base, np.vstack([fixed, turned(max(stop, reading - tol), reading + tol)])),
                sweep=(base, np.vstack([fixed, turned(stop, max(reading, closed_pos) + tol)])))


def _groups(tool) -> tuple:
    """Tool samples as a tuple of (N, 3) groups (one array: one group)."""
    return tuple(np.asarray(g, dtype=float).reshape(-1, 3) for g in tool) if isinstance(tool, (tuple, list)) \
        else (np.asarray(tool, dtype=float).reshape(-1, 3),)


def _stacked(tool) -> np.ndarray:
    return np.vstack(_groups(tool))


def _as_jaws(jaws) -> Jaws:
    if isinstance(jaws, Jaws):
        return jaws
    groups = _groups(jaws)
    return Jaws(closed=groups, now=groups, sweep=groups)


@dataclass(frozen=True)
class PoseClearance:
    """What one solved pose keeps (m): the jaws from the column, the lowest
    jaw sample above the table, the joint origins from the column, the aim
    point's distance from the approach ray, and the jaws and joint origins
    from the nearest known neighbour (`near` names it)."""

    tool_m: float
    table_m: float
    links_m: float
    aim_m: float
    near_m: float = math.inf
    near: str = ""

    def admissible(self, margin: float) -> bool:
        # written so that NaN refuses
        return bool(self.tool_m >= margin and self.table_m >= margin and self.links_m >= margin
                    and self.near_m >= margin and self.aim_m <= AIM_TOLERANCE_M)

    def worst(self, margin: float) -> str:
        """The first failed condition, in words."""
        if not self.tool_m >= margin:
            return f"the jaws {self.tool_m:.3f} m from it"
        if not self.table_m >= margin:
            return f"the jaws {self.table_m:.3f} m above the table"
        if not self.links_m >= margin:
            return f"a joint {self.links_m:.3f} m from it"
        if not self.near_m >= margin:
            return f"the arm {self.near_m:.3f} m from {self.near}"
        return f"the aim {self.aim_m:.3f} m off its top's centre"


def _link_points(kin, q) -> np.ndarray:
    """The joint origins the harness checks (`link_positions(q)[1:]`)."""
    return np.asarray(kin.link_positions(q), dtype=float)[1:]


def pose_clearance(kin, q, column: ObjectColumn, tool, approach_col: int,
                   table_z: float, neighbours=()) -> PoseClearance:
    T = np.asarray(kin.fk(q), dtype=float)
    world = _stacked(tool) @ T[:3, :3].T + T[:3, 3]
    links = _link_points(kin, q)
    a = T[:3, approach_col]
    v = column.aim - T[:3, 3]
    along = max(float(v @ a), 0.0)
    near_m, near = math.inf, ""
    for other in neighbours:
        d = column_min(other, world, links)
        if not d >= near_m:          # NaN wins: it refuses
            near_m, near = d, other.name
    return PoseClearance(tool_m=float(column.distance(world).min()),
                         table_m=float(world[:, 2].min()) - float(table_z),
                         links_m=float(column.distance(links).min()) if len(links) else math.inf,
                         aim_m=float(np.linalg.norm(v - along * a)), near_m=near_m, near=near)


def column_min(column: ObjectColumn, world: np.ndarray, links: np.ndarray) -> float:
    """The smallest distance from the jaw samples and joint origins to it."""
    d = column.distance(world).min()
    if len(links):
        d = min(d, column.distance(links).min()) if not np.isnan(d) else d
    return float(d)


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


def lean_in(kin, column: ObjectColumn, tool, approach_col: int, q_seed, *, margin: float,
            table_z: float, vet=None, neighbours=()) -> LeanIn:
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
            clr = pose_clearance(kin, res.q, column, tool, approach_col, table_z, neighbours)
            if clr.admissible(margin) and (vet is None or vet(res.q) is None):
                return LeanIn(q=np.asarray(res.q, dtype=float), tilt_deg=tilt, distance_m=dist,
                              clearance=clr, solved=solved)
            if best is None or _score(clr) > _score(best):
                best = clr
    return LeanIn(q=None, clearance=best, solved=solved)


def _score(c: PoseClearance) -> float:
    vals = [c.tool_m, c.table_m, c.links_m, c.near_m]
    return min(vals) if not any(math.isnan(v) for v in vals) else -math.inf


# ── the route ──────────────────────────────────────────────────────────────


_WORDS = ("the jaws {} m from {}", "the jaws {} m above the table", "a joint {} m from {}")


def _cull_box(world: np.ndarray, box) -> float:
    """A lower bound on the distance from `world` points to a column inside
    the axis-aligned `box`: the distance between the two boxes."""
    lo, hi = box
    gap = np.maximum(np.maximum(lo - world.max(axis=0), world.min(axis=0) - hi), 0.0)
    return float(np.linalg.norm(gap))


def _pair_clearances(kin, q, columns, tool, table_z: float, margin: float | None = None):
    """The clearances at q, one per pair that must not close in, and what
    each pair is: `kinds` (an index into _WORDS), `owners` (the column's
    index, -1 the table) and `links` (the joint origin's index, -1 the jaws).
    Per column: each tool sample group (`_groups`) from it (signed: negative
    inside), then each joint origin from it (signed); last the jaws above
    the table. Column 0 is the target. With `margin`, a neighbour a group's
    bounding box keeps >= margin + 1 cm from gets that bound instead of its
    distance (both clear it)."""
    T = np.asarray(kin.fk(q), dtype=float)
    groups = [g @ T[:3, :3].T + T[:3, 3] for g in _groups(tool)]
    links = _link_points(kin, q)
    values, kinds, owners, which = [], [], [], []
    for c, column in enumerate(columns):
        for world in groups:
            jaws = None
            if margin is not None and c > 0:
                bound = _cull_box(world, column.box())
                if bound >= margin + 0.01:
                    jaws = bound
            values.append([column.distance(world).min() if jaws is None else jaws])
        values.append(column.distance(links) if len(links) else np.empty(0))
        kinds += [0] * len(groups) + [2] * len(links)
        owners += [c] * (len(groups) + len(links))
        which += [-1] * len(groups) + list(range(len(links)))
    values.append([min(world[:, 2].min() for world in groups) - float(table_z)])
    kinds.append(1)
    owners.append(-1)
    which.append(-1)
    values = np.concatenate([np.asarray(v, dtype=float).reshape(-1) for v in values])
    values[~np.isfinite(values)] = np.nan
    return values, np.asarray(kinds), np.asarray(owners), np.asarray(which)


def _words(kind: int, owner: int, value: float, columns) -> str:
    text = "nan" if np.isnan(value) else f"{value:.3f}"
    if kind == 1:
        return _WORDS[1].format(text)
    return _WORDS[kind].format(text, columns[owner].name)


def _levers(kin, q_a, q_b, tool):
    """Per joint j, upper bounds on how far each pair's points can move per
    radian of joint j, valid at every pose of the segment q_a -> q_b: (R_tool
    (n,), R_links (m, n)) for the jaw samples and each checked joint origin.
    A kinematics may provide `lever_arms(q, tool)` itself; otherwise they come
    from a serial chain of revolute joints, `link_positions` giving each
    joint's origin (on its axis): a point moves at most |dq_j| times its
    distance from joint j's origin, which the chain's link lengths bound at
    every pose. None when the chain is not that (a link length or the jaws'
    distance from the last joint differs between the two ends)."""
    tool = _stacked(tool)
    custom = getattr(kin, "lever_arms", None)
    if callable(custom):
        return custom(q_a, tool)
    n = len(np.asarray(q_a).reshape(-1))
    oa = np.asarray(kin.link_positions(q_a), dtype=float)
    ob = np.asarray(kin.link_positions(q_b), dtype=float)
    if oa.shape != ob.shape or len(oa) < n or n < 1:
        return None
    seg_a = np.linalg.norm(np.diff(oa, axis=0), axis=1)
    seg_b = np.linalg.norm(np.diff(ob, axis=0), axis=1)
    rigid_a = np.linalg.norm(oa[n - 1:] - oa[n - 1], axis=1)
    rigid_b = np.linalg.norm(ob[n - 1:] - ob[n - 1], axis=1)
    Ta, Tb = np.asarray(kin.fk(q_a), dtype=float), np.asarray(kin.fk(q_b), dtype=float)
    da = np.linalg.norm(tool @ Ta[:3, :3].T + Ta[:3, 3] - oa[n - 1], axis=1)
    db = np.linalg.norm(tool @ Tb[:3, :3].T + Tb[:3, 3] - ob[n - 1], axis=1)
    if not (np.allclose(seg_a, seg_b, rtol=0.0, atol=1e-6) and np.allclose(rigid_a, rigid_b, rtol=0.0, atol=1e-6)
            and np.allclose(da, db, rtol=0.0, atol=1e-6)):
        return None
    seg = np.maximum(seg_a, seg_b)
    chain = np.array([seg[j:n - 1].sum() for j in range(n)])      # joint j's origin -> joint n-1's
    r_tool = chain + float(np.max(np.maximum(da, db)))
    m = len(oa) - 1                                                 # the checked origins: oa[1:]
    r_links = np.zeros((m, n))
    for k in range(1, len(oa)):
        for j in range(min(k, n)):
            r_links[k - 1, j] = seg[j:k].sum() if k <= n - 1 else chain[j] + max(rigid_a[k - n + 1],
                                                                                rigid_b[k - n + 1])
    return r_tool, r_links


def _dip(evaluate, values_a, values_b, s_a: float, s_b: float, watch: np.ndarray, floors: np.ndarray,
         speed: np.ndarray, budget: list) -> tuple[int, float, float] | None:
    """Between two samples s_a < s_b: each watched pair i keeps
    >= floors[i] at every pose (it is 1-Lipschitz in its points, which move
    at most speed[i] per unit s, so on [a, b] it is >= (v_a + v_b -
    speed (b - a)) / 2); an interval that bound does not clear is halved and
    its midpoint checked. Returns None, or (pair, s, value) of a midpoint
    below its floor (value) or of an interval it could not clear (NaN)."""
    stack = [(s_a, s_b, values_a, values_b, 0)]
    while stack:
        a, b, va, vb, depth = stack.pop()
        lower = (va[watch] + vb[watch] - speed[watch] * (b - a)) / 2.0
        need = watch[~(lower >= floors[watch])]
        if not need.size:
            continue
        if depth >= DIP_MAX_DEPTH or budget[0] <= 0:
            return int(need[0]), 0.5 * (a + b), math.nan
        mid = 0.5 * (a + b)
        vm = evaluate(mid)
        budget[0] -= 1
        low = need[~(vm[need] >= floors[need])]
        if low.size:
            return int(low[0]), mid, float(vm[low[0]])
        stack.append((mid, b, vm, vb, depth + 1))
        stack.append((a, mid, va, vm, depth + 1))
    return None


def segment_violation(kin, q_from, q_to, columns, tool, margin: float,
                      table_z: float, step_rad: float = ROUTE_STEP_RAD, from_inside: bool = False,
                      rise: bool = False) -> str | None:
    """Why the straight joint segment q_from -> q_to does not keep the margin
    (the first failing FK sample), or None. `columns`: the target's column
    first, then each neighbour's (or a single column). Samples at most
    `step_rad` apart, both ends included.

    `from_inside`: q_from is where the arm IS. Pair by pair (the jaws and an
    object, the jaws and the table, each joint and an object): a pair
    already inside the margin (the SO-101's home holds its jaw box 5 mm into
    the mock_small cube's top) may only separate, never close in (beyond
    float noise, ESCAPE_TOL_M), until it is out; a pair out of the margin
    never comes in. The object distance is signed, so deeper is closer, and
    one pair's slack never covers another closing in. Between two samples a
    pair still inside never dips more than DIP_TOL_M below its value where
    the arm starts (`_dip`). `rise` (a rise-first leg, from where the arm
    is): the jaws never come closer to the target than the previous sample,
    inside the margin or not, and never dip more than DIP_TOL_M below where
    they started between two samples. Every other segment keeps the margin
    for every pair at every sample."""
    columns = _columns(columns)
    q_from = np.asarray(q_from, dtype=float)
    q_to = np.asarray(q_to, dtype=float)
    span = float(np.max(np.abs(q_to - q_from))) if q_from.size else 0.0
    if not math.isfinite(span):
        return "the joint targets are not finite"
    n = max(1, int(math.ceil(span / step_rad)))
    cull = None if (from_inside or rise) else margin

    def evaluate(s):
        return _pair_clearances(kin, q_from + s * (q_to - q_from), columns, tool, table_z, cull)

    start = previous = None
    speed = None
    budget = [DIP_MAX_EVALS]
    for i in range(n + 1):
        s = i / n
        values, kinds, owners, links = evaluate(s)
        # the rise rule's pairs: every jaw group and the target
        jaws_target = (owners == 0) & (kinds == 0) if rise else np.zeros(len(values), dtype=bool)
        if from_inside and previous is None:
            start = previous = values             # where the arm is: judged by where it goes
            continue
        was_in = np.zeros(len(values), dtype=bool) if previous is None else ~(previous >= margin)
        # written so that NaN refuses; `previous` is only kept from_inside
        floor = np.where(was_in, (previous if previous is not None else values) - ESCAPE_TOL_M, margin)
        if previous is not None:
            floor = np.where(jaws_target, previous - ESCAPE_TOL_M, floor)
        bad = ~(values >= floor)
        if bad.any():
            j = int(np.flatnonzero(bad)[np.argmin(np.nan_to_num(values[bad], nan=-np.inf))])
            where = "where it starts" if i == 0 else f"{100.0 * s:.0f}% of the way"
            closing = was_in[j] or jaws_target[j]
            return f"{_words(kinds[j], owners[j], values[j], columns)} {where}" + (
                (", closing in from inside the margin" if was_in[j] else ", closing in on the rise")
                if closing else "")
        if from_inside and previous is not None and start is not None:
            watch = was_in | jaws_target
            if watch.any():
                if speed is None:
                    levers = _levers(kin, q_from, q_to, tool)
                    if levers is None:
                        return "cannot bound the motion between two samples (not a serial revolute chain)"
                    r_tool, r_links = levers
                    dq = np.abs(q_to - q_from)
                    per_link = np.concatenate([np.asarray(r_links, dtype=float).reshape(-1, len(dq)) @ dq, [0.0]])
                    speed = np.where(links >= 0, per_link[links], float(np.asarray(r_tool, dtype=float) @ dq))
                found = _dip(lambda t: evaluate(t)[0], previous, values, (i - 1) / n, s,
                             np.flatnonzero(watch), start - DIP_TOL_M, speed, budget)
                if found is not None:
                    j, at, value = found
                    if math.isnan(value):
                        who = "a joint" if kinds[j] == 2 else "the jaws"
                        return f"cannot bound how far {who} dips between two samples near {100.0 * at:.0f}% of the way"
                    return (f"{_words(kinds[j], owners[j], value, columns)} {100.0 * at:.0f}% of the way, "
                            f"between two samples, closing in from where it started")
            previous = values
    return None


def close_violation(kin, q, columns, sweep, margin: float, table_z: float) -> str | None:
    """Why the jaws may not close at q (their `sweep` samples, every pose a
    close sweeps, within `margin` of an object or the table), or None."""
    columns = _columns(columns)
    T = np.asarray(kin.fk(q), dtype=float)
    world = _stacked(sweep) @ T[:3, :3].T + T[:3, 3]
    for column in columns:
        d = float(column.distance(world).min())
        if not d >= margin:
            return f"the jaws, closing, {'nan' if math.isnan(d) else f'{d:.3f}'} m from {column.name}"
    low = float(world[:, 2].min()) - float(table_z)
    if not low >= margin:
        return f"the jaws, closing, {'nan' if math.isnan(low) else f'{low:.3f}'} m above the table"
    return None


@dataclass(frozen=True)
class Route:
    """The joint targets to command, in order, and where the jaws close:
    before targets[close_at] (0: before the arm moves), after the last one
    when close_at == len(targets); None: no jaw command (holding)."""

    targets: list
    close_at: int | None


def plan_route(kin, q_start, q_goal, column: ObjectColumn, jaws, approach_col: int, *,
               margin: float, table_z: float, neighbours=(), direct_neighbours: bool = True,
               close: bool = True, vet=None, preflight=None, preflight_direct: bool = False) -> Route:
    """The route to `q_goal` and where the jaws close (`Route`): [q_goal]
    when the straight segment keeps the margin, else [rise waypoint, q_goal].
    An arm that starts inside the margin may only move away until it is out
    (`segment_violation(from_inside=True)`); the jaws never come closer to
    the target on the rise (`rise=True`). Raises RouteRefused (nothing is
    commanded) when no route clears it or the jaws can close nowhere on it.

    `jaws`: `Jaws` (or one sample array for both). `close`: the jaws close
    on the way (False while holding: no close, the jaws as they are,
    `jaws.now`, throughout); they close at the first of the start, the
    waypoint and the goal where `close_violation` of `jaws.sweep` clears
    every column and the table; legs before the close are checked with
    `jaws.now`, legs after it with `jaws.closed`.
    `neighbours`: the other known objects' columns, kept on every leg except
    the direct one when `direct_neighbours` is False (a top-down pose: the
    leg main streamed).

    `vet(q)` (the harness's static `vet_pose`) vetoes a waypoint;
    `preflight(q_from, q_to)` (the harness's static route vet,
    `safety.trajectory.vet_segment`) vetoes a leg the harness would stop
    mid-way (a TCP leaving the workspace box between two vetted ends). Every
    rise leg is preflighted; the direct leg only with `preflight_direct` (a
    new pose: the lean-in), so a top-down route that main streamed is
    streamed exactly as before."""
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)
    jaws = _as_jaws(jaws)
    every = (column, *neighbours)
    first = (close_violation(kin, q_start, every, jaws.sweep, margin, table_z) if close else None)
    early = not close or first is None          # nothing left to close on the way
    before = jaws.closed if (close and early) else jaws.now
    at_start = (0 if close else None)

    def closes(q):
        return close_violation(kin, q, every, jaws.sweep, margin, table_z)

    direct = segment_violation(kin, q_start, q_goal, every if direct_neighbours else (column,), before, margin,
                               table_z, from_inside=True)
    if direct is None and preflight_direct and preflight is not None:
        harness = preflight(q_start, q_goal)
        direct = None if harness is None else f"the harness would stop it ({harness})"
    if direct is None:
        if early:
            return Route([q_goal], at_start)
        at_goal = closes(q_goal)
        if at_goal is None:
            return Route([q_goal], 1)
        direct = f"the jaws may close neither where it starts ({first}) nor at the pointing pose ({at_goal})"
    T0 = np.asarray(kin.fk(q_start), dtype=float)
    T1 = np.asarray(kin.fk(q_goal), dtype=float)
    base = {"start": float(T0[2, 3]), "higher": max(float(T0[2, 3]), float(T1[2, 3]))}
    heights = sorted({round(base[which] + extra, 9) for which, extra in RISE_WAYPOINTS})
    away = T0[:2, 3] - np.asarray(column.aim, dtype=float)[:2]
    away = away / np.linalg.norm(away) if float(np.linalg.norm(away)) > 1e-6 else np.zeros(2)
    axis = np.eye(3)[:, approach_col]
    candidates = [(z, back) for z in heights for back in RISE_BACK_OFF_M]
    candidates += [(round(base["start"] + up, 9), back) for back, up in BACK_OFF_WAYPOINTS]
    for z, back in candidates:
        if back and not away.any():
            continue
        position = np.array([*(T0[:2, 3] + back * away), z])
        for waypoint in _rise_waypoints(kin, q_start, T0, T1, position, approach_col, axis):
            if vet is not None and vet(waypoint) is not None:
                continue
            if segment_violation(kin, q_start, waypoint, every, before, margin, table_z, from_inside=True,
                                 rise=True) is not None:
                continue
            if early:
                tool, close_at = before, at_start
            elif closes(waypoint) is None:
                tool, close_at = jaws.closed, 1
            elif closes(q_goal) is None:
                tool, close_at = jaws.now, 2
            else:
                continue
            if (segment_violation(kin, waypoint, q_goal, every, tool, margin, table_z) is None
                    and (preflight is None or (preflight(q_start, waypoint) is None
                                               and preflight(waypoint, q_goal) is None))):
                return Route([waypoint, q_goal], close_at)
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
