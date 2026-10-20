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
(`neighbour_column`), kept the same margin as the target by every pose and leg
point_at commands (the top-down pose too, `neighbour_clearance`; the direct
leg, every rise-first leg, a withdrawal) and by the jaws' close.

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

The rest of the tool the harness models (`safety.tool_volume`: the reBot's
wrist camera box, a measured housing or fingers) rides along as `Part`s
(`tool_body`): each part's box sampled on its faces at most BODY_SPACING_M
apart, its own pair in every rule below (from each object and from the
table), named in a refusal. A part's distance from an object is its sampled
distance less `slack_m` (`part_slack`), a lower bound on the BOX's distance
wherever it keeps the margin, so a pose or leg a part clears keeps the whole
box the margin away. A profile without a tool volume has no part; a part
whose box the volume does not give refuses (its points alone bound nothing).

A pose is admissible (`PoseClearance.admissible`) when, on its solved FK,
every closed-jaw sample and every part keeps `margin` from the column and
from every neighbour and stays `margin` above the table, every joint origin
the harness checks keeps `margin` from the column and every neighbour, and
the approach ray passes within AIM_TOLERANCE_M of the aim point.

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
table: jaw samples, the parts and joint origins. An arm that already starts
inside the margin (the SO-101's home holds its jaw box over the mock_small
cube) may only move away until it is out, never closer, and between two
samples it never dips more than DIP_TOL_M below where it started (`_dip`, a
bound from the joints' lever arms, not more samples). A failing direct
segment gets one rise-first waypoint: over the start TCP (or backed off from
the object), at the start's height + 3 or 6 cm or at the higher of the start
and the goal (+0/3/6 cm), lowest first, then backed off 6 or 9 cm at the
start's height + 0-2 cm (`BACK_OFF_WAYPOINTS`), at the start's rotation or
tilted in the arm's plane; on the rise leg the jaws never come closer to the
target than where it started (sampled, and bounded between samples the same
way).

A jaw that starts within TOUCH_M of an object (judged on the jaws' real
geometry, `touch_distance`: a prop touching or inside the open jaws, where the
model's escape rule cannot promise the real parts do not push it) first backs
straight out along its approach, the TCP on a line (`withdrawal`), until it
keeps WITHDRAWN_M from every object; the route starts there. Never while
holding: the payload in the jaws is not modelled, so that start refuses.
Planning runs against a budget (`Budget`): PLAN_BUDGET_S of the planning
thread's CPU time (stretched when that thread's core, timed once before
planning, is slower than the host's fastest, `core_factor`) and PLAN_WALL_S of
wall time; past either, point_at refuses.
When more than FRESH_PLAN_S of wall time passed between the target's
localization and the end of planning (the budget's core-speed search
included), point_at looks at the target again before the first command, on a
detection newer than the plan only (never the remembered belief, never an
image received before that look started (FRESH_IN_FLIGHT_S aside) or whose
capture was first received more than FRESH_OBSERVED_S before it, never a
producer capture no newer than the one the plan was made from on that camera or
on a producer clock both share: not seen again, it refuses); moved
more than FRESH_MOVE_M (`fix_moved_m`), it is planned once more from where it
is now, on what is left of the same budget, and refused if it moved again;
never a move toward a stale pose.

The jaws close (`Route.close_at`) at the first pose of the route where their
`sweep` samples keep `margin` from the target, every neighbour and the table
(`close_violation`): where the arm starts, else at the rise-first waypoint or
the pointing pose. Every leg travelled before the close is checked with the
jaws as they are (`now`), every leg after it with the closed ones. A route
with no such close, or no clear waypoint, is refused before any command.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np

#: lean-in tilt from straight down, tried in this order (smallest first)
LEAN_IN_TILTS_DEG = (15.0, 25.0, 35.0, 45.0)
#: TCP distance from the aim point along the approach, per tilt, in order
LEAN_IN_DISTANCES_M = (0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10)
#: largest spacing of the jaw samples (faces and interior)
TOOL_SPACING_M = 0.005
#: largest spacing of a tool part's face samples (`tool_body`): the reBot's
#: 15 x 14 x 8 cm camera box is ~900 samples; its slack (`part_slack`) at the
#: 3 cm margin is 0.8 mm
BODY_SPACING_M = 0.01
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
#: a jaw (its real geometry, `touch_groups`) this close to an object where
#: the arm starts touches it, or nearly: the arm first withdraws straight back
#: along its approach (`withdrawal`), or nothing moves
TOUCH_M = 0.005
#: ... until every jaw keeps more than this from every object (no longer
#: touching: from there the escape rule holds the real parts off, since the
#: jaw model holds them)
WITHDRAWN_M = TOUCH_M
#: it backs out in steps of this (the TCP on its line), at most this far
WITHDRAW_STEP_M = 0.005
WITHDRAW_MAX_M = 0.08
#: the CPU time the planning thread may spend planning (s): past it, point_at
#: refuses. CPU time, not wall time: a loaded host (other processes on the
#: cores, other threads holding the GIL) stretches the wall time planning
#: waits, not the work it does (#385 review M2: rows that planned in < 2.5 s
#: at load 3 ran past a 3 s wall budget at load 8-10)
PLAN_BUDGET_S = 3.0
#: ... and never more than this wall-clock time, however loaded the host
#: (s): planning stays bounded (main once planned for 13.3 s)
PLAN_WALL_S = 8.0
#: A loaded GB10 also runs the planner on its Cortex-A725 cores, which take
#: 2.5-3x an X925's CPU time for the same plan (at load 2 pinned to one:
#: 6 of 32 so101_mock and 14 of 32 g1_mock rows ran past 3 s; none on an
#: X925). The CPU budget stretches by how much slower the planning thread's
#: core is than the host's fastest (`core_factor`, timed once, before
#: planning), never past PLAN_WALL_S; a core within this ratio of the fastest
#: counts as it (timing noise)
SLOWER_CORE_MIN = 1.25
#: Up to PLAN_WALL_S can pass between the target's localization and the
#: first command. When longer than this (s, wall, from the localization: the
#: budget's core-speed search counts, #406 review Minor 1) passed, point_at
#: looks at the target again before that command (#399 review Minor 3,
#: `fix_moved_m`), on a detection newer than the plan (#406 review) ...
FRESH_PLAN_S = 3.0
#: ... and plans once more, from where it is now, when it moved more than
#: this (m: its centre or its observed top), the pose's own aim tolerance
FRESH_MOVE_M = AIM_TOLERANCE_M
#: The look again's image must have been received after that look started,
#: or at most this long before it (s, its `Frame.t`): an image the camera's
#: pump had received but not yet published when the look started (its pose
#: and depth work comes after the stamp; the #413 review refused from 25 ms of
#: it on). A driver that stalled earlier hands back an older one: refused, and
#: an image from before the plan started is refused however recent.
FRESH_IN_FLIGHT_S = 0.1
#: ... and its capture FIRST received at most this long before that look
#: started (s, `freshness.observed_at`): a bridge re-sends a render in a new
#: Frame with a new receipt `Frame.t`, so a sim that froze during the plan
#: looks new (#413 review Minor 2: first received 3.03 s before the look; a
#: sim rendering slower than its bridge serves, 0.09 s). A bound 5x the benign
#: case and a sixth of FRESH_PLAN_S (the age a short plan acts on unseen).
FRESH_OBSERVED_S = 0.5
#: the planning clocks (tests replace them): the calling thread's CPU time,
#: and wall time
_clock = time.thread_time
_wall = time.monotonic


#: `ObjectColumn.min_distance` evaluates the exact distance this many points at a time
_MIN_CHUNK = 64


class RouteRefused(Exception):
    """No route to the pointing pose keeps the margin; nothing was commanded."""


class PlanningTimeout(RouteRefused):
    """Planning ran past its time budget; nothing was commanded."""


#: `_fastest_probe_s`, kept for the process once a search completed ("s"),
#: and the incomplete searches' answer with when it was taken ("partial")
_FASTEST: dict = {}
#: an incomplete core-speed search (a loaded host: a helper not given its core
#: in time) serves every call for this long (s, `_wall`) before a call
#: searches again: a search takes up to ~1.2 s, never on every call (#406
#: review Minor 1)
SEARCH_AGAIN_S = 60.0


def _speed_probe_s() -> float:
    """CPU seconds this thread takes for a fixed slice of the planner's own
    work (`ObjectColumn.min_distance`, 216 points, 8 times; ~0.4 ms on a
    GB10 X925)."""
    probe = _FASTEST.get("probe")
    if probe is None:
        square = np.array([[-0.02, -0.02], [0.02, -0.02], [0.02, 0.02], [-0.02, 0.02]])
        grid = np.stack(np.meshgrid(*[np.linspace(-0.05, 0.05, 6)] * 3, indexing="ij"), -1).reshape(-1, 3)
        probe = _FASTEST["probe"] = (ObjectColumn(footprints=(square,), tops=(0.05,), z_lo=0.0,
                                                  aim=np.array([0.0, 0.0, 0.05])), grid + [0.0, 0.1, 0.1])
    column, points = probe
    t0 = time.thread_time()
    for i in range(8):
        column.min_distance(points + i * 1e-5)
    return time.thread_time() - t0


def _fastest_probe_s() -> float:
    """`_speed_probe_s` on the host's fastest core (`core_speed.fastest`:
    timed by helper threads pinned to one core of each capacity class of the
    process's CPU set, fastest class first, the planning thread's affinity
    untouched), measured at the first use and kept for the process once a
    search completed. A search that ran out of time is never kept for good
    (it can only understate how much slower a core is, #399 review Minor 1):
    its answer, the least of every such search so far, serves every call for
    SEARCH_AGAIN_S, then a call searches again (#406 review Minor 1: never a
    search on every call, nor on every re-measure of `Budget.check`)."""
    if "s" in _FASTEST:
        return _FASTEST["s"]
    partial = _FASTEST.get("partial")
    if partial is not None and _wall() - partial[1] < SEARCH_AGAIN_S:      # NaN searches again
        return partial[0]
    from ..core_speed import fastest

    ref = fastest(lambda: _speed_probe_s())
    if ref.complete:
        _FASTEST["s"] = ref.seconds
        return ref.seconds
    seconds = ref.seconds if partial is None else min(partial[0], ref.seconds)
    _FASTEST["partial"] = (seconds, _wall())
    return seconds


def core_factor() -> float:
    """How many times slower than the host's fastest core this thread's
    core runs the planner's own work now (>= 1; 1 within SLOWER_CORE_MIN, or
    when it cannot tell). This thread's core is timed FIRST: finding the
    fastest (`_fastest_probe_s`) can search the host's cores with helper
    threads while this one sleeps, and the scheduler may wake it on an idle
    slower core that it leaves within milliseconds once busy (#423 review
    follow-up 5: an idle GB10 timed 2.6-2.7 on a Cortex-A725 there, then
    planned on X925s)."""
    now = min(_speed_probe_s() for _ in range(2))
    ref = _fastest_probe_s()
    ratio = now / ref if ref > 0.0 else math.nan
    return float(ratio) if math.isfinite(ratio) and ratio >= SLOWER_CORE_MIN else 1.0


class Budget:
    """point_at's planning deadline: `check()` raises PlanningTimeout once
    the thread that made it has spent `seconds` of CPU time since (`_clock`,
    PLAN_BUDGET_S) times `factor` (`core_factor`: how much slower its core
    is than the host's fastest, timed once, here, before planning), or `wall`
    seconds have passed (`_wall`, PLAN_WALL_S); the factor never stretches
    the CPU budget past `wall`. Nothing read later stretches it: a core timed
    again once the budget ran out read 1.29x on the fastest core (timing
    noise right after seconds of work) and admitted CPU time already spent
    there (#423 review follow-up 5); a thread moved to a slower core
    mid-plan can run out (an availability cost: the next call times the core
    it then runs on). Read
    from another thread, whose CPU clock says nothing about this one,
    `seconds` counts wall time (the strict rule)."""

    def __init__(self, seconds: float | None = None, wall: float | None = None):
        self.seconds = float(PLAN_BUDGET_S if seconds is None else seconds)
        self.wall_s = float(PLAN_WALL_S if wall is None else wall)
        self.thread = threading.get_ident()
        self.most = max(1.0, self.wall_s / self.seconds) if self.seconds > 0.0 else 1.0
        self.factor = min(core_factor(), self.most)
        self.start = _clock()
        start = _wall()
        self.wall_end = start + self.wall_s
        self.strict_end = start + self.seconds

    def check(self) -> None:
        # written so that NaN refuses
        now = _wall()
        if threading.get_ident() == self.thread:
            within = _clock() - self.start <= self.seconds * self.factor
        else:
            within = now <= self.strict_end
        if not within:
            slower = f" (x{self.factor:.1f} on a slower core)" if self.factor > 1.0 else ""
            raise PlanningTimeout(f"planning took longer than its {self.seconds:.1f} s budget{slower}")
        if not now <= self.wall_end:
            raise PlanningTimeout(f"planning took longer than its {self.wall_s:.1f} s wall-clock cap "
                                  f"(the host is loaded)")


def _check(budget) -> None:
    if budget is not None:
        budget.check()


def fix_moved_m(before, after) -> float:
    """How far a target moved between two fixes (`ObjectFix`): its centre,
    or its observed top (the highest point, what point_at keeps the margin
    above), whichever moved more (m). NaN when either is not finite."""
    def top(fix):
        pts = np.asarray(fix.points, dtype=float).reshape(-1, 3)
        return float(pts[:, 2].max()) if len(pts) else math.nan

    centre = float(np.linalg.norm(np.asarray(after.position, dtype=float) - np.asarray(before.position, dtype=float)))
    raised = abs(top(after) - top(before))
    return max(centre, raised) if math.isfinite(centre) and math.isfinite(raised) else math.nan


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

    def min_distance(self, points: np.ndarray) -> float:
        """`distance(points).min()`, exactly, evaluating only the points that
        can hold it: each point's signed distance to the column's bounding
        box (`box`, the table no way out) is a lower bound on its distance to
        the column, so points are taken nearest-bound first until the bound
        passes the least distance found."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if len(points) <= _MIN_CHUNK or not np.isfinite(points).all():
            return float(self.distance(points).min())
        lo, hi = self.box()
        gap = np.maximum(np.maximum(lo - points, points - hi), 0.0)
        depth = np.minimum.reduce([points[:, 0] - lo[0], hi[0] - points[:, 0], points[:, 1] - lo[1],
                                   hi[1] - points[:, 1]])
        bound = np.where(gap.any(axis=1), np.linalg.norm(gap, axis=1), np.maximum(-depth, points[:, 2] - hi[2]))
        order = np.argsort(bound, kind="stable")
        best = math.inf
        for start in range(0, len(order), _MIN_CHUNK):
            chunk = order[start:start + _MIN_CHUNK]
            if bound[chunk[0]] - 1e-12 >= best:
                break
            best = min(best, float(self.distance(points[chunk]).min()))
        return best


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
    """Tool samples as a tuple of (N, 3) groups (one array: one group; a
    `Part`: its samples)."""
    if isinstance(tool, (tuple, list)):
        return tuple(np.asarray(g.samples if isinstance(g, Part) else g, dtype=float).reshape(-1, 3) for g in tool)
    return (np.asarray(tool, dtype=float).reshape(-1, 3),)


def _stacked(tool) -> np.ndarray:
    return np.vstack(_groups(tool))


# ── the rest of the tool: the harness's parts beyond the jaws ──────────────


@dataclass(frozen=True, eq=False)
class Part:
    """A rigid part of the tool beyond the jaws (`tool_body`): TCP-frame
    `samples` on its box's faces, the `name` a refusal gives it, and
    `slack_m`, taken off every distance from it to an object."""

    samples: np.ndarray
    name: str
    slack_m: float = 0.0


def part_slack(spacing: float, margin: float) -> float:
    """How far a grid at most `spacing` apart on a box's faces can overstate
    the box's distance from a convex object, where the box keeps `margin`.

    The box point nearest the object (a corner, on an edge or on a face) has
    a sample at most spacing / sqrt(2) from it along the box, square to the
    direction of the object, so the sampled distance is at most
    sqrt(d^2 + spacing^2 / 2) <= d + spacing^2 / (4 d): a sampled distance
    of at least margin + spacing^2 / (4 margin) proves d >= margin. Never
    more than spacing / sqrt(2), which bounds it at any distance (also the
    slack without a positive margin)."""
    worst = spacing / math.sqrt(2.0)
    return min(worst, spacing * spacing / (4.0 * margin)) if margin > 0.0 else worst


def tool_body(tool_volume, margin: float, spacing: float = BODY_SPACING_M) -> tuple:
    """The harness's tool volume beyond the jaws as `Part`s, () without one.

    `tool_volume` is the arm harness's `safety.tool_volume.ToolVolume` (or
    None). Its "jaws" part is the open-jaw sweep box, which every jaw sample
    group already holds (`tool_samples`: the same `jaw_box`), so it is not
    repeated. Every other part is its box (`ToolVolume.boxes`) sampled on its
    faces at most `spacing` apart, with `part_slack` at `margin`. A part
    without its box raises ValueError (#399 review Minor 4): its own samples
    alone (the harness's 2 cm grid) could overstate its distance from an
    object by up to their spacing / sqrt(2), with no slack to cover it.
    (`build_tool_volume` gives every part its box.)"""
    if tool_volume is None:
        return ()
    names = tuple(tool_volume.names)
    boxes = tuple(getattr(tool_volume, "boxes", ()) or ())
    parts = []
    for k, name in enumerate(names):
        if name == "jaws":
            continue
        if len(boxes) != len(names):
            raise ValueError(f"the harness's tool part {name!r} has no box (safety.tool_volume's "
                             f"ToolVolume.boxes): point_at keeps the margin from a part's box, never from its "
                             f"sample points alone")
        parts.append(Part(_box_surface(*boxes[k], spacing), str(name), part_slack(spacing, margin)))
    return tuple(parts)


def _items(tool) -> list:
    return list(tool) if isinstance(tool, (tuple, list)) else [tool]


def _part_of(tool, which: int) -> str:
    """The name of the tool group a pair's `which` (`_pair_clearances`)
    names: "jaws" for a jaw group, else the part's own name."""
    if which > -2:
        return "jaws"
    return [g for g in _items(tool) if isinstance(g, Part)][-2 - which].name


def _posed(tool, T) -> list:
    """The tool at the TCP pose T as (points, slack, name): the jaw groups as
    one point set ("jaws", no slack), then each `Part`."""
    R, p = np.asarray(T, dtype=float)[:3, :3], np.asarray(T, dtype=float)[:3, 3]
    items = _items(tool)
    plain = [g for g in items if not isinstance(g, Part)]
    out = [(_stacked(plain) @ R.T + p, 0.0, "jaws")] if plain else []
    return out + [(np.asarray(g.samples, dtype=float).reshape(-1, 3) @ R.T + p, float(g.slack_m), g.name)
                  for g in items if isinstance(g, Part)]


def _least(column, posed) -> tuple[float, str]:
    """(distance, name) of the posed tool group nearest `column`, a part's
    distance less its slack. NaN wins."""
    best, name = math.inf, "jaws"
    for world, slack, part in posed:
        d = column.min_distance(world) - slack
        if math.isnan(d):
            return math.nan, part
        if d < best:
            best, name = d, part
    return best, name


def _lowest(posed, table_z: float) -> tuple[float, str]:
    """(height above the table, name) of the posed tool group lowest down. NaN wins."""
    best, name = math.inf, "jaws"
    for world, _, part in posed:
        h = float(world[:, 2].min()) - float(table_z)
        if math.isnan(h):
            return math.nan, part
        if h < best:
            best, name = h, part
    return best, name


def _as_jaws(jaws) -> Jaws:
    if isinstance(jaws, Jaws):
        return jaws
    groups = _groups(jaws)
    return Jaws(closed=groups, now=groups, sweep=groups)


@dataclass(frozen=True)
class PoseClearance:
    """What one solved pose keeps (m): the tool (jaws and parts) from the
    column, its lowest sample above the table, the joint origins from the
    column, the aim point's distance from the approach ray, and the tool and
    joint origins from the nearest known neighbour (`near` names it);
    `tool_part` / `table_part` name the tool group that set the first two.
    A part's distance from an object is less its slack."""

    tool_m: float
    table_m: float
    links_m: float
    aim_m: float
    near_m: float = math.inf
    near: str = ""
    tool_part: str = "jaws"
    table_part: str = "jaws"

    def admissible(self, margin: float) -> bool:
        # written so that NaN refuses
        return bool(self.tool_m >= margin and self.table_m >= margin and self.links_m >= margin
                    and self.near_m >= margin and self.aim_m <= AIM_TOLERANCE_M)

    def worst(self, margin: float) -> str:
        """The first failed condition, in words."""
        if not self.tool_m >= margin:
            return f"the {self.tool_part} {self.tool_m:.3f} m from it"
        if not self.table_m >= margin:
            return f"the {self.table_part} {self.table_m:.3f} m above the table"
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
    posed = _posed(tool, T)
    links = _link_points(kin, q)
    a = T[:3, approach_col]
    v = column.aim - T[:3, 3]
    along = max(float(v @ a), 0.0)
    near_m, near = _nearest(neighbours, posed, links)
    tool_m, tool_part = _least(column, posed)
    table_m, table_part = _lowest(posed, table_z)
    return PoseClearance(tool_m=tool_m, table_m=table_m,
                         links_m=float(column.distance(links).min()) if len(links) else math.inf,
                         aim_m=float(np.linalg.norm(v - along * a)), near_m=near_m, near=near,
                         tool_part=tool_part, table_part=table_part)


def column_min(column: ObjectColumn, world: np.ndarray, links: np.ndarray) -> float:
    """The smallest distance from the jaw samples and joint origins to it."""
    return _posed_min(column, [(world, 0.0, "jaws")], links)


def _posed_min(column: ObjectColumn, posed, links: np.ndarray) -> float:
    """The smallest distance from the posed tool (`_posed`: a part's less
    its slack) and the joint origins to it."""
    d, _ = _least(column, posed)
    if len(links):
        d = min(d, column.distance(links).min()) if not np.isnan(d) else d
    return float(d)


def _nearest(neighbours, posed, links: np.ndarray) -> tuple[float, str]:
    """(distance, name) of the neighbour nearest the posed tool (`_posed`)
    and the joint origins; (inf, "") without neighbours. NaN wins: it
    refuses."""
    near_m, near = math.inf, ""
    for other in neighbours:
        d = _posed_min(other, posed, links)
        if not d >= near_m:
            near_m, near = d, other.name
    return near_m, near


def neighbour_clearance(kin, q, tool, neighbours=()) -> tuple[float, str]:
    """(distance, name) from the tool (`tool`, TCP frame: the jaws and any
    `Part`) and the joint origins at q to the nearest known neighbour: what
    a top-down pose keeps from every other known object."""
    T = np.asarray(kin.fk(q), dtype=float)
    return _nearest(neighbours, _posed(tool, T), _link_points(kin, q))


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
            table_z: float, vet=None, neighbours=(), budget=None) -> LeanIn:
    """The first admissible lean-in pose (see the module docstring), or
    LeanIn(q=None, ...). `vet(q)` (the harness's static `vet_pose`) may veto
    a candidate with a reason; the next candidate is tried. `budget`
    (`Budget`) is checked before every candidate."""
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
            _check(budget)
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


def _cull_box(world: np.ndarray, box) -> float:
    """A lower bound on the distance from `world` points to a column inside
    the axis-aligned `box`: the distance between the two boxes."""
    lo, hi = box
    gap = np.maximum(np.maximum(lo - world.max(axis=0), world.min(axis=0) - hi), 0.0)
    return float(np.linalg.norm(gap))


def _pair_clearances(kin, q, columns, tool, table_z: float, margin: float | None = None):
    """The clearances at q, one per pair that must not close in, and what
    each pair is: `kinds` (0 a tool group from an object, 1 a tool group
    above the table, 2 a joint origin from an object), `owners` (the
    column's index, -1 the table) and `links` (the joint origin's index, -1
    a jaw group, -2 - k the k-th `Part`, `_part_of`).
    Per column: each tool sample group (`_groups`) from it (signed: negative
    inside; a part's less its slack), then each joint origin from it
    (signed); last the jaws above the table, then each part above it. Column
    0 is the target. With `margin`, a neighbour a group's bounding box keeps
    >= margin + 1 cm from gets that bound (a part's less its slack) instead
    of its distance (both clear it)."""
    T = np.asarray(kin.fk(q), dtype=float)
    groups = [g @ T[:3, :3].T + T[:3, 3] for g in _groups(tool)]
    items = _items(tool)
    slack = [float(g.slack_m) if isinstance(g, Part) else 0.0 for g in items]
    tags, k = [], 0
    for g in items:
        tags.append(-2 - k if isinstance(g, Part) else -1)
        k += isinstance(g, Part)
    links = _link_points(kin, q)
    values, kinds, owners, which = [], [], [], []
    for c, column in enumerate(columns):
        for g, world in enumerate(groups):
            jaws = None
            if margin is not None and c > 0:
                bound = _cull_box(world, column.box())
                if bound >= margin + 0.01:
                    jaws = bound
            values.append([(column.min_distance(world) if jaws is None else jaws) - slack[g]])
        values.append(column.distance(links) if len(links) else np.empty(0))
        kinds += [0] * len(groups) + [2] * len(links)
        owners += [c] * (len(groups) + len(links))
        which += tags + list(range(len(links)))
    plain = [world for world, tag in zip(groups, tags) if tag == -1]
    if plain:
        values.append([min(world[:, 2].min() for world in plain) - float(table_z)])
        kinds.append(1)
        owners.append(-1)
        which.append(-1)
    for world, tag in zip(groups, tags):
        if tag <= -2:
            values.append([world[:, 2].min() - float(table_z)])
            kinds.append(1)
            owners.append(-1)
            which.append(tag)
    values = np.concatenate([np.asarray(v, dtype=float).reshape(-1) for v in values])
    values[~np.isfinite(values)] = np.nan
    return values, np.asarray(kinds), np.asarray(owners), np.asarray(which)


def _words(kind: int, owner: int, value: float, columns, part: str = "jaws") -> str:
    text = "nan" if np.isnan(value) else f"{value:.3f}"
    if kind == 1:
        return f"the {part} {text} m above the table"
    if kind == 0:
        return f"the {part} {text} m from {columns[owner].name}"
    return f"a joint {text} m from {columns[owner].name}"


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
                      rise: bool = False, budget=None, held=None) -> str | None:
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
    is): each tool group (jaws and parts) never comes closer to the target
    than where the leg started, inside the margin or not (at every sample,
    and between two samples by more than DIP_TOL_M). `held` (a withdrawal
    step, from where
    the arm is: every pair's value where the withdrawal started, as
    `_pair_clearances` orders them): a pair that was inside the margin there
    never comes more than DIP_TOL_M closer than it was there (at every sample
    and between samples) instead of only separating; one that was out never
    comes in. Every other segment keeps the margin for every pair at every
    sample. `budget` (`Budget`) is checked at every pose evaluated."""
    columns = _columns(columns)
    q_from = np.asarray(q_from, dtype=float)
    q_to = np.asarray(q_to, dtype=float)
    span = float(np.max(np.abs(q_to - q_from))) if q_from.size else 0.0
    if not math.isfinite(span):
        return "the joint targets are not finite"
    n = max(1, int(math.ceil(span / step_rad)))
    cull = None if (from_inside or rise) else margin

    def evaluate(s):
        _check(budget)
        return _pair_clearances(kin, q_from + s * (q_to - q_from), columns, tool, table_z, cull)

    start = previous = None
    speed = None
    evals = [DIP_MAX_EVALS]
    for i in range(n + 1):
        s = i / n
        values, kinds, owners, links = evaluate(s)
        # the rise rule's pairs: every jaw group and the target
        jaws_target = (owners == 0) & (kinds == 0) if rise else np.zeros(len(values), dtype=bool)
        if from_inside and previous is None:
            previous = values                     # where the arm is: judged by where it goes
            start = values if held is None else np.asarray(held, dtype=float)
            continue
        if held is not None:
            # written so that NaN refuses
            was_in = ~(start >= margin)
            floor = np.where(was_in, start - DIP_TOL_M, margin)
        else:
            was_in = np.zeros(len(values), dtype=bool) if previous is None else ~(previous >= margin)
            # written so that NaN refuses; `previous` is only kept from_inside
            floor = np.where(was_in, (previous if previous is not None else values) - ESCAPE_TOL_M, margin)
        if start is not None:
            floor = np.where(jaws_target, np.maximum(floor, start - ESCAPE_TOL_M), floor)
        bad = ~(values >= floor)
        if bad.any():
            j = int(np.flatnonzero(bad)[np.argmin(np.nan_to_num(values[bad], nan=-np.inf))])
            where = "where it starts" if i == 0 else f"{100.0 * s:.0f}% of the way"
            closing = was_in[j] or jaws_target[j]
            return f"{_words(kinds[j], owners[j], values[j], columns, _part_of(tool, links[j]))} {where}" + (
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
                    speed = np.where(links >= 0, per_link[np.where(links >= 0, links, -1)],
                                     float(np.asarray(r_tool, dtype=float) @ dq))
                found = _dip(lambda t: evaluate(t)[0], previous, values, (i - 1) / n, s,
                             np.flatnonzero(watch), start - DIP_TOL_M, speed, evals)
                if found is not None:
                    j, at, value = found
                    part = _part_of(tool, links[j])
                    if math.isnan(value):
                        who = "a joint" if kinds[j] == 2 else f"the {part}"
                        return f"cannot bound how far {who} dips between two samples near {100.0 * at:.0f}% of the way"
                    return (f"{_words(kinds[j], owners[j], value, columns, part)} {100.0 * at:.0f}% of the way, "
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
        d = column.min_distance(world)
        if not d >= margin:
            return f"the jaws, closing, {'nan' if math.isnan(d) else f'{d:.3f}'} m from {column.name}"
    low = float(world[:, 2].min()) - float(table_z)
    if not low >= margin:
        return f"the jaws, closing, {'nan' if math.isnan(low) else f'{low:.3f}'} m above the table"
    return None


@dataclass(frozen=True)
class Route:
    """The joint targets to command, in order, and where the jaws close:
    before targets[close_at] (0: before the arm moves, or right after the
    withdrawal), after the last one when close_at == len(targets); None: no
    jaw command (holding). `withdraw`: a dense joint path that first backs
    the jaws straight out along their approach, `withdraw_m` far (empty: the
    arm starts clear of every object, `withdrawal`)."""

    targets: list
    close_at: int | None
    withdraw: list = field(default_factory=list)
    withdraw_m: float = 0.0


def touch_groups(jaws) -> tuple:
    """The jaws as they are, on their real geometry: a hinged jaw's measured
    `gripper.jaw_volume` (it holds the real parts, every face at most 5 mm
    out), else the sweep box (parallel jaws close inside it)."""
    jaws = _as_jaws(jaws)
    return jaws.now[1:] or jaws.now


def touch_distance(kin, q, columns, jaws) -> tuple[float, str]:
    """(distance, name) from the jaws' real geometry (`touch_groups`) at q to
    the nearest object column (signed: negative inside). NaN wins."""
    T = np.asarray(kin.fk(q), dtype=float)
    world = np.vstack(touch_groups(jaws)) @ T[:3, :3].T + T[:3, 3]
    best, name = math.inf, ""
    for column in _columns(columns):
        d = column.min_distance(world)
        if not d >= best:
            best, name = d, column.name
    return best, name


def withdrawal(kin, q_start, columns, jaws, approach_col: int, *, margin: float, table_z: float,
               preflight=None, budget=None, body=()) -> tuple[list, float]:
    """(dense joint path, distance): the jaws backed straight out along their
    approach axis, WITHDRAW_STEP_M at a time with the TCP on that line, until
    their real geometry keeps more than WITHDRAWN_M from every object (at most
    WITHDRAW_MAX_M). Each step is solved from the previous pose with no
    restart and no joint moving more than `planning.cartesian`'s branch-flip
    bound: the whole tool pose held (`Kinematics.ik`), else, on a 5-joint arm
    that cannot hold it, the approach held and its roll free
    (`Kinematics.ik_approach`). Each step is a route leg from where the arm
    is (`segment_violation(from_inside=True, held=...)`, the jaws as they
    are and the `body` parts): a pair inside the margin where the withdrawal
    started never comes
    more than DIP_TOL_M closer than it was there (a face sliding along the
    line wobbles by FK noise and the wrist's turn, it never only separates),
    one outside never comes in, and the harness (`preflight`) would let it
    run. Raises RouteRefused (nothing is commanded) when a step does not
    solve or closes in, or the line ends before the jaws are clear."""
    from ..planning.cartesian import MAX_JOINT_STEP_RAD

    q_prev = np.asarray(q_start, dtype=float)
    jaws = _as_jaws(jaws)
    T0 = np.asarray(kin.fk(q_prev), dtype=float)
    approach = T0[:3, approach_col]
    axis = np.eye(3)[:, approach_col]
    tool = jaws.now + tuple(body)
    held = _pair_clearances(kin, q_prev, _columns(columns), tool, table_z)[0]
    path = []
    for k in range(1, int(round(WITHDRAW_MAX_M / WITHDRAW_STEP_M)) + 1):
        _check(budget)
        dist = k * WITHDRAW_STEP_M
        goal = T0.copy()
        goal[:3, 3] = T0[:3, 3] - dist * approach
        q = _line_step(kin, goal, approach, axis, q_prev, MAX_JOINT_STEP_RAD)
        if q is None:
            raise RouteRefused(f"backing straight out along the approach, no pose solves {dist * 100:.1f} cm back")
        bad = segment_violation(kin, q_prev, q, columns, tool, margin, table_z, from_inside=True, budget=budget,
                                held=held)
        if bad is None and preflight is not None:
            stop = preflight(q_prev, q)
            bad = None if stop is None else f"the harness would stop it ({stop})"
        if bad is not None:
            raise RouteRefused(f"backing straight out along the approach, {dist * 100:.1f} cm back: {bad}")
        path.append(q)
        q_prev = q
        gap, name = touch_distance(kin, q, columns, jaws)
        if gap > WITHDRAWN_M:
            return path, dist
    raise RouteRefused(f"backing straight out along the approach {WITHDRAW_MAX_M * 100:.0f} cm still leaves the jaws "
                       f"{gap:.3f} m from {name}")


def _line_step(kin, goal, approach, axis, q_prev, max_jump: float):
    """The joint pose at `goal` solved from q_prev with no restart (the whole
    pose, else the approach with its roll free), or None when neither solves
    within `max_jump` of q_prev on every joint."""
    solvers = [lambda: kin.ik(goal, q_prev, retries=0)]
    if callable(getattr(kin, "ik_approach", None)):
        solvers.append(lambda: kin.ik_approach(goal[:3, 3], approach, q_prev, axis, retries=0))
    for solve in solvers:
        res = solve()
        q = np.asarray(res.q, dtype=float).reshape(-1)[: q_prev.size]
        if res.success and float(np.max(np.abs(q - q_prev))) <= max_jump:
            return q
    return None


def plan_route(kin, q_start, q_goal, column: ObjectColumn, jaws, approach_col: int, *,
               margin: float, table_z: float, neighbours=(),
               close: bool = True, vet=None, preflight=None, preflight_direct: bool = False,
               withdraw: bool = True, budget=None, body=(), holding: str | None = None) -> Route:
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
    `neighbours`: the other known objects' columns, kept on every leg (the
    direct one too, top-down or lean-in). `body`: the tool's `Part`s beyond
    the jaws (`tool_body`), carried on every leg and the withdrawal with the
    jaws as they are or closed (never part of a close or of a touch).

    A jaw that starts within TOUCH_M of an object (its real geometry,
    `touch_distance`: a prop touching or inside the open jaws) first backs
    straight out along the approach (`withdrawal`, `Route.withdraw`); the
    route is planned from where that ends. `withdraw` False (the arm cannot
    stream a joint path) refuses instead, and so does `holding` (the name of
    the object in the jaws: the payload is not modelled, so it never backs
    out with it).

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
    back, back_m = [], 0.0
    gap, name = touch_distance(kin, q_start, every, jaws)
    if not gap > TOUCH_M:          # NaN refuses too
        if holding:
            raise RouteRefused(f"the jaws start {gap:.3f} m from {name} while holding {holding!r}; they back out "
                               f"only empty (the payload is not modelled)")
        if not withdraw:
            raise RouteRefused(f"the jaws start {gap:.3f} m from {name} and this arm cannot back them straight out")
        try:
            back, back_m = withdrawal(kin, q_start, every, jaws, approach_col, margin=margin, table_z=table_z,
                                      preflight=preflight, budget=budget, body=body)
        except PlanningTimeout:
            raise
        except RouteRefused as exc:
            raise RouteRefused(f"the jaws start {gap:.3f} m from {name}; {exc}") from None
        q_start = back[-1]
    route = _plan_from(kin, q_start, q_goal, column, jaws, approach_col, every, margin=margin, table_z=table_z,
                       close=close, vet=vet, preflight=preflight, preflight_direct=preflight_direct or bool(back),
                       budget=budget, body=tuple(body))
    return Route(route.targets, route.close_at, back, back_m) if back else route


def _plan_from(kin, q_start, q_goal, column, jaws, approach_col, every, *, margin, table_z, close, vet, preflight,
               preflight_direct, budget, body=()) -> Route:
    """plan_route from a start the jaws do not touch."""
    first = (close_violation(kin, q_start, every, jaws.sweep, margin, table_z) if close else None)
    early = not close or first is None          # nothing left to close on the way
    before = (jaws.closed if (close and early) else jaws.now) + body
    at_start = (0 if close else None)

    def closes(q):
        return close_violation(kin, q, every, jaws.sweep, margin, table_z)

    direct = segment_violation(kin, q_start, q_goal, every, before, margin, table_z, from_inside=True, budget=budget)
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
        _check(budget)
        if back and not away.any():
            continue
        position = np.array([*(T0[:2, 3] + back * away), z])
        for waypoint in _rise_waypoints(kin, q_start, T0, T1, position, approach_col, axis):
            if vet is not None and vet(waypoint) is not None:
                continue
            if segment_violation(kin, q_start, waypoint, every, before, margin, table_z, from_inside=True,
                                 rise=True, budget=budget) is not None:
                continue
            if early:
                tool, close_at = before, at_start
            elif closes(waypoint) is None:
                tool, close_at = jaws.closed + body, 1
            elif closes(q_goal) is None:
                tool, close_at = jaws.now + body, 2
            else:
                continue
            if (segment_violation(kin, waypoint, q_goal, every, tool, margin, table_z, budget=budget) is None
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
