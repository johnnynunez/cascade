"""The pregrasp hover counts from the target's top in the MAP (spec §3.2 rules 3-4).

On the physical reBot the occupancy map put a target's top 0.2-0.9 cm above
the top perception measured (1 cm voxels + depth noise), so a hover computed
from the perception top alone left the TCP 2.1-2.8 cm from the map's top,
inside the 3 cm margin. With `grasp.pregrasp_map_top` the hover is

    fix_top + min(max(map_top - fix_top, 0), 2 cm) + pregrasp_clear_top_m + voxel/2

where `map_top` is the top face of the highest occupied voxel in the target's
own column (the approach column's footprint around the fix XY), read under the
map's refresh lock, and the half-voxel term is at most 5 mm. It only ever rises
compared with the perception-only hover, the raise is capped so that one residue
voxel cannot push it out of reach, and a stale map, an empty column or a
non-finite answer keeps today's hover.

The rule holds for the pose that EXECUTES: a raised hover goes to select_grasp
as a descending ladder (map hover, today's hover), so a map hover that misses
IK or vetting falls to today's hover, and only then to the fixed offset.
"""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.grasping.selector import select_grasp
from cascade.skills.runtime import _pregrasp_min_z
from cascade.types import Grasp
from occupancy_grid_bridge import TABLE, GridBridge, centred_box, depth_frame, make_map

FIX_TOP, CLEAR, VOXEL = .093, .03, .01
CUP_XY = np.array([0.305, 0.005])       # on a voxel centre of make_map's 1 cm grid
GCFG_ON = {"pregrasp_clear_top_m": CLEAR, "pregrasp_map_top": True, "exempt_radius_m": 0.15}
TABLE_FLOOR = 0.01                      # table_z + table_clearance of the stand-in harness

# the map's cup: a box whose top voxel centres sit at 0.095 (top face 0.100)
CUP_BOX = centred_box([*CUP_XY, 0.095 / 2], [0.06, 0.06, 0.095])
# a taller neighbour 8 cm away (centre to centre), its near face 6 cm from the cup's axis
NEIGHBOUR_BOX = centred_box([CUP_XY[0] + 0.08, CUP_XY[1], 0.15 / 2], [0.04, 0.04, 0.15])
# a 15 cm stick touching the cup: its voxels lie inside the cup's 4 cm column
STICK_BOX = centred_box([CUP_XY[0] + 0.04, CUP_XY[1], 0.15 / 2], [0.02, 0.02, 0.15])
TODAY = FIX_TOP + CLEAR                 # today's hover: the observed top + clear_top


def cup_fix(top=FIX_TOP, r=0.03, n=40):
    """Perception's cup: wall points on a ring of radius `r` around the fix
    XY, from the table to `top` (the 95th-percentile footprint is exactly `r`)."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    side = np.c_[CUP_XY[0] + r * np.cos(a), CUP_XY[1] + r * np.sin(a), np.linspace(0.0, top, n)]
    return SimpleNamespace(position=np.array([*CUP_XY, top / 2]), points=side)


def harness_with(column_top=None, *, voxel=VOXEL, calls=None):
    """A harness whose map answers `column_top` with (column_top, voxel), or
    None (stale map / empty column); every query is recorded in `calls`."""

    def query(center_xy, radius_m, z_min):
        if calls is not None:
            calls.append((np.asarray(center_xy, dtype=float).copy(), float(radius_m), float(z_min)))
        return None if column_top is None else (column_top, voxel)

    return SimpleNamespace(limits=SimpleNamespace(table_z=0.0, table_clearance=TABLE_FLOOR),
                           occupancy=SimpleNamespace(column_top=query))


def fresh_map(*boxes):
    occ = make_map(GridBridge([TABLE, *boxes], voxel=VOXEL), unobserved="free")
    occ.refresh(depth_frame(), np.eye(4))
    assert occ.last_error is None and occ.usable()
    return occ


# ── the hover rule ────────────────────────────────────────────────────────


@pytest.mark.parametrize("map_top, expected", [
    (None,           TODAY),                                         # stale map / empty column: today's hover
    (FIX_TOP + .009, (FIX_TOP + .009 + CLEAR + VOXEL / 2, TODAY)),   # the recorded 0.2-0.9 cm gaps
    (FIX_TOP + .05,  (FIX_TOP + .02 + CLEAR + VOXEL / 2, TODAY)),    # one residue voxel: raise capped at 2 cm
    (FIX_TOP - .01,  (FIX_TOP + CLEAR + VOXEL / 2, TODAY)),          # never below the fix top
])
def test_hover_from_the_map_top(map_top, expected):
    """A raised hover is the ladder (map hover, today's hover): select_grasp
    falls back to today's hover before the fixed offset."""
    assert _pregrasp_min_z(GCFG_ON, harness_with(column_top=map_top), cup_fix()) == pytest.approx(expected)


def test_a_coarse_voxel_adds_at_most_half_a_centimetre():
    """The half-voxel term is min(voxel/2, 5 mm): on a 5 cm-voxel bridge the
    hover rises by the 2 cm cap + 5 mm, never by 2 + 2.5 cm."""
    harness = harness_with(column_top=FIX_TOP + .05, voxel=.05)
    assert _pregrasp_min_z(GCFG_ON, harness, cup_fix()) == pytest.approx((FIX_TOP + .02 + CLEAR + .005, TODAY))
    harness = harness_with(column_top=FIX_TOP + .009, voxel=.05)
    assert _pregrasp_min_z(GCFG_ON, harness, cup_fix()) == pytest.approx((FIX_TOP + .009 + CLEAR + .005, TODAY))


@pytest.mark.parametrize("answer", [
    (float("nan"), VOXEL),             # a NaN top
    (FIX_TOP + .005, float("nan")),    # a NaN voxel
    (FIX_TOP + .005, -float("inf")),   # a voxel that would sink the hover without bound
])
def test_a_non_finite_map_answer_keeps_todays_hover(answer):
    """A duck-typed map whose column_top answers garbage: today's hover, a
    plain float (no ladder), never NaN."""
    harness = SimpleNamespace(limits=SimpleNamespace(table_z=0.0, table_clearance=TABLE_FLOOR),
                              occupancy=SimpleNamespace(column_top=lambda center_xy, radius_m, z_min: answer))
    result = _pregrasp_min_z(GCFG_ON, harness, cup_fix())
    assert isinstance(result, float) and result == pytest.approx(TODAY)


def test_flag_off_or_clear_top_null_keeps_todays_hover():
    high = harness_with(column_top=FIX_TOP + .05)
    assert _pregrasp_min_z({**GCFG_ON, "pregrasp_map_top": False}, high, cup_fix()) == pytest.approx(TODAY)
    off_by_default = {k: v for k, v in GCFG_ON.items() if k != "pregrasp_map_top"}
    assert _pregrasp_min_z(off_by_default, high, cup_fix()) == pytest.approx(TODAY)
    assert _pregrasp_min_z({**GCFG_ON, "pregrasp_clear_top_m": None}, high, cup_fix()) is None
    no_clear_top = {k: v for k, v in GCFG_ON.items() if k != "pregrasp_clear_top_m"}
    assert _pregrasp_min_z(no_clear_top, high, cup_fix()) is None


def test_no_map_keeps_todays_hover():
    limits = SimpleNamespace(table_z=0.0, table_clearance=TABLE_FLOOR)
    for occupancy in (None, SimpleNamespace(clearance=lambda points: None)):   # no map; a map without column_top
        harness = SimpleNamespace(limits=limits, occupancy=occupancy)
        assert _pregrasp_min_z(GCFG_ON, harness, cup_fix()) == pytest.approx(TODAY)


def test_unusable_target_points_keep_todays_answer():
    calls = []
    empty = SimpleNamespace(position=np.array([*CUP_XY, 0.05]), points=np.zeros((0, 3)))
    assert _pregrasp_min_z(GCFG_ON, harness_with(FIX_TOP + .05, calls=calls), empty) is None
    # a footprint that cannot be measured (non-finite XY) gives no column: the
    # observed top still sets today's hover, and the map is never asked
    fix = cup_fix()
    fix.points[0, 0] = np.nan
    assert _pregrasp_min_z(GCFG_ON, harness_with(FIX_TOP + .05, calls=calls), fix) == pytest.approx(TODAY)
    assert calls == []


# ── which column the map is asked about (ruling R13) ─────────────────────


@pytest.mark.parametrize("top, z_min", [
    (FIX_TOP, FIX_TOP - 0.02),     # 2 cm under the observed top
    (0.015, TABLE_FLOOR),          # a flat target keeps the table floor
])
def test_the_map_is_asked_about_the_targets_own_column(top, z_min):
    """Centre = the fix XY; radius = the observed footprint (95th percentile,
    here the 3 cm ring) + 1 cm, clipped to [2 cm, exempt_radius_m]; floor =
    max(top - 2 cm, table + table_clearance): the approach column's formula,
    never the 15 cm descent cylinder, which would reach the neighbours."""
    calls = []
    fix = cup_fix(top=top)
    _pregrasp_min_z(GCFG_ON, harness_with(top + .005, calls=calls), fix)
    assert len(calls) == 1
    center, radius, floor = calls[0]
    np.testing.assert_allclose(center, fix.position[:2])
    assert radius == pytest.approx(0.04)
    assert floor == pytest.approx(z_min)


def test_the_column_radius_is_clipped():
    calls = []
    _pregrasp_min_z(GCFG_ON, harness_with(FIX_TOP, calls=calls), cup_fix(r=0.002))   # a thin stick
    _pregrasp_min_z({**GCFG_ON, "exempt_radius_m": 0.07}, harness_with(FIX_TOP, calls=calls),
                    cup_fix(r=0.30))                                                 # a wide tray
    assert [c[1] for c in calls] == pytest.approx([0.02, 0.07])


# ── OccupancyMap.column_top ──────────────────────────────────────────────


def test_column_top_reads_only_the_target_column():
    occ = fresh_map(CUP_BOX, NEIGHBOUR_BOX)
    top, voxel = occ.column_top(CUP_XY, .05, .07)
    assert voxel == .01 and .095 <= top < .095 + .01 + 1e-9
    # premise: the neighbour's voxels are in the map, just outside this column
    top_wide, _ = occ.column_top(CUP_XY, .10, .07)
    assert top_wide > .14


def test_column_top_ignores_voxels_below_the_floor():
    occ = fresh_map(CUP_BOX)
    assert occ.column_top(CUP_XY, .05, .10) is None          # every cup voxel centre is at or below 0.095
    top, _ = occ.column_top(CUP_XY, .05, .09)                # the top layer's centres (0.095) are above it
    assert .095 <= top < .105


def test_column_top_is_none_when_stale_failed_or_empty():
    bridge = GridBridge([TABLE, CUP_BOX], voxel=VOXEL)
    occ = make_map(bridge, unobserved="free")
    assert occ.column_top(CUP_XY, .05, .07) is None          # never refreshed: no map yet
    occ.refresh(depth_frame(), np.eye(4))
    assert occ.column_top(CUP_XY, .05, .07) is not None      # premise: a fresh map answers
    assert occ.column_top(CUP_XY + [0.0, 0.3], .05, .07) is None   # an empty column

    occ._last_refresh -= occ.max_age_s + 1.0                  # the grid aged out
    assert occ.is_stale() and occ.column_top(CUP_XY, .05, .07) is None

    occ.refresh(depth_frame(), np.eye(4))
    assert occ.column_top(CUP_XY, .05, .07) is not None
    bridge.fail = "bridge timed out"                          # a failed refresh keeps the old grid
    occ.refresh(depth_frame(), np.eye(4))
    assert occ.last_error and not occ.is_stale()
    assert occ.column_top(CUP_XY, .05, .07) is None

    bridge.fail = None
    occ.refresh(depth_frame(), np.eye(4))
    assert occ.column_top(CUP_XY, .05, .07) is not None
    occ.add_robot_body_mask(lambda depth, K, T: np.zeros(depth.shape, bool))   # body mask not yet validated
    assert occ.column_top(CUP_XY, .05, .07) is None

    assert fresh_map().column_top(CUP_XY, .05, -1.0) is not None   # premise: the table is there
    empty = make_map(GridBridge([], voxel=VOXEL), unobserved="free")
    empty.refresh(depth_frame(), np.eye(4))
    assert empty.last_error is None and empty.column_top(CUP_XY, .05, -1.0) is None


def test_column_top_computes_outside_the_refresh_lock():
    """column_top checks the map's health under `_refresh_lock`, then filters
    the cloud outside it (`_commit_grid` replaces the arrays, never mutates
    them): a refresh is never held up by the column scan."""
    occ = fresh_map(CUP_BOX)
    free = []

    def try_lock():
        ok = occ._refresh_lock.acquire(timeout=0.5)
        if ok:
            occ._refresh_lock.release()
        free.append(ok)

    class Cloud:
        """The occupied cloud, which asks another thread for the lock when it is read."""

        def __init__(self, points):
            self.points = points

        def __len__(self):
            return len(self.points)

        def __array__(self, dtype=None, copy=None):
            t = threading.Thread(target=try_lock)
            t.start()
            t.join(5)
            return np.asarray(self.points, dtype=dtype)

    occ._occupied = Cloud(occ._occupied)
    top, voxel = occ.column_top(CUP_XY, .05, .07)
    assert voxel == .01 and .095 <= top < .105                # the same answer as before
    assert free and all(free)


# ── end to end: the runtime's rule on a real map ─────────────────────────


def real_map_harness(*boxes):
    return SimpleNamespace(limits=SimpleNamespace(table_z=0.0, table_clearance=TABLE_FLOOR),
                           occupancy=fresh_map(*boxes))


def test_hover_on_a_real_map_ignores_a_taller_neighbour():
    """Perception says 9.3 cm, the map's cup top face is at 10 cm and a 15 cm
    neighbour stands 8 cm away: the hover rises by the 0.7 cm gap only."""
    expected = FIX_TOP + (0.10 - FIX_TOP) + CLEAR + VOXEL / 2
    assert _pregrasp_min_z(GCFG_ON, real_map_harness(CUP_BOX, NEIGHBOUR_BOX), cup_fix()) == \
        pytest.approx((expected, TODAY), abs=1e-6)


def test_a_tall_neighbour_inside_the_column_raises_the_hover_to_the_cap_only():
    """Per spec: the column is the target's footprint + 1 cm, so a 15 cm stick
    touching the cup is in it. Its top raises the hover by the 2 cm cap and
    no further."""
    harness = real_map_harness(CUP_BOX, STICK_BOX)
    assert harness.occupancy.column_top(CUP_XY, .04, FIX_TOP - .02)[0] > .14   # premise: the stick is in the column
    assert _pregrasp_min_z(GCFG_ON, harness, cup_fix()) == \
        pytest.approx((FIX_TOP + .02 + CLEAR + VOXEL / 2, TODAY), abs=1e-6)


@pytest.mark.parametrize("map_top_on, expected", [
    (True, (0.10 + CLEAR + VOXEL / 2, FIX_TOP + CLEAR)),
    (False, FIX_TOP + CLEAR),
])
def test_grasp_object_selects_with_the_map_top_hover(monkeypatch, map_top_on, expected):
    """skill_grasp_object hands select_grasp the hover _pregrasp_min_z
    computes from its own harness's map (stripped runtime, as in
    tests/test_grasp_home_failure.py); selection stops the skill before any
    motion."""
    from unittest.mock import Mock

    from cascade.config import load_demo_config
    from cascade.safety.harness import SafetyHarness, SafetyLimits
    from cascade.skills.runtime import SkillRuntime
    from cascade.types import SkillError

    runtime = SkillRuntime.__new__(SkillRuntime)
    runtime.cfg = load_demo_config(camera="mock", arm="fr3_mock", llm="mock")
    runtime.cfg._data["grasp"].update(pregrasp_clear_top_m=CLEAR, pregrasp_map_top=map_top_on)
    harness = SafetyHarness(SafetyLimits(workspace_min=np.array([.2, -.6, -.01]), workspace_max=np.ones(3),
                                         table_z=0.0, table_clearance=TABLE_FLOOR),
                            occupancy=fresh_map(CUP_BOX, NEIGHBOUR_BOX))
    moves = []
    runtime.arm = SimpleNamespace(harness=harness, n_joints=7, stop=lambda: None,
                                  get_state=lambda: SimpleNamespace(q=np.zeros(7), physics_clock=None),
                                  set_gripper=lambda pos, effort: None,
                                  move_joints=lambda q, **kw: moves.append(q))
    runtime.arm.raw = runtime.arm
    runtime.kin = object()
    runtime.memory = Mock()
    runtime.grasp_memory = Mock()
    runtime._call_measurements = None
    runtime.held_object = None
    runtime._grip_open = .04
    runtime._max_width = .08
    runtime._reconcile_held = lambda: None
    runtime._plan_grasps = lambda *a, **kw: [SimpleNamespace(position=np.array([*CUP_XY, .06]))]
    seen = []

    def select(grasps, kin, seed, **kwargs):
        seen.append(kwargs["pregrasp_min_z"])
        raise SkillError("stop after selection")

    monkeypatch.setattr("cascade.skills.runtime.select_grasp", select)
    fix = cup_fix()
    fix.detection = SimpleNamespace(label="paper cup")
    with pytest.raises(SkillError, match="stop after selection"):
        runtime.skill_grasp_object("paper cup", _fix=fix, _frame=object())
    assert seen == [pytest.approx(expected, abs=1e-6)]
    assert moves == []


# ── the EXECUTED hover: select_grasp's descending ladder ─────────────────

GRASP_Z, OFFSET = 0.06, 0.04                   # the fixed offset alone: pregrasp at z = 0.10
MAP_Z = FIX_TOP + .012 + CLEAR + VOXEL / 2     # the map's hover, 0.140


class CeilingKin:
    """Top-down IK that solves up to TCP z = `ceiling` (`twin_ceiling` for the
    jaws-swapped twin, whose TCP x axis is flipped). q = (TCP z, wrist):
    `wrist(z, twin)` sets the joint travel the twin ranking compares."""

    def __init__(self, ceiling=1.0, twin_ceiling=None, wrist=lambda z, twin: 0.0):
        self.ceiling, self.twin_ceiling, self.wrist = ceiling, twin_ceiling, wrist

    def ik(self, T, seed):
        z, twin = float(T[2, 3]), bool(T[0, 0] < 0)
        ceiling = self.twin_ceiling if twin and self.twin_ceiling is not None else self.ceiling
        ok = z <= ceiling + 1e-9
        return SimpleNamespace(success=ok, q=np.array([z, self.wrist(z, twin)]), error=0.0 if ok else 1.0)


def down_grasp():
    return Grasp(position=np.array([*CUP_XY, GRASP_Z]), rotation=np.diag([1.0, -1.0, -1.0]),
                 width_m=0.05, approach=np.array([0.0, 0.0, -1.0]), quality=1.0, label="paper cup")


def select(kin, min_z, **kw):
    return select_grasp([down_grasp()], kin, np.zeros(2), pregrasp_offset_m=OFFSET, pregrasp_min_z=min_z, **kw)


def executed_z(kin, min_z, **kw):
    return float(select(kin, min_z, **kw)[1][0])


@pytest.mark.parametrize("ceiling, expected", [
    (1.0, MAP_Z),             # the map's hover is reachable: it executes
    (0.13, TODAY),            # an IK ceiling between the two hovers: today's hover, never lower
    (0.11, GRASP_Z + OFFSET),  # today's hover is out of reach too: the fixed offset
])
def test_the_executed_hover_never_drops_below_todays(ceiling, expected):
    kin = CeilingKin(ceiling)
    assert executed_z(kin, (MAP_Z, TODAY)) == pytest.approx(expected)
    assert executed_z(kin, (MAP_Z, TODAY)) >= executed_z(kin, TODAY)   # today = the single hover


def test_a_ladder_is_tried_highest_first_whatever_its_order():
    """select_grasp sorts the rungs itself and drops a NaN rung (it could
    never be solved, and it would break the sort): the highest reachable
    rung executes."""
    assert executed_z(CeilingKin(), (TODAY, float("nan"), MAP_Z)) == pytest.approx(MAP_Z)
    assert executed_z(CeilingKin(0.13), (TODAY, float("nan"), MAP_Z)) == pytest.approx(TODAY)


def test_a_refused_map_hover_falls_to_todays_hover():
    refuse_high = lambda g, q_pre, q_grasp: "blocked above" if q_pre[0] > 0.13 else None
    assert executed_z(CeilingKin(), (MAP_Z, TODAY), validate=refuse_high) == pytest.approx(TODAY)


def test_when_todays_hover_is_out_of_reach_too_the_offset_is_exactly_base():
    kin = CeilingKin(0.11)
    ladder, base = select(kin, (MAP_Z, TODAY)), select(kin, TODAY)
    np.testing.assert_array_equal(ladder[0].rotation, base[0].rotation)
    for got, want in zip(ladder[1:], base[1:]):
        np.testing.assert_array_equal(got, want)


def test_a_twin_at_the_fixed_offset_never_beats_a_hover():
    """The jaws-swapped twin reaches only the fixed offset; its wrist travel
    (1.3) is below the original's at the map hover (1.40) but above it at
    today's (1.23). Today executes the original at today's hover, so the
    ladder must not trade the hover for travel and drop to the offset."""
    kin = CeilingKin(twin_ceiling=0.11, wrist=lambda z, twin: 1.3 if twin else 10 * z)
    assert executed_z(kin, TODAY) == pytest.approx(TODAY)          # premise: today's choice
    assert executed_z(kin, (MAP_Z, TODAY)) == pytest.approx(MAP_Z)


def test_between_hovers_the_twin_with_less_travel_wins():
    """Travel still picks the orientation among the hovers: the twin at
    today's hover beats a half-turn of the wrist at the map hover."""
    kin = CeilingKin(twin_ceiling=0.13, wrist=lambda z, twin: 0.2 if twin else 3.0)
    grasp, q_pre, _ = select(kin, (MAP_Z, TODAY))
    assert grasp.rotation[0, 0] < 0 and q_pre[0] == pytest.approx(TODAY)


def test_a_single_float_keeps_todays_twin_choice():
    """Every caller that passes one float (no map raise) keeps today's
    selection exactly: there travel alone picks the orientation, even a twin
    at the fixed offset over the original at the hover. Only a ladder adds
    the hover-beats-offset rule."""
    kin = CeilingKin(twin_ceiling=0.11, wrist=lambda z, twin: 1.3 if twin else 2.0)
    grasp, q_pre, _ = select(kin, TODAY)
    assert grasp.rotation[0, 0] < 0 and q_pre[0] == pytest.approx(GRASP_Z + OFFSET)
    grasp, q_pre, _ = select(kin, (MAP_Z, TODAY))
    assert grasp.rotation[0, 0] > 0 and q_pre[0] == pytest.approx(MAP_Z)
