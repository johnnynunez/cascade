"""The pregrasp hover counts from the target's top in the MAP (spec §3.2 rules 3-4).

On the physical reBot the occupancy map put a target's top 0.2-0.9 cm above
the top perception measured (1 cm voxels + depth noise), so a hover computed
from the perception top alone left the TCP 2.1-2.8 cm from the map's top,
inside the 3 cm margin. With `grasp.pregrasp_map_top` the hover is

    fix_top + min(max(map_top - fix_top, 0), 2 cm) + pregrasp_clear_top_m + voxel/2

where `map_top` is the top face of the highest occupied voxel in the target's
own column (the approach column's footprint around the fix XY), read under the
map's refresh lock. It only ever rises compared with the perception-only hover,
the raise is capped so that one residue voxel cannot push it out of reach, and a
stale map or an empty column keeps today's hover.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from cascade.skills.runtime import MAP_TOP_RAISE_CAP_M, _pregrasp_min_z
from occupancy_grid_bridge import TABLE, GridBridge, centred_box, depth_frame, make_map

FIX_TOP, CLEAR, VOXEL = .093, .03, .01
CUP_XY = np.array([0.305, 0.005])       # on a voxel centre of make_map's 1 cm grid
GCFG_ON = {"pregrasp_clear_top_m": CLEAR, "pregrasp_map_top": True, "exempt_radius_m": 0.15}
TABLE_FLOOR = 0.01                      # table_z + table_clearance of the stand-in harness

# the map's cup: a box whose top voxel centres sit at 0.095 (top face 0.100)
CUP_BOX = centred_box([*CUP_XY, 0.095 / 2], [0.06, 0.06, 0.095])
# a taller neighbour 8 cm away (centre to centre), its near face 6 cm from the cup's axis
NEIGHBOUR_BOX = centred_box([CUP_XY[0] + 0.08, CUP_XY[1], 0.15 / 2], [0.04, 0.04, 0.15])


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
    (None,           FIX_TOP + CLEAR),                     # stale map / empty column: today's hover
    (FIX_TOP + .009, FIX_TOP + .009 + CLEAR + VOXEL / 2),  # the recorded 0.2-0.9 cm gaps
    (FIX_TOP + .05,  FIX_TOP + .02 + CLEAR + VOXEL / 2),   # one residue voxel: raise capped at 2 cm
    (FIX_TOP - .01,  FIX_TOP + CLEAR + VOXEL / 2),         # never below the fix top
])
def test_hover_from_the_map_top(map_top, expected):
    assert _pregrasp_min_z(GCFG_ON, harness_with(column_top=map_top), cup_fix()) == pytest.approx(expected)


def test_the_raise_cap_is_two_centimetres():
    assert MAP_TOP_RAISE_CAP_M == 0.02


def test_flag_off_or_clear_top_null_keeps_todays_hover():
    high = harness_with(column_top=FIX_TOP + .05)
    assert _pregrasp_min_z({**GCFG_ON, "pregrasp_map_top": False}, high, cup_fix()) == pytest.approx(FIX_TOP + CLEAR)
    off_by_default = {k: v for k, v in GCFG_ON.items() if k != "pregrasp_map_top"}
    assert _pregrasp_min_z(off_by_default, high, cup_fix()) == pytest.approx(FIX_TOP + CLEAR)
    assert _pregrasp_min_z({**GCFG_ON, "pregrasp_clear_top_m": None}, high, cup_fix()) is None
    no_clear_top = {k: v for k, v in GCFG_ON.items() if k != "pregrasp_clear_top_m"}
    assert _pregrasp_min_z(no_clear_top, high, cup_fix()) is None


def test_no_map_keeps_todays_hover():
    limits = SimpleNamespace(table_z=0.0, table_clearance=TABLE_FLOOR)
    for occupancy in (None, SimpleNamespace(clearance=lambda points: None)):   # no map; a map without column_top
        harness = SimpleNamespace(limits=limits, occupancy=occupancy)
        assert _pregrasp_min_z(GCFG_ON, harness, cup_fix()) == pytest.approx(FIX_TOP + CLEAR)


def test_unusable_target_points_keep_todays_answer():
    calls = []
    empty = SimpleNamespace(position=np.array([*CUP_XY, 0.05]), points=np.zeros((0, 3)))
    assert _pregrasp_min_z(GCFG_ON, harness_with(FIX_TOP + .05, calls=calls), empty) is None
    # a footprint that cannot be measured (non-finite XY) gives no column: the
    # observed top still sets today's hover, and the map is never asked
    fix = cup_fix()
    fix.points[0, 0] = np.nan
    assert _pregrasp_min_z(GCFG_ON, harness_with(FIX_TOP + .05, calls=calls), fix) == pytest.approx(FIX_TOP + CLEAR)
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


# ── end to end: the runtime's rule on a real map ─────────────────────────


def test_hover_on_a_real_map_ignores_a_taller_neighbour():
    """Perception says 9.3 cm, the map's cup top face is at 10 cm and a 15 cm
    neighbour stands 8 cm away: the hover rises by the 0.7 cm gap only."""
    harness = SimpleNamespace(limits=SimpleNamespace(table_z=0.0, table_clearance=TABLE_FLOOR),
                              occupancy=fresh_map(CUP_BOX, NEIGHBOUR_BOX))
    expected = FIX_TOP + (0.10 - FIX_TOP) + CLEAR + VOXEL / 2
    assert _pregrasp_min_z(GCFG_ON, harness, cup_fix()) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("map_top_on, expected", [
    (True, 0.10 + CLEAR + VOXEL / 2),
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
