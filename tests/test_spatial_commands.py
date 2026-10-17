"""Natural-language spatial motion through the real runtime on the mock stack.

`move_relative` (frames, vague sizes, free vectors, "again"), `undo_move`,
`rotate_gripper` and `move_near_object` run end to end here: the real
`build_runtime` wiring, the real SafeArm/SafetyHarness, the MockArm
recording every command it receives. The assertions that matter most are
the negative ones -- a refused or invalid request sends ZERO commands, and
the offered smaller move was found by vetting, not by moving.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_pin


@pytest.fixture
def rig(demo_cfg, tmp_path):
    from cascade.apps.demo import build_runtime, shutdown_runtime

    runtime, arm = build_runtime(demo_cfg, tmp_path / "run")
    try:
        yield runtime, arm
    finally:
        shutdown_runtime(runtime, arm)


def _tcp(runtime):
    return runtime.arm.harness.kin.fk(runtime.arm.get_state().q)


def _hold(runtime, arm, label="red cube", drop_m=0.03):
    """Put the rig in a holding state the way a finished grasp leaves it:
    jaws stalled on something (MockArm object_stop_frac) and the runtime's
    held markers set. Done on the RAW arm: this is test setup, not a skill."""
    arm.object_stop_frac = 0.5
    arm.set_gripper(1.0)
    runtime.held_object = label
    runtime._held_det_label = label
    runtime._held_width_m = 0.045
    runtime._held_support_offset_m = drop_m


def _park_low(runtime, z=0.10, xy=(0.30, 0.0)):
    """Test setup: a top-down pose `z` above the real table (through SafeArm,
    so the harness vets it like any motion) -- the pose a grasp leaves the
    gripper in. Raising `table_z` instead would put the elbow 'under the
    table' and refuse everything for the wrong reason."""
    from cascade.grasping.obb_grasp import _yaw_rotation
    from cascade.types import make_transform

    kin = runtime.arm.harness.kin
    yaw = float(np.arctan2(xy[1], xy[0]))          # radial jaws, like point_at
    T = make_transform(_yaw_rotation(yaw, axis_order=runtime._tool_axis_order), [xy[0], xy[1], z])
    ik = kin.ik(T, runtime.arm.get_state().q)
    assert ik.success
    assert runtime.arm.move_joints(ik.q, duration_s=2.0)
    np.testing.assert_allclose(_tcp(runtime)[:3, 3], [xy[0], xy[1], z], atol=2e-3)


# ── frames and the reported interpretation ──────────────────────────────────


@needs_pin
def test_bare_left_is_the_robots_left_and_the_result_says_so(rig):
    runtime, _ = rig
    start = _tcp(runtime)[:3, 3]
    r = runtime.execute("move_relative", {"direction": "left", "amount": "small"})
    assert r["ok"], r
    assert r["frame"] == "base"
    np.testing.assert_allclose(r["displacement_base_m"], [0.0, 0.02, 0.0], atol=1e-9)
    np.testing.assert_allclose(_tcp(runtime)[:3, 3] - start, [0.0, 0.02, 0.0], atol=2e-3)
    assert "robot's left" in r["interpretation"]
    # the same move as the person at the screen sees it (mock camera is
    # top-down: robot +y is image left)
    assert "left on the screen" in r["as_seen_on_screen"]


@needs_pin
def test_toward_the_viewer_uses_the_scene_camera(rig):
    """Mock camera: image up = base +x, so 'toward the viewer' (down on the
    screen) is base -x."""
    runtime, _ = rig
    start = _tcp(runtime)[:3, 3]
    r = runtime.execute("move_relative", {"direction": "toward_viewer", "distance_m": 0.03})
    assert r["ok"], r
    assert r["frame"] == "viewer" and r["camera"] == "mock"
    np.testing.assert_allclose(_tcp(runtime)[:3, 3] - start, [-0.03, 0.0, 0.0], atol=2e-3)


@needs_pin
def test_a_free_vector_moves_by_exactly_that_vector(rig):
    runtime, _ = rig
    start = _tcp(runtime)[:3, 3]
    r = runtime.execute("move_relative", {"vector": [0.0, -0.02, -0.03]})
    assert r["ok"], r
    np.testing.assert_allclose(_tcp(runtime)[:3, 3] - start, [0.0, -0.02, -0.03], atol=2e-3)


@needs_pin
@pytest.mark.parametrize("args", [
    {"direction": "left", "distance_m": float("nan")},
    {"direction": "left", "distance_m": 0.5},
    {"direction": "left", "distance_m": -0.05},
    {"direction": "left", "distance_m": True},
    {"vector": [0.0, float("inf"), 0.0]},
    {"vector": [0.2, 0.0, 0.0]},
    {"direction": "sideways"},
    {"direction": "go back"},
    {"direction": "in", "frame": "viewer"},
    {"direction": "left", "frame": "world"},
    {},
])
def test_absurd_or_ambiguous_requests_never_move_the_arm(rig, args):
    runtime, arm = rig
    n0 = len(arm.commands)
    r = runtime.execute("move_relative", args)
    assert r["ok"] is False, r
    assert len(arm.commands) == n0


# ── refusals: vetted, not probed ────────────────────────────────────────────


@needs_pin
def test_a_harness_refusal_sends_nothing_and_offers_the_vetted_fraction(rig):
    runtime, arm = rig
    h = runtime.arm.harness
    tcp = _tcp(runtime)[:3, 3]
    # a keep-out slab 6..12 cm below the gripper (a tripod arm, a fixture)
    h.limits.keep_out.append((tcp + [-0.2, -0.2, -0.12], tcp + [0.2, 0.2, -0.06]))
    n0, v0 = len(arm.commands), list(h.violations)
    r = runtime.execute("move_relative", {"direction": "down", "distance_m": 0.10})
    assert r["ok"] is False
    assert r["refused_by"] == "safety_harness"
    assert "keep-out" in r["error"]
    assert r["arm_moved"] is False
    assert len(arm.commands) == n0          # the arm stayed put
    assert h.violations == v0               # vetting records no violation
    feasible = r["feasible_distance_m"]
    assert 0.03 <= feasible < 0.06
    assert "distance_m" in r["next_action"]
    # the offer is real: taking it passes every gate
    ok = runtime.execute("move_relative", {"direction": "down", "distance_m": feasible})
    assert ok["ok"], ok


@needs_pin
def test_table_clearance_is_enforced_before_motion(rig):
    runtime, arm = rig
    _park_low(runtime, z=0.10)
    n0 = len(arm.commands)
    r = runtime.execute("move_relative", {"direction": "down", "distance_m": 0.10})
    assert r["ok"] is False and r["refused_by"] == "safety_harness"
    assert "table" in r["error"]
    assert len(arm.commands) == n0
    # the harness keeps the TCP 2 cm above the table: 8 of the 10 cm vet
    assert r["feasible_distance_m"] == pytest.approx(0.08, abs=0.004)


class _PointObstacle:
    """Occupancy-map double: one obstacle point, distances to it."""

    required = False

    def __init__(self, point):
        self.point = np.asarray(point, dtype=float)

    def clearance(self, points):
        return np.linalg.norm(np.asarray(points, dtype=float) - self.point, axis=1)


@needs_pin
def test_occupancy_clearance_is_enforced_before_motion(rig):
    runtime, arm = rig
    h = runtime.arm.harness
    tcp = _tcp(runtime)[:3, 3]
    h.occupancy = _PointObstacle(tcp + [0.0, 0.0, -0.07])
    n0 = len(arm.commands)
    r = runtime.execute("move_relative", {"direction": "down", "distance_m": 0.06})
    assert r["ok"] is False and r["refused_by"] == "safety_harness"
    assert "occupancy" in r["error"]
    assert len(arm.commands) == n0


@needs_pin
def test_a_nudge_that_ends_inside_the_map_margin_is_refused(rig):
    runtime, arm = rig
    runtime.arm.harness.occupancy = _PointObstacle(_tcp(runtime)[:3, 3] + [0., 0., -.020])
    n0 = len(arm.commands)
    r = runtime.execute("move_relative", {"direction": "up", "distance_m": .005})   # ends at 2.5 cm
    assert r["ok"] is False and r["refused_by"] == "safety_harness"
    assert "route ends inside the margin" in r["error"] and len(arm.commands) == n0
    assert runtime.execute("move_relative", {"direction": "up", "distance_m": .02})["ok"] is True


@needs_pin
def test_every_executed_step_passes_through_the_harness(rig):
    runtime, arm = rig
    h = runtime.arm.harness
    calls = []
    original = h.approve
    h.approve = lambda *a, **k: calls.append(a) or original(*a, **k)
    n0 = len(arm.commands)
    r = runtime.execute("move_relative", {"direction": "up", "amount": "small"})
    assert r["ok"], r
    assert len(arm.commands) > n0
    assert len(calls) >= len(arm.commands) - n0


# ── holding something ───────────────────────────────────────────────────────


@needs_pin
def test_while_holding_it_never_touches_the_jaws_and_moves_slower(rig):
    runtime, arm = rig
    _hold(runtime, arm)
    durations = []
    move = runtime.arm.move_joints
    runtime.arm.move_joints = lambda q, duration_s=2.0, **k: durations.append(duration_s) or move(
        q, duration_s=duration_s, **k)

    def no_gripper(*a, **k):
        raise AssertionError("a spatial move commanded the jaws while holding")

    runtime.arm.set_gripper = no_gripper
    r = runtime.execute("move_relative", {"direction": "up", "amount": "small"})
    assert r["ok"], r
    assert r["holding"] == "red cube"
    assert "slow" in r["speed"]
    assert durations and durations[0] == pytest.approx(2.0 * runtime.cfg.get("relative_motion", {}).get(
        "duration_s", 1.0))
    assert runtime.held_object == "red cube"


@needs_pin
def test_a_held_object_keeps_its_bottom_off_the_table(rig):
    """The harness keeps the TCP 2 cm above the table; a held object hangs
    BELOW the TCP, so the move is checked for the object's bottom too."""
    runtime, arm = rig
    _park_low(runtime, z=0.10)
    empty = runtime.execute("move_relative", {"direction": "down", "distance_m": 0.05})
    assert empty["ok"], empty                   # TCP 5 cm above the table: fine
    runtime.execute("move_relative", {"direction": "up", "distance_m": 0.05})
    _hold(runtime, arm, drop_m=0.03)
    n0 = len(arm.commands)
    r = runtime.execute("move_relative", {"direction": "down", "distance_m": 0.07})
    assert r["ok"] is False and r["refused_by"] == "held_object_clearance"
    assert "red cube" in r["error"]
    assert len(arm.commands) == n0
    # bottom 3 cm under the TCP must stay 2 cm over the table: TCP >= 5 cm
    assert r["feasible_distance_m"] == pytest.approx(0.05, abs=0.004)


# ── "again" and "go back" ───────────────────────────────────────────────────


@needs_pin
def test_again_repeats_the_last_direction_and_size(rig):
    runtime, _ = rig
    start = _tcp(runtime)[:3, 3]
    assert runtime.execute("move_relative", {"direction": "left", "amount": "small"})["ok"]
    r = runtime.execute("move_relative", {"direction": "again"})
    assert r["ok"], r
    assert r["interpretation"].startswith("again:")
    np.testing.assert_allclose(_tcp(runtime)[:3, 3] - start, [0.0, 0.04, 0.0], atol=3e-3)
    # "a tiny bit more": same direction, new size
    r = runtime.execute("move_relative", {"direction": "more", "amount": "tiny"})
    assert r["ok"], r
    np.testing.assert_allclose(_tcp(runtime)[:3, 3] - start, [0.0, 0.05, 0.0], atol=3e-3)


@needs_pin
def test_again_with_nothing_to_repeat_is_refused_without_motion(rig):
    runtime, arm = rig
    n0 = len(arm.commands)
    r = runtime.execute("move_relative", {"direction": "again"})
    assert r["ok"] is False and "nothing to repeat" in r["error"]
    assert len(arm.commands) == n0


@needs_pin
def test_undo_retraces_moves_one_at_a_time_back_to_the_start(rig):
    runtime, arm = rig
    start = _tcp(runtime)
    assert runtime.execute("move_relative", {"direction": "up", "amount": "small"})["ok"]
    assert runtime.execute("move_relative", {"direction": "left", "amount": "small"})["ok"]
    r = runtime.execute("undo_move", {})
    assert r["ok"], r
    assert len(r["undone"]) == 1 and "left" in r["undone"][0]
    np.testing.assert_allclose(_tcp(runtime)[:3, 3] - start[:3, 3], [0.0, 0.0, 0.02], atol=2e-3)
    assert r["undo_available"] == 1
    assert runtime.execute("undo_move", {})["ok"]
    np.testing.assert_allclose(_tcp(runtime), start, atol=2e-3)
    n0 = len(arm.commands)
    r = runtime.execute("undo_move", {})
    assert r["ok"] is False and "nothing to undo" in r["error"]
    assert len(arm.commands) == n0


@needs_pin
def test_undo_several_steps_in_one_call(rig):
    runtime, _ = rig
    start = _tcp(runtime)
    for direction in ("up", "left", "forward"):
        assert runtime.execute("move_relative", {"direction": direction, "amount": "small"})["ok"]
    r = runtime.execute("undo_move", {"steps": 3})
    assert r["ok"], r
    assert len(r["undone"]) == 3
    np.testing.assert_allclose(_tcp(runtime), start, atol=2e-3)


@needs_pin
def test_a_refused_move_does_not_change_the_history(rig):
    runtime, _ = rig
    start = _tcp(runtime)
    assert runtime.execute("move_relative", {"direction": "up", "amount": "small"})["ok"]
    h = runtime.arm.harness
    tcp = _tcp(runtime)[:3, 3]
    h.limits.keep_out.append((tcp + [-0.2, -0.2, -0.12], tcp + [0.2, 0.2, -0.08]))
    refused = runtime.execute("move_relative", {"direction": "down", "distance_m": 0.12})
    assert refused["ok"] is False and refused["undo_available"] == 1
    h.limits.keep_out.clear()
    r = runtime.execute("undo_move", {})
    assert r["ok"], r
    np.testing.assert_allclose(_tcp(runtime), start, atol=2e-3)


@needs_pin
def test_undo_refuses_and_forgets_after_another_command_moved_the_arm(rig):
    runtime, arm = rig
    assert runtime.execute("move_relative", {"direction": "up", "amount": "medium"})["ok"]
    assert runtime.execute("move_home", {})["ok"]
    n0 = len(arm.commands)
    r = runtime.execute("undo_move", {})
    assert r["ok"] is False and "moved since" in r["error"]
    assert len(arm.commands) == n0
    assert "nothing to undo" in runtime.execute("undo_move", {})["error"]


@needs_pin
def test_an_undo_is_vetted_like_any_move(rig):
    runtime, arm = rig
    start = _tcp(runtime)[:3, 3]
    assert runtime.execute("move_relative", {"direction": "up", "amount": "medium"})["ok"]
    # something now occupies the spot the gripper came from
    runtime.arm.harness.limits.keep_out.append((start - 0.01, start + 0.01))
    n0 = len(arm.commands)
    r = runtime.execute("undo_move", {})
    assert r["ok"] is False and r["refused_by"] == "safety_harness"
    assert len(arm.commands) == n0
    assert r["undo_available"] == 1          # still there for when it is clear


@needs_pin
@pytest.mark.parametrize("steps", [0, 6, 1.5, True, "two", float("nan")])
def test_undo_rejects_bad_step_counts(rig, steps):
    runtime, arm = rig
    assert runtime.execute("move_relative", {"direction": "up", "amount": "small"})["ok"]
    n0 = len(arm.commands)
    r = runtime.execute("undo_move", {"steps": steps})
    assert r["ok"] is False
    assert len(arm.commands) == n0


# ── orientation nudges, wrist camera on top ─────────────────────────────────


def _angle_deg(R0, R1):
    cos = (np.trace(R0.T @ R1) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


@needs_pin
def test_roll_turns_about_the_gripper_axis_in_place(rig):
    runtime, _ = rig
    T0 = _tcp(runtime)
    r = runtime.execute("rotate_gripper", {"direction": "clockwise", "amount": "small"})
    assert r["ok"], r
    assert r["kind"] == "roll" and r["angle_deg"] == pytest.approx(10.0)
    T1 = _tcp(runtime)
    np.testing.assert_allclose(T1[:3, 3], T0[:3, 3], atol=2e-3)       # in place
    np.testing.assert_allclose(T1[:3, 0], T0[:3, 0], atol=2e-3)       # approach kept
    assert _angle_deg(T0[:3, :3], T1[:3, :3]) == pytest.approx(10.0, abs=0.5)
    assert T1[2, 2] > 0.95                                            # camera still on top


@needs_pin
def test_tilt_down_points_the_jaws_toward_the_table(rig):
    runtime, _ = rig
    T0 = _tcp(runtime)                     # home: jaws point forward (+x)
    r = runtime.execute("rotate_gripper", {"direction": "down", "angle_deg": 15})
    assert r["ok"], r
    assert r["kind"] == "tilt"
    T1 = _tcp(runtime)
    np.testing.assert_allclose(T1[:3, 3], T0[:3, 3], atol=2e-3)
    assert T1[2, 0] == pytest.approx(-np.sin(np.radians(15.0)), abs=5e-3)


@needs_pin
@pytest.mark.parametrize("args", [{"direction": "clockwise", "angle_deg": 180},
                                  {"direction": "counterclockwise", "angle_deg": 120},
                                  {"direction": "upside down"},
                                  {"direction": "flip"}])
def test_rotate_it_upside_down_is_refused_before_any_motion(rig, args):
    """User safety requirement: the wrist camera sits on TOP of the gripper
    and must never be rolled under (it hits the table)."""
    runtime, arm = rig
    n0 = len(arm.commands)
    r = runtime.execute("rotate_gripper", args)
    assert r["ok"] is False
    assert "camera must stay on top" in r["error"]
    assert len(arm.commands) == n0


@needs_pin
def test_rolls_stop_where_the_wrist_camera_would_go_below_horizontal(rig):
    runtime, arm = rig
    for angle in (30, 30, 25):                                         # 85 deg, camera still up
        r = runtime.execute("rotate_gripper", {"direction": "clockwise", "angle_deg": angle})
        assert r["ok"], r
    n0 = len(arm.commands)
    r = runtime.execute("rotate_gripper", {"direction": "clockwise", "angle_deg": 10})
    assert r["ok"] is False and r["refused_by"] == "wrist_camera"
    assert "camera must stay on top" in r["error"]
    assert len(arm.commands) == n0
    assert r["feasible_angle_deg"] == pytest.approx(5.0, abs=1.0)
    back = runtime.execute("rotate_gripper", {"direction": "counterclockwise", "angle_deg": 30})
    assert back["ok"], back


@needs_pin
@pytest.mark.parametrize("args", [{"direction": "clockwise", "angle_deg": 31},
                                  {"direction": "clockwise", "angle_deg": float("nan")},
                                  {"direction": "clockwise", "angle_deg": -5},
                                  {"direction": "clockwise", "frame": "viewer"},
                                  {"direction": "sideways"},
                                  {}])
def test_rotation_caps_and_validation_never_move(rig, args):
    runtime, arm = rig
    n0 = len(arm.commands)
    r = runtime.execute("rotate_gripper", args)
    assert r["ok"] is False, r
    assert len(arm.commands) == n0


@needs_pin
def test_rotate_again_and_undo_back_to_the_start(rig):
    runtime, _ = rig
    T0 = _tcp(runtime)
    assert runtime.execute("rotate_gripper", {"direction": "counterclockwise", "amount": "small"})["ok"]
    r = runtime.execute("rotate_gripper", {"direction": "again"})
    assert r["ok"] and r["interpretation"].startswith("again:")
    assert _angle_deg(T0[:3, :3], _tcp(runtime)[:3, :3]) == pytest.approx(20.0, abs=0.7)
    assert runtime.execute("undo_move", {"steps": 2})["ok"]
    np.testing.assert_allclose(_tcp(runtime), T0, atol=2e-3)


@needs_pin
def test_rotating_while_holding_is_slow_and_never_touches_the_jaws(rig):
    runtime, arm = rig
    _hold(runtime, arm)

    def no_gripper(*a, **k):
        raise AssertionError("a rotation commanded the jaws while holding")

    runtime.arm.set_gripper = no_gripper
    r = runtime.execute("rotate_gripper", {"direction": "clockwise", "amount": "tiny"})
    assert r["ok"], r
    assert r["holding"] == "red cube" and "slow" in r["speed"]


# ── positions relative to objects ───────────────────────────────────────────


def _seen(runtime, label="red cube"):
    import time

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if runtime.beliefs.find(label) is not None:
            return runtime.beliefs.find(label)
        time.sleep(0.05)
    pytest.fail(f"the watcher never saw {label!r}")


@needs_pin
def test_hold_above_an_object_hovers_over_its_top(rig):
    runtime, _ = rig
    _seen(runtime)
    _park_low(runtime, z=0.14, xy=(0.32, -0.10))
    r = runtime.execute("move_near_object", {"label": "red cube", "relation": "above"})
    assert r["ok"], r
    top = r["target"]["top_z_m"]
    tcp = _tcp(runtime)[:3, 3]
    np.testing.assert_allclose(tcp[:2], r["target"]["center_xy_m"], atol=3e-3)
    assert tcp[2] == pytest.approx(top + 0.05, abs=3e-3)       # default 5 cm over its top
    assert "red cube" in r["interpretation"]


@needs_pin
def test_holding_keeps_the_carried_bottom_above_the_targets_top(rig):
    runtime, arm = rig
    _seen(runtime)
    _park_low(runtime, z=0.14, xy=(0.32, -0.10))
    _hold(runtime, arm, label="blue block", drop_m=0.03)
    r = runtime.execute("move_near_object", {"label": "red cube", "relation": "above", "clearance_m": 0.04})
    assert r["ok"], r
    assert _tcp(runtime)[2, 3] == pytest.approx(r["target"]["top_z_m"] + 0.04 + 0.03, abs=3e-3)
    assert r["holding"] == "blue block"


@needs_pin
def test_going_back_after_a_hover_retraces_the_carry_route(rig):
    """The hover went over at the start height and then straight down; the
    undo must climb straight up first, never cut diagonally across the
    target it is hovering over."""
    runtime, arm = rig
    _seen(runtime)
    _park_low(runtime, z=0.14, xy=(0.32, -0.10))
    start = _tcp(runtime)
    r = runtime.execute("move_near_object", {"label": "red cube", "relation": "above"})
    assert r["ok"], r
    centre = np.asarray(r["target"]["center_xy_m"])
    n0 = len(arm.commands)
    assert runtime.execute("undo_move", {})["ok"]
    np.testing.assert_allclose(_tcp(runtime), start, atol=2e-3)
    kin = runtime.arm.harness.kin
    path = np.array([kin.fk(q)[:3, 3] for q in arm.commands[n0:]])
    away = np.linalg.norm(path[:, :2] - centre, axis=1) > 0.02
    assert away.any() and path[away, 2].min() > start[2, 3] - 0.01


@needs_pin
def test_just_above_the_table_while_holding(rig):
    runtime, arm = rig
    _park_low(runtime, z=0.15)
    _hold(runtime, arm, drop_m=0.03)
    before = _tcp(runtime)[:3, 3]
    r = runtime.execute("move_near_object", {"label": "table", "relation": "just_above"})
    assert r["ok"], r
    tcp = _tcp(runtime)[:3, 3]
    np.testing.assert_allclose(tcp[:2], before[:2], atol=2e-3)
    # the held bottom (3 cm under the TCP) ends just over the harness clearance
    assert tcp[2] - 0.03 == pytest.approx(0.025, abs=3e-3)


@needs_pin
def test_next_to_an_object_stops_beside_it_on_the_named_side(rig):
    runtime, _ = rig
    _seen(runtime)
    _park_low(runtime, z=0.15)
    r = runtime.execute("move_near_object", {"label": "red cube", "relation": "next_to", "side": "left"})
    assert r["ok"], r
    tcp = _tcp(runtime)[:3, 3]
    centre = np.asarray(r["target"]["center_xy_m"])
    assert tcp[1] > centre[1]                                  # the robot's left
    assert np.linalg.norm(tcp[:2] - centre) == pytest.approx(r["target"]["standoff_m"], abs=3e-3)
    assert tcp[2] > r["target"]["top_z_m"]                      # never lower than its top


@needs_pin
def test_closer_moves_toward_it_and_stops_at_the_standoff(rig):
    runtime, arm = rig
    _seen(runtime)
    centre = None
    r = runtime.execute("move_near_object", {"label": "red cube", "relation": "closer", "amount": "large"})
    assert r["ok"], r
    centre = np.asarray(r["target"]["center_xy_m"])
    d = np.linalg.norm(_tcp(runtime)[:2, 3] - centre)
    assert d == pytest.approx(r["target"]["standoff_m"], abs=3e-3)   # stopped short of 10 cm
    n0 = len(arm.commands)
    again = runtime.execute("move_near_object", {"label": "red cube", "relation": "closer"})
    assert again["ok"] is False and "already" in again["error"]
    assert len(arm.commands) == n0


@needs_pin
@pytest.mark.parametrize("args", [
    {"label": "red cube", "relation": "inside"},
    {"label": "red cube"},
    {"relation": "above"},
    {"label": "table", "relation": "next_to"},
    {"label": "red cube", "relation": "above", "clearance_m": float("nan")},
    {"label": "red cube", "relation": "above", "clearance_m": 0.5},
    {"label": "red cube", "relation": "next_to", "side": "up"},
    {"label": "purple elephant", "relation": "above"},
])
def test_object_relative_requests_that_cannot_be_honoured_never_move(rig, args):
    runtime, arm = rig
    _seen(runtime)
    n0 = len(arm.commands)
    r = runtime.execute("move_near_object", args)
    assert r["ok"] is False, r
    assert len(arm.commands) == n0


@needs_pin
def test_the_object_in_the_gripper_is_not_a_target(rig):
    runtime, arm = rig
    _seen(runtime)
    _hold(runtime, arm, label="red cube")
    n0 = len(arm.commands)
    r = runtime.execute("move_near_object", {"label": "red cube", "relation": "above"})
    assert r["ok"] is False and "holding" in r["error"]
    assert len(arm.commands) == n0


@needs_pin
def test_a_target_too_far_for_one_call_is_refused(rig):
    runtime, arm = rig
    _seen(runtime)
    runtime.cfg._data["relative_motion"]["max_object_move_m"] = 0.10
    n0 = len(arm.commands)
    r = runtime.execute("move_near_object", {"label": "red cube", "relation": "above"})
    assert r["ok"] is False and "per call" in r["error"]
    assert len(arm.commands) == n0


# ── straight-line profiles and the viewer calibration ───────────────────────


@pytest.fixture
def cartesian_rig(demo_cfg, tmp_path):
    from cascade.apps.demo import build_runtime, shutdown_runtime

    demo_cfg._data["arm"]["cartesian_relative_moves"] = True
    runtime, arm = build_runtime(demo_cfg, tmp_path / "run")
    try:
        yield runtime, arm
    finally:
        shutdown_runtime(runtime, arm)


@needs_pin
def test_a_cartesian_profile_moves_on_a_straight_line(cartesian_rig):
    runtime, arm = cartesian_rig
    start = _tcp(runtime)[:3, 3]
    n0 = len(arm.commands)
    r = runtime.execute("move_relative", {"vector": [0.0, 0.03, -0.03]})
    assert r["ok"], r
    assert r["path"].startswith("straight line")
    kin = runtime.arm.harness.kin
    path = np.array([kin.fk(q)[:3, 3] for q in arm.commands[n0:]])
    d = np.array([0.0, 0.03, -0.03]) / np.linalg.norm([0.0, 0.03, -0.03])
    rel = path - start
    off_line = np.linalg.norm(rel - np.outer(rel @ d, d), axis=1)
    assert off_line.max() < 1e-3


@needs_pin
def test_an_unplannable_straight_line_is_refused_never_replaced_by_joint_space(cartesian_rig, monkeypatch):
    runtime, arm = cartesian_rig
    from cascade.planning import cartesian
    from cascade.types import SkillError

    def no_line(*a, **k):
        raise SkillError("cartesian path: IK failed at 3/10")

    monkeypatch.setattr(cartesian, "plan_cartesian_path", no_line)
    joint_moves = []
    runtime.arm.move_joints = lambda *a, **k: joint_moves.append(a) or True
    n0 = len(arm.commands)
    r = runtime.execute("move_relative", {"direction": "left", "amount": "small"})
    assert r["ok"] is False and r["refused_by"] == "kinematics"
    assert "straight-line" in r["error"]
    assert joint_moves == [] and len(arm.commands) == n0


@needs_pin
def test_a_straight_line_cannot_carry_a_retained_attachment(cartesian_rig):
    runtime, arm = cartesian_rig

    class _Attachment:          # an armed NV carry guard that is still valid
        def guard(self):
            pass

    runtime._carry_attachment = _Attachment()
    n0 = len(arm.commands)
    r = runtime.execute("move_relative", {"direction": "up", "amount": "small"})
    assert r["ok"] is False and r["refused_by"] == "carry_attachment"
    assert len(arm.commands) == n0


@needs_pin
def test_an_invalidated_scene_camera_disables_only_the_viewer_frame(rig):
    """The drift monitor can invalidate the scene camera mid-session: screen
    words must then refuse (no guessed transform), robot words keep working."""
    runtime, arm = rig
    runtime.extrinsics.invalidate("extrinsic drift: camera moved 3 cm")
    n0 = len(arm.commands)
    r = runtime.execute("move_relative", {"direction": "toward_viewer", "amount": "small"})
    assert r["ok"] is False and "viewer frame unavailable" in r["error"] and "frame='base'" in r["error"]
    assert len(arm.commands) == n0
    ok = runtime.execute("move_relative", {"direction": "left", "amount": "small"})
    assert ok["ok"], ok
    assert "unavailable" in ok["as_seen_on_screen"]


@needs_pin
def test_an_interrupted_route_can_be_undone_from_where_it_stopped(rig):
    """A halt mid-route (halt_motion, a superseded call) leaves the arm
    part-way along a vetted segment. That partial path is recorded, so 'go
    back' retraces it instead of refusing as stale."""
    from cascade.types import SafetyViolation

    runtime, _ = rig
    _seen(runtime)
    _park_low(runtime, z=0.14, xy=(0.32, -0.10))
    start = _tcp(runtime)
    real = runtime.arm.move_joints
    calls = []

    def halted_on_the_descent(q, duration_s=2.0, **k):
        calls.append(q)
        if len(calls) == 2:
            q0 = np.asarray(runtime.arm.get_state().q, dtype=float)
            real(q0 + (np.asarray(q, dtype=float) - q0) * 0.5, duration_s=duration_s, **k)
            raise SafetyViolation("halted: superseded")
        return real(q, duration_s=duration_s, **k)

    runtime.arm.move_joints = halted_on_the_descent
    r = runtime.execute("move_near_object", {"label": "red cube", "relation": "above"})
    assert r["ok"] is False and "halted" in r["error"]
    runtime.arm.move_joints = real
    u = runtime.execute("undo_move", {})
    assert u["ok"], u
    np.testing.assert_allclose(_tcp(runtime), start, atol=2e-3)


@needs_pin
def test_move_relative_keeps_its_previous_result_keys(rig):
    """Callers of the old move_relative read `distance_m` and `tcp_xyz`."""
    runtime, _ = rig
    r = runtime.execute("move_relative", {"direction": "up", "amount": "small"})
    assert r["ok"] and r["distance_m"] == pytest.approx(0.02)
    np.testing.assert_allclose(r["tcp_xyz"], _tcp(runtime)[:3, 3], atol=1e-3)


# ── the wrist camera's volume (it sticks out 6-8 cm above the gripper) ──────


def _park_forward(runtime, x=0.48, z=0.13, roll_deg=0.0):
    """Test setup: jaws pointing forward (+x), rolled `roll_deg` about them."""
    from cascade.skills import relative_motion as rm_

    kin = runtime.arm.harness.kin
    T = np.eye(4)
    T[:3, 3] = [x, 0.0, z]
    T[:3, :3] = rm_.rotation_matrix([1.0, 0.0, 0.0], np.radians(roll_deg))
    ik = kin.ik(T, runtime.arm.get_state().q)
    assert ik.success
    # a straight joint-space sweep from home leaves the workspace box on the
    # way; move_planned finds a vetted corner route
    assert runtime.arm.move_planned(ik.q, duration_s=2.0)
    np.testing.assert_allclose(_tcp(runtime), T, atol=2e-3)


@needs_pin
def test_unrolled_the_camera_rides_on_top_and_a_descent_is_fine(rig):
    runtime, _ = rig
    _park_forward(runtime, roll_deg=0.0)
    runtime.arm.harness.limits.table_z = 0.05        # TCP floor 0.07
    r = runtime.execute("move_relative", {"direction": "down", "distance_m": 0.03})
    assert r["ok"], r


@needs_pin
def test_a_rolled_gripper_cannot_drop_the_side_of_the_camera_into_the_table(rig):
    """Rolled 80 deg the camera-up axis still points up, and the TCP would
    end 6 cm over its floor -- but the camera's side (half its padded width,
    7.5 cm) hangs below the TCP and would reach the table."""
    runtime, arm = rig
    half_w = float(runtime.cfg.relative_motion.wrist_camera.half_width_m)
    _park_forward(runtime, z=0.16, roll_deg=80.0)
    runtime.arm.harness.limits.table_z = 0.05        # TCP floor 0.07
    edge = 0.16 - half_w * np.sin(np.radians(80.0))  # camera's lowest corner now
    assert edge > 0.07                                # still above the floor here
    n0 = len(arm.commands)
    r = runtime.execute("move_relative", {"direction": "down", "distance_m": 0.03})
    assert r["ok"] is False and r["refused_by"] == "wrist_camera", r
    assert "table" in r["error"] and "camera" in r["error"]
    assert len(arm.commands) == n0
    assert r["feasible_distance_m"] == pytest.approx(edge - 0.07, abs=0.003)


@needs_pin
def test_a_camera_already_too_low_may_only_climb(rig):
    runtime, arm = rig
    _park_forward(runtime, roll_deg=80.0)
    runtime.arm.harness.limits.table_z = 0.08        # floor 0.10: camera edge 0.091 is under it
    n0 = len(arm.commands)
    down = runtime.execute("move_relative", {"direction": "left", "amount": "small"})
    assert down["ok"] is False and down["refused_by"] == "wrist_camera"
    assert len(arm.commands) == n0
    up = runtime.execute("move_relative", {"direction": "up", "amount": "small"})
    assert up["ok"], up


@needs_pin
def test_the_camera_is_kept_out_of_a_keep_out_zone_the_tcp_never_enters(rig):
    runtime, arm = rig
    tcp = _tcp(runtime)[:3, 3]                       # home: jaws forward, camera on top
    zone = (tcp + [-0.05, -0.01, 0.092], tcp + [-0.01, 0.01, 0.112])
    runtime.arm.harness.limits.keep_out.append(zone)
    n0 = len(arm.commands)
    r = runtime.execute("move_relative", {"direction": "up", "distance_m": 0.03})
    assert r["ok"] is False and r["refused_by"] == "wrist_camera", r
    assert "keep-out" in r["error"]
    assert len(arm.commands) == n0
    assert r["feasible_distance_m"] == pytest.approx(0.012, abs=0.003)   # 9.2 cm gap - 8 cm camera
    # nothing else objects: without the camera model the same move runs
    runtime.cfg._data["relative_motion"]["wrist_camera"] = None
    assert runtime.execute("move_relative", {"direction": "up", "distance_m": 0.03})["ok"]


@needs_pin
def test_checking_the_camera_against_the_occupancy_map_is_opt_in(rig):
    runtime, arm = rig
    tcp = _tcp(runtime)[:3, 3]
    runtime.arm.harness.occupancy = _PointObstacle(tcp + [-0.06, 0.035, 0.07])   # inside the camera box
    assert runtime.execute("move_relative", {"direction": "left", "amount": "small"})["ok"]
    assert runtime.execute("undo_move", {})["ok"]
    runtime.cfg._data["relative_motion"]["wrist_camera"]["check_occupancy"] = True
    n0 = len(arm.commands)
    r = runtime.execute("move_relative", {"direction": "left", "amount": "small"})
    assert r["ok"] is False and r["refused_by"] == "wrist_camera" and "occupancy" in r["error"]
    assert len(arm.commands) == n0


@needs_pin
@pytest.mark.parametrize("bad", [{"protrusion_m": float("nan")}, {"behind_tip_m": -0.1},
                                 {"half_width_m": "wide"}, {"protrusion_m": 0.8},
                                 {"behind_m": 0.1}])   # a typo must not silently keep the default
def test_a_bad_camera_size_refuses_every_move(rig, bad):
    runtime, arm = rig
    runtime.cfg._data["relative_motion"]["wrist_camera"].update(bad)
    n0 = len(arm.commands)
    r = runtime.execute("move_relative", {"direction": "up", "amount": "small"})
    assert r["ok"] is False and "wrist_camera" in r["error"]
    assert len(arm.commands) == n0


@needs_pin
def test_the_camera_volume_is_checked_at_every_sample_not_only_the_ends(rig):
    """A path whose ends are fine but whose middle drops the camera's side
    into the table (e.g. a joint-space arc while rolled) must be refused."""
    from cascade.skills import relative_motion as rm_
    from cascade.skills import spatial_moves

    runtime, _ = rig
    s = spatial_moves.settings(runtime)

    def pose(z, roll_deg):
        T = np.eye(4)
        T[:3, 3] = [0.45, 0.0, z]
        T[:3, :3] = rm_.rotation_matrix([1.0, 0.0, 0.0], np.radians(roll_deg))
        return T

    high = pose(0.10, 0.0)                     # camera bottom 10 cm over the table
    dip = pose(0.05, 80.0)                     # its side at ~1 cm: under the 2 cm clearance
    assert spatial_moves._camera_volume(runtime, [high, high], s) is None
    found = spatial_moves._camera_volume(runtime, [high, dip, high], s)
    assert found is not None and found[0] == "wrist_camera" and "table" in found[1]


# ── the shipped camera box must cover the camera as measured ────────────────


@pytest.mark.parametrize("arm", ["rebot_rs", "mock"])
def test_the_shipped_camera_box_covers_the_measured_reBot_camera(arm):
    """Measured on the rig's photos (2026-10-11, tape measure, +-1-2 cm):
    a D455F (124 x 29 x 26 mm) on top of the wrist, padded with bubble wrap
    and tape, ~15 cm long across the jaws with the padding; its front sits
    ~8 cm behind the fingertips and the padding reaches 13-14 cm back; its
    top is 5-7 cm above the finger axis (user: 6-8 cm). The box may be
    larger -- that only refuses more -- never smaller."""
    from cascade.config import load_demo_config
    from cascade.skills import spatial_moves

    cam = load_demo_config(camera="mock", arm=arm, llm="mock").relative_motion.wrist_camera
    assert float(cam.protrusion_m) >= 0.08
    assert float(cam.behind_tip_m) >= 0.14
    assert 2.0 * float(cam.half_width_m) >= 0.15
    # a config without the key falls back to the same, measured box
    d = spatial_moves.DEFAULT_WRIST_CAMERA
    assert (d["protrusion_m"], d["behind_tip_m"], d["half_width_m"]) == (
        float(cam.protrusion_m), float(cam.behind_tip_m), float(cam.half_width_m))
