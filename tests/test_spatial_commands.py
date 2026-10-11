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


def _park_low(runtime, z=0.10):
    """Test setup: a top-down pose `z` above the real table (through SafeArm,
    so the harness vets it like any motion). Raising `table_z` instead would
    put the elbow 'under the table' and refuse everything for the wrong
    reason."""
    from cascade.grasping.obb_grasp import _yaw_rotation
    from cascade.types import make_transform

    kin = runtime.arm.harness.kin
    T = make_transform(_yaw_rotation(0.0, axis_order=runtime._tool_axis_order), [0.30, 0.0, z])
    ik = kin.ik(T, runtime.arm.get_state().q)
    assert ik.success
    assert runtime.arm.move_joints(ik.q, duration_s=2.0)
    np.testing.assert_allclose(_tcp(runtime)[:3, 3], [0.30, 0.0, z], atol=2e-3)


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
