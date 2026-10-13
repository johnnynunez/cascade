"""Reachable placement endpoints can still sweep the TCP through an obstacle."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cascade.safety.harness import SafetyHarness, SafetyLimits
from cascade.skills.runtime import SkillRuntime
from cascade.skills.place_geometry import plan as plan_place_geometry
from cascade.types import SkillError


class Kin:
    joint_limits = (np.full(7, -3.), np.full(7, 3.))

    def __init__(self, all_blocked=False):
        self.calls = 0
        self.all_blocked = all_blocked

    def ik(self, T, seed):
        self.calls += 1
        # First reachable orientation dips down mid-approach; the next stays up.
        branch = 1. if self.all_blocked or self.calls <= 2 else -1.
        q = np.zeros(7)
        q[:2] = [branch, T[2, 3]]
        return SimpleNamespace(success=True, q=q)

    def fk(self, q):
        T = np.eye(4)
        dip = .18 * np.sin(np.pi * q[0]) ** 2 if q[0] > 0 else 0.
        T[:3, 3] = [.4, .1, q[1] - dip]
        return T

    def link_positions(self, q):
        return np.array([[0., 0., .3], [0., 0., .3], self.fk(q)[:3, 3]])


def runtime(all_blocked=False):
    rt = SkillRuntime.__new__(SkillRuntime)
    rt.cfg = SimpleNamespace(camera={}, arm={"tool_axis_order": "third_open_down", "type": "ros2",
                                  "ros_gripper_grasp_action": "franka_gripper/grasp",
                                  "gripper": {"force_contact_close": True}},
                             grasp={"topdown_z_max": .45, "pregrasp_offset_m": .1},
                             safety={"table_z": -1.})
    rt.kin = Kin(all_blocked)
    obstacle = np.array([.4, .1, 0.])
    occ = SimpleNamespace(clearance=lambda points: np.linalg.norm(points - obstacle, axis=1))
    h = SafetyHarness(SafetyLimits(workspace_min=np.full(3, -1.), workspace_max=np.ones(3),
                                    table_z=-1., min_clearance_m=.03),
                       kinematics=rt.kin, occupancy=occ)
    q = np.array([0., .2, 0., 0., 0., 0., 0.])
    rt.arm = SimpleNamespace(harness=h, get_state=lambda: SimpleNamespace(q=q))
    rt.memory = Mock()
    rt.held_object = "cork"
    rt._held_object_offset = lambda: None
    rt._move_joints = Mock()
    rt._set_gripper = Mock()
    return rt, q


def test_place_skips_clear_endpoints_with_blocked_approach():
    rt, q = runtime()
    target, hover = np.array([.4, .1, .1]), np.array([.4, .1, .2])
    assert rt.arm.harness.vet_pose(q) is None
    assert rt.arm.harness.vet_pose(np.array([1., .2, 0., 0., 0., 0., 0.])) is None
    geometry = plan_place_geometry(rt, q, target, x=target[0], y=target[1],
                                   release_z=target[2], z_cap=.445,
                                   validate=rt._place_geometry_rejection)
    pre, low = geometry.pre, geometry.low
    assert pre.q[0] == low.q[0] == -1.
    assert rt.arm.harness.violations == []  # candidate rejection never latches STOP
    assert rt.arm.harness._grasp_exempt is None
    rt._move_joints.assert_not_called()
    rt._set_gripper.assert_not_called()


def test_all_blocked_place_paths_keep_object_and_send_no_motion():
    rt, _ = runtime(all_blocked=True)
    with pytest.raises(SkillError, match="no collision-free place path"):
        rt.skill_place_at(.4, .1, .1)
    assert rt.held_object == "cork"
    assert rt.arm.harness.limits.min_clearance_m == .03
    assert rt.arm.harness._grasp_exempt is None
    assert rt.arm.harness.violations == []
    rt._move_joints.assert_not_called()
    rt._set_gripper.assert_not_called()


def test_place_descent_gets_only_the_execution_contact_exemption():
    rt, q = runtime()
    rt._joint_segment_rejection = Mock(side_effect=[None, "blocked descent", None, None])
    target, hover = np.array([.4, .1, .1]), np.array([.4, .1, .2])
    geometry = plan_place_geometry(rt, q, target, x=target[0], y=target[1],
                                   release_z=target[2], z_cap=.445,
                                   validate=rt._place_geometry_rejection)
    pre, low = geometry.pre, geometry.low
    assert pre.q[0] == low.q[0] == -1.
    calls = rt._joint_segment_rejection.call_args_list
    assert calls[0].kwargs == calls[2].kwargs == {}
    for call in (calls[1], calls[3]):
        np.testing.assert_array_equal(call.kwargs['exempt_xy'], target[:2])
        assert call.kwargs['exempt_radius_m'] == .07
        assert call.kwargs['exempt_z_min'] == pytest.approx(.08)
    assert rt.arm.harness._grasp_exempt is None


def test_stop_interrupts_candidate_search():
    rt, q = runtime()
    rt.motion_cancel_requested = lambda: True
    with pytest.raises(SkillError, match="cancelled"):
        plan_place_geometry(rt, q, np.array([.4, .1, .1]), x=.4, y=.1,
                            release_z=.1, z_cap=.445, validate=rt._place_geometry_rejection)
    rt._move_joints.assert_not_called()
    rt._set_gripper.assert_not_called()


def test_clearance_message_exposes_a_submillimetre_shortfall_without_relaxing_gate():
    rt, _ = runtime()
    h = rt.arm.harness
    reason = h._occupancy_violation(np.array([[.4, .1, .02975]]), None)
    assert "0.029750000 m below 0.030000000 m" in reason
    assert "0.250000 mm" in reason
    # Even a float32 representation just below 30 mm remains below the limit.
    assert h._occupancy_violation(np.array([[.4, .1, float(np.float32(.03))]]), None)
    assert h._occupancy_violation(np.array([[.4, .1, .03001]]), None) is None
