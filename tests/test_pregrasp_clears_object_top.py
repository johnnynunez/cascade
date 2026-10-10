"""The pregrasp, and so the lift, can be held above the object's top.

Found on the physical reBot (front D435i, warp occupancy): a paper cup whose
observed top was 9.3 cm was grasped at 5.2 cm (its widest visible section).
With the fixed 4 cm pregrasp offset the lift ended with the fingertips (they
end at gripper_end) at the rim, inside the 3 cm occupancy margin, and every
route home was refused: "no safe joint route found; direct route: point 0
clearance 0.015 m below 0.030 m (occupancy map)".
"""

from types import SimpleNamespace

import numpy as np
import pytest

from conftest import needs_pin
from cascade.grasping.obb_grasp import _yaw_rotation
from cascade.grasping.selector import select_grasp
from cascade.safety.harness import SafetyHarness, SafetyLimits
from cascade.safety.trajectory import plan_route
from cascade.types import Grasp

CUP_XY = np.array([0.3449, 0.1287])
CUP_TOP = 0.093
GRASP_Z = 0.0516
OFFSET = 0.04


@pytest.fixture
def rebot():
    from cascade.config import load_demo_config
    from cascade.control.kinematics import Kinematics

    cfg = load_demo_config(arm="rebot_rs", camera="mock", llm="mock")
    arm = cfg.arm
    kin = Kinematics(arm.model, arm.get("ee_frame", "gripper_end"), 6,
                     joint_signs=arm.get("joint_signs"))
    return SimpleNamespace(cfg=cfg, kin=kin, home=np.asarray(arm.home_q, dtype=float))


def cup_grasp(approach=(0.0, 0.0, -1.0)):
    yaw = float(np.arctan2(CUP_XY[1], CUP_XY[0]))
    return Grasp(position=np.array([*CUP_XY, GRASP_Z]), rotation=_yaw_rotation(yaw, axis_order="down_open"),
                 width_m=0.062, approach=np.array(approach), quality=1.0, label="paper cup")


def tcp_z(rebot, q):
    return float(rebot.kin.fk(q)[2, 3])


@needs_pin
def test_default_pregrasp_is_the_fixed_offset(rebot):
    _, q_pre, _ = select_grasp([cup_grasp()], rebot.kin, rebot.home, pregrasp_offset_m=OFFSET)
    assert tcp_z(rebot, q_pre) == pytest.approx(GRASP_Z + OFFSET, abs=1e-3)


@needs_pin
def test_pregrasp_rises_above_the_object_top(rebot):
    _, q_pre, q_grasp = select_grasp([cup_grasp()], rebot.kin, rebot.home, pregrasp_offset_m=OFFSET,
                                     pregrasp_min_z=CUP_TOP + 0.03)
    assert tcp_z(rebot, q_pre) == pytest.approx(CUP_TOP + 0.03, abs=1e-3)
    assert tcp_z(rebot, q_grasp) == pytest.approx(GRASP_Z, abs=1e-3)


@needs_pin
def test_a_lower_minimum_never_lowers_the_pregrasp(rebot):
    _, q_pre, _ = select_grasp([cup_grasp()], rebot.kin, rebot.home, pregrasp_offset_m=OFFSET,
                               pregrasp_min_z=GRASP_Z + 0.01)
    assert tcp_z(rebot, q_pre) == pytest.approx(GRASP_Z + OFFSET, abs=1e-3)


@needs_pin
def test_unreachable_or_refused_raise_falls_back_to_the_fixed_offset(rebot):
    # top-down IK fails far above the B601-RS wrist range
    _, q_pre, _ = select_grasp([cup_grasp()], rebot.kin, rebot.home, pregrasp_offset_m=OFFSET,
                               pregrasp_min_z=0.60)
    assert tcp_z(rebot, q_pre) == pytest.approx(GRASP_Z + OFFSET, abs=1e-3)
    # validation refuses the raised pose (e.g. something above the object)
    refuse_high = lambda g, q_pre, q_grasp: "blocked above" if tcp_z(rebot, q_pre) > 0.1 else None
    _, q_pre, _ = select_grasp([cup_grasp()], rebot.kin, rebot.home, pregrasp_offset_m=OFFSET,
                               pregrasp_min_z=CUP_TOP + 0.03, validate=refuse_high)
    assert tcp_z(rebot, q_pre) == pytest.approx(GRASP_Z + OFFSET, abs=1e-3)


def test_only_downward_approaches_are_raised():
    seen = []

    class Kin:
        def ik(self, T, seed):
            seen.append(T[:3, 3].copy())
            return SimpleNamespace(success=True, q=np.zeros(6), error=0.0)

    side = Grasp(position=np.array([0.3, 0.0, 0.05]), rotation=np.eye(3), width_m=0.05,
                 approach=np.array([1.0, 0.0, 0.0]), quality=1.0, label="box")
    select_grasp([side], Kin(), np.zeros(6), pregrasp_offset_m=OFFSET, pregrasp_min_z=0.5)
    assert seen[0] == pytest.approx([0.3 - OFFSET, 0.0, 0.05])


class PointObstacles:
    required = False

    def __init__(self, points):
        self.points = np.asarray(points, dtype=float)

    def clearance(self, query):
        query = np.atleast_2d(np.asarray(query, dtype=float))
        return np.linalg.norm(query[:, None, :] - self.points[None, :, :], axis=2).min(axis=1)


def cup_in_map(r=0.035, n=48):
    """Cup wall cells from the table to the observed top. The front camera sees
    the cup's front half, so the grasp sits ~2 cm off the real axis, toward the
    camera (+x): the TCP then has the front wall 1.5 cm away, as on the rig."""
    axis = CUP_XY - [0.02, 0.0]
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.vstack([np.c_[axis[0] + r * np.cos(a), axis[1] + r * np.sin(a), np.full(n, z)]
                      for z in np.arange(0.01, CUP_TOP + 1e-9, 0.01)])


@needs_pin
def test_the_rig_trap_is_gone_after_a_raised_lift(rebot):
    """After the lift (back to the pregrasp) the arm can go home through the map."""
    h = SafetyHarness(SafetyLimits.from_config(rebot.cfg.safety), kinematics=rebot.kin,
                      occupancy=PointObstacles(cup_in_map()))
    h.heartbeat()
    _, q_low, _ = select_grasp([cup_grasp()], rebot.kin, rebot.home, pregrasp_offset_m=OFFSET)
    reason = h.vet_pose(q_low)
    assert reason and "occupancy map" in reason          # premise: today's lift pose is trapped
    _, q_high, _ = select_grasp([cup_grasp()], rebot.kin, rebot.home, pregrasp_offset_m=OFFSET,
                                pregrasp_min_z=CUP_TOP + 0.03)
    assert h.vet_pose(q_high) is None
    route = plan_route(h, q_high, rebot.home)
    assert np.allclose(route[-1], rebot.home)


@needs_pin
def test_only_the_rebot_profile_turns_it_on():
    from cascade.config import load_demo_config

    assert load_demo_config(arm="rebot_rs", camera="mock", llm="mock").grasp.get("pregrasp_clear_top_m") == 0.03
    assert load_demo_config(arm="mock", camera="mock", llm="mock").grasp.get("pregrasp_clear_top_m") is None
