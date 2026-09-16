"""CPU kinematic regressions from physically recorded Isaac carry failures.

The recorder below checks planned paths only; it is not grasp evidence.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.control.kinematics import Kinematics
from cascade.skills.runtime import SkillRuntime
from cascade.types import SkillError


MODEL = (Path(__file__).resolve().parents[1] / "assets" / "urdf"
         / "00-arm-rs_asm-v3" / "urdf" / "00-arm-rs_asm-v3.urdf")


def runtime(q, *, settle=True):
    rt = SkillRuntime.__new__(SkillRuntime)
    rt.cfg = SimpleNamespace(
        arm={}, safety={"table_z": 0.0},
        grasp={"pregrasp_offset_m": 0.04, "topdown_z_max": 0.15,
               "release_height_m": 0.05},
    )
    rt.kin = Kinematics(str(MODEL), "gripper_end", 6, [-1] * 6)
    rt.held_object = "test object"
    rt._held_det_label = "test object"
    rt._held_color = None
    rt._held_offset = np.array([0.0, 0.0, -0.032])
    rt._held_object_offset = lambda: rt._held_offset
    rt._grip_open = 1.0
    rt.memory = SimpleNamespace(add=lambda *args, **kwargs: None)
    rt.beliefs = SimpleNamespace(update=lambda *args, **kwargs: None)
    state = SimpleNamespace(q=np.asarray(q, dtype=float))
    goals, gripper = [], []

    def move(goal, **kwargs):
        goals.append(np.asarray(goal).copy())
        state.q = np.asarray(goal).copy()
        return settle

    rt.arm = SimpleNamespace(
        get_state=lambda: state,
        move_joints=move,
        set_gripper=lambda *args, **kwargs: gripper.append(args),
        harness=SimpleNamespace(allow_grasp_descent=lambda *args, **kwargs: None,
                                clear_grasp_exemption=lambda: None),
    )
    return rt, goals, gripper


def yaw(T):
    return np.arctan2(T[1, 1], T[0, 1])


def angle_difference(a, b):
    return abs(np.arctan2(np.sin(a - b), np.cos(a - b)))


def test_pink_carry_keeps_the_held_yaw_and_lift_height():
    # Pink-02's stable pre-transfer configuration, in LOCAL joint signs.
    q = [-0.632842, 1.557939, 1.287957, -1.312603, -0.000014562, -2.798962]
    rt, goals, _ = runtime(q)
    start = rt.kin.fk(q)
    rt.skill_place_at(0.14, -0.27)
    hover = rt.kin.fk(goals[0])
    assert angle_difference(yaw(hover), yaw(start)) < 0.002
    assert hover[2, 3] >= min(start[2, 3], 0.145) - 0.0002
    assert np.allclose(hover[:2, 3], [0.14, -0.27], atol=0.0002)
    assert max(np.abs(goals[0] - q)) < np.pi


def test_can_near_wrist_limit_uses_a_nearby_yaw_without_unwinding():
    # Can-01, sample 100: top-down and physically carrying the can.
    q = [-0.756133, 1.849315, 1.701903, -1.439553, -0.000003778, 1.660184]
    rt, goals, _ = runtime(q)
    start = rt.kin.fk(q)
    rt.skill_place_at(0.14, -0.27)
    hover = rt.kin.fk(goals[0])
    # Preserving the exact endpoint yaw needs a 4.43 rad wrist winding.
    # A 45-degree yaw adjustment avoids that while keeping the lifted can up.
    assert angle_difference(yaw(hover), yaw(start)) < np.pi / 4 + 0.002
    assert max(np.abs(goals[0] - q)) < np.pi
    assert hover[2, 3] >= start[2, 3] - 0.0002
    assert np.allclose(hover[:2, 3], [0.14, -0.27], atol=0.0002)
    assert max(np.abs(goals[1] - goals[0])) < np.pi


def test_hover_settle_failure_still_keeps_the_jaws_closed():
    rt, goals, gripper = runtime([-0.756, 1.849, 1.702, -1.440, 0.0, 1.660],
                                settle=False)
    with pytest.raises(SkillError, match="did not settle above the place target"):
        rt.skill_place_at(0.14, -0.27)
    assert len(goals) == 1
    assert gripper == []
    assert rt.held_object == "test object"
