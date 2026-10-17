"""Spatial-command axes follow the FR3 TCP without inventing mount geometry."""
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.config import load_demo_config
from cascade.grasping.obb_grasp import _yaw_rotation
from cascade.skills import relative_motion
from cascade.skills.spatial_moves import CameraDims, settings, wrist_camera_dims


@pytest.mark.parametrize("arm", ["fr3", "fr3_mock", "fr3_mujoco"])
@pytest.mark.parametrize("yaw", [-np.pi, 0.0, np.pi / 2])
def test_fr3_spatial_camera_up_accepts_topdown_and_refuses_inversion(arm, yaw):
    cfg = load_demo_config(arm=arm, camera="d435i", llm="mock")
    spatial = settings(SimpleNamespace(cfg=cfg))
    rotation = _yaw_rotation(yaw, axis_order=cfg.arm.tool_axis_order)
    # +Z approaches the object while the back of the wrist remains above it.
    np.testing.assert_allclose(rotation[:, 2], [0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(rotation @ spatial.tool_up_axis, [0, 0, 1], atol=1e-12)
    assert relative_motion.camera_up_violation(
        [rotation], spatial.tool_up_axis, spatial.tool_up_min_z) is None
    inverted = rotation @ np.diag([1.0, -1.0, -1.0])
    assert relative_motion.camera_up_violation(
        [rotation, inverted], spatial.tool_up_axis, spatial.tool_up_min_z) is not None
    # The existing escape policy still permits recovery from an inverted start.
    assert relative_motion.camera_up_violation(
        [inverted, rotation], spatial.tool_up_axis, spatial.tool_up_min_z) is None


@pytest.mark.parametrize("arm", ["fr3", "fr3_mock", "fr3_mujoco"])
def test_fr3_does_not_inherit_the_rebot_camera_collision_box(arm):
    cfg = load_demo_config(arm=arm, camera="d435i", llm="mock")
    spatial = settings(SimpleNamespace(cfg=cfg))
    assert wrist_camera_dims(cfg.relative_motion) == (None, (0.0, 0.0, -1.0))
    assert spatial.wrist_camera is None
    assert spatial.camera_occupancy is False  # Only the absent camera volume.
    # Camera geometry overrides do not replace the robot's motion safety data.
    assert cfg.safety.max_joint_vel == 1.0
    assert cfg.safety.min_clearance_m > 0
    assert cfg.safety.table_clearance > 0
    assert cfg.arm.ros_max_joint_velocity_rad_s == 0.5

    rebot = load_demo_config(arm="rebot_rs", camera="mock", llm="mock")
    dimensions, up = wrist_camera_dims(rebot.relative_motion)
    assert dimensions == CameraDims(0.08, 0.14, 0.075)
    assert up == (0.0, 0.0, 1.0)
