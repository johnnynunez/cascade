"""FR3 profile numbers are DERIVED from the vendored URDF, and the MuJoCo
twin (Menagerie fr3 + attached Franka Hand) agrees with that URDF.

Both models come from different upstreams (franka_description vs
mujoco_menagerie) and the hand is attached by our own composer, so the
agreement test is what stops a mount-pose or TCP-offset slip from turning
into "the sim grasps 2 cm beside the object".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cascade.config import load_demo_config
from cascade.control.kinematics import Kinematics
from cascade.grasping.obb_grasp import _yaw_rotation

_ASSETS = Path(__file__).resolve().parents[1] / "assets"
_URDF = _ASSETS / "urdf" / "fr3" / "fr3_arm.urdf"
_MJCF = _ASSETS / "mjcf" / "fr3" / "scene_hand.xml"


def _has_mujoco() -> bool:
    try:
        import mujoco  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


needs_mujoco = pytest.mark.skipif(
    not (_has_mujoco() and _MJCF.exists()),
    reason="needs mujoco + `python scripts/fetch_robot_assets.py fr3`",
)


@pytest.fixture(scope="module")
def kin():
    return Kinematics(str(_URDF), ee_frame="fr3_hand_tcp", n_controlled=7)


def test_fr3_home_is_a_top_down_pose_with_third_open_down(kin):
    """The profile says third_open_down and a home over (0.45, 0, 0.30):
    check both against FK rather than trusting the comment."""
    cfg = load_demo_config(camera="mock", arm="fr3_mock", llm="mock")
    q = np.asarray(cfg.arm.home_q, dtype=float)
    T = kin.fk(q)
    np.testing.assert_allclose(T[:3, 3], [0.45, 0.0, 0.30], atol=2e-3)
    # third_open_down: col2 is the approach axis and points down at the table
    assert cfg.arm.tool_axis_order == "third_open_down"
    np.testing.assert_allclose(T[:3, 2], [0, 0, -1], atol=0.02)
    # and the jaw opens along a horizontal axis (col1)
    assert abs(T[2, 1]) < 0.02
    lo, hi = kin.joint_limits
    assert np.min(np.minimum(q - lo, hi - q)) > 0.4


def test_fr3_strict_top_down_ik_covers_the_declared_annulus(kin):
    """workspace_filter says base_radius 0.20 .. reach 0.85 and the grasp
    band topdown_z_max 0.45; a strict top-down solve must succeed across the
    band the demo actually uses (r 0.30..0.75 measured 1.00)."""
    rng = np.random.default_rng(3)
    cfg = load_demo_config(camera="mock", arm="fr3_mock", llm="mock")
    seed = np.asarray(cfg.arm.home_q, dtype=float)
    ok = 0
    n = 40
    for _ in range(n):
        r = rng.uniform(0.30, 0.70)
        th = rng.uniform(-1.0, 1.0)
        T = np.eye(4)
        T[:3, :3] = _yaw_rotation(rng.uniform(-1.2, 1.2), axis_order="third_open_down")
        T[:3, 3] = [r * np.cos(th), r * np.sin(th), rng.uniform(0.03, 0.25)]
        ok += int(kin.ik(T, seed).success)
    assert ok / n >= 0.95, f"strict top-down solve rate {ok/n:.2f}"


def test_yaw_rotation_third_open_down_geometry():
    R = _yaw_rotation(0.0, axis_order="third_open_down")
    np.testing.assert_allclose(R[:, 2], [0, 0, -1], atol=1e-12)     # approach
    np.testing.assert_allclose(abs(R[2, 1]), 0.0, atol=1e-12)      # opening is horizontal
    assert np.linalg.det(R) == pytest.approx(1.0)
    # yaw rotates the opening axis about the world z
    R2 = _yaw_rotation(np.pi / 2, axis_order="third_open_down")
    assert abs(float(R[:, 1] @ R2[:, 1])) < 1e-9
    with pytest.raises(ValueError, match="axis_order"):
        _yaw_rotation(0.0, axis_order="sideways")


@needs_mujoco
def test_mjcf_hand_tcp_matches_urdf_tcp_over_random_q(kin):
    """scene_hand.xml (composed by fetch_robot_assets.py) vs fr3_arm.urdf:
    same TCP position AND orientation to sub-mm over the joint range."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(_MJCF.resolve()))
    d = mujoco.MjData(m)
    jn = [f"fr3_joint{i}" for i in range(1, 8)]
    qadr = [m.jnt_qposadr[m.joint(n).id] for n in jn]
    hand = m.body("fh_hand").id
    lo, hi = kin.joint_limits
    rng = np.random.default_rng(0)
    worst_p, worst_R = 0.0, 0.0
    for _ in range(50):
        q = rng.uniform(lo, hi)
        for a, v in zip(qadr, q):
            d.qpos[a] = v
        mujoco.mj_forward(m, d)
        R = d.xmat[hand].reshape(3, 3)
        tcp_mj = d.xpos[hand] + R @ np.array([0, 0, 0.1034])
        T = kin.fk(q)
        worst_p = max(worst_p, float(np.linalg.norm(tcp_mj - T[:3, 3])))
        worst_R = max(worst_R, float(np.abs(R - T[:3, :3]).max()))
    assert worst_p < 1e-4, f"TCP mismatch {worst_p*1e3:.2f} mm"
    assert worst_R < 1e-6


@needs_mujoco
def test_mjcf_gripper_ctrl_map_reaches_full_finger_travel():
    """fr3_mujoco.yaml maps finger joint metres -> tendon ctrl 0..255.
    ctrl 255 must OPEN both fingers to 0.04 m and ctrl 0 close them."""
    import mujoco

    cfg = load_demo_config(camera="mock", arm="fr3_mujoco", llm="mock")
    m = mujoco.MjModel.from_xml_path(str(_MJCF.resolve()))
    d = mujoco.MjData(m)
    act = m.actuator(cfg.arm.mj_gripper_actuator).id
    j1 = m.jnt_qposadr[m.joint("fh_finger_joint1").id]
    j2 = m.jnt_qposadr[m.joint("fh_finger_joint2").id]
    scale = float(cfg.arm.mj_gripper_ctrl_scale)
    for target, expect in ((cfg.arm.gripper.open_pos, 0.04), (cfg.arm.gripper.closed_pos, 0.0)):
        d.ctrl[act] = scale * float(target) + float(cfg.arm.mj_gripper_ctrl_offset)
        assert 0.0 <= d.ctrl[act] <= 255.0 + 1e-9
        for _ in range(1500):
            mujoco.mj_step(m, d)
        assert d.qpos[j1] == pytest.approx(expect, abs=1.5e-3)
        assert d.qpos[j2] == pytest.approx(expect, abs=1.5e-3)


@needs_mujoco
def test_fr3_mujoco_profile_names_exist_in_the_composed_scene():
    import mujoco

    cfg = load_demo_config(camera="mock", arm="fr3_mujoco", llm="mock")
    m = mujoco.MjModel.from_xml_path(str(_MJCF.resolve()))
    for n in cfg.arm.mj_joints:
        assert m.joint(n).id >= 0
    for n in cfg.arm.mj_actuators:
        assert m.actuator(n).id >= 0
    assert m.joint(cfg.arm.mj_gripper_joint).id >= 0
    assert m.actuator(cfg.arm.mj_gripper_actuator).id >= 0
