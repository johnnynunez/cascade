"""Check the mount against a stationary real FR3 RGB-D table capture."""
import numpy as np
import pytest

from cascade.config import load_profile
from cascade.control.kinematics import Kinematics
from cascade.perception.grounding import Extrinsics


def test_rgb_origin_contains_mount_measurements_and_one_lateral_lens_offset():
    cfg = load_profile('cameras', 'd435i')
    transform = Extrinsics.from_config(cfg.extrinsics)
    np.testing.assert_allclose(transform.T[:3, 3], [.030, -.0325, -.035])
    nominal = transform.T.copy()
    nominal[1, 3] = 0.  # measured mount centerline, before the RGB lens offset
    # Datasheet: 17.5 mm mount-to-left-IR + 15 mm left-IR-to-RGB.
    # Optical -X maps to TCP -Y; the translation rotates with the hand.
    # Repeated composition must not accumulate the offset.
    for yaw in [0., .6, np.pi / 2, np.pi]:
        c, s = np.cos(yaw), np.sin(yaw)
        hand = np.array([[c, -s, 0., .4], [s, c, 0., -.1],
                         [0., 0., 1., .3], [0., 0., 0., 1.]])
        transform._fk = lambda: hand
        for _ in range(2):
            delta = transform.cam_to_base()[:3, 3] - (hand @ nominal)[:3, 3]
            np.testing.assert_allclose(hand[:3, :3].T @ delta, [0., -.0325, 0.], atol=1e-12)


def test_no_additional_grasp_correction():
    from cascade.config import load_demo_config
    cfg = load_demo_config(camera='d435i', arm='fr3', llm='mock')
    assert cfg.grasp.get('target_offset_base_m') is None
    assert cfg.grasp.get('backoff_tool_x_m') is None
    assert cfg.grasp.get('backoff_toward_base_m') is None


def test_d435i_roll_matches_measured_horizontal_table_and_far_image_top():
    pytest.importorskip("pinocchio")
    # 2026-10-01 aligned D435i RGB-D capture, no robot motion. Largest table
    # plane: 226,971 inliers, 2.4 mm RMSE. These measurements are independent
    # of the configured camera rotation.
    q = np.array([0.0846556574, -1.2900493145, 0.0424589477,
                  -2.8778376579, 0.0390989408, 2.0713210106, 0.8384267688])
    normal_cam = np.array([-0.0218954071, -0.4646328369, -0.8852326915])
    plane_offset = -0.4626861221
    arm = load_profile("arms", "fr3")
    kin = Kinematics(arm.model, arm.ee_frame, arm.n_joints, arm.joint_signs)
    camera = load_profile("cameras", "d435i")
    T = kin.fk(q) @ np.array(camera.extrinsics.T)
    normal_base = T[:3, :3] @ normal_cam
    assert np.dot(normal_base, [0, 0, 1]) > np.cos(np.deg2rad(0.2))
    assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0)
    # Points on the observed table at upper/lower centre-image pixels.
    rays = np.array([[0, (50 - 244.7605133) / 607.6699829, 1],
                     [0, (430 - 244.7605133) / 607.6699829, 1]])
    depths = plane_offset / (rays @ normal_cam)
    points_base = (rays * depths[:, None]) @ T[:3, :3].T + T[:3, 3]
    assert depths[0] > depths[1]
    assert points_base[0, 0] > points_base[1, 0]
    assert np.ptp(points_base[:, 2]) < 0.002
    assert np.max(np.abs(points_base[:, 2])) < 0.02
