"""Synthetic D455-like depth of the arm (calibration/synthetic_depth.py).

The markerless solver and the drift monitor are tested against rendered
depth, so the renderer's own geometry is pinned here first: the arm sits
where FK + the camera transform put it, the table is a plane at its height,
and the noise model has the range-proportional sigma and dropout it claims.
A renderer that drew the arm in the wrong place would make every downstream
"recovered within N mm" claim meaningless.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import JOINT_SIGNS, URDF, needs_pin

from cascade.calibration.frames import se3_inv, so3_exp

pytestmark = needs_pin

HOME = np.array([0.0, 1.2, 1.2, 0.0, 0.0, 0.0])
REACH = np.array([0.3, 0.9, 0.7, 0.2, 0.6, -0.4])


def _down_camera(t=(0.30, -0.03, 0.85), tilt=(0.05, -0.06, 0.0)):
    down = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    T = np.eye(4)
    T[:3, :3] = down @ so3_exp(np.asarray(tilt, dtype=float))
    T[:3, 3] = t
    return T


@pytest.fixture(scope="module")
def kin():
    from cascade.control.kinematics import Kinematics

    return Kinematics(str(URDF), "gripper_end", 6, JOINT_SIGNS)


def _camera(kin, q, **kw):
    from cascade.calibration.synthetic_depth import SyntheticDepthCamera

    state = {"q": np.asarray(q, dtype=float)}
    cam = SyntheticDepthCamera(kin, URDF, _down_camera(), lambda: state["q"], **kw)
    return cam, state


def test_clean_render_puts_the_visible_arm_where_fk_and_the_camera_say(kin):
    from cascade.calibration.robot_surface import RobotSurface

    cam, _ = _camera(kin, REACH, noise_frac=0.0, correlated_frac=0.0, dropout=0.0,
                     edge_dropout=0.0)
    depth = cam.render_clean(REACH)
    assert depth.shape == (cam.image_size[1], cam.image_size[0]) and depth.dtype == np.float32
    # Sparse model points facing the camera: most land on a pixel whose
    # rendered depth matches their own camera-frame depth.
    cloud = RobotSurface.from_kinematics(kin, URDF).points_at(REACH)
    Tinv = se3_inv(cam.T_cam2base)
    pc = cloud.points @ Tinv[:3, :3].T + Tinv[:3, 3]
    nc = cloud.normals @ Tinv[:3, :3].T
    facing = np.einsum("ij,ij->i", nc, pc) < -0.3 * np.linalg.norm(pc, axis=1)
    K = cam.K
    u = np.round(K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]).astype(int)
    w, h = cam.image_size
    inside = facing & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    assert inside.sum() > 500
    d = depth[v[inside], u[inside]]
    # Nothing the model says faces the camera may be IN FRONT of the render
    # (that would be the arm drawn in the wrong place, showing table behind).
    behind = d - pc[inside, 2] > 0.01
    assert behind.mean() < 0.03, behind.mean()
    # The links nearest the camera are unoccluded and must match closely;
    # the base and shoulder are legitimately hidden under the forearm here.
    near = np.isin(cloud.link_ids[inside], [cloud_link(kin, "gripper_end"), cloud_link(kin, "link4")])
    match = np.abs(d - pc[inside, 2]) < 0.01
    assert near.sum() > 200 and match[near].mean() > 0.7, match[near].mean()


def cloud_link(kin, name):
    from cascade.calibration.robot_surface import RobotSurface

    return RobotSurface.from_kinematics(kin, URDF).links.index(name)


def test_table_is_a_plane_at_its_height(kin):
    cam, _ = _camera(kin, HOME, noise_frac=0.0, correlated_frac=0.0, dropout=0.0,
                     edge_dropout=0.0, table_z=0.0)
    depth = cam.render_clean(HOME)
    # A corner pixel far from the arm sees the table: back-project and lift.
    v, u = 5, 5
    z = float(depth[v, u])
    assert z > 0
    K = cam.K
    p = np.array([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
    pb = cam.T_cam2base[:3, :3] @ p + cam.T_cam2base[:3, 3]
    assert pb[2] == pytest.approx(0.0, abs=1e-3)


def test_noise_grows_with_range_and_pixels_drop_out(kin):
    cam, _ = _camera(kin, HOME, noise_frac=0.01, correlated_frac=0.0, dropout=0.05,
                     edge_dropout=0.0, seed=4)
    clean = cam.render_clean(HOME)
    frame = cam.get_frame()
    noisy = frame.depth_m
    both = (clean > 0) & (noisy > 0)
    rel = (noisy[both] - clean[both]) / clean[both]
    assert np.std(rel) == pytest.approx(0.01, rel=0.15)
    dropped = np.mean(noisy[clean > 0] == 0)
    assert dropped == pytest.approx(0.05, abs=0.015)
    # Depth units are 1 mm, like the D455's z16 stream.
    assert np.allclose(noisy[both] * 1000, np.round(noisy[both] * 1000), atol=1e-3)


def test_frames_follow_the_arm_and_are_sensor_depth(kin):
    cam, state = _camera(kin, HOME)
    a = cam.get_frame()
    state["q"] = REACH
    b = cam.get_frame()
    assert a.depth_source == "sensor" and b.has_depth and np.allclose(a.K, cam.K)
    assert a.rgb.shape == (cam.image_size[1], cam.image_size[0], 3)
    assert np.mean(np.abs(a.depth_m - b.depth_m) > 0.05) > 0.01     # the arm moved
    # Fresh noise each grab at the same pose; same seed -> same sequence.
    c = cam.get_frame()
    assert not np.array_equal(b.depth_m, c.depth_m)
    cam2, _ = _camera(kin, HOME)
    first = cam2.get_frame()
    assert np.array_equal(first.depth_m, a.depth_m)


def test_bumping_the_camera_changes_what_it_sees(kin):
    cam, _ = _camera(kin, REACH, noise_frac=0.0, correlated_frac=0.0, dropout=0.0,
                     edge_dropout=0.0)
    before = cam.render_clean(REACH).copy()
    bump = np.eye(4)
    bump[:3, :3] = so3_exp([0.0, 0.03, 0.0])
    cam.T_cam2base = cam.T_cam2base @ bump
    after = cam.render_clean(REACH)
    assert np.mean(np.abs(after - before) > 0.02) > 0.01
