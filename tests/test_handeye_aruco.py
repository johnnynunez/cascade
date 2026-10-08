"""ArUco detection + square-marker PnP (ported from WRC tests/test_aruco_session.py).

WRC's test could not pin the recovered pose: its renderer padded the marker
with white INSIDE the projected quadrilateral, so the black square was 77 %
of the size handed to PnP and every pose came back wrong by construction.
The renderer here maps the marker's own black border onto the projected
corners, so the recovered T_marker2cam is checked against ground truth.

Marker frame = OpenCV ArUco convention: origin at the centre, x right, y up,
z out of the printed face (towards a camera that can read it).
"""

from __future__ import annotations

import numpy as np
import pytest

from cascade.calibration.aruco import ArucoSession, marker_object_points
from cascade.calibration.frames import is_se3, pose_error, se3_inv, so3_exp
from cascade.calibration.synthetic import render_marker
from cascade.types import Frame

K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
SIZE = (640, 480)


def _pose(t, w):
    T = np.eye(4)
    T[:3, :3] = so3_exp(np.asarray(w, dtype=float))
    T[:3, 3] = t
    return T


# Marker 0.5 m in front of the camera, facing it: marker z must point back
# at the camera, i.e. a half-turn about x plus a little tilt.
T_TRUE = _pose([0.02, -0.01, 0.50], [np.pi - 0.25, 0.15, 0.3])


def test_object_points_follow_the_opencv_aruco_corner_order():
    h = 0.05
    assert np.allclose(marker_object_points(0.10),
                       [[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])


def test_detects_and_recovers_the_true_pose():
    img = render_marker(K, SIZE, T_TRUE, size_m=0.10, marker_id=0)
    assert img is not None and img.shape == (480, 640, 3) and img.dtype == np.uint8
    det = ArucoSession("4x4_50").detect(img, K, None, 0.10)
    assert det is not None and det.marker_id == 0
    assert det.corners_px.shape == (4, 2)
    assert is_se3(det.T_marker2cam)
    err = pose_error(se3_inv(T_TRUE) @ det.T_marker2cam)
    assert np.linalg.norm(err[:3]) < 0.003, err
    assert np.degrees(np.linalg.norm(err[3:])) < 2.0, err
    assert det.reprojection_px < 1.0


def test_detect_frame_takes_intrinsics_from_the_frame():
    img = render_marker(K, SIZE, T_TRUE, size_m=0.10)
    frame = Frame(rgb=img, depth_m=None, K=K)
    a = ArucoSession().detect_frame(frame, 0.10)
    b = ArucoSession().detect(img, K, np.zeros(5), 0.10)
    assert a is not None and np.allclose(a.T_marker2cam, b.T_marker2cam)


def test_marker_size_scales_the_translation():
    """The single most common real-rig error: a printer scaled the marker.
    PnP with the wrong size returns a proportionally wrong distance."""
    img = render_marker(K, SIZE, T_TRUE, size_m=0.10)
    det = ArucoSession().detect(img, K, None, 0.095)
    assert det.T_marker2cam[2, 3] == pytest.approx(0.50 * 0.95, abs=0.004)


def test_target_id_filter():
    img = render_marker(K, SIZE, T_TRUE, size_m=0.10, marker_id=0)
    s = ArucoSession()
    assert s.detect(img, K, None, 0.10, target_id=7) is None
    img3 = render_marker(K, SIZE, T_TRUE, size_m=0.10, marker_id=3)
    det = s.detect(img3, K, None, 0.10, target_id=3)
    assert det is not None and det.marker_id == 3


def test_blank_image_is_none():
    assert ArucoSession().detect(np.full((480, 640, 3), 200, np.uint8), K, None, 0.1) is None


def test_unknown_dictionary_raises():
    with pytest.raises(ValueError, match="dictionary"):
        ArucoSession("not_a_dict")


def test_overlay_is_a_copy():
    img = render_marker(K, SIZE, T_TRUE, size_m=0.10)
    s = ArucoSession()
    det = s.detect(img, K, None, 0.10)
    out = s.draw(img, det)
    assert out.shape == img.shape and out is not img and not np.array_equal(out, img)


@pytest.mark.parametrize("T", [
    _pose([0.0, 0.0, 0.5], [0.0, 0.0, 0.0]),        # printed face pointing away
    _pose([0.0, 0.0, -0.5], [np.pi, 0.0, 0.0]),     # behind the camera
    _pose([0.6, 0.0, 0.5], [np.pi, 0.0, 0.0]),      # outside the image
])
def test_renderer_refuses_invisible_markers(T):
    assert render_marker(K, SIZE, T, size_m=0.10) is None


def test_synthetic_camera_follows_the_arm_and_can_inject_an_outlier():
    from cascade.calibration.handeye import EYE_TO_HAND
    from cascade.calibration.synthetic import SyntheticMarkerCamera

    X = _pose([0.30, 0.0, 0.80], [np.pi, 0.0, 0.0])      # looking straight down
    Y = _pose([0.0, 0.0, 0.03], [0.0, 0.0, 0.2])          # marker face up on the TCP
    tcp = {"T": _pose([0.30, 0.02, 0.25], [0.1, -0.1, 0.3])}
    cam = SyntheticMarkerCamera(EYE_TO_HAND, X, Y, lambda: tcp["T"],
                                image_size=(1280, 720), corrupt={1})
    s = ArucoSession()
    f0 = cam.get_frame()
    det = s.detect_frame(f0, 0.10, D=cam.dist_coeffs)
    err = pose_error(se3_inv(cam.marker_in_camera()) @ det.T_marker2cam)
    assert np.linalg.norm(err[:3]) < 0.003
    det1 = s.detect_frame(cam.get_frame(), 0.10)       # grab #1 is corrupted
    assert np.linalg.norm(det1.T_marker2cam[:3, 3] - det.T_marker2cam[:3, 3]) > 0.03
    tcp["T"] = _pose([0.30, 0.02, 0.25], [2.5, 0.0, 0.0])   # marker turned away
    assert s.detect_frame(cam.get_frame(), 0.10) is None
