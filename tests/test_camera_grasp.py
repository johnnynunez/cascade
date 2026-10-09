"""Camera-frame mask grasp planner (`grasping/camera_grasp.py`).

Ported from Seeed's WRC fork (`tests/test_camera_grasp.py` and
`tests/test_camera_grasp_mask_cleaning.py`), which pinned the algorithm of
`rebot_grasp/utils/ordinary_grasp.py::estimate_grasp`: minAreaRect on the
mask, depth-quantile back-projection, approach along the camera's line of
sight, reBot TCP axes, lift by T_cam2base, push `insertion_depth_m` into the
object. Cascade additions: the arm's `tool_axis_order` (columns are not
universal, AGENTS.md), `finger_drop_m` as an argument instead of WRC's
`WRCSIM_FINGER_DROP_M` environment variable, and a stronger finger-cleaning
test (WRC's put the finger exactly ON the drop threshold, so it was kept and
the test passed for an unrelated reason).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cascade.grasping.camera_grasp import (
    DEFAULT_INSERTION_DEPTH_M,
    _grasp_axes_to_rebot_tcp_rotation,
    _normalize,
    plan_grasp_from_mask,
)

_R_TOP = np.diag([1.0, -1.0, -1.0])   # camera looking straight down at the base


def _K(w=640, h=480, f=600.0):
    return np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])


def _top_T(height=1.0):
    T = np.eye(4)
    T[:3, :3] = _R_TOP
    T[:3, 3] = [0.0, 0.0, height]
    return T


def _tilted_T(height=1.0, tilt=math.radians(20.0)):
    c, s = math.cos(tilt), math.sin(tilt)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    T = np.eye(4)
    T[:3, :3] = (Rx @ _R_TOP).T
    T[:3, 3] = [0.0, 0.0, height]
    return T


def _rect(cx, cy, w_px, h_px, H=480, W=640, yaw=0.0):
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    dx, dy = xx - cx, yy - cy
    rx = math.cos(yaw) * dx + math.sin(yaw) * dy
    ry = -math.sin(yaw) * dx + math.cos(yaw) * dy
    return (np.abs(rx) <= w_px / 2.0) & (np.abs(ry) <= h_px / 2.0)


def _depth(z=0.5, H=480, W=640):
    return np.full((H, W), z, dtype=np.float32)


def _plan(**kw):
    base = dict(mask=_rect(320, 240, 100, 50), depth_m=_depth(), K=_K(), T_cam2base=_top_T())
    base.update(kw)
    return plan_grasp_from_mask(**base)


# ── preconditions -> [] (the caller decides what happens next) ─────────────


@pytest.mark.parametrize("kw", [{"mask": None}, {"depth_m": None}, {"K": None},
                                {"T_cam2base": None}])
def test_missing_inputs_return_empty(kw):
    assert _plan(**kw) == []


def test_empty_mask_and_all_invalid_depth_return_empty():
    assert _plan(mask=np.zeros((480, 640), bool)) == []
    assert _plan(depth_m=np.zeros((480, 640), np.float32)) == []


def test_a_mask_at_another_resolution_is_resized_to_the_depth_image():
    """rebot_grasp resizes the mask INTER_NEAREST to the depth image."""
    small = _rect(160, 120, 50, 25, H=240, W=320)
    g = _plan(mask=small)[0]
    np.testing.assert_allclose(g.position[:2], [0.0, 0.0], atol=2e-3)


# ── geometry ───────────────────────────────────────────────────────────────


def test_top_camera_gives_a_top_down_grasp_inserted_into_the_object():
    (g,) = _plan()
    np.testing.assert_allclose(g.approach, [0.0, 0.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(g.position[2], 0.5 - DEFAULT_INSERTION_DEPTH_M, atol=1e-6)
    np.testing.assert_allclose(g.position[:2], [0.0, 0.0], atol=1e-3)
    assert np.isclose(np.linalg.det(g.rotation), 1.0, atol=1e-6)
    np.testing.assert_allclose(g.rotation[:, 0], g.approach, atol=1e-9)   # down_open


@pytest.mark.parametrize("yaw", [0.0, 0.3, -0.7, 1.2])
def test_top_camera_approach_ignores_the_masks_image_yaw(yaw):
    (g,) = _plan(mask=_rect(320, 240, 100, 50, yaw=yaw))
    np.testing.assert_allclose(g.approach, [0.0, 0.0, -1.0], atol=1e-6)


def test_jaws_open_across_the_objects_narrow_side():
    """Long along image x, short along image y; image y is base -y for this
    camera, so the jaw OPENING axis (down_open column 1) is base +-y."""
    (g,) = _plan(mask=_rect(320, 240, 100, 50))
    assert abs(g.rotation[1, 1]) > 0.999
    (g,) = _plan(mask=_rect(320, 240, 50, 100))
    assert abs(g.rotation[0, 1]) > 0.999


@pytest.mark.parametrize("insertion", [0.0, 0.01, 0.015, 0.03])
def test_insertion_depth_pushes_the_tcp_into_the_object(insertion):
    (g,) = _plan(insertion_depth_m=insertion)
    np.testing.assert_allclose(g.position[2], 0.5 - insertion, atol=1e-6)


def test_tilted_camera_approach_follows_the_line_of_sight():
    """The point of the planner: the tool dives along the camera ray, not a
    fixed base-frame vector (WRC's B601 top camera is tilted ~30 deg)."""
    T = _tilted_T(tilt=math.radians(25.0))
    (g,) = _plan(T_cam2base=T)
    surface = g.position + g.approach * DEFAULT_INSERTION_DEPTH_M
    ray = surface - T[:3, 3]
    np.testing.assert_allclose(g.approach, ray / np.linalg.norm(ray), atol=1e-6)
    assert g.approach[2] < -0.85                   # still a downward dive


def test_jaw_width_is_the_short_edge_in_metres_plus_the_pad():
    (g,) = _plan(mask=_rect(320, 240, 100, 50), width_pad_m=0.015)
    np.testing.assert_allclose(g.width_m, 50 * 0.5 / 600.0 + 0.015, atol=2e-3)


def test_too_wide_for_the_jaw_is_reported_with_demoted_quality_not_dropped():
    (g,) = _plan(mask=_rect(320, 240, 200, 200), max_width_m=0.05, confidence=0.9)
    assert g.width_m > 0.05
    assert g.quality == pytest.approx(0.2 * 0.9)
    (ok,) = _plan(confidence=0.9)
    assert ok.quality == pytest.approx(0.9)


def test_label_rides_on_the_grasp_and_pregrasp_retreats_along_the_approach():
    (g,) = _plan(label="banana")
    assert g.label == "banana"
    for offset in (0.05, 0.08, 0.12):
        np.testing.assert_allclose(g.pregrasp_position(offset),
                                   g.position - g.approach * offset, atol=1e-12)


def test_random_scenes_always_give_proper_rotations():
    rng = np.random.default_rng(0)
    for _ in range(20):
        T = _tilted_T(tilt=rng.uniform(-0.4, 0.4))
        mask = _rect(rng.uniform(200, 440), rng.uniform(180, 300), rng.integers(60, 200),
                     rng.integers(30, 100), yaw=rng.uniform(-1.0, 1.0))
        out = _plan(mask=mask, depth_m=_depth(rng.uniform(0.3, 0.8)), T_cam2base=T)
        if not out:
            continue
        R = out[0].rotation
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


# ── the arm's tool-axis convention ─────────────────────────────────────────


def test_open_down_arms_get_approach_in_column_two_and_opening_in_column_zero():
    (ref,) = _plan(T_cam2base=_tilted_T())
    (g,) = _plan(T_cam2base=_tilted_T(), axis_order="open_down")
    np.testing.assert_allclose(g.rotation[:, 2], ref.approach, atol=1e-9)
    np.testing.assert_allclose(g.rotation[:, 0], ref.rotation[:, 1], atol=1e-9)
    np.testing.assert_allclose(g.approach, ref.approach, atol=1e-12)
    np.testing.assert_allclose(g.position, ref.position, atol=1e-12)
    assert np.isclose(np.linalg.det(g.rotation), 1.0)


def test_third_open_down_arms_get_opening_in_column_one():
    (ref,) = _plan(T_cam2base=_tilted_T())
    (g,) = _plan(T_cam2base=_tilted_T(), axis_order="third_open_down")
    np.testing.assert_allclose(g.rotation[:, 1], ref.rotation[:, 1], atol=1e-9)
    np.testing.assert_allclose(g.rotation[:, 2], ref.approach, atol=1e-9)
    assert np.isclose(np.linalg.det(g.rotation), 1.0)


def test_an_unknown_axis_order_is_refused():
    with pytest.raises(ValueError, match="tool_axis_order"):
        _plan(axis_order="sideways")


# ── finger-depth cleaning (WRC: longer fingers inside the YOLO mask) ───────


def _finger_scene(finger_depth):
    """Object 200x60 px at 0.65 m; a 20x20 px finger patch inside the mask,
    off-centre and nearer the camera."""
    mask = np.zeros((480, 640), bool)
    mask[170:230, 220:420] = True
    depth = _depth(0.65)
    mask[220:240, 400:440] = True                # finger, partly outside the object
    depth[220:240, 400:440] = finger_depth
    return mask, depth


def test_finger_pixels_nearer_than_the_drop_threshold_are_removed():
    mask, depth = _finger_scene(finger_depth=0.60)          # 50 mm nearer
    T = np.eye(4)
    (clean,) = plan_grasp_from_mask(mask, depth, _K(), T, insertion_depth_m=0.0,
                                    finger_drop_m=0.030)
    (dirty,) = plan_grasp_from_mask(mask, depth, _K(), T, insertion_depth_m=0.0,
                                    finger_drop_m=0.0)
    true_xy = np.array([(320 - 320) * 0.65 / 600.0, (200 - 240) * 0.65 / 600.0])
    assert np.linalg.norm(clean.position[:2] - true_xy) < 1.5e-3
    assert np.linalg.norm(dirty.position[:2] - true_xy) > 5e-3   # pulled to the finger
    assert clean.position[2] == pytest.approx(0.65, abs=1e-6)


def test_cleaning_keeps_every_pixel_of_an_uncontaminated_mask():
    mask, depth = _finger_scene(finger_depth=0.65)          # no depth tail
    (a,) = plan_grasp_from_mask(mask, depth, _K(), np.eye(4), finger_drop_m=0.030)
    (b,) = plan_grasp_from_mask(mask, depth, _K(), np.eye(4), finger_drop_m=0.0)
    np.testing.assert_allclose(a.position, b.position, atol=1e-12)
    assert a.width_m == pytest.approx(b.width_m)


# ── helpers ────────────────────────────────────────────────────────────────


def test_normalize_handles_the_zero_vector():
    assert _normalize(np.zeros(3)) is None
    np.testing.assert_allclose(_normalize(np.array([3.0, 0.0, 4.0])), [0.6, 0.0, 0.8])


def test_rebot_tcp_rotation_is_right_handed_with_tool_x_against_the_approach():
    rng = np.random.default_rng(42)
    for _ in range(20):
        a = rng.normal(size=3)
        a /= np.linalg.norm(a)
        o = rng.normal(size=3)
        o -= o @ a * a
        o /= np.linalg.norm(o)
        R = _grasp_axes_to_rebot_tcp_rotation(np.cross(o, a), o, a)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)
        np.testing.assert_allclose(R[:, 0], -a, atol=1e-9)


def test_rebot_tcp_z_stays_aligned_with_the_grip_axis():
    grip = np.array([1.0, 0.0, 0.0])
    R = _grasp_axes_to_rebot_tcp_rotation(grip, np.array([0.0, 1.0, 0.0]),
                                          np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(R[:, 0], [0.0, 0.0, -1.0], atol=1e-9)
    assert R[:, 2] @ grip > 0
