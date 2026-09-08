import numpy as np

from cascade.perception.depth_provider import DepthProvider
from cascade.perception.detector import MockDetector
from cascade.perception.grounding import (
    Extrinsics,
    localize_object,
    mask_to_points_cam,
    oriented_bbox,
)
from cascade.perception.mock_camera import MockCamera, synthetic_tabletop
from cascade.config import Cfg


T_CAM2BASE = np.array(
    [
        [0.0, -1.0, 0.0, 0.28],
        [-1.0, 0.0, 0.0, 0.00],
        [0.0, 0.0, -1.0, 0.60],
        [0.0, 0.0, 0.0, 1.00],
    ]
)


def test_synthetic_frame_geometry():
    f = synthetic_tabletop()
    assert f.rgb.shape == (480, 640, 3)
    assert f.depth_m.shape == (480, 640)
    # Box region is nearer than the table.
    assert f.depth_m[230, 320] < f.depth_m[100, 100]


def test_mock_detector_finds_red_box():
    f = synthetic_tabletop()
    dets = MockDetector().detect(f)
    assert len(dets) == 1
    d = dets[0]
    x0, y0, x1, y1 = d.bbox
    assert 270 <= x0 <= 290 and 350 <= x1 <= 370
    assert d.mask is not None and d.mask.sum() > 1000


def test_mask_backprojection_scale():
    f = synthetic_tabletop()
    d = MockDetector().detect(f)[0]
    pts = mask_to_points_cam(f, d.mask)
    assert pts.shape[0] > 100
    # Box top sits at 0.55 m; spread in x should be ~7 cm (80 px at fx=600).
    assert abs(pts[:, 2].mean() - 0.55) < 0.01
    assert 0.05 < np.ptp(pts[:, 0]) < 0.09


def test_localize_object_in_base_frame():
    f = synthetic_tabletop()
    extr = Extrinsics(mode="eye_to_hand", T=T_CAM2BASE)
    fix = localize_object(f, "red cube", MockDetector(), extr)
    # Camera above (0.28, 0); the box center is near the image center.
    assert abs(fix.position[0] - 0.29) < 0.03
    assert abs(fix.position[1]) < 0.03
    # z: BETWEEN the true centre (0.025) and the top face (0.05), and
    # derivably so. The mid-height skirt gives the cloud a measured z-span of
    # h/2 = 0.025 (top face 0.05 down to the skirt at 0.025 -- a straight-down
    # camera sees no lower), and _recentre_by_size steps half the smallest
    # measured extent in from the near face: 0.05 - 0.0125 = 0.0375. The old
    # flat-lid scene had zero z-span, so localize reported the TOP (~0.045)
    # and the planner aimed the jaws at the upper corner (a measured 2.29 cm
    # shove in MuJoCo). Pinning ~0.05 here would reintroduce that.
    assert 0.025 <= fix.position[2] < 0.045
    assert abs(fix.position[2] - 0.0375) < 0.005


def test_localize_off_center_object_stays_lateral():
    """An object away from the image centre must not be dragged toward it.

    The camera-bias correction anchors on the near surface, but with a
    RANGE-decile anchor an off-centre flat face contributes its near EDGE
    (range grows with lateral offset too), shifting the fix ~14 mm toward
    the principal point -- measured on mock_small + so101_mujoco, where the
    finger then caught the edge, shoved the 35 mm prop, and every grasp
    failed as "air grasp". The along-ray/lateral decomposition keeps the
    lateral coordinate at the face centroid. mock_small's box_px (301..339 x
    308..346) backprojects to base x ~ 0.200 at the top-face depth; 5 mm of
    lateral slack covers mask edge effects, far below the 14 mm defect.
    """
    box_px = (301, 308, 339, 346)
    f = synthetic_tabletop(box_px=box_px, table_depth_m=0.55)
    extr = Extrinsics(mode="eye_to_hand", T=T_CAM2BASE)
    fix = localize_object(f, "red cube", MockDetector(), extr)
    # Ground truth from the same backprojection the scene generator uses:
    # centre pixel (320, 327), fx=600, top face at 0.55-0.05 = 0.50 m.
    u, v = (box_px[0] + box_px[2]) / 2, (box_px[1] + box_px[3]) / 2
    z = 0.55 - 0.05
    x_cam, y_cam = (u - 320) / 600 * z, (v - 240) / 600 * z
    expect = T_CAM2BASE @ np.array([x_cam, y_cam, z, 1.0])
    assert abs(fix.position[0] - expect[0]) < 0.005, (
        f"x pulled {1000 * abs(fix.position[0] - expect[0]):.1f} mm off the "
        "object: the near-surface anchor is biased toward the image centre"
    )
    assert abs(fix.position[1] - expect[1]) < 0.005


def test_localize_fails_readably():
    f = synthetic_tabletop()
    extr = Extrinsics(mode="eye_to_hand", T=T_CAM2BASE)
    det = MockDetector(detections=[])
    try:
        localize_object(f, "unicorn", det, extr)
        raised = False
    except Exception as e:
        raised = True
        assert "unicorn" in str(e)
    assert raised


def test_oriented_bbox_of_box_points(rng):
    pts = rng.uniform([-0.05, -0.02, 0.0], [0.05, 0.02, 0.03], size=(500, 3))
    center, extents, axes = oriented_bbox(pts)
    assert np.allclose(center, [0, 0, 0.015], atol=0.01)
    assert extents[0] > extents[1] > 0


def test_plane_depth_fallback():
    cfg = Cfg({"table_plane_cam": [0.0, 0.0, -1.0, 0.6]})
    dp = DepthProvider(cfg)
    f = synthetic_tabletop()
    f.depth_m = None
    f.depth_source = "none"
    f = dp.ensure_depth(f)
    assert f.depth_source == "plane"
    # Fronto-parallel plane: z-depth is 0.6 m at every pixel (depth maps
    # store z-depth, not ray length).
    assert abs(f.depth_m[240, 320] - 0.6) < 1e-6
    assert abs(f.depth_m[0, 0] - 0.6) < 1e-6


def test_plane_fit_recovers_plane():
    f = synthetic_tabletop()
    plane = DepthProvider.fit_table_plane(f)
    n, d = plane[:3], plane[3]
    assert n[2] < -0.9  # normal points at the camera
    # Table at z=0.6: n.p + d = 0 -> d ~ 0.6 for n=(0,0,-1).
    assert abs(d - 0.6) < 0.03


def test_mock_camera_replay(tmp_path):
    f = synthetic_tabletop()
    np.savez_compressed(tmp_path / "frame_0000.npz", rgb=f.rgb, depth_m=f.depth_m, K=f.K)
    cam = MockCamera(Cfg({"dataset": str(tmp_path)}))
    cam.open()
    g = cam.get_frame()
    assert np.array_equal(g.rgb, f.rgb)
    assert g.frame_id == 1
    cam.close()
