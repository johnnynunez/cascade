"""Robot-body masking before occupancy integration (perception/robot_mask.py).

Measured failure this guards: with a live distance-field bridge and NO
masking, the SO-101 at HOME read clearance 0.000 / 0.007 / 0.001 m at three
of its own link points (the camera sees the arm; the arm became an
obstacle) and the very first grasp was refused by the harness.
"""

from __future__ import annotations

import numpy as np

from cascade.perception.occupancy import OccupancyError, OccupancyMap
from cascade.perception.robot_mask import arm_link_points, robot_mask
from cascade.types import Frame

W, H, FX = 160, 120, 150.0
K = np.array([[FX, 0, W / 2], [0, FX, H / 2], [0, 0, 1]], dtype=float)
# camera 0.6 m above the table looking straight down (cam z -> -base z)
T = np.array([[0, -1, 0, 0.28], [-1, 0, 0, 0.0], [0, 0, -1, 0.60], [0, 0, 0, 1]], dtype=float)


def _depth_with_arm_and_cube():
    """Table at 0.60; an 'arm' as a vertical post at base (0.20, 0.00) from
    z=0 to 0.25 (reads as depth 0.35 over its footprint); a 5 cm cube at
    base (0.28, 0.10) (depth 0.55)."""
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    d = np.full((H, W), 0.60, dtype=np.float32)
    # cube: cam x = -(base y - 0) ... use the inverse mapping: base (x,y) -> cam (x_c = -(y - 0), y_c = -(x - 0.28))
    def footprint(bx, by, half, depth_val):
        xc = -(by - 0.0); yc = -(bx - 0.28)
        m = (np.abs((us - W / 2) / FX * depth_val - xc) < half) & (np.abs((vs - H / 2) / FX * depth_val - yc) < half)
        d[m] = depth_val
    footprint(0.28, 0.10, 0.025, 0.55)   # cube
    footprint(0.20, 0.00, 0.02, 0.35)    # arm post top at z=0.25
    return d


def test_robot_mask_covers_the_arm_and_spares_the_object():
    depth = _depth_with_arm_and_cube()
    post = np.array([[0.20, 0.0, 0.0], [0.20, 0.0, 0.25]])   # base -> tip
    m = robot_mask(depth, K, T, post, radius_m=0.04)
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    arm_px = np.isclose(depth, 0.35)
    cube_px = np.isclose(depth, 0.55)
    assert m[arm_px].mean() > 0.95, "arm pixels not masked"
    assert m[cube_px].sum() == 0, "the object next to the arm must NOT be masked"
    # mask does not leak far onto the table
    assert m.sum() < arm_px.sum() * 4


def test_robot_mask_radius_is_a_real_knob():
    depth = _depth_with_arm_and_cube()
    post = np.array([[0.20, 0.0, 0.0], [0.20, 0.0, 0.25]])
    small = robot_mask(depth, K, T, post, radius_m=0.01).sum()
    big = robot_mask(depth, K, T, post, radius_m=0.08).sum()
    assert big > small > 0


def test_robot_mask_handles_no_links_and_empty_depth():
    depth = np.zeros((H, W), dtype=np.float32)
    assert not robot_mask(depth, K, T, np.zeros((2, 3))).any()
    assert not robot_mask(_depth_with_arm_and_cube(), K, T, np.zeros((0, 3))).any()


class _RecordingClient:
    """Captures what would go over the wire."""

    def __init__(self):
        self.depths: list[np.ndarray] = []

    def request(self, payload, timeout_ms=None):
        if payload["action"] == "integrate_depth":
            self.depths.append(np.asarray(payload["depth"]))
            return {"ms": 0.1}
        if payload["action"] == "query":
            return {"points": np.zeros((0, 3), np.float32), "grid": np.full((2, 2, 2), np.inf, np.float32),
                    "origin": np.zeros(3, np.float32), "voxel": 0.02}
        if payload["action"] == "probe":
            return {"ok": True, "backend": "fake", "device": "cpu", "voxel": 0.02, "esdf": True}
        raise OccupancyError("unknown action")

    def probe(self, timeout_ms=300):
        return self.request({"action": "probe"})


def test_map_zeroes_the_registered_arm_before_shipping_the_frame():
    depth = _depth_with_arm_and_cube()
    frame = Frame(rgb=np.zeros((H, W, 3), np.uint8), depth_m=depth, K=K)
    c = _RecordingClient()
    m = OccupancyMap(client=c, region_min=np.array([0, -0.3, -0.02]), region_max=np.array([0.6, 0.3, 0.4]), depth_stride=1)
    post = np.array([[0.20, 0.0, 0.0], [0.20, 0.0, 0.25]])
    m.add_robot_body(lambda: post, radius_m=0.04)
    m.refresh(frame, T_base_cam=T)
    assert m.last_error is None
    shipped = c.depths[-1]
    arm_px = np.isclose(depth, 0.35)
    assert (shipped[arm_px] == 0).all(), "arm depth reached the bridge"
    assert np.isclose(shipped[np.isclose(depth, 0.55)], 0.55).all(), "cube depth was altered"
    assert m.last_masked_px == int(arm_px.sum()) or m.last_masked_px >= int(arm_px.sum() * 0.95)


def test_an_arm_in_standby_defers_integration_until_it_can_be_masked():
    depth = _depth_with_arm_and_cube()
    frame = Frame(rgb=np.zeros((H, W, 3), np.uint8), depth_m=depth, K=K)
    c = _RecordingClient()
    m = OccupancyMap(client=c, region_min=np.zeros(3), region_max=np.ones(3), depth_stride=1)
    m.add_robot_body(lambda: None, radius_m=0.04)             # LazyArm not up yet

    def boom():
        raise RuntimeError("arm not connected")
    m.add_robot_body(boom, radius_m=0.04)
    m.refresh(frame, T_base_cam=T)
    assert m.last_error is not None and "robot body pose" in m.last_error
    assert c.depths == [], "unknown pose must never send unmasked arm depth"
    assert m.last_masked_px == 0


def test_arm_link_points_appends_the_tcp():
    class FakeKin:
        def link_positions(self, q):
            return np.array([[0, 0, 0.1], [0.1, 0, 0.2]])

        def fk(self, q):
            T = np.eye(4); T[:3, 3] = [0.2, 0, 0.1]; return T

    pts = arm_link_points(FakeKin(), np.zeros(5))
    assert pts.shape == (3, 3)
    np.testing.assert_allclose(pts[-1], [0.2, 0, 0.1])
