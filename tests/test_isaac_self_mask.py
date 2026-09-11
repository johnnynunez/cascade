"""The self filter must select only this robot's rendered pixels."""
from pathlib import Path
import runpy
import base64
import zlib

import numpy as np
import pytest

from test_isaac_frame_snapshot import capture_bridge, ROBOT  # noqa: F401
from cascade.sim.bridge_client import BridgeClient
from cascade.perception.occupancy import OccupancyMap

HELPER = Path(__file__).resolve().parents[1] / "scripts/isaac_self_mask.py"


def test_render_self_mask_preserves_other_objects_and_prefix_neighbors():
    encode = runpy.run_path(str(HELPER))["encode_robot_mask"]
    ids = np.array([[1, 2, 3], [4, 0, 1]], dtype=np.uint32)
    labels = {"1": "/Robot/link/mesh", "2": "/World_Props/pink_cube",
              "3": "/RobotOther/mesh", "4": "/Robot"}
    p = encode(ids, {"idToLabels": labels}, "/Robot", 12.0)
    actual = np.frombuffer(zlib.decompress(base64.b64decode(p["data"])), np.uint8).reshape(2, 3)
    np.testing.assert_array_equal(actual, [[1, 0, 0], [1, 0, 1]])
    assert p["robot_id"] == "/Robot" and p["t"] == 12.0


def test_explicit_contact_permission_does_not_hide_neighbors():
    encode = runpy.run_path(str(HELPER))["encode_robot_mask"]
    ids = np.array([[1, 2, 3, 4]], np.uint32)
    labels = {"1": "/Robot/link", "2": "/World_Props/pink_cube", "3": "/World_Props/green_cube", "4": "/World_Props/pink_cube_other"}
    p = encode(ids, {"idToLabels": labels}, "/Robot", 1.0, contact_paths=["/World_Props/pink_cube"])
    m = np.frombuffer(zlib.decompress(base64.b64decode(p["data"])), np.uint8)
    np.testing.assert_array_equal(m, [1, 1, 0, 0])
    assert p["contact_paths"] == ["/World_Props/pink_cube"]


def test_unapproved_contact_mask_is_rejected():
    from types import SimpleNamespace
    from cascade.perception.occupancy import OccupancyError

    f = SimpleNamespace(robot_mask=np.zeros((2, 3), bool), depth_m=np.ones((2, 3)),
                        capture={"contact_paths": ["/World_Props/pink_cube"]})
    om = OccupancyMap()
    with pytest.raises(OccupancyError, match="contact"):
        om._render_robot_mask(f)
    om.allowed_contact_paths = {"/World_Props/pink_cube"}
    np.testing.assert_array_equal(om._render_robot_mask(f), f.robot_mask)


def test_missing_render_labels_are_not_an_empty_safe_mask():
    encode = runpy.run_path(str(HELPER))["encode_robot_mask"]
    with pytest.raises(ValueError, match="labels"):
        encode(np.zeros((2, 3), dtype=np.uint32), {}, "/Robot", 12.0)


@pytest.mark.parametrize("depth_supported", [True, False])
def test_real_wire_mask_filters_only_robot_depth(capture_bridge, depth_supported):
    b = capture_bridge
    encode = runpy.run_path(str(HELPER))["encode_robot_mask"]
    ids = np.full(b.depth.shape[:2], 2, np.uint32)
    ids[0, 0] = 1
    payload = b.env["_frames"]["cam0"]
    payload["robot_pixel_mask"] = encode(ids, {"idToLabels": {"1": ROBOT + "/mesh", "2": "/World_Props/pink_cube"}}, ROBOT, payload["t"])
    c = BridgeClient(host=b.host, port=b.port)
    c.connect()
    try:
        frame = c.observation("cam0")
    finally:
        c.close()
    expected_mask = ids == 1
    np.testing.assert_array_equal(getattr(frame, "robot_mask", None), expected_mask)
    packets = []

    class Client:
        def request(self, value):
            packets.append(value)
            return {"points": np.empty((0, 3)), "ok": True}

    om = OccupancyMap(client=Client(), region_min=np.array([-1, -1, -1]), region_max=np.array([20, 20, 20]), stride=1, depth_stride=1)
    om.status = {"backend": "test"}
    om._depth_supported = depth_supported
    # Deliberately over-wide geometric proxy: the exact render mask, not this
    # coarse volume, decides which pixels belong to self.
    om.add_robot_body(lambda: np.array([[0, 0, 0], [0, 0, 1]]), radius_m=100)
    om.refresh(frame, np.eye(4))
    assert om.last_error is None
    if depth_supported:
        expected_depth = frame.depth_m.copy()
        expected_depth[expected_mask] = 0
        np.testing.assert_array_equal(packets[0]["depth"], expected_depth)
    else:
        assert len(packets[0]["points"]) == int((~expected_mask).sum())
    np.testing.assert_allclose(frame.depth_m, .6)  # raw sensor data unchanged


def test_producer_captures_render_mask_in_same_frame(capture_bridge):
    from types import SimpleNamespace

    b = capture_bridge
    ids = np.full(b.depth.shape[:2], 2, np.uint32)
    ids[2, 3] = 1
    old_sensor, K = b.env["_annotators"]["cam0"]

    def get_data(name):
        if name == "instance_id_segmentation":
            return SimpleNamespace(numpy=lambda: ids), {"idToLabels": {"1": ROBOT + "/mesh", "2": "/World_Props/pink_cube"}}
        return old_sensor.get_data(name)

    b.env["_annotators"]["cam0"] = (SimpleNamespace(get_data=get_data), K)
    b.env["_PIXEL_MASK_ENABLED"] = True
    b.env["_encode_robot_mask"] = runpy.run_path(str(HELPER))["encode_robot_mask"]
    b.env["_refresh_frames"]()
    p = b.env["_frames"]["cam0"]
    assert p.get("robot_pixel_mask") is not None
    assert p["robot_pixel_mask"]["t"] == p["proprioception"]["t"]


def test_required_render_mask_cannot_fall_back_to_capsules(capture_bridge):
    from cascade.control.isaac_arm import IsaacArm
    from cascade.config import Cfg
    from cascade.sim.bridge_client import BridgeError

    b = capture_bridge
    c = BridgeClient(host=b.host, port=b.port)
    c.connect()
    try:
        f = c.observation("cam0")
    finally:
        c.close()
    arm = IsaacArm(Cfg({"bridge_host": b.host, "bridge_port": b.port, "bridge_robot_id": ROBOT,
                        "n_joints": 6, "joint_signs": [-1]*6, "require_robot_pixel_mask": True}))
    with pytest.raises(BridgeError, match="pixel mask"):
        arm.state_from_frame(f)


