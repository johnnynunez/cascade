"""Payload masking must use contact evidence from the rendered physics state."""
import base64
from pathlib import Path
import runpy
from types import SimpleNamespace
import zlib

import numpy as np
import pytest

from test_isaac_frame_snapshot import capture_bridge, ROBOT  # noqa: F401


def configure(b):
    helper = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/isaac_self_mask.py"))
    b.env["os"] = SimpleNamespace(environ={"CASCADE_ISAAC_CONTACT_MASK": "1"}, path=b.env["os"].path)
    b.env["_PIXEL_MASK_ENABLED"] = True
    b.env["_encode_robot_mask"] = helper["encode_robot_mask"]
    b.env["_bilateral_contact_paths"] = helper["bilateral_contact_paths"]
    b.env["_PROP_SPAWNS"] = {"pink_cube": None, "green_cube": None}
    ids = np.full(b.depth.shape[:2], 2, np.uint32)
    ids[0, 0] = 1
    ids[0, 1] = 3
    for name, (sensor, K) in list(b.env["_annotators"].items()):
        def get_data(kind, original=sensor):
            if kind == "instance_id_segmentation":
                return ids, {"idToLabels": {"1": ROBOT + "/mesh", "2": "/World_Props/pink_cube/mesh",
                                            "3": "/World_Props/green_cube/mesh"}}
            return original.get_data(kind)
        b.env["_annotators"][name] = (SimpleNamespace(get_data=get_data,
            get_render_times=sensor.get_render_times, render_product_id=sensor.render_product_id), K)
    return ids


def contacts(attached, b):
    def read(name):
        held = attached and name == "pink_cube"
        return {"channel": b.env["NEWTON_CONTACT_CHANNEL"], "physics_step": b.physics_index[0],
                "jaw_forces_n": [[1., 0., 0.], [1., 0., 0.]] if held else np.zeros((2, 3)),
                "jaw_contact_counts": [1, 1] if held else [0, 0]}
    return read


@pytest.mark.parametrize("historically_attached", [True, False])
def test_delayed_mask_uses_historical_attachment_and_never_current_contacts(capture_bridge, historically_attached):
    b = capture_bridge
    ids = configure(b)
    b.env["_gpu_contact_snapshot"] = contacts(historically_attached, b)
    b.publish()
    historical_index = b.physics_index[0]
    historical_t = b.env["_frames"]["cam0"]["t"]
    b.env["_gpu_contact_snapshot"] = contacts(not historically_attached, b)
    b.env["_step_with_frame_history"]()
    b.render_index[0] = historical_index
    b.env["_frames"].clear()
    b.env["_published_frame_tokens"].clear()
    b.env["_gpu_contact_snapshot"] = lambda _name: pytest.fail("current contacts read during historical publication")
    b.env["_refresh_frames"]()
    packet = b.env["_frames"]["cam0"]
    mask = packet["robot_pixel_mask"]
    assert mask["t"] == packet["proprioception"]["t"] == packet["t"] == historical_t
    assert mask["contact_paths"] == (["/World_Props/pink_cube"] if historically_attached else [])
    payload = np.frombuffer(zlib.decompress(base64.b64decode(mask["payload_data"])), np.uint8).reshape(ids.shape)
    np.testing.assert_array_equal(payload, ids == 2 if historically_attached else np.zeros_like(ids))
    assert not payload[0, 1]  # A neighboring prop never becomes payload.


def test_historical_contact_failure_remains_unknown_after_current_read_recovers(capture_bridge):
    b = capture_bridge
    configure(b)
    def unavailable(_name):
        raise RuntimeError("historical contact sensor unavailable")
    b.env["_gpu_contact_snapshot"] = unavailable
    b.publish()
    historical_index = b.physics_index[0]
    b.env["_gpu_contact_snapshot"] = contacts(True, b)
    b.env["_step_with_frame_history"]()
    b.render_index[0] = historical_index
    b.env["_frames"].clear()
    b.env["_published_frame_tokens"].clear()
    b.env["_refresh_frames"]()
    packet = b.env["_frames"]["cam0"]
    assert "robot_pixel_mask" not in packet
    assert "historical contact sensor unavailable" in packet["robot_pixel_mask_error"]


def test_new_contact_failure_cannot_erase_old_valid_payload_evidence(capture_bridge):
    b = capture_bridge
    configure(b)
    b.env["_gpu_contact_snapshot"] = contacts(True, b)
    b.publish()
    historical_index = b.physics_index[0]
    original = b.env["_frames"]["cam0"]
    def unavailable(_name):
        raise RuntimeError("new contact sensor unavailable")
    b.env["_gpu_contact_snapshot"] = unavailable
    b.env["_step_with_frame_history"]()
    b.render_index[0] = historical_index
    b.env["_refresh_frames"]()
    assert b.env["_frames"]["cam0"] is original
    assert original["robot_pixel_mask"]["contact_paths"] == ["/World_Props/pink_cube"]
