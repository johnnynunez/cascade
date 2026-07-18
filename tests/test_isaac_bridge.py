"""Isaac bridge protocol tests: a fake in-process TCP server speaks the
bridge protocol, so IsaacCamera/IsaacArm are covered without Isaac Sim."""

import base64
import json
import socketserver
import threading
import zlib

import cv2
import numpy as np
import pytest

from wrc_demo.config import Cfg
from wrc_demo.control.isaac_arm import IsaacArm
from wrc_demo.perception.isaac_camera import IsaacCamera
from wrc_demo.sim.bridge_client import BridgeClient, BridgeError


class FakeBridge(socketserver.StreamRequestHandler):
    def handle(self):
        for raw in self.rfile:
            req = json.loads(raw)
            self.wfile.write(json.dumps(self._dispatch(req)).encode() + b"\n")
            self.wfile.flush()

    def _dispatch(self, req):
        op = req["op"]
        state = self.server.state
        if op == "ping":
            return {"ok": True}
        if op == "frame":
            h, w = 48, 64
            bgr = np.full((h, w, 3), 90, dtype=np.uint8)
            bgr[10:30, 20:40] = (203, 105, 255)  # a pink block
            _, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            depth = np.full((h, w), 0.6, dtype=np.float32)
            return {
                "ok": True, "width": w, "height": h,
                "K": [[60.0, 0, 32], [0, 60.0, 24], [0, 0, 1]],
                "rgb_jpeg_b64": base64.b64encode(jpg.tobytes()).decode(),
                "depth_z_b64": base64.b64encode(zlib.compress(depth.tobytes())).decode(),
                "t": 1.5,
            }
        if op == "state":
            return {"ok": True, "q": state["q"], "dq": [0.0] * 6,
                    "gripper_pos": state["gripper"]}
        if op == "set_joints":
            state["q"] = list(req["q"])
            return {"ok": True}
        if op == "gripper":
            state["gripper"] = req["pos"]
            return {"ok": True}
        if op == "stop":
            state["stopped"] = True
            return {"ok": True}
        return {"ok": False, "error": f"unknown op {op!r}"}


@pytest.fixture
def bridge_port():
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), FakeBridge)
    srv.daemon_threads = True
    srv.state = {"q": [0.0, 1.2, 1.2, 0.0, 0.75, 0.0], "gripper": -6.8, "stopped": False}
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1], srv.state
    srv.shutdown()
    srv.server_close()


def test_bridge_client_roundtrip(bridge_port):
    port, state = bridge_port
    c = BridgeClient(port=port)
    c.connect()
    try:
        assert c.ping()
        bgr, depth, K = c.frame()
        assert bgr.shape == (48, 64, 3) and depth.shape == (48, 64)
        assert depth.dtype == np.float32 and abs(float(depth[0, 0]) - 0.6) < 1e-6
        assert K[0, 0] == 60.0
        c.set_joints(np.array([0.1] * 6))
        assert state["q"] == [0.1] * 6
        with pytest.raises(BridgeError):
            c.request({"op": "warp"})
    finally:
        c.close()


def test_bridge_client_connection_refused():
    c = BridgeClient(port=1)  # nothing listens there
    with pytest.raises(BridgeError, match="isaac_bridge"):
        c.connect()


def test_isaac_camera_serves_frames(bridge_port):
    port, _ = bridge_port
    cam = IsaacCamera(Cfg({"bridge_port": port}))
    with cam:
        f = cam.get_frame()
        assert f.has_depth and f.depth_source == "sensor"
        assert f.rgb.shape == (48, 64, 3)
        # pink block survives the JPEG round trip
        from wrc_demo.perception.colors import mask_color

        mask = np.zeros((48, 64), dtype=bool)
        mask[12:28, 22:38] = True
        assert mask_color(f.rgb, mask) == "pink"


def test_isaac_arm_state_and_targets(bridge_port):
    port, state = bridge_port
    arm = IsaacArm(Cfg({"bridge_port": port, "n_joints": 6}))
    arm.connect()
    try:
        s = arm.get_state()
        assert s.q.shape == (6,) and s.gripper_valid
        arm.send_joint_target(np.array([0.0, 1.0, 1.0, 0.0, 0.5, 0.0]))
        assert state["q"][1] == 1.0
        arm.set_gripper(-3.0)
        assert state["gripper"] == -3.0
        arm.stop()
        assert state["stopped"]
        with pytest.raises(BridgeError):
            arm.send_joint_target(np.zeros(6))  # soft-stopped
        arm.resume()
        arm.send_joint_target(np.zeros(6))
    finally:
        arm.disconnect()
