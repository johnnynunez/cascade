"""Real bridge definitions + TCP protocol; Kit boundaries only are doubles.

These fixtures do not simulate physics or establish exposure synchronization.
They pin ownership and same-completed-update capture through the real consumer.
"""
from __future__ import annotations

import ast
import base64
import json
import socketserver
import threading
import time
import zlib
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from conftest import REPO, load_isaac_bridge_definitions
from cascade.config import Cfg
from cascade.perception.isaac_camera import IsaacCamera

ROBOT = "/tn__00armrs_asmv3_hJ6D/Geometry/base_link"


@pytest.fixture
def capture_bridge(loopback):
    """Execute actual producer and Handler, without importing/booting Kit."""
    q = np.array([[0., -1.2, -1.2, 0., -0.75, 0., 0.5]])
    rgba = np.full((12, 16, 4), 90, np.uint8)
    depth = np.full((12, 16, 1), 0.6, np.float32)
    wrist_T = np.eye(4).tolist()
    sensor = SimpleNamespace(get_data=lambda name: (
        rgba if name == "rgb" else depth, {}))
    env = dict(np=np, time=time, cv2=cv2, base64=base64, zlib=zlib, json=json,
               socketserver=socketserver, _frames={}, _wrist_T=wrist_T,
               _annotators={n: (sensor, np.eye(3).tolist()) for n in ("cam0", "side", "wrist")},
               MPU=1., args=SimpleNamespace(prim=ROBOT), engine="newton",
               art=SimpleNamespace(get_dof_positions=lambda: SimpleNamespace(numpy=lambda: q),
                                   get_dof_velocities=lambda: SimpleNamespace(numpy=lambda: np.zeros_like(q))),
               _grip_frac_now=lambda q: 0.5,
               ARM_IDX=list(range(6)), names=[f"joint{i}" for i in range(7)])
    load_isaac_bridge_definitions({"_refresh_frames", "Handler"}, env)
    env["_refresh_frames"]()
    srv = socketserver.ThreadingTCPServer((loopback, 0), env["Handler"])
    srv.daemon_threads = True
    thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    try:
        yield SimpleNamespace(env=env, q=q, rgba=rgba, depth=depth, wrist_T=wrist_T,
                              host=loopback, port=srv.server_address[1])
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=1)


def test_capture_snapshot_survives_real_tcp_camera_delayed_consumption(capture_bridge):
    b = capture_bridge
    before = b.q[0, :6].copy()
    # The physics/renderer buffers are reused AFTER _refresh_frames, before
    # any client reads. This is the aliasing regression, not robot motion.
    b.q[:] = 9.
    b.depth[:] = 3.
    b.rgba[:] = 255
    b.wrist_T[0][3] = 8.
    for _, K in b.env["_annotators"].values():
        K[0][0] = 999.
    frames = []
    for name in ("cam0", "side", "wrist"):
        cam = IsaacCamera(Cfg(dict(bridge_host=b.host, bridge_port=b.port, sim_camera=name)))
        # open() also pings the real Handler; never an actuator op.
        with cam:
            f = cam.get_frame()
        assert getattr(f, "capture", None) is not None, "Isaac discarded capture-time proprioception"
        assert f.capture["source"] == (b.host, b.port)
        assert f.capture["camera"] == name
        snapshot = f.capture["proprioception"]
        assert snapshot["version"] == 1 and snapshot["backend"] == "isaac"
        assert snapshot["robot_id"] == ROBOT
        assert snapshot["joint_convention"] == "asset"
        assert snapshot["time_source"] == "physics_loop_monotonic"
        assert snapshot["t"] == f.capture["t"]
        assert np.isfinite(snapshot["t"])
        np.testing.assert_allclose(snapshot["q"], before)
        np.testing.assert_allclose(f.depth_m, 0.6)
        np.testing.assert_allclose(f.K, np.eye(3))
        assert f.rgb.max() < 100
        if name == "wrist":
            np.testing.assert_allclose(f.T_base_cam, np.eye(4))
        frames.append(f)
    # A subsequent publication must not rewrite any previously returned Frame.
    b.env["_refresh_frames"]()
    for f in frames:
        np.testing.assert_allclose(f.capture["proprioception"]["q"], before)


def test_producer_state_failure_publishes_unmaskable_frame_not_previous_q(capture_bridge):
    b = capture_bridge

    def unavailable():
        raise RuntimeError("stale articulation")

    b.env["art"].get_dof_positions = unavailable
    # RGB-D remains usable for viewing, but never silently reuse the previous
    # q or tear down the entire simulator for an unavailable articulation view.
    b.env["_refresh_frames"]()
    assert b.env["_frames"]["cam0"]["proprioception"] is None


@pytest.mark.parametrize("depth_supported", [True, False])
@pytest.mark.parametrize("ready", [False, True])
def test_demo_masks_delayed_frame_with_captured_not_current_joints(capture_bridge, monkeypatch, depth_supported, ready):
    from cascade.apps import demo
    from cascade.config import load_demo_config
    from cascade.perception.occupancy import OccupancyMap
    from cascade.perception.robot_mask import arm_link_points, robot_mask
    from test_occupancy_lazy_prewarm import RecordingClient

    b = capture_bridge
    cfg = load_demo_config(arm="isaac", camera="isaac", llm="mock")
    cfg.arm._data.update(bridge_host=b.host, bridge_port=b.port)
    client = RecordingClient()
    occ = OccupancyMap(client, region_min=np.zeros(3), region_max=np.ones(3),
                       depth_stride=1, stride=1)
    occ._depth_supported = depth_supported
    monkeypatch.setattr(demo, "make_arm", lambda *a, **kw: pytest.fail("materialized lazy motors"))
    arm, _, kin = demo._build_arm(cfg.arm, True, occ, cfg)
    q_capture = -b.q[0, :6].copy()  # shipped Isaac profile is raw -> local negation
    # Synthetic sensor boundary places a visible patch on a REAL FK link.
    # This exercises the real geometric mask, not a fake mask callback.
    links = arm_link_points(kin, q_capture)
    T = np.eye(4)
    T[:3, 3] = links[-2] - [0., 0., .6]
    b.env["_annotators"]["cam0"] = (b.env["_annotators"]["cam0"][0],
                                    [[100., 0., 8.], [0., 100., 6.], [0., 0., 1.]])
    b.env["_refresh_frames"]()
    cam = IsaacCamera(Cfg(dict(bridge_host=b.host, bridge_port=b.port)))
    try:
        with cam:
            f = cam.get_frame()
        good = robot_mask(f.depth_m, f.K, T, links, radius_m=.06)
        bad_links = arm_link_points(kin, q_capture + [1., 0., 0., 0., 0., 0.])
        wrong = robot_mask(f.depth_m, f.K, T, bad_links, radius_m=.06)
        assert good.any() and np.any(good != wrong), "fixture cannot expose a pose-epoch error"
        # Detector/scheduling delay: current joints change, frame stays fixed.
        # The actual Handler can serve the different current pose, but the
        # mask MUST use the captured q, no current-state RPC.
        b.q[0, 0] -= 1.
        if ready:
            arm._arm = SimpleNamespace(n_joints=6, disconnect=lambda: None,
                                       get_state=lambda: pytest.fail("read current q for old frame"))
        occ.refresh(f, T)
        assert occ.last_error is None, occ.last_error
        assert arm.connected == ready
        if depth_supported:
            packet = next(p for p in client.calls if p["action"] == "integrate_depth")
            np.testing.assert_array_equal(packet["depth"] == 0, good)
        else:
            expected = occ._depth_to_base_points(f, T)[~good[f.depth_m > 0]]
            packets = [p for p in client.calls if p["action"] == "integrate"]
            if len(expected):
                np.testing.assert_allclose(packets[0]["points"], expected)
            else:
                assert not packets
    finally:
        arm.disconnect()


@pytest.mark.parametrize("require_cuda", [False, True])
def test_actual_main_loop_captures_before_jobs_and_keeps_rendered_wrist_pose(capture_bridge, require_cuda):
    """Run one real main-loop iteration; do not copy its scheduling logic."""
    b = capture_bridge
    events, rendered_T, rendered_q = [], [], []

    def wrist():
        events.append("wrist")
        b.wrist_T[0][3] += .1

    def update():
        events.append("update")
        b.q[0, 0] += .2
        rendered_q.append(b.q[0, :6].copy())
        rendered_T.append(np.array(b.wrist_T))

    def jobs():
        events.append("jobs")
        b.q[0, 0] += .7

    tree = ast.parse((REPO / "scripts/isaac_bridge.py").read_text())
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.While)
             and any(isinstance(test, ast.Call) and ast.unparse(test) == "app.is_running()"
                     for test in ast.walk(node.test))]
    assert len(loops) == 1, "expected one simulator main loop"
    loop = loops[0]
    running = iter([True, False])
    refresh = b.env["_refresh_frames"]

    def capture():
        events.append("capture")
        refresh()

    b.env.update(app=SimpleNamespace(is_running=lambda: next(running), update=update),
                 _tl=SimpleNamespace(is_playing=lambda: True), _was_playing=True,
                 _REQUIRE_CUDA=require_cuda, _gpu_log_guard=SimpleNamespace(check=lambda: None),
                 _state_lock=threading.Lock(), _targets=dict(q=None, grip_frac=None, stopped=True),
                 _run_exec_jobs=jobs, _update_wrist_cam=wrist, _refresh_frames=capture, step=1)
    b.env["args"].cam_every = 2
    exec(compile(ast.Module(body=[loop], type_ignores=[]), "isaac_actual_loop", "exec"), b.env)
    assert events == ["wrist", "update", "capture", "jobs"]
    entry = b.env["_frames"]["wrist"]
    np.testing.assert_array_equal(entry["proprioception"]["q"], rendered_q[0])
    np.testing.assert_array_equal(entry["T_base_cam"], rendered_T[0])


@pytest.mark.parametrize("depth_supported", [True, False])
@pytest.mark.parametrize("bad", [
    "missing", "wrong_robot", "wrong_backend", "wrong_source", "wrong_version",
    "wrong_convention", "short_q", "long_q", "nan_q", "nested_q", "bool_q",
    "missing_q", "bad_time", "different_time", "wrong_clock", "bad_envelope",
])
def test_bad_snapshot_latches_mask_fault_without_fallback(capture_bridge, monkeypatch, bad, depth_supported):
    import copy

    from cascade.apps import demo
    from cascade.config import load_demo_config
    from cascade.perception.occupancy import OccupancyMap
    from cascade.types import Frame, SafetyViolation
    from test_occupancy_lazy_prewarm import RecordingClient

    b = capture_bridge
    cfg = load_demo_config(arm="isaac", camera="isaac", llm="mock")
    cfg.arm._data.update(bridge_host=b.host, bridge_port=b.port)
    client = RecordingClient()
    occ = OccupancyMap(client, region_min=np.zeros(3), region_max=np.ones(3))
    occ._depth_supported = depth_supported
    monkeypatch.setattr(demo, "make_arm", lambda *a, **kw: pytest.fail("materialized lazy motors"))
    arm, _, _ = demo._build_arm(cfg.arm, True, occ, cfg)
    cam = IsaacCamera(Cfg(dict(bridge_host=b.host, bridge_port=b.port)))
    try:
        with cam:
            good = cam.get_frame()
        occ.refresh(good, np.eye(4))
        assert occ.last_error is None
        occ._depth_supported = depth_supported
        client.calls.clear()
        packet = b.env["_frames"]["cam0"]
        s = packet["proprioception"]
        if bad == "missing":
            del packet["proprioception"]
        elif bad == "bad_envelope":
            packet["proprioception"] = "invalid"
        elif bad == "wrong_robot":
            s["robot_id"] = "/World/another_arm"
        elif bad == "wrong_backend":
            s["backend"] = "mujoco"
        elif bad == "wrong_version":
            s["version"] = 2
        elif bad == "wrong_convention":
            s["joint_convention"] = "local"
        elif bad == "short_q":
            s["q"] = [0.]
        elif bad == "long_q":
            s["q"] += [0.]
        elif bad == "nan_q":
            s["q"][0] = float("nan")
        elif bad == "nested_q":
            s["q"] = [s["q"]]
        elif bad == "bool_q":
            s["q"] = [False] * 6
        elif bad == "missing_q":
            del s["q"]
        elif bad == "bad_time":
            s["t"] = float("nan")
        elif bad == "different_time":
            s["t"] += 1.
        elif bad == "wrong_clock":
            s["time_source"] = "exposure_guess"
        with cam:
            f = cam.get_frame()
        if bad == "wrong_source":
            # A camera on another bridge, same robot prim spelling.
            f.capture["source"] = (b.host, b.port + 1)
        occ.refresh(f, np.eye(4))
        assert client.calls == [], f"{bad} proprioception reached integration"
        assert "robot body pose" in occ.last_error
        occ._last_refresh = time.monotonic() - occ.max_age_s - 1
        with pytest.raises(SafetyViolation, match="robot body pose"):
            occ.clearance([[0., 0., .6]])
        # A query-only observation cannot clear this failure.
        occ.refresh(Frame(rgb=f.rgb, depth_m=None, K=f.K), np.eye(4))
        with pytest.raises(SafetyViolation, match="robot body pose"):
            occ.clearance([[0., 0., .6]])
        occ.refresh(copy.deepcopy(good), np.eye(4))
        assert occ.last_error is None
        occ.clearance([[0., 0., .6]])
    finally:
        arm.disconnect()


def test_other_registered_arm_cannot_borrow_same_bridge_snapshot(capture_bridge, monkeypatch):
    from cascade.apps import demo
    from cascade.config import load_demo_config
    from cascade.perception.occupancy import OccupancyMap
    from cascade.types import SafetyViolation
    from test_occupancy_lazy_prewarm import RecordingClient

    b = capture_bridge
    cfg = load_demo_config(arm="isaac", camera="isaac", llm="mock")
    cfg.arm._data.update(bridge_host=b.host, bridge_port=b.port)
    client = RecordingClient()
    occ = OccupancyMap(client, region_min=np.zeros(3), region_max=np.ones(3))
    monkeypatch.setattr(demo, "make_arm", lambda *a, **kw: pytest.fail("motor factory"))
    left, _, _ = demo._build_arm(cfg.arm, True, occ, cfg)
    right_cfg = Cfg({**cfg.arm._data, "bridge_robot_id": "/World/right"})
    right, _, _ = demo._build_arm(right_cfg, True, occ, cfg)
    cam = IsaacCamera(Cfg(dict(bridge_host=b.host, bridge_port=b.port)))
    try:
        with cam:
            f = cam.get_frame()
        occ.refresh(f, np.eye(4))
        assert client.calls == []
        with pytest.raises(SafetyViolation, match="robot body pose 1.*identity"):
            occ.clearance([[0., 0., .6]])
        assert not left.connected and not right.connected
    finally:
        left.disconnect()
        right.disconnect()


@pytest.mark.parametrize("depth", [True, False])
def test_stream_and_depth_provider_preserve_capture(capture_bridge, depth):
    from cascade.perception.depth_provider import DepthProvider
    from cascade.perception.stream import CameraStream

    b = capture_bridge
    stream = CameraStream(IsaacCamera(Cfg(dict(bridge_host=b.host, bridge_port=b.port, depth=depth))),
                          rate_hz=10)
    stream.open()
    try:
        f, _ = stream.wait_newer(0, timeout_s=2.)
        assert f is not None and f.capture is not None
        snapshot = f.capture
        provider = DepthProvider(Cfg({"table_plane_cam": [0., 0., -1., .6]}))
        filled = provider.ensure_depth(f)
        assert filled.capture is snapshot
        assert filled.capture["source"] == (b.host, b.port)
        np.testing.assert_allclose(filled.capture["proprioception"]["q"], b.q[0, :6])
    finally:
        stream.close()
