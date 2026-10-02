"""JPEG/software RPC fixtures. NOT images captured from Isaac or physics."""
import base64
import copy
import threading
import time

import cv2
import numpy as np
import pytest
from mobile_support_fixture import support_contract

from cascade.sim.mobile_bridge import MobileBridgeController, MobileBridgeServer


def jpeg_fixture(width=24, height=16, value=80):
    ok, encoded = cv2.imencode(".jpg", np.full((height, width, 3), value, np.uint8))
    assert ok
    return encoded.tobytes()


@pytest.fixture
def frame_endpoint():
    from mobile_support_fixture import support
    c = MobileBridgeController(robot_id="microduck", source="isaac-microduck", engine="physx",
        device="cuda:0", asset_sha256="a" * 64, policy_sha256="b" * 64, model_identity_sha256="e" * 64, support_contract=support_contract(),
        max_linear_speed=.15, max_angular_speed=.6, max_duration_s=3., lease_s=.5,
        max_state_age_s=2., max_action_wall_s=8.)
    c.publish({"step": 1, "sim_time": .005, "position": [0., 0., .3],
               "orientation_wxyz": [1., 0., 0., 0.], "linear_velocity": [0., 0., 0.],
               "angular_velocity": [0., 0., 0.], "q": [0.] * 14, "dq": [0.] * 14,
               "joint_names": [f"fixture-{i}" for i in range(14)], "contacts": [],
               "fallen": False, "balance_active": True, "support": support(1, .005)})
    jpeg = jpeg_fixture()
    packet = {key: c.hello()[key] for key in ("robot_id", "source", "epoch", "engine", "device",
                                             "asset_sha256", "policy_sha256", "model_identity_sha256")}
    packet.update(camera="side", step=1, sim_time_s=.005, width=24, height=16,
                  rgb_jpeg_b64=base64.b64encode(jpeg).decode(), producer_age_s=0.)
    operations = []
    server = MobileBridgeServer(c, port=0, frame_callback=lambda req: {"ok": True, "frame": copy.deepcopy(packet)})
    dispatch = server.dispatch
    def record(req):
        operations.append({k: copy.deepcopy(v) for k, v in req.items() if not k.startswith("_")})
        return dispatch(req)
    server.dispatch = record
    server.start()
    profile = {key: c.hello()[key] for key in ("robot_id", "source", "engine", "device",
                                              "asset_sha256", "policy_sha256", "model_identity_sha256")}
    profile.update(type="isaac", bridge_host=server.address[0], bridge_port=server.address[1], timeout_s=.2,
                   support_contract=support_contract(),
                   cameras={"side": dict(max_age_s=.3, max_jpeg_bytes=65536, max_pixels=4096)})
    try:
        yield c, server, profile, packet, jpeg, operations
    finally:
        server.close()


def test_frame_reader_real_rpc_is_passive_and_keeps_exact_provenance(frame_endpoint):
    from cascade.sim.mobile_frames import MobileFrameReader

    c, server, profile, packet, jpeg, operations = frame_endpoint
    reader = MobileFrameReader(profile)
    assert not operations  # lazy construction
    generation = c.hello()["generation"]
    try:
        frame = reader("side")
        assert frame is not None, reader.last_error
        assert frame.jpeg == jpeg
        meta = frame.as_dict()
        assert all(meta[key] == packet[key] for key in ("source", "epoch", "step", "width", "height", "camera"))
        assert "depth" not in meta and "K" not in meta and "T_base_cam" not in meta
        assert reader.cached("side").jpeg == jpeg
        assert reader("side").as_dict()["step"] == 1  # reads never advance clocks
    finally:
        reader.close()
    assert operations == [{"op": "hello", "role": "reader"}, {"op": "frame", "camera": "side"},
                          {"op": "frame", "camera": "side"}]
    assert c.hello()["generation"] == generation
    assert reader.cached("side") is None
    assert reader("side") is None


@pytest.mark.parametrize("key,value", [
    ("robot_id", "other"), ("source", "other"), ("epoch", "old"), ("engine", "newton"),
    ("device", "cpu"), ("asset_sha256", "c" * 64), ("policy_sha256", "c" * 64), ("model_identity_sha256", "f" * 64),
    ("camera", "other"), ("step", True), ("step", -1), ("step", 1.5),
    ("sim_time_s", True), ("sim_time_s", -1.), ("sim_time_s", float("nan")),
    ("producer_age_s", True), ("producer_age_s", -1.), ("producer_age_s", .31),
    ("width", 25), ("height", 0), ("width", True), ("width", 100000),
    ("rgb_jpeg_b64", "not base64"), ("rgb_jpeg_b64", base64.b64encode(b"not jpeg").decode()),
    ("depth", "uncontracted field"),
])
def test_malformed_frames_fail_closed_and_clear_cache(frame_endpoint, key, value):
    from cascade.sim.mobile_frames import MobileFrameReader
    _, _, profile, packet, _, _ = frame_endpoint
    reader = MobileFrameReader(profile)
    try:
        assert reader("side") is not None
        packet[key] = value
        assert reader("side") is None, key
        assert reader.last_error
        assert reader.cached("side") is None
    finally:
        reader.close()


@pytest.mark.parametrize("change", [
    {"type": "mock"}, {"cameras": []}, {"cameras": {" side": {}}},
    {"cameras": {"side": {"max_age_s": .3}}},
    {"cameras": {"side": dict(max_age_s=True, max_jpeg_bytes=10, max_pixels=10)}},
    {"cameras": {"side": dict(max_age_s=.3, max_jpeg_bytes=0, max_pixels=10)}},
    {"cameras": {"side": dict(max_age_s=.3, max_jpeg_bytes=10, max_pixels=10.5)}},
    {"cameras": {"side": dict(max_age_s=.3, max_jpeg_bytes=10, max_pixels=2000000000)}},
])
def test_bad_camera_profile_rejected_before_any_connect(frame_endpoint, monkeypatch, change):
    from cascade.sim.mobile_frames import MobileFrameReader
    import socket
    _, _, profile, _, _, _ = frame_endpoint
    profile.update(change)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("profile dialled"))
    with pytest.raises(ValueError):
        MobileFrameReader(profile)


def test_frame_repeats_cannot_rejuvenate_or_change_clock_or_pixels(frame_endpoint):
    from cascade.sim.mobile_frames import MobileFrameReader
    _, _, profile, packet, _, operations = frame_endpoint
    profile["cameras"]["side"]["max_age_s"] = .03
    reader = MobileFrameReader(profile)
    try:
        assert reader("side")
        count = len(operations)
        assert reader("missing") is None
        assert len(operations) == count
        assert reader("side")
        time.sleep(.04)
        assert reader.cached("side") is None
        assert reader("side") is None  # even if producer dishonestly resets age
        packet.update(step=2, sim_time_s=.01)
        assert reader("side")
        packet["rgb_jpeg_b64"] = base64.b64encode(jpeg_fixture(value=140)).decode()
        assert reader("side") is None
        packet.update(step=3, sim_time_s=.005)
        assert reader("side") is None
    finally:
        reader.close()


def test_jpeg_header_size_and_actual_size_bounded_before_decode(frame_endpoint, monkeypatch):
    from cascade.sim.mobile_frames import MobileFrameReader
    _, _, profile, packet, jpeg, _ = frame_endpoint
    # JPEG header itself exceeds profile max_pixels despite small advertised size.
    packet["rgb_jpeg_b64"] = base64.b64encode(jpeg_fixture(100, 100)).decode()
    reader = MobileFrameReader(profile)
    monkeypatch.setattr(cv2, "imdecode", lambda *a: pytest.fail("unbounded decode"))
    try:
        assert reader("side") is None
        packet["rgb_jpeg_b64"] = base64.b64encode(jpeg + b"x" * 70000).decode()
        assert reader("side") is None
    finally:
        reader.close()


def test_slow_frame_age_includes_whole_rtt_and_close_joins(frame_endpoint):
    from cascade.sim.mobile_frames import MobileFrameReader
    _, server, profile, packet, _, _ = frame_endpoint
    profile["cameras"]["side"]["max_age_s"] = .02
    def delayed(req):
        time.sleep(.04)
        return {"ok": True, "frame": copy.deepcopy(packet)}
    server._frame = delayed
    reader = MobileFrameReader(profile)
    try:
        assert reader("side") is None
    finally:
        reader.close()


@pytest.mark.parametrize("key,value", [("protocol", True), ("kind", "rebot"), ("measurement_kind", "kinematic_mock"),
    ("robot_id", "wrong"), ("source", "wrong"), ("engine", "newton"), ("device", "cpu"),
    ("asset_sha256", "c" * 64), ("policy_sha256", "c" * 64), ("model_identity_sha256", "f" * 64), ("physics_dt", .01),
    ("policy_dt", True), ("capabilities", []), ("epoch", "wrong")])
def test_frame_hello_is_bound_like_truth_reader(frame_endpoint, monkeypatch, key, value):
    from cascade.sim.mobile_frames import MobileFrameReader
    c, server, profile, _, _, operations = frame_endpoint
    profile["epoch"] = c.hello()["epoch"]
    dispatch = server.dispatch
    def altered(req):
        result = dispatch(req)
        if req["op"] == "hello":
            result[key] = value
        return result
    monkeypatch.setattr(server, "dispatch", altered)
    reader = MobileFrameReader(profile)
    try:
        assert reader("side") is None
        assert reader.last_error
        assert not any(r["op"] == "frame" for r in operations)
    finally:
        reader.close()


@pytest.mark.parametrize("wire", [b'{"ok":true,"ok":true,"frame":{}}\n',
    b'{"ok":true,"frame":{"step":NaN}}\n', b'{"ok":1,"frame":{}}\n',
    b'{"ok":true,"frame":{}}\n{}\n', b'x' * 120000])
def test_camera_wire_limits_duplicate_json_and_nonfinite_fail_closed(frame_endpoint, monkeypatch, wire):
    from cascade.sim.mobile_frames import MobileFrameReader
    _, server, profile, _, _, _ = frame_endpoint
    send = server._send
    def altered(conn, response):
        if "frame" in response:
            conn.sendall(wire)
        else:
            send(conn, response)
    monkeypatch.setattr(server, "_send", altered)
    reader = MobileFrameReader(profile)
    try:
        assert reader("side") is None
        assert reader.last_error
    finally:
        reader.close()


def test_blocked_frame_rpc_has_deadline_close_and_cache_never_waits(frame_endpoint):
    from cascade.sim.mobile_frames import MobileFrameReader
    _, server, profile, packet, _, _ = frame_endpoint
    profile["timeout_s"] = .05
    entered, release = threading.Event(), threading.Event()
    def blocked(req):
        entered.set()
        release.wait(1)
        return {"ok": True, "frame": copy.deepcopy(packet)}
    server._frame = blocked
    reader = MobileFrameReader(profile)
    results = []
    thread = threading.Thread(target=lambda: results.append(reader("side")))
    try:
        thread.start()
        assert entered.wait(1)
        started = time.monotonic()
        assert reader.cached("side") is None
        assert time.monotonic() - started < .02
        reader.close()
        thread.join(.3)
        assert not thread.is_alive() and results == [None]
        assert time.monotonic() - started < .2
    finally:
        release.set()
        thread.join(1)
        reader.close()


def test_jpeg_duplicate_sof_cannot_bypass_allocation_bound(frame_endpoint, monkeypatch):
    from cascade.sim.mobile_frames import MobileFrameReader
    _, _, profile, packet, jpeg, _ = frame_endpoint
    sof = jpeg.index(b"\xff\xc0")
    size = int.from_bytes(jpeg[sof + 2:sof + 4], "big")
    duplicate = bytearray(jpeg[sof:sof + size + 2])
    duplicate[5:9] = b"\xff\xff\xff\xff"  # second SOF wants an enormous decoded allocation
    altered = jpeg[:sof + size + 2] + duplicate + jpeg[sof + size + 2:]
    packet["rgb_jpeg_b64"] = base64.b64encode(altered).decode()
    monkeypatch.setattr(cv2, "imdecode", lambda *a: pytest.fail("duplicate SOF reached native allocation"))
    reader = MobileFrameReader(profile)
    try:
        assert reader("side") is None
    finally:
        reader.close()
