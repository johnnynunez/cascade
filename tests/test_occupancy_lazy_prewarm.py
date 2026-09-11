"""No unmasked robot may enter occupancy while LazyArm is in standby."""
import time

import numpy as np
import pytest

from cascade.perception.occupancy import OccupancyMap
from cascade.types import Frame, SafetyViolation


class RecordingClient:
    def __init__(self):
        self.calls = []

    def request(self, payload, timeout_ms=None):
        self.calls.append(payload)
        if payload['action'] == 'query':
            return {'points': np.array([[0.3, 0.0, 0.8]], dtype=np.float32)}
        return {}

    def probe(self, timeout_ms=300):
        return {'ok': True, 'backend': 'recording'}


def frame():
    return Frame(rgb=np.zeros((2, 2, 3), np.uint8),
                 depth_m=np.ones((2, 2), np.float32), K=np.eye(3))


@pytest.mark.parametrize('depth_supported', [True, False])
def test_unknown_robot_pose_cannot_be_integrated_or_age_into_a_clearance_bypass(depth_supported):
    client = RecordingClient()
    occupancy = OccupancyMap(client, region_min=np.zeros(3), region_max=np.ones(3))
    occupancy._depth_supported = depth_supported
    occupancy.add_robot_body(lambda: None)
    occupancy.refresh(frame(), np.eye(4))

    assert client.calls == [], 'standby arm pixels were sent to the obstacle map'
    assert 'robot body pose' in occupancy.last_error
    # No cache (or an old cache) must NOT turn this known safety fault into
    # None = skip the guard. The error persists until a masked frame succeeds.
    occupancy._last_refresh = time.monotonic() - occupancy.max_age_s - 1
    with pytest.raises(SafetyViolation, match='robot body pose'):
        occupancy.clearance([[0.0, 0.0, 0.5]])


def test_isaac_prewarm_reads_current_normalized_pose_without_materializing_arm(monkeypatch, loopback):
    import json
    import socketserver
    import threading

    from cascade.apps import demo
    from cascade.config import load_demo_config
    from cascade.perception.robot_mask import arm_link_points
    from cascade.types import pose_to_transform

    q_raw = np.array([0.0, -1.2, -1.2, 0.0, -0.75, 0.0])
    ops = []

    class ReadOnlyBridge(socketserver.StreamRequestHandler):
        def handle(self):
            for line in self.rfile:
                req = json.loads(line)
                ops.append(req['op'])
                if req['op'] == 'state':
                    response = {'ok': True, 'q': q_raw.tolist()}
                elif req['op'] == 'ping':
                    response = {'ok': True}
                else:
                    response = {'ok': False, 'error': 'actuation forbidden in prewarm'}
                self.wfile.write(json.dumps(response).encode() + b'\n')
                self.wfile.flush()

    srv = socketserver.ThreadingTCPServer((loopback, 0), ReadOnlyBridge)
    srv.daemon_threads = True
    thread = threading.Thread(target=srv.serve_forever, kwargs={'poll_interval': 0.01}, daemon=True)
    thread.start()
    cfg = load_demo_config(arm='isaac', camera='isaac', llm='mock')
    cfg.arm._data.update(bridge_host=loopback, bridge_port=srv.server_address[1],
                         base_pose=[0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    occupancy = OccupancyMap()

    def no_motor_factory(*args, **kwargs):
        pytest.fail('prewarm called the actuator factory')

    monkeypatch.setattr(demo, 'make_arm', no_motor_factory)
    arm = None
    try:
        arm, _, kin = demo._build_arm(cfg.arm, True, occupancy, cfg)
        assert not arm.connected
        assert ops == [], 'construction should not read the bridge yet'
        for turn in range(2):
            q_raw[0] = -0.1 * turn
            points = occupancy._body_fns[0]()
            assert points is not None, 'lazy Isaac body mask has no pose during prewarm'
            T = pose_to_transform(cfg.arm.base_pose)
            expected = arm_link_points(kin, -q_raw) @ T[:3, :3].T + T[:3, 3]
            np.testing.assert_allclose(points, expected)
            assert not arm.connected
        assert ops == ['ping', 'state', 'ping', 'state']
    finally:
        if arm is not None:
            arm.disconnect()
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=1)


@pytest.mark.parametrize('bad_pose', [None, [], [[float('nan'), 0, 0]], [[0, 0]], 'broken'])
def test_invalid_body_pose_cannot_contaminate_after_a_good_refresh(bad_pose):
    client = RecordingClient()
    occupancy = OccupancyMap(client, region_min=np.zeros(3), region_max=np.ones(3))
    pose = np.array([[0.0, 0.0, 0.8], [0.0, 0.0, 1.2]])
    occupancy.add_robot_body(lambda: pose)
    occupancy.refresh(frame(), np.eye(4))
    assert occupancy.last_error is None
    np.testing.assert_allclose(occupancy.clearance(np.array([[0.3, 0, 0.8]])), 0, atol=1e-7)

    client.calls.clear()
    pose = bad_pose
    occupancy.refresh(frame(), np.eye(4))
    assert client.calls == []
    with pytest.raises(SafetyViolation, match='robot body pose'):
        occupancy.clearance(np.array([[0.0, 0.0, 0.5]]))
    # A query-only RGB refresh must not mark the failed mask as repaired.
    occupancy.refresh(Frame(rgb=frame().rgb, depth_m=None, K=np.eye(3)), np.eye(4))
    with pytest.raises(SafetyViolation, match='robot body pose'):
        occupancy.clearance(np.array([[0.0, 0.0, 0.5]]))

    pose = np.array([[0.0, 0.0, 0.8], [0.0, 0.0, 1.2]])
    occupancy.refresh(frame(), np.eye(4))
    assert occupancy.last_error is None
    np.testing.assert_allclose(occupancy.clearance(np.array([[0.3, 0, 0.8]])), 0, atol=1e-7)


def test_unreadable_second_body_stops_integration_of_the_whole_frame():
    client = RecordingClient()
    occupancy = OccupancyMap(client, region_min=np.zeros(3), region_max=np.ones(3))
    occupancy.add_robot_body(lambda: np.array([[0, 0, 0.8], [0, 0, 1.2]]))

    def disconnected():
        raise RuntimeError('standby')

    occupancy.add_robot_body(disconnected)
    occupancy.refresh(frame(), np.eye(4))
    assert client.calls == []
    assert 'standby' in occupancy.last_error
    with pytest.raises(SafetyViolation, match='robot body pose 1'):
        occupancy.clearance(np.array([[0.0, 0.0, 0.5]]))


def test_real_arm_prewarm_never_connects_a_motor_driver(monkeypatch):
    from cascade.apps import demo
    from cascade.config import load_demo_config

    cfg = load_demo_config(arm='rebot_rs', camera='mock', llm='mock')
    occupancy = OccupancyMap(RecordingClient(), region_min=np.zeros(3), region_max=np.ones(3))

    def no_motor_factory(*args, **kwargs):
        pytest.fail('perception attempted to power the physical arm')

    monkeypatch.setattr(demo, 'make_arm', no_motor_factory)
    raw, safe, _ = demo._build_arm(cfg.arm, True, occupancy, cfg)
    try:
        occupancy.refresh(frame(), np.eye(4))
        assert not raw.connected
        assert occupancy.last_error is not None
        with pytest.raises(SafetyViolation, match='robot body pose'):
            safe.harness.vet_pose(np.asarray(cfg.arm.home_q))
    finally:
        raw.disconnect()

