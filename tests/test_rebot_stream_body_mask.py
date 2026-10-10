"""The occupancy body mask must not read the reBot's bus during a stream.

Found on the physical reBot: one measured pose polls the six joint motors in
turn (mechPos, 8.07 ms each over motorbridge, measured), ~48 ms with the
driver lock held. The occupancy watcher masks the arm on every refresh (3 Hz)
and keeps refreshing during motion, so each mask read held up two to three
20 ms waypoints, which then went out back to back: the arm moved in steps.
With the occupancy map off the same moves were smooth, and a 3 Hz map refresh
without bus reads left a 50 Hz loop's largest gap at 22 ms (20 ms without).
"""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.apps.demo import _body_mask_q
from cascade.config import Cfg
from cascade.control import rebot_rs_arm as rs
from cascade.control.lazy_arm import LazyArm
from cascade.control.rebot_rs_arm import RebotRSArm
from cascade.types import RobotState

MEASURED = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


def connected_arm(monkeypatch):
    """RebotRSArm on a fake SDK: MIT sends are recorded, position reads counted."""
    arm = RebotRSArm(Cfg({}))
    sends, reads = [], []
    arm._arm = SimpleNamespace(arm=SimpleNamespace(send_mit=lambda q, **kw: sends.append(np.copy(q))),
                               has_gripper=False)
    arm._stopped = False
    arm._read_failures = 0
    arm._last_cmd_q, arm._last_cmd_t = None, None
    arm._mit_kp = arm._mit_kd = None
    arm.settle_timeout_s = 0.2

    def read_positions():
        reads.append(1)
        return MEASURED.copy()

    monkeypatch.setattr(arm, "_read_positions", read_positions)
    return arm, sends, reads


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(rs.time, "monotonic", lambda: now[0])
    return now


def test_streaming_mask_uses_the_target_without_bus_reads(monkeypatch, clock):
    arm, _, reads = connected_arm(monkeypatch)
    target = np.array([0.0, 1.2, 1.2, 0.0, 0.0, 0.0])
    arm.send_joint_target(target)
    clock[0] += 0.02                       # next waypoint is due: still streaming
    np.testing.assert_allclose(arm.body_mask_q(), target)
    assert reads == []


def test_idle_or_stopped_mask_reads_the_measured_pose(monkeypatch, clock):
    arm, _, reads = connected_arm(monkeypatch)
    np.testing.assert_allclose(arm.body_mask_q(), MEASURED)        # nothing commanded yet
    arm.send_joint_target(np.zeros(6))
    clock[0] += 0.5                                                # stream over: settling / idle
    np.testing.assert_allclose(arm.body_mask_q(), MEASURED)
    arm.send_joint_target(np.zeros(6))
    arm.stop()                                                     # frozen: measured, not the target
    np.testing.assert_allclose(arm.body_mask_q(), MEASURED)
    assert len(reads) == 3


def test_safety_feedback_still_reads_the_bus_while_streaming(monkeypatch, clock):
    arm, _, reads = connected_arm(monkeypatch)
    arm.send_joint_target(np.zeros(6))
    np.testing.assert_allclose(arm.get_state().q, MEASURED)
    assert len(reads) == 1


def test_no_bus_reads_while_the_mask_is_queried_during_a_real_stream(monkeypatch):
    arm, sends, reads = connected_arm(monkeypatch)
    in_stream = threading.Event()
    reads_in_stream = []
    send = arm.send_joint_target

    def tracked_send(q):
        in_stream.set()
        send(q)
    monkeypatch.setattr(arm, "send_joint_target", tracked_send)
    orig = arm._read_positions

    def tracked_read():
        if in_stream.is_set() and len(sends) < 25:     # strictly inside the stream
            reads_in_stream.append(1)
        return orig()
    monkeypatch.setattr(arm, "_read_positions", tracked_read)

    stop = threading.Event()
    masks = []

    def watcher():                                      # the occupancy refresh, flat out
        while not stop.is_set():
            if in_stream.is_set() and len(sends) < 25:
                masks.append(arm.body_mask_q())

    t = threading.Thread(target=watcher, daemon=True)
    t.start()
    try:
        arm.stream_to(MEASURED + 0.05, duration_s=0.5, rate_hz=50.0, settle_tol=1.0)
    finally:
        stop.set()
        t.join(2)
    assert len(sends) == 25 and masks, "the mask was never queried during the stream"
    assert reads_in_stream == []


def test_runtime_mask_helper_keeps_other_arms_on_measured_pose():
    plain = SimpleNamespace(get_state=lambda: RobotState(q=MEASURED.copy(), gripper_pos=0.0))
    np.testing.assert_allclose(_body_mask_q(plain), MEASURED)
    with_hook = SimpleNamespace(body_mask_q=lambda: np.ones(6), get_state=lambda: pytest.fail("bus read"))
    np.testing.assert_allclose(_body_mask_q(with_hook), np.ones(6))


def test_runtime_mask_helper_through_a_connected_lazy_arm():
    backend = SimpleNamespace(get_state=lambda: RobotState(q=MEASURED.copy(), gripper_pos=0.0),
                              connect=lambda: None, n_joints=6)
    lazy = LazyArm(lambda: backend, n_joints=6)
    lazy._ensure()
    np.testing.assert_allclose(_body_mask_q(lazy), MEASURED)   # backend without the hook


# ── the motorbridge transport (rebot_rs_mb) has the same lock and polls ──


class _Motor:
    def __init__(self, sends):
        self.sends = sends

    def send_mit(self, *a, **kw):
        self.sends.append(a)

    def robstride_get_param_f32(self, *a, **kw):
        return 0.0


def mb_arm(monkeypatch):
    from cascade.control.rebot_rs_mb_arm import RebotRSMotorBridgeArm

    arm = RebotRSMotorBridgeArm(Cfg({
        "n_joints": 6, "joint_ids": [1, 2, 3, 4, 5, 6], "gripper_id": 7,
        "mit_kp": [1.0] * 6, "mit_kd": [0.1] * 6,
        "gripper": {"open_pos": 6.2, "closed_pos": 0.0},
    }))
    sends, reads = [], []
    arm._ctrl = object()
    arm._motors = {mid: _Motor(sends) for mid in range(1, 8)}
    arm.settle_timeout_s = 0.2

    def read_positions():
        reads.append(1)
        return MEASURED.copy()
    monkeypatch.setattr(arm, "_read_positions", read_positions)
    return arm, sends, reads


def test_mb_streaming_mask_uses_the_target_and_idle_reads(monkeypatch):
    import cascade.control.rebot_rs_mb_arm as mb

    now = [100.0]
    monkeypatch.setattr(mb.time, "monotonic", lambda: now[0])
    arm, _, reads = mb_arm(monkeypatch)
    target = np.array([0.0, 1.2, 1.2, 0.0, 0.0, 0.0])
    arm.send_joint_target(target)
    now[0] += 0.02
    np.testing.assert_allclose(arm.body_mask_q(), target)
    assert reads == []
    now[0] += 0.5
    np.testing.assert_allclose(arm.body_mask_q(), MEASURED)
    assert len(reads) == 1


def test_mb_no_bus_reads_while_the_mask_is_queried_during_a_real_stream(monkeypatch):
    arm, sends, reads = mb_arm(monkeypatch)
    in_stream, reads_in_stream, masks = threading.Event(), [], []
    send = arm.send_joint_target

    def tracked_send(q):
        in_stream.set()
        send(q)
    monkeypatch.setattr(arm, "send_joint_target", tracked_send)
    orig = arm._read_positions

    def tracked_read():
        if in_stream.is_set() and len(sends) < 6 * 25:
            reads_in_stream.append(1)
        return orig()
    monkeypatch.setattr(arm, "_read_positions", tracked_read)
    stop = threading.Event()

    def watcher():
        while not stop.is_set():
            if in_stream.is_set() and len(sends) < 6 * 25:
                masks.append(arm.body_mask_q())
    t = threading.Thread(target=watcher, daemon=True)
    t.start()
    try:
        arm.stream_to(MEASURED + 0.05, duration_s=0.5, rate_hz=50.0, settle_tol=1.0)
    finally:
        stop.set()
        t.join(2)
    assert masks, "the mask was never queried during the stream"
    assert reads_in_stream == []
