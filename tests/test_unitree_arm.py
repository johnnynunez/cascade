"""UnitreeArm (`type: unitree_arm`) against a stubbed rclpy + unitree_hg /
unitree_go message stack, plus the wire-format facts the backend depends on.

No ROS2 in CI, so -- like test_ros2_arm.py -- the whole surface is faked in
sys.modules and driven by hand. What is asserted is BEHAVIOUR on the wire:
which motor slot gets which q/kp/kd, where the arm-SDK weight rides, that
the weight ramps instead of stepping, that mode_machine is echoed from
LowState, and that the CRC matches a reference computed independently
from the C struct layout unitree_ros2 hashes.
"""

from __future__ import annotations

import struct
import sys
import threading
import time
import types
import zlib

import numpy as np
import pytest

from cascade.config import Cfg, load_demo_config, load_profile
from cascade.control.unitree_arm import crc32_core, hg_lowcmd_crc


# ── the rclpy / unitree message stub ─────────────────────────────────────

class _FakeMotorCmd:
    def __init__(self):
        self.mode = 0
        self.q = self.dq = self.tau = self.kp = self.kd = 0.0
        self.reserve = 0


class _FakeLowCmdHG:
    def __init__(self):
        self.mode_pr = 0
        self.mode_machine = 0
        self.motor_cmd = [_FakeMotorCmd() for _ in range(35)]
        self.reserve = [0, 0, 0, 0]
        self.crc = 0


class _FakeLowCmdGO:
    def __init__(self):
        self.motor_cmd = [_FakeMotorCmd() for _ in range(20)]
        self.crc = 0


class _FakeMotorState:
    def __init__(self, q=0.0):
        self.q = q
        self.dq = 0.0
        self.tau_est = 0.0


class _FakeLowState:
    def __init__(self, n=35, mode_machine=0):
        self.motor_state = [_FakeMotorState() for _ in range(n)]
        self.mode_machine = mode_machine


class _FakePublisher:
    def __init__(self, msg_type, topic):
        self.msg_type, self.topic = msg_type, topic
        self.published: list = []

    def publish(self, msg):
        self.published.append(msg)


class _FakeSubscription:
    def __init__(self, msg_type, topic, cb, qos):
        self.msg_type, self.topic, self.cb, self.qos = msg_type, topic, cb, qos


class _FakeNode:
    instances: list["_FakeNode"] = []

    def __init__(self, name):
        self.name = name
        self.subs: list[_FakeSubscription] = []
        self.pubs: list[_FakePublisher] = []
        self.destroyed = False
        _FakeNode.instances.append(self)

    def create_subscription(self, msg_type, topic, cb, qos):
        sub = _FakeSubscription(msg_type, topic, cb, qos)
        self.subs.append(sub)
        return sub

    def create_publisher(self, msg_type, topic, depth):
        pub = _FakePublisher(msg_type, topic)
        self.pubs.append(pub)
        return pub

    def destroy_node(self):
        self.destroyed = True


class _FakeExecutor:
    def __init__(self):
        self._shutdown = threading.Event()

    def add_node(self, node):
        pass

    def spin(self):
        self._shutdown.wait()

    def shutdown(self, timeout_sec=None):
        self._shutdown.set()


class _RclpyState:
    ok_flag = False


@pytest.fixture
def unitree_stub(monkeypatch):
    _FakeNode.instances = []
    _RclpyState.ok_flag = False
    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda: setattr(_RclpyState, "ok_flag", True)
    rclpy.ok = lambda: _RclpyState.ok_flag
    executors = types.ModuleType("rclpy.executors")
    executors.SingleThreadedExecutor = _FakeExecutor
    node_mod = types.ModuleType("rclpy.node")
    node_mod.Node = _FakeNode
    qos_mod = types.ModuleType("rclpy.qos")
    qos_mod.qos_profile_sensor_data = object()
    hg = types.ModuleType("unitree_hg")
    hg_msg = types.ModuleType("unitree_hg.msg")
    hg_msg.LowCmd, hg_msg.LowState, hg_msg.MotorCmd = _FakeLowCmdHG, _FakeLowState, _FakeMotorCmd
    go = types.ModuleType("unitree_go")
    go_msg = types.ModuleType("unitree_go.msg")
    go_msg.LowCmd, go_msg.LowState, go_msg.MotorCmd = _FakeLowCmdGO, _FakeLowState, _FakeMotorCmd
    for name, mod in {
        "rclpy": rclpy, "rclpy.executors": executors, "rclpy.node": node_mod,
        "rclpy.qos": qos_mod, "unitree_hg": hg, "unitree_hg.msg": hg_msg,
        "unitree_go": go, "unitree_go.msg": go_msg,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    yield


def _mk_arm(**over):
    from cascade.control.arm_base import make_arm

    data = {
        "type": "unitree_arm", "name": "g1_right", "n_joints": 7,
        "unitree_msg_family": "hg",
        "unitree_joint_index": [22, 23, 24, 25, 26, 27, 28],
        "unitree_weight_index": 29,
        "unitree_kp": [60] * 7, "unitree_kd": [1.5] * 7,
        "unitree_weight_ramp_s": 0.1,     # fast ramp for tests (5 steps)
        "wait_state_s": 2.0,
        "gripper": {"open_pos": 0.0, "closed_pos": 0.0, "max_width_m": 0.0},
    }
    data.update(over)
    return make_arm(Cfg(data))


def _feed(node, q_by_index: dict[int, float], n=35, mode_machine=5):
    msg = _FakeLowState(n=n, mode_machine=mode_machine)
    for i, v in q_by_index.items():
        msg.motor_state[i].q = float(v)
    for sub in node.subs:
        sub.cb(msg)


def _connect(arm, q_by_index=None, n=35):
    def feeder():
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not _FakeNode.instances:
            time.sleep(0.01)
        for _ in range(100):
            if _FakeNode.instances:
                _feed(_FakeNode.instances[-1], q_by_index or {}, n=n)
                if getattr(arm, "connected", False):
                    return
            time.sleep(0.02)

    t = threading.Thread(target=feeder, daemon=True)
    t.start()
    arm.connect()
    t.join(timeout=1.0)


def _cmd_pub(node):
    return next(p for p in node.pubs if p.topic == "rt/arm_sdk")


# ── factory + profile validation ─────────────────────────────────────────

def test_make_arm_dispatches_unitree(unitree_stub):
    from cascade.control.unitree_arm import UnitreeArm

    assert isinstance(_mk_arm(), UnitreeArm)


def test_joint_index_must_match_dof_and_range(unitree_stub):
    with pytest.raises(ValueError, match="unitree_joint_index"):
        _mk_arm(unitree_joint_index=[22, 23])
    with pytest.raises(ValueError, match="out of range"):
        _mk_arm(unitree_joint_index=[22, 23, 24, 25, 26, 27, 40])
    with pytest.raises(ValueError, match="hg|go"):
        _mk_arm(unitree_msg_family="dds")


# ── wire behaviour ───────────────────────────────────────────────────────

def test_state_reads_the_indexed_motors_in_profile_order(unitree_stub):
    arm = _mk_arm()
    _connect(arm, {22: 0.1, 23: 0.2, 24: 0.3, 25: 0.4, 26: 0.5, 27: 0.6, 28: 0.7, 0: 9.0})
    try:
        st = arm.get_state()
        np.testing.assert_allclose(st.q, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        # no hand on the wire: the skill layer must not read "open"
        assert st.gripper_valid is False
    finally:
        arm.disconnect()


def test_connect_ramps_the_arm_sdk_weight_while_holding_current_pose(unitree_stub):
    """The weight slot must go 0 -> 1 in STEPS (the locomotion controller
    blends on it; a step is a shoulder jerk), and every ramp message must
    command the pose the arm is already in, not zeros."""
    arm = _mk_arm()
    _connect(arm, {22: 0.5, 25: -0.4})
    try:
        pub = _cmd_pub(_FakeNode.instances[-1])
        weights = [m.motor_cmd[29].q for m in pub.published]
        assert len(weights) >= 3
        assert weights == sorted(weights) and weights[0] < 0.5 and weights[-1] == pytest.approx(1.0)
        for m in pub.published:
            assert m.motor_cmd[22].q == pytest.approx(0.5)
            assert m.motor_cmd[25].q == pytest.approx(-0.4)
            assert m.motor_cmd[22].kp == pytest.approx(60.0)
            assert m.motor_cmd[22].kd == pytest.approx(1.5)
    finally:
        arm.disconnect()


def test_send_joint_target_lands_on_the_indexed_slots_only(unitree_stub):
    arm = _mk_arm()
    _connect(arm)
    try:
        pub = _cmd_pub(_FakeNode.instances[-1])
        n0 = len(pub.published)
        arm.send_joint_target(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))
        msg = pub.published[-1]
        assert len(pub.published) == n0 + 1
        for j, mi in enumerate([22, 23, 24, 25, 26, 27, 28]):
            assert msg.motor_cmd[mi].q == pytest.approx(0.1 * (j + 1))
            assert msg.motor_cmd[mi].mode == 1
        # untouched motors (legs, waist, left arm) carry NO gains: a zero-kp
        # command is "no torque from the SDK", the locomotion controller keeps them
        for mi in list(range(0, 22)) + list(range(30, 35)):
            assert msg.motor_cmd[mi].kp == 0.0 and msg.motor_cmd[mi].q == 0.0
        assert msg.motor_cmd[29].q == pytest.approx(1.0)      # weight stays up
        assert msg.mode_machine == 5                          # echoed from LowState
        assert msg.mode_pr == 0
    finally:
        arm.disconnect()


def test_hg_crc_is_recomputed_per_message_and_matches_reference(unitree_stub):
    """The firmware drops any LowCmd whose crc does not match. Check the
    published crc against a from-scratch computation over the struct."""
    arm = _mk_arm()
    _connect(arm)
    try:
        pub = _cmd_pub(_FakeNode.instances[-1])
        arm.send_joint_target(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))
        msg = pub.published[-1]
        motors = [{"mode": m.mode, "q": m.q, "dq": m.dq, "tau": m.tau, "kp": m.kp, "kd": m.kd}
                  for m in msg.motor_cmd]
        assert msg.crc == hg_lowcmd_crc(msg.mode_pr, msg.mode_machine, motors)
        assert msg.crc != 0
        # and a different command hashes differently
        arm.send_joint_target(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]))
        assert pub.published[-1].crc != msg.crc
    finally:
        arm.disconnect()


def test_stop_holds_last_target_and_keeps_weight(unitree_stub):
    """stop() must NOT drop the weight (that hands the arm to the
    locomotion controller mid-motion = a large motion); it re-asserts the
    last commanded pose and suppresses waypoints until resume()."""
    arm = _mk_arm()
    _connect(arm)
    try:
        pub = _cmd_pub(_FakeNode.instances[-1])
        arm.send_joint_target(np.array([0.3, 0, 0, 0, 0, 0, 0]))
        arm.stop()
        msg = pub.published[-1]
        assert msg.motor_cmd[22].q == pytest.approx(0.3)
        assert msg.motor_cmd[22].kp == pytest.approx(60.0)
        assert msg.motor_cmd[29].q == pytest.approx(1.0)
        n = len(pub.published)
        arm.send_joint_target(np.zeros(7))
        assert len(pub.published) == n
        arm.resume()
        arm.send_joint_target(np.zeros(7))
        assert len(pub.published) == n + 1
    finally:
        arm.disconnect()


def test_disconnect_ramps_weight_down_then_stops_spinning(unitree_stub):
    arm = _mk_arm()
    _connect(arm)
    pub = _cmd_pub(_FakeNode.instances[-1])
    spin = arm._spin_thread
    arm.disconnect()
    tail = [m.motor_cmd[29].q for m in pub.published[-3:]]
    assert tail == sorted(tail, reverse=True) and tail[-1] == pytest.approx(0.0)
    assert not spin.is_alive()
    assert _FakeNode.instances[-1].destroyed


def test_connect_times_out_without_lowstate(unitree_stub):
    arm = _mk_arm(wait_state_s=0.3)
    with pytest.raises(RuntimeError, match="no LowState"):
        arm.connect()
    assert not arm.connected


def test_go_family_uses_20_slot_message(unitree_stub):
    """Gen-1 H1 speaks unitree_go: 20 motor slots, weight in slot 9, no
    mode bytes / crc fields to fill."""
    arm = _mk_arm(unitree_msg_family="go", n_joints=4,
                  unitree_joint_index=[12, 13, 14, 15], unitree_weight_index=9,
                  unitree_kp=[60] * 4, unitree_kd=[1.5] * 4)
    _connect(arm, {12: 0.2}, n=20)
    try:
        pub = _cmd_pub(_FakeNode.instances[-1])
        msg = pub.published[-1]
        assert isinstance(msg, _FakeLowCmdGO) and len(msg.motor_cmd) == 20
        assert msg.motor_cmd[12].q == pytest.approx(0.2)
        assert msg.motor_cmd[9].q == pytest.approx(1.0)
    finally:
        arm.disconnect()


# ── CRC reference ────────────────────────────────────────────────────────

def _crc32_mpeg2_words(words: list[int]) -> int:
    """Independent reference: Unitree's crc32_core is CRC-32/MPEG-2 over the
    big-endian bytes of each 32-bit word (poly 0x04C11DB7, init 0xFFFFFFFF,
    no reflection, no final xor). Implemented table-free from the definition."""
    crc = 0xFFFFFFFF
    for w in words:
        for b in struct.pack(">I", w):
            crc ^= b << 24
            for _ in range(8):
                crc = ((crc << 1) ^ 0x04C11DB7) & 0xFFFFFFFF if crc & 0x80000000 else (crc << 1) & 0xFFFFFFFF
    return crc


def test_crc32_core_matches_mpeg2_reference():
    rng = np.random.default_rng(1)
    for n in (1, 3, 8, 250):
        words = [int(x) for x in rng.integers(0, 2**32, size=n, dtype=np.uint64)]
        assert crc32_core(words) == _crc32_mpeg2_words(words)


def test_hg_lowcmd_struct_is_hashed_at_c_layout_size():
    """motor_crc_hg.cpp hashes sizeof(LowCmd)/4 - 1 words: 2 mode bytes (+2
    pad), 35 x 28-byte MotorCmd, 4 reserve words. 4 + 980 + 16 = 1000 bytes
    = 250 words. A layout slip (e.g. packing `mode` as 1 byte) would hash a
    different number of words and every command would be silently dropped."""
    motors = [{"mode": 1, "q": 0.5, "dq": 0.0, "tau": 0.0, "kp": 60.0, "kd": 1.5}] * 35
    buf = bytearray(struct.pack("<BBxx", 0, 5))
    for m in motors:
        buf += struct.pack("<Bxxxfffff I", m["mode"], m["q"], m["dq"], m["tau"], m["kp"], m["kd"], 0)
    buf += struct.pack("<4I", 0, 0, 0, 0)
    assert len(buf) == 1000
    words = list(struct.unpack("<250I", bytes(buf)))
    assert hg_lowcmd_crc(0, 5, motors) == crc32_core(words)


# ── profile completeness ─────────────────────────────────────────────────

@pytest.mark.parametrize("name, n, idx, weight", [
    ("g1", 7, [22, 23, 24, 25, 26, 27, 28], 29),
    ("h1_2_sdk", 7, [20, 21, 22, 23, 24, 25, 26], 27),
    ("h1_sdk", 4, [12, 13, 14, 15], 9),
])
def test_unitree_profiles_pin_the_sdk_motor_indices(name, n, idx, weight):
    prof = load_profile("arms", name).as_dict()
    assert prof["type"] == "unitree_arm"
    assert prof["n_joints"] == n and len(prof["home_q"]) == n
    assert prof["unitree_joint_index"] == idx
    assert prof["unitree_weight_index"] == weight
    assert len(prof["unitree_kp"]) == n and len(prof["unitree_kd"]) == n
    assert prof["gripper"]["max_width_m"] == 0.0     # handless: no pinch grasps


def test_g1_home_is_a_top_down_pose_on_the_vendored_urdf():
    """The profile numbers are DERIVED from the URDF; pin the claim."""
    from cascade.control.kinematics import Kinematics

    cfg = load_demo_config(camera="mock", arm="g1_mock", llm="mock")
    kin = Kinematics(cfg.arm.model, ee_frame=cfg.arm.ee_frame, n_controlled=7)
    T = kin.fk(np.asarray(cfg.arm.home_q, dtype=float))
    np.testing.assert_allclose(T[:3, 3], [0.128, -0.161, -0.121], atol=2e-3)
    # down_open: column 0 is the approach axis and it points DOWN (-z torso)
    assert cfg.arm.tool_axis_order == "down_open"
    np.testing.assert_allclose(T[:3, 0], [0, 0, -1], atol=0.02)
    lo, hi = kin.joint_limits
    q = np.asarray(cfg.arm.home_q)
    assert np.min(np.minimum(q - lo, hi - q)) > 0.5
