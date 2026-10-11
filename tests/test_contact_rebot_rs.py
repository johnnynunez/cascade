"""The contact monitor on the reBot RS drivers (both transports), offline.

Fake SDK motors over a PD plant on a fake clock -- no CAN bus, no
motorbridge, no real arm object. What the RS backends add:

* `contact_joint_reader`: ONE motor's mechPos param read per call, with the
  monitor's short timeout (a full pose polls six motors, ~48 ms with the
  driver lock held; the 50 Hz stream only has slack for one or two). A
  reading that cannot be a position raises -- it is never handed on.
* `read_positions_for_monitor`: a fresh six-motor read with the drivers'
  reading guards that RAISES on failure (get_state would serve the last
  known pose, which may be the stream's start: holding that would snap the
  arm back).
* after a trip the MIT hold is the measured pose, and a later soft stop()
  re-asserts that hold -- never the command the arm was pressing toward.
"""

from __future__ import annotations

import threading
import types

import numpy as np
import pytest
from test_contact_stream import INERTIA, KD, KP

from cascade.config import Cfg
from cascade.control import arm_base
from cascade.control import rebot_rs_arm as rs
from cascade.control import rebot_rs_mb_arm as mb
from cascade.control.rebot_rs_arm import MECH_POS, RebotRSArm
from cascade.control.rebot_rs_mb_arm import RebotRSMotorBridgeArm
from cascade.control.robstride import UnsafeReading
from cascade.safety.contact import ContactMonitor, ContactMonitorConfig
from cascade.types import ContactDetected, SafetyViolation

HOME = np.array([0.0, 1.2, 1.2, 0.0, 0.0, 0.0])
GARBAGE = 2.3354605539310961e18          # the 2026-10-10 mechPos read


class Plant:
    """Six PD joints (RS MIT gains, no velocity feedforward) on a fake clock."""

    def __init__(self, q0=HOME):
        self.t = 500.0
        self.q = np.asarray(q0, float).copy()
        self.dq = np.zeros(6)
        self.target = self.q.copy()
        self.walls: dict[int, tuple[float, float]] = {}
        self.garbage: set[int] = set()
        self.reads: list[tuple[int, object]] = []   # (joint, timeout arg)

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.advance(max(0.0, float(s)))

    def advance(self, s):
        steps = max(1, round(s / 0.0005))
        h = s / steps
        for _ in range(steps):
            self.dq += h * (KP * (self.target - self.q) - KD * self.dq) / INERTIA
            self.q += h * self.dq
            for j, (wall, start) in self.walls.items():
                if (wall - start) * (self.q[j] - wall) > 0:
                    self.q[j], self.dq[j] = wall, 0.0
        self.t += s

    def block(self, j, at):
        self.walls[j] = (float(at), float(self.q[j]))


class Motor:
    def __init__(self, plant, j, sign=1.0):
        self.plant, self.j, self.sign = plant, j, sign

    def robstride_get_param_f32(self, param, *timeout):
        assert param == MECH_POS
        self.plant.reads.append((self.j, timeout[0] if timeout else None))
        self.plant.advance(0.008)                      # measured: 8.07 ms per read
        if self.j in self.plant.garbage:
            return GARBAGE
        return float(self.plant.q[self.j]) * self.sign


@pytest.fixture
def plant(monkeypatch):
    p = Plant()
    clock = types.SimpleNamespace(monotonic=p.monotonic, sleep=p.sleep)
    for module in (arm_base, rs, mb):
        monkeypatch.setattr(module, "time", clock)
    return p


def sdk_arm(plant):
    """RebotRSArm on a fake reBotArm_control_py: MIT sends drive the plant."""
    arm = RebotRSArm(Cfg({"n_joints": 6}))
    names = [f"joint{i}" for i in range(1, 7)]
    sends = []

    def send_mit(q, **kw):
        plant.advance(0.001)
        plant.target = np.asarray(q, float).copy()
        sends.append(plant.target.copy())
    group = types.SimpleNamespace(send_mit=send_mit, joint_names=names,
                                  _mm={n: Motor(plant, i) for i, n in enumerate(names)})
    arm._arm = types.SimpleNamespace(arm=group, has_gripper=False)
    arm._stopped = False
    arm._read_failures = 0
    arm._last_q, arm._last_q_t = plant.q.copy(), plant.t
    arm._last_cmd_q = arm._last_cmd_t = None
    arm._mit_kp = arm._mit_kd = None
    arm.settle_timeout_s = 2.0
    return arm, sends


def mb_arm(plant, signs=(-1, 1, -1, 1, 1, -1)):
    arm = RebotRSMotorBridgeArm(Cfg({"n_joints": 6, "joint_ids": [1, 2, 3, 4, 5, 6],
                                     "gripper_id": None, "wire_signs": list(signs),
                                     "mit_kp": KP.tolist(), "mit_kd": KD.tolist()}))
    sends = []
    arm._ctrl = object()
    arm._motors = {i + 1: Motor(plant, i, sign=signs[i]) for i in range(6)}

    def send_mit(q):
        plant.advance(0.001)
        plant.target = np.asarray(q, float).copy()
        sends.append(plant.target.copy())
    arm._send_mit = send_mit
    arm._last_q, arm._last_q_t = plant.q.copy(), plant.t
    return arm, sends


def monitor(**kw):
    base = {"enabled": True}
    base.update(kw)
    return ContactMonitor(ContactMonitorConfig.from_config(base), label="test")


# ── RebotRSArm ───────────────────────────────────────────────────────────


def test_rs_mid_stream_reads_are_single_motors_with_a_short_timeout(plant):
    arm, _ = sdk_arm(plant)
    m = monitor(read_timeout_s=0.05)
    assert arm.stream_to(HOME + 0.2, 1.0, contact_monitor=m)
    stream = [r for r in plant.reads if r[1] is not None]
    assert stream and {t for _, t in stream} == {50}
    s = m.summary()
    assert 1.0 <= s["reads"]["per_cycle"] <= 2.0 and m.tripped is None
    assert s["reads"]["cost_ms_mean"] == pytest.approx(8.0, abs=0.5)


def test_rs_blocked_joint_trips_holds_measured_and_stop_keeps_that_hold(plant):
    arm, sends = sdk_arm(plant)
    target = HOME.copy()
    target[3] = 0.9
    plant.block(3, 0.4)                       # the wrist meets clutter
    m = monitor()
    with pytest.raises(ContactDetected, match="contact detected at joint 4"):
        arm.stream_to(target, 2.1, contact_monitor=m)
    hold = sends[-1]
    assert hold[3] == pytest.approx(0.4, abs=0.01)
    np.testing.assert_allclose(arm._last_cmd_q, hold)
    pressing = max(s[3] for s in sends[:-1])
    assert pressing > 0.45
    arm.stop()                                # soft stop re-asserts the hold
    np.testing.assert_allclose(sends[-1], hold)
    assert sends[-1][3] < pressing


def test_rs_implausible_mid_stream_reading_is_never_used(plant):
    arm, sends = sdk_arm(plant)
    plant.garbage.add(2)
    m = monitor()
    with pytest.raises(SafetyViolation, match="lost joint 3 feedback") as exc:
        arm.stream_to(HOME + 0.3, 2.0, contact_monitor=m)
    assert not isinstance(exc.value, ContactDetected)
    assert all(np.all(np.abs(s) < 10) for s in sends)
    assert "implausible" in (m.last_read_failure or "")
    with pytest.raises(UnsafeReading, match="joint 3"):   # the driver itself refuses it
        arm.contact_joint_reader()(2, 0.05)


def test_rs_fresh_read_raises_instead_of_serving_a_stale_pose(plant):
    arm, _ = sdk_arm(plant)
    q = arm.read_positions_for_monitor()
    np.testing.assert_allclose(q, plant.q, atol=0.02)
    plant.garbage.add(0)
    with pytest.raises(Exception, match="implausible"):
        arm.read_positions_for_monitor()
    # get_state still serves the last good pose on one transient failure
    np.testing.assert_allclose(arm.get_state().q, q, atol=0.02)


def test_rs_reads_keep_the_driver_lock(plant):
    arm, _ = sdk_arm(plant)
    held = []
    lock = arm._lock

    class Spy:
        def __enter__(self):
            held.append(True)
            return lock.__enter__()

        def __exit__(self, *a):
            return lock.__exit__(*a)
    arm._lock = Spy()
    arm.contact_joint_reader()(1, 0.05)
    assert held == [True]
    assert isinstance(lock, type(threading.Lock()))


# ── RebotRSMotorBridgeArm ────────────────────────────────────────────────


def test_mb_reader_applies_the_wire_signs_and_timeout(plant):
    arm, _ = mb_arm(plant)
    plant.q[:] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    plant.target = plant.q.copy()                 # at rest there
    arm._last_q, arm._last_q_t = plant.q.copy(), plant.t
    read = arm.contact_joint_reader()
    assert [round(read(j, 0.03), 6) for j in range(6)] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert {t for _, t in plant.reads[-6:]} == {30}
    np.testing.assert_allclose(arm.read_positions_for_monitor(), plant.q)


def test_mb_blocked_joint_trips_and_holds_measured(plant):
    arm, sends = mb_arm(plant)
    target = HOME.copy()
    target[0] = 0.8
    plant.block(0, 0.3)
    with pytest.raises(ContactDetected, match="joint 1"):
        arm.stream_to(target, 2.0, contact_monitor=monitor())
    assert sends[-1][0] == pytest.approx(0.3, abs=0.01)
    arm.stop()
    assert sends[-1][0] == pytest.approx(0.3, abs=0.01)
