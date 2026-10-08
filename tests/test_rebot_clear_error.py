"""Both reBot RS transports clear latched RobStride faults before enabling.

Rig finding (WRC docs/HARDWARE_VERIFICATION_HANDOVER.md, Finding #1, ported
from Seeed's WRC fork `src/wrc_demo/control/__init__.py::make_arm`): RobStride
motors keep their fault bits (`fault_raw=0x00000004`, status_code=1) across
sessions, and the SDK's `enable_all()` does NOT clear them. A faulted motor
then silently refuses every MIT command -- motion "streams" while nothing
moves. `motor.clear_error()` is the fix.

In the RobStride private protocol `clear_error` is the type-4 *stop* frame
with the clear-fault flag, i.e. it also leaves the motor in reset (no torque).
So the clear must happen while the motors are still disabled -- after the run
mode is set (which on the motorbridge path proves reset state) and strictly
before enable -- or it would drop a holding arm.

Fully offline: the SDK (`reBotArm_control_py`) and motorbridge are faked in
`sys.modules`; nothing here opens a bus.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from cascade.config import Cfg


# ── a shared, ordered event log ─────────────────────────────────────────────


class _Log(list):
    def index_of(self, pred):
        for i, ev in enumerate(self):
            if pred(ev):
                return i
        raise AssertionError(f"no event matching predicate in {self!r}")


class _Motor:
    def __init__(self, name, log, clear_failures=()):
        self.name = name
        self.log = log
        self._failures = list(clear_failures)

    def clear_error(self):
        self.log.append(("clear_error", self.name))
        if self._failures:
            raise self._failures.pop(0)

    # motorbridge surface used by the mb backend
    def ensure_mode(self, mode, *a):
        self.log.append(("ensure_mode", self.name))

    def robstride_get_param_f32(self, param, *a):
        return 0.0

    def send_mit(self, *a, **kw):
        self.log.append(("send_mit", self.name))


# ── fake reBotArm_control_py SDK (rebot_rs backend) ─────────────────────────


class _Group:
    def __init__(self, names, mm, log, label):
        self._jn = list(names)
        self._mm = mm
        self._log = log
        self._label = label
        self._mit_kp = np.ones(len(names))
        self._mit_kd = np.ones(len(names))

    @property
    def joint_names(self):
        return list(self._jn)

    def mode_mit(self, kp=None, kd=None):
        self._log.append(("mode_mit", self._label))
        return True

    def enable(self):
        self._log.append(("enable", self._label))

    def send_mit(self, *a, **kw):
        self._log.append(("group_send_mit", self._label))


def _install_fake_sdk(monkeypatch, log, failures=None):
    failures = failures or {}
    joint_names = [f"joint{i}" for i in range(1, 7)]

    class FakeRebotArm:
        instances = []

        def __init__(self, hw_yaml=None):
            self.hw_yaml = hw_yaml
            self._motor_map = {}
            names = joint_names + ["gripper"]
            for n in names:
                self._motor_map[n] = _Motor(n, log, failures.get(n, ()))
            self.arm = _Group(joint_names, self._motor_map, log, "arm")
            self.gripper = _Group(["gripper"], self._motor_map, log, "gripper")
            self.has_gripper = True
            FakeRebotArm.instances.append(self)

        def connect(self):
            log.append(("connect", None))

        def start_control_loop(self, *a, **kw):  # the SDK's 500 Hz loop
            raise AssertionError("cascade's RS driver must not start the SDK control loop")

        def disconnect(self):
            log.append(("disconnect", None))

        def estop(self):
            log.append(("estop", None))

    pkg = types.ModuleType("reBotArm_control_py")
    pkg.__path__ = []  # mark as a package
    actuator = types.ModuleType("reBotArm_control_py.actuator")
    actuator.RebotArm = FakeRebotArm
    pkg.actuator = actuator
    monkeypatch.setitem(sys.modules, "reBotArm_control_py", pkg)
    monkeypatch.setitem(sys.modules, "reBotArm_control_py.actuator", actuator)
    return FakeRebotArm


def _rs_arm():
    from cascade.control.rebot_rs_arm import RebotRSArm

    # gravity comp OFF so connect() does not need pinocchio/the URDF here
    return RebotRSArm(Cfg({"hw_yaml": "rebotarm_rs.yaml", "n_joints": 6}))


@pytest.fixture
def no_sleep(monkeypatch):
    from cascade.control import robstride

    slept = []
    monkeypatch.setattr(robstride.time, "sleep", lambda s: slept.append(s))
    return slept


def test_rs_sdk_connect_clears_every_motor_before_enable(monkeypatch, no_sleep):
    log = _Log()
    fake = _install_fake_sdk(monkeypatch, log)
    arm = _rs_arm()
    arm.connect()

    motors = fake.instances[-1]._motor_map
    cleared = [name for ev, name in log if ev == "clear_error"]
    # every motor on the bus -- six joints AND the gripper -- exactly once
    assert sorted(cleared) == sorted(motors)
    first_enable = log.index_of(lambda ev: ev[0] == "enable")
    last_clear = max(i for i, ev in enumerate(log) if ev[0] == "clear_error")
    assert last_clear < first_enable, f"clear_error must precede enable: {log}"
    # and after the run mode is set (motors provably still in reset)
    last_mode = max(i for i, ev in enumerate(log) if ev[0] == "mode_mit")
    assert last_mode < log.index_of(lambda ev: ev[0] == "clear_error")


def test_rs_sdk_connect_retries_a_transient_ack_timeout(monkeypatch, no_sleep):
    """WRC: some RobStride firmwares ACK the clear late right after power-up or
    a bus-off; a comm_type=4 ack timeout is retried (bounded), then succeeds."""
    log = _Log()
    _install_fake_sdk(monkeypatch, log, failures={
        "joint2": [RuntimeError("control ack timeout: comm_type=4")],
    })
    arm = _rs_arm()
    arm.connect()
    assert [n for ev, n in log if ev == "clear_error"].count("joint2") == 2
    assert no_sleep == [pytest.approx(0.1)]
    assert any(ev[0] == "enable" for ev in log)


def test_rs_sdk_connect_fails_closed_when_a_fault_cannot_be_cleared(monkeypatch, no_sleep):
    """A clear that keeps failing is a hardware fault: connect raises a plain
    RuntimeError (taxonomy: crash/hardware) and NOTHING is enabled -- a motor
    that may still hold a latched fault would silently ignore every command."""
    log = _Log()
    _install_fake_sdk(monkeypatch, log, failures={
        "joint3": [RuntimeError("control ack timeout: comm_type=4")] * 3,
    })
    arm = _rs_arm()
    with pytest.raises(RuntimeError, match="joint3"):
        arm.connect()
    assert not any(ev[0] == "enable" for ev in log)
    assert arm._arm is None  # not left half-connected
    assert ("disconnect", None) in log  # the bus handle is released


def test_rs_sdk_connect_does_not_retry_a_non_timeout_error(monkeypatch, no_sleep):
    log = _Log()
    _install_fake_sdk(monkeypatch, log, failures={
        "gripper": [RuntimeError("motor is null")],
    })
    arm = _rs_arm()
    with pytest.raises(RuntimeError, match="gripper"):
        arm.connect()
    assert [n for ev, n in log if ev == "clear_error"].count("gripper") == 1
    assert no_sleep == []
    assert not any(ev[0] == "enable" for ev in log)


# ── fake motorbridge (rebot_rs_mb backend) ──────────────────────────────────


def _install_fake_motorbridge(monkeypatch, log, failures=None):
    failures = failures or {}

    class Mode:
        MIT = "mit"

    class Controller:
        def __init__(self, channel=None):
            self.channel = channel
            self.motors = {}

        def add_robstride_motor(self, motor_id, feedback_id, model):
            m = _Motor(motor_id, log, failures.get(motor_id, ()))
            self.motors[motor_id] = m
            return m

        def enable_all(self):
            log.append(("enable_all", None))

        def disable_all(self):
            log.append(("disable_all", None))

        def close_bus(self):
            log.append(("close_bus", None))

        def close(self):
            log.append(("close", None))

    mod = types.ModuleType("motorbridge")
    mod.Controller = Controller
    mod.Mode = Mode
    monkeypatch.setitem(sys.modules, "motorbridge", mod)
    return Controller


def _mb_arm():
    from cascade.control.rebot_rs_mb_arm import RebotRSMotorBridgeArm

    return RebotRSMotorBridgeArm(Cfg({
        "n_joints": 6, "joint_ids": [1, 2, 3, 4, 5, 6], "gripper_id": 7,
        "mit_kp": [1.0] * 6, "mit_kd": [0.1] * 6,
        "gripper": {"open_pos": 6.2, "closed_pos": 0.0},
    }))


def test_mb_connect_clears_every_motor_after_mode_and_before_enable(monkeypatch, no_sleep):
    log = _Log()
    _install_fake_motorbridge(monkeypatch, log)
    arm = _mb_arm()
    arm.connect()

    cleared = [mid for ev, mid in log if ev == "clear_error"]
    assert sorted(cleared) == [1, 2, 3, 4, 5, 6, 7]
    enable = log.index_of(lambda ev: ev[0] == "enable_all")
    clears = [i for i, ev in enumerate(log) if ev[0] == "clear_error"]
    modes = [i for i, ev in enumerate(log) if ev[0] == "ensure_mode"]
    # ensure_mode succeeding proves reset state, so the type-4 stop frame
    # cannot drop a holding arm; and the clear lands before torque arrives
    assert max(modes) < min(clears)
    assert max(clears) < enable


def test_mb_connect_fails_closed_and_releases_the_bus(monkeypatch, no_sleep):
    log = _Log()
    _install_fake_motorbridge(monkeypatch, log, failures={
        4: [RuntimeError("control ack timeout: comm_type=4")] * 3,
    })
    arm = _mb_arm()
    with pytest.raises(RuntimeError, match="4"):
        arm.connect()
    assert not any(ev[0] == "enable_all" for ev in log)
    # the existing failure path still closes the bus
    assert ("close_bus", None) in log and ("close", None) in log
    assert arm._ctrl is None
