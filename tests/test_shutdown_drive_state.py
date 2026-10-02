"""Retained simulation loads must not change hardware park/torque-off ordering."""
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.apps.demo import shutdown_runtime
from cascade.config import Cfg
from cascade.control.arm_rig import ArmRig
from cascade.control.feetech import ADDR_GOAL_POSITION, ADDR_PRESENT_POSITION, ADDR_TORQUE_ENABLE
from cascade.control.feetech_arm import FeetechArm
from cascade.control.isaac_arm import IsaacArm
from cascade.control.lazy_arm import LazyArm
from cascade.safety.harness import SafeArm, SafetyHarness
from test_safety import limits


RETAINED = ("held_object", "_held_provisional", "_contact_episode",
            "_carry_attachment", "_release_episode")


def runtime_for(*arms, retained=None):
    runtime = SimpleNamespace(arm=arms[0],
        arm_rig=ArmRig(list(arms), [str(i) for i in range(len(arms))]),
        watcher=None, camera=SimpleNamespace(close=lambda: None))
    if retained is not None:
        setattr(runtime, retained, object())
    return runtime


def safe(raw):
    harness = SafetyHarness(limits())
    harness.park_q = np.zeros(raw.n_joints)
    return SafeArm(raw, harness)


def hardware(events):
    """Actual Feetech streaming/disconnect, with register I/O confined to memory."""
    arm = FeetechArm(Cfg({"n_joints": 5, "servo_ids": [1, 2, 3, 4, 5], "gripper_id": 6}))
    positions = {sid: 2113 for sid in range(1, 7)}

    class Bus:
        def read_reg(self, sid, address):
            assert address == ADDR_PRESENT_POSITION
            return positions[sid]

        def sync_write_reg(self, address, values):
            assert address == ADDR_GOAL_POSITION
            events.append(("joint_target", dict(values)))
            positions.update(values)

        def write_reg(self, sid, address, value):
            assert address == ADDR_TORQUE_ENABLE and value == 0
            events.append(("torque_off", sid))

    arm._bus = Bus()
    arm._port = SimpleNamespace(close=lambda: events.append(("port_closed",)))
    arm._connected = True
    return safe(arm)


def simulation(events, monkeypatch, lazy=False):
    raw = IsaacArm(Cfg({"n_joints": 6}))
    monkeypatch.setattr(raw._client, "close", lambda: events.append(("isaac_disconnected",)))
    if lazy:
        wrapper = LazyArm(lambda: pytest.fail("shutdown must not construct a backend"))
        wrapper._arm = raw
        raw = wrapper
    arm = safe(raw)
    monkeypatch.setattr(arm, "move_joints", lambda *a, **kw: pytest.fail("loaded simulator parked"))
    monkeypatch.setattr(arm, "set_gripper", lambda *a, **kw: pytest.fail("loaded jaws changed"))
    return arm


def assert_park_before_torque_off(events):
    targets = [i for i, event in enumerate(events) if event[0] == "joint_target"]
    disabled = [i for i, event in enumerate(events) if event[0] == "torque_off"]
    assert len(targets) == 100  # Real two-second, 50 Hz park stream.
    assert events[targets[-1]][1] == {sid: 2048 for sid in range(1, 6)}
    assert len(disabled) == 6 and max(targets) < min(disabled)
    assert {events[i][1] for i in disabled} == set(range(1, 7))
    assert events[-1] == ("port_closed",)


@pytest.mark.parametrize("retained", RETAINED)
def test_retained_hardware_load_keeps_park_before_actual_torque_off(monkeypatch, retained):
    monkeypatch.setattr("cascade.control.arm_base.time.sleep", lambda _seconds: None)
    events = []
    arm = hardware(events)
    assert arm.raw.disconnect_preserves_drive_state is False
    shutdown_runtime(runtime_for(arm, retained=retained), arm.raw)
    assert_park_before_torque_off(events)
    assert not arm.raw._connected


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("retained", (*RETAINED, "_pending_contact_episode", "_pending_release_episode"))
def test_loaded_isaac_disconnects_without_joint_or_gripper_commands(monkeypatch, lazy, retained):
    events = []
    arm = simulation(events, monkeypatch, lazy=lazy)
    runtime = runtime_for(arm)
    setattr(arm.harness if retained.startswith("_pending_") else runtime, retained, object())
    assert arm.raw.disconnect_preserves_drive_state is True
    shutdown_runtime(runtime, arm.raw)
    assert events == [("isaac_disconnected",)]


def test_standby_lazy_capability_and_shutdown_never_connect():
    raw = LazyArm(lambda: pytest.fail("standby backend materialized"), profile_type="isaac")
    assert raw.disconnect_preserves_drive_state is False
    arm = safe(raw)
    shutdown_runtime(runtime_for(arm, retained="held_object"), raw)
    assert not raw.connected


@pytest.mark.parametrize("hardware_first", [False, True])
def test_mixed_rig_retained_load_skips_only_transport_only_backend(monkeypatch, hardware_first):
    monkeypatch.setattr("cascade.control.arm_base.time.sleep", lambda _seconds: None)
    hardware_events, sim_events = [], []
    real = hardware(hardware_events)
    sim = simulation(sim_events, monkeypatch, lazy=True)
    arms = (real, sim) if hardware_first else (sim, real)
    shutdown_runtime(runtime_for(*arms, retained="_held_provisional"), arms[0].raw)
    assert_park_before_torque_off(hardware_events)
    assert sim_events == [("isaac_disconnected",)]
