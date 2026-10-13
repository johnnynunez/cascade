"""Exiting the ROS client retains the controller's hold, never parks or closes."""
import signal

import pytest

from cascade.apps import demo
from cascade.config import Cfg
from cascade.control.lazy_arm import LazyArm
from cascade.control.ros2_arm import Ros2Arm
from test_shutdown_drive_state import hardware, assert_park_before_torque_off, runtime_for, safe


def native_arm(monkeypatch, *, lazy=False):
    raw = Ros2Arm(Cfg({"n_joints": 7, "ros_joints": [f"fr3_joint{i}" for i in range(1, 8)]}))
    # No rclpy, ROS node, controller connection, or robot I/O is created.
    raw.connected = True
    backend = raw
    if lazy:
        backend = LazyArm(lambda: pytest.fail("shutdown materialized a backend"), n_joints=7)
        backend._arm = raw
    arm = safe(backend)
    for name in ("move_joints", "set_gripper", "get_state"):
        monkeypatch.setattr(arm, name, lambda *a, **kw: pytest.fail("shutdown commanded/read the arm"))
    monkeypatch.setattr(raw, "close_gripper_torque", lambda: pytest.fail("shutdown closed jaws"), raising=False)
    return arm, raw


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("retained", [None, "held_object"])
def test_native_ros_shutdown_never_parks_even_with_empty_gripper(monkeypatch, lazy, retained):
    arm, raw = native_arm(monkeypatch, lazy=lazy)
    runtime = runtime_for(arm, retained=retained)
    receipt = demo.shutdown_runtime(runtime, arm.raw)
    assert receipt["complete"] and not raw.connected
    assert not raw._stopped  # idle transport disconnect alone never commands a hold


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_native_ros_cli_signal_stops_then_disconnects_without_parking(monkeypatch, tmp_path, signum):
    monkeypatch.delenv("CASCADE_START_STOPPED", raising=False)
    arm, raw = native_arm(monkeypatch, lazy=True)
    runtime = runtime_for(arm)
    events = []
    raw_stop = raw.stop

    def stop():
        events.append("stop")
        raw_stop()

    monkeypatch.setattr(raw, "stop", stop)
    monkeypatch.setattr(demo, "build_runtime", lambda *a, **kw: (runtime, arm.raw))
    monkeypatch.setattr(demo, "_run_demo", lambda *a: signal.raise_signal(signum))
    status = demo.main(["--arm", "fr3", "--camera", "mock", "--llm", "mock",
                        "--no-view", "--no-serve", "--run-dir", str(tmp_path)])
    assert status == (0 if signum == signal.SIGINT else 128 + signum)
    assert events == ["stop"]
    assert arm.harness.estopped and raw._stopped and not raw.connected


def test_mixed_ros_and_serial_rig_still_parks_before_serial_torque_off(monkeypatch):
    monkeypatch.setattr("cascade.control.arm_base.time.sleep", lambda _: None)
    events = []
    serial = hardware(events)
    native, raw = native_arm(monkeypatch, lazy=True)
    demo.shutdown_runtime(runtime_for(native, serial), native.raw)
    assert_park_before_torque_off(events)
    assert not raw.connected
