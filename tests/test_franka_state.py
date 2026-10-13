"""Native Franka readiness stays distinct from streaming joint feedback."""

from types import SimpleNamespace

import pytest

from cascade.control.franka_state import FrankaStateMonitor


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class _RosErrors:
    """Generated ROS messages use private slots and public properties."""
    __slots__ = ("_cartesian_reflex", "_communication_constraints_violation")

    def __init__(self, cartesian_reflex=False, communication_constraints_violation=False):
        self._cartesian_reflex = cartesian_reflex
        self._communication_constraints_violation = communication_constraints_violation

    @classmethod
    def get_fields_and_field_types(cls):
        return {"cartesian_reflex": "boolean", "communication_constraints_violation": "boolean"}

    @property
    def cartesian_reflex(self):
        return self._cartesian_reflex

    @property
    def communication_constraints_violation(self):
        return self._communication_constraints_violation


def _state(mode=1, current=None, previous=None):
    return SimpleNamespace(robot_mode=mode,
                           current_errors=current if current is not None else _RosErrors(),
                           last_motion_errors=previous if previous is not None else _RosErrors())


def test_no_native_state_is_unavailable():
    monitor = FrankaStateMonitor()
    assert monitor.snapshot()["motion_ready"] is False
    assert monitor.snapshot()["state_age_s"] is None
    with pytest.raises(RuntimeError, match="hardware state unavailable"):
        monitor.ensure_ready()


@pytest.mark.parametrize("mode,name", [(1, "Idle"), (2, "Move")])
def test_idle_and_move_are_ready(mode, name):
    clock = _Clock()
    monitor = FrankaStateMonitor(clock=clock)
    monitor.update(_state(mode))
    clock.now += 0.5
    status = monitor.snapshot()
    assert status["motion_ready"] is True
    assert status["robot_mode"] == name
    assert status["robot_mode_code"] == mode
    assert status["state_age_s"] == 0.5
    monitor.ensure_ready()


def test_current_error_blocks_even_move_mode():
    monitor = FrankaStateMonitor()
    monitor.update(_state(2, current=_RosErrors(cartesian_reflex=True)))
    assert monitor.snapshot()["current_errors"] == ["cartesian_reflex"]
    assert monitor.snapshot()["motion_ready"] is False
    with pytest.raises(RuntimeError, match="current_errors: cartesian_reflex"):
        monitor.ensure_ready()


def test_reflex_reports_last_error_even_after_current_flags_clear():
    monitor = FrankaStateMonitor()
    monitor.update(_state(4, previous=_RosErrors(cartesian_reflex=True)))
    with pytest.raises(RuntimeError, match="mode Reflex.*last_motion_errors: cartesian_reflex"):
        monitor.ensure_ready()


def test_historical_error_does_not_block_recovered_idle():
    monitor = FrankaStateMonitor()
    monitor.update(_state(1, previous=_RosErrors(cartesian_reflex=True)))
    assert monitor.snapshot()["motion_ready"] is True
    assert monitor.snapshot()["last_motion_errors"] == ["cartesian_reflex"]
    monitor.ensure_ready()


def test_receipt_time_not_message_stamp_defines_freshness():
    clock = _Clock()
    monitor = FrankaStateMonitor(timeout_s=1.0, clock=clock)
    msg = _state()
    msg.header = SimpleNamespace(stamp=SimpleNamespace(sec=10**20))
    monitor.update(msg)
    clock.now += 1.01
    assert monitor.snapshot()["motion_ready"] is False
    with pytest.raises(RuntimeError, match="native robot state stale"):
        monitor.ensure_ready()
    monitor.update(msg)
    monitor.ensure_ready()


@pytest.mark.parametrize("mode,name", [(0, "Other"), (3, "Guiding"), (4, "Reflex"),
                                      (5, "UserStopped"), (6, "AutomaticErrorRecovery"),
                                      (99, "Unknown(99)")])
def test_non_motion_modes_refuse_motion(mode, name):
    monitor = FrankaStateMonitor()
    monitor.update(_state(mode))
    assert monitor.snapshot()["robot_mode"] == name
    assert monitor.snapshot()["motion_ready"] is False
    with pytest.raises(RuntimeError, match="does not permit motion"):
        monitor.ensure_ready()


def test_snapshot_and_input_message_are_not_mutable_monitor_storage():
    monitor = FrankaStateMonitor()
    msg = _state(4, current=_RosErrors(cartesian_reflex=True))
    monitor.update(msg)
    first = monitor.snapshot()
    first["current_errors"].clear()
    first["last_motion_errors"].append("invented")
    first["robot_mode"] = "Idle"
    msg.current_errors._cartesian_reflex = False
    msg.robot_mode = 1
    second = monitor.snapshot()
    assert second["current_errors"] == ["cartesian_reflex"]
    assert second["last_motion_errors"] == []
    assert second["robot_mode"] == "Reflex"
    assert second["motion_ready"] is False


def test_only_boolean_fields_are_errors():
    monitor = FrankaStateMonitor()
    monitor.update(_state(2, current=SimpleNamespace(cartesian_reflex=False, header="present",
                                                   ROBOT_MODE_MOVE=2,
                                                   communication_constraints_violation=True)))
    assert monitor.snapshot()["current_errors"] == ["communication_constraints_violation"]


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout_is_rejected(timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        FrankaStateMonitor(timeout_s=timeout)
