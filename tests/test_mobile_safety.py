"""Fail-closed CPU safety tests. Kinematic fixtures NEVER prove locomotion."""
import importlib.util
from dataclasses import replace
import math
import threading
import time

import pytest

from cascade.control.mobile_base import BaseState, VelocityCommand
from cascade.control.mock_base import MockMobileBase


def safety_module():
    assert importlib.util.find_spec("cascade.safety.base_harness"), "mobile safety missing"
    return __import__("cascade.safety.base_harness", fromlist=["SafeBase"])


def limits(**changes):
    result = dict(max_vx=.2, max_vy=.1, max_wz=1., max_duration_s=2.,
                  max_state_age_s=.1, max_no_progress_s=.1, max_wall_duration_s=2.,
                  poll_interval_s=.002, turn_speed_rad_s=.8, turn_tolerance_rad=.01,
                  max_turn_angle_rad=math.pi)
    result.update(changes)
    return result


@pytest.mark.parametrize("key,value", [("max_vx", 0), ("max_vy", -1),
    ("max_duration_s", float("inf")), ("max_wall_duration_s", True),
    ("turn_speed_rad_s", 2), ("poll_interval_s", 3), ("max_turn_angle_rad", 4)])
def test_limits_fail_before_any_backend_side_effect(key, value):
    config = limits()
    config[key] = value
    with pytest.raises(ValueError):
        safety_module().BaseSafetyHarness(config)


def test_limits_must_be_explicit_and_commands_are_never_clipped():
    mod = safety_module()
    with pytest.raises(ValueError):
        mod.BaseSafetyHarness({})
    config = limits()
    config["max_vy"] = 0
    harness = mod.BaseSafetyHarness(config)
    for command in [VelocityCommand(.3, 0, 0, 1), VelocityCommand(0, .01, 0, 1),
                    VelocityCommand(0, 0, 2, 1), VelocityCommand(0, 0, 0, 3)]:
        with pytest.raises(ValueError):
            harness.validate_command(command)
    harness.validate_command(VelocityCommand(.2, 0, -1, 2))


def test_walk_uses_advancing_snapshots_but_mock_cannot_confirm_physics():
    mod = safety_module()
    raw = MockMobileBase(wall_lease_s=2., dt_s=.002)
    safe = mod.SafeBase(raw, limits())
    safe.connect()
    try:
        result = safe.walk_velocity(.1, 0, 0, .04)
        assert result["execution_ok"], result
        assert result["ok"] is False and result["outcome"] == "unverified"
        measured = result["measured"]
        before = BaseState.from_dict(measured["before"])
        after = BaseState.from_dict(measured["after"])
        assert after.position_world[0] > before.position_world[0]
        assert after.step > before.step and after.sim_time_s > before.sim_time_s
        assert all(s["measurement_kind"] == "kinematic_mock" for s in measured["samples"])
        assert raw.get_state().linear_velocity_world == (0., 0., 0.)
        assert raw.get_state().controller_status == "ready"
        assert not raw.get_state().latched
    finally:
        safe.disconnect()


def test_stationary_clock_never_authorizes_a_command():
    mod = safety_module()

    class Counted(MockMobileBase):
        commands = 0

        def command_velocity(self, command, *, generation):
            self.commands += 1
            return super().command_velocity(command, generation=generation)

    raw = Counted(wall_lease_s=2., auto_step=False)
    safe = mod.SafeBase(raw, limits())
    safe.connect()
    try:
        result = safe.walk_velocity(.1, 0, 0, .03)
        assert not result["execution_ok"]
        assert "advanc" in result["error"] or "stale" in result["error"]
        assert raw.commands == 0
        assert raw.get_state().latched
    finally:
        safe.disconnect()


def test_stop_before_connect_survives_connect_normal_stop_and_explicit_reset():
    mod = safety_module()
    raw = MockMobileBase(wall_lease_s=2., dt_s=.002)
    safe = mod.SafeBase(raw, limits())
    assert safe.stop()["ok"]
    safe.connect()
    try:
        assert not safe.walk_velocity(.1, 0, 0, .03)["execution_ok"]
        assert safe.stop(latch=False)["latched"]
        assert safe.reset_stop()["ok"]
        stopped = raw.get_state().position_world
        time.sleep(.01)
        assert raw.get_state().position_world == stopped
        assert safe.walk_velocity(.1, 0, 0, .03)["execution_ok"]
    finally:
        safe.disconnect()


@pytest.mark.parametrize("angle", [.04, -.04])
def test_turn_closes_on_measured_yaw_across_wrap_without_physics_claim(angle):
    mod = safety_module()
    raw = MockMobileBase(wall_lease_s=2., dt_s=.002)
    # Initial condition belongs to the toy fixture, never the physical backend.
    raw._yaw = math.copysign(math.pi - .01, angle)
    safe = mod.SafeBase(raw, limits())
    safe.connect()
    try:
        result = safe.turn(angle)
        assert result["execution_ok"], result
        assert abs(result["measured_angle_rad"] - angle) <= .01
        assert not result["ok"] and result["outcome"] == "unverified"
        assert raw.get_state().angular_velocity_body == (0., 0., 0.)
    finally:
        safe.disconnect()


def test_turn_does_not_integrate_command_to_invent_rotation():
    mod = safety_module()

    class InertYaw(MockMobileBase):
        def get_state(self):
            return replace(super().get_state(), orientation_wxyz=(1., 0., 0., 0.),
                           angular_velocity_body=(0., 0., 0.))

    raw = InertYaw(wall_lease_s=2., dt_s=.002)
    safe = mod.SafeBase(raw, limits(max_duration_s=.04))
    safe.connect()
    try:
        result = safe.turn(.2)
        assert not result["execution_ok"] and not result["ok"]
        assert "yaw" in result["error"]
        assert raw.get_state().latched
    finally:
        safe.disconnect()


@pytest.mark.parametrize("angle", [True, "1", float("nan"), float("inf"), 4.])
def test_turn_invalid_input_never_connects_or_sends(angle):
    mod = safety_module()
    raw = MockMobileBase(wall_lease_s=1.)
    safe = mod.SafeBase(raw, limits())
    assert not safe.turn(angle)["execution_ok"]
    assert not raw.connected


def launch_call(function):
    result = {}
    thread = threading.Thread(target=lambda: result.update(function()), daemon=True)
    thread.start()
    return thread, result


@pytest.mark.parametrize("block", ["command", "read"])
def test_priority_stop_bypasses_blocked_io_and_reset_never_replays(block):
    mod = safety_module()
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()

    class Blocked(MockMobileBase):
        commands = 0
        reads = 0

        def command_velocity(self, command, *, generation):
            self.commands += 1
            if block == "command":
                entered.set()
                assert release.wait(2.)
            return super().command_velocity(command, generation=generation)

        def get_state(self):
            self.reads += 1
            if block == "read" and self.reads == 3:
                entered.set()
                assert release.wait(2.)
            return super().get_state()

        def stop(self, *, latch=True):
            result = super().stop(latch=latch)
            stopped.set()
            return result

    raw = Blocked(wall_lease_s=2., dt_s=.002)
    safe = mod.SafeBase(raw, limits())
    safe.connect()
    thread, result = launch_call(lambda: safe.walk_velocity(.1, 0, 0, .1))
    try:
        assert entered.wait(1.)
        second = safe.walk_velocity(.1, 0, 0, .1)
        assert not second["execution_ok"] and "concurrent" in second["error"]
        stop_thread, stop_result = launch_call(safe.stop)
        assert stopped.wait(.5), "priority stop waited behind blocked IO"
        stop_thread.join(1.)
        assert not stop_thread.is_alive() and stop_result["ok"]
        assert not safe.reset_stop()["ok"]  # pending delivery must finish, not be queued
        release.set()
        thread.join(1.)
        assert not thread.is_alive() and not result["execution_ok"]
        assert raw.get_state().latched
        assert safe.reset_stop()["ok"]
        position = raw.get_state().position_world
        time.sleep(.02)
        assert raw.get_state().position_world == position
        assert raw.commands <= 1  # no automatic retry or replay
    finally:
        release.set()
        thread.join(2.)
        safe.disconnect()


def test_wall_deadline_stops_during_blocked_command_independently_of_polling():
    mod = safety_module()
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()

    class Delayed(MockMobileBase):
        def command_velocity(self, command, *, generation):
            entered.set()
            assert release.wait(2.)
            return super().command_velocity(command, generation=generation)

        def stop(self, *, latch=True):
            result = super().stop(latch=latch)
            stopped.set()
            return result

    raw = Delayed(wall_lease_s=1., dt_s=.002)
    safe = mod.SafeBase(raw, limits(max_wall_duration_s=.12))
    safe.connect()
    thread, result = launch_call(lambda: safe.walk_velocity(.1, 0, 0, 1.))
    try:
        assert entered.wait(1.) and stopped.wait(.5)
        assert raw.get_state().latched
        release.set()
        thread.join(1.)
        assert not result["execution_ok"] and "wall" in result["error"]
        assert raw.get_state().position_world[0] == 0
    finally:
        release.set()
        thread.join(2.)
        safe.disconnect()


def test_uncertain_delivery_is_not_retried_even_if_final_snapshot_looks_good():
    mod = safety_module()

    class LostAck(MockMobileBase):
        commands = 0

        def command_velocity(self, command, *, generation):
            self.commands += 1
            super().command_velocity(command, generation=generation)
            raise TimeoutError("ACK lost after acceptance")

    raw = LostAck(wall_lease_s=1., dt_s=.002)
    safe = mod.SafeBase(raw, limits())
    safe.connect()
    try:
        result = safe.walk_velocity(.1, 0, 0, .1)
        assert not result["execution_ok"] and result["delivery_uncertain"]
        assert raw.commands == 1 and raw.get_state().latched
        assert safe.reset_stop()["ok"]
        time.sleep(.01)
        assert raw.commands == 1 and raw.get_state().linear_velocity_world == (0., 0., 0.)
    finally:
        safe.disconnect()


@pytest.mark.parametrize("change", [
    {"robot_id": "other"}, {"source": "other"}, {"measurement_kind": "physics"},
    {"epoch": "new-epoch"}, {"generation": 100}, {"producer_age_s": 100.},
    {"received_monotonic_s": 1e12}, {"fallen": True}, {"controller_status": "fault"},
    {"controller_status": "disabled"}, {"step": 0, "sim_time_s": 0.},
])
def test_corrupt_feedback_mid_motion_stops_and_never_confirms(change):
    mod = safety_module()

    class Corrupt(MockMobileBase):
        sent = False

        def command_velocity(self, command, *, generation):
            ack = super().command_velocity(command, generation=generation)
            self.sent = True
            return ack

        def get_state(self):
            state = super().get_state()
            return replace(state, **change) if self.sent else state

    raw = Corrupt(wall_lease_s=1., dt_s=.002)
    safe = mod.SafeBase(raw, limits())
    safe.connect()
    try:
        result = safe.walk_velocity(.1, 0, 0, .05)
        assert not result["execution_ok"] and not result["ok"]
        assert safe.latched and MockMobileBase.get_state(raw).latched
    finally:
        safe.disconnect()


def test_stop_racing_a_delayed_reset_keeps_permission_latched():
    mod = safety_module()
    entered, release = threading.Event(), threading.Event()

    class DelayedReset(MockMobileBase):
        def reset_stop(self):
            entered.set()
            assert release.wait(2.)
            return super().reset_stop()

    raw = DelayedReset(wall_lease_s=1., dt_s=.002)
    safe = mod.SafeBase(raw, limits())
    safe.connect()
    safe.stop()
    thread, result = launch_call(safe.reset_stop)
    try:
        assert entered.wait(1.)
        assert safe.stop()["ok"]
        release.set()
        thread.join(1.)
        assert not result["ok"] and safe.latched and raw.get_state().latched
        assert not safe.walk_velocity(.1, 0, 0, .01)["execution_ok"]
    finally:
        release.set()
        thread.join(2.)
        safe.disconnect()


def test_late_ack_after_cancel_is_reported_uncertain_not_completed():
    mod = safety_module()
    entered, release = threading.Event(), threading.Event()

    class DelayedAck(MockMobileBase):
        def command_velocity(self, command, *, generation):
            ack = super().command_velocity(command, generation=generation)
            entered.set()
            assert release.wait(2.)
            return ack

    raw = DelayedAck(wall_lease_s=1., dt_s=.002)
    safe = mod.SafeBase(raw, limits())
    safe.connect()
    thread, result = launch_call(lambda: safe.walk_velocity(.1, 0, 0, .1))
    try:
        assert entered.wait(1.)
        safe.stop()
        release.set()
        thread.join(1.)
        assert not result["execution_ok"] and result["delivery_uncertain"]
    finally:
        release.set()
        thread.join(2.)
        safe.disconnect()


def test_delayed_normal_stop_response_cannot_claim_emergency_latch_cleared():
    mod = safety_module()
    entered, release = threading.Event(), threading.Event()

    class DelayedStop(MockMobileBase):
        def stop(self, *, latch=True):
            ack = super().stop(latch=latch)
            if not latch:
                entered.set()
                assert release.wait(2.)
            return ack

    raw = DelayedStop(wall_lease_s=1., dt_s=.002)
    safe = mod.SafeBase(raw, limits())
    safe.connect()
    thread, result = launch_call(lambda: safe.stop(latch=False))
    try:
        assert entered.wait(1.)
        assert safe.stop()["latched"]
        release.set()
        thread.join(1.)
        assert result["latched"] and safe.latched and raw.get_state().latched
    finally:
        release.set()
        thread.join(2.)
        safe.disconnect()


def test_command_ack_with_collapsed_duration_is_rejected():
    mod = safety_module()

    class BadDuration(MockMobileBase):
        def command_velocity(self, command, *, generation):
            ack = super().command_velocity(command, generation=generation)
            ack["end_sim_time_s"] = ack["start_sim_time_s"]
            return ack

    raw = BadDuration(wall_lease_s=1., dt_s=.002)
    safe = mod.SafeBase(raw, limits())
    safe.connect()
    try:
        result = safe.walk_velocity(.1, 0, 0, 1e-10)
        assert not result["execution_ok"] and result["delivery_uncertain"]
        assert safe.latched
    finally:
        safe.disconnect()


def test_turn_failure_preserves_caller_angle_and_measured_zero():
    mod = safety_module()

    class Inert(MockMobileBase):
        def get_state(self):
            return replace(super().get_state(), orientation_wxyz=(1., 0., 0., 0.))

    raw = Inert(wall_lease_s=1., dt_s=.002)
    safe = mod.SafeBase(raw, limits(max_duration_s=.02))
    safe.connect()
    try:
        result = safe.turn(.2)
        assert not result["execution_ok"]
        assert result["requested_angle_rad"] == .2
        assert result["measured_angle_rad"] == 0.
    finally:
        safe.disconnect()


@pytest.mark.parametrize("bad", [None, {"ok": True}, {"ok": "true"}])
def test_ack_only_or_malformed_receipt_cannot_complete_motion(bad):
    mod = safety_module()

    class BadAck(MockMobileBase):
        def command_velocity(self, command, *, generation):
            super().command_velocity(command, generation=generation)
            return bad

    raw = BadAck(wall_lease_s=1., dt_s=.002)
    safe = mod.SafeBase(raw, limits())
    safe.connect()
    try:
        result = safe.walk_velocity(.1, 0, 0, .03)
        assert not result["execution_ok"] and result["delivery_uncertain"]
        assert raw.get_state().latched
    finally:
        safe.disconnect()


def test_stop_during_connect_survives_a_backend_startup_latch_reset():
    mod = safety_module()
    entered, release = threading.Event(), threading.Event()

    class SlowStartup(MockMobileBase):
        def connect(self):
            entered.set()
            assert release.wait(2.)
            super().connect()
            # Simulate a faulty startup resetting permission. SafeBase must
            # reapply the stop pending during startup before returning.
            super().reset_stop()

    raw = SlowStartup(wall_lease_s=1., dt_s=.002)
    safe = mod.SafeBase(raw, limits())
    thread = threading.Thread(target=safe.connect, daemon=True)
    thread.start()
    try:
        assert entered.wait(1.)
        safe.stop()
        release.set()
        thread.join(1.)
        assert not thread.is_alive() and safe.latched and raw.get_state().latched
    finally:
        release.set()
        thread.join(2.)
        safe.disconnect()


def test_python_interruption_stops_backend_before_propagating():
    mod = safety_module()

    class Interrupted(MockMobileBase):
        def command_velocity(self, command, *, generation):
            super().command_velocity(command, generation=generation)
            raise KeyboardInterrupt("interrupted after acceptance")

    raw = Interrupted(wall_lease_s=1., dt_s=.002)
    safe = mod.SafeBase(raw, limits())
    safe.connect()
    try:
        with pytest.raises(KeyboardInterrupt):
            safe.walk_velocity(.1, 0, 0, .2)
        assert raw.get_state().latched and safe.latched
        assert raw.get_state().linear_velocity_world == (0., 0., 0.)
    finally:
        safe.disconnect()
