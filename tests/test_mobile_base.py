"""CPU mobile contracts; no simulator or physical evidence is used here."""
import importlib
import importlib.util
import json
import time

import pytest


def mobile_module():
    spec = importlib.util.find_spec("cascade.control.mobile_base")
    assert spec is not None, "mobile contracts are not implemented"
    return importlib.import_module("cascade.control.mobile_base")


def state_fields():
    return dict(
        robot_id="microduck-test", source="test-source", epoch="episode-1",
        step=2, sim_time_s=0.01, received_monotonic_s=1.0, producer_age_s=0.0,
        position_world=[0.0, 0.0, 0.2], orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
        linear_velocity_world=[0.0, 0.0, 0.0], angular_velocity_body=[0.0, 0.0, 0.0],
        joint_names=["left", "right"], joint_positions=[0.1, -0.1],
        joint_velocities=[0.0, 0.0], controller_status="ready", generation=0,
        contacts=["left_foot", "right_foot"], fallen=False, latched=False,
        measurement_kind="kinematic_mock",
    )


def test_state_roundtrip_preserves_identity_clocks_and_copies_buffers():
    mod = mobile_module()
    data = state_fields()
    state = mod.BaseState.from_dict(data)
    data["position_world"][0] = 99.0
    assert state.position_world[0] == 0.0
    assert state.orientation_wxyz == (1.0, 0.0, 0.0, 0.0)
    assert mod.BaseState.from_dict(json.loads(json.dumps(state.as_dict()))) == state
    assert state.as_dict()["measurement_kind"] == "kinematic_mock"
    command = mod.VelocityCommand(0.1, 0.0, -0.2, 1.0)
    assert command.as_dict() == dict(vx=0.1, vy=0.0, wz=-0.2, duration_s=1.0)


@pytest.mark.parametrize("field,value", [
    ("step", True), ("step", -1), ("generation", 1.5), ("sim_time_s", float("nan")),
    ("producer_age_s", -0.1), ("position_world", [1, 2]),
    ("orientation_wxyz", [0, 0, 0, 0]), ("orientation_wxyz", [2, 0, 0, 0]),
    ("joint_positions", [0.1]), ("joint_names", ["same", "same"]),
    ("joint_velocities", [0.0, float("inf")]), ("fallen", "false"),
    ("robot_id", ""), ("epoch", " episode-1 "), ("source", None),
    ("controller_status", "whatever"), ("measurement_kind", "assumed_physics"),
])
def test_state_rejects_malformed_values(field, value):
    mod = mobile_module()
    data = state_fields()
    data[field] = value
    with pytest.raises((TypeError, ValueError)):
        mod.BaseState.from_dict(data)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_state_requires_exact_wire_schema(change):
    mod = mobile_module()
    data = state_fields()
    if change == "missing":
        del data["producer_age_s"]
    else:
        data["physics_confirmed"] = True
    with pytest.raises((TypeError, ValueError)):
        mod.BaseState.from_dict(data)


@pytest.mark.parametrize("args", [
    (float("nan"), 0, 0, 1), (0, float("inf"), 0, 1), (0, 0, "1", 1),
    (True, 0, 0, 1), (0, 0, 0, 0), (0, 0, 0, -1), (0, 0, 0, float("inf")),
])
def test_velocity_command_rejects_nonfinite_coercions_and_unbounded_duration(args):
    mod = mobile_module()
    with pytest.raises((ValueError, TypeError)):
        mod.VelocityCommand(*args)


def mock_class():
    assert importlib.util.find_spec("cascade.control.mock_base"), "kinematic mock missing"
    return importlib.import_module("cascade.control.mock_base").MockMobileBase


def test_mock_is_explicit_kinematic_and_reads_do_not_advance():
    mod = mobile_module()
    raw = mock_class()(wall_lease_s=1.0, auto_step=False)
    assert isinstance(raw, mod.MobileBase)
    assert raw.metadata["measurement_kind"] == "kinematic_mock"
    assert raw.capabilities == frozenset({"walk_velocity", "turn", "stop_navigation"})
    raw.connect()
    try:
        before = raw.get_state()
        assert raw.get_state().step == before.step
        ack = raw.command_velocity(mod.VelocityCommand(.1, 0, 0, .1), generation=0)
        assert ack["accepted"] and ack["generation"] == 1
        assert ack["end_sim_time_s"] - ack["start_sim_time_s"] == pytest.approx(.1)
        raw.advance(.2)
        after = raw.get_state()
        assert after.position_world[0] == pytest.approx(.01)
        assert after.linear_velocity_world == (0., 0., 0.)
        assert after.step > before.step and after.sim_time_s > before.sim_time_s
        assert after.measurement_kind == "kinematic_mock"
        assert not raw.command_velocity(mod.VelocityCommand(.1, 0, 0, .1), generation=0)["ok"]
    finally:
        raw.disconnect()
    assert not raw.connected


def test_backend_stop_reset_fences_pending_commands_without_reviving_them():
    mod = mobile_module()
    raw = mock_class()(wall_lease_s=1., auto_step=False)
    pre = raw.stop()  # before connect must latch, not activate anything
    raw.connect()
    try:
        assert raw.get_state().latched
        assert not raw.command_velocity(mod.VelocityCommand(.1, 0, 0, .1), generation=pre["generation"])["ok"]
        normal = raw.stop(latch=False)
        assert normal["latched"]
        reset = raw.reset_stop()
        assert reset["generation"] > normal["generation"] and not reset["latched"]
        raw.advance(.1)
        assert raw.get_state().position_world[0] == 0
        assert not raw.command_velocity(mod.VelocityCommand(.1, 0, 0, .1), generation=normal["generation"])["ok"]
        ack = raw.command_velocity(mod.VelocityCommand(.1, 0, 0, .1), generation=reset["generation"])
        assert ack["ok"]
        raw.stop(latch=False)
        raw.advance(.1)
        assert raw.get_state().position_world[0] == 0
    finally:
        raw.disconnect()


def test_mock_wall_lease_expires_without_simulation_or_client_reads():
    mod = mobile_module()
    raw = mock_class()(wall_lease_s=.04, dt_s=.002, auto_step=False)
    raw.connect()
    try:
        raw.command_velocity(mod.VelocityCommand(.1, 0, 0, 10), generation=0)
        time.sleep(.08)
        state = raw.get_state()
        assert state.step == 0
        assert state.latched and state.generation > 1
        assert state.controller_status == "fault"
        assert not raw.reset_stop()["ok"]  # resetting permission cannot revive a bad controller
        raw.advance(.1)
        assert raw.get_state().position_world[0] == 0.
    finally:
        raw.disconnect()
    assert not any(t.name.startswith("mock-mobile-") for t in __import__("threading").enumerate())


def test_rig_rejects_ambiguity_and_continues_after_one_backend_failure():
    assert importlib.util.find_spec("cascade.control.mobile_rig"), "MobileRig missing"
    rig_cls = importlib.import_module("cascade.control.mobile_rig").MobileRig
    calls = []

    class Member:
        def __init__(self, name, fail=False):
            self.name, self.fail = name, fail

        def stop(self, *, latch=True):
            calls.append((self.name, "stop", latch))
            if self.fail:
                raise OSError("stop unavailable")
            return {"ok": True}

        def disconnect(self):
            calls.append((self.name, "close"))
            if self.fail:
                raise OSError("close unavailable")

    first, second = Member("first", True), Member("second")
    rig = rig_cls([first, second], ["a", "b"])
    assert rig.primary is first and rig.get() is first and rig.get("b") is second
    assert list(rig) == [first, second] and len(rig) == 2
    with pytest.raises(KeyError):
        rig.get("missing")
    for bases, names in (([], []), ([first], []), ([first, second], ["a", "a"]), ([first], [""])):
        with pytest.raises(ValueError):
            rig_cls(bases, names)
    assert rig.stop(latch=False)["ok"] is False
    assert rig.close()["ok"] is False
    assert ("second", "stop", False) in calls and ("second", "close") in calls
