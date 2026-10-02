"""Measured-distance control regressions. Kinematic fixtures prove no gait."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import math
import threading

import pytest

from cascade.control.mock_base import MockMobileBase
from cascade.safety.base_harness import SafeBase
from cascade.skills.mobile_runtime import tool_specs_for_profiles
from test_mobile_safety import limits
from test_mobile_base import state_fields
from mobile_support_fixture import support_contract, support


def distance_limits(**updates):
    value = dict(speed_m_s=.1, max_distance_m=.05, tolerance_m=.002,
                 max_lateral_drift_m=.01, max_heading_drift_rad=.08,
                 min_height_m=.06, max_tilt_rad=.5)
    value.update(updates)
    return value


def configured(raw, **updates):
    return SafeBase(raw, limits(max_duration_s=.4, max_state_age_s=.5,
                               max_no_progress_s=.5),
                    distance_control=distance_limits(**updates))


@pytest.mark.parametrize("distance", [.02, -.02])
def test_distance_stops_on_measured_travel_and_never_claims_mock_physics(distance):
    raw = MockMobileBase(wall_lease_s=2., dt_s=.002)
    safe = configured(raw)
    safe.connect()
    try:
        result = safe.walk_distance(distance)
        assert result["execution_ok"], result
        assert result["ok"] is False and result["outcome"] == "unverified"
        assert result["command"]["vx"] == math.copysign(.1, distance)
        assert result["command"]["duration_s"] == .4
        assert result["requested_distance_m"] == distance
        assert abs(result["measured_distance_m"] - distance) <= .002
        baseline = result["distance_baseline"]
        assert baseline["sim_time_s"] > result["ack"]["start_sim_time_s"]
        assert baseline["generation"] == result["ack"]["generation"]
        assert result["measured"]["after"]["sim_time_s"] < result["ack"]["end_sim_time_s"]
        assert raw.get_state().linear_velocity_world == (0., 0., 0.)
    finally:
        safe.disconnect()


def test_profile_is_opt_in_and_original_velocity_semantics_are_unchanged():
    raw = MockMobileBase(wall_lease_s=2.)
    default = SafeBase(raw, limits())
    assert "walk_distance" not in default.capabilities
    assert not default.walk_distance(.02)["execution_ok"]
    profile = {"capabilities": ["walk_velocity", "turn", "stop_navigation"]}
    assert "walk_distance" not in {s["name"] for s in tool_specs_for_profiles([profile])}
    profile["capabilities"] = ["walk_distance", "turn", "stop_navigation"]
    tools = {s["name"] for s in tool_specs_for_profiles([profile])}
    assert "walk_distance" in tools and "walk_velocity" not in tools
    assert configured(raw).harness.limits["max_vx"] == default.harness.limits["max_vx"]


@pytest.mark.parametrize("bad", [None, True, 0., .001, .051, float("nan"), float("inf")])
def test_invalid_distance_refused_before_connect_or_motion(bad):
    raw = MockMobileBase(wall_lease_s=2.)
    result = configured(raw).walk_distance(bad)
    assert not result["execution_ok"] and not raw.connected


@pytest.mark.parametrize("field,value", [("speed_m_s", .3), ("tolerance_m", .06),
    ("max_tilt_rad", math.pi), ("max_heading_drift_rad", math.pi),
    ("max_distance_m", 0), ("speed_m_s", True)])
def test_distance_contract_rejects_inconsistent_bounds(field, value):
    with pytest.raises(ValueError):
        configured(MockMobileBase(wall_lease_s=2.), **{field: value})


def test_preadmission_drift_and_commanded_velocity_cannot_substitute_for_motion():
    class DriftingInert(MockMobileBase):
        dispatched = False

        def command_velocity(self, command, *, generation):
            self.dispatched = True
            return super().command_velocity(command, generation=generation)

        def get_state(self):
            state = super().get_state()
            # Distinct pre-admission0 and post-admission.02, then stationary.
            return replace(state, position_world=(.02 if self.dispatched else 0., 0., .2),
                           linear_velocity_world=(0., 0., 0.))

    raw = DriftingInert(wall_lease_s=2., dt_s=.002)
    safe = configured(raw)
    safe.connect()
    try:
        result = safe.walk_distance(.02)
        assert not result["execution_ok"] and "did not reach" in result["error"]
        assert result["measured"]["before"]["position_world"][0] == 0.
        assert result["distance_baseline"]["position_world"][0] == .02
        assert result["measured_distance_m"] == 0.
        assert raw.get_state().latched
    finally:
        safe.disconnect()


@pytest.mark.parametrize("fault", ["lateral", "heading", "low", "tilt"])
def test_distance_control_stops_on_geometric_fault(fault):
    class Broken(MockMobileBase):
        reads_after_command = 0

        def get_state(self):
            state = super().get_state()
            if state.controller_status == "active":
                self.reads_after_command += 1
                if self.reads_after_command > 2:
                    if fault == "lateral":
                        return replace(state, position_world=(state.position_world[0], .02, .2))
                    if fault == "low":
                        return replace(state, position_world=(state.position_world[0], 0., .04))
                    angle = .2 if fault == "heading" else .7
                    q = ((math.cos(angle/2), 0., 0., math.sin(angle/2)) if fault == "heading"
                         else (math.cos(angle/2), math.sin(angle/2), 0., 0.))
                    return replace(state, orientation_wxyz=q)
            return state

    raw = Broken(wall_lease_s=2., dt_s=.002)
    safe = configured(raw)
    safe.connect()
    try:
        result = safe.walk_distance(.02)
        assert not result["execution_ok"], result
        assert "drift exceeded" in result["error"] or "posture bound" in result["error"]
        assert raw.get_state().latched
    finally:
        safe.disconnect()


def test_priority_stop_invalidates_blocked_distance_dispatch_without_replay():
    entered, release = threading.Event(), threading.Event()

    class Delayed(MockMobileBase):
        def command_velocity(self, command, *, generation):
            entered.set()
            assert release.wait(2.)
            return super().command_velocity(command, generation=generation)

    raw = Delayed(wall_lease_s=2., dt_s=.002)
    safe = configured(raw)
    safe.connect()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(safe.walk_distance, .02)
            assert entered.wait(2.)
            stop = safe.stop(latch=True)
            release.set()
            result = pending.result(2.)
        assert stop["ok"] and stop["latched"]
        assert not result["execution_ok"] and result["delivery_uncertain"]
        assert raw.get_state().position_world[0] == 0.
    finally:
        release.set()
        safe.disconnect()


@pytest.mark.parametrize("fault", ["missing", "unavailable", "forbidden", "identity"])
def test_physical_distance_requires_complete_bound_contact_channel(fault):
    from cascade.control.mobile_base import BaseState

    raw = MockMobileBase(wall_lease_s=2.)
    safe = SafeBase(raw, limits(), distance_control=distance_limits(), support_contract=support_contract())
    fields = state_fields()
    fields.update(measurement_kind="physics", model_identity_sha256='e'*64)
    observed = support(fields['step'], fields['sim_time_s'])
    observed.update(epoch=fields['epoch'], model_identity_sha256='e'*64)
    fields['support'] = observed
    if fault == "missing":
        fields['support'] = None
    elif fault == "unavailable":
        observed.update(status="unavailable", reason="force channel missing", contacts=[])
    elif fault == "forbidden":
        observed['contacts'][0]['shape_b'] = '/Fixture/trunk'
    else:
        fields['model_identity_sha256'] = observed['model_identity_sha256'] = 'f'*64
    with pytest.raises(ValueError, match="contact|identity"):
        safe._distance_state(BaseState.from_dict(fields))


def test_known_empty_contact_solve_is_allowed_during_swing_not_counted_as_rest():
    from cascade.control.mobile_base import BaseState

    safe = SafeBase(MockMobileBase(wall_lease_s=2.), limits(),
                    distance_control=distance_limits(), support_contract=support_contract())
    fields = state_fields()
    fields.update(measurement_kind="physics", model_identity_sha256='e'*64)
    fields['support'] = {**support(fields['step'], fields['sim_time_s']),
                         'epoch': fields['epoch'], 'model_identity_sha256': 'e'*64, 'contacts': []}
    safe._distance_state(BaseState.from_dict(fields))
    # This is only the moving-phase veto. Independent rest still needs load.
    with pytest.raises(ValueError, match="positive solved sole support"):
        safe._distance_state(BaseState.from_dict(fields), require_load=True)
