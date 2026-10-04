"""Observed-turn admission regressions; synthetic feedback proves no gait."""
from dataclasses import replace
import math

import pytest

from cascade.control.mock_base import MockMobileBase
from cascade.safety.base_harness import SafeBase
from mobile_support_fixture import support, support_contract
from test_mobile_safety import limits


@pytest.mark.parametrize('angle', [.2, -.2])
@pytest.mark.parametrize('when', ['delivery', 'after_expiry'])
def test_turn_cannot_complete_from_delivery_drift_or_late_rotation(angle, when):
    class RecordedTurn(MockMobileBase):
        ack = None

        def command_velocity(self, command, *, generation):
            self.ack = super().command_velocity(command, generation=generation)
            return self.ack

        def get_state(self):
            self.advance(.03)
            state = super().get_state()
            rotated = self.ack is not None and (
                when == 'delivery' or state.sim_time_s > self.ack['end_sim_time_s'])
            yaw = angle if rotated else 0.
            return replace(state, orientation_wxyz=(math.cos(yaw/2), 0., 0., math.sin(yaw/2)),
                           angular_velocity_body=(0., 0., 0.))

    raw = RecordedTurn(wall_lease_s=2., auto_step=False)
    safe = SafeBase(raw, limits(max_duration_s=.1))
    safe.connect()
    try:
        result = safe.turn(angle)
        assert not result['execution_ok']
        assert 'before simulation deadline' in result['error']
        assert result['requested_angle_rad'] == angle
        assert result['measured_angle_rad'] == 0.
        # Retain the late observed pose even though it supplies no progress.
        assert result['measured']['after']['sim_time_s'] > raw.ack['end_sim_time_s']
        assert abs(SafeBase._yaw(raw.get_state())-angle) < 1e-12
        assert raw.get_state().latched
    finally:
        safe.disconnect()


def turn_limits():
    return dict(max_translation_path_m=.02, min_height_m=.06, max_tilt_rad=.5)


class TurnFeedback(MockMobileBase):
    """Synthetic completed feedback; physical-shaped fields test guards only."""
    def __init__(self, change=lambda state, index: state):
        super().__init__(wall_lease_s=2., auto_step=False)
        self.change, self.commands, self.index = change, 0, 0

    @property
    def metadata(self):
        return {**super().metadata, 'measurement_kind': 'physics'}

    def command_velocity(self, command, *, generation):
        self.commands += 1
        return super().command_velocity(command, generation=generation)

    def get_state(self):
        self.advance(.02)
        state = super().get_state()
        if self.commands:
            self.index += 1
        observed = {**support(state.step, state.sim_time_s), 'epoch': state.epoch,
                    'model_identity_sha256': 'e'*64}
        state = replace(state, measurement_kind='physics', model_identity_sha256='e'*64,
                        support=observed)
        return self.change(state, self.index)


def guarded(raw):
    return SafeBase(raw, limits(max_duration_s=.3), turn_control=turn_limits(),
                    support_contract=support_contract())


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_turn_counts_returning_3d_path_and_stops_before_yaw_goal(axis):
    def path(state, index):
        position = list(state.position_world)
        position[axis] += .011 if index == 1 else 0.
        return replace(state, position_world=position)

    raw = TurnFeedback(path)
    safe = guarded(raw)
    safe.connect()
    try:
        result = safe.turn(.1)
        assert not result['execution_ok'] and 'translation path' in result['error']
        assert result['measured_translation_path_m'] == pytest.approx(.022)
        assert result['measured']['after']['position_world'] == [0., 0., .2]
        assert raw.commands == 1 and raw.index == 2 and safe.latched
    finally:
        safe.disconnect()


@pytest.mark.parametrize('fault', ['height', 'tilt', 'missing', 'unavailable', 'identity', 'nonsole'])
@pytest.mark.parametrize('during', [False, True])
def test_turn_posture_and_support_veto_before_or_during_command(fault, during):
    def change(state, index):
        if during and index != 2:
            return state
        if fault == 'height':
            return replace(state, position_world=(0., 0., .04))
        if fault == 'tilt':
            return replace(state, orientation_wxyz=(math.cos(.3), math.sin(.3), 0., 0.))
        if fault == 'missing':
            return replace(state, support=None)
        observed = state.support.as_observation_dict()
        if fault == 'unavailable':
            observed.update(status='unavailable', reason='missing channel', contacts=[])
        elif fault == 'identity':
            observed['model_identity_sha256'] = 'f'*64
            return replace(state, support=observed, model_identity_sha256='f'*64)
        else:
            observed['contacts'][0].update(shape_b='/Fixture/trunk', normal_force_n=0.,
                                           force_on_b_world_n=[0., 0., 0.])
        return replace(state, support=observed)

    raw = TurnFeedback(change)
    safe = guarded(raw)
    safe.connect()
    try:
        result = safe.turn(.1)
        assert not result['execution_ok']
        assert ('posture' if fault in ('height', 'tilt') else
                'identity' if fault == 'identity' else 'contact') in result['error']
        assert raw.commands == int(during) and safe.latched
        if during:
            assert raw.index == 2
    finally:
        safe.disconnect()


@pytest.mark.parametrize('angle', [.08, -.08])
def test_guarded_turn_keeps_command_and_unverified_outcome(angle):
    raw = TurnFeedback()
    safe = guarded(raw)
    safe.connect()
    try:
        result = safe.turn(angle)
        assert result['execution_ok'], result.get('error')
        assert result['outcome'] == 'unverified' and not result['ok']
        assert result['command']['wz'] == math.copysign(.8, angle)
        assert abs(result['measured_angle_rad']-angle) <= .01
        assert result['measured_translation_path_m'] == 0.
        assert not safe.latched
    finally:
        safe.disconnect()


@pytest.mark.parametrize('config', [{}, {'max_translation_path_m': 0, 'min_height_m': .06, 'max_tilt_rad': .5},
                                   {'max_translation_path_m': .02, 'min_height_m': True, 'max_tilt_rad': .5},
                                   {'max_translation_path_m': .02, 'min_height_m': .06, 'max_tilt_rad': math.pi}])
def test_invalid_turn_contract_is_rejected_without_opening_backend(config):
    raw = TurnFeedback()
    with pytest.raises(ValueError):
        SafeBase(raw, limits(), turn_control=config)
    assert not raw.connected and raw.commands == 0


def test_candidate_turn_bounds_match_original_verifier_and_runtime_wiring(tmp_path):
    from cascade.apps.mobile_runtime import build_mobile_runtime
    from cascade.config import load_demo_config, load_profile

    profile = load_profile('bases', 'microduck_distance_native_slow').as_dict()
    control = profile['turn_control']
    assert control['max_translation_path_m'] == profile['verifier']['max_lateral_drift_m'] == .02
    assert all(control[k] == profile['verifier'][k] for k in ('min_height_m', 'max_tilt_rad'))
    cfg = load_demo_config(base='microduck_mock', llm='mock')
    cfg._data['bases'][0]['turn_control'] = control
    runtime, owner = build_mobile_runtime(cfg, tmp_path)
    try:
        assert owner.primary.turn_control == control
        assert not owner.primary.raw.connected
    finally:
        runtime.close()


@pytest.mark.parametrize('during', [False, True])
def test_turn_known_empty_support_is_allowed_only_after_loaded_preflight(during):
    def empty(state, index):
        if during and index == 0:
            return state
        return replace(state, support={**state.support.as_observation_dict(), 'contacts': []})

    raw = TurnFeedback(empty)
    safe = guarded(raw)
    safe.connect()
    try:
        result = safe.turn(.08)
        assert result['execution_ok'] is during
        assert raw.commands == int(during)
        assert result['outcome'] == 'unverified'
        if not during:
            assert 'positive solved sole support' in result['error']
    finally:
        safe.disconnect()


def test_turn_late_observation_retains_translation_veto_without_progress_credit():
    raw = TurnFeedback(lambda state, index: replace(state, position_world=(.03 if index else 0., 0., .2)))
    safe = SafeBase(raw, limits(max_duration_s=.01), turn_control=turn_limits(),
                    support_contract=support_contract())
    safe.connect()
    try:
        result = safe.turn(.08)
        assert not result['execution_ok'] and 'translation path' in result['error']
        assert result['measured_translation_path_m'] == .03
        assert result['measured_angle_rad'] == 0.
        assert safe.latched
    finally:
        safe.disconnect()
