"""Kinematic transport fixtures retain receipts, never prove physical locomotion."""
from copy import deepcopy
from dataclasses import replace

import pytest

from cascade.control.mock_base import MockMobileBase
from cascade.safety.base_harness import SafeBase
from test_mobile_safety import limits
from test_walk_distance_control import distance_limits


class RecordedBase(MockMobileBase):
    def __init__(self, *, stop_uncertain=False, invalid_ack=False, cancel_after_ack=False):
        super().__init__(wall_lease_s=2., dt_s=.01, auto_step=False)
        self.command_ack = None
        self.stop_uncertain = stop_uncertain
        self.invalid_ack = invalid_ack
        self.cancel_after_ack = cancel_after_ack
        self.safe = None

    def command_velocity(self, command, *, generation):
        ack = super().command_velocity(command, generation=generation)
        if self.invalid_ack:
            ack['generation'] += 10
        self.command_ack = deepcopy(ack)
        return ack

    def get_state(self):
        self.advance(.05)
        state = super().get_state()
        if self.cancel_after_ack and self.command_ack is not None:
            self.safe.stop()
        # Explicit inert kinematic fixture; real clock/control/ACK paths remain.
        return replace(state, position_world=(0., 0., .2), linear_velocity_world=(0.,0.,0.))

    def stop(self, *, latch=True):
        result = super().stop(latch=latch)
        if self.stop_uncertain and self.command_ack is not None:
            raise TimeoutError('stop reply unavailable')
        return result


def setup(raw):
    safe = SafeBase(raw, limits(max_duration_s=3., max_state_age_s=.5, max_no_progress_s=.5),
                    distance_control=distance_limits())
    raw.safe=safe
    safe.connect()
    return safe


@pytest.mark.parametrize('uncertain', [False, True])
def test_three_sim_second_distance_failure_keeps_admission_separate_from_stop(uncertain):
    raw=RecordedBase(stop_uncertain=uncertain);safe=setup(raw)
    try:
        result=safe.walk_distance(.025)
        assert result['error']=='measured distance did not reach target before simulation deadline'
        assert result['ack']==raw.command_ack and result['ack'] is not raw.command_ack
        assert result['ack']['accepted'] and not result['ack']['latched']
        assert result['ack']['end_sim_time_s']==result['ack']['start_sim_time_s']+3.
        assert result['measured']['after']['sim_time_s']>=result['ack']['end_sim_time_s']
        assert result['distance_baseline']['generation']==result['ack']['generation']
        assert result['measured_distance_m']==0. and not result['delivery_uncertain']
        assert result['stop_ack']['latched'] and result['stop_ack']['ok'] is not uncertain
        if uncertain:
            assert result['stop_ack']['delivery_uncertain'] and 'stop reply unavailable' in result['stop_ack']['error']
        else:
            assert result['stop_ack']['generation']==result['ack']['generation']+1
        assert not result['execution_ok'] and not result['ok'] and result['outcome']=='unverified'
    finally:
        raw.stop_uncertain=False  # Restore only the toy transport for fixture teardown.
        safe.disconnect()


def test_cancel_after_validated_admission_keeps_original_receipt():
    raw=RecordedBase(cancel_after_ack=True);safe=setup(raw)
    try:
        result=safe.walk_distance(.025)
        assert result['error']=='cancelled by stop'
        assert result['ack']==raw.command_ack and result['ack']['accepted']
        assert not result['execution_ok'] and not result['ok'] and not result['delivery_uncertain']
        assert result['outcome']=='unverified' and 'stop_ack' not in result
    finally:
        safe.disconnect()


def test_invalid_admission_is_not_promoted_to_a_preserved_validated_ack():
    raw=RecordedBase(invalid_ack=True);safe=setup(raw)
    try:
        result=safe.walk_distance(.025)
        assert result['error']=='ACK generation mismatch'
        assert 'ack' not in result and result['delivery_uncertain']
        assert not result['execution_ok'] and not result['ok']
    finally:
        safe.disconnect()


@pytest.mark.parametrize('uncertain', [False, True])
def test_composed_domain_retains_failed_admission_without_task_or_stop_credit(tmp_path, monkeypatch, uncertain):
    from cascade.apps import mobile_runtime
    from cascade.apps.robot_runtime import build_robot_runtime
    from cascade.config import Cfg, load_demo_config
    from test_mobile_effects import limits as verifier_limits

    # Only the actuator is an inert kinematic fixture. Composition, SafeBase,
    # independent checker, domain classification and task accounting are real.
    resolved=load_demo_config(base='microduck_mock', llm='mock').as_dict()
    profile=resolved['bases'][0]
    profile['distance_control']=distance_limits()
    profile['capabilities'].append('walk_distance')
    profile['resolved']['safety']=limits(max_duration_s=3., max_state_age_s=.5, max_no_progress_s=.5)
    profile['verifier']=verifier_limits()
    raw=RecordedBase(stop_uncertain=uncertain)
    monkeypatch.setattr(mobile_runtime, 'make_base', lambda _profile: raw)
    cfg=Cfg({'robot_mode':'composed','robot_id':'receipt-fixture','domains':{
        'locomotion':{'kind':'locomotion','robot_id':'receipt-fixture','resolved':resolved}}})
    runtime,_=build_robot_runtime(cfg,tmp_path)
    try:
        result=runtime.execute('locomotion.walk_distance', {'distance_m':.025})
        assert result['ack']==raw.command_ack and result['ack']['accepted']
        assert result['postcondition']['admission']['generation']==result['ack']['generation']
        assert result['postcondition']['admission']['accepted'] is True
        assert result['postcondition']['execution_failed'] is True
        assert result['postcondition']['status']=='unverified'
        assert not result['execution_ok'] and not result['ok'] and result['outcome']=='unverified'
        assert not runtime.execute('task_done', {'success':True,'summary':'must not promote receipt'})['success']
        stopped=runtime.stop()
        assert stopped['physical_stop_verified'] is False
        assert stopped['domains']['locomotion']['physical_stop_verified'] is False
    finally:
        raw.stop_uncertain=False
        assert runtime.close()['complete']
