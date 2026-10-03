"""Post-completion vetoes, using synthetic records and the real verifier.

No policy, solver, actuator or production clock is run or patched here.
The initial 30 mm + 50 mm counterexample comes from the independent PR87 review.
"""
from copy import deepcopy
import math

import pytest

from cascade.agent.base_effects import BasePostconditionChecker, _Window
from cascade.config import load_profile
from test_mobile_effects import fixture_motion_receipt, fixture_support_contract, state


def outcome(*, active=.03, coast=(0., 0., 0.), target=.03, heading=0.,
            return_coast=False, corrupt=None, result=None, late_coast=False):
    limits = load_profile('bases', 'microduck_distance_native_slow').as_dict()['verifier']
    checker = BasePostconditionChecker(lambda: None, limits=limits,
                                       support_contract=fixture_support_contract())
    op = _Window('coast-fixture', 'walk_distance', {'distance_m': target},
                 started=100., deadline=129.)
    receipt = fixture_motion_receipt() if result is None else deepcopy(result)
    receipt['ack'].update(start_sim_time_s=.02, end_sim_time_s=3.02)
    for n in range(1, 65 if late_coast else 62):
        if n == 12:
            op.finished = 100.055
        if late_coast and n == 62:
            op.cancel.set()  # Valid in-flight returns may veto but cannot earn credit.
        stamp = 100. + n * .005
        fraction = max(0, min(n - (61 if late_coast else 11), 10)) / 10
        if return_coast:
            fraction -= max(0, min(n - 21, 10)) / 10
        x = active * min(n - 1, 10) / 10 + coast[0] * fraction
        y = coast[1] * fraction
        yaw = heading + coast[2] * fraction
        kwargs = dict(generation=0 if n == 1 else 1 if n <= 10 else 2,
                      controller_status='active' if 1 < n <= 10 else 'ready',
                      position_world=(math.cos(heading)*x - math.sin(heading)*y,
                                      math.sin(heading)*x + math.cos(heading)*y, .3),
                      orientation_wxyz=(math.cos(yaw/2), 0., 0., math.sin(yaw/2)),
                      received_monotonic_s=stamp + .001)
        value = state(n, **kwargs)
        if corrupt is not None:
            value = corrupt(n, value)
        op.attempts += 1
        checker._accept(op, value, stamp, stamp + .001)
    verdict = checker._receipt(op, receipt)
    checker._evaluate(op, verdict)
    checker.close()
    return verdict


@pytest.mark.parametrize('sign', [-1., 1.])
@pytest.mark.parametrize('heading', [0., math.pi/2, -math.pi/2, math.pi-.01])
def test_coasting_overshoot_after_valid_distance_cannot_confirm(sign, heading):
    verdict = outcome(active=sign*.03, coast=(sign*.05, 0., 0.), target=sign*.03, heading=heading)
    assert verdict['status'] == 'refuted', verdict
    assert 'post-completion' in verdict['reason'] and 'excess progress' in verdict['reason']
    assert verdict['metrics']['body_displacement_m'] == pytest.approx([sign*.03, 0.])
    assert verdict['metrics']['distance_outcome']['body_displacement_m'] == pytest.approx([sign*.08, 0.])
    assert verdict['metrics']['settle_drift_m'] == pytest.approx(0.)


def test_healthy_goal_and_small_coast_keep_same_limits():
    for coast in (0., .005):
        verdict = outcome(coast=(coast, 0., 0.))
        assert verdict['status'] == 'confirmed', verdict
        assert verdict['metrics']['body_displacement_m'] == pytest.approx([.03, 0.])
        assert verdict['metrics']['distance_outcome']['body_displacement_m'] == pytest.approx([.03+coast, 0.])
        assert verdict['limits'] == load_profile('bases', 'microduck_distance_native_slow').as_dict()['verifier']


@pytest.mark.parametrize('coast,reason', [
    ((-.025, 0., 0.), 'insufficient progress'),
    ((-.05, 0., 0.), 'insufficient progress'),
    ((0., .03, 0.), 'vy drift'),
    ((0., 0., .12), 'wz drift'),
])
def test_post_completion_retreat_and_unrequested_drift_veto(coast, reason):
    verdict = outcome(coast=coast)
    assert verdict['status'] == 'refuted', verdict
    assert reason in verdict['reason']


@pytest.mark.parametrize('coast', [(.05, 0., 0.), (0., .03, 0.), (0., 0., .12)])
def test_excursion_cannot_be_erased_by_returning_before_quiet_suffix(coast):
    verdict = outcome(coast=coast, return_coast=True)
    assert verdict['status'] == 'refuted', verdict
    assert 'post-completion' in verdict['reason']
    assert verdict['evidence']['distance_outcome_interval']['first_veto']['step'] < 31


@pytest.mark.parametrize('coast', [0., .03])
def test_missing_admitted_travel_is_not_repaired_by_later_goal(coast):
    verdict = outcome(active=0., coast=(coast, 0., 0.))
    assert verdict['status'] == 'refuted' and 'no effect' in verdict['reason']
    assert verdict['metrics']['body_displacement_m'] == [0., 0.]


def test_cancelled_inflight_fresh_observations_can_only_veto():
    # Three late states add30mm in12mm-or-smaller physical increments; they
    # satisfy plausibility but must not disappear behind the completed quiet window.
    verdict = outcome(coast=(.1, 0., 0.), late_coast=True)
    assert verdict['status'] == 'refuted'
    assert 'excess progress' in verdict['evidence']['distance_outcome_interval']['first_veto']['reason']
    assert verdict['metrics']['body_displacement_m'] == pytest.approx([.03, 0.])
    assert verdict['metrics']['distance_outcome']['body_displacement_m'] == pytest.approx([.06, 0.])
    late = [e for e in verdict['evidence']['observations'] if e['late']]
    assert len(late) == 3 and all(e['valid'] and not e['confirmation_eligible'] for e in late)


@pytest.mark.parametrize('fault,reason', [('support', 'forbidden external'),
    ('unavailable', 'support unavailable'), ('stale', 'stale'),
    ('generation', 'generation'), ('epoch', 'epoch')])
def test_coasting_does_not_hide_support_or_channel_failures(fault, reason):
    from dataclasses import replace
    def corrupt(n, value):
        if n != 15:
            return value
        if fault in ('support', 'unavailable'):
            support = value.as_dict()['support']
            if fault == 'support':
                support['contacts'][0].update(shape_b='/Fixture/Robot/body', shape_b_id=3)
            else:
                support.update(status='unavailable', reason='fixture unavailable', contacts=[])
            return replace(value, support=support)
        if fault == 'stale':
            return replace(value, producer_age_s=1.)
        if fault == 'generation':
            return replace(value, generation=4)
        raw = value.as_dict()
        raw.pop('support')
        raw.pop('step')
        raw['epoch'] = 'another-epoch'
        return state(n, **raw)
    verdict = outcome(corrupt=corrupt)
    assert verdict['status'] != 'confirmed', verdict
    assert reason in verdict['reason']


@pytest.mark.parametrize('which,value', [('execution_ok', False), ('delivery_uncertain', True)])
def test_valid_outcome_cannot_repair_failed_or_uncertain_execution(which, value):
    receipt = fixture_motion_receipt()
    receipt[which] = value
    verdict = outcome(result=receipt)
    assert verdict['status'] == 'unverified' and 'execution failed' in verdict['reason']


def test_old_stop_ack_cannot_bind_new_outcome():
    receipt = fixture_motion_receipt()
    receipt['stop_ack']['generation'] = 0
    verdict = outcome(result=receipt)
    assert verdict['status'] == 'unverified' and 'generation' in verdict['reason']
