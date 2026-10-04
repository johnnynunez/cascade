"""A cached observation never caches permission, age, or caller-owned objects."""
import copy
import math

import pytest

from mobile_support_fixture import support
from test_mobile_bridge import command, control, publish


def test_contact_validation_once_per_publish_and_live_permission_on_every_read(control, monkeypatch):
    from cascade.control.mobile_support import SolvedContact

    c, clock = control
    calls = []
    original = SolvedContact.__post_init__

    def checked(contact):
        calls.append(contact)
        original(contact)

    monkeypatch.setattr(SolvedContact, '__post_init__', checked)
    observed = support(1, .005)
    observed['contacts'] = [copy.deepcopy(observed['contacts'][0]) for _ in range(192)]
    for index, contact in enumerate(observed['contacts']):
        contact['point_world_m'][0] = index * .001
    publish(c, support=observed, balance_active=True)
    assert len(calls) == 192
    first = c.state()['state']
    clock.now += .1
    command(c)
    active = c.state()['state']
    assert active['controller_status'] == 'active' and active['generation'] > first['generation']
    ack = c.stop()
    stopped = c.state()['state']
    assert stopped['generation'] == ack['generation'] and stopped['latched']
    assert stopped['controller_status'] == 'ready'
    c.reset_stop()
    reset = c.state()['state']
    assert reset['generation'] > stopped['generation'] and not reset['latched']
    assert reset['step'] == first['step'] and reset['sim_time_s'] == first['sim_time_s']
    assert reset['received_monotonic_s'] == first['received_monotonic_s']
    assert reset['producer_age_s'] == pytest.approx(.1)
    assert reset['support'] == first['support'] and len(reset['support']['contacts']) == 192
    assert len(calls) == 192
    publish(c, step=2, sim_time=.01)
    assert len(calls) == 193
    assert c.state()['state']['support']['step'] == 2


def test_input_and_both_wire_views_are_detached_from_completed_snapshot(control):
    c, _ = control
    observed, q = support(1, .005), [0.] * 14
    publish(c, support=observed, q=q)
    expected = c.state()
    observed['contacts'][0]['force_on_b_world_n'][2] = 1000.
    observed['contacts'].append(copy.deepcopy(observed['contacts'][0]))
    q[0] = 9.
    response = c.state()
    response['q'][0] = 8.
    response['support']['contacts'][0]['point_world_m'][0] = 7.
    response['state']['joint_positions'][0] = 6.
    response['state']['support']['contacts'][0]['force_on_b_world_n'] = (0., 0., 500.)
    response['state']['support']['epoch'] = 'foreign'
    assert c.state() == expected


def test_new_completed_measurement_replaces_every_cached_observation(control):
    c, clock = control
    publish(c)
    clock.now += .1
    observed = support(2, .02)
    observed['contacts'][0]['force_on_b_world_n'] = [1., 2., 4.]
    observed['contacts'][0]['normal_force_n'] = 4.
    publish(c, step=2, sim_time=.02, position=[1., 2., .13],
            orientation_wxyz=[math.sqrt(.5), 0., 0., math.sqrt(.5)],
            linear_velocity=[.1, .2, .3], angular_velocity=[.4, .5, .6],
            q=[.1]*14, dq=[.2]*14, joint_names=[f'new_{i}' for i in range(14)],
            contacts=['new_foot'], support=observed)
    state = c.state()['state']
    assert state['step'] == 2 and state['sim_time_s'] == .02 and state['producer_age_s'] == 0.
    assert state['received_monotonic_s'] == clock.now
    for field, expected in dict(position_world=[1., 2., .13],
            orientation_wxyz=[math.sqrt(.5), 0., 0., math.sqrt(.5)],
            linear_velocity_world=[.1, .2, .3], angular_velocity_body=[.4, .5, .6],
            joint_positions=[.1]*14, joint_velocities=[.2]*14,
            joint_names=[f'new_{i}' for i in range(14)], contacts=['new_foot']).items():
        assert state[field] == expected
    assert state['support']['contacts'][0]['force_on_b_world_n'] == (1., 2., 4.)


@pytest.mark.parametrize('field,value', [('step', 1), ('epoch', 'foreign'),
    ('model_identity_sha256', 'f'*64), ('sim_time_s', .03)])
def test_bad_new_support_faults_without_replacing_last_completed_observation(control, field, value):
    c, clock = control
    publish(c)
    first = c.state()['state']
    clock.now += .1
    bad = support(2, .01)
    bad[field] = value
    with pytest.raises(ValueError):
        publish(c, step=2, sim_time=.01, support=bad)
    failed = c.state()['state']
    assert failed['step'] == 1 and failed['support'] == first['support']
    assert failed['latched'] and failed['controller_status'] == 'fault'
    assert failed['generation'] > first['generation'] and failed['producer_age_s'] == pytest.approx(.1)


def test_epoch_reset_discards_cached_feedback_and_unavailable_support_stays_unknown(control):
    c, _ = control
    publish(c)
    old = c.state()['state']
    c.begin_epoch()
    assert not c.state()['feedback_available'] and c.state()['state'] is None
    observed = support(1, .005)
    observed.update(status='unavailable', reason='no solved sample', contacts=[])
    publish(c, support=observed)
    state = c.state()['state']
    assert state['epoch'] != old['epoch'] and state['support']['epoch'] == state['epoch']
    assert state['support']['status'] == 'unavailable' and not state['support']['contacts']


def test_cache_does_not_bypass_current_clock_validation_or_staleness_gate(control):
    c, clock = control
    publish(c)
    clock.now += .501
    assert c.state()['state']['producer_age_s'] == pytest.approx(.501)
    with pytest.raises(ValueError, match='stale'):
        command(c)
    clock.now = math.inf
    with pytest.raises(ValueError, match='finite'):
        c.state()


def test_zero_clock_missing_support_and_nested_raw_telemetry_remain_truthful(control):
    c, clock = control
    clock.now = 0.
    telemetry = {'timestamps': [0., {'sample': 0.}]}
    publish(c, support=None, telemetry=telemetry)
    first = c.state()
    assert first['state']['received_monotonic_s'] == first['state']['producer_age_s'] == 0.
    assert first['state']['support'] is None
    telemetry['timestamps'][1]['sample'] = 9.
    first['telemetry']['timestamps'][1]['sample'] = 8.
    clock.now = .1
    current = c.state()
    assert current['telemetry'] == {'timestamps': [0., {'sample': 0.}]}
    assert current['state']['received_monotonic_s'] == 0.
    assert current['state']['producer_age_s'] == .1
    c.stop(latch=False)
    assert not c.state()['state']['latched']


def test_fallen_completed_observation_is_retained_under_fault(control):
    c, clock = control
    publish(c)
    clock.now += .1
    with pytest.raises(ValueError, match='fallen'):
        publish(c, step=2, sim_time=.01, fallen=True)
    state = c.state()['state']
    assert state['step'] == 2 and state['support']['step'] == 2
    assert state['fallen'] and state['latched'] and state['controller_status'] == 'fault'
    assert state['received_monotonic_s'] == clock.now and state['producer_age_s'] == 0.
