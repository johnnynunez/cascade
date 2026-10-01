"""Release observation uses one bounded window and never commands settling."""
import copy

import numpy as np
import pytest

from cascade.skills import release_episode as release
from cascade.types import RobotState, SafetyViolation, SkillError
from test_occupancy_payload import PROP
from test_release_episode import atomic_attachment, case, jaws
from test_release_open_sync import Clock


def stream(monkeypatch, produce):
    rt, ep, events = case()
    release.open_hand(rt, ep)
    events.clear()
    clock = Clock()
    monkeypatch.setattr(release, 'time', clock)
    calls = []
    step = ep['clock_validator'].step

    def read(**kwargs):
        nonlocal step
        row = produce(len(calls))
        step += row.get('steps', 6)
        pc = dict(ep['clock'], physics_step=step, sim_time=step/120.)
        q = rt.arm.raw.q.copy()
        q[-1] += row.get('offset', 0.)
        state = RobotState(q=q, dq=np.zeros(3), gripper_joints=jaws(row.get('jaw', .05)), physics_clock=pc)
        atomic_attachment(state, paths=row.get('paths', ()), signs=rt.arm.raw._signs)
        if 'fault' in row:
            row['fault'](state)
        calls.append({'timeout': kwargs['timeout_s'], 'state': copy.deepcopy(state), 'wall_before': clock.now})
        if 'rpc_finish_at' in row:
            # Deliberately late successful transport reply for boundary tests.
            clock.now = row['rpc_finish_at']
        else:
            clock.sleep(min(row.get('rpc_wall_s', 0.), kwargs['timeout_s']))
        return state

    rt.arm.raw.get_state = read
    return rt, ep, events, clock, calls


def assert_pending_only(ep, events):
    assert not events and not ep['released'] and ep['barrier'] is None
    assert ep['withdrawal_counter'] is None and ep['q_failure'] is None


def test_transient_reanchors_then_observes_fixed_physical_window(monkeypatch):
    offsets = [0., .0006, .0012, .0014, .0015]
    rt, ep, events, clock, calls = stream(monkeypatch, lambda i: {'offset': offsets[i]})
    observed = []
    original = ep['clock_validator'].observe
    ep['clock_validator'].observe = lambda value: (observed.append(value), original(value))[1]
    monkeypatch.setattr(rt.arm.harness.occupancy, 'wait_released_ready',
                        lambda *a, **k: pytest.fail('opening must not depend on mapper'))
    start = clock.now
    release.wait_open(rt, ep, timeout_s=8.)
    assert len(calls) == len(observed) == 5
    assert clock.now-start == pytest.approx(.2)
    assert all(x['timeout'] == 1. for x in calls)
    assert_pending_only(ep, events)


def test_sub_milliradian_increments_cannot_slide_anchor_until_acceptance(monkeypatch):
    rt, ep, events, clock, calls = stream(monkeypatch, lambda i: {'offset': .0006*i, 'rpc_wall_s': .15})
    start = clock.now
    with pytest.raises(SkillError, match='deadline'):
        release.wait_open(rt, ep, timeout_s=8.)
    assert clock.now-start == pytest.approx(8.)
    assert len(calls) > 10 and all(0 < x['timeout'] <= 1. for x in calls)
    assert_pending_only(ep, events)


def test_duplicate_steps_do_not_count_as_distinct_stability_samples(monkeypatch):
    steps = [6, 0, 0, 6, 0, 6]
    rt, ep, events, clock, calls = stream(monkeypatch, lambda i: {'steps': steps[i]})
    release.wait_open(rt, ep, timeout_s=8.)
    assert len(calls) == 6
    assert_pending_only(ep, events)


def test_two_samples_spanning_hold_window_are_not_enough(monkeypatch):
    steps = [6, 12, 0, 6]
    rt, ep, events, clock, calls = stream(monkeypatch, lambda i: {'steps': steps[i]})
    release.wait_open(rt, ep, timeout_s=8.)
    assert len(calls) == 4
    assert_pending_only(ep, events)


def test_reused_feedback_array_cannot_move_the_fixed_anchor(monkeypatch):
    rt, ep, events, clock, calls = stream(monkeypatch,
        lambda i: {'offset': .0006*i, 'rpc_wall_s': .15})
    original = rt.arm.raw.get_state
    shared = np.empty(3)
    def read(**kwargs):
        state = original(**kwargs)
        shared[:] = state.q
        state.q = shared
        return atomic_attachment(state)
    rt.arm.raw.get_state = read
    with pytest.raises(SkillError, match='deadline'):
        release.wait_open(rt, ep, timeout_s=8.)
    assert_pending_only(ep, events)


def test_frozen_clock_expires_without_dwell_or_extended_deadline(monkeypatch):
    rt, ep, events, clock, calls = stream(monkeypatch,
        lambda i: {'steps': 6 if i == 0 else 0, 'rpc_wall_s': .15})
    start = clock.now
    with pytest.raises(SkillError, match='deadline'):
        release.wait_open(rt, ep, timeout_s=8.)
    assert clock.now-start == pytest.approx(8.)
    assert_pending_only(ep, events)


def test_large_gap_restarts_dwell_even_if_positions_are_identical(monkeypatch):
    steps = [6, 24, 6, 6]
    rt, ep, events, clock, calls = stream(monkeypatch, lambda i: {'steps': steps[i]})
    release.wait_open(rt, ep, timeout_s=8.)
    assert len(calls) == 4
    assert_pending_only(ep, events)


@pytest.mark.parametrize('physical_steps', [[120, 132, 144], [120, 133, 139, 145]])
def test_hold_boundary_uses_numeric_slack_but_extra_step_restarts(monkeypatch, physical_steps):
    # 132/120 - 120/120 is 0.10000000000000009, but still exactly 12 steps.
    rt, ep, events, clock, calls = stream(monkeypatch, lambda i: {
        'steps': physical_steps[i]-ep['clock_validator'].step if i < len(physical_steps) else 6})
    release.wait_open(rt, ep, timeout_s=8.)
    assert [row['state'].physics_clock['physics_step'] for row in calls] == physical_steps
    assert_pending_only(ep, events)


@pytest.mark.parametrize('restart', ['drift', 'gap'])
@pytest.mark.parametrize('regression', ['jaws', 'attachment'])
def test_eligibility_latch_survives_candidate_window_restart(monkeypatch, restart, regression):
    rows = [{}, {'offset': .002} if restart == 'drift' else {'steps': 24},
            {'jaw': .048} if regression == 'jaws' else {'paths': [PROP]}]
    rt, ep, events, clock, calls = stream(monkeypatch, lambda i: rows[i])
    with pytest.raises(SafetyViolation, match='regressed'):
        release.wait_open(rt, ep, timeout_s=8.)
    assert len(calls) == 3
    assert_pending_only(ep, events)


def test_same_step_attachment_cannot_turn_empty_without_advancing(monkeypatch):
    rows = [{'paths': [PROP]}, {'steps': 0}]
    rt, ep, events, clock, calls = stream(monkeypatch, lambda i: rows[i])
    with pytest.raises(SafetyViolation, match='without a new physics step'):
        release.wait_open(rt, ep, timeout_s=8.)
    assert len(calls) == 2
    assert_pending_only(ep, events)


@pytest.mark.parametrize('field', ['q', 'jaws', 'dq'])
def test_same_step_feedback_change_is_terminal(monkeypatch, field):
    change = {'q': {'offset': .0001}, 'jaws': {'jaw': .0499},
              'dq': {'fault': lambda s: setattr(s, 'dq', np.full(3, .001))}}[field]
    rows = [{}, {'steps': 0, **change}]
    rt, ep, events, clock, calls = stream(monkeypatch, lambda i: rows[i])
    with pytest.raises(SafetyViolation, match='without a new physics step'):
        release.wait_open(rt, ep, timeout_s=8.)
    assert len(calls) == 2
    assert_pending_only(ep, events)


@pytest.mark.parametrize('fault', [
    lambda s: setattr(s, 'dq', None),
    lambda s: setattr(s, 'dq', np.zeros(4)),
    lambda s: setattr(s, 'dq', np.array([0., np.nan, 0.])),
    lambda s: setattr(s, 'dq', np.zeros(3, dtype=bool)),
    lambda s: setattr(s, 'dq', [[0.], [0., 0.]]),
    lambda s: delattr(s, 'dq'),
    lambda s: setattr(s, 'q', np.array([0., np.inf, 0.])),
    lambda s: setattr(s, 'q', [[0.], [0., 0.]]),
    lambda s: delattr(s, 'q'),
    lambda s: delattr(s, 'attachment'),
    lambda s: s.attachment.update(tracking=False),
    lambda s: s.attachment.update(error='sensor unavailable'),
    lambda s: s.attachment.update(source=('other', 1)),
    lambda s: s.attachment.update(producer_epoch='other'),
    lambda s: s.attachment.update(physics_step=True),
    lambda s: s.attachment.update(sim_time=s.attachment['sim_time']+.1),
    lambda s: s.attachment.update(sensor_channel='other'),
    lambda s: s.attachment.update(q=[0., 0., 0.]),
    lambda s: s.attachment.update(q=[[0.], [0., 0.]]),
    lambda s: s.attachment.update(gripper_joints={'position_m': np.zeros(3)}),
    lambda s: s.attachment.update(gripper_joints=jaws(.04)),
    lambda s: s.attachment.update(paths=None),
    lambda s: s.attachment.update(paths=[PROP, PROP]),
    lambda s: s.attachment.update(paths=['/World_Props/other']),
])
def test_invalid_atomic_state_is_terminal_on_first_read(monkeypatch, fault):
    rt, ep, events, clock, calls = stream(monkeypatch, lambda i: {'fault': fault})
    with pytest.raises(SafetyViolation):
        release.wait_open(rt, ep, timeout_s=8.)
    assert len(calls) == 1
    assert_pending_only(ep, events)


@pytest.mark.parametrize('change', ['signs', 'config_signs', 'dof'])
def test_joint_convention_change_refuses_before_state_request(monkeypatch, change):
    rt, ep, events, clock, calls = stream(monkeypatch, lambda i: {})
    if change == 'signs': rt.arm.raw._signs[-1] = -1
    if change == 'config_signs': rt.cfg.arm._data['joint_signs'][-1] = -1
    if change == 'dof': rt.arm.raw.n_joints = 4
    with pytest.raises(SafetyViolation, match='identity changed'):
        release.wait_open(rt, ep, timeout_s=8.)
    assert not calls
    assert_pending_only(ep, events)


@pytest.mark.parametrize('hold', [None, float('nan'), float('inf'), 0., -.1, True])
def test_invalid_physical_window_cannot_disable_stability(monkeypatch, hold):
    rt, ep, events, clock, calls = stream(monkeypatch, lambda i: {})
    rt.arm.raw.settle_hold_s = hold
    with pytest.raises(SafetyViolation, match='physical hold'):
        release.wait_open(rt, ep, timeout_s=8.)
    assert not calls
    assert_pending_only(ep, events)


def test_hold_window_is_captured_once_not_shortened_during_poll(monkeypatch):
    rt, ep, events, clock, calls = stream(monkeypatch, lambda i: {})
    original = rt.arm.raw.get_state
    def read(**kwargs):
        rt.arm.raw.settle_hold_s = .001
        return original(**kwargs)
    rt.arm.raw.get_state = read
    release.wait_open(rt, ep, timeout_s=8.)
    assert len(calls) == 3
    assert calls[-1]['state'].physics_clock['sim_time']-calls[0]['state'].physics_clock['sim_time'] == pytest.approx(.1)
    assert_pending_only(ep, events)


def test_zero_deadline_makes_no_state_request(monkeypatch):
    rt, ep, events, clock, calls = stream(monkeypatch, lambda i: {})
    with pytest.raises(SkillError, match='deadline'):
        release.wait_open(rt, ep, timeout_s=0.)
    assert not calls
    assert_pending_only(ep, events)


@pytest.mark.parametrize('lateness', [0., .1])
def test_eligible_window_cannot_return_at_or_after_wall_deadline(monkeypatch, lateness):
    rt, ep, events, clock, calls = stream(monkeypatch,
        lambda i: {'rpc_finish_at': start+8.+lateness} if i == 2 else {})
    start = clock.now
    with pytest.raises(SkillError, match='deadline'):
        release.wait_open(rt, ep, timeout_s=8.)
    assert len(calls) == 3
    assert_pending_only(ep, events)


def test_completed_open_window_does_not_authorize_later_geometry_drift(monkeypatch):
    rt, ep, events, clock, calls = stream(monkeypatch, lambda i: {})
    release.wait_open(rt, ep, timeout_s=8.)
    original = rt.arm.harness.occupancy.wait_released_ready
    def geometry(*args, **kwargs):
        result = original(*args, **kwargs)
        rt.arm.raw.q[-1] += .002
        return result
    rt.arm.harness.occupancy.wait_released_ready = geometry
    with pytest.raises(SafetyViolation, match='moved while waiting for geometry'):
        release.wait_geometry(rt, ep)
    assert not events and ep['barrier'] is None and ep['withdrawal_counter'] is None
