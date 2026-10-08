"""Candidate revision 4 of the H2 turn (B29b): budget the goal ramp against the command time left.

The owner A/B of revision 3 (docs/evidence/h2-turn-ramp-live-20261008) measured a policy property: the yaw
rate dips below 0.2 rad/s about 1.0-1.75 s into most turns (24 of 39, both arms). A ramp that starts on the
measured remaining yaw after such a dip still spends its whole deceleration and overruns the unchanged 3 s
command (1.0 rad: 1/5 with the ramp, 5/6 without). Software fixtures only: the toy below reproduces that dip
and pins the control law, the one-admission/stop invariants and every fail-closed path. They prove no gait,
no settle and no physical acceptance.
"""
from __future__ import annotations

import math
from dataclasses import replace

import pytest

from cascade.control.mock_base import MockMobileBase
from cascade.safety.base_harness import SafeBase
from mobile_support_fixture import support, support_contract
from mobile_tick_fixture import healthy_episode_gc as healthy_episode_gc
from test_turn_goal_ramp import RAMP, control, h2_limits, kinds

BUDGET = dict(reserve_s=.3, tracking=.93)
RAMP_V2 = {**RAMP, 'time_budget': BUDGET}
DT = .02  # sim seconds per published state of the toy
pytestmark = pytest.mark.usefixtures('healthy_episode_gc')


class DipFeedback(MockMobileBase):
    """Toy yaw kinematics with the measured mid-turn dip; never a gait.

    The REPORTED yaw advances at ``tracking`` x the commanded rate (0.93: 0.465 of 0.5 rad/s, geo3-v2),
    except during ``dip`` (seconds after the admitted start) where it advances at most ``dip_rate``
    whatever the command: the dip is the policy's, present with and without the ramp. Physics-shaped
    synthetic support so the geometric vetoes run; ``change(state, index)`` injects faults.
    """

    def __init__(self, *, tracking=.93, dip=(1., 1.75), dip_rate=.1, change=None):
        super().__init__(wall_lease_s=60., auto_step=False)
        self.tracking, self.dip, self.dip_rate = tracking, dip, dip_rate
        self.change = change or (lambda state, index: state)
        self.calls, self.index, self.yaw, self.start = [], 0, 0., None

    @property
    def metadata(self):
        return {**super().metadata, 'measurement_kind': 'physics'}

    def command_velocity(self, command, *, generation):
        self.calls.append(('command_velocity', command.as_dict(), generation))
        ack = super().command_velocity(command, generation=generation)
        if ack.get('accepted'):
            self.start = ack['start_sim_time_s']
        return ack

    def scale_velocity(self, scale, *, generation):
        self.calls.append(('scale_velocity', scale, generation))
        return super().scale_velocity(scale, generation=generation)

    def stop(self, *, latch=True):
        self.calls.append(('stop', latch))
        return super().stop(latch=latch)

    def get_state(self):
        before = super().get_state()
        wz = before.angular_velocity_body[2]  # the (scaled) admitted twist in force over the next interval
        self.advance(DT)
        if wz:
            rate = self.tracking * abs(wz)
            if self.dip is not None and self.dip[0] <= before.sim_time_s - self.start < self.dip[1]:
                rate = min(rate, self.dip_rate)
            self.yaw += math.copysign(rate, wz) * DT
        state = super().get_state()
        self.index += 1
        observed = {**support(state.step, state.sim_time_s), 'epoch': state.epoch, 'model_identity_sha256': 'e' * 64}
        state = replace(state, measurement_kind='physics', model_identity_sha256='e' * 64, support=observed,
                        position_world=(state.position_world[0], state.position_world[1], 1.),
                        orientation_wxyz=(math.cos(self.yaw / 2), 0., 0., math.sin(self.yaw / 2)))
        return self.change(state, self.index)


def guarded(raw, ramp=RAMP_V2, **changes):
    return SafeBase(raw, h2_limits(**changes), turn_control=control(ramp), support_contract=support_contract())


def run(angle, ramp=RAMP_V2, **feedback):
    raw = DipFeedback(**feedback)
    safe = guarded(raw, ramp)
    safe.connect()
    try:
        result = safe.turn(angle)
        latched, calls = safe.latched, list(raw.calls)
    finally:
        safe.disconnect()
    raw.calls = calls  # the turn's calls; the disconnect's own latched stop is teardown
    return raw, result, latched


def used_s(result):
    return result['measured']['after']['sim_time_s'] - result['ack']['start_sim_time_s']


def assert_one_admission_then_unchanged_stop(raw, result, angle, *, latched):
    """ONE admission of the profile's rate for the profile's command, scaled inside it, then the stop."""
    assert kinds(raw, 'command_velocity') == [
        ('command_velocity', {'vx': 0., 'vy': 0., 'wz': math.copysign(.5, angle), 'duration_s': 3.}, 0)]
    assert result['command'] == {'vx': 0., 'vy': 0., 'wz': math.copysign(.5, angle), 'duration_s': 3.}
    scales, updates = kinds(raw, 'scale_velocity'), result['turn_rate_updates']
    assert len(scales) == len(updates) and all(call[2] == 1 for call in scales)  # same generation, never a new one
    rates = [update['rate_rad_s'] for update in updates]
    assert [call[1] for call in scales] == pytest.approx([rate / .5 for rate in rates])
    assert all(later < earlier for earlier, later in zip([.5, *rates], rates))  # never re-accelerates
    assert all(.15 <= rate < .5 for rate in rates)                               # never below the floor
    samples = {s['step']: s for s in result['measured']['samples']}
    assert all(samples[update['step']]['generation'] == 1 for update in updates)
    assert raw.calls[-1] == ('stop', latched)


def test_time_budget_contract_is_exact_bounded_and_null_is_off():
    safe = guarded(DipFeedback())
    assert safe.turn_control == control(RAMP_V2) and safe.turn_control['goal_ramp']['time_budget'] == BUDGET
    # An explicit null keeps revision 3 (no budget), so an `extends:` child can A/B the budget alone.
    off = guarded(DipFeedback(), ramp={**RAMP, 'time_budget': None})
    assert off.turn_control['goal_ramp'] == {**RAMP, 'time_budget': None}
    assert guarded(DipFeedback(), ramp=RAMP).turn_control['goal_ramp'] == RAMP  # revision 3 unchanged
    for bad in ({}, {'reserve_s': .3}, {'tracking': .93}, {**BUDGET, 'extra': 1.},
                {**BUDGET, 'reserve_s': 0.}, {**BUDGET, 'reserve_s': -.1},
                {**BUDGET, 'reserve_s': 3.},           # the whole command: nothing left to budget
                {**BUDGET, 'tracking': 0.}, {**BUDGET, 'tracking': 1.01},  # never assumes the robot outruns it
                {**BUDGET, 'reserve_s': float('nan')}, {**BUDGET, 'tracking': True}, 'late', [BUDGET]):
        raw = DipFeedback()
        with pytest.raises(ValueError):
            guarded(raw, ramp={**RAMP, 'time_budget': bad})
        assert not raw.connected and raw.calls == []


@pytest.mark.parametrize('angle', [.6, .8, 1., -1.])
def test_the_dip_overruns_the_revision_3_ramp_and_the_time_budget_keeps_the_turn_inside_the_command(angle):
    old_raw, old, old_latched = run(angle, RAMP)
    raw, new, latched = run(angle)
    # Premise, reproduced on the toy: at 1.0 rad the revision-3 ramp starts decelerating only after the
    # dip and runs out of the unchanged 3 s command (the owner's 3 timeouts); shorter turns fit.
    if abs(angle) == 1.:
        assert not old['execution_ok'] and 'before simulation deadline' in old['error']
        assert old_latched and old_raw.calls[-1] == ('stop', True)
    else:
        assert old['execution_ok'], old.get('error')
    # Revision 4: inside the same command for every angle, never by a longer one.
    assert new['execution_ok'], new.get('error')
    assert abs(new['measured_angle_rad'] - angle) <= .03 and not latched
    assert used_s(new) <= 3. - BUDGET['reserve_s'] + 2 * DT
    assert new['measured']['after']['sim_time_s'] <= new['ack']['end_sim_time_s']
    assert_one_admission_then_unchanged_stop(raw, new, angle, latched=False)
    assert .15 <= new['commanded_rate_at_stop_rad_s'] <= .5
    if abs(angle) < 1.:  # the budget never binds: revision 3's ramp, decision for decision
        assert new['turn_rate_budget']['held_samples'] == 0
        assert [u['rate_rad_s'] for u in new['turn_rate_updates']] == [u['rate_rad_s'] for u in old['turn_rate_updates']]
        assert new['commanded_rate_at_stop_rad_s'] == old['commanded_rate_at_stop_rad_s'] == .15


def test_the_budget_law_holds_on_every_update_and_its_cost_is_a_higher_cut_rate():
    # A milder dip (0.2 rad/s, the evidence's own dip threshold) leaves room for part of the ramp:
    # the distance law sets the first updates, the time budget the last one.
    raw, result, _ = run(1., dip_rate=.2)
    assert result['execution_ok'], result.get('error')
    budget, rate, by_budget, by_distance = result['turn_rate_budget'], .5, 0, 0
    for update in result['turn_rate_updates']:
        distance = abs(update['remaining_rad']) - .03
        left = update['command_time_left_s']
        assert left == pytest.approx(result['ack']['end_sim_time_s'] - update['sim_time_s'])
        assert left > BUDGET['reserve_s']  # inside the reserve the rate is only ever held
        need = distance / (BUDGET['tracking'] * (left - BUDGET['reserve_s']))
        assert update['budget_rate_rad_s'] == pytest.approx(need)
        assert update['rate_rad_s'] == pytest.approx(max(.15, min(rate, max(math.sqrt(.8 * distance), need))))
        by_budget += need > math.sqrt(.8 * distance)
        by_distance += need < math.sqrt(.8 * distance)
        rate = update['rate_rad_s']
    assert by_budget and by_distance
    # The budget withheld decelerations the distance law alone would have made, only after the dip ...
    assert budget['held_samples'] > 0
    first = budget['first_held']
    assert first['sim_time_s'] - result['ack']['start_sim_time_s'] >= 1.75
    assert first['budget_rate_rad_s'] > max(.15, first['distance_rate_rad_s'])
    # ... the turn ends inside the command, short of the reserve ...
    assert used_s(result) <= 3. - BUDGET['reserve_s'] + 2 * DT
    # ... and that is the price, stated: the cut happens above the floor (never above the admitted rate).
    assert .15 < result['commanded_rate_at_stop_rad_s'] < .5


@pytest.mark.parametrize('angle', [.6, .8, 1., -.8])
def test_a_budget_that_never_binds_reproduces_the_revision_3_ramp_exactly(angle):
    old_raw, old, _ = run(angle, RAMP, dip=None)
    raw, new, _ = run(angle, dip=None)
    assert old['execution_ok'] and new['execution_ok']
    assert new['turn_rate_budget'] == {'held_samples': 0, 'first_held': None}
    strip = [{k: u[k] for k in ('step', 'sim_time_s', 'remaining_rad', 'rate_rad_s')} for u in new['turn_rate_updates']]
    assert strip == old['turn_rate_updates']
    assert kinds(raw, 'scale_velocity') == kinds(old_raw, 'scale_velocity')
    assert raw.calls == old_raw.calls and new['commanded_rate_at_stop_rad_s'] == .15
    assert sorted(new) == sorted([*old, 'turn_rate_budget'])


@pytest.mark.parametrize('angle', [.8, 1.])
def test_the_budget_never_extends_the_command_and_a_dip_too_long_for_it_still_fails_closed(angle):
    _, old, _ = run(angle, RAMP, dip=(1., 2.6))
    raw, result, latched = run(angle, dip=(1., 2.6))
    assert not old['execution_ok'] and not result['execution_ok']  # no budget can buy back 1.6 s of dip
    assert 'before simulation deadline' in result['error']
    assert latched and raw.calls[-1] == ('stop', True)
    assert_one_admission_then_unchanged_stop(raw, result, angle, latched=True)
    assert all(u['command_time_left_s'] > BUDGET['reserve_s'] for u in result['turn_rate_updates'])
    assert result['turn_rate_budget']['held_samples'] > 0
    # Revision 3 kept decelerating into the deadline; the budget held the rate instead.
    assert len(result['turn_rate_updates']) < len(old['turn_rate_updates'])


def test_the_budget_fails_closed_on_a_stale_state_without_computing_from_it():
    seen = {}

    def stale_after_first_hold(state, index):
        if seen.get('index') is None and state.sim_time_s >= 2.3:
            seen['index'], seen['scales'] = index, len([c for c in raw.calls if c[0] == 'scale_velocity'])
            return replace(state, producer_age_s=1.)  # older than the profile's 0.5 s budget
        return state

    raw = DipFeedback(change=stale_after_first_hold)
    safe = guarded(raw)
    safe.connect()
    try:
        result = safe.turn(1.)
        assert not result['execution_ok'] and 'stale' in result['error']
        assert result['turn_rate_budget']['held_samples'] > 0  # the fault arrived while the budget held
        assert len(kinds(raw, 'scale_velocity')) == seen['scales'] == len(result['turn_rate_updates'])
        assert raw.calls[-1] == ('stop', True) and safe.latched
    finally:
        safe.disconnect()


def test_candidate_values_replayed_on_the_owner_ramp_updates_bind_only_where_the_command_ran_out():
    """The YAML derivation, checked on the committed owner evidence (verifier clock, which starts before the
    admission, so this over-states binding). A decision differs from revision 3 only where revision 3 sent a
    logged update the budget would have withheld; holds between logged updates change nothing."""
    import json

    from cascade.config import load_profile
    from conftest import REPO

    ramp = load_profile('bases', 'h2_velocity_candidate').as_dict()['turn_control']['goal_ramp']
    safety = load_profile('bases', 'h2_velocity_candidate').as_dict()['safety']
    budget, tol, command = ramp['time_budget'], safety['turn_tolerance_rad'], safety['max_duration_s']
    evidence = REPO / 'docs/evidence/h2-turn-ramp-live-20261008'
    profiles = json.loads((evidence / 'yaw_rate_profiles.json').read_text())
    status = {f"{t['run']}:{t['tag']}": t['status'] for t in json.loads((evidence / 'turns.json').read_text())}
    bound = {}
    for key, value in profiles.items():
        for update in value.get('rate_updates') or []:
            distance = max(0., abs(update['remaining_rad']) - tol)
            usable = command - update['t_s'] - budget['reserve_s']
            need = distance / (budget['tracking'] * usable) if usable > 0 else math.inf
            if need > max(ramp['min_rate_rad_s'], math.sqrt(2 * ramp['decel_rad_s2'] * distance)):
                bound[key] = update['t_s']
                break
    ramped = [key for key, value in profiles.items() if value.get('rate_updates')]
    assert len(ramped) == 23
    assert sorted(bound) == ['1:R1-3-+1.0', '2:R1-2--0.8', '2:R1-3-+1.0', '2:R1-4--1.0', '2:R2-3-+1.0']
    assert [status[k] for k in ('1:R1-3-+1.0', '2:R1-3-+1.0', '2:R2-3-+1.0')] == ['unverified'] * 3  # the timeouts
    assert bound['2:R1-2--0.8'] >= 2.49  # only that confirmed 0.8 rad turn's last update
    assert all(bound[k] >= 2. for k in bound)  # never before the dip window has passed


def test_candidate_profile_carries_the_budget_and_an_extends_child_switches_only_it_off(tmp_path):
    import shutil

    import yaml

    from cascade.config import CONFIG_DIR, load_profile

    candidate = load_profile('bases', 'h2_velocity_candidate').as_dict()
    assert candidate['turn_control']['goal_ramp'] == RAMP_V2
    names = sorted(path.stem for path in (CONFIG_DIR / 'bases').glob('*.yaml'))
    budgeted = {n for n in names if ((load_profile('bases', n).as_dict().get('turn_control') or {})
                                     .get('goal_ramp') or {}).get('time_budget')}
    assert budgeted == {'h2_velocity_candidate'}
    # The live A/B switch: a private child that nulls ONLY the budget resolves to revision 3.
    shutil.copytree(CONFIG_DIR / 'bases', tmp_path / 'bases')
    (tmp_path / 'bases' / 'h2_ab_v1.yaml').write_text(yaml.safe_dump(
        {'extends': 'h2_velocity_candidate', 'turn_control': {'goal_ramp': {'time_budget': None}}}))
    child = load_profile('bases', 'h2_ab_v1', tmp_path).as_dict()
    assert child['turn_control']['goal_ramp'] == {**RAMP, 'time_budget': None}
    child['turn_control']['goal_ramp']['time_budget'] = candidate['turn_control']['goal_ramp']['time_budget']
    assert child == candidate  # nothing else differs
    # Both arms through the real loop with the profile's own limits (wall lease for slow runners only):
    # 1.0 rad, the longest admitted turn, with the measured dip at the measured tracking.
    for turn_control, inside in ((candidate['turn_control'], True),
                                 ({**candidate['turn_control'], 'goal_ramp': {**RAMP, 'time_budget': None}}, False)):
        safety = dict(candidate['safety'], max_wall_duration_s=60.)
        raw = DipFeedback()
        safe = SafeBase(raw, safety, turn_control=turn_control, support_contract=support_contract())
        safe.connect()
        try:
            result = safe.turn(safety['max_turn_angle_rad'])
            assert result['execution_ok'] is inside, result.get('error')
            assert kinds(raw, 'command_velocity')[0][1]['duration_s'] == safety['max_duration_s'] == 3.
        finally:
            safe.disconnect()
