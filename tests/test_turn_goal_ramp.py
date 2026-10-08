"""Candidate revision 3 of the H2 turn: ramp the commanded yaw rate down on MEASURED yaw.

Software fixtures only. They pin the control law, the opt-in in-envelope
``scale_velocity`` transport primitive and every fail-closed path; they prove no
gait, no settle and no physical acceptance (docs/HUMANOID_H2.md, revision 3 is
NOT physically measured).
"""
from __future__ import annotations

import copy
from dataclasses import replace
import math
from types import SimpleNamespace as NS

import pytest

from cascade.control.mobile_base import VelocityCommand
from cascade.control.mock_base import MockMobileBase
from cascade.safety.base_harness import SafeBase
from mobile_support_fixture import support, support_contract
from mobile_tick_fixture import healthy_episode_gc as healthy_episode_gc
from test_mobile_bridge import Clock, command, publish
from test_mobile_safety import limits

RAMP = dict(decel_rad_s2=.4, min_rate_rad_s=.15, rate_step_rad_s=.025)
pytestmark = pytest.mark.usefixtures('healthy_episode_gc')


def h2_limits(**changes):
    """The H2 profile's turn numbers; the fixture's wall lease covers a slow runner."""
    values = dict(max_wz=.6, turn_speed_rad_s=.5, turn_tolerance_rad=.03, max_turn_angle_rad=1.,
                  max_duration_s=3., max_state_age_s=.5, max_no_progress_s=.4, max_wall_duration_s=60.)
    values.update(changes)
    return limits(**values)


def control(ramp=RAMP):
    result = dict(max_translation_path_m=.35, min_height_m=.615, max_tilt_rad=.5236)
    if ramp is not None:
        result['goal_ramp'] = copy.deepcopy(ramp)
    return result


class RampFeedback(MockMobileBase):
    """Toy yaw kinematics with physics-shaped synthetic support; never a gait.

    ``tracking`` scales the REPORTED yaw so a robot slower than its command can be
    exercised; ``change(state, index)`` injects faults into one sample.
    """

    def __init__(self, *, tracking=1., change=None):
        super().__init__(wall_lease_s=60., auto_step=False)
        self.tracking = tracking
        self.change = change or (lambda state, index: state)
        self.calls, self.index = [], 0

    @property
    def metadata(self):
        return {**super().metadata, 'measurement_kind': 'physics'}

    def command_velocity(self, command, *, generation):
        self.calls.append(('command_velocity', command.as_dict(), generation))
        return super().command_velocity(command, generation=generation)

    def scale_velocity(self, scale, *, generation):
        self.calls.append(('scale_velocity', scale, generation))
        return super().scale_velocity(scale, generation=generation)

    def stop(self, *, latch=True):
        self.calls.append(('stop', latch))
        return super().stop(latch=latch)

    def get_state(self):
        self.advance(.02)
        state = super().get_state()
        self.index += 1
        yaw = self.tracking * SafeBase._yaw(state)
        observed = {**support(state.step, state.sim_time_s), 'epoch': state.epoch, 'model_identity_sha256': 'e' * 64}
        state = replace(state, measurement_kind='physics', model_identity_sha256='e' * 64, support=observed,
                        position_world=(state.position_world[0], state.position_world[1], 1.),
                        orientation_wxyz=(math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)))
        return self.change(state, self.index)


def guarded(raw, ramp=RAMP, **changes):
    return SafeBase(raw, h2_limits(**changes), turn_control=control(ramp), support_contract=support_contract())


def kinds(raw, kind):
    return [call for call in raw.calls if call[0] == kind]


def test_goal_ramp_contract_is_exact_bounded_and_kept_explicit():
    safe = guarded(RampFeedback())
    assert safe.turn_control == control(RAMP) and safe.turn_control['goal_ramp'] == RAMP
    # An explicit null is OFF, so an `extends:` child can A/B the ramp away.
    assert guarded(RampFeedback(), ramp=None).turn_control == control(None)
    off = SafeBase(RampFeedback(), h2_limits(), turn_control={**control(None), 'goal_ramp': None},
                   support_contract=support_contract())
    assert off.turn_control['goal_ramp'] is None
    for bad in ({}, {'decel_rad_s2': .4, 'min_rate_rad_s': .15}, {**RAMP, 'extra': 1.},
                {**RAMP, 'decel_rad_s2': 0.}, {**RAMP, 'min_rate_rad_s': -.1},
                {**RAMP, 'min_rate_rad_s': .5},     # not below the admitted turn speed
                {**RAMP, 'rate_step_rad_s': .4},    # one step larger than the whole ramp
                {**RAMP, 'decel_rad_s2': float('inf')}, {**RAMP, 'min_rate_rad_s': True}, 'fast', [RAMP]):
        raw = RampFeedback()
        with pytest.raises(ValueError):
            guarded(raw, ramp=bad)
        assert not raw.connected and raw.calls == []


@pytest.mark.parametrize('angle', [.8, -.8])
def test_goal_ramp_decelerates_monotonically_on_measured_yaw_then_stops_at_zero_twist(angle):
    raw = RampFeedback()
    safe = guarded(raw)
    safe.connect()
    try:
        result = safe.turn(angle)
        assert result['execution_ok'], result.get('error')
        assert result['outcome'] == 'unverified' and not result['ok']
        assert abs(result['measured_angle_rad'] - angle) <= .03
        # ONE admission: the profile's turn speed for the profile's command budget ...
        assert kinds(raw, 'command_velocity') == [
            ('command_velocity', {'vx': 0., 'vy': 0., 'wz': math.copysign(.5, angle), 'duration_s': 3.}, 0)]
        assert result['command'] == {'vx': 0., 'vy': 0., 'wz': math.copysign(.5, angle), 'duration_s': 3.}
        # ... scaled down INSIDE that admission: same generation, never a new one ...
        scales = kinds(raw, 'scale_velocity')
        updates = result['turn_rate_updates']
        assert scales and len(scales) == len(updates) and all(call[2] == 1 for call in scales)
        rates = [update['rate_rad_s'] for update in updates]
        assert [call[1] for call in scales] == pytest.approx([rate / .5 for rate in rates])
        assert all(later < earlier for earlier, later in zip([.5, *rates], rates))  # monotone
        assert all(.15 <= rate < .5 for rate in rates) and rates[-1] == .15         # bounded, floor reached
        decrements = [earlier - later for earlier, later in zip([.5, *rates], rates)]
        assert all(step >= .025 - 1e-12 for step in decrements[:-1])
        for update in updates:  # the law, on the measured remaining angle of that very sample
            remaining = abs(update['remaining_rad'])
            assert remaining > .03 and math.copysign(1., update['remaining_rad']) == math.copysign(1., angle)
            assert update['rate_rad_s'] == pytest.approx(max(.15, math.sqrt(2 * .4 * (remaining - .03))))
        samples = {s['step']: s for s in result['measured']['samples']}
        assert all(samples[u['step']]['generation'] == 1 for u in updates)
        # ... then the UNCHANGED zero-twist stop, after the last update.
        assert raw.calls[-1] == ('stop', False)
        assert result['commanded_rate_at_stop_rad_s'] == .15
        after = raw.get_state()
        assert after.angular_velocity_body == (0., 0., 0.) and after.controller_status == 'ready'
        assert not safe.latched
    finally:
        safe.disconnect()


def test_goal_ramp_never_drops_below_its_floor_so_a_lagging_robot_still_reaches_the_goal():
    raw = RampFeedback(tracking=.5)  # measured yaw advances at half the commanded rate
    safe = guarded(raw)
    safe.connect()
    try:
        result = safe.turn(.4)
        assert result['execution_ok'], result.get('error')
        assert abs(result['measured_angle_rad'] - .4) <= .03
        rates = [update['rate_rad_s'] for update in result['turn_rate_updates']]
        assert min(rates) == .15
        commanded = [abs(s['angular_velocity_body'][2]) for s in result['measured']['samples']
                     if s['controller_status'] == 'active']
        assert commanded and min(commanded) >= .15 - 1e-12  # held at the floor until the goal tolerance
        assert raw.calls[-1] == ('stop', False)
    finally:
        safe.disconnect()


def test_goal_ramp_never_reaccelerates_when_the_measured_yaw_falls_back():
    def fall_back(state, index):  # every 7th sample reads 0.05 rad behind (a yaw oscillation dip)
        if index % 7:
            return state
        yaw = SafeBase._yaw(state) - .05
        return replace(state, orientation_wxyz=(math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)))

    raw = RampFeedback(change=fall_back)
    safe = guarded(raw)
    safe.connect()
    try:
        result = safe.turn(.8)
        assert result['execution_ok'], result.get('error')
        rates = [update['rate_rad_s'] for update in result['turn_rate_updates']]
        dips = [u for u in result['turn_rate_updates'] if u['step'] % 7 == 0]
        assert len(rates) >= 5 and not dips  # a dip never triggers an update, let alone a raise
        assert all(later < earlier for earlier, later in zip([.5, *rates], rates)) and rates[-1] == .15
        assert [call[1] for call in kinds(raw, 'scale_velocity')] == pytest.approx([r / .5 for r in rates])
    finally:
        safe.disconnect()


@pytest.mark.parametrize('fault', ['stale', 'missing'])
def test_goal_ramp_fails_closed_on_stale_or_missing_state_without_extrapolating(fault):
    seen = {}

    class Faulty(RampFeedback):
        def get_state(self):
            if seen.get('scales') is None and kinds(self, 'scale_velocity'):
                seen['scales'] = len(kinds(self, 'scale_velocity'))
                if fault == 'missing':
                    raise RuntimeError('state channel lost')
            state = super().get_state()
            if seen.get('scales') is not None and 'stale' not in seen:
                seen['stale'] = state.step
                return replace(state, producer_age_s=1.)  # older than the profile's 0.5 s budget
            return state

    raw = Faulty()
    safe = guarded(raw)
    safe.connect()
    try:
        result = safe.turn(.8)
        assert not result['execution_ok']
        assert ('stale' if fault == 'stale' else 'state channel lost') in result['error']
        # No update was computed from the faulty sample or invented after it.
        assert len(kinds(raw, 'scale_velocity')) == seen['scales'] == len(result['turn_rate_updates'])
        assert raw.calls[-1] == ('stop', True) and safe.latched
    finally:
        safe.disconnect()


def test_goal_ramp_turn_is_refused_before_motion_when_the_backend_cannot_scale():
    class Unscalable(RampFeedback):
        velocity_scaling = False

    raw = Unscalable()
    safe = guarded(raw)
    safe.connect()
    try:
        result = safe.turn(.8)
        assert not result['execution_ok'] and 'velocity scaling' in result['error']
        assert raw.calls == [] and raw.index == 0 and not safe.latched  # no read, no command, no stop
        assert result['requested_angle_rad'] == .8 and result['measured_angle_rad'] is None
    finally:
        safe.disconnect()


def test_goal_ramp_scale_refusal_latches_and_existing_vetoes_still_fire():
    class Refusing(RampFeedback):
        def scale_velocity(self, scale, *, generation):
            self.calls.append(('scale_velocity', scale, generation))
            return {'ok': False, 'error': 'refused by backend'}

    raw = Refusing()
    safe = guarded(raw)
    safe.connect()
    try:
        result = safe.turn(.8)
        assert not result['execution_ok'] and 'refused by backend' in result['error']
        assert len(kinds(raw, 'scale_velocity')) == 1 and raw.calls[-1] == ('stop', True) and safe.latched
    finally:
        safe.disconnect()

    def drift(state, index):  # the unchanged translation veto still owns the ramp phase
        return replace(state, position_world=(.01 * max(0, index - 40), 0., 1.))

    raw = RampFeedback(change=drift)
    safe = guarded(raw)
    safe.connect()
    try:
        result = safe.turn(.8)
        assert not result['execution_ok'] and 'translation path' in result['error']
        assert kinds(raw, 'scale_velocity') and raw.calls[-1] == ('stop', True) and safe.latched
    finally:
        safe.disconnect()


#: Result keys of a default-off turn, measured on the pre-change tree (c5012e7) with this fixture.
GOLDEN_TURN_KEYS = ['ack', 'command', 'execution_ok', 'measured', 'measured_angle_rad',
                    'measured_translation_path_m', 'ok', 'outcome', 'reason', 'requested_angle_rad', 'stop_ack']


def test_default_off_turns_keep_the_golden_command_sequence_and_only_the_h2_candidate_opts_in():
    from cascade.config import CONFIG_DIR, load_profile

    names = sorted(path.stem for path in (CONFIG_DIR / 'bases').glob('*.yaml'))
    profiles = {name: load_profile('bases', name).as_dict() for name in names}
    ramped = {name for name, p in profiles.items() if (p.get('turn_control') or {}).get('goal_ramp')}
    assert ramped == {'h2_velocity_candidate'}
    for name, profile in profiles.items():
        if 'turn' not in profile['capabilities']:
            continue
        # Production limits except the WALL budget: a fixture lease for slow runners only
        # (the command sequence and the sim-time budget do not depend on it).
        safety = dict(profile['safety'], max_wall_duration_s=max(60., profile['safety']['max_wall_duration_s']))
        angle = -min(.3, safety['max_turn_angle_rad'])
        admitted = ('command_velocity', {'vx': 0., 'vy': 0., 'wz': -safety['turn_speed_rad_s'],
                                         'duration_s': safety['max_duration_s']}, 0)
        turn = profile.get('turn_control')
        # The ramped profile is also run with an explicit null ramp (the extends: A/B switch), and the
        # budgeted one with an explicit null budget (revision 3, B29b's A/B switch).
        variants = [turn] + ([{**turn, 'goal_ramp': None}] if name in ramped else [])
        if (turn or {}).get('goal_ramp') and turn['goal_ramp'].get('time_budget'):
            variants.append({**turn, 'goal_ramp': {**turn['goal_ramp'], 'time_budget': None}})
        for variant in variants:
            raw = RampFeedback()
            safe = SafeBase(raw, safety, turn_control=variant,
                            support_contract=support_contract() if variant is not None else None)
            safe.connect()
            try:
                result = safe.turn(angle)
                assert result['execution_ok'], (name, result.get('error'))
                if (variant or {}).get('goal_ramp'):
                    assert raw.calls[0] == admitted and kinds(raw, 'scale_velocity'), name
                    assert raw.calls[-1] == ('stop', False)
                    budget = ['turn_rate_budget'] if variant['goal_ramp'].get('time_budget') else []
                    assert sorted(result) == sorted(GOLDEN_TURN_KEYS + ['turn_rate_updates',
                                                                        'commanded_rate_at_stop_rad_s', *budget])
                else:  # byte-for-byte the pre-ramp sequence and result schema
                    assert raw.calls == [admitted, ('stop', False)], name
                    assert sorted(result) == GOLDEN_TURN_KEYS, name
            finally:
                safe.disconnect()


def test_candidate_ramp_finishes_the_longest_turn_inside_the_unchanged_budget_at_measured_tracking():
    """The YAML derivation through the real loop: the longest admitted H2 turn (1.0 rad) on a toy
    that tracks 0.465 of every 0.5 rad/s (geo3-v2) still ends inside the unchanged 3 s command."""
    from cascade.config import load_profile

    profile = load_profile('bases', 'h2_velocity_candidate').as_dict()
    safety = dict(profile['safety'], max_wall_duration_s=60.)  # wall lease for slow runners; sim budget unchanged
    raw = RampFeedback(tracking=.465 / .5)
    safe = SafeBase(raw, safety, turn_control=profile['turn_control'], support_contract=support_contract())
    safe.connect()
    try:
        result = safe.turn(safety['max_turn_angle_rad'])
        assert result['execution_ok'], result.get('error')
        used = result['measured']['after']['sim_time_s'] - result['ack']['start_sim_time_s']
        assert used < safety['max_duration_s'] - .3  # >= 0.3 s of the 3 s command left
        assert len(result['turn_rate_updates']) <= 14 and result['commanded_rate_at_stop_rad_s'] == .15
    finally:
        safe.disconnect()


def test_turn_tool_surface_is_unchanged_and_runtime_wiring_carries_the_ramp(tmp_path):
    from cascade.apps.mobile_runtime import build_mobile_runtime
    from cascade.config import load_demo_config, load_profile
    from cascade.skills.mobile_runtime import TOOL_SPECS, tool_specs_for_profiles

    candidate = load_profile('bases', 'h2_velocity_candidate').as_dict()
    # Revision 3's ramp, plus revision 4's opt-in time budget (B29b, tests/test_turn_goal_ramp_budget.py).
    assert {k: v for k, v in candidate['turn_control']['goal_ramp'].items() if k != 'time_budget'} == RAMP
    without = copy.deepcopy(candidate)
    del without['turn_control']['goal_ramp']
    specs = tool_specs_for_profiles([candidate])
    assert specs == tool_specs_for_profiles([without])
    turn = next(spec for spec in specs if spec['name'] == 'turn')
    assert turn == next(spec for spec in TOOL_SPECS if spec['name'] == 'turn')
    assert set(turn['parameters']['properties']) == {'base', 'angle_rad'}
    assert turn['parameters']['required'] == ['angle_rad']
    cfg = load_demo_config(base='microduck_mock', llm='mock')
    cfg._data['bases'][0]['turn_control'] = control(RAMP)
    runtime, owner = build_mobile_runtime(cfg, tmp_path)
    try:
        assert owner.primary.turn_control['goal_ramp'] == RAMP
        assert not owner.primary.raw.connected
    finally:
        runtime.close()


def scaling_bridge(**changes):
    from cascade.sim.mobile_bridge import MobileBridgeController
    clock = Clock()
    return MobileBridgeController(
        robot_id='duck', source='isolated-bridge', engine='physx', device='cuda:0',
        asset_sha256='a' * 64, policy_sha256='b' * 64, model_identity_sha256='e' * 64,
        support_contract=support_contract(), max_linear_speed=.2, max_angular_speed=.8, max_duration_s=10.,
        lease_s=.5, max_state_age_s=.5, clock=clock, **changes), clock


def scale(c, **changes):
    request = {'robot_id': 'duck', 'source': 'isolated-bridge', 'epoch': c.hello()['epoch'],
               'generation': c.hello()['generation'], 'owner': 'test-client', 'command_id': 'move-1', 'scale': .5}
    request.update(changes)
    return c.scale_velocity(request)


def test_bridge_velocity_scaling_is_opt_in_and_stays_inside_the_admitted_envelope():
    off, _ = scaling_bridge()
    assert off.hello()['capabilities'] == ['state', 'velocity', 'stop', 'reset_stop']  # default unchanged
    publish(off)
    command(off, vx=0., wz=.5)
    with pytest.raises(ValueError, match='not enabled'):
        scale(off)
    assert off.control_at(.01) == (0., 0., .5)

    c, clock = scaling_bridge(velocity_scaling=True)
    assert c.hello()['capabilities'] == ['state', 'velocity', 'stop', 'reset_stop', 'velocity_scaling']
    publish(c)
    with pytest.raises(ValueError, match='no active command'):
        scale(c)
    ack = command(c, vx=0., wz=.5)
    generation = ack['generation']
    for bad in ({'scale': 0.}, {'scale': -.1}, {'scale': 1.01}, {'scale': float('nan')}, {'scale': True},
                {'scale': '0.5'}, {'owner': 'intruder'}, {'command_id': 'other'},
                {'generation': generation - 1}, {'epoch': 'old'}, {'robot_id': 'other'}):
        with pytest.raises(ValueError):
            scale(c, **bad)
    assert c.control_at(.01) == (0., 0., .5)  # every refusal left the admitted twist untouched
    scaled = scale(c, scale=.3)
    assert scaled['accepted'] is True and scaled['generation'] == generation  # not an admission
    assert scaled['end_sim_time_s'] == ack['end_sim_time_s'] and scaled['latched'] is False
    assert c.control_at(.02) == pytest.approx((0., 0., .15))
    assert c.state()['state']['generation'] == generation
    scale(c, scale=1.)  # never above the admitted twist
    assert c.control_at(.03) == pytest.approx((0., 0., .5))
    done = ack['end_sim_time_s']
    c.control_at(done)  # normal completion: never extended by a scale
    with pytest.raises(ValueError, match='no active command'):
        scale(c)
    second = command(c, command_id='move-2', vx=0., wz=-.5)
    c.stop(latch=False)  # a stop consumes the generation every later scale must carry
    with pytest.raises(ValueError):
        scale(c, command_id='move-2', generation=second['generation'])
    assert c.control_at(done + .01) == (0., 0., 0.)  # (sim clock advances; a regression would fault)
    third = command(c, command_id='move-3', vx=0., wz=.5)
    clock.now += 1.  # stale physics state: refused, twist untouched
    with pytest.raises(ValueError, match='stale'):
        scale(c, command_id='move-3', generation=third['generation'])
    publish(c, step=2, sim_time=.01)  # fresh state again, but the 0.5 s wall lease has expired
    with pytest.raises(ValueError, match='expired'):
        scale(c, command_id='move-3', generation=third['generation'])
    assert c.hello()['generation'] == third['generation'] and not c.state()['latched']


def test_isaac_base_scales_only_an_advertised_active_command_over_the_real_socket():
    from cascade.control.isaac_base import IsaacBase
    from cascade.sim.bridge_client import BridgeClient, BridgeError
    from cascade.sim.mobile_bridge import MobileBridgeController, MobileBridgeServer
    from test_isaac_base import profile

    for enabled in (False, True):
        c = MobileBridgeController(
            robot_id='duck', source='isolated-bridge', engine='physx', device='cuda:0',
            asset_sha256='a' * 64, policy_sha256='b' * 64, model_identity_sha256='e' * 64,
            support_contract=support_contract(), max_linear_speed=.2, max_angular_speed=.8,
            max_duration_s=5., lease_s=1., max_state_age_s=5., max_action_wall_s=10., velocity_scaling=enabled)
        server = MobileBridgeServer(c, port=0)
        server.start()
        raw = IsaacBase(profile(server.address[1], timeout_s=1.))
        try:
            assert raw.velocity_scaling is False  # unknown before the hello
            raw.connect()
            assert raw.velocity_scaling is enabled
            publish(c)
            before = raw.get_state()
            ack = raw.command_velocity(VelocityCommand(0., 0., .5, 1.), generation=before.generation)
            assert ack['accepted'] is True
            result = raw.scale_velocity(.4, generation=ack['generation'])
            if not enabled:
                assert result['ok'] is False and 'velocity_scaling' in result['error']
                assert c.control_at(.01) == (0., 0., .5)
                continue
            assert result['ok'] is True and result['generation'] == ack['generation']
            assert c.control_at(.01) == pytest.approx((0., 0., .2))
            # Only the owner's control channel may scale; a reader cannot.
            reader = BridgeClient(*server.address, timeout_s=1.)
            reader.connect()
            try:
                reader.request({'op': 'hello', 'role': 'reader'})
                with pytest.raises(BridgeError, match='forbidden'):
                    reader.request({'op': 'scale_velocity', 'scale': .1, 'generation': ack['generation']})
            finally:
                reader.close()
            assert c.control_at(.02) == pytest.approx((0., 0., .2))
            assert raw.stop(latch=False)['ok']
            late = raw.scale_velocity(.2, generation=ack['generation'])  # crossing a stop: refused
            assert late['ok'] is False
            assert c.control_at(.03) == (0., 0., 0.)
        finally:
            raw.disconnect()
            server.close()


def test_h2_owner_offers_velocity_scaling_only_behind_its_explicit_flag(tmp_path, monkeypatch, contract):
    import json
    from cascade.sim import h2_identity
    from cascade.sim.bridge_client import BridgeClient
    from cascade.sim.mobile_bridge import MobileBridgeServer
    from test_h2_owner import REPO, SoftwareBackend, _software_identity, cli

    required = ['--release', 'r', '--policy', 'p', '--limits', 'l', '--robot-id', 'h2', '--source', 's',
                '--device', 'cuda:0', '--port', '0', '--out', 'o', '--max-wall-s', '1', '--max-steps', '1']
    assert cli().parse_args(required).velocity_scaling is False
    assert cli().parse_args(required + ['--velocity-scaling']).velocity_scaling is True
    monkeypatch.setattr(h2_identity, 'build_h2_model_identity', _software_identity)
    manifest = json.loads((REPO / 'configs/h2/bundle.json').read_text())
    admission = {'bundle': manifest, 'asset_sha256': manifest['files']['H2.usda']['sha256'],
                 'policy_sha256': manifest['files']['policy.pt']['sha256'], 'limits': json.loads(
                     (REPO / 'configs/h2/controller-limits-candidate.json').read_text())}
    for flag in (False, True):
        seen = {}

        class Backend(SoftwareBackend):
            def __init__(self, *unused):
                super().__init__(contract)
                self.receipt['support_contract'] = {'version': 1, 'robot_shapes': ['/World/H2/pelvis/c'],
                                                    'foot_shapes': ['/World/H2/pelvis/c'],
                                                    'ground_shapes': ['/World/Ground'],
                                                    'gravity_world_m_s2': [0., 0., -9.81]}

        class Server(MobileBridgeServer):
            def start(self):
                super().start()
                client = BridgeClient(self.address[0], self.address[1], timeout_s=1.)
                client.connect()
                try:
                    seen['hello'] = client.request({'op': 'hello', 'role': 'reader'})
                finally:
                    client.close()

        args = NS(out=tmp_path / f'run-{flag}', device='cuda:0', robot_id='h2', source='software-only',
                  max_wall_s=20., max_steps=5, port=0, camera_every=4, max_jpeg_bytes=100000, physics_row_every=1,
                  policy=tmp_path / 'policy.pt', overview_resolution='32x24', camera_eye=(3., -3., 2.),
                  camera_target=(0., 0., .8), ground_visual_m=40., camera_follow=False, velocity_scaling=flag)
        result = cli().run(args, admission, contract, backend_factory=Backend, server_factory=Server)
        assert result['completed'] is True and result['physical_acceptance'] is False
        assert ('velocity_scaling' in seen['hello']['capabilities']) is flag
        startup = json.loads((args.out / 'startup.json').read_text())
        assert startup['arguments']['velocity_scaling'] is flag  # bound into configuration_sha256


@pytest.fixture(scope='module')
def contract():
    from cascade.control.h2_policy_contract import H2PolicyContract
    return H2PolicyContract.from_bundle()
