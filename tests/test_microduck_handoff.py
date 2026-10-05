"""Standing handoff: software doubles only; no physics, Kit or locomotion claim."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from test_microduck_bridge_cli import cli, digest, forbid_live_imports, software_bundle, software_limits  # noqa: F401
from test_microduck_stepper import command, controller, make_stepper
from test_mobile_identity import recipe_inputs  # noqa: F401

from cascade.control.microduck_handoff import HANDOFF_PROFILES, HANDOFF_RULE, StandingHandoff
from cascade.control.microduck_policy import HOME_Q
from cascade.sim.microduck_policy_admission import target_contract
from cascade.sim.mobile_identity import build_model_identity

MOTION, STANDING = 'b' * 64, 'c' * 64


class FakePolicy:
    """Single-policy interface with a recognisable constant output per network."""

    def __init__(self, sha, value, profile='direct-v1', events=None):
        self.sha256 = sha
        self.value = np.float32(value)
        self.target_contract = target_contract(sha, profile)
        self.events = events if events is not None else []
        self.inputs = []
        self.resets = 0
        self.reset()

    def preview(self, obs):
        self.inputs.append(obs.copy())
        self.events.append(('preview', self.sha256[0]))
        return np.full(14, self.value, np.float32)

    def infer(self, obs):
        action = self.preview(obs)
        self.commit(action)
        return action

    def commit(self, action):
        self.previous_action = action.copy()
        self._committed = HOME_Q + action

    def targets(self, action):
        return HOME_Q + action

    def reset(self):
        self.previous_action = np.zeros(14, np.float32)
        self._committed = None
        self.resets += 1

    @property
    def committed_targets(self):
        return self._committed


def pair(events=None):
    return FakePolicy(MOTION, .1, events=events), FakePolicy(STANDING, -.2, events=events)


def zero():
    return np.zeros(13, np.float32)


def forward():
    return np.array([.2] + [0.] * 12, np.float32)


def obs_for(command):
    return np.concatenate([np.zeros(48, np.float32), command]).reshape(1, 61)


# --- selector ----------------------------------------------------------------------
def test_construction_requires_two_bound_policies_sharing_the_target_transform():
    motion, standing = pair()
    with pytest.raises(ValueError, match='different'):
        StandingHandoff(motion, FakePolicy(MOTION, 0.))
    with pytest.raises(ValueError, match='target transform'):
        StandingHandoff(motion, FakePolicy(STANDING, 0., profile='robotd-targets-v1'))
    with pytest.raises(ValueError, match='profile'):
        StandingHandoff(motion, standing, profile='stand-always')
    unbound = FakePolicy(STANDING, 0.)
    unbound.target_contract = target_contract('d' * 64, 'direct-v1')
    with pytest.raises(ValueError, match='bound'):
        StandingHandoff(motion, unbound)
    with pytest.raises(ValueError, match='lacks'):
        StandingHandoff(motion, object())
    handoff = StandingHandoff(motion, standing)
    assert HANDOFF_PROFILES == ('stand-on-zero-twist-v1',) and handoff.profile == HANDOFF_PROFILES[0]
    assert handoff.sha256 == MOTION and handoff.standing_sha256 == STANDING
    assert handoff.target_contract == motion.target_contract
    assert handoff.contract() == {'profile': 'stand-on-zero-twist-v1', 'rule': HANDOFF_RULE,
                                  'motion_policy_sha256': MOTION, 'standing_policy_sha256': STANDING}
    assert handoff.committed_targets is None and handoff.active_role is None


def test_selection_follows_the_commanded_twist_and_precedes_evaluation():
    motion, standing = pair()
    handoff = StandingHandoff(motion, standing)
    with pytest.raises(RuntimeError, match='select'):
        handoff.preview(obs_for(zero()))
    assert handoff.select(zero()) == {'policy_role': 'standing', 'policy_sha256': STANDING, 'switch_pending': False}
    assert handoff.select(forward()) == {'policy_role': 'motion', 'policy_sha256': MOTION, 'switch_pending': False}
    for twist in ([0., .05, 0.], [0., 0., -.1], [1e-9, 0., 0.]):
        assert handoff.select(np.array(twist + [0.] * 10, np.float32))['policy_role'] == 'motion'
    # Nonzero head/body pose terms alone are not motion; only the body twist selects.
    assert handoff.select(np.array([0., 0., 0.] + [.3] * 10, np.float32))['policy_role'] == 'standing'
    for bad in (np.zeros(12, np.float32), np.array([np.nan] + [0.] * 12), np.zeros((1, 13)), np.array(['0'] * 13)):
        with pytest.raises(ValueError):
            handoff.select(bad)


def test_previous_action_is_the_last_committed_action_across_switches():
    events = []
    motion, standing = pair(events)
    handoff = StandingHandoff(motion, standing)
    handoff.select(forward())
    action = handoff.preview(obs_for(forward()))
    np.testing.assert_array_equal(handoff.targets(action), HOME_Q + np.float32(.1))
    handoff.commit(action)
    assert handoff.commits == {'motion': 1, 'standing': 0} and handoff.switches == 0
    np.testing.assert_array_equal(handoff.committed_targets, HOME_Q + np.float32(.1))
    # Switch to standing: the carried history is the motion action actually sent,
    # not the standing network's own (zero) memory.
    assert handoff.select(zero())['switch_pending'] is True
    np.testing.assert_array_equal(handoff.previous_action, np.full(14, .1, np.float32))
    np.testing.assert_array_equal(standing.previous_action, np.zeros(14, np.float32))
    stand = handoff.preview(obs_for(zero()))
    handoff.commit(stand)
    assert handoff.switches == 1 and handoff.last_switch == {'from': 'motion', 'to': 'standing', 'commit_index': 1}
    np.testing.assert_array_equal(handoff.previous_action, np.full(14, -.2, np.float32))
    np.testing.assert_array_equal(handoff.committed_targets, HOME_Q + np.float32(-.2))
    # A re-selection without a commit (the stepper's bounded retry) is not a switch.
    handoff.select(forward())
    handoff.select(zero())
    assert handoff.switches == 1 and handoff.selections == 4
    handoff.commit(handoff.preview(obs_for(zero())))
    assert handoff.switches == 1 and handoff.commits == {'motion': 1, 'standing': 2}
    assert events == [('preview', 'b'), ('preview', 'c'), ('preview', 'c')]
    telemetry = handoff.telemetry()
    assert telemetry['active_role'] == 'standing' and telemetry['last_committer'] == 'standing'
    assert telemetry['motion_policy_sha256'] == MOTION and telemetry['standing_policy_sha256'] == STANDING
    handoff.reset()
    assert motion.resets == 2 and standing.resets == 2 and handoff.active_role is None
    np.testing.assert_array_equal(handoff.previous_action, np.zeros(14, np.float32))
    assert handoff.committed_targets is None


# --- stepper --------------------------------------------------------------------
def roles(stepper):
    return [r['policy_role'] for r in stepper.policy_records]


def test_stepper_switches_network_with_the_lease_and_keeps_cadence():
    motion, standing = pair()
    handoff = StandingHandoff(motion, standing)
    s, b, c, p, a = make_stepper(policy=handoff)
    s.start()
    s.tick()
    assert roles(s) == ['standing'] and s.policy_records[0]['policy_sha256'] == STANDING
    assert len(standing.inputs) == 1 and not motion.inputs
    np.testing.assert_array_equal(a.targets[-1], HOME_Q + np.float32(-.2))
    command(c)
    for _ in range(4):
        s.tick()
    assert roles(s) == ['motion'] and s.policy_records[0]['switch_pending'] is True
    assert motion.inputs[-1][0, 48] == np.float32(.2)
    np.testing.assert_array_equal(motion.inputs[-1][0, 34:48], np.full(14, -.2, np.float32))
    np.testing.assert_array_equal(a.targets[-1], HOME_Q + np.float32(.1))
    c.stop(latch=True)
    for _ in range(4):
        s.tick()
    assert roles(s) == ['standing'] and s.policy_records[0]['switch_pending'] is True
    assert not standing.inputs[-1][0, 48:].any()
    np.testing.assert_array_equal(standing.inputs[-1][0, 34:48], np.full(14, .1, np.float32))
    assert handoff.switches == 2 and handoff.commits == {'motion': 1, 'standing': 2}
    # Cadence and accounting are those of a single policy: one evaluation per slot.
    assert s.policy_attempts == s.policy_evaluations == s.policy_commits == 3
    assert len([e for e in b.events if e[0] == 'before']) == s.steps
    s.close()


def test_stepper_switches_back_when_the_distance_completes_in_simulation_time():
    motion, standing = pair()
    s, b, c, p, a = make_stepper(policy=StandingHandoff(motion, standing))
    s.start()
    s.tick()
    h = c.hello()
    c.command_velocity(dict(robot_id=h['robot_id'], source=h['source'], epoch=h['epoch'],
                            generation=h['generation'], owner='test-owner', command_id='short',
                            vx=.2, vy=0., wz=0., duration_s=.05))  # ten physics steps: slots at steps 4 and 8
    seen = []
    for _ in range(16):
        s.tick()
        seen += roles(s)
    assert seen[:2] == ['motion', 'motion'] and seen[-1] == 'standing' and seen.count('motion') == 2
    assert c.state()['command_id'] is None and not c.state()['latched']
    s.close()


def test_retry_after_a_crossed_stop_re_selects_the_standing_network():
    motion, standing = pair()
    handoff = StandingHandoff(motion, standing)
    s, b, c, p, a = make_stepper(policy=handoff)
    s.start()
    s.tick()
    command(c)
    for _ in range(7):
        s.tick()  # motion commits at step 4; step 8 is the next slot
    last = handoff.previous_action.copy()
    np.testing.assert_array_equal(last, np.full(14, .1, np.float32))

    def crossed(obs):
        motion.inputs.append(obs.copy())
        c.stop(latch=True)
        return np.full(14, .4, np.float32)
    motion.preview = crossed
    before_step = b.step_count
    s.tick()
    assert [r['status'] for r in s.policy_records] == ['discarded', 'evaluated']
    assert roles(s) == ['motion', 'standing']
    assert [r['policy_sha256'] for r in s.policy_records] == [MOTION, STANDING]
    assert not standing.inputs[-1][0, 48:].any()
    # The discarded motion output never became history for the retry.
    np.testing.assert_array_equal(standing.inputs[-1][0, 34:48], last)
    np.testing.assert_array_equal(handoff.previous_action, np.full(14, -.2, np.float32))
    np.testing.assert_array_equal(a.targets[-1], HOME_Q + np.float32(-.2))
    assert b.step_count == before_step + 1 and not b.contained
    assert handoff.switches == 2 and handoff.commits == {'motion': 1, 'standing': 2}
    s.close()


def test_single_policy_records_are_unchanged_without_a_handoff():
    s, b, c, p, a = make_stepper()
    s.start()
    s.tick()
    assert 'policy_role' not in s.policy_records[0] and 'switch_pending' not in s.policy_records[0]
    s.close()


# --- bridge admission and construction ---------------------------------------------
def handoff_argv(tmp_path, monkeypatch, software_bundle):
    module, root, receipt_hash, _ = software_bundle
    from cascade.control import newton_bam
    from cascade.sim import microduck_policy_admission as admission
    forbid_live_imports(monkeypatch)
    release = tmp_path / 'release'
    for folder in ('apps', 'exts', 'extscache', 'extsUser', 'extsDeprecated'):
        (release / folder).mkdir(parents=True, exist_ok=True)
    (release / 'python.sh').write_text('software fixture; never executed')
    bam_root = tmp_path / 'bam'
    bam_root.mkdir()
    (bam_root / 'fixture.py').write_text('not executed')
    monkeypatch.setattr(newton_bam, 'SOURCE_SHA256', {'fixture.py': digest(b'not executed')})
    standing = tmp_path / 'velstand.onnx'
    standing.write_bytes(b'software fixture standing policy; never executed')
    motion = tmp_path / 'candidate.onnx'
    motion.write_bytes(b'software fixture motion candidate; never executed')
    manifest, _ = module.source_manifest()
    manifest['files'].append({'path': 'microduck-policies/velstand.onnx',
                              'sha256': digest(standing.read_bytes()), 'size': standing.stat().st_size})
    candidates = json.loads((admission.REPO / 'assets/microduck/policy-candidates.json').read_text())
    candidates['profiles']['rough_walk_e'].update(sha256=digest(motion.read_bytes()), size=motion.stat().st_size)
    destination = tmp_path / 'repo/assets/microduck/policy-candidates.json'
    destination.parent.mkdir(parents=True)
    destination.write_text(json.dumps(candidates))
    monkeypatch.setattr(admission, 'REPO', tmp_path / 'repo')
    limits = tmp_path / 'limits.json'
    limits.write_text(json.dumps(software_limits()))
    base = ['--engine', 'newton', '--release', str(release), '--bundle', str(root), '--bundle-sha256', receipt_hash,
            '--policy', str(motion), '--policy-sha256', digest(motion.read_bytes()), '--policy-profile', 'rough_walk_e',
            '--bam-source-root', str(bam_root), '--bam-profile', 'nominal_no_current_limit_no_delay',
            '--limits', str(limits), '--out', str(tmp_path / 'output'), '--port', '0', '--robot-id', 'microduck',
            '--source', 'software-only', '--device', 'cuda:0', '--max-wall-s', '60', '--max-steps', '10', '--check-only']
    extra = ['--handoff-profile', 'stand-on-zero-twist-v1', '--standing-policy', str(standing),
             '--standing-policy-sha256', digest(standing.read_bytes())]
    return base, extra, motion, standing


def test_bridge_admits_the_handoff_only_when_all_three_arguments_select_it(software_bundle, tmp_path, monkeypatch):
    base, extra, motion, standing = handoff_argv(tmp_path, monkeypatch, software_bundle)
    plain = cli().admit(cli().parse_args(base))
    assert 'handoff' not in plain
    admitted = cli().admit(cli().parse_args(base + extra))
    handoff = admitted['handoff']
    assert handoff['profile'] == 'stand-on-zero-twist-v1' and handoff['rule'] == HANDOFF_RULE
    assert handoff['motion_policy_sha256'] == digest(motion.read_bytes())
    assert handoff['standing_policy_sha256'] == digest(standing.read_bytes())
    assert handoff['standing_policy_profile'] == 'velstand'
    assert handoff['standing_policy_admission'] == {'profile': 'velstand', 'sha256': digest(standing.read_bytes()),
                                                    'size': standing.stat().st_size, 'physical_admission': False}
    assert admitted['policy_admission']['profile'] == 'rough_walk_e'
    assert 'src/cascade/control/microduck_handoff.py' in admitted['source_sha256']
    assert admitted['target_contract'] == plain['target_contract']
    for partial in (extra[:2], extra[2:4], extra[:4], extra[2:]):
        with pytest.raises(ValueError, match='together'):
            cli().admit(cli().parse_args(base + partial))
    with pytest.raises(SystemExit):
        cli().parse_args(base + ['--handoff-profile', 'stand-always'] + extra[2:])
    # The standing policy is the official pin, never the motion candidate again.
    same = list(extra)
    same[3], same[5] = str(motion), digest(motion.read_bytes())
    with pytest.raises(ValueError, match='different'):
        cli().admit(cli().parse_args(base + same))
    wrong = list(extra)
    wrong[5] = 'a' * 64
    with pytest.raises(ValueError, match='velstand ONNX'):
        cli().admit(cli().parse_args(base + wrong))
    assert cli().main(base + extra) == 0
    assert not (tmp_path / 'output').exists()


def test_bridge_constructs_the_handoff_from_admission(software_bundle, tmp_path, monkeypatch):
    base, extra, motion, standing = handoff_argv(tmp_path, monkeypatch, software_bundle)
    built = []

    def factory(path, sha, target_profile='direct-v1'):
        built.append((str(path), sha, target_profile))
        return FakePolicy(sha, .1 if sha == digest(motion.read_bytes()) else -.2, target_profile)
    args = cli().parse_args(base)
    plain = cli().create_policy(args, cli().admit(args), factory)
    assert isinstance(plain, FakePolicy) and built == [(str(motion), digest(motion.read_bytes()), 'direct-v1')]
    args = cli().parse_args(base + extra)
    admission = cli().admit(args)
    policy = cli().create_policy(args, admission, factory)
    assert isinstance(policy, StandingHandoff)
    assert policy.sha256 == digest(motion.read_bytes()) and policy.standing_sha256 == digest(standing.read_bytes())
    assert built[1:] == [(str(motion), digest(motion.read_bytes()), 'direct-v1'),
                         (str(standing), digest(standing.read_bytes()), 'direct-v1')]
    assert policy.target_contract == admission['target_contract']
    # A construction that does not match the admitted standing digest is refused.
    changed = copy.deepcopy(admission)
    changed['handoff']['standing_policy_sha256'] = 'a' * 64
    with pytest.raises(ValueError, match='differs from admission'):
        cli().create_policy(args, changed, factory)


# --- identity ------------------------------------------------------------------------
def handoff_admission(sha=STANDING):
    return {'profile': 'stand-on-zero-twist-v1', 'rule': HANDOFF_RULE, 'motion_policy_sha256': MOTION,
            'standing_policy_sha256': sha, 'standing_policy_profile': 'velstand',
            'standing_policy_admission': {'profile': 'velstand', 'sha256': sha, 'size': 1, 'physical_admission': False}}


def test_handoff_enters_the_model_identity_only_when_admitted(recipe_inputs):
    admission, native, paths = recipe_inputs
    plain = build_model_identity(admission, native, **paths)
    assert 'handoff' not in plain['recipe']
    admission['handoff'] = handoff_admission()
    bound = build_model_identity(admission, native, **paths)
    assert bound['model_identity_sha256'] != plain['model_identity_sha256']
    assert bound['recipe']['handoff'] == {k: v for k, v in handoff_admission().items() if k != 'standing_policy_admission'}
    other = copy.deepcopy(admission)
    other['handoff']['standing_policy_sha256'] = 'd' * 64
    assert build_model_identity(other, native, **paths)['model_identity_sha256'] != bound['model_identity_sha256']
    other = copy.deepcopy(admission)
    other['handoff']['profile'] = 'stand-on-zero-twist-v2'
    assert build_model_identity(other, native, **paths)['model_identity_sha256'] != bound['model_identity_sha256']


@pytest.mark.parametrize('mutation', ['motion_mismatch', 'same_policy', 'not_a_mapping', 'missing_rule'])
def test_identity_refuses_an_unbound_handoff(recipe_inputs, mutation):
    admission, native, paths = recipe_inputs
    admission['handoff'] = handoff_admission()
    if mutation == 'motion_mismatch':
        admission['handoff']['motion_policy_sha256'] = 'f' * 64
    elif mutation == 'same_policy':
        admission['handoff']['standing_policy_sha256'] = MOTION
    elif mutation == 'not_a_mapping':
        admission['handoff'] = 'stand-on-zero-twist-v1'
    else:
        del admission['handoff']['rule']
    with pytest.raises((ValueError, KeyError)):
        build_model_identity(admission, native, **paths)
