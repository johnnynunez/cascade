"""Output-profile binding and actual staging; synthetic graph, no inference."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from cascade.control.microduck_policy import MicroduckPolicy, MicroduckTargets
from cascade.sim.microduck_policy_admission import target_contract, verify_target_contract
from test_microduck_bridge_cli import cli
from test_microduck_policy import fake_runtime, model_file


@pytest.mark.parametrize('change', ['missing', 'unknown', 'scale', 'head', 'legs', 'source',
                                  'policy', 'extra', 'bool'])
def test_contract_cannot_omit_or_override_fixed_transform(change):
    value = target_contract('a'*64, 'robotd-targets-v1')
    if change == 'missing': del value['head_lowpass']
    elif change == 'unknown': value['profile'] = 'other'
    elif change == 'scale': value['action_scale'] = 1.
    elif change == 'head': value['head_lowpass'] = .7
    elif change == 'legs': value['legs_lowpass'] = .5
    elif change == 'source': value['upstream_commit'] = 'a'*40
    elif change == 'policy': value['policy_sha256'] = 'b'*64
    elif change == 'extra': value['command_ema'] = .2
    else: value['physical_admission'] = 0
    with pytest.raises(ValueError):verify_target_contract(value, 'a'*64)
    with pytest.raises(ValueError):verify_target_contract(None, 'a'*64)


def test_launcher_passes_exact_profile_and_refuses_fallback(monkeypatch, tmp_path):
    fake_runtime(monkeypatch)
    path, digest = model_file(tmp_path)
    args = NS(policy=path, policy_sha256=digest, target_profile='robotd-targets-v1')
    admission = {'target_contract':target_contract(digest, args.target_profile)}
    policy = cli().create_policy(args, admission)
    assert policy.target_contract == admission['target_contract']
    def fallback(*args, **kwargs):
        return MicroduckPolicy(path, digest)  # A constructor ignoring the flag cannot admit.
    with pytest.raises(ValueError):cli().create_policy(args, admission, fallback)
    calls = []
    def forbidden(*args, **kwargs):calls.append(True)
    args.target_profile = 'direct-v1'
    with pytest.raises(ValueError):cli().create_policy(args, admission, forbidden)
    assert not calls


def test_shared_withdrawn_preview_never_advances_filter_or_applies_double_ema(monkeypatch, tmp_path):
    from test_microduck_shared_scene import shared
    from test_microduck_stepper import command
    fake_runtime(monkeypatch)
    path, digest = model_file(tmp_path)
    fleet, owner, steppers = shared(2)
    for s in steppers:
        s.policy = MicroduckPolicy(path, digest, target_profile='robotd-targets-v1')
        s.policy._session.run = lambda names, feed: [np.full((1,14), feed['obs'][0,48], np.float32)]
    # Bind the cohort only after its immutable policy members are selected.
    fleet = type(fleet)(owner, steppers, layout=fleet.layout)
    fleet.start()
    fleet.tick()
    for s in steppers:command(s.controller)
    for _ in range(3):fleet.tick()
    before = [s.policy.committed_targets for s in steppers]
    step = owner.step_count
    crossed = False
    def crossing(names, feed):
        nonlocal crossed
        for s, anchor in zip(steppers, before):
            np.testing.assert_array_equal(s.policy.committed_targets, anchor)
            assert s.policy_commits == 1
        if not crossed:
            crossed = True
            assert steppers[0].controller.stop()['cancelled']
        return [np.full((1,14), feed['obs'][0,48], np.float32)]
    steppers[1].policy._session.run = crossing
    fleet.tick()
    assert owner.step_count == step + 1 and not fleet.failure
    for index, s in enumerate(steppers):
        assert s.policy_commits == 2 and s.policy_evaluations == 3
        assert [r['status'] for r in s.policy_records] == ['discarded', 'evaluated']
        reference = MicroduckTargets('robotd-targets-v1')
        reference.commit(np.zeros(14, np.float32))
        raw = np.full(14, .2 if index else 0., np.float32)
        expected = reference.preview(raw)
        np.testing.assert_array_equal(s.policy.committed_targets, expected)
        np.testing.assert_array_equal(s.actuator.targets[-1], expected)
        np.testing.assert_array_equal(s.policy.previous_action, raw)
    fleet.close()
