"""Policy identity admission is independent of native behavior acceptance."""
import hashlib
import json
from pathlib import Path

import pytest

from cascade.sim import microduck_policy_admission as admission


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    data = b'not an ONNX; admission tests never execute this fixture'
    policy = tmp_path / 'candidate.onnx'
    policy.write_bytes(data)
    manifest = json.loads((Path(__file__).parents[1] / 'assets/microduck/policy-candidates.json').read_text())
    record = manifest['profiles']['rough_walk_e']
    record.update(sha256=hashlib.sha256(data).hexdigest(), size=len(data))
    destination = tmp_path / 'assets/microduck/policy-candidates.json'
    destination.parent.mkdir(parents=True)
    destination.write_text(json.dumps(manifest))
    monkeypatch.setattr(admission, 'REPO', tmp_path)
    monkeypatch.setattr(admission.native, 'source_manifest', lambda: ({'files': []}, b''))
    return policy, record['sha256'], destination, manifest


def test_alternative_requires_explicit_selection(candidate):
    policy, digest, _, _ = candidate
    with pytest.raises(ValueError, match='official VelStand'):
        admission.admit_policy(policy, digest)
    result = admission.admit_policy(policy, digest, 'rough_walk_e')
    assert result['profile'] == 'rough_walk_e' and result['physical_admission'] is False
    assert result['sha256'] == digest and result['size'] == policy.stat().st_size


@pytest.mark.parametrize('mutation', ['unknown_profile', 'caller_digest', 'same_size_bytes', 'size', 'symlink'])
def test_caller_cannot_authorize_another_checkpoint(candidate, mutation):
    policy, digest, _, _ = candidate
    profile = 'rough_walk_e'
    if mutation == 'unknown_profile':
        profile = 'arbitrary'
    elif mutation == 'caller_digest':
        digest = 'a' * 64
    elif mutation == 'same_size_bytes':
        policy.write_bytes(b'x' * policy.stat().st_size)
    elif mutation == 'size':
        policy.write_bytes(policy.read_bytes() + b'x')
    else:
        link = policy.with_suffix('.link')
        link.symlink_to(policy)
        policy = link
    with pytest.raises(ValueError):
        admission.admit_policy(policy, digest, profile)


@pytest.mark.parametrize('key,value', [('model_api', 2), ('obs', [1, 62]), ('actions', [1, 15]),
                                      ('action_scale', .9), ('policy_hz', 100), ('body_command', 'unknown')])
def test_candidate_contract_change_refused(candidate, key, value):
    policy, digest, path, manifest = candidate
    manifest['profiles']['rough_walk_e']['contract'][key] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='contract'):
        admission.admit_policy(policy, digest, 'rough_walk_e')


def test_shipped_candidate_pin_and_default_are_explicit():
    manifest = json.loads((Path(__file__).parents[1] / 'assets/microduck/policy-candidates.json').read_text())
    assert manifest['default_profile'] == 'velstand'
    record = manifest['profiles']['rough_walk_e']
    assert record['revision'] == 'fa7b27eeb5610d3b351362f4bd71691ee8be3d7d'
    assert record['sha256'] == '5aa423bd693e431b19e2ead77f99cbae6184e40a529eb2f7c1b4f85bb7f57040'
    assert record['size'] == 793772
    assert record['status'] == 'opt_in_research_candidate_not_native_or_hardware_admitted'
