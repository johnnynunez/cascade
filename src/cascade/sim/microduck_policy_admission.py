"""Explicit, byte-pinned MicroDuck policy selection; no download or execution.

The default remains the official VelStand pin. Alternative weights require a
named reviewed entry, never a caller-supplied SHA alone. Matching these bytes
establishes provenance/layout intent, not native locomotion admission. The ONNX
runner separately checks the actual graph signature before inference.
"""
from __future__ import annotations

from pathlib import Path
import json

from cascade.sim import microduck_newton as native
from cascade.sim.microduck_newton import REPO, digest_token, safe_file, sha256, strict_json


POLICY_PROFILES = ('velstand', 'rough_walk_e', 'isaaclab_velocity_flat', 'isaaclab_velocity_rough')


def target_contract(policy_sha256, profile):
    """Fixed output transform, separately selected from the weight profile."""
    from cascade.control.microduck_policy import MicroduckTargets
    digest_token(policy_sha256)
    if type(profile) is not str:
        raise ValueError('explicit target profile name required')
    return dict(MicroduckTargets(profile).contract, policy_sha256=policy_sha256)


def verify_target_contract(value, policy_sha256):
    if type(value) is not dict:
        raise ValueError('complete target contract required')
    expected = target_contract(policy_sha256, value.get('profile'))
    try:
        equal = json.dumps(value, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True)
    except (TypeError, ValueError):
        equal = False
    if not equal:
        raise ValueError('target contract differs from fixed profile/policy')
    return expected


def admit_policy(path, expected_sha256, profile='velstand'):
    digest_token(expected_sha256)
    if profile not in POLICY_PROFILES:
        raise ValueError('explicit known policy profile required')
    path = Path(path).absolute()
    safe_file(path.parent, path.name)
    if profile == 'velstand':
        manifest, _ = native.source_manifest()
        rows = [row for row in manifest['files'] if row['path'] == 'microduck-policies/velstand.onnx']
        if len(rows) != 1:
            raise ValueError('official VelStand source manifest is ambiguous')
        record = rows[0]
    else:
        manifest = strict_json((REPO / 'assets/microduck/policy-candidates.json').read_bytes())
        if manifest['schema_version'] != 1 or manifest['default_profile'] != 'velstand':
            raise ValueError('unsupported policy candidate manifest')
        record = manifest['profiles'][profile]
        if record['contract'] != {
            'model_api': 1, 'obs': [1, 61], 'actions': [1, 14], 'dtype': 'float32',
            'policy_hz': 50, 'action_scale': 1.0, 'normalizer': 'embedded',
            'entry_pose': 'standing', 'body_command': 'zeros', 'mouth': 'outside_policy',
        }:
            raise ValueError('unsupported policy candidate contract')
    if (record['sha256'] != expected_sha256 or path.stat().st_size != record['size']
            or sha256(path) != expected_sha256):
        raise ValueError(f'explicit policy must match pinned {profile} ONNX SHA/size; no checkpoint substitution')
    return {'profile': profile, 'sha256': expected_sha256, 'size': record['size'],
            'physical_admission': False}
