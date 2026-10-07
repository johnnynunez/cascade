"""Content identity of one opened H2 PhysX recipe (provenance binding, not an outcome).

Built after ``backend.open()`` from the pinned bundle bytes (re-hashed, so a file
changed while Kit initialized cannot inherit its admission), the consumed stage
layers, the contract the owner enforces and the drives/solver settings the
articulation actually carries. Paths, ports, run directories and clocks are
absent; consumers pin the digest independently.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from cascade.sim.mobile_identity import IDENTITY_KEY, canonical_bytes, digest_token


def _file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract_record(contract) -> dict:
    """JSON view of the enforced contract (everything the owner reads from it)."""
    return {
        'model_sha256': contract.model_sha256, 'joint_names': list(contract.joint_names),
        'policy_joint_names': list(contract.policy_joint_names), 'held_joint_names': list(contract.held_joint_names),
        'observation_terms': [{'name': t.name, 'width': t.width, 'history_length': t.history_length,
                               'scale': t.scale} for t in contract.observation_terms],
        'observation_width': contract.observation_width, 'action_scale': contract.action_scale,
        'action_offset': list(contract.action_offset), 'action_clip': [list(c) for c in contract.action_clip],
        'default_joint_pos': list(contract.default_joint_pos), 'physics_dt': contract.physics_dt,
        'control_dt': contract.control_dt, 'decimation': contract.decimation,
        'command_ranges': {k: list(v) for k, v in contract.command_ranges.items()},
        'gains': {j: g.__dict__ for j, g in contract.gains.items()},
        'init_root_pos': list(contract.init_root_pos), 'init_root_rot_wxyz': list(contract.init_root_rot_wxyz),
        'fall_tilt_rad': contract.fall_tilt_rad, 'fall_pelvis_height_m': contract.fall_pelvis_height_m,
        'root_body': contract.root_body, 'illegal_contact_bodies': list(contract.illegal_contact_bodies),
        'foot_bodies': list(contract.foot_bodies), 'enabled_self_collisions': contract.enabled_self_collisions,
        'solver_iterations': list(contract.solver_iterations),
    }


def build_h2_model_identity(admission, native, *, repo):
    bundle = admission['bundle']
    files = bundle['files']
    for name, path_key in (('env.yaml', 'env_yaml'), ('IO_descriptors.yaml', 'descriptor_yaml')):
        if _file_digest(admission[path_key]) != files[name]['sha256']:
            raise ValueError(f'{name} changed after admission')
    if _file_digest(admission['policy']) != files['policy.pt']['sha256']:
        raise ValueError('policy.pt changed after admission')
    if _file_digest(admission['manifest_path']) != admission['manifest_sha256']:
        raise ValueError('bundle manifest changed after admission')
    sources = {}
    for relative, expected in admission['source_sha256'].items():
        if _file_digest(Path(repo) / relative) != expected:
            raise ValueError('runtime source changed during bootstrap: ' + relative)
        sources[relative] = digest_token(expected)
    layers = native['consumed_asset_layers']
    if not layers or len({r['identifier'] for r in layers}) != len(layers):
        raise ValueError('identity requires distinct consumed asset layers')
    root_url = f"{bundle['asset_root']}/{files['H2.usda']['path']}"
    if not any(r['identifier'] == root_url and r['sha256'] == files['H2.usda']['sha256'] for r in layers):
        raise ValueError('consumed root USD is not the pinned asset')
    native_fields = ('physics_device', 'gains_check', 'articulation', 'ground', 'contact_tracking',
                     'active_colliders', 'asset', 'deployment_chain', 'runtime_versions', 'support_contract',
                     'overview_camera')
    recipe = {
        'schema': 'cascade.h2.effective-model.v1', 'engine': 'physx',
        'bundle_manifest_sha256': digest_token(admission['manifest_sha256']),
        'bundle_files': {name: {'path': files[name]['path'], 'sha256': files[name]['sha256']} for name in sorted(files)},
        'asset_root': bundle['asset_root'], 'asset_sha256': digest_token(files['H2.usda']['sha256']),
        'policy_sha256': digest_token(files['policy.pt']['sha256']),
        'consumed_asset_layers': sorted(({'identifier': r['identifier'], 'sha256': digest_token(r['sha256'])}
                                         for r in layers), key=lambda r: r['identifier']),
        'contract': admission['contract_record'], 'policy_dt_s': admission['contract_record']['control_dt'],
        'limits': admission['limits'], 'source_sha256': sources,
        'native': {k: copy.deepcopy(native[k]) for k in native_fields},
    }
    payload = canonical_bytes(recipe)
    digest = hashlib.sha256(payload).hexdigest()
    return {IDENTITY_KEY: digest, 'recipe': json.loads(payload),
            'support_contract': {**copy.deepcopy(native['support_contract']), IDENTITY_KEY: digest}}
