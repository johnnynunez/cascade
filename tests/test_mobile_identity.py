"""Recipe and wire binding tests; synthetic native metadata is not physics."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

from cascade.sim.mobile_identity import build_model_identity, canonical_bytes


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def recipe_inputs(tmp_path):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    outputs = []
    for name, data in [('usd/microduck.usda', b'root reference to payload'),
                       ('usd/physics.usda', b'effective masses and collision shapes')]:
        path = bundle / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)
        outputs.append(dict(path=name, sha256=sha(data), size=len(data)))
    receipt = dict(usd_path='usd/microduck.usda', outputs=outputs)
    receipt_bytes = canonical_bytes(receipt)
    (bundle / 'receipt.json').write_bytes(receipt_bytes)
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'controller.py').write_bytes(b'controller source')
    scene = tmp_path / 'runtime-scene.usda'
    scene.write_bytes(b'gravity = 9.81; reference=@' + str(bundle).encode() + b'/usd/microduck.usda@')
    admission = dict(bundle=str(bundle), receipt=receipt, asset_receipt_sha256=sha(receipt_bytes),
        asset_sha256=outputs[0]['sha256'], policy_sha256='b'*64,
        bam_config_sha256='c'*64, bam_params={'kp_fw': 200., 'max_delay': 0},
        bam_source_sha256={'drive.py': 'd'*64}, limits={'max_linear_speed': .15, 'min_height_m': .06},
        source_sha256={'controller.py': sha(b'controller source')})
    native = dict(consumed_asset_layers=[{'path': str(bundle/r['path']), 'sha256': r['sha256']} for r in outputs],
        runtime_layer_sha256=sha(scene.read_bytes()), solver='SolverMuJoCo',
        newton_version='1.6.0', warp_version='1.17.0', actual_physics_dt=.004999999888241291,
        physics_dt_s=.005, physics_device='cuda:0', gpu_attestation={'uuid': 'software-fixture-only'},
        configuration={'use_mujoco_contacts': True}, native_labels={'shapes': ['foot', 'ground']},
        native_model_properties={'joint_damping': [.005]}, native_body_properties={'body_mass_kg': [.3]},
        disabled_source_actuators=['source-drive'],
        initialization={'root_z_m': .125}, runtime_versions={'mujoco': '3.12.0'},
        support_contract={'version': 1, 'robot_shapes': ['foot'], 'foot_shapes': ['foot'],
                          'ground_shapes': ['ground'], 'gravity_world_m_s2': [0., 0., -9.81]},
        support_extraction={'semantics': 'solved contact forces'},
        bam=dict(implementation='synthetic-native-receipt', revision='fixed-source',
            source_sha256=copy.deepcopy(admission['bam_source_sha256']), device='cuda:0', newton_version='1.6.0',
            params=copy.deepcopy(admission['bam_params']), q_indices=list(range(7,21)), dof_indices=list(range(6,20)),
            friction_reference='bam_mjlab', env_dof_stride=14, mechanical_damping=.005, armature=.0018))
    return admission, native, dict(repo=repo, runtime_scene=scene)


def test_recipe_relocation_and_incidental_counters_do_not_change_identity(recipe_inputs, tmp_path):
    import shutil
    admission, native, paths = recipe_inputs
    before = build_model_identity(admission, native, **paths)
    relocated = tmp_path / 'elsewhere'
    shutil.copytree(admission['bundle'], relocated)
    old = admission['bundle']
    admission['bundle'] = str(relocated)
    admission.update(out='/incidental/new-output', port=9999)
    for layer in native['consumed_asset_layers']:
        layer['path'] = layer['path'].replace(old, str(relocated))
    paths['runtime_scene'].write_bytes(paths['runtime_scene'].read_bytes().replace(old.encode(), str(relocated).encode()))
    native.update(runtime_layer_sha256=sha(paths['runtime_scene'].read_bytes()), bootstrap_step=999)
    native['bam'].update(armed=True, steps_since_reset=999, source_root='/relocated/source')
    after = build_model_identity(admission, native, **paths)
    assert after == before
    assert str(tmp_path) not in json.dumps(before)
    assert before['support_contract']['model_identity_sha256'] == before['model_identity_sha256']


@pytest.mark.parametrize('field,change', [
    ('limits', lambda a,n: a['limits'].update(max_linear_speed=.2)),
    ('policy', lambda a,n: a.update(policy_sha256='f'*64)),
    ('BAM', lambda a,n: (a['bam_params'].update(max_delay=6), n['bam']['params'].update(max_delay=6))),
    ('solver', lambda a,n: n.update(solver='different-solver')),
    ('dt', lambda a,n: n.update(actual_physics_dt=.006)),
    ('versions', lambda a,n: n['runtime_versions'].update(mujoco='changed')),
    ('native_damping', lambda a,n: n['native_model_properties'].update(joint_damping=[.01])),
    ('device', lambda a,n: (n.update(physics_device='cuda:1'), n['bam'].update(device='cuda:1'))),
    ('GPU', lambda a,n: n['gpu_attestation'].update(uuid='other-GPU')),
    ('contact_support', lambda a,n: n['support_contract'].update(foot_shapes=['other-foot'])),
])
def test_effective_recipe_changes_cannot_reuse_identity(recipe_inputs, field, change):
    admission, native, paths = recipe_inputs
    before = build_model_identity(admission, native, **paths)['model_identity_sha256']
    change(admission, native)
    assert build_model_identity(admission, native, **paths)['model_identity_sha256'] != before, field


def test_changed_payload_with_identical_root_usd_gets_distinct_binding(recipe_inputs):
    admission, native, paths = recipe_inputs
    before = build_model_identity(admission, native, **paths)['model_identity_sha256']
    row = admission['receipt']['outputs'][1]
    from pathlib import Path
    data = b'new collision geometry'
    (Path(admission['bundle']) / row['path']).write_bytes(data)
    row.update(size=len(data), sha256=sha(data))
    native['consumed_asset_layers'][1]['sha256'] = row['sha256']
    receipt = canonical_bytes(admission['receipt'])
    (Path(admission['bundle']) / 'receipt.json').write_bytes(receipt)
    admission['asset_receipt_sha256'] = sha(receipt)
    assert build_model_identity(admission, native, **paths)['model_identity_sha256'] != before


@pytest.mark.parametrize('change', ['payload', 'source', 'scene', 'bam_mismatch', 'external_layer', 'nonfinite'])
def test_changes_during_bootstrap_fail_closed(recipe_inputs, change):
    admission, native, paths = recipe_inputs
    from pathlib import Path
    if change == 'payload':
        (Path(admission['bundle']) / 'usd/physics.usda').write_bytes(b'corruption')
    elif change == 'source':
        (paths['repo'] / 'controller.py').write_bytes(b'new-code')
    elif change == 'scene':
        paths['runtime_scene'].write_bytes(b'new scene')
    elif change == 'bam_mismatch':
        native['bam']['params']['kp_fw'] = 1.
    elif change == 'external_layer':
        native['consumed_asset_layers'][1]['path'] = '/unbound/physics.usda'
    else:
        native['native_model_properties']['joint_damping'][0] = float('nan')
    with pytest.raises(ValueError):
        build_model_identity(admission, native, **paths)


def test_recipe_is_a_defensive_snapshot(recipe_inputs):
    admission, native, paths = recipe_inputs
    identity = build_model_identity(admission, native, **paths)
    native['bam']['params']['kp_fw'] = 0
    assert identity['recipe']['bam']['params']['kp_fw'] == 200
    assert sha(canonical_bytes(identity['recipe'])) == identity['model_identity_sha256']
