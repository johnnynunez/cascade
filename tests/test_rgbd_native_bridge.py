"""CPU lifecycle/identity controls; no SDK bootstrap or native acceptance."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from benchmark.rgbd import native_bridge as module
from cascade.sim.mobile_identity import build_model_identity
from test_mobile_identity import recipe_inputs as recipe_inputs


def test_usd_snapshot_preserves_listop_content_and_infinite_joint_limits():
    class TokenListOp:
        isExplicit = False
        explicitItems = []
        prependedItems = ['PhysicsCollisionAPI']
        appendedItems = []
        addedItems = []
        deletedItems = ['RemovedAPI']
        orderedItems = []
    first, second = TokenListOp(), TokenListOp()
    assert repr(first) != repr(second)
    assert module.usd_value(first) == module.usd_value(second)
    second.prependedItems = ['DifferentAPI']
    assert module.usd_value(first) != module.usd_value(second)
    bounds = module.usd_value([-float('inf'), float('inf')])
    assert bounds['items'] == [{'type': 'float', 'nonfinite': '-inf'},
                               {'type': 'float', 'nonfinite': 'inf'}]
    module.canonical(bounds)


def test_prior_identity_requires_document_and_canonical_recipe(tmp_path):
    recipe = {'native': {'solver': 'test'}, 'bam': {'q_indices': [1]}}
    model = hashlib.sha256(module.canonical(recipe)).hexdigest()
    path = tmp_path / 'identity.json'
    value = {'recipe': recipe, 'model_identity_sha256': model}
    path.write_bytes(module.canonical(value))
    assert module.reference_identity(path, module.sha(path), model) == value
    with pytest.raises(ValueError, match='file SHA'):
        module.reference_identity(path, 'a' * 64, model)
    with pytest.raises(ValueError, match='model identity'):
        module.reference_identity(path, module.sha(path), 'a' * 64)
    value['recipe']['native']['solver'] = 'changed'
    path.write_bytes(module.canonical(value))
    with pytest.raises(ValueError, match='model identity'):
        module.reference_identity(path, module.sha(path), model)


@pytest.mark.parametrize('field', ['native_labels', 'native_body_properties', 'configuration',
                                    'support_contract', 'rgbd_camera', 'bam'])
def test_no_filtering_of_changed_native_physics(field):
    expected = {'native_labels': {'shapes': ['foot', 'ground']},
                'native_body_properties': {'masses': [1.]}, 'configuration': {'steps': 1},
                'support_contract': {'foot_shapes': ['foot']}, 'rgbd_camera': {'digest': 'a'}}
    reference = {'recipe': {'native': expected, 'bam': {'q_indices': [1, 2]}},
                 'model_identity_sha256': 'b' * 64}
    native = copy.deepcopy(expected) | {'bam': copy.deepcopy(reference['recipe']['bam'])}
    assert module.native_invariance(native, reference)['passed']
    if field == 'native_labels':
        native[field]['shapes'].append('/World/RgbdMetricBoard/quad_0')
    else:
        native[field] = {'changed': True}
    rejected = module.native_invariance(native, reference)
    assert not rejected['passed'] and rejected['differences']


def test_fixture_precedes_camera_export_play_and_native_check(monkeypatch, tmp_path):
    events = []
    reference = {'recipe': {'native': {'solver': 'cpu-double'}, 'bam': {'q_indices': [1]}},
                 'model_identity_sha256': 'a' * 64}
    class Parent:
        def __init__(self):
            self.receipt = {'solver': 'cpu-double', 'bam': {'q_indices': [1]}}
            self.args = SimpleNamespace(out=tmp_path)
            self.admission = {'planar_reference_identity': reference}
        def _checkpoint(self):
            pass
        def _create_camera(self, stage):
            events.append('ordinary-camera')
        def open(self):
            self._create_camera(object())
            events.extend(['export', 'play-original-bootstrap'])
    monkeypatch.setattr(module, 'author_checked_board', lambda stage: events.append('board') or {})
    Backend = module.reference_backend(Parent)
    backend = Backend()
    backend.open()
    assert events == ['board', 'ordinary-camera', 'export', 'play-original-bootstrap']
    assert json.loads((tmp_path / 'planar-physics-invariance.json').read_text())['passed']
    events.clear()
    backend.receipt['solver'] = 'different'
    with pytest.raises(ValueError, match='native.solver'):
        backend.open()
    assert not json.loads((tmp_path / 'planar-physics-invariance.json').read_text())['passed']


def test_source_and_consumed_geometry_change_identity_and_tamper_rejects(recipe_inputs, monkeypatch):
    admission, native, paths = recipe_inputs
    original = build_model_identity(admission, native, **paths)
    monkeypatch.setattr(module, 'REPO', paths['repo'])
    for relative in module.SOURCES:
        path = paths['repo'] / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    module.extend_admission(admission, original)
    scene = paths['runtime_scene']
    scene.write_bytes(scene.read_bytes() + b'; mesh=/World/RgbdMetricBoard metric-points=(.08,.01,.002)')
    native['runtime_layer_sha256'] = module.sha(scene)
    observed = build_model_identity(admission, native, **paths)
    assert observed['model_identity_sha256'] != original['model_identity_sha256']
    assert observed['recipe']['runtime_scene_sha256'] != original['recipe']['runtime_scene_sha256']
    assert set(module.SOURCES) <= observed['recipe']['source_sha256'].keys()
    (paths['repo'] / module.SOURCES[-1]).write_bytes(b'changed during bootstrap')
    with pytest.raises(ValueError, match='source changed'):
        build_model_identity(admission, native, **paths)


def test_check_only_never_constructs_backend_or_output(monkeypatch, tmp_path, capsys):
    reference = {'model_identity_sha256': 'a' * 64}
    args = SimpleNamespace(camera_rgbd=True, check_only=True, out=tmp_path / 'absent')
    bridge = SimpleNamespace(parse_args=lambda argv: args,
        admit=lambda args: {'source_sha256': {}},
        run=lambda *a, **k: pytest.fail('check-only invoked run'))
    monkeypatch.setattr(module, 'bind_bridge', lambda: bridge)
    monkeypatch.setattr(module, 'reference_identity', lambda *args: reference)
    monkeypatch.setattr(module, 'reference_backend', lambda *a: pytest.fail('constructed SDK backend'))
    assert module.main(['--reference-identity', 'unused', '--reference-identity-sha256', 'a' * 64,
                        '--reference-model-sha256', 'a' * 64]) == 0
    assert json.loads(capsys.readouterr().out)['no_native_run']
    assert not args.out.exists()


def test_usd_existing_values_time_samples_and_collision_gate(monkeypatch):
    pytest.importorskip('pxr', reason='OpenUSD SDK unavailable in ordinary CPU environment')
    from pxr import Usd, UsdGeom
    from benchmark.rgbd import planar_reference
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.Xform.Define(stage, '/World')
    old = UsdGeom.Xform.Define(stage, '/World/existing')
    attr = old.AddTranslateOp()
    attr.Set((0., 0., 0.))
    attr.Set((1., 0., 0.), 1.)
    before = module.stage_snapshot(stage)
    result = module.author_checked_board(stage)
    assert result['existing_stage_unchanged'] and result['visual_meshes'] == 53
    assert before == module.stage_snapshot(stage)
    with pytest.raises(ValueError, match='already exists'):
        module.author_checked_board(stage)
    stage.RemovePrim(module.BOARD_PATH)
    original = planar_reference.author_visual_board
    def changed(stage, **kwargs):
        value = original(stage, **kwargs)
        attr.Set((2., 0., 0.), 1.)
        return value
    monkeypatch.setattr(planar_reference, 'author_visual_board', changed)
    with pytest.raises(ValueError, match='existing stage'):
        module.author_checked_board(stage)
