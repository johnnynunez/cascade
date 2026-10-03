"""CPU binding/lifecycle controls; no Kit or renderer claim."""
import copy
import json
from types import SimpleNamespace as NS

import pytest

from benchmark.rgbd import ground_native_bridge as module
from benchmark.rgbd.ground_texture import GroundTextureBoard, MATERIAL
from cascade.sim.mobile_identity import build_model_identity
from test_mobile_identity import recipe_inputs as recipe_inputs


def test_consumer_requires_exact_producer_board_png_and_quantized_geometry():
    value = module.texture_descriptor()
    assert module.board_from_fixture(value).description() == GroundTextureBoard().description()
    assert value['texture_sha256'] == '2638e36175318b42f2b477f3e8539b9afb441738f4737fb4f4177ba9dab05253'
    for key in value:
        bad = copy.deepcopy(value)
        bad.pop(key)
        with pytest.raises(ValueError, match='does not match'):
            module.board_from_fixture(bad)
    changed = copy.deepcopy(value)
    changed['board']['origin_xyz_m'][2] = .002
    with pytest.raises(ValueError, match='does not match'):
        module.board_from_fixture(changed)


def test_bound_png_does_not_inherit_admission_when_bytes_change(tmp_path, monkeypatch):
    original = module.REPO
    path = tmp_path / module.TEXTURE
    path.parent.mkdir(parents=True)
    path.write_bytes((original / module.TEXTURE).read_bytes() + b'\0')
    monkeypatch.setattr(module, 'REPO', tmp_path)
    with pytest.raises(ValueError, match='PNG differs'):
        module.texture_descriptor()


def test_model_identity_binds_new_entrypoint_and_actual_texture(recipe_inputs, monkeypatch):
    admission, native, paths = recipe_inputs
    original = build_model_identity(admission, native, **paths)
    source_root = module.REPO
    for rel in module.SOURCES:
        p = paths['repo'] / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((source_root / rel).read_bytes())
    monkeypatch.setattr(module, 'REPO', paths['repo'])
    module.extend_admission(admission, original)
    current = build_model_identity(admission, native, **paths)
    assert current['model_identity_sha256'] != original['model_identity_sha256']
    assert current['recipe']['source_sha256'][module.TEXTURE] == module.texture_descriptor()['texture_sha256']
    (paths['repo'] / module.TEXTURE).write_bytes(b'changed')
    with pytest.raises(ValueError, match='source changed'):
        build_model_identity(admission, native, **paths)


def backend_fixture(monkeypatch, tmp_path):
    events = []
    descriptor = module.texture_descriptor()
    reference = {'recipe': {'native': {'native_labels': ['ground', 'foot'],
        'native_body_properties': {'mass': [1.]}}, 'bam': {'q_indices': [1]}},
        'model_identity_sha256': 'a' * 64}
    class Parent:
        def __init__(self):
            self.receipt = copy.deepcopy(reference['recipe']['native']) | {'bam': {'q_indices': [1]}}
            self.admission = {'planar_reference_identity': reference,
                              'ground_reference_descriptor': descriptor}
            self.args = NS(out=tmp_path)
        def _checkpoint(self):
            events.append('checkpoint')
        def _create_camera(self, stage):
            events.append('ordinary-camera')
        def open(self):
            self._create_camera(object())
            events.extend(['export', 'ordinary-bootstrap'])
    def author(stage, expected):
        assert expected == descriptor
        events.append('texture')
        return descriptor
    monkeypatch.setattr(module, 'author_checked_ground', author)
    monkeypatch.setattr(module, 'ground_snapshot', lambda stage: {'surface': 'unchanged'})
    return module.reference_backend(Parent)(), events


def test_native_camera_order_unchanged_and_complete_invariance_has_no_label_exemption(monkeypatch, tmp_path):
    backend, events = backend_fixture(monkeypatch, tmp_path)
    backend.open()
    assert events == ['checkpoint', 'texture', 'checkpoint', 'ordinary-camera',
                      'checkpoint', 'export', 'ordinary-bootstrap', 'checkpoint']
    assert backend.receipt['ground_reference_fixture'] == module.texture_descriptor()
    backend.receipt['native_labels'].append('/World/RgbdMetricBoard/extra')
    with pytest.raises(ValueError, match='native.native_labels'):
        backend.open()
    assert not json.loads((tmp_path / 'planar-physics-invariance.json').read_text())['passed']


@pytest.mark.parametrize('phase', ['after_camera', 'after_bootstrap'])
@pytest.mark.parametrize('field', ['shader', 'st', 'descriptor', 'geometry', 'ancestors'])
def test_parent_change_refuses_before_model_listener_boundary(monkeypatch, tmp_path, phase, field):
    backend, events = backend_fixture(monkeypatch, tmp_path)
    calls = []
    initial = {k: 'authored' for k in ('shader', 'st', 'descriptor', 'geometry', 'ancestors')}
    def observed(stage):
        calls.append(stage)
        value = initial.copy()
        if len(calls) >= (2 if phase == 'after_camera' else 3):
            value[field] = 'changed-by-parent'
        return value
    monkeypatch.setattr(module, 'ground_snapshot', observed)
    with pytest.raises(ValueError, match='appearance/geometry changed'):
        backend.open()
        pytest.fail('model/listener admission was reached after changed appearance')
    check = json.loads((tmp_path / f'ground-appearance-{phase}.json').read_text())
    assert check['passed'] is False and check['differences'] == [field]
    assert check['authored_sha256'] != check['observed_sha256']
    if phase == 'after_camera':
        assert 'export' not in events


def test_check_only_is_passive_and_requires_rgbd(monkeypatch, tmp_path, capsys):
    args = NS(camera_rgbd=True, check_only=True, out=tmp_path / 'absent')
    bridge = NS(parse_args=lambda _: args, admit=lambda _: {'source_sha256': {}},
        run=lambda *a, **k: pytest.fail('check-only invoked native'))
    monkeypatch.setattr(module.common, 'bind_bridge', lambda: bridge)
    monkeypatch.setattr(module.common, 'reference_identity', lambda *a: {'model_identity_sha256': 'a'*64})
    monkeypatch.setattr(module, 'reference_backend', lambda *a: pytest.fail('SDK constructed'))
    argv = ['--reference-identity', 'unused', '--reference-identity-sha256', 'a'*64,
            '--reference-model-sha256', 'a'*64]
    assert module.main(argv) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['no_native_run'] and module.TEXTURE in result['source_sha256']
    assert not args.out.exists()
    args.camera_rgbd = False
    assert module.main(argv) == 1


def test_cpu_usd_metadata_is_in_consumed_scene_and_keeps_physics_binding():
    pytest.importorskip('pxr', reason='OpenUSD unavailable in ordinary CPU environment')
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.Xform.Define(stage, '/World')
    plane = UsdGeom.Plane.Define(stage, '/World/Ground')
    plane.GetAxisAttr().Set('Z')
    plane.GetWidthAttr().Set(2.)
    plane.GetLengthAttr().Set(2.)
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    material = UsdShade.Material.Define(stage, '/World/GroundMaterial')
    UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(plane.GetPrim()).Bind(material, materialPurpose='physics')
    expected = module.texture_descriptor()
    authored = module.author_checked_ground(stage, expected)
    bound = json.loads(stage.GetPrimAtPath(MATERIAL).GetCustomDataByKey(module.METADATA_KEY))
    assert bound == expected
    assert module.board_from_fixture(authored).description() == expected['board']
    assert expected['texture_sha256'] in stage.GetRootLayer().ExportToString()
    assert authored['physics_material'] == '/World/GroundMaterial'
    prims = [p for p in stage.TraverseAll() if p.IsA(UsdGeom.Gprim)]
    assert [str(p.GetPath()) for p in prims] == ['/World/Ground']
    stage.GetPrimAtPath(MATERIAL).SetCustomDataByKey(module.METADATA_KEY, '{}')
    with pytest.raises(ValueError, match='appearance or original'):
        module.validate_ground_texture(stage, module.REPO / module.TEXTURE,
            board=GroundTextureBoard(), receipt=authored)


def test_cpu_usd_stronger_ancestor_material_changes_effective_binding_and_is_rejected(tmp_path):
    pytest.importorskip('pxr', reason='OpenUSD unavailable in ordinary CPU environment')
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    world = UsdGeom.Xform.Define(stage, '/World').GetPrim()
    plane = UsdGeom.Plane.Define(stage, '/World/Ground')
    plane.GetAxisAttr().Set('Z')
    plane.GetWidthAttr().Set(2.)
    plane.GetLengthAttr().Set(2.)
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    physics = UsdShade.Material.Define(stage, '/World/GroundMaterial')
    UsdPhysics.MaterialAPI.Apply(physics.GetPrim())
    binding = UsdShade.MaterialBindingAPI.Apply(plane.GetPrim())
    binding.Bind(physics, materialPurpose='physics')
    module.author_checked_ground(stage, module.texture_descriptor())
    before = module.ground_snapshot(stage)
    assert str(binding.ComputeBoundMaterial()[0].GetPath()) == MATERIAL
    alternative = UsdShade.Material.Define(stage, '/World/AlternativeMaterial')
    surface = UsdShade.Shader.Define(stage, '/World/AlternativeMaterial/surface')
    surface.CreateIdAttr('UsdPreviewSurface')
    surface.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.18))
    alternative.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), 'surface')
    UsdShade.MaterialBindingAPI.Apply(world).Bind(
        alternative, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    assert str(binding.ComputeBoundMaterial()[0].GetPath()) == '/World/AlternativeMaterial'
    after = module.ground_snapshot(stage)
    assert before['surface_prims'] == after['surface_prims']
    assert before['world_from_ground'] == after['world_from_ground']
    assert before['ancestors'] != after['ancestors']
    backend = module.reference_backend(object)()
    backend._ground_stage, backend._ground_authored = stage, before
    backend.args, backend.receipt = NS(out=tmp_path), {}
    with pytest.raises(ValueError, match='ancestors'):
        backend._check_ground('after_bootstrap')
