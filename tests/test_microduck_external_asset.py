"""Isaac Lab USD folder (schema 2) admission, actuation neutralization and sole registry.

CPU doubles only: no Kit, Newton, network or physical claim. The fixtures build a
fake downloaded folder and manifests, run the real admission tool and the real
verifier against them, and drive the runtime-layer edits with fake prims.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from cascade.control.microduck_policy import POLICY_JOINTS

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / 'scripts/admit_microduck_usd.py'


def tool():
    spec = importlib.util.spec_from_file_location('admit_microduck_usd_test', TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    """Fake downloaded folder plus manifests, wired into the tool and the verifier."""
    import cascade.sim.microduck_newton as native
    files = {'microduck_allcollisions.usd': b'#usda 1.0\n# fixture all-collisions robot\n',
             'microduck_walk.usd': b'#usda 1.0\n# fixture walking robot\n',
             'LICENSE': b'Apache License 2.0 (fixture)\n',
             'ATTRIBUTION.txt': b'MicroDuck local USD assets (fixture)\n'}
    mirror = tmp_path / 'download' / 'isaaclab-microduck-usd'
    mirror.mkdir(parents=True)
    rows = []
    for name, data in sorted(files.items()):
        (mirror / name).write_bytes(data)
        rows.append({'source': 'isaaclab-microduck-usd', 'source_path': name,
                     'path': f'isaaclab-microduck-usd/{name}', 'size': len(data), 'sha256': digest(data),
                     'license': 'models' if name.endswith('.usd') else 'isaac-assets'})
    lines = sorted(f"{r['source_path']} {r['size']} {r['sha256']}" for r in rows)
    manifest = {'schema_version': 1,
                'sources': {'isaaclab-microduck-usd': {'kind': 'folder', 'relative_path': 'Robots/Fixture/MicroDuck',
                                                       'reference': 'https://example.invalid/isaaclab/pull/1',
                                                       'listing_sha256': digest(('\n'.join(lines) + '\n').encode())}},
                'licenses': {'models': {'license': 'Creative Commons BY-SA-NC', 'version': None,
                                        'evidence': 'https://example.invalid/models'},
                             'isaac-assets': {'license': 'Apache-2.0', 'evidence': 'https://example.invalid/isaac'}},
                'files': rows}
    folder_manifest = tmp_path / 'isaaclab-microduck-usd-manifest.json'
    folder_manifest.write_bytes(json.dumps(manifest, indent=2).encode() + b'\n')
    base = {'schema_version': 1,
            'sources': {'fixture': {'kind': 'github', 'repository': 'owner/robot', 'revision': 'a' * 40}},
            'licenses': {'code': {'license': 'Apache-2.0', 'evidence': 'https://example.invalid/code'}},
            'files': [{'source': 'fixture', 'source_path': 'robot.xml', 'path': 'fixture/robot.xml', 'size': 6,
                       'sha256': digest(b'<xml/>'), 'license': 'code'}]}
    base_manifest = tmp_path / 'manifest.json'
    base_manifest.write_bytes(json.dumps(base, indent=2).encode() + b'\n')
    monkeypatch.setattr(native, 'USD_FOLDER_MANIFEST', folder_manifest)
    monkeypatch.setattr(native, 'BASE_MANIFEST', base_manifest)
    return NS(mirror=mirror.parent, folder_manifest=folder_manifest, base_manifest=base_manifest,
              files=files, rows=rows, native=native, manifest=manifest)


def build(snapshot, destination, **overrides):
    options = dict(source_directory=snapshot.mirror, destination=destination, variant='allcollisions',
                   manifest=snapshot.folder_manifest, base_manifest=snapshot.base_manifest,
                   accept_model_license=True)
    options.update(overrides)
    return tool().build(**options)


def rewrite_receipt(bundle, mutate):
    receipt = json.loads((bundle / 'receipt.json').read_bytes())
    mutate(receipt)
    raw = (json.dumps(receipt, indent=2, sort_keys=True) + '\n').encode()
    (bundle / 'receipt.json').write_bytes(raw)
    (bundle / 'receipt.sha256').write_text(digest(raw) + '\n')
    return digest(raw)


def test_tool_builds_a_bundle_the_runtime_verifier_accepts(snapshot, tmp_path):
    result = build(snapshot, tmp_path / 'bundle')
    bundle = Path(result['bundle'])
    receipt = json.loads((bundle / 'receipt.json').read_bytes())
    assert result['ok'] and result['variant'] == 'allcollisions' and result['outputs'] == 6
    assert receipt['schema_version'] == 2 and receipt['kind'] == 'external-usd'
    assert receipt['usd_path'] == 'usd/microduck_allcollisions.usd'
    assert receipt['origin'] == {'kind': 'folder', 'source': 'isaaclab-microduck-usd', 'relative_path': 'Robots/Fixture/MicroDuck',
                                 'reference': 'https://example.invalid/isaaclab/pull/1', 'file': 'microduck_allcollisions.usd',
                                 'listing_sha256': snapshot.manifest['sources']['isaaclab-microduck-usd']['listing_sha256']}
    assert receipt['statuses']['physical_validation'] == 'none'
    assert all(value == 'unverified' for key, value in receipt['statuses'].items() if key != 'physical_validation')
    assert receipt['versions']['admission_tool_sha256'] == digest(TOOL.read_bytes())
    assert {row['path'] for row in receipt['outputs']} == {
        'usd/microduck_allcollisions.usd', 'usd/LICENSE', 'usd/ATTRIBUTION.txt',
        'provenance/manifest.json', 'provenance/base-manifest.json', 'provenance/folder.json'}
    assert (bundle / 'provenance/manifest.json').read_bytes() == snapshot.folder_manifest.read_bytes()
    admitted = snapshot.native.verify_bundle(bundle, expected_sha256=result['receipt_sha256'])
    assert admitted['kind'] == 'external-usd' and admitted['variant'] == 'allcollisions'
    assert admitted['asset'] == str(bundle / 'usd/microduck_allcollisions.usd')
    assert admitted['asset_sha256'] == digest(snapshot.files['microduck_allcollisions.usd'])
    assert admitted['asset_receipt_sha256'] == result['receipt_sha256']
    # The explicit --asset path must be the admitted root USD, not the reference walk file.
    snapshot.native.verify_bundle(bundle, expected_sha256=result['receipt_sha256'],
                                  asset=bundle / 'usd/microduck_allcollisions.usd')
    with pytest.raises(ValueError, match='admitted bundle root USD'):
        snapshot.native.verify_bundle(bundle, expected_sha256=result['receipt_sha256'], asset=bundle / 'usd/LICENSE')
    assert tool().check(bundle)['kind'] == 'external-usd'


def test_tool_refuses_unacknowledged_license_existing_destination_unknown_variant_and_bad_mirror(snapshot, tmp_path):
    with pytest.raises(ValueError, match='accept-model-license'):
        build(snapshot, tmp_path / 'a', accept_model_license=False)
    assert not (tmp_path / 'a').exists()
    with pytest.raises(ValueError, match='unknown variant'):
        build(snapshot, tmp_path / 'b', variant='walk')
    build(snapshot, tmp_path / 'c')
    with pytest.raises(ValueError, match='never overwritten'):
        build(snapshot, tmp_path / 'c')
    (snapshot.mirror / 'isaaclab-microduck-usd' / 'LICENSE').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='manifest check'):
        build(snapshot, tmp_path / 'd')
    assert not (tmp_path / 'd').exists()


@pytest.mark.parametrize('bad', ['kind', 'variant', 'usd_bytes', 'extra_file', 'tool_drift', 'manifest_drift',
                                 'physical_claim', 'origin', 'source_files', 'receipt_hash', 'symlink'])
def test_bad_schema2_bundle_fails_before_kit(snapshot, tmp_path, monkeypatch, bad):
    result = build(snapshot, tmp_path / 'bundle')
    bundle, receipt_hash = Path(result['bundle']), result['receipt_sha256']
    if bad == 'kind':
        receipt_hash = rewrite_receipt(bundle, lambda r: r.update(kind='converted-mjcf'))
    elif bad == 'variant':
        receipt_hash = rewrite_receipt(bundle, lambda r: r.update(variant='walk'))
    elif bad == 'usd_bytes':
        (bundle / 'usd/microduck_allcollisions.usd').write_bytes(b'#usda 1.0\n# different bytes\n')
    elif bad == 'extra_file':
        (bundle / 'usd/extra.usda').write_text('extra')
    elif bad == 'tool_drift':
        other = tmp_path / 'other_tool.py'
        other.write_text('# a different admission tool')
        monkeypatch.setattr(snapshot.native, 'ADMISSION_TOOL', other)
    elif bad == 'manifest_drift':
        data = json.loads(snapshot.folder_manifest.read_bytes())
        data['scope'] = 'edited after admission'
        snapshot.folder_manifest.write_bytes(json.dumps(data, indent=2).encode() + b'\n')
    elif bad == 'physical_claim':
        receipt_hash = rewrite_receipt(bundle, lambda r: r['statuses'].update(physical_validation='claimed'))
    elif bad == 'origin':
        receipt_hash = rewrite_receipt(bundle, lambda r: r['origin'].update(reference='https://other.example.invalid/pull/2'))
    elif bad == 'source_files':
        receipt_hash = rewrite_receipt(bundle, lambda r: r['source_files'].pop())
    elif bad == 'receipt_hash':
        receipt_hash = '0' * 64
    else:
        target = bundle / 'usd/LICENSE'
        target.unlink()
        target.symlink_to(bundle / 'receipt.json')
    with pytest.raises((ValueError, FileNotFoundError)):
        snapshot.native.verify_bundle(bundle, expected_sha256=receipt_hash)


# --- runtime-layer neutralization of the authored actuation --------------------------

REVOLUTE = object()


class FakeDriveAPI:
    @staticmethod
    def Get(prim, name):
        assert name == 'angular'
        return prim.drive


class FakeAttr:
    def __init__(self, value):
        self.value = value

    def Get(self):
        return self.value


class FakeDrive:
    def __init__(self, stiffness=0.0, damping=0.0, max_force=0.9599999785423279):
        self.stiffness, self.damping, self.max_force = stiffness, damping, max_force

    def GetStiffnessAttr(self):
        return FakeAttr(self.stiffness)

    def GetDampingAttr(self):
        return FakeAttr(self.damping)

    def GetMaxForceAttr(self):
        return FakeAttr(self.max_force)


class FakePrim:
    def __init__(self, path, type_name, *, drive=None, target=None, remove_ok=True):
        self.path, self.type_name, self.drive, self.target = path, type_name, drive, target
        self.active, self.has_drive, self.remove_ok = True, drive is not None, remove_ok

    def GetTypeName(self):
        return self.type_name

    def GetPath(self):
        return self.path

    def IsValid(self):
        return True

    def IsA(self, cls):
        return cls is REVOLUTE and self.type_name == 'PhysicsRevoluteJoint'

    def HasAPI(self, api, name=None):
        return api is FakeDriveAPI and name == 'angular' and self.has_drive

    def RemoveAPI(self, api, name=None):
        assert api is FakeDriveAPI and name == 'angular'
        if self.remove_ok:
            self.has_drive = False
        return self.remove_ok

    def SetActive(self, active):
        self.active = active


class FakeStage:
    def __init__(self, prims, anonymous=True):
        self.prims, self.layer = prims, NS(anonymous=anonymous)

    def Traverse(self):
        return iter(self.prims)

    def GetRootLayer(self):
        return self.layer

    def GetEditTarget(self):
        return NS(GetLayer=lambda: self.layer)

    def GetPrimAtPath(self, path):
        return next((p for p in self.prims if p.path == path), None)


def deployment():
    return dict(kp_fw=200.0, vin=7.400000095367432, vin_min=6.0, sag_gain=0.0, max_current=1.75, min_delay=3,
                max_delay=6, delay_seed=0, delay_hold_prob=0.0, delay_update_period=0, max_effort=1.067553162574768,
                kp_scale=1.0, kd_scale=1.0, friction_scale=1.0)


def asset_scene(root='/World/MicroDuck', **overrides):
    """Fourteen actuator prims targeting fourteen policy joints, like the Isaac Lab USD."""
    from cascade.control.newton_bam import motor_coefficients
    joints, actuators, kwargs = [], [], {}
    for name in POLICY_JOINTS:
        joint = FakePrim(f'{root}/Geometry/trunk_base/{name}', 'PhysicsRevoluteJoint', drive=FakeDrive())
        joints.append(joint)
        values = dict(motor_coefficients())
        values.update(deployment())
        kwargs[joint.path] = values
        actuators.append(FakePrim(f'{root}/servos_{name}_actuator', 'NewtonActuator', target=joint.path))
    overrides.get('mutate', lambda joints, actuators, kwargs: None)(joints, actuators, kwargs)
    DriveBam = type('DriveBam', (), {})
    Other = type('DrivePD', (), {})

    def parse(prim):
        if prim.type_name != 'NewtonActuator':
            return None
        cls = Other if overrides.get('other_class') and prim is actuators[0] else DriveBam
        values = kwargs.get(prim.target) or next(iter(kwargs.values()))  # foreign targets parse like any BAM prim
        return NS(drive_class=cls, drive_kwargs=dict(values), target_path=prim.target)
    stage = FakeStage([*joints, *actuators], anonymous=overrides.get('anonymous', True))
    return stage, joints, actuators, parse


@pytest.fixture
def fake_pxr(monkeypatch):
    monkeypatch.setitem(sys.modules, 'pxr', NS(UsdPhysics=NS(RevoluteJoint=REVOLUTE, DriveAPI=FakeDriveAPI)))


def test_asset_actuators_are_verified_recorded_then_deactivated_with_their_drives(fake_pxr):
    from cascade.sim.microduck_newton import disable_asset_actuators
    stage, joints, actuators, parse = asset_scene()
    record = disable_asset_actuators(stage, root_path='/World/MicroDuck', parse_actuator_prim=parse)
    assert record['kind'] == 'external-usd' and record['schema'] == 'NewtonBamDriveAPI'
    assert len(record['actuators']) == 14 and len(record['drives_removed']) == 14
    assert {a['target'].rsplit('/', 1)[-1] for a in record['actuators']} == set(POLICY_JOINTS)
    assert record['actuators'][0]['deployment'] == {k: deployment()[k] for k in sorted(deployment())}
    assert set(record['coefficients_checked']) >= {'kt', 'resistance', 'friction_base', 'load_friction_motor_quad'}
    assert record['drives_removed'][0] == {'path': joints[0].path, 'stiffness': 0.0, 'damping': 0.0,
                                           'max_force': 0.9599999785423279}
    assert all(not p.active for p in actuators) and all(not j.has_drive for j in joints)
    assert all(j.active for j in joints)


def _mutations():
    def coefficient(joints, actuators, kwargs):
        kwargs[joints[3].path]['kt'] *= 1.01

    def unknown(joints, actuators, kwargs):
        kwargs[joints[0].path]['extra_gain'] = 1.0

    def missing(joints, actuators, kwargs):
        del kwargs[joints[5].path]['friction_base']

    def backlash(joints, actuators, kwargs):
        kwargs[joints[1].path]['has_backlash'] = 1

    def foreign_target(joints, actuators, kwargs):
        actuators[2].target = '/World/Other/Geometry/trunk_base/left_knee'

    def duplicate_target(joints, actuators, kwargs):
        actuators[2].target = actuators[1].target
        kwargs.setdefault(actuators[1].target, kwargs[actuators[1].target])

    def gain(joints, actuators, kwargs):
        joints[7].drive.stiffness = 5.0

    def cap(joints, actuators, kwargs):
        joints[7].drive.max_force = 1.5

    def no_drive(joints, actuators, kwargs):
        joints[9].drive, joints[9].has_drive = None, False

    def count(joints, actuators, kwargs):
        actuators.pop()
    return dict(coefficient=coefficient, unknown=unknown, missing=missing, backlash=backlash,
                foreign_target=foreign_target, duplicate_target=duplicate_target, gain=gain, cap=cap,
                no_drive=no_drive, count=count)


@pytest.mark.parametrize('bad', sorted(_mutations()) + ['other_class', 'layer'])
def test_invalid_asset_actuation_is_rejected_before_any_mutation(fake_pxr, bad):
    from cascade.sim.microduck_newton import disable_asset_actuators
    options = {'mutate': _mutations()[bad]} if bad in _mutations() else {bad if bad == 'other_class' else 'anonymous': bad == 'other_class'}
    stage, joints, actuators, parse = asset_scene(**options)
    with pytest.raises(ValueError):
        disable_asset_actuators(stage, root_path='/World/MicroDuck', parse_actuator_prim=parse)
    assert all(p.active for p in actuators)
    assert all(j.has_drive for j in joints if j.drive is not None)


def test_drive_removal_failure_is_not_silent(fake_pxr):
    from cascade.sim.microduck_newton import disable_asset_actuators
    stage, joints, actuators, parse = asset_scene()
    joints[4].remove_ok = False
    with pytest.raises(RuntimeError, match='could not remove'):
        disable_asset_actuators(stage, root_path='/World/MicroDuck', parse_actuator_prim=parse)


def test_neutralization_dispatches_by_admitted_asset_kind(monkeypatch):
    import cascade.sim.microduck_newton as native
    calls = []
    monkeypatch.setattr(native, 'disable_source_actuators', lambda stage, root_path=None: calls.append(('mjcf', root_path)) or ['m'])
    monkeypatch.setattr(native, 'disable_asset_actuators', lambda stage, root_path=None: calls.append(('usd', root_path)) or {'u': 1})
    assert native.neutralize_asset_actuation('stage', 'converted-mjcf', root_path='/World/A') == ['m']
    assert native.neutralize_asset_actuation('stage', 'external-usd', root_path='/World/B') == {'u': 1}
    assert calls == [('mjcf', '/World/A'), ('usd', '/World/B')]
    with pytest.raises(ValueError, match='asset kind'):
        native.neutralize_asset_actuation('stage', 'physx-usd')


# --- native model preparation per asset kind -----------------------------------------

class Array:
    def __init__(self, values, dtype):
        self.value = np.array(values, dtype)

    def numpy(self):
        return self.value.copy()

    def assign(self, values):
        self.value[...] = values


def model_with(properties):
    model = NS()
    for key, value in properties.items():
        setattr(model, key, Array([value] * 20, np.int32 if key == 'joint_target_mode' else np.float32))
    return model


def test_external_asset_source_properties_are_asserted_then_replaced_by_the_same_m6_values():
    from cascade.sim.microduck_newton import ASSET_SOURCE_PROPERTIES, prepare_native_model
    expected = ASSET_SOURCE_PROPERTIES['external-usd']
    assert expected['joint_damping'] == pytest.approx(0.005359668274599504, rel=1e-6)   # BAM friction_viscous (SI)
    assert expected['joint_friction'] == pytest.approx(0.004771183165566, rel=1e-6)     # BAM friction_base
    assert expected['joint_effort_limit'] == 1e6 and expected['joint_target_mode'] == 0  # drive removed in the runtime layer
    model = model_with(expected)
    notifications = []
    ns = NS(model=model, solver=NS(notify_model_changed=notifications.append))
    newton = NS(ModelFlags=NS(JOINT_DOF_PROPERTIES='properties'))
    result = prepare_native_model(ns, np.arange(6, 20), source_cap=.96, newton=newton, asset_kind='external-usd')
    assert notifications == ['properties'] and result['asset_kind'] == 'external-usd'
    assert result['source_properties'] == expected
    np.testing.assert_allclose(result['after']['joint_damping'], [.005359668274599504] * 14)
    np.testing.assert_allclose(result['after']['joint_armature'], [.0018077432831600838] * 14)
    np.testing.assert_allclose(result['after']['joint_effort_limit'], [.96] * 14)
    # The converted bundle's expectations are not accepted for the Isaac asset, and vice versa.
    with pytest.raises(ValueError, match='unexpected native source property'):
        prepare_native_model(NS(model=model_with(ASSET_SOURCE_PROPERTIES['converted-mjcf']), solver=ns.solver),
                             np.arange(6, 20), source_cap=.96, newton=newton, asset_kind='external-usd')
    with pytest.raises(ValueError, match='unexpected native source property'):
        prepare_native_model(NS(model=model_with(expected), solver=ns.solver), np.arange(6, 20),
                             source_cap=.96, newton=newton)
    with pytest.raises(ValueError, match='asset kind'):
        prepare_native_model(None, (), source_cap=.96, newton=None, asset_kind='physx-usd')


# --- sole registry per asset kind -----------------------------------------------------

def test_sole_registry_is_explicit_per_asset_kind():
    from cascade.sim.microduck_contact_support import (FOOT_SHAPES, FOOT_SHAPES_BY_ASSET, GROUND_SHAPE,
                                                       foot_shapes_for, support_contract)
    external = foot_shapes_for('external-usd')
    assert external == (FOOT_SHAPES[0] + '/sole_left', FOOT_SHAPES[1] + '/sole_right')
    assert foot_shapes_for('converted-mjcf') == FOOT_SHAPES and set(FOOT_SHAPES_BY_ASSET) == {'converted-mjcf', 'external-usd'}
    with pytest.raises(ValueError, match='sole registry'):
        foot_shapes_for('physx-usd')
    sites = (FOOT_SHAPES[0].rsplit('/', 1)[0] + '/left_foot', FOOT_SHAPES[1].rsplit('/', 1)[0] + '/right_foot')
    labels = [GROUND_SHAPE, *external, *sites, '/World/MicroDuck/Geometry/trunk_base/np_f970_1/np_f970']
    contract = support_contract(labels, foot_shapes=external)
    assert contract['foot_shapes'] == list(external) and contract['ground_shapes'] == [GROUND_SHAPE]
    assert set(contract['robot_shapes']) == set(labels) - {GROUND_SHAPE}
    # Site spheres and the converted registry are never accepted as Isaac soles.
    with pytest.raises(ValueError, match='exact admitted sole'):
        support_contract(labels)
    with pytest.raises(ValueError, match='exact admitted sole'):
        support_contract([GROUND_SHAPE, *sites], foot_shapes=external)
    with pytest.raises(ValueError, match='exact admitted sole'):
        support_contract(labels, foot_shapes=sites[:1])


def test_shared_scene_binding_uses_the_explicit_sole_registry():
    from test_microduck_shared_scene import layout_fixture
    from cascade.sim.microduck_contact_support import FOOT_SHAPES, foot_shapes_for
    from cascade.sim.microduck_shared import bind_scene
    external = foot_shapes_for('external-usd')
    layout = layout_fixture(2)
    layout['shape_labels'] = [s + ('/sole_left' if s.endswith('left_foot_collision') else '/sole_right' if s.endswith('right_foot_collision') else '')
                              for s in layout['shape_labels']]
    bound = bind_scene(**layout, foot_shapes=external)
    for binding in bound.robots:
        assert binding.support_contract()['foot_shapes'] == [p.replace('/World/MicroDuck', binding.root_path, 1) for p in external]
    with pytest.raises(ValueError, match='missing exact sole shape'):
        bind_scene(**layout)
    with pytest.raises(ValueError, match='foot shapes'):
        bind_scene(**layout, foot_shapes=('/Elsewhere/sole',) + FOOT_SHAPES[1:])


# --- ground enrollment per asset kind ------------------------------------------------

COLLISION_GROUP, COLLISION_API = object(), object()


class GroundPrim:
    def __init__(self, path, *, group=False, collision=False):
        self.path, self.group, self.collision = path, group, collision

    def GetPath(self):
        return self.path

    def IsA(self, cls):
        return cls is COLLISION_GROUP and self.group

    def HasAPI(self, api, name=None):
        return api is COLLISION_API and self.collision

    def __bool__(self):
        return True


class GroundStage:
    def __init__(self, prims):
        self.prims = prims

    def Traverse(self):
        return iter(self.prims)

    def GetPrimAtPath(self, path):
        return next((p for p in self.prims if p.path == path), None)


@pytest.fixture
def ground_pxr(monkeypatch):
    monkeypatch.setitem(sys.modules, 'pxr', NS(UsdPhysics=NS(CollisionGroup=COLLISION_GROUP, CollisionAPI=COLLISION_API)))


def test_ground_enrollment_follows_the_asset_contact_model(ground_pxr):
    from cascade.sim.microduck_newton import bind_scene_ground
    ground = GroundPrim('/World/Ground', collision=True)
    calls = []
    converted = GroundStage([GroundPrim('/World/MicroDuck/CollisionGroups/mask_1_1', group=True), ground])
    record = bind_scene_ground(converted, '/World/Ground', root_path='/World/MicroDuck', asset_kind='converted-mjcf',
                               bind_ground=lambda stage, paths, root_path: calls.append((paths, root_path)))
    assert calls == [(['/World/Ground'], '/World/MicroDuck')] and record['enrollment'].startswith('collision group')
    external = GroundStage([GroundPrim('/World/MicroDuck/Geometry/trunk_base', collision=True), ground])
    record = bind_scene_ground(external, '/World/Ground', root_path='/World/MicroDuck', asset_kind='external-usd',
                               bind_ground=lambda *a, **k: pytest.fail('external asset must not be enrolled'))
    assert record['enrollment'].startswith('none') and record['ground'] == '/World/Ground'
    with pytest.raises(ValueError, match='collision groups'):
        bind_scene_ground(converted, '/World/Ground', root_path='/World/MicroDuck', asset_kind='external-usd')
    with pytest.raises(ValueError, match='external CollisionAPI'):
        bind_scene_ground(GroundStage([GroundPrim('/World/Ground')]), '/World/Ground', root_path='/World/MicroDuck',
                          asset_kind='external-usd')
    with pytest.raises(ValueError, match='asset kind'):
        bind_scene_ground(external, '/World/Ground', root_path='/World/MicroDuck', asset_kind='physx-usd')
