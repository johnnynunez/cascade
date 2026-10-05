"""Opt-in native Newton/Kit MicroDuck backend; optional imports stay lazy.

Bundle checks establish identity, not locomotion or physical acceptance. No
PhysX/PD fallback, SDK mutation, automatic download, or external probe imports.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import stat

REPO = Path(__file__).resolve().parents[3]
MODEL_PREFIX = 'microduck_rl/src/mjlab_microduck/robot/microduck/'
USD_FOLDER_MANIFEST = REPO / 'assets/microduck/isaaclab-microduck-usd-manifest.json'
BASE_MANIFEST = REPO / 'assets/microduck/manifest.json'
ADMISSION_TOOL = REPO / 'scripts/admit_microduck_usd.py'
EXTERNAL_USD_SOURCE = 'isaaclab-microduck-usd'
# Admitted Isaac Lab USD variants (schema 2 bundles built by scripts/admit_microduck_usd.py).
EXTERNAL_USD_VARIANTS = {'allcollisions': 'microduck_allcollisions.usd'}
EXTERNAL_USD_REFERENCE_FILES = ('LICENSE', 'ATTRIBUTION.txt')
CONVERTED_KIND, EXTERNAL_KIND = 'converted-mjcf', 'external-usd'
# Joint properties Newton 1.6.1rc1 imports for each admitted asset kind with Isaac
# Sim's add_usd arguments (Newton, Mjc, PhysX schema resolvers; forced
# position/velocity actuation) after the runtime-layer edits below. The converted
# bundle keeps the MJCF degree-based damping; the Isaac Lab USD authors the SI BAM
# viscous term as mjc:damping and friction_base as newton:friction, and its
# zero-gain drive is removed so the joints import unactuated like the converted
# bundle. Every value is asserted before the explicit BAM replacements.
ASSET_SOURCE_PROPERTIES = {
    CONVERTED_KIND: {'joint_damping': .053, 'joint_armature': .0018, 'joint_effort_limit': 1e6,
                     'joint_friction': .0048, 'joint_target_mode': 0, 'joint_target_ke': 0, 'joint_target_kd': 0},
    EXTERNAL_KIND: {'joint_damping': .005359668284654617, 'joint_armature': .0018, 'joint_effort_limit': 1e6,
                    'joint_friction': .004771183244884014, 'joint_target_mode': 0, 'joint_target_ke': 0,
                    'joint_target_kd': 0},
}
OVERVIEW_CAMERA = '/World/Overview'
OVERVIEW_RENDER_PRODUCT = '/World/OverviewRenderProduct'

OVERVIEW_SHAPE = (480, 640)
"""Retained overview (height, width); an opt-in resolution changes the receipt and identity."""


def overview_shape(value):
    """Validate an overview (height, width) pair; the default is the retained 640x480."""
    if (type(value) not in (tuple, list) or len(value) != 2 or any(type(v) is not int for v in value)
            or not all(64 <= v <= 4096 for v in value)):
        raise ValueError('overview resolution must be two integers (height, width) within 64..4096')
    return int(value[0]), int(value[1])


def parse_overview_resolution(text):
    """``WxH`` operator text (e.g. ``1920x1080``) to the (height, width) pair."""
    if type(text) is not str or text.count('x') != 1 or not all(part.isdigit() for part in text.split('x')):
        raise ValueError('overview resolution must be WIDTHxHEIGHT digits, e.g. 1920x1080')
    width, height = (int(part) for part in text.split('x'))
    return overview_shape((height, width))



def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest_token(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('SHA-256 must be 64 exact lowercase hex characters')
    return value


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON constant')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def source_manifest():
    data = (REPO / 'assets/microduck/manifest.json').read_bytes()
    return strict_json(data), data


def safe_file(root, relative):
    if (not isinstance(relative, str) or not relative or relative.startswith('/')
            or any(c in relative for c in '\\:%?#') or any(ord(c) < 32 for c in relative)
            or any(part in ('', '.', '..') for part in relative.split('/'))):
        raise ValueError('unsafe bundle path')
    path = root / relative
    for p in (path, *path.parents):
        if p.is_symlink():
            raise ValueError('symlink in bundle')
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError('bundle artifact is not a regular file')
    return path


def verify_bundle(bundle, *, expected_sha256, asset=None):
    """Verify every bundle output and pinned provenance WITHOUT USD/converter imports.

    The operator supplies the offline-admitted receipt digest. Schema 1 is the
    converted MJCF bundle (134 outputs); schema 2 is the downloaded Isaac Lab
    USD folder admitted by ``scripts/admit_microduck_usd.py``. Neither path redoes offline
    semantic validation; both refuse changed provenance, source files, layers,
    dependencies, or a different root USD.
    """
    root = Path(bundle).expanduser().absolute()
    receipt_path = safe_file(root, 'receipt.json')
    expected_sha256 = digest_token(expected_sha256)
    if (sha256(receipt_path) != expected_sha256
            or safe_file(root, 'receipt.sha256').read_text().strip() != expected_sha256):
        raise ValueError('bundle receipt SHA mismatch')
    receipt = strict_json(receipt_path.read_bytes())
    if receipt.get('schema_version') == 2:
        return _verify_external_usd_bundle(root, receipt, expected_sha256, asset)
    return _verify_converted_bundle(root, receipt, expected_sha256, asset)


def _verify_outputs(root, receipt):
    """Every listed output exists with the pinned bytes; nothing else is present."""
    outputs = receipt.get('outputs')
    if not isinstance(outputs, list) or not outputs:
        raise ValueError('bundle receipt lists no outputs')
    listed = {}
    for row in outputs:
        if not isinstance(row, dict) or set(row) != {'path', 'size', 'sha256'} or row['path'] in listed:
            raise ValueError('invalid/duplicate bundle output')
        path = safe_file(root, row['path'])
        if type(row['size']) is not int or row['size'] < 0:
            raise ValueError('invalid artifact size')
        if path.stat().st_size != row['size'] or sha256(path) != digest_token(row['sha256']):
            raise ValueError(f"bundle output hash/size mismatch: {row['path']}")
        listed[row['path']] = row
    actual = set()
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('symlink in bundle')
        if not path.is_dir():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(listed) | {'receipt.json', 'receipt.sha256'}:
        raise ValueError('bundle file set mismatch')
    return listed


def _verify_external_usd_bundle(root, receipt, expected_sha256, asset):
    """Schema 2: the admitted USD must be the exact folder bytes pinned in the manifest."""
    if receipt.get('kind') != EXTERNAL_KIND:
        raise ValueError('unsupported external bundle kind')
    variant = receipt.get('variant')
    if variant not in EXTERNAL_USD_VARIANTS:
        raise ValueError('external USD variant is not an admitted candidate')
    usd_name = EXTERNAL_USD_VARIANTS[variant]
    origin = receipt.get('origin')
    if (not isinstance(origin, dict) or origin.get('kind') != 'folder' or origin.get('source') != EXTERNAL_USD_SOURCE
            or origin.get('file') != usd_name
            or not all(isinstance(origin.get(k), str) and origin.get(k) for k in ('relative_path', 'reference', 'listing_sha256'))):
        raise ValueError('external USD origin record is incomplete')
    outputs = _verify_outputs(root, receipt)
    manifest_bytes = USD_FOLDER_MANIFEST.read_bytes()
    manifest = strict_json(manifest_bytes)
    base_bytes = BASE_MANIFEST.read_bytes()
    provenance = receipt.get('provenance', {})
    if (not isinstance(provenance, dict) or provenance.get('sources') != manifest['sources']
            or provenance.get('licenses') != manifest['licenses']
            or provenance.get('manifest_sha256') != hashlib.sha256(manifest_bytes).hexdigest()
            or provenance.get('base_manifest_sha256') != hashlib.sha256(base_bytes).hexdigest()
            or safe_file(root, 'provenance/manifest.json').read_bytes() != manifest_bytes
            or safe_file(root, 'provenance/base-manifest.json').read_bytes() != base_bytes):
        raise ValueError('external bundle provenance mismatch')
    source = manifest['sources'].get(EXTERNAL_USD_SOURCE)
    if (set(manifest['sources']) != {EXTERNAL_USD_SOURCE} or not isinstance(source, dict)
            or source.get('kind') != 'folder' or source.get('relative_path') != origin['relative_path']
            or source.get('reference') != origin['reference']
            or source.get('listing_sha256') != origin['listing_sha256']):
        raise ValueError('external bundle origin differs from the pinned asset folder')
    rows = {row['source_path']: row for row in manifest['files'] if row['source'] == EXTERNAL_USD_SOURCE}
    expected_sources = sorted(({'path': r['source_path'], 'size': r['size'], 'sha256': r['sha256']} for r in rows.values()),
                              key=lambda r: r['path'])
    source_files = receipt.get('source_files')
    if (not isinstance(source_files, list) or not all(isinstance(r, dict) and 'path' in r for r in source_files)
            or sorted(source_files, key=lambda r: r['path']) != expected_sources):
        raise ValueError('external bundle folder listing mismatch')
    for name in (usd_name, *EXTERNAL_USD_REFERENCE_FILES):
        row = rows.get(name)
        if row is None or outputs.get(f'usd/{name}') != {'path': f'usd/{name}', 'size': row['size'], 'sha256': row['sha256']}:
            raise ValueError(f'admitted file is not the pinned folder bytes: {name}')
    if receipt.get('versions', {}).get('admission_tool_sha256') != sha256(ADMISSION_TOOL):
        raise ValueError('bundle admission tool changed; re-admit explicitly')
    statuses = receipt.get('statuses')
    if not isinstance(statuses, dict) or statuses.get('physical_validation') != 'none':
        raise ValueError('external bundle must not claim physical validation')
    usd = safe_file(root, receipt['usd_path'])
    if receipt['usd_path'] != f'usd/{usd_name}' or (asset is not None and Path(asset).absolute() != usd):
        raise ValueError('asset must be the admitted bundle root USD')
    return {'bundle': str(root), 'asset': str(usd), 'asset_sha256': sha256(usd),
            'asset_receipt_sha256': expected_sha256, 'receipt': receipt, 'kind': EXTERNAL_KIND, 'variant': variant}


def _verify_converted_bundle(root, receipt, expected_sha256, asset):
    """Schema 1: the 134-output conversion of the pinned microduck_rl MJCF."""
    if receipt.get('schema_version') != 1 or receipt.get('collision_profile') not in ('source', 'velstand'):
        raise ValueError('unsupported bundle schema/profile')
    outputs = receipt.get('outputs')
    if not isinstance(outputs, list) or len(outputs) != 134:
        raise ValueError('require the admitted 134-output bundle')
    _verify_outputs(root, receipt)
    manifest, manifest_bytes = source_manifest()
    provenance = receipt.get('provenance', {})
    mh = hashlib.sha256(manifest_bytes).hexdigest()
    if (provenance.get('sources') != manifest['sources'] or provenance.get('licenses') != manifest['licenses']
            or provenance.get('manifest_sha256') != mh
            or safe_file(root, 'provenance/manifest.json').read_bytes() != manifest_bytes):
        raise ValueError('bundle source provenance mismatch')
    expected_sources = [{k: row[k] for k in ('path', 'size', 'sha256')} for row in manifest['files']
                        if row['path'].startswith(MODEL_PREFIX)]
    expected_sources = [{**row, 'path': row['path'][len(MODEL_PREFIX):]} for row in expected_sources]
    if sorted(receipt['source_files'], key=lambda r: r['path']) != sorted(expected_sources, key=lambda r: r['path']):
        raise ValueError('source file manifest mismatch')
    for row in expected_sources:
        path = safe_file(root, 'source/' + row['path'])
        if sha256(path) != row['sha256'] or path.stat().st_size != row['size']:
            raise ValueError('pinned source file hash/size mismatch')
    if receipt.get('source_xml') != 'robot_allcollisions.xml':
        raise ValueError('unsupported source XML')
    if receipt.get('versions', {}).get('adapter_sha256') != sha256(REPO / 'scripts/convert_microduck.py'):
        raise ValueError('bundle converter source changed; reconvert explicitly')
    usd = safe_file(root, receipt['usd_path'])
    if receipt['usd_path'] != 'usd/microduck.usda' or (asset is not None and Path(asset).absolute() != usd):
        raise ValueError('asset must be the admitted bundle root USD')
    return {'bundle': str(root), 'asset': str(usd), 'asset_sha256': sha256(usd),
            'asset_receipt_sha256': expected_sha256, 'receipt': receipt, 'kind': CONVERTED_KIND}


def experience_text(release, *, sdk_recipe=None):
    root = Path(release).expanduser().absolute()  # preserve source/shadow layout
    folders = [str(root / name) for name in ('apps', 'exts', 'extscache', 'extsUser', 'extsDeprecated')]
    if sdk_recipe is not None:
        from .microduck_sdk import require_recipe
        require_recipe(sdk_recipe)
        folders.append(str(root / 'extsInternal'))
    if not all(Path(p).is_dir() for p in folders):
        raise ValueError('--release must provide the Isaac extension folders')
    template = (REPO / 'configs/isaac/microduck.newton.kit').read_text()
    return template + '\n[settings.app.exts.folders]\n\'++\' = ' + json.dumps(folders) + '\n'


def _read_native_scene(ns, *, max_contacts, max_constraints, q_count, dof_count):
    """Capture and validate one completed scene; never retain swapped tensors."""
    import numpy as np
    model, state = ns.model, ns.state_0  # reacquire each time
    contacts, solver = ns.contacts, ns.solver
    data = solver.mjw_data
    clock = ns.simulation_step_count, float(ns.sim_time)
    arrays = {k: getattr(state, k).numpy() for k in ('body_q', 'body_qd', 'joint_q', 'joint_qd')}
    for name, a in arrays.items():
        if a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError(f'nonfinite or wrong native dtype: {name}')
    bodies = tuple(model.body_label)
    if arrays['body_q'].shape != (len(bodies), 7) or arrays['body_qd'].shape != (len(bodies), 6):
        raise ValueError('native body layout changed')
    if arrays['joint_q'].shape != (q_count,) or arrays['joint_qd'].shape != (dof_count,):
        raise ValueError('native free root + fourteen hinges required')
    body_com = model.body_com.numpy()
    counts = contacts.rigid_contact_count.numpy()
    nefc = data.nefc.numpy()
    for count, limit in ((counts, max_contacts), (nefc, max_constraints)):
        if count.shape != (1,) or count.dtype != np.int32 or not 0 <= count[0] < limit:
            raise RuntimeError('contact/constraint capacity or configured solver safety limit reached')
    ncon, nconstraints = int(counts[0]), int(nefc[0])
    shapes = tuple(model.shape_label)
    shape_body = model.shape_body.numpy()
    if shape_body.shape != (len(shapes),) or shape_body.dtype != np.int32:
        raise ValueError('invalid native shape/body map')
    pairs = [getattr(contacts, name).numpy() for name in ('rigid_contact_shape0', 'rigid_contact_shape1')]
    for a in pairs:
        if a.ndim != 1 or a.dtype != np.int32 or len(a) < ncon or (a[:ncon] < 0).any() or (a[:ncon] >= len(shapes)).any():
            raise ValueError('unidentified native contact shape')
    types = data.efc.type.numpy()
    if types.ndim != 2 or types.shape[0] != 1 or types.shape[1] < nconstraints or types.dtype != np.int32:
        raise ValueError('invalid constraint storage')
    # SolverMuJoCo.update_contacts copies ALL nacon candidates in identical
    # order, including separated geoms with no contact constraint. Only a
    # valid contact-row address is contact evidence; never report those
    # candidate pairs as bodies physically touching the floor.
    native_count = data.nacon.numpy()
    addresses = data.contact.efc_address.numpy()
    worlds = data.contact.worldid.numpy()
    if native_count.shape != (1,) or native_count.dtype != np.int32 or int(native_count[0]) != ncon:
        raise ValueError('contact candidate count differs from native solver')
    if (addresses.ndim != 2 or addresses.dtype != np.int32 or addresses.shape[0] < ncon
            or addresses.shape[1] < 1 or worlds.ndim != 1 or worlds.dtype != np.int32
            or len(worlds) < ncon or (worlds[:ncon] != 0).any()):
        raise ValueError('invalid contact constraint/world layout')
    first_rows = addresses[:ncon, 0]
    if (first_rows < -1).any() or (first_rows >= nconstraints).any():
        raise ValueError('invalid contact constraint address')
    active = np.flatnonzero(first_rows >= 0)
    # MuJoCo constraint enum: contact frictionless / pyramidal / elliptic.
    if not np.isin(types[0, first_rows[active]], (5, 6, 7)).all():
        raise ValueError('contact address does not identify a contact constraint')
    names = set()
    for a in pairs:
        for index in a[active]:
            body = int(shape_body[index])
            if body < -1 or body >= len(bodies):
                raise ValueError('unidentified contact body')
            names.add(shapes[index] if body == -1 else bodies[body])
    if (ns.model is not model or ns.state_0 is not state or ns.contacts is not contacts
            or ns.solver is not solver or ns.solver.mjw_data is not data
            or (ns.simulation_step_count, float(ns.sim_time)) != clock):
        raise RuntimeError('native scene changed during completed state capture')
    return dict(arrays=arrays, body_com=body_com, common=dict(
        step=clock[0], sim_time=clock[1], contacts=tuple(sorted(names)),
        contact_count=len(active), contact_candidate_count=ncon, constraint_count=nconstraints,
        contact_semantics='active_solver_contact_constraints_not_support_force',
        contact_constraint_addresses=first_rows[active].tolist(),
        contact_pairs=[[shapes[int(pairs[0][i])], shapes[int(pairs[1][i])]] for i in active]))


def _native_robot_slice(scene, *, q_indices, dof_indices, root_index):
    import copy
    import numpy as np
    from cascade.control.microduck_policy import POLICY_JOINTS
    from cascade.sim.microduck_state import body_frame_vectors

    arrays = scene['arrays']
    pose, vel = arrays['body_q'][root_index], arrays['body_qd'][root_index]
    quat = np.array([pose[6], *pose[3:6]])
    vectors = body_frame_vectors(quat, vel[3:], vel[:3], scene['body_com'][root_index])
    return dict(copy.deepcopy(scene['common']),
                position=pose[:3].astype(float).tolist(), orientation_wxyz=quat.astype(float).tolist(),
                linear_velocity=vectors['linear_velocity_origin_world'].tolist(),
                angular_velocity=vectors['angular_velocity_body'].tolist(),
                gravity_body=vectors['gravity_body'].tolist(),
                q=arrays['joint_q'][q_indices].copy(), dq=arrays['joint_qd'][dof_indices].copy(),
                joint_names=POLICY_JOINTS)


def read_native_state(ns, *, q_indices, dof_indices, root_index, max_contacts, max_constraints,
                      q_count=21, dof_count=20):
    """Read current state_0, never an experimental persistent swapped tensor."""
    scene = _read_native_scene(ns, max_contacts=max_contacts, max_constraints=max_constraints,
                               q_count=q_count, dof_count=dof_count)
    return _native_robot_slice(scene, q_indices=q_indices, dof_indices=dof_indices, root_index=root_index)


def read_native_states(ns, *, robots, max_contacts, max_constraints, q_count, dof_count):
    """One global read and validation, detached slices for every declared robot.

    The snapshot is local to this call; no array, contact row or binding is
    cached across completed physics states. Each result retains all contacts.
    """
    return _read_native_states(ns, robots=robots, max_contacts=max_contacts,
        max_constraints=max_constraints, q_count=q_count, dof_count=dof_count,
        private_contacts=False)


def _read_native_states(ns, *, robots, max_contacts, max_constraints, q_count, dof_count,
                        private_contacts=True):
    scene = _read_native_scene(ns, max_contacts=max_contacts, max_constraints=max_constraints,
                               q_count=q_count, dof_count=dof_count)
    if private_contacts:
        from cascade.control.mobile_telemetry import _freeze_contacts
        _freeze_contacts(scene['common'])
    return {robot: _native_robot_slice(scene, **indices) for robot, indices in robots.items()}


def prepare_native_model(ns, dof_indices, *, source_cap, newton, effort_cap=None, dof_count=20,
                         asset_kind=CONVERTED_KIND):
    """Explicit startup-only source -> BAM replacements, with fail-closed readback.

    ``asset_kind`` selects which imported joint properties are the expected
    source state (``ASSET_SOURCE_PROPERTIES``); the replacements are the same
    pinned M6 coefficients for every admitted asset.
    """
    import math
    import numpy as np
    from cascade.control.microduck_actuator import M6_PARAMETERS
    if asset_kind not in ASSET_SOURCE_PROPERTIES:
        raise ValueError('unknown admitted asset kind')
    if not math.isclose(source_cap, .96, rel_tol=0, abs_tol=1e-12):
        raise ValueError('verified XML effort cap must be 0.96 Nm')
    # The official inference loader explicitly replaces the XML position
    # actuator with a motor bounded by V*kt/R. Preserve the old XML-cap recipe
    # separately; neither setting is chosen from observed motion quality.
    if effort_cap is None:
        effort_cap = source_cap
    official_cap = 7.4 * M6_PARAMETERS['kt'] / M6_PARAMETERS['R']
    if (type(effort_cap) not in (int, float) or not math.isfinite(effort_cap)
            or not any(math.isclose(effort_cap, cap, rel_tol=0, abs_tol=1e-12)
                       for cap in (source_cap, official_cap))):
        raise ValueError('effort cap must match the XML or pinned official nominal inference recipe')
    expected = dict(ASSET_SOURCE_PROPERTIES[asset_kind])
    before, arrays = {}, {}
    for name, value in expected.items():
        a = getattr(ns.model, name).numpy()
        dtype = np.int32 if name == 'joint_target_mode' else np.float32
        if a.dtype != dtype or a.shape != (dof_count,) or not np.isfinite(a[dof_indices]).all():
            raise ValueError(f'unexpected native property layout: {name}')
        if not np.array_equal(a[dof_indices], np.full(14, value, dtype=dtype)):
            observed = sorted(set(a[dof_indices].tolist()))
            raise ValueError(f'unexpected native source property: {name} expected {value} for {asset_kind}, observed {observed}')
        before[name], arrays[name] = a[dof_indices].tolist(), a
    overrides = {'joint_damping': M6_PARAMETERS['friction_viscous'],
                 'joint_armature': M6_PARAMETERS['armature'], 'joint_effort_limit': effort_cap}
    for name, value in overrides.items():
        arrays[name][dof_indices] = value
        getattr(ns.model, name).assign(arrays[name])
    ns.solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
    after = {name: getattr(ns.model, name).numpy()[dof_indices].tolist() for name in expected}
    for name, value in overrides.items():
        if not np.array_equal(np.array(after[name], np.float32), np.full(14, value, np.float32)):
            raise RuntimeError(f'BAM property readback mismatch: {name}')
    return {'before': before, 'overrides': overrides, 'after': after,
            'asset_kind': asset_kind, 'source_properties': expected}


def read_native_body_properties(ns, *, root_path="/World/MicroDuck"):
    """Startup-only measurements for model identity and load diagnostics."""
    import numpy as np
    labels = list(ns.model.body_label)
    masses = ns.model.body_mass.numpy()
    gravity = ns.model.gravity.numpy()
    solver_gravity = ns.solver.mjw_model.opt.gravity.numpy()
    if (not labels or len(labels) != len(set(labels)) or masses.dtype != np.float32
            or masses.shape != (len(labels),) or not np.isfinite(masses).all()
            or (masses < 0).any()):
        raise ValueError('invalid measured native body mass/label layout')
    if (gravity.dtype != np.float32 or gravity.ndim != 2 or gravity.shape[1] != 3
            or len(gravity) not in (1, 2) or not np.isfinite(gravity).all()
            or solver_gravity.dtype != np.float32 or solver_gravity.shape != (1, 3)
            or not np.array_equal(gravity, np.repeat(solver_gravity, len(gravity), axis=0))
            or not np.array_equal(solver_gravity[0], np.array([0., 0., -9.81], np.float32))):
        raise ValueError('native Newton/MJWarp gravity differs from the declared single-world scene')
    robot = [i for i, label in enumerate(labels) if label.startswith(root_path + '/')]
    if not robot or float(np.sum(masses[robot], dtype=np.float64)) <= 0:
        raise ValueError('native MicroDuck mass unavailable')
    return dict(body_labels=labels, body_mass_kg=masses.astype(float).tolist(),
                gravity_world_m_s2=solver_gravity[0].astype(float).tolist(),
                newton_gravity_vectors_m_s2=gravity.astype(float).tolist())


def capture_bound_rgb(ns, app, readback, *, updates, checkpoint=lambda: None, calibration=None, pose_reader=None,
                      shape=OVERVIEW_SHAPE):
    """Main-thread capture with both physical clocks held fixed during render.

    ``shape`` is the authored overview (height, width); the default is the retained 640x480.
    """
    import time
    import numpy as np
    from cascade.sim.microduck_stepper import validate_render_times
    checkpoint()
    before = (ns.simulation_step_count, float(ns.sim_time))
    captured_at = time.monotonic()  # conservative: includes render and encoding latency
    camera_calibration = calibration() if calibration is not None else None
    checkpoint()
    ns.update_fabric()
    for _ in range(updates):
        checkpoint()
        app.update()
        # Kit can consume a SignalRequest inside its callbacks. Fence each
        # native return, not only the return of this whole capture operation.
        checkpoint()
        if (ns.simulation_step_count, float(ns.sim_time)) != before:
            raise RuntimeError('render advanced uncontrolled physics')
    before_render_times = None
    capture_objects = getattr(pose_reader,'capture_objects',None)
    pose_objects = capture_objects() if capture_objects is not None else None
    before_pose = pose_reader(camera_calibration, checkpoint) if pose_reader is not None else None
    if calibration is not None:
        data, _, before_render_times = readback.get_data_bound('rgb', checkpoint=checkpoint)
    else:
        data, _ = readback.get_data('rgb')
    checkpoint()
    if data is None:
        raise RuntimeError('overview RGB unavailable')
    rgb = np.asarray(data.numpy() if hasattr(data, 'numpy') else data).copy()
    if rgb.dtype != np.uint8 or rgb.shape != (*shape, 3):
        raise ValueError(f'overview must produce uint8 RGB[{shape[0]},{shape[1]},3]')
    extra = {}
    if calibration is not None:
        checkpoint()
        depth, _, depth_times = readback.get_data_bound('distance_to_image_plane', checkpoint=checkpoint)
        checkpoint()
        if before_render_times != depth_times:
            raise RuntimeError('RGB-D channels have different render product references')
        if depth is None:
            raise RuntimeError('overview registered metric depth unavailable')
        checkpoint()
        depth = np.asarray(depth.numpy() if hasattr(depth, 'numpy') else depth).copy()
        checkpoint()
        if depth.dtype != np.float32 or depth.shape not in (tuple(shape), (*shape, 1)):
            raise ValueError(f'overview depth must be float32[{shape[0]},{shape[1]}]')
        depth = depth.reshape(*shape)
        if (depth < 0).any():
            raise ValueError('negative overview depth')
        depth[~np.isfinite(depth)] = 0.  # renderer far clip/unavailable pixels, never inferred depth
        checkpoint()
        after_calibration = calibration()
        checkpoint()
        if camera_calibration != after_calibration:
            raise RuntimeError('camera calibration changed during RGB-D capture')
        extra = dict(depth_m=depth, calibration=camera_calibration,
                     rgbd_render_times={'rgb': before_render_times, 'depth': depth_times})
        if pose_reader is not None:
            after_pose = pose_reader(camera_calibration, checkpoint)
            if before_pose != after_pose or after_pose['render_reference'] != before_render_times:
                raise RuntimeError('RGB-D rig pose and channels do not share one render completion')
            extra['capture_pose'] = {k:v for k,v in after_pose.items() if k != 'evidence'}
            extra['pose_evidence'] = after_pose['evidence']
    checkpoint()
    times = readback.get_render_times()
    checkpoint()
    validate_render_times(times, before[1])
    if before_render_times is not None and times != before_render_times:
        raise RuntimeError('RGB-D render product changed during readback')
    if (ns.simulation_step_count, float(ns.sim_time)) != before:
        raise RuntimeError('camera read advanced physics')
    if pose_objects is not None and any(a is not b for a,b in zip(pose_objects,capture_objects(),strict=True)):
        raise RuntimeError('RGB-D capture changed registered native pose objects')
    return dict(rgb=rgb, step=before[0], sim_time_s=before[1], captured_at=captured_at, render_times=times, **extra)


def disable_asset_actuators(stage, *, root_path=None, parse_actuator_prim=None):
    """Isaac Lab USD: verify, record and deactivate the authored BAM actuation.

    The fourteen ``NewtonActuator`` prims resolve through Newton's own actuator
    registry (``newton.actuators.parse_actuator_prim``; ``load_pinned_bam``
    registers the pinned ``DriveBam`` under ``NewtonBamDriveAPI``). Their
    motor/gearbox coefficients must equal the pinned Rhoban m6 fit, their
    deployment settings are recorded next to the admitted profile, and they are
    deactivated before the stage parse so no second actuator pipeline exists. The
    zero-gain ``PhysicsDriveAPI:angular`` on the servo joints is removed in the
    runtime layer so Newton imports them unactuated (target mode NONE, effort
    limit 1e6), exactly like the converted bundle. All validation precedes any
    mutation; the adapter still owns every servo.
    """
    import math
    from pxr import UsdPhysics
    from cascade.control.microduck_policy import POLICY_JOINTS
    from cascade.control.newton_bam import BAM_DRIVE_API, DEPLOYMENT_KEYS, motor_coefficients
    if parse_actuator_prim is None:
        from newton.actuators import parse_actuator_prim
    layer = stage.GetRootLayer()
    if not layer.anonymous or stage.GetEditTarget().GetLayer() != layer:
        raise ValueError('asset actuator removal requires our anonymous runtime layer')
    prefix = None if root_path is None else root_path + '/'
    prims = [p for p in stage.Traverse() if p.GetTypeName() == 'NewtonActuator'
             and (prefix is None or str(p.GetPath()).startswith(prefix))]
    if len(prims) != 14:
        raise ValueError('unexpected asset NewtonActuator count')
    expected = motor_coefficients()
    allowed = set(expected) | set(DEPLOYMENT_KEYS) | {'has_backlash'}
    actuators, targets = [], {}
    for prim in prims:
        parsed = parse_actuator_prim(prim)
        if parsed is None or getattr(parsed.drive_class, '__name__', None) != 'DriveBam':
            raise ValueError(f'asset actuator is not a {BAM_DRIVE_API} drive: {prim.GetPath()}')
        kwargs = dict(parsed.drive_kwargs)
        unknown = set(kwargs) - allowed
        missing = set(expected) - set(kwargs)
        if unknown or missing:
            raise ValueError(f'unexpected asset BAM authoring (unknown={sorted(unknown)}, missing={sorted(missing)})')
        if kwargs.get('has_backlash', 0) not in (0, 0.0, False):
            raise ValueError('backlash asset variants are not admitted')
        for key, value in expected.items():
            authored = kwargs[key]
            if isinstance(value, int) and not isinstance(value, bool):
                ok = isinstance(authored, (int, float)) and float(authored) == float(value)
            else:
                ok = (isinstance(authored, (int, float)) and math.isfinite(authored)
                      and math.isclose(float(authored), float(value), rel_tol=1e-6, abs_tol=0))
            if not ok:
                raise ValueError(f'asset BAM coefficient differs from the pinned m6 fit: {key}')
        target = parsed.target_path
        joint = stage.GetPrimAtPath(target)
        name = str(target).rsplit('/', 1)[-1]
        if (not joint or not joint.IsValid() or not joint.IsA(UsdPhysics.RevoluteJoint) or name not in POLICY_JOINTS
                or name in targets or (prefix is not None and not str(target).startswith(prefix))):
            raise ValueError(f'asset actuator target is not an owned policy joint: {target}')
        targets[name] = joint
        actuators.append({'path': str(prim.GetPath()), 'target': str(target), 'drive_class': parsed.drive_class.__name__,
                          'deployment': {k: kwargs[k] for k in sorted(DEPLOYMENT_KEYS) if k in kwargs}})
    if set(targets) != set(POLICY_JOINTS):
        raise ValueError('asset actuators do not cover exactly the fourteen policy joints')
    drives = []
    for name in POLICY_JOINTS:
        joint = targets[name]
        if not joint.HasAPI(UsdPhysics.DriveAPI, 'angular'):
            raise ValueError(f'expected an authored angular drive on {joint.GetPath()}')
        drive = UsdPhysics.DriveAPI.Get(joint, 'angular')
        stiffness, damping, max_force = (drive.GetStiffnessAttr().Get(), drive.GetDampingAttr().Get(),
                                         drive.GetMaxForceAttr().Get())
        if (not all(isinstance(v, (int, float)) for v in (stiffness, damping, max_force))
                or stiffness != 0 or damping != 0
                or not math.isclose(float(max_force), .96, rel_tol=0, abs_tol=1e-6)):
            raise ValueError(f'unexpected authored drive on {joint.GetPath()}')
        drives.append({'path': str(joint.GetPath()), 'stiffness': float(stiffness), 'damping': float(damping),
                       'max_force': float(max_force)})
    for prim in prims:
        prim.SetActive(False)
    for name in POLICY_JOINTS:
        joint = targets[name]
        if not joint.RemoveAPI(UsdPhysics.DriveAPI, 'angular') or joint.HasAPI(UsdPhysics.DriveAPI, 'angular'):
            raise RuntimeError(f'could not remove the authored drive on {joint.GetPath()}')
    return {'kind': EXTERNAL_KIND, 'schema': BAM_DRIVE_API, 'actuators': actuators,
            'coefficients_checked': sorted(expected), 'drives_removed': drives}


def bind_scene_ground(stage, ground_path, *, root_path, asset_kind, bind_ground=None):
    """Enroll the scene ground the way the admitted asset expresses contact filtering.

    The converted MJCF bundle filters contacts through authored collision groups,
    so its ground must join the 1/1 mask group (``convert_microduck.bind_ground``).
    The Isaac Lab USD authors no collision groups at all: every enabled collider
    may touch the ground, and enrolling it anywhere would invent a filter. Any
    group found under such a root is refused rather than guessed about.
    """
    from pxr import UsdPhysics
    if asset_kind == CONVERTED_KIND:
        if bind_ground is None:
            from convert_microduck import bind_ground
        bind_ground(stage, [ground_path], root_path=root_path)
        return {'kind': asset_kind, 'root': str(root_path), 'ground': str(ground_path),
                'enrollment': 'collision group mask_1_1'}
    if asset_kind != EXTERNAL_KIND:
        raise ValueError('unknown admitted asset kind')
    prefix = str(root_path) + '/'
    groups = [str(p.GetPath()) for p in stage.Traverse()
              if str(p.GetPath()).startswith(prefix) and p.IsA(UsdPhysics.CollisionGroup)]
    if groups:
        raise ValueError(f'external USD authors collision groups; ground enrollment policy unknown: {groups[:3]}')
    ground = stage.GetPrimAtPath(ground_path)
    if not ground or not ground.HasAPI(UsdPhysics.CollisionAPI) or str(ground_path).startswith(prefix):
        raise ValueError('ground must be an external CollisionAPI prim')
    return {'kind': asset_kind, 'root': str(root_path), 'ground': str(ground_path),
            'enrollment': 'none; the asset authors no collision groups, every enabled collider may touch the ground'}


def neutralize_asset_actuation(stage, asset_kind, *, root_path=None):
    """Dispatch the startup-only runtime-layer edits by admitted asset kind."""
    if asset_kind == EXTERNAL_KIND:
        return disable_asset_actuators(stage, root_path=root_path)
    if asset_kind == CONVERTED_KIND:
        return disable_source_actuators(stage, root_path=root_path)
    raise ValueError('unknown admitted asset kind')


# Newton's USD importer (Isaac Sim 6.2 / Newton 1.6.1rc1) resolves a convex-hull
# vertex limit only from an AUTHORED attribute, in this resolver order, and
# otherwise substitutes its own cap, newton.Mesh.MAX_HULL_VERTICES = 64. Both
# admitted assets are MJCF conversions whose collision meshes author no limit; in
# MJCF that means a complete hull (-1). qhull's partial 64-vertex hull depends on
# vertex order, so the mirrored soles came out asymmetric and a straight command
# curled (+3.3 rad in 20 s on Isaac 6.2). Isaac Sim passes no mesh_maxhullvert.
HULL_LIMIT_ATTRIBUTES = ('newton:maxHullVertices', 'mjc:maxhullvert', 'physxConvexHullCollision:hullVertexLimit')
HULL_LIMIT_API = 'NewtonMeshCollisionAPI'
COMPLETE_HULL = -1


def author_collision_hull_limits(stage, *, root_path):
    """Give every enabled convex-hull collision mesh under ``root_path`` an explicit limit.

    Startup-only runtime-layer edit, before the stage parse. An authored limit is
    kept and recorded; conflicting authored limits are refused; an unauthored one
    is authored as the MJCF default, a complete hull (-1). The Isaac Lab USD
    instances its colliders, and USD refuses edits through instance proxies, so
    the enclosing instance prims are de-instanced first (composition only: the
    same prototype prims compose at the same paths with the same geometry).
    """
    from pxr import Sdf, Usd, UsdPhysics
    layer = stage.GetRootLayer()
    if not layer.anonymous or stage.GetEditTarget().GetLayer() != layer:
        raise ValueError('collision hull authoring requires our anonymous runtime layer')
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise ValueError(f'robot root missing: {root_path}')

    def colliders():
        found = []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not (prim.HasAPI(UsdPhysics.CollisionAPI) and prim.HasAPI(UsdPhysics.MeshCollisionAPI)):
                continue
            enabled = prim.GetAttribute('physics:collisionEnabled')
            if enabled and enabled.HasAuthoredValue() and not enabled.Get():
                continue
            approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            if approximation != UsdPhysics.Tokens.convexHull:
                raise ValueError(f'unexpected collision approximation {approximation!r}: {prim.GetPath()}')
            found.append(prim)
        return found

    def authored_limit(prim):
        values = {}
        for name in HULL_LIMIT_ATTRIBUTES:
            attr = prim.GetAttribute(name)
            if attr and attr.HasAuthoredValue():
                values[name] = int(attr.Get())
        if len(set(values.values())) > 1:
            raise ValueError(f'conflicting collision hull limits {values}: {prim.GetPath()}')
        return values

    deinstanced = []
    for prim in colliders():
        if not prim.IsInstanceProxy():
            continue
        instance = prim.GetParent()
        while instance and instance.IsValid() and not instance.IsInstance():
            instance = instance.GetParent()
        if not instance or not instance.IsValid():
            raise RuntimeError(f'instance proxy without an instance root: {prim.GetPath()}')
        if str(instance.GetPath()) not in deinstanced:
            if not instance.SetInstanceable(False) or instance.IsInstance():
                raise RuntimeError(f'could not de-instance {instance.GetPath()}')
            deinstanced.append(str(instance.GetPath()))
    authored, kept = [], []
    prims = colliders()
    if not prims:
        raise ValueError(f'no enabled convex-hull collision mesh under {root_path}')
    schema_known = bool(Usd.SchemaRegistry.GetTypeFromSchemaTypeName(HULL_LIMIT_API))
    for prim in prims:
        if prim.IsInstanceProxy():
            raise RuntimeError(f'collision mesh still an instance proxy: {prim.GetPath()}')
        values = authored_limit(prim)
        if values:
            kept.append({'path': str(prim.GetPath()), 'limits': values})
            continue
        # Apply Newton's mesh collision schema when its plugin is registered (it is in
        # Kit); the importer reads the attribute by name, so a bare attribute also binds.
        if schema_known:
            prim.ApplyAPI(HULL_LIMIT_API)
        attr = prim.GetAttribute(HULL_LIMIT_ATTRIBUTES[0])
        if not attr or not attr.IsValid():
            attr = prim.CreateAttribute(HULL_LIMIT_ATTRIBUTES[0], Sdf.ValueTypeNames.Int, custom=False)
        if not attr.Set(COMPLETE_HULL) or not attr.HasAuthoredValue() or int(attr.Get()) != COMPLETE_HULL:
            raise RuntimeError(f'could not author {HULL_LIMIT_ATTRIBUTES[0]} on {prim.GetPath()}')
        authored.append(str(prim.GetPath()))
    return {'root': str(root_path), 'default_limit': COMPLETE_HULL, 'attribute': HULL_LIMIT_ATTRIBUTES[0],
            'resolver_order': list(HULL_LIMIT_ATTRIBUTES), 'engine_fallback_limit': 64,
            'collision_meshes': len(prims), 'authored': authored, 'kept': kept, 'deinstanced': deinstanced}


def read_native_collision_hulls(model, *, root_path, limits, mesh_types, collide_flag):
    """Startup-only readback: the hull Newton built for every colliding mesh shape.

    ``physics:approximation = convexHull`` makes the importer replace each mesh
    by its hull (``ModelBuilder.approximate_meshes``), so the shape source holds
    the hull vertices and the limit it was built with. Every authored or kept
    limit must have reached a colliding native shape; visual and
    collision-disabled meshes are imported too but never collide, so they are
    only counted. The vertex counts are evidence (a complete hull of a small
    part may legitimately have few vertices).
    """
    expected = {path: COMPLETE_HULL for path in limits['authored']}
    for row in limits['kept']:
        expected[row['path']] = next(iter(row['limits'].values()))
    labels = list(model.shape_label)
    types = model.shape_type.numpy().tolist()
    flags = model.shape_flags.numpy().tolist()
    prefix = str(root_path) + '/'
    hulls, passive = [], 0
    for index, label in enumerate(labels):
        if not label.startswith(prefix) or types[index] not in mesh_types:
            continue
        if not int(flags[index]) & int(collide_flag):
            passive += 1
            continue
        source = model.shape_source[index]
        vertices = getattr(source, 'vertices', None)
        limit = getattr(source, 'maxhullvert', None)
        if vertices is None or type(limit) is not int:
            raise ValueError(f'native mesh shape without hull data: {label}')
        if label not in expected:
            raise ValueError(f'native collision mesh without an explicit hull limit: {label}')
        if limit != expected[label]:
            raise ValueError(f'native hull limit {limit} differs from the authored {expected[label]}: {label}')
        hulls.append({'label': label, 'maxhullvert': limit, 'hull_vertices': int(len(vertices))})
    if {row['label'] for row in hulls} != set(expected):
        missing = sorted(set(expected) - {row['label'] for row in hulls})
        raise ValueError(f'authored hull limits did not reach colliding native mesh shapes: {missing[:3]}')
    return {'root': str(root_path), 'shapes': hulls, 'non_colliding_meshes': passive,
            'total_hull_vertices': int(sum(row['hull_vertices'] for row in hulls))}


def disable_source_actuators(stage, *, root_path=None):
    import math
    from cascade.control.microduck_policy import POLICY_JOINTS
    layer = stage.GetRootLayer()
    if not layer.anonymous or stage.GetEditTarget().GetLayer() != layer:
        raise ValueError('source actuator removal requires our anonymous runtime layer')
    prims = [p for p in stage.Traverse() if p.GetTypeName() == 'MjcActuator'
             and (root_path is None or str(p.GetPath()).startswith(root_path + '/'))]
    if len(prims) != 14 or {p.GetName() for p in prims} != set(POLICY_JOINTS):
        raise ValueError('unexpected source MjcActuator names/count')
    records = []
    for prim in prims:
        lo = float(prim.GetAttribute('mjc:forceRange:min').Get())
        hi = float(prim.GetAttribute('mjc:forceRange:max').Get())
        if not math.isclose(lo, -.96, rel_tol=0, abs_tol=1e-12) or not math.isclose(hi, .96, rel_tol=0, abs_tol=1e-12):
            raise ValueError('unexpected source actuator effort cap')
        records.append({'path': str(prim.GetPath()), 'force_range': [lo, hi]})
    for prim in prims:
        prim.SetActive(False)
    return records


def synchronize_camera_authoring(app, timeline, manager, native_stage, *, checkpoint=lambda: None):
    """Absorb authored USD once, before any physical initialization or solve."""
    def clocks():
        return dict(stopped=bool(timeline.is_stopped()), timeline_time_s=float(timeline.get_current_time()),
                    manager_step=int(manager.get_num_physics_steps()),
                    manager_time_s=float(manager.get_simulation_time()),
                    native_initialized=bool(native_stage.initialized),
                    native_step=int(native_stage.simulation_step_count), native_time_s=float(native_stage.sim_time))
    expected = dict(stopped=True, timeline_time_s=0., manager_step=0, manager_time_s=0.,
                    native_initialized=False, native_step=0, native_time_s=0.)
    checkpoint()
    before = clocks()
    if before != expected:
        raise RuntimeError('camera authoring requires stopped zero clocks: ' + json.dumps(before, sort_keys=True))
    app.update()
    checkpoint()
    after = clocks()
    if after != before:
        raise RuntimeError('camera authoring advanced physics: ' + json.dumps(after, sort_keys=True))
    return dict(app_updates=1, before=before, after=after)


def create_overview_sensor(stage, camera, sensor_factory, *, sync_renderer, rgbd=False, camera_path=OVERVIEW_CAMERA,
                           shape=OVERVIEW_SHAPE):
    """Author a stable product for CameraSensor's supported asset-RP path.

    Isaac Sim 6.1 SensorRuntime._find_asset_render_product discovers a
    RenderProduct outside /Render with a camera relationship to this sensor.
    CameraSensor adopts its authored resolution. Its default creation path
    instead embeds hash(self), making otherwise identical scene hashes vary.
    Hydra's USD ABI requires a complete product, including its RenderVar, and
    one update to absorb the edits before attachment (omni.kit.hydra_texture's
    test_hydra_texture.setup_custom_product). Run that update while stopped.
    Verify adoption: no fallback product or scene-text normalization.
    """
    from pxr import Gf, Sdf
    if stage.GetPrimAtPath(OVERVIEW_RENDER_PRODUCT).IsValid():
        raise ValueError('overview render product path is already occupied')
    if (tuple(camera.paths) != (camera_path,)
            or stage.GetPrimAtPath(camera_path).GetTypeName() != 'Camera'):
        raise ValueError('overview requires the exact owned camera prim')
    product = stage.DefinePrim(OVERVIEW_RENDER_PRODUCT, 'RenderProduct')
    product.CreateRelationship('camera', custom=False).SetTargets([Sdf.Path(camera_path)])
    height, width = overview_shape(shape)
    product.CreateAttribute('resolution', Sdf.ValueTypeNames.Int2, custom=False,
                            variability=Sdf.VariabilityUniform).Set(Gf.Vec2i(width, height))
    color_path = Sdf.Path(OVERVIEW_RENDER_PRODUCT + '/LdrColor')
    color = stage.DefinePrim(color_path, 'RenderVar')
    color.CreateAttribute('sourceName', Sdf.ValueTypeNames.String, custom=False,
                          variability=Sdf.VariabilityUniform).Set('LdrColor')
    product.CreateRelationship('orderedVars', custom=False).SetTargets([color_path])
    sync_renderer()
    sensor = sensor_factory(camera, resolution=(height, width),
                            annotators=['rgb', 'distance_to_image_plane'] if rgbd else ['rgb'])
    actual = sensor.render_product.GetPrim()
    resolution = actual.GetAttribute('resolution').Get()
    observed = dict(path=str(actual.GetPath()),
                    camera_targets=[str(p) for p in actual.GetRelationship('camera').GetTargets()],
                    render_resolution=None if resolution is None else list(resolution),
                    sensor_resolution=None if sensor.resolution is None else list(sensor.resolution),
                    ordered_vars=[str(p) for p in actual.GetRelationship('orderedVars').GetTargets()])
    if (observed['path'] != OVERVIEW_RENDER_PRODUCT or observed['camera_targets'] != [camera_path]
            or observed['render_resolution'] != [width, height] or observed['sensor_resolution'] != [height, width]):
        raise RuntimeError('CameraSensor did not adopt the exact authored overview render product: '
                           + json.dumps(observed, sort_keys=True))
    return sensor


class KitNewtonBackend:
    """Kit main-thread lifecycle. Constructing this object imports no SDK.

    No world-reset method is exposed. Start a new process/epoch for another
    episode. Initial HOME/FK is the only pose write; faults pause and terminate.
    """
    def __init__(self, args, admission, experience):
        from .microduck_integrator import selected
        self._integrator_selection = selected(args, admission)
        self.args, self.admission, self.experience = args, admission, experience
        self.app = self.ns = self.readback = self.timeline = None
        self.receipt = {}
        self._closed = False
        self._captures = 0
        self._last_support_solve = None
        self._calibration_reader = None
        self._pose_reader = self._pose_annotator = self._capture_identity = None
        self.signals = None
        self._solver_graph = None
        self._sdk_recipe = getattr(args, 'sdk_recipe', None)
        self._solved_read = None
        self._reuse_solved_read = getattr(args, 'reuse_solved_read', False)
        if type(self._reuse_solved_read) is not bool or (self._reuse_solved_read
                and getattr(args, 'solver_cuda_graph', False) is not True):
            raise ValueError('same-solve read reuse requires explicit bound solver graph mode')
        # Opt-in overview resolution (operator WxH text or a (height, width) pair); default retained.
        selected_shape = getattr(args, 'overview_resolution', None)
        self.overview_shape = OVERVIEW_SHAPE if selected_shape is None else (
            parse_overview_resolution(selected_shape) if isinstance(selected_shape, str) else overview_shape(selected_shape))
        if self.overview_shape != OVERVIEW_SHAPE:
            self.receipt['overview_resolution'] = list(self.overview_shape)

    def _checkpoint(self):
        if self.signals is not None:
            self.signals.checkpoint(persistent=True)

    def _app_config(self):
        config = {'headless': True, 'disable_viewport_updates': True,
                'multi_gpu': False, 'width': 320, 'height': 240, 'renderer': 'RayTracedLighting',
                'physics_gpu': int(self.args.device.split(':')[1])}
        if 'private_rtx_cache' in self.admission:
            from .private_rtx_cache import extra_args
            config['extra_args'] = extra_args(self.admission['private_rtx_cache'])
        return config

    def _solver_capacity(self):
        return 512, 2400

    def open(self):
        import sys
        rtx_cache = self.admission.get('private_rtx_cache')
        if rtx_cache is not None:
            from .private_rtx_cache import prepare
            self.receipt['private_rtx_cache'] = prepare(self.args, rtx_cache)
        from isaacsim import SimulationApp
        saved = sys.argv
        sys.argv = [saved[0], '--/exts/isaacsim.physics.newton/auto_switch_on_startup=false',
                    '--/exts/omni.services.transport.server.http/http/enabled=false',
                    '--/exts/omni.services.transport.server.http/https/enabled=false']
        try:
            from contextlib import nullcontext
            defer = self.signals.defer if self.signals is not None else nullcontext
            with defer():
                # Acquire the handle so cleanup can find it even if interrupted.
                self.app = SimulationApp(self._app_config(), experience=str(self.experience))
            if rtx_cache is not None:
                import carb.settings
                from .private_rtx_cache import verify_effective
                if type(self.app) is not SimulationApp:
                    raise ValueError('private RTX cache constructed SDK class mismatch')
                self.receipt['private_rtx_cache_effective'] = verify_effective(
                    rtx_cache, carb.settings.get_settings(), type(self.app), self.args.release)
            self._checkpoint()
            self._initialize()
        except BaseException:
            self.close()
            raise
        finally:
            sys.argv = saved

    def _initialize(self):
        self._solved_read = None
        self._checkpoint()
        import copy
        import math
        import numpy as np
        import warp as wp
        import newton
        import mujoco
        import mujoco_warp
        import omni.usd
        import omni.timeline
        from pxr import Gf, UsdGeom, UsdPhysics, UsdShade
        from isaacsim.core.simulation_manager import SimulationManager as SM
        from isaacsim.core.version import get_version
        from isaacsim.core.experimental.utils import app as app_utils
        from isaacsim.physics.newton import acquire_stage, get_newton_config, configure_newton, MuJoCoSolverConfig
        from isaac_runtime import setup_physics, ensure_time_code_range, physics_device_identity, physics_timestep_identity
        from convert_microduck import bind_ground
        from cascade.sim.microduck_sdk import admit_release, configure_outputs, verify_runtime_recipe

        if self._sdk_recipe is not None:
            if self.admission.get('sdk_recipe') != admit_release(self.args.release, self._sdk_recipe):
                raise RuntimeError('MicroDuck SDK release changed after offline admission')
            verify_runtime_recipe(self._sdk_recipe, newton_version=newton.__version__)

        self._checkpoint()
        self.SM = SM
        self.timeline = omni.timeline.get_timeline_interface()
        manager = self.app._app.get_extension_manager()
        unrelated = ('omni.kit.asset_converter', 'isaacsim.asset.importer.mjcf.ui',
                     'isaacsim.asset.importer.urdf.ui', 'isaacsim.robot_motion.pink', 'omni.physx.pvd')
        enabled = [name for name in unrelated if manager.is_extension_enabled(name)]
        if enabled:
            raise RuntimeError(f'unexpected importer/UI dependencies enabled: {enabled}')
        SM.switch_physics_engine('newton')
        self._checkpoint()
        omni.usd.get_context().new_stage()
        self._checkpoint()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.Xform.Define(stage, '/World')
        # Load (and register with Newton's actuator registry) the pinned BAM sources
        # before any asset prim is parsed: asset-authored BAM prims must resolve
        # through that registry, and the stage parse must see no active actuator.
        from cascade.control.newton_bam import load_pinned_bam
        load_pinned_bam(self.args.bam_source_root, sdk_recipe=self._sdk_recipe)
        self._checkpoint()
        roots = self._author_robots(stage)
        ground = UsdGeom.Plane.Define(stage, '/World/Ground')
        ground.CreateAxisAttr('Z')
        ground_visual = getattr(self.args, 'ground_visual_m', None)
        if ground_visual is not None:
            # Display size of the ground only: Newton imports a PlaneShape collider as an
            # infinite plane (width=0, length=0) whatever the prim's width/length say.
            if type(ground_visual) not in (int, float) or not math.isfinite(ground_visual) or not 2. <= ground_visual <= 1000.:
                raise ValueError('ground visual size must be a finite length in 2..1000 m')
            ground.CreateWidthAttr(float(ground_visual))
            ground.CreateLengthAttr(float(ground_visual))
            self.receipt['ground_visual_m'] = float(ground_visual)
        UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
        material = UsdShade.Material.Define(stage, '/World/GroundMaterial')
        mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        mat.CreateStaticFrictionAttr(1.)
        mat.CreateDynamicFrictionAttr(1.)
        mat.CreateRestitutionAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(material, materialPurpose='physics')
        self.receipt['ground_binding'] = [
            bind_scene_ground(stage, ground.GetPath(), root_path=root_path, asset_kind=self.asset_kind,
                              bind_ground=bind_ground)
            for root_path in roots]
        scene = UsdPhysics.Scene.Define(stage, '/World/PhysicsScene')
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0., 0., -1.))
        scene.CreateGravityMagnitudeAttr(9.81)
        setup_physics(SM, dt=.005, device=self.args.device, require_cuda=True)
        cfg = copy.deepcopy(get_newton_config())
        cfg.solver_cfg = MuJoCoSolverConfig()
        cfg.num_substeps, cfg.use_cuda_graph, cfg.time_step_app = 1, False, False
        contacts, constraints = self._solver_capacity()
        cfg.solver_cfg.nconmax, cfg.solver_cfg.njmax = contacts, constraints
        cfg.collision_cfg.rigid_contact_max = contacts
        cfg.solver_cfg.use_mujoco_contacts = True
        configure_outputs(cfg, self._sdk_recipe)
        configure_newton(cfg)
        if self._integrator_selection is not None:
            # setup_physics has attached MjcSceneAPI; author its declared token
            # before exporting/importing the scene or performing bootstrap.
            from .microduck_integrator import author as author_integrator
            author_integrator(scene.GetPrim(), self._integrator_selection)
        self._checkpoint()
        self.receipt['configuration'] = dict(num_substeps=1, use_cuda_graph=False, time_step_app=False,
                                             nconmax=contacts, njmax=constraints,
                                             rigid_contact_max=contacts, use_mujoco_contacts=True)
        if 'private_rtx_cache' in self.admission:
            self.receipt['configuration']['private_rtx_cache'] = self.admission['private_rtx_cache']['policy']
        if self._sdk_recipe is not None:
            self.receipt['configuration'].update(sdk_recipe=self.admission['sdk_recipe'],
                                                contact_forces=True, link_incoming_joint_force=False)
        self._create_camera(stage)
        self._checkpoint()
        ensure_time_code_range(stage)
        layers = []
        allowed = {str((Path(self.admission['bundle']) / row['path']).resolve()): row['sha256']
                   for row in self.admission['receipt']['outputs']}
        for layer in stage.GetUsedLayers():
            if not layer.anonymous:
                path = Path(layer.realPath).resolve()
                if str(path) not in allowed or sha256(path) != allowed[str(path)]:
                    raise ValueError('USD consumed an unbound external layer')
                layers.append({'path': str(path), 'sha256': sha256(path)})
        self.receipt['consumed_asset_layers'] = layers
        runtime_layer = Path(self.args.out) / 'runtime-scene.usda'
        stage.GetRootLayer().Export(str(runtime_layer))
        self.receipt['runtime_layer_sha256'] = sha256(runtime_layer)
        self._checkpoint()
        app_utils.play(commit=True)
        self._checkpoint()
        self.ns = ns = acquire_stage()
        if not ns.initialized:
            raise RuntimeError('Newton did not initialize')
        self.receipt.update(physics_device_identity(SM, require_cuda=True, newton_stage=ns))
        self.receipt.update(physics_timestep_identity(SM))
        self._dt = self.dt
        self.receipt['bootstrap_clocks'] = {
            'manager_nominal_dt_s': float(SM.get_physics_dt()),
            'native_interface_dt_s': self._dt,
            'native_step': ns.simulation_step_count, 'native_time_s': float(ns.sim_time)}
        if self._dt != float(np.float32(.005)) or ns.simulation_step_count != 2 or not math.isclose(
                ns.sim_time, 2*self._dt, rel_tol=0, abs_tol=1e-10):
            raise RuntimeError('unexpected native bootstrap clocks/dt')
        if str(ns.model.device) != self.args.device or not ns.model.device.is_cuda or ns.solver.use_mujoco_cpu:
            raise RuntimeError('native Newton CUDA attestation failed')
        self.receipt.update(newton_version=newton.__version__, warp_version=wp.__version__,
                            solver=type(ns.solver).__name__, actual_physics_dt=self._dt,
                            bootstrap_step=ns.simulation_step_count, bootstrap_time_s=float(ns.sim_time))
        self.receipt['runtime_versions'] = dict(isaac_sim=list(get_version()),
                                               mujoco=mujoco.__version__, mujoco_warp=mujoco_warp.__version__)
        self._bind_native_model(ns)
        self._checkpoint()
        mesh_types = {int(newton.GeoType.MESH), int(newton.GeoType.CONVEX_MESH)}
        self.receipt['collision_hulls'] = [
            read_native_collision_hulls(ns.model, root_path=limits['root'], limits=limits, mesh_types=mesh_types,
                                        collide_flag=int(newton.ShapeFlags.COLLIDE_SHAPES))
            for limits in self.receipt['collision_hull_limits']]
        if [row['root'] for row in self.receipt['collision_hulls']] != list(roots):
            raise RuntimeError('collision hull records do not cover the authored robot roots')
        self._checkpoint()
        if self._integrator_selection is not None:
            from .microduck_integrator import observe
            self.receipt['integrator'] = {'contract': self._integrator_selection,
                                          'after_bootstrap': observe(ns)}
        if self.admission.get('camera_mount') is not None:
            from .mobile_camera_pose import FabricRigReader
            from .mobile_camera_encoding import verify_encoding_sources
            encoding = verify_encoding_sources(self.args.release,self._sdk_recipe)
            if encoding != self.admission['camera_pose_encoding']:
                raise ValueError('actual camera encoding differs from source admission')
            self._pose_reader = FabricRigReader(ns, self.admission['camera_mount']['definition'],
                                               self.readback, self._pose_annotator)
            self.receipt['rgbd_camera']['fabric_body_index'] = self._pose_reader.index
            self.receipt['rgbd_camera']['pose_encoding'] = encoding
        import inspect
        from cascade.sim.microduck_solver_graph import SolverGraphContract
        self._solver_graph = SolverGraphContract(ns,
            enabled=getattr(self.args, 'solver_cuda_graph', False), wp=wp, dt=self._dt,
            source_path=inspect.getfile(type(ns)), sdk_recipe=self._sdk_recipe)
        self.receipt['configuration']['use_cuda_graph'] = self._solver_graph.enabled
        self.receipt['configuration']['solver_graph_stage_sha256'] = self._solver_graph.source_sha256
        self.receipt['configuration']['reuse_solved_read'] = self._reuse_solved_read

    def verify_integrator_identity(self):
        """Recheck after identity construction and before policy/episode work."""
        if self._integrator_selection is None:
            return
        from .microduck_integrator import identity, observe
        expected = identity(self.admission, self.receipt)['after_bootstrap']
        observed = observe(self.ns)
        if observed != expected:
            raise ValueError('effective integrator changed while binding model identity')
        self.receipt['integrator_after_identity'] = observed

    def _author_robots(self, stage):
        from pxr import UsdGeom
        root = UsdGeom.Xform.Define(stage, '/World/MicroDuck').GetPrim()
        root.GetReferences().AddReference(self.admission['asset'])
        self.receipt['disabled_source_actuators'] = neutralize_asset_actuation(stage, self.asset_kind)
        self.receipt['collision_hull_limits'] = [author_collision_hull_limits(stage, root_path='/World/MicroDuck')]
        return ('/World/MicroDuck',)

    @property
    def asset_kind(self):
        return self.admission.get('kind', CONVERTED_KIND)

    def _bind_native_model(self, ns):
        import numpy as np
        import newton
        from cascade.control.microduck_policy import HOME_Q
        from cascade.control.newton_bam import NewtonBamAdapter
        from cascade.sim.microduck_state import newton_joint_indices
        from cascade.sim.microduck_contact_support import extraction_provenance, foot_shapes_for, support_contract
        qs, ds = newton_joint_indices(ns.model.joint_label, ns.model.joint_q_start.numpy(), ns.model.joint_qd_start.numpy())
        self.q_indices, self.dof_indices = qs, ds
        self.root_index = list(ns.model.body_label).index('/World/MicroDuck/Geometry/trunk_base')
        self._model = ns.model
        self._layout = (tuple(ns.model.joint_label), tuple(ns.model.body_label), tuple(ns.model.shape_label))
        self.receipt['native_labels'] = dict(zip(('joints', 'bodies', 'shapes'), self._layout))
        self.receipt['native_body_properties'] = read_native_body_properties(ns)
        self.receipt['support_contract'] = support_contract(ns.model.shape_label,
                                                            foot_shapes=foot_shapes_for(self.asset_kind))
        self.receipt['support_contract']['gravity_world_m_s2'] = self.receipt['native_body_properties']['gravity_world_m_s2'][:]
        self.receipt['support_extraction'] = extraction_provenance(sdk_recipe=self._sdk_recipe)
        self._checkpoint()
        self.receipt['native_model_properties'] = prepare_native_model(ns, ds, source_cap=.96, newton=newton,
            effort_cap=self.admission['bam_params']['joint_effort_limit'], asset_kind=self.asset_kind)
        self._checkpoint()
        self.bam = NewtonBamAdapter(ns, source_root=self.args.bam_source_root,
                                    q_indices=qs, dof_indices=ds, params=self.admission['bam_params'],
                                    sdk_recipe=self._sdk_recipe)
        self._checkpoint()
        # INITIALIZATION ONLY, outside all command admission/episode loops.
        q0 = ns.model.joint_q.numpy().copy()
        q0[qs] = HOME_Q
        free = np.flatnonzero(ns.model.joint_type.numpy() == int(newton.JointType.FREE))
        if len(free) != 1 or q0.shape != (21,):
            raise ValueError('exactly one free root required')
        start = int(ns.model.joint_q_start.numpy()[free[0]])
        q0[start:start+7] = [0., 0., .125, 0., 0., 0., 1.]
        for state in (ns.state_0, ns.state_1):
            self._checkpoint()
            state.joint_q.assign(q0)
            state.joint_qd.zero_()
            newton.eval_fk(ns.model, state.joint_q, state.joint_qd, state)
            state.clear_forces()
        self.bam.reset()  # no armed target until successful ONNX inference
        self.receipt['initialization'] = {'root_z_m': .125, 'home_q': HOME_Q.tolist(), 'pose_writes_in_episode': False}
        self.receipt['bam'] = self.bam.telemetry()

    def _camera_pose(self):
        return (.50, .45, .32), (.06, 0., .10)

    def _create_camera(self, stage):
        self._checkpoint()
        from pxr import Gf, UsdGeom, UsdLux
        from isaacsim.sensors.experimental.rtx import RtxCamera, CameraSensor
        from isaacsim.physics.newton import acquire_stage
        from isaac_camera_readback import CpuCameraReadback
        import omni.replicator.core as rep
        UsdLux.DomeLight.Define(stage, '/World/DomeLight').CreateIntensityAttr(300.)
        sun = UsdLux.DistantLight.Define(stage, '/World/Sun')
        sun.CreateIntensityAttr(1200.)
        sun.AddRotateXYZOp().Set(Gf.Vec3f(-50., 20., 0.))
        mount = self.admission.get('camera_mount')
        path = OVERVIEW_CAMERA
        if mount is None:
            eye, target = (Gf.Vec3d(*v) for v in self._camera_pose())
            quat = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0., 0., 1.)).GetInverse().ExtractRotationQuat()
        else:
            import numpy as np
            from pxr import UsdPhysics
            from .mobile_camera_pose import camera_path
            definition = mount['definition']
            rig = stage.GetPrimAtPath(definition['rig_prim_path'])
            if not rig or not rig.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError('camera mount must name an existing physical rigid body')
            path = camera_path(definition)
            if stage.GetPrimAtPath(path).IsValid():
                raise ValueError('owned mounted camera path already exists')
            local = np.array(definition['rig_from_camera']).reshape(4,4) @ np.diag([1.,-1.,-1.,1.])
            eye = local[:3,3].tolist()
            quat = Gf.Matrix4d(tuple(tuple(float(v) for v in row) for row in local.T)).ExtractRotationQuat()
        camera = RtxCamera(path, tick_rate=0., translations=list(eye),
                           orientations=[quat.GetReal(), *quat.GetImaginary()])
        camera.camera.set_clipping_ranges(.005, 20.)
        optics = UsdGeom.Camera(stage.GetPrimAtPath(path))
        optics.GetFocalLengthAttr().Set(18.)
        optics.GetHorizontalApertureAttr().Set(20.955)
        height, width = self.overview_shape
        optics.GetVerticalApertureAttr().Set(20.955 * height / width)
        def sync_renderer():
            self.receipt['camera_authoring_sync'] = synchronize_camera_authoring(
                self.app, self.timeline, self.SM, acquire_stage(), checkpoint=self._checkpoint)
        rgbd = getattr(self.args, 'camera_rgbd', False)
        sensor = create_overview_sensor(stage, camera, CameraSensor, sync_renderer=sync_renderer,
                                        rgbd=rgbd, camera_path=path, shape=self.overview_shape)
        self._checkpoint()
        product = str(sensor.render_product.GetPath())
        self.readback = CpuCameraReadback(sensor, render_product_id=product)
        times = self.readback._render_times
        for name in ('rpFabricTime', 'IsaacReadSimulationTime'):
            annotator = rep.AnnotatorRegistry.get_annotator(name)
            annotator.attach([product])
            times[name] = annotator
        self.receipt['camera'] = dict(name='overview', render_product=product, resolution=[height, width])
        self._calibration_reader = None
        if rgbd:
            from cascade.sim.mobile_rgbd import calibration_record, read_static_calibration, read_mount_calibration
            self._calibration_reader = (lambda: read_static_calibration(stage, path)) if mount is None else (
                lambda: read_mount_calibration(stage, path, mount['definition']))
            observed, digest = calibration_record(self._calibration_reader())
            self.receipt['rgbd_camera'] = {'calibration': observed, 'calibration_sha256': digest,
                                         'render_product': product, 'annotator': 'distance_to_image_plane'}
            if mount is not None:
                self._pose_annotator = rep.AnnotatorRegistry.get_annotator('camera_params')
                self._pose_annotator.attach([product])
                self.receipt['rgbd_camera'].update(mount={k:v for k,v in mount.items() if k != 'path'},
                    pose_source='registered Fabric body and same-render camera_params', camera_prim_path=path)

    @property
    def physics_clock(self):
        return self.ns.simulation_step_count, float(self.ns.sim_time)

    @property
    def dt(self):
        # The manager publishes decimal scene dt; the C++ physics interface
        # passes float32 to Newton. Keep both identities distinct and confirm
        # the native value against bootstrap and every completed solve.
        import math
        import numpy as np
        nominal = self.SM.get_physics_dt()
        if type(nominal) not in (int, float) or not math.isfinite(nominal) or nominal != .005:
            raise ValueError('nominal MicroDuck timestep changed or invalid')
        return float(np.float32(nominal))

    def _guard(self):
        from .microduck_sdk import check_outputs
        ns = self.ns
        check_outputs(ns.cfg, self._sdk_recipe)
        if (self._closed or not ns.initialized or ns.model is not self._model or ns.cfg.time_step_app
                or ns.cfg.num_substeps != 1
                or self.dt != self._dt or str(self.SM.get_active_physics_engine()).lower() != 'newton'
                or self._layout != (tuple(ns.model.joint_label), tuple(ns.model.body_label), tuple(ns.model.shape_label))):
            raise RuntimeError('frozen Newton model/clock/manual-step contract changed')
        if self._solver_graph is None:
            raise RuntimeError('solver execution mode was not bound during initialization')
        self._solver_graph.check()

    def read(self):
        # This backend has a single main-thread state writer and exposes no
        # pose/reset API between solves. The duplicate pre-tick read can reuse
        # the preceding post-solve payload after guards revalidate clock/model/
        # buffer ownership. This never publishes or refreshes a sample's age.
        import copy
        from cascade.sim.microduck_contact_support import read_support
        self._guard()
        clock = self.physics_clock
        key = (clock, self._last_support_solve)
        if self._reuse_solved_read and self._solved_read is not None and self._solved_read[0] == key:
            return copy.deepcopy(self._solved_read[1])
        sample = read_native_state(self.ns, q_indices=self.q_indices, dof_indices=self.dof_indices,
            root_index=self.root_index, max_contacts=self.admission['limits']['max_contacts'],
            max_constraints=self.admission['limits']['max_constraints'])
        sample['support'] = read_support(self.ns, last_solved_clock=self._last_support_solve,
            source_admitted=self.receipt['support_extraction']['source_admitted'])
        sample['solver_graph'] = self._solver_graph.telemetry()
        if self.physics_clock != clock:
            raise RuntimeError('physics advanced during native state/support read')
        if self._reuse_solved_read:
            self._solved_read = (key, copy.deepcopy(sample))
        return sample

    def step(self):
        # No app update, target write, model notification or rendering here.
        import math
        from cascade.sim.microduck_stepper import clock_tolerance
        self._checkpoint()
        self._guard()
        before = self.physics_clock
        self._solved_read = None
        self._last_support_solve = None
        self.SM.step(steps=1)
        self._guard()
        after = self.physics_clock
        if (after[0] != before[0] + 1 or not math.isclose(after[1], before[1] + self._dt,
                                                       rel_tol=0, abs_tol=clock_tolerance(after[1]))):
            raise RuntimeError('native step did not complete exactly one physical solve')
        self._last_support_solve = after

    def support_probe(self):
        """Read-only comparison with the SDK force API for native validation."""
        from cascade.sim.microduck_contact_support import compare_native_force_api
        self._guard()
        if self._last_support_solve != self.physics_clock:
            raise RuntimeError('support diagnostic requires a completed physical solve')
        return compare_native_force_api(self.ns)

    def bind_capture_identity(self, identity):
        """Bind only producer identity, never a controller pose or receipt clock."""
        from ..robotics.contracts import identifier
        if self._capture_identity is not None:
            raise ValueError('capture identity already bound')
        self._capture_identity = {'epoch': identifier(identity['epoch']),
                                  'model_identity_sha256': digest_token(identity['model_identity_sha256'])}

    def capture(self):
        self._checkpoint()
        self._guard()
        result = capture_bound_rgb(self.ns, self.app, self.readback,
            updates=16 if self._captures == 0 else 3, checkpoint=self._checkpoint,
            calibration=self._calibration_reader, pose_reader=self._pose_reader, shape=self.overview_shape)
        if self._pose_reader is not None:
            if self._capture_identity is None:
                raise RuntimeError('mounted capture requires its opened model/epoch identity')
            result['capture_pose'].update(self._capture_identity, step=result['step'], sim_time_s=result['sim_time_s'],
                                          world_frame_id='world')
        self._captures += 1
        return result

    def contain(self, reason):
        self._solved_read = None
        self.receipt['containment'] = str(reason)
        if self.timeline is not None:
            self.timeline.pause()  # no stop/reset callback or pose write

    def close(self):
        if self._closed:
            return
        self._closed = True
        errors = []
        try:
            self.contain('lifecycle teardown; not physical stop acceptance')
        except Exception as exc:
            errors.append(str(exc))
        if self._pose_annotator is not None:
            try:
                self._pose_annotator.detach()
            except Exception as exc:
                errors.append(str(exc))
            self._pose_annotator = None
        if self.readback is not None:
            for cleanup in (self.readback.detach_render_times,
                            lambda: self.readback.detach_annotators(
                                ['rgb', 'distance_to_image_plane'] if self._calibration_reader else ['rgb']),
                            self.readback._invalidate_sensor):
                # CameraSensor 6.1 has no public close: its own destructor calls
                # _invalidate_sensor (subscription + render product teardown).
                try:
                    cleanup()
                except Exception as exc:
                    errors.append(str(exc))
            self.readback = None
        self.receipt['owned_resources_closed'] = not errors
        if errors:
            raise RuntimeError('backend teardown errors: ' + '; '.join(errors))

    def shutdown(self, exit_code):
        """Attest an actual SDK.close return; fast shutdown may never return."""
        if not self._closed:
            raise RuntimeError('close owned resources before final SDK shutdown')
        if self.app is None:
            return False
        app, self.app = self.app, None
        app.close(exit_code=exit_code)
        return True
