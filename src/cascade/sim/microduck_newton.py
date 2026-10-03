"""Opt-in native Newton/Kit MicroDuck backend; optional imports stay lazy.

Bundle checks establish identity, not locomotion or physical acceptance. No
PhysX/PD fallback, SDK mutation, automatic download, or external probe imports.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import stat

REPO = Path(__file__).resolve().parents[3]
MODEL_PREFIX = 'microduck_rl/src/mjlab_microduck/robot/microduck/'
OVERVIEW_CAMERA = '/World/Overview'
OVERVIEW_RENDER_PRODUCT = '/World/OverviewRenderProduct'


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
    """Verify all 134 outputs and pinned source bytes WITHOUT USD/converter imports.

    The operator supplies the offline-admitted receipt digest. This does not
    redo offline semantic conversion validation; it refuses changed provenance,
    source files, layers, dependencies, or a different root USD.
    """
    root = Path(bundle).expanduser().absolute()
    receipt_path = safe_file(root, 'receipt.json')
    expected_sha256 = digest_token(expected_sha256)
    if (sha256(receipt_path) != expected_sha256
            or safe_file(root, 'receipt.sha256').read_text().strip() != expected_sha256):
        raise ValueError('bundle receipt SHA mismatch')
    receipt = strict_json(receipt_path.read_bytes())
    if receipt.get('schema_version') != 1 or receipt.get('collision_profile') not in ('source', 'velstand'):
        raise ValueError('unsupported bundle schema/profile')
    outputs = receipt.get('outputs')
    if not isinstance(outputs, list) or len(outputs) != 134:
        raise ValueError('require the admitted 134-output bundle')
    listed = set()
    for row in outputs:
        if set(row) != {'path', 'size', 'sha256'} or row['path'] in listed:
            raise ValueError('invalid/duplicate bundle output')
        path = safe_file(root, row['path'])
        if type(row['size']) is not int or row['size'] < 0:
            raise ValueError('invalid artifact size')
        if path.stat().st_size != row['size'] or sha256(path) != digest_token(row['sha256']):
            raise ValueError(f"bundle output hash/size mismatch: {row['path']}")
        listed.add(row['path'])
    actual = set()
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('symlink in bundle')
        if not path.is_dir():
            actual.add(path.relative_to(root).as_posix())
    if actual != listed | {'receipt.json', 'receipt.sha256'}:
        raise ValueError('bundle file set mismatch')
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
            'asset_receipt_sha256': expected_sha256, 'receipt': receipt}


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


def read_native_state(ns, *, q_indices, dof_indices, root_index, max_contacts, max_constraints,
                      q_count=21, dof_count=20):
    """Read current state_0, never an experimental persistent swapped tensor."""
    import numpy as np
    from cascade.control.microduck_policy import POLICY_JOINTS
    from cascade.sim.microduck_state import body_frame_vectors

    model, state = ns.model, ns.state_0  # reacquire each time
    arrays = {k: getattr(state, k).numpy() for k in ('body_q', 'body_qd', 'joint_q', 'joint_qd')}
    for name, a in arrays.items():
        if a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError(f'nonfinite or wrong native dtype: {name}')
    bodies = tuple(model.body_label)
    if arrays['body_q'].shape != (len(bodies), 7) or arrays['body_qd'].shape != (len(bodies), 6):
        raise ValueError('native body layout changed')
    if arrays['joint_q'].shape != (q_count,) or arrays['joint_qd'].shape != (dof_count,):
        raise ValueError('native free root + fourteen hinges required')
    pose, vel = arrays['body_q'][root_index], arrays['body_qd'][root_index]
    quat = np.array([pose[6], *pose[3:6]])
    vectors = body_frame_vectors(quat, vel[3:], vel[:3], model.body_com.numpy()[root_index])
    contacts = ns.contacts
    counts = contacts.rigid_contact_count.numpy()
    nefc = ns.solver.mjw_data.nefc.numpy()
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
    data = ns.solver.mjw_data
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
    return dict(step=ns.simulation_step_count, sim_time=float(ns.sim_time),
                position=pose[:3].astype(float).tolist(), orientation_wxyz=quat.astype(float).tolist(),
                linear_velocity=vectors['linear_velocity_origin_world'].tolist(),
                angular_velocity=vectors['angular_velocity_body'].tolist(),
                gravity_body=vectors['gravity_body'].tolist(),
                q=arrays['joint_q'][q_indices].copy(), dq=arrays['joint_qd'][dof_indices].copy(),
                joint_names=POLICY_JOINTS, contacts=tuple(sorted(names)),
                contact_count=len(active), contact_candidate_count=ncon, constraint_count=nconstraints,
                contact_semantics='active_solver_contact_constraints_not_support_force',
                contact_constraint_addresses=first_rows[active].tolist(),
                contact_pairs=[[shapes[int(pairs[0][i])], shapes[int(pairs[1][i])]] for i in active])


def prepare_native_model(ns, dof_indices, *, source_cap, newton, effort_cap=None, dof_count=20):
    """Explicit startup-only XML -> BAM replacements, with fail-closed readback."""
    import math
    import numpy as np
    from cascade.control.microduck_actuator import M6_PARAMETERS
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
    expected = {'joint_damping': .053, 'joint_armature': .0018, 'joint_effort_limit': 1e6,
                'joint_friction': .0048, 'joint_target_mode': 0, 'joint_target_ke': 0, 'joint_target_kd': 0}
    before, arrays = {}, {}
    for name, value in expected.items():
        a = getattr(ns.model, name).numpy()
        dtype = np.int32 if name == 'joint_target_mode' else np.float32
        if a.dtype != dtype or a.shape != (dof_count,) or not np.isfinite(a[dof_indices]).all():
            raise ValueError(f'unexpected native property layout: {name}')
        if not np.array_equal(a[dof_indices], np.full(14, value, dtype=dtype)):
            raise ValueError(f'unexpected native source property: {name}')
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
    return {'before': before, 'overrides': overrides, 'after': after}


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


def capture_bound_rgb(ns, app, readback, *, updates, checkpoint=lambda: None, calibration=None):
    """Main-thread capture with both physical clocks held fixed during render."""
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
    if calibration is not None:
        data, _, before_render_times = readback.get_data_bound('rgb', checkpoint=checkpoint)
    else:
        data, _ = readback.get_data('rgb')
    checkpoint()
    if data is None:
        raise RuntimeError('overview RGB unavailable')
    rgb = np.asarray(data.numpy() if hasattr(data, 'numpy') else data).copy()
    if rgb.dtype != np.uint8 or rgb.shape != (480, 640, 3):
        raise ValueError('overview must produce uint8 RGB[480,640,3]')
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
        if depth.dtype != np.float32 or depth.shape not in ((480, 640), (480, 640, 1)):
            raise ValueError('overview depth must be float32[480,640]')
        depth = depth.reshape(480, 640)
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
    checkpoint()
    times = readback.get_render_times()
    checkpoint()
    validate_render_times(times, before[1])
    if before_render_times is not None and times != before_render_times:
        raise RuntimeError('RGB-D render product changed during readback')
    if (ns.simulation_step_count, float(ns.sim_time)) != before:
        raise RuntimeError('camera read advanced physics')
    return dict(rgb=rgb, step=before[0], sim_time_s=before[1], captured_at=captured_at, render_times=times, **extra)


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


def create_overview_sensor(stage, camera, sensor_factory, *, sync_renderer, rgbd=False):
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
    if (tuple(camera.paths) != (OVERVIEW_CAMERA,)
            or stage.GetPrimAtPath(OVERVIEW_CAMERA).GetTypeName() != 'Camera'):
        raise ValueError('overview requires the exact owned camera prim')
    product = stage.DefinePrim(OVERVIEW_RENDER_PRODUCT, 'RenderProduct')
    product.CreateRelationship('camera', custom=False).SetTargets([Sdf.Path(OVERVIEW_CAMERA)])
    product.CreateAttribute('resolution', Sdf.ValueTypeNames.Int2, custom=False,
                            variability=Sdf.VariabilityUniform).Set(Gf.Vec2i(640, 480))
    color_path = Sdf.Path(OVERVIEW_RENDER_PRODUCT + '/LdrColor')
    color = stage.DefinePrim(color_path, 'RenderVar')
    color.CreateAttribute('sourceName', Sdf.ValueTypeNames.String, custom=False,
                          variability=Sdf.VariabilityUniform).Set('LdrColor')
    product.CreateRelationship('orderedVars', custom=False).SetTargets([color_path])
    sync_renderer()
    sensor = sensor_factory(camera, resolution=(480, 640),
                            annotators=['rgb', 'distance_to_image_plane'] if rgbd else ['rgb'])
    actual = sensor.render_product.GetPrim()
    resolution = actual.GetAttribute('resolution').Get()
    observed = dict(path=str(actual.GetPath()),
                    camera_targets=[str(p) for p in actual.GetRelationship('camera').GetTargets()],
                    render_resolution=None if resolution is None else list(resolution),
                    sensor_resolution=None if sensor.resolution is None else list(sensor.resolution),
                    ordered_vars=[str(p) for p in actual.GetRelationship('orderedVars').GetTargets()])
    if (observed['path'] != OVERVIEW_RENDER_PRODUCT or observed['camera_targets'] != [OVERVIEW_CAMERA]
            or observed['render_resolution'] != [640, 480] or observed['sensor_resolution'] != [480, 640]):
        raise RuntimeError('CameraSensor did not adopt the exact authored overview render product: '
                           + json.dumps(observed, sort_keys=True))
    return sensor


class KitNewtonBackend:
    """Kit main-thread lifecycle. Constructing this object imports no SDK.

    No world-reset method is exposed. Start a new process/epoch for another
    episode. Initial HOME/FK is the only pose write; faults pause and terminate.
    """
    def __init__(self, args, admission, experience):
        self.args, self.admission, self.experience = args, admission, experience
        self.app = self.ns = self.readback = self.timeline = None
        self.receipt = {}
        self._closed = False
        self._captures = 0
        self._last_support_solve = None
        self._calibration_reader = None
        self.signals = None
        self._solver_graph = None
        self._sdk_recipe = getattr(args, 'sdk_recipe', None)
        self._solved_read = None
        self._reuse_solved_read = getattr(args, 'reuse_solved_read', False)
        if type(self._reuse_solved_read) is not bool or (self._reuse_solved_read
                and getattr(args, 'solver_cuda_graph', False) is not True):
            raise ValueError('same-solve read reuse requires explicit bound solver graph mode')

    def _checkpoint(self):
        if self.signals is not None:
            self.signals.checkpoint(persistent=True)

    def _app_config(self):
        return {'headless': True, 'disable_viewport_updates': True,
                'multi_gpu': False, 'width': 320, 'height': 240, 'renderer': 'RayTracedLighting',
                'physics_gpu': int(self.args.device.split(':')[1])}

    def open(self):
        import sys
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
        roots = self._author_robots(stage)
        ground = UsdGeom.Plane.Define(stage, '/World/Ground')
        ground.CreateAxisAttr('Z')
        UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
        material = UsdShade.Material.Define(stage, '/World/GroundMaterial')
        mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        mat.CreateStaticFrictionAttr(1.)
        mat.CreateDynamicFrictionAttr(1.)
        mat.CreateRestitutionAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(material, materialPurpose='physics')
        for root_path in roots:
            bind_ground(stage, [ground.GetPath()], root_path=root_path)
        scene = UsdPhysics.Scene.Define(stage, '/World/PhysicsScene')
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0., 0., -1.))
        scene.CreateGravityMagnitudeAttr(9.81)
        setup_physics(SM, dt=.005, device=self.args.device, require_cuda=True)
        cfg = copy.deepcopy(get_newton_config())
        cfg.solver_cfg = MuJoCoSolverConfig()
        cfg.num_substeps, cfg.use_cuda_graph, cfg.time_step_app = 1, False, False
        cfg.solver_cfg.nconmax, cfg.solver_cfg.njmax = 512, 2400
        cfg.collision_cfg.rigid_contact_max = 512
        cfg.solver_cfg.use_mujoco_contacts = True
        configure_outputs(cfg, self._sdk_recipe)
        configure_newton(cfg)
        self._checkpoint()
        self.receipt['configuration'] = dict(num_substeps=1, use_cuda_graph=False, time_step_app=False,
                                             nconmax=512, njmax=2400, rigid_contact_max=512, use_mujoco_contacts=True)
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
        import inspect
        from cascade.sim.microduck_solver_graph import SolverGraphContract
        self._solver_graph = SolverGraphContract(ns,
            enabled=getattr(self.args, 'solver_cuda_graph', False), wp=wp, dt=self._dt,
            source_path=inspect.getfile(type(ns)), sdk_recipe=self._sdk_recipe)
        self.receipt['configuration']['use_cuda_graph'] = self._solver_graph.enabled
        self.receipt['configuration']['solver_graph_stage_sha256'] = self._solver_graph.source_sha256
        self.receipt['configuration']['reuse_solved_read'] = self._reuse_solved_read

    def _author_robots(self, stage):
        from pxr import UsdGeom
        root = UsdGeom.Xform.Define(stage, '/World/MicroDuck').GetPrim()
        root.GetReferences().AddReference(self.admission['asset'])
        self.receipt['disabled_source_actuators'] = disable_source_actuators(stage)
        return ('/World/MicroDuck',)

    def _bind_native_model(self, ns):
        import numpy as np
        import newton
        from cascade.control.microduck_policy import HOME_Q
        from cascade.control.newton_bam import NewtonBamAdapter
        from cascade.sim.microduck_state import newton_joint_indices
        from cascade.sim.microduck_contact_support import extraction_provenance, support_contract
        qs, ds = newton_joint_indices(ns.model.joint_label, ns.model.joint_q_start.numpy(), ns.model.joint_qd_start.numpy())
        self.q_indices, self.dof_indices = qs, ds
        self.root_index = list(ns.model.body_label).index('/World/MicroDuck/Geometry/trunk_base')
        self._model = ns.model
        self._layout = (tuple(ns.model.joint_label), tuple(ns.model.body_label), tuple(ns.model.shape_label))
        self.receipt['native_labels'] = dict(zip(('joints', 'bodies', 'shapes'), self._layout))
        self.receipt['native_body_properties'] = read_native_body_properties(ns)
        self.receipt['support_contract'] = support_contract(ns.model.shape_label)
        self.receipt['support_contract']['gravity_world_m_s2'] = self.receipt['native_body_properties']['gravity_world_m_s2'][:]
        self.receipt['support_extraction'] = extraction_provenance(sdk_recipe=self._sdk_recipe)
        self._checkpoint()
        self.receipt['native_model_properties'] = prepare_native_model(ns, ds, source_cap=.96, newton=newton,
            effort_cap=self.admission['bam_params']['joint_effort_limit'])
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
        eye, target = (Gf.Vec3d(*v) for v in self._camera_pose())
        quat = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0., 0., 1.)).GetInverse().ExtractRotationQuat()
        camera = RtxCamera(OVERVIEW_CAMERA, tick_rate=0., translations=list(eye),
                           orientations=[quat.GetReal(), *quat.GetImaginary()])
        camera.camera.set_clipping_ranges(.005, 20.)
        optics = UsdGeom.Camera(stage.GetPrimAtPath(OVERVIEW_CAMERA))
        optics.GetFocalLengthAttr().Set(18.)
        optics.GetHorizontalApertureAttr().Set(20.955)
        optics.GetVerticalApertureAttr().Set(20.955 * 480 / 640)
        def sync_renderer():
            self.receipt['camera_authoring_sync'] = synchronize_camera_authoring(
                self.app, self.timeline, self.SM, acquire_stage(), checkpoint=self._checkpoint)
        rgbd = getattr(self.args, 'camera_rgbd', False)
        sensor = create_overview_sensor(stage, camera, CameraSensor, sync_renderer=sync_renderer, rgbd=rgbd)
        self._checkpoint()
        product = str(sensor.render_product.GetPath())
        self.readback = CpuCameraReadback(sensor, render_product_id=product)
        times = self.readback._render_times
        for name in ('rpFabricTime', 'IsaacReadSimulationTime'):
            annotator = rep.AnnotatorRegistry.get_annotator(name)
            annotator.attach([product])
            times[name] = annotator
        self.receipt['camera'] = dict(name='overview', render_product=product, resolution=[480, 640])
        self._calibration_reader = None
        if rgbd:
            from cascade.sim.mobile_rgbd import calibration_record, read_static_calibration
            self._calibration_reader = lambda: read_static_calibration(stage, OVERVIEW_CAMERA)
            observed, digest = calibration_record(self._calibration_reader())
            self.receipt['rgbd_camera'] = {'calibration': observed, 'calibration_sha256': digest,
                                         'render_product': product, 'annotator': 'distance_to_image_plane'}

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

    def capture(self):
        self._checkpoint()
        self._guard()
        result = capture_bound_rgb(self.ns, self.app, self.readback,
            updates=16 if self._captures == 0 else 3, checkpoint=self._checkpoint,
            calibration=self._calibration_reader)
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
