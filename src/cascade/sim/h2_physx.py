"""Kit main-thread PhysX owner for the Unitree H2 bundle (Isaac Sim 6.2 build).

Constructing this object imports no SDK; ``open()`` does. The deployment chain
(spawn, exported gains, observation history, inference, target write) is
NVIDIA's ``isaacsim.robot.policy.examples.RobotPolicyRunner`` pointed at the
PINNED local bundle files; CASCADE owns everything around it: the stage, the
asset root, manual physics stepping (``SimulationManager.step``), the state
reads the controller and verifier consume, contact tracking, fall criteria,
rendering and the receipt. The runner is handed only the admitted planar
twist; it is never a second command path.

Two independent validations must agree before an episode starts: the exported
contract parsed by ``H2PolicyContract`` and what the runner actually configured
on the articulation (joint names, timestep, decimation, PD gains, effort and
velocity limits, armature, self-collision, solver iterations).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
import urllib.request

import numpy as np

OVERVIEW_CAMERA = '/World/OverviewCamera'
ROBOT_PATH = '/World/H2'
GROUND_PATH = '/World/Ground'
CONTACT_FORCE_THRESHOLD_N = 1.0
MAX_CONTACTS = 256  # solved contact points per step the PhysX view may report
# NVIDIA's h2_standalone.py removes exactly this zero-width Agile action record
# (upper-body joints commanded autonomously inside the training environment, no
# Isaac Sim deployment); the owner holds those joints at their defaults instead.
AUTONOMOUS_ACTION_PATH = 'agile.rl_env.mdp.actions.random_actions.RandomPositionAction'


def derive_h2_policy_binding(artifact, env_config):
    """Descriptor-derived binding minus the autonomous upper-body record (fails on any drift)."""
    from isaacsim.robot.policy.examples.binding import derive_binding
    del env_config
    descriptor = artifact.load_descriptor()
    records = list(descriptor.get('actions', ()))
    autonomous = [r for r in records if r.get('full_path') == AUTONOMOUS_ACTION_PATH]
    if len(autonomous) != 1 or int(np.prod(autonomous[0].get('shape', ()))) != 0:
        raise ValueError('expected exactly one zero-width Agile RandomPositionAction in the descriptor')
    policy_descriptor = dict(descriptor)
    policy_descriptor['actions'] = [r for r in records if r.get('full_path') != AUTONOMOUS_ACTION_PATH]
    return derive_binding(policy_descriptor)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_layer_bytes(identifier: str, *, timeout_s: float = 120.0) -> bytes:
    """Bytes of a consumed stage layer: local file or the public asset root over HTTPS."""
    if identifier.startswith(('http://', 'https://')):
        with urllib.request.urlopen(identifier, timeout=timeout_s) as response:
            if response.status != 200:
                raise RuntimeError(f'{identifier}: HTTP {response.status}')
            return response.read()
    return Path(identifier).read_bytes()


def _np(value):
    """Host copy of a Warp/torch/numpy array."""
    if hasattr(value, 'numpy'):
        value = value.numpy()
    return np.asarray(value)


class KitH2PhysxBackend:
    def __init__(self, args, admission, contract):
        self.args, self.admission, self.contract = args, admission, contract
        self.receipt = {'engine': 'physx', 'owner': 'cascade.sim.h2_physx.KitH2PhysxBackend'}
        self.app = self.runner = self.articulation = self.contacts = None
        self.SM = self.RM = self.timeline = None
        self._steps_since_bootstrap = 0
        self._bootstrap = None
        self._dof_indices = None
        self._contact_names = ()
        self._annotator = self._render_product = None
        self.signals = None
        self.opened = self.closed = False

    # ---- lifecycle -----------------------------------------------------------------------

    def _checkpoint(self):
        if self.signals is not None:
            self.signals.checkpoint(persistent=True)

    def open(self):
        if self.opened:
            raise RuntimeError('backend already opened')
        self.opened = True
        from isaacsim import SimulationApp
        self.app = SimulationApp({'headless': True, 'extra_args': [
            '--/exts/isaacsim.core.simulation_manager/default_engine=physx']})
        self._checkpoint()
        self._initialize()

    def _initialize(self):
        from functools import partial
        import carb
        import omni.timeline
        import omni.usd
        from isaacsim.core.simulation_manager import SimulationManager as SM
        from isaacsim.core.rendering_manager import RenderingManager as RM
        from isaacsim.core.experimental.utils import app as app_utils
        from isaacsim.core.experimental.prims import RigidPrim
        from isaacsim.core.version import get_version
        from isaacsim.robot.policy.examples import PolicyArtifact, PolicyEnvConfig, PolicySpec, RobotPolicyRunner
        from pxr import Gf, UsdGeom, UsdLux, UsdPhysics, UsdShade

        args, contract, bundle = self.args, self.contract, self.admission['bundle']
        self.SM, self.RM = SM, RM
        self.timeline = omni.timeline.get_timeline_interface()
        settings = carb.settings.get_settings()
        # Pin the public asset root: the dev build may otherwise resolve an internal server.
        settings.set('/persistent/isaac/asset_root/default', bundle['asset_root'])
        if SM.get_active_physics_engine() != 'physx':
            SM.switch_physics_engine('physx')
        SM.set_physics_sim_device(args.device)
        self._checkpoint()
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.Xform.Define(stage, '/World')
        scene = UsdPhysics.Scene.Define(stage, '/World/PhysicsScene')
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0., 0., -1.))
        scene.CreateGravityMagnitudeAttr(9.81)
        ground = UsdGeom.Plane.Define(stage, GROUND_PATH)
        ground.CreateAxisAttr('Z')
        ground.CreateWidthAttr(float(args.ground_visual_m))
        ground.CreateLengthAttr(float(args.ground_visual_m))
        UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
        material = UsdShade.Material.Define(stage, '/World/GroundMaterial')
        mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        mat.CreateStaticFrictionAttr(1.)
        mat.CreateDynamicFrictionAttr(1.)
        mat.CreateRestitutionAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(material, materialPurpose='physics')
        light = UsdLux.DistantLight.Define(stage, '/World/DistantLight')
        light.CreateIntensityAttr(1500.)
        dome = UsdLux.DomeLight.Define(stage, '/World/DomeLight')
        dome.CreateIntensityAttr(400.)
        # Visual-only distance markers (no collision API, no physics): the policy sees
        # the same flat ground, the video gets a metric reference along +x and +y.
        markers = []
        for metres in range(-2, 13):
            for axis, (x, y) in (('x', (float(metres), 0.)), ('y', (0., float(metres)))):
                if metres == 0 and axis == 'y':
                    continue
                path = f'/World/Markers/{axis}_{metres + 2:02d}'
                cube = UsdGeom.Cube.Define(stage, path)
                cube.CreateSizeAttr(1.0)
                cube.CreatePurposeAttr(UsdGeom.Tokens.render)  # never a collider, never in physics
                xf = UsdGeom.Xformable(cube.GetPrim())
                xf.ClearXformOpOrder()
                xf.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.005))
                xf.AddScaleOp().Set(Gf.Vec3f(0.08, 0.08, 0.01) if metres % 5 else Gf.Vec3f(0.14, 0.14, 0.01))
                colour = (0.95, 0.35, 0.1) if axis == 'x' else (0.1, 0.45, 0.95)
                cube.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
                markers.append(path)
        lane = UsdGeom.Cube.Define(stage, '/World/Markers/lane_x')
        lane.CreateSizeAttr(1.0)
        lane.CreatePurposeAttr(UsdGeom.Tokens.render)
        lane_xf = UsdGeom.Xformable(lane.GetPrim())
        lane_xf.ClearXformOpOrder()
        lane_xf.AddTranslateOp().Set(Gf.Vec3d(5.0, 0., 0.002))
        lane_xf.AddScaleOp().Set(Gf.Vec3f(14.0, 0.02, 0.004))
        lane.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.85, 0.85)])
        self.receipt['visual_markers'] = {'count': len(markers) + 1, 'collision': False, 'purpose': 'render',
                                          'spacing_m': 1.0}
        self.receipt['ground'] = {'path': GROUND_PATH, 'collider': 'UsdGeom.Plane', 'static_friction': 1.,
                                  'dynamic_friction': 1., 'restitution': 0., 'visual_m': float(args.ground_visual_m)}
        self._checkpoint()

        files = bundle['files']
        artifact = PolicyArtifact.from_files(str(args.policy), str(self.admission['env_yaml']),
                                             str(self.admission['descriptor_yaml']),
                                             model_sha256=files['policy.pt']['sha256'])
        env_config = PolicyEnvConfig.from_file(artifact.env_config_path)
        usd_url = f"{bundle['asset_root']}/{files['H2.usda']['path']}"
        spec = PolicySpec(name='unitree_h2_velocity_history', engines={'physx': artifact}, usd_path=usd_url,
                          binding=partial(derive_h2_policy_binding, artifact))
        self.runner = RobotPolicyRunner(spec, prim_path=ROBOT_PATH, training_engine='physx')
        self.articulation = self.runner.spawn()
        self._env_config = env_config
        self._checkpoint()
        if not math.isclose(float(self.runner.physics_dt), contract.physics_dt, rel_tol=0, abs_tol=1e-12):
            raise RuntimeError('deployment chain physics dt differs from the pinned contract')
        if int(self.runner.decimation) != contract.decimation:
            raise RuntimeError('deployment chain decimation differs from the pinned contract')
        SM.set_physics_dt(contract.physics_dt)
        RM.set_dt(contract.physics_dt * int(env_config.timing.render_interval))
        self.receipt['asset'] = {'usd': usd_url, 'prim_path': ROBOT_PATH,
                                 'root_sha256_pinned': files['H2.usda']['sha256']}
        variant_sets = stage.GetPrimAtPath(ROBOT_PATH).GetVariantSets()
        self.receipt['asset']['variant_sets'] = {name: variant_sets.GetVariantSet(name).GetVariantSelection()
                                                 for name in variant_sets.GetNames()}
        # Hash every consumed layer so the identity binds the composed stage, not a URL.
        layers = []
        for layer in stage.GetUsedLayers():
            if layer.anonymous:
                continue
            data = fetch_layer_bytes(layer.identifier)
            layers.append({'identifier': layer.identifier, 'sha256': sha256_bytes(data), 'bytes': len(data)})
        if not any(row['identifier'] == usd_url and row['sha256'] == files['H2.usda']['sha256'] for row in layers):
            raise RuntimeError('the composed stage did not consume the pinned H2.usda root')
        self.receipt['consumed_asset_layers'] = sorted(layers, key=lambda r: r['identifier'])
        self._checkpoint()

        # Solved contact reactions between EVERY robot link and the ground: the support
        # evidence the independent verifier consumes (a non-foot link on the ground is a
        # forbidden contact, so partial coverage would hide exactly what it must see).
        link_names = list(self.articulation.link_names)
        link_paths = list(self.articulation.link_paths[0])
        wanted = tuple(contract.foot_bodies) + tuple(contract.illegal_contact_bodies)
        missing = [b for b in wanted if b not in link_names]
        if missing:
            raise RuntimeError(f'articulation lacks the training bodies {missing}')
        self.contacts = RigidPrim(link_paths, contact_filter_paths=[GROUND_PATH], max_contact_count=MAX_CONTACTS)
        self.contacts.set_enabled_contact_tracking(True)
        self._contact_names = tuple(link_names)
        self._link_paths = tuple(link_paths)
        self.receipt['contact_tracking'] = {'bodies': dict(zip(link_names, link_paths)), 'filter': [GROUND_PATH],
                                            'threshold_n': CONTACT_FORCE_THRESHOLD_N, 'max_contact_count': MAX_CONTACTS,
                                            'granularity': 'PhysX reports contacts per rigid link; shapes are link paths'}
        # The asset composes its collision geometry through instanceable payloads:
        # plain Traverse() skips instance proxies and sees four colliders of ~forty.
        from pxr import Usd
        colliders = []
        root_prim = stage.GetPrimAtPath(ROBOT_PATH)
        for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                colliders.append(str(prim.GetPath()))
        self.receipt['active_colliders'] = colliders
        foot_paths = [link_paths[link_names.index(b)] for b in contract.foot_bodies]
        if not any(any(c.startswith(f + '/') for c in colliders) for f in foot_paths):
            raise RuntimeError('no collider found under the training foot links; variants must be checked')
        self.receipt['support_contract'] = {
            'version': 1, 'robot_shapes': list(link_paths), 'foot_shapes': foot_paths,
            'ground_shapes': [GROUND_PATH], 'gravity_world_m_s2': [0., 0., -9.81]}
        self._create_camera(stage, UsdGeom, Gf)
        self._checkpoint()

        # Physics initialization owns the first app update; manual stepping afterwards.
        app_utils.play(commit=True)
        self.app.update()
        self._checkpoint()
        if SM.get_active_physics_engine() != 'physx' or not SM.is_simulating():
            raise RuntimeError('PhysX did not start simulating')
        device = str(SM.get_physics_sim_device())
        if not device.startswith('cuda'):
            raise RuntimeError(f'physics is not on CUDA: {device}')
        # Build the runtime (exported defaults and gains), reset to the default standing
        # pose, then reset the runtime from that clean state so the first owned solve is
        # the first policy tick (the runner's tick counter starts at zero).
        self.runner.initialize()
        self.articulation.reset_to_default_state()
        view = SM.get_physics_simulation_view()
        if view is not None:
            view.update_articulations_kinematic()
        self.runner.initialize()
        self._checkpoint()
        dof_names = list(self.articulation.dof_names)
        if set(contract.joint_names) - set(dof_names):
            raise RuntimeError('articulation DOF names do not cover the contract joints')
        self._dof_indices = np.array([dof_names.index(j) for j in contract.policy_joint_names])
        self._all_indices = np.array([dof_names.index(j) for j in contract.joint_names])
        self.receipt['gains_check'] = self._verify_gains(dof_names)
        self_collisions = bool(_np(self.articulation.get_enabled_self_collisions())[0])
        iterations = tuple(int(_np(v)[0]) for v in self.articulation.get_solver_iteration_counts())
        self.receipt['articulation'] = {'dof_names': dof_names, 'link_names': link_names,
                                        'enabled_self_collisions': self_collisions,
                                        'solver_iterations': list(iterations)}
        if self_collisions != contract.enabled_self_collisions or iterations != tuple(contract.solver_iterations):
            raise RuntimeError('articulation self-collision/solver iterations differ from the training export')
        self._bootstrap = (int(SM.get_num_physics_steps()), float(SM.get_simulation_time()))
        self._steps_since_bootstrap = 0
        self.receipt['bootstrap_clocks'] = {'manager_step': self._bootstrap[0], 'manager_time_s': self._bootstrap[1],
                                            'manager_dt_s': float(SM.get_physics_dt())}
        self.receipt['physics_device'] = device
        self.receipt['runtime_versions'] = {'isaac_sim': list(get_version())}
        self.receipt['deployment_chain'] = {
            'runner': 'isaacsim.robot.policy.examples.RobotPolicyRunner', 'training_engine': 'physx',
            'actuator_model': type(getattr(self.runner, '_actuators', None)).__name__,
            'controller': type(getattr(self.runner, '_controller', None)).__name__,
            'policy_sha256': files['policy.pt']['sha256']}

    def _verify_gains(self, dof_names):
        """The deployment chain's actuation must equal the exported contract, read back two ways.

        The runner deploys the training's explicit ``DelayedDCMotor`` groups as an actuator
        model (PD inside the model, torque clamps, delay) and leaves the physics drives of
        those joints at zero gain in effort mode; joints the policy does not command keep the
        asset-authored drives. Both facts are recorded; the model parameters are compared
        with the contract per joint and any difference refuses the episode.
        """
        contract = self.contract
        stiff, damp = (_np(v)[0] for v in self.articulation.get_dof_gains())
        efforts = _np(self.articulation.get_dof_max_efforts())[0]
        velocities = _np(self.articulation.get_dof_max_velocities())[0]
        armatures = _np(self.articulation.get_dof_armatures())[0]
        drives = {}
        for joint in contract.joint_names:
            i = dof_names.index(joint)
            drives[joint] = {'stiffness': float(stiff[i]), 'damping': float(damp[i]), 'effort_limit': float(efforts[i]),
                             'velocity_limit': float(velocities[i]), 'armature': float(armatures[i]),
                             'commanded_by_policy': joint in contract.policy_joint_names}
        specs = list(self._env_config.actuator_model_specs(list(contract.policy_joint_names)))
        models, mismatches = {}, []
        for spec in specs:
            for k, joint in enumerate(spec.joints):
                models[joint] = {'class': spec.class_name, 'stiffness': float(spec.stiffness[k]),
                                 'damping': float(spec.damping[k]), 'effort_limit': float(spec.effort_limit[k]),
                                 'velocity_limit': float(spec.velocity_limit[k]),
                                 'saturation_effort': float(spec.saturation_effort[k]), 'max_delay': int(spec.max_delay)}
        for joint in contract.policy_joint_names:
            expected = contract.gains[joint]
            if joint in models:
                actual = models[joint]
                for key in ('stiffness', 'damping', 'effort_limit', 'velocity_limit'):
                    if not math.isclose(actual[key], getattr(expected, key), rel_tol=1e-3, abs_tol=1e-6):
                        mismatches.append((joint, 'model.' + key, actual[key], getattr(expected, key)))
                # Effort mode: the physics drive must not add a second PD on top of the model.
                if drives[joint]['stiffness'] != 0. or drives[joint]['damping'] != 0.:
                    mismatches.append((joint, 'drive gains with an actuator model', drives[joint]['stiffness'], 0.))
            else:
                for key in ('stiffness', 'damping', 'effort_limit'):
                    if not math.isclose(drives[joint][key], getattr(expected, key), rel_tol=1e-3, abs_tol=1e-6):
                        mismatches.append((joint, 'drive.' + key, drives[joint][key], getattr(expected, key)))
            if not math.isclose(drives[joint]['armature'], expected.armature, rel_tol=1e-3, abs_tol=1e-6):
                mismatches.append((joint, 'armature', drives[joint]['armature'], expected.armature))
        if mismatches:
            raise RuntimeError('deployed actuation differs from the exported actuator contract: '
                               + json.dumps(mismatches[:8]))
        held_deviation = {j: {'authored_drive': {k: drives[j][k] for k in ('stiffness', 'damping', 'effort_limit')},
                              'training': {'stiffness': contract.gains[j].stiffness, 'damping': contract.gains[j].damping,
                                           'effort_limit': contract.gains[j].effort_limit}}
                          for j in contract.held_joint_names}
        return {'policy_joints_checked': len(contract.policy_joint_names), 'actuator_models': models,
                'drives': drives, 'actuator_model_in_use': bool(models),
                'held_joints_keep_asset_drives': held_deviation}

    def _create_camera(self, stage, UsdGeom, Gf):
        import omni.replicator.core as rep
        args = self.args
        camera = UsdGeom.Camera.Define(stage, OVERVIEW_CAMERA)
        camera.CreateFocalLengthAttr(18.0)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 200.0))
        xform = UsdGeom.Xformable(camera.GetPrim())
        xform.ClearXformOpOrder()
        self._camera_op = xform.AddTransformOp()
        self._Gf = Gf
        self._aim_camera(Gf.Vec3d(*args.camera_eye), Gf.Vec3d(*args.camera_target))
        height, width = self.overview_shape
        self._render_product = rep.create.render_product(OVERVIEW_CAMERA, (width, height))
        self._annotator = rep.AnnotatorRegistry.get_annotator('rgb')
        self._annotator.attach(self._render_product)
        self.receipt['overview_camera'] = {'path': OVERVIEW_CAMERA, 'eye': list(args.camera_eye),
                                           'target': list(args.camera_target), 'resolution': [width, height],
                                           'focal_length_mm': 18.0}

    def _aim_camera(self, eye, target):
        Gf = self._Gf
        # Camera looks down -Z: build a look-at frame with Z up in the world.
        forward = (target - eye).GetNormalized()
        right = Gf.Cross(forward, Gf.Vec3d(0, 0, 1)).GetNormalized()
        up = Gf.Cross(right, forward)
        self._camera_op.Set(Gf.Matrix4d(right[0], right[1], right[2], 0, up[0], up[1], up[2], 0,
                                        -forward[0], -forward[1], -forward[2], 0, eye[0], eye[1], eye[2], 1))

    def _follow(self):
        """Chase camera: keep the authored eye/target offsets relative to the robot's x/y."""
        if not getattr(self.args, 'camera_follow', False):
            return
        Gf = self._Gf
        positions, _ = (_np(v)[0] for v in self.articulation.get_world_poses())
        base = Gf.Vec3d(float(positions[0]), float(positions[1]), 0.)
        eye = base + Gf.Vec3d(*self.args.camera_eye)
        target = base + Gf.Vec3d(*self.args.camera_target)
        self._aim_camera(eye, target)

    @property
    def overview_shape(self):
        width, height = (int(v) for v in str(self.args.overview_resolution).lower().split('x'))
        return height, width

    # ---- stepper protocol ----------------------------------------------------------------

    @property
    def dt(self):
        return float(self.contract.physics_dt)

    @property
    def physics_clock(self):
        return (self._bootstrap[0] + self._steps_since_bootstrap, float(self.SM.get_simulation_time()))

    @property
    def policy_due(self):
        return int(getattr(self.runner, '_tick', 0)) % self.contract.decimation == 0

    def read(self):
        contract = self.contract
        positions, orientations = (_np(v)[0] for v in self.articulation.get_world_poses())
        linear, angular_world = (_np(v)[0] for v in self.articulation.get_velocities())
        q_all = _np(self.articulation.get_dof_positions())[0]
        dq_all = _np(self.articulation.get_dof_velocities())[0]
        from cascade.sim.microduck_state import body_frame_vectors
        vectors = body_frame_vectors(orientations.astype(float).tolist(), angular_world.astype(float).tolist(),
                                     [0., 0., 0.], [0., 0., 0.])
        forces = _np(self.contacts.get_net_contact_forces(dt=self.dt))
        magnitudes = np.linalg.norm(forces.reshape(len(self._contact_names), 3), axis=1)
        contacts = tuple(name for name, f in zip(self._contact_names, magnitudes) if f > CONTACT_FORCE_THRESHOLD_N)
        step, sim_time = self.physics_clock
        return {'step': step, 'sim_time': sim_time, 'joint_names': contract.policy_joint_names,
                'q': q_all[self._dof_indices].astype(np.float32), 'dq': dq_all[self._dof_indices].astype(np.float32),
                'position': positions.astype(float), 'orientation_wxyz': orientations.astype(float),
                'linear_velocity': linear.astype(float), 'angular_velocity': vectors['angular_velocity_body'],
                'gravity_body': vectors['gravity_body'], 'contacts': contacts,
                'contact_forces_n': {name: float(f) for name, f in zip(self._contact_names, magnitudes) if f > 0.},
                'support_contacts': self._solved_contacts(),
                'held_joint_positions': q_all[self._all_indices].astype(float).tolist()}

    def _solved_contacts(self):
        """Per-point solved normal reactions (ground -> link) from PhysX's contact view.

        Shape A is the ground, shape B the robot link; the A->B normal is oriented
        along the ground's up axis (exact for the plane z=0), the force on B is the
        normal reaction only (friction is reported separately by PhysX and omitted).
        """
        normal_forces, points, normals, _distances, counts, starts = self.contacts.get_contact_force_data(dt=self.dt)
        normal_forces, points, normals = _np(normal_forces).reshape(-1), _np(points).reshape(-1, 3), _np(normals).reshape(-1, 3)
        counts, starts = _np(counts).reshape(len(self._link_paths), -1), _np(starts).reshape(len(self._link_paths), -1)
        rows = []
        for link, path in enumerate(self._link_paths):
            for k in range(int(counts[link, 0])):
                index = int(starts[link, 0]) + k
                force = float(normal_forces[index])
                if not force > 0.:
                    continue
                normal = normals[index].astype(float)
                norm = float(np.linalg.norm(normal))
                if norm == 0.:
                    continue
                normal = normal / norm
                if normal[2] < 0.:
                    normal = -normal
                rows.append({'shape_a_id': 0, 'shape_b_id': link + 1, 'shape_a': GROUND_PATH, 'shape_b': path,
                             'force_on_b_world_n': (force * normal).tolist(), 'normal_force_n': force,
                             'normal_a_to_b_world': normal.tolist(), 'point_world_m': points[index].astype(float).tolist()})
        return rows

    def control(self, command):
        tick_before = int(self.runner._tick)
        due = tick_before % self.contract.decimation == 0
        self.runner.step(self.dt, [float(v) for v in command])
        record = {'runner_tick': tick_before}
        controller = getattr(self.runner, '_controller', None)
        if due and controller is not None and getattr(controller, '_last_action', None) is not None:
            record['raw_action'] = [float(v) for v in np.asarray(controller._last_action).reshape(-1)]
        return record

    def step(self):
        # One PhysX solve; no app update, render or target write here.
        self.SM.step(steps=1, update_fabric=bool(self.SM.is_fabric_enabled()))
        self._steps_since_bootstrap += 1

    def capture(self):
        before = self.physics_clock
        captured_at = time.monotonic()
        height, width = self.overview_shape
        rgb = None
        # The render product delivers its first frames asynchronously; render (no physics:
        # /app/player/playSimulations is off inside RenderingManager.render) until a frame
        # of the authored resolution arrives, within a small bound.
        self._follow()
        for attempt in range(12):
            self.RM.render()
            self._checkpoint()
            data = self._annotator.get_data()
            candidate = np.asarray(data.numpy() if hasattr(data, 'numpy') else data)
            if candidate.ndim == 3 and candidate.shape[:2] == (height, width) and candidate.shape[2] >= 3:
                rgb = np.ascontiguousarray(candidate[:, :, :3]).astype(np.uint8)
                break
        if rgb is None:
            raise RuntimeError('overview RGB unavailable after bounded renders')
        if self.physics_clock != before:
            raise RuntimeError('render advanced physics')
        return {'rgb': rgb, 'step': before[0], 'sim_time_s': before[1], 'captured_at': captured_at,
                'renders': attempt + 1}

    def contain(self, reason):
        self.receipt['containment'] = reason
        if self.timeline is not None:
            try:
                self.timeline.pause()  # no stop/reset callback or pose write
            except Exception as exc:  # noqa: BLE001 - recorded, never masks the fault
                self.receipt['containment_error'] = str(exc)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.runner is not None:
            try:
                self.runner.close()
            except Exception as exc:  # noqa: BLE001
                self.receipt['runner_close_error'] = str(exc)
        if self._annotator is not None:
            try:
                self._annotator.detach()
            except Exception as exc:  # noqa: BLE001
                self.receipt['annotator_detach_error'] = str(exc)

    def shutdown(self, exit_code):
        if self.app is None:
            return False
        if self.timeline is not None:
            try:
                self.timeline.stop()
            except Exception as exc:  # noqa: BLE001
                self.receipt['timeline_stop_error'] = str(exc)
        self.app.close(exit_code=exit_code)
        return True
