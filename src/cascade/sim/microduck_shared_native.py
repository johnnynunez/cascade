"""Bounded native shared-scene foundation; no network command admission.

Each robot has its own policy, battery group, coordinates and model identity.
The single Kit owner alone solves and captures; initial placement is the only
pose write. This diagnostic does not establish walking or fleet task success.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import numpy as np

from cascade.control.mobile_telemetry import _public_contacts

from .microduck_newton import (OVERVIEW_CAMERA, KitNewtonBackend, author_collision_hull_limits, neutralize_asset_actuation,
                              prepare_native_model, read_native_body_properties,
                              _read_native_states)
from .microduck_shared import _SharedSupport, bind_scene


CHOREOGRAPHY_MIN_SEPARATION_M = .35
"""Floor between explicit spawn positions (the MicroDuck footprint is about 0.15 m wide)."""


def placements(count, spacing, layout='grid', route_m=0., positions=None):
    """Disjoint robot namespaces and initial positions; ``layout`` is an explicit choice.

    ``grid`` (the retained layout) is a square grid of ``spacing`` metres starting at the
    origin. ``line`` puts the robots on x=0 along y, centred on the origin, so that every
    robot owns a lane along +x: a robot walking ``route_m`` metres forward crosses no
    other robot's start position. Lanes need less separation than the grid (robots
    never share a lane), so ``line`` accepts a 1 m spacing; both floors are explicit.
    ``choreography`` takes explicit ``positions`` (one [x, y] per robot, at least 0.35 m
    apart, all facing +x) from a showcase choreography; ``spacing`` is ignored there.
    ``route_m`` only frames the overview camera; it writes no pose and bounds no motion.
    """
    if type(count) is not int or not 1 <= count <= 12:
        raise ValueError('robot count must be in 1..12')
    if layout not in ('grid', 'line', 'choreography'):
        raise ValueError("layout must be 'grid', 'line' or 'choreography'")
    if type(route_m) not in (int, float) or not math.isfinite(route_m) or route_m < 0.:
        raise ValueError('route_m must be a finite nonnegative length')
    if layout == 'choreography':
        if not isinstance(positions, (list, tuple)) or len(positions) != count:
            raise ValueError('choreography layout needs one explicit [x, y] position per robot')
        points = []
        for p in positions:
            if (not isinstance(p, (list, tuple)) or len(p) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in p)):
                raise ValueError('choreography positions must be finite [x, y] pairs')
            points.append((float(p[0]), float(p[1])))
        for i, a in enumerate(points):
            for b in points[i + 1:]:
                if math.dist(a, b) < CHOREOGRAPHY_MIN_SEPARATION_M:
                    raise ValueError(f'choreography positions must be at least {CHOREOGRAPHY_MIN_SEPARATION_M} m apart')
        return {f'duck{i:02d}': (f'/World/Duck{i:02d}', [x, y, .125]) for i, (x, y) in enumerate(points)}
    floor = 2. if layout == 'grid' else 1.
    if type(spacing) not in (int, float) or not math.isfinite(spacing) or spacing < floor:
        raise ValueError(f'explicit finite separation of at least {floor:g} m required for the {layout} layout')
    if layout == 'line':
        return {f'duck{i:02d}': (f'/World/Duck{i:02d}', [0., float((i - (count - 1) / 2) * spacing), .125])
                for i in range(count)}
    columns = math.ceil(math.sqrt(count))
    return {f'duck{i:02d}': (f'/World/Duck{i:02d}',
                            [float(i % columns * spacing), float(i // columns * spacing), .125])
            for i in range(count)}


class SharedKitNewtonBackend(KitNewtonBackend):
    def __init__(self, args, admission, experience):
        super().__init__(args, admission, experience)
        if not self._reuse_solved_read:
            raise ValueError('shared native reads require explicit bound graph/read-reuse mode')
        self.choreography = self._presenter = None
        choreography = getattr(args, 'choreography', None)
        if choreography is not None:
            # Showcase choreography (opt-in): robots spawn at their formation slots behind the
            # presenter's start pose; the stage dressing and scripted twists come later.
            from .microduck_choreography import Choreography
            if getattr(args, 'layout', 'grid') != 'choreography':
                raise ValueError('a choreography requires --layout choreography')
            self.choreography = Choreography.load(choreography, args.robots)
            spawn = self.choreography.spawn_positions()
            self.placements = placements(args.robots, args.spacing, 'choreography',
                                         getattr(args, 'route_m', 0.) or 0., positions=spawn)
        else:
            self.placements = placements(args.robots, args.spacing, getattr(args, 'layout', 'grid'),
                                         getattr(args, 'route_m', 0.) or 0.)
        self._route_m = float(getattr(args, 'route_m', 0.) or 0.)
        self._shared_read = None
        self._bound_identity = False
        self._support_layout = None

    def _app_config(self):
        if self._sdk_recipe is None:
            raise ValueError('shared foundation requires the explicit internal SDK recipe')
        # Internal SimulationApp supports CUDA-index renderer selection, honoring
        # CUDA_VISIBLE_DEVICES rather than selecting physical Vulkan GPU zero.
        return super()._app_config() | {'active_cuda_gpus': [int(self.args.device.split(':')[1])]}

    def _solver_capacity(self):
        return tuple(value * len(self.placements) for value in super()._solver_capacity())

    def _author_robots(self, stage):
        from pxr import Gf, UsdGeom
        removed, hulls = [], []
        for path, position in self.placements.values():
            root = UsdGeom.Xform.Define(stage, path)
            root.GetPrim().GetReferences().AddReference(self.admission['asset'])
            root.AddTranslateOp().Set(Gf.Vec3d(position[0], position[1], 0.))
            removed.append(neutralize_asset_actuation(stage, self.asset_kind, root_path=path))
            hulls.append(author_collision_hull_limits(stage, root_path=path))
        # One record per robot; the converted bundle keeps its flat prim list.
        if self.asset_kind == 'converted-mjcf':
            removed = [record for records in removed for record in records]
        self.receipt['disabled_source_actuators'] = removed
        self.receipt['collision_hull_limits'] = hulls
        return tuple(path for path, _ in self.placements.values())

    def _camera_pose(self):
        if self.choreography is not None and self.choreography.camera is not None:
            return list(self.choreography.camera['eye']), list(self.choreography.camera['target'])
        points = np.array([p for _, p in self.placements.values()])
        if self._route_m > 0.:
            # Frame the start positions plus the forward route (+x), from behind, above and
            # to the side, so every lane and its whole length stay in view.
            low, high = points.min(axis=0), points.max(axis=0)
            high = high + [self._route_m, 0., 0.]
            center = (low + high) / 2
            extent = max(float((high - low)[:2].max()), .4)
            return (center + [-extent*.55, -extent*.65, extent*.6]).tolist(), center.tolist()
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        extent = max(float(np.ptp(points[:, :2], axis=0).max()), .4)
        return (center + [extent*.8, extent*.8, extent*.85]).tolist(), center.tolist()

    def dress_showcase(self):
        """Visual-only set dressing after the Newton model exists: stage, presenter proxy, lights.

        Nothing physical changes: the model was built from the exported runtime scene, the
        new prims carry no physics API (checked), the ground collider stays and only its
        display is hidden under the stage floor. Receipted with the layers' digests.
        """
        from pxr import Usd, UsdGeom, UsdLux
        from .microduck_choreography import PresenterProxy
        from .microduck_newton import sha256
        import omni.usd
        choreo = self.choreography
        if choreo is None:
            raise ValueError('no choreography to dress')
        if self._presenter is not None:
            raise ValueError('showcase already dressed')
        ns = self.ns
        model_id, before = id(ns.model), (ns.simulation_step_count, float(ns.sim_time))
        shape_count, body_count = int(ns.model.shape_count), int(ns.model.body_count)
        stage = omni.usd.get_context().get_stage()
        layers = {}
        for key, path, root in (('stage_asset', choreo.stage_asset, '/World/Stage'),
                                ('presenter_asset', choreo.presenter_asset, '/World/Presenter')):
            if path is None:
                continue
            path = str(Path(path).resolve())
            if stage.GetPrimAtPath(root).IsValid():
                raise ValueError(f'{root} already exists')
            prim = stage.DefinePrim(root, 'Xform')
            prim.GetReferences().AddReference(path)
            for p in Usd.PrimRange(prim):
                if any(api.startswith('Physics') for api in p.GetAppliedSchemas()):
                    raise ValueError(f'showcase asset must carry no physics API: {p.GetPath()}')
            layers[key] = {'path': path, 'sha256': sha256(Path(path)), 'root': root,
                           'prims': sum(1 for _ in Usd.PrimRange(prim))}
        if choreo.presenter_asset is not None:
            self._presenter = PresenterProxy(stage, str(Path(choreo.presenter_asset).resolve()), root='/World/Presenter',
                                             referenced=True)
            start = choreo.presenter.pose_at(0.)
            self._presenter.write(start, choreo.walk.pose(0., False))
        lights = {}
        if choreo.camera is not None and 'focal_length_mm' in choreo.camera:
            optics = UsdGeom.Camera(stage.GetPrimAtPath(OVERVIEW_CAMERA))
            lights['focal_length_mm'] = {'before': float(optics.GetFocalLengthAttr().Get()),
                                         'after': float(choreo.camera['focal_length_mm'])}
            optics.GetFocalLengthAttr().Set(float(choreo.camera['focal_length_mm']))
        if choreo.stage_asset is not None:
            # The keynote stage brings its own lights; the retained bright dome/sun would wash it out.
            ground = UsdGeom.Imageable(stage.GetPrimAtPath('/World/Ground'))
            ground.MakeInvisible()
            for path, intensity in (('/World/DomeLight', choreo.lighting['dome_intensity']),
                                    ('/World/Sun', choreo.lighting['sun_intensity'])):
                light = UsdLux.LightAPI(stage.GetPrimAtPath(path))
                lights[path] = {'before': float(light.GetIntensityAttr().Get()), 'after': float(intensity)}
                light.GetIntensityAttr().Set(float(intensity))
        if id(ns.model) != model_id or (ns.simulation_step_count, float(ns.sim_time)) != before:
            raise RuntimeError('showcase dressing touched the native model or clock')
        if (int(ns.model.shape_count), int(ns.model.body_count)) != (shape_count, body_count):
            raise RuntimeError('showcase dressing changed the native shape/body count')
        showcase_layer = Path(self.args.out) / 'showcase-scene.usda'
        stage.GetRootLayer().Export(str(showcase_layer))
        self.receipt['showcase'] = {'layers': layers, 'ground_display': 'hidden' if choreo.stage_asset else 'retained',
                                    'lights': lights, 'showcase_scene_sha256': sha256(showcase_layer),
                                    'native_model_unchanged': True, 'choreography': choreo.receipt()}
        return self.receipt['showcase']

    def _write_camera(self, sim_time):
        """Follow-camera pose for this capture (visual only; the overview's intrinsics are unchanged)."""
        from pxr import Gf, UsdGeom
        import omni.usd
        last = getattr(self, '_camera_last_t', None)
        pose = self.choreography.camera_pose_at(sim_time, None if last is None else sim_time - last)
        self._camera_last_t = sim_time
        if pose is None:
            return None
        eye, target = (Gf.Vec3d(*v) for v in pose)
        quat = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0., 0., 1.)).GetInverse().ExtractRotationQuat()
        stage = omni.usd.get_context().get_stage()
        ops = {op.GetOpName(): op for op in UsdGeom.Xformable(stage.GetPrimAtPath(OVERVIEW_CAMERA)).GetOrderedXformOps()}
        ops['xformOp:translate'].Set(eye)
        ops['xformOp:orient'].Set(Gf.Quatd(quat.GetReal(), quat.GetImaginary()))
        return {'eye': list(pose[0]), 'target': list(pose[1])}

    def capture(self):
        if self._presenter is not None:
            # Presenter pose for the frame about to be rendered: a function of the completed sim clock.
            sim_time = float(self.physics_clock[1])
            pose = self.choreography.presenter.pose_at(sim_time)
            presenter = self._presenter.write(pose, self.choreography.walk.pose(pose[3], pose[4]))
            camera = self._write_camera(sim_time)
            result = super().capture()
            result['presenter'] = {**presenter, 'travelled_m': pose[3], 'walking': pose[4], 'sim_time_s': sim_time}
            if camera is not None:
                result['camera_pose'] = camera
            return result
        return super().capture()

    def _layout_for(self, digest):
        import newton
        from .microduck_contact_support import foot_shapes_for
        m = self.ns.model
        return bind_scene(robots={k:v[0] for k,v in self.placements.items()},
            foot_shapes=foot_shapes_for(self.asset_kind),
            scene_model_sha256=digest, joint_labels=m.joint_label,
            joint_q_start=m.joint_q_start.numpy(), joint_qd_start=m.joint_qd_start.numpy(),
            joint_types=m.joint_type.numpy(), joint_parent=m.joint_parent.numpy(),
            joint_child=m.joint_child.numpy(), body_labels=m.body_label,
            shape_labels=m.shape_label, shape_body=m.shape_body.numpy(),
            free_type=int(newton.JointType.FREE), hinge_type=int(newton.JointType.REVOLUTE),
            ground_shapes=('/World/Ground',), world_count=m.world_count)

    def _bind_native_model(self, ns):
        import newton
        from cascade.control.microduck_policy import HOME_Q
        from cascade.control.newton_bam import NewtonBamAdapter
        from .microduck_contact_support import extraction_provenance
        self._model = ns.model
        self._layout = (tuple(ns.model.joint_label), tuple(ns.model.body_label), tuple(ns.model.shape_label))
        self.layout = self._layout_for('0'*64)
        self.receipt['native_labels'] = dict(zip(('joints', 'bodies', 'shapes'), self._layout))
        self.receipt['native_body_properties'] = read_native_body_properties(
            ns, root_path=self.layout.robots[0].root_path)
        self.receipt['support_extraction'] = extraction_provenance(sdk_recipe=self._sdk_recipe)
        self.actuators, properties = [], []
        # Finish all model-property edits before any adapter freezes its channels.
        for binding in self.layout.robots:
            properties.append(prepare_native_model(ns, np.array(binding.dof_indices),
                source_cap=.96, newton=newton, effort_cap=self.admission['bam_params']['joint_effort_limit'],
                dof_count=self.layout.dof_count, asset_kind=self.asset_kind))
        for binding in self.layout.robots:
            self._checkpoint()
            self.actuators.append(NewtonBamAdapter(ns, source_root=self.args.bam_source_root,
                q_indices=binding.q_indices, dof_indices=binding.dof_indices,
                params=self.admission['bam_params'], sdk_recipe=self._sdk_recipe))
        q0 = ns.model.joint_q.numpy().copy()
        for binding in self.layout.robots:
            q0[list(binding.q_indices)] = HOME_Q
            q0[list(binding.free_q_indices)] = self.placements[binding.robot_id][1] + [0., 0., 0., 1.]
        for state in (ns.state_0, ns.state_1):
            self._checkpoint()
            state.joint_q.assign(q0)
            state.joint_qd.zero_()
            newton.eval_fk(ns.model, state.joint_q, state.joint_qd, state)
            state.clear_forces()
        for actuator in self.actuators:
            actuator.reset()
        self.receipt['initialization'] = {'placements': self.placements, 'home_q': HOME_Q.tolist(),
                                          'pose_writes_in_episode': False}
        self.receipt['shared_actuators'] = [a.telemetry() for a in self.actuators]
        self.receipt['shared_model_properties'] = properties
        # The existing content-identity builder validates the common native and
        # asset recipe. bind_identity adds every independently owned BAM group.
        self.receipt['bam'] = self.receipt['shared_actuators'][0]
        self.receipt['native_model_properties'] = properties[0]
        contract = self.layout.robots[0].support_contract()
        contract.pop('model_identity_sha256')
        self.receipt['support_contract'] = contract

    def bind_identity(self, *, repo, runtime_scene):
        from .mobile_identity import build_model_identity, canonical_bytes
        if self._bound_identity:
            raise RuntimeError('shared model identity already bound')
        recipe = build_model_identity(self.admission, self.receipt,
                                      repo=repo, runtime_scene=runtime_scene)['recipe']
        recipe.update(schema='cascade.microduck.shared-scene.v1', shared={
            'placements': self.placements,
            'actuators': [{key: member[key] for key in recipe['bam']}
                          for member in self.receipt['shared_actuators']],
            'native_model_properties': self.receipt['shared_model_properties']})
        digest = hashlib.sha256(canonical_bytes(recipe)).hexdigest()
        self.layout = self._layout_for(digest)
        self._support_layout = self.layout
        self._bound_identity = True
        self.verify_integrator_identity()
        return {'scene_model_sha256': digest, 'recipe': recipe,
                'robots': {b.robot_id: {'model_identity_sha256': b.model_identity_sha256,
                                       'support_contract': b.support_contract()}
                           for b in self.layout.robots}}

    def _read_completed_scene(self):
        from .microduck_contact_support import read_support
        self._guard()
        if not self._bound_identity:
            raise RuntimeError('bind the shared model identity before observations')
        if self.layout is not self._support_layout:
            raise RuntimeError('shared support layout changed after identity binding')
        clock = self.physics_clock
        source_admitted = self.receipt['support_extraction']['source_admitted']
        if type(source_admitted) is not bool:
            raise ValueError('support source admission must be boolean')
        key = clock, self._last_support_solve, source_admitted
        if self._shared_read is not None and self._shared_read[0] == key:
            return self._shared_read[1]
        support = read_support(self.ns, last_solved_clock=self._last_support_solve,
                              source_admitted=source_admitted)
        support = self.layout.capture_support(support, clock=clock)
        result = _read_native_states(self.ns, robots={
            binding.robot_id: dict(q_indices=np.array(binding.q_indices),
                dof_indices=np.array(binding.dof_indices), root_index=binding.root_body_index)
            for binding in self.layout.robots},
            max_contacts=self.admission['limits']['max_contacts'] * len(self.layout.robots),
            max_constraints=self.admission['limits']['max_constraints'] * len(self.layout.robots),
            q_count=self.layout.q_count, dof_count=self.layout.dof_count)
        for sample in result.values():
            sample.update(support=support, solver_graph=self._solver_graph.telemetry())
        if self.physics_clock != clock:
            raise RuntimeError('physics advanced during shared native state/support read')
        self._shared_read = key, copy.deepcopy(result)
        return self._shared_read[1]

    def read_robots(self):
        result = copy.deepcopy(self._read_completed_scene())
        for sample in result.values():
            _public_contacts(sample)
            if type(sample.get('support')) is _SharedSupport:
                sample['support'] = sample['support'].as_observation_dict()
        return result

    def read_robot(self, robot_id):
        sample = _public_contacts(copy.deepcopy(self._read_completed_scene()[robot_id]))
        if type(sample.get('support')) is _SharedSupport:
            sample['support'] = sample['support'].as_observation_dict()
        return sample

    def read_bound_robot(self, binding, epoch):
        return _public_contacts(self._read_bound_robot(binding, epoch))

    def _read_bound_robot(self, binding, epoch):
        sample = copy.deepcopy(self._read_completed_scene()[binding.robot_id])
        sample['support'] = self.layout.robot_support(sample['support'], binding,
            epoch=epoch, clock=(sample['step'], sample['sim_time']))
        return sample

    def step(self):
        self._shared_read = None
        return super().step()

    def contain(self, reason):
        self._shared_read = None
        return super().contain(reason)


class SharedRobotView:
    """One identity/epoch over an immutable completed global sample; cannot step."""
    def __init__(self, owner, binding, epoch):
        self.owner, self.binding, self.epoch = owner, binding, epoch

    @property
    def dt(self):
        return self.owner.dt

    @property
    def physics_clock(self):
        return self.owner.physics_clock

    def read(self):
        return self._read(private=False)

    def _read_for_stepper(self):
        return self._read(private=True)

    def _read(self, *, private):
        bound = getattr(self.owner, 'read_bound_robot', None)
        if private:
            bound = getattr(self.owner, '_read_bound_robot', bound)
        sample = (bound(self.binding, self.epoch) if bound is not None
                  else self.owner.read_robot(self.binding.robot_id))
        sample.update(robot_id=self.binding.robot_id, epoch=self.epoch,
                      model_identity_sha256=self.binding.model_identity_sha256)
        if bound is None:
            sample['support'] = self.owner.layout.robot_support(sample['support'], self.binding,
                epoch=self.epoch, clock=(sample['step'], sample['sim_time']))
        return sample

    def contain(self, reason):
        self.owner.contain(reason)
