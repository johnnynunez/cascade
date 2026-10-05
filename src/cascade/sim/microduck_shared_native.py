"""Bounded native shared-scene foundation; no network command admission.

Each robot has its own policy, battery group, coordinates and model identity.
The single Kit owner alone solves and captures; initial placement is the only
pose write. This diagnostic does not establish walking or fleet task success.
"""
from __future__ import annotations

import copy
import hashlib
import math

import numpy as np

from cascade.control.mobile_telemetry import _public_contacts

from .microduck_newton import (KitNewtonBackend, neutralize_asset_actuation,
                              prepare_native_model, read_native_body_properties,
                              _read_native_states)
from .microduck_shared import _SharedSupport, bind_scene


def placements(count, spacing):
    if type(count) is not int or not 1 <= count <= 12:
        raise ValueError('robot count must be in 1..12')
    if type(spacing) not in (int, float) or not math.isfinite(spacing) or spacing < 2.:
        raise ValueError('explicit finite separation of at least 2 m required')
    columns = math.ceil(math.sqrt(count))
    return {f'duck{i:02d}': (f'/World/Duck{i:02d}',
                            [float(i % columns * spacing), float(i // columns * spacing), .125])
            for i in range(count)}


class SharedKitNewtonBackend(KitNewtonBackend):
    def __init__(self, args, admission, experience):
        super().__init__(args, admission, experience)
        if not self._reuse_solved_read:
            raise ValueError('shared native reads require explicit bound graph/read-reuse mode')
        self.placements = placements(args.robots, args.spacing)
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
        removed = []
        for path, position in self.placements.values():
            root = UsdGeom.Xform.Define(stage, path)
            root.GetPrim().GetReferences().AddReference(self.admission['asset'])
            root.AddTranslateOp().Set(Gf.Vec3d(position[0], position[1], 0.))
            removed.append(neutralize_asset_actuation(stage, self.asset_kind, root_path=path))
        # One record per robot; the converted bundle keeps its flat prim list.
        if self.asset_kind == 'converted-mjcf':
            removed = [record for records in removed for record in records]
        self.receipt['disabled_source_actuators'] = removed
        return tuple(path for path, _ in self.placements.values())

    def _camera_pose(self):
        points = np.array([p for _, p in self.placements.values()])
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        extent = max(float(np.ptp(points[:, :2], axis=0).max()), .4)
        return (center + [extent*.8, extent*.8, extent*.85]).tolist(), center.tolist()

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
