"""Optional MuJoCo C destination regions and passive batch-end evidence.

The arm is the sole physics writer. Contact forces describe the last solved
interval; detached kinematics describes its final qpos. No observer step or
forward on live data is performed, and no all-substep coverage is claimed.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np

from ..config import Cfg


def mapping(value):
    return value.as_dict() if isinstance(value, Cfg) else value


@dataclass(frozen=True)
class Region:
    name: str
    bounds_xy_m: tuple
    version: int = 1
    planning_margin_m: float = 0.

    @classmethod
    def parse(cls, value):
        value = mapping(value)
        base = {'version', 'name', 'frame', 'bounds_xy_m'}
        if not isinstance(value, dict):
            raise ValueError('placement region requires its explicit version/name/base-frame contract')
        version = value.get('version')
        keys = base if version == 1 else base | {'planning_margin_m', 'capacity_scope'}
        if (set(value) != keys or type(version) is not int or version not in (1, 2)
                or value['frame'] != 'robot_base' or not isinstance(value['name'], str)
                or not value['name'].strip()):
            raise ValueError('placement region requires its explicit version/name/base-frame contract')
        bounds = np.asarray(value['bounds_xy_m'], float)
        if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or (bounds[1] <= bounds[0]).any():
            raise ValueError('placement region requires finite increasing XY bounds')
        margin = 0.
        if version == 2:
            margin = value['planning_margin_m']
            if (type(margin) not in (int, float) or not np.isfinite(margin) or margin <= 0
                    or value['capacity_scope'] != 'all_free_bodies'
                    or (bounds[1]-bounds[0] <= 2*margin).any()):
                raise ValueError('region2 requires positive interior margin and complete free-body capacity')
        return cls(value['name'], tuple(tuple(float(x) for x in row) for row in bounds), version, float(margin))

    def as_dict(self):
        value = {'version': self.version, 'name': self.name, 'frame': 'robot_base',
                 'bounds_xy_m': [list(row) for row in self.bounds_xy_m]}
        if self.version == 2:
            value.update(planning_margin_m=self.planning_margin_m, capacity_scope='all_free_bodies')
        return value

    def contains(self, lower, upper):
        bounds = np.asarray(self.bounds_xy_m)
        return bool((np.asarray(lower)[:2] >= bounds[0]).all()
                    and (np.asarray(upper)[:2] <= bounds[1]).all())

    def candidates(self, offsets, separation):
        """Deterministic edge packing, not label-specific slots.

        The area boundary is semantic on a plane: erode by the observed
        footprint alone. Separation is between objects, not a border margin.
        Remaining gaps are tested with measured geometry and full-path checks.
        """
        if self.version != 1:
            raise ValueError('region2 requires measured whole-inventory capacity planning')
        offsets, bounds = np.asarray(offsets, float), np.asarray(self.bounds_xy_m)
        if (offsets.shape != (2, 2) or not np.isfinite(offsets).all()
                or (offsets[1] <= offsets[0]).any()
                or not np.isfinite(separation) or separation <= 0):
            raise ValueError('invalid measured placement footprint or separation')
        low, high = bounds[0]-offsets[0], bounds[1]-offsets[1]
        if (low > high).any():
            return []
        # Edge plus size-derived interior placements; the opposite edge is
        # also evaluated. Exact endpoint deduplication, no random search.
        axes = []
        for axis in range(2):
            step = float(offsets[1, axis]-offsets[0, axis]+separation)
            count = int(np.floor((high[axis]-low[axis])/step))
            if count > 64:
                raise ValueError('placement region exceeds bounded candidate capacity')
            values = [float(low[axis]+i*step) for i in range(count+1)] + [float(high[axis])]
            axes.append(list(dict.fromkeys(values)))
        if len(axes[0])*len(axes[1]) > 64:
            raise ValueError('placement region exceeds bounded candidate capacity')
        return [[x, y] for y in axes[1] for x in axes[0]]


def model_digest(model, descriptor):
    # MuJoCo's own complete compiled-model serialization, not a selective
    # list that could miss a changed actuator/contact/inertia parameter.
    import mujoco
    buffer = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    digest = hashlib.sha256(json.dumps({'descriptor': descriptor, 'sdk': mujoco.__version__},
                                      sort_keys=True, allow_nan=False).encode())
    digest.update(buffer.tobytes())
    return digest.hexdigest()


def state_digest(data):
    digest = hashlib.sha256(np.asarray([data.time]).tobytes())
    for name in ('qpos', 'qvel', 'ctrl'):
        digest.update(np.asarray(getattr(data, name)).tobytes())
    return digest.hexdigest()


class PlacementHistory:
    """One model/epoch, finite passive records; a channel error stays unknown."""

    def __init__(self, world, arm):
        self.world, self.model, self.mj, self.arm = world, world.model, world.mj, arm
        self.data = world.data
        self.region = Region.parse(arm._cfg.get('mj_delivery_area'))
        plane = mapping(arm._cfg.get('mj_release_support_plane'))
        self.descriptor = {'region': self.region.as_dict(), 'support': deepcopy(plane),
                           'root': arm._cfg.get('mj_release_articulation_root'),
                           'joints': list(arm._joint_names), 'gripper': arm._grip_joint,
                           'open_pos': arm._grip_open, 'closed_pos': arm._grip_closed,
                           'qadr': list(arm._qadr), 'dadr': list(arm._dadr), 'grip_qadr': arm._grip_qadr}
        if not isinstance(plane, dict) or not isinstance(self.descriptor['root'], str):
            raise ValueError('placement history requires explicit root/support binding')
        model, mj = self.model, self.mj
        self.root = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, self.descriptor['root'])
        if (model.nmocap or model.npair or self.root < 1 or model.body_parentid[self.root] != 0
                or model.body_jntnum[self.root] != 0 or np.any(model.body_pos[self.root])
                or not np.array_equal(model.body_quat[self.root], [1., 0., 0., 0.])):
            raise ValueError('placement history requires a fixed root at the explicit base frame')
        required, optional = plane.get('geoms'), plane.get('optional_geoms', [])
        if (not isinstance(required, list) or not required or not isinstance(optional, list)
                or any(not isinstance(n, str) for n in required+optional)
                or len(set(required+optional)) != len(required+optional)):
            raise ValueError('placement support names missing or duplicated')
        self.support = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, n) for n in required]
        self.support += [g for n in optional
                         if (g := mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, n)) >= 0]
        pos, quat = np.asarray(plane.get('position_m'), float), np.asarray(plane.get('quaternion_wxyz'), float)
        if (pos.shape != (3,) or quat.shape != (4,) or not np.isfinite(pos).all()
                or not np.array_equal(quat, [1., 0., 0., 0.])):
            raise ValueError('placement support must declare a horizontal plane')
        for geom in self.support:
            if (geom < 0 or model.geom_type[geom] != mj.mjtGeom.mjGEOM_PLANE
                    or model.geom_bodyid[geom] != 0
                    or not np.array_equal(model.geom_pos[geom], pos)
                    or not np.array_equal(model.geom_quat[geom], quat)
                    or not (model.geom_contype[geom] or model.geom_conaffinity[geom])):
                raise ValueError('placement support differs from its model binding')
        self.support_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) for g in self.support]
        names = self.descriptor['joints']+[self.descriptor['gripper']]
        joints = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n) for n in names]
        actual = {j for j in range(model.njnt) if self.descendant(int(model.jnt_bodyid[j]), self.root)}
        if (set(joints) != actual or any(j < 0 or model.jnt_type[j] != mj.mjtJoint.mjJNT_HINGE for j in joints)
                or list(model.jnt_qposadr[joints[:-1]]) != list(arm._qadr)
                or int(model.jnt_qposadr[joints[-1]]) != arm._grip_qadr
                or not np.isfinite([arm._grip_open, arm._grip_closed]).all()
                or arm._grip_open == arm._grip_closed):
            raise ValueError('placement articulation/gripper coordinate binding differs')
        self.bodies = {mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[j])):
                       (int(model.jnt_bodyid[j]), int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j]))
                       for j in range(model.njnt) if model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE}
        self.objects = {name: [] for name in self.bodies}
        self.robot = set()
        for geom in range(model.ngeom):
            if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            body = int(model.geom_bodyid[geom])
            if self.descendant(body, self.root):
                self.robot.add(geom)
            elif geom in self.support:
                continue
            else:
                matches = [name for name, (bid, _, _) in self.bodies.items() if self.descendant(body, bid)]
                if len(matches) != 1:
                    raise ValueError('placement history has an unmodeled collider')
                self.objects[matches[0]].append(geom)
        if (not self.robot or any(not v for v in self.objects.values())
                or model.geom_aabb.shape != (model.ngeom, 6)
                or not np.isfinite(model.geom_aabb).all() or (model.geom_aabb[:, 3:] < 0).any()):
            raise ValueError('placement collision bounds unavailable')
        if model.opt.integrator not in (mj.mjtIntegrator.mjINT_EULER,
                                        mj.mjtIntegrator.mjINT_IMPLICIT, mj.mjtIntegrator.mjINT_IMPLICITFAST):
            raise ValueError('placement history does not admit a multistage integrator')
        self.identity = model_digest(model, self.descriptor)
        self.scratch = mj.MjData(model)
        self.rows = deque(maxlen=256)
        self.reset()

    def descendant(self, child, parent):
        while child:
            if child == parent:
                return True
            child = int(self.model.body_parentid[child])
        return False

    def reset(self):
        self.epoch = uuid.uuid4().hex
        self.rows.clear()
        self.closed_seen = False
        self.open_since = None
        self.error = None
        self.last_final_time = float(self.world.data.time)
        self.step = 0
        self.confirmed_prefix = set()
        self.prefix_faults = {}
        self.retired_obligations = []

    def guard(self):
        if (self.world.model is not self.model or self.world.data is not self.data
                or model_digest(self.model, self.descriptor) != self.identity):
            raise ValueError('placement native model identity changed')
        if (Region.parse(self.arm._cfg.get('mj_delivery_area')) != self.region
                or mapping(self.arm._cfg.get('mj_release_support_plane')) != self.descriptor['support']
                or self.arm._cfg.get('mj_release_articulation_root') != self.descriptor['root']
                or list(self.arm._qadr) != self.descriptor['qadr']
                or list(self.arm._dadr) != self.descriptor['dadr']
                or self.arm._grip_qadr != self.descriptor['grip_qadr']
                or list(self.arm._joint_names) != self.descriptor['joints']
                or self.arm._grip_joint != self.descriptor['gripper']
                or self.arm._grip_open != self.descriptor['open_pos']
                or self.arm._grip_closed != self.descriptor['closed_pos']):
            raise ValueError('placement profile binding changed')
        if self.error:
            raise ValueError(self.error)

    def geometry(self, qpos=None):
        """Detached final-state FK, never a dynamics update of live data."""
        with self.world.lock:
            self.guard()
            return self._validated_geometry(qpos)

    def _validated_geometry(self, qpos=None):
        """Scratch FK within this admission's validated world-lock scope.

        No model validation is reused across calls. The capture only invokes
        this local helper between its complete fingerprint and saved output;
        it has no external callback or model writer in that interval.
        """
        self.scratch.qpos[:] = self.world.data.qpos if qpos is None else qpos
        if not np.isfinite(self.scratch.qpos).all():
            raise ValueError('placement coordinates are non-finite')
        if any(abs(np.linalg.norm(self.scratch.qpos[qa+3:qa+7])-1.) > 1e-6
               for _, qa, _ in self.bodies.values()):
            raise ValueError('placement body quaternion is not unit length')
        self.mj.mj_kinematics(self.model, self.scratch)
        rotation = self.scratch.geom_xmat.reshape(-1, 3, 3)
        center = self.scratch.geom_xpos + np.einsum('gij,gj->gi', rotation, self.model.geom_aabb[:, :3])
        half = np.einsum('gij,gj->gi', np.abs(rotation), self.model.geom_aabb[:, 3:])
        return center-half, center+half

    def capture(self, steps, before):
        """Called only by the existing arm step writer, once per nonempty batch."""
        with self.world.lock:
            self._capture_locked(steps, before)

    def _capture_locked(self, steps, before):
        try:
            self.guard()
            d, mj, m = self.world.data, self.mj, self.model
            clock = float(d.time)
            if (type(steps) is not int or steps <= 0 or not np.isfinite(clock)
                    or abs(clock-self.last_final_time-steps*float(m.opt.timestep)) > 1e-9):
                raise ValueError('placement batch clock did not advance')
            if (not np.isfinite(d.qvel).all() or not np.isfinite(d.ctrl).all()
                    or any(d.warning[getattr(mj.mjtWarning, 'mjWARN_'+name)].number
                           for name in ('CONTACTFULL', 'CNSTRFULL', 'BADQPOS', 'BADQVEL', 'BADQACC', 'BADCTRL'))):
                raise ValueError('placement native contact/state channel unavailable')
            if (not isinstance(before, dict) or before['qpos'].shape != (m.nq,)
                    or before['qvel'].shape != (m.nv,)
                    or not np.isfinite(before['qpos']).all() or not np.isfinite(before['qvel']).all()
                    or abs(clock-before['time_s']-float(m.opt.timestep)) > 1e-9):
                raise ValueError('placement last-solve input binding unavailable')
            self.step += steps
            final_lower, final_upper = self._validated_geometry()
            final_footprints = {name: {'lower_m': final_lower[g].min(axis=0).tolist(),
                                      'upper_m': final_upper[g].max(axis=0).tolist()}
                                for name, g in self.objects.items()}
            for name, (_, _, va) in self.bodies.items():
                final_footprints[name].update(linear_speed_m_s=float(np.linalg.norm(d.qvel[va:va+3])),
                                             angular_speed_rad_s=float(np.linalg.norm(d.qvel[va+3:va+6])))
            lower, upper = self._validated_geometry(before['qpos'])
            if (not np.allclose(self.scratch.geom_xpos, d.geom_xpos, rtol=0, atol=1e-10)
                    or not np.allclose(self.scratch.geom_xmat, d.geom_xmat, rtol=0, atol=1e-10)):
                raise ValueError('placement solved geometry differs from captured input phase')
            grip = (float(before['qpos'][self.arm._grip_qadr])-self.arm._grip_closed)/(self.arm._grip_open-self.arm._grip_closed)
            if not np.isfinite(grip):
                raise ValueError('placement physical gripper coordinate unavailable')
            if grip < .98:
                self.closed_seen, self.open_since = True, None
            elif self.closed_seen and self.open_since is None:
                self.open_since = before['time_s']
            objects = {}
            for name, geoms in self.objects.items():
                body, _, va = self.bodies[name]
                velocity = before['qvel'][va:va+6]
                if velocity.shape != (6,):
                    raise ValueError('placement free-body velocity unavailable')
                objects[name] = {'position_m': self.scratch.xpos[body].tolist(),
                    'lower_m': lower[geoms].min(axis=0).tolist(), 'upper_m': upper[geoms].max(axis=0).tolist(),
                    'linear_speed_m_s': float(np.linalg.norm(velocity[:3])),
                    'angular_speed_rad_s': float(np.linalg.norm(velocity[3:])),
                    'support_up_n': 0., 'support_contacts': 0, 'other_loaded_contacts': 0, 'arm_active_contacts': 0}
            # Iterate each actual contact once, even with two coincident named
            # planes. Nothing is duplicated per support alias.
            contacts = []
            for i in range(int(d.ncon)):
                contact = d.contact[i]
                if int(contact.efc_address) < 0:
                    continue
                force = np.zeros(6)
                mj.mj_contactForce(m, d, i, force)
                frame = np.asarray(contact.frame).reshape(3, 3)
                if (not np.isfinite(force).all() or not np.isfinite(frame).all()
                        or not np.isfinite(contact.pos).all() or force[0] < 0
                        or not np.allclose(frame @ frame.T, np.eye(3), rtol=0, atol=1e-8)):
                    raise ValueError('placement solved contact force unavailable')
                a, b = int(contact.geom1), int(contact.geom2)
                if not 0 <= a < m.ngeom or not 0 <= b < m.ngeom:
                    raise ValueError('placement contact geometry unavailable')
                contacts.append({'index': i, 'geom': [a, b], 'efc_address': int(contact.efc_address),
                    'position_m': np.asarray(contact.pos).tolist(), 'frame_a_to_b': frame.tolist(),
                    'force_torque_contact': force.tolist()})
                world_force = frame.T @ force[:3]  # wrench on geom2
                for name, geoms in self.objects.items():
                    if (a in geoms) == (b in geoms):
                        continue
                    other = b if a in geoms else a
                    sign = -1 if a in geoms else 1
                    if other in self.robot:
                        objects[name]['arm_active_contacts'] += 1
                    if other in self.support:
                        if force[0] > 0 and sign*frame[0, 2] > 0 and sign*world_force[2] > 0:
                            objects[name]['support_up_n'] += float(sign*world_force[2])
                            objects[name]['support_contacts'] += 1
                    elif force[0] > 0:
                        objects[name]['other_loaded_contacts'] += 1
            for name in self.confirmed_prefix:
                value = objects[name]
                if (not self.region.contains(value['lower_m'], value['upper_m'])
                        or value['support_up_n'] <= 0 or value['other_loaded_contacts'] or value['arm_active_contacts']
                        or value['linear_speed_m_s'] > .02 or value['angular_speed_rad_s'] > .2):
                    self.prefix_faults.setdefault(name, {'step': self.step, 'solve_time_s': before['time_s'],
                                                         'measurement': deepcopy(value)})
            self.rows.append({'epoch': self.epoch, 'model_sha256': self.identity,
                'step': self.step, 'time_s': before['time_s'], 'final_state_time_s': clock,
                'last_force_interval_s': [before['time_s'], clock], 'final_footprints': final_footprints,
                'batch_steps': steps, 'gripper_open_fraction': grip, 'open_since_s': self.open_since,
                'solve_qpos': before['qpos'].tolist(), 'solve_qvel': before['qvel'].tolist(),
                'final_qpos': d.qpos.tolist(), 'final_qvel': d.qvel.tolist(), 'ctrl': d.ctrl.tolist(),
                'native_contact_count': int(d.ncon), 'solved_contacts': contacts,
                'final_gripper_open_fraction': (float(d.qpos[self.arm._grip_qadr])-self.arm._grip_closed)/(self.arm._grip_open-self.arm._grip_closed),
                'state_sha256': state_digest(d), 'objects': objects})
            self.last_final_time = clock
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'

    def retire_explicit(self, context, result, *, harness):
        """Replace an old area goal only after a separately requested point succeeds.

        This changes the goal ledger, never motion authorization or the point
        verifier. Previous disturbances remain in the explicit retirement row.
        """
        from ..agent.effects import SAME_PLACE_M
        with self.world.lock:
            self.guard()
            pc = result.get('postcondition') or {}
            if (context['epoch'] != self.epoch or context['model_sha256'] != self.identity
                    or result.get('ok') is not True or pc.get('status') != 'confirmed'
                    or pc.get('channel') != 'physics'):
                return {'retired': False, 'reason': 'explicit relocation not independently confirmed in this epoch'}
            name = context['object']
            if len(self.retired_obligations) >= 256:
                return {'retired': False, 'reason': 'bounded explicit-obligation journal is full'}
            if name not in self.confirmed_prefix:
                return {'retired': False, 'reason': 'no active prior area obligation'}
            if (not self.rows or self.rows[-1]['state_sha256'] != state_digest(self.world.data)
                    or self.rows[-1]['final_state_time_s'] != float(self.world.data.time)):
                return {'retired': False, 'reason': 'explicit relocation has no current bound history'}
            post_request = [row for row in self.rows if row['time_s'] >= context['request_final_time_s']]
            if (not post_request or post_request[-1]['open_since_s'] is None
                    or post_request[-1]['open_since_s'] < context['request_final_time_s']):
                return {'retired': False, 'reason': 'explicit request lacks its own measured release'}
            checked = audit(post_request, name, self.region, self.identity, self.epoch, containment=False)
            current = np.asarray(self.world.data.qpos[self.bodies[name][1]:self.bodies[name][1]+2])
            if checked['status'] != 'confirmed' or np.linalg.norm(current-context['target_xy_m']) > SAME_PLACE_M*2:
                return {'retired': False, 'reason': 'explicit point lacks current released/stable support evidence',
                        'verification': checked}
            event = {'retired': True, 'scope': 'old region goal only; no new motion authority',
                     'request': deepcopy(context), 'verification': checked,
                     'previous_disturbance': deepcopy(self.prefix_faults.get(name))}
            # Audit, fingerprint, copies and SDK reads have finished. Only
            # these ledger mutations share the short stop-state lock, never
            # the slow observation work. Stop does not take world.lock.
            with harness._observation_lock:
                if (context['generation'] != harness._halt_generation
                        or context['cancellation_token'] != harness._observation_cancel_generation
                        or harness.estopped or harness.halted is not None):
                    return {'retired': False, 'reason': 'explicit region-obligation retirement cancelled'}
                self.retired_obligations.append(event)
                self.confirmed_prefix.remove(name)
                self.prefix_faults.pop(name, None)
            return deepcopy(event)

    def read(self, label):
        from .truth import _match_label
        with self.world.lock:
            self.guard()
            rows = deepcopy(list(self.rows))
            if (not rows or rows[-1]['state_sha256'] != state_digest(self.world.data)
                    or rows[-1]['final_state_time_s'] != float(self.world.data.time)):
                raise ValueError('placement history does not end at the current native state')
            name = _match_label(label, {name: name for name in self.objects})
            if name is None:
                raise ValueError('placement object identity is ambiguous or missing')
            report = audit(rows, name, self.region, self.identity, self.epoch)
            if name in self.prefix_faults:
                report['status'] = 'refuted'
                report['evidence'] = 'previously placed object was disturbed in a sampled intervening solve'
                report['measured']['first_disturbance'] = deepcopy(self.prefix_faults[name])
            prefix = {}
            for previous in sorted(self.confirmed_prefix - {name}):
                checked = audit(rows, previous, self.region, self.identity, self.epoch)
                if previous in self.prefix_faults:
                    checked['status'] = 'refuted'
                    checked['evidence'] = 'previously placed prefix was disturbed in a sampled intervening solve'
                    checked['measured']['first_disturbance'] = deepcopy(self.prefix_faults[previous])
                prefix[previous] = checked
                if checked['status'] != 'confirmed':
                    report['status'] = checked['status']
                    report['evidence'] = 'previously placed object no longer satisfies the region: '+previous
            report['measured']['previously_confirmed'] = prefix
            report['measured']['retired_area_obligations'] = deepcopy(self.retired_obligations)
            if report['status'] == 'confirmed':
                self.confirmed_prefix.add(name)
            return report, rows

    def diagnostic_snapshot(self):
        """Copy only previously produced records; never validate a new verdict.

        This remains usable after a sticky capture/identity failure. No SDK,
        native-state read, guard, FK, audit or ledger mutation belongs here.
        """
        with self.world.lock:
            return deepcopy({
                'version': 1, 'diagnostic_only': True, 'physical_task_verdict': False,
                'scope': 'recorded batch history; current model/state not revalidated',
                'scene_path': self.world.path, 'epoch': self.epoch,
                'model_sha256': self.identity, 'descriptor': self.descriptor,
                'support_planes': self.support_names,
                'last_captured_step': self.step, 'last_captured_final_time_s': self.last_final_time,
                'capture_error': self.error, 'record_capacity': 256,
                'records': list(self.rows)[-256:],
                'confirmed_prefix': sorted(self.confirmed_prefix),
                'prefix_faults': self.prefix_faults,
                'retired_area_obligations': self.retired_obligations[-256:],
            })


def save_placement_diagnostic(arm, evidence_dir):
    """Persist an existing native owner's history without activating a driver."""
    from ..control.lazy_arm import LazyArm
    from ..control.mujoco_arm import MujocoArm
    raw = arm.raw
    backend = raw.__dict__.get('_arm') if isinstance(raw, LazyArm) else raw
    if not isinstance(backend, MujocoArm):
        return None
    world = backend.world
    history = getattr(world, 'placement_history', None)
    if history is None:
        return None
    metadata = {'diagnostic_only': True, 'physical_task_verdict': False}
    try:
        if type(history) is not PlacementHistory or history.arm is not backend or history.world is not world:
            raise ValueError('diagnostic history owner binding differs')
        snapshot = history.diagnostic_snapshot()
        # The world lock has been released: encoding and file I/O must not
        # delay a driver holding or stopping its physical state.
        encoded = json.dumps(snapshot, allow_nan=False)+'\n'
        path = Path(evidence_dir)/f'diagnostic-{uuid.uuid4().hex}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as handle:
            handle.write(encoded)
        metadata.update(evidence_path=str(path), evidence_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
                        records=len(snapshot['records']), epoch=snapshot['epoch'],
                        model_sha256=snapshot['model_sha256'])
    except Exception as exc:
        metadata['error'] = f'placement diagnostic unavailable: {type(exc).__name__}: {exc}'
    return metadata


def audit(rows, name, region, identity, epoch, *, containment=True):
    measured = {'destination': region.name, 'region': region.as_dict(), 'object': name,
                'model_sha256': identity, 'epoch': epoch, 'containment_required': containment, 'sampling': 'last-solve of existing arm batches'}
    def verdict(status, reason):
        return {'status': status, 'evidence': reason, 'measured': measured}
    if not rows:
        return verdict('unverified', 'no passive placement samples')
    last = rows[-1]
    start = len(rows)-1
    while start > 0 and last['time_s']-rows[start]['time_s'] < .6:
        start -= 1
    tail = rows[start:]
    measured.update(samples=len(tail), duration_s=last['time_s']-tail[0]['time_s'])
    if len(tail) < 6 or measured['duration_s'] < .6:
        return verdict('unverified', 'placement settling history is shorter than six samples/0.6 physical seconds')
    clocks, steps = np.asarray([r['time_s'] for r in tail]), np.asarray([r['step'] for r in tail])
    if (not np.isfinite(clocks).all() or (np.diff(clocks) <= 0).any() or (np.diff(clocks) > .25).any()
            or (np.diff(steps) <= 0).any() or any(r['epoch'] != epoch or r['model_sha256'] != identity for r in tail)):
        return verdict('unverified', 'placement settling clock or identity continuity unavailable')
    if any(r['open_since_s'] is None or r['open_since_s'] > tail[0]['time_s'] for r in tail):
        return verdict('unverified', 'placement window lacks measured post-opening history')
    final = last['final_footprints'][name]
    if not np.isfinite([final['linear_speed_m_s'], final['angular_speed_rad_s'], last['final_gripper_open_fraction']]).all():
        return verdict('unverified', 'integrated final release/velocity measurement unavailable')
    if (final['linear_speed_m_s'] > .02 or final['angular_speed_rad_s'] > .2
            or last['final_gripper_open_fraction'] < .98):
        return verdict('refuted', 'integrated current state is moving or jaws are not open')
    if containment and not region.contains(final['lower_m'], final['upper_m']):
        return verdict('refuted', 'current integrated footprint is outside the configured region')
    values = [r['objects'][name] for r in tail]
    if any(not np.isfinite([v['support_up_n'], v['linear_speed_m_s'], v['angular_speed_rad_s'],
                               r['gripper_open_fraction']]).all()
           or type(v['support_contacts']) is not int or type(v['other_loaded_contacts']) is not int
           or type(v['arm_active_contacts']) is not int or v['arm_active_contacts'] < 0
           or v['support_contacts'] < 0 or v['other_loaded_contacts'] < 0
           for v, r in zip(values, tail)):
        return verdict('unverified', 'placement measured contact/gripper channel unavailable')
    if containment and any(not region.contains(v['lower_m'], v['upper_m']) for v in values):
        return verdict('refuted', 'full collision footprint is outside the configured destination region')
    if any(r['gripper_open_fraction'] < .98 for r in tail):
        return verdict('refuted', 'physical jaws are not fully open throughout placement window')
    if any(v['support_contacts'] < 1 or v['support_up_n'] <= 0 or v['other_loaded_contacts'] or v['arm_active_contacts'] for v in values):
        return verdict('refuted', 'placement requires loaded support, no active arm contact and no other loaded contact throughout the sampled window')
    positions = np.asarray([v['position_m'] for v in values])
    spread = float(np.linalg.norm(positions[:, None, :]-positions[None, :, :], axis=2).max())
    linear = max(v['linear_speed_m_s'] for v in values)
    angular = max(v['angular_speed_rad_s'] for v in values)
    measured.update(position_spread_m=spread, max_linear_speed_m_s=linear, max_angular_speed_rad_s=angular,
                    first_step=tail[0]['step'], last_step=last['step'], support_planes='effective declared colliders')
    if not np.isfinite([spread, linear, angular]).all():
        return verdict('unverified', 'placement kinematic measurements unavailable')
    if spread > .005 or linear > .02 or angular > .2:
        return verdict('refuted', 'object did not remain stable in the post-release physical window')
    return verdict('confirmed', ('full footprint, ' if containment else '')
                   + 'actual release, loaded support and sampled physical settling confirmed')


def read_placement(scene_path, cfg, label, destination, *, evidence_dir=None):
    from .mujoco_world import peek
    report = {'status': 'unverified', 'evidence': 'placement region history unavailable',
              'measured': {'destination': destination}}
    rows = []
    try:
        region = Region.parse(cfg.get('mj_delivery_area'))
        world = peek(scene_path)
        history = getattr(world, 'placement_history', None)
        if destination != region.name or history is None or history.region != region:
            raise ValueError('placement world/configured region binding unavailable')
        report, rows = history.read(label)
        report['measured']['support_planes'] = history.support_names
        report['measured']['native_registry'] = {
            'scene_path': history.world.path, 'sdk': history.mj.__version__,
            'support_geom_ids': list(history.support), 'robot_geom_ids': sorted(history.robot),
            'object_geom_ids': history.objects, 'body_qpos_dof': history.bodies,
            'geom_names': [history.mj.mj_id2name(history.model, history.mj.mjtObj.mjOBJ_GEOM, g)
                           for g in range(history.model.ngeom)], 'descriptor': history.descriptor}
    except Exception as exc:
        report['evidence'] = f'placement observation unavailable: {type(exc).__name__}: {exc}'
    if evidence_dir is not None:
        path = Path(evidence_dir)/f'{uuid.uuid4().hex}.json'
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = json.dumps({'version': 1, 'records': rows, 'verdict': report}, allow_nan=False)+'\n'
            path.write_text(raw)
            report['measured'].update(evidence_path=str(path), evidence_sha256=hashlib.sha256(raw.encode()).hexdigest())
        except (OSError, ValueError, TypeError) as exc:
            report['status'] = 'unverified'
            report['evidence'] = f'placement evidence persistence failed: {type(exc).__name__}'
    return report
