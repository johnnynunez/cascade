"""Shared-scene ownership and one-solve scheduling; no SDK imports on import.

These bindings do not create a scene or admit physical motion. A native owner
must supply the measured model, independent robot views and disjoint BAM groups.
The existing single-robot entrypoint remains the default.
"""
from __future__ import annotations

import hashlib
import json
import re
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np

from cascade.control.microduck_policy import POLICY_JOINTS
from cascade.robotics.contracts import identifier

from .microduck_contact_support import FOOT_SHAPES
from .microduck_state import newton_joint_indices


def _labels(values, name):
    values = tuple(values)
    if any(type(x) is not str or not x for x in values) or len(set(values)) != len(values):
        raise ValueError(f'invalid or duplicate {name}')
    return values


def _ints(values, size, name):
    values = np.asarray(values)
    if values.shape != (size,) or values.dtype.kind not in 'iu':
        raise ValueError(f'invalid {name} layout')
    return values


@dataclass(frozen=True)
class RobotBinding:
    robot_id: str
    root_path: str
    root_body_index: int
    q_indices: tuple[int, ...]
    dof_indices: tuple[int, ...]
    free_q_indices: tuple[int, ...]
    free_dof_indices: tuple[int, ...]
    body_indices: tuple[int, ...]
    shape_indices: tuple[int, ...]
    robot_shapes: tuple[str, ...]
    foot_shapes: tuple[str, ...]
    ground_shapes: tuple[str, ...]
    scene_model_sha256: str

    @property
    def model_identity_sha256(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(',', ':'),
                                         allow_nan=False).encode()).hexdigest()

    def support_contract(self):
        return {'version': 1, 'model_identity_sha256': self.model_identity_sha256,
                    'robot_shapes': list(self.robot_shapes), 'foot_shapes': list(self.foot_shapes),
                    'ground_shapes': list(self.ground_shapes), 'gravity_world_m_s2': [0., 0., -9.81]}


@dataclass(frozen=True)
class SceneLayout:
    robots: tuple[RobotBinding, ...]
    q_count: int
    dof_count: int
    shape_labels: tuple[str, ...]

    def robot_support(self, observed, binding, *, epoch, clock):
        """Keep the full shared contact set: another robot is never ground.

        Global completeness comes only from the original solved decoder. This
        method binds/copies that result; it cannot turn unavailable into known.
        """
        from cascade.control.mobile_support import SupportObservation
        if binding not in self.robots or (observed['step'], observed['sim_time_s']) != clock:
            raise ValueError('support robot/clock binding mismatch')
        result = deepcopy(observed)
        for key, expected in (('epoch', epoch), ('model_identity_sha256', binding.model_identity_sha256)):
            if key in result and result[key] != expected:
                raise ValueError('support identity cannot be rebound')
            result[key] = expected
        parsed = SupportObservation.from_dict(result)
        for contact in parsed.contacts:
            for sid, name in ((contact.shape_a_id, contact.shape_a), (contact.shape_b_id, contact.shape_b)):
                if sid >= len(self.shape_labels) or self.shape_labels[sid] != name:
                    raise ValueError('support shape differs from shared model')
        return result


def bind_scene(*, robots, scene_model_sha256, joint_labels, joint_q_start, joint_qd_start,
               joint_types, joint_parent, joint_child, body_labels, shape_labels, shape_body,
               free_type, hinge_type, ground_shapes, world_count):
    """Bind one world containing only the explicit robots and static ground.

    Generated free-joint names have no USD namespace. Resolve them through the
    native child-body map, never traversal order or a synthetic prefix. Reject
    cross-robot joints/shapes, extra articulated bodies and unclassified shapes.
    The caller supplies SDK enum values explicitly; no numeric enum is guessed.
    """
    if (type(world_count) is not int or world_count != 1
            or not isinstance(robots, dict) or not 1 <= len(robots) <= 12
            or re.fullmatch('[0-9a-f]{64}', scene_model_sha256 or '') is None
            or type(free_type) is not int or type(hinge_type) is not int or free_type == hinge_type):
        raise ValueError('explicit robot/model/type contract required')
    roots = tuple(robots.values())
    for robot_id in robots:
        identifier(robot_id, 'shared robot identity')
    if any(type(p) is not str or re.fullmatch(r'/[A-Za-z_][A-Za-z0-9_]*(/[A-Za-z_][A-Za-z0-9_]*)+', p) is None for p in roots):
        raise ValueError('explicit absolute robot namespaces required')
    if any(a == b or a.startswith(b + '/') or b.startswith(a + '/')
           for i, a in enumerate(roots) for b in roots[i+1:]):
        raise ValueError('overlapping robot namespaces')
    joints = _labels(joint_labels, 'joint labels')
    bodies = _labels(body_labels, 'body labels')
    shapes = _labels(shape_labels, 'shape labels')
    ground = _labels(ground_shapes, 'ground shapes')
    if not ground or not set(ground).issubset(shapes):
        raise ValueError('explicit ground shapes required')
    n = len(joints)
    qs = _ints(joint_q_start, n+1, 'joint q starts')
    ds = _ints(joint_qd_start, n+1, 'joint dof starts')
    jt = _ints(joint_types, n, 'joint types')
    parent = _ints(joint_parent, n, 'joint parent')
    child = _ints(joint_child, n, 'joint child')
    sb = _ints(shape_body, len(shapes), 'shape body')
    if (qs[0] != 0 or ds[0] != 0 or (np.diff(qs) <= 0).any() or (np.diff(ds) <= 0).any()
            or qs[-1] != 21*len(robots) or ds[-1] != 20*len(robots)
            or (qs > qs[-1]).any() or (ds > ds[-1]).any()
            or (child < 0).any() or (child >= len(bodies)).any()
            or (parent < -1).any() or (parent >= len(bodies)).any()
            or (sb < -1).any() or (sb >= len(bodies)).any()):
        raise ValueError('invalid shared native coordinate/body layout')
    def owner(label):
        return next((i for i, root in enumerate(roots) if label.startswith(root + '/')), -1)
    body_owners = [owner(label) for label in bodies]
    if -1 in body_owners:
        raise ValueError('unclassified native body')
    shape_owners = [owner(label) for label in shapes]
    for i, label in enumerate(shapes):
        if label in ground:
            if sb[i] != -1 or shape_owners[i] != -1:
                raise ValueError('ground cannot be a robot shape or dynamic body')
        elif (shape_owners[i] == -1 or sb[i] == -1
              or body_owners[sb[i]] != shape_owners[i]):
            raise ValueError('unclassified or cross-robot collision shape')
    if len(set(map(int, child))) != len(bodies) or set(map(int, child)) != set(range(len(bodies))):
        raise ValueError('each native body must have exactly one joint')
    joint_by_child = {int(body): i for i, body in enumerate(child)}
    for body in range(len(bodies)):
        visited = set()
        while body != -1:
            if body in visited:
                raise ValueError('cyclic native articulation')
            visited.add(body)
            body = int(parent[joint_by_child[body]])
    bindings = []
    covered = set()
    for robot_index, (robot_id, root) in enumerate(robots.items()):
        root_body = root + '/Geometry/trunk_base'
        if root_body not in bodies:
            raise ValueError('missing native root body')
        root_index = bodies.index(root_body)
        owned = [i for i in range(n) if body_owners[child[i]] == robot_index]
        free = [i for i in owned if jt[i] == free_type]
        hinges = [i for i in owned if jt[i] == hinge_type]
        if (len(owned) != 15 or len(free) != 1 or len(hinges) != 14
                or child[free[0]] != root_index or parent[free[0]] != -1
                or qs[free[0]+1] - qs[free[0]] != 7 or ds[free[0]+1] - ds[free[0]] != 6):
            raise ValueError('each robot requires one free root and fourteen hinges')
        if any(parent[i] == -1 or body_owners[parent[i]] != robot_index
               or owner(joints[i]) != robot_index for i in hinges):
            raise ValueError('cross-robot joint or missing parent')
        selected = {joints[i].rsplit('/', 1)[-1] for i in hinges}
        if selected != set(POLICY_JOINTS):
            raise ValueError('policy joints differ from robot articulation')
        qi, di = newton_joint_indices(joints, qs, ds, root_path=root)
        feet = tuple(p.replace('/World/MicroDuck', root, 1) for p in FOOT_SHAPES)
        if not set(feet).issubset(shapes):
            raise ValueError('missing exact sole shape')
        si = tuple(i for i, value in enumerate(shape_owners) if value == robot_index)
        bindings.append(RobotBinding(robot_id, root, root_index, tuple(map(int, qi)), tuple(map(int, di)),
            tuple(range(int(qs[free[0]]), int(qs[free[0]+1]))),
            tuple(range(int(ds[free[0]]), int(ds[free[0]+1]))),
            tuple(i for i, value in enumerate(body_owners) if value == robot_index), si,
            tuple(shapes[i] for i in si), feet, ground, scene_model_sha256))
        covered.update(owned)
    if covered != set(range(n)):
        raise ValueError('unclassified native joint')
    return SceneLayout(tuple(bindings), int(qs[-1]), int(ds[-1]), shapes)


class SharedMicroduckStepper:
    """One physics owner; independent policy/history/controller per robot.

    All preparations finish before the one solve. A late stop/lease change
    during a peer's preparation vetoes the whole pending solve. After the final
    memory-only fence, that solve is admitted; a concurrent stop cannot abort
    SDK work, and the next slot uses its new intent. Faults contain the entire
    scene, without asserting physical rest. Readers must not step physics.
    """
    def __init__(self, owner, steppers, *, layout):
        self.owner, self.steppers, self.layout = owner, tuple(steppers), layout
        if not 1 <= len(self.steppers) <= 12 or len(self.steppers) != len(layout.robots):
            raise ValueError('one controller/policy/actuator per bound robot required')
        for field in ('controller', 'policy', 'actuator', 'backend'):
            if len({id(getattr(s, field)) for s in self.steppers}) != len(self.steppers):
                raise ValueError(f'robots cannot share {field} state')
        for stepper, binding in zip(self.steppers, layout.robots):
            view = stepper.backend
            hello = stepper.controller.hello()
            if (view.owner is not owner or view.binding != binding or stepper.started or stepper.closed
                    or hello['robot_id'] != binding.robot_id
                    or hello['model_identity_sha256'] != binding.model_identity_sha256
                    or stepper.dt != owner.dt
                    or stepper.actuator.coordinate_indices != (binding.q_indices, binding.dof_indices)):
                raise ValueError('shared robot view/controller/model binding mismatch')
        self._members = tuple((s.backend, s.controller, s.policy, s.actuator) for s in self.steppers)
        self.started = self.closed = False
        self.failure = ''
        self.containment_errors = []

    def _fail(self, exc):
        try:
            self.failure = str(exc) or type(exc).__name__
        except BaseException:  # noqa: BLE001 - diagnostics must not mask containment
            self.failure = type(exc).__name__
        for s, members in zip(self.steppers, self._members):
            s.failure = self.failure
            try:
                members[1].fault('shared MicroDuck containment; lifecycle restart required')
            except BaseException as secondary:  # noqa: BLE001 - try every owner, preserve primary
                self.containment_errors.append(('controller', type(secondary).__name__))
        try:
            self.owner.contain('shared MicroDuck containment; no physical stop verdict')
        except BaseException as secondary:  # noqa: BLE001 - caller must retain the primary failure
            self.containment_errors.append(('owner', type(secondary).__name__))

    def _check_bindings(self):
        for s, binding, members in zip(self.steppers, self.layout.robots, self._members):
            if (any(current is not saved for current, saved in
                    zip((s.backend, s.controller, s.policy, s.actuator), members))
                    or s.backend.owner is not self.owner or s.backend.binding != binding
                    or s.actuator.coordinate_indices != (binding.q_indices, binding.dof_indices)):
                raise RuntimeError('shared native ownership binding changed')

    def _identity(self, stepper, sample):
        for key in ('robot_id', 'epoch', 'model_identity_sha256'):
            if sample.get(key) != stepper.identity[key]:
                raise RuntimeError('shared observation identity/epoch mismatch')

    def start(self):
        if self.started or self.closed or self.failure:
            raise RuntimeError('shared stepper already started/closed/faulted')
        try:
            self._check_bindings()
            initial = self.owner.physics_clock
            for s in self.steppers:
                s.start()
                self._identity(s, s.last)
                if (s.last['step'], s.last['sim_time']) != initial:
                    raise RuntimeError('shared initial observations disagree with owner clock')
            self.started = True
        except BaseException as exc:
            self._fail(exc)
            raise

    def tick(self):
        if not self.started or self.closed or self.failure:
            raise RuntimeError('shared stepper not running; lifecycle restart required')
        try:
            self._check_bindings()
            prepared = []
            for s in self.steppers:
                item = s._prepare_tick(prepare_actuator=False)
                self._identity(s, item[0])
                prepared.append(item)
            # Every policy has completed before any group writes its owned
            # effort/friction. No adapter may clear the global control array.
            for s, item in zip(self.steppers, prepared):
                s._prepare_actuator(item)
            for s in self.steppers:
                s._check_wall()
            # No inference, device writes, reads or callbacks while these locks
            # are held. Stop can always interrupt expensive peer preparation.
            with ExitStack() as locks:
                for s in self.steppers:
                    locks.enter_context(s.controller._lock)
                for s, (sample, identity, _) in zip(self.steppers, prepared):
                    _, current = s._control_snapshot(sample['sim_time'])
                    if current['generation'] != identity['generation']:
                        raise RuntimeError('command changed during shared preparation; solve withheld')
            self.owner.step()
            results = []
            for s, item in zip(self.steppers, prepared):
                result = s._validate_tick(item)
                self._identity(s, result)
                results.append(result)
            # All observations must validate before any controller publication.
            for s, item, result in zip(self.steppers, prepared, results):
                s._commit_tick(item, result)
            return {s.identity['robot_id']: result for s, result in zip(self.steppers, results)}
        except BaseException as exc:
            self._fail(exc)
            raise

    def stop(self, robot_id=None):
        """Revoke selected intent; neither this ACK nor close proves rest."""
        selected = [(binding.robot_id, members[1])
                    for binding, members in zip(self.layout.robots, self._members)
                    if robot_id is None or binding.robot_id == robot_id]
        if not selected:
            raise ValueError('unknown robot identity')
        result, first = {}, None
        for name, controller in selected:
            try:
                result[name] = controller.stop(latch=True)
            except BaseException as exc:
                self.containment_errors.append(('stop:' + name, type(exc).__name__))
                if first is None:
                    first = exc
        if first is not None:
            raise first
        return result

    def close(self):
        if not self.closed:
            self.closed = True
            first = None
            try:
                self.stop()
            except BaseException as exc:
                first = exc
            for s in self.steppers:
                s.closed = True
            for name, action in (('owner_contain', lambda: self.owner.contain(
                    'shared lifecycle shutdown; no physical stop verdict')),
                                 ('owner_close', self.owner.close)):
                try:
                    action()
                except BaseException as exc:
                    self.containment_errors.append((name, type(exc).__name__))
                    if first is None:
                        first = exc
            if first is not None:
                raise first
