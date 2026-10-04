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
from dataclasses import asdict, dataclass, fields, replace

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

    def capture_support(self, observed, *, clock):
        """One immutable full-scene decode; robot provenance is attached later."""
        from cascade.control.mobile_support import SupportObservation
        if (not isinstance(observed, dict) or 'epoch' in observed or 'model_identity_sha256' in observed
                or (observed.get('step'), observed.get('sim_time_s')) != clock):
            raise ValueError('unbound completed scene support required')
        data = deepcopy(observed)
        data.update(epoch='shared-scene-support', model_identity_sha256=self.robots[0].scene_model_sha256)
        parsed = SupportObservation.from_dict(data)
        return _SharedSupport(self, parsed)

    def robot_support(self, observed, binding, *, epoch, clock):
        """Keep the full shared contact set: another robot is never ground.

        Global completeness comes only from the original solved decoder. This
        method binds/copies that result; it cannot turn unavailable into known.
        """
        from cascade.control.mobile_support import SupportObservation
        if type(observed) is _SharedSupport:
            if (observed.layout is not self or binding not in self.robots
                    or (observed.observation.step, observed.observation.sim_time_s) != clock
                    or observed.observation.model_identity_sha256 != binding.scene_model_sha256):
                raise ValueError('support scene/robot/clock binding mismatch')
            return replace(observed.observation, epoch=epoch,
                           model_identity_sha256=binding.model_identity_sha256)
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


@dataclass(frozen=True)
class _SharedSupport:
    layout: SceneLayout
    observation: object

    def __post_init__(self):
        from cascade.control.mobile_support import immutable_support
        def immutable(value):
            if type(value) in (str, int):
                return True
            if type(value) is tuple:
                return all(immutable(item) for item in value)
            if type(value) in (SceneLayout, RobotBinding):
                return all(immutable(getattr(value, f.name)) for f in fields(value))
            return False

        if (type(self.layout) is not SceneLayout or not immutable(self.layout)
                or not self.layout.robots or type(self.layout.shape_labels) is not tuple
                or not immutable_support(self.observation)
                or self.observation.epoch != 'shared-scene-support'
                or any(b.scene_model_sha256 != self.observation.model_identity_sha256 for b in self.layout.robots)):
            raise ValueError('immutable shared support/model binding required')
        for contact in self.observation.contacts:
            for sid, name in ((contact.shape_a_id, contact.shape_a), (contact.shape_b_id, contact.shape_b)):
                if sid >= len(self.layout.shape_labels) or self.layout.shape_labels[sid] != name:
                    raise ValueError('support shape differs from shared model')

    def __deepcopy__(self, memo):
        # Created only by capture_support after strict schema/shape validation;
        # every reachable field is frozen, including all contact vectors.
        return self

    def as_observation_dict(self):
        result = self.observation.as_observation_dict()
        del result['epoch'], result['model_identity_sha256']
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

    All policy previews finish before a memory-only cohort fence. A crossing
    stop discards pure staging and retries once; repeated revocation returns
    None without a solve or publication. After the fence, that solve is in
    flight: a stop cannot abort SDK work, and the next regular policy slot uses
    its new intent. Native/partial commit faults contain the entire scene.
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
        self.withheld_ticks = 0
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
        staged = []
        try:
            self._check_bindings()
            for retry in range(2):
                staged = []
                for s in self.steppers:
                    item = s._stage_tick(retry=retry)
                    self._identity(s, item.sample)
                    staged.append(item)
                for s in self.steppers:
                    s._check_wall()
                self._check_bindings()
                completed = self.owner.physics_clock
                if any(completed != (item.sample['step'], item.sample['sim_time']) for item in staged):
                    raise RuntimeError('shared physics clock changed before cohort admission')
                # Only bounded memory checks/copies under permission locks.
                # ONNX, target uploads and BAM/device work remain outside.
                with ExitStack() as locks:
                    for s in self.steppers:
                        locks.enter_context(s.controller._lock)
                    valid = True
                    for s, item in zip(self.steppers, staged):
                        command, current = s._control_snapshot(item.sample['sim_time'])
                        s._check_wall(checkpoint=False)
                        valid &= (current['generation'] == item.identity['generation']
                                  and np.array_equal(command, item.command))
                    if valid:
                        for s, item in zip(self.steppers, staged):
                            if item.candidate is not None:
                                action, _, record = item.candidate
                                record['commit_outcome'] = 'in_progress'
                                s.policy.commit(action)
                                s.policy_commits += 1
                                s.policy_target_generation = item.identity['generation']
                                record.update(status='evaluated', committed=True, commit_outcome='returned',
                                              commit_generation=item.identity['generation'])
                if valid:
                    break
                self._discard(staged, 'cohort invalidated before shared fence')
            else:
                self.withheld_ticks += 1
                return None
            # The cohort is admitted. Do not re-veto or rewind a partially
            # applied BAM history if stop arrives after this linearization.
            for s, item in zip(self.steppers, staged):
                if item.candidate is not None:
                    s.actuator.set_targets(item.candidate[1])
                s._prepare_actuator(item.prepared)
            self.owner.step()
            results = []
            for s, item in zip(self.steppers, staged):
                result = s._validate_tick(item.prepared)
                self._identity(s, result)
                results.append(result)
            # All observations must validate before any controller publication.
            for s, item, result in zip(self.steppers, staged, results):
                s._commit_tick(item.prepared, result)
            return {s.identity['robot_id']: result for s, result in zip(self.steppers, results)}
        except BaseException as exc:
            self._discard(staged, 'shared episode failed before solve completion')
            self._fail(exc)
            raise

    @staticmethod
    def _discard(staged, reason):
        for item in staged:
            if item.candidate is None:
                continue
            record = item.candidate[2]
            if record.get('commit_outcome') == 'in_progress':
                record.update(status='failed', committed=None, commit_outcome='unknown_due_to_failure')
            elif record['committed'] is False:
                record.update(status='discarded', reason=reason)

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
