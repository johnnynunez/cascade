"""CPU doubles: shared ownership/cadence, never a physical fleet admission."""
from __future__ import annotations

import numpy as np
import pytest
from test_microduck_stepper import SoftwareActuator, SoftwareBackend, SoftwarePolicy

from cascade.control.microduck_policy import POLICY_JOINTS
from cascade.sim.microduck_contact_support import FOOT_SHAPES
from cascade.sim.microduck_shared import SharedMicroduckStepper, bind_scene
from cascade.sim.microduck_stepper import MicroduckStepper


def layout_fixture(count=2):
    roots = {f'duck{i}': f'/World/Duck{i}' for i in range(count)}
    bodies = [root + '/Geometry/trunk_base' for root in roots.values()]
    labels, children, parents, types, qs, ds = [], [], [], [], [0], [0]
    # Roots first, then joints interleaved across robots. Never assume contiguous DOFs.
    for i in range(count):
        labels.append(f'generated_free_{i}')
        children.append(i); parents.append(-1); types.append(4)
        qs.append(qs[-1] + 7); ds.append(ds[-1] + 6)
    for name in reversed(POLICY_JOINTS):
        for i, root in enumerate(roots.values()):
            bodies.append(root + '/Geometry/' + name + '_body')
            labels.append(root + '/Geometry/' + name)
            parents.append(i); children.append(len(bodies) - 1); types.append(1)
            qs.append(qs[-1] + 1); ds.append(ds[-1] + 1)
    shapes = ['/World/Ground']
    shape_body = [-1]
    for i, root in enumerate(roots.values()):
        shapes.extend(s.replace('/World/MicroDuck', root) for s in FOOT_SHAPES)
        shapes.append(root + '/Geometry/shell')
        shape_body.extend([i] * 3)
    return {'robots': roots, 'scene_model_sha256': 'a'*64, 'joint_labels': labels,
        'joint_q_start': qs, 'joint_qd_start': ds, 'joint_types': types,
        'joint_parent': parents, 'joint_child': children, 'body_labels': bodies,
        'shape_labels': shapes, 'shape_body': shape_body, 'free_type': 4, 'hinge_type': 1,
        'ground_shapes': ('/World/Ground',), 'world_count': 1}


@pytest.mark.parametrize('count', [1, 2, 12])
def test_scoped_indices_cover_each_coordinate_once_and_support_is_disjoint(count):
    layout = bind_scene(**layout_fixture(count))
    assert layout.q_count == 21*count and layout.dof_count == 20*count
    q, dofs, digests = set(), set(), set()
    for binding in layout.robots:
        assert not q.intersection(binding.q_indices + binding.free_q_indices)
        assert not dofs.intersection(binding.dof_indices + binding.free_dof_indices)
        q.update(binding.q_indices + binding.free_q_indices)
        dofs.update(binding.dof_indices + binding.free_dof_indices)
        digests.add(binding.model_identity_sha256)
        assert binding.support_contract()['ground_shapes'] == ['/World/Ground']
        assert all(s.startswith(binding.root_path + '/') for s in binding.robot_shapes)
    assert len(q) == layout.q_count and len(dofs) == layout.dof_count
    assert len(digests) == count


@pytest.mark.parametrize('bad', ['nested', 'duplicate', 'cross_parent', 'cross_shape', 'static_robot',
                                 'ground_robot', 'unknown_body', 'missing_foot', 'wrong_type', 'q_layout', 'worlds', 'cycle'])
def test_ambiguous_or_foreign_mapping_fails_closed(bad):
    kw = layout_fixture()
    if bad == 'nested': kw['robots']['duck1'] = '/World/Duck0/nested'
    elif bad == 'duplicate': kw['joint_labels'][-1] = kw['joint_labels'][-3]
    elif bad == 'cross_parent': kw['joint_parent'][2] = 1
    elif bad == 'cross_shape': kw['shape_body'][1] = 1
    elif bad == 'static_robot': kw['shape_body'][1] = -1
    elif bad == 'ground_robot': kw['shape_body'][0] = 0
    elif bad == 'unknown_body': kw['body_labels'][0] = '/World/Other'
    elif bad == 'missing_foot': kw['shape_labels'][1] += '_other'
    elif bad == 'wrong_type': kw['joint_types'][2] = 4
    elif bad == 'worlds': kw['world_count'] = 2
    elif bad == 'cycle': kw['joint_parent'][2] = kw['joint_child'][2]
    else: kw['joint_q_start'][-1] += 1
    with pytest.raises(ValueError): bind_scene(**kw)


class View:
    def __init__(self, owner, binding, epoch):
        self.owner, self.binding, self.epoch = owner, binding, epoch
    @property
    def dt(self): return self.owner.dt
    @property
    def physics_clock(self): return self.owner.physics_clock
    def read(self):
        return self.owner.read() | {'robot_id': self.binding.robot_id, 'epoch': self.epoch,
            'model_identity_sha256': self.binding.model_identity_sha256}
    def contain(self, reason): self.owner.contain(reason)


def shared(count=2):
    owner = SoftwareBackend()
    layout = bind_scene(**layout_fixture(count))
    steppers = []
    for binding in layout.robots:
        # Constructor identity is immutable in the real controller; each fixture
        # creates it with distinct kwargs through the same production class.
        from cascade.sim.mobile_bridge import MobileBridgeController
        ctrl = MobileBridgeController(robot_id=binding.robot_id, source='software-test-not-physics',
            engine='newton', device='cuda:0', asset_sha256='a'*64, policy_sha256='b'*64,
            model_identity_sha256=binding.model_identity_sha256,
            support_contract=binding.support_contract(), max_linear_speed=.3,
            max_angular_speed=.5, max_duration_s=10., lease_s=.3,
            max_state_age_s=1., max_action_wall_s=10., clock=lambda: 1.)
        view = View(owner, binding, ctrl.hello()['epoch'])
        actuator = SoftwareActuator(owner)
        actuator.coordinate_indices = binding.q_indices, binding.dof_indices
        steppers.append(MicroduckStepper(view, ctrl, SoftwarePolicy(owner), actuator,
            max_steps=100, max_wall_s=100., min_height_m=.05, max_height_m=.3,
            max_tilt_rad=.7, clock=lambda: 1.))
    fleet = SharedMicroduckStepper(owner, steppers, layout=layout)
    return fleet, owner, steppers


@pytest.mark.parametrize('count', [1, 2, 12])
def test_all_policy_and_bam_preparations_precede_one_global_solve(count):
    fleet, owner, steppers = shared(count)
    fleet.start()
    for _ in range(8): fleet.tick()
    assert owner.step_count == 10
    assert len([e for e in owner.events if e[0] == 'solve']) == 8
    for step in range(2, 10):
        events = [e[0] for e in owner.events if e[1] == step]
        assert events[-1] == 'solve' and events.count('before') == count
        assert events.count('infer') == (count if step in (2, 6) else 0)
        if step in (2, 6): assert events[:count] == ['infer'] * count
    for item in steppers:
        assert item.steps == 8 and item.policy_commits == 2
        assert item.controller.state()['state']['step'] == 10
    fleet.close(); fleet.close()
    assert owner.closed == 1


def test_preparation_failure_never_solves_partial_batch_or_publishes():
    fleet, owner, steppers = shared()
    fleet.start()
    def fail(_): raise RuntimeError('second actuator failed')
    steppers[1].actuator.before_step = fail
    with pytest.raises(RuntimeError, match='second actuator'): fleet.tick()
    assert owner.step_count == 2
    assert all(s.controller.state()['state'] is None for s in steppers)
    assert all(s.failure for s in steppers)
    with pytest.raises(RuntimeError): fleet.tick()


def test_repeated_stop_during_peer_inference_withholds_without_mutating_history():
    fleet, owner, steppers = shared()
    fleet.start()
    original = steppers[1].policy.preview
    def crossed(obs):
        steppers[0].controller.stop(latch=True)
        return original(obs)
    steppers[1].policy.preview = crossed
    assert fleet.tick() is None
    assert owner.step_count == 2
    assert all(not s.failure and s.policy_commits == 0 and s.steps == 0 for s in steppers)
    assert all(not s.actuator.targets for s in steppers)
    assert all([r['status'] for r in s.policy_records] == ['discarded', 'discarded'] for s in steppers)
    steppers[1].policy.preview = original
    assert fleet.tick()['duck0']['step'] == 3


@pytest.mark.parametrize('bad', ['epoch', 'model', 'robot', 'clock'])
def test_crosswired_or_stale_robot_observation_faults_every_participant(bad):
    fleet, owner, steppers = shared()
    fleet.start()
    original = steppers[1].backend.read
    def wrong():
        out = original()
        out[{'epoch':'epoch', 'model':'model_identity_sha256', 'robot':'robot_id', 'clock':'step'}[bad]] = (
            1 if bad == 'clock' else 'wrong')
        return out
    steppers[1].backend.read = wrong
    with pytest.raises(RuntimeError): fleet.tick()
    assert owner.step_count == 2
    assert all(s.failure for s in steppers)


def test_single_robot_original_tick_and_shared_tick_preserve_policy_bam_cadence():
    from test_microduck_stepper import make_stepper
    original, baseline, _, policy, actuator = make_stepper(clock=lambda: 1.)
    fleet, owner, steppers = shared(1)
    original.start(); fleet.start()
    for _ in range(9):
        a = original.tick()
        b = fleet.tick()['duck0']
        for key in ('step', 'sim_time', 'position', 'contacts', 'fallen', 'policy_target_held'):
            assert a[key] == b[key]
    assert baseline.events == owner.events
    np.testing.assert_array_equal(policy.inputs, steppers[0].policy.inputs)
    np.testing.assert_array_equal(actuator.targets, steppers[0].actuator.targets)


def test_individual_command_and_stop_never_replace_another_robots_intent_or_history():
    from test_microduck_stepper import command
    fleet, owner, steppers = shared()
    fleet.start(); fleet.tick()
    command(steppers[0].controller)
    for s in steppers:
        s.policy.infer = lambda obs: np.full(14, obs[0, 48], np.float32)
    for _ in range(4): fleet.tick()
    np.testing.assert_array_equal(steppers[0].policy.previous_action, np.full(14, .2, np.float32))
    np.testing.assert_array_equal(steppers[1].policy.previous_action, np.zeros(14, np.float32))
    other = steppers[1].controller.hello()
    ack = fleet.stop('duck0')
    assert not ack['duck0']['physical_stop_verified']
    assert steppers[1].controller.hello() == other
    # Keep physical delay/50Hz cadence rather than erase histories on stop.
    for _ in range(4): fleet.tick()
    assert not steppers[0].policy.previous_action.any()
    assert len(steppers[0].policy.inputs) == 1  # infer was replaced after first tick
    assert steppers[0].policy_commits == steppers[1].policy_commits == 3
    before = owner.step_count
    stopped = fleet.stop()
    assert set(stopped) == {'duck0', 'duck1'} and owner.step_count == before
    with pytest.raises(ValueError): fleet.stop('unknown')


def contact(layout, a, b):
    return {'shape_a_id': a, 'shape_b_id': b, 'shape_a': layout.shape_labels[a], 'shape_b': layout.shape_labels[b],
        'force_on_b_world_n': [0., 0., 3.], 'normal_force_n': 3., 'normal_a_to_b_world': [0., 0., 1.],
        'point_world_m': [0., 0., 0.]}


@pytest.mark.parametrize('pair', ['ground', 'neighbor', 'zero_neighbor', 'unavailable'])
def test_real_support_checker_never_treats_a_neighbor_as_ground(pair):
    from test_mobile_effects import limits

    from cascade.agent.base_effects import BasePostconditionChecker
    layout = bind_scene(**layout_fixture())
    raw = {'version': 1, 'status': 'known', 'reason': '', 'step': 3, 'sim_time_s': .015,
        'contacts': [contact(layout, 0, 1), contact(layout, 0, 4)]}
    if pair in ('neighbor', 'zero_neighbor'):
        row = contact(layout, 1, 4)
        if pair == 'zero_neighbor':
            row['normal_force_n'] = 0.; row['force_on_b_world_n'] = [0., 0., 0.]
        raw['contacts'].append(row)
    if pair == 'unavailable':
        raw.update(status='unavailable', reason='global constraint overflow', contacts=[])
    for binding in layout.robots:
        bound = layout.robot_support(raw, binding, epoch='fixture', clock=(3, .015))
        checker = BasePostconditionChecker(lambda: None, limits=limits(), support_contract=binding.support_contract())
        verdict, _ = checker._support({'support': bound,
            'model_identity_sha256': binding.model_identity_sha256}, require_load=True)
        expected = {'ground':'confirmed', 'neighbor':'refuted', 'zero_neighbor':'confirmed', 'unavailable':'unverified'}[pair]
        assert verdict == expected
        assert len(bound['contacts']) == len(raw['contacts'])  # no filtering/hidden neighbor rows
        bound['contacts'].clear()
    assert len(raw['contacts']) == (0 if pair == 'unavailable' else 3 if 'neighbor' in pair else 2)


@pytest.mark.parametrize('bad', ['step', 'epoch', 'model', 'shape'])
def test_support_cannot_be_restamped_or_rebound(bad):
    layout = bind_scene(**layout_fixture())
    raw = {'version': 1, 'status': 'known', 'reason': '', 'step': 3, 'sim_time_s': .015, 'contacts': [contact(layout, 0, 1)]}
    if bad == 'step': raw['step'] = 2
    elif bad == 'epoch': raw['epoch'] = 'old'
    elif bad == 'model': raw['model_identity_sha256'] = 'b'*64
    else: raw['contacts'][0]['shape_b'] = layout.shape_labels[4]
    with pytest.raises(ValueError): layout.robot_support(raw, layout.robots[0], epoch='fixture', clock=(3, .015))


@pytest.mark.parametrize('field', ['actuator', 'controller', 'policy', 'backend', 'indices'])
def test_shared_binding_rejects_aliasing_or_mutation_before_next_prepare(field):
    fleet, owner, steppers = shared()
    fleet.start()
    if field == 'indices':
        steppers[0].actuator.coordinate_indices = steppers[1].actuator.coordinate_indices
    else:
        setattr(steppers[0], field, getattr(steppers[1], field))
    with pytest.raises(RuntimeError, match='ownership binding'): fleet.tick()
    assert not owner.events and owner.step_count == 2


def test_completed_shared_solve_with_bad_peer_read_publishes_no_partial_batch():
    fleet, owner, steppers = shared()
    fleet.start()
    original = steppers[1].backend.read
    def late():
        row = original()
        if owner.step_count == 3: row['epoch'] = 'alien-epoch'
        return row
    steppers[1].backend.read = late
    with pytest.raises(RuntimeError, match='identity/epoch'): fleet.tick()
    assert owner.step_count == 3
    assert all(s.controller.state()['state'] is None for s in steppers)


def test_interrupted_policy_contains_before_a_solve_and_does_not_resume_partial_bam():
    from cascade.apps.signal_stop import SignalRequest
    fleet, owner, steppers = shared()
    fleet.start()
    def interrupted(_): raise SignalRequest(15)
    steppers[1].policy.preview = interrupted
    with pytest.raises(SignalRequest): fleet.tick()
    assert owner.step_count == 2
    with pytest.raises(RuntimeError, match='not running'): fleet.tick()
    fleet.close()
    assert owner.closed == 1


@pytest.mark.parametrize('secondary', [None, 'controller', 'owner', 'broken_str'])
def test_failure_containment_attempts_every_owner_and_preserves_primary_exception(secondary):
    fleet, owner, steppers = shared()
    fleet.start()
    class BrokenError(RuntimeError):
        def __str__(self): raise KeyboardInterrupt('broken formatting')
    error = BrokenError() if secondary == 'broken_str' else RuntimeError('BAM failure\nsecond line')
    original_fault = steppers[0].controller.fault
    faults, contained = [], []
    def first_fault(reason):
        faults.append(reason)
        if secondary == 'controller': raise OSError('controller fixture')
        original_fault(reason)
    steppers[0].controller.fault = first_fault
    def contain(reason):
        contained.append(reason)
        if secondary == 'owner': raise OSError('owner fixture')
    owner.contain = contain
    def fail(_): raise error
    steppers[1].actuator.before_step = fail
    with pytest.raises(RuntimeError) as caught: fleet.tick()
    assert caught.value is error
    assert len(faults) == len(contained) == 1
    assert steppers[1].controller.state()['controller'] == 'fault'
    assert owner.step_count == 2
    assert len(fleet.containment_errors) == (1 if secondary in ('controller', 'owner') else 0)


def test_fleet_identity_is_not_a_usd_path_or_short_slug():
    kw = layout_fixture(1)
    name = 'fleet/duck+front@' + 'a'*100
    kw['robots'] = {name: '/World/Duck0'}
    binding, = bind_scene(**kw).robots
    assert binding.robot_id == name and binding.root_path == '/World/Duck0'


@pytest.mark.parametrize('operation', ['stop', 'close'])
def test_stop_failure_still_revokes_peers_and_close_attempts_native_cleanup(operation):
    fleet, owner, steppers = shared()
    fleet.start()
    primary = OSError('first controller cannot stop')
    def stop_failure(*, latch):
        raise primary
    def containment_failure(reason):
        raise RuntimeError('containment also failed')
    steppers[0].controller.stop = stop_failure
    owner.contain = containment_failure
    with pytest.raises(OSError) as caught:
        getattr(fleet, operation)()
    assert caught.value is primary
    assert steppers[1].controller.state()['latched']
    if operation == 'close':
        assert all(s.closed for s in steppers)
        assert owner.closed == 1
        assert fleet.containment_errors == [('stop:duck0', 'OSError'), ('owner_contain', 'RuntimeError')]
        fleet.close()
        assert owner.closed == 1
    else:
        assert owner.closed == 0


class SnapshotActuator(SoftwareActuator):
    """Software double of an adapter that offers a per-step host snapshot."""
    captures = []

    def host_snapshot(self):
        token = object()
        SnapshotActuator.captures.append((self.backend.step_count, token))
        return token

    def before_step(self, dt, *, snapshot=None):
        self.backend.events.append(('snapshot', self.backend.step_count, id(snapshot)))
        super().before_step(dt)


def test_shared_tick_hands_one_host_snapshot_to_every_actuator():
    SnapshotActuator.captures = []
    fleet, owner, steppers = shared(3)
    for stepper in steppers:
        actuator = SnapshotActuator(owner)
        actuator.coordinate_indices = stepper.actuator.coordinate_indices
        stepper.actuator = actuator
    fleet._members = tuple((s.backend, s.controller, s.policy, s.actuator) for s in steppers)
    fleet.start()
    for _ in range(4): fleet.tick()
    # Exactly one capture per solved tick, on the pre-solve step count, and the
    # same object reached all three adapters before that tick's solve.
    assert [step for step, _ in SnapshotActuator.captures] == [2, 3, 4, 5]
    for step, token in SnapshotActuator.captures:
        seen = [e[2] for e in owner.events if e[0] == 'snapshot' and e[1] == step]
        assert seen == [id(token)] * 3


def test_shared_tick_without_snapshot_support_keeps_the_plain_actuation_call():
    fleet, owner, steppers = shared(2)
    fleet.start()
    fleet.tick()
    assert [e for e in owner.events if e[0] == 'snapshot'] == []
    assert [e[0] for e in owner.events if e[1] == 2].count('before') == 2
