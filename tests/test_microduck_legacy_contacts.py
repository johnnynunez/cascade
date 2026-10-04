"""Internal contact sharing preserves public lists, isolation and every row."""
import copy
import json
from types import SimpleNamespace

import pytest

from cascade.control.mobile_telemetry import _FrozenList, _freeze_contacts, _public_contacts
from cascade.sim.microduck_newton import _read_native_states, read_native_states
from cascade.sim.microduck_shared_native import SharedRobotView
from test_microduck_scene_read import scene
from test_microduck_shared_scene import shared
from test_microduck_shared_support_cache import owner_fixture
from test_microduck_row_serialization import runner, original  # noqa: F401


def encoded(value):
    return json.dumps(value, allow_nan=False, default=lambda item: item.tolist())


@pytest.mark.parametrize('count', [1, 2, 12])
def test_native_private_capture_shares_only_immutable_contacts_for_one_solve(count):
    ns, kwargs = scene(count)
    public, private = read_native_states(ns, **kwargs), _read_native_states(ns, **kwargs)
    assert encoded(private) == encoded(public)
    first = private['robot0']
    for robot, sample in private.items():
        for field in ('contact_pairs', 'contact_constraint_addresses'):
            assert type(sample[field]) is _FrozenList
            assert sample[field] is first[field]
            assert copy.deepcopy(sample[field]) is sample[field]
            assert type(public[robot][field]) is list
        detached = _public_contacts(copy.deepcopy(sample))
        assert encoded(detached) == encoded(public[robot])
        detached['contact_pairs'][0][0] = 'caller mutation'
        detached['contact_constraint_addresses'][0] = 999
        assert encoded(sample) == encoded(public[robot])
    assert type(public['robot0']['contact_pairs'][0]) is list
    ns.simulation_step_count, ns.sim_time = 3, .015
    ns.contacts.rigid_contact_shape0.value[0] = 0
    ns.contacts.rigid_contact_shape1.value[0] = 1
    second = _read_native_states(ns, **kwargs)
    assert second['robot0']['contact_pairs'] is not first['contact_pairs']
    assert second['robot0']['step'] == 3 and first['step'] == 2
    assert encoded(second) == encoded(read_native_states(ns, **kwargs))


class MutableString(str):
    pass


class MutableTuple(tuple):
    pass


@pytest.mark.parametrize('leaf', [[], {}, ('plain',), MutableTuple(('plain',)),
                                 MutableString('shape'), True, 1.0])
def test_mutable_or_nonplain_leaf_cannot_gain_deepcopy_sharing(leaf):
    with pytest.raises(ValueError, match='plain immutable'):
        _FrozenList([leaf])


def test_private_sequence_has_no_mutable_attributes_or_subclasses():
    value = _FrozenList([_FrozenList(['shape-a', 'shape-b']), 3])
    with pytest.raises(AttributeError):
        value.mutable = []
    with pytest.raises(TypeError, match='cannot be subclassed'):
        type('MutablePrivate', (_FrozenList,), {})
    assert json.loads(json.dumps(value)) == [['shape-a', 'shape-b'], 3]


@pytest.mark.parametrize('count', [1, 2, 12])
def test_all_public_reads_detach_and_private_record_keeps_complete_json(monkeypatch, runner, count):
    import cascade.sim.microduck_shared_native as native
    owner, _ = owner_fixture(monkeypatch, count)
    capture = native._read_native_states
    common = dict(contact_pairs=[[b.foot_shapes[0], '/World/Ground'] for b in owner.layout.robots],
                  contact_constraint_addresses=list(range(count)))
    _freeze_contacts(common)

    def with_contacts(*args, **kwargs):
        return {key: sample | common for key, sample in capture(*args, **kwargs).items()}

    monkeypatch.setattr(native, '_read_native_states', with_contacts)
    _, _, steppers = shared(count)
    public_rows, private_rows = {}, {}
    for binding, stepper in zip(owner.layout.robots, steppers):
        view = SharedRobotView(owner, binding, stepper.controller.hello()['epoch'])
        stepper.backend = view
        private = stepper._read()
        assert private['contact_pairs'] is common['contact_pairs']
        stepper._publish(private, True)
        public_rows[binding.robot_id] = view.read() | {'controller': stepper.controller.state()}
        private_rows[binding.robot_id] = view._read_for_stepper() | {
            'controller': stepper.controller._state_for_record()}
        assert private_rows[binding.robot_id]['controller']['contact_pairs'] is common['contact_pairs']
        replies = [owner.read_robots()[binding.robot_id], owner.read_robot(binding.robot_id),
                   owner.read_bound_robot(binding, view.epoch), view.read(), stepper.controller.state()]
        for reply in replies:
            assert type(reply['contact_pairs']) is type(reply['contact_pairs'][0]) is list
            assert type(reply['contact_constraint_addresses']) is list
            assert len(reply['contact_pairs']) == count
            reply['contact_pairs'][0][0] = 'changed'
            reply['contact_constraint_addresses'].clear()
        assert stepper.controller.state()['contact_pairs'][0][0] == owner.layout.robots[0].foot_shapes[0]
        assert len(owner.read_robot(binding.robot_id)['contact_constraint_addresses']) == count
        stepper.controller.stop()
        stopped = stepper.controller.state()
        assert stopped['latched'] and type(stopped['contact_pairs'][0]) is list
        stepper.controller.fault('diagnostic fault')
        assert type(stepper.controller.state()['contact_pairs'][0]) is list
    assert runner._physics_json(private_rows) == original(runner, public_rows)


def test_legacy_contact_values_retain_full_deepcopy_isolation():
    from cascade.sim.mobile_bridge import _copy_observation
    value = MutableString('legacy')
    value.mutable = []
    original = {'contact_pairs': [[value, 'ground']],
                'contact_constraint_addresses': MutableTuple(([1],))}
    reply = _public_contacts(_copy_observation(original))
    assert type(reply['contact_pairs'][0][0]) is MutableString
    assert type(reply['contact_constraint_addresses']) is MutableTuple
    reply['contact_pairs'][0][0].mutable.append('changed')
    reply['contact_constraint_addresses'][0].append(2)
    assert value.mutable == [] and original['contact_constraint_addresses'][0] == [1]


def test_private_view_falls_back_to_an_existing_public_bound_reader():
    binding = SimpleNamespace(robot_id='robot', model_identity_sha256='a'*64)
    sample = {'contact_pairs': [['shape', 'ground']], 'support': {'unavailable': True}}
    owner = SimpleNamespace(read_bound_robot=lambda b, e: copy.deepcopy(sample))
    view = SharedRobotView(owner, binding, 'epoch')
    private, public = view._read_for_stepper(), view.read()
    assert private == public
    private['contact_pairs'][0][0] = 'caller'
    assert public['contact_pairs'] == sample['contact_pairs']
