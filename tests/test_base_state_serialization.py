"""Exact wire shape and copy isolation for the mobile snapshot serializer."""
from dataclasses import asdict, dataclass
import json

import pytest

from cascade.control.mobile_base import BaseState
from mobile_support_fixture import support
from test_mobile_base import state_fields


def observation(kind='known'):
    data = state_fields() | {'model_identity_sha256': 'e' * 64}
    solved = support(data['step'], data['sim_time_s']) | {
        'epoch': data['epoch'], 'model_identity_sha256': data['model_identity_sha256']}
    if kind == 'unavailable':
        solved.update(status='unavailable', reason='no completed solve', contacts=[])
    data['support'] = None if kind == 'none' else solved
    return data


def legacy(state):
    return {key: list(value) if isinstance(value, tuple) else value
            for key, value in asdict(state).items()}


@pytest.mark.parametrize('kind', ['known', 'unavailable', 'none'])
def test_snapshot_preserves_exact_json_container_types_and_detached_outputs(kind):
    state = BaseState(**observation(kind))
    expected, reply = legacy(state), state.as_dict()
    assert reply == expected
    assert json.dumps(reply, allow_nan=False) == json.dumps(expected, allow_nan=False)
    reply['joint_positions'][0] = 90.
    if state.support is not None:
        assert type(reply['support']['contacts']) is tuple
        reply['support']['epoch'] = 'foreign'
        if state.support.contacts:
            contact = reply['support']['contacts'][0]
            assert type(contact['point_world_m']) is tuple
            contact['point_world_m'] = (90., 0., 0.)
    assert state.as_dict() == expected


@pytest.mark.parametrize('path', [('robot_id',), ('joint_names', 0), ('contacts', 0),
    ('controller_status',), ('support', 'reason'), ('support', 'contacts', 0, 'shape_a')])
def test_mutable_string_subclasses_keep_legacy_copy_semantics(path):
    class MutableString(str):
        pass

    data = observation()
    target = data
    for key in path[:-1]:
        target = target[key]
    value = MutableString(target[path[-1]])
    value.mutable = ['original']
    target[path[-1]] = value
    state = BaseState(**data)
    reply, expected = state.as_dict(), legacy(state)
    assert json.dumps(reply) == json.dumps(expected)
    leaf = reply
    for key in path:
        leaf = leaf[key]
    assert type(leaf) is MutableString and leaf is not value
    leaf.mutable.append('caller')
    fresh = state.as_dict()
    for key in path:
        fresh = fresh[key]
    assert fresh.mutable == value.mutable == ['original']


def test_extended_state_keeps_extra_fields_and_nested_legacy_isolation():
    @dataclass(frozen=True)
    class ExtendedState(BaseState):
        telemetry: tuple = ()

    state = ExtendedState(**observation(), telemetry=({'samples': [1.]},))
    reply = state.as_dict()
    assert reply == legacy(state)
    assert json.dumps(reply) == json.dumps(legacy(state))
    reply['telemetry'][0]['samples'].append(2.)
    assert state.as_dict()['telemetry'] == [{'samples': [1.]}]
