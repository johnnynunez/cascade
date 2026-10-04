"""Private per-row sharing preserves every public observation and JSON byte."""
from dataclasses import replace
import importlib
import json
from pathlib import Path

import pytest

from cascade.control.mobile_support import SupportObservation
from cascade.control.mobile_base import BaseState
from cascade.sim.microduck_shared_native import SharedRobotView
from cascade.sim.mobile_bridge import _RecordSnapshot
from test_microduck_shared_support_cache import owner_fixture
from test_microduck_shared_scene import shared


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('isaac_microduck_shared')


def original(runner, value):
    return json.dumps(value, allow_nan=False, default=runner.json_default)


@pytest.mark.parametrize('count', [1, 2, 12])
def test_each_row_converts_shared_contacts_once_with_distinct_robot_headers(monkeypatch, runner, count):
    owner, _ = owner_fixture(monkeypatch, count)
    _, _, steppers = shared(count)
    rows, public = {}, {}
    for binding, stepper in zip(owner.layout.robots, steppers):
        controller = stepper.controller
        sample = SharedRobotView(owner, binding, controller.hello()['epoch']).read()
        controller.publish(sample | {'q': sample['q'].tolist(), 'dq': sample['dq'].tolist(),
                                     'fallen': False, 'balance_active': True})
        public[binding.robot_id] = sample | {'controller': controller.state()}
        rows[binding.robot_id] = sample | {'controller': controller._state_for_record()}
        assert type(rows[binding.robot_id]['controller']['state']) is _RecordSnapshot
    expected = original(runner, public)
    calls, encode = [], SupportObservation.as_observation_dict
    snapshots, canonical = [], BaseState.as_dict
    def counted(value):
        calls.append(value)
        return encode(value)
    monkeypatch.setattr(SupportObservation, 'as_observation_dict', counted)
    def counted_snapshot(value):
        snapshots.append(value)
        return canonical(value)
    monkeypatch.setattr(BaseState, 'as_dict', counted_snapshot)
    for _ in range(2):
        assert runner._physics_json(rows) == expected
    assert len(calls) == 2  # One conversion per row, never a cross-row cache.
    assert len(snapshots) == 2  # Canonical contact key order has its own projection.
    decoded = json.loads(expected)
    for name, row in rows.items():
        assert decoded[name]['support']['epoch'] == row['support'].epoch
        assert decoded[name]['support']['model_identity_sha256'] == row['model_identity_sha256']
    first = next(iter(rows.values()))['support']
    reply = first.as_observation_dict()
    reply['contacts'][0]['point_world_m'][0] = 999.
    assert runner._physics_json(rows) == expected


def test_same_first_contact_never_reuses_a_different_interior_or_order(monkeypatch, runner):
    owner, _ = owner_fixture(monkeypatch, 1)
    observed = SharedRobotView(owner, owner.layout.robots[0], 'epoch').read()['support']
    changed = list(observed.contacts)
    changed[1] = replace(changed[1], point_world_m=(42., 0., 0.))
    other = replace(observed, contacts=tuple(changed), step=3, sim_time_s=.015)
    reordered = replace(observed, contacts=tuple(reversed(observed.contacts)))
    row = [observed, other, observed, reordered]
    assert runner._physics_json(row) == original(runner, row)


def test_legacy_mutable_string_support_keeps_original_conversion(monkeypatch, runner):
    class MutableString(str):
        pass
    owner, _ = owner_fixture(monkeypatch, 1)
    observed = SharedRobotView(owner, owner.layout.robots[0], 'epoch').read()['support']
    reason = MutableString('legacy')
    reason.mutable = []
    legacy = replace(observed, reason=reason)
    row = [observed, legacy, legacy, observed]
    expected = original(runner, row)
    calls, encode = [], SupportObservation.as_observation_dict
    def counted(value):
        calls.append(value)
        return encode(value)
    monkeypatch.setattr(SupportObservation, 'as_observation_dict', counted)
    assert runner._physics_json(row) == expected
    assert len(calls) == 3  # Each legacy record retains its complete conversion.
    assert reason.mutable == []


def test_foreign_base_state_in_raw_telemetry_remains_unsupported(runner):
    from test_mobile_base import state_fields
    row = {'foreign_raw_telemetry': BaseState(**state_fields())}
    with pytest.raises(TypeError):
        original(runner, row)
    with pytest.raises(TypeError):
        runner._physics_json(row)


def test_private_controller_record_preserves_legacy_copy_isolation(monkeypatch, runner):
    class MutableString(str):
        pass
    owner, _ = owner_fixture(monkeypatch, 1)
    _, _, steppers = shared(1)
    controller = steppers[0].controller
    assert controller._state_for_record() == controller.state()
    sample = SharedRobotView(owner, owner.layout.robots[0], controller.hello()['epoch']).read()
    reason = MutableString('legacy')
    reason.mutable = []
    sample['support'] = replace(sample['support'], reason=reason)
    controller.publish(sample | {'q': sample['q'].tolist(), 'dq': sample['dq'].tolist(),
                                 'fallen': False, 'balance_active': True})
    private, public = controller._state_for_record(), controller.state()
    assert type(private['state']) is dict and type(private['support']) is dict
    assert runner._physics_json(private) == original(runner, public)
    private['support']['reason'].mutable.append('caller')
    private['state']['support']['reason'].mutable.append('caller')
    fresh = controller.state()
    assert fresh['support']['reason'].mutable == fresh['state']['support']['reason'].mutable == []
    assert reason.mutable == []
