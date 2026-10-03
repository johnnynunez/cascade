"""Existing-record diagnostics only: no simulator construction or integration."""
from collections import deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest
from test_placement_reporting import runtime as runtime  # noqa: F401

from cascade.control.lazy_arm import LazyArm
from cascade.control.mujoco_arm import MujocoArm
from cascade.sim import mujoco_placement as placement
from cascade.types import SkillError


@pytest.fixture
def recorded(monkeypatch):
    class RecordedWorld:
        path = 'synthetic-records-not-a-native-scene'
        lock = threading.RLock()

        def __getattr__(self, name):
            raise AssertionError('diagnostic attempted native/unknown read: '+name)

    world = RecordedWorld()
    arm = MujocoArm.__new__(MujocoArm)  # No constructor, SDK or physics.
    arm._engine = SimpleNamespace(_world=world)
    history = placement.PlacementHistory.__new__(placement.PlacementHistory)
    history.world, history.arm = world, arm
    history.epoch, history.identity = 'captured-epoch', 'a'*64
    history.descriptor, history.support_names = {'frame': 'declared'}, ['floor']
    history.step, history.last_final_time = 50, 1.25
    history.error = 'recorded capture failed; do not revalidate'
    history.rows = deque([{'step': 50, 'value': [1., 2.], 'epoch': history.epoch}], maxlen=256)
    history.confirmed_prefix = {'blue'}
    history.prefix_faults = {'blue': {'step': 49, 'reason': 'recorded disturbance'}}
    history.retired_obligations = [{'object': 'previous', 'retired': True}]
    world.placement_history = history

    def prohibited(*args, **kwargs):
        raise AssertionError('diagnostic called validation, SDK or verdict code')

    for name in ('guard', 'read', 'geometry', '_validated_geometry', 'capture', 'reset'):
        monkeypatch.setattr(history, name, prohibited)
    for name in ('audit', 'model_digest', 'state_digest'):
        monkeypatch.setattr(placement, name, prohibited)
    return SimpleNamespace(raw=arm), history


def test_snapshot_copies_under_lock_without_revalidation_or_ledger_mutation(recorded):
    arm, history = recorded
    before = deepcopy((list(history.rows), history.prefix_faults, history.retired_obligations))

    class LockedRecord(dict):
        def __deepcopy__(self, memo):
            assert history.world.lock._is_owned()
            return deepcopy(dict(self), memo)

    history.rows[0] = LockedRecord(history.rows[0])
    value = history.diagnostic_snapshot()
    assert value['diagnostic_only'] and value['physical_task_verdict'] is False
    assert value['capture_error'] == history.error
    assert value['epoch'] == 'captured-epoch' and value['model_sha256'] == 'a'*64
    assert value['confirmed_prefix'] == ['blue']
    assert not history.world.lock._is_owned()
    value['records'][0]['value'].append(3.)
    value['prefix_faults']['blue']['reason'] = 'consumer mutation'
    value['retired_area_obligations'].clear()
    assert (list(history.rows), history.prefix_faults, history.retired_obligations) == before


def test_snapshot_bounds_records_and_does_not_invent_empty_history_support(recorded):
    _, history = recorded
    history.rows = deque(({'step': i} for i in range(300)), maxlen=300)
    value = history.diagnostic_snapshot()
    assert len(value['records']) == 256
    assert [r['step'] for r in value['records']] == list(range(44, 300))
    history.rows.clear()
    empty = history.diagnostic_snapshot()
    assert empty['records'] == [] and not empty['physical_task_verdict']
    assert history.prefix_faults['blue']['reason'] == 'recorded disturbance'


def test_encoding_and_file_io_follow_lock_release_and_keep_copied_rows(recorded, monkeypatch, tmp_path):
    arm, history = recorded
    dumps, open_file = placement.json.dumps, Path.open

    def encode(value, **kwargs):
        assert not history.world.lock._is_owned()
        # A later owner record must not mutate the already-copied payload.
        history.rows.append({'step': 51})
        return dumps(value, **kwargs)

    def write(path, *args, **kwargs):
        assert not history.world.lock._is_owned()
        return open_file(path, *args, **kwargs)

    monkeypatch.setattr(placement.json, 'dumps', encode)
    monkeypatch.setattr(Path, 'open', write)
    result = placement.save_placement_diagnostic(arm, tmp_path)
    data = Path(result['evidence_path']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == result['evidence_sha256']
    assert json.loads(data)['records'] == [{'step': 50, 'value': [1., 2.], 'epoch': 'captured-epoch'}]
    assert result['records'] == 1 and history.rows[-1] == {'step': 51}
    assert result['diagnostic_only'] and not result['physical_task_verdict']


@pytest.mark.parametrize('failure', ['encode', 'write', 'owner'])
def test_diagnostic_failure_preserves_owner_error_and_ledger(recorded, monkeypatch, tmp_path, failure):
    arm, history = recorded
    before = deepcopy((list(history.rows), history.error, history.prefix_faults, history.confirmed_prefix))
    if failure == 'encode':
        history.rows[0]['bad'] = float('nan')
        before[0][0]['bad'] = history.rows[0]['bad']
    elif failure == 'write':
        def failed(*args, **kwargs):
            assert not history.world.lock._is_owned()
            raise OSError('synthetic storage fault')
        monkeypatch.setattr(Path, 'open', failed)
    else:
        monkeypatch.setattr(history, 'arm', object())
    result = placement.save_placement_diagnostic(arm, tmp_path)
    assert result['diagnostic_only'] and not result['physical_task_verdict']
    assert 'error' in result and 'evidence_path' not in result
    assert (list(history.rows), history.error, history.prefix_faults, history.confirmed_prefix) == before


def test_diagnostics_never_materialize_a_lazy_backend(tmp_path):
    lazy = LazyArm(lambda: pytest.fail('unexpected backend activation'))
    assert placement.save_placement_diagnostic(SimpleNamespace(raw=lazy), tmp_path) is None
    assert not lazy.connected


def test_materialized_lazy_wrapper_copies_only_its_existing_owner(recorded, tmp_path, monkeypatch):
    arm, history = recorded
    lazy = LazyArm(lambda: pytest.fail('unexpected backend activation'))
    lazy._arm = arm.raw
    monkeypatch.setattr(lazy, '_ensure', lambda: pytest.fail('diagnostic must use the existing registry'))
    result = placement.save_placement_diagnostic(SimpleNamespace(raw=lazy), tmp_path)
    assert result['epoch'] == history.epoch and result['records'] == 1
    assert result['diagnostic_only'] and not result['physical_task_verdict']


@pytest.mark.parametrize('failure', ['actor', 'exception', 'storage'])
def test_ordinary_failed_pick_trace_retains_diagnostic_without_new_verdict(runtime, recorded, monkeypatch, failure):
    rt, (arm, history) = runtime, recorded
    original = rt.arm.raw
    # The runtime/camera remain mock; only the pre-existing diagnostic owner is
    # represented by this explicit double. No physical task is claimed here.
    arm.raw.get_state = original.get_state
    arm.raw.disconnect = original.disconnect
    arm.raw.stop = original.stop
    arm.raw._stopped = False
    arm.raw._cfg = rt.cfg.arm
    monkeypatch.setattr(rt.arm, '_arm', arm.raw)
    rt.attach_verifier(object_pose=lambda _: [.2, -.12, .02])
    pending = object()
    rt.arm.harness._pending_model_withdrawal = pending

    def failed(**args):
        if failure == 'exception':
            raise SkillError('release route refused')
        return {'ok': False, 'error': 'release route refused', 'home_skipped': True}

    rt.skill_pick_and_place = failed
    if failure == 'storage':
        (rt.trace.run_dir/'placement').write_text('not a directory')
    before_ledger = deepcopy((history.prefix_faults, history.confirmed_prefix, history.retired_obligations))
    result = rt.execute('pick_and_place', {'object': 'red cube'})
    assert result['ok'] is False and result['verified'] is False
    assert result['postcondition']['status'] == 'refuted'
    assert result['postcondition']['measured']['moved_m'] == 0.
    assert 'release route refused' in result['error']
    assert result['placement_diagnostic']['physical_task_verdict'] is False
    if failure == 'storage':
        assert 'error' in result['placement_diagnostic']
    else:
        payload = json.loads(Path(result['placement_diagnostic']['evidence_path']).read_text())
        assert payload['capture_error'] == history.error and payload['records']
    assert rt.arm.harness._pending_model_withdrawal is pending
    assert (history.prefix_faults, history.confirmed_prefix, history.retired_obligations) == before_ledger
    trace = [json.loads(row) for row in (rt.trace.run_dir/'trace.jsonl').read_text().splitlines()]
    assert trace[-1]['result'] == result


def test_ordinary_place_at_unverified_result_cannot_gain_credit_from_diagnostic(runtime, recorded, monkeypatch):
    from cascade.agent.effects import Postcondition
    from cascade.skills import mujoco_region
    rt, (arm, history) = runtime, recorded
    original = rt.arm.raw
    arm.raw.get_state, arm.raw.disconnect = original.get_state, original.disconnect
    arm.raw.stop, arm.raw._stopped = original.stop, False
    arm.raw._cfg = rt.cfg.arm
    monkeypatch.setattr(rt.arm, '_arm', arm.raw)
    # Mock action/independent-reader outcome. Explicit area-goal retirement is
    # separately tested; diagnostics must not turn this missing proof positive.
    monkeypatch.setattr(mujoco_region, 'explicit_context', lambda *args: None)
    rt.skill_place_at = lambda **args: {'ok': True, 'at': [.2, -.12, .03]}
    pc = Postcondition(skill='place_at', kind='released', status='unverified',
                       evidence='synthetic independent reader unavailable')
    rt.effects.verify = lambda *args, **kwargs: pc
    result = rt.execute('place_at', {'x': .2, 'y': -.12, 'z': .03})
    assert result['ok'] is True and result['verified'] is False
    assert result['postcondition'] == pc.as_dict()
    diagnostic = result['placement_diagnostic']
    assert not diagnostic['physical_task_verdict'] and diagnostic['epoch'] == history.epoch
    assert history.confirmed_prefix == {'blue'} and history.prefix_faults['blue']['step'] == 49
    trace = json.loads((rt.trace.run_dir/'trace.jsonl').read_text().splitlines()[-1])
    assert trace['result'] == result
