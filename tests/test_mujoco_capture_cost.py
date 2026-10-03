"""Recorded-channel replay; no integration, renderer or new physical proof."""
from collections import deque
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.sim import mujoco_placement as placement
from test_mujoco_withdrawal import model_runtime as model_runtime  # noqa: F401

FIXTURE = json.loads((Path(__file__).parent/'fixtures/placement_capture_07080fa.json').read_text())


@pytest.fixture
def replay(model_runtime):
    history = model_runtime.arm.raw.world.placement_history
    record = deepcopy(FIXTURE['row'])
    # The fixture uses the same compiled native geometry as the saved row.
    # Only solver outputs below are replayed doubles, explicitly not new truth.
    assert history.identity == record['model_sha256']
    mj = history.mj
    scratch = mj.MjData(history.model)
    scratch.qpos[:] = record['solve_qpos']
    mj.mj_kinematics(history.model, scratch)
    contacts = [SimpleNamespace(efc_address=-1) for _ in range(record['native_contact_count'])]
    forces = {}
    for c in record['solved_contacts']:
        contacts[c['index']] = SimpleNamespace(efc_address=c['efc_address'],
            geom1=c['geom'][0], geom2=c['geom'][1], pos=np.asarray(c['position_m']),
            frame=np.asarray(c['frame_a_to_b']).reshape(-1))
        forces[c['index']] = np.asarray(c['force_torque_contact'])
    data = SimpleNamespace(qpos=np.asarray(record['final_qpos']), qvel=np.asarray(record['final_qvel']),
        ctrl=np.asarray(record['ctrl']), time=record['final_state_time_s'], contact=contacts,
        ncon=record['native_contact_count'], geom_xpos=scratch.geom_xpos.copy(),
        geom_xmat=scratch.geom_xmat.copy(), warning=[SimpleNamespace(number=0) for _ in range(8)])
    history.world.data = history.data = data

    class RecordedContacts:
        def __getattr__(self, name):
            return getattr(mj, name)

        def mj_contactForce(self, model, observed, index, output):
            assert model is history.model and observed is data
            output[:] = forces[index]

    history.mj = RecordedContacts()
    before = {'qpos': np.asarray(record['solve_qpos']), 'qvel': np.asarray(record['solve_qvel']),
              'time_s': record['time_s']}

    def reset():
        history.rows = deque(maxlen=256)
        history.error = None
        history.epoch = record['epoch']
        history.step = record['step']-record['batch_steps']
        history.last_final_time = record['final_state_time_s']-record['batch_steps']*history.model.opt.timestep
        history.closed_seen = True
        history.open_since = record['open_since_s']

    reset()
    return history, before, record['batch_steps'], reset


def test_one_full_fingerprint_per_capture_matches_exact_legacy_journal(replay, monkeypatch):
    history, before, count, reset = replay
    namespace = dict(vars(placement))
    exec(compile(FIXTURE['capture_source'], '<capture at 07080fa>', 'exec'), namespace)
    legacy_capture = namespace['capture']
    original, calls = placement.model_digest, []

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(placement, 'model_digest', counted)
    initial = placement.state_digest(history.data)
    with history.world.lock:
        legacy_capture(history, count, before)
    assert history.error is None
    assert len(calls) == 3
    old = deepcopy(list(history.rows))
    old_counters = (history.step, history.last_final_time, history.open_since)
    reset()
    calls.clear()
    history.capture(count, before)
    assert history.error is None
    assert len(calls) == 1
    assert list(history.rows) == old
    assert (history.step, history.last_final_time, history.open_since) == old_counters
    assert placement.state_digest(history.data) == initial
    assert old[0]['solved_contacts'] == FIXTURE['row']['solved_contacts']


@pytest.mark.parametrize('entry', ['capture', 'geometry', 'read'])
def test_model_change_is_refused_at_the_next_admission(replay, entry):
    history, before, count, _ = replay
    history.capture(count, before)
    assert history.error is None and len(history.rows) == 1
    rows = deepcopy(list(history.rows))
    old = float(history.model.body_mass[1])
    try:
        history.model.body_mass[1] = old+.001
        if entry == 'capture':
            history.capture(count, before)
            assert 'model identity changed' in history.error
        else:
            with pytest.raises(ValueError, match='model identity changed'):
                getattr(history, entry)(*(['red cube'] if entry == 'read' else []))
        assert list(history.rows) == rows
    finally:
        history.model.body_mass[1] = old


def test_public_geometry_and_read_keep_their_own_full_checks(replay, monkeypatch):
    history, before, count, _ = replay
    history.capture(count, before)
    original, calls = placement.model_digest, []

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(placement, 'model_digest', counted)
    history.geometry()
    history.read('red cube')
    assert len(calls) == 2


def test_private_fk_work_stays_in_the_world_lock(replay, monkeypatch):
    history, before, count, _ = replay
    original, calls = history._validated_geometry, []

    def checked(*args):
        assert history.world.lock._is_owned()
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(history, '_validated_geometry', checked)
    history.capture(count, before)
    assert history.error is None and len(calls) == 2
