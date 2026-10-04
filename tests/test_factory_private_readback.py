"""Owned immutable readback equivalence and failure fences; no SDK/model work."""
from dataclasses import asdict, replace
import json
import pickle
import queue

import numpy as np
import pytest

from cascade.control.fastening import FasteningFault, FasteningUpload
from cascade.control.fastening_seat import check_seating_solve, seating_solve
from cascade.sim.factory_observation import (
    _FactoryReadback, _Float32Snapshot, _RawMetadata, _contact_snapshot,
    _seating_from_snapshot, _ZeroLoadPairs, contact_records,
)
from cascade.sim.factory_owner import _RawRecord
from cascade.sim.microduck_contact_support import _ContactSnapshot, _solved_contact_snapshot
from test_factory_observation import observer_fixture
from test_factory_owner import backend_fixture, owner_fixture
from test_fastening_runtime import binding, row
from test_fastening_seating import configuration
from test_microduck_contact_support import Buffer, contact_population


def encoded(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False).encode()


def snapshot_for_row(value):
    """Synthetic archive fixture, not a reconstruction of native constraints."""
    labels = binding().collider_names
    shapes = np.asarray([(labels.index(p.collider_a), labels.index(p.collider_b)) for p in value.contacts], np.int32)
    forces = np.asarray([p.normal_force_n for p in value.contacts], np.float64)
    vectors = np.zeros((len(forces), 3, 3), np.float64)
    vectors[:, 0, 2] = forces
    vectors[:, 2, 2] = 1.
    return _ContactSnapshot(labels, len(forces), shapes.tobytes(),
        np.arange(len(forces), dtype=np.int32).tobytes(), forces.tobytes(), vectors.tobytes())


def envelope(value, stamp):
    return _FactoryReadback.capture({'solve': value, 'upload': stamp, 'contacts': snapshot_for_row(value),
        'metadata': {'exact_zero': -0., 'valid': True},
        'channel': _Float32Snapshot.capture(np.array([-0., 1.25], np.float32))})


@pytest.mark.parametrize('cone,dim', [(0, 1), (0, 3), (0, 4), (0, 6), (1, 1), (1, 3), (1, 4), (1, 6)])
def test_private_ledger_keeps_full_public_bytes_and_detached_candidate_rows(cone, dim):
    ns = contact_population(32, cone=cone, dim=dim, inactive=(0, 8, 31))
    ns.solver.mjw_data.overflow = Buffer([0], np.int32)
    ns.solver.update_contacts = lambda output: None
    pairs, public = contact_records(ns, ns.contacts)
    private_pairs, snapshot = _contact_snapshot(ns, ns.contacts, _ZeroLoadPairs())
    assert private_pairs == pairs and encoded(snapshot.factory_records()) == encoded(public)
    before = encoded(snapshot.factory_records())
    ns.contacts.force.value[:] = np.nan
    ns.solver.mjw_data.contact.pos.value[:] = np.nan
    public[1]['force_on_b_world_n'][0] = 999.
    for value in snapshot._arrays():
        with pytest.raises(ValueError): value.setflags(write=True)
    assert encoded(snapshot.factory_records()) == before
    assert encoded(pickle.loads(pickle.dumps(snapshot)).factory_records()) == before


def test_private_observer_matches_public_types_key_order_and_raw_archive():
    left, right = observer_fixture(), observer_fixture()
    stamp = FasteningUpload(2, 0, .03, 9.9, 9.91)
    receipt = {'contacts': {'count': 1, 'capacity': 4}}
    public_row, public = left.read(stamp, 10., receipt)
    private_row, private = right._read_for_owner(stamp, 10., receipt)
    assert private_row == public_row and type(private) is _FactoryReadback
    expected = encoded(public)
    record = _RawRecord(private)
    receipt['contacts']['count'] = 99
    right.scene.state.body_q.value[:] = np.nan
    right.scene.solver.mjw_data.ctrl.value[:] = np.nan
    expanded = record.expand()
    assert encoded(expanded) == expected
    assert type(expanded['solve']['contacts']) is tuple
    assert type(expanded['body_poses_xyzw']) is list
    expanded['contacts'][0]['force_on_b_world_n'][0] = 999.
    assert encoded(record.expand()) == expected


@pytest.mark.parametrize('bad', ['count', 'ids', 'index', 'force', 'vector', 'metadata', 'channel', 'subclass'])
def test_private_snapshot_rejects_malformed_or_mutable_members(bad):
    snap = snapshot_for_row(row(1))
    with pytest.raises((ValueError, TypeError)):
        if bad == 'count': replace(snap, count=snap.count+1)
        elif bad == 'ids': replace(snap, shape_ids=bytearray(snap.shape_ids))
        elif bad == 'index': replace(snap, active_indices=np.array([0, 0], np.int32).tobytes())
        elif bad == 'force': replace(snap, normal_forces=np.array([np.nan, 1.], np.float64).tobytes())
        elif bad == 'vector': replace(snap, vectors=snap.vectors[:-8])
        elif bad == 'metadata': _RawMetadata((('mutable', []),))
        elif bad == 'channel': _Float32Snapshot((2,), b'bad')
        else:
            class Fake(_FactoryReadback): pass
            original = envelope(row(1), FasteningUpload(0, 0, 0., 10., 10.))
            _RawRecord(Fake(original.entries))


def test_archive_rejects_cross_solve_and_omitted_contact_binding():
    value = row(1)
    good = envelope(value, FasteningUpload(0, 0, 0., 10., 10.))
    for changed in (replace(value, step=2), replace(value, generation=1),
                    replace(value, contacts=value.contacts[:1], solver_count=1, tool_contacts=0)):
        with pytest.raises(ValueError, match='mismatch'):
            _FactoryReadback(tuple((key, changed if key == 'solve' else item) for key, item in good.entries))


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('force', [0., 4.])
def test_seat_witnesses_match_public_force_side_and_zero_selection(reverse, force):
    bind, _ = configuration()
    ns = contact_population(1, cone=1, dim=3)
    ns.model.shape_label = bind.collider_names
    seat, nut = (bind.collider_names.index(k) for k in ('seat', 'nut'))
    ids = (nut, seat) if reverse else (seat, nut)
    ns.solver.mjc_geom_to_newton_shape.value = np.arange(len(bind.collider_names), dtype=np.int32)[None, :]
    ns.solver.mjw_data.contact.geom.value[0] = ids
    ns.contacts.rigid_contact_shape0.value[0], ns.contacts.rigid_contact_shape1.value[0] = ids
    ns.contacts.force.value[0] = [0., 0., -force, 0., 0., 0.]
    ns.solver.mjw_data.efc.force.value[0, 0] = force
    snap = _solved_contact_snapshot(ns)
    from cascade.control.fastening import SolvedPair
    value = row(1, bind=bind, contacts=(SolvedPair(*(bind.collider_names[i] for i in ids), force),),
                solver_count=1, collision_count=1, thread_contacts=0, tool_contacts=0)
    public = seating_solve(value, snap.factory_records(), bind)
    private = _seating_from_snapshot(value, snap, bind)
    assert encoded(asdict(private)) == encoded(asdict(public))
    check_seating_solve(private, bind)
    if force:
        broken = replace(private, shoulder_contacts=())
        with pytest.raises(FasteningFault, match='complete positive seat ledger'):
            check_seating_solve(broken, bind)


def test_native_private_advance_preserves_final_fence_clock_and_public_advance(monkeypatch):
    backend, events = backend_fixture(monkeypatch)
    def read(stamp, capture, coverage):
        events.append('private-observe')
        return row(1, generation=stamp.generation, captured=capture), envelope(
            row(1, generation=stamp.generation, captured=capture), stamp)
    backend.observer._read_for_owner = read
    stamp = FasteningUpload(0, 0, 0., 9.9, 9.91)
    def upload():
        events.append('fence')
        return stamp
    value, raw = backend._advance_for_owner(upload)
    assert events == ['identity', 'clear', 'collide', 'coverage', 'fence', 'solve', 'private-observe']
    assert raw.materialize()['native_clock']['step'] == value.step == 1
    assert raw.materialize()['collision_interval'] == {'before_step': 0, 'generation': 0, 'after_step': 1}
    with pytest.raises(FasteningFault, match='upload stamp'):
        backend._advance_for_owner(upload)
    assert events.count('solve') == 1


@pytest.mark.parametrize('failure', ['stale', 'encoding', 'queue', 'expansion'])
def test_private_archive_failures_do_not_publish_or_lose_a_rejected_solve(monkeypatch, failure):
    now, backend, owner = owner_fixture(active=False)
    original = backend.advance
    def advance(upload):
        value, _ = original(upload)
        packed = envelope(value, backend.stamps[-1])
        if failure == 'stale': now[0] += 1.
        return value, packed
    backend.advance = advance
    if failure == 'encoding':
        monkeypatch.setattr('cascade.sim.factory_owner.pickle.dumps', lambda *a, **k: (_ for _ in ()).throw(MemoryError('encode')))
    elif failure == 'queue':
        owner._records = queue.Queue(maxsize=1)
        owner._records.put(object())
    if failure != 'expansion':
        with pytest.raises((FasteningFault, MemoryError, queue.Full)):
            owner.cycle()
        assert owner.journal._rows[-1].step == 0
        if failure == 'stale':
            with pytest.raises(FasteningFault, match='stale'): owner.journal.read()
            archived, = owner.records()
            assert archived['solve']['step'] == 1 and archived['solve']['captured_monotonic_s'] == now[0]-1.
    else:
        owner.cycle()
        monkeypatch.setattr(_FactoryReadback, 'materialize', lambda self: (_ for _ in ()).throw(MemoryError('expand')))
        with pytest.raises(MemoryError, match='expand'): owner.records()
        with pytest.raises(FasteningFault, match='raw record expansion'): owner.records()
        with pytest.raises(FasteningFault, match='raw record expansion'): owner.journal.read()
