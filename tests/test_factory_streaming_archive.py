"""Private native retention boundaries using CPU fixtures, without an SDK."""
import gc
import json
import queue
import weakref

import pytest

from cascade.apps.factory_runtime import RecordedFactoryDomain
from cascade.control.fastening import FasteningFault, FasteningUpload
from cascade.sim import factory_owner as module
from test_factory_private_readback import envelope
from test_fastening_runtime import binding, limits, row


def native_owner(count=3):
    backend = object.__new__(module.FactoryNewtonBackend)
    backend.binding, backend.limits = binding(), limits()
    values = [row(step) for step in range(1, count + 1)]
    raw = [envelope(value, FasteningUpload(0, value.step - 1, 0., 9.9, 9.91))
           for value in values]
    remaining = iter(zip(values, raw, strict=True))
    # Exercise the exact owner's archive/accept path; this is no SDK solve.
    backend._advance_for_owner = lambda upload: next(remaining)
    owner = module.FactorySolveOwner(backend, clock=lambda: 10.)
    return owner, raw


def test_native_queue_holds_untracked_bytes_and_public_records_remains_detached_list():
    owner, raw = native_owner()
    for _ in raw:
        owner.cycle()
    assert all(type(value) is bytes and not gc.is_tracked(value)
               for value in owner._records.queue)
    expected = [value.materialize() for value in raw]
    actual = owner.records()
    assert type(actual) is list and actual == expected and owner.records() == []
    actual[0]['contacts'][0]['force_on_b_world_n'][0] = 99.
    assert raw[0].materialize() == expected[0]


def test_native_stale_solve_remains_archived_but_unaccepted():
    owner, raw = native_owner(1)
    owner.clock = lambda: 11.
    owner.controller.clock = lambda: 11.
    owner.controller.guard.clock = lambda: 11.
    with pytest.raises(FasteningFault, match='stale'):
        owner.cycle()
    assert len(owner.journal._rows) == 0
    assert owner.records() == [raw[0].materialize()]


def test_native_full_raw_queue_prevents_journal_acceptance():
    owner, raw = native_owner(1)
    owner._records = queue.Queue(maxsize=1)
    owner._records.put(module._pack_native_record(raw[0]))
    with pytest.raises(queue.Full):
        owner.cycle()
    assert len(owner.journal._rows) == 0 and owner._records.qsize() == 1


def test_native_json_drain_does_not_keep_expanded_history(tmp_path, monkeypatch):
    owner, raw = native_owner(5)
    for _ in raw:
        owner.cycle()
    expected = ''.join(json.dumps(value.materialize(), sort_keys=True, allow_nan=False) + '\n'
                       for value in raw)
    live = []
    real = module._expand_native_record
    class TrackedDict(dict):
        pass
    def expand(payload):
        assert sum(ref() is not None for ref in live) <= 1
        value = TrackedDict(real(payload))
        live.append(weakref.ref(value))
        return value
    monkeypatch.setattr(module, '_expand_native_record', expand)
    monkeypatch.setattr(owner, 'records', lambda: pytest.fail('public eager drain used'))
    domain = RecordedFactoryDomain(owner, tmp_path, domain_id='fastening')
    domain.flush_records()
    assert (tmp_path / 'solves.jsonl').read_text() == expected
    assert len(live) == 5 and all(ref() is None for ref in live)


def test_decode_failure_preserves_written_prefix_and_refuses_later_evidence(tmp_path, monkeypatch):
    owner, raw = native_owner()
    for _ in raw:
        owner.cycle()
    real = module._expand_native_record
    calls = []
    def expand(payload):
        calls.append(1)
        if len(calls) == 2:
            raise MemoryError('second snapshot')
        return real(payload)
    monkeypatch.setattr(module, '_expand_native_record', expand)
    domain = RecordedFactoryDomain(owner, tmp_path, domain_id='fastening')
    with pytest.raises(MemoryError, match='second snapshot'):
        domain.flush_records()
    assert len((tmp_path / 'solves.jsonl').read_text().splitlines()) == 1
    assert owner._records.qsize() == 1  # Undecoded tail remains in the bounded queue.
    assert owner.controller.guard._latched
    with pytest.raises(FasteningFault, match='raw record expansion'):
        owner.journal.read()
    with pytest.raises(FasteningFault, match='raw record expansion'):
        owner.records()
    with pytest.raises(FasteningFault, match='new owned epoch'):
        domain.reset_stop()


def test_private_flush_finishes_initial_prefix_while_producer_replenishes(tmp_path, monkeypatch):
    owner, raw = native_owner()
    for _ in raw:
        owner.cycle()
    payload = module._pack_native_record(raw[-1])
    real = module._expand_native_record
    calls = []
    def replenish(value):
        calls.append(1)
        assert len(calls) <= 3, 'new rows extended the current drain'
        owner._records.put_nowait(payload)
        return real(value)
    domain = RecordedFactoryDomain(owner, tmp_path, domain_id='fastening')
    monkeypatch.setattr(module, '_expand_native_record', replenish)
    domain.flush_records()
    assert len(calls) == owner._records.qsize() == 3
    assert len((tmp_path / 'solves.jsonl').read_text().splitlines()) == 3
    monkeypatch.setattr(module, '_expand_native_record', real)
    domain.flush_records()
    assert owner._records.empty()
    assert len((tmp_path / 'solves.jsonl').read_text().splitlines()) == 6


def test_writer_failure_leaves_undecoded_tail_and_sticky_domain_fault(tmp_path, monkeypatch):
    owner, raw = native_owner()
    for _ in raw:
        owner.cycle()
    real = json.dumps
    def encode(value, **kwargs):
        if value['solve']['step'] == 2:
            raise OSError('writer encoding failed')
        return real(value, **kwargs)
    domain = RecordedFactoryDomain(owner, tmp_path, domain_id='fastening')
    monkeypatch.setattr(json, 'dumps', encode)
    with pytest.raises(OSError, match='writer encoding'):
        domain.flush_records()
    assert owner._records.qsize() == 1 and owner.controller.guard._latched
    assert len((tmp_path / 'solves.jsonl').read_text().splitlines()) == 1
    assert domain.execute('turn_screw', {'turns': 1., 'direction': 'tighten'})['execution_ok'] is False
