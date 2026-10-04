"""Compact retention preserves evidence and fences; no SDK or model execution."""
from dataclasses import asdict, FrozenInstanceError, replace
import json
import threading
import weakref

import pytest

from cascade.control import _fastening_retention as compact
from cascade.control.fastening import (FasteningController, FasteningFault,
    FasteningObservationAgeFault, SolveJournal, check_solve)
from cascade.sim.factory_owner import FactoryNewtonBackend, FactorySolveOwner
from cascade.sim.threading_verification import ThreadContract, verify_threading
from test_factory_owner import SyntheticBackend
from test_fastening_runtime import binding, limits, row
from test_fastening_seating import SeatingOwner, domain, observed


def encoded(value):
    return json.dumps(asdict(value), separators=(',', ':'), allow_nan=False).encode()


@pytest.mark.parametrize('seating', [False, True])
def test_roundtrip_exact_types_values_order_signed_zero_and_no_row_cache(seating):
    original = observed(1, loaded=True) if seating else row(1, spindle_effort_nm=-0.)
    packed = compact._PackedRecord(original)
    restored = packed.expand()
    assert type(restored) is type(original)
    assert encoded(restored) == encoded(original)
    assert all(type(a) is type(b) for a, b in zip(restored.contacts, original.contacts, strict=True))
    with pytest.raises(FrozenInstanceError): packed.step = 99
    assert packed.expand() is not restored


def test_journal_retains_only_latest_typed_row_and_cursor_batch_is_lazy(monkeypatch):
    journal = SolveJournal(binding(), _compact=True)
    first = row(0)
    ref = weakref.ref(first)
    journal.publish(first)
    journal.publish(row(1))
    del first
    assert ref() is None  # No explicit collection or GC configuration change.
    calls = []
    real = compact._PackedRecord.expand
    def tracked(self):
        calls.append(self.step)
        return real(self)
    monkeypatch.setattr(compact._PackedRecord, 'expand', tracked)
    assert journal.read()[0].step == 1
    batch = journal.read(0)
    assert len(batch) == 1 and not calls
    with pytest.raises(FrozenInstanceError): batch._records = ()
    value = batch[0]
    ref = weakref.ref(value)
    del value
    assert ref() is None and calls == [1]
    assert batch[0].step == 1 and calls == [1, 1]


@pytest.mark.parametrize('enabled', [False, True])
def test_capacity_gap_cursor_order_and_first_invalid_identity_error(enabled):
    journal = SolveJournal(binding(), capacity=3, _compact=enabled)
    for step in range(5): journal.publish(row(step))
    assert [v.step for v in journal.read(1)] == [2, 3, 4]
    assert not journal.read(4)
    with pytest.raises(FasteningFault, match='lost solves'): journal.read(0)
    with pytest.raises(ValueError): journal.read(True)
    with pytest.raises(FasteningFault, match='invalid identity/epoch/clock'):
        journal.publish(replace(row(7), binding_sha256='b'*64))
    with pytest.raises(FasteningFault, match='invalid identity/epoch/clock'):
        journal.read()


@pytest.mark.parametrize('fault', ['encode', 'decode'])
def test_codec_fault_is_sticky_and_never_returns_partial_evidence(monkeypatch, fault):
    journal = SolveJournal(binding(), _compact=True)
    journal.publish(row(0))
    def broken(*args, **kwargs): raise OSError('storage unavailable')
    if fault == 'encode':
        monkeypatch.setattr(compact.pickle, 'dumps', broken)
        operation = lambda: journal.publish(row(1))
    else:
        batch = journal.read(0)
        journal.publish(row(1))
        batch = journal.read(0)
        monkeypatch.setattr(compact._PackedRecord, 'expand', broken)
        operation = lambda: batch[0]
    with pytest.raises(FasteningFault, match='retention failed'): operation()
    with pytest.raises(FasteningFault, match='retention failed'): journal.read()
    with pytest.raises(FasteningFault, match='retention failed'): journal.publish(row(2))


def test_stop_and_publish_do_not_wait_for_reader_decode_and_fault_is_rechecked(monkeypatch):
    journal = SolveJournal(binding(), _compact=True)
    controller = FasteningController(binding(), limits(), journal, synthetic=True,
        close_owner=lambda: {'ok': True}, clock=lambda: 10.)
    journal.publish(row(0)); journal.publish(row(1))
    entered, release = threading.Event(), threading.Event()
    real = compact._PackedRecord.expand
    errors = []
    def blocked(self):
        entered.set()
        assert release.wait(2)
        return real(self)
    monkeypatch.setattr(compact._PackedRecord, 'expand', blocked)
    batch = journal.read(0)
    def read():
        try: batch[0]
        except Exception as error: errors.append(error)
    reader = threading.Thread(target=read)
    reader.start()
    try:
        assert entered.wait(1)
        assert controller.stop()['ok']
        journal.publish(row(2))
        journal.fail('owner failed while reader decoded')
    finally:
        release.set(); reader.join(2)
    assert not reader.is_alive()
    assert len(errors) == 1 and 'owner failed' in str(errors[0])


def test_decoding_does_not_refresh_capture_or_hide_expiry(monkeypatch):
    journal = SolveJournal(binding(), _compact=True)
    journal.publish(row(0)); journal.publish(row(1, captured=10.))
    now = [10.]
    real = compact._PackedRecord.expand
    def delayed(self):
        now[0] = 10.201
        return real(self)
    monkeypatch.setattr(compact._PackedRecord, 'expand', delayed)
    decoded = journal.read(0)[0]
    assert decoded.captured_monotonic_s == 10.
    with pytest.raises(FasteningObservationAgeFault) as error:
        check_solve(decoded, binding(), limits(), now[0])
    assert error.value.observation_age['age_s'] > .2


def test_original_safety_failure_precedes_packing(monkeypatch):
    journal = SolveJournal(binding(), _compact=True)
    controller = FasteningController(binding(), limits(), journal, synthetic=True,
        close_owner=lambda: {'ok': True}, clock=lambda: 10.)
    def forbidden(*args, **kwargs): raise AssertionError('must reject before packing')
    monkeypatch.setattr(compact.pickle, 'dumps', forbidden)
    with pytest.raises(FasteningFault, match='joint speed'):
        controller.accept_solve(row(joint_velocity_rad_s=(.81,)))
    with pytest.raises(FasteningFault, match='joint speed'): journal.read()


def test_unknown_nested_object_or_subclass_is_refused_before_pickle(monkeypatch):
    calls = []
    monkeypatch.setattr(compact.pickle, 'dumps', lambda *a, **k: calls.append(1))
    class Unexpected:
        def __reduce__(self): raise AssertionError('must not execute')
    value = row(1)
    object.__setattr__(value, 'contacts', (Unexpected(),))
    with pytest.raises(TypeError): compact._PackedRecord(value)
    class Subclass(type(row())): pass
    with pytest.raises(TypeError): compact._PackedRecord(Subclass(**vars(row())))
    assert not calls


@pytest.mark.parametrize('mode', ['positive', 'pitch', 'gap', 'identity'])
def test_threading_sequence_preserves_full_existing_verifier_and_slice(mode):
    journal = SolveJournal(binding(), _compact=True)
    history = compact._ThreadHistory(200, journal._check_error, journal.fail)
    values = [row(i, turns=i/120).thread_sample(binding()) for i in range(121)]
    if mode == 'pitch': values = [replace(v, fastener_position_m=(0., 0., .069)) for v in values]
    if mode == 'gap': values[-1] = replace(values[-1], time_s=5.)
    if mode == 'identity': values[-1] = replace(values[-1], epoch='other')
    for value in values: history.append(value)
    assert [encoded(v) for v in history] == [encoded(v) for v in values]
    contract = ThreadContract(.0025)
    assert verify_threading(history, contract) == verify_threading(values, contract)
    assert verify_threading(history[:100], contract) == verify_threading(values[:100], contract)
    assert isinstance(history[:100], compact._RetainedBatch)


def test_thread_history_capacity_fault_is_sticky_without_losing_prefix():
    journal = SolveJournal(binding(), _compact=True)
    history = compact._ThreadHistory(3, journal._check_error, journal.fail)
    for step in range(3): history.append(row(step).thread_sample(binding()))
    with pytest.raises(FasteningFault, match='OverflowError'):
        history.append(row(3).thread_sample(binding()))
    assert len(history) == 3
    with pytest.raises(FasteningFault): history[0]


def test_only_exact_native_owner_opts_into_compact_storage():
    synthetic = SyntheticBackend([10.])
    owner = FactorySolveOwner(synthetic)
    assert not owner.journal._compact and owner.controller._new_thread_history() == []
    native = object.__new__(FactoryNewtonBackend)
    native.binding, native.limits = binding(), limits()
    owner = FactorySolveOwner(native)
    assert owner.journal._compact
    assert type(owner.controller._new_thread_history()) is compact._ThreadHistory
    class Derived(FactoryNewtonBackend): pass
    derived = object.__new__(Derived)
    derived.binding, derived.limits = binding(), limits()
    assert not FactorySolveOwner(derived).journal._compact


@pytest.mark.parametrize('mode', ['healthy', 'late_command', 'lost_support', 'late_rest'])
def test_seating_real_domain_preserves_terminal_result_with_compact_history(mode):
    ordinary, packed = SeatingOwner(mode), SeatingOwner(mode)
    journal = SolveJournal(packed.binding, _compact=True)
    packed._new_thread_history = lambda: compact._ThreadHistory(5000, journal._check_error, journal.fail)
    assert domain(packed).execute('seat_fastener', {}) == domain(ordinary).execute('seat_fastener', {})
    assert packed.stopped and ordinary.stopped
