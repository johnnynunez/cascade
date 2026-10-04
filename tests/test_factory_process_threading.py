"""Host historical evidence only: no model construction or new physical gates."""
import math
import struct
import threading
from dataclasses import replace

import pytest
from test_fastening_runtime import binding, row

from cascade.control.fastening import FasteningFault
from cascade.sim import factory_process_threading as retention
from cascade.sim.factory_process_wire import WireFault
from cascade.sim.threading_verification import (
    ThreadContract,
    ThreadSample,
    verify_threading,
)


class Authority:
    def __init__(self):
        self.fault = None

    def check(self):
        if self.fault:
            raise FasteningFault(self.fault)

    def fail(self, reason):
        self.fault = self.fault or reason


def archive(**kwargs):
    authority = Authority()
    return retention._ThreadArchive(authority.check, authority.fail, **kwargs), authority


def sample(step):
    return row(step).thread_sample(binding())


def test_complete_schema_preserves_values_types_float_bits_and_detachment():
    value = replace(sample(1), fastener_position_m=(-0., 0., 1.), time_s=-0.)
    payload = retention._pack_sample(value)
    restored = retention._unpack_sample(payload)
    assert type(restored) is ThreadSample and vars(restored) == vars(value)
    assert struct.pack('d', restored.time_s) == struct.pack('d', -0.)
    assert struct.pack('d', restored.fastener_position_m[0]) == struct.pack('d', -0.)
    object.__setattr__(value, 'time_s', 1000.)
    assert retention._pack_sample(restored) == payload


def test_worst_bound_identity_and_native_vectors_fit_2048_bytes():
    value = replace(sample(1), epoch='x'*512, fastener_id='y'*512, fixture_id='z'*512,
                    step=2**63-1, thread_contacts=2**63-1, tool_contacts=2**63-1)
    payload = retention._pack_sample(value)
    assert len(payload) <= 2048
    assert vars(retention._unpack_sample(payload)) == vars(value)


@pytest.mark.parametrize('changes', [
    {'step': True}, {'step': 2**63}, {'thread_contacts': -1},
    {'epoch': 'a'*513}, {'time_s': math.inf},
    {'fixture_position_m': (0., 0.)}, {'fixture_position_m': (0., False, 0.)},
])
def test_invalid_schema_refuses_without_executing_a_reducer(changes):
    with pytest.raises(WireFault):
        retention._pack_sample(replace(sample(1), **changes))


def test_no_new_freshness_gate_or_changed_physical_verifier_for_old_history():
    rows = [sample(i) for i in range(1, 11)]
    contract = ThreadContract(pitch_m=.0025)
    expected = verify_threading(rows, contract)
    retained, authority = archive()
    for value in rows:
        retained.append(value)
    assert verify_threading(retained, contract) == expected
    with retained[:5] as first, retained[5:] as last:
        assert verify_threading(first, contract) == verify_threading(rows[:5], contract)
        assert list(first) + list(last) == rows
    assert authority.fault is None
    retained.close()
    retained.require_drained()


def test_finite_views_do_not_copy_bytes_or_follow_later_appends():
    retained, _ = archive()
    for step in range(1, 4):
        retained.append(sample(step))
    initial_bytes = retained.retained['bytes']
    with retained[:] as view, view[::-1] as reverse:
        assert retained.retained['bytes'] == initial_bytes
        retained.append(sample(4))
        assert [s.step for s in view] == [1, 2, 3]
        assert [s.step for s in reverse] == [3, 2, 1]
    retained.close()
    retained.require_drained()


@pytest.mark.parametrize('limit', ['rows', 'bytes'])
def test_row_or_byte_limit_faults_before_overwrite_and_remains_after_cleanup(limit):
    kwargs = {'capacity': 1} if limit == 'rows' else {'byte_capacity': len(retention._pack_sample(sample(1)))}
    retained, authority = archive(**kwargs)
    retained.append(sample(1))
    before = retained.retained.copy()
    with pytest.raises(FasteningFault, match='OverflowError'):
        retained.append(sample(2))
    assert retained.retained == before and authority.fault
    retained.close()
    assert retained.retained == {'rows': 0, 'bytes': 0, 'views': 0, 'active_readers': 0}
    with pytest.raises(FasteningFault, match='OverflowError'):
        retained.require_drained()


def test_empty_views_also_consume_global_slots_and_can_clean_up_after_failure():
    retained, _ = archive(view_capacity=2)
    first, second = retained[:], retained[:]
    with pytest.raises(FasteningFault, match='OverflowError'):
        retained[:]
    retained.close()
    first.close()
    second.close()
    second.close()
    assert retained.retained == {'rows': 0, 'bytes': 0, 'views': 0, 'active_readers': 0}


@pytest.mark.parametrize('failure', ['fault', 'close_view', 'close_archive'])
def test_no_delivery_after_fault_or_close_during_decode(monkeypatch, failure):
    retained, authority = archive()
    retained.append(sample(1))
    view = retained[:]
    original = retention._unpack_sample

    def decode(payload):
        value = original(payload)
        if failure == 'fault':
            authority.fail('child refusal during decode')
        elif failure == 'close_view':
            view.close()
        else:
            retained.close()
        return value

    monkeypatch.setattr(retention, '_unpack_sample', decode)
    with pytest.raises(FasteningFault, match='refusal|closed'):
        view[0]
    view.close()
    retained.close()
    assert retained.retained == {'rows': 0, 'bytes': 0, 'views': 0, 'active_readers': 0}


def test_active_decode_remains_charged_until_exit_and_reader_cap_is_sticky(monkeypatch):
    retained, _ = archive(reader_capacity=1)
    retained.append(sample(1))
    entered, released = threading.Event(), threading.Event()
    original, failures = retention._unpack_sample, []

    def decode(payload):
        entered.set()
        assert released.wait(1.)
        return original(payload)

    def read():
        try:
            retained[0]
        except FasteningFault as error:
            failures.append(str(error))

    monkeypatch.setattr(retention, '_unpack_sample', decode)
    worker = threading.Thread(target=read)
    worker.start()
    try:
        assert entered.wait(1.)
        with pytest.raises(FasteningFault, match='OverflowError'):
            retained[0]
        retained.close()
        assert retained.retained['bytes'] > 0 and retained.retained['active_readers'] == 1
    finally:
        released.set()
        worker.join(1.)
    assert not worker.is_alive() and len(failures) == 1
    assert retained.retained == {'rows': 0, 'bytes': 0, 'views': 0, 'active_readers': 0}
