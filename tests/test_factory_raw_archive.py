"""Private raw retention only: no SDK, physical solves or new admission."""
from dataclasses import asdict
import json
import math
import pickle
import queue
import threading

import pytest

from cascade.apps.factory_runtime import RecordedFactoryDomain
from cascade.control.fastening import FasteningFault
from cascade.sim import factory_owner
from test_factory_owner import owner_fixture


def capture_once(owner, backend, raw):
    original = backend.advance

    def advance(upload):
        state, _ = original(upload)
        return state, raw

    backend.advance = advance
    owner.cycle()


@pytest.mark.parametrize("mutation", ["root", "nested"])
def test_backend_raw_mutation_cannot_rewrite_an_archived_capture(mutation):
    _, backend, owner = owner_fixture(active=False)
    raw = {"capture": 10.001, "step": 1, "contacts": [{"force": 1.25}]}
    expected = json.loads(json.dumps(raw))
    capture_once(owner, backend, raw)
    if mutation == "root":
        raw["capture"] = 900.
    else:
        raw["contacts"][0]["force"] = 900.
    assert owner.records() == [expected]
    assert owner.records() == []


def test_types_aliases_signed_zero_and_original_clock_roundtrip():
    _, backend, owner = owner_fixture(active=False)
    shared = [1.25]
    raw = {"capture": 10.001, "shape": (None, True, 2, -0., "µ", shared),
           "other": [shared]}
    capture_once(owner, backend, raw)
    restored, = owner.records()
    assert restored == raw and restored is not raw
    assert type(restored["shape"]) is tuple and type(restored["other"]) is list
    assert restored["shape"][-1] is restored["other"][0]
    assert math.copysign(1., restored["shape"][3]) == -1.
    assert restored["capture"] == 10.001
    assert owner.journal.read()[-1] is owner._row


def test_original_python_and_persisted_json_shape(tmp_path):
    _, backend, owner = owner_fixture(active=False)
    from test_fastening_runtime import row
    raw = {"solve": asdict(row()), "contacts": [{"point": [0., .2, .3]}]}
    capture_once(owner, backend, raw)
    domain = RecordedFactoryDomain(owner, tmp_path, domain_id="fastening")
    domain.flush_records()
    assert (tmp_path / "solves.jsonl").read_text() == json.dumps(raw, sort_keys=True, allow_nan=False) + "\n"


def test_nonfinite_raw_still_fails_existing_json_persistence_and_stays_sticky(tmp_path):
    _, backend, owner = owner_fixture(active=False)
    capture_once(owner, backend, {"force": float("nan")})
    domain = RecordedFactoryDomain(owner, tmp_path, domain_id="fastening")
    with pytest.raises(ValueError):
        domain.flush_records()
    with pytest.raises(FasteningFault, match="new owned epoch"):
        domain.reset_stop()
    assert not domain.close()["ok"]
    assert not domain.execute("turn_screw", {"turns": 1., "direction": "tighten"})["ok"]


class Unprintable(BaseException):
    def __str__(self):
        raise RuntimeError("formatting failed")


@pytest.mark.parametrize("error", [RuntimeError("codec failure"), KeyboardInterrupt("cancel"), Unprintable()])
def test_encoder_fault_prevents_accept_and_reaches_existing_owner_finalizer(monkeypatch, error):
    now, backend, owner = owner_fixture(active=False)
    accepted = owner.journal.read()

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(pickle, "dumps", fail)
    owner._start = now[0]
    owner._run()
    assert owner.controller.guard._latched and backend.step == 1 and backend.effort == 0.
    assert tuple(owner.journal._rows) == accepted
    assert owner.records() == []
    assert owner._error.startswith(type(error).__name__ + ":")
    assert not owner.close()["ok"]


@pytest.mark.parametrize("error", [RuntimeError("decode failed"), KeyboardInterrupt("cancel"), Unprintable()])
def test_expansion_failure_latches_before_formatting_and_never_becomes_clean_drain(monkeypatch, error, tmp_path):
    _, backend, owner = owner_fixture(active=False)
    capture_once(owner, backend, {"step": 1})
    owner._zero()  # Make a failed close meaningful even without a running worker.

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(pickle, "loads", fail)
    generation = owner.controller.generation
    with pytest.raises(type(error)) as exc:
        owner.records()
    assert exc.value is error
    assert owner.controller.guard._latched and owner.controller.generation == generation + 1
    with pytest.raises(FasteningFault, match="raw record expansion"):
        owner.journal.read()
    assert not owner.close()["ok"]
    with pytest.raises(FasteningFault, match="raw record expansion"):
        owner.records()
    domain = RecordedFactoryDomain(owner, tmp_path, domain_id="fastening")
    with pytest.raises(FasteningFault, match="raw record expansion"):
        domain.flush_records()
    assert not domain.close()["ok"]
    with pytest.raises(FasteningFault, match="new owned epoch"):
        domain.reset_stop()


def test_queue_before_accept_retains_original_stale_capture(monkeypatch):
    _, backend, owner = owner_fixture(active=False)
    original = pickle.dumps
    calls = []

    def encode(value, **kwargs):
        calls.append("encode")
        return original(value, **kwargs)

    def reject(state):
        calls.append("accept")
        assert owner._records.qsize() == 1
        raise FasteningFault("stale solve")

    monkeypatch.setattr(pickle, "dumps", encode)
    monkeypatch.setattr(owner.controller, "accept_solve", reject)
    with pytest.raises(FasteningFault, match="stale solve"):
        capture_once(owner, backend, {"capture": 1., "step": 1})
    assert calls == ["encode", "accept"]
    assert owner.records() == [{"capture": 1., "step": 1}]


def test_encoding_delay_is_included_in_the_existing_age_veto(monkeypatch):
    now, backend, owner = owner_fixture(active=False)
    original = pickle.dumps

    def delayed(value, **kwargs):
        now[0] += backend.limits.max_observation_age_s + .001
        return original(value, **kwargs)

    monkeypatch.setattr(pickle, "dumps", delayed)
    with pytest.raises(FasteningFault, match="stale or future"):
        owner.cycle()
    assert owner._row.step == 0  # Rejected solve never becomes the next baseline.
    assert owner.records() == [{"step": 1, "generation": 0, "effort": 0.}]


def test_original_capacity_overflow_stops_without_accepting_or_dropping_earlier_rows():
    now, backend, owner = owner_fixture(active=False)
    assert owner._records.maxsize == 20000
    owner._records = queue.Queue(maxsize=3)
    owner._start = now[0]
    owner._run()
    assert backend.step == 4 and owner._row.step == 3
    assert owner._error.startswith("Full:") and owner.controller.guard._latched
    assert backend.effort == 0. and not owner.close()["ok"]
    assert [r["step"] for r in owner.records()] == [1, 2, 3]


def test_priority_stop_reaches_real_controller_while_owner_encoder_is_blocked(monkeypatch):
    now, backend, owner = owner_fixture()
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
    original = pickle.dumps
    stop_ack = []

    def held(value, **kwargs):
        entered.set()
        assert release.wait(2.)
        # Finish this one in-flight row, then observe the original lifetime fault.
        now[0] += owner.max_wall_s
        return original(value, **kwargs)

    def stop():
        stop_ack.append(owner.controller.stop())
        stopped.set()

    monkeypatch.setattr(pickle, "dumps", held)
    owner.start()
    stopper = threading.Thread(target=stop)
    try:
        assert entered.wait(1.)
        stopper.start()
        assert stopped.wait(1.), "stop waited for raw encoding"
        assert owner._thread.is_alive() and not release.is_set()
        assert backend.effort == .03  # Stop ACK is not physical zero.
    finally:
        release.set()
        stopper.join(2.)
        owner.close()
    assert not owner._thread.is_alive() and backend.effort == 0.
    assert not stop_ack[0]["physical_stop_verified"]
    assert owner.records()[0]["generation"] < stop_ack[0]["generation"]


def test_external_encoded_bytes_cannot_supply_a_pickle_stream():
    # No deserialize-bytes API: even bytes used as a (non-production) value
    # are wrapped by the constructor and expanded only to those same bytes.
    supplied = b"not a valid pickle stream"
    record = factory_owner._RawRecord(supplied)
    assert record.expand() == supplied
    assert not any(hasattr(record, key) for key in ("load", "loads", "from_bytes", "payload"))


@pytest.mark.parametrize("held_phase", ["solve", "error_formatter"])
def test_first_retained_decoder_fault_survives_concurrent_owner_failure(monkeypatch, held_phase):
    _, backend, owner = owner_fixture(active=False)
    entered, release = threading.Event(), threading.Event()

    class SlowFormatError(RuntimeError):
        def __str__(self):
            entered.set()
            assert release.wait(2.)
            return "later owner fault"

    if held_phase == "solve":
        def held():
            if backend.step == 1:
                entered.set()
                assert release.wait(2.)
        backend.solve_hook = held
    else:
        original_advance = backend.advance
        def advance(upload):
            if backend.step == 1:
                raise SlowFormatError()
            return original_advance(upload)
        backend.advance = advance

    error = MemoryError("first decoder failure")
    def fail(*args, **kwargs):
        raise error

    owner.start()
    try:
        assert entered.wait(1.)
        monkeypatch.setattr(pickle, "loads", fail)
        with pytest.raises(MemoryError) as exc:
            owner.records()
        assert exc.value is error
        first = owner._error
        assert first == "raw record expansion failed: MemoryError: first decoder failure"
        # A formatter waits outside both stop and first-error assignment locks.
        assert owner.controller.stop()["latched"] and not release.is_set()
    finally:
        release.set()
        owner._thread.join(2.)
        closure = owner.close()
    assert not owner._thread.is_alive() and not closure["ok"] and backend.effort == 0.
    assert owner._error == closure["error"] == first
    with pytest.raises(FasteningFault) as exc:
        owner.journal.read()
    assert str(exc.value) == first
    with pytest.raises(FasteningFault, match="raw record expansion"):
        owner.records()
