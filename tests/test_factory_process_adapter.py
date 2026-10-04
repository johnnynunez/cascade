"""Real CPU subprocess/socket/owner/domain wiring; all solves are synthetic."""
import json
import multiprocessing
import socket
import threading
import time
from concurrent.futures import Future
from dataclasses import replace

import numpy as np
import pytest
from test_factory_process_snapshot import fixture
from test_fastening_runtime import binding, limits, row

from cascade.control.fastening import FasteningFault
from cascade.sim.factory_observation import _FactoryReadback, _RawMetadata
from cascade.sim.factory_owner import FactorySolveOwner, _Request
from cascade.sim.factory_process_adapter import (
    _Caps,
    _ChildService,
    _FileArchive,
    _HostSession,
    _OwnerTransport,
    _ProcessDomain,
)
from cascade.sim.factory_process_protocol import RawOutbox
from cascade.sim.factory_process_wire import WireFault
from cascade.sim.microduck_contact_support import _ContactSnapshot


class _Synthetic:
    synthetic = True

    def __init__(self, *, stale_step=None):
        self.binding, self.limits = binding(), limits()
        layout, template = fixture(inactive=False)
        self.layout = replace(layout, epoch="b"*32)
        self.template = dict(template.entries)
        contacts = self.template["contacts"]
        self.template["contacts"] = _ContactSnapshot(contacts.labels, 2,
            np.array([[0, 1], [0, 2]], np.int32).tobytes(), contacts.active_indices,
            contacts.normal_forces, contacts.vectors)
        self.step, self.turns = 0, 0.
        self.stale_step = stale_step

    def enter_owner(self):
        pass

    def upload_zero(self, effort):
        assert effort == 0.

    def upload(self, target, effort):
        assert target == (0.,) and effort == .01

    def plan(self, previous):
        return (0.,), (previous.geometry_min_m, previous.geometry_max_m), .01

    def advance(self, upload):
        stamp = upload()
        # Explicit deterministic synthetic plant data, not simulated physics.
        time.sleep(.002)
        self.step += 1
        if stamp.effort_nm:
            self.turns += .01
        captured = time.monotonic() - (.3 if self.step == self.stale_step else 0.)
        value = replace(row(self.step, turns=self.turns, generation=stamp.generation, captured=captured),
                        epoch=self.layout.epoch, commanded_spindle_effort_nm=stamp.effort_nm,
                        spindle_effort_nm=stamp.effort_nm)
        data = dict(self.template, solve=value, upload=stamp)
        data["native_clock"] = _RawMetadata.capture({"step": self.step,
            "time_s": float(np.float32(value.simulation_time_s)),
            "timestep_s": float(np.float32(self.binding.dt_s)),
            "interval_clock_s": value.simulation_time_s, "cuda_graph": False})
        data["collision_interval"] = _RawMetadata.capture({"before_step": self.step-1,
            "generation": stamp.generation, "after_step": self.step})
        return value, _FactoryReadback(tuple(data.items()))


def _child(sockets, identity, caps, deadline, receipt, stale_step, close_failure):
    backend = _Synthetic(stale_step=stale_step)
    transport = _OwnerTransport(backend.layout, identity, caps)
    owner = FactorySolveOwner(backend, max_wall_s=4., _process_transport=transport)
    if close_failure is not None:
        def fail(*_args, **_kwargs):
            raise RuntimeError("synthetic close failure")
        if close_failure == "guard":
            owner.controller.guard.close = fail
        else:
            owner.close = fail
    result = _ChildService(owner, transport, sockets, deadline=deadline).serve()
    with open(receipt, "x") as stream:
        json.dump(result, stream)


def _start(tmp_path, *, stale_step=None, raw_sink=None, outcome_sink=None, close_failure=None):
    backend = _Synthetic()
    caps = replace(_Caps(), raw_slots=512, history_slots=512)
    identity = ("a"*32, backend.layout.epoch, backend.binding.sha256)
    pairs = tuple(socket.socketpair() for _ in range(3))
    parent, child = tuple(p[0] for p in pairs), tuple(p[1] for p in pairs)
    deadline = time.monotonic()+6.
    receipt = tmp_path/"child.json"
    process = multiprocessing.get_context("spawn").Process(target=_child,
        args=(child, identity, caps, deadline, receipt, stale_step, close_failure))
    process.start()
    for endpoint in child:
        endpoint.close()
    raw, outcomes = [], []
    try:
        session = _HostSession(parent, layout=backend.layout, identity=identity,
            binding=backend.binding, limits=backend.limits, caps=caps, deadline=deadline,
            synthetic=True, persist_raw=raw_sink or (lambda seq, record: raw.append((seq, record.materialize()))),
            persist_outcome=outcome_sink or (lambda seq, outcome: outcomes.append((seq, outcome))))
    except BaseException:
        for endpoint in parent:
            endpoint.close()
        process.join(7)
        if process.is_alive():
            process.terminate()
            process.join(2)
        raise
    return process, session, raw, outcomes, receipt


def _finish(process, session):
    try:
        return session.close()
    finally:
        process.join(7)
        if process.is_alive():
            process.terminate()  # Only this test's owned child, never foreign PID.
            process.join(2)
        assert not process.is_alive()


def test_real_owner_child_and_domain_start_turn_stop_rest_archive_without_physical_credit(tmp_path):
    process, session, raw, outcomes, receipt = _start(tmp_path)
    domain = _ProcessDomain(session, controller_id="synthetic-process")
    try:
        ready = domain.ready(timeout_s=2.)
        assert ready["ready"] and not ready["physical_task_admission"]
        reset = domain.reset_stop()
        assert reset["ok"]
        result = domain.execute("turn_screw", {"turns": 1., "direction": "tighten"})
        assert result["ok"], json.dumps(result, sort_keys=True)
        assert result["synthetic"] and not result["physical_stop_verified"]
        assert result["rest"]["window_sim_s"] >= .5-1e-9
    finally:
        closed = _finish(process, session)
    assert closed["ok"] and not closed["physical_stop_verified"]
    assert process.exitcode == 0 and json.loads(receipt.read_text())["ok"]
    assert len(raw) == len(outcomes) > 100
    assert [seq for seq, _ in raw] == list(range(1, len(raw)+1))
    assert outcomes == [(seq, RawOutbox.ACCEPTED) for seq, _ in raw]
    assert session._pool.retained_bytes == 0


def test_rejected_original_age_row_is_archived_but_never_delivered(tmp_path):
    process, session, raw, outcomes, receipt = _start(tmp_path, stale_step=8)
    try:
        assert session._bulk_done.wait(3)
        assert [seq for seq, _ in raw] == list(range(1, 9))
        assert outcomes[-1] == (8, RawOutbox.REJECTED)
        with pytest.raises((FasteningFault, TimeoutError, WireFault)):
            batch = session._read_before(0, timeout_s=.05, deadline=time.monotonic()+1.)
            list(batch)
    finally:
        with pytest.raises((WireFault, FasteningFault, EOFError)):
            _finish(process, session)
    result = json.loads(receipt.read_text())
    assert "stale" in result["owner"]["owner"]["error"]


def test_archive_failure_revokes_child_without_persistence_ack(tmp_path):
    def fail(_seq, _record):
        raise OSError("synthetic disk failure")
    process, session, _, _, receipt = _start(tmp_path, raw_sink=fail)
    try:
        assert session._bulk_done.wait(3)
        with pytest.raises(FasteningFault, match="archive"):
            session._check()
    finally:
        with pytest.raises((WireFault, FasteningFault, EOFError)):
            _finish(process, session)
    result = json.loads(receipt.read_text())
    assert not result["ok"]
    assert result["owner"]["owner"]["pending_records"] > 0
    assert result["owner"]["owner"]["zero_spindle"]["uploaded"]


@pytest.mark.parametrize("error", [TimeoutError("original reader deadline"),
                                  WireFault("fault fence refused delivery"),
                                  ValueError("invalid snapshot: " + "x"*1000)])
def test_reader_failure_retains_bounded_cause_through_observation_cleanup(tmp_path, monkeypatch, error):
    process, session, _, _, _ = _start(tmp_path)
    expected = f"Factory reader delivery failed: {type(error).__name__}: {error}"[:800]
    try:
        # Allocate the real domain archive so cleanup crosses the same sticky
        # fault check as execute(), after it has caught a reader exception.
        history = session._new_thread_history()
        batch = session._read_before(0, timeout_s=2., deadline=time.monotonic()+2.)
        assert batch
        def fail(*_args, **_kwargs):
            raise error
        monkeypatch.setattr(session._decoder, "read", fail)
        with pytest.raises(type(error)) as caught:
            next(batch)
        assert caught.value is error
        with pytest.raises(FasteningFault) as cleanup:
            session.release_observations()
        assert str(cleanup.value) == expected
        assert history.retained == {"rows": 0, "bytes": 0, "views": 0, "active_readers": 0}
        with pytest.raises(FasteningFault) as sticky:
            session._check()
        assert str(sticky.value) == expected
    finally:
        with pytest.raises((WireFault, FasteningFault, EOFError)):
            _finish(process, session)


def test_priority_stop_does_not_wait_for_blocked_host_archive(tmp_path):
    entered, release = threading.Event(), threading.Event()
    def pause(_seq, _record):
        entered.set()
        assert release.wait(3)
    process, session, _, _, receipt = _start(tmp_path, raw_sink=pause)
    try:
        assert entered.wait(2)
        started = time.monotonic()
        stop = session.stop()
        assert stop["ok"] and not stop["physical_stop_verified"]
        assert started <= stop["accepted_monotonic_s"] <= time.monotonic()
        assert not release.is_set()  # Persistence remains blocked throughout ACK.
    finally:
        release.set()
        assert _finish(process, session)["ok"]
    assert json.loads(receipt.read_text())["ok"]


def test_file_archive_wired_to_subprocess_is_complete_and_exclusive(tmp_path):
    archive = _FileArchive(tmp_path/"archive", byte_capacity=8*1024**2)
    persisted_rows = threading.Event()
    def persist_outcome(sequence, outcome):
        archive.outcome(sequence, outcome)
        if sequence >= 8:
            persisted_rows.set()
    process, session, _, _, receipt = _start(tmp_path, raw_sink=archive.raw, outcome_sink=persist_outcome)
    try:
        # Wait for actual outcome persistence before closing the subprocess.
        # Physical readiness and stale-row rejection have separate tests.
        assert persisted_rows.wait(2.)
    finally:
        closed = _finish(process, session)
        persisted = archive.close()
    assert closed["ok"] and persisted["ok"]
    solves = [json.loads(line) for line in (tmp_path/"archive/solves.jsonl").read_text().splitlines()]
    outcomes = [json.loads(line) for line in (tmp_path/"archive/outcomes.jsonl").read_text().splitlines()]
    assert len(solves) == len(outcomes) == persisted["raw"] >= 8
    assert [item["solve"]["step"] for item in solves] == list(range(1, len(solves)+1))
    assert all(item["outcome"] == RawOutbox.ACCEPTED for item in outcomes)
    with pytest.raises(FileExistsError):
        _FileArchive(tmp_path/"archive", byte_capacity=8*1024**2)
    assert json.loads(receipt.read_text())["ok"]


def test_file_archive_partial_failure_cannot_ack_or_heal(tmp_path):
    _, record = fixture()
    archive = _FileArchive(tmp_path/"archive", byte_capacity=32)
    with pytest.raises(WireFault, match="cap"):
        archive.raw(1, record)
    with pytest.raises(WireFault, match="faulted"):
        archive.raw(1, record)
    closed = archive.close()
    assert not closed["ok"] and closed["raw"] == closed["outcomes"] == 0


def test_aggregate_budget_rejects_individually_legal_caps():
    backend = _Synthetic()
    caps = replace(_Caps(), total_bytes=16*1024**3)
    with pytest.raises(ValueError, match="aggregate"):
        caps.validate(backend.layout)


def test_private_owner_rejects_untyped_transport_without_changing_default():
    backend = _Synthetic()
    with pytest.raises(TypeError, match="exact"):
        FactorySolveOwner(backend, _process_transport=object())
    owner = FactorySolveOwner(backend)
    assert type(owner.journal).__name__ == "SolveJournal"


@pytest.mark.parametrize("failure", ["guard", "owner"])
def test_close_failure_still_joins_owner_channels_and_returns_negative(tmp_path, failure):
    process, session, _, _, receipt = _start(tmp_path, close_failure=failure)
    assert not _finish(process, session)["ok"]
    result = json.loads(receipt.read_text())
    assert not result["ok"] and result["channels_closed"]
    assert result["owner"]["owner"]["owner_thread_closed"]
    assert f"Factory {failure} close failed" in result["errors"]


def test_cleanup_exception_cannot_skip_other_channel_close(monkeypatch):
    backend = _Synthetic()
    caps = _Caps()
    identity = ("a"*32, backend.layout.epoch, backend.binding.sha256)
    transport = _OwnerTransport(backend.layout, identity, caps)
    owner = FactorySolveOwner(backend, _process_transport=transport)
    pairs = [socket.socketpair() for _ in range(3)]
    service = _ChildService(owner, transport, tuple(pair[0] for pair in pairs),
                            deadline=time.monotonic()+1.)
    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic cleanup failure")
    monkeypatch.setattr(service.priority, "send", fail)
    monkeypatch.setattr(service, "_close_owner", fail)
    original = service.admission.close
    def close_then_fail():
        original()
        fail()
    monkeypatch.setattr(service.admission, "close", close_then_fail)
    try:
        result = service.serve()
        assert not result["ok"]
        assert "Factory child owner cleanup failed" in result["errors"]
        assert all(pair[0].fileno() == -1 for pair in pairs)
    finally:
        for pair in pairs:
            for endpoint in pair:
                endpoint.close()


def test_process_authority_fault_prevents_next_owner_cycle_before_upload():
    backend = _Synthetic()
    transport = _OwnerTransport(backend.layout, ("a"*32, backend.layout.epoch, backend.binding.sha256), _Caps())
    owner = FactorySolveOwner(backend, _process_transport=transport)
    transport.fault.fail("synthetic EOF")
    with pytest.raises(WireFault, match="EOF"):
        owner.cycle()
    assert backend.step == 0


@pytest.mark.parametrize("primary_failure", [False, True])
def test_owner_pending_eof_retains_error_and_resolves_all_admission_futures(monkeypatch, primary_failure):
    backend = _Synthetic()
    transport = _OwnerTransport(backend.layout, ("a"*32, backend.layout.epoch, backend.binding.sha256), _Caps())
    owner = FactorySolveOwner(backend, _process_transport=transport)
    transport.outbox.append_raw(b"pending synthetic frame")
    futures = [Future(), Future()]
    for future in futures:
        owner._requests.put_nowait(_Request("reset", {}, 0, time.monotonic()+1., future))
    owner._exit.set()  # Close without another synthetic solve or admission.
    if primary_failure:
        def fail():
            raise RuntimeError("primary synthetic owner fault")
        monkeypatch.setattr(backend, "enter_owner", fail)
    owner._run()
    assert transport.ended.is_set() and owner._requests.empty() and backend.step == 0
    for future in futures:
        assert future.done()
        with pytest.raises(FasteningFault, match="closed before admission"):
            future.result(timeout=0)
    closed = owner.close()
    assert not closed["ok"] and closed["owner_thread_closed"]
    assert closed["pending_records"] == 1 and closed["zero_spindle"]["uploaded"]
    expected = "primary synthetic owner fault" if primary_failure else "EOF before every raw acceptance outcome"
    assert expected in closed["error"]
    with pytest.raises(WireFault):
        transport.fault.check()
