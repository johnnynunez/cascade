"""Private CPU process seam over the unchanged Factory controller/owner loop.

No process is spawned here and no native backend is constructed or selected.
The reviewed launcher must supply three inherited socketpairs, one absolute
lifetime deadline, a bound layout and explicit caps. Separate priority requests
never queue behind RAW persistence or normal admission. All timestamps belong
to the producing owner; the host does not replace them with receipt times.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, fields
from types import SimpleNamespace

from ..control.fastening import FasteningFault, FasteningPermit
from ..skills.fastening_runtime import FasteningDomain
from .factory_process_history import _AcceptedHistory, _ByteLeases, _LatestJournal
from .factory_process_protocol import FaultState, FencedDecoder, RawOutbox
from .factory_process_snapshot import SnapshotLayout, decode_snapshot, encode_snapshot
from .factory_process_wire import FrameChannel, Kind, WireFault, WireLimits, remaining


class _FileArchive:
    """Exclusive host files; ACK means complete write+fsync, not buffered IO.

    A failed partial row is retained and sticky. This sink never rolls evidence
    back, overwrites a file, retains a decoded row or acknowledges its failure.
    Its explicit disk cap is additional to the process memory reservation.
    """

    def __init__(self, directory, *, byte_capacity, row_capacity=32768):
        if (type(byte_capacity) is not int or not 1 <= byte_capacity <= 64*1024**3
                or type(row_capacity) is not int or not 1 <= row_capacity <= 32768):
            raise ValueError("bounded exclusive archive capacities required")
        directory.mkdir(mode=0o700)
        self.capacity, self.row_capacity = byte_capacity, row_capacity
        self.bytes = self.raw_count = self.outcome_count = 0
        self.fault = None
        self._files = {}
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for name in ("solves.jsonl", "outcomes.jsonl"):
                self._files[name] = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                           0o600, dir_fd=fd)
        except BaseException:
            for output in self._files.values():
                os.close(output)
            self._files.clear()
            raise
        finally:
            os.close(fd)

    def _write(self, name, value):
        if self.fault is not None or not self._files:
            raise WireFault("archive is faulted or closed")
        try:
            buffer = bytearray()
            def flush():
                if self.bytes + len(buffer) > self.capacity:
                    raise WireFault("archive disk byte cap exceeded")
                view = memoryview(buffer)
                while view:
                    count = os.write(self._files[name], view)
                    if not count:
                        raise OSError("archive write made no progress")
                    self.bytes += count
                    view = view[count:]
            for text in json.JSONEncoder(sort_keys=True, allow_nan=False).iterencode(value):
                buffer.extend(text.encode("utf-8"))
                if len(buffer) >= 16384:
                    flush()
                    buffer = bytearray()
            buffer.extend(b"\n")
            flush()
            os.fsync(self._files[name])
        except BaseException:
            self.fault = "archive persistence failed"
            raise

    def raw(self, sequence, record):
        if sequence != self.raw_count+1 or self.raw_count != self.outcome_count or sequence > self.row_capacity:
            self.fault = "archive RAW sequence or capacity refused"
            raise WireFault(self.fault)
        self._write("solves.jsonl", record.materialize())
        self.raw_count = sequence

    def outcome(self, sequence, outcome):
        if sequence != self.raw_count or sequence != self.outcome_count+1 or outcome not in (1, 2):
            self.fault = "archive outcome sequence refused"
            raise WireFault(self.fault)
        self._write("outcomes.jsonl", {"sequence": sequence, "outcome": outcome})
        self.outcome_count = sequence

    def close(self):
        errors = []
        for fd in self._files.values():
            try:
                os.close(fd)
            except OSError as error:
                errors.append(type(error).__name__)
        self._files.clear()
        return {"ok": self.fault is None and not errors and self.raw_count == self.outcome_count,
                "raw": self.raw_count, "outcomes": self.outcome_count, "bytes": self.bytes,
                "error": self.fault, "close_errors": errors}


@dataclass(frozen=True)
class _Caps:
    raw_slots: int = 32768
    raw_bytes: int = 1024**3
    history_slots: int = 20000
    history_bytes: int = 8805880000 + 256*1024**2
    thread_bytes: int = 64*1024**2
    working_bytes: int = 64*1024**2
    metadata_bytes: int = 32*1024**2
    sdk_reserve: int = 12*1024**3
    host_reserve: int = 2*1024**3
    total_bytes: int = 24*1024**3

    def validate(self, layout):
        maxima = type(self)()
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or not 1 <= value <= getattr(maxima, field.name):
                raise ValueError("process capacity exceeds reviewed bound")
        frame = layout.maximum_frame_bytes
        if (self.raw_slots < 3 or self.history_slots < 3
                or frame > min(self.raw_bytes, self.history_bytes)
                or 16*frame > self.working_bytes
                or sum(getattr(self, name) for name in (
                    "raw_bytes", "history_bytes", "thread_bytes", "working_bytes",
                    "metadata_bytes", "sdk_reserve", "host_reserve")) > self.total_bytes):
            raise ValueError("aggregate Factory process budget exceeded")
        # Reservations are planning bounds, not a measured native admission.
        # The launcher must still measure SDK/host headroom at preparation.


class _AuthorityJournal(_LatestJournal):
    def __init__(self, binding, authority):
        super().__init__(binding)
        self.authority = authority

    def fail(self, reason):
        super().fail(reason)
        self.authority.fail(self._error[:512])


class _OwnerTransport:
    """Nonblocking owner archive seam; the sender, never owner, performs IO."""

    def __init__(self, layout, identity, caps):
        if type(layout) is not SnapshotLayout or type(caps) is not _Caps:
            raise TypeError("exact process layout/caps required")
        caps.validate(layout)
        self.layout, self.caps = layout, caps
        self.fault = FaultState(boot_id=identity[0], epoch=identity[1], binding=identity[2])
        if identity[1:] != (layout.epoch, layout.binding_sha256):
            raise ValueError("process and snapshot identity differ")
        self.outbox = RawOutbox(caps.raw_slots, caps.raw_bytes, layout.maximum_frame_bytes, self.fault)
        self.changed, self.ended = threading.Event(), threading.Event()
        self.owner = None

    def attach(self, owner):
        if self.owner is not None or owner.backend.binding.sha256 != self.layout.binding_sha256:
            raise ValueError("process owner identity mismatch or reuse")
        self.owner = owner
        return _AuthorityJournal(owner.backend.binding, self.fault)

    def capture(self, row, raw, accept):
        # Exactly the snapshot accepted by the controller, not a parallel row.
        from .factory_observation import _FactoryReadback
        if type(raw) is not _FactoryReadback or dict(raw.entries).get("solve") is not row:
            raise WireFault("RAW and acceptance row differ")
        sequence = self.outbox.append_raw(encode_snapshot(raw, self.layout))
        self.changed.set()
        try:
            accept()
        except BaseException:
            self.outbox.finish(sequence, accepted=False)
            raise
        else:
            self.outbox.finish(sequence, accepted=True)
        finally:
            self.changed.set()

    def end(self):
        try:
            self.outbox.end_of_stream()
        finally:
            self.ended.set()
            self.changed.set()

    def fail(self, reason):
        self.fault.fail(reason)
        if self.owner is not None:
            self.owner.controller.stop()
            self.owner.journal.fail(reason)
        self.changed.set()


def _channel_limits(layout):
    # The snapshot itself remains limited by the inner schema. The outer RAW
    # sequence/payload envelope and RPC metadata need a small explicit margin.
    return WireLimits(frame_bytes=layout.maximum_frame_bytes+4096, nodes=512,
                      depth=8, text_bytes=4096)


def _channels(sockets, layout):
    if type(sockets) is not tuple or len(sockets) != 3:
        raise ValueError("three separate inherited channels required")
    if len({socket.fileno() for socket in sockets}) != 3:
        raise ValueError("bulk/admission/priority must be separate sockets")
    return tuple(FrameChannel(socket, _channel_limits(layout)) for socket in sockets)


def _close_each(channels):
    errors = []
    for channel in channels:
        try:
            channel.close()
        except BaseException:  # noqa: BLE001 - one close cannot skip other owned descriptors
            errors.append("Factory channel close failed")
    return errors


def _result(value):
    if type(value) is FasteningPermit:
        return ("permit", tuple(getattr(value, field.name) for field in fields(value)))
    if type(value) is dict:
        return ("receipt", value)
    raise WireFault("unsupported owner result type")


def _restore(value):
    if type(value) is not tuple or len(value) != 2:
        raise WireFault("invalid owner result envelope")
    if value[0] == "permit" and type(value[1]) is tuple and len(value[1]) == len(fields(FasteningPermit)):
        return FasteningPermit(*value[1])
    if value[0] == "receipt" and type(value[1]) is dict:
        return value[1]
    raise WireFault("invalid owner result schema")


class _ChildService:
    """Run on an already bound owner. Close stays within the external deadline."""

    def __init__(self, owner, transport, sockets, *, deadline):
        if type(transport) is not _OwnerTransport or transport.owner is not owner:
            raise TypeError("transport must be attached to this owner")
        remaining(deadline)
        self.owner, self.transport, self.deadline = owner, transport, deadline
        self.bulk, self.admission, self.priority = _channels(sockets, transport.layout)
        self.closing = threading.Event()
        self.bulk_done = threading.Event()
        self.errors = []

    def _failure(self):
        self.errors.append("Factory child channel failed")
        try:
            self.transport.fail("Factory child channel failed")
        except BaseException:  # noqa: BLE001 - still close every owned channel
            self.errors.append("Factory child fault/stop delivery failed")
        finally:
            self.closing.set()
            self.errors.extend(_close_each((self.admission, self.priority)))

    def _close_owner(self):
        # Neither guard failure nor SDK-close failure may skip the remaining
        # cleanup. The fallback join spends the SAME two-second interval.
        end = min(self.deadline, time.monotonic()+2.)
        try:
            stop = self.owner.controller.guard.close()
        except BaseException:  # noqa: BLE001 - retain failure and attempt owner close
            self.errors.append("Factory guard close failed")
            stop = {"ok": False, "physical_stop_verified": False}
        try:
            closure = self.owner.close(timeout_s=min(2., max(.000001, end-time.monotonic())))
        except BaseException:  # noqa: BLE001 - bounded fallback, no invented SDK receipt
            self.errors.append("Factory owner close failed")
            self.owner._exit.set()
            thread = self.owner._thread
            if thread is not None and thread.ident is not None:
                thread.join(max(0., end-time.monotonic()))
            closure = {"ok": False, "owner_thread_closed": thread is None or not thread.is_alive(),
                       "error": "owner close raised", "physical_stop_verified": False}
        return {"ok": closure["ok"] and stop["ok"], "stop": stop, "owner": closure,
                "physical_stop_verified": False}

    def _admissions(self):
        try:
            while not self.closing.is_set():
                kind, request = self.admission.receive(deadline=self.deadline)
                if (kind is not Kind.REQUEST or type(request) is not tuple or len(request) != 4
                        or request[0] not in ("reset", "turn", "seat") or type(request[1]) is not dict):
                    raise WireFault("invalid normal admission request")
                name, arguments, generation, original_deadline = request
                if type(original_deadline) is not float or original_deadline > self.deadline:
                    raise WireFault("admission deadline exceeds original scope")
                try:
                    result = self.owner._request_before(name, arguments, generation, original_deadline)
                    reply = (True, _result(result))
                except FasteningFault as error:
                    reply = (False, str(error)[:512])
                self.admission.send(Kind.ACK, reply, deadline=original_deadline)
        except BaseException:  # noqa: BLE001 - latch and close the child on channel failure
            if not self.closing.is_set():
                self._failure()

    def _bulk(self):
        try:
            while True:
                prefix = self.transport.outbox.prefix(1)
                if not prefix:
                    if self.transport.ended.is_set():
                        self.transport.outbox.require_drained()
                        self.bulk.send(Kind.CLOSE, {"complete": True}, deadline=self.deadline)
                        kind, receipt = self.bulk.receive(deadline=self.deadline)
                        if kind is not Kind.ACK or receipt != {"complete": True}:
                            raise WireFault("missing host archive closure")
                        return
                    self.transport.changed.wait(min(.01, remaining(self.deadline)))
                    self.transport.changed.clear()
                    continue
                sequence, payload, outcome = prefix[0]
                self.bulk.send(Kind.RAW, (sequence, payload), deadline=self.deadline)
                kind, ack = self.bulk.receive(deadline=self.deadline)
                if kind is not Kind.ACK or ack != (sequence, RawOutbox.ARCHIVED):
                    raise WireFault("missing exact persistence ACK")
                self.transport.outbox.acknowledge(sequence, RawOutbox.ARCHIVED)
                while outcome == RawOutbox.PENDING:
                    remaining(self.deadline)
                    self.transport.changed.wait(min(.01, remaining(self.deadline)))
                    self.transport.changed.clear()
                    current, = self.transport.outbox.prefix(1)
                    outcome = current[2]
                self.bulk.send(Kind.OUTCOME, (sequence, outcome), deadline=self.deadline)
                for consumer in (RawOutbox.OUTCOME_SEEN, RawOutbox.READER_DONE):
                    kind, ack = self.bulk.receive(deadline=self.deadline)
                    if kind is not Kind.ACK or ack != (sequence, consumer):
                        raise WireFault("missing exact outcome/reader ACK")
                    self.transport.outbox.acknowledge(sequence, consumer)
        except BaseException:  # noqa: BLE001 - no bulk failure may retain live authority
            self._failure()
        finally:
            self.bulk_done.set()

    def serve(self):
        threads = []
        closure = None
        close_attempted = False
        try:
            hello = {"identity": self.transport.fault.identity,
                     "layout": self.transport.layout.sha256, "caps": asdict(self.transport.caps),
                     "synthetic": self.owner.backend.synthetic}
            self.priority.send(Kind.HELLO, hello, deadline=self.deadline)
            for target in (self._admissions, self._bulk):
                thread = threading.Thread(target=target, daemon=True)
                thread.start()
                threads.append(thread)
            self.owner.start()
            while not self.closing.is_set():
                kind, argument = self.priority.receive(deadline=self.deadline)
                if kind is Kind.FENCE:
                    result = {"status": self.transport.fault.status(argument),
                              "generation": self.owner.controller.generation}
                elif kind is Kind.STOP and argument is None:
                    result = self.owner.controller.stop()
                elif kind is Kind.CLOSE and argument is None:
                    self.closing.set()
                    close_attempted = True
                    closure = self._close_owner()
                    result = closure
                else:
                    raise WireFault("unsupported priority operation")
                self.priority.send(Kind.ACK, result, deadline=self.deadline)
        except BaseException:  # noqa: BLE001 - all exceptions enter owned stop/cleanup
            self._failure()
        finally:
            self.closing.set()
            self.errors.extend(_close_each((self.admission,)))
            try:
                if not close_attempted:
                    close_attempted = True
                    # No renewed interval/retry if the first close raised.
                    try:
                        closure = self._close_owner()
                    except BaseException:  # noqa: BLE001 - still join and close channel threads
                        self.errors.append("Factory child owner cleanup failed")
                for thread in threads:
                    try:
                        thread.join(max(0., self.deadline-time.monotonic()))
                    except BaseException:  # noqa: BLE001 - still join/close other owned resources
                        self.errors.append("Factory child thread join failed")
            except BaseException:  # noqa: BLE001 - close failure cannot skip descriptor cleanup
                self.errors.append("Factory child owner cleanup failed")
            finally:
                self.errors.extend(_close_each((self.bulk, self.admission, self.priority)))
        if closure is None:
            closure = {"ok": False, "error": "owner cleanup did not return",
                       "physical_stop_verified": False}
        drained = self.bulk_done.is_set() and not any(thread.is_alive() for thread in threads)
        return {"ok": closure["ok"] and drained and not self.errors,
                "owner": closure, "channels_closed": drained, "errors": tuple(self.errors)}


class _HostSession:
    """Private actuator/reader over the child; persistent archive is mandatory.

    persist_raw(sequence, record) and persist_outcome(sequence, outcome) must
    finish their durable write before returning. They execute only on the host
    bulk thread. No ACK is sent after a failed write. They may not retain decoded
    records; the adapter retains only immutable packet leases between records.
    """

    def __init__(self, sockets, *, layout, identity, binding, limits, caps,
                 deadline, synthetic, persist_raw, persist_outcome):
        caps.validate(layout)
        remaining(deadline)
        if binding.sha256 != layout.binding_sha256 or type(synthetic) is not bool:
            raise ValueError("explicit bound host identity required")
        self.layout, self.identity, self.caps = layout, identity, caps
        self.binding, self.limits, self.synthetic = binding, limits, synthetic
        self.deadline = deadline
        self.bulk, self.admission, self.priority = _channels(sockets, layout)
        self._rpc_locks = (threading.Lock(), threading.Lock())
        self._condition = threading.Condition()
        self._fault = None
        self._closing = False
        self._bulk_done = threading.Event()
        self._batch = None
        self._histories = []
        self._persist_raw, self._persist_outcome = persist_raw, persist_outcome
        self._pool = _ByteLeases(capacity=min(65536, caps.history_slots+caps.raw_slots),
                                byte_capacity=caps.history_bytes, frame_bytes=layout.maximum_frame_bytes)
        self._history = _AcceptedHistory(binding, self._pool, capacity=caps.history_slots)
        self._last_step = 0
        self._decoder = FencedDecoder(identity, lambda nonce, end: self._status(nonce, end)["status"])
        kind, hello = self.priority.receive(deadline=deadline)
        expected = {"identity": identity, "layout": layout.sha256,
                    "caps": asdict(caps), "synthetic": synthetic}
        if kind is not Kind.HELLO or hello != expected:
            self.abort("Factory process handshake differs from admitted contract")
            raise WireFault(self._fault)
        self._thread = threading.Thread(target=self._consume, name="factory-host-archive", daemon=True)
        self._thread.start()

    def _check(self):
        with self._condition:
            if self._fault:
                raise FasteningFault(self._fault)

    def abort(self, reason="Factory host failed"):
        with self._condition:
            self._fault = self._fault or reason
            self._condition.notify_all()
        # Child priority EOF goes through its real stop and owner close paths.
        _close_each((self.priority, self.admission))

    def _rpc(self, kind, argument, deadline, *, normal=False):
        channel = self.admission if normal else self.priority
        lock = self._rpc_locks[int(normal)]
        acquired = False
        try:
            acquired = lock.acquire(timeout=remaining(deadline))
            if not acquired:
                raise TimeoutError("Factory RPC lock exceeded original deadline")
            channel.send(kind, argument, deadline=deadline)
            reply_kind, result = channel.receive(deadline=deadline)
            if reply_kind is not Kind.ACK:
                raise WireFault("missing Factory RPC acknowledgment")
            return result
        except BaseException:
            self.abort("Factory host RPC failed")
            raise
        finally:
            if acquired:
                lock.release()

    def _status(self, nonce, deadline):
        self._check()
        result = self._rpc(Kind.FENCE, nonce, deadline)
        if (type(result) is not dict or set(result) != {"status", "generation"}
                or type(result["generation"]) is not int or result["generation"] < 0):
            raise WireFault("invalid authoritative status envelope")
        status = result["status"]
        if (type(status) is not dict
                or set(status) != {"nonce", "boot_id", "epoch", "binding", "revision", "fault"}
                or type(status["nonce"]) is not bytes or status["nonce"] != nonce
                or any(type(status[key]) is not str for key in ("boot_id", "epoch", "binding"))
                or (status["boot_id"], status["epoch"], status["binding"]) != self.identity
                or type(status["revision"]) is not int or status["revision"] < 0
                or not (status["fault"] is None or type(status["fault"]) is str)):
            raise WireFault("invalid authoritative status identity")
        return result

    @property
    def generation(self):
        import secrets
        result = self._status(secrets.token_bytes(16), min(self.deadline, time.monotonic()+1.))
        if result["status"]["fault"] is not None:
            raise FasteningFault("child authority is faulted")
        return result["generation"]

    def _request(self, name, arguments):
        self._check()
        # One deadline begins BEFORE generation query, serialization and IPC.
        deadline = min(self.deadline, time.monotonic()+1.)
        import secrets
        generation = self._status(secrets.token_bytes(16), deadline)["generation"]
        reply = self._rpc(Kind.REQUEST, (name, arguments, generation, deadline), deadline, normal=True)
        if type(reply) is not tuple or len(reply) != 2 or type(reply[0]) is not bool:
            raise WireFault("invalid admission ACK schema")
        if not reply[0]:
            raise FasteningFault(reply[1])
        return _restore(reply[1])

    def request_turn(self, **arguments):
        return self._request("turn", arguments)

    def request_seating(self, **arguments):
        return self._request("seat", arguments)

    def reset_stop(self):
        return self._request("reset", {})

    def stop(self):
        return self._rpc(Kind.STOP, None, min(self.deadline, time.monotonic()+1.))

    def _consume(self):
        lease = None
        try:
            while True:
                kind, value = self.bulk.receive(deadline=self.deadline)
                if kind is Kind.CLOSE and value == {"complete": True}:
                    self.bulk.send(Kind.ACK, value, deadline=self.deadline)
                    return
                if kind is not Kind.RAW or type(value) is not tuple or len(value) != 2:
                    raise WireFault("expected exact Factory RAW envelope")
                sequence, payload = value
                lease = self._pool.register(sequence, payload)
                record = decode_snapshot(payload, self.layout)
                row = dict(record.entries)["solve"]
                self._persist_raw(sequence, record)
                # At most one decoded RAW tree is held by this bulk consumer;
                # the ring retains only the packet, never the decoded history.
                metadata = row
                del record, row
                self.bulk.send(Kind.ACK, (sequence, RawOutbox.ARCHIVED), deadline=self.deadline)
                kind, value = self.bulk.receive(deadline=self.deadline)
                if (kind is not Kind.OUTCOME or type(value) is not tuple or len(value) != 2
                        or type(value[0]) is not int or value[0] != sequence
                        or type(value[1]) is not int or value[1] not in (RawOutbox.ACCEPTED, RawOutbox.REJECTED)):
                    raise WireFault("missing exact Factory acceptance outcome")
                outcome = value[1]
                self._persist_outcome(sequence, outcome)
                if outcome == RawOutbox.ACCEPTED:
                    with self._condition:
                        self._history.publish(lease, metadata)
                        self._last_step = metadata.step
                        self._condition.notify_all()
                del metadata
                self.bulk.send(Kind.ACK, (sequence, RawOutbox.OUTCOME_SEEN), deadline=self.deadline)
                # Host owns a full immutable lease now; no child buffer alias is
                # reachable by any reader. That transfer is the reader ACK.
                self.bulk.send(Kind.ACK, (sequence, RawOutbox.READER_DONE), deadline=self.deadline)
                lease.close()
                lease = None
        except BaseException:  # noqa: BLE001 - failed archive can grant no acknowledgment
            self.abort("Factory RAW/outcome archive failed")
        finally:
            if lease is not None:
                lease.close()
            self._bulk_done.set()
            with self._condition:
                self._condition.notify_all()

    def _read_before(self, after_step, *, timeout_s, deadline):
        if (type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s < 0
                or type(deadline) is not float or not math.isfinite(deadline)):
            raise ValueError("original observation deadline required")
        deadline = min(deadline, self.deadline)
        self.release_batch()
        with self._condition:
            self._condition.wait_for(lambda: self._fault or self._last_step > after_step or self._bulk_done.is_set(),
                                     timeout=min(timeout_s, remaining(deadline)))
            self._check()
            batch = self._history.snapshot(after_step)
            self._batch = _ReadBatch(self, batch, deadline)
            return self._batch

    def release_batch(self):
        if self._batch is not None:
            self._batch.close()
            self._batch = None

    def _new_thread_history(self):
        from .factory_process_threading import _ThreadArchive
        if self._histories:
            self.abort("overlapping Factory thread archives")
            raise FasteningFault(self._fault)
        archive = _ThreadArchive(self._check, self.abort, byte_capacity=self.caps.thread_bytes,
                                 reader_capacity=1)
        self._histories.append(archive)
        return archive

    def release_observations(self):
        try:
            self.release_batch()
        finally:
            histories, self._histories = self._histories, []
            for history in histories:
                history.close()
                history.require_drained()

    def close(self):
        receipt = error = None
        try:
            self.release_observations()
            receipt = self._rpc(Kind.CLOSE, None, self.deadline)
            self._closing = True
            self._thread.join(remaining(self.deadline))
            if self._thread.is_alive() or not self._bulk_done.is_set():
                raise WireFault("host archive did not drain within original scope")
        except BaseException as exc:  # noqa: BLE001 - retain primary through owned cleanup
            error = exc
        finally:
            if _close_each((self.bulk, self.admission, self.priority)):
                error = error or WireFault("Factory host channel close failed")
            try:
                self._thread.join(max(0., self.deadline-time.monotonic()))
            except BaseException as exc:  # noqa: BLE001 - continue lease/history cleanup
                error = error or exc
            try:
                self.release_observations()
                self._history.close()
            except BaseException as exc:  # noqa: BLE001 - all channels already closed
                error = error or exc
            if self._thread.is_alive():
                error = error or WireFault("host archive thread remains alive")
        if error is not None:
            raise error
        self._pool.require_drained()
        self._check()
        return receipt


class _ReadBatch:
    """Finite private delivery; one decoded row at a time, no decoded cache."""
    def __init__(self, host, batch, deadline):
        self.host, self.batch, self.deadline = host, batch, deadline
        self.closed = False
        self._position = 0
        self._state = threading.Lock()
        self._reading = threading.Lock()

    def __bool__(self):
        return bool(self.batch._leases)

    def __iter__(self):
        return self

    def __next__(self):
        # No generator retains a lease across a caller's break. Each delivered
        # row is fully detached; its packet lease is released BEFORE return.
        if not self._reading.acquire(blocking=False):
            self.host.abort("concurrent Factory batch decoding")
            raise FasteningFault("concurrent Factory batch decoding")
        active = None
        try:
            with self._state, self.batch._lock:
                if self.closed:
                    raise FasteningFault("reader batch already closed")
                if self._position == len(self.batch._leases):
                    raise StopIteration
                active = self.batch._leases[self._position].retain()
            def decode(value):
                return dict(decode_snapshot(value, self.host.layout).entries)["solve"]
            def validate(_row):
                self.host._check()
                with self._state:
                    if self.closed:
                        raise FasteningFault("reader batch closed during decode")
            value = self.host._decoder.read(active._payload(), RawOutbox.ACCEPTED, decode=decode,
                                           validate=validate, deadline=self.deadline)
            with self._state:
                if self.closed:
                    raise FasteningFault("reader batch closed before delivery")
                self._position += 1
                return value
        except StopIteration:
            raise
        except BaseException:
            self.host.abort("Factory reader delivery failed")
            raise
        finally:
            if active is not None:
                active.close()
            self._reading.release()

    def close(self):
        with self._state:
            if not self.closed:
                self.closed = True
                self.batch.close()


class _ProcessDomain(FasteningDomain):
    """Ordinary domain algorithm over IPC; explicit finite evidence lifetimes.

    This is the only supported host reader consumer. It serializes operations,
    keeps at most the current/previous decoded rows and never exposes batches
    or mutable archive internals through the public runtime tool surface.
    """

    def __init__(self, session, *, controller_id, domain_id="fastening"):
        if type(session) is not _HostSession:
            raise TypeError("exact private host session required")
        self._operation = threading.Lock()
        super().__init__(session, session, controller_id=controller_id, domain_id=domain_id)

    def _read(self, cursor, deadline):
        return self.actuator._read_before(cursor, deadline=deadline,
                    timeout_s=min(.05, max(0., deadline-self.clock())))

    def _new_samples(self):
        return self.actuator._new_thread_history()

    def _verify_threading(self, samples, contract, *, stop=None):
        from .factory_process_threading import _ThreadArchive
        from .threading_verification import verify_threading
        if type(samples) is not _ThreadArchive:
            raise TypeError("process domain requires bounded thread archive")
        with samples[:stop] as finite:
            return verify_threading(finite, contract)

    def ready(self, *, timeout_s=10.):
        from ..apps.factory_runtime import _ready
        if not self._operation.acquire(blocking=False):
            raise FasteningFault("Factory observation already in use")
        try:
            return _ready(SimpleNamespace(backend=self.actuator), timeout_s=timeout_s,
                          _reader=self.actuator._read_before)
        finally:
            try:
                self.actuator.release_observations()
            finally:
                self._operation.release()

    def execute(self, name, args):
        if not self._operation.acquire(blocking=False):
            raise FasteningFault("Factory observation already in use")
        result = None
        try:
            result = super().execute(name, args)
            return result
        finally:
            try:
                self.actuator.release_observations()
            except BaseException:
                self.actuator.abort("Factory observation release failed")
                if result is not None:
                    result.update(ok=False, verified=False, physical_stop_verified=False,
                                  postcondition={"status": "unverified", "reason": "observation release failed"})
                raise
            finally:
                self._operation.release()
