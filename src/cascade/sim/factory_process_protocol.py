"""CPU-only transaction/fault primitives for the optional Factory transport.

No native adapter is selected here. In particular, a byte payload is not a
solved observation: the adapter must apply the bound snapshot schema and the
unchanged controller checks before declaring its acceptance outcome.
"""
from __future__ import annotations

import secrets
import threading
from array import array

from .factory_process_wire import WireFault, remaining


def _positive(value, upper):
    if type(value) is not int or not 1 <= value <= upper:
        raise ValueError("invalid Factory transport capacity")
    return value


class FaultState:
    """Child authority, serialized with publication of its sticky fault.

    A status reply linearizes while this lock is held. It describes that point,
    not a promise that no later fault can occur before the host returns. There
    is deliberately no reset method and no reconnect under the same boot ID.
    boot_id is a fresh 128-bit hex process-incarnation nonce; it is NOT Linux's
    36-character /proc/sys/kernel/random/boot_id or a substitute for PID birth.
    """

    def __init__(self, *, boot_id: str, epoch: str, binding: str):
        for value, size in ((boot_id, 32), (epoch, 32), (binding, 64)):
            if type(value) is not str or len(value) != size or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("invalid Factory process identity")
        self.identity = (boot_id, epoch, binding)
        self._lock = threading.Lock()
        self._revision = 0
        self._reason = None

    def fail(self, reason: str):
        if type(reason) is not str or not reason or len(reason) > 512:
            raise ValueError("bounded fault reason required")
        with self._lock:
            if self._reason is None:
                self._reason = reason
                self._revision += 1

    def check(self):
        with self._lock:
            if self._reason is not None:
                raise WireFault(self._reason)

    def status(self, nonce: bytes):
        if type(nonce) is not bytes or len(nonce) != 16:
            raise WireFault("exact fence nonce required")
        with self._lock:
            return {"nonce": nonce, "boot_id": self.identity[0],
                    "epoch": self.identity[1], "binding": self.identity[2],
                    "revision": self._revision, "fault": self._reason}


class RawOutbox:
    """Bounded raw bytes, outcome and acknowledgments in preallocated slots.

    RAW is committed before accept_solve. finish() is called only after that
    call returned/raised. Archive acknowledgment alone never frees a slot: both
    outcome and reader acknowledgment are also required. The host can archive
    rejected rows after a sticky fault, but cannot treat them as accepted data.
    """

    ARCHIVED = 1
    OUTCOME_SEEN = 2
    READER_DONE = 4
    PENDING = 0
    ACCEPTED = 1
    REJECTED = 2

    def __init__(self, capacity: int, byte_capacity: int, frame_bytes: int, fault: FaultState):
        self.capacity = _positive(capacity, 32768)
        self.byte_capacity = _positive(byte_capacity, 24 * 1024**3)
        self.frame_bytes = _positive(frame_bytes, 16 * 1024**2)
        if frame_bytes > byte_capacity or type(fault) is not FaultState:
            raise ValueError("invalid Factory outbox bounds/authority")
        self.fault = fault
        self._lock = threading.Lock()
        self._payloads = [None] * capacity
        self._outcomes = array("B", [0]) * capacity
        self._acks = array("B", [0]) * capacity
        self._head = self._next = 1
        self._bytes = 0
        self._ended = False

    def _refuse(self, reason):
        self.fault.fail(reason)
        raise WireFault(reason)

    def _index(self, sequence):
        if type(sequence) is not int or not self._head <= sequence < self._next:
            self._refuse("unknown or already released raw sequence")
        return (sequence - 1) % self.capacity

    def append_raw(self, payload: bytes) -> int:
        with self._lock:
            self.fault.check()
            if self._ended:
                self._refuse("raw publication after EOF")
            if type(payload) is not bytes or not 0 < len(payload) <= self.frame_bytes:
                self._refuse("invalid raw frame bytes")
            if self._next - self._head == self.capacity or self._bytes + len(payload) > self.byte_capacity:
                self._refuse("Factory raw outbox capacity exceeded")
            sequence = self._next
            index = (sequence - 1) % self.capacity
            self._payloads[index] = payload
            self._outcomes[index] = self._acks[index] = 0
            self._bytes += len(payload)
            self._next += 1
            return sequence

    def finish(self, sequence: int, *, accepted: bool):
        with self._lock:
            index = self._index(sequence)
            if type(accepted) is not bool or self._outcomes[index] != self.PENDING:
                self._refuse("missing or duplicate raw acceptance outcome")
            # Record what the actual accept call did. A concurrent sticky fault
            # refuses delivery at the authoritative fence; it must not rewrite
            # an accept call that already returned as one that raised.
            self._outcomes[index] = self.ACCEPTED if accepted else self.REJECTED
            if not accepted:
                self.fault.fail("solver refused the archived raw snapshot")

    def transact(self, payload: bytes, accept):
        """Publish RAW before calling the unchanged acceptance function once."""
        sequence = self.append_raw(payload)
        try:
            result = accept()
        except BaseException:
            self.finish(sequence, accepted=False)
            raise
        self.finish(sequence, accepted=True)
        return sequence, result

    def prefix(self, max_records: int):
        """Finite detached prefix; later producer rows are left for next drain."""
        _positive(max_records, self.capacity)
        with self._lock:
            end = min(self._next, self._head + max_records)
            return tuple((sequence, self._payloads[(sequence - 1) % self.capacity],
                          self._outcomes[(sequence - 1) % self.capacity])
                         for sequence in range(self._head, end))

    def acknowledge(self, sequence: int, consumer: int):
        with self._lock:
            index = self._index(sequence)
            if type(consumer) is not int or consumer not in (self.ARCHIVED, self.OUTCOME_SEEN, self.READER_DONE):
                self._refuse("unknown Factory raw consumer")
            if consumer != self.ARCHIVED and self._outcomes[index] == self.PENDING:
                self._refuse("raw outcome has not been published")
            if self._acks[index] & consumer:
                self._refuse("duplicate raw acknowledgment")
            self._acks[index] |= consumer
            while self._head < self._next:
                head = (self._head - 1) % self.capacity
                if self._acks[head] != 7:
                    break
                self._bytes -= len(self._payloads[head])
                self._payloads[head] = None
                self._outcomes[head] = self._acks[head] = 0
                self._head += 1

    def end_of_stream(self):
        """Report incomplete acceptance even if raw persistence itself succeeded."""
        with self._lock:
            self._ended = True
            if any(self._outcomes[(seq - 1) % self.capacity] == self.PENDING
                   for seq in range(self._head, self._next)):
                self._refuse("EOF before every raw acceptance outcome")

    def require_drained(self):
        with self._lock:
            if not self._ended or self._head != self._next or self._bytes:
                self._refuse("raw stream closed without all consumer acknowledgments")


class FencedDecoder:
    """Decode between authoritative nonce replies within one original deadline.

    request_status(nonce, deadline) must use the separate priority channel and
    the child FaultState.status operation. validate(value) is the unchanged
    original schema/freshness validation, called with a fresh local clock on
    each invocation. No timestamp is supplied or rewritten by this layer.

    The second child status is the delivery linearization point. A fault after
    that point is observed on the next read; it cannot retroactively retract a
    completed reply. Local deadline/freshness checks still run after the reply.
    """

    def __init__(self, identity: tuple[str, str, str], request_status):
        # Validate identity through the same narrow format, without retaining
        # or using this separate instance as a local source of fault authority.
        if type(identity) is not tuple or len(identity) != 3:
            raise ValueError("exact process identity required")
        FaultState(boot_id=identity[0], epoch=identity[1], binding=identity[2])
        self.identity = identity
        self._request = request_status
        self._fault = None
        self._lock = threading.Lock()
        self._state = threading.Lock()

    def _check_local(self):
        with self._state:
            if self._fault is not None:
                raise WireFault(self._fault)

    def _status(self, deadline):
        remaining(deadline)
        nonce = secrets.token_bytes(16)
        reply = self._request(nonce, deadline)
        remaining(deadline)
        if (type(reply) is not dict or set(reply) != {"nonce", "boot_id", "epoch", "binding", "revision", "fault"}
                or type(reply["nonce"]) is not bytes or reply["nonce"] != nonce
                or any(type(reply[key]) is not str for key in ("boot_id", "epoch", "binding"))
                or (reply["boot_id"], reply["epoch"], reply["binding"]) != self.identity
                or type(reply["revision"]) is not int or reply["revision"] != 0
                or reply["fault"] is not None):
            raise WireFault("Factory authoritative fault fence refused delivery")
        return reply["revision"]

    def read(self, payload: bytes, outcome: int, *, decode, validate, deadline: float):
        acquired = False
        try:
            acquired = self._lock.acquire(timeout=remaining(deadline))
            if not acquired:
                raise TimeoutError("reader lock exceeded original deadline")
            self._check_local()
            if type(payload) is not bytes or type(outcome) is not int or outcome != RawOutbox.ACCEPTED:
                raise WireFault("raw snapshot lacks an accepted outcome")
            before = self._status(deadline)
            value = decode(payload)
            remaining(deadline)
            self._check_local()
            validate(value)
            after = self._status(deadline)
            if before != after:
                raise WireFault("fault revision changed during decode")
            validate(value)
            remaining(deadline)
            # A sibling read can expire waiting for _lock while this read is
            # decoding. Child status cannot observe that host-local failure.
            # Serialize its publication with this final local delivery point.
            with self._state:
                if self._fault is not None:
                    raise WireFault(self._fault)
                return value
        except BaseException:
            with self._state:
                if self._fault is None:
                    self._fault = "Factory accepted-snapshot delivery failed"
            raise
        finally:
            if acquired:
                self._lock.release()
