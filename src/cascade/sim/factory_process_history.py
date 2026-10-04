"""Private CPU candidate journals and byte leases for the process boundary.

These classes do not construct an owner or admit wire observations. A future
host adapter must verify RAW/outcome and both fault fences before delivery.
The ordinary in-process SolveJournal and every public default remain intact.
"""
from __future__ import annotations

import math
import threading
from collections import deque

from ..control.fastening import FasteningFault, FasteningSolve


class _LatestJournal:
    """Child-local latest row and original publish/fault semantics, no history.

    Only FasteningController._latest may read this journal. Startup readiness,
    task observations and rest verifiers must use the host history instead.
    The unchanged controller still validates every solve before publish.
    """

    def __init__(self, binding):
        self.binding = binding
        self._condition = threading.Condition()
        self._error = None
        self._latest_row = None
        self._epoch = self._step = self._time = None

    def _check_error(self):
        with self._condition:
            if self._error:
                raise FasteningFault(self._error)

    def fail(self, reason):
        with self._condition:
            self._error = self._error or str(reason) or "producer failed without an error message"
            self._condition.notify_all()

    def publish(self, row):
        with self._condition:
            self._check_error()
            if (not isinstance(row, FasteningSolve) or row.binding_sha256 != self.binding.sha256
                    or self._step is not None and (row.epoch != self._epoch or row.step != self._step + 1
                        or not math.isclose(row.simulation_time_s - self._time,
                                            self.binding.dt_s, rel_tol=1e-6, abs_tol=1e-9))):
                self.fail("invalid identity/epoch/clock in producer stream")
                raise FasteningFault(self._error)
            # Metadata cannot follow later object.__setattr__ on an exposed row.
            self._epoch, self._step, self._time = row.epoch, row.step, row.simulation_time_s
            self._latest_row = row
            self._condition.notify_all()

    def read(self, after_step=None, *, timeout_s=0.):
        if (type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s < 0):
            raise ValueError("invalid latest-journal read timeout")
        if after_step is not None:
            raise FasteningFault("child latest journal cannot provide history; use the host reader")
        with self._condition:
            self._condition.wait_for(lambda: self._error or self._latest_row is not None, timeout=timeout_s)
            self._check_error()
            return () if self._latest_row is None else (self._latest_row,)


class _ByteLeases:
    """One charge per immutable host packet, across archive/ring/reader leases.

    Internal callers must keep a lease for every retained payload reference.
    This is not an API that hands uncharged bytes to arbitrary callers. Decoded
    public records belong to the separately bounded working-set budget.
    """

    def __init__(self, *, capacity, byte_capacity, frame_bytes,
                 lease_capacity=98304, batch_capacity=8):
        for value, upper in ((capacity, 65536), (byte_capacity, 24 * 1024**3),
                             (frame_bytes, 16 * 1024**2), (lease_capacity, 98304),
                             (batch_capacity, 8)):
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError("invalid host lease bounds")
        if frame_bytes > byte_capacity:
            raise ValueError("frame exceeds host lease byte capacity")
        self.capacity, self.byte_capacity, self.frame_bytes = capacity, byte_capacity, frame_bytes
        self.lease_capacity, self.batch_capacity = lease_capacity, batch_capacity
        self._lock = threading.RLock()
        self._entries = {}
        self._bytes = 0
        self._leases = self._batches = 0
        self._last_sequence = 0
        self._fault = None

    def _check(self):
        if self._fault:
            raise FasteningFault(self._fault)

    def _refuse(self, reason):
        self._fault = self._fault or reason
        raise FasteningFault(self._fault)

    def _new_lease(self, sequence):
        # Called under the store lock, including active iterator references.
        if self._leases >= self.lease_capacity:
            self._refuse("host payload lease metadata capacity reached")
        lease = _ByteLease(self, sequence)
        self._leases += 1
        return lease

    def _open_batch(self):
        # Empty snapshots consume a slot too: no unbounded empty objects.
        self._check()
        if self._batches >= self.batch_capacity:
            self._refuse("host history batch metadata capacity reached")
        self._batches += 1

    def _close_batch(self):
        with self._lock:
            if self._batches <= 0:
                self._refuse("duplicate history batch release")
            self._batches -= 1

    def register(self, sequence, payload):
        with self._lock:
            self._check()
            if (type(sequence) is not int or sequence != self._last_sequence + 1
                    or type(payload) is not bytes or not 0 < len(payload) <= self.frame_bytes):
                self._refuse("invalid host RAW sequence or payload")
            if len(self._entries) == self.capacity or self._bytes + len(payload) > self.byte_capacity:
                self._refuse("host RAW/history byte or slot capacity reached")
            lease = self._new_lease(sequence)
            try:
                self._entries[sequence] = [payload, 1]
            except BaseException:
                self._leases -= 1
                self._fault = self._fault or "host payload registration failed"
                raise
            self._bytes += len(payload)
            self._last_sequence = sequence
            return lease

    def _retain(self, sequence):
        with self._lock:
            self._check()
            if sequence not in self._entries:
                self._refuse("payload lease no longer exists")
            lease = self._new_lease(sequence)
            self._entries[sequence][1] += 1
            return lease

    def _release(self, sequence):
        # Cleanup remains possible after a sticky capacity/transport fault.
        with self._lock:
            if sequence not in self._entries:
                self._refuse("duplicate payload lease release")
            entry = self._entries[sequence]
            entry[1] -= 1
            self._leases -= 1
            if entry[1] == 0:
                self._bytes -= len(entry[0])
                del self._entries[sequence]

    @property
    def retained_bytes(self):
        with self._lock:
            return self._bytes

    @property
    def retained_metadata(self):
        with self._lock:
            return {"entries": len(self._entries), "leases": self._leases,
                    "batches": self._batches}

    def require_drained(self):
        with self._lock:
            if self._entries or self._bytes or self._leases or self._batches:
                self._refuse("host payload leases remain at closure")
            self._check()


class _ByteLease:
    """Internal owned reference; explicit release, no destructor/GC dependency."""
    __slots__ = ("_closed", "_store", "sequence")

    def __init__(self, store, sequence):
        self._store, self.sequence, self._closed = store, sequence, False

    def _check(self):
        if self._closed:
            raise FasteningFault("payload lease already released")

    def retain(self):
        with self._store._lock:
            self._check()
            return self._store._retain(self.sequence)

    def _payload(self):
        with self._store._lock:
            self._check()
            self._store._check()
            return self._store._entries[self.sequence][0]

    def close(self):
        with self._store._lock:
            self._check()
            self._closed = True
            self._store._release(self.sequence)


class _AcceptedHistory:
    """Host ring references accepted packets; pinned batches retain byte charge.

    publish() is private to the future RAW/outcome consumer. It must only be
    called after validating the explicit ACCEPTED outcome for this exact packet.
    It does not confer acceptance or execute a physical/freshness check itself.
    """

    def __init__(self, binding, store, *, capacity=20000):
        if type(store) is not _ByteLeases or type(capacity) is not int or not 3 <= capacity <= 20000:
            raise ValueError("invalid accepted-history capacity/store")
        self.binding, self.store, self.capacity = binding, store, capacity
        self._ring = deque()
        self._metadata = None
        self._closed = False

    def publish(self, lease, row):
        with self.store._lock:
            self.store._check()
            if self._closed or type(lease) is not _ByteLease or lease._store is not self.store:
                self.store._refuse("unowned accepted-history publication")
            lease._check()
            previous = self._metadata
            if (not isinstance(row, FasteningSolve) or row.binding_sha256 != self.binding.sha256
                    or previous is not None and (row.epoch != previous[0] or row.step != previous[1] + 1
                        or lease.sequence != previous[3] + 1
                        or not math.isclose(row.simulation_time_s - previous[2],
                                            self.binding.dt_s, rel_tol=1e-6, abs_tol=1e-9))):
                self.store._refuse("invalid identity/epoch/clock in accepted host history")
            retained = lease.retain()
            try:
                self._ring.append((row.step, retained))
            except BaseException:
                retained.close()
                self.store._fault = self.store._fault or "accepted host history publication failed"
                raise
            self._metadata = row.epoch, row.step, row.simulation_time_s, lease.sequence
            if len(self._ring) > self.capacity:
                # This releases only this ring's reference. Archive and reader
                # leases keep both the payload and its byte charge alive.
                _, expired = self._ring.popleft()
                expired.close()

    def snapshot(self, after_step):
        with self.store._lock:
            self.store._check()
            if self._closed or type(after_step) is not int or after_step < 0:
                raise FasteningFault("invalid or closed host history cursor")
            if self._ring and after_step < self._ring[0][0] - 1:
                raise FasteningFault("reader lost solves to journal capacity")
            self.store._open_batch()
            leases = []
            try:
                for step, lease in self._ring:
                    if step > after_step:
                        retained = lease.retain()
                        try:
                            leases.append(retained)
                        except BaseException:
                            retained.close()
                            raise
                return _HistoryBatch(self.store, tuple(leases))
            except BaseException:
                try:
                    for lease in leases:
                        lease.close()
                finally:
                    self.store._close_batch()
                raise

    def close(self):
        with self.store._lock:
            if not self._closed:
                self._closed = True
                while self._ring:
                    _, lease = self._ring.popleft()
                    lease.close()


class _HistoryBatch:
    """Finite private lease batch; close explicitly after the fenced iteration."""
    __slots__ = ("_closed", "_leases", "_lock", "_store")

    def __init__(self, store, leases):
        self._store = store
        self._leases, self._closed = leases, False
        self._lock = threading.RLock()

    def __enter__(self):
        if self._closed:
            raise FasteningFault("history batch already closed")
        return self

    def __exit__(self, *_):
        self.close()

    def __iter__(self):
        # Do not retain a tuple iterator over all leases: close() must drop
        # their metadata even while one decoded row remains in flight.
        for index in range(len(self._leases)):
            with self._lock:
                if self._closed:
                    raise FasteningFault("history batch already closed")
                active = self._leases[index].retain()
            try:
                yield active._payload()
            finally:
                active.close()

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                error = None
                try:
                    for lease in self._leases:
                        try:
                            lease.close()
                        except BaseException as exc:  # noqa: BLE001 - rethrow after all owned releases
                            error = error or exc
                finally:
                    self._leases = ()
                    self._store._close_batch()
                if error is not None:
                    raise error
