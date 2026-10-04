"""Bounded host retention of already admitted threading evidence, CPU candidate.

This does not wire a process owner or replace verify_threading. The future
adapter must close each finite view explicitly, then close its archive. No
destructor or cyclic-GC event returns charged capacity.
"""
from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from dataclasses import fields
from operator import index as as_index

from ..control.fastening import FasteningFault
from .factory_process_wire import WireFault, WireLimits, decode, encode
from .threading_verification import ThreadSample

_NAMES = tuple(field.name for field in fields(ThreadSample))
_KEYS = frozenset(_NAMES)
_LIMITS = WireLimits(frame_bytes=2048, nodes=40, depth=3, text_bytes=512)


def _values(sample):
    if type(sample) is not ThreadSample or vars(sample).keys() != _KEYS:
        raise WireFault("exact complete threading sample required")
    values = tuple(getattr(sample, name) for name in _NAMES)
    for value in (sample.epoch, sample.fastener_id, sample.fixture_id):
        if type(value) is not str or len(value) > 512 or len(value.encode("utf-8")) > 512:
            raise WireFault("threading identity exceeds schema")
    for value in (sample.step, sample.thread_contacts, sample.tool_contacts):
        if type(value) is not int or not 0 <= value < 2**63:
            raise WireFault("invalid threading clock/count schema")
    if type(sample.time_s) not in (int, float) or not math.isfinite(sample.time_s):
        raise WireFault("invalid threading time schema")
    for vector, count in ((sample.fastener_position_m, 3), (sample.fastener_quaternion_xyzw, 4),
                          (sample.fixture_position_m, 3), (sample.fixture_quaternion_xyzw, 4)):
        if (type(vector) is not tuple or len(vector) != count
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in vector)):
            raise WireFault("invalid threading vector schema")
    return values


def _pack_sample(sample):
    # Maximum: 3*(5+512) identity bytes, 18 nine-byte numeric leaves,
    # four five-byte vector headers and two tuple headers/version: < 2048.
    return encode((1, _values(sample)), _LIMITS)


def _unpack_sample(payload):
    value = decode(payload, _LIMITS)
    if (type(value) is not tuple or len(value) != 2 or type(value[0]) is not int or value[0] != 1
            or type(value[1]) is not tuple or len(value[1]) != len(_NAMES)):
        raise WireFault("unknown threading sample schema")
    sample = ThreadSample(*value[1])
    _values(sample)
    return sample


class _ThreadArchive(Sequence):
    """Append-only bytes with bounded finite views, never a decoded-row cache.

    check/fail are the future host adapter's sticky fault observation path.
    They must not apply a fresh check_solve(now) to historical samples or renew
    their timestamps. Existing current-observation gates run before append.
    """

    def __init__(self, check, fail, *, capacity=32768, byte_capacity=64 * 1024**2,
                 view_capacity=8, reader_capacity=8):
        for value, maximum in ((capacity, 32768), (byte_capacity, 64 * 1024**2),
                               (view_capacity, 8), (reader_capacity, 8)):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError("invalid host threading archive bounds")
        if not callable(check) or not callable(fail):
            raise TypeError("threading archive requires fault callbacks")
        self._check, self._fail = check, fail
        self._capacity, self._byte_capacity = capacity, byte_capacity
        self._view_capacity, self._reader_capacity = view_capacity, reader_capacity
        self._records = [None] * capacity
        self._count = self._bytes = self._views = self._active = 0
        self._closed = False
        self._fault = None
        self._lock = threading.RLock()

    def _check_fault(self):
        if self._fault is not None:
            raise FasteningFault(self._fault)
        self._check()

    def _refuse(self, error):
        # Never format an arbitrary error; preserve the primary failure cause.
        reason = "host threading retention failed: " + type(error).__name__
        self._fault = self._fault or reason
        self._fail(self._fault)
        raise FasteningFault(self._fault) from error

    def _check_open(self):
        self._check_fault()
        if self._closed:
            raise FasteningFault("host threading archive already closed")

    def append(self, sample):
        with self._lock:
            self._check_open()
            try:
                if self._count >= self._capacity:
                    raise OverflowError("host threading row capacity reached")
                payload = _pack_sample(sample)
                if self._bytes + len(payload) > self._byte_capacity:
                    raise OverflowError("host threading byte capacity reached")
            except Exception as error:  # noqa: BLE001 - latch and rethrow every retention failure
                self._refuse(error)
            self._check_fault()
            self._records[self._count] = payload
            self._bytes += len(payload)
            self._count += 1

    def __len__(self):
        with self._lock:
            return self._count

    def __getitem__(self, index):
        with self._lock:
            self._check_open()
            if isinstance(index, slice):
                return self._view(range(self._count)[index])
            position = as_index(index)
            if position < 0:
                position += self._count
            if not 0 <= position < self._count:
                raise IndexError("threading history index out of range")
        return self._read(position)

    def _view(self, positions):
        with self._lock:
            self._check_open()
            if self._views >= self._view_capacity:
                self._refuse(OverflowError("threading view metadata capacity reached"))
            view = _ThreadView(self, positions)
            self._views += 1
            return view

    def _read(self, position, view=None):
        with self._lock:
            self._check_open()
            if view is not None:
                view._check_open()
            if self._active >= self._reader_capacity:
                self._refuse(OverflowError("threading concurrent reader capacity reached"))
            payload = self._records[position]
            self._active += 1
        try:
            try:
                sample = _unpack_sample(payload)
            except Exception as error:  # noqa: BLE001 - latch and rethrow every decode failure
                with self._lock:
                    self._refuse(error)
            with self._lock:
                self._check_open()  # No fault or close during decode may escape.
                if view is not None:
                    view._check_open()
                return sample
        finally:
            payload = None
            with self._lock:
                self._active -= 1
                self._release_if_closed()

    def _release_if_closed(self):
        if self._closed and self._views == self._active == 0:
            self._records = []
            self._count = self._bytes = 0

    def close(self):
        with self._lock:
            self._closed = True
            self._release_if_closed()

    @property
    def retained(self):
        with self._lock:
            return {"rows": self._count, "bytes": self._bytes,
                    "views": self._views, "active_readers": self._active}

    def require_drained(self):
        with self._lock:
            if not self._closed or self._count or self._bytes or self._views or self._active:
                self._refuse(RuntimeError("host threading retention remains at closure"))
            self._check_fault()


class _ThreadView(Sequence):
    """Finite range over one charged archive; slicing never copies payloads."""
    __slots__ = ("_archive", "_closed", "_positions")

    def __init__(self, archive, positions):
        self._archive, self._positions, self._closed = archive, positions, False

    def _check_open(self):
        if self._closed:
            raise FasteningFault("threading view already closed")
        self._archive._check_open()

    def __len__(self):
        with self._archive._lock:
            self._check_open()
            return len(self._positions)

    def __getitem__(self, index):
        with self._archive._lock:
            self._check_open()
            position = self._positions[index]
            if isinstance(index, slice):
                return self._archive._view(position)
        return self._archive._read(position, self)

    def __enter__(self):
        with self._archive._lock:
            self._check_open()
            return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        with self._archive._lock:
            if not self._closed:
                self._closed = True
                self._archive._views -= 1
                self._archive._release_if_closed()
