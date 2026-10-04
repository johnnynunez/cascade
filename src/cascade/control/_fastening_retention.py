"""Private, in-process retention of already validated Factory observations.

Only exact known records with built-in immutable leaves enter this codec. Its
constructor creates the pickle bytes; no file, transport or encoded-input API
loads caller-provided pickle. This is not an untrusted serialization format.
Reads reconstruct one record at a time, never a full live contact history.
"""
from collections.abc import Sequence
from dataclasses import dataclass, fields
import math
import pickle

from .fastening import FasteningFault, FasteningSolve, SolvedPair
from .fastening_seat import SeatingSolve, ShoulderContact
from ..sim.threading_verification import ThreadSample


_FIELDS = {kind: tuple(field.name for field in fields(kind)) for kind in
           (FasteningSolve, SeatingSolve, SolvedPair, ShoulderContact, ThreadSample)}
_FIELD_KEYS = {kind: frozenset(names) for kind, names in _FIELDS.items()}


def _trusted(value):
    kind = type(value)
    if kind in (str, int, float, bool, type(None)):
        return
    if kind is tuple:
        for item in value:
            _trusted(item)
        return
    if kind in _FIELDS and vars(value).keys() == _FIELD_KEYS[kind]:
        for name in _FIELDS[kind]:
            _trusted(getattr(value, name))
        return
    raise TypeError("compact Factory retention requires exact immutable record types")


def _pack(value):
    if type(value) not in (FasteningSolve, SeatingSolve, ThreadSample):
        raise TypeError("unknown compact Factory record")
    _trusted(value)
    return pickle.dumps(value, protocol=5)


def _unpack(payload):
    # Only _pack produces these internal references. No encoded-input API is
    # exposed to callers, files or transports; this is not a safe pickle reader.
    if type(payload) is not bytes:
        raise TypeError("exact private Factory bytes required")
    return pickle.loads(payload)


class _RetainedRing:
    """Preallocated reference/step slots, owned under the journal condition.

    Exact bytes and scalar metadata need no per-row tracked Python wrapper.
    Captured metadata supplies epoch/time validation before any slot is changed.
    A cursor snapshot owns bytes references across subsequent ring overwrites.
    """
    __slots__ = ("_payloads", "_steps", "_epochs", "_times", "_capacity", "_start", "_count")

    def __init__(self, capacity):
        self._payloads = [None] * capacity
        self._steps = [0] * capacity
        self._epochs = [None] * capacity
        self._times = [0.] * capacity
        self._capacity, self._start, self._count = capacity, 0, 0

    def __len__(self):
        return self._count

    @property
    def first_step(self):
        return self._steps[self._start]

    @property
    def last_step(self):
        return self._steps[(self._start + self._count - 1) % self._capacity]

    def follows(self, value, dt_s):
        slot = (self._start + self._count - 1) % self._capacity
        return (value.epoch == self._epochs[slot] and value.step == self._steps[slot] + 1
                and math.isclose(value.simulation_time_s - self._times[slot],
                                 dt_s, rel_tol=1e-6, abs_tol=1e-9))

    def append(self, value):
        payload = _pack(value)  # A failed codec must not overwrite a valid slot.
        slot = (self._start + self._count) % self._capacity
        self._payloads[slot], self._steps[slot] = payload, value.step
        self._epochs[slot], self._times[slot] = value.epoch, value.simulation_time_s
        if self._count == self._capacity:
            self._start = (self._start + 1) % self._capacity
        else:
            self._count += 1

    def after(self, step):
        # Journal validation guarantees consecutive steps, including across
        # wraparound. Avoid constructing metadata objects for every retained row.
        offset = max(0, step - self.first_step + 1)
        return tuple(self._payloads[(self._start + i) % self._capacity]
                     for i in range(offset, self._count))


def _failure(fail, error):
    # Even an exception with broken formatting must leave a sticky refusal.
    reason = "compact Factory retention failed: " + type(error).__name__
    fail(reason)
    raise FasteningFault(reason) from error


def _expand(record, check, fail):
    check()
    try:
        value = _unpack(record)
    except Exception as error:
        _failure(fail, error)
    check()
    return value


@dataclass(frozen=True, slots=True, init=False)
class _RetainedBatch(Sequence):
    """Immutable snapshot of packed references, with no decoded-row cache.

    Decoding is outside the journal condition lock. GC remains process-wide;
    this does not establish a latency bound or renew any observation deadline.
    The journal fault is checked again after each decode before exposing it.
    """
    _records: tuple
    _check: object
    _fail: object

    def __init__(self, records, check, fail):
        object.__setattr__(self, '_records', tuple(records))
        object.__setattr__(self, '_check', check)
        object.__setattr__(self, '_fail', fail)

    def __len__(self):
        return len(self._records)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return _RetainedBatch(self._records[index], self._check, self._fail)
        record = self._records[index]  # Preserve ordinary sequence IndexError.
        return _expand(record, self._check, self._fail)


class _ThreadHistory(Sequence):
    """Bounded append-only private history; slices keep only packed references."""
    __slots__ = ("_records", "_capacity", "_count", "_check", "_fail")

    def __init__(self, capacity, check, fail):
        self._records, self._capacity = [None] * capacity, capacity
        self._count = 0
        self._check, self._fail = check, fail

    def append(self, sample):
        self._check()
        try:
            if self._count >= self._capacity:
                raise OverflowError("compact threading history capacity reached")
            if type(sample) is not ThreadSample:
                raise TypeError("exact threading sample required")
            packed = _pack(sample)
        except Exception as error:
            _failure(self._fail, error)
        self._check()
        self._records[self._count] = packed
        self._count += 1

    def __len__(self):
        return self._count

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, stride = index.indices(self._count)
            return _RetainedBatch((self._records[i] for i in range(start, stop, stride)),
                                  self._check, self._fail)
        # Index only the populated prefix, preserving ordinary Sequence rules.
        from operator import index as as_index
        index = as_index(index)
        if index < 0:
            index += self._count
        if not 0 <= index < self._count:
            raise IndexError("threading history index out of range")
        return _expand(self._records[index], self._check, self._fail)
