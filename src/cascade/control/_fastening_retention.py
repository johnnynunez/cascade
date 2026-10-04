"""Private, in-process retention of already validated Factory observations.

Only exact known records with built-in immutable leaves enter this codec. Its
constructor creates the pickle bytes; no file, transport or encoded-input API
loads caller-provided pickle. This is not an untrusted serialization format.
Reads reconstruct one record at a time, never a full live contact history.
"""
from collections.abc import Sequence
from dataclasses import dataclass, fields
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


@dataclass(frozen=True, slots=True, init=False)
class _PackedRecord:
    __payload: bytes
    step: int
    epoch: str
    simulation_time_s: float

    def __init__(self, value):
        if type(value) not in (FasteningSolve, SeatingSolve, ThreadSample):
            raise TypeError("unknown compact Factory record")
        _trusted(value)
        object.__setattr__(self, '_PackedRecord__payload', pickle.dumps(value, protocol=5))
        object.__setattr__(self, 'step', value.step)
        object.__setattr__(self, 'epoch', value.epoch)
        object.__setattr__(self, 'simulation_time_s',
                           value.time_s if type(value) is ThreadSample else value.simulation_time_s)

    def expand(self):
        return pickle.loads(self.__payload)


def _failure(fail, error):
    # Even an exception with broken formatting must leave a sticky refusal.
    reason = "compact Factory retention failed: " + type(error).__name__
    fail(reason)
    raise FasteningFault(reason) from error


def _expand(record, check, fail):
    check()
    try:
        value = record.expand()
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
    __slots__ = ("_records", "_capacity", "_check", "_fail")

    def __init__(self, capacity, check, fail):
        self._records, self._capacity = [], capacity
        self._check, self._fail = check, fail

    def append(self, sample):
        self._check()
        try:
            if len(self._records) >= self._capacity:
                raise OverflowError("compact threading history capacity reached")
            if type(sample) is not ThreadSample:
                raise TypeError("exact threading sample required")
            packed = _PackedRecord(sample)
        except Exception as error:
            _failure(self._fail, error)
        self._check()
        self._records.append(packed)

    def __len__(self):
        return len(self._records)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return _RetainedBatch(self._records[index], self._check, self._fail)
        return _expand(self._records[index], self._check, self._fail)
