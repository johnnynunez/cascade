"""Bind render-product time to owned, completed-update physics snapshots.

Isaac's current-time reader truncates double times to integer microseconds.
The render product can instead expose nanoseconds. Only that documented
microsecond conversion is allowed here; other clock resolutions require an
exact rational match. This is timestamp representation, not a freshness budget.
"""
from collections import OrderedDict
from copy import deepcopy
from fractions import Fraction
import math


_SDK_DENOMINATOR = 1_000_000


class ClockDiscontinuity(ValueError):
    """The history was cleared because its clock or producer changed."""


def reference_key(reference):
    """Validate a raw SDK/render-product pair without losing its precision."""
    if not isinstance(reference, (tuple, list)) or len(reference) != 2:
        raise ValueError("reference must be a numerator/denominator pair")
    numerator, denominator = reference
    if (type(numerator) is not int or type(denominator) is not int
            or numerator < 0 or denominator <= 0):
        raise ValueError("reference must contain nonnegative/positive integers")
    return Fraction(numerator, denominator)


def _finite_time(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be a finite nonnegative time")
    return float(value)


def _state_payload(payload):
    value = deepcopy(payload)
    # This one field is the publication anchor, not a physical measurement.
    # Do not recursively remove 't': another field may represent real state.
    proprioception = value.get("proprioception")
    if isinstance(proprioception, dict):
        proprioception.pop("t", None)
    return value


class FrameHistory:
    """Bounded snapshots; contradictory samples cannot replace old evidence."""

    def __init__(self, maxlen=256):
        if type(maxlen) is not int or maxlen <= 0:
            raise ValueError("maxlen must be a positive integer")
        self.maxlen = maxlen
        self.clear()

    def clear(self):
        self._entries = OrderedDict()
        self._ambiguous = set()
        self._last = None
        self._resolution = None

    def record(self, *, reference, simulation_time, physics_step,
               started_monotonic, finished_monotonic, epoch, payload):
        """Own a snapshot, preserving the first anchor for an identical repeat.

        Return True only for a new key. A conflicting repeat poisons that key:
        resolve returns None until it is evicted or the history is cleared.
        Discontinuities clear the entire history and raise; callers must also
        invalidate previously published frames before collecting a new epoch.
        """
        key = reference_key(reference)
        simulation_time = _finite_time(simulation_time, "simulation_time")
        started_monotonic = _finite_time(started_monotonic, "started_monotonic")
        finished_monotonic = _finite_time(finished_monotonic, "finished_monotonic")
        if finished_monotonic < started_monotonic:
            raise ValueError("finished_monotonic precedes started_monotonic")
        if type(physics_step) is not int or physics_step < 0:
            raise ValueError("physics_step must be a nonnegative integer")
        if not isinstance(epoch, str) or not epoch:
            raise ValueError("epoch must be a nonempty string")
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dictionary")

        raw_reference = tuple(reference)
        last = self._last
        if last is not None and (
                epoch != last["epoch"]
                or raw_reference[1] != self._resolution
                or key < reference_key(last["reference"])
                or physics_step < last["physics_step"]
                or simulation_time < last["simulation_time"]
                or started_monotonic < last["started_monotonic"]
                or finished_monotonic < last["finished_monotonic"]):
            self.clear()
            raise ClockDiscontinuity("frame clock, physics clock, or producer changed")

        entry = dict(reference=raw_reference, simulation_time=simulation_time,
                     physics_step=physics_step, started_monotonic=started_monotonic,
                     finished_monotonic=finished_monotonic, epoch=epoch,
                     payload=deepcopy(payload))
        self._last = {k: v for k, v in entry.items() if k != "payload"}
        self._resolution = raw_reference[1]  # Keep RAW resolution, never Fraction.denominator.
        existing = self._entries.get(key)
        if existing is not None:
            if (existing["physics_step"] != physics_step
                    or existing["simulation_time"] != simulation_time
                    or _state_payload(existing["payload"]) != _state_payload(entry["payload"])):
                self._ambiguous.add(key)
            return False
        self._entries[key] = entry
        while len(self._entries) > self.maxlen:
            old, _ = self._entries.popitem(last=False)
            self._ambiguous.discard(old)
        return True

    def resolve(self, *, reference, simulation_time, epoch):
        """Return an owned matching snapshot, or None for missing/ambiguous data.

        The render-product clock selects the snapshot. Its simulation-time
        annotator only checks coherence; it can never select a current pose
        as a fallback when the render-product clock is absent or unknown.
        """
        try:
            key = reference_key(reference)
            simulation_time = _finite_time(simulation_time, "simulation_time")
        except (ValueError, TypeError, OverflowError):
            return None
        if self._resolution == _SDK_DENOMINATOR:
            key = Fraction(key.numerator * _SDK_DENOMINATOR // key.denominator,
                           _SDK_DENOMINATOR)
        entry = self._entries.get(key)
        if entry is None or key in self._ambiguous or entry["epoch"] != epoch:
            return None
        if self._resolution == _SDK_DENOMINATOR:
            try:
                coherent = (math.trunc(simulation_time * _SDK_DENOMINATOR)
                            == math.trunc(entry["simulation_time"] * _SDK_DENOMINATOR))
            except (ValueError, OverflowError):
                return None
        else:
            coherent = simulation_time == entry["simulation_time"]
        return deepcopy(entry) if coherent else None
