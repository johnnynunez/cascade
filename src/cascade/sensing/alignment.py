"""Capture-time alignment of sample streams (backlog B50).

A camera frame and an IMU / joint sample describe one instant only when they
were CAPTURED at (nearly) the same time on the same clock. This module pairs a
reference capture time with the nearest recorded sample of another stream and
says how good the pairing is, in one of four states:

- `aligned`: a sample within `max_skew_s` of the reference time (the bound is
  inclusive), on a comparable clock, with no reason for doubt.
- `uncertain`: a sample is returned, but it cannot be taken as the value at the
  reference instant: its clock is not comparable with the reference's (the skew
  is unknowable), its measurement is flagged saturated, or the stream's own
  measured rate times the skew (a first-order displacement estimate, not a
  bound) exceeds a declared tolerance.
- `stale`: the nearest sample is farther than `max_skew_s` (or, on an
  incomparable clock, older than the sensor's max age). Its identity and skew
  are reported; its VALUE is withheld (`Pairing.sample is None`).
- `missing`: no sample at all. Nothing is made up.

Two rules hold everywhere: nothing is interpolated or extrapolated (a returned
sample is one recorded sample, unchanged), and missing data is never upgraded
to present (a stale or missing pairing has no value; a channel a producer did
not measure stays absent, never zero). Ties keep the EARLIER sample.

Generalized from B39's link self-mask (`perception/link_mask.py`), which pairs
each camera frame with the arm's joint sample nearest its capture time and now
uses `SampleBuffer` and `classify` from here. The sensing domain's opt-in
`read_aligned` tool (`domain.py`) applies `align_capture` to admitted
`ObservationEnvelope`s. Stdlib only: imported by the minimal install.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import threading

ALIGNED = "aligned"
STALE = "stale"
MISSING = "missing"
UNCERTAIN = "uncertain"
STATUSES = (ALIGNED, STALE, MISSING, UNCERTAIN)

#: Clock domains that are THIS process's clock (`time.monotonic`), shared by
#: every in-process provider: comparable across provider epochs (a synthetic
#: provider mints one epoch per instance). Any other domain, e.g.
#: `simulation`, is one clock per epoch: a reset world restarts its time.
LOCAL_CLOCK_DOMAINS = frozenset({"monotonic"})

#: A rate unit and the displacement unit its rate x |skew| estimate is in.
_DISPLACEMENT = {"rad/s": "rad", "m/s": "m"}

#: Upper bound of `max_skew_s`: a second apart is not one instant.
MAX_SKEW_LIMIT_S = 1.0


def nearest(items, t, time_of=lambda item: item[0]):
    """`(item, skew_s)` of the item whose time is nearest `t`, or None.

    `skew_s = time_of(item) - t`. Ties keep the EARLIER item in iteration
    order. The item is returned as recorded: never interpolated.
    """
    best = None
    for item in items:
        skew = float(time_of(item)) - t
        if best is None or abs(skew) < abs(best[1]):
            best = (item, skew)
    return best


class SampleBuffer:
    """Bounded, thread-safe history of `(time, sample)`; nearest lookups only."""

    def __init__(self, maxlen: int = 64):
        self._buf: deque = deque(maxlen=int(maxlen))
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def add(self, t, sample) -> bool:
        """Keep `sample` captured at `t`; False (nothing kept) for a non-finite time."""
        try:
            t = float(t)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(t):
            return False
        with self._lock:
            self._buf.append((t, sample))
        return True

    def nearest(self, t: float):
        """`(sample, skew_s)` of the sample nearest `t` (skew = sample time - t), or None."""
        with self._lock:
            items = list(self._buf)
        got = nearest(items, t)
        return None if got is None else (got[0][1], got[1])


@dataclass(frozen=True)
class Pairing:
    """One stream's pairing with a reference capture time.

    `candidate` is the sample that was considered (its identity is reported
    even when stale); `sample` is the value a consumer may use: None unless the
    pairing is aligned or uncertain. `motion` maps a displacement unit ("rad",
    "m") to rate x |skew| from the candidate's own measured rate.
    """
    status: str
    reference_time_s: float
    skew_s: float | None = None
    candidate: object = None
    reasons: tuple = ()
    motion: dict = field(default_factory=dict)

    @property
    def sample(self):
        return self.candidate if self.status in (ALIGNED, UNCERTAIN) else None

    def as_dict(self) -> dict:
        return {"status": self.status, "skew_s": self.skew_s, "reasons": list(self.reasons),
                "motion": dict(self.motion)}


def classify(reference_time_s: float, found, *, max_skew_s: float, rates=None, tolerances=None,
             saturated=None) -> Pairing:
    """The pairing of `found = (sample, skew_s)` (a `nearest` result over
    samples on the reference's clock) or None with the reference time.

    `rates`: displacement unit -> measured rate magnitude of the sample;
    `tolerances`: displacement unit -> largest acceptable rate x |skew| (None
    or absent: not checked); `saturated`: the sample's own saturation flag.
    """
    if found is None:
        return Pairing(MISSING, reference_time_s, reasons=("no_sample",))
    sample, skew = found
    if abs(skew) > max_skew_s:
        return Pairing(STALE, reference_time_s, skew, sample, ("skew_exceeds_bound",))
    motion = {unit: rate * abs(skew) for unit, rate in (rates or {}).items()}
    reasons = []
    if saturated is True:
        reasons.append("saturated")
    if any(limit is not None and unit in motion and motion[unit] > limit
           for unit, limit in (tolerances or {}).items()):
        reasons.append("motion_exceeds_tolerance")
    return Pairing(UNCERTAIN if reasons else ALIGNED, reference_time_s, skew, sample, tuple(reasons), motion)


# ── typed sensor captures (cascade.sensing.models) ──────────────────────────


def _positive(block, key, *, required=False, maximum=None):
    value = block.get(key)
    if value is None and not required:
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value <= 0 or (maximum is not None and value > maximum)):
        bound = "" if maximum is None else f" (at most {maximum})"
        raise ValueError(f"alignment.{key} must be a positive number{bound}, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class AlignmentPolicy:
    """The sensors-domain `alignment:` block (opt-in `read_aligned`)."""
    max_skew_s: float
    max_rotation_rad: float | None = None
    max_translation_m: float | None = None

    @classmethod
    def from_profile(cls, block) -> "AlignmentPolicy":
        if not isinstance(block, dict):
            raise ValueError(f"alignment must be a mapping, got {block!r}")
        unknown = set(block) - {"max_skew_s", "max_rotation_rad", "max_translation_m"}
        if unknown:
            raise ValueError(f"unknown alignment settings: {sorted(unknown)}")
        return cls(_positive(block, "max_skew_s", required=True, maximum=MAX_SKEW_LIMIT_S),
                   _positive(block, "max_rotation_rad"), _positive(block, "max_translation_m"))

    @property
    def tolerances(self) -> dict:
        return {"rad": self.max_rotation_rad, "m": self.max_translation_m}

    def as_dict(self) -> dict:
        return {"max_skew_s": self.max_skew_s, "max_rotation_rad": self.max_rotation_rad,
                "max_translation_m": self.max_translation_m,
                "selection": "nearest_admitted_capture", "interpolation": "none"}


def comparable(reference, candidate) -> bool:
    """Whether two envelopes' capture times are on ONE clock instance."""
    if reference.clock_domain != candidate.clock_domain:
        return False
    return reference.clock_domain in LOCAL_CLOCK_DOMAINS or reference.epoch == candidate.epoch


def _merge(out: dict, unit: str, magnitude: float) -> None:
    out[unit] = max(out.get(unit, 0.0), magnitude)


def rates(payload) -> dict:
    """Displacement unit -> the largest measured rate magnitude in `payload`.

    IMU: |angular velocity| (rad/s); joints: the fastest joint, per unit (a
    multi-DoF joint's angular / linear components as vector norms). Payloads
    that measure no rate (images, contact, tactile) have none: {}.
    """
    from .models import (GeneralizedJointStatePayload, ImuPayload, JointStatePayload,
                         ProprioceptionPayload)
    out: dict = {}
    if isinstance(payload, ImuPayload):
        out["rad"] = math.sqrt(sum(w * w for w in payload.angular_velocity_rad_s))
    elif isinstance(payload, ProprioceptionPayload):
        out["rad"] = max(abs(v) for v in payload.velocity_rad_s)
    elif isinstance(payload, JointStatePayload):
        for joint in payload.joints:
            _merge(out, _DISPLACEMENT[joint.velocity_unit], abs(joint.velocity))
    elif isinstance(payload, GeneralizedJointStatePayload):
        for joint in payload.joints:
            groups: dict = {}
            for unit, v in zip(joint.coordinates["v_units"], joint.v):
                groups.setdefault(_DISPLACEMENT[unit], []).append(v)
            for unit, values in groups.items():
                _merge(out, unit, math.sqrt(sum(v * v for v in values)))
    return out


def absent_channels(payload) -> list:
    """Optional channels the producer did not measure (absent, never zero)."""
    from .models import (GeneralizedJointStatePayload, ImuPayload, JointStatePayload,
                         ProprioceptionPayload)
    if isinstance(payload, ImuPayload):
        return [] if payload.linear_acceleration_m_s2 is not None else ["linear_acceleration_m_s2"]
    if isinstance(payload, ProprioceptionPayload):
        return [] if payload.effort_nm is not None else ["effort_nm"]
    if isinstance(payload, (JointStatePayload, GeneralizedJointStatePayload)):
        return [f"{joint.joint_id}.effort" for joint in payload.joints if joint.effort is None]
    return []


def align_capture(reference, history, policy: AlignmentPolicy, *, now: float, max_age_s: float) -> Pairing:
    """Pair the admitted capture `reference` with ONE sensor's admitted
    captures `history` (oldest first, e.g. `SensorHub.history(sensor_id)`).

    Captures on the reference's clock (`comparable`) are classified by skew,
    saturation and motion. If none is comparable, the skew is unknowable: the
    LATEST capture is `uncertain` (skew None) while younger than `max_age_s` at
    local time `now`, else `stale`. No capture at all is `missing`.
    """
    t = reference.capture_time_s
    usable = [o for o in history if comparable(reference, o)]
    if usable:
        found = nearest(usable, t, time_of=lambda o: o.capture_time_s)
        payload = found[0].payload
        return classify(t, found, max_skew_s=policy.max_skew_s, rates=rates(payload),
                        tolerances=policy.tolerances, saturated=payload.metadata.saturated)
    if not history:
        return Pairing(MISSING, t, reasons=("no_admitted_capture",))
    latest = history[-1]
    if latest.age_s(now) > max_age_s:
        return Pairing(STALE, t, None, latest, ("clock_not_comparable", "older_than_max_age"))
    return Pairing(UNCERTAIN, t, None, latest, ("clock_not_comparable",))
