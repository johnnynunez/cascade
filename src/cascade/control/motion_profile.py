"""One nominal curve for actuator targets and conservative safety sampling.

The rational union retains every legacy 50 Hz sample exactly, including when
floor(duration*rate) makes command/safety counts incommensurate. Validation is
still discrete geometry checking, not proof of continuous collision freedom.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ..types import SafetyViolation

LEGACY_SAFETY_RATE_HZ = 50.0
MAX_PROFILE_STEPS = 100000


def positive_rate(value):
    if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
        raise SafetyViolation("motion rate must be finite and positive")
    return float(value)


def resolve_motion_rate(arm, explicit=None):
    return positive_rate(getattr(arm, "motion_rate_hz", 50.0) if explicit is None else explicit)


def min_jerk(s: float) -> float:
    return 10 * s**3 - 15 * s**4 + 6 * s**5


@dataclass(frozen=True)
class ProfileTarget:
    q: np.ndarray
    dt: float
    # Actual edge first, original 50 Hz edges, and virtual curve subedges.
    # Each tuple is (previous q, next q, its own nominal physical dt).
    checks: tuple


def profile_counts(duration_s, rate_hz, *, max_steps=MAX_PROFILE_STEPS):
    rate_hz = positive_rate(rate_hz)
    if type(duration_s) not in (int, float) or not np.isfinite(duration_s) or duration_s <= 0:
        raise SafetyViolation("motion duration must be finite and positive")
    counts = []
    for rate in (rate_hz, LEGACY_SAFETY_RATE_HZ):
        count = duration_s * rate
        if not np.isfinite(count) or count > max_steps:
            raise SafetyViolation("motion profile exceeds waypoint budget")
        counts.append(max(2, int(count)))
    return tuple(counts)


def nominal_profile(start, target, duration_s, rate_hz=50.0, *, max_steps=MAX_PROFILE_STEPS):
    """Keep original edges too: escape tolerances apply per complete edge."""
    start, target = np.asarray(start, dtype=float), np.asarray(target, dtype=float)
    if (start.ndim != 1 or start.shape != target.shape or not start.size
            or not np.isfinite(start).all() or not np.isfinite(target).all()):
        raise SafetyViolation("motion profile requires finite matching joint vectors")
    def evaluate(fraction):
        return start + (target - start) * min_jerk(float(fraction))

    yield from curve_profile(evaluate, duration_s, rate_hz, max_steps=max_steps)


def curve_profile(evaluate, duration_s, rate_hz=50.0, *, max_steps=MAX_PROFILE_STEPS):
    """Sample one curve, retaining command, 50 Hz and union safety edges.

    ``evaluate`` receives a normalized rational time. It must evaluate the
    original curve, not interpolate exported samples or solve another route.
    The caller owns its lifetime; execution callers freeze the result first.
    """
    count, legacy = profile_counts(duration_s, rate_hz, max_steps=max_steps)
    dt = duration_s / count
    previous = np.asarray(evaluate(Fraction(0)), dtype=float).copy()
    shape = previous.shape
    if previous.ndim != 1 or not previous.size or not np.isfinite(previous).all():
        raise SafetyViolation("trajectory requires finite joint vectors")

    def position(fraction):
        q = np.asarray(evaluate(fraction), dtype=float)
        if q.shape != shape or not np.isfinite(q).all():
            raise SafetyViolation("trajectory requires finite matching joint vectors")
        return q.copy()

    lower = Fraction(0)
    for index in range(1, count + 1):
        upper = Fraction(index, count)
        q = position(upper)
        checks = [(previous, q, dt)]
        seen_edges = {(lower, upper)}

        def add_edge(a, b, edge_dt):
            if (a, b) in seen_edges:
                return
            seen_edges.add((a, b))
            qa = position(a)
            qb = position(b)
            checks.append((qa, qb, edge_dt))

        inner = []
        # Include every legacy edge intersected by this command interval,
        # even one ending just AFTER the command. Otherwise a coarse target
        # could enter partway into an edge that the legacy gate would refuse.
        first = (lower.numerator * legacy) // lower.denominator + 1
        last = (upper.numerator * legacy + upper.denominator - 1) // upper.denominator
        for legacy_index in range(first, last + 1):
            s = Fraction(legacy_index, legacy)
            # Two tolerated 0.8 mm excursions must never replace an original
            # 1.6 mm excursion that the harness would reject.
            add_edge(Fraction(legacy_index - 1, legacy), s, duration_s / legacy)
            if lower < s < upper:
                inner.append(s)
        if inner:
            last_s = lower
            for s in (*inner, upper):
                add_edge(last_s, s, float(s - last_s) * duration_s)
                last_s = s
        yield ProfileTarget(q, dt, tuple(checks))
        previous, lower = q, upper
