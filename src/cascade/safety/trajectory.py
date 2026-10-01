"""Bounded, deterministic joint-route preflight; this never commands a robot.

The gate models the same TCP/link/payload geometry as SafetyHarness. A vetted
route is not a continuous collision proof and never replaces streaming checks.
"""
from __future__ import annotations

from contextlib import contextmanager
import time

import numpy as np

from ..control.motion_profile import nominal_profile, profile_counts
from ..types import SkillError, SafetyViolation

PLAN_BUDGET_S = 3.0
MAX_ROUTE_CANDIDATES = 15


def motion_duration(harness, start, goal, duration_s):
    delta = float(np.max(np.abs(np.asarray(goal) - np.asarray(start))))
    return max(float(duration_s), 1.875 * delta / (.9 * harness.limits.max_joint_vel))


@contextmanager
def geometry_guard(harness, *, deadline=None):
    # OccupancyMap refresh and payload transitions use this same reentrant
    # lock. Prevent a preflight from mixing two different captured maps.
    occupancy = getattr(harness, "occupancy", None)
    lock = getattr(occupancy, "_refresh_lock", None)
    if lock is None:
        yield
        return
    timeout = max(0., deadline - time.monotonic()) if deadline is not None else PLAN_BUDGET_S
    if not lock.acquire(timeout=timeout):
        raise SkillError("route preflight could not acquire fresh geometry within its time budget")
    try:
        yield
    finally:
        lock.release()


def _joints(start, goal):
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    if (start.ndim != 1 or start.shape != goal.shape or not start.size
            or not np.isfinite(start).all() or not np.isfinite(goal).all()):
        raise SkillError("route requires finite, matching joint vectors")
    return start, goal


def vet_segment(harness, start, goal, duration_s, *, deadline=None, stretch=True, rate_hz=50.0):
    """Vet the executor's real edges and retained legacy safety samples.

    Callers vetting several segments hold geometry_guard around the whole
    route. SafetyHarness.vet_step retains the ordinary escape rules.
    """
    start, goal = _joints(start, goal)
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise SkillError("route duration must be finite and positive")
    duration = (motion_duration(harness, start, goal, duration_s)
                if stretch else float(duration_s))
    try:
        profile_counts(duration, rate_hz, max_steps=10000)
    except SafetyViolation as exc:
        raise SkillError(f"route exceeds or violates its bounded waypoint profile: {exc}") from exc
    for target in nominal_profile(start, goal, duration, rate_hz, max_steps=10000):
        for previous, waypoint, dt in target.checks:
            if deadline is not None and time.monotonic() >= deadline:
                raise SkillError("route preflight exceeded its time budget; no route authorized")
            reason = harness.vet_step(previous, waypoint, dt)
            if reason:
                return reason
    return None


def vet_route(harness, start, goals, duration_s, *, deadline=None, rate_hz=50.0):
    previous = start
    for goal in goals:
        reason = vet_segment(harness, previous, goal, duration_s, deadline=deadline, rate_hz=rate_hz)
        if reason:
            return reason
        previous = goal
    return None


def plan_route(harness, start, goal, duration_s=3.0, *, rate_hz=50.0):
    """Try a direct route, then joint-first/joint-last corners, without retries.

    Corners use only the measured start and requested goal. Every segment must
    pass before any part can execute; candidate count and wall time are capped.
    """
    start, goal = _joints(start, goal)
    deadline = time.monotonic() + PLAN_BUDGET_S

    def candidates():
        yield [goal.copy()]
        for joint in range(len(start)):
            if abs(goal[joint] - start[joint]) < 1e-9:
                continue
            first = start.copy()
            first[joint] = goal[joint]
            yield [first, goal.copy()]
            last = goal.copy()
            last[joint] = start[joint]
            yield [last, goal.copy()]

    first_reason = None
    with geometry_guard(harness, deadline=deadline):
        for index, route in enumerate(candidates()):
            if index >= MAX_ROUTE_CANDIDATES:
                break
            reason = vet_route(harness, start, route, duration_s, deadline=deadline, rate_hz=rate_hz)
            if reason is None:
                return route
            if first_reason is None:
                first_reason = reason
    raise SkillError(f"no safe joint route found; direct route: {first_reason}")
