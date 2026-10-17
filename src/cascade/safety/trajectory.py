"""Bounded, deterministic joint-route preflight; this never commands a robot.

The gate models the same TCP/link/payload geometry as SafetyHarness. A vetted
route is not a continuous collision proof and never replaces streaming checks.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np

from ..control.motion_profile import nominal_profile, profile_counts
from ..types import SafetyViolation, SkillError

PLAN_BUDGET_S = 3.0
MAX_ROUTE_CANDIDATES = 15
#: TCP lifts in base z, in order: plan_route's model-only fallback and the
#: air-grasp exit (`SkillRuntime.AIR_GRASP_EXIT_LIFTS_M`) share them.
ROUTE_LIFT_HEIGHTS_M = (0.03, 0.06)


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


def route_end_reason(harness, end):
    """Why a route may not end at joint pose `end`, or None (spec 3.1).

    A step may retreat inside the map margin, but a route must never leave
    the arm there. A GeometryHarness (`mujoco_withdrawal.py`) forwards
    attribute lookups to the arm's harness, so the check resolves there.
    Harnesses without `route_end_violation` (test doubles) have no such
    escape and need no end check.
    """
    check = getattr(harness, "route_end_violation", None)
    reason = check(end) if check is not None else None
    return None if reason is None else "route ends inside the margin: " + reason


def vet_route(harness, start, goals, duration_s, *, deadline=None, rate_hz=50.0):
    previous = start
    for goal in goals:
        reason = vet_segment(harness, previous, goal, duration_s, deadline=deadline, rate_hz=rate_hz)
        if reason:
            return reason
        previous = goal
    return route_end_reason(harness, goals[-1] if len(goals) else start)


def lift_ik(kin, start, height_m: float, *, pose=None) -> np.ndarray | None:
    """Joint pose that raises the TCP of `start` by `height_m` in base z at
    the start orientation, or None.

    Bounded, seed-only model IK (one solve from `start`, no random
    restarts), shared by plan_route's lift fallback and the runtime's
    air-grasp exit. None when the solve fails or its result is not finite.
    The pose is a proposal only: every route to it is vetted. `pose` is
    FK of `start` when the caller already has it (plan_route computes it
    once, inside its deadline checks).
    """
    start = np.asarray(start, dtype=float)
    target = np.array(kin.fk(start) if pose is None else pose, dtype=float)
    target[2, 3] += float(height_m)
    lift = kin.ik(target, q_init=start.copy(), max_iter=80, retries=0)
    if not lift.success:
        return None
    q = np.asarray(lift.q, dtype=float)
    return q if np.isfinite(q).all() else None


def plan_route(harness, start, goal, duration_s=3.0, *, rate_hz=50.0):
    """Try direct, joint-first/last corners, then at most two TCP lifts.

    Corners use only the measured start and requested goal, in their original
    order. Lifts request +3/+6 cm in base z at the start orientation, using
    only remaining candidate slots and bounded, seed-only model IK. The shared
    deadline is checked around IK; an individual solver call is not preempted.
    Every segment must pass before any part can execute.
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

        # Model-only fallback, after every existing corner has failed. Raising
        # the TCP endpoint does NOT make the interpolated joint path vertical;
        # both legs still pass through the same full-route vetting below.
        kin = getattr(harness, "kin", None)
        if not callable(getattr(kin, "fk", None)) or not callable(getattr(kin, "ik", None)):
            return
        used = 1 + 2 * int(np.count_nonzero(np.abs(goal - start) >= 1e-9))
        if used >= MAX_ROUTE_CANDIDATES:
            return
        if time.monotonic() >= deadline:
            raise SkillError("route preflight exceeded its time budget; no route authorized")
        pose = kin.fk(start)
        for height in ROUTE_LIFT_HEIGHTS_M[:MAX_ROUTE_CANDIDATES - used]:
            if time.monotonic() >= deadline:
                raise SkillError("route preflight exceeded its time budget; no route authorized")
            lift = lift_ik(kin, start, height, pose=pose)
            if time.monotonic() >= deadline:
                raise SkillError("route preflight exceeded its time budget; no route authorized")
            if lift is not None:
                yield [lift, goal.copy()]

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
