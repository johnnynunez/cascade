"""Motion paced by authoritative physics, with a separate bounded wall budget.

Durations and approval dt are seconds of simulation. A frame/render slowdown
cannot compress a trajectory into fewer physical seconds. A late clock sample
authorizes at most ONE nominal waypoint: missed time is never caught up in a
burst. Missing or inconsistent clocks fail closed; there is no wall-clock mode.
"""

from __future__ import annotations

import time

import numpy as np

from ..types import SafetyViolation
from .arm_base import PREFLIGHT_MAX_DRIFT_RAD, prepare_stream
from .motion_profile import nominal_profile, profile_counts


def positive(value, name):
    if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
        raise SafetyViolation(f"{name} must be finite and positive")
    return float(value)


class PhysicsClock:
    def __init__(self, source, robot_id):
        if not isinstance(robot_id, str) or not robot_id:
            raise SafetyViolation("simulation motion requires a configured robot identity")
        if (not isinstance(source, tuple) or len(source) != 2
                or not isinstance(source[0], str) or type(source[1]) is not int):
            raise SafetyViolation("simulation motion requires a bound bridge endpoint")
        self.source, self.robot_id = source, robot_id
        self.identity = None
        self.time = None
        self.step = None
        self.delta = None

    def observe(self, value):
        if not isinstance(value, dict):
            raise SafetyViolation("authoritative physics clock missing")
        engine = value.get("engine")
        expected_clock = ({"physx": "SimulationManager", "newton": "newton_stage"}.get(engine)
                          if isinstance(engine, str) else None)
        if (type(value.get("version")) is not int or value["version"] != 1
                or expected_clock is None or value.get("clock") != expected_clock
                or value.get("robot_id") != self.robot_id
                or value.get("source") != self.source
                or not isinstance(value.get("epoch"), str) or not value["epoch"]):
            raise SafetyViolation("physics clock source, robot, or schema mismatch")
        t, step, dt = value.get("sim_time"), value.get("physics_step"), value.get("physics_dt_s")
        if type(t) not in (int, float) or not np.isfinite(t) or t < 0:
            raise SafetyViolation("invalid simulation time")
        if type(step) is not int or step < 0:
            raise SafetyViolation("invalid physics step")
        dt = positive(dt, "physics step duration")
        identity = (engine, expected_clock, value["epoch"], dt)
        if self.identity is not None and identity != self.identity:
            raise SafetyViolation("physics clock changed during motion")
        self.identity = identity
        if self.time is not None:
            delta, steps = float(t) - self.time, step - self.step
            if delta < 0 or steps < 0:
                raise SafetyViolation("physics clock regressed during motion")
            if steps == 0:
                if delta != 0:
                    raise SafetyViolation("physics time advanced without a new step")
                self.delta = 0.
                return False
            if delta <= 0 or not np.isclose(delta, steps * dt, rtol=1e-5, atol=1e-7):
                raise SafetyViolation("physics time and step count disagree")
            self.delta = delta
        else:
            self.delta = None
        self.time, self.step = float(t), step
        return True


class SimulationMotion:
    def __init__(self, arm, *, approve=None, before_stream=None, feedback_guard=None):
        self.arm, self.approve, self.before_stream = arm, approve, before_stream
        self.feedback_guard = feedback_guard
        self.deadline = time.monotonic() + positive(arm.motion_wall_timeout_s, "motion wall budget")
        self.clock = PhysicsClock(arm._client._addr, arm._cfg.get("bridge_robot_id"))
        self.last_state = None
        self.q_previous = None
        self.approval_edges = ()
        self.dt = .02

    def remaining(self):
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise SafetyViolation("simulation motion exceeded its wall-time budget")
        return left

    def check(self):
        self.remaining()
        if self.arm._stopped:
            raise SafetyViolation("simulation motion stopped")
        # Run the existing harness cancellation/freshness gates even when
        # simulation is paused. Re-vet the pending/last nominal edge, not an
        # artificial q->q hold: a hold could reject an authorized escape
        # from a joint limit. This sends no actuator data.
        if self.approve is not None:
            for previous, target, dt in self.approval_edges:
                self.remaining()
                if self.arm._stopped:
                    raise SafetyViolation("simulation motion stopped")
                self.approve(previous, target, dt)
        self.remaining()

    def check_start(self):
        self.check()
        if self.before_stream is not None:
            self.before_stream()
        self.check()

    def get_state(self):
        self.check()
        budget = min(self.remaining(), positive(self.arm.motion_rpc_timeout_s, "motion RPC budget"))
        try:
            state = self.arm.get_state(timeout_s=budget)
        except (TypeError, ValueError, KeyError) as exc:
            raise SafetyViolation("invalid simulation feedback") from exc
        self.check()
        q = np.asarray(state.q)
        dq = np.asarray(state.dq)
        if (q.shape != (self.arm.n_joints,) or q.dtype.kind not in "fiu" or not np.isfinite(q).all()
                or dq.shape != q.shape or dq.dtype.kind not in "fiu" or not np.isfinite(dq).all()):
            raise SafetyViolation("simulation motion requires finite, exact-DOF q and dq")
        fresh = self.clock.observe(state.physics_clock)
        if not fresh and self.last_state is not None and not np.array_equal(q, self.last_state.q):
            raise SafetyViolation("joint feedback changed without a new physics step")
        self.last_state = state
        self.fresh = fresh
        if self.feedback_guard is not None:
            self.feedback_guard(state)
            self.check()
        return state

    def stream(self, q_target, duration_s, rate_hz, settle_tol, settle_timeout_s, preflight):
        duration_s = positive(duration_s, "simulation motion duration")
        rate_hz = positive(rate_hz, "simulation waypoint rate")
        positive(settle_tol, "settle position tolerance")
        positive(settle_timeout_s, "simulation settle timeout")
        try:
            target = np.asarray(q_target, dtype=float)
        except (TypeError, ValueError) as exc:
            raise SafetyViolation("invalid simulation joint target") from exc
        if target.shape != (self.arm.n_joints,) or not np.isfinite(target).all():
            raise SafetyViolation("invalid simulation joint target")
        steps, _ = profile_counts(duration_s, rate_hz)
        self.dt = duration_s / steps
        # Same bounded feedback rebind and callback ordering as planned NV
        # routes; get_state here additionally applies the total wall budget.
        start, target = prepare_stream(self, target, duration_s, preflight, self.check_start)
        self.q_previous = start
        last_command_time = self.clock.time
        for waypoint in nominal_profile(start, target, duration_s, rate_hz):
            command = waypoint.q
            self.approval_edges = waypoint.checks
            while True:
                self.get_state()
                if self.fresh and self.clock.time - last_command_time >= self.dt - 1e-9:
                    break
                time.sleep(min(.005, self.remaining()))
            self.check()
            self.arm.send_joint_target(command, timeout_s=min(self.remaining(), self.arm.motion_rpc_timeout_s))
            self.remaining()
            self.q_previous = command
            # Anchor after acknowledgement AND a newer physics step. Time
            # spent waiting for the command's transport lock/ACK cannot be
            # credited to the next waypoint's pacing interval.
            while True:
                self.get_state()
                if self.fresh:
                    break
                time.sleep(min(.005, self.remaining()))
            # Discard surplus simulated time. Never emit multiple commands
            # against one resumed step or leap to a late profile position.
            last_command_time = self.clock.time
        return self.settle(target, settle_tol, settle_timeout_s)

    def settle(self, target, tol, timeout_s):
        try:
            target = np.asarray(target, dtype=float)
        except (TypeError, ValueError) as exc:
            raise SafetyViolation("invalid simulation joint target") from exc
        if target.shape != (self.arm.n_joints,) or not np.isfinite(target).all():
            raise SafetyViolation("invalid simulation joint target")
        tol = positive(tol, "settle position tolerance")
        timeout_s = positive(timeout_s, "simulation settle timeout")
        hold_s = positive(self.arm.settle_hold_s, "simulation settle hold")
        if self.clock.time is None:
            self.get_state()
        deadline = self.clock.time + timeout_s
        anchor = None
        anchor_q = None
        samples = 0
        while self.clock.time < deadline:
            state = self.get_state()
            if not self.fresh:
                time.sleep(min(.005, self.remaining()))
                continue
            if self.clock.time >= deadline:
                return False
            # A large unsampled gap is not a dwell observation.
            if self.clock.delta is not None and self.clock.delta > hold_s:
                anchor, samples = None, 0
            quiet = np.max(np.abs(state.q - target)) < tol
            if quiet:
                # Contact solvers can report nonzero dq while successive
                # tensor positions remain fixed. Require finite dq above,
                # but measure stationarity from fresh q over physics time.
                # Reuse the route preflight's 1 mrad drift contract.
                if (anchor is None or np.max(np.abs(state.q - anchor_q)) >
                        min(PREFLIGHT_MAX_DRIFT_RAD, tol)):
                    anchor = self.clock.time
                    anchor_q = state.q.copy()
                    samples = 0
                samples += 1
                if samples >= 3 and self.clock.time - anchor >= hold_s - 1e-9:
                    return True
            else:
                anchor, samples = None, 0
        return False
