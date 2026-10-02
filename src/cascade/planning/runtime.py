"""Opt-in cuMotion binding for ordinary SafeArm manipulation calls.

The SDK's static world only proposes a curve. Current harness geometry and
the skill's observed-scene/attachment/release callbacks retain veto authority.
"""
from copy import deepcopy
from pathlib import Path
from threading import RLock
import time
import xml.etree.ElementTree as ET

import numpy as np

from ..control.arm_base import PREFLIGHT_MAX_DRIFT_RAD
from ..control.motion_profile import resolve_motion_rate
from ..safety.trajectory import PLAN_BUDGET_S, geometry_guard
from ..types import SafetyViolation
from . import PlanningError, make_motion_planner
from .trajectory import TrajectoryProfile


def wait_for_contact_stability(safe, state, *, source, robot_id, timeout_s, check, observe,
                               rpc_timeout_s=1., progress=None):
    """Observe a bounded, physically advancing hold; never command or rebase."""
    from ..control.simulation_motion import PhysicsClock, positive
    progress = {} if progress is None else progress
    started = time.monotonic()
    # Empirical contact jitter may exceed 0.25 mrad. Reserve half of the
    # unchanged 1 mrad final start-drift allowance; this is only a prefilter.
    tolerance = PREFLIGHT_MAX_DRIFT_RAD / 2.
    progress.update(status="observing", required_window_physics_s=.5,
                    limit_rad=tolerance, distinct_samples=0)
    try:
        deadline = started + positive(timeout_s, "post-close stability timeout")
        rpc_timeout_s = positive(rpc_timeout_s, "post-close feedback RPC timeout")
        clock = PhysicsClock(source, robot_id)
        clock.observe(state.physics_clock)
        q = np.asarray(state.q, float)
        if q.ndim != 1 or not len(q) or not np.isfinite(q).all():
            raise SafetyViolation("invalid initial post-close joint feedback")
        samples = [(clock.time, q.copy())]
        while True:
            check()
            left = deadline - time.monotonic()
            if left <= 0:
                raise SafetyViolation("post-close joints did not stabilize before the wall deadline")
            try:
                current = safe.get_state(timeout_s=min(left, rpc_timeout_s))
            except Exception as exc:
                if time.monotonic() >= deadline:
                    raise SafetyViolation("post-close joints did not stabilize before the wall deadline") from exc
                raise
            if time.monotonic() >= deadline:
                raise SafetyViolation("post-close joints did not stabilize before the wall deadline")
            check()
            observe(current)
            if clock.observe(current.physics_clock):
                q = np.asarray(current.q, float)
                if q.shape != samples[0][1].shape or not np.isfinite(q).all():
                    raise SafetyViolation("invalid post-close joint feedback")
                samples.append((clock.time, q.copy()))
                # Retain one sample at/before the exact window boundary.
                while len(samples) > 2 and samples[1][0] <= clock.time - .5:
                    samples.pop(0)
                spread = float(np.max(np.ptp([item[1] for item in samples], axis=0)))
                progress.update(window_physics_s=clock.time - samples[0][0],
                    max_joint_range_rad=spread, distinct_samples=len(samples),
                    last_q=q.tolist(), last_clock=dict(current.physics_clock))
                if len(samples) >= 3 and clock.time - samples[0][0] >= .5 and spread <= tolerance:
                    progress['status'] = 'passed'
                    return current, progress
            time.sleep(min(.01, max(0., deadline - time.monotonic())))
    except Exception as exc:
        progress.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        progress['wall_elapsed_s'] = time.monotonic() - started


def contact_lift_target(kin, measured_q, pregrasp_q):
    """Lift to the vetted position without undoing measured contact rotation."""
    pose = kin.fk(measured_q).copy()
    pose[:3, 3] = kin.fk(pregrasp_q)[:3, 3]
    solution = kin.ik(pose, measured_q)
    if not solution.success:
        raise SafetyViolation("cuMotion contact lift cannot preserve measured orientation")
    actual = kin.fk(solution.q)
    position_error = float(np.linalg.norm(actual[:3, 3] - pose[:3, 3]))
    angle = float(np.arccos(np.clip((np.trace(pose[:3, :3].T @ actual[:3, :3]) - 1.) / 2., -1., 1.)))
    if (not np.isfinite(actual).all() or position_error > 1e-4 or angle > 1e-4):
        raise SafetyViolation("cuMotion contact lift IK failed independent FK validation")
    return solution.q


class RuntimeMotionPlanner:
    """Model-bound lazy SDK owner; constructing this never connects an arm."""

    def __init__(self, config, arm_config, kin):
        self.config = deepcopy(config.as_dict() if hasattr(config, 'as_dict') else dict(config))
        if self.config.get('type') != 'cumotion':
            raise PlanningError("runtime motion planner must explicitly select cumotion")
        if arm_config.get('type') != 'isaac':
            raise PlanningError("cuMotion runtime currently requires Isaac physical-clock execution")
        try:
            self.model_bytes = Path(arm_config.model).read_bytes()
            if Path(self.config['urdf']).read_bytes() != self.model_bytes:
                raise PlanningError("cuMotion URDF differs from the arm kinematics model")
            root = ET.fromstring(self.model_bytes)
            roots = {e.attrib['name'] for e in root.findall('link')} - {
                e.attrib['link'] for e in root.findall('joint/child')}
        except (OSError, KeyError, ET.ParseError) as exc:
            raise PlanningError("runtime cuMotion requires the arm's readable URDF") from exc
        joints = [(joint.idx_q, str(kin.model.names[i]), joint.nq, joint.nv)
                  for i, joint in enumerate(kin.model.joints) if i and joint.idx_q < kin.n]
        joints.sort()
        if (len(joints) != kin.n or [j[0] for j in joints] != list(range(kin.n))
                or any(j[2:] != (1, 1) for j in joints)):
            raise PlanningError("runtime cuMotion requires exact scalar controlled joint coordinates")
        self.joint_names = tuple(j[1] for j in joints)
        signs = arm_config.get('joint_signs') or [1] * kin.n
        if (tuple(self.config.get('joint_names', ())) != self.joint_names
                or self.config.get('joint_signs') != list(signs)
                or roots != {self.config.get('base_frame')}
                or self.config.get('tool_frame') != arm_config.get('ee_frame')
                or self.config.get('tool_frame') != kin.ee_frame):
            raise PlanningError("runtime cuMotion joint order, signs or model frames differ")
        self._planner = None
        self._lock = RLock()
        self._closed = False

    def prepare(self):
        """Load the explicitly selected SDK before any arm can materialize."""
        with self._lock:
            if self._closed:
                raise PlanningError("runtime planner is closed")
            # Recheck the same bytes immediately before the lazy SDK load.
            try:
                current_model = Path(self.config['urdf']).read_bytes()
            except OSError as exc:
                raise PlanningError("runtime cuMotion model became unreadable") from exc
            if current_model != self.model_bytes:
                raise PlanningError("runtime cuMotion model changed after composition")
            if self._planner is None:
                self._planner = make_motion_planner(self.config)
            return self._planner

    def plan_profile(self, *args, **kwargs):
        with self._lock:
            self.prepare()
            return self._planner.plan_profile(*args, **kwargs)

    def close(self):
        with self._lock:
            self._closed = True
            if self._planner is not None:
                self._planner.close()
                self._planner = None


def execute(safe, target, duration_s, *, joint_margin=None, legacy_preflight=None,
            trajectory_preflight=None, halt_generation=None, linear_tool_path=False, **kwargs):
    """One solve and one physical stream; every error remains terminal."""
    allowed = {'preflight', 'before_stream', 'feedback_guard', 'rate_hz', 'bias_compensate'}
    if set(kwargs) - allowed:
        raise SafetyViolation("unsupported cuMotion execution options; no motion sent")
    if legacy_preflight is not None and kwargs.get('preflight') is not None:
        raise SafetyViolation("ambiguous motion preflight callbacks; no motion sent")
    if (legacy_preflight is not None or kwargs.get('preflight') is not None) and trajectory_preflight is None:
        raise SafetyViolation("motion safety callback does not support the planned curve")
    h = safe.harness
    generation = h._halt_generation if halt_generation is None else halt_generation
    h.check_release_episode(target=target, duration=duration_s)
    h._check_halt_generation(generation)
    before = kwargs.get('before_stream')
    feedback = kwargs.get('feedback_guard')

    def guard():
        h.check_stream_start(halt_generation=generation)
        if before is not None:
            before()

    h.begin_motion(halt_generation=generation)
    guard()
    state = safe.get_state()
    if feedback is not None:
        feedback(state)
    guard()
    started = time.monotonic()
    profile = safe.motion_planner.plan_profile(state.q, target, duration_s=duration_s,
        rate_hz=resolve_motion_rate(safe.raw, kwargs.get('rate_hz')),
        max_velocity=h.limits.max_joint_vel,
        **({'linear_tool_path': True} if linear_tool_path else {}),
        **({'joint_margin': joint_margin} if joint_margin is not None else {}))
    guard()
    if time.monotonic() - started >= PLAN_BUDGET_S:
        raise SafetyViolation("cuMotion planning exceeded its time budget; no motion sent")
    if (not isinstance(profile, TrajectoryProfile)
            or profile.start.shape != np.asarray(state.q).shape
            or profile.end.shape != np.asarray(target).shape
            or not np.isfinite(profile.start).all() or not np.isfinite(profile.end).all()
            or np.max(abs(profile.start - state.q)) > PREFLIGHT_MAX_DRIFT_RAD
            or np.max(abs(profile.end - np.asarray(target))) > PREFLIGHT_MAX_DRIFT_RAD):
        raise SafetyViolation("cuMotion profile does not match its measured request")

    def preflight(start, actual_profile):
        if actual_profile is not profile:
            raise SafetyViolation("planned profile identity changed")
        deadline = time.monotonic() + PLAN_BUDGET_S
        def check():
            guard()
            if time.monotonic() >= deadline:
                raise SafetyViolation("planned curve preflight exceeded its time budget")
        with geometry_guard(h, deadline=deadline):
            for index, item in enumerate(profile):
                checks = item.checks
                if index == 0:
                    checks = ((start, item.q, item.dt), *checks)
                for a, b, dt in checks:
                    check()
                    reason = h.vet_step(a, b, dt, joint_margin=joint_margin)
                    if reason:
                        raise SafetyViolation(f"planned curve became unsafe: {reason}")
            if trajectory_preflight is not None:
                trajectory_preflight(start, profile, check)
            check()

    def approve(a, b, dt):
        h._check_halt_generation(generation)
        h.approve(a, b, dt, joint_margin=joint_margin)

    try:
        return safe.raw.stream_profile(profile, planned_state=state, approve=approve,
            preflight=preflight, before_stream=guard, feedback_guard=feedback)
    except TypeError as exc:
        raise SafetyViolation("planned motion callback failed; no unguarded retry") from exc
