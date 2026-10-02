"""Opt-in SafeArm execution: native-shaped paths retain live motion authority."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from cascade.config import Cfg
from cascade.control.simulation_motion import SimulationMotion
from cascade.planning import PlanningError
from cascade.planning.runtime import RuntimeMotionPlanner
from cascade.planning.trajectory import TrajectoryProfile
from cascade.safety.harness import SafeArm, SafetyHarness
from cascade.types import SafetyViolation
from test_isaac_simulation_motion import Sim
from test_safety import limits


class CurvePlanner:
    def __init__(self):
        self.calls = []

    def plan_profile(self, start, target, **kwargs):
        self.calls.append((start.copy(), np.asarray(target).copy(), kwargs))
        return TrajectoryProfile.from_curve(
            lambda s: start + (target - start) * float(s)
            + np.array([0., .025 * np.sin(np.pi * float(s))]),
            kwargs['duration_s'], kwargs['rate_hz'], None)


def safe_sim(monkeypatch):
    sim = Sim(monkeypatch)
    get_state = sim.get_state
    sim.get_state = lambda **kw: get_state(timeout_s=kw.get('timeout_s', .2))
    sim.motion_rate_hz = 30.
    sim.stream_profile = lambda profile, planned_state, preflight, **kwargs: SimulationMotion(
        sim, **kwargs).stream_profile(profile, planned_state, .045, 1., preflight)
    sim.stream_to = lambda *a, **kw: pytest.fail('planner failure must never fall back')
    harness = SafetyHarness(limits())
    planner = CurvePlanner()
    return SafeArm(sim, harness, motion_planner=planner), sim, planner


def test_ordinary_motion_streams_one_original_curve_and_live_edges(monkeypatch):
    arm, sim, planner = safe_sim(monkeypatch)
    edges, scene_checks, feedback = [], [], []
    approve = arm.harness.approve
    def gate(a, b, dt, **kwargs):
        edges.append((a.copy(), b.copy(), dt))
        return approve(a, b, dt, **kwargs)
    monkeypatch.setattr(arm.harness, 'approve', gate)
    def scene(start, profile, check):
        for item in profile:
            check()
            scene_checks.append(item.q.copy())
    assert arm.move_joints(np.array([.1, 0.]), .5,
        _trajectory_preflight=scene, feedback_guard=lambda state: feedback.append(state.q.copy()))
    assert len(planner.calls) == 1
    assert len(sim.sent) == 15
    assert max(q[1] for _, _, q in sim.sent) > .024
    np.testing.assert_allclose(sim.sent[-1][2], [.1, 0.], atol=1e-12)
    np.testing.assert_array_equal(scene_checks, [q for _, _, q in sim.sent])
    assert len(edges) > len(sim.sent) and len(feedback) > len(sim.sent)
    assert not arm.harness._motion_active


@pytest.mark.parametrize('stage', ['solver', 'scene', 'feedback', 'stream'])
def test_errors_never_retry_and_always_restore_watchdog(monkeypatch, stage):
    arm, sim, planner = safe_sim(monkeypatch)
    def fail(*args, **kwargs):
        raise TypeError('deliberate failure')
    options = {}
    if stage == 'solver':
        planner.plan_profile = fail
    elif stage == 'scene':
        options['_trajectory_preflight'] = fail
    elif stage == 'feedback':
        options['feedback_guard'] = fail
    else:
        sim.stream_profile = fail
    with pytest.raises((SafetyViolation, TypeError)):
        arm.move_joints(np.array([.1, 0.]), .5, **options)
    assert not sim.sent
    assert not arm.harness._motion_active


def test_legacy_callback_cannot_silently_validate_different_curve(monkeypatch):
    arm, sim, planner = safe_sim(monkeypatch)
    with pytest.raises(SafetyViolation, match='does not support the planned curve'):
        arm.move_joints(np.array([.1, 0.]), _preflight=lambda *a: None)
    assert not sim.sent and not planner.calls


def test_release_and_contact_episodes_keep_authority_before_solving(monkeypatch):
    arm, sim, planner = safe_sim(monkeypatch)
    arm.harness._pending_release_episode = {'episode': 'retained'}
    with pytest.raises(SafetyViolation, match='release'):
        arm.move_joints(np.array([.1, 0.]), .5)
    assert not sim.sent and not planner.calls


def test_halt_while_solver_runs_invalidates_unchanged_feedback(monkeypatch):
    arm, sim, planner = safe_sim(monkeypatch)
    plan = planner.plan_profile
    def halted(*a, **kw):
        value = plan(*a, **kw)
        arm.harness.halt('cancelled while planning')
        return value
    planner.plan_profile = halted
    with pytest.raises(SafetyViolation, match='cancel|halt'):
        arm.move_joints(np.array([.1, 0.]), .5)
    assert not sim.sent and len(planner.calls) == 1
    assert not arm.harness._motion_active


def model_binding(tmp_path):
    model = tmp_path / 'arm.urdf'
    model.write_text('<robot name="arm"><link name="base"/><link name="tip"/>'
                     '<joint name="a"><child link="tip"/></joint></robot>')
    config = dict(type='cumotion', urdf=str(model), xrdf='explicit.xrdf',
        joint_names=['a', 'b'], joint_signs=[-1, 1], base_frame='base', tool_frame='tip',
        obstacles=[])
    arm = Cfg(dict(type='isaac', model=str(model), ee_frame='tip', joint_signs=[-1, 1]))
    kin = NS(n=2, ee_frame='tip', model=NS(names=['universe', 'a', 'b'], joints=[
        NS(idx_q=0, nq=0, nv=0), NS(idx_q=0, nq=1, nv=1), NS(idx_q=1, nq=1, nv=1)]))
    return config, arm, kin


def test_runtime_binding_is_lazy_and_rechecks_urdf_bytes(tmp_path, monkeypatch):
    config, arm, kin = model_binding(tmp_path)
    factory = []
    monkeypatch.setattr('cascade.planning.runtime.make_motion_planner', lambda cfg: factory.append(cfg))
    bound = RuntimeMotionPlanner(config, arm, kin)
    assert not factory
    # Exact model binding cannot drift between app composition and first motion.
    from pathlib import Path
    Path(arm.model).write_text('<robot name="changed"/>')
    with pytest.raises(PlanningError, match='model changed'):
        bound.plan_profile([0, 0], [0, .1])
    assert not factory
    bound.close()


@pytest.mark.parametrize('key,value', [('joint_names', ['b', 'a']),
    ('joint_signs', [1, 1]), ('tool_frame', 'flange'), ('base_frame', 'world')])
def test_runtime_rejects_wrong_joint_or_frame_binding(tmp_path, key, value):
    config, arm, kin = model_binding(tmp_path)
    config[key] = value
    with pytest.raises(PlanningError, match='differ'):
        RuntimeMotionPlanner(config, arm, kin)


def test_runtime_rejects_nonphysical_backend(tmp_path):
    config, arm, kin = model_binding(tmp_path)
    arm._data['type'] = 'mock'
    with pytest.raises(PlanningError, match='physical-clock'):
        RuntimeMotionPlanner(config, arm, kin)
