"""A planned curve is streamed once, with physical and safety clocks intact."""
import numpy as np
import pytest

from cascade.control.simulation_motion import SimulationMotion
from cascade.planning import make_motion_planner
from cascade.planning import PlanningError
from cascade.planning.trajectory import TrajectoryProfile
from cascade.types import RobotState, SafetyViolation
from test_cumotion_planner import setup as setup
from test_isaac_simulation_motion import Sim, clock


def curved_profile():
    return TrajectoryProfile.from_curve(
        lambda s: np.array([.1 * float(s), .05 * np.sin(np.pi * float(s))]),
        .2, 30., None)


def planned_state():
    return RobotState(q=np.zeros(2), dq=np.zeros(2), physics_clock=clock())


def test_native_curve_keeps_shape_names_and_signs_after_uniform_slowing(setup):
    cfg, sdk = setup
    def curve(t):
        old_eval = t.eval
        t.eval = lambda x, derivative_order=0: old_eval(x, derivative_order) + np.array([
            .1 * np.sin(np.pi * (x - 7)) if derivative_order == 0
            else .1 * np.pi * np.cos(np.pi * (x - 7)), 0.])
        t.min_position = lambda: np.array([0., -.2])
        t.max_position = lambda: np.array([.2, 0.])
        t.max_velocity_magnitude = lambda: np.array([.42, .2])
    sdk.mutate_trajectory = curve
    with make_motion_planner(cfg) as planner:
        profile = planner.plan_profile([0, 0], [.2, .1], duration_s=2.,
                                       rate_hz=30., max_velocity=1.)
    # The native owner has been released. All execution samples remain copied.
    assert profile.duration_s == 2.
    assert len(sdk.requests) == 1
    midpoint = profile.targets[29].q
    np.testing.assert_allclose(midpoint, [.1, .15], atol=1e-12)
    np.testing.assert_allclose(profile.end, [.2, .1], atol=1e-12)
    assert not profile.plan.as_dict()['execution_authorized']
    with pytest.raises(ValueError):
        profile.targets[0].q.setflags(write=True)
    # 50 Hz times not on the 30 Hz command grid are retained from this curve.
    edge = profile.targets[0].checks[1]
    np.testing.assert_allclose(edge[1], [.002, .001 + .1*np.sin(.01*np.pi)])


def test_frozen_curve_executes_exact_targets_without_minjerk_between_them(monkeypatch):
    sim = Sim(monkeypatch)
    profile = curved_profile()
    edges, preflights = [], []
    motion = SimulationMotion(sim, approve=lambda a,b,dt: edges.append((a.copy(),b.copy(),dt)))
    assert motion.stream_profile(profile, planned_state(), .045, 1.,
                                 lambda start,p: preflights.append(p))
    assert preflights == [profile]
    np.testing.assert_array_equal([row[2] for row in sim.sent], [p.q for p in profile])
    assert len(sim.sent) == 6
    assert max(row[2][1] for row in sim.sent) == pytest.approx(.05)
    assert all(b[1] > a[1] for a,b in zip(sim.sent,sim.sent[1:]))
    assert any(np.allclose(b,[.01,.05*np.sin(.1*np.pi)]) for a,b,dt in edges)


@pytest.mark.parametrize('when', ['planning', 'preflight'])
def test_feedback_drift_rejects_without_replanning_or_rebasing(monkeypatch, when):
    sim = Sim(monkeypatch)
    if when == 'planning':
        sim.q[0] = .002
    def preflight(start, profile):
        sim.q[0] = .002
    with pytest.raises(SafetyViolation, match='changed|moved'):
        SimulationMotion(sim).stream_profile(curved_profile(), planned_state(), .045, 1., preflight)
    assert sim.sent == []


def test_epoch_change_during_planning_is_not_hidden_by_same_joint_pose(monkeypatch):
    sim = Sim(monkeypatch)
    state = planned_state()
    state.physics_clock['epoch'] = 'before-reset'
    with pytest.raises(SafetyViolation, match='changed'):
        SimulationMotion(sim).stream_profile(curved_profile(), state, .045, 1., None)
    assert not sim.sent


@pytest.mark.parametrize('guard', ['raw', 'feedback', 'edge'])
def test_stop_or_live_veto_stops_the_single_stream_without_retry(monkeypatch, guard):
    sim = Sim(monkeypatch)
    def check():
        if len(sim.sent) >= 2:
            raise SafetyViolation('retained live veto')
    if guard == 'raw':
        sim.on_read = lambda s: setattr(s, '_stopped', len(s.sent) >= 2)
    motion = SimulationMotion(sim,
        approve=(lambda *a: check()) if guard == 'edge' else None,
        feedback_guard=(lambda state: check()) if guard == 'feedback' else None)
    with pytest.raises(SafetyViolation, match='stopped|live veto'):
        motion.stream_profile(curved_profile(), planned_state(), .045, 1., None)
    assert len(sim.sent) == 2


def test_paused_physics_cannot_advance_planned_curve(monkeypatch):
    sim = Sim(monkeypatch, step_advance=0)
    sim.motion_wall_timeout_s = .12
    with pytest.raises(SafetyViolation, match='wall-time'):
        SimulationMotion(sim).stream_profile(curved_profile(), planned_state(), .045, 1., None)
    assert not sim.sent


@pytest.mark.parametrize('deviates', [False, True])
def test_linear_tool_constraint_is_requested_and_independently_checked(setup, deviates):
    from types import SimpleNamespace as NS
    cfg, sdk = setup
    constraints = []
    class Target:
        TranslationPathConstraint = NS(linear=lambda tolerance: ('line', tolerance))
        OrientationPathConstraint = NS(constant=lambda tolerance: ('orientation', tolerance))
        def __new__(cls, q, translation, orientation):
            constraints.append((translation, orientation))
            return q.copy()
    sdk.TrajectoryOptimizer = NS(CSpaceTarget=Target, Results=sdk.TrajectoryOptimizer.Results)
    def pose(q, frame):
        matrix = np.eye(4)
        matrix[:2, 3] = q
        if deviates:
            matrix[2, 3] = .01 * np.sin(np.pi * q[0] / .1)
        return NS(matrix=lambda: matrix)
    sdk.kin.pose = pose
    with make_motion_planner(cfg) as planner:
        planner._optimizer.plan_to_cspace_target = lambda *a: pytest.fail(
            'contact must select its declared native optimizer before solving')
        if deviates:
            with pytest.raises(PlanningError, match='violates the requested linear tool path'):
                planner.plan_profile([0, 0], [.2, .1], duration_s=2., rate_hz=30.,
                                     max_velocity=1., linear_tool_path=True)
        else:
            profile = planner.plan_profile([0, 0], [.2, .1], duration_s=2., rate_hz=30.,
                                           max_velocity=1., linear_tool_path=True)
            assert profile.plan.path_constraint == 'linear_tool'
    assert constraints == [(('line', .0001), ('orientation', .025))]


def test_authorized_zero_margin_is_per_request_and_keeps_hard_limits(setup):
    cfg, sdk = setup
    with make_motion_planner(cfg) as planner:
        with pytest.raises(PlanningError, match='margin'):
            planner.plan_profile([1.9, 0.], [2., 0.], duration_s=2., rate_hz=30., max_velocity=1.)
        permitted = planner.plan_profile([1.9, 0.], [2., 0.], duration_s=2., rate_hz=30.,
                                         max_velocity=1., joint_margin=0.)
        np.testing.assert_allclose(permitted.end, [2., 0.])
        with pytest.raises(PlanningError, match='margin'):
            planner.plan_profile([1.9, 0.], [2., 0.], duration_s=2., rate_hz=30., max_velocity=1.)
        with pytest.raises(PlanningError, match='limits'):
            planner.plan_profile([1.9, 0.], [2.001, 0.], duration_s=2., rate_hz=30.,
                                 max_velocity=1., joint_margin=0.)
        default = planner.plan_profile([0., 0.], [.1, 0.], duration_s=2., rate_hz=30., max_velocity=1.)
        override = planner.plan_profile([0., 0.], [.1, 0.], duration_s=2., rate_hz=30.,
                                        max_velocity=1., joint_margin=0.)
        assert default.plan.request_sha256 != override.plan.request_sha256
