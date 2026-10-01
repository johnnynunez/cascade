"""The planned grid must match real and virtual Isaac checks at every rate."""
import numpy as np
import pytest

from cascade.control.lazy_arm import LazyArm
from cascade.control.motion_profile import min_jerk, nominal_profile
from cascade.control.simulation_motion import SimulationMotion
from cascade.safety.harness import SafeArm
from cascade.safety.trajectory import plan_route, vet_segment
from cascade.types import SafetyViolation
from test_home_routes import case
from test_isaac_simulation_motion import Sim

START=np.array([.3,0.,.3]);GOAL=np.array([.4,.1,.35])


def obstacle(harness, fraction):
    q=START+(GOAL-START)*min_jerk(fraction)
    harness.limits.keep_out=[(q-.0002,q+.0002)]


def test_30hz_only_obstacle_is_found_in_preflight_before_any_command():
    _,raw,harness=case();obstacle(harness,14/30)
    assert vet_segment(harness,START,GOAL,1.,rate_hz=50.) is None
    assert 'keep-out' in vet_segment(harness,START,GOAL,1.,rate_hz=30.)
    route=plan_route(harness,START,GOAL,1.,rate_hz=30.)
    assert len(route)>1
    previous=START
    for goal in route:
        assert vet_segment(harness,previous,goal,1.,rate_hz=30.) is None
        previous=goal
    assert raw.commands==[]


@pytest.mark.parametrize('fraction',[14/30,23/50])
def test_real_harness_direct_stream_checks_both_grids(monkeypatch,fraction):
    sim=Sim(monkeypatch);sim.n_joints=3;sim.q=START.copy();sim.dq=np.zeros(3)
    _,_,harness=case();obstacle(harness,fraction);harness.begin_motion()
    with pytest.raises(SafetyViolation,match='keep-out'):
        SimulationMotion(sim,approve=harness.approve).stream(GOAL,1.,30.,.045,1.,None)
    assert len(sim.sent)==13


@pytest.mark.parametrize('explicit,expected',[(None,30.),(50.,50.)])
def test_planned_rate_survives_lazy_backend_preflight_rebind_and_stream(monkeypatch,explicit,expected):
    _,raw,harness=case();raw.motion_rate_hz=30.
    lazy=LazyArm(lambda:raw,n_joints=3);safe=SafeArm(lazy,harness)
    seen=[];original=harness.vet_step
    def vet(a,b,dt):seen.append((a.copy(),b.copy(),dt));return original(a,b,dt)
    monkeypatch.setattr(harness,'vet_step',vet)
    assert safe.move_planned(GOAL,duration_s=1.17,rate_hz=explicit,
                             _halt_generation=harness._halt_generation)
    profile=list(nominal_profile(START,GOAL,1.17,expected))
    assert len(raw.commands)==len(profile)
    for command,w in zip(raw.commands,profile):np.testing.assert_array_equal(command,w.q)
    for w in profile:
        for a,b,dt in w.checks:
            assert any(np.array_equal(x,a) and np.array_equal(y,b) and z==dt for x,y,z in seen)


def test_preflight_counterexample_refuses_without_sending_partial_stream(monkeypatch):
    sim=Sim(monkeypatch);sim.n_joints=3;sim.q=START.copy();sim.dq=np.zeros(3)
    _,_,harness=case();obstacle(harness,14/30)
    def preflight(start,duration):
        reason=vet_segment(harness,start,GOAL,duration,rate_hz=30.)
        if reason:raise SafetyViolation(reason)
    with pytest.raises(SafetyViolation,match='keep-out'):
        SimulationMotion(sim,approve=harness.approve).stream(GOAL,1.,30.,.045,1.,preflight)
    assert sim.sent==[]


def test_original_50hz_edges_preserve_workspace_escape_refusals(monkeypatch):
    # Synthetic FK isolates a per-edge +1mm tolerance. Splitting a forbidden
    # +1.6mm edge into two +0.8mm edges must not make a route permissible.
    from cascade.safety.harness import SafetyHarness,SafetyLimits
    fractions=[0,23/50,14/30,24/50,15/30,1]
    qs=[.1+.1*min_jerk(s) for s in fractions]
    class Kin:
        joint_limits=(np.array([-2.]),np.array([2.]))
        def fk(self,q):
            p=np.eye(4);p[:3,3]=[np.interp(q[0],qs,[.60,.60,.6008,.6016,.6008,.59]),0,.3];return p
        def link_positions(self,q):return np.array([[0,0,.1],self.fk(q)[:3,3]])
    harness=SafetyHarness(SafetyLimits(np.array([.1,-.3,-.01]),np.array([.5,.3,.55])),Kin())
    assert 'outside workspace' in vet_segment(harness,[.1],[.2],1.,rate_hz=50.,stretch=False)
    assert 'outside workspace' in vet_segment(harness,[.1],[.2],1.,rate_hz=30.,stretch=False)
    sim=Sim(monkeypatch);sim.n_joints=1;sim.q=np.array([.1]);sim.dq=np.zeros(1)
    harness.heartbeat();harness.begin_motion()
    with pytest.raises(SafetyViolation,match='outside workspace'):
        SimulationMotion(sim,approve=harness.approve).stream(np.array([.2]),1.,30.,.045,1.,None)
    assert len(sim.sent)==13  # Refuse before entering legacy edge23/50->24/50.


def test_planner_preserves_skillerror_for_oversized_waypoint_budget():
    from cascade.types import SkillError
    _,raw,harness=case()
    with pytest.raises(SkillError,match='bounded waypoint'):
        vet_segment(harness,START,GOAL,201.,rate_hz=30.)
    assert raw.commands==[]
