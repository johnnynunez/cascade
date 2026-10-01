"""Changing Isaac command rate must not discard legacy safety samples."""
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.config import Cfg
from cascade.control.arm_base import ArmBase
from cascade.control.isaac_arm import IsaacArm
from cascade.control.lazy_arm import LazyArm
from cascade.control.motion_profile import min_jerk, nominal_profile, resolve_motion_rate
from cascade.control.simulation_motion import SimulationMotion
from cascade.types import SafetyViolation
from test_isaac_simulation_motion import Sim


@pytest.mark.parametrize('duration,rate', [(7.5,30.),(7.53,30.),(.017,30.),(1.13,73.),(2.,50.)])
def test_union_retains_exact_legacy_samples_and_actual_edges(duration, rate):
    start=np.array([.3,0,.3]);goal=np.array([.4,.1,.35])
    profile=list(nominal_profile(start,goal,duration,rate))
    n=max(2,int(duration*rate));legacy=max(2,int(duration*50))
    assert len(profile)==n
    samples=[q for waypoint in profile for _,q,_ in waypoint.checks]
    for i in range(1,legacy+1):
        expected=start+(goal-start)*min_jerk(i/legacy)
        assert any(np.array_equal(q,expected) for q in samples)
        old_previous=start+(goal-start)*min_jerk((i-1)/legacy)
        assert any(np.array_equal(a,old_previous) and np.array_equal(b,expected)
                   and dt==duration/legacy for w in profile for a,b,dt in w.checks)
    previous=start
    for i,w in enumerate(profile,1):
        np.testing.assert_array_equal(w.q,start+(goal-start)*min_jerk(i/n))
        np.testing.assert_array_equal(w.checks[0][0],previous)
        np.testing.assert_array_equal(w.checks[0][1],w.q)
        assert w.checks[0][2]==duration/n
        for a,b,dt in w.checks:
            assert dt>0
            assert dt<=max(duration/n,duration/legacy)+1e-12
            assert np.max(abs((b-a)/dt))<=1.875*np.max(abs(goal-start))/duration+1e-10
        previous=w.q


def test_explicit_50hz_preserves_original_commands_and_one_check_per_edge():
    start=np.zeros(2);goal=np.array([1.,-1.]);duration=1.17;n=int(duration*50.)
    profile=list(nominal_profile(start,goal,duration,50.))
    for i,w in enumerate(profile,1):
        assert len(w.checks)==1
        np.testing.assert_array_equal(w.q,start+(goal-start)*min_jerk(i/n))


@pytest.mark.parametrize('configured,explicit,expected',[(30,None,30),(24,None,24),(30,50,50),(50,30,30)])
def test_isaac_default_and_explicit_rate_resolution(monkeypatch,configured,explicit,expected):
    arm=IsaacArm(Cfg({'bridge_robot_id':'/robot','motion_rate_hz':configured}))
    seen=[]
    def stream(self,q,duration,rate,*args):seen.append(rate);return True
    monkeypatch.setattr(SimulationMotion,'stream',stream)
    assert arm.stream_to(np.zeros(6),1.,rate_hz=explicit)
    assert seen==[expected]


def test_defaults_are_isaac_30_hardware_50_and_lazy_inspection_has_no_side_effect():
    arm=IsaacArm(Cfg({'bridge_robot_id':'/robot'}))
    assert resolve_motion_rate(arm)==30.
    assert ArmBase.motion_rate_hz==50.
    assert inspect.signature(ArmBase.stream_to).parameters['rate_hz'].default==50.
    def forbidden():raise AssertionError('must not connect hardware merely to inspect')
    lazy=LazyArm(forbidden)
    assert resolve_motion_rate(lazy)==50.
    lazy._arm=arm  # A planned route has already read feedback/materialized.
    assert resolve_motion_rate(lazy)==30.


@pytest.mark.parametrize('value',[0,-1,True,float('nan'),float('inf'),'30'])
def test_invalid_explicit_rate_never_falls_back(value):
    arm=IsaacArm(Cfg({'bridge_robot_id':'/robot'}))
    with pytest.raises(SafetyViolation,match='rate'):
        arm.stream_to(np.zeros(6),1.,rate_hz=value)


def test_direct_stream_catches_legacy_only_geometry_before_next_real_target(monkeypatch):
    sim=Sim(monkeypatch);legacy_q=.1*min_jerk(23/50)
    def approve(a,b,dt):
        if abs(b[0]-legacy_q)<1e-12:
            raise SafetyViolation('obstacle at retained legacy sample')
    with pytest.raises(SafetyViolation,match='retained legacy'):
        SimulationMotion(sim,approve=approve).stream(np.array([.1,-.1]),1.,30.,.045,1.,None)
    assert len(sim.sent)==13
    assert all(abs(q[0]-legacy_q)>1e-6 for _,_,q in sim.sent)


@pytest.mark.parametrize('cancel',['harness','raw'])
def test_cancellation_inside_virtual_sample_emits_no_next_command(monkeypatch,cancel):
    sim=Sim(monkeypatch);legacy_q=.1*min_jerk(23/50)
    def approve(a,b,dt):
        if abs(b[0]-legacy_q)<1e-12:
            if cancel=='raw':sim._stopped=True
            else:raise SafetyViolation('cancelled on virtual sample')
    with pytest.raises(SafetyViolation,match='stopped|cancelled'):
        SimulationMotion(sim,approve=approve).stream(np.array([.1,-.1]),1.,30.,.045,1.,None)
    assert len(sim.sent)==13


def test_virtual_edges_keep_escape_direction_and_their_own_dt(monkeypatch):
    sim=Sim(monkeypatch);seen=[]
    def approve(a,b,dt):
        assert b[0]>a[0]  # Escape must be forward; never invent q->q holds.
        assert np.max(abs((b-a)/dt))<=.1875+1e-10
        seen.append(dt)
    assert SimulationMotion(sim,approve=approve).stream(np.array([.1,-.1]),1.,30.,.045,1.,None)
    assert len(sim.sent)==30
    assert min(seen)<.02 and max(seen)==pytest.approx(1/30)
    assert all(b[1]-a[1]>=4 for a,b in zip(sim.sent,sim.sent[1:]))


@pytest.mark.parametrize('mode',['frozen','regression','slow_ack'])
def test_30hz_retains_clock_failure_and_post_ack_pacing_contract(monkeypatch,mode):
    sim=Sim(monkeypatch)
    if mode=='frozen':
        sim.step_advance=0;sim.motion_wall_timeout_s=.12
    elif mode=='regression':
        def regress(s):
            if len(s.reads)==5:s.step=0
        sim.on_read=regress
    else:sim.send_steps=100
    motion=SimulationMotion(sim)
    if mode=='slow_ack':
        assert motion.stream(np.array([.1,-.1]),.2,30.,.045,1.,None)
        assert len(sim.sent)==6
        assert all(b[1]-a[1]>=105 for a,b in zip(sim.sent,sim.sent[1:]))
    else:
        with pytest.raises(SafetyViolation,match='wall-time budget|regressed'):
            motion.stream(np.array([.1,-.1]),.2,30.,.045,1.,None)
        assert len(sim.sent)<=1
