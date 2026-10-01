"""Released withdrawal never borrows a global exemption or reopens the jaws."""
from types import SimpleNamespace
import copy
import threading
import time

import numpy as np
import pytest

from cascade.config import Cfg
from cascade.safety.harness import SafetyHarness, SafetyLimits, SafeArm
from cascade.skills import release_episode as release
from cascade.skills.runtime import SkillRuntime
from cascade.types import RobotState, SafetyViolation, SkillError
from cascade.perception.occupancy import OccupancyError
from test_occupancy_payload import mapping, frame, PROP


def jaws(pos=.05):
    return {'version':1,'names':['joint_left','joint_right'],'position_m':[pos,pos],
            'lower_m':[0.,0.],'upper_m':[.05,.05]}


def atomic_attachment(state, *, paths=(), signs=None):
    clock = state.physics_clock
    signs = np.ones(len(state.q)) if signs is None else signs
    state.attachment = {'version': 1, 'backend': 'isaac', 'source': clock['source'],
        'robot_id': clock['robot_id'], 'producer_epoch': clock['epoch'],
        'physics_step': clock['physics_step'], 'sim_time': clock['sim_time'],
        'joint_convention': 'asset', 'q': (state.q * signs).tolist(),
        'gripper_joints': copy.deepcopy(state.gripper_joints),
        'tracking': True, 'error': None, 'paths': list(paths),
        'channel': 'completed_update_bilateral_contact', 'sensor_channel': 'physx_gpu_contact_tensor'}
    return state


def capture(name, stamp, *, attached=False):
    f=frame(attached,stamp);f.capture['camera']=name
    f.capture['proprioception'].update(producer_epoch='epoch',gripper_joints=jaws())
    return f


def case():
    class Kin:
        joint_limits=(np.full(3,-2.),np.full(3,2.))
        def fk(self,q):
            out=np.eye(4);out[:3,3]=q;return out
        def link_positions(self,q):return np.array([[0.,0.,.8]])
    events=[]
    class Raw:
        n_joints=3
        _signs=np.ones(3)
        settle_hold_s=.1
        _acknowledged_joint_targets=0
        q=np.array([.2,.1,.3]);step=0;opening=.025
        def get_state(self, *, timeout_s=None):
            self.step+=1
            state = RobotState(q=self.q.copy(),dq=np.zeros(3),gripper_joints=jaws(self.opening),
                physics_clock={'version':1,'source':('test',1),'robot_id':'/Robot','engine':'physx',
                  'clock':'SimulationManager','epoch':'epoch','physics_step':self.step,
                  'sim_time':self.step/120.,'physics_dt_s':1/120.})
            return atomic_attachment(state, paths=() if self.opening >= .049 else (PROP,))
        def validate_simulation_clock(self):return self.get_state().physics_clock
        def set_gripper(self,p,effort):self.opening=.05*p;events.append(('jaw',p))
        def stream_to(self,target,duration_s,approve,**kw):
            if kw.get('preflight'):kw['preflight'](self.q.copy(),duration_s)
            if kw.get('before_stream'):kw['before_stream']()
            approve(self.q,target,duration_s)
            self._acknowledged_joint_targets+=1
            events.append(('move',target.copy(),duration_s,h._grasp_exempt))
            self.q=target.copy();return True
    raw=Raw();h=SafetyHarness(SafetyLimits(np.full(3,-1.),np.ones(3),watchdog_s=100.),Kin())
    h.occupancy=m=mapping();arm=SafeArm(raw,h)
    for name in ('cam0','side','proof'):m.refresh(capture(name,1.,attached=True),np.eye(4))
    h.allow_grasp_descent(raw.q[:2],radius_m=.07,z_min=.25)
    streams=[SimpleNamespace(name=name,get_fresh_frame=lambda name=name,**kw:capture(name,10.))
             for name in ('cam0','side','proof')]
    rt=SkillRuntime.__new__(SkillRuntime)
    rt.arm=arm;rt.kin=h.kin;rt.cfg=Cfg({'arm':{'type':'isaac','bridge_host':'test','bridge_port':1,
        'bridge_robot_id':'/Robot','joint_signs':[1,1,1]},'grasp':{'descend_duration_s':2.}})
    rt.held_object='green cube';rt._grip_open=1.;rt.camera=streams[0]
    rt.watcher=SimpleNamespace(_cams=[SimpleNamespace(stream=s,maps_depth=True) for s in streams])
    original=m.wait_released_ready
    def commit(floors,**kw):
        for f in floors:m.refresh(capture(f.capture['camera'],11.),np.eye(4))
        return original(floors,**kw)
    m.wait_released_ready=commit
    ep=release.begin(rt,np.array([.2,.1,.4]),2.)
    return rt,ep,events


def retained():
    rt,ep,events=case();release.open_hand(rt,ep);rt.held_object=None
    release.wait_geometry(rt,ep)
    release.finish(rt,ep,completed=False);events.clear()
    return rt,ep,events


def test_original_release_and_exact_withdrawal_with_fresh_empty_commits():
    rt,ep,events=case();release.open_hand(rt,ep);rt.held_object=None
    evidence=release.wait_geometry(rt,ep)
    assert evidence['contact_paths']==[] and min(m['t'] for m in evidence['integrated'])>10.
    assert rt.arm.harness.occupancy._prop_history_floor[PROP]==11.
    assert release.withdraw(rt,ep)
    release.finish(rt,ep,completed=True)
    assert [x[0] for x in events]==['jaw','move']
    assert rt._release_episode is None and rt.arm.harness._pending_release_episode is None


def test_explicit_recovery_uses_exact_original_cylinder_endpoint_without_open():
    rt,ep,events=retained();result=release.recover(rt)
    assert result['retreated_to_original_release_target'] and len(events)==1
    _,q,duration,cylinder=events[0]
    np.testing.assert_array_equal(q,ep['q_retreat']);assert duration==ep['duration_s']
    np.testing.assert_array_equal(cylinder[0],ep['cylinder'][0]);assert cylinder[1:]==ep['cylinder'][1:]
    assert rt.arm.harness._grasp_exempt is None and rt._release_episode is None


@pytest.mark.parametrize('change',['q','backend','robot','epoch','camera','scene','halt','jaws','contact'])
def test_changed_state_refuses_recovery_without_any_actuator(change):
    rt,ep,events=retained();h=rt.arm.harness
    if change=='q':rt.arm.raw.q[0]+=.002
    elif change=='backend':rt.arm._arm=SimpleNamespace(get_state=lambda:None)
    elif change=='robot':rt.cfg.arm._data['bridge_robot_id']='/Other'
    elif change=='epoch':
        original=rt.arm.raw.get_state
        def altered(**kw):
            s=original(**kw);s.physics_clock['epoch']='other';return s
        rt.arm.raw.get_state=altered
    elif change=='camera':rt.watcher._cams.pop()
    elif change=='scene':h.occupancy.begin_scene_reset()
    elif change=='halt':h.halt('late stop')
    elif change=='jaws':rt.arm.raw.opening=.04
    else:rt.camera.get_fresh_frame=lambda **kw:capture('cam0',10.,attached=True)
    with pytest.raises((SkillError,SafetyViolation)):release.recover(rt)
    assert not events and rt._release_episode is ep and h._grasp_exempt is None


def test_scope_blocks_concurrent_commands_wrong_endpoint_duration_and_jaws():
    rt,ep,events=retained();h=rt.arm.harness;refused=[]
    with pytest.raises(SafetyViolation,match='unfinished release'):rt.arm.move_joints(ep['q_retreat'])
    def worker():
        for f in (lambda:rt.arm.move_joints(ep['q_retreat']),lambda:rt.arm.set_gripper(1.)):
            try:f()
            except SafetyViolation:refused.append(True)
    h._release_scope.value=(ep,'retreat')
    h.allow_grasp_descent(ep['cylinder'][0],radius_m=ep['cylinder'][1],z_min=ep['cylinder'][2])
    thread=threading.Thread(target=worker);thread.start();thread.join(1);assert not thread.is_alive()
    with pytest.raises(SafetyViolation,match='exact original'):rt.arm.move_joints(ep['q_retreat']+.01)
    with pytest.raises(SafetyViolation,match='exact original'):rt.arm.move_joints(ep['q_retreat'],duration_s=3.)
    with pytest.raises(SafetyViolation,match='unfinished release'):rt.arm.set_gripper(1.)
    assert len(refused)==2 and not events


@pytest.mark.parametrize('fault',['unacknowledged','one_jaw','floor_contact','floor_epoch','floor_camera_timeout'])
def test_ambiguous_release_is_never_recovered_using_held_none(fault):
    rt,ep,events=case()
    if fault!='unacknowledged':release.open_hand(rt,ep)
    rt.held_object=None
    if fault=='one_jaw':
        original=rt.arm.raw.get_state
        def state(**kw):
            s=original(**kw);s.gripper_joints['position_m'][1]=.025;return s
        rt.arm.raw.get_state=state
    if fault.startswith('floor_'):
        def floor(**kw):
            if fault=='floor_camera_timeout':raise TimeoutError('frozen')
            f=capture('cam0',10.,attached=fault=='floor_contact')
            if fault=='floor_epoch':f.capture['proprioception']['producer_epoch']='new'
            return f
        rt.camera.get_fresh_frame=floor
    with pytest.raises((SkillError,SafetyViolation)):release.wait_geometry(rt,ep)
    assert not ep['released']
    release.finish(rt,ep,completed=False);events.clear()
    with pytest.raises(SkillError,match='confirmed release'):release.recover(rt)
    assert not events


def test_mapping_failure_retains_eligible_release_and_failed_feedback():
    rt,ep,events=case();release.open_hand(rt,ep)
    rt.arm.harness.occupancy.wait_released_ready=lambda *a,**kw:(_ for _ in ()).throw(OccupancyError('500 ms'))
    with pytest.raises(SkillError,match='500 ms'):release.wait_geometry(rt,ep)
    assert ep['released']
    release.finish(rt,ep,completed=False)
    np.testing.assert_array_equal(ep['q_failure'],rt.arm.raw.q)
    assert [v[0] for v in events]==['jaw']


def test_partial_retreat_failure_rebinds_only_measured_failure_pose():
    rt,ep,events=retained()
    def partial(*args,**kw):
        rt.arm.raw.q[2]+=.02;rt.arm.raw._acknowledged_joint_targets+=1
        events.append(('partial',));raise SafetyViolation('mapping fault')
    rt.arm.raw.stream_to=partial
    with pytest.raises(SafetyViolation,match='mapping fault'):release.recover(rt)
    assert ep['q_failure'][2]==pytest.approx(.32) and rt.arm.harness._grasp_exempt is None
    rt.arm.raw.q[2]+=.002;events.clear()
    with pytest.raises(SafetyViolation,match='feedback changed'):release.recover(rt)
    assert not events


def test_halt_on_last_feedback_read_and_after_geometry_cannot_be_cleared():
    rt,ep,events=retained();original=rt.arm.raw.get_state;count=[]
    def state(**kw):
        count.append(1)
        if len(count)==3:rt.arm.harness.halt('last feedback')
        return original(**kw)
    rt.arm.raw.get_state=state
    with pytest.raises(SafetyViolation,match='halt'):release.recover(rt)
    assert not events


def test_public_reset_finishes_release_before_home_without_exemption():
    rt,ep,events=retained()
    def home(**kw):
        assert rt._release_episode is None and rt.arm.harness._grasp_exempt is None
        assert kw['_halt_generation']==ep['halt_generation']
        raise SkillError('normal home remains rejected')
    rt.skill_move_home=home
    result=rt.skill_reset_scene()
    assert result['release_recovery']['retreated_to_original_release_target']
    assert result['stage']=='home' and len(events)==1


def test_halt_after_recovery_home_prevents_prop_reset_or_open():
    rt,ep,events=retained()
    rt.skill_move_home=lambda **kw:rt.arm.harness.halt('after home')
    result=rt.skill_reset_scene()
    assert result['stage']=='reset_recovery' and len(events)==1


def test_one_of_three_missing_commit_cannot_authorize_empty_withdrawal():
    m=mapping();floors=[capture(n,10.) for n in ('cam0','side','proof')]
    for name in ('cam0','side','proof'):m.refresh(capture(name,1.,attached=True),np.eye(4))
    for name in ('cam0','side'):m.refresh(capture(name,11.),np.eye(4))
    with pytest.raises(OccupancyError,match='deadline'):
        m.wait_released_ready(floors,deadline=time.monotonic()+.01,guard=lambda:None,
                             producer_epoch='epoch',prop_floors={PROP:1.})
    m.refresh(capture('proof',11.),np.eye(4))
    assert m.wait_released_ready(floors,deadline=time.monotonic()+.1,guard=lambda:None,
                                 producer_epoch='epoch',prop_floors={PROP:1.})['contact_paths']==[]


def test_repeated_reset_after_uncommanded_drift_never_rebases_failure_pose():
    rt,ep,events=retained();failure=ep['q_failure'].copy();rt.arm.raw.q[0]+=.002
    for _ in range(2):
        result=rt.skill_reset_scene()
        assert result['stage']=='release_recovery'
        np.testing.assert_array_equal(ep['q_failure'],failure)
    assert not events


@pytest.mark.parametrize('field,value',[('version',True),('position_m',['.05','.05']),('position_m',[True,True]),('lower_m',['0','0']),('upper_m',[float('nan'),.05])])
def test_malformed_individual_feedback_is_not_confirmed_open(field,value):
    snapshot=jaws();snapshot[field]=value
    with pytest.raises(SafetyViolation):release._jaws(snapshot,require_open=True)


def test_stream_camera_swap_after_confirmed_release_is_rejected():
    rt,ep,events=retained()
    a,b=rt.watcher._cams[:2]
    a.stream.get_fresh_frame,b.stream.get_fresh_frame=b.stream.get_fresh_frame,a.stream.get_fresh_frame
    with pytest.raises(SafetyViolation,match='stream source'):release.recover(rt)
    assert not events


def test_missing_or_wrong_epoch_failure_feedback_blocks_future_reset():
    rt,ep,events=retained()
    def partial(*a,**kw):
        rt.arm.raw.q[2]+=.02;rt.arm.raw._acknowledged_joint_targets+=1
        rt.arm.raw.get_state=lambda **kw:RobotState(q=rt.arm.raw.q.copy(),physics_clock=None)
        raise SafetyViolation('partial failure')
    rt.arm.raw.stream_to=partial
    with pytest.raises(SafetyViolation):release.recover(rt)
    assert ep['q_failure'] is None
    with pytest.raises(SkillError,match='measured failure'):release.recover(rt)
    assert not events


def test_withdraw_cannot_skip_its_fresh_release_barrier():
    rt,ep,events=case();release.open_hand(rt,ep);events.clear()
    with pytest.raises(SafetyViolation,match='fresh confirmed'):release.withdraw(rt,ep)
    assert not events


def test_real_reset_after_ready_cancels_even_inside_withdraw_scope():
    rt,ep,events=case();release.open_hand(rt,ep);release.wait_geometry(rt,ep);events.clear()
    rt.arm.harness._release_scope.value=(ep,'retreat')
    rt.arm.harness.occupancy.begin_scene_reset()
    with pytest.raises(SafetyViolation,match='scene identity'):
        rt.arm.move_joints(ep['q_retreat'],duration_s=ep['duration_s'])
    assert not events


def test_old_open_capture_before_attachment_cannot_confirm_release():
    rt,ep,events=case();release.open_hand(rt,ep)
    rt.camera.get_fresh_frame=lambda **kw:capture('cam0',.5)
    with pytest.raises(SafetyViolation,match='capture identity'):release.wait_geometry(rt,ep)
    assert not ep['released']


def test_released_map_preserves_each_prop_history_floor():
    m=mapping();floors=[capture(n,10.) for n in ('cam0','side','proof')]
    for name in ('cam0','side','proof'):m.refresh(capture(name,11.),np.eye(4))
    with pytest.raises(OccupancyError,match='deadline'):
        m.wait_released_ready(floors,deadline=time.monotonic()+.01,guard=lambda:None,
                             producer_epoch='epoch',prop_floors={PROP:9.})


def test_last_guard_expiring_deadline_never_publishes_release_barrier():
    m=mapping();floors=[capture(n,10.) for n in ('cam0','side','proof')]
    for name in ('cam0','side','proof'):m.refresh(capture(name,11.),np.eye(4))
    count=[]
    def guard():
        count.append(1)
        if len(count)>1:time.sleep(.015)
    with pytest.raises(OccupancyError,match='deadline'):
        m.wait_released_ready(floors,deadline=time.monotonic()+.01,guard=guard,
                             producer_epoch='epoch',prop_floors={})


def test_subthreshold_drift_accumulation_is_compared_with_original_failure():
    rt,ep,events=retained();original=rt.arm.raw.get_state;reads=[]
    def state(**kw):
        reads.append(1)
        if len(reads) in (1,2,3):rt.arm.raw.q[0]+=.0004
        return original(**kw)
    rt.arm.raw.get_state=state
    with pytest.raises(SafetyViolation,match='feedback moved'):release.recover(rt)
    assert not events


def test_preflight_rejects_feedback_drift_after_barrier_without_targets():
    rt,ep,events=retained();original=rt.arm.raw.stream_to
    def stream(*a,**kw):
        rt.arm.raw.q[0]+=.002
        return original(*a,**kw)
    rt.arm.raw.stream_to=stream
    with pytest.raises(SafetyViolation,match='before withdrawal stream'):release.recover(rt)
    assert not events


@pytest.mark.parametrize('changed',['jaws','q'])
def test_changed_atomic_feedback_on_frozen_physics_step_is_rejected(changed):
    rt,ep,events=case();clock=rt.arm.raw.get_state().physics_clock
    # Bind this step with the original closed fingers first.
    ep['clock_validator'].observe(clock)
    release.open_hand(rt,ep);events.clear()
    q=rt.arm.raw.q.copy()
    if changed=='q':q[0]+=.001
    rt.arm.raw.get_state=lambda **kw:RobotState(q=q,gripper_joints=jaws(),physics_clock=clock)
    with pytest.raises(SafetyViolation,match='without a new physics step'):release.wait_geometry(rt,ep)
    assert not events


def test_second_recovery_cannot_adopt_preflight_drift_with_no_target_ack():
    rt,ep,events=retained();original=rt.arm.raw.stream_to;failure=ep['q_failure'].copy()
    def stream(*a,**kw):
        rt.arm.raw.q[0]+=.002
        return original(*a,**kw)
    rt.arm.raw.stream_to=stream
    with pytest.raises(SafetyViolation,match='before withdrawal stream'):release.recover(rt)
    assert not events and rt.arm.raw._acknowledged_joint_targets==0
    np.testing.assert_array_equal(ep['q_failure'],failure)
    rt.arm.raw.stream_to=original
    with pytest.raises(SafetyViolation,match='retained failure'):release.recover(rt)
    assert not events


def test_real_isaac_ack_counter_requires_transport_success_and_precedes_logging(monkeypatch):
    from cascade.control.isaac_arm import IsaacArm
    from cascade.control import isaac_arm
    arm=IsaacArm(Cfg({'n_joints':3}));sent=[]
    arm._client.set_joints=lambda *a,**kw:(_ for _ in ()).throw(TimeoutError('lost ack'))
    with pytest.raises(TimeoutError):arm.send_joint_target(np.ones(3))
    assert arm._acknowledged_joint_targets==0
    arm._client.set_joints=lambda *a,**kw:sent.append(a)
    monkeypatch.setattr(isaac_arm.grasp_evidence,'event',lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('log')))
    with pytest.raises(RuntimeError,match='log'):arm.send_joint_target(np.ones(3))
    assert len(sent)==1 and arm._acknowledged_joint_targets==1


def test_concurrent_setters_cannot_change_or_clear_retained_cylinder():
    rt,ep,events=case();release.open_hand(rt,ep);release.wait_geometry(rt,ep);events.clear()
    h=rt.arm.harness;refusals=[];original=h._grasp_exempt
    h._release_scope.value=(ep,'retreat')
    def other():
        for setter in (lambda:h.allow_grasp_descent([0.,0.],radius_m=9.),h.clear_grasp_exemption):
            try:setter()
            except SafetyViolation:refusals.append(True)
    worker=threading.Thread(target=other);worker.start();worker.join(1)
    assert not worker.is_alive() and len(refusals)==2 and h._grasp_exempt is original
    with pytest.raises(SafetyViolation,match='original scoped'):
        h.allow_grasp_descent(ep['cylinder'][0],radius_m=ep['cylinder'][1]+.01,z_min=ep['cylinder'][2])
    assert not events


def test_original_contact_centre_does_not_alias_callers_array():
    rt,ep,events=case();release.finish(rt,ep,completed=True)
    xy=np.array([.2,.1]);rt.arm.harness.allow_grasp_descent(xy,z_min=.25)
    xy[:]=9.
    np.testing.assert_array_equal(rt.arm.harness._grasp_exempt[0],[.2,.1])


def test_runtime_place_failure_retains_release_then_public_reset_withdraws():
    rt,ep,events=case();release.finish(rt,ep,completed=True)
    rt.cfg._data['safety']={'table_z':0.}
    rt.cfg._data['grasp'].update(pregrasp_offset_m=.1,topdown_z_max=.8,
        post_place_retreat_offset_m=.1,close_settle_s=0.,move_duration_s=2.)
    rt.kin.ik=lambda pose,q:SimpleNamespace(q=pose[:3,3].copy(),success=True)
    rt._held_det_label='green cube';rt._held_color=None;rt._held_support_offset_m=None
    rt._held_object_offset=lambda:np.zeros(3)
    rt.memory=SimpleNamespace(add=lambda *a:None)
    rt.beliefs=SimpleNamespace(update=lambda *a,**kw:None)
    original=rt.arm.harness.occupancy.wait_released_ready
    rt.arm.harness.occupancy.wait_released_ready=lambda *a,**kw:(_ for _ in ()).throw(OccupancyError('500 ms'))
    result=rt.skill_place_at(.2,.1,.3)
    assert result['stage']=='retreat' and result['home_skipped'] and rt.held_object is None
    assert [event[0] for event in events]==['move','move','jaw']
    assert rt._release_episode['released'] and rt.arm.harness._grasp_exempt is None
    rt.arm.harness.occupancy.wait_released_ready=original;events.clear()
    rt.skill_move_home=lambda **kw:(_ for _ in ()).throw(SkillError('home normal'))
    result=rt.skill_reset_scene()
    assert result['release_recovery']['retreated_to_original_release_target']
    assert result['stage']=='home' and [e[0] for e in events]==['move']


def test_normal_pick_place_preserves_halt_between_successful_release_and_home():
    rt,ep,events=case();release.finish(rt,ep,completed=True)
    rt.cfg.arm._data['home_q']=[.2,0.,.5]
    rt._reconcile_held=lambda:None
    rt.memory=SimpleNamespace(add=lambda *a:None)
    def placed(label):
        rt.held_object=None
        rt.arm.harness.halt('between successful release and home')
        return {'placed':'green cube','at':[.2,.1,.3],'post_place_retreat':{'ok':True}}
    rt.skill_place_on_object=placed
    result=rt.skill_pick_and_place('green cube',destination='shelf')
    assert result['return_home']['ok'] is False and 'halt' in result['return_home']['error']
    assert not events and rt.arm.raw._acknowledged_joint_targets==0


@pytest.mark.parametrize('replace',['harness','occupancy'])
def test_recovery_cannot_rebind_harness_or_map_and_cleans_original_only(replace):
    rt,ep,events=retained();original=ep['harness']
    if replace=='harness':
        other=SafetyHarness(original.limits,original.kin);other.occupancy=mapping()
        other.allow_grasp_descent([.8,.8],radius_m=.01,z_min=.5)
        untouched=other._grasp_exempt;rt.arm.harness=other
    else:original.occupancy=mapping()
    with pytest.raises(SafetyViolation,match='identity changed'):release.recover(rt)
    assert not events and original._grasp_exempt is None
    if replace=='harness':assert other._grasp_exempt is untouched
