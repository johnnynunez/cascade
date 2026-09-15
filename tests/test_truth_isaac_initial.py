"""First Isaac truth reads must not materialize or resume the lazy actuator."""
import json
from types import SimpleNamespace

import pytest

from cascade.agent.effects import PostconditionChecker
from cascade.apps.demo import _truth_pose_fn
from cascade.config import Cfg
from cascade.control.lazy_arm import LazyArm
from cascade.perception.isaac_camera import IsaacCamera
from cascade.sim.truth import TruthPoseReader


def setup_camera_truth():
    cfg=Cfg({'type':'isaac','bridge_host':'127.0.0.1','bridge_port':8611,'bridge_robot_id':'/test_robot'})
    factory_calls=[]
    def forbidden_factory():
        factory_calls.append(True)
        raise AssertionError('a verification read materialized the actuator')
    arm=LazyArm(forbidden_factory,profile_type='isaac')
    class CameraClient:
        _addr=('127.0.0.1',8611)
        poses={'pink_cube':[.17,.15,.04]}
        failed=False
        calls=[]
        def request(self,request):
            self.calls.append(request['op'])
            assert request['op']=='exec'
            if self.failed:raise RuntimeError('test bridge failure')
            return {'ok':True,'stdout':'CASCADE_TRUTH_POSES '+json.dumps(self.poses)}
        def connect(self):raise AssertionError('verification connected a camera/actuator')
        def close(self):raise AssertionError('verification closed a camera-owned client')
    client=CameraClient()
    camera=IsaacCamera(cfg)  # Constructor allocates no network connection.
    camera._client=client
    frame=SimpleNamespace(capture={'backend':'isaac','source':client._addr,'t':1.,'proprioception':{
        'version':1,'backend':'isaac','joint_convention':'asset','robot_id':'/test_robot',
        't':1.,'time_source':'physics_loop_monotonic'}})
    rig=[SimpleNamespace(_camera=camera,latest=lambda:frame)]
    return cfg,arm,client,frame,rig,factory_calls


def test_demo_verifier_has_physics_before_first_arm_use():
    cfg,arm,client,frame,rig,calls=setup_camera_truth()
    fn=_truth_pose_fn(SimpleNamespace(raw=arm),camera_rig=rig,arm_cfg=cfg)
    checker=PostconditionChecker(object_pose=fn,belief_pose=lambda label:[.01,.02,.04] if label=='pink cube' else None)
    before=checker.snapshot('pink cube')
    assert before=={'label':'pink cube','pose':[.17,.15,.04],'channel':'physics'}
    assert arm.connected is False and calls==[] and client.calls==['exec']
    client.poses={'pink_cube':[.14,-.27,.04]}
    after=checker.verify('pick_and_place',{'object':'pink cube','destination':'drop zone'},
        {'ok':True,'picked':'pink cube','placed_at':[.14,-.27,.1]},before=before)
    assert after.channel=='physics' and after.confirmed
    assert after.measured['moved_m']==pytest.approx(.4211,abs=1e-4)
    assert 'start_channel_mismatch' not in after.measured
    assert arm.connected is False and calls==[] and set(client.calls)=={'exec'}


@pytest.mark.parametrize('mismatch',['profile','endpoint','capture_source','robot','missing_clock',
    'clock_source','backend','convention','camera_missing'])
def test_first_truth_rejects_unbound_camera_without_powering_arm(mismatch):
    cfg,arm,client,frame,rig,calls=setup_camera_truth()
    if mismatch=='profile':arm._profile_type='rebot_rs'
    elif mismatch=='endpoint':client._addr=('127.0.0.1',9999)
    elif mismatch=='capture_source':frame.capture['source']=('127.0.0.1',9999)
    elif mismatch=='robot':frame.capture['proprioception']['robot_id']='/another_robot'
    elif mismatch=='missing_clock':frame.capture['t']=frame.capture['proprioception']['t']=None
    elif mismatch=='clock_source':frame.capture['proprioception']['time_source']='not_physics'
    elif mismatch=='backend':frame.capture['proprioception']['backend']='mock'
    elif mismatch=='convention':frame.capture['proprioception']['joint_convention']='unknown'
    elif mismatch=='camera_missing':rig=[]
    fn=_truth_pose_fn(SimpleNamespace(raw=arm),camera_rig=rig,arm_cfg=cfg)
    assert fn is None or fn('pink cube') is None
    assert not arm.connected and calls==[] and client.calls==[]


@pytest.mark.parametrize('ttl',[0.,.25])
def test_failed_fresh_truth_probe_invalidates_expired_cache(monkeypatch,ttl):
    _,_,client,_,_,_=setup_camera_truth()
    clock=[100.]
    monkeypatch.setattr('cascade.sim.truth.time.monotonic',lambda:clock[0])
    reader=TruthPoseReader(client,ttl_s=ttl)
    assert reader.pose('pink cube')==[.17,.15,.04]
    client.failed=True;clock[0]+=ttl+.1
    assert reader.pose('pink cube') is None
    assert reader.all_poses()=={}
    client.failed=False;client.poses={'pink_cube':[.14,-.27,.04]}
    assert reader.pose('pink cube')==[.14,-.27,.04]


def test_old_no_camera_call_signature_remains_lazy():
    _,arm,client,_,_,calls=setup_camera_truth()
    fn=_truth_pose_fn(SimpleNamespace(raw=arm))
    assert fn('pink cube') is None and not arm.connected and calls==[]
