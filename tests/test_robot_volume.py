"""Whole-body geometric contracts only; no model/SDK/physical admission."""
from dataclasses import replace
import math
import threading
import time
from types import SimpleNamespace

import pytest

from cascade.robotics.embodiment import EmbodimentDescriptor
from cascade.spatial.frames import SpatialStamp, TransformSample
from cascade.spatial.robot_volume import CollisionBox, LinkReach, RobotVolume, RobotVolumeSample
from cascade.spatial.navigation import navigation_settings
from test_spatial_navigation import navigation as navigation, arguments


def volume(robot_id='duck'):
    body=EmbodimentDescriptor(robot_id,'base','floating',('base','arm','leg'),tuple(
        dict(joint_id=link,parent='base',child=link,type='continuous',axis=[0,0,1],
             units={'position':'rad','velocity':'rad/s','effort':'N*m'}) for link in ('arm','leg')))
    return RobotVolume(robot_id,None,'a'*64,'base','joint-frames','e'*64,body,
        ('base_box','arm_box','leg_box'),
        (CollisionBox('base_box','base',(-.04,-.04,-.04),(.04,.04,.04),.001),
         CollisionBox('arm_box','arm',(.2,-.03,-.03),(.3,.03,.03),.001),
         CollisionBox('leg_box','leg',(-.03,-.03,-.1),(.03,.03,.1),.001)),
        (LinkReach('base',(0,0,0),(0,0,0),0,0),
         LinkReach('arm',(-.01,-.1,.1),(.3,.1,.3),.005,.02),
         LinkReach('leg',(-.1,-.1,-.3),(.1,.1,-.1),.005,.02)))


def observations(v, *, now=10.):
    stamp=SpatialStamp('map','map-epoch','capture',1.,'localizer','b'*64,'c'*64,'synthetic')
    base=TransformSample('map','base',(0,0,.5),(1,0,0,0),stamp)
    navigation=SimpleNamespace(robot_id=v.robot_id,model_identity_sha256=None,pose=base,
                              sensor_epoch='robot-epoch',sequence=3)
    return navigation, sample(v,navigation,now)


def sample(v,navigation,now):
    stamp=replace(navigation.pose.stamp,source_id=v.pose_source_id,calibration_id=v.pose_calibration_sha256)
    poses=tuple(TransformSample('base',link,position,(1,0,0,0),stamp,
        position_error_m=.001,angular_error_rad=.01)
        for link,position in [('arm',(.1,0,.2)),('leg',(0,0,-.2))])
    return RobotVolumeSample(v.sha256,None,navigation.sensor_epoch,navigation.sequence,stamp,now,0.,poses)


def test_full_reach_cylinder_does_not_shrink_to_instantaneous_pose_and_outputs_detach():
    v=volume(); nav,obs=observations(v)
    bound=v.validate(obs,nav,now=10.,max_age_s=.2)
    arm=v.colliders[1]
    assert v.cylinder()[0]==pytest.approx(math.hypot(.3,.1)+arm.radius_m)
    # A local box displaced from the joint origin has a real angular lever arm.
    assert bound['maximum_m'][0] > .4+.001+.001
    moved=replace(obs,link_poses=(replace(obs.link_poses[0],translation_m=(.299,0,.299)),obs.link_poses[1]))
    after=v.validate(moved,nav,now=10.,max_age_s=.2)
    assert after['maximum_m'][0]>bound['maximum_m'][0]
    assert v.cylinder()[0]>after['maximum_m'][0]
    raw=v.as_dict(); before=v.sha256
    raw['colliders'][0]['minimum_m']=(-100,-100,-100)
    raw['embodiment']['links'].append('foreign')
    assert v.sha256==before and RobotVolume.from_dict(v.as_dict())==v
    bound['maximum_m']=(999,)*3
    assert v.validate(obs,nav,now=10.,max_age_s=.2)['maximum_m'][0]<1


@pytest.mark.parametrize('fault',['collider_missing','extra_inventory','link_missing','reach_missing','duplicate','unknown_geometry'])
def test_complete_registered_inventory_cannot_omit_body_or_appendages(fault):
    value=volume().as_dict()
    if fault=='collider_missing':value['colliders'].pop()
    if fault=='extra_inventory':value['collider_inventory'].append('payload')
    if fault=='link_missing':value['colliders'][1]['link_id']='base'
    if fault=='reach_missing':value['reach'].pop()
    if fault=='duplicate':value['colliders'].append(value['colliders'][0])
    if fault=='unknown_geometry':value['colliders'][0]['error_m']=None
    with pytest.raises(ValueError):RobotVolume.from_dict(value)


@pytest.mark.parametrize('fault',['stale','future','epoch','sequence','model','calibration','capture',
                                'unknown_position','unknown_angle','reach','missing','static','frame','synthetic'])
def test_geometry_capture_must_be_complete_current_registered_and_bounded(fault):
    v=volume();nav,obs=observations(v)
    if fault=='stale':obs=replace(obs,producer_age_s=.3)
    if fault=='future':obs=replace(obs,received_monotonic_s=11.)
    if fault=='epoch':obs=replace(obs,sensor_epoch='reset')
    if fault=='sequence':obs=replace(obs,sequence=4)
    if fault=='model':obs=replace(obs,model_identity_sha256='d'*64)
    if fault=='calibration':obs=replace(obs,stamp=replace(obs.stamp,calibration_id='f'*64))
    if fault=='capture':obs=replace(obs,stamp=replace(obs.stamp,time_s=1.01))
    if fault=='unknown_position':obs=replace(obs,link_poses=(replace(obs.link_poses[0],position_error_m=None),obs.link_poses[1]))
    if fault=='unknown_angle':obs=replace(obs,link_poses=(replace(obs.link_poses[0],angular_error_rad=None),obs.link_poses[1]))
    if fault=='reach':obs=replace(obs,link_poses=(replace(obs.link_poses[0],translation_m=(.3,0,.2)),obs.link_poses[1]))
    if fault=='missing':obs=replace(obs,link_poses=obs.link_poses[:1])
    if fault=='static':obs=replace(obs,link_poses=(replace(obs.link_poses[0],static=True),obs.link_poses[1]))
    if fault=='frame':obs=replace(obs,link_poses=(replace(obs.link_poses[0],parent='optical'),obs.link_poses[1]))
    if fault=='synthetic':nav.pose=replace(nav.pose,stamp=replace(nav.pose.stamp,measurement_kind='estimated'))
    with pytest.raises(ValueError):v.validate(obs,nav,now=10.,max_age_s=.2)


def install_volume(runtime,source):
    v=volume(runtime.profile['robot_id'])
    r,lo,hi=v.cylinder()
    settings={**runtime.settings,'robot_volume':v.as_dict(),'geometry_sha256':v.sha256,
              'body_radius_m':r,'body_z_min_m':lo,'body_z_max_m':hi}
    runtime.settings,_,_=navigation_settings(settings,runtime.mobile.cfg.bases)
    runtime.robot_volume=v
    source.mutate=lambda nav:replace(nav,geometry_sha256=v.sha256,
        robot_volume=sample(v,nav,source.clock()))
    return v,settings


def test_native_configuration_requires_whole_geometry_before_io_and_checks_enclosure(navigation):
    runtime,source=navigation
    physical={**runtime.profile,'type':'isaac','model_identity_sha256':'f'*64}
    with pytest.raises(ValueError,match='whole-robot volume'):
        navigation_settings(runtime.settings,[physical])
    v,settings=install_volume(runtime,source)
    with pytest.raises(ValueError,match='omits'):
        navigation_settings({**settings,'body_radius_m':.05},[runtime.profile])
    with pytest.raises(ValueError,match='identity'):
        navigation_settings(settings,[physical])


def test_registered_geometry_flows_through_route_and_query_hash(navigation):
    runtime,source=navigation
    install_volume(runtime,source)
    result=runtime.execute('go_to',arguments(source,(0.,0.)))
    assert result.get('software_complete'),result
    assert not result['ok'] and not result['physical_admission']
    assert source.queries and all('geometry_sample_sha256' in q for q in source.queries)


def test_unknown_link_geometry_stops_route_before_mobile_dispatch(navigation,monkeypatch):
    runtime,source=navigation
    install_volume(runtime,source)
    good=source.mutate
    def unknown(nav):
        nav=good(nav)
        poses=nav.robot_volume.link_poses
        return replace(nav,robot_volume=replace(nav.robot_volume,
            link_poses=(replace(poses[0],angular_error_rad=None),poses[1])))
    source.mutate=unknown
    monkeypatch.setattr(runtime.mobile,'execute',lambda *a,**k:pytest.fail('motion must not dispatch'))
    result=runtime.execute('go_to',arguments(source))
    assert not result['execution_ok'] and 'uncertainty' in result['error']


def test_retained_geometry_is_rechecked_at_clearance_boundary(navigation,monkeypatch):
    runtime,source=navigation
    install_volume(runtime,source)
    nav=source.read(deadline_monotonic_s=time.monotonic()+1)
    nav=replace(nav,robot_volume=replace(nav.robot_volume,producer_age_s=1.))
    monkeypatch.setattr(source,'swept_clearance',lambda *a,**k:pytest.fail('stale geometry query'))
    with pytest.raises(ValueError,match='stale'):
        runtime._clearance(nav,(0.,0.),time.monotonic()+1)


def test_motion_authority_expires_with_older_geometry_even_when_base_is_fresh(navigation,monkeypatch):
    from cascade.spatial import navigation as module
    runtime,source=navigation
    install_volume(runtime,source)
    nav=source.read(deadline_monotonic_s=time.monotonic()+1)
    nav=replace(nav,received_monotonic_s=0.,producer_age_s=0.,
        robot_volume=replace(nav.robot_volume,received_monotonic_s=0.,producer_age_s=.4))
    clock=[.09]
    monkeypatch.setattr(module,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    op=dict(cancel=threading.Event(),serial=runtime._serial,deadline=1.,samples=[nav])
    assert runtime._current(op)
    clock[0]=.11
    assert runtime._age(nav)<runtime.limits['max_state_age_s']
    assert not runtime._current(op)


@pytest.mark.parametrize('boundary',['geometry','provider'])
def test_clearance_does_not_publish_geometry_that_expires_during_work(navigation,monkeypatch,boundary):
    from cascade.spatial import navigation as module
    runtime,source=navigation
    install_volume(runtime,source)
    nav=source.read(deadline_monotonic_s=time.monotonic()+1)
    nav=replace(nav,received_monotonic_s=0.,producer_age_s=0.,
        robot_volume=replace(nav.robot_volume,received_monotonic_s=0.,producer_age_s=.4))
    clock=[0.]
    monkeypatch.setattr(module,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    if boundary=='geometry':
        original=RobotVolume.validate
        def validate(*args,**kwargs):
            result=original(*args,**kwargs)
            clock[0]=.11
            return result
        monkeypatch.setattr(RobotVolume,'validate',validate)
    else:
        original=source.swept_clearance
        def clearance(*args,**kwargs):
            result=original(*args,**kwargs)
            clock[0]=.11
            return result
        monkeypatch.setattr(source,'swept_clearance',clearance)
    with pytest.raises(ValueError,match='stale'):
        runtime._clearance(nav,(0.,0.),1.)
    assert len(source.queries)==(boundary=='provider')
