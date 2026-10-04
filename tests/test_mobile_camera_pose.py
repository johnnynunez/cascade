"""SDK-shaped records and real USD local mounts; no Kit or rendered-pixel proof."""
import copy
import hashlib
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from cascade.sim.mobile_camera_pose import (FabricRigReader, admit_mount, camera_params_record,
                                          camera_path, mount_record)
from cascade.sim.mobile_rgbd import read_mount_calibration, read_static_calibration
from test_mobile_rgbd_pose import mount_calibration
from test_microduck_stepper import render_times
from test_mobile_identity import recipe_inputs as recipe_inputs


def mount():
    cal = mount_calibration()
    return dict(schema='cascade.rigid-render-camera.v1', rig_prim_path='/World/Robot/base',
        rig_frame_id=cal['rig_frame_id'], rig_from_camera=cal['rig_from_camera'],
        position_error_m=cal['mount_position_error_m'], angular_error_rad=cal['mount_angular_error_rad'])


def params(cal, world_from_rig):
    optical = world_from_rig @ np.array(cal['rig_from_camera']).reshape(4,4)
    usd_camera = optical @ np.diag([1.,-1.,-1.,1.])
    w,h = cal['width'],cal['height']; fx,fy = cal['intrinsics'][0],cal['intrinsics'][4]
    projection = np.zeros((4,4)); projection[0,0]=2*fx/w; projection[1,1]=2*fy/h
    projection[2,3]=-1.; projection[3,2]=1.
    return dict(cameraModel='pinhole', cameraViewTransform=np.linalg.inv(usd_camera).T.reshape(-1),
        cameraProjection=projection.reshape(-1), cameraFocalLength=1.,
        cameraAperture=np.array([w/fx,h/fy]), cameraApertureOffset=np.zeros(2),
        renderProductResolution=np.array([w,h]), metersPerSceneUnit=1.)


def test_render_view_convention_matches_nontrivial_rig_rotation_and_lever_arm():
    cal=mount_calibration(); rig=np.eye(4)
    rig[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]; rig[:3,3]=[1,2,3]
    raw=params(cal,rig)
    record, optical=camera_params_record(raw,cal)
    assert np.array(optical).reshape(4,4) == pytest.approx(rig @ np.array(cal['rig_from_camera']).reshape(4,4))
    raw['cameraViewTransform'][:]=99
    assert record['cameraViewTransform'][0] != 99
    # The SDK's example has translation -100 in the last row for eye +100.
    raw=params(cal,np.eye(4)); raw['cameraViewTransform']=np.eye(4).reshape(-1)
    raw['cameraViewTransform'][12]=-100
    assert np.array(camera_params_record(raw,cal)[1]).reshape(4,4)[0,3] == 100


@pytest.mark.parametrize('w', [1. - 8*np.spacing(1.), 1. + 8*np.spacing(1.)])
def test_render_homogeneous_roundoff_normalizes_all_entries_and_preserves_raw(w):
    cal=mount_calibration(); rig=np.eye(4); rig[:3,3]=[.5,.45,.125]
    raw=params(cal,rig); raw['cameraViewTransform']*=w
    saved=raw['cameraViewTransform'].copy()
    record,optical=camera_params_record(raw,cal)
    assert np.array_equal(raw['cameraViewTransform'],saved)
    assert record['cameraViewTransform']==saved.tolist()
    result=np.array(optical).reshape(4,4)
    assert np.array_equal(result[3],[0.,0.,0.,1.])
    assert result == pytest.approx(rig@np.array(cal['rig_from_camera']).reshape(4,4),abs=1e-14)


def test_actual_gf_affine_inverse_with_retained_bootstrap_mount():
    Gf=pytest.importorskip('pxr.Gf')
    # Authored local optics from failed Fabric reference02. Gf's affine inverse
    # reproduces homogeneous roundoff; no native rejected matrix was retained.
    mount=np.array([
        [-.7150066493027298,.2306943989907678,-.6599587757785872,.5],
        [.6991176520821676,.23593746522544112,-.6749577920508149,.44999998807907104],
        [1.8529186796012453e-8,-.9439881391083694,-.32997938302675434,.19499999284744263],
        [0.,0.,0.,1.]])
    rig=np.eye(4);rig[2,3]=.125
    usd=rig@mount@np.diag([1.,-1.,-1.,1.])
    raw=params(mount_calibration(),rig)
    raw['cameraViewTransform']=np.array(Gf.Matrix4d(tuple(map(tuple,usd.T))).GetInverse()).reshape(-1)
    record,optical=camera_params_record(raw,mount_calibration())
    assert record['cameraViewTransform']==raw['cameraViewTransform'].tolist()
    assert np.array(optical).reshape(4,4)==pytest.approx(rig@mount,abs=1e-14)


@pytest.mark.parametrize('kind',['perspective','w','scale','shear','reflection'])
def test_homogeneous_canonicalization_cannot_repair_nonrigid_or_projective_view(kind):
    cal=mount_calibration();raw=params(cal,np.eye(4))
    view=raw['cameraViewTransform'].reshape(4,4)
    if kind=='perspective': view[0,3]=np.spacing(1.)
    if kind=='w': view[3,3]=1.+9*np.spacing(1.)
    if kind=='scale': view[:3,:3]*=1.001
    if kind=='shear': view[0,1]+=.001
    if kind=='reflection': view[0,:3]*=-1.
    with pytest.raises(ValueError,match='render world-to-view'):
        camera_params_record(raw,cal)


@pytest.mark.parametrize('change', [
    {'cameraModel':'fisheyePolynomial'}, {'metersPerSceneUnit':.01},
    {'cameraApertureOffset':[.1,0]}, {'renderProductResolution':[480,640]},
    {'cameraFocalLength':2.}, {'cameraViewTransform':[0.]*16},
    {'cameraProjection':[0.]*16}, {'cameraAperture':[float('nan'),1.]},
])
def test_render_optics_and_transform_cannot_override_calibration(change):
    cal=mount_calibration()
    with pytest.raises(ValueError):
        camera_params_record(params(cal,np.eye(4)) | change, cal)


def fabric_fixture():
    cal=mount_calibration(); rig=np.eye(4); index=[0]; reference=[render_times(.05)]
    attr=lambda getter:NS(Get=getter)
    prim=NS(IsValid=lambda:True, GetAttribute=lambda name:attr(
        (lambda:index[0]) if name=='newton:index' else (lambda:rig.T.copy())))
    stage=NS(GetPrimAtPath=lambda path:prim)
    body=np.array([[0,0,0,0,0,0,1]],dtype=np.float32)
    scales=np.ones((1,3),dtype=np.float32)
    # The fixture explicitly supplies the co-captured native pose; Fabric is
    # never allowed to synthesize a missing quaternion from its matrix.
    def bodies():
        body[0,:3]=rig[:3,3]
        return body.copy()
    native=NS(model=NS(body_label=[mount()['rig_prim_path']]),
        fabric_manager=NS(stage=stage,_body_scales=NS(numpy=lambda:scales.copy())),
        state_0=NS(body_q=NS(numpy=bodies)),simulation_step_count=10,sim_time=.05,scene_scale=1.)
    raw=params(cal,rig)
    reader=FabricRigReader(native,mount(),NS(get_render_times=lambda:reference[0]),NS(get_data=lambda:raw))
    return NS(**locals())


def test_fabric_registration_and_same_render_are_rechecked_and_detached():
    f=fabric_fixture(); first=f.reader(f.cal,lambda:None)
    assert first['position_error_m'] is first['angular_error_rad'] is None
    assert first['world_from_rig'] == list(np.eye(4).flat)
    f.rig[0,3]=float(np.float32(.3))
    with pytest.raises(RuntimeError,match='differs'):
        f.reader(f.cal,lambda:None)
    f.raw.update(params(f.cal,f.rig))
    second=f.reader(f.cal,lambda:None)
    assert second['world_from_rig'][3] == float(np.float32(.3)) and first['world_from_rig'][3] == 0
    f.index[0]=1
    with pytest.raises(ValueError,match='index'):
        f.reader(f.cal,lambda:None)
    f.index[0]=False
    with pytest.raises(ValueError,match='index'):
        f.reader(f.cal,lambda:None)
    f.index[0]=0; f.native.model=copy.deepcopy(f.native.model)
    with pytest.raises(ValueError,match='model'):
        f.reader(f.cal,lambda:None)


@pytest.mark.parametrize('what',['rig','render','cancel'])
def test_mid_read_change_or_checkpoint_failure_rejects_capture(what):
    f=fabric_fixture()
    def during():
        if what=='rig': f.rig[0,3]=float(np.float32(.001))
        if what=='render': f.reference[0]=render_times(.055)
        if what=='cancel': raise RuntimeError('cancelled')
        return f.raw
    f.reader.annotator.get_data=during
    with pytest.raises(RuntimeError): f.reader(f.cal,lambda:None)


def test_mount_sha_and_unknown_bounds_are_explicit(tmp_path):
    value=mount(); value.update(position_error_m=None,angular_error_rad=None)
    path=tmp_path/'mount.json'; path.write_text(json.dumps(value))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert admit_mount(path,digest)['definition']==mount_record(value)
    path.write_text(json.dumps(value | {'rig_frame_id':'changed'}))
    with pytest.raises(ValueError,match='SHA256'): admit_mount(path,digest)


@pytest.mark.parametrize('change', [None,'pose_between_channels','reference_between_channels','buffer_between_channels'])
def test_native_capture_keeps_pose_and_both_aovs_at_one_render_completion(change):
    from cascade.sim.microduck_newton import capture_bound_rgb
    cal=mount_calibration(); cal.update(width=640,height=480)
    pose=dict(world_from_rig=list(np.eye(4).flat),position_error_m=None,angular_error_rad=None,
              render_reference=render_times(.05),evidence={'source':'fixture'})
    ns=NS(simulation_step_count=10,sim_time=.05,update_fabric=lambda:None)
    captured_objects=[object()]
    def data(name,checkpoint):
        if name=='distance_to_image_plane':
            if change=='pose_between_channels': pose['world_from_rig'][3]=1.
            if change=='reference_between_channels': pose['render_reference']=render_times(.055)
            if change=='buffer_between_channels':captured_objects[0]=object()
            return np.ones((480,640),np.float32),{},render_times(.05)
        return np.zeros((480,640,3),np.uint8),{},render_times(.05)
    reader=NS(get_render_times=lambda:render_times(.05),get_data_bound=data)
    def pose_reader(cal,check):return copy.deepcopy(pose)
    pose_reader.capture_objects=lambda:tuple(captured_objects)
    def capture():
        return capture_bound_rgb(ns,NS(update=lambda:None),reader,updates=1,
            calibration=lambda:copy.deepcopy(cal),pose_reader=pose_reader)
    if change:
        with pytest.raises(RuntimeError,match='one render completion|native pose objects'): capture()
    else:
        value=capture()
        assert value['capture_pose']['render_reference']==value['render_times']
        assert value['pose_evidence']=={'source':'fixture'}
        assert (ns.simulation_step_count,ns.sim_time)==(10,.05)


def test_model_identity_rehashes_mount_and_refuses_unadmitted_native_mount(recipe_inputs,tmp_path):
    from cascade.sim.mobile_identity import build_model_identity
    from cascade.sim.mobile_rgbd import calibration_record
    admission,native,paths=recipe_inputs
    path=tmp_path/'mount.json'; path.write_text(json.dumps(mount()))
    admitted=admit_mount(path,hashlib.sha256(path.read_bytes()).hexdigest())
    cal,digest=calibration_record(mount_calibration())
    native['rgbd_camera']={'calibration':cal,'calibration_sha256':digest,
        'mount':{k:v for k,v in admitted.items() if k!='path'}}
    with pytest.raises(ValueError,match='lacks explicit'): build_model_identity(admission,native,**paths)
    admission['camera_mount']=admitted
    from cascade.sim.mobile_camera_encoding import encoding_descriptor
    admission['camera_pose_encoding']=encoding_descriptor()
    native['rgbd_camera']['pose_encoding']=encoding_descriptor()
    before=build_model_identity(admission,native,**paths)
    assert str(path) not in json.dumps(before)
    changed=copy.deepcopy(native)
    changed['rgbd_camera']['pose_encoding']['fast_math']=True
    with pytest.raises(ValueError,match='encoding'):build_model_identity(admission,changed,**paths)
    bad_admission=copy.deepcopy(admission)
    bad_admission['camera_pose_encoding']=changed['rgbd_camera']['pose_encoding']
    with pytest.raises(ValueError,match='encoding'):build_model_identity(bad_admission,changed,**paths)
    assert before['recipe']['native']['rgbd_camera']['pose_encoding']==encoding_descriptor()
    path.write_text(json.dumps(mount() | {'rig_frame_id':'other'}))
    with pytest.raises(ValueError,match='SHA256'): build_model_identity(admission,native,**paths)


def test_real_usd_local_mount_ignores_world_pose_but_refuses_mount_or_lens_animation():
    Gf, Usd, UsdGeom, UsdPhysics = (pytest.importorskip('pxr.'+name)
                                 for name in ('Gf', 'Usd', 'UsdGeom', 'UsdPhysics'))
    stage=Usd.Stage.CreateInMemory(); UsdGeom.SetStageMetersPerUnit(stage,1.)
    stage.DefinePrim('/World','Xform'); stage.DefinePrim('/World/Robot','Xform')
    definition=mount(); rig=UsdGeom.Xform.Define(stage,definition['rig_prim_path'])
    UsdPhysics.RigidBodyAPI.Apply(rig.GetPrim())
    move=rig.AddTranslateOp(); move.Set(Gf.Vec3d(100,200,300))
    camera=UsdGeom.Camera.Define(stage,camera_path(definition))
    camera.GetFocalLengthAttr().Set(18.); camera.GetHorizontalApertureAttr().Set(20.955)
    camera.GetVerticalApertureAttr().Set(20.955*480/640)
    local=np.array(definition['rig_from_camera']).reshape(4,4) @ np.diag([1.,-1.,-1.,1.])
    op=camera.AddTransformOp(); op.Set(Gf.Matrix4d(tuple(map(tuple,local.T))))
    first=read_mount_calibration(stage,camera_path(definition),definition)
    move.Set(Gf.Vec3d(1,2,3),Usd.TimeCode(1.))
    assert read_mount_calibration(stage,camera_path(definition),definition)==first
    with pytest.raises(ValueError,match='static'): read_static_calibration(stage,camera_path(definition))
    op.Set(Gf.Matrix4d(1.),Usd.TimeCode(1.))
    with pytest.raises(ValueError,match='static'): read_mount_calibration(stage,camera_path(definition),definition)


@pytest.mark.parametrize('change', [
    {'camera_mount':None}, {'camera_mount_sha256':None}, {'camera_rgbd':False},
    {'sdk_recipe':None},
])
def test_mount_opt_in_requires_paired_hash_rgbd_and_current_sdk_before_live_imports(change,monkeypatch):
    from test_microduck_bridge_cli import cli, forbid_live_imports
    forbid_live_imports(monkeypatch)
    args=NS(engine='newton',camera_mount='/missing/mount.json',camera_mount_sha256='a'*64,
            camera_rgbd=True,sdk_recipe='isaac62_48b2d951')
    vars(args).update(change)
    with pytest.raises(ValueError,match='paired file/SHA'):
        cli().admit(args)
