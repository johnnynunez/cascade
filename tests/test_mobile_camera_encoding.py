"""Float32 source encoding and capture fences; no Kit or model construction."""
import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.sim.mobile_camera_encoding import canonical_rig, NORM2_MAX
from cascade.sensing.models import rigid_transform


POSE22 = np.array([.0011678141308948398,.00021551286044996232,.11616756021976471,
    -.0013580911327153444,.003615526482462883,-1.4238458788895514e-5,.9999925494194031],np.float32)
MATRIX22 = np.array([[.9999738931655884,1.8656275642570108e-5,.007231037598103285,POSE22[0]],
    [-3.829713386949152e-5,.9999963641166687,.002716059098020196,POSE22[1]],
    [-.007230960298329592,-.002716264920309186,.9999701976776123,POSE22[2]], [0,0,0,1]])


def compose(pose, *, fused=False):
    # Separate float32 operations and one final rounding of the corresponding
    # real expression are both enclosed. Neither test computes a physics pose.
    q=pose[3:].astype(float if fused else np.float32)
    w,v=q[3],q[:3]; two=q.dtype.type(2);one=q.dtype.type(1)
    c=two*w*w-one
    rotation=np.column_stack([axis*c+v*(two*np.dot(v,axis))+np.cross(v,axis)*w*two
        for axis in np.eye(3,dtype=q.dtype)])
    matrix=np.eye(4);matrix[:3,:3]=rotation.astype(np.float32);matrix[:3,3]=pose[:3]
    return matrix


def test_retained_native22_reproduces_failure_then_decodes_same_quaternion():
    assert np.array_equal(compose(POSE22),MATRIX22)
    with pytest.raises(ValueError,match='rigid'):rigid_transform(tuple(MATRIX22.flat),'retained')
    result,evidence=canonical_rig(MATRIX22,POSE22,np.ones(3,np.float32))
    result=np.array(result).reshape(4,4)
    assert np.array_equal(result[:3,3],POSE22[:3])
    assert np.max(abs(result[:3,:3]@result[:3,:3].T-np.eye(3)))<1e-14
    assert evidence['raw_fabric_world_from_rig']==MATRIX22.tolist()
    assert evidence['raw_body_pose_xyzw']==POSE22.astype(float).tolist()
    actual=np.linalg.norm(result[:3,:3]-MATRIX22[:3,:3])
    assert actual<=evidence['point_encoding_error_per_m']<1e-5
    assert evidence['physical_uncertainty_estimated'] is False
    saved=copy.deepcopy(evidence);matrix=MATRIX22.copy();pose=POSE22.copy()
    other=canonical_rig(matrix,pose,np.ones(3,np.float32))[1]
    matrix[:]=99;pose[:]=99
    assert other==saved


@pytest.mark.parametrize('q',[(0,0,0,1),(1,0,0,0),(0,0,0,-1),(.5,.5,.5,.5),
    (0,np.sqrt(.5),0,np.sqrt(.5)),(np.nextafter(np.float32(0),np.float32(1)),0,0,1)])
def test_outward_intervals_cover_zero_subnormal_and_fused_arithmetic(q):
    pose=np.array([0,0,0,*q],np.float32)
    for fused in (False,True):
        result,evidence=canonical_rig(compose(pose,fused=fused),pose,np.ones(3,np.float32))
        assert len(result)==16 and np.isfinite(evidence['point_encoding_error_per_m'])


@pytest.mark.parametrize('fault',['missing_q','nonfinite','nonfloat32','norm','translation','scale',
                                  'perspective','shear','other_rotation','overflow'])
def test_decode_cannot_repair_wrong_pose_or_nonrigid_encoding(fault):
    matrix=MATRIX22.copy();pose=POSE22.copy();scale=np.ones(3,np.float32)
    if fault=='missing_q':pose=pose[:3]
    if fault=='nonfinite':pose[3]=np.nan
    if fault=='nonfloat32':pose=pose.astype(float)
    if fault=='norm':pose[6]=np.float32(np.sqrt(NORM2_MAX)+.001)
    if fault=='translation':matrix[0,3]+=1
    if fault=='scale':scale[0]=np.float32(1.000001)
    if fault=='perspective':matrix[3,0]=np.finfo(np.float32).tiny
    if fault=='shear':matrix[0,1]=np.float32(matrix[0,1]+1e-5)
    if fault=='other_rotation':pose[3:]*=-1;pose[3]=0
    if fault=='overflow':matrix[0,0]=np.finfo(float).max
    with np.errstate(over='raise'), pytest.raises(ValueError):canonical_rig(matrix,pose,scale)


@pytest.mark.parametrize('changed',['state','body_buffer','scale_buffer','clock','render'])
def test_same_values_do_not_hide_capture_object_or_clock_swap(changed):
    from test_mobile_camera_pose import fabric_fixture
    f=fabric_fixture()
    def during():
        if changed=='state':f.native.state_0=copy.copy(f.native.state_0)
        if changed=='body_buffer':f.native.state_0.body_q=copy.copy(f.native.state_0.body_q)
        if changed=='scale_buffer':f.native.fabric_manager._body_scales=copy.copy(f.native.fabric_manager._body_scales)
        if changed=='clock':f.native.simulation_step_count+=1
        if changed=='render':f.reference[0]={'changed':True}
        return f.raw
    f.reader.annotator.get_data=during
    with pytest.raises(RuntimeError):f.reader(f.cal,lambda:None)


@pytest.mark.parametrize('changed',['body_buffer','scale_buffer'])
def test_first_capture_rejects_buffer_swap_inside_first_fabric_read(changed):
    from cascade.sim.mobile_camera_pose import FABRIC_MATRIX, FabricRigReader
    from test_mobile_camera_pose import fabric_fixture
    f=fabric_fixture(); original=f.prim.GetAttribute
    def attribute(name):
        found=original(name)
        if name!=FABRIC_MATRIX:return found
        def read():
            if changed=='body_buffer':f.native.state_0.body_q=copy.copy(f.native.state_0.body_q)
            else:f.native.fabric_manager._body_scales=copy.copy(f.native.fabric_manager._body_scales)
            return found.Get()
        return SimpleNamespace(Get=read)
    f.prim.GetAttribute=attribute
    reader=FabricRigReader(f.native,f.reader.mount,f.reader.readback,f.reader.annotator)
    with pytest.raises(RuntimeError,match='state buffer'):
        reader(f.cal,lambda:None)


def test_constructor_registers_but_only_same_solve_capture_admits_encoding():
    from cascade.sim.mobile_camera_pose import FABRIC_MATRIX, FabricRigReader
    from cascade.sim.mobile_camera_encoding import FabricEncodingError
    from test_mobile_camera_pose import fabric_fixture, params
    f=fabric_fixture(); original=f.prim.GetAttribute; reads=[]
    pose=np.array([[.125,0,0,0,0,0,1]],np.float32)
    f.native.state_0.body_q.numpy=lambda:pose.copy()
    def attribute(name):
        found=original(name)
        if name!=FABRIC_MATRIX:return found
        def read():reads.append(1);return found.Get()
        return SimpleNamespace(Get=read)
    f.prim.GetAttribute=attribute
    reader=FabricRigReader(f.native,f.reader.mount,f.reader.readback,f.reader.annotator)
    assert not reads
    with pytest.raises(FabricEncodingError) as rejected:reader(f.cal,lambda:None)
    diagnostic=json.loads(str(rejected.value).split('encoding_diagnostic=',1)[1])
    assert diagnostic['native_clock']=={'step':10,'simulation_time_s':.05}
    assert diagnostic['body_pose']['values']==pose[0].tolist()
    assert diagnostic['matrix']['values']==np.eye(4).ravel().tolist()
    # A render synchronization may update Fabric without advancing physics.
    f.rig[0,3]=.125; f.raw.update(params(f.cal,f.rig))
    assert reader(f.cal,lambda:None)['world_from_rig'][3]==.125
    assert (f.native.simulation_step_count,f.native.sim_time)==(10,.05)


def test_refusal_diagnostics_are_bounded_detached_and_json_safe():
    from cascade.sim.mobile_camera_encoding import FabricEncodingError
    matrix=np.zeros((100,100));pose=np.full(30,np.nan,np.float32);scale=np.ones(12,np.float32)
    with pytest.raises(FabricEncodingError) as rejected:canonical_rig(matrix,pose,scale)
    saved=str(rejected.value);matrix[:]=1;pose[:]=1;scale[:]=2
    assert str(rejected.value)==saved
    diagnostic=json.loads(saved.split('encoding_diagnostic=',1)[1])
    assert [len(diagnostic[k]['values']) for k in ('matrix','body_pose','body_scale')]==[16,7,3]
    assert all(v['truncated'] for v in diagnostic.values())
    assert diagnostic['body_pose']['dtype']=='float32' and diagnostic['body_pose']['values']==['nan']*7


@pytest.mark.parametrize('fault',[None,'source','origin','fast_math'])
def test_source_and_actual_compiler_options_are_required(tmp_path,monkeypatch,fault):
    from cascade.sim import mobile_camera_encoding as module
    paths={name:(f'warp/{name}' if name=='warp/native/quat.h' else name.replace('.','/')+'.py')
        for name in module.SOURCE_PATHS}
    paths['warp/native/quat.h']='warp/native/quat.h'
    digests={}
    for name,relative in paths.items():
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(name);digests[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(module,'SOURCE_PATHS',paths)
    monkeypatch.setattr(module,'SOURCE_SHA256',digests)
    modules={name:SimpleNamespace(__file__=str(tmp_path/path)) for name,path in paths.items()}
    modules['warp']=SimpleNamespace(__file__=str(tmp_path/'warp/__init__.py'),
        get_module=lambda name:SimpleNamespace(options={'fast_math':fault=='fast_math'}))
    monkeypatch.setattr(module.importlib,'import_module',lambda name:modules[name])
    if fault=='source':(tmp_path/paths['mujoco_warp._src.smooth']).write_text('changed')
    if fault=='origin':
        other=tmp_path/'elsewhere.py';other.write_text('mujoco_warp._src.smooth')
        modules['mujoco_warp._src.smooth'].__file__=str(other)
    if fault is None:
        assert module.verify_encoding_sources(tmp_path,'isaac62_48b2d951')==module.encoding_descriptor()
    else:
        with pytest.raises(ValueError):module.verify_encoding_sources(tmp_path,'isaac62_48b2d951')
