"""Endpoint depth veto: analytic rays, retained failure and cancellation."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.grasping.observed_scene import ObservedFingerGate
from cascade.types import MotionHalted, SafetyViolation
from test_observed_finger_scene import construct, scene_at


def box(lo=(-.1, -.1, 2.), hi=(.1, .1, 3.)):
    lo, hi = np.array(lo), np.array(hi)
    planes = np.r_[np.c_[np.eye(3), -hi], np.c_[-np.eye(3), lo]]
    return [('right', 8, planes, lo, hi, (lo + hi) / 2, np.linalg.norm(hi - lo) / 2)]


def scene():
    return construct(*scene_at([0, 0, 1], target=False))


def test_hidden_convex_has_no_surface_point_but_endpoint_is_rejected():
    s = scene()
    assert s.conflict(np.eye(4), box()) is None
    hit = s.occluded(np.eye(4), box())
    assert hit['pixel_rc'] == [0, 0]
    assert hit['observed_depth_z_m'] == 1.
    assert hit['finger_ray_near_z_m'] == 2.
    assert hit['finger_ray_far_z_m'] == 3.


@pytest.mark.parametrize('lo,hi,hit', [
    ((-.1,-.1,.2),(.1,.1,.8), False),  # observed free segment
    ((-.1,-.1,.2),(.1,.1,1.), False),  # touching surface remains the surface gate's job
    ((0.,-.1,2.),(.1,.1,3.), True),  # tangent side ray
    ((.1,-.1,2.),(.2,.1,3.), False),  # parallel outside plane
    ((100.,100.,2.),(101.,101.,3.), False),  # outside image, no free-space claim
    ((-.1,-.1,-3.),(.1,.1,-2.), False),  # behind camera, no free-space claim
])
def test_ray_boundaries(lo, hi, hit):
    assert (scene().occluded(np.eye(4), box(lo, hi)) is not None) is hit


def test_camera_plane_crossing_is_explicit_rejection():
    hit = scene().occluded(np.eye(4), box((-.1,-.1,-1.),(.1,.1,2.)))
    assert hit['surface'] == 'endpoint envelope crosses camera plane'


@pytest.mark.parametrize('change', ['target','robot','nan','zero'])
def test_unobserved_or_exempt_pixels_make_no_extra_claim(change):
    frame, mask, T = scene_at([0,0,1], target=False)
    if change == 'target': mask[0,0] = True
    if change == 'robot': frame.robot_mask[0,0] = True
    if change == 'nan': frame.depth_m[0,0] = np.nan
    if change == 'zero': frame.depth_m[0,0] = 0
    assert construct(frame,mask,T).occluded(np.eye(4),box()) is None


def test_snapshot_arrays_are_frozen_after_construction():
    frame, mask, T = scene_at([0,0,1], target=False)
    s = construct(frame,mask,T)
    frame.depth_m[:]=100.;frame.robot_mask[:]=True;mask[:]=True;T[:3,3]=100
    assert s.occluded(np.eye(4),box()) is not None


def test_true_inverse_for_rounded_extrinsics_and_noncentral_pixel_ray():
    frame, mask, T = scene_at([0,0,1], target=False)
    T[:3,:3] = np.diag([1.+1e-7,1.,1.])
    frame.K = np.array([[2.,.3,-.2],[0.,3.,0.],[0.,0.,1.]])
    s = construct(frame,mask,T)
    assert s.occluded(np.eye(4),box((.19,-.1,2.),(.31,.1,3.))) is not None


@pytest.mark.parametrize('change', ['nan','scale','last_row'])
def test_invalid_tcp_rejected(change):
    T=np.eye(4)
    if change=='nan':T[0,0]=np.nan
    if change=='scale':T[0,0]=2
    if change=='last_row':T[3,0]=1
    with pytest.raises(SafetyViolation):scene().occluded(T,box())


def test_check_runs_inside_block_and_after_cache_hit():
    s=scene();calls=[]
    def check():
        calls.append(1)
        if len(calls)==4:raise MotionHalted('cancelled in ray block')
    with pytest.raises(MotionHalted,match='ray block'):
        s.occluded(np.eye(4),box(),check=check)
    gate=ObservedFingerGate.__new__(ObservedFingerGate)
    gate._q_shape=(1,);gate._occlusion_cache={};gate.scene=s
    gate.kin=SimpleNamespace(fk=lambda q:np.eye(4));gate.envelopes=box();gate.closing_envelopes=box()
    assert gate.occluded_pose([0.])
    calls.clear()
    def late():
        calls.append(1)
        if len(calls)==2:raise SafetyViolation('deadline at cache return')
    with pytest.raises(SafetyViolation,match='cache return'):
        gate.occluded_pose([0.],check=late)


def test_recorded_orange07_endpoint_behind_pink_depth():
    data=json.loads((Path(__file__).parent/'fixtures/orange07_occluded_endpoint.json').read_text())
    frame, mask, _ = scene_at([0,0,1],target=False)
    frame.depth_m[:]=data['depth_m'];frame.K=np.array(data['shifted_K'])
    s=construct(frame,mask,np.array(data['T_base_cam']))
    part=data['envelope'];env=[(part[0],part[1],*[np.array(x)for x in part[2:6]],part[6])]
    T=np.array(data['T_base_tcp'])
    assert s.conflict(T,env) is None
    hit=s.occluded(T,env)
    assert hit and hit['finger']=='joint_right' and hit['component']==8
    assert hit['finger_ray_far_z_m']>hit['observed_depth_z_m']
