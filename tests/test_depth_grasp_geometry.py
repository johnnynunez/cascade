from types import SimpleNamespace

import numpy as np
import pytest

from cascade.grasping.obb_grasp import plan_grasps_from_fix
from cascade.perception.detector import OpenVocabDetector
from cascade.perception.support_surface import estimate_support_z
from cascade.types import Detection, Frame, ObjectFix, SkillError


def scene(height):
    h, w = 240, 320
    K = np.array([[400., 0, w / 2], [0, 400., h / 2], [0, 0, 1]])
    T = np.diag([1., -1., -1., 1.]); T[:3, 3] = [.5, 0, .6]
    depth = np.full((h, w), .6 - height, np.float32)
    mask = np.zeros((h, w), bool); mask[100:140, 140:180] = True
    depth[mask] -= .04
    frame = Frame(np.zeros((h,w,3),np.uint8),depth,K,T_base_cam=T)
    points = np.array([[.49,-.01,height+.04],[.51,-.01,height+.04],
                       [.49,.01,height+.04],[.51,.01,height+.04]])
    fix = ObjectFix('block', np.array([.5,0,height+.04]), points,
                    Detection('block', .9, np.array([140,100,180,140]), mask),
                    np.array([.02,.02,.001]), np.eye(3))
    return frame, fix


@pytest.mark.parametrize('height', [0., .005, .018, .09])
def test_grasp_tracks_measured_support_even_with_top_only_depth(height):
    frame, fix = scene(height)
    support = estimate_support_z(frame, fix)
    assert support == pytest.approx(height, abs=1e-6)
    g = plan_grasps_from_fix(fix, table_z=-.2, support_z=support, depth_fraction=.5)[0]
    assert g.position[2] == pytest.approx(height+.02, abs=1e-6)


def test_missing_support_depth_does_not_fall_back_to_configured_table():
    frame, fix = scene(.018)
    frame.depth_m[~fix.detection.mask] = 0
    with pytest.raises(SkillError, match='not enough nearby depth'):
        estimate_support_z(frame, fix)
    frame, fix = scene(.018)
    frame.T_base_cam = None
    with pytest.raises(SkillError, match='captured camera pose'):
        estimate_support_z(frame, fix)


def test_support_fit_ignores_neighbor_depth_outliers():
    frame, fix = scene(.018)
    rng = np.random.default_rng(1)
    noise = (rng.random(frame.depth_m.shape) < .2) & ~fix.detection.mask
    frame.depth_m[noise] -= rng.uniform(.05,.15,noise.sum())
    assert estimate_support_z(frame,fix) == pytest.approx(.018, abs=.001)


def test_letterbox_mask_is_unpadded_before_depth_sampling():
    import torch
    # Original object occupies y=500..560. At half scale, padded by 12 rows.
    mask = torch.zeros((1,384,640)); mask[0,262:292,200:260] = 1
    boxes = SimpleNamespace(cls=[0], conf=[.9], xyxy=[torch.tensor([400,500,520,560])])
    class Boxes:
        cls,conf,xyxy = boxes.cls,boxes.conf,boxes.xyxy
        def __len__(self): return 1
    result = SimpleNamespace(names={0:'block'},boxes=Boxes(),
                             masks=SimpleNamespace(data=mask),obb=None)
    frame=Frame(np.zeros((720,1280,3),np.uint8),None,np.eye(3))
    det=OpenVocabDetector.__new__(OpenVocabDetector)._parse(result,frame)[0]
    yy,xx=np.nonzero(det.mask)
    assert [xx.min(),yy.min(),xx.max()+1,yy.max()+1]==[400,500,520,560]


@pytest.mark.parametrize('frame_arg', ['frame', '_frame'])
def test_runtime_depth_support_survives_upstream_frame_backend_api(frame_arg):
    from cascade.config import Cfg
    from cascade.skills.runtime import SkillRuntime

    frame, fix = scene(.018)
    runtime = SkillRuntime.__new__(SkillRuntime)
    runtime.cfg = Cfg({'arm': {'tool_axis_order': 'third_open_down'},
                       'grasp': {'backend': 'obb', 'support_height_source': 'depth',
                                 'apply_learned_z_nudge': False, 'depth_fraction': .5},
                       'safety': {'table_z': -.2}})
    runtime._max_width = .08
    runtime.memory = SimpleNamespace(add=lambda *args: None)
    prior = {'nudges': {'grasp_z_delta': .017}, 'profile': 'block',
             'wins': 0, 'losses': 1, 'success_rate': 0.}
    runtime.grasp_memory = SimpleNamespace(prior=lambda *args: prior,
                                         rerank=lambda candidates, *args: candidates)
    candidates = runtime._plan_grasps(fix, **{frame_arg: frame})
    assert candidates[0].position[2] == pytest.approx(.038, abs=1e-6)
    assert runtime._last_grasp_z_nudge == 0.
