"""B32: semantic fusion must not turn the robot's own body into an object.

Measured on Isaac 6.2 PhysX with YOLOE prompt-free (`docs/evidence/
b32-fusion-self-mask-20261008/`): the side camera sees the arm's upper link at
the top of its image, YOLOE names it "biplane"/"fighter jet", 98-99 % of the
detection's pixels are robot pixels in the bridge's render self-mask, and the
link is outside the workspace filter's base cylinder. Fusion never consulted
the self-mask, so the link became a belief 0.42 m above the table (all of the
bare scene's 17.9-19.6 % phantom rate). Props measured <= 4.6 % robot pixels,
robot detections >= 88.5 %.

The gate lives in WorkspaceFilter (class-agnostic, a property of the rig) and
both fusion paths (WorldWatcher._tick and get_observation) apply it.
"""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cascade.memory.beliefs import BeliefStore
from cascade.perception.workspace import WorkspaceFilter
from cascade.perception.world import WatchedCamera, WorldWatcher
from cascade.types import Detection, Frame

H, W = 120, 160
K = np.array([[100.0, 0.0, 80.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]])
# A camera 0.6 m above the table looking straight down at (0.3, 0, 0).
T = np.eye(4)
T[:3, :3] = [[0, -1, 0], [-1, 0, 0], [0, 0, -1]]
T[:3, 3] = [0.3, 0.0, 0.6]


def _box(r0, r1, c0, c1):
    m = np.zeros((H, W), dtype=bool)
    m[r0:r1, c0:c1] = True
    return m


def _det(label, mask, conf=0.6):
    ys, xs = np.nonzero(mask)
    bbox = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)
    return Detection(label, conf, bbox, mask=mask)


def _base_z(depth_m):
    return 0.6 - depth_m


CUBE = _box(50, 70, 30, 50)      # a prop, top at z = 0.05
LINK = _box(10, 30, 100, 140)    # the robot's upper link, z = 0.25, far outside the base cylinder
CUP = _box(80, 100, 100, 120)    # a prop, top at z = 0.10 ...
CUP_ON_ROBOT = _box(80, 86, 100, 120)  # ... whose sloppy mask bleeds 30 % onto a link at z = 0.30
HELD = _box(50, 70, 120, 140)    # a prop in the jaws (z = 0.15): robot mask AND payload mask


def _frame(*, robot=True, payload=True, frame_id=1):
    depth = np.full((H, W), 0.6, dtype=np.float32)
    depth[CUBE] = 0.55
    depth[LINK] = 0.35
    depth[CUP] = 0.50
    depth[CUP_ON_ROBOT] = 0.30
    depth[HELD] = 0.45
    f = Frame(rgb=np.zeros((H, W, 3), dtype=np.uint8), depth_m=depth, K=K.copy(),
              frame_id=frame_id, t=1000.0 + frame_id, depth_source="sensor", T_base_cam=T.copy())
    if robot:
        f.robot_mask = LINK | CUP_ON_ROBOT | HELD
        if payload:
            f.payload_mask = HELD.copy()
    return f


def _dets():
    return [_det("cube", CUBE), _det("biplane", LINK), _det("cup", CUP), _det("block", HELD)]


def _watch(workspace=None):
    stream = SimpleNamespace(name="side", latest=Mock(), set_overlay=Mock())
    depth = SimpleNamespace(ensure_depth=Mock(side_effect=lambda frame: frame))
    detector = SimpleNamespace(detect=Mock(return_value=_dets()))
    cam = WatchedCamera(stream, depth, None, fuse=True)
    beliefs = BeliefStore()
    watch = WorldWatcher([cam], detector, beliefs, workspace=workspace or WorkspaceFilter(),
                         harness=SimpleNamespace(heartbeat=Mock()))
    return watch, cam, beliefs


def _by_label(beliefs):
    out = {}
    for b in beliefs.all():
        for name in {b.label, *b.aliases}:
            out[name] = b
    return out


def _fuse_watcher(frame, workspace=None):
    watch, cam, beliefs = _watch(workspace)
    cam.stream.latest.return_value = frame
    watch._tick(cam)
    return _by_label(beliefs)


def _fuse_observation(frame, workspace=None):
    from cascade.skills.runtime import SkillRuntime
    rt = SimpleNamespace(_workspace=workspace or WorkspaceFilter(), beliefs=BeliefStore(), extrinsics=None)
    summaries = SkillRuntime._update_beliefs_from_frame(rt, frame, _dets(), T=frame.T_base_cam)
    return _by_label(rt.beliefs), summaries


@pytest.fixture(params=["watcher", "get_observation"])
def fuse(request):
    if request.param == "watcher":
        return _fuse_watcher
    return lambda frame, workspace=None: _fuse_observation(frame, workspace)[0]


def test_a_detection_on_the_robots_own_link_is_not_an_object(fuse):
    seen = fuse(_frame())
    assert "biplane" not in seen
    assert "cube" in seen


def test_robot_pixels_do_not_drag_a_props_position(fuse):
    seen = fuse(_frame())
    cup = seen["cup"]
    # Only the cup's own pixels are lifted: its top, not the link above it.
    assert cup.position[2] == pytest.approx(_base_z(0.50), abs=0.005)
    assert cup.extent is None or max(cup.extent) < 0.2


def test_a_held_payload_is_still_an_object(fuse):
    seen = fuse(_frame())
    assert "block" in seen
    assert seen["block"].position[2] == pytest.approx(_base_z(0.45), abs=0.005)


def test_without_a_payload_mask_the_whole_render_mask_is_the_robot(fuse):
    seen = fuse(_frame(payload=False))
    assert "block" not in seen and "biplane" not in seen
    assert "cube" in seen


def test_a_frame_without_a_render_self_mask_is_fused_as_before(fuse):
    seen = fuse(_frame(robot=False))
    # No self-mask, no evidence: the geometric gate alone decides (the real
    # rig until a link-geometry mask reaches fusion, ROADMAP B32 follow-up).
    assert {"cube", "biplane", "cup", "block"} <= set(seen)


def test_the_gate_can_be_switched_off_for_an_ab_baseline(fuse):
    seen = fuse(_frame(), WorkspaceFilter(self_mask=False))
    assert "biplane" in seen


def test_get_observation_does_not_report_the_robot():
    _, summaries = _fuse_observation(_frame())
    assert "biplane" not in {s["label"] for s in summaries}
    assert "cube" in {s["label"] for s in summaries}


@pytest.mark.parametrize("robot_rows, kept", [(10, True), (12, False)])
def test_the_robot_fraction_threshold_is_inclusive_at_half(robot_rows, kept):
    gate = WorkspaceFilter()
    mask = _box(0, 20, 0, 20)
    robot = _box(0, robot_rows, 0, 20)
    out, frac = gate.exclude_self(mask, robot)
    assert frac == pytest.approx(robot_rows / 20)
    if kept:
        assert out is not None and int(out.sum()) == (20 - robot_rows) * 20
        assert not (out & robot).any()
    else:
        assert out is None


def test_no_self_mask_or_a_mismatched_one_leaves_the_detection_untouched():
    gate = WorkspaceFilter()
    mask = _box(0, 20, 0, 20)
    assert gate.exclude_self(mask, None) == (mask, None)
    out, frac = gate.exclude_self(mask, np.ones((5, 5), dtype=bool))
    assert out is mask and frac is None


def test_self_pixels_is_the_render_mask_minus_the_payload():
    gate = WorkspaceFilter()
    f = _frame()
    px = gate.self_pixels(f)
    assert px.dtype == bool and px.shape == (H, W)
    assert not (px & HELD).any() and px[LINK].all() and px[CUP_ON_ROBOT].all()
    assert gate.self_pixels(_frame(robot=False)) is None
    assert WorkspaceFilter(self_mask=False).self_pixels(f) is None
    bad = _frame()
    bad.payload_mask = np.zeros((4, 4), dtype=bool)
    assert gate.self_pixels(bad) is None  # cannot tell the payload apart: no gate


def test_config_block_sets_the_gate():
    gate = WorkspaceFilter.from_config({"self_mask": False, "self_mask_max_frac": 0.3})
    assert gate.self_mask is False and gate.self_mask_max_frac == 0.3
    default = WorkspaceFilter.from_config({})
    assert default.self_mask is True and default.self_mask_max_frac == 0.5
    with pytest.raises(ValueError):
        WorkspaceFilter.from_config({"self_mask_max_frac": 0.0})
    with pytest.raises(ValueError):
        WorkspaceFilter.from_config({"self_mask_max_frac": 1.5})


def test_demo_config_enables_the_gate_at_the_measured_threshold():
    from cascade.config import load_demo_config
    cfg = load_demo_config(cameras=["isaac"], arm="isaac", llm="mock")
    gate = WorkspaceFilter.from_config(cfg.get("workspace_filter"))
    assert gate.self_mask is True and gate.self_mask_max_frac == 0.5


def test_torch_masks_from_the_strict_cuda_detector_are_gated_on_their_device():
    torch = pytest.importorskip("torch")
    gate = WorkspaceFilter()
    mask = torch.as_tensor(_box(0, 20, 0, 20))
    out, frac = gate.exclude_self(mask, _box(0, 5, 0, 20))
    assert isinstance(out, torch.Tensor) and out.device == mask.device
    assert frac == pytest.approx(0.25) and int(out.sum()) == 300
    out, frac = gate.exclude_self(mask, _box(0, 15, 0, 20))
    assert out is None and frac == pytest.approx(0.75)


@pytest.mark.parametrize("path", ["watcher", "get_observation"])
def test_the_scene_fraction_is_judged_on_the_detection_not_on_its_robot_free_rest(path):
    # A detection over max_frame_frac of the image IS the scene, even when
    # removing the robot's pixels would leave less than that: 52 % of the
    # frame, a 0.3 x 0.3 m patch 0.3 m up; its robot-free rest is 44 %.
    big = _box(10, 110, 30, 130)
    depth = np.full((H, W), 0.6, dtype=np.float32)
    depth[big] = 0.3
    f = Frame(rgb=np.zeros((H, W, 3), dtype=np.uint8), depth_m=depth, K=K.copy(), frame_id=1,
              t=1001.0, depth_source="sensor", T_base_cam=T.copy())
    f.robot_mask = _box(10, 25, 30, 130)  # 15 % of the detection
    dets = [_det("studio shot", big)]
    if path == "watcher":
        watch, cam, beliefs = _watch()
        watch._detector.detect.return_value = dets
        cam.stream.latest.return_value = f
        watch._tick(cam)
    else:
        from cascade.skills.runtime import SkillRuntime
        rt = SimpleNamespace(_workspace=WorkspaceFilter(), beliefs=BeliefStore(), extrinsics=None)
        SkillRuntime._update_beliefs_from_frame(rt, f, dets, T=f.T_base_cam)
        beliefs = rt.beliefs
    assert beliefs.all() == []
