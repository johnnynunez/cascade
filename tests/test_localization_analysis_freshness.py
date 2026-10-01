"""Localization must expire the image it analyzed, without actuating or retrying."""
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.agent.trace import TraceLogger
from cascade.config import Cfg
from cascade.control.lazy_arm import LazyArm
from cascade.memory import BeliefStore, EpisodicMemory
from cascade.perception.depth_provider import DepthProvider
from cascade.perception.grounding import Extrinsics
from cascade.skills import runtime as module
from cascade.skills.runtime import SkillRuntime, SlowPerceptionError
from cascade.types import Detection, Frame, SkillError


def image(t=100., x=0.):
    T = np.eye(4)
    T[0, 3] = x
    return Frame(np.zeros((20, 20, 3), np.uint8),
                 np.full((20, 20), .5, np.float32),
                 np.array([[100., 0., 10.], [0., 100., 10.], [0., 0., 1.]]),
                 t=t, T_base_cam=T, depth_source="sensor",
                 capture={"t": -9999999.})  # Foreign clock must never be compared.


def detection():
    return Detection("object", .9, np.array([3, 3, 17, 17]),
                     mask=np.pad(np.ones((14, 14), bool), 3))


@pytest.fixture
def rig(monkeypatch, demo_cfg, tmp_path):
    clock = SimpleNamespace(now=100.)
    monkeypatch.setattr(module, "time", SimpleNamespace(
        monotonic=lambda: clock.now, time=lambda: clock.now))
    primary = image()
    calls = []
    def never_materialize():
        pytest.fail("Localization materialized LazyArm")
    lazy = LazyArm(never_materialize)
    arm = SimpleNamespace(raw=lazy, harness=SimpleNamespace(heartbeat=lambda: None))
    demo_cfg._data.update(perception_loop={"localize_frames": 3}, grounder=None)
    rt = SkillRuntime(SimpleNamespace(get_frame=lambda: primary),
        DepthProvider(Cfg({})), None, Extrinsics(), None, arm,
        EpisodicMemory(), BeliefStore(), TraceLogger(tmp_path / "trace"), demo_cfg)
    rt._resolve_query = lambda query: dict(prompts=[query], color=None, near_xyz=None,
        vocab=None, prefer_label=None, configured_description=False, belief=None)
    rt._other_cams = lambda: []
    def set_detector(delay=0., found=True):
        def detect(frame, **kwargs):
            calls.append(frame)
            clock.now += delay
            return [detection()] if found else []
        rt.detector = SimpleNamespace(detect=detect)
    set_detector()
    return SimpleNamespace(rt=rt, clock=clock, primary=primary, calls=calls,
                           detector=set_detector, lazy=lazy)


@pytest.mark.parametrize("found", [True, False])
def test_cold_primary_analysis_or_miss_is_terminal_not_memory_fallback(rig, found):
    rig.detector(104., found)
    rig.rt.beliefs.find = lambda _: pytest.fail("Slow analysis fell back to memory")
    rig.rt._other_cams = lambda: pytest.fail("Slow analysis tried another camera")
    with pytest.raises(SlowPerceptionError, match="Perception took 104.0s"):
        rig.rt._localize("object")
    assert rig.calls == [rig.primary]
    assert not rig.lazy.connected
    assert rig.primary.t == 100. and rig.primary.capture["t"] == -9999999.


@pytest.mark.parametrize("delay", [0., 4.99, 5.])
def test_fresh_primary_retains_exact_frame_without_comparing_remote_time(rig, delay):
    rig.detector(delay)
    frame, fix = rig.rt._localize("object")
    assert frame is rig.primary
    assert fix.detection.label == "object"
    assert not rig.lazy.connected


@pytest.mark.parametrize("bad_t", [94., float("nan"), float("inf"), 101.])
def test_invalid_initial_frame_cannot_reach_detector(rig, bad_t):
    rig.primary.t = bad_t
    with pytest.raises(SkillError, match="timestamp|old"):
        rig.rt._localize("object")
    assert not rig.calls and not rig.lazy.connected


def secondary(rig):
    frame = image(x=.3)
    cam = SimpleNamespace(stream=SimpleNamespace(get_frame=lambda: frame, name="side"),
                          depth=rig.rt.depth, extrinsics=Extrinsics(T=frame.T_base_cam))
    rig.rt._other_cams = lambda: [cam]
    return frame


@pytest.mark.parametrize("delay", [.5, 6.])
def test_secondary_success_or_expiry_uses_secondary_frame(rig, delay):
    side = secondary(rig)
    rig.rt.cfg._data["perception_loop"]["localize_frames"] = 1
    def detect(frame, **kwargs):
        rig.calls.append(frame)
        if frame is rig.primary:
            return []
        rig.clock.now += delay
        return [detection()]
    rig.rt.detector = SimpleNamespace(detect=detect)
    if delay > 5:
        with pytest.raises(SlowPerceptionError):
            rig.rt._localize("object")
    else:
        frame, fix = rig.rt._localize("object")
        assert frame is side and fix.position[0] > .25
    assert rig.calls == [rig.primary, side]
    assert not rig.lazy.connected


@pytest.mark.parametrize("slow_view", ["primary", "secondary"])
def test_vlm_slow_success_is_not_swallowed_as_a_miss(rig, slow_view):
    secondary(rig)
    rig.rt.cfg._data["grounder"] = {"base_url": "unused"}
    seen = []
    def ground(frame, query):
        seen.append(frame)
        if frame is rig.primary and slow_view == "secondary":
            return None
        rig.clock.now += 6.
        return detection()
    rig.rt._grounder = SimpleNamespace(ground=ground)
    with pytest.raises(SlowPerceptionError):
        rig.rt._vlm_ground_fix(rig.primary, "object")
    assert len(seen) == (1 if slow_view == "primary" else 2)
    assert not rig.lazy.connected


def test_vlm_secondary_fix_returns_actual_calibrated_frame_through_localize(rig):
    side = secondary(rig)
    side.T_base_cam = None  # Profile calibration must travel with this view.
    rig.detector(found=False)
    rig.rt.cfg._data["grounder"] = {"base_url": "unused"}
    rig.rt._grounder = SimpleNamespace(
        ground=lambda frame, query: None if frame is rig.primary else detection())
    frame, fix = rig.rt._localize("object")
    assert frame.rgb is side.rgb and frame.t == side.t
    assert frame.capture is side.capture
    assert frame.T_base_cam[0, 3] == .3 and fix.position[0] > .25
    assert not rig.lazy.connected


def test_vlm_slow_miss_does_not_try_next_view(rig):
    secondary(rig)
    rig.rt.cfg._data["grounder"] = {"base_url": "unused"}
    def ground(frame, query):
        rig.calls.append(frame)
        rig.clock.now += 6.
        return None
    rig.rt._grounder = SimpleNamespace(ground=ground)
    with pytest.raises(SlowPerceptionError):
        rig.rt._vlm_ground_fix(rig.primary, "object")
    assert rig.calls == [rig.primary]


def test_pick_failure_does_not_home_open_or_retry_without_observed_gate(rig, monkeypatch):
    monkeypatch.delenv("CASCADE_OBSERVED_FINGER_GATE", raising=False)
    rig.detector(104.)
    rig.rt._reconcile_held = lambda: None
    calls = []
    def grasp(*args, **kwargs):
        calls.append("grasp")
        return rig.rt._localize("object")
    rig.rt.skill_grasp_object = grasp
    rig.rt.skill_move_home = lambda: pytest.fail("Expired perception triggered home")
    rig.rt._reobserve = lambda: pytest.fail("Expired perception triggered retry")
    rig.rt.skill_place_at = lambda *a, **k: pytest.fail("Expired perception triggered place")
    result = rig.rt.skill_pick_and_place("object")
    assert result["ok"] is False and result["home_skipped"] is True
    assert result["grasp_attempts"] == 1 and calls == ["grasp"]
    assert not rig.lazy.connected


def test_multiple_short_detector_passes_still_expire_the_retained_image(rig):
    rig.detector(2., found=False)
    with pytest.raises(SlowPerceptionError):
        rig.rt._localize("object")
    assert len(rig.calls) == 3 and rig.clock.now == 106.


@pytest.mark.parametrize("use_belief", [False, True])
def test_other_view_misses_cannot_attach_expired_primary_to_fallback(rig, use_belief):
    side = secondary(rig)
    rig.rt.cfg._data["perception_loop"]["localize_frames"] = 1
    side.t = 102.  # The side image is acquired after the primary pass.
    if use_belief:
        rig.rt.beliefs.update("object", [.2, .1, .05], .9, t=100.)
        rig.rt._motion_t0 = 100.  # Keep the belief itself eligible.
    def detect(frame, **kwargs):
        rig.calls.append(frame)
        rig.clock.now += 2. if frame is rig.primary else 4.
        return []
    rig.rt.detector = SimpleNamespace(detect=detect)
    rig.rt._vlm_ground_fix = lambda *a: pytest.fail("Expired primary reached VLM")
    with pytest.raises(SlowPerceptionError):
        rig.rt._localize("object")
    assert rig.calls == [rig.primary, side] and rig.clock.now == 106.
    assert not rig.lazy.connected


def test_fresh_frame_keeps_existing_motion_epoch_belief_policy(rig):
    rig.detector(found=False)
    belief = rig.rt.beliefs.update("object", [.2, .1, .05], .9, t=20.)
    rig.rt._motion_t0 = 20.5
    frame, fix = rig.rt._localize("object")
    assert frame is rig.primary
    np.testing.assert_allclose(fix.position, belief.position, rtol=0., atol=1e-15)
    assert belief.last_seen_t == 20. and frame.t == 100.
    assert not rig.lazy.connected


def test_vlm_accumulated_view_time_is_terminal(rig):
    side = secondary(rig)
    rig.rt.cfg._data["grounder"] = {"base_url": "unused"}
    def ground(frame, query):
        rig.calls.append(frame)
        rig.clock.now += 4. if frame is rig.primary else 2.
        return None if frame is rig.primary else detection()
    rig.rt._grounder = SimpleNamespace(ground=ground)
    with pytest.raises(SlowPerceptionError):
        rig.rt._vlm_ground_fix(rig.primary, "object")
    assert rig.calls == [rig.primary, side] and rig.clock.now == 106.


def test_vlm_cold_constructor_cannot_bypass_age_check(rig, monkeypatch):
    from cascade.perception import vlm_ground
    rig.rt.cfg._data["grounder"] = {"base_url": "unused"}
    def cold_init(**kwargs):
        rig.clock.now += 6.
        return SimpleNamespace(ground=lambda *a: pytest.fail("Expired image analyzed"))
    monkeypatch.setattr(vlm_ground, "VLMGrounder", cold_init)
    with pytest.raises(SlowPerceptionError):
        rig.rt._vlm_ground_fix(rig.primary, "object")


def test_vlm_depth_processing_is_included_in_age_budget(rig, monkeypatch):
    rig.rt.cfg._data["grounder"] = {"base_url": "unused"}
    rig.rt._grounder = SimpleNamespace(ground=lambda *a: detection())
    real_bbox = module.oriented_bbox
    def slow_bbox(points):
        rig.clock.now += 6.
        return real_bbox(points)
    monkeypatch.setattr(module, "oriented_bbox", slow_bbox)
    with pytest.raises(SlowPerceptionError):
        rig.rt._vlm_ground_fix(rig.primary, "object")
