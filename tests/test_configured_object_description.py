"""Explicit booth names retain identity while using current measured color/depth."""
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.agent.trace import TraceLogger
from cascade.config import Cfg
from cascade.memory import BeliefStore, EpisodicMemory
from cascade.perception.depth_provider import DepthProvider
from cascade.perception.detector import MockDetector
from cascade.perception.grounding import Extrinsics
from cascade.skills.runtime import SkillRuntime
from cascade.types import Detection, Frame, SkillError


def runtime_for(colors, cfg, tmp_path):
    frame = Frame(
        rgb=np.zeros((32, 40 * len(colors), 3), dtype=np.uint8),
        depth_m=np.full((32, 40 * len(colors)), .5, dtype=np.float32),
        K=np.array([[100., 0., 20.], [0., 100., 16.], [0., 0., 1.]]),
        depth_source="sensor",
    )
    dets = []
    for i, color in enumerate(colors):
        mask = np.zeros(frame.rgb.shape[:2], dtype=bool)
        mask[8:24, 40 * i + 8:40 * i + 24] = True
        frame.rgb[mask] = color
        dets.append(Detection("lime" if i == 0 else "toy", .9 - .1 * i,
                              np.array([40 * i + 8, 8, 40 * i + 24, 24]), mask=mask))
    cfg._data.update(object_descriptions={"lemon": "yellow object"},
                     perception_loop={"localize_frames": 1}, grounder=None)
    runtime = SkillRuntime(
        camera=SimpleNamespace(get_frame=lambda: frame), depth_provider=DepthProvider(Cfg({})),
        detector=MockDetector(detections=dets), extrinsics=Extrinsics(), kin=None,
        safe_arm=SimpleNamespace(harness=SimpleNamespace(heartbeat=lambda: None)),
        memory=EpisodicMemory(), beliefs=BeliefStore(), trace=TraceLogger(tmp_path / "run"), cfg=cfg,
    )
    return runtime, frame


def test_named_lemon_uses_current_yellow_pixels_without_relabeling(demo_cfg, tmp_path):
    runtime, _ = runtime_for([(0, 255, 255), (0, 255, 0)], demo_cfg, tmp_path)
    result = runtime.skill_localize_object("lemon")
    assert result["label"] == "lemon"
    assert result["detected_as"] == "lime"
    assert result["color"] == "yellow"
    assert result["position"] == pytest.approx([-.02, -.002, .5], abs=.005)
    assert result["target_resolution"] == {
        "requested": "lemon", "visual_query": "yellow object", "detected_as": "lime",
        "measured_color": "yellow", "unique_reachable_match": True,
    }
    assert runtime.beliefs.find("lime") is not None
    assert runtime.beliefs.find("lemon") is None


def test_two_yellow_objects_refuse_instead_of_ranking(demo_cfg, tmp_path):
    runtime, _ = runtime_for([(0, 255, 255)] * 2, demo_cfg, tmp_path)
    with pytest.raises(SkillError, match="2 reachable objects"):
        runtime.skill_localize_object("lemon")


def test_outside_workspace_yellow_does_not_hide_reachable_one(demo_cfg, tmp_path):
    runtime, _ = runtime_for([(0, 255, 255)] * 2, demo_cfg, tmp_path)
    runtime._localization_workspace_bounds = lambda: (np.array([.1, -1., 0.]), np.array([1., 1., 1.]))
    result = runtime.skill_localize_object("lemon")
    assert result["detected_as"] == "toy"
    assert result["position"][0] > .1


@pytest.mark.parametrize("missing", ["wrong_color", "no_depth", "no_detection"])
def test_missing_current_evidence_cannot_use_memory_or_vlm(demo_cfg, tmp_path, missing):
    runtime, frame = runtime_for([(0, 255, 255)], demo_cfg, tmp_path)
    runtime.beliefs.update("lemon", np.array([.2, 0., .04]), .99, color="yellow")
    runtime.beliefs.update("lime", np.array([.2, 0., .04]), .99, color="yellow")
    def forbidden(*_):
        raise AssertionError("Configured descriptions cannot use an unfiltered VLM fallback")
    runtime._vlm_ground_fix = forbidden
    if missing == "wrong_color":
        frame.rgb[:] = (0, 255, 0)
    elif missing == "no_depth":
        frame.depth_m[:] = 0
    else:
        runtime.detector = MockDetector(detections=[])
    with pytest.raises(SkillError):
        runtime.skill_localize_object("lemon")


def test_description_does_not_override_other_queries_or_references(demo_cfg, tmp_path):
    runtime, _ = runtime_for([(0, 255, 255)], demo_cfg, tmp_path)
    for query in ("green cube", "the second lemon", "lemon not the yellow one"):
        assert runtime._visual_query(query) == query


@pytest.mark.parametrize("description", ["fruit", "", 123, {"position": [1, 2, 3]}])
def test_invalid_description_cannot_supply_a_grasp_target(demo_cfg, tmp_path, description):
    runtime, _ = runtime_for([(0, 255, 255)], demo_cfg, tmp_path)
    runtime.cfg._data["object_descriptions"]["lemon"] = description
    with pytest.raises(SkillError, match="explicit measured color"):
        runtime.skill_localize_object("lemon")
