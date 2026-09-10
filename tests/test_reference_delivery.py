"""Reference safety through real RGB-D grounding and the skill dispatcher.

Only the detector/camera/hardware boundaries are doubled. Reference parsing,
mask backprojection, ObjectFix construction and runtime fallback stay real.
No model service, simulator or physical arm is started.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.agent.trace import TraceLogger
from cascade.config import Cfg
from cascade.memory import BeliefStore, EpisodicMemory
from cascade.perception.depth_provider import DepthProvider
from cascade.perception.detector import MockDetector
from cascade.perception.grounding import Extrinsics, localize_object
from cascade.skills.runtime import SkillRuntime
from cascade.types import Detection, Frame, SkillError


def _scene(labels):
    """Small deterministic RGB-D patches; no mocked grounding results."""
    height, width = 16 * len(labels) + 8, 32
    frame = Frame(
        rgb=np.full((height, width, 3), (0, 0, 255), dtype=np.uint8),
        depth_m=np.full((height, width), 0.5, dtype=np.float32),
        K=np.array([[80.0, 0.0, width / 2],
                    [0.0, 80.0, height / 2], [0.0, 0.0, 1.0]]),
        depth_source="sensor",
    )
    detections = []
    for i, label in enumerate(labels):
        y0 = 4 + 16 * i
        mask = np.zeros((height, width), dtype=bool)
        mask[y0:y0 + 8, 8:18] = True
        detections.append(Detection(
            label=label, conf=0.9 - 0.1 * i,
            bbox=np.array([8, y0, 18, y0 + 8], dtype=np.float32), mask=mask,
        ))
    return frame, MockDetector(detections=detections), Extrinsics()


def test_localize_refuses_exclusion_that_removes_every_candidate():
    frame, detector, extrinsics = _scene(["red cup", "red cup"])

    with pytest.raises(SkillError, match="exclusion.*red.*no candidates"):
        localize_object(frame, "the cup, not the red one", detector, extrinsics)


def test_localize_refuses_ordinal_beyond_available_candidates():
    frame, detector, extrinsics = _scene(["cup", "cup"])

    with pytest.raises(SkillError, match="ordinal 5.*out of range.*2 candidate"):
        localize_object(frame, "the fifth cup from the left", detector, extrinsics)


@pytest.mark.parametrize("query, reason", [
    ("the cup, not the cup", "exclusion.*cup.*no candidates"),
    ("the second cup from the left", "ordinal 2.*out of range.*1 candidate"),
])
@pytest.mark.parametrize("depthless_extra", [False, True], ids=["singleton", "depth-filtered"])
def test_localize_enforces_reference_on_a_single_candidate(query, reason, depthless_extra):
    labels = ["cup", "cup"] if depthless_extra else ["cup"]
    frame, detector, extrinsics = _scene(labels)
    if depthless_extra:
        # Two detections but only one usable 3D fix must still obey the phrase.
        assert frame.depth_m is not None
        frame.depth_m[detector.detect(frame)[1].mask] = 0.0

    with pytest.raises(SkillError, match=reason):
        localize_object(frame, query, detector, extrinsics)


def _runtime(frame, detector, extrinsics, cfg, tmp_path):
    cfg._data["perception_loop"] = {"localize_frames": 2, "belief_fallback_age_s": 3.0}
    cfg._data["grounder"] = None  # no model service in this CPU contract test
    return SkillRuntime(
        camera=SimpleNamespace(get_frame=lambda: frame),
        depth_provider=DepthProvider(Cfg({})), detector=detector,
        extrinsics=extrinsics, kin=None,
        safe_arm=SimpleNamespace(harness=SimpleNamespace(heartbeat=lambda: None)),
        memory=EpisodicMemory(), beliefs=BeliefStore(),
        trace=TraceLogger(tmp_path / "run"), cfg=cfg,
    )


@pytest.mark.parametrize("query, reason", [
    ("the cup, not the cup", "exclusion 'cup' leaves no candidates"),
    ("the fifth cup from the left", "ordinal 5 is out of range"),
], ids=["exclude-all", "ordinal-out-of-range"])
@pytest.mark.parametrize("count", [1, 2], ids=["singleton", "multiple"])
@pytest.mark.parametrize("warm_belief", [False, True], ids=["no-belief", "fresh-belief"])
def test_dispatcher_preserves_reference_refusal(
    demo_cfg, tmp_path, query, reason, count, warm_belief,
):
    frame, detector, extrinsics = _scene(["cup"] * count)
    runtime = _runtime(frame, detector, extrinsics, demo_cfg, tmp_path)
    if warm_belief:
        belief = runtime.beliefs.update(
            "cup", np.array([0.2, 0.0, 0.04]), 0.9,
            extent=np.array([0.04, 0.04, 0.04]), top_z=0.06,
        )
        # Prove the real query resolver can take this otherwise-eligible fallback.
        assert runtime.beliefs.find(query) is belief

    result = runtime.execute("localize_object", {"label": query})

    assert result["ok"] is False, (result, runtime.memory.digest())
    assert "ReferenceResolutionError" in result["error"], result
    assert reason in result["error"], result
    assert "Clarify the reference" in result["error"], result
    assert "position" not in result
    assert "using the world-model fix" not in runtime.memory.digest()
    row = json.loads((tmp_path / "run" / "trace.jsonl").read_text())
    assert row["skill"] == "localize_object"
    assert row["result"] == result


@pytest.mark.parametrize("query", ["the second cup", "the cup, not the cup"])
def test_memory_only_detection_miss_cannot_resolve_a_constrained_reference(
    demo_cfg, tmp_path, monkeypatch, query,
):
    frame, detector, extrinsics = _scene(["cup"])
    runtime = _runtime(frame, detector, extrinsics, demo_cfg, tmp_path)
    runtime.beliefs.update("cup", np.array([0.2, 0.0, 0.04]), 0.9,
                           extent=np.array([0.04, 0.04, 0.04]), top_z=0.06)
    monkeypatch.setattr(detector, "detect", lambda *args, **kwargs: [])

    # A plain noun may use a fresh remembered fix when detection flickers.
    plain = runtime.execute("localize_object", {"label": "cup"})
    assert plain["ok"] is True, plain
    # That single remembered fix cannot establish ordinal/exclusion identity.
    result = runtime.execute("localize_object", {"label": query})
    assert result["ok"] is False, result
    assert "ReferenceResolutionError" in result["error"]
    assert "Clarify the reference" in result["error"]
    assert "position" not in result


@pytest.mark.parametrize("query", [
    "cup", "the first cup", "the last cup", "the biggest cup", "cup, not red",
])
def test_valid_singleton_references_still_localize(query):
    frame, detector, extrinsics = _scene(["cup"])
    expected = detector.detect(frame)[0]

    fix = localize_object(frame, query, detector, extrinsics)

    assert fix.detection is expected
    assert fix.points.shape[0] >= 10
    assert np.all(np.isfinite(fix.position))


def test_valid_ordinal_keeps_its_axis_order():
    frame, detector, extrinsics = _scene(["cup", "cup", "cup"])
    expected = detector.detect(frame)[1]  # middle y: second from +y (left)

    fix = localize_object(frame, "the second cup from the left", detector, extrinsics)

    assert fix.detection is expected


def test_exclusion_can_leave_one_valid_candidate():
    frame, detector, extrinsics = _scene(["red cup", "blue cup"])
    expected = detector.detect(frame)[1]

    fix = localize_object(frame, "the cup, not the red one", detector, extrinsics)

    assert fix.detection is expected


def test_ordinal_counts_candidates_after_exclusion():
    frame, detector, extrinsics = _scene(["red cup", "blue cup"])

    with pytest.raises(SkillError, match="ordinal 2.*out of range.*1 candidate"):
        localize_object(frame, "the second cup, not red", detector, extrinsics)


def test_dispatcher_keeps_plain_query_belief_fallback(demo_cfg, tmp_path):
    """An ordinary detection miss may still use memory; only refusals are terminal."""
    frame, _, extrinsics = _scene(["cup"])
    runtime = _runtime(frame, MockDetector(detections=[]), extrinsics, demo_cfg, tmp_path)
    position = np.array([0.2, 0.0, 0.04])
    runtime.beliefs.update(
        "cup", position, 0.9, extent=np.array([0.04, 0.04, 0.04]), top_z=0.06,
    )

    result = runtime.execute("localize_object", {"label": "cup"})

    assert result["ok"] is True, result
    assert result["position"] == pytest.approx(position)
    assert "using the world-model fix" in runtime.memory.digest()


def test_dispatcher_preserves_secondary_camera_reference_refusal(
    demo_cfg, tmp_path, monkeypatch,
):
    frame, detector, extrinsics = _scene(["cup"])
    primary, _, _ = _scene(["cup"])
    runtime = _runtime(primary, detector, extrinsics, demo_cfg, tmp_path)
    detect = detector.detect
    monkeypatch.setattr(detector, "detect", lambda f, classes=None: (
        [] if f is primary else detect(f, classes=classes)
    ))
    monkeypatch.setattr(runtime, "watcher", SimpleNamespace(
        _cams=[SimpleNamespace(), SimpleNamespace(
            stream=SimpleNamespace(get_frame=lambda: frame), depth=runtime.depth,
            extrinsics=extrinsics,
        )],
        ignore_label=lambda label: None,
    ))

    result = runtime.execute("localize_object", {"label": "the second cup"})

    assert result["ok"] is False, result
    assert "ReferenceResolutionError" in result["error"], result
    assert "ordinal 2 is out of range" in result["error"], result


def test_dispatcher_does_not_overwrite_reference_refusal_with_later_miss(
    demo_cfg, tmp_path, monkeypatch,
):
    frame, detector, extrinsics = _scene(["cup"])
    runtime = _runtime(frame, detector, extrinsics, demo_cfg, tmp_path)
    # Real grounding refuses the first observation; the next detection flickers.
    rounds = iter([detector.detect(frame), []])
    monkeypatch.setattr(detector, "detect", lambda *a, **k: next(rounds, []))

    result = runtime.execute("localize_object", {"label": "the second cup"})

    assert result["ok"] is False, result
    assert "ReferenceResolutionError" in result["error"], result
    assert "ordinal 2 is out of range" in result["error"], result
