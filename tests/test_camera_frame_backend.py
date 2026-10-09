"""`grasp.backend: camera_frame` -- the WRC/rebot_grasp camera-frame mask
planner as an EXPLICIT, opt-in grasp backend.

The rules this pins:

  - opt-in only (`grasp.backend: camera_frame` or
    `CASCADE_GRASP_BACKEND=camera_frame`); the shipped default stays
    GraspGen-X and no shipped profile selects it (the golden profile pin in
    tests/test_hug_runtime.py keeps every non-HUG profile on graspgenx/obb,
    and the Spark presenter on REQUIRED GraspGen-X);
  - it plans from the RGB-D frame the object was localized in and that
    frame's own camera->base transform, in the arm's `tool_axis_order`;
  - its candidates are ordinary Grasps: memory re-ranking, the selector's
    jaw-width/IK checks and the harness vetting all still apply;
  - nothing is substituted silently: a `required` profile fails visibly when
    the mask planner has nothing, an optional one reports its OBB fallback
    (`grasp_planner_used`, a memory note);
  - a frame/transform that does not reproduce the localized object is
    refused, never aimed at.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.config import load_demo_config
from cascade.types import Grasp, SkillError


@pytest.fixture(autouse=True)
def _isolated_ports(monkeypatch):
    monkeypatch.delenv("CASCADE_GRASP_BACKEND", raising=False)
    # nothing listens on these; a runtime built here never dials :5556
    monkeypatch.setenv("CASCADE_GRASPGENX_PORT", "43741")
    monkeypatch.setenv("CASCADE_OCCUPANCY_PORT", "43742")
    monkeypatch.setenv("CASCADE_BRIDGE_PORT", "43743")


def _scene():
    from test_hug_backend import _scene as hug_scene

    frame, fix, T = hug_scene()            # 4 cm red cube under a top-down camera
    frame.T_base_cam = T
    return frame, fix, T


def _runtime(*, required=False, include_obb=True, notes=None, memory=None, **ccfg):
    cfg = load_demo_config()
    cfg._data["grasp"]["backend"] = "camera_frame"
    block = cfg._data["grasp"]["camera_frame"]
    block.update(required=required, include_obb=include_obb, **ccfg)
    return SimpleNamespace(
        cfg=cfg, _max_width=0.09, _tool_axis_order="down_open",
        extrinsics=SimpleNamespace(cam_to_base=lambda: pytest.fail("frame carries T_base_cam")),
        memory=SimpleNamespace(add=lambda kind, text: (notes if notes is not None else []).append(text)),
        grasp_memory=memory or SimpleNamespace(prior=lambda *a: None))


def _plan(rt, fix, **kw):
    from cascade.skills.runtime import SkillRuntime

    return SkillRuntime._plan_grasps(rt, fix, **kw)


# ── config ─────────────────────────────────────────────────────────────────


def test_launcher_env_can_select_camera_frame_and_typos_still_fail(monkeypatch):
    monkeypatch.setenv("CASCADE_GRASP_BACKEND", "camera_frame")
    cfg = load_demo_config(arms=["rebot_rs", "mock"])
    assert cfg.grasp.backend == "camera_frame"
    assert all(a["resolved"]["grasp"]["backend"] == "camera_frame" for a in cfg.arms)
    monkeypatch.setenv("CASCADE_GRASP_BACKEND", "camera-frame")
    with pytest.raises(ValueError, match="CASCADE_GRASP_BACKEND"):
        load_demo_config()


def test_defaults_live_in_demo_yaml_and_the_default_backend_is_unchanged():
    g = load_demo_config().grasp
    assert g.backend == "graspgenx"
    c = g.camera_frame
    assert c.required is False and c.include_obb is True
    assert c.insertion_depth_m == pytest.approx(0.015)
    assert c.depth_quantile == pytest.approx(0.5)
    assert c.finger_drop_m == pytest.approx(0.030)
    assert c.max_fix_offset_m == pytest.approx(0.03)


# ── planning ───────────────────────────────────────────────────────────────


def test_the_mask_grasp_leads_and_obb_alternates_follow():
    frame, fix, _ = _scene()
    notes = []
    rt = _runtime(notes=notes)
    out = _plan(rt, fix, _frame=frame)
    first = out[0]
    np.testing.assert_allclose(first.approach, [0.0, 0.0, -1.0], atol=1e-6)
    top_z = float(fix.points[:, 2].max())
    assert first.position[2] == pytest.approx(top_z - 0.015, abs=2e-3)
    np.testing.assert_allclose(first.position[:2], fix.points[:, :2].mean(axis=0), atol=3e-3)
    assert first.width_m == pytest.approx(0.04 + rt.cfg.grasp.get("width_pad_m", 0.015), abs=4e-3)
    assert len(out) > 1                                   # OBB alternates behind it
    assert rt.grasp_planner_used == "camera_frame (mask + depth)"
    assert any(n.startswith("camera_frame:") for n in notes)


def test_without_obb_alternates_only_the_mask_grasp_is_offered():
    frame, fix, _ = _scene()
    out = _plan(_runtime(include_obb=False), fix, _frame=frame)
    assert len(out) == 1


def test_the_frame_is_required_and_never_guessed():
    _, fix, _ = _scene()
    with pytest.raises(SkillError, match="frame"):
        _plan(_runtime(), fix)


def test_the_frames_own_transform_wins_and_extrinsics_are_the_fallback():
    frame, fix, T = _scene()
    frame.T_base_cam = None
    rt = _runtime()
    rt.extrinsics = SimpleNamespace(cam_to_base=lambda: T)
    out = _plan(rt, fix, _frame=frame)
    assert rt.grasp_planner_used == "camera_frame (mask + depth)"
    np.testing.assert_allclose(out[0].approach, [0.0, 0.0, -1.0], atol=1e-6)


def test_the_arms_tool_axis_order_is_used():
    frame, fix, _ = _scene()
    rt = _runtime(include_obb=False)
    rt._tool_axis_order = "open_down"
    (g,) = _plan(rt, fix, _frame=frame)
    np.testing.assert_allclose(g.rotation[:, 2], g.approach, atol=1e-9)


@pytest.mark.parametrize("break_it, reason", [
    (lambda f, fx: setattr(fx.detection, "mask", None), "no segmentation mask"),
    (lambda f, fx: setattr(f, "depth_m", None), "no depth"),
    (lambda f, fx: setattr(f, "depth_source", "plane"), "plane"),
    (lambda f, fx: setattr(fx.detection, "mask", np.ones((10, 10), bool)), "mask"),
])
def test_optional_falls_back_to_a_labelled_obb(break_it, reason):
    frame, fix, _ = _scene()
    break_it(frame, fix)
    notes = []
    rt = _runtime(notes=notes)
    out = _plan(rt, fix, _frame=frame)
    assert out and all(np.allclose(g.approach, [0.0, 0.0, -1.0]) for g in out)
    assert rt.grasp_planner_used.startswith("obb (camera_frame: ")
    assert reason in rt.grasp_planner_used
    assert any("OBB fallback" in n and reason in n for n in notes)


def test_required_fails_visibly_and_never_substitutes_obb():
    frame, fix, _ = _scene()
    fix.detection.mask = None
    rt = _runtime(required=True)
    with pytest.raises(SkillError, match="camera-frame grasp planner required.*no segmentation mask"):
        _plan(rt, fix, _frame=frame)
    assert rt.grasp_planner_used == "camera_frame (unavailable)"


def test_a_transform_that_does_not_reproduce_the_object_is_refused():
    """Wrong extrinsics / a frame from elsewhere: the planned surface point
    lands off the localized object, so the grasp is dropped, not aimed."""
    frame, fix, T = _scene()
    moved = T.copy()
    moved[0, 3] += 0.10                                   # 10 cm off
    frame.T_base_cam = moved
    rt = _runtime(required=True)
    with pytest.raises(SkillError, match="off the localized object"):
        _plan(rt, fix, _frame=frame)


def test_memory_reranking_still_owns_the_final_order():
    frame, fix, _ = _scene()
    seen = {}

    def rerank(grasps, label, fx):
        seen["in"] = list(grasps)
        return list(reversed(grasps))

    memory = SimpleNamespace(
        prior=lambda *a: {"profile": "p", "wins": 1, "losses": 0, "success_rate": 1.0,
                          "nudges": {}, "top_fail": None},
        rerank=rerank)
    rt = _runtime(memory=memory)
    out = _plan(rt, fix, _frame=frame)
    assert seen["in"][0].approach is not None and out == list(reversed(seen["in"]))


# ── call sites hand it the frame; other backends unchanged ─────────────────


@pytest.mark.parametrize("backend", ["obb", "graspgenx", "hug", "camera_frame"])
def test_preview_hands_the_frame_only_to_frame_backends(backend):
    from cascade.skills.runtime import SkillRuntime

    frame, fix, _ = _scene()
    cfg = load_demo_config()
    cfg._data["grasp"]["backend"] = backend
    calls = []
    grasp = Grasp(fix.position, np.eye(3), 0.05, np.array([0.0, 0.0, -1.0]), quality=0.5)
    fake = SimpleNamespace(
        cfg=cfg, _localize=lambda label, spatial_hint=None: (frame, fix),
        _plan_grasps=lambda f, **kw: calls.append(kw) or [grasp],
        grasp_memory=SimpleNamespace(prior=lambda *a: None),
        memory=SimpleNamespace(add=lambda *a: None))
    assert SkillRuntime.skill_preview_grasp(fake, "red cube")["ok"] is True
    wants_frame = backend in ("hug", "camera_frame")
    assert calls == [{"label": "red cube", **({"_frame": frame} if wants_frame else {})}]


def test_capability_matrix_says_grasps_are_analytic_camera_frame():
    from cascade.apps.capabilities import _grasp_capability

    rt = _runtime()
    cell = _grasp_capability(rt, {"grasp_planner": "camera_frame (mask + depth)"})
    assert cell["available"] is False
    assert "camera-frame" in cell["why"] and "GraspGen-X" not in cell["why"]


# ── end to end on the mock stack: selection + harness still decide ─────────


def test_grasp_object_executes_a_vetted_camera_frame_grasp(tmp_path, monkeypatch):
    pytest.importorskip("pinocchio")
    import cascade.skills.runtime as rtmod
    from cascade.apps.demo import build_runtime, shutdown_runtime

    offered, chosen = [], []
    real_select = rtmod.select_grasp

    def spy_select(grasps, *a, **kw):
        offered.append(list(grasps))
        out = real_select(grasps, *a, **kw)
        chosen.append(out[0])
        return out

    monkeypatch.setattr(rtmod, "select_grasp", spy_select)
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    cfg._data["grasp"]["backend"] = "camera_frame"
    runtime, arm = build_runtime(cfg, tmp_path / "run")
    try:
        assert runtime.backends()["grasp_planner"] == "camera_frame"
        arm.object_stop_frac = 0.5
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and runtime.beliefs.find("red cube") is None:
            time.sleep(0.05)
        with runtime.watcher.paused():
            result = runtime.execute("grasp_object", {"label": "red cube"})
        assert result["ok"] is True, result
        assert runtime.grasp_planner_used == "camera_frame (mask + depth)"
        # the camera-frame grasp led the offer and passed IK + harness vetting
        lead = offered[0][0]
        np.testing.assert_allclose(lead.approach, [0.0, 0.0, -1.0], atol=1e-6)
        assert np.allclose(chosen[0].position, lead.position)
    finally:
        shutdown_runtime(runtime, arm)
