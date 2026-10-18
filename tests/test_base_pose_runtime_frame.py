"""base_pose Option 2: the skill runtime works in the running arm's base frame
(spec docs/superpowers/specs/2026-10-17-base-pose-runtime-frame-design.md, §3.3-3.4, §6.1).

Perception section (Task 3): every frame localization analyses is in the
running arm's base frame (`_frame_in_arm`), belief positions used as targets
go through `_to_arm`, and every report goes back out in the TABLE frame
(localize_object, point_at, preview_grasp, get_observation's `tcp_xyz`, the
configured zones). The equivariance oracle (`tests/base_pose_oracle.py`) is
the core check: an arm at base_pose T with the scene moved by T behaves
exactly like the same arm without a base_pose.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from base_pose_oracle import (  # noqa: F401  (fixtures by name)
    RAISED,
    SHIPPED_RIGHT,
    YAWED,
    assert_equivariant,
    lift,
    make_pair_rigs,
    make_runtime,
    moved_cfg,
    run_pair,
    transform,
)

#: Task 3's oracle rows: one call per perception skill.
ORACLE_T3 = {"localize_object": {"label": "red cube"}, "point_at": {"label": "red cube"},
             "preview_grasp": {"label": "red cube"}, "get_observation": {}, "list_objects": {},
             "describe_scene": {}, "count_objects": {}}

#: mock_small's red cube, with the camera shifted 10 cm in -y in the table
#: frame: W (0.198, -0.10), r = 0.231 from so101_right's base (spec §6.2).
WORLD_OBJECT_XY = (0.198, -0.10)


def _world_object_rig():
    return moved_cfg("so101_right", SHIPPED_RIGHT, move_cameras=False, shift_xy=(0.0, -0.10))


# ── 1. the oracle ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("pose", [SHIPPED_RIGHT, YAWED], ids=["so101_right", "yawed"])
@pytest.mark.parametrize("skill", sorted(ORACLE_T3))
def test_perception_skills_are_equivariant(make_pair_rigs, pose, skill):  # noqa: F811
    rt_a, rt_b, T = make_pair_rigs(pose)
    run = run_pair(rt_a, rt_b, T, skill, ORACLE_T3[skill])
    if skill in ("localize_object", "point_at", "preview_grasp"):
        assert run["result_a"].get("ok", True) is True, run["result_a"]     # rig A really ran it
    if skill == "point_at":
        assert run["cmds_a"], "rig A pointed: the arm moved"
    assert_equivariant(run, T)


# ── 2. the world point ────────────────────────────────────────────────────


def _belief_xy(rt, label):
    belief = rt.beliefs.find(label)
    assert belief is not None, [b.label for b in rt.beliefs.all()]
    return np.asarray(belief.position, dtype=float)[:2]


def test_localize_reports_the_world_position_on_so101_right(make_runtime):  # noqa: F811
    rt = make_runtime(_world_object_rig())
    result = rt.execute("localize_object", {"label": "red cube"})
    assert result.get("ok", True) is True, result
    assert np.hypot(*(np.asarray(result["position"][:2]) - WORLD_OBJECT_XY)) <= 0.005, result["position"]
    assert np.hypot(*(_belief_xy(rt, "red cube") - WORLD_OBJECT_XY)) <= 0.005


def test_point_at_hovers_over_the_world_object(make_runtime, monkeypatch):  # noqa: F811
    rt = make_runtime(_world_object_rig())
    targets = []
    original = rt._move_joints

    def move_joints(q, *a, **k):
        targets.append(np.asarray(q, dtype=float).copy())
        return original(q, *a, **k)

    monkeypatch.setattr(rt, "_move_joints", move_joints)
    result = rt.execute("point_at", {"label": "red cube"})          # main: refused by #352's guard
    assert result["ok"] is True, result
    assert targets, "point_at commanded its pointing pose"
    hover = lift(transform(SHIPPED_RIGHT), rt.kin.fk(targets[0])[:3, 3])
    assert np.hypot(*(hover[:2] - WORLD_OBJECT_XY)) <= 0.005, hover
    assert np.hypot(*(np.asarray(result["position"][:2]) - WORLD_OBJECT_XY)) <= 0.005, result


def test_the_belief_fallback_aims_in_the_arm_frame(make_pair_rigs, monkeypatch):  # noqa: F811
    """The detector misses; a fresh belief (table frame) becomes the target,
    converted into the arm frame (`_localize`'s world-model fix)."""
    rt_a, rt_b, T = make_pair_rigs(YAWED)
    misses = []
    for rt in (rt_a, rt_b):
        # point_at's hold and the IK search take seconds on rig A; keep rig B's
        # belief (seeded with A's) inside the fallback's freshness window
        rt.cfg._data.setdefault("perception_loop", {})["belief_fallback_age_s"] = 60.0
        monkeypatch.setattr(rt.detector, "detect", lambda *a, **k: (misses.append(1), [])[1])
    run = run_pair(rt_a, rt_b, T, "point_at", {"label": "red cube"})
    assert misses, "the detector was asked and missed"
    assert run["result_a"]["ok"] is True, run["result_a"]
    assert any("world-model fix" in e.text for e in rt_b.memory.events(("note",))), "rig B used the belief"
    assert_equivariant(run, T)


@pytest.mark.parametrize("pose", [YAWED, RAISED], ids=["yawed", "raised"])
@pytest.mark.parametrize("memory", ["cloud", "box"])
def test_the_remembered_fix_is_built_in_the_arm_frame(make_pair_rigs, monkeypatch, memory, pose):  # noqa: F811
    """A belief becomes a fix from its remembered cloud, or (without one) an
    axis-aligned box from its extent and `top_z`. Either is built in the
    RUNNING arm's frame (the belief is converted first), so a moved arm plans
    on exactly the fix of the arm without a base_pose."""
    rt_a, rt_b, T = make_pair_rigs(pose)
    fixes = {}
    for tag, rt in (("a", rt_a), ("b", rt_b)):
        rt.cfg._data.setdefault("perception_loop", {})["belief_fallback_age_s"] = 60.0
        if memory == "box":
            rt.beliefs.find("red cube").points = None
        monkeypatch.setattr(rt.detector, "detect", lambda *a, **k: [])
        real = rt._plan_grasps

        def plan(fix, *a, _real=real, _tag=tag, **k):
            fixes[_tag] = fix
            return _real(fix, *a, **k)

        monkeypatch.setattr(rt, "_plan_grasps", plan)
    run = run_pair(rt_a, rt_b, T, "preview_grasp", {"label": "red cube"})
    assert run["result_a"]["ok"] is True, run["result_a"]
    if memory == "box":
        np.testing.assert_allclose(fixes["a"].axes, np.eye(3))
    # 1e-6: the remembered cloud is float32 (a cube's near-equal PCA spreads
    # turn its 1e-8 rounding into ~1e-7 on the axes); a frame error moves the
    # points by centimetres and turns the axes by the 0.4 rad yaw
    np.testing.assert_allclose(fixes["b"].axes, fixes["a"].axes, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(fixes["b"].position, fixes["a"].position, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(fixes["b"].points, fixes["a"].points, rtol=0.0, atol=1e-6)
    assert_equivariant(run, T)


def test_the_belief_fallback_checks_the_workspace_in_the_arm_frame(make_runtime, monkeypatch):  # noqa: F811
    """The fallback admits a belief inside the running arm's (base-frame)
    workspace: W (0.198, -0.40) is B (0.198, -0.18) for so101_right, inside
    its y >= -0.36, although -0.40 read as a base-frame y is outside."""
    rt = make_runtime(moved_cfg("so101_right", SHIPPED_RIGHT, move_cameras=False, shift_xy=(0.0, -0.40)))
    rt.cfg._data.setdefault("perception_loop", {})["belief_fallback_age_s"] = 60.0
    with rt.watcher.paused():
        rt._reobserve(1)
        assert np.allclose(_belief_xy(rt, "red cube"), (0.198, -0.40), atol=0.005)
        monkeypatch.setattr(rt.detector, "detect", lambda *a, **k: [])
        result = rt.execute("point_at", {"label": "red cube"})
    assert result["ok"] is True, result
    assert any("world-model fix" in e.text for e in rt.memory.events(("note",)))
    assert np.hypot(*(np.asarray(result["position"][:2]) - (0.198, -0.40))) <= 0.005, result


def _side_camera_only(cfg):
    """The primary camera sees an empty table; a second camera ("side", the
    same pose) sees the red cube, so `_localize` finds it in its other-camera
    pass."""
    import copy

    side = copy.deepcopy(cfg._data["cameras"][0])
    side["name"] = "side"
    for cam in (cfg._data["camera"], cfg._data["cameras"][0]):
        cam["box_px"] = [700, 500, 710, 510]          # outside the 640x480 image: no prop
    cfg._data["cameras"].append(side)


def test_a_side_camera_localizes_in_the_arm_frame(make_pair_rigs):  # noqa: F811
    rt_a, rt_b, T = make_pair_rigs(YAWED, edit=_side_camera_only)
    assert len(rt_b._other_cams()) == 1
    run = run_pair(rt_a, rt_b, T, "point_at", {"label": "red cube"})
    assert run["result_a"]["ok"] is True, run["result_a"]
    assert any("found through" in e.text for e in rt_b.memory.events(("note",))), "the side camera found it"
    assert_equivariant(run, T)


class _BoxGrounder:
    """A VLM grounder double that names mock_small's box (the detector missed)."""

    def ground(self, frame, query):
        from cascade.types import Detection

        return Detection(label=query, conf=0.9, bbox=np.array([301, 308, 339, 346], dtype=np.float32))


def test_the_vlm_fallback_grounds_in_the_arm_frame(make_pair_rigs, monkeypatch):  # noqa: F811
    rt_a, rt_b, T = make_pair_rigs(YAWED)
    for rt in (rt_a, rt_b):
        rt.cfg._data["grounder"] = {"base_url": "http://127.0.0.1:9/v1", "model": "double"}   # never dialled
        rt._grounder = _BoxGrounder()
        rt.cfg._data.setdefault("perception_loop", {})["belief_fallback_age_s"] = -1.0       # no memory fix
        monkeypatch.setattr(rt.detector, "detect", lambda *a, **k: [])
    run = run_pair(rt_a, rt_b, T, "point_at", {"label": "red cube"})
    assert run["result_a"]["ok"] is True, run["result_a"]
    assert any("the VLM (2nd filter) found it" in e.text for e in rt_b.memory.events(("note",)))
    assert_equivariant(run, T)


@pytest.mark.parametrize("skill", ["localize_object", "get_observation"])
def test_perception_is_equivariant_on_a_raised_base(make_pair_rigs, skill):  # noqa: F811
    """A base 3 cm up with the scene raised alike: the remembered top_z is a
    table-frame height (B's is A's + 3 cm)."""
    rt_a, rt_b, T = make_pair_rigs(RAISED)
    run = run_pair(rt_a, rt_b, T, skill, ORACLE_T3[skill])
    assert_equivariant(run, T)


@pytest.mark.parametrize("camera", ["primary", "side"])
def test_two_phase_localization_is_equivariant(make_runtime, monkeypatch, camera):  # noqa: F811
    """The brain names the cube on frame0 and SAM re-measures it on a fresh
    frame1 (spec 2026-10-16 two-phase): both frames, on the primary camera
    and on the other-camera pass, are analysed in the running arm's frame.
    Fakes and time scale of tests/test_brain_two_phase_localize.py."""
    import functools
    from types import SimpleNamespace

    pytest.importorskip("pinocchio")
    from test_brain_two_phase_localize import (
        SCALE,
        _ImageBrain,
        _ObjectSam,
        _Scene,
        _side_cameras,
    )

    from cascade.apps import demo
    from cascade.perception.brain_detector import BrainDetector
    from cascade.skills import runtime as runtime_mod

    monkeypatch.setattr(runtime_mod, "_analyzed_frame_age_s",
                        functools.partial(runtime_mod._analyzed_frame_age_s, max_age_s=5.0 * SCALE))
    rigs = []
    for pose in (None, YAWED):
        brain, sam = _ImageBrain(), _ObjectSam()
        detector = BrainDetector(base_url="http://brain/v1", model="qwen3.8-27b", api_key_file=None, client=brain,
                                 segmenter=sam, refresh_s=4.0 * SCALE, max_cache_age_s=30.0 * SCALE,
                                 idle_refresh_s=30.0 * SCALE)
        monkeypatch.setattr(demo, "_make_detector", lambda cfg, _d=detector: _d)
        cfg = moved_cfg("so101_right", pose, camera="mock")
        cfg._data["perception_loop"] = {"enabled": False, "semantic_max_age_s": 20.0 * SCALE}   # no watcher
        rt = make_runtime(cfg)
        t = SimpleNamespace(rt=rt, scene=_Scene(rt.camera._camera))
        rt.camera.get_frame()
        rt.camera.get_frame()
        if camera == "side":
            _side_cameras(t)
        rigs.append((rt, sam))
    (rt_a, sam_a), (rt_b, sam_b) = rigs
    T = transform(YAWED)
    run = run_pair(rt_a, rt_b, T, "localize_object", {"label": "red cube"})
    assert run["result_a"].get("ok", True) is True, run["result_a"]
    assert_equivariant(run, T)
    assert len(sam_a.calls) == len(sam_b.calls) == 3, "named on frame0, re-masked, measured on frame1"


def test_the_near_hint_is_in_the_arm_frame(make_runtime, monkeypatch):  # noqa: F811
    import cascade.skills.runtime as runtime_mod

    rt = make_runtime(moved_cfg("so101_right", YAWED))
    with rt.watcher.paused():
        rt._reobserve(1)
        belief = rt.beliefs.find("red cube")
        assert belief is not None
        position = np.asarray(belief.position, dtype=float).copy()
        hints = []
        real = runtime_mod.localize_object

        def spy(*a, **k):
            hints.append(k.get("near_xyz"))
            return real(*a, **k)

        monkeypatch.setattr(runtime_mod, "localize_object", spy)
        result = rt.execute("localize_object", {"label": "red cube"})
    assert result.get("ok", True) is True, result
    assert hints and hints[0] is not None
    want = (np.linalg.inv(transform(YAWED)) @ np.append(position, 1.0))[:3]
    np.testing.assert_allclose(np.asarray(hints[0], dtype=float), want, rtol=0.0, atol=1e-9)


# ── 3. configured zones are reported in the table frame ───────────────────


def _zone_xy(result):
    assert result.get("ok", True) is True, result
    return [round(v, 6) for v in result["position_xy_m"]]


def test_configured_zones_are_reported_in_the_table_frame(make_runtime):  # noqa: F811
    pair = make_runtime(moved_cfg(["so101_left", "so101_right"], {}, move_cameras=False))
    # each arm's drop_zone is in its own base frame: right [0.20, -0.12] -> W (0.20, -0.34)
    assert _zone_xy(pair.execute("localize_object", {"label": "drop zone", "arm": "so101_right"})) == [0.20, -0.34]
    assert _zone_xy(pair.execute("localize_object", {"label": "drop zone", "arm": "so101_left"})) == [0.20, 0.34]
    assert _zone_xy(pair.execute("localize_object", {"label": "drop zone"})) == [0.20, 0.34]   # the primary
    observed = pair.execute("get_observation", {"arm": "so101_right"})
    assert [[round(v, 6) for v in z["position_xy_m"]] for z in observed["configured_zones"]] == [[0.20, -0.34]]
    single = make_runtime(moved_cfg("so101_mock", None, move_cameras=False))
    assert _zone_xy(single.execute("localize_object", {"label": "drop zone"})) == [0.20, -0.12]


# ── 4. grasp evidence records the arm pose ────────────────────────────────


@pytest.mark.parametrize("rig", ["yawed", "so101_mock"])
def test_grasp_evidence_records_the_arm_pose(make_runtime, monkeypatch, tmp_path, rig):  # noqa: F811
    """The `localized` event's points and camera pose are in the arm frame now;
    `T_world_base` lets an offline replay lift them (spec §3.4 Traces)."""
    import cascade.skills.runtime as runtime_mod

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    monkeypatch.setenv("CASCADE_GRASP_EVIDENCE_DIR", str(evidence))
    # grasp_object stays refused on a base_pose arm until Task 4 converts its
    # path; only its localization (converted here) is under test
    monkeypatch.setattr(runtime_mod, "_WORLD_TARGET_SKILLS", runtime_mod._WORLD_TARGET_SKILLS - {"grasp_object"})
    cfg = moved_cfg("so101_right", YAWED) if rig == "yawed" else moved_cfg("so101_mock", None)
    rt = make_runtime(cfg)
    rt.execute("grasp_object", {"label": "red cube"})
    events = [e["data"] for path in sorted(evidence.glob("*.json"))
              for e in json.loads(path.read_text())["events"] if e["kind"] == "localized"]
    assert events, sorted(p.name for p in evidence.iterdir())
    for event in events:
        if rig == "yawed":
            np.testing.assert_allclose(np.asarray(event["T_world_base"], dtype=float), transform(YAWED),
                                       rtol=0.0, atol=1e-12)
            # the recorded camera pose is the arm-frame one: T^-1 T_table_cam
            assert not np.allclose(np.asarray(event["effective_T_base_cam"], dtype=float)[:3, 3],
                                   np.asarray(cfg.camera.extrinsics.T, dtype=float)[:3, 3])
        else:
            assert event["T_world_base"] is None
