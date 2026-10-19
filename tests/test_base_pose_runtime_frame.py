"""base_pose Option 2: the skill runtime works in the running arm's base frame
(spec docs/superpowers/specs/2026-10-17-base-pose-runtime-frame-design.md, §3.3-3.4, §6.1).

Perception section (Task 3): every frame localization analyses is in the
running arm's base frame (`_frame_in_arm`), belief positions used as targets
go through `_to_arm`, and every report goes back out in the TABLE frame
(localize_object, point_at, preview_grasp, get_observation's `tcp_xyz`, the
configured zones). The equivariance oracle (`tests/base_pose_oracle.py`) is
the core check: an arm at base_pose T with the scene moved by T behaves
exactly like the same arm without a base_pose. Agent-facing text quotes
table-frame points too (a workspace refusal, the VLM's refusal and note), and
another camera is taken into the arm frame by its OWN extrinsics.

Grasp section (Task 4): grasp_object and grasp_at_pixel plan in the running
arm's frame; the descent and target columns, the observed-finger gate's scene
and tracked objects, the held offset (legacy pose channel and image path) and
PR-A's map-top hover (M6) follow; grasp_at_pixel's workspace filter and the
belief removal read table-frame points; a provisional hold is judged only by
the jaws of the arm that closed on it (R5).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import re
from types import SimpleNamespace

import numpy as np
import pytest
from base_pose_oracle import (  # noqa: F401  (fixtures by name)
    RAISED,
    SHIPPED_RIGHT,
    YAWED,
    assert_equivariant,
    lift,
    lifted_tcps,
    make_pair_rigs,
    make_runtime,
    moved_cfg,
    raised_table,
    raw_arm,
    run_pair,
    transform,
)

#: Task 3's oracle rows: one call per perception skill.
ORACLE_T3 = {"localize_object": {"label": "red cube"}, "point_at": {"label": "red cube"},
             "preview_grasp": {"label": "red cube"}, "get_observation": {}, "list_objects": {},
             "describe_scene": {}, "count_objects": {}}
#: The arm each oracle row runs on. point_at runs on the `mock` profile: #369
#: refuses it on every SO-101 (its 0.04 m top-down ceiling is below a 5 cm
#: cube's top + `safety.min_clearance_m` + the jaws' reach), so on so101_right
#: both rigs would only compare a refusal.
POINT_AT_ARM = "mock"
#: The arm push_object runs on: `mock` too. Its hover 10 cm above the contact
#: (z 0.117 m) is above every SO-101's top-down ceiling, so push_object
#: refuses "push poses unreachable" on so101 with or without a base_pose
#: (pre-existing; probe bp-t5/probe_push.py).
PUSH_ARM = "mock"
#: mock_small's red box: 5 cm tall on the table (z = 0 in the table frame).
WORLD_TOP_Z = 0.05

#: mock_small's red cube, with the camera shifted 10 cm in -y in the table
#: frame: W (0.198, -0.10), r = 0.231 from so101_right's base (spec §6.2).
WORLD_OBJECT_XY = (0.198, -0.10)
#: A yawed, raised and shifted mount (the Task 3 review's COMBO): the cube at
#: W (0.198, -0.10) is behind this arm, B (-0.238, 0.076, 0.018), outside its
#: workspace, so localization refuses it and quotes where it saw it.
COMBO = [0.05, -0.30, 0.02, 0.0, 0.0, -1.9]
#: The point a refusal quotes: "... at [x, y, z] (is) outside the active arm workspace".
_QUOTED = re.compile(r"at \[(-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?)\] (?:is )?outside the active arm "
                     r"workspace")
#: The point the VLM's memory note quotes: "... found it at [x, y, z]" (cm).
_NOTED = re.compile(r"the VLM \(2nd filter\) found it at \[(-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?)\]")


def _quoted(text, pattern=_QUOTED):
    found = pattern.findall(str(text))
    assert len(found) == 1, text
    return np.asarray(found[0], dtype=float)


def _world_object_rig():
    return moved_cfg("so101_right", SHIPPED_RIGHT, move_cameras=False, shift_xy=(0.0, -0.10))


# ── 1. the oracle ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("pose", [SHIPPED_RIGHT, YAWED], ids=["so101_right", "yawed"])
@pytest.mark.parametrize("skill", sorted(ORACLE_T3))
def test_perception_skills_are_equivariant(make_pair_rigs, pose, skill):  # noqa: F811
    arm = POINT_AT_ARM if skill == "point_at" else "so101_right"
    rt_a, rt_b, T = make_pair_rigs(pose, arm=arm)
    run = run_pair(rt_a, rt_b, T, skill, ORACLE_T3[skill])
    if skill in ("localize_object", "point_at", "preview_grasp"):
        assert run["result_a"].get("ok", True) is True, run["result_a"]     # rig A really ran it
    if skill == "point_at":
        assert run["cmds_a"], "rig A pointed: the arm moved"
        for rt, cmds, Tr in ((rt_a, run["cmds_a"], np.eye(4)), (rt_b, run["cmds_b"], T)):
            _assert_pointing_clears(rt, cmds, Tr)
    assert_equivariant(run, T)


# ── 2. the world point ────────────────────────────────────────────────────


def _lowest_lifted_tool_z(rt, q, T) -> float:
    """Lowest table-frame z of the TCP and the jaws (the profile's open-jaw
    sweep box, which bounds the closed jaws: #369's `_pointing_tool_points`)
    at the joint command q, lifted by the arm's base_pose T."""
    from cascade.skills.runtime import _pointing_tool_points

    tool = _pointing_tool_points(rt.cfg.grasp, rt._tool_axis_order)
    pose = rt.kin.fk(np.asarray(q, dtype=float))
    return float(lift(T, tool @ pose[:3, :3].T + pose[:3, 3])[:, 2].min())


def _assert_pointing_clears(rt, cmds, T, top=WORLD_TOP_Z) -> None:
    """The T3 review's Important 1 (#369's rule, in the table frame): no pose
    point_at commanded put the TCP or the jaws within `safety.min_clearance_m`
    of the object's top on the table."""
    margin = float(rt.cfg.safety.get("min_clearance_m"))
    assert margin == 0.03, margin                          # the shipped margin, never relaxed
    assert cmds, "point_at commanded a pose"
    low = min(_lowest_lifted_tool_z(rt, q, T) for q in cmds)
    assert low >= top + margin - 1e-9, (low, top + margin)


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
    """The world cube, seen by a camera fixed in the table frame, on an arm at
    the shipped right-arm pose (the `mock` profile: #369 refuses pointing on
    the SO-101's ceiling). The hover sits over the WORLD object and keeps the
    jaws the margin above its world top."""
    rt = make_runtime(moved_cfg(POINT_AT_ARM, SHIPPED_RIGHT, move_cameras=False, shift_xy=(0.0, -0.10)))
    raw = raw_arm(rt.arm)
    n0 = len(raw.commands)
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
    _assert_pointing_clears(rt, raw.commands[n0:], transform(SHIPPED_RIGHT))


def test_the_belief_fallback_aims_in_the_arm_frame(make_pair_rigs, monkeypatch):  # noqa: F811
    """The detector misses; a fresh belief (table frame) becomes the target,
    converted into the arm frame (`_localize`'s world-model fix). The frame
    the fallback returns is in the arm frame too: Task 4's grasp and observed
    gate read its camera pose."""
    rt_a, rt_b, T = make_pair_rigs(YAWED)
    misses, returned = [], {}
    for tag, rt in (("a", rt_a), ("b", rt_b)):
        # keep rig B's belief (seeded with A's) inside the fallback's
        # freshness window while rig A runs
        rt.cfg._data.setdefault("perception_loop", {})["belief_fallback_age_s"] = 60.0
        monkeypatch.setattr(rt.detector, "detect", lambda *a, **k: (misses.append(1), [])[1])
        real = rt._localize

        def localize(*a, _real=real, _tag=tag, _rt=rt, **k):
            frame, fix = _real(*a, **k)
            returned[_tag] = (frame.T_base_cam if frame.T_base_cam is not None
                              else _rt.extrinsics.cam_to_base())
            return frame, fix

        monkeypatch.setattr(rt, "_localize", localize)
    run = run_pair(rt_a, rt_b, T, "preview_grasp", {"label": "red cube"})
    assert misses, "the detector was asked and missed"
    assert run["result_a"]["ok"] is True, run["result_a"]
    assert any("world-model fix" in e.text for e in rt_b.memory.events(("note",))), "rig B used the belief"
    assert_equivariant(run, T)
    # rig B's camera, moved with its base by T, sits where rig A's does in the arm frame
    np.testing.assert_allclose(returned["b"], returned["a"], rtol=0.0, atol=1e-9)


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
        result = rt.execute("localize_object", {"label": "red cube"})
    assert result.get("ok", True) is True, result
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
    run = run_pair(rt_a, rt_b, T, "preview_grasp", {"label": "red cube"})
    assert run["result_a"]["ok"] is True, run["result_a"]
    assert any("found through" in e.text for e in rt_b.memory.events(("note",))), "the side camera found it"
    assert_equivariant(run, T)


#: The side camera's own pose relative to the primary's: 3 cm along the image x
#: and -2 cm along its y, in the camera's frame (so it moves with the arm in rig B).
_SIDE_OFFSET = np.array([[1.0, 0.0, 0.0, 0.03], [0.0, 1.0, 0.0, -0.02], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


def _side_camera_elsewhere(cfg):
    """`_side_camera_only`, with the side camera at its OWN pose (not the
    primary's): a wrong camera's extrinsics then put the cube 3.6 cm off."""
    _side_camera_only(cfg)
    ext = cfg._data["cameras"][-1]["extrinsics"]
    ext = ext._data if hasattr(ext, "_data") else ext
    ext["T"] = (np.asarray(ext["T"], dtype=float) @ _SIDE_OFFSET).tolist()


def _unstamped_other_cameras(rt):
    """The other cameras' frames carry no per-capture `T_base_cam` (a camera
    without capture geometry): their pose is the camera's static extrinsics."""
    for cam in rt._other_cams():
        real = cam.stream.get_frame
        cam.stream.get_frame = lambda *a, _real=real, **k: dataclasses.replace(_real(*a, **k), T_base_cam=None)


class _RedBoxGrounder:
    """A VLM grounder double that reads the image: the box around the red
    cube's pixels (mock_small's BGR red), or None on a view without it."""

    def ground(self, frame, query):
        from cascade.types import Detection

        ys, xs = np.nonzero(np.all(frame.rgb == (40, 40, 200), axis=-1))
        if not len(xs):
            return None
        return Detection(label=query, conf=0.9,
                         bbox=np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32))


@pytest.mark.parametrize("path", ["detector", "vlm"])
def test_another_camera_is_placed_by_its_own_extrinsics(make_pair_rigs, monkeypatch, path):  # noqa: F811
    """The other camera at its own pose, its frames unstamped: its detector
    pass and its VLM view take it into the arm frame with ITS extrinsics."""
    rt_a, rt_b, T = make_pair_rigs(YAWED, edit=_side_camera_elsewhere)
    for rt in (rt_a, rt_b):
        assert len(rt._other_cams()) == 1
        _unstamped_other_cameras(rt)
        assert rt._other_cams()[0].stream.get_frame().T_base_cam is None
        if path == "vlm":
            rt.cfg._data["grounder"] = {"base_url": "http://127.0.0.1:9/v1", "model": "double"}   # never dialled
            rt._grounder = _RedBoxGrounder()
            rt.cfg._data.setdefault("perception_loop", {})["belief_fallback_age_s"] = -1.0       # no memory fix
            monkeypatch.setattr(rt.detector, "detect", lambda *a, **k: [])
    run = run_pair(rt_a, rt_b, T, "preview_grasp", {"label": "red cube"})
    assert run["result_a"]["ok"] is True, run["result_a"]
    note = "found through" if path == "detector" else "the VLM (2nd filter) found it"
    assert any(note in e.text for e in rt_b.memory.events(("note",))), "the side camera found it"
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
    run = run_pair(rt_a, rt_b, T, "preview_grasp", {"label": "red cube"})
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


def _two_phase_rigs(make_runtime, monkeypatch, *, side=False):  # noqa: F811
    """Rig A (no base_pose) and rig B (YAWED, cameras moved) on the brain
    detector with SAM: the brain names the cube on frame0 and SAM re-measures
    it on a fresh frame1 (spec 2026-10-16 two-phase). Fakes and time scale of
    tests/test_brain_two_phase_localize.py. `side`: the primary sees no cube;
    another camera, at its OWN pose with unstamped frames, does. Returns
    ([(rt_a, sam_a), (rt_b, sam_b)], T)."""
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
    from cascade.perception.grounding import Extrinsics
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
        if side:
            _side_cameras(t)   # unstamped MockCamera frames
            for cam in rt._other_cams():
                cam.extrinsics = Extrinsics(T=np.asarray(cam.extrinsics.T, dtype=float) @ _SIDE_OFFSET)
        rigs.append((rt, sam))
    return rigs, transform(YAWED)


@pytest.mark.parametrize("camera", ["primary", "side"])
def test_two_phase_localization_is_equivariant(make_runtime, monkeypatch, camera):  # noqa: F811
    """Both frames of the two-phase localization, on the primary camera and
    on the other-camera pass (its own pose), are analysed in the running
    arm's frame."""
    ((rt_a, sam_a), (rt_b, sam_b)), T = _two_phase_rigs(make_runtime, monkeypatch, side=camera == "side")
    run = run_pair(rt_a, rt_b, T, "localize_object", {"label": "red cube"})
    assert run["result_a"].get("ok", True) is True, run["result_a"]
    assert_equivariant(run, T)
    assert len(sam_a.calls) == len(sam_b.calls) == 3, "named on frame0, re-masked, measured on frame1"


# ── 2b. agent text quotes table-frame points (Q1 a) ──────────────────────


def _combo_rig():
    return moved_cfg("so101_right", COMBO, move_cameras=False, shift_xy=(0.0, -0.10))


def test_a_workspace_refusal_quotes_the_table_frame_point(make_runtime):  # noqa: F811
    """The cube behind the COMBO arm is refused for its workspace; the point
    the agent reads is where the cube is on the table (the same cube seen by
    an arm without a base_pose), not its arm-frame coordinates."""
    reference = make_runtime(moved_cfg("so101_right", None, move_cameras=False, shift_xy=(0.0, -0.10)))
    seen = reference.execute("localize_object", {"label": "red cube"})
    assert seen.get("ok", True) is True, seen
    rt = make_runtime(_combo_rig())
    result = rt.execute("localize_object", {"label": "red cube"})
    assert result["ok"] is False, result
    np.testing.assert_allclose(_quoted(result["error"]), seen["position"], rtol=0.0, atol=0.005)


def test_a_vlm_refusal_and_its_note_quote_the_table_frame_point(make_runtime, monkeypatch):  # noqa: F811
    rt = make_runtime(_combo_rig())
    rt.cfg._data["grounder"] = {"base_url": "http://127.0.0.1:9/v1", "model": "double"}   # never dialled
    rt._grounder = _RedBoxGrounder()
    rt.cfg._data.setdefault("perception_loop", {})["belief_fallback_age_s"] = -1.0       # no memory fix
    monkeypatch.setattr(rt.detector, "detect", lambda *a, **k: [])
    grounded = []
    real = rt._vlm_ground_fix
    monkeypatch.setattr(rt, "_vlm_ground_fix", lambda *a, **k: grounded.append(real(*a, **k)) or grounded[-1])
    result = rt.execute("localize_object", {"label": "red cube"})
    assert result["ok"] is False, result
    assert len(grounded) == 1 and grounded[0] is not None, "the VLM found the cube (outside the workspace)"
    world = lift(transform(COMBO), grounded[0][1].position)
    assert np.hypot(*(world[:2] - WORLD_OBJECT_XY)) <= 0.005, world
    np.testing.assert_allclose(_quoted(result["error"]), world, rtol=0.0, atol=1.1e-3)
    notes = [e.text for e in rt.memory.events(("note",)) if "the VLM (2nd filter) found it" in e.text]
    assert notes
    np.testing.assert_allclose(_quoted(notes[-1], _NOTED), world, rtol=0.0, atol=1.1e-2)


def test_a_two_phase_scene_change_quotes_the_table_frame_point(make_runtime, monkeypatch):  # noqa: F811
    """frame1's fix refused for the workspace ("out of the workspace now",
    `_SceneChanged`, twice: SceneChangedError): the point in the text is rig
    A's, moved by T."""
    from cascade.skills import runtime as runtime_mod

    rigs, T = _two_phase_rigs(make_runtime, monkeypatch)
    real = runtime_mod.fix_from_detection
    nowhere = (np.full(3, 5.0), np.full(3, 6.0))
    monkeypatch.setattr(runtime_mod, "fix_from_detection",
                        lambda *a, **k: real(*a, **{**k, "workspace_bounds": nowhere}))
    errors = [rt.execute("localize_object", {"label": "red cube"}).get("error") for rt, _ in rigs]
    for error in errors:
        assert "scene changed during perception of 'red cube'" in str(error), error
    np.testing.assert_allclose(_quoted(errors[1]), lift(T, _quoted(errors[0])), rtol=0.0, atol=1.5e-3)


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
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    monkeypatch.setenv("CASCADE_GRASP_EVIDENCE_DIR", str(evidence))
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


# ── 5. the grasp path (Task 4) ────────────────────────────────────────────

#: Task 4's oracle rows. (320, 327) is the centre of mock_small's box_px
#: [301, 308, 339, 346]. Its top face is SQUARE, so the OBB's two horizontal
#: axes tie and grasp_at_pixel's yaw is a numerical tie-break: at a large yaw
#: (1.2 rad) a ~1e-16 camera-pose difference flips it (400 vs 405 commands),
#: and rig A with the same perturbation reproduces rig B exactly (Task 4
#: review Minor 3). That is not a frame error; the poses here (SHIPPED_RIGHT,
#: YAWED) are tie-free, and a new pose that flips the tie needs a non-square
#: target, not a looser tolerance.
ORACLE_T4 = {"grasp_object": {"label": "red cube"},
             "grasp_at_pixel": {"u": 320, "v": 327, "normalized": False}}


@pytest.fixture
def gate_off(monkeypatch):
    """CASCADE_OBSERVED_FINGER_GATE pinned off: the real observed-finger gate
    needs a calibrated Isaac reBot, so on the mock stack its on-path runs
    through the gate double (`observed_gate` rows) instead."""
    monkeypatch.delenv("CASCADE_OBSERVED_FINGER_GATE", raising=False)


@pytest.fixture(params=["gate off", "gate on"])
def observed_gate(request, monkeypatch):
    """CASCADE_OBSERVED_FINGER_GATE set explicitly, 0 or 1. On: the mock-stack
    gate double (tests/test_agent_results_compose.py `_gate_on`), installed on
    so101_right's backend of each runtime by the returned `install(rt)`."""
    if request.param == "gate off":
        monkeypatch.delenv("CASCADE_OBSERVED_FINGER_GATE", raising=False)
        return lambda rt: None
    from test_agent_results_compose import _gate_on

    monkeypatch.setenv("CASCADE_OBSERVED_FINGER_GATE", "1")
    return lambda rt: _gate_on(monkeypatch, raw_arm(_right(rt)))


def _right(rt):
    """so101_right's SafeArm (the pair's second arm), or the lone arm (a
    so101_right or a `mock` rig)."""
    rig = rt.arm_rig
    return rig.get("so101_right") if rig is not None and "so101_right" in rig.arms else rt.arm


def _lowest(tcps):
    tcps = np.asarray(tcps, dtype=float).reshape(-1, 3)
    assert len(tcps), "the arm moved"
    return tcps[int(np.argmin(tcps[:, 2]))]


@pytest.mark.parametrize("pose", [SHIPPED_RIGHT, YAWED], ids=["so101_right", "yawed"])
@pytest.mark.parametrize("skill", sorted(ORACLE_T4))
def test_grasps_are_equivariant(make_pair_rigs, observed_gate, pose, skill):  # noqa: F811
    rt_a, rt_b, T = make_pair_rigs(pose)
    for rt in (rt_a, rt_b):
        observed_gate(rt)
    run = run_pair(rt_a, rt_b, T, skill, ORACLE_T4[skill])
    assert run["result_a"]["ok"] is True, run["result_a"]          # rig A really grasped
    assert run["cmds_a"] and run["jaws_a"]
    assert_equivariant(run, T)


@pytest.mark.parametrize("pose", [SHIPPED_RIGHT, YAWED], ids=["so101_right", "yawed"])
def test_the_grasped_belief_is_removed_near_its_table_frame_position(make_pair_rigs, gate_off, pose):  # noqa: F811
    """Two "red cube" beliefs: the grasp removes the one it took
    (`mark_removed(near=)` the TABLE-frame fix position). The second cube sits
    where rig B's arm-frame fix position reads as a table-frame point, so a
    near hint left in the arm frame removes the wrong belief."""
    rt_a, rt_b, T = make_pair_rigs(pose)
    for rt, M in ((rt_a, np.eye(4)), (rt_b, T)):
        seen = rt.beliefs.find("red cube")
        assert seen is not None
        decoy = dataclasses.replace(seen, position=lift(M, [0.198, 0.22, float(seen.position[2])]),
                                    points=None, top_z=None if seen.top_z is None else seen.top_z + M[2, 3])
        rt.beliefs._beliefs.append(decoy)
    run = run_pair(rt_a, rt_b, T, "grasp_object", {"label": "red cube"})
    # rig A grasped (the verifier then reads the second cube's belief, which
    # did not rise: its verdict is the same on both rigs)
    assert run["result_a"]["held"] == "red cube", json.dumps(run["result_a"], default=str)
    assert [b["label"] for b in run["beliefs_a"]].count("red cube") == 1      # rig A removed the grasped one
    assert_equivariant(run, T)


@pytest.mark.parametrize("rig", ["so101_right", "pair, arm=so101_right"])
def test_the_grasp_descends_on_the_world_object(make_runtime, observed_gate, rig):  # noqa: F811
    """Camera fixed in W, the cube at W (0.198, -0.10): the lowest commanded
    TCP, lifted by the base_pose, is rig A's (the same arm without a base_pose,
    the cube at the same arm-frame place) moved by T, and on the world cube."""
    T = transform(SHIPPED_RIGHT)
    reference = make_runtime(moved_cfg("so101_right", None, move_cameras=False, shift_xy=(0.0, 0.12)))
    observed_gate(reference)
    n0 = len(raw_arm(reference.arm).commands)
    assert reference.execute("grasp_object", {"label": "red cube"})["ok"] is True
    low_a = _lowest(lifted_tcps(reference, reference.arm, n0, np.eye(4)))
    if rig == "so101_right":
        rt, args = make_runtime(_world_object_rig()), {}
    else:
        rt = make_runtime(moved_cfg(["so101_left", "so101_right"], {}, move_cameras=False, shift_xy=(0.0, -0.10)))
        args = {"arm": "so101_right"}
    observed_gate(rt)
    safe = _right(rt)
    others = [s for s in (rt.arm_rig.arms.values() if rt.arm_rig is not None else []) if s is not safe]
    m0 = [len(raw_arm(s).commands) for s in others]
    n0 = len(raw_arm(safe).commands)
    result = rt.execute("grasp_object", {"label": "red cube", **args})     # main: refused by #352's guard
    assert result["ok"] is True, result
    low_b = _lowest(lifted_tcps(rt, safe, n0, T))
    np.testing.assert_allclose(low_b, lift(T, low_a), rtol=0.0, atol=0.005)
    assert np.hypot(*(low_b[:2] - WORLD_OBJECT_XY)) <= 0.025, low_b
    assert [len(raw_arm(s).commands) for s in others] == m0               # the left arm never moved


def _select_spy(monkeypatch):
    import cascade.skills.runtime as runtime_mod

    selected = []
    real = runtime_mod.select_grasp

    def spy(*a, **k):
        selected.append(real(*a, **k))
        return selected[-1]

    monkeypatch.setattr(runtime_mod, "select_grasp", spy)
    return selected


def test_the_descent_and_target_columns_sit_on_the_target(make_runtime, gate_off, monkeypatch):  # noqa: F811
    """The descent cylinder and the target column are centred on the grasp
    point in the arm frame: lifted, they stand on the world cube."""
    T = transform(SHIPPED_RIGHT)
    rt = make_runtime(_world_object_rig())
    rt.cfg.grasp._data["approach_target_exemption"] = True
    harness = rt.arm.harness
    descents, columns = [], []
    real_descent, real_column = harness.allow_grasp_descent, harness.target_column_exemption

    def descent(center_xy, *a, **k):
        descents.append(np.asarray(center_xy, dtype=float)[:2].copy())
        return real_descent(center_xy, *a, **k)

    def column(cylinder, **k):
        if cylinder is not None:
            columns.append(np.asarray(cylinder[0], dtype=float)[:2].copy())
        return real_column(cylinder, **k)

    monkeypatch.setattr(harness, "allow_grasp_descent", descent)
    monkeypatch.setattr(harness, "target_column_exemption", column)
    selected = _select_spy(monkeypatch)
    result = rt.execute("grasp_object", {"label": "red cube"})
    assert result["ok"] is True, result
    assert descents and columns and selected
    grasp_w = lift(T, selected[-1][0].position)
    for centre in descents + columns:
        centre_w = lift(T, [centre[0], centre[1], 0.0])[:2]
        assert np.hypot(*(centre_w - grasp_w[:2])) <= 0.005, (centre_w, grasp_w)
        assert np.hypot(*(centre_w - WORLD_OBJECT_XY)) <= 0.02, centre_w


def _observed_gate_on(monkeypatch, rt):
    """CASCADE_OBSERVED_FINGER_GATE=1 on the mock stack (tests/test_agent_results_compose.py's
    `_GateDouble` / `_gate_on`); `for_runtime` records the frame and fix it receives."""
    from test_agent_results_compose import _gate_on

    from cascade.grasping import observed_scene

    gate = _gate_on(monkeypatch, raw_arm(rt.arm))
    seen = []

    def for_runtime(runtime, frame, fix, state):
        seen.append((frame, fix))
        return gate

    monkeypatch.setattr(observed_scene, "for_runtime", for_runtime)
    return seen


def test_the_observed_gate_reads_the_scene_in_the_arm_frame(make_runtime, monkeypatch):  # noqa: F811
    """With the gate on, `for_runtime` receives the arm-frame frame (camera
    pose T^-1 T_table_cam) and the arm-frame fix: the finger hulls it checks
    come from the arm's own FK."""
    T = transform(YAWED)
    rt = make_runtime(moved_cfg("so101_right", YAWED))
    seen = _observed_gate_on(monkeypatch, rt)
    rt.execute("grasp_object", {"label": "red cube"})                     # main: refused by #352's guard
    assert seen, "the observed-finger gate was built"
    frame, fix = seen[0]
    np.testing.assert_allclose(frame.T_base_cam, np.linalg.inv(T) @ rt.extrinsics.cam_to_base(),
                               rtol=0.0, atol=1e-9)
    # the cube moved with the arm: at B (0.198, 0) whatever T is (W (0.212, -0.143) here)
    assert np.hypot(*(np.asarray(fix.position[:2]) - (0.198, 0.0))) <= 0.005, fix.position


def _factory_inputs():
    from test_observed_finger_scene import runtime_factory_inputs

    return runtime_factory_inputs()


def _scene_recorder(monkeypatch):
    from cascade.grasping import observed_scene

    recorded = {}
    real = observed_scene.ObservedScene

    def scene(frame, mask, T_base_cam, **kw):
        recorded.update(T=np.asarray(T_base_cam, dtype=float).copy(), tracked=list(kw.get("tracked") or ()))
        return real(frame, mask, T_base_cam, **kw)

    monkeypatch.setattr(observed_scene, "ObservedScene", scene)
    return recorded


def test_for_runtime_converts_tracked_beliefs(monkeypatch):
    """Unit (stub runtime, T = YAWED): the world-model objects that name a
    refused point are taken into the arm frame (centres and remembered
    clouds), so the target itself is recognised and skipped there."""
    from types import SimpleNamespace

    from cascade.grasping import observed_scene

    T = transform(YAWED)
    to_arm = np.linalg.inv(T)
    runtime, frame, fix, state = _factory_inputs()
    runtime.T_world_base = T
    fix.position, fix.label = np.array([0.20, 0.01, 0.03]), "red cube"
    target = SimpleNamespace(label="red cube", position=lift(T, fix.position), points=None,
                             diameter_m=0.05, extent=None)
    lemon = SimpleNamespace(label="lemon", position=np.array([0.316, 0.011, 0.03]),
                            points=np.array([[0.30, 0.01, 0.04], [0.33, 0.01, 0.04]]), diameter_m=None, extent=None)
    cup = SimpleNamespace(label="cup", position=np.array([0.40, 0.10, 0.05]), points=None,
                          diameter_m=0.06, extent=None)
    runtime.beliefs = SimpleNamespace(all=lambda: [target, lemon, cup])
    recorded = _scene_recorder(monkeypatch)
    observed_scene.for_runtime(runtime, frame, fix, state)
    tracked = {t.label: t for t in recorded["tracked"]}
    assert sorted(tracked) == ["cup", "lemon"]                       # the target (in W) is skipped in B
    np.testing.assert_allclose(tracked["lemon"].points, lift(to_arm, lemon.points), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(tracked["cup"].points[0], lift(to_arm, cup.position), rtol=0.0, atol=1e-12)
    assert tracked["cup"].radius_m == pytest.approx(0.03)
    np.testing.assert_allclose(lemon.points[0], [0.30, 0.01, 0.04])   # the beliefs themselves are untouched


def test_for_runtime_takes_the_static_camera_pose_into_the_arm_frame(monkeypatch):
    """A frame without a per-capture pose borrows the primary camera's static
    calibration (table frame): taken into the arm frame, T^-1 E."""
    from types import SimpleNamespace

    from cascade.grasping import observed_scene

    T = transform(YAWED)
    runtime, frame, fix, state = _factory_inputs()
    E = frame.T_base_cam.copy()
    frame.T_base_cam = None
    frame.capture["camera"] = runtime.cfg.camera.get("sim_camera")
    runtime.extrinsics = SimpleNamespace(mode="eye_to_hand", T=E)
    runtime.T_world_base = T
    recorded = _scene_recorder(monkeypatch)
    observed_scene.for_runtime(runtime, frame, fix, state)
    np.testing.assert_allclose(recorded["T"], np.linalg.inv(T) @ E, rtol=0.0, atol=1e-12)
    runtime.T_world_base = None                                       # no base_pose: the calibration as is
    observed_scene.for_runtime(runtime, frame, fix, state)
    np.testing.assert_array_equal(recorded["T"], E)


def test_the_held_offset_is_measured_in_the_arm_frame(make_pair_rigs, gate_off, monkeypatch):  # noqa: F811
    """The offset of the held object from the TCP is an arm-frame vector: the
    grasp's, the legacy pose channel's (a table-frame pose) and the image
    path's (the frame taken into the arm frame) all equal rig A's."""
    from cascade.skills import held_observation
    from cascade.skills import runtime as runtime_mod

    rt_a, rt_b, T = make_pair_rigs(YAWED)
    run = run_pair(rt_a, rt_b, T, "grasp_object", {"label": "red cube"})
    assert run["result_a"]["ok"] is True and run["result_b"]["ok"] is True, run["result_b"]
    np.testing.assert_allclose(rt_b._held_offset, rt_a._held_offset, rtol=0.0, atol=1e-3)
    # the image path: the camera frame goes into the arm frame before the cube
    # is measured (MockArm has no capture-time state: the current one stands in)
    for rt in (rt_a, rt_b):
        raw = raw_arm(rt.arm)
        monkeypatch.setattr(raw, "state_from_frame", lambda _frame, _raw=raw: _raw.get_state(), raising=False)
    image = {tag: held_observation.measure(rt, runtime_mod.localize_object) for tag, rt in (("a", rt_a), ("b", rt_b))}
    assert image["a"].channel == image["b"].channel == "captured_image", image
    np.testing.assert_allclose(image["b"].offset, image["a"].offset, rtol=0.0, atol=1e-3)
    # the legacy pose channel: a table-frame object pose
    tcp_b = rt_b.kin.fk(rt_b.arm.get_state().q)[:3, 3]
    want = np.array([0.01, 0.02, -0.03])
    world_point = lift(T, tcp_b + want)
    rt_b._object_pose = lambda _label: world_point.copy()
    legacy = held_observation.measure(rt_b, runtime_mod.localize_object)
    assert legacy.channel == "legacy_pose", legacy
    np.testing.assert_allclose(legacy.offset, want, rtol=0.0, atol=1e-9)


def test_grasp_at_pixel_filters_in_the_table_frame(make_runtime, gate_off, monkeypatch):  # noqa: F811
    """The cube sits 5 cm from so101_right's base in the table frame (W (0.05,
    -0.22)): the workspace filter (table frame, D6) refuses it as the robot.
    Its arm-frame centre B (0.05, 0) is 23 cm from that base: read as a
    table-frame point it would pass."""
    rt = make_runtime(moved_cfg("so101_right", SHIPPED_RIGHT, move_cameras=False,
                                shift_xy=(0.05 - 0.198, -0.22)))
    raw = raw_arm(rt.arm)
    n0, jaws = len(raw.commands), []
    real = raw.set_gripper
    monkeypatch.setattr(raw, "set_gripper", lambda *a, **k: (jaws.append(a), real(*a, **k))[1])
    result = rt.execute("grasp_at_pixel", dict(ORACLE_T4["grasp_at_pixel"]))
    assert result["ok"] is False, result
    assert "the object at pixel (320, 327) is not graspable: robot" in result["error"], result
    assert len(raw.commands) == n0 and jaws == []


def _cup(T_points_top=0.093, r=0.03, n=40, xy=(0.198, 0.12)):
    """A cup in the ARM frame: wall points on a ring around `xy`, from z 0 to
    its top (tests/test_pregrasp_map_top.py's `cup_fix`)."""
    from types import SimpleNamespace

    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    side = np.c_[xy[0] + r * np.cos(a), xy[1] + r * np.sin(a), np.linspace(0.0, T_points_top, n)]
    return SimpleNamespace(position=np.array([*xy, T_points_top / 2]), points=side)


@pytest.mark.parametrize("pose", [RAISED, YAWED], ids=["raised", "yawed"])
def test_pregrasp_map_top_reads_the_map_in_the_map_frame(pose):                       # M6
    """PR-A's map-top hover: the target's column is queried where it stands in
    the map (the table frame: centre and floor lifted by the harness's own
    `_to_map_frame`), and the map's top comes back into the arm frame."""
    from types import SimpleNamespace

    from cascade.safety.harness import SafetyHarness, SafetyLimits
    from cascade.skills.runtime import _column_geometry, _pregrasp_min_z

    T = transform(pose)
    gcfg = {"pregrasp_clear_top_m": 0.03, "pregrasp_map_top": True, "exempt_radius_m": 0.15}
    harness = SafetyHarness(SafetyLimits(np.full(3, -1.0), np.full(3, 1.0), table_z=-float(T[2, 3]),
                                         table_clearance=0.01), base_pose=T)
    fix = _cup()
    fix_top = float(fix.points[:, 2].max())
    world_top = float(lift(T, [*fix.position[:2], fix_top])[2]) + 0.009     # the map's top: 9 mm above the observed one
    calls = []

    def column_top(center_xy, radius_m, z_min):
        calls.append((np.asarray(center_xy, dtype=float)[:2].copy(), float(radius_m), float(z_min)))
        return world_top, 0.01

    harness.occupancy = SimpleNamespace(column_top=column_top)
    hover = _pregrasp_min_z(gcfg, harness, fix)
    centre_b, radius, z_min_b = _column_geometry(gcfg, harness, fix.position[:2], fix.points)
    centre_w = lift(T, [centre_b[0], centre_b[1], z_min_b])
    ((centre, r, z_min),) = calls
    np.testing.assert_allclose(centre, centre_w[:2], rtol=0.0, atol=1e-12)
    assert r == radius and z_min == pytest.approx(centre_w[2], abs=1e-12)
    today = fix_top + 0.03
    assert hover == pytest.approx((today + 0.009 + 0.005, today), abs=1e-9)   # (map hover, today's hover), B


def test_the_air_grasp_exit_column_and_ceiling_share_the_arm_frame(make_runtime, gate_off, monkeypatch):  # noqa: F811  (M6)
    """An air grasp's exit: its column is centred on the cube in the arm frame
    (lifted: on the world cube) and its ceiling is the hover TCP + 5 mm in the
    arm frame (a base 3 cm up: not the table-frame height)."""
    T = transform(RAISED)
    rt = make_runtime(raised_table(moved_cfg("so101_right", RAISED, move_cameras=False, shift_xy=(0.0, -0.10))))
    rt.cfg.grasp._data["approach_target_exemption"] = True
    raw_arm(rt.arm).object_stop_frac = None                              # the jaws close on air
    harness = rt.arm.harness
    exits, hovers = [], []
    real_column, real_exit = harness.target_column_exemption, rt._air_grasp_exit

    def column(cylinder, **k):
        if "z_max" in k:
            exits.append((cylinder, k["z_max"]))
        return real_column(cylinder, **k)

    def air_exit(col, move, **k):
        hovers.append(np.asarray(k["hover_q"], dtype=float).copy())
        return real_exit(col, move, **k)

    monkeypatch.setattr(harness, "target_column_exemption", column)
    monkeypatch.setattr(rt, "_air_grasp_exit", air_exit)
    result = rt.execute("grasp_object", {"label": "red cube"})
    assert result["ok"] is False and "air grasp" in result["error"], result
    ((cylinder, z_max),) = exits
    centre_w = lift(T, [cylinder[0][0], cylinder[0][1], 0.0])
    assert np.hypot(*(centre_w[:2] - WORLD_OBJECT_XY)) <= 0.005, centre_w
    assert z_max == pytest.approx(float(harness.kin.fk(hovers[0])[2, 3]) + 0.005, abs=1e-9)


def test_the_harness_compares_map_voxels_with_the_arm_frame_ceiling():   # M6, harness unit
    """A voxel 1 cm above the lifted ceiling is not the target's (checked at
    full strength); one 1 cm below is (base 3 cm up)."""
    from cascade.safety.harness import SafetyHarness, SafetyLimits

    T = transform(RAISED)
    harness = SafetyHarness(SafetyLimits(np.full(3, -1.0), np.full(3, 1.0), table_z=-0.03), base_pose=T)
    centre, z_max = np.array([0.198, 0.12]), 0.10
    ceiling_w = lift(T, [centre[0], centre[1], z_max])
    with harness.target_column_exemption((centre, 0.05, -0.01), z_max=z_max):
        exempt = harness._grasp_exempt
        for dz, inside in ((0.01, False), (-0.01, True)):
            voxel_w = ceiling_w + [0.0, 0.0, dz]
            assert bool(harness._in_cylinder(harness._to_base_frame(voxel_w[None])[0], exempt)) is inside, dz


def test_a_yawed_grasp_column_refusal_replays_with_the_same_verdict():   # R20
    """A grasp-path refusal on a YAWED reBot mount: the target column comes
    from Task 4's arm-frame fix (`_column_geometry`), the obstacle stands 2 cm
    ahead of the fingertips in the map, outside that column. The recorded
    report replays through scripts/replay_clearance.py with the same verdict
    and clearance: reports and replay stay as #354 shipped them."""
    pytest.importorskip("pinocchio")
    import importlib.util
    from pathlib import Path

    from test_harness_tool_volume import side_grasp_geometry
    from test_replay_clearance import WHOLE, at_x, lattice, recorder, voxel_box

    from cascade.skills.runtime import _column_geometry, _column_kw

    script = Path(__file__).resolve().parents[1] / "scripts" / "replay_clearance.py"
    spec = importlib.util.spec_from_file_location("replay_clearance_t4", script)
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    rebot = side_grasp_geometry()
    Tb = transform(YAWED)
    q = at_x(rebot, 0.4498)
    T_tcp = rebot.kin.fk(q)
    T_map = Tb @ T_tcp
    voxel = lattice(T_map[:3, 3] + T_map[:3, 0] * 0.02)                 # 2 cm ahead of the fingertips (map)
    voxel_b = lift(np.linalg.inv(Tb), voxel)
    away = T_tcp[:2, 3] - voxel_b[:2]
    away = away / np.linalg.norm(away)
    # the grasped target: a 1 cm cup 3 cm from the TCP, on the far side from the obstacle (arm frame)
    fix = _cup(T_points_top=float(T_tcp[2, 3]) - 0.03, r=0.005, xy=T_tcp[:2, 3] + 0.03 * away)
    h, _ = recorder(rebot, voxel_box(voxel), base_pose=Tb, snapshot=WHOLE)
    column = _column_geometry({"exempt_radius_m": 0.07}, h, fix.position[:2], fix.points)
    assert np.linalg.norm(voxel_b[:2] - column[0]) > column[1] + 0.005       # premise: outside the column (B)
    reason = h.vet_pose(q, **_column_kw(column))
    assert reason is not None and "clearance" in reason, reason
    reports = h.drain_clearance_reports()
    assert reports
    report = reports[0]
    assert report["exemption"] == {"cylinder": [float(column[0][0]), float(column[0][1]), column[1], column[2]]}
    out = rc.replay(report, arm="rebot_rs")
    assert out["old"].startswith("refused: "), out["old"]
    assert out["replay_clearance_m"] == pytest.approx(report["clearance_m"], abs=1e-6)
    assert out["model_mismatch"] is False


def test_a_provisional_hold_is_judged_by_the_arm_that_closed(make_runtime, gate_off, monkeypatch):  # noqa: F811  (R5)
    """A grasp interrupted after its close on so101_right leaves a provisional
    hold. A call on so101_left reads the LEFT jaws (open): it neither promotes
    nor refutes the hold; a call on so101_right promotes it from the right
    jaws (stalled on the cube)."""
    rt = make_runtime(moved_cfg(["so101_left", "so101_right"], {}, move_cameras=False, shift_xy=(0.0, -0.10)))
    real_lift = rt._move_carry

    def interrupted_lift(*a, **k):
        from cascade.types import SkillError

        raise SkillError("lift refused (test)")

    monkeypatch.setattr(rt, "_move_carry", interrupted_lift)
    result = rt.execute("grasp_object", {"label": "red cube", "arm": "so101_right"})
    assert result["ok"] is False and "lift refused (test)" in str(result.get("error")), result
    prov = rt._held_provisional
    assert prov is not None and prov[0] == "red cube" and not rt.held_object
    monkeypatch.setattr(rt, "_move_carry", real_lift)
    seen = []

    def spy():
        rt._reconcile_held()
        seen.append((rt._held_provisional, rt.held_object))
        return {"ok": True}

    monkeypatch.setattr(rt, "skill_list_objects", spy)
    rt.execute("list_objects", {"arm": "so101_left"})      # the left jaws are open: not this arm's hold
    rt.execute("list_objects", {})                           # the primary (left) again, unselected
    rt.execute("list_objects", {"arm": "so101_right"})     # the right jaws: stalled on the cube
    assert seen[0] == (prov, None) and seen[1] == (prov, None), seen
    assert seen[2] == (None, "red cube"), seen


# ── 6. place, push and pick_and_place (Task 5) ───────────────────────────

#: Task 5's oracle sequences: each step runs on both rigs and is compared.
#: place_at's (x, y, z) is rig A's arm-frame point, mapped to the table
#: frame for rig B (`map_args`).
ORACLE_T5 = {
    "grasp+place_at": [("grasp_object", {"label": "red cube"}), ("place_at", {"x": 0.20, "y": -0.08, "z": 0.06})],
    "grasp+place_on_object": [("grasp_object", {"label": "red cube"}), ("place_on_object", {"label": "red cube"})],
    "push_object": [("push_object", {"label": "red cube", "direction": "left", "distance_m": 0.05})],
    "pick_and_place": [("pick_and_place", {"object": "red cube"})],
}
#: The release target of the world-point rows (table frame) and rig A's
#: matching arm-frame point (so101_right's shipped base_pose is a pure -0.22 m
#: shift in y).
PLACE_W = (0.22, -0.30)
PLACE_A = (0.22, -0.08)
#: so101_right's drop_zone [0.20, -0.12] (its base frame) in the table frame.
RIGHT_DROP_W = (0.20, -0.34)
HELD_TEXT = "is held by arm 'so101_right'"


@pytest.mark.parametrize("pose", [SHIPPED_RIGHT, YAWED], ids=["so101_right", "yawed"])
@pytest.mark.parametrize("seq", sorted(ORACLE_T5))
def test_place_and_push_are_equivariant(make_pair_rigs, observed_gate, pose, seq):  # noqa: F811
    rt_a, rt_b, T = make_pair_rigs(pose, arm=PUSH_ARM if seq == "push_object" else "so101_right")
    for rt in (rt_a, rt_b):
        observed_gate(rt)
    for skill, args in ORACLE_T5[seq]:
        run = run_pair(rt_a, rt_b, T, skill, args)
        assert run["result_a"].get("ok", True) is True, (skill, run["result_a"])      # rig A really ran it
        assert run["cmds_a"], f"rig A moved for {skill}"
        assert_equivariant(run, T)


def _pair_world_rig():
    """The shipped pair (left primary at its base_pose, right at
    SHIPPED_RIGHT), camera fixed in the table frame, the cube at W (0.198, -0.10)."""
    return moved_cfg(["so101_left", "so101_right"], {}, move_cameras=False, shift_xy=(0.0, -0.10))


def _rig_and_args(make_runtime, rig):  # noqa: F811
    if rig == "so101_right":
        return make_runtime(_world_object_rig()), {}
    return make_runtime(_pair_world_rig()), {"arm": "so101_right"}


def _fusion_held(rt):
    """The watcher's belief fusion held (the mock cube never moves: a fusion
    pass after a place or a push would put the belief back where the camera
    still sees it)."""
    return rt.watcher.paused() if rt.watcher is not None else contextlib.nullcontext()


def _offset_spy(rt, monkeypatch):
    """The held offset place_at aims with (arm frame), per call."""
    offsets = []
    real = rt._held_object_observation

    def observe(*a, **k):
        out = real(*a, **k)
        offsets.append(None if out.offset is None else np.asarray(out.offset, dtype=float).copy())
        return out

    monkeypatch.setattr(rt, "_held_object_observation", observe)
    return offsets


def _release_point(rt, safe, n0, T, offsets):
    """(lifted release TCP, the object point there): the lowest commanded TCP of
    the place, and that TCP plus the held offset, rotated into the table frame (R16)."""
    tcp = _lowest(lifted_tcps(rt, safe, n0, T))
    offset = offsets[-1] if offsets and offsets[-1] is not None else np.zeros(3)
    return tcp, tcp + np.asarray(T)[:3, :3] @ offset


@pytest.mark.parametrize("rig", ["so101_right", "pair, arm=so101_right"])
def test_place_at_releases_at_the_world_point(make_runtime, gate_off, monkeypatch, rig):  # noqa: F811
    """grasp, then place_at the table-frame point (0.22, -0.30): the release
    TCP is rig A's (the same arm without a base_pose, the cube at the same
    arm-frame place) moved by T; the OBJECT point (TCP + held offset, R16)
    and the belief are on the world target."""
    T = transform(SHIPPED_RIGHT)
    reference = make_runtime(moved_cfg("so101_right", None, move_cameras=False, shift_xy=(0.0, 0.12)))
    assert reference.execute("grasp_object", {"label": "red cube"})["ok"] is True
    offsets_a = _offset_spy(reference, monkeypatch)
    n0 = len(raw_arm(reference.arm).commands)
    placed_a = reference.execute("place_at", {"x": PLACE_A[0], "y": PLACE_A[1]})
    assert placed_a.get("ok", True) is True, placed_a
    tcp_a, _ = _release_point(reference, reference.arm, n0, np.eye(4), offsets_a)

    rt, args = _rig_and_args(make_runtime, rig)
    safe = _right(rt)
    others = [s for s in (rt.arm_rig.arms.values() if rt.arm_rig is not None else []) if s is not safe]
    assert rt.execute("grasp_object", {"label": "red cube", **args})["ok"] is True
    offsets = _offset_spy(rt, monkeypatch)
    m0 = [len(raw_arm(s).commands) for s in others]
    n0 = len(raw_arm(safe).commands)
    with _fusion_held(rt):
        placed = rt.execute("place_at", {"x": PLACE_W[0], "y": PLACE_W[1], **args})     # main: refused by #352's guard
        belief = _belief_xy(rt, "red cube")
    assert placed.get("ok", True) is True, placed
    tcp_b, object_b = _release_point(rt, safe, n0, T, offsets)
    np.testing.assert_allclose(tcp_b, lift(T, tcp_a), rtol=0.0, atol=0.005)
    assert np.hypot(*(object_b[:2] - PLACE_W)) <= 0.005, (object_b, tcp_b)
    assert np.hypot(*(belief - PLACE_W)) <= 0.005
    np.testing.assert_allclose(placed["at"][:2], PLACE_W, rtol=0.0, atol=1.1e-3)
    assert [len(raw_arm(s).commands) for s in others] == m0               # the left arm never moved


def test_push_passes_through_the_world_object(make_runtime, gate_off, monkeypatch):  # noqa: F811
    """The straight push segment (lifted) runs through the world cube, and the
    estimate it reports is the cube moved along the arm's +y in the table frame."""
    T = transform(SHIPPED_RIGHT)
    rt = make_runtime(moved_cfg(PUSH_ARM, SHIPPED_RIGHT, move_cameras=False, shift_xy=(0.0, -0.10)))
    targets = []
    real = rt._move_joints

    def move_joints(q, *a, **k):
        targets.append(np.asarray(q, dtype=float).copy())
        return real(q, *a, **k)

    monkeypatch.setattr(rt, "_move_joints", move_joints)
    with _fusion_held(rt):
        rt._reobserve(1)                                                  # the masked depth refresh the body check needs
        result = rt.execute("push_object", {"label": "red cube", "direction": "left", "distance_m": 0.05})
        belief = _belief_xy(rt, "red cube")
    assert result.get("ok", True) is True, result                         # main: refused by #352's guard
    start, end = (lift(T, rt.kin.fk(q)[:3, 3]) for q in targets[1:3])     # hover, START, END, hover
    d = (end - start)[:2] / np.linalg.norm((end - start)[:2])
    rel = np.asarray(WORLD_OBJECT_XY) - start[:2]
    assert abs(rel[0] * d[1] - rel[1] * d[0]) <= 0.005, (start, end)      # perpendicular distance
    np.testing.assert_allclose(result["new_position_estimate"][:2],
                               (WORLD_OBJECT_XY[0], WORLD_OBJECT_XY[1] + 0.05), rtol=0.0, atol=0.006)
    # The belief store blends the estimate (conf 0.6) with what it held: the
    # belief lies between the cube and the table-frame estimate (an arm-frame
    # estimate, y 0.17, would pull it past the estimate).
    cube, est = np.asarray(WORLD_OBJECT_XY), np.asarray(result["new_position_estimate"][:2])
    u = float(np.dot(belief - cube, est - cube) / np.dot(est - cube, est - cube))
    assert 0.0 < u <= 1.0 + 0.02 and np.hypot(*(cube + u * (est - cube) - belief)) <= 0.006, (belief, est)


def test_pick_and_place_uses_the_running_arms_drop_zone(make_runtime, gate_off, monkeypatch):  # noqa: F811
    """pick_and_place on the pair's right arm: its OWN drop zone (its base
    frame, [0.20, -0.12]) is W (0.20, -0.34); the object point lands there,
    the result reports it in the table frame, and the left arm never moves."""
    T = transform(SHIPPED_RIGHT)
    rt = make_runtime(_pair_world_rig())
    left, right = rt.arm_rig.get("so101_left"), _right(rt)
    offsets = _offset_spy(rt, monkeypatch)
    l0, n0 = len(raw_arm(left).commands), len(raw_arm(right).commands)
    places = []
    real = rt.skill_place_at

    def place_at(*a, **k):
        places.append(len(raw_arm(right).commands))
        return real(*a, **k)

    monkeypatch.setattr(rt, "skill_place_at", place_at)
    result = rt.execute("pick_and_place", {"object": "red cube", "arm": "so101_right"})
    assert result.get("ok", True) is True, result                         # main: refused by #352's guard
    assert places
    commands = raw_arm(right).commands[places[-1]:]
    tcp = _lowest(lift(T, np.array([right.harness.kin.fk(np.asarray(q, dtype=float))[:3, 3] for q in commands])))
    object_point = tcp + np.asarray(T)[:3, :3] @ offsets[-1]
    assert np.hypot(*(object_point[:2] - RIGHT_DROP_W)) <= 0.005, (object_point, tcp)
    np.testing.assert_allclose(result["placed_at"][:2], RIGHT_DROP_W, rtol=0.0, atol=1.1e-3)
    assert len(raw_arm(left).commands) == l0 and len(raw_arm(right).commands) > n0


class _JawLog:
    """Every arm and jaw command of each raw MockArm of a runtime."""

    def __init__(self, rt, monkeypatch):
        self.rt, self.jaws = rt, {}
        for name, safe in rt.arm_rig.arms.items():
            raw = raw_arm(safe)
            self.jaws[name] = []
            real = raw.set_gripper
            monkeypatch.setattr(raw, "set_gripper",
                                lambda *a, _real=real, _log=self.jaws[name], **k: (_log.append(a), _real(*a, **k))[1])

    def mark(self):
        return ({n: len(raw_arm(s).commands) for n, s in self.rt.arm_rig.arms.items()},
                {n: len(j) for n, j in self.jaws.items()})

    def since(self, mark):
        arms, jaws = mark
        return ({n: len(raw_arm(s).commands) - arms[n] for n, s in self.rt.arm_rig.arms.items()},
                {n: len(j) - jaws[n] for n, j in self.jaws.items()})


def _assert_one_failed_line(result):
    text = json.dumps(result, default=str)
    assert text.count("this attempt FAILED") <= 1, text
    assert "move_home" not in str(result.get("next_action", "")) and "arm_left_in_place" not in result, result


@pytest.mark.parametrize("skill,args", [
    ("place_at", {"x": PLACE_W[0], "y": PLACE_W[1]}),
    ("place_on_object", {"label": "red cube"}),
    ("pick_and_place", {"object": "red cube"}),
])
def test_a_place_on_the_arm_that_does_not_hold_is_refused(make_runtime, gate_off, monkeypatch, skill, args):  # noqa: F811
    """Review Focus 2 (spec D9 / §5): the right arm holds the cube; a release
    skill on the LEFT arm is refused before any work (no command on either
    arm, no perception), naming the holding arm; the right arm still holds it."""
    rt = make_runtime(_pair_world_rig())
    assert rt.execute("grasp_object", {"label": "red cube", "arm": "so101_right"})["held"] == "red cube"
    log = _JawLog(rt, monkeypatch)
    mark = log.mark()
    looked = []
    real_localize = rt._localize
    monkeypatch.setattr(rt, "_localize", lambda *a, **k: (looked.append(a), real_localize(*a, **k))[1])
    result = rt.execute(skill, {**args, "arm": "so101_left"})
    assert looked == [], looked                                           # refused before perceiving
    assert result["ok"] is False, result
    assert HELD_TEXT in result["error"] and "not 'so101_left'" in result["error"], result
    assert log.since(mark) == ({"so101_left": 0, "so101_right": 0}, {"so101_left": 0, "so101_right": 0})
    assert rt.held_object == "red cube" and rt._held_arm == "so101_right"
    _assert_one_failed_line(result)


#: so101_mujoco's delivery area (configs/arms/so101_mujoco.yaml): a declared model region.
_MJ_AREA = {"name": "drop zone", "center_xy_m": [0.20, -0.12], "half_extent_xy_m": [0.04, 0.04]}
REGION_TEXT = "region placement / model release is not supported on a base_pose arm"


@pytest.mark.parametrize("declared", ["mj_delivery_area", "mj_release_tool_body", "_model_plan"])
def test_a_region_plan_on_a_base_pose_arm_is_refused(make_runtime, gate_off, monkeypatch, declared):  # noqa: F811
    """R9: the MuJoCo placement and withdrawal observers admit only an
    articulation root at the kinematic origin, so a base_pose arm with a
    region plan, or with a declared model release, refuses place_at before
    any command; the cube stays held. Without a base_pose a stray plan meets
    today's refusal."""
    rt = make_runtime(_world_object_rig())
    assert rt.execute("grasp_object", {"label": "red cube"})["held"] == "red cube"
    raw = raw_arm(rt.arm)
    jaws = []
    real = raw.set_gripper
    monkeypatch.setattr(raw, "set_gripper", lambda *a, **k: (jaws.append(a), real(*a, **k))[1])
    n0 = len(raw.commands)
    if declared == "_model_plan":
        with pytest.raises(Exception, match=re.escape(REGION_TEXT)):
            rt.skill_place_at(PLACE_W[0], PLACE_W[1], _model_plan=object())
    else:
        rt.cfg.arm._data[declared] = _MJ_AREA if declared == "mj_delivery_area" else "gripper"
        result = rt.execute("place_at", {"x": PLACE_W[0], "y": PLACE_W[1]})
        assert result["ok"] is False and REGION_TEXT in result["error"], result
        _assert_one_failed_line(result)
        # Task 5 review Minor 3: the refusal names the way out for the cube in the jaws.
        assert "open_gripper" in result["next_action"] and "reset_scene" in result["next_action"], result
    assert len(raw.commands) == n0 and jaws == []
    assert rt.held_object == "red cube"
    plain = make_runtime(moved_cfg("so101_right", None, move_cameras=False, shift_xy=(0.0, 0.12)))
    assert plain.execute("grasp_object", {"label": "red cube"})["held"] == "red cube"
    with pytest.raises(Exception, match="placement requires its private bound region plan"):
        plain.skill_place_at(*PLACE_A, _model_plan=object())


def test_pick_and_place_with_a_model_region_on_a_base_pose_arm_is_refused_before_the_grasp(make_runtime, gate_off):  # noqa: F811
    """The same refusal starts pick_and_place: it would grasp, then never
    place (R9), and strand the cube in the jaws."""
    rt = make_runtime(_world_object_rig())
    rt.cfg.arm._data["mj_delivery_area"] = _MJ_AREA
    raw = raw_arm(rt.arm)
    n0 = len(raw.commands)
    result = rt.execute("pick_and_place", {"object": "red cube"})
    assert result["ok"] is False and REGION_TEXT in result["error"], result
    assert len(raw.commands) == n0 and not rt.held_object
    _assert_one_failed_line(result)
    assert "open_gripper" in result["next_action"] and "reset_scene" in result["next_action"], result


# ── 7. the holding arm (spec D9, R5; Task 4 review Importants 1-2) ────────


def _in_bound_call(rt, arm, fn):
    """Run `fn()` inside execute()'s bind window for `arm` (None: the
    primary), through list_objects (no motion)."""
    out = []

    def spy():
        out.append(fn())
        return {"ok": True}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(rt, "skill_list_objects", spy)
        rt.execute("list_objects", {} if arm is None else {"arm": arm})
    assert len(out) == 1, "the bound call ran"
    return out[0]


def _interrupted_right_grasp(rt, monkeypatch):
    """so101_right closes on the cube and its lift is refused: a provisional hold."""
    from cascade.types import SkillError

    real = rt._move_carry
    monkeypatch.setattr(rt, "_move_carry", lambda *a, **k: (_ for _ in ()).throw(SkillError("lift refused (test)")))
    result = rt.execute("grasp_object", {"label": "red cube", "arm": "so101_right"})
    monkeypatch.setattr(rt, "_move_carry", real)
    assert result["ok"] is False and rt._held_provisional is not None and not rt.held_object, result
    return rt._held_provisional


@pytest.mark.parametrize("site", ["grasp", "reconcile", "adopt"])
def test_the_holding_arm_is_recorded_at_every_set_site(make_runtime, gate_off, monkeypatch, site):  # noqa: F811
    """R5: `_held_arm` is set wherever `held_object` is: the grasp's promotion,
    `_reconcile_held`'s promotion of a provisional hold, and
    `_adopt_unknown_held`; a place on the other arm is then refused."""
    rt = make_runtime(_pair_world_rig())
    if site == "grasp":
        rt.execute("grasp_object", {"label": "red cube", "arm": "so101_right"})
    elif site == "reconcile":
        _interrupted_right_grasp(rt, monkeypatch)
        _in_bound_call(rt, "so101_right", rt._reconcile_held)
    else:
        rt.execute("close_gripper", {"arm": "so101_right"})                # the jaws stall half open (on "something")
        _in_bound_call(rt, "so101_right", rt._adopt_unknown_held)
    assert rt.held_object and rt._held_arm == "so101_right", (rt.held_object, rt._held_arm)
    result = rt.execute("place_at", {"x": 0.22, "y": 0.30, "arm": "so101_left"})
    assert result["ok"] is False and HELD_TEXT in result["error"], result


def test_a_place_on_the_holding_arm_clears_the_holding_arm(make_runtime, gate_off):  # noqa: F811
    rt = make_runtime(_pair_world_rig())
    rt.execute("grasp_object", {"label": "red cube", "arm": "so101_right"})
    result = rt.execute("place_at", {"x": PLACE_W[0], "y": PLACE_W[1], "arm": "so101_right"})
    assert result.get("ok", True) is True, result
    assert rt.held_object is None and rt._held_arm is None


def test_open_gripper_on_the_other_arm_keeps_the_hold(make_runtime, gate_off, monkeypatch):  # noqa: F811
    """R5: open_gripper on the arm that does not hold the object opens ITS
    jaws only; the held state, the holding arm, the measured width and the
    carry attachment stay. open_gripper on the holding arm releases it."""
    rt = make_runtime(_pair_world_rig())
    rt.execute("grasp_object", {"label": "red cube", "arm": "so101_right"})
    attachment = SimpleNamespace(failure=None, guard=lambda: None, observe=lambda state: None)
    rt._carry_attachment = attachment
    width = rt._held_width_m
    log = _JawLog(rt, monkeypatch)
    mark = log.mark()
    result = rt.execute("open_gripper", {"arm": "so101_left"})
    assert result.get("ok", True) is True, result
    arms, jaws = log.since(mark)
    assert jaws == {"so101_left": 1, "so101_right": 0} and arms["so101_right"] == 0, (arms, jaws)
    assert (rt.held_object, rt._held_width_m) == ("red cube", width)
    assert rt._held_arm == "so101_right"
    assert rt._carry_attachment is attachment
    rt._carry_attachment = None
    rt.execute("open_gripper", {"arm": "so101_right"})
    assert rt.held_object is None and rt._held_arm is None


def test_open_gripper_on_the_other_arm_keeps_a_provisional_hold(make_runtime, gate_off, monkeypatch):  # noqa: F811
    """Task 4 review Important 2: a right grasp interrupted after its close
    leaves a provisional hold; open_gripper on the LEFT arm must not clear it,
    or the right arm's next grasp opens its jaws on the cube (a drop)."""
    rt = make_runtime(_pair_world_rig())
    prov = _interrupted_right_grasp(rt, monkeypatch)
    rt.execute("open_gripper", {"arm": "so101_left"})
    assert rt._held_provisional is prov
    raw = raw_arm(_right(rt))
    stall = float(raw._gripper)                                           # the right jaws, stalled on the cube
    opening = np.sign(raw._grip_open - raw._grip_closed)
    log = _JawLog(rt, monkeypatch)
    rt.execute("grasp_object", {"label": "red cube", "arm": "so101_right"})
    opened = [a for a in log.jaws["so101_right"] if opening * (float(a[0]) - stall) > 1e-6]
    assert opened == [], (stall, log.jaws["so101_right"])                 # never opened on the cube
    assert rt.held_object == "red cube" and rt._held_arm == "so101_right"


def test_reset_scene_while_another_arm_holds_is_refused(make_runtime, gate_off, monkeypatch):  # noqa: F811
    rt = make_runtime(_pair_world_rig())
    rt.execute("grasp_object", {"label": "red cube", "arm": "so101_right"})
    log = _JawLog(rt, monkeypatch)
    mark = log.mark()
    result = rt.execute("reset_scene", {"arm": "so101_left"})
    assert result["ok"] is False and "'red cube' " + HELD_TEXT in result["error"], result
    # Task 5 fix review Minor 2: the default `then` (reset_scene) verbatim
    assert result["error"] == ("'red cube' is held by arm 'so101_right', not 'so101_left': place it with "
                               "arm='so101_right' first (place_at), then reset_scene; no motion was sent"), result
    assert result["stage"] == "release" and result["home_skipped"] is True, result
    assert log.since(mark) == ({"so101_left": 0, "so101_right": 0}, {"so101_left": 0, "so101_right": 0})
    assert rt.held_object == "red cube" and rt._held_arm == "so101_right"
    _assert_one_failed_line(result)


def test_another_arms_closed_jaws_never_clear_the_hold(make_runtime, gate_off, monkeypatch):  # noqa: F811
    """Task 4 review Important 1 (the slip branch): the right arm holds the
    cube; the LEFT jaws shut fully. A left-bound reconcile reads the left jaws:
    it must not decide the cube slipped. The right arm's own shut jaws do."""
    rt = make_runtime(_pair_world_rig())
    rt.execute("grasp_object", {"label": "red cube", "arm": "so101_right"})
    left = raw_arm(rt.arm_rig.get("so101_left"))
    left.object_stop_frac = None
    left.set_gripper(left._grip_closed)                                   # the left jaws read fully shut (0 %)
    assert _in_bound_call(rt, "so101_left", rt._gripper_width_frac) == pytest.approx(0.0, abs=1e-9)
    _in_bound_call(rt, "so101_left", rt._reconcile_held)
    assert rt.held_object == "red cube"                                   # not "slipped" by the left jaws
    _in_bound_call(rt, None, rt._reconcile_held)          # the primary, unselected
    assert rt.held_object == "red cube"
    assert rt._held_arm == "so101_right"
    right = raw_arm(_right(rt))
    right.object_stop_frac = None
    right.set_gripper(right._grip_closed)                                 # the cube slipped out of the right jaws
    _in_bound_call(rt, "so101_right", rt._reconcile_held)
    assert (rt.held_object, rt._held_arm) == (None, None)


def test_a_real_gripper_reset_names_the_arm_whose_close_left_the_provisional(make_runtime, gate_off, monkeypatch):  # noqa: F811
    """Merge follow-up (R5 x #363): reset_scene on a real gripper refuses with
    a provisional hold; when ANOTHER arm's close left it, the refusal names
    that arm (\"held by arm 'so101_right'\") instead of sending the agent to
    open_gripper on the arm it runs on. Nothing moves."""
    rt = make_runtime(_pair_world_rig())
    prov = _interrupted_right_grasp(rt, monkeypatch)
    monkeypatch.setattr(rt, "_drives_real_gripper", lambda: True)
    log = _JawLog(rt, monkeypatch)
    mark = log.mark()
    result = rt.execute("reset_scene", {"arm": "so101_left"})
    assert result["ok"] is False and "held by arm 'so101_right'" in result["error"], result
    assert "open_gripper" not in result["error"].split("Error:")[0] or "arm='so101_right'" in result["error"], result
    assert len(result["error"]) <= 240, len(result["error"])
    assert log.since(mark) == ({"so101_left": 0, "so101_right": 0}, {"so101_left": 0, "so101_right": 0})
    assert rt._held_provisional is prov


def test_an_unreachable_place_quotes_the_table_frame_point(make_runtime, gate_off):  # noqa: F811
    """place_geometry's refusal quotes the TABLE-frame point the agent asked
    for: on the yawed so101_right the world point (0.45, -0.22) is
    B (0.386, -0.163), out of its reach; the text names (0.45, -0.22)."""
    rt = make_runtime(moved_cfg("so101_right", YAWED, move_cameras=False))
    rt.held_object = "red cube"                                           # a hand-set flag: this arm's hold
    raw = raw_arm(rt.arm)
    n0 = len(raw.commands)
    result = rt.execute("place_at", {"x": 0.45, "y": -0.22, "z": 0.05})
    assert result["ok"] is False and "place pose unreachable at" in result["error"], result
    number = r"(-?\d+(?:\.\d+)?)"
    quoted = re.search(rf"unreachable at \[{number}, {number}, {number}\] \(hover \[{number}, {number}, {number}\]",
                       result["error"])
    assert quoted, result["error"]
    values = [float(v) for v in quoted.groups()]
    np.testing.assert_allclose(values[0:2], (0.45, -0.22), rtol=0.0, atol=0.002)
    np.testing.assert_allclose(values[3:5], (0.45, -0.22), rtol=0.0, atol=0.002)
    assert len(raw.commands) == n0


def _kept_column(safe, *, age_s=0.0):
    """A stop column kept for `safe` (stop_retrace.Kept) at its current pose, `age_s` old."""
    from cascade.skills import stop_retrace

    q = np.asarray(safe.get_state().q, dtype=float)
    p = safe.harness.kin.fk(q)[:3, 3]
    return stop_retrace.Kept(arm=safe, raw=safe.raw, column=(p[:2].copy(), 0.05, 0.0), q_pre=q.copy(),
                             hover_z=float(p[2]) + 0.05, z_max=float(p[2]) + 0.055, duration_s=2.0,
                             scene=stop_retrace._scene(safe.harness), kept_at=stop_retrace._clock() - age_s)


# ── 8. merge follow-ups (#359 stop retrace, #369 point_at) ────────────────


def test_a_stop_column_is_forgotten_only_by_its_own_arm(make_runtime, gate_off):  # noqa: F811
    """#359's kept approach column belongs to one arm: another arm's motion
    skill keeps it; the arm's own motion forfeits it; reset_scene (the scene
    changes for every arm) forgets every arm's that still applies to nothing:
    another arm's LIVE column refuses the reset instead (Task 5 review Minor 1,
    tests/test_base_pose_other_arm_hold.py
    `test_reset_scene_is_refused_while_another_arms_stop_column_is_live`)."""
    from cascade.skills import stop_retrace

    rt = make_runtime(_pair_world_rig())
    left, right = rt.arm_rig.get("so101_left"), _right(rt)
    rt._stop_retraces = {id(left): "left column", id(right): "right column"}
    rt.execute("open_gripper", {"arm": "so101_left"})
    assert rt._stop_retraces == {id(right): "right column"}
    rt._stop_retraces[id(left)] = "left column"
    rt.execute("open_gripper", {"arm": "so101_right"})
    assert rt._stop_retraces == {id(left): "left column"}
    rt._stop_retraces[id(right)] = "right column"
    assert stop_retrace.forget(rt, every_arm=True) == 2 and rt._stop_retraces == {}
    # Unpinned (Task 5 review Minor 1): this row used to forget a LIVE right
    # column from a left reset, as intended; the right arm then could no
    # longer retrace up it. Now only a column that no longer applies (here:
    # expired) is forgotten by another arm's reset.
    rt._stop_retraces = {id(right): _kept_column(right, age_s=stop_retrace.MAX_AGE_S + 1.0)}
    result = rt.execute("reset_scene", {"arm": "so101_left"})
    assert result.get("home_skipped") is not True, result                 # it got as far as its home move
    assert rt._stop_retraces == {}, result


def test_a_point_at_refusal_quotes_table_frame_heights(make_runtime, gate_off, monkeypatch):  # noqa: F811
    """#369's ceiling refusal on a RAISED so101_right (base 3 cm above the
    table): the heights it quotes are table-frame (the cube's top 0.05 m,
    the ceiling 0.07 m), not the arm frame's (0.02 m, 0.04 m). Since #378 a
    pose that clears from above is no longer the only one: point_at leans in
    on this rig, so the lean-in finds no pose here and the refusal names both."""
    from cascade.skills import pointing

    monkeypatch.setattr(pointing, "lean_in", lambda *a, **k: pointing.LeanIn(q=None))
    rt = make_runtime(raised_table(moved_cfg("so101_right", RAISED, move_cameras=False, shift_xy=(0.0, -0.10))))
    raw = raw_arm(rt.arm)
    n0 = len(raw.commands)
    result = rt.execute("point_at", {"label": "red cube"})
    assert result["ok"] is False and len(raw.commands) == n0, result
    ceiling = re.search(r"top-down ceiling is (-?\d+\.\d+) m", result["error"])
    top = re.search(r"top (-?\d+\.\d+) m \+ clearance", result["error"])
    need = re.search(r"object top \+ margin \((-?\d+\.\d+) m", result["error"])
    assert ceiling and top and need, result["error"]
    assert float(ceiling.group(1)) == pytest.approx(0.07, abs=1e-3)
    assert float(top.group(1)) == pytest.approx(WORLD_TOP_Z, abs=0.003)
    assert float(need.group(1)) - float(top.group(1)) == pytest.approx(0.03 + 0.015, abs=0.002)


def test_a_bad_close_stall_band_in_a_resolved_arm_view_fails_the_build(monkeypatch, tmp_path):
    """The close on a non-primary reads its RESOLVED arm section (the bound
    profile): a bad band there fails the build before any arm exists, even
    when the arm's top-level entry is fine."""
    from cascade.apps import demo

    cfg = moved_cfg(["so101_left", "so101_right"], {}, move_cameras=False)
    second = cfg._data["arms"][1]
    resolved = second["resolved"]
    resolved = getattr(resolved, "_data", resolved)
    gripper = dict(resolved["arm"].get("gripper") or {})
    gripper["close_stall_band"] = 0.2
    resolved["arm"] = {**resolved["arm"], "gripper": gripper}
    assert (second.get("gripper") or {}).get("close_stall_band") != 0.2       # premise: the entry itself is fine
    monkeypatch.setattr(demo, "make_arm", lambda *a, **k: pytest.fail("arm made"))
    with pytest.raises(ValueError, match="so101_right.*close_stall_band"):
        demo.build_runtime(cfg, tmp_path / "run", lazy_arm=False)


# ── 9. Task 4 review Minors 2 and 4: the running arm on a pair, doubles ──

#: A pair whose PRIMARY (so101_left) has its own shipped base_pose and whose
#: right arm is YAWED: a call bound to so101_right must read so101_right's
#: frame, never the primary's.
PAIR_LEFT = [0.0, 0.22, 0.0, 0.0, 0.0, 0.0]


def _yawed_pair_and_rig_a(make_runtime):  # noqa: F811
    """(pair, rig A, T): the pair's cameras show rig A's scene moved by T_right
    (the cube at T (0.198, 0)); rig A is so101_right alone without a base_pose."""
    from base_pose_oracle import _camera_dicts, _data

    T = transform(YAWED)
    cfg = moved_cfg(["so101_left", "so101_right"], {"so101_right": YAWED, "so101_left": PAIR_LEFT},
                    move_cameras=False)
    for cam in _camera_dicts(cfg._data):
        ext = _data(cam["extrinsics"])
        if ext.get("T") is not None:
            ext["T"] = (T @ np.asarray(ext["T"], dtype=float)).tolist()
    pair, rig_a = make_runtime(cfg), make_runtime(moved_cfg("so101_right", None))
    for rt in (pair, rig_a):
        rt._reobserve(1)
    return pair, rig_a, T


def test_the_observed_gate_reads_the_running_arms_frame_on_a_pair(make_runtime, monkeypatch):  # noqa: F811
    """Gate on, grasp on the pair's YAWED right arm: `for_runtime` runs with
    so101_right's T and Kinematics bound (not the primary's), its frame's
    camera pose is T_right^-1 T_table_cam, and the fix is rig A's."""
    from test_agent_results_compose import _gate_on

    from cascade.grasping import observed_scene

    monkeypatch.setenv("CASCADE_OBSERVED_FINGER_GATE", "1")
    pair, rig_a, T = _yawed_pair_and_rig_a(make_runtime)
    right = pair.arm_rig.get("so101_right")
    gates = {id(rig_a): _gate_on(monkeypatch, raw_arm(rig_a.arm)), id(pair): _gate_on(monkeypatch, raw_arm(right))}
    seen = {id(rig_a): [], id(pair): []}

    def for_runtime(runtime, frame, fix, state):
        seen[id(runtime)].append((np.array(runtime.T_world_base if runtime.T_world_base is not None
                                           else np.eye(4)), runtime.kin, frame, fix))
        return gates[id(runtime)]

    monkeypatch.setattr(observed_scene, "for_runtime", for_runtime)
    l0 = len(raw_arm(pair.arm_rig.get("so101_left")).commands)
    assert rig_a.execute("grasp_object", {"label": "red cube"})["ok"] is True
    assert pair.execute("grasp_object", {"label": "red cube", "arm": "so101_right"})["ok"] is True
    assert seen[id(pair)] and len(seen[id(pair)]) == len(seen[id(rig_a)]), {k: len(v) for k, v in seen.items()}
    for (T_bound, kin, frame, fix), (_, _, _, fix_a) in zip(seen[id(pair)], seen[id(rig_a)]):    # every gate built
        np.testing.assert_allclose(T_bound, T, rtol=0.0, atol=1e-12)
        assert kin is right.harness.kin and kin is not pair.arm_rig.get("so101_left").harness.kin
        np.testing.assert_allclose(frame.T_base_cam, np.linalg.inv(T) @ pair.extrinsics.cam_to_base(),
                                   rtol=0.0, atol=1e-9)
        np.testing.assert_allclose(fix.position, fix_a.position, rtol=0.0, atol=1e-6)
    assert len(raw_arm(pair.arm_rig.get("so101_left")).commands) == l0


def test_the_held_offset_is_the_running_arms_on_a_pair(make_runtime, gate_off, monkeypatch):  # noqa: F811
    """The pair's YAWED right arm grasps: its held offset is rig A's; inside a
    call bound to so101_right the legacy pose channel (a table-frame pose) and
    the image path measure in so101_right's frame, not the primary's."""
    from cascade.skills import held_observation
    from cascade.skills import runtime as runtime_mod

    pair, rig_a, T = _yawed_pair_and_rig_a(make_runtime)
    assert rig_a.execute("grasp_object", {"label": "red cube"})["ok"] is True
    assert pair.execute("grasp_object", {"label": "red cube", "arm": "so101_right"})["ok"] is True
    np.testing.assert_allclose(pair._held_offset, rig_a._held_offset, rtol=0.0, atol=1e-3)
    for rt in (pair, rig_a):
        raw = raw_arm(rt.arm_rig.get("so101_right") if rt.arm_rig is not None else rt.arm)
        monkeypatch.setattr(raw, "state_from_frame", lambda _frame, _raw=raw: _raw.get_state(), raising=False)
    want = np.array([0.01, 0.02, -0.03])

    def measured():
        tcp = pair.kin.fk(pair.arm.get_state().q)[:3, 3]
        point = lift(T, tcp + want)
        pair._object_pose = lambda _label: point.copy()
        legacy = held_observation.measure(pair, runtime_mod.localize_object)
        pair._object_pose = None
        return legacy, held_observation.measure(pair, runtime_mod.localize_object)

    legacy, image = _in_bound_call(pair, "so101_right", measured)
    assert legacy.channel == "legacy_pose", legacy
    np.testing.assert_allclose(legacy.offset, want, rtol=0.0, atol=1e-9)
    image_a = held_observation.measure(rig_a, runtime_mod.localize_object)
    assert image.channel == image_a.channel == "captured_image", (image, image_a)
    np.testing.assert_allclose(image.offset, image_a.offset, rtol=0.0, atol=1e-3)


def _plain_double(T=None):
    """A runtime double that is NOT a SkillRuntime (no `_to_arm`, no
    `_frame_in_arm`): a held object, a legacy table-frame pose channel, a TCP
    at (0.20, 0, 0.10), no camera. `T` (optional) is its `T_world_base`."""
    from cascade.types import RobotState

    def fk(q):
        out = np.eye(4)
        out[:3, 3] = [0.20, 0.0, 0.10]
        return out

    state = RobotState(q=np.zeros(6))
    rt = SimpleNamespace(held_object="red cube", _held_det_label=None, _held_offset=None,
                         _held_observation_floor=None, cfg=SimpleNamespace(arm={}, grasp={}, safety={}),
                         arm=SimpleNamespace(get_state=lambda **kw: state, raw=None, harness=None),
                         kin=SimpleNamespace(fk=fk), detector=None, extrinsics=None,
                         _localization_workspace_bounds=lambda: None,
                         observe=lambda: (_ for _ in ()).throw(RuntimeError("no camera")))
    if T is not None:
        rt.T_world_base = T
    # the object 1 cm / 2 cm / -3 cm from the TCP, as a table-frame pose
    point = [0.21, 0.02, 0.07] if T is None else lift(T, [0.21, 0.02, 0.07]).tolist()
    rt._object_pose = lambda _label: list(point)
    return rt


@pytest.mark.parametrize("T", [None, "yawed"])
def test_a_runtime_double_without_the_arm_frame_measures_as_before(T):
    """Task 4 review Minor 4: `measure` reads the arm frame the way the grasp
    path and `for_runtime` do (`T_world_base` by getattr; None: no transform).
    A double without `_to_arm` keeps its legacy offset instead of silently
    ending up \"rejected\"; one with a `T_world_base` measures in that frame."""
    from cascade.skills import held_observation

    out = held_observation.measure(_plain_double(None if T is None else transform(YAWED)), lambda *a, **k: None)
    assert out.channel == "legacy_pose", out
    np.testing.assert_allclose(out.offset, [0.01, 0.02, -0.03], rtol=0.0, atol=1e-12)


# ── 10. the remaining world-target skills and the viewer frame (Task 6) ─────

#: rig A's snapshot entries moved by this (rig A's frame) before restore_scene,
#: rig B's by R of it: the cube is then displaced from where the snapshot
#: remembers it (the mock camera never shows a moved cube, so a push could
#: not displace it).
_SNAPSHOT_SHIFT = np.array([0.0, -0.06, 0.0])
_SHIFT_SNAPSHOT = ("shift the snapshot", None)


def _no_visitor_sides(cfg):
    """safety.people.visitor_sides: [] (every throw is refused toward a visitor
    side, and every side is one by default): the throw rows compare the skill."""
    from test_people_guard_base_pose import _people

    return _people({"visitor_sides": []})(cfg)


def _throw_rig(cfg):
    """The throw row's rig: no visitor side, and the table 3 cm below the
    arm's base (`raised_table`). At the shipped table_z 0 the SO-101's
    wind-up passes 1.9 cm over the table, under its 2 cm clearance: rig A is
    refused mid-throw and the release would never be compared (#400 review
    Minor 2)."""
    return raised_table(_no_visitor_sides(cfg))


#: Task 6's oracle sequences: (arm, steps, config edit). Each step runs on both
#: rigs and is compared (`run_pair`). move_near_object moves `closer` on
#: `mock`: above / next_to are out of every shipped chain's reach from home
#: (refused by the arm's reach on both rigs). search_for_object's occluder row
#: runs on `mock`: on so101 the park spot's place is refused (place pose
#: unreachable, pre-existing).
ORACLE_T6 = {
    "sort_by_color": ("so101_right", [("sort_by_color", {})], None),
    "search_for_object": ("so101_right", [("search_for_object", {"label": "red cube"})], None),
    "search_for_object, occluder": ("mock", [("search_for_object", {"label": "blue ball"})], None),
    "restore_scene": ("so101_right", [("snapshot_scene", {"name": "scene"}), _SHIFT_SNAPSHOT,
                                      ("restore_scene", {"name": "scene", "max_moves": 1})], None),
    "turn_screw": ("so101_right", [("turn_screw", {"label": "red cube"})], None),
    "throw": ("so101_right", [("throw", {"label": "red cube"})], _throw_rig),
    "handover": ("so101_right", [("handover", {"label": "red cube"})], None),
    "move_near_object": ("mock", [("move_near_object", {"label": "red cube", "relation": "closer",
                                                        "distance_m": 0.03})], None),
    "probe_point": ("so101_right", [("probe_point", {"u": 320, "v": 327})], None),
}
#: Steps that move nothing (they still compare their results).
_LOOKS = {"snapshot_scene", "probe_point", "search_for_object"}
#: Rows whose rig A verdict is not ok by design; their motion completes and is
#: compared: the occluder search never sees the ball (exhausted after lifting
#: the occluder), and turn_screw cannot verify a fastening on the mock stack.
#: Every other step completes on rig A (#400 review Minor 2: a refusal on both
#: rigs would compare nothing the skill does).
_NOT_OK_BY_DESIGN = {"search_for_object, occluder", "turn_screw"}


def _shift_snapshots(rt_a, rt_b, T):
    for rt, d in ((rt_a, _SNAPSHOT_SHIFT), (rt_b, np.asarray(T)[:3, :3] @ _SNAPSHOT_SHIFT)):
        snap = rt.beliefs.get_snapshot("scene")
        assert snap is not None and snap.entries, "snapshot_scene memorized the cube"
        for entry in snap.entries:
            entry["position"] = (np.asarray(entry["position"], dtype=float) + d).tolist()


@pytest.mark.parametrize("pose", [SHIPPED_RIGHT, YAWED], ids=["so101_right", "yawed"])
@pytest.mark.parametrize("seq", sorted(ORACLE_T6))
def test_remaining_skills_are_equivariant(make_pair_rigs, observed_gate, pose, seq):  # noqa: F811
    """Task 6: the arm at base_pose T with the scene moved by T sends rig A's
    commands and reports rig A's results moved by T (main: refused by #352's
    guard). The sort zones and the park spots are the arm's; the snapshot
    targets, beliefs and agent text stay in the table frame."""
    arm, steps, edit = ORACLE_T6[seq]
    rt_a, rt_b, T = make_pair_rigs(pose, arm=arm, edit=edit)
    for rt in (rt_a, rt_b):
        observed_gate(rt)
    for skill, args in steps:
        if (skill, args) == _SHIFT_SNAPSHOT:
            _shift_snapshots(rt_a, rt_b, T)
            continue
        run = run_pair(rt_a, rt_b, T, skill, args)
        if skill not in _LOOKS or seq == "search_for_object, occluder":
            assert run["cmds_a"], f"rig A moved for {skill}: {run['result_a']}"
        assert run["result_a"].get("ok", True) is (seq not in _NOT_OK_BY_DESIGN), (skill, run["result_a"])
        assert_equivariant(run, T)


def test_viewer_directions_follow_the_camera_on_a_yawed_base(make_pair_rigs):  # noqa: F811
    """The viewer frame is the camera's, taken into the arm frame: on a YAWED
    base (the camera moved with it) "right" / "left" as seen on the screen
    send rig A's commands (main: rotated by the 0.4 rad yaw), and the undo
    retraces them. Arm profile `mock` (its relative moves complete at home)."""
    rt_a, rt_b, T = make_pair_rigs(YAWED, arm="mock")
    for skill, args in (("move_relative", {"direction": "right", "frame": "viewer", "distance_m": 0.02}),
                        ("undo_move", {}),
                        ("rotate_gripper", {"direction": "left", "angle_deg": 10, "frame": "viewer"})):
        run = run_pair(rt_a, rt_b, T, skill, args)
        assert run["result_a"].get("ok", True) is True and run["cmds_a"], (skill, run["result_a"])
        assert_equivariant(run, T)


@pytest.mark.parametrize("skill", ["throw", "handover"])
def test_throw_and_handover_on_the_arm_that_does_not_hold_are_refused(make_runtime, gate_off, monkeypatch,  # noqa: F811
                                                                      skill):
    """Review Focus 2 (spec D9 / §5): the right arm holds the cube; throw or
    handover on the LEFT arm would swing the left arm and open its empty jaws
    (the hold is skipped: "already holding"). Refused before any perception or
    command on either arm, naming the holding arm; the right arm still holds it."""
    cfg = _pair_world_rig()
    _no_visitor_sides(cfg)
    rt = make_runtime(cfg)
    assert rt.execute("grasp_object", {"label": "red cube", "arm": "so101_right"})["held"] == "red cube"
    log = _JawLog(rt, monkeypatch)
    mark = log.mark()
    looked = []
    real_localize = rt._localize
    monkeypatch.setattr(rt, "_localize", lambda *a, **k: (looked.append(a), real_localize(*a, **k))[1])
    result = rt.execute(skill, {"label": "red cube", "arm": "so101_left"})
    assert result["ok"] is False, result
    assert "'red cube' " + HELD_TEXT + ", not 'so101_left'; no motion was sent" in result["error"], result
    assert looked == [] and log.since(mark) == ({"so101_left": 0, "so101_right": 0},
                                                {"so101_left": 0, "so101_right": 0})
    assert rt.held_object == "red cube" and rt._held_arm == "so101_right"
    _assert_one_failed_line(result)


def test_turn_screw_on_the_arm_that_does_not_hold_names_the_holding_arm(make_runtime, gate_off,  # noqa: F811
                                                                        monkeypatch):
    """The right arm holds the cube: turn_screw on the LEFT arm is refused
    before any perception or command, naming the holding arm and the way out
    on it (Task 5 review Minor 2's text for every grasp skill), not "already
    holding ...; place it first" (the left arm holds nothing)."""
    rt = make_runtime(_pair_world_rig())
    assert rt.execute("grasp_object", {"label": "red cube", "arm": "so101_right"})["held"] == "red cube"
    log = _JawLog(rt, monkeypatch)
    mark = log.mark()
    looked = []
    real_localize = rt._localize
    monkeypatch.setattr(rt, "_localize", lambda *a, **k: (looked.append(a), real_localize(*a, **k))[1])
    result = rt.execute("turn_screw", {"label": "red cube", "arm": "so101_left"})
    assert result["ok"] is False, result
    assert ("'red cube' " + HELD_TEXT + ", not 'so101_left': place it with arm='so101_right' first (place_at), "
            "then turn_screw; no motion was sent") in result["error"], result
    assert looked == [] and log.since(mark) == ({"so101_left": 0, "so101_right": 0},
                                                {"so101_left": 0, "so101_right": 0})
    assert rt.held_object == "red cube" and rt._held_arm == "so101_right"
    _assert_one_failed_line(result)


def test_probe_point_checks_the_workspace_in_the_arm_frame(make_runtime):  # noqa: F811
    """R10: the probed point is table-frame (the camera's); the configured
    workspace is the running arm's. so101_right (base at W y -0.22) and the
    cube at W (0.198, 0.20): B y 0.42, outside its workspace (|y| <= 0.36),
    though the table-frame numbers read in the arm frame would be inside. The
    gripper distance is measured from the TCP lifted into the table frame."""
    T = transform(SHIPPED_RIGHT)
    rt = make_runtime(moved_cfg("so101_right", SHIPPED_RIGHT, move_cameras=False, shift_xy=(0.0, 0.20)))
    out = rt.execute("probe_point", {"u": 320, "v": 327})
    assert out.get("ok") is True, out
    p = np.asarray(out["position"], dtype=float)
    assert abs(p[1] - 0.20) < 0.03, p                                        # table frame
    lo, hi = (np.asarray(rt.cfg.safety.workspace.get(k), dtype=float) for k in ("min", "max"))
    p_arm = p - np.asarray(T)[:3, 3]
    assert not np.all((lo <= p_arm) & (p_arm <= hi)) and np.all((lo <= p) & (p <= hi)), (p, lo, hi)
    assert out["reachable"]["in_workspace"] is False, out["reachable"]
    tcp = lift(T, rt.kin.fk(rt.arm.get_state().q)[:3, 3])
    np.testing.assert_allclose(out["from_gripper"]["delta_xyz_m"], p - tcp, rtol=0.0, atol=1.1e-3)
    assert out["from_gripper"]["distance_m"] == pytest.approx(float(np.linalg.norm(p - tcp)), abs=1.1e-3)


def _in_arm_frame(T, p):
    """A table-frame point in the frame of the arm at T: T^-1 p."""
    T = np.asarray(T, dtype=float)
    return (np.asarray(p, dtype=float) - T[:3, 3]) @ T[:3, :3]


def test_restore_checks_the_target_in_the_arm_frame(make_runtime, gate_off, monkeypatch):  # noqa: F811
    """restore_scene's reachability pre-check (an IK solve) gets the snapshot
    target (table frame) in the running arm's frame; the move it starts goes
    to the table-frame target (place_at's agent coordinates)."""
    T = transform(YAWED)
    rt = make_runtime(moved_cfg("so101_right", YAWED, move_cameras=False, shift_xy=(0.0, -0.10)))
    checked, placed = [], []
    real_check, real_move = rt._reachable_place_xy, rt._move_object_to
    monkeypatch.setattr(rt, "_reachable_place_xy", lambda x, y: (checked.append((x, y)), real_check(x, y))[1])
    monkeypatch.setattr(rt, "_move_object_to",
                        lambda query, to, **k: (placed.append(list(to)), real_move(query, to, **k))[1])
    with _fusion_held(rt):
        rt._reobserve(1)
        assert rt.execute("snapshot_scene", {"name": "scene"}).get("ok", True) is True
        entry = rt.beliefs.get_snapshot("scene").entries[0]
        target = np.asarray(entry["position"], dtype=float) + np.asarray(T)[:3, :3] @ _SNAPSHOT_SHIFT
        entry["position"] = target.tolist()
        rt.execute("restore_scene", {"name": "scene", "max_moves": 1})
    assert checked and placed, (checked, placed)
    np.testing.assert_allclose(checked[0], _in_arm_frame(T, target)[:2], rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(placed[0], target[:2], rtol=0.0, atol=1e-9)


@pytest.mark.parametrize("pose", [SHIPPED_RIGHT, YAWED, COMBO], ids=["so101_right", "yawed", "combo"])
def test_the_park_spot_avoids_its_neighbours_in_the_arm_frame(make_pair_rigs, pose):  # noqa: F811
    """#400 review Minor 1: `_free_park_spot` takes its neighbours (`avoid`,
    table-frame (xy, radius)) into the arm frame with `around`. A neighbour on
    rig A's first candidate (its first direction at the first offset) moves
    rig A's spot off it; rig B gets the same neighbour moved by T and parks at
    T of rig A's spot. Left in the table frame, rig B's neighbour sits away
    from the spot it should block (arm profile `mock`: every candidate in reach)."""
    rt_a, rt_b, T = make_pair_rigs(pose, arm="mock")
    around = np.asarray(rt_a.beliefs.find("red cube").position, dtype=float)
    first = around + np.array([0.0, 0.20, 0.0])
    assert np.allclose(rt_a._free_park_spot(around[:2], []), first[:2], rtol=0.0, atol=1e-9)
    spot_a = rt_a._free_park_spot(around[:2], [(first[:2], 0.07)])
    spot_b = rt_b._free_park_spot(lift(T, around)[:2], [(lift(T, first)[:2], 0.07)])
    assert spot_a is not None and spot_b is not None, (spot_a, spot_b)
    assert float(np.linalg.norm(np.asarray(spot_a) - first[:2])) > 0.07, (spot_a, first)
    np.testing.assert_allclose(spot_b, lift(T, [float(spot_a[0]), float(spot_a[1]), 0.0])[:2], rtol=0.0, atol=1e-9)


@pytest.mark.parametrize("cube_y", [-0.10, -0.20], ids=["cube at W y -0.10", "cube at the zone's arm numbers"])
def test_sort_by_color_zones_are_the_arms_and_reported_in_the_table_frame(make_runtime, gate_off, cube_y):  # noqa: F811
    """sort_by_color's zones run along the running arm's front edge (its
    drop_zone x, its base frame); they are placed at and reported as table-frame
    points: on so101_right (a pure -0.22 m shift in y) every zone's y is 0.22 m
    lower in the table frame. The "already sorted" check compares in the arm
    frame: a cube at W (0.198, -0.20) (B y 0.02) is NOT in the first zone (B y
    -0.20) and moves, though its table-frame numbers equal the zone's arm-frame
    ones."""
    rt = make_runtime(moved_cfg("so101_right", SHIPPED_RIGHT, move_cameras=False, shift_xy=(0.0, cube_y)))
    result = rt.execute("sort_by_color", {})
    assert result.get("ok", True) is True and result["moved"], result
    # the arm's zones, B (0.20, -0.20 / -0.10 / 0 / 0.10 / 0.20), first colour first, in the table frame
    table_zones = [[0.20, y - 0.22] for y in (-0.20, -0.10, 0.0, 0.10, 0.20)]
    zones = list(result["zones"].values())
    assert zones and np.allclose(zones, table_zones[:len(zones)], rtol=0.0, atol=1e-9), zones
    for m in result["moved"]:
        assert np.allclose(m["zone"], result["zones"][m["color"]], rtol=0.0, atol=1e-9), m
