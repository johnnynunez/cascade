"""Pigey scene memory as composite skills over the BeliefStore (ROADMAP #11):
`snapshot_scene` (memorize), `restore_scene` (diff -> restore blocker-first)
and `search_for_object` (occlusion search), after github.com/lianegalanti/
Pigey `real/agent-system.md`.

Three layers of tests, each pinning a different claim:

* belief-store snapshots are advisory data (visible objects only, named,
  persisted, dropped by a scene reset);
* the composite logic on a scripted runtime: only displaced objects move,
  blockers first, swap cycles park one object, moves and the task budget
  are bounded, failures and missing objects are reported, never papered over;
* the real mock stack (and the rendered MuJoCo two-prop world when the SO-101
  meshes + offscreen GL are available): every move goes through the normal
  `SafeArm`/harness path and the composite verdict is honest -- UNVERIFIED on
  the static mock camera, physics-confirmed in MuJoCo.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import REPO, needs_pin
from mujoco_gl_probe import probe_offscreen_gl

from cascade.agent.effects import POSTCONDITIONS
from cascade.agent.reflex import parse_command
from cascade.config import Cfg, load_demo_config
from cascade.memory import BeliefStore, EpisodicMemory
from cascade.skills.runtime import _MOTION_SKILLS, TOOL_SPECS, SkillRuntime
from cascade.types import SkillError

MJCF = REPO / "assets" / "mjcf" / "so101" / "scene.xml"
_offscreen_gl = probe_offscreen_gl(MJCF)
needs_gl = pytest.mark.skipif(
    not _offscreen_gl.available,
    reason=f"needs mujoco + SO-101 assets + offscreen GL: {_offscreen_gl.reason}",
)


# ── registration: specs, motion set, postconditions, reflexes ─────────────


def test_pigey_skills_are_registered_with_the_right_motion_and_postcondition_flags():
    specs = {t["name"]: t for t in TOOL_SPECS}
    for name in ("snapshot_scene", "restore_scene", "search_for_object"):
        assert name in specs, f"{name} missing from TOOL_SPECS"
        assert hasattr(SkillRuntime, f"skill_{name}")
    # snapshot moves nothing; restore and search move the arm, so they must
    # pause belief fusion and record a memory frame like every motion skill
    assert "snapshot_scene" not in _MOTION_SKILLS
    assert {"restore_scene", "search_for_object"} <= _MOTION_SKILLS
    assert "arm" in specs["restore_scene"]["parameters"]["properties"]
    assert "arm" in specs["search_for_object"]["parameters"]["properties"]
    assert "arm" not in specs["snapshot_scene"]["parameters"]["properties"]
    # three-state verdicts for the two skills that claim a physical effect
    assert POSTCONDITIONS["restore_scene"] == "restored"
    assert POSTCONDITIONS["search_for_object"] == "searched"
    assert "snapshot_scene" not in POSTCONDITIONS
    # the schema says what the snapshot is NOT
    assert "advisory" in specs["snapshot_scene"]["description"].lower()
    assert "original task" in specs["search_for_object"]["description"].lower()
    from cascade.apps.robot_runtime import _GRIPPER

    assert {"restore_scene", "search_for_object"} <= _GRIPPER  # both pick and place


def test_pigey_phrases_are_reflexes_in_english_and_spanish():
    for text in ("memorize the scene", "snapshot the scene", "remember where everything is",
                 "memoriza la escena", "recuerda la escena"):
        p = parse_command(text)
        assert p is not None and p.calls == [("snapshot_scene", {})], text
    for text in ("restore the scene", "put everything back", "put everything back where it was",
                 "restaura la escena", "pon todo donde estaba"):
        p = parse_command(text)
        assert p is not None and p.calls == [("restore_scene", {})], text
    for text, label in (("find the red cube", "red cube"), ("search for the banana", "banana"),
                        ("look for the blue cube", "blue cube"), ("busca el cubo rojo", "cubo rojo"),
                        ("encuentra la banana", "banana")):
        p = parse_command(text)
        assert p is not None and p.calls == [("search_for_object", {"label": label})], text
    # the existing reflexes keep their meaning
    assert parse_command("reset the scene").calls == [("reset_scene", {})]
    assert parse_command("look around").calls == [("get_observation", {})]
    assert parse_command("find the red cube and then put it in the bowl") is None  # compound -> LLM


# ── belief-store snapshots ────────────────────────────────────────────────


def _store_with(now, *objs):
    bs = BeliefStore()
    for label, color, xyz, age in objs:
        bs.update(label, np.asarray(xyz, float), 0.9, extent=np.array([0.035, 0.035, 0.035]),
                  top_z=xyz[2] + 0.0175, t=now - age, color=color)
    return bs


def test_belief_store_snapshot_is_named_advisory_and_holds_visible_objects_only():
    now = time.monotonic()
    bs = _store_with(now, ("red cube", "red", [0.20, 0.10, 0.0175], 0.1),
                     ("blue cube", "blue", [0.16, 0.16, 0.0175], 0.5),
                     ("bowl", None, [0.30, -0.10, 0.03], 40.0))  # remembered, not visible
    snap = bs.snapshot("layout", now=now)
    assert snap.name == "layout"
    assert sorted(e["label"] for e in snap.entries) == ["blue cube", "red cube"]
    red = next(e for e in snap.entries if e["label"] == "red cube")
    assert red["color"] == "red" and np.allclose(red["position"], [0.20, 0.10, 0.0175])
    assert np.allclose(red["extent"], [0.035, 0.035, 0.035])
    assert bs.get_snapshot("layout") is snap and bs.snapshots() == ["layout"]
    assert bs.get_snapshot("nope") is None
    # taking it again under the same name replaces, not appends
    bs.snapshot("layout", now=now)
    assert bs.snapshots() == ["layout"]
    # the snapshot is a COPY: later fusion must not rewrite the memorized layout
    bs.update("red cube", np.array([0.25, 0.10, 0.0175]), 0.9, color="red", t=now)
    assert np.allclose(red["position"], [0.20, 0.10, 0.0175])
    # a scene reset forgets the layout too: the next visitor must not inherit
    # the previous visitor's targets
    bs.clear()
    assert bs.snapshots() == [] and bs.get_snapshot("layout") is None


def test_belief_store_snapshot_round_trips_through_save_and_load(tmp_path):
    now = time.monotonic()
    bs = _store_with(now, ("red cube", "red", [0.20, 0.10, 0.0175], 0.1))
    bs.snapshot("layout", now=now)
    path = tmp_path / "beliefs.json"
    assert bs.save(path) == 1
    restored = BeliefStore()
    assert restored.load(path) == 1
    snap = restored.get_snapshot("layout")
    assert snap is not None and [e["label"] for e in snap.entries] == ["red cube"]
    assert np.allclose(snap.entries[0]["position"], [0.20, 0.10, 0.0175])
    assert snap.age_s() >= 0.0
    # a file written before snapshots existed still loads
    old = BeliefStore()
    old.update("red cube", np.array([0.2, 0.1, 0.0]), 0.9, color="red")
    old.save(path)
    again = BeliefStore()
    assert again.load(path) == 1 and again.snapshots() == []


# ── scripted runtime: the composite logic without a rig ───────────────────


def _stub(beliefs: BeliefStore, *, bounds=None, place_fails: int = 0, reveal=None):
    """A SkillRuntime whose motion primitives are scripted: grasping removes
    the belief (as the real grasp does), placing re-registers it at the
    target (as the real place_at does). Everything the composite skills
    decide -- what moves, in which order, how far -- happens for real."""
    rt = SkillRuntime.__new__(SkillRuntime)
    rt.beliefs = beliefs
    rt.memory = EpisodicMemory()
    rt.cfg = Cfg({"grasp": {}, "safety": {"table_z": 0.0}, "arm": {}})
    rt.held_object = None
    rt._held_det_label = None
    rt._held_color = None
    rt.effects = None
    rt._task_deadline = None
    rt.watcher = None
    rt.kin = None  # no IK pre-check in the stub: every target is reachable
    log = SimpleNamespace(grasps=[], places=[], homes=0, reobserves=0, place_fails=place_fails)
    rt.log = log

    def grasp(query, material=None, budget_s=None):
        log.grasps.append(query)
        b = beliefs.find(query)
        if b is None:
            return {"ok": False, "held": False, "error": "no detections", "grasp_attempts": 2}
        rt.held_object = query
        rt._held_det_label = b.label
        rt._held_color = b.color
        beliefs.mark_removed(b.label, near=b.position)
        return {"ok": True, "held": query, "grasp_attempts": 1}

    def place_at(x, y, z=None):
        if log.place_fails > 0:
            log.place_fails -= 1
            raise SkillError("place pose unreachable")
        log.places.append((round(float(x), 3), round(float(y), 3)))
        label, color = rt._held_det_label, rt._held_color
        beliefs.update(label, np.array([x, y, 0.0175]), 0.8,
                       extent=np.array([0.035, 0.035, 0.035]), color=color)
        rt.held_object = None
        rt._held_det_label = None
        rt._held_color = None
        return {"placed": label, "at": [round(float(x), 3), round(float(y), 3), 0.05]}

    def move_home(**kw):
        log.homes += 1
        return {"at": "home"}

    def reobserve(frames=2):
        # a detector pass re-confirms what the camera can see (anything
        # seen within the visible horizon); remembered objects stay hidden
        log.reobserves += 1
        now = time.monotonic()
        for b in beliefs.all():
            if now - b.last_seen_t <= 1.5:
                b.last_seen_t = now
        if reveal is not None:
            reveal(log.reobserves)

    rt._grasp_with_persistence = grasp
    rt.skill_place_at = place_at
    rt.skill_move_home = move_home
    rt._reobserve = reobserve
    rt._gripper_width_frac = lambda: 0.5
    rt._max_width = 0.09
    rt._localization_workspace_bounds = lambda: bounds
    return rt


A = [0.20, 0.10, 0.0175]
B = [0.16, -0.16, 0.0175]
C = [0.30, 0.00, 0.0175]
BOUNDS = (np.array([-0.15, -0.36, -0.02]), np.array([0.40, 0.36, 0.40]))


def _two_cubes(now, red_at=A, blue_at=B):
    return _store_with(now, ("red cube", "red", red_at, 0.1), ("blue cube", "blue", blue_at, 0.1))


def test_restore_moves_only_displaced_objects_to_their_snapshot_positions():
    now = time.monotonic()
    bs = _two_cubes(now)
    rt = _stub(bs, bounds=BOUNDS)
    snap = rt.skill_snapshot_scene(name="t")
    assert snap["count"] == 2 and snap["snapshot"] == "t"
    # the red cube is nudged 8 cm; the blue one stays
    bs.find("red cube").position = np.array(A) + np.array([0.0, 0.08, 0.0])
    out = rt.skill_restore_scene(name="t")
    assert out.get("ok", True), out
    assert rt.log.grasps == ["red cube"], rt.log.grasps
    assert rt.log.places == [(0.20, 0.10)]
    assert [m["label"] for m in out["moves"]] == ["red cube"]
    assert out["moves"][0]["kind"] == "restore" and out["moves"][0]["ok"]
    assert np.allclose(out["moves"][0]["to"], [0.20, 0.10], atol=1e-3)
    assert [d["label"] for d in out["displaced"]] == ["red cube"]
    assert [u["label"] for u in out["unchanged"]] == ["blue cube"]
    assert out["still_displaced"] == [] and out["missing"] == []
    assert out["tolerance_m"] == pytest.approx(0.05)
    assert rt.held_object is None and rt.log.homes >= 1
    # the belief store now matches the snapshot
    assert np.allclose(bs.find("red cube").position[:2], A[:2], atol=1e-6)
    # and a second restore has nothing to do and moves nothing
    out2 = rt.skill_restore_scene(name="t")
    assert out2.get("ok", True) and out2["moves"] == [] and len(rt.log.places) == 1


def test_restore_is_blocker_first():
    """Blue sits ON red's snapshot position; red was carried elsewhere. Blue
    must move first (to its own, free, target) or red's place would land on
    it (Pigey: 'restore blocker-first')."""
    now = time.monotonic()
    bs = _two_cubes(now)
    rt = _stub(bs, bounds=BOUNDS)
    rt.skill_snapshot_scene(name="t")
    bs.find("blue cube").position = np.array(A) + np.array([0.01, 0.0, 0.0])  # blocker
    bs.find("red cube").position = np.array(C)
    out = rt.skill_restore_scene(name="t")
    assert out.get("ok", True), out
    assert [m["label"] for m in out["moves"]] == ["blue cube", "red cube"], out["moves"]
    assert [m["kind"] for m in out["moves"]] == ["restore", "restore"]
    assert rt.log.places == [(0.16, -0.16), (0.20, 0.10)]
    assert np.allclose(bs.find("blue cube").position[:2], B[:2]) and np.allclose(bs.find("red cube").position[:2], A[:2])


def test_restore_parks_one_object_to_break_a_swap_cycle():
    """Red and blue swapped places: both targets are blocked by each other,
    so one is parked at a free spot first, then both go home: 3 moves."""
    now = time.monotonic()
    bs = _two_cubes(now)
    rt = _stub(bs, bounds=BOUNDS)
    rt.skill_snapshot_scene(name="t")
    bs.find("red cube").position = np.array(B)
    bs.find("blue cube").position = np.array(A)
    out = rt.skill_restore_scene(name="t")
    assert out.get("ok", True), out
    kinds = [m["kind"] for m in out["moves"]]
    assert kinds == ["park", "restore", "restore"], out["moves"]
    park = np.asarray(out["moves"][0]["to"])
    # the park spot is clear of both targets and inside the workspace
    assert np.linalg.norm(park - np.array(A[:2])) > 0.06 and np.linalg.norm(park - np.array(B[:2])) > 0.06
    assert np.all(park >= BOUNDS[0][:2]) and np.all(park <= BOUNDS[1][:2])
    assert np.allclose(bs.find("red cube").position[:2], A[:2]) and np.allclose(bs.find("blue cube").position[:2], B[:2])
    assert out["still_displaced"] == []


def test_restore_bounds_the_number_of_moves_and_respects_the_task_budget():
    now = time.monotonic()
    bs = _two_cubes(now)
    rt = _stub(bs, bounds=BOUNDS)
    rt.skill_snapshot_scene(name="t")
    bs.find("red cube").position = np.array(C)
    bs.find("blue cube").position = np.array(C) + np.array([0.0, 0.12, 0.0])
    out = rt.skill_restore_scene(name="t", max_moves=1)
    assert out["ok"] is False and len(out["moves"]) == 1 and len(rt.log.places) == 1
    assert "budget" in out["error"] and len(out["still_displaced"]) == 1, out
    # the per-task persistence cap already spent by an earlier tier: no motion at all
    rt2 = _stub(_two_cubes(now), bounds=BOUNDS)
    rt2.skill_snapshot_scene(name="t")
    rt2.beliefs.find("red cube").position = np.array(C)
    rt2.begin_task_budget(seconds=0.0)
    out2 = rt2.skill_restore_scene(name="t")
    assert out2["ok"] is False and out2["moves"] == [] and rt2.log.grasps == []
    assert "task persistence budget" in out2["error"], out2


def test_restore_refuses_without_a_snapshot_or_with_full_hands_and_reports_missing_objects():
    now = time.monotonic()
    rt = _stub(_two_cubes(now), bounds=BOUNDS)
    with pytest.raises(SkillError, match="snapshot_scene"):
        rt.skill_restore_scene(name="never-taken")
    rt.skill_snapshot_scene(name="t")
    rt.held_object = "banana"
    with pytest.raises(SkillError, match="holding"):
        rt.skill_restore_scene(name="t")
    rt.held_object = None
    # an object from the snapshot that is nowhere to be seen is reported, not
    # silently dropped from the goal; the hint names the search skill
    rt.beliefs.mark_removed("blue cube")
    out = rt.skill_restore_scene(name="t")
    assert out["ok"] is False and out["missing"] == ["blue cube"], out
    assert "search_for_object" in out["error"] and out["moves"] == []


def test_restore_stops_when_a_place_fails_and_reports_what_is_in_the_jaws():
    now = time.monotonic()
    bs = _two_cubes(now)
    rt = _stub(bs, bounds=BOUNDS, place_fails=1)
    rt.skill_snapshot_scene(name="t")
    bs.find("red cube").position = np.array(C)
    bs.find("blue cube").position = np.array(C) + np.array([0.0, 0.12, 0.0])
    out = rt.skill_restore_scene(name="t")
    assert out["ok"] is False and out["stage"] == "place", out
    assert out["holding"] is not None and rt.held_object == out["holding"]
    assert len(rt.log.grasps) == 1  # nothing else was attempted with an object in the jaws
    assert "place" in out["error"]


# ── scripted runtime: occlusion search ────────────────────────────────────


def _cluttered(now, with_target=False):
    objs = [("bowl", None, [0.22, 0.00, 0.03], 0.1),
            ("red cube", "red", [0.16, 0.16, 0.0175], 0.1)]
    if with_target:
        objs.append(("green cube", "green", [0.22, 0.00, 0.0175], 0.1))
    bs = _store_with(now, *objs)
    bs.find("bowl").extent = np.array([0.15, 0.15, 0.06])
    return bs


def test_search_returns_without_motion_when_the_object_is_already_visible():
    now = time.monotonic()
    bs = _cluttered(now, with_target=True)

    def reveal(n):  # the fresh pass sees the green cube
        bs.find("green cube").last_seen_t = time.monotonic()

    rt = _stub(bs, bounds=BOUNDS, reveal=reveal)
    out = rt.skill_search_for_object(label="green cube")
    assert out["found"] is True and out["occluders_moved"] == [] and rt.log.grasps == []
    assert np.allclose(out["position"][:2], [0.22, 0.00], atol=1e-3)
    assert out["task_complete"] is False
    assert "original task" in out["next_action"].lower()


def test_search_lifts_the_largest_hollow_occluder_parks_it_in_the_workspace_and_resumes():
    now = time.monotonic()
    bs = _cluttered(now)

    def reveal(n):
        # the target appears only once the bowl has been lifted (pass #2 is
        # the re-perception after the park)
        if n >= 2 and bs.find("green cube") is None:
            bs.update("green cube", np.array([0.22, 0.0, 0.0175]), 0.9, color="green",
                      extent=np.array([0.035, 0.035, 0.035]))

    rt = _stub(bs, bounds=BOUNDS, reveal=reveal)
    out = rt.skill_search_for_object(label="green cube")
    assert out["found"] is True, out
    assert rt.log.grasps == ["bowl"], rt.log.grasps  # hollow + largest first, never the red cube
    assert len(out["occluders_moved"]) == 1
    mv = out["occluders_moved"][0]
    assert mv["label"] == "bowl" and mv["ok"]
    park = np.asarray(mv["to"])
    assert np.linalg.norm(park - np.array([0.22, 0.0])) == pytest.approx(0.20, abs=0.01)
    assert np.all(park >= BOUNDS[0][:2]) and np.all(park <= BOUNDS[1][:2])
    # parked clear of the other object on the table
    assert np.linalg.norm(park - np.array([0.16, 0.16])) > 0.06
    assert np.allclose(out["position"][:2], [0.22, 0.0], atol=1e-3)
    assert out["attempts"] == 1 and out["task_complete"] is False
    assert "original task" in out["next_action"].lower() and "green cube" in out["next_action"]
    assert rt.held_object is None


def test_search_reports_stuck_honestly_when_nothing_reveals_the_target():
    now = time.monotonic()
    bs = _cluttered(now)
    rt = _stub(bs, bounds=BOUNDS)
    out = rt.skill_search_for_object(label="green cube", max_occluders=2)
    assert out["ok"] is False and out["found"] is False and out["stuck"] is True, out
    assert out["status"] == "not_found"
    assert [m["label"] for m in out["occluders_moved"]] == ["bowl", "red cube"]
    assert out["attempts"] == 2 and "green cube" in out["error"]
    assert out["task_complete"] is False and rt.held_object is None
    assert len(rt.log.places) == 2


def test_search_never_lifts_the_remembered_target_itself_and_refuses_with_full_hands():
    now = time.monotonic()
    # only the (remembered, not visible) target is known: nothing to lift
    bs = _store_with(now, ("green cube", "green", [0.22, 0.0, 0.0175], 30.0))
    rt = _stub(bs, bounds=BOUNDS)
    out = rt.skill_search_for_object(label="green cube")
    assert out["ok"] is False and out["found"] is False and out["occluders_moved"] == []
    assert rt.log.grasps == [] and out["attempts"] == 0
    rt.held_object = "bowl"
    with pytest.raises(SkillError, match="holding"):
        rt.skill_search_for_object(label="green cube")


# ── the real mock stack: through SafeArm, honest verdicts ─────────────────


@pytest.fixture
def mock_runtime(demo_cfg, tmp_path):
    from cascade.apps.demo import build_runtime, shutdown_runtime

    runtime, arm = build_runtime(demo_cfg, tmp_path / "run")
    try:
        yield runtime, arm
    finally:
        shutdown_runtime(runtime, arm)


def _wait_for_belief(runtime, label, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        b = runtime.beliefs.find(label)
        if b is not None:
            return b
        time.sleep(0.05)
    raise AssertionError(f"no belief for {label!r}")


def _count_safe_arm_motions(monkeypatch, runtime):
    safe_arm = runtime.arm
    calls = {"move_joints": 0}
    real = safe_arm.move_joints

    def counting(*a, **kw):
        calls["move_joints"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(safe_arm, "move_joints", counting)
    return calls


@needs_pin
def test_snapshot_then_restore_on_the_mock_stack_moves_through_the_harness_and_stays_unverified(
        mock_runtime, monkeypatch):
    runtime, arm = mock_runtime
    arm.object_stop_frac = 0.5  # the jaws stop on something: a held object
    b0 = _wait_for_belief(runtime, "red cube")
    calls = _count_safe_arm_motions(monkeypatch, runtime)
    snap = runtime.execute("snapshot_scene", {"name": "booth"})
    assert snap["ok"] and snap["count"] == 1 and snap["objects"][0]["label"] == "red cube", snap
    assert calls["move_joints"] == 0  # memorizing moves nothing
    assert "postcondition" not in snap
    # memorized layout says the cube belongs 10 cm toward the drop side of
    # where the (static) camera sees it: it is displaced and must come back
    target = np.asarray(b0.position[:2]) + np.array([0.0, -0.10])
    entry = runtime.beliefs.get_snapshot("booth").entries[0]
    entry["position"] = [float(target[0]), float(target[1]), float(entry["position"][2])]
    with runtime.watcher.paused():
        out = runtime.execute("restore_scene", {"name": "booth"})
    assert out["ok"], out
    assert len(out["moves"]) == 1 and out["moves"][0]["kind"] == "restore"
    assert np.allclose(out["moves"][0]["to"], target, atol=1e-3)
    assert calls["move_joints"] > 0, "the restore must drive the arm through SafeArm"
    assert runtime.held_object is None
    # per-move verdict inherited from place_at; composite verdict honest:
    # the static mock camera and the belief the skill itself wrote cannot
    # independently confirm the layout
    assert out["moves"][0]["postcondition"]["skill"] == "place_at"
    pc = out["postcondition"]
    assert pc["kind"] == "restored" and pc["status"] == "unverified", pc
    assert "verification_note" in out
    # traced as one restore_scene row, with a memory frame like every motion skill
    rows = (runtime.trace.run_dir / "trace.jsonl").read_text()
    assert '"skill": "restore_scene"' in rows
    assert any("restore_scene" in f["text"] for f in runtime.memory.memory_frames(8))


@needs_pin
def test_search_on_the_mock_stack_parks_the_only_occluder_inside_the_workspace_and_reports_stuck(
        mock_runtime, monkeypatch):
    runtime, arm = mock_runtime
    arm.object_stop_frac = 0.5
    b0 = _wait_for_belief(runtime, "red cube")
    start = np.asarray(b0.position[:2]).copy()
    calls = _count_safe_arm_motions(monkeypatch, runtime)
    with runtime.watcher.paused():
        out = runtime.execute("search_for_object", {"label": "green ball", "max_occluders": 1})
    assert out["ok"] is False and out["found"] is False and out["stuck"] is True, out
    assert out["status"] == "not_found" and out["task_complete"] is False
    assert len(out["occluders_moved"]) == 1 and out["occluders_moved"][0]["label"] == "red cube"
    park = np.asarray(out["occluders_moved"][0]["to"])
    lim = runtime.arm.harness.limits
    assert np.all(park >= np.asarray(lim.workspace_min)[:2]) and np.all(park <= np.asarray(lim.workspace_max)[:2])
    assert 0.10 <= np.linalg.norm(park - start) <= 0.25
    assert calls["move_joints"] > 0 and runtime.held_object is None
    assert out["postcondition"]["kind"] == "searched"


# ── rendered MuJoCo two-prop world: physics is the judge ──────────────────


@pytest.fixture
def two_prop_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("CASCADE_OCCUPANCY", "0")
    from cascade.apps.demo import build_runtime

    cfg = load_demo_config(camera="mujoco_scene_two", arm="so101_mujoco", llm="mock")
    cfg._data["stream"] = {"enabled": False}
    runtime, arm = build_runtime(cfg, tmp_path / "run")
    yield cfg, runtime, arm
    runtime.camera.close()
    arm.disconnect()


@needs_pin
@needs_gl
def test_snapshot_matches_physics_truth_on_the_two_prop_world(two_prop_runtime):
    cfg, runtime, arm = two_prop_runtime
    world = arm.world
    snap = runtime.execute("snapshot_scene", {"name": "spawn"})
    assert snap["ok"] and snap["count"] == 2, snap
    for o in snap["objects"]:
        truth = np.asarray(world.body_pos(o["label"].replace(" ", "_")))
        assert np.linalg.norm(np.asarray(o["position"])[:2] - truth[:2]) < 0.01, (o, truth)


@needs_pin
@needs_gl
def test_restore_puts_a_physically_displaced_prop_back_and_physics_confirms_it(two_prop_runtime):
    """The Pigey memorize -> LookAway -> restore beat, with physics as the
    judge: pick_and_place carries the red cube ~22 cm to the drop zone
    (physics-confirmed); that layout is memorized; then the world changes
    while the robot is not looking (`reset_props` teleports both props back
    to spawn -- the belief store is NOT told). restore_scene must see the red
    cube at spawn in its fresh look (not trust the remembered drop-zone
    belief), move ONLY the red cube back to its memorized spot through the
    harness, leave the blue cube alone, and its composite verdict must come
    from physics, not from the belief the place itself wrote.

    The memorized layout is the post-pick one on purpose: the two props
    spawn 7 cm apart and the MuJoCo release-escape planner refuses a place
    next to a neighbour that close (it also refuses several free spots, see
    the PR), so a restore to the spawn layout is an honest `holding`
    failure, not a test of the restore logic."""
    cfg, runtime, arm = two_prop_runtime
    world = arm.world
    r = runtime.execute("pick_and_place", {"object": "red cube"})
    assert r["ok"] and (r.get("postcondition") or {}).get("status") == "confirmed", r
    snap = runtime.execute("snapshot_scene", {"name": "after_pick"})
    assert snap["ok"] and snap["count"] == 2, snap
    layout = {n: np.asarray(world.body_pos(n)).copy() for n in world.free_body_names()}
    red_entry = next(o for o in snap["objects"] if o["label"] == "red cube")
    assert np.linalg.norm(np.asarray(red_entry["position"])[:2] - layout["red_cube"][:2]) < 0.01
    # the world changes while the robot is not looking: props back on spawn
    world.reset_props()
    q = arm.get_state().q
    for _ in range(50):
        arm.send_joint_target(q)  # let physics settle the teleported props
    assert np.linalg.norm(np.asarray(world.body_pos("red_cube"))[:2] - layout["red_cube"][:2]) > 0.15
    assert any(b.label == "red cube"
               and np.linalg.norm(np.asarray(b.position)[:2] - layout["red_cube"][:2]) < 0.03
               for b in runtime.beliefs.all()), \
        "the belief store still remembers the drop-zone position: the restore must not trust it"
    out = runtime.execute("restore_scene", {"name": "after_pick"})
    assert out["ok"], out
    assert len(out["moves"]) == 1 and out["moves"][0]["label"] == "red cube", out["moves"]
    assert out["moves"][0]["kind"] == "restore" and out["moves"][0]["ok"]
    assert [d["label"] for d in out["displaced"]] == ["red cube"]
    assert [u["label"] for u in out["unchanged"]] == ["blue cube"]
    red = np.asarray(world.body_pos("red_cube"))
    assert np.linalg.norm(red[:2] - layout["red_cube"][:2]) <= out["tolerance_m"], (red, layout["red_cube"])
    assert red[2] > 0.0
    assert np.linalg.norm(np.asarray(world.body_pos("blue_cube")) - layout["blue_cube"]) < 0.01
    pc = out["postcondition"]
    assert pc["kind"] == "restored" and pc["status"] == "confirmed" and pc["channel"] == "physics", pc
    assert out["moves"][0]["postcondition"]["channel"] == "physics"
    assert runtime.held_object is None
