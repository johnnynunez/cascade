"""A TILTED base_pose refuses the world-target skills; a level one runs them
(spec 2026-10-17 base_pose runtime frame, D4 / D10 / §5; plan Task 6).

Every world-target skill now converts its targets into the running arm's base
frame and its reports back into the table frame (tests/test_base_pose_runtime_
frame.py, the equivariance oracle), so #352's world-target refusal is gone.
The conversion is exact only on a level mount (one `table_z` plane, the
top-down yaw, the reach annuli): an arm whose base_pose has any roll or pitch
refuses the brief's 15 world-target skills (`TILT_REFUSED_CALLS`, frozen as
the plan started) before any work, with one FAILED line; halts, joint-space
and relative skills still run there. Arms without a base_pose (the reBot, the
kitchen arm, every single-arm profile) never reach either.

Ported from the deleted tests/test_base_pose_world_target_guard.py: the
refusal before any work (no perception, no arm or jaw command), the MCP relay
with one FAILED line, a LazyArm never powered, the AST early-return pin, the
joint-space and relative skills still running, the running arm deciding (not
the primary), and the call-graph derivation, widened (Minor 4, R10): every
skill that reaches a world source is in the equivariance oracle or listed as
frame-free with a reason.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from base_pose_oracle import ORACLE_SKILLS, TILTED, YAWED

from cascade.config import load_demo_config

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "cascade"
SKILLS_DIR = SRC / "skills"

#: The brief's 15 world-target skills, one valid call each: the tilt scope,
#: FROZEN as the plan started (spec D4 / §5).
TILT_REFUSED_CALLS = {
    "point_at": {"label": "red cube"},
    "preview_grasp": {"label": "red cube"},
    "grasp_object": {"label": "red cube"},
    "grasp_at_pixel": {"u": 0.5, "v": 0.5, "normalized": True},
    "pick_and_place": {"object": "red cube"},
    "place_at": {"x": 0.25, "y": -0.10, "z": 0.05},
    "place_on_object": {"label": "red cube"},
    "push_object": {"label": "red cube", "direction": "left", "distance_m": 0.05},
    "move_near_object": {"label": "red cube", "relation": "above"},
    "sort_by_color": {},
    "search_for_object": {"label": "red cube"},
    "restore_scene": {"name": "scene"},
    "turn_screw": {"label": "red cube"},
    "throw": {"label": "red cube"},
    "handover": {"label": "red cube"},
}
#: The refusal text of a tilted mount (agent-facing).
TILT = "its base_pose is tilted"
#: #352's withdrawn refusal: no call returns it any more.
OLD_GUARD = "mounted off the table origin"

#: Joint-space and relative skills: never world targets, never refused.
JOINT_AND_RELATIVE_CALLS = {
    "move_home": {},
    "rotate_joint": {"joint": 1, "degrees": 5},
    "wave": {},
    "open_gripper": {},
    "close_gripper": {},
    "move_to_rest": {},
    "move_relative": {"direction": "up", "distance_m": 0.02},
    "rotate_gripper": {"direction": "clockwise", "angle_deg": 10},
    "undo_move": {},
}
#: Of those, the ones that complete with a motion on the mock so101 (the
#: relative nudges refuse on its home pose for a profile reason on main too:
#: "relative_motion.tool_up_axis must not be parallel to the approach").
COMPLETES_ON_MOCK = {"move_home", "rotate_joint", "wave", "open_gripper", "close_gripper", "move_to_rest"}

IDENTITY = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
PITCHED = [0.0, -0.22, 0.0, 0.0, 0.1, 0.0]
TILTED_LEFT = [0.0, 0.22, 0.0, 0.1, 0.0, 0.0]
#: Tilted mounts (TILTED rolls 0.1 rad): rig id -> (arm profiles, base_pose
#: patches, arm= selector or None, the roll/pitch the refusal quotes).
TILTED_RIGS = {
    "tilted": (["so101_right"], {"so101_right": TILTED}, None, "(roll 5.73 deg, pitch 0.00 deg)"),
    "pitched": (["so101_right"], {"so101_right": PITCHED}, None, "(roll 0.00 deg, pitch 5.73 deg)"),
    "pair tilted, arm=so101_right": (["so101_left", "so101_right"], {"so101_right": TILTED}, "so101_right",
                                     "(roll 5.73 deg, pitch 0.00 deg)"),
}
#: Level mounts: rig id -> (arm profiles, base_pose patches, arm= selector or None).
LEVEL_RIGS = {
    "so101_right": (["so101_right"], {}, None),
    "yawed": (["so101_right"], {"so101_right": YAWED}, None),
    "pair, arm=so101_right": (["so101_left", "so101_right"], {}, "so101_right"),
}


def _cfg(arms, poses, *, people=None):
    if len(arms) == 1:
        cfg = load_demo_config(arm=arms[0], camera="mock_small", llm="mock")
    else:
        cfg = load_demo_config(arms=arms, camera="mock_small", llm="mock")
    entries = [cfg._data["arm"], *(cfg._data.get("arms") or [])]
    for name, pose in poses.items():
        hit = [e for e in entries if e.get("name") == name]
        assert hit, name
        for entry in hit:
            entry["base_pose"] = list(pose)
            resolved = entry.get("resolved")
            if resolved is not None:
                arm = (resolved._data if hasattr(resolved, "_data") else resolved).get("arm")
                if isinstance(arm, dict):
                    arm["base_pose"] = list(pose)
    if people is not None:
        from test_people_guard_base_pose import _people

        _people(people)(cfg)
    return cfg


@pytest.fixture
def make_runtime(tmp_path, monkeypatch):
    from cascade.apps.demo import build_runtime, shutdown_runtime

    monkeypatch.setenv("CASCADE_GRASP_BACKEND", "obb")
    built = []

    def make(arms, poses=None, *, lazy=False, people=None):
        cfg = _cfg(arms, poses or {}, people=people)
        rt, arm = build_runtime(cfg, tmp_path / f"run{len(built)}", view=False, serve=False, lazy_arm=lazy)
        built.append((rt, arm))
        return rt

    try:
        yield make
    finally:
        for rt, arm in built:
            shutdown_runtime(rt, arm)


def _raw(safe):
    raw = safe.raw
    return raw.__dict__.get("_arm", raw)


def _safe_arms(rt):
    if rt.arm_rig is not None:
        return list(rt.arm_rig.arms.values())
    return [rt.arm]


class Spy:
    """Every arm command, jaw command and perception call on a runtime."""

    def __init__(self, rt, monkeypatch):
        self.raws = [_raw(a) for a in _safe_arms(rt)]
        for raw in self.raws:
            if hasattr(raw, "object_stop_frac"):
                raw.object_stop_frac = 0.5
        self.n0 = [len(r.commands) for r in self.raws]
        self.jaw = []
        self.perception = []
        for raw in self.raws:
            original = raw.set_gripper

            def jaw(*a, _original=original, **k):
                self.jaw.append(a)
                return _original(*a, **k)

            monkeypatch.setattr(raw, "set_gripper", jaw)
        for name in ("observe", "observe_fresh", "_localize", "_reobserve"):
            original = getattr(rt, name)

            def seen(*a, _original=original, _name=name, **k):
                self.perception.append(_name)
                return _original(*a, **k)

            monkeypatch.setattr(rt, name, seen)
        detector = getattr(rt, "detector", None)
        if detector is not None:
            original_detect = detector.detect

            def detect(*a, **k):
                self.perception.append("detect")
                return original_detect(*a, **k)

            monkeypatch.setattr(detector, "detect", detect)
        if rt.effects is not None:
            original_snapshot = rt.effects.snapshot

            def snapshot(*a, **k):
                self.perception.append("effects.snapshot")
                return original_snapshot(*a, **k)

            monkeypatch.setattr(rt.effects, "snapshot", snapshot)
        # A later-placed guard would take the BEFORE keyframe with observe().
        rt.last_frame = None

    def arm_commands(self):
        return sum(len(r.commands) - n for r, n in zip(self.raws, self.n0))


def _entered(rt, monkeypatch, skill):
    """Replace `skill`'s body with a recorder: the call reached the skill (no
    guard refused it), and nothing else runs."""
    calls = []

    def body(*a, **k):
        calls.append(k)
        return {"ok": True, "entered": skill}

    monkeypatch.setattr(rt, f"skill_{skill}", body)
    return calls


def _assert_tilt_refused(result, skill, angles):
    from cascade.skills.runtime import _REFUSED_GRASP_LEFT_IN_PLACE, FAILED_INSTRUCTION

    assert result["ok"] is False, result
    assert f"{skill} refused on " in result["error"], result
    assert f"{TILT} {angles}; table-frame targets need a level mount; no motion was sent" in result["error"], result
    action = result["next_action"]
    assert action.startswith(FAILED_INSTRUCTION), action
    assert action.count(FAILED_INSTRUCTION) == 1, action
    assert TILT in action  # the FAILED line quotes the whole refusal
    assert "Level its base_pose in the profile" in action
    assert "move_relative" not in action and "rotate_gripper" not in action
    assert "arm_left_in_place" not in result
    assert _REFUSED_GRASP_LEFT_IN_PLACE not in action
    assert "move_home before" not in action


# ── 1. a tilted mount refuses the frozen 15 before any work ───────────────


@pytest.mark.parametrize("skill", sorted(TILT_REFUSED_CALLS))
@pytest.mark.parametrize("rig", sorted(TILTED_RIGS))
def test_a_tilted_mount_refuses_every_world_target_skill_before_any_work(make_runtime, monkeypatch, rig, skill):
    """All 15, before any perception, status line or command, with one FAILED
    line, though every one of them converts on a level mount."""
    arms, poses, select, angles = TILTED_RIGS[rig]
    rt = make_runtime(arms, poses)
    spy = Spy(rt, monkeypatch)
    args = dict(TILT_REFUSED_CALLS[skill])
    if select:
        args["arm"] = select
    _assert_tilt_refused(rt.execute(skill, args), skill, angles)
    assert spy.arm_commands() == 0
    assert spy.jaw == []
    assert spy.perception == []
    assert rt.held_object is None


@pytest.mark.parametrize("skill", sorted(TILT_REFUSED_CALLS))
def test_the_mcp_server_relays_the_tilt_refusal_with_one_failed_line(make_runtime, monkeypatch, skill):
    from cascade.apps.mcp_server import McpSkillServer
    from cascade.skills.runtime import FAILED_INSTRUCTION

    rt = make_runtime(["so101_right"], {"so101_right": TILTED})
    spy = Spy(rt, monkeypatch)
    server = McpSkillServer()
    server._runtime = rt  # bypass the lazy build: the real runtime is already up
    out = server.call_tool(skill, dict(TILT_REFUSED_CALLS[skill]))
    assert out["isError"] is True
    text = next(b["text"] for b in out["content"] if b["type"] == "text")
    assert text.count(FAILED_INSTRUCTION) == 1
    _assert_tilt_refused(json.loads(text), skill, TILTED_RIGS["tilted"][3])
    assert spy.arm_commands() == 0 and spy.jaw == [] and spy.perception == []


@pytest.mark.parametrize("skill", ["move_near_object", "throw", "turn_screw", "grasp_object"])
def test_the_tilt_refusal_never_powers_a_lazy_arm(make_runtime, skill):
    """The MCP server builds every arm lazily (standby until a motion)."""
    rt = make_runtime(["so101_left", "so101_right"], {"so101_left": TILTED_LEFT, "so101_right": TILTED}, lazy=True)
    raws = [a.raw for a in _safe_arms(rt)]
    assert [r.connected for r in raws] == [False, False]
    _assert_tilt_refused(rt.execute(skill, {**TILT_REFUSED_CALLS[skill], "arm": "so101_right"}), skill,
                         "(roll 5.73 deg, pitch 0.00 deg)")
    _assert_tilt_refused(rt.execute(skill, dict(TILT_REFUSED_CALLS[skill])), skill,
                         "(roll 5.73 deg, pitch 0.00 deg)")
    assert [r.connected for r in raws] == [False, False]


# ── 2. halts, joint-space and relative skills keep working there ──────────


@pytest.mark.parametrize("skill", sorted({**JOINT_AND_RELATIVE_CALLS, "halt_motion": {}}))
def test_a_tilted_mount_still_runs_halts_joint_space_and_relative_skills(make_runtime, monkeypatch, skill):
    rt = make_runtime(["so101_right"], {"so101_right": TILTED})
    spy = Spy(rt, monkeypatch)
    result = rt.execute(skill, dict({**JOINT_AND_RELATIVE_CALLS, "halt_motion": {}}[skill]))
    assert TILT not in str(result.get("error")), result
    assert TILT not in str(result.get("next_action"))
    if skill == "halt_motion":
        assert result["ok"] is True, result          # a halt is never deniable
    if skill in COMPLETES_ON_MOCK:
        assert result["ok"] is True, result
        assert spy.arm_commands() > 0 or spy.jaw


# ── 3. a level mount runs every one of them (#352's refusal is gone) ──────


@pytest.mark.parametrize("skill", sorted(TILT_REFUSED_CALLS))
@pytest.mark.parametrize("rig", sorted(LEVEL_RIGS))
def test_a_level_mount_runs_every_world_target_skill(make_runtime, monkeypatch, rig, skill):
    """The skill body is entered: neither the tilt refusal nor #352's
    world-target refusal stops it on a level base_pose (yawed included).
    safety.people.visitor_sides is [] so the throw reaches its body too."""
    arms, poses, select = LEVEL_RIGS[rig]
    rt = make_runtime(arms, poses, people={"visitor_sides": []})
    calls = _entered(rt, monkeypatch, skill)
    args = dict(TILT_REFUSED_CALLS[skill])
    if select:
        args["arm"] = select
    result = rt.execute(skill, args)
    assert TILT not in str(result.get("error")) and OLD_GUARD not in str(result.get("error")), result
    assert calls == [{k: v for k, v in args.items() if k != "arm"}], (calls, result)


def test_the_tilt_scope_is_the_frozen_fifteen():
    """Spec D4 / §5: a tilted mount refuses the brief's 15, frozen when the
    plan started; never a halt, a joint-space or a relative skill. #352's
    world-target refusal and its set are gone."""
    import cascade.skills.runtime as runtime_mod

    assert len(TILT_REFUSED_CALLS) == 15
    assert runtime_mod._TILT_REFUSED_SKILLS == frozenset(TILT_REFUSED_CALLS)
    assert not runtime_mod._TILT_REFUSED_SKILLS & (set(JOINT_AND_RELATIVE_CALLS) | {"halt_motion"})
    assert not runtime_mod._TILT_REFUSED_SKILLS & runtime_mod._FRAME_FREE_ALWAYS   # those may run with no frame
    for gone in ("_WORLD_TARGET_SKILLS", "_mounted_off_table_origin", "_world_target_refusal",
                 "_BASE_POSE_WORLD_TARGET_REFUSAL"):
        assert not hasattr(runtime_mod, gone), gone


#: Near the level tolerance (1e-9, #349's): base_pose -> the angles the
#: refusal quotes, or None when the mount counts as level and the skill runs.
#: A tilt below the 0.01 deg print step is quoted with its own digits, never
#: as "0.00" (a refusal that says "tilted (roll 0.00 deg, pitch 0.00 deg)"
#: contradicts itself).
NEAR_LEVEL = {
    "roll 2e-9": ([0.0, -0.22, 0.0, 2e-9, 0.0, 0.0], "(roll 1.1e-07 deg, pitch 0.00 deg)"),
    "pitch -2e-9": ([0.0, -0.22, 0.0, 0.0, -2e-9, 0.0], "(roll 0.00 deg, pitch -1.1e-07 deg)"),
    "roll 5e-10": ([0.0, -0.22, 0.0, 5e-10, 0.0, 0.0], None),
    "pitch 5e-10": ([0.0, -0.22, 0.0, 0.0, 5e-10, 0.0], None),
}


@pytest.mark.parametrize("rig", sorted(NEAR_LEVEL))
def test_the_tilt_refusal_starts_at_the_level_tolerance(make_runtime, monkeypatch, rig):
    """2e-9 rad of roll or pitch is tilted and refused before any work, with
    honest digits; 5e-10 is level within 1e-9 and the skill runs."""
    pose, angles = NEAR_LEVEL[rig]
    rt = make_runtime(["so101_right"], {"so101_right": pose})
    spy = Spy(rt, monkeypatch)
    result = rt.execute("point_at", dict(TILT_REFUSED_CALLS["point_at"]))
    if angles is not None:
        _assert_tilt_refused(result, "point_at", angles)
        assert spy.arm_commands() == 0 and spy.jaw == [] and spy.perception == []
        return
    assert TILT not in str(result.get("error")), result
    assert "_localize" in spy.perception


def test_the_tilt_text_quotes_a_sub_print_step_angle_with_its_digits():
    """Unit: angles that round to 0.00 but tilt beyond the level tolerance keep
    two significant digits; level axes (within 1e-9) and visible angles keep
    the 2-decimal form (a rounded -0.0 prints as 0.00)."""
    from cascade.skills.runtime import _tilted_mount_refusal

    def quoted(roll, pitch):
        text = _tilted_mount_refusal("grasp_object", None, roll, pitch)["error"]
        return text[text.index("(roll"):text.index(")") + 1]

    assert quoted(5.729577951308232, 0.0) == "(roll 5.73 deg, pitch 0.00 deg)"
    assert quoted(-0.0, -0.004) == "(roll 0.00 deg, pitch -0.004 deg)"
    assert quoted(np.degrees(2e-9), np.degrees(-3e-11)) == "(roll 1.1e-07 deg, pitch 0.00 deg)"
    assert quoted(-0.006, 0.0) == "(roll -0.01 deg, pitch 0.00 deg)"


# ── 4. the RUNNING arm decides (execute(arm=...) included) ────────────────


@pytest.mark.parametrize("skill", ["move_near_object", "search_for_object"])
def test_the_running_arm_decides_not_the_primary(make_runtime, monkeypatch, skill):
    """A pair whose primary is level and whose second arm is tilted: only
    calls that RUN on the second arm are refused; the reverse too."""
    rt = make_runtime(["so101_left", "so101_right"], {"so101_right": TILTED})
    spy = Spy(rt, monkeypatch)
    calls = _entered(rt, monkeypatch, skill)
    _assert_tilt_refused(rt.execute(skill, {**TILT_REFUSED_CALLS[skill], "arm": "so101_right"}), skill,
                         "(roll 5.73 deg, pitch 0.00 deg)")
    assert spy.arm_commands() == 0 and spy.jaw == [] and spy.perception == [] and calls == []
    allowed = rt.execute(skill, dict(TILT_REFUSED_CALLS[skill]))
    assert TILT not in str(allowed.get("error")) and len(calls) == 1, allowed
    reverse = make_runtime(["so101_left", "so101_right"], {"so101_left": TILTED_LEFT})
    calls = _entered(reverse, monkeypatch, skill)
    allowed = reverse.execute(skill, {**TILT_REFUSED_CALLS[skill], "arm": "so101_right"})
    assert TILT not in str(allowed.get("error")) and len(calls) == 1, allowed
    _assert_tilt_refused(reverse.execute(skill, dict(TILT_REFUSED_CALLS[skill])), skill,
                         "(roll 5.73 deg, pitch 0.00 deg)")


def _parse(path):
    return ast.parse(path.read_text(), filename=str(path))


def _execute_skill_body():
    tree = _parse(SKILLS_DIR / "runtime.py")
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SkillRuntime")
    return next(f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == "_execute_skill")


def test_the_tilt_refusal_is_one_early_return_ahead_of_any_work_in_execute():
    """AST pin: one `if <frozen 15> and <the bound frame is not level>: return
    <refusal>`, after the frame is resolved and before anything that reads,
    shows or moves. Without a base_pose `is_level(None)` is True, so nothing
    else changes. #352's early return is gone."""
    fn = _execute_skill_body()
    assert "_mounted_off_table_origin" not in ast.unparse(fn) and "_WORLD_TARGET_SKILLS" not in ast.unparse(fn)
    guards = [s for s in fn.body if isinstance(s, ast.If) and "_TILT_REFUSED_SKILLS" in ast.unparse(s.test)]
    assert len(guards) == 1
    guard = guards[0]
    assert ast.unparse(guard.test) == "name in _TILT_REFUSED_SKILLS and (not arm_frame.is_level(frame.T_world_base))"
    assert len(guard.body) == 1 and isinstance(guard.body[0], ast.Return) and not guard.orelse
    assert ast.unparse(guard.body[0].value) == (
        "_tilted_mount_refusal(name, resolved_arm, *arm_frame.tilt_deg(frame.T_world_base))")
    index = fn.body.index(guard)
    before = "\n".join(ast.unparse(s) for s in fn.body[:index])
    after = "\n".join(ast.unparse(s) for s in fn.body[index + 1:])
    assert "frame = self._resolve_arm_frame(effective_arm, resolved_arm)" in before
    for work in ("drain_contact_reports", "_show_status", "observe(", "save_keyframe",
                 "effects.snapshot", "watcher.paused", "fn(**args)", "ensure_motion_ready"):
        assert work not in before, work
        assert work in after, work


def _shipped_arm_profiles():
    return sorted(p.stem for p in (REPO / "configs" / "arms").glob("*.yaml"))


def test_no_shipped_profile_is_tilted():
    """Walk every shipped arm profile: none is tilted (none meets the tilt
    refusal); the ones that declare a non-identity base_pose are the four
    so101 pair arms, and the live rig's reBot and the kitchen arm declare none."""
    from cascade.apps.demo import _base_transform
    from cascade.skills import arm_frame

    profiles = _shipped_arm_profiles()
    assert {"rebot_rs", "so101_mock", "mock"} <= set(profiles)
    assert any(n.startswith("isaac_kitchen") for n in profiles)
    mounted = set()
    for name in profiles:
        T = _base_transform(load_demo_config(arm=name, camera="mock", llm="mock").arm)
        T = None if T is None or np.array_equal(np.asarray(T, dtype=float), np.eye(4)) else T
        assert arm_frame.is_level(T), name
        if T is not None:
            mounted.add(name)
    assert mounted == {"so101_left", "so101_right", "so101_left_mujoco", "so101_right_mujoco"}


@pytest.mark.parametrize("arm", ["mock", "so101_mock"])
@pytest.mark.parametrize("skill", ["grasp_object", "place_at", "point_at", "preview_grasp"])
def test_an_arm_without_a_base_pose_runs_every_world_target_skill(make_runtime, monkeypatch, arm, skill):
    rt = make_runtime([arm])
    spy = Spy(rt, monkeypatch)
    result = rt.execute(skill, dict(TILT_REFUSED_CALLS[skill]))
    assert TILT not in str(result.get("error")) and OLD_GUARD not in str(result.get("error")), result
    assert spy.perception or spy.arm_commands() > 0


def test_list_arms_no_longer_mentions_base_pose():
    """#352's list_arms sentence described the withdrawn refusal: the text is
    pre-#352's again (42eaddd), and no tool spec mentions base_pose."""
    from cascade.skills.runtime import TOOL_SPECS

    assert [s["name"] for s in TOOL_SPECS if "base_pose" in json.dumps(s)] == []
    text = next(s for s in TOOL_SPECS if s["name"] == "list_arms")["description"]
    assert text == ("List the robot arms available and which one motion commands use by default. Read-only and "
                    "safe to call at any time. Call this before using the `arm` parameter, so you use a real name: "
                    "on a single-arm rig it reports one arm and `arm` can be omitted entirely.")


# ── 5. every skill that reaches a world source is covered ─────────────────

#: Skills the derivation finds that the equivariance oracle does not run, each
#: with why its results need no conversion (or the task that converts them).
FRAME_FREE = {
    "analyze_scene": "Task 7 (R14): its overlay lifts the TCP and the workspace box; its text reads "
                     "table-frame beliefs",
    "annotated_view": "Task 7 (R14, the T3 re-review's in_workspace flag): an overlay image of table-frame "
                      "beliefs",
    "locate_pixel": "projects a table-frame belief through the table-frame camera: no arm geometry",
    "recall_memory": "reports remembered table-frame beliefs as they are: no arm geometry",
    "reset_scene": "a joint-space home and release; beliefs are forgotten and re-observed in the table frame",
}

#: Modules the derivation follows (R10), by the name a caller uses: every
#: skills/ module, the probe, the visual interface and the observed scene.
_FOLLOWED = {
    "observed_scene": SRC / "grasping" / "observed_scene.py",
    "visual_interface": SRC / "perception" / "visual_interface.py",
    "probe": SRC / "perception" / "probe.py",
    **{p.stem: p for p in SKILLS_DIR.glob("*.py")},
}
#: A world source: a perceived position, any belief read, a camera pose, the
#: viewer axes (or x / y parameters: agent world coordinates).
WORLD_SOURCES = frozenset({"_localize", "_belief_pose", "fix_at_pixel", "beliefs", "T_base_cam", "cam_to_base",
                           "extrinsics", "viewer_axes"})


def _derive_world_source_skills(runtime_source: str | None = None) -> dict[str, set[str]]:
    """Every `skill_*` that reaches a world source, through any runtime method,
    function or class of a followed module it calls (a `self`-passing call
    included), with the sources it reaches."""
    parsed: dict[Path, ast.Module] = {}

    def module(stem):
        path = _FOLLOWED.get(stem)
        if path is None or not path.exists():
            return {}, {}
        if path not in parsed:
            parsed[path] = _parse(path)
        body = parsed[path].body
        return ({f.name: f for f in body if isinstance(f, ast.FunctionDef)},
                {c.name: {f.name: f for f in c.body if isinstance(f, ast.FunctionDef)}
                 for c in body if isinstance(c, ast.ClassDef)})

    tree = (ast.parse(runtime_source) if runtime_source is not None else _parse(SKILLS_DIR / "runtime.py"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SkillRuntime")
    methods = {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}

    def edges_to(stem, name):
        funcs, classes = module(stem)
        if name in funcs:
            return {(stem, name)}
        return {(stem, f"{name}.{m}") for m in classes.get(name, ())}

    def refs(fn, mod):
        names, edges, imported = set(), set(), {}
        for node in ast.walk(fn):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[-1] in _FOLLOWED:
                for alias in node.names:
                    imported[alias.asname or alias.name] = (node.module.split(".")[-1], alias.name)
        for node in ast.walk(fn):
            if isinstance(node, ast.Attribute):
                names.add(node.attr)
                if node.attr in methods:
                    edges.add(("rt", node.attr))
                if isinstance(node.value, ast.Name):
                    edges |= edges_to(node.value.id, node.attr)
            elif isinstance(node, ast.Name):
                names.add(node.id)
                if mod is not None:
                    edges |= edges_to(mod, node.id)
                if node.id in imported:
                    edges |= edges_to(*imported[node.id])
        return names, edges

    def reach(start):
        seen, stack, names = set(), [start], set()
        while stack:
            key = stack.pop()
            if key in seen:
                continue
            seen.add(key)
            kind, name = key
            if kind == "rt":
                fn = methods[name]
            else:
                funcs, classes = module(kind)
                fn = classes[name.split(".")[0]][name.split(".")[1]] if "." in name else funcs[name]
            n, e = refs(fn, None if kind == "rt" else kind)
            names |= n
            stack.extend(e)
        return names

    out = {}
    for method, fn in methods.items():
        if method.startswith("skill_"):
            hit = reach(("rt", method)) & WORLD_SOURCES
            if {"x", "y"} <= {a.arg for a in fn.args.args}:
                hit.add("x/y")
            if hit:
                out[method.removeprefix("skill_")] = hit
    return out


def test_every_skill_reaching_a_world_source_is_covered():
    """Minor 4 / R10: the widened call-graph derivation (beliefs, camera poses,
    the viewer axes; probe.py, visual_interface.py and observed_scene.py
    followed) finds exactly the oracle's skills and the listed frame-free ones,
    each with a reason; every one of the frozen 15 is in the oracle."""
    derived = set(_derive_world_source_skills())
    assert derived == set(ORACLE_SKILLS) | set(FRAME_FREE), (sorted(derived - set(ORACLE_SKILLS) - set(FRAME_FREE)),
                                                             sorted((set(ORACLE_SKILLS) | set(FRAME_FREE)) - derived))
    assert not set(ORACLE_SKILLS) & set(FRAME_FREE)
    assert all(isinstance(why, str) and why.strip() for why in FRAME_FREE.values())
    assert set(TILT_REFUSED_CALLS) <= set(ORACLE_SKILLS)


def test_the_derivation_sees_a_new_belief_or_camera_reader():
    """The rule is not vacuous: a skill reading the belief store or a frame's
    camera pose through a helper counts, a joint-space one does not."""
    source = (SKILLS_DIR / "runtime.py").read_text()
    extra = ("\n    def skill_poke(self, label):\n        return self._poke_helper(label)\n\n"
             "    def _poke_helper(self, label):\n        return self.beliefs.find(label)\n\n"
             "    def skill_peek(self):\n        return self.last_frame.T_base_cam\n\n"
             "    def skill_nod(self):\n        return self._move_joints(None)\n")
    anchor = "    def skill_wave(self, cycles: int = 2) -> dict:\n"
    assert source.count(anchor) == 1
    derived = _derive_world_source_skills(source.replace(anchor, extra + anchor))
    assert derived["poke"] == {"beliefs"} and derived["peek"] == {"T_base_cam"}
    assert "nod" not in derived
