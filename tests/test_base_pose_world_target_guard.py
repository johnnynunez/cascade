"""World-target skills refuse, fail closed, on an arm mounted with a base_pose,
until their path converts (spec 2026-10-17 base_pose runtime frame).

Perception, beliefs and agent coordinates are in the shared TABLE frame; the
skill runtime solves IK in the RUNNING arm's base frame. On a base_pose arm
(so101_left / so101_right / *_mujoco, y = +-0.22) grasp_object and place_at
used to land 22 cm off and still report ok, and the grasp exemption column sat
where the arm really descends, so the map gate ignored whatever really stood
there. execute() refuses every skill that turns a perceived or world position
into an arm target on such an arm, before any work (no perception call, no arm
or jaw command, no move_home hint), until the task that converts its path
removes it from `_WORLD_TARGET_SKILLS` (`CONVERTED_CALLS` below run there).
Joint-space and relative skills keep working there. Arms without a base_pose
(the reBot, the kitchen arm, every single-arm profile) or with the identity
pose are exactly main's path.

A TILTED base_pose (any roll or pitch) refuses the brief's 15 world-target
skills (`TILT_REFUSED_CALLS`, frozen: the converted ones included, spec D4 /
§5), before any work and with one FAILED line; halts, joint-space and relative
skills still run there.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from base_pose_oracle import ORACLE_SKILLS, TILTED

from cascade.config import load_demo_config

REPO = Path(__file__).resolve().parents[1]
SKILLS_DIR = REPO / "src" / "cascade" / "skills"

#: The refusal text the brief fixes (agent-facing, relayed to the user).
GUARD = "this arm is mounted off the table origin (base_pose); world targets are not converted yet"

#: One valid call per world-target skill still refused (the brief's list,
#: verified against the call graph in
#: `test_the_guarded_list_is_every_skill_that_aims_at_a_world_position`).
WORLD_TARGET_CALLS = {
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
#: World-target skills whose path converts (spec 2026-10-17 base_pose runtime
#: frame): they left the guard in the task that proved they reach the world
#: point (tests/test_base_pose_runtime_frame.py) and run on a base_pose arm.
CONVERTED_CALLS = {
    "point_at": {"label": "red cube"},          # Task 3
    "preview_grasp": {"label": "red cube"},     # Task 3
    "grasp_object": {"label": "red cube"},      # Task 4
    "grasp_at_pixel": {"u": 0.5, "v": 0.5, "normalized": True},     # Task 4
}

#: The tilt refusal's scope (spec D4, §5): the brief's 15 world-target skills,
#: FROZEN as the plan started. It never shrinks with `_WORLD_TARGET_SKILLS`:
#: a tilted base_pose refuses all 15 even after every one has converted.
TILT_REFUSED_CALLS = {**WORLD_TARGET_CALLS, **CONVERTED_CALLS}
#: The refusal text of a tilted mount (agent-facing).
TILT = "its base_pose is tilted"

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

YAWED = [0.03, -0.22, 0.0, 0.0, 0.0, 0.4]
IDENTITY = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

#: rig id -> (arm profiles, base_pose patches by arm name, arm= selector or None)
RIGS = {
    "so101_right": (["so101_right"], {}, None),
    "yawed": (["so101_right"], {"so101_right": YAWED}, None),
    "pair, arm=so101_right": (["so101_left", "so101_right"], {}, "so101_right"),
}
#: Tilted mounts (TILTED rolls 0.1 rad): rig id -> (arm profiles, base_pose
#: patches, arm= selector or None, the roll/pitch the refusal quotes).
PITCHED = [0.0, -0.22, 0.0, 0.0, 0.1, 0.0]
TILTED_RIGS = {
    "tilted": (["so101_right"], {"so101_right": TILTED}, None, "(roll 5.73 deg, pitch 0.00 deg)"),
    "pitched": (["so101_right"], {"so101_right": PITCHED}, None, "(roll 0.00 deg, pitch 5.73 deg)"),
    "pair tilted, arm=so101_right": (["so101_left", "so101_right"], {"so101_right": TILTED}, "so101_right",
                                     "(roll 5.73 deg, pitch 0.00 deg)"),
}


def _cfg(arms, poses):
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
    return cfg


@pytest.fixture
def make_runtime(tmp_path, monkeypatch):
    from cascade.apps.demo import build_runtime, shutdown_runtime

    monkeypatch.setenv("CASCADE_GRASP_BACKEND", "obb")
    built = []

    def make(arms, poses=None, *, lazy=False):
        cfg = _cfg(arms, poses or {})
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


def _assert_refused(result, skill):
    from cascade.skills.runtime import _REFUSED_GRASP_LEFT_IN_PLACE, FAILED_INSTRUCTION

    assert result["ok"] is False, result
    assert GUARD in result["error"], result
    assert skill in result["error"]
    action = result["next_action"]
    assert action.startswith(FAILED_INSTRUCTION), action
    assert action.count(FAILED_INSTRUCTION) == 1, action
    assert GUARD in action  # the FAILED line quotes the whole refusal
    assert "arm_left_in_place" not in result
    assert _REFUSED_GRASP_LEFT_IN_PLACE not in action
    assert "move_home before" not in action


def _assert_tilt_refused(result, skill, angles):
    from cascade.skills.runtime import _REFUSED_GRASP_LEFT_IN_PLACE, FAILED_INSTRUCTION

    assert result["ok"] is False, result
    assert f"{skill} refused on " in result["error"], result
    assert f"{TILT} {angles}; table-frame targets need a level mount; no motion was sent" in result["error"], result
    assert GUARD not in result["error"]
    action = result["next_action"]
    assert action.startswith(FAILED_INSTRUCTION), action
    assert action.count(FAILED_INSTRUCTION) == 1, action
    assert TILT in action  # the FAILED line quotes the whole refusal
    assert "Level its base_pose in the profile" in action
    assert "move_relative" not in action and "rotate_gripper" not in action
    assert "arm_left_in_place" not in result
    assert _REFUSED_GRASP_LEFT_IN_PLACE not in action
    assert "move_home before" not in action


# ── 1. every world-target skill refuses before any work ───────────────────


@pytest.mark.parametrize("skill", sorted(WORLD_TARGET_CALLS))
@pytest.mark.parametrize("rig", sorted(RIGS))
def test_every_world_target_skill_refuses_on_a_base_pose_arm_before_any_work(make_runtime, monkeypatch, rig, skill):
    arms, poses, select = RIGS[rig]
    rt = make_runtime(arms, poses)
    spy = Spy(rt, monkeypatch)
    args = dict(WORLD_TARGET_CALLS[skill])
    if select:
        args["arm"] = select
    result = rt.execute(skill, args)
    _assert_refused(result, skill)
    assert spy.arm_commands() == 0
    assert spy.jaw == []
    assert spy.perception == []
    assert rt.held_object is None


@pytest.mark.parametrize("skill", sorted(WORLD_TARGET_CALLS))
def test_the_mcp_server_relays_the_refusal_with_one_failed_line(make_runtime, monkeypatch, skill):
    from cascade.apps.mcp_server import McpSkillServer
    from cascade.skills.runtime import FAILED_INSTRUCTION

    rt = make_runtime(["so101_left", "so101_right"])
    spy = Spy(rt, monkeypatch)
    server = McpSkillServer()
    server._runtime = rt  # bypass the lazy build: the real runtime is already up
    out = server.call_tool(skill, dict(WORLD_TARGET_CALLS[skill]))
    assert out["isError"] is True
    text = next(b["text"] for b in out["content"] if b["type"] == "text")
    assert text.count(FAILED_INSTRUCTION) == 1
    _assert_refused(json.loads(text), skill)
    assert spy.arm_commands() == 0 and spy.jaw == [] and spy.perception == []


@pytest.mark.parametrize("skill", ["place_at", "throw", "push_object"])
def test_the_refusal_never_powers_a_lazy_arm(make_runtime, skill):
    """The MCP server builds every arm lazily (standby until a motion)."""
    rt = make_runtime(["so101_left", "so101_right"], lazy=True)
    raws = [a.raw for a in _safe_arms(rt)]
    assert [r.connected for r in raws] == [False, False]
    _assert_refused(rt.execute(skill, {**WORLD_TARGET_CALLS[skill], "arm": "so101_right"}), skill)
    _assert_refused(rt.execute(skill, dict(WORLD_TARGET_CALLS[skill])), skill)
    assert [r.connected for r in raws] == [False, False]


# ── 2. joint-space and relative skills keep working there ─────────────────


@pytest.mark.parametrize("skill", sorted(JOINT_AND_RELATIVE_CALLS))
@pytest.mark.parametrize("rig", sorted(RIGS))
def test_joint_space_and_relative_skills_still_run_on_a_base_pose_arm(make_runtime, monkeypatch, rig, skill):
    arms, poses, select = RIGS[rig]
    rt = make_runtime(arms, poses)
    spy = Spy(rt, monkeypatch)
    args = dict(JOINT_AND_RELATIVE_CALLS[skill])
    if select:
        args["arm"] = select
    result = rt.execute(skill, args)
    assert GUARD not in str(result.get("error")), result
    assert GUARD not in str(result.get("next_action"))
    if skill in COMPLETES_ON_MOCK:
        assert result["ok"] is True, result
        assert spy.arm_commands() > 0 or spy.jaw


@pytest.mark.parametrize("skill", sorted(CONVERTED_CALLS))
@pytest.mark.parametrize("rig", sorted(RIGS) + sorted(TILTED_RIGS))
def test_a_converted_skill_runs_on_a_base_pose_arm(make_runtime, monkeypatch, rig, skill):
    """A level mount runs it; a TILTED one still refuses it before any work
    (the tilt scope is the frozen 15, spec D4 / §5)."""
    arms, poses, select, *tilt = RIGS[rig] if rig in RIGS else TILTED_RIGS[rig]
    rt = make_runtime(arms, poses)
    spy = Spy(rt, monkeypatch)
    args = dict(CONVERTED_CALLS[skill])
    if select:
        args["arm"] = select
    result = rt.execute(skill, args)
    assert GUARD not in str(result.get("error")), result
    if tilt:
        _assert_tilt_refused(result, skill, tilt[0])
        assert spy.arm_commands() == 0 and spy.jaw == [] and spy.perception == []
        return
    assert TILT not in str(result.get("error")), result
    # it ran: perceived (the object here is W (0.198, 0), 22 cm across from
    # so101_right's base, where a pointing pose or a grasp may honestly be
    # refused); grasp_at_pixel reads the camera frame itself
    assert {"_localize", "observe"} & set(spy.perception), spy.perception


# ── 2b. a tilted mount refuses the frozen 15, and only those ──────────────


@pytest.mark.parametrize("skill", sorted(TILT_REFUSED_CALLS))
@pytest.mark.parametrize("rig", sorted(TILTED_RIGS) + ["tilted, every skill converted"])
def test_a_tilted_mount_refuses_every_world_target_skill_before_any_work(make_runtime, monkeypatch, rig, skill):
    """All 15, before any perception, status line or command, with one FAILED
    line. \"every skill converted\" is Task 6's end state (`_WORLD_TARGET_SKILLS`
    empty): the tilt scope does not shrink with it."""
    import cascade.skills.runtime as runtime_mod

    if rig == "tilted, every skill converted":
        monkeypatch.setattr(runtime_mod, "_WORLD_TARGET_SKILLS", frozenset())
        rig = "tilted"
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


def test_the_tilt_scope_is_the_frozen_fifteen():
    """Spec D4 / §5: a tilted mount refuses the brief's 15, frozen when the
    plan started, whatever `_WORLD_TARGET_SKILLS` still holds; never a halt."""
    import cascade.skills.runtime as runtime_mod

    assert len(TILT_REFUSED_CALLS) == 15
    assert runtime_mod._TILT_REFUSED_SKILLS == frozenset(TILT_REFUSED_CALLS)
    assert runtime_mod._WORLD_TARGET_SKILLS <= runtime_mod._TILT_REFUSED_SKILLS
    assert not runtime_mod._TILT_REFUSED_SKILLS & (set(JOINT_AND_RELATIVE_CALLS) | {"halt_motion"})
    assert not runtime_mod._TILT_REFUSED_SKILLS & runtime_mod._FRAME_FREE_ALWAYS   # those may run with no frame


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
    honest digits; 5e-10 is level within 1e-9 and the converted skill runs."""
    pose, angles = NEAR_LEVEL[rig]
    rt = make_runtime(["so101_right"], {"so101_right": pose})
    spy = Spy(rt, monkeypatch)
    result = rt.execute("point_at", dict(CONVERTED_CALLS["point_at"]))
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


# ── 3. the RUNNING arm decides (execute(arm=...) included) ────────────────


@pytest.mark.parametrize("skill", ["place_at", "push_object"])
def test_the_running_arm_decides_not_the_primary(make_runtime, monkeypatch, skill):
    """A pair whose primary has the identity pose and whose second arm has a
    real base_pose: only calls that RUN on the second arm are refused."""
    rt = make_runtime(["so101_left", "so101_right"], {"so101_left": IDENTITY})
    spy = Spy(rt, monkeypatch)
    refused = rt.execute(skill, {**WORLD_TARGET_CALLS[skill], "arm": "so101_right"})
    _assert_refused(refused, skill)
    assert spy.arm_commands() == 0 and spy.jaw == [] and spy.perception == []
    allowed = rt.execute(skill, dict(WORLD_TARGET_CALLS[skill]))
    assert GUARD not in str(allowed.get("error")), allowed
    assert spy.perception, "the primary's call ran its skill"


@pytest.mark.parametrize("skill", ["place_at", "push_object"])
def test_a_selected_arm_without_a_base_pose_runs_beside_a_mounted_primary(make_runtime, monkeypatch, skill):
    rt = make_runtime(["so101_left", "so101_right"], {"so101_right": IDENTITY})
    spy = Spy(rt, monkeypatch)
    allowed = rt.execute(skill, {**WORLD_TARGET_CALLS[skill], "arm": "so101_right"})
    assert GUARD not in str(allowed.get("error")), allowed
    assert spy.perception, "the selected arm's call ran its skill"
    _assert_refused(rt.execute(skill, dict(WORLD_TARGET_CALLS[skill])), skill)


def test_a_grasp_on_the_identity_primary_still_moves_the_arm(make_runtime, monkeypatch):
    """The identity pose is main's path: the grasp really runs (commands)."""
    rt = make_runtime(["so101_left", "so101_right"], {"so101_left": IDENTITY})
    spy = Spy(rt, monkeypatch)
    result = rt.execute("place_at", dict(WORLD_TARGET_CALLS["place_at"]))
    assert GUARD not in str(result.get("error")), result
    assert spy.arm_commands() > 0


# ── 4. the guarded list is exactly the skills that aim at world positions ─


def _parse(path):
    return ast.parse(path.read_text(), filename=str(path))


def _derive_world_target_skills():
    """Every `skill_*` that turns a perceived position (`_localize`,
    `_belief_pose`, `fix_at_pixel`, through any helper or skills/ module it
    delegates to) or agent world coordinates (x, y parameters) into an arm
    target (a motion skill, or one that plans grasps)."""
    from cascade.skills.runtime import _MOTION_SKILLS

    tree = _parse(SKILLS_DIR / "runtime.py")
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SkillRuntime")
    methods = {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}
    modules: dict[str, dict] = {}

    def module_functions(mod):
        if mod not in modules:
            path = SKILLS_DIR / f"{mod}.py"
            modules[mod] = ({f.name: f for f in _parse(path).body if isinstance(f, ast.FunctionDef)}
                            if path.exists() else {})
        return modules[mod]

    def refs(fn, owner, mod=None):
        names, edges = set(), set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == owner:
                    names.add(node.attr)
                    if node.attr in methods:
                        edges.add(("rt", node.attr))
                elif node.attr in module_functions(node.value.id):
                    edges.add((node.value.id, node.attr))
            elif isinstance(node, ast.Name):
                names.add(node.id)
                if mod and node.id in module_functions(mod):
                    edges.add((mod, node.id))
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
                n, e = refs(methods[name], "self")
            else:
                fn = module_functions(kind)[name]
                n, e = refs(fn, fn.args.args[0].arg if fn.args.args else None, kind)
            names |= n
            stack.extend(e)
        return names

    perceived = {"_localize", "_belief_pose", "fix_at_pixel"}
    out = set()
    for method, fn in methods.items():
        if not method.startswith("skill_"):
            continue
        skill = method.removeprefix("skill_")
        names = reach(("rt", method))
        params = {a.arg for a in fn.args.args}
        aims = skill in _MOTION_SKILLS or "_plan_grasps" in names
        if aims and (names & perceived or {"x", "y"} <= params):
            out.add(skill)
    return out


def test_the_guarded_list_is_every_skill_that_aims_at_a_world_position():
    """The call graph still finds the brief's 15; each is either still refused
    or converted (and then pinned by the equivariance oracle)."""
    import cascade.skills.runtime as runtime_mod

    derived = _derive_world_target_skills()
    assert derived == set(WORLD_TARGET_CALLS) | set(CONVERTED_CALLS)  # the brief's 15
    assert not set(WORLD_TARGET_CALLS) & set(CONVERTED_CALLS)
    guarded = set(getattr(runtime_mod, "_WORLD_TARGET_SKILLS", ()))
    assert guarded <= derived
    assert guarded == set(WORLD_TARGET_CALLS)
    assert set(CONVERTED_CALLS) <= ORACLE_SKILLS      # a skill leaves the guard only with oracle rows


def test_the_derivation_sees_a_new_world_target_skill(monkeypatch):
    """The call-graph rule is not vacuous: a skill reaching `_localize`
    through a helper counts, a joint-space one does not."""
    import cascade.skills.runtime as runtime_mod

    source = (SKILLS_DIR / "runtime.py").read_text()
    extra = ("\n    def skill_poke(self, label):\n        return self._poke_helper(label)\n\n"
             "    def _poke_helper(self, label):\n        return self._localize(label)\n\n"
             "    def skill_nod(self):\n        return self._move_joints(None)\n")
    anchor = "    def skill_wave(self, cycles: int = 2) -> dict:\n"
    assert source.count(anchor) == 1
    patched = source.replace(anchor, extra + anchor)
    real_read = Path.read_text

    def read_text(self, *a, **k):
        return patched if self == SKILLS_DIR / "runtime.py" else real_read(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(runtime_mod, "_MOTION_SKILLS", runtime_mod._MOTION_SKILLS | {"poke", "nod"})
    derived = _derive_world_target_skills()
    assert "poke" in derived and "nod" not in derived


# ── 5. arms without a base_pose: the guard is a no-op ─────────────────────


class _Harness:
    def __init__(self, pose):
        self.base_pose = None if pose is None else np.asarray(pose, dtype=float)


class _Arm:
    def __init__(self, harness):
        self.harness = harness


def _decide(arm):
    from cascade.skills.runtime import _mounted_off_table_origin

    return _mounted_off_table_origin(arm)


def test_the_guard_reads_only_a_real_non_identity_base_pose():
    from cascade.types import pose_to_transform

    assert _decide(None) is False
    assert _decide(object()) is False                       # no harness at all
    assert _decide(_Arm(None)) is False                     # harness-less SafeArm double
    assert _decide(_Arm(_Harness(None))) is False           # every single-arm profile
    assert _decide(_Arm(_Harness(np.eye(4)))) is False      # identity
    assert _decide(_Arm(_Harness(pose_to_transform(IDENTITY)))) is False
    # A Mock arm / harness answers every attribute: only a real instance
    # attribute counts (registry read, like `_arm_harnesses`).
    assert _decide(mock.Mock()) is False
    assert _decide(_Arm(mock.Mock())) is False
    # anything else is mounted off the origin: fail closed
    assert _decide(_Arm(_Harness(pose_to_transform([0, -0.22, 0, 0, 0, 0])))) is True
    assert _decide(_Arm(_Harness(pose_to_transform(YAWED)))) is True
    assert _decide(_Arm(_Harness(pose_to_transform([0, 0, 0, 0, 0, 1e-6])))) is True
    garbled = _Harness(None)
    garbled.base_pose = "not a matrix"
    assert _decide(_Arm(garbled)) is True
    shape = _Harness(None)
    shape.base_pose = np.eye(3)
    assert _decide(_Arm(shape)) is True


def test_the_guard_never_materializes_a_lazy_arm():
    """The MCP server builds every arm lazily: deciding must not power one."""
    from cascade.control.lazy_arm import LazyArm
    from cascade.safety.harness import SafeArm
    from cascade.types import pose_to_transform

    made = []
    for pose, expected in ((None, False), (pose_to_transform([0, -0.22, 0, 0, 0, 0]), True)):
        lazy = LazyArm(lambda: made.append(1))
        assert _decide(lazy) is False
        assert _decide(SafeArm(lazy, _Harness(pose))) is expected
        assert lazy.connected is False
    assert made == []


def test_the_guard_is_one_early_return_ahead_of_any_work_in_execute():
    """AST pin of the no-op claim: in `_execute_skill` the guard is one
    `if <world-target skill> and <mounted>: return <refusal>` placed right
    after the arm is resolved and before anything that reads, shows or moves
    (contact-report drain, profile checks, status line, keyframes, the
    pre-state snapshot, the watcher pause, the skill). With the predicate
    False nothing else in the dispatch changes."""
    tree = _parse(SKILLS_DIR / "runtime.py")
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SkillRuntime")
    fn = next(f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == "_execute_skill")
    guards = [s for s in fn.body if isinstance(s, ast.If)
              and "_mounted_off_table_origin" in ast.unparse(s.test)]
    assert len(guards) == 1
    guard = guards[0]
    assert ast.unparse(guard.test) == "name in _WORLD_TARGET_SKILLS and _mounted_off_table_origin(effective_arm)"
    assert len(guard.body) == 1 and isinstance(guard.body[0], ast.Return) and not guard.orelse
    assert ast.unparse(guard.body[0].value) == "_world_target_refusal(name, resolved_arm)"
    index = fn.body.index(guard)
    before = "\n".join(ast.unparse(s) for s in fn.body[:index])
    after = "\n".join(ast.unparse(s) for s in fn.body[index + 1:])
    assert "effective_arm =" in before and "resolved_arm =" in before
    for work in ("drain_contact_reports", "_show_status", "observe(", "save_keyframe",
                 "effects.snapshot", "watcher.paused", "fn(**args)", "ensure_motion_ready"):
        assert work not in before, work
        assert work in after, work


def test_the_tilt_refusal_is_one_early_return_ahead_of_any_work_in_execute():
    """The same AST pin for the tilt refusal: one `if <frozen 15> and <the
    bound frame is not level>: return <refusal>`, after the frame is
    resolved and before anything that reads, shows or moves. Without a
    base_pose `is_level(None)` is True, so nothing else changes."""
    tree = _parse(SKILLS_DIR / "runtime.py")
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SkillRuntime")
    fn = next(f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == "_execute_skill")
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


def test_the_guard_engages_exactly_on_the_profiles_that_declare_a_base_pose():
    """Walk every shipped arm profile: the guard's decision equals "declares a
    non-identity base_pose"; the live rig's reBot and the kitchen arm are not
    among them."""
    from cascade.apps.demo import _base_transform

    profiles = _shipped_arm_profiles()
    assert {"rebot_rs", "so101_mock", "mock"} <= set(profiles)
    assert any(n.startswith("isaac_kitchen") for n in profiles)
    engaged = {name for name in profiles
               if _decide(_Arm(_Harness(_base_transform(load_demo_config(arm=name, camera="mock", llm="mock").arm))))}
    assert engaged == {"so101_left", "so101_right", "so101_left_mujoco", "so101_right_mujoco"}


@pytest.mark.parametrize("arm", ["mock", "so101_mock"])
@pytest.mark.parametrize("skill", ["grasp_object", "place_at", "point_at", "preview_grasp"])
def test_an_arm_without_a_base_pose_runs_every_world_target_skill(make_runtime, monkeypatch, arm, skill):
    rt = make_runtime([arm])
    spy = Spy(rt, monkeypatch)
    result = rt.execute(skill, dict({**WORLD_TARGET_CALLS, **CONVERTED_CALLS}[skill]))
    assert GUARD not in str(result.get("error")), result
    assert spy.perception or spy.arm_commands() > 0


def test_the_agent_is_told_once_in_the_tool_specs():
    """One sentence, in the one spec that is about choosing an arm."""
    from cascade.skills.runtime import TOOL_SPECS

    mentions = [s["name"] for s in TOOL_SPECS if "base_pose" in json.dumps(s)]
    assert mentions == ["list_arms"]
    text = next(s for s in TOOL_SPECS if s["name"] == "list_arms")["description"]
    assert "mounted off the table origin" in text and "refuses" in text
    assert "joint-space and relative skills still run" in text
