"""The natural-language motion tools are registered everywhere a motion
tool must be, and nowhere a stop tool must not be.

A tool that moves the arm but is missing from `_MOTION_SKILLS` keeps the
WorldWatcher fusing mid-motion, gets no cancel->e-stop latch in the MCP
server, and -- the reason this file exists -- is NOT denied by OpenShell's
`motion lock` (scripts/openshell_rig.py reads the set from source with
`ast`). These pins name the tools explicitly, so a rename or a removal
from the set fails here rather than silently re-opening motion under an
observe-only lock.
"""

from __future__ import annotations

import importlib.util
import inspect
import re
import string
import sys
from pathlib import Path

import pytest

from cascade.skills import relative_motion as rm
from cascade.skills.runtime import _MOTION_SKILLS, TOOL_SPECS, SkillRuntime

REPO = Path(__file__).resolve().parents[1]
SPATIAL_TOOLS = ("move_relative", "rotate_gripper", "undo_move", "move_near_object")


def _specs() -> dict:
    return {t["name"]: t for t in TOOL_SPECS}


def _openshell():
    spec = importlib.util.spec_from_file_location("openshell_rig", REPO / "scripts" / "openshell_rig.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve their module by name
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("tool", SPATIAL_TOOLS)
def test_every_spatial_tool_is_a_registered_motion_skill_with_a_spec(tool):
    assert tool in _MOTION_SKILLS
    assert callable(getattr(SkillRuntime, f"skill_{tool}", None))
    assert "arm" in _specs()[tool]["parameters"]["properties"]   # injected over _MOTION_SKILLS


def test_motion_lock_denies_every_spatial_tool_and_never_a_stop():
    rig = _openshell()
    assert set(SPATIAL_TOOLS) <= rig.motion_tools()
    deny = set(rig.deny_set(observe_only=True))
    assert set(SPATIAL_TOOLS) <= deny
    assert not deny & {"emergency_stop", "halt_motion"}
    assert not set(SPATIAL_TOOLS) & set(rig.deny_set(observe_only=False))


def test_the_structural_registration_check_sees_the_delegating_skills():
    """test_motion_skill_registration finds motion skills by the arm calls in
    their bodies; these skills delegate to skills/spatial_moves.py, so its
    pattern must recognise that delegation (and move_cartesian), or the
    NEXT delegating motion skill could be left out of _MOTION_SKILLS."""
    sys.path.insert(0, str(REPO / "tests"))
    import test_motion_skill_registration as reg

    for tool in SPATIAL_TOOLS:
        assert reg._MOVES.search(inspect.getsource(getattr(SkillRuntime, f"skill_{tool}"))), tool
    assert reg._MOVES.search("self.arm.move_cartesian(T)")


def test_spec_enums_match_what_the_code_accepts():
    specs = _specs()
    move = specs["move_relative"]["parameters"]["properties"]
    assert set(move["direction"]["enum"]) == set(rm.DIRECTIONS) | {rm.AGAIN}
    assert set(move["amount"]["enum"]) == set(rm.AMOUNTS_M)
    assert set(move["frame"]["enum"]) == set(rm.FRAMES)
    rot = specs["rotate_gripper"]["parameters"]["properties"]
    assert set(rot["direction"]["enum"]) == set(rm.ROTATION_DIRECTIONS) | {rm.AGAIN}
    assert set(rot["amount"]["enum"]) == set(rm.AMOUNTS_DEG)


def test_descriptions_carry_the_safety_guidance_the_model_needs():
    specs = _specs()
    move = specs["move_relative"]["description"]
    assert "undo_move" in move and "NOT undo" in move           # "go back" is not direction=back
    assert "frame=viewer" in move and "ROBOT" in move           # the default frame is stated
    assert "did NOT" in move and "feasible_distance_m" in move  # refusals are relayed
    rot = specs["rotate_gripper"]["description"]
    assert "camera" in rot and "upside down" in rot and "never offer" in rot and "30" in rot
    near = specs["move_near_object"]["description"]
    assert "just_above" in near and "table" in near and "bottom" in near
    assert "go back" in specs["undo_move"]["description"]


def test_object_relative_positioning_is_withheld_without_measured_heights():
    """With plane-cast depth every object's top reads as the table, so
    "above X" would hover INSIDE a tall X: no measured heights, no tool."""
    from cascade.apps.capabilities import (
        CAP_DEPTH_3D,
        CAP_DEPTH_HEIGHTS,
        TOOL_REQUIREMENTS,
    )

    assert set(TOOL_REQUIREMENTS["move_near_object"]) == {CAP_DEPTH_3D, CAP_DEPTH_HEIGHTS}
    for tool in ("move_relative", "rotate_gripper", "undo_move"):
        assert tool not in TOOL_REQUIREMENTS          # no perception needed: never withheld


@pytest.mark.parametrize("tool", SPATIAL_TOOLS)
def test_each_tool_has_a_plausibility_question_over_its_own_arguments(tool):
    from cascade.agent.milestones import _PLAUSIBILITY_QUESTIONS

    question = _PLAUSIBILITY_QUESTIONS[tool]
    fields = {f for _, f, _, _ in string.Formatter().parse(question) if f}
    assert fields <= set(_specs()[tool]["parameters"]["properties"]), fields


@pytest.mark.parametrize("tool", SPATIAL_TOOLS)
def test_spatial_tools_keep_a_grasp_verdict(tool):
    """They can never start a new grasp (no jaw command at all), so a
    verdict about the current hold stays valid while they run."""
    from cascade.apps.wrist_narration import CARRIES_GRASP_VERDICT

    assert tool in CARRIES_GRASP_VERDICT


def test_the_spatial_modules_never_command_the_gripper():
    for module in ("relative_motion.py", "spatial_moves.py"):
        src = (REPO / "src" / "cascade" / "skills" / module).read_text()
        assert not re.search(r"set_gripper|close_gripper|open_gripper|_close_two_stage", src), module
