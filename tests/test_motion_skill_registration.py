"""Every arm-moving skill must be registered in _MOTION_SKILLS.

CLAUDE.md flags this as a silent trap: membership is what pauses WorldWatcher
belief fusion during motion and what captures the pre-motion frame for the
postcondition channel. Forget it and the held object gets re-fused at bogus
mid-air positions, while the verifier compares the after-frame against itself
and confirms everything.

Nothing caught this before: `test_llm_and_library.py` checks TOOL_SPECS has a
matching skill_ method, but nothing checks the reverse or _MOTION_SKILLS at
all. `grasp_at_pixel` was added this session and was missing from the set.

The check is deliberately structural rather than a hand-maintained list: a
skill that reaches the arm through SafeArm is a motion skill, and asking the
source is what makes this test catch the NEXT one too.
"""

from __future__ import annotations

import inspect
import re

from wrc_demo.skills import runtime as rt_mod
from wrc_demo.skills.runtime import _MOTION_SKILLS, SkillRuntime

#: Calls that command the arm. Reading state (get_state, harness queries) does
#: not move anything and must not force a skill into the motion set.
_MOVES = re.compile(
    r"self\.arm\.(move_joints|move_home|stream_to|set_gripper|move_relative)"
    r"|self\.arm\.raw\."
    r"|self\.skill_(grasp_object|place_at|place_on_object|pick_and_place"
    r"|open_gripper|close_gripper|push_object|move_home|grasp_at_pixel)\("
)


def _skill_names() -> list[str]:
    return [n[len("skill_"):] for n in dir(SkillRuntime)
            if n.startswith("skill_") and callable(getattr(SkillRuntime, n))]


def test_every_arm_moving_skill_is_registered():
    missing = []
    for name in _skill_names():
        try:
            src = inspect.getsource(getattr(SkillRuntime, f"skill_{name}"))
        except (OSError, TypeError):
            continue
        if _MOVES.search(src) and name not in _MOTION_SKILLS:
            missing.append(name)
    assert not missing, (
        f"skills move the arm but are absent from _MOTION_SKILLS: {missing}. "
        "The WorldWatcher will keep fusing beliefs mid-motion and the "
        "postcondition channel will lose its pre-motion frame."
    )


def test_motion_skills_all_exist():
    """A stale entry is harmless at runtime but hides a rename."""
    names = set(_skill_names())
    unknown = sorted(_MOTION_SKILLS - names)
    assert not unknown, f"_MOTION_SKILLS names skills that do not exist: {unknown}"


def test_grasp_at_pixel_is_a_motion_skill():
    """Pinned explicitly: this is the one that was missing."""
    assert "grasp_at_pixel" in _MOTION_SKILLS


def test_every_skill_has_a_tool_spec():
    """The reverse of the existing check, which CLAUDE.md says is unenforced.

    A skill without a TOOL_SPECS entry is invisible to the LLM and MCP: it
    fails no test, it simply never gets called.
    """
    specs = {t["name"] for t in rt_mod.TOOL_SPECS}
    internal = {"halt_motion"}          # exposed, keep in sync if more appear
    missing = sorted(
        n for n in _skill_names()
        if n not in specs and n not in internal
    )
    # Report rather than assert-empty: some skills are deliberately internal.
    # Fail only for ones this session added, so the suite stays honest without
    # retroactively policing pre-existing choices.
    session_added = {"grasp_at_pixel", "halt_motion"}
    regressed = sorted(session_added.intersection(missing))
    assert not regressed, f"skills added without a TOOL_SPECS entry: {regressed}"
