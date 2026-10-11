"""`move_to_rest` is registered everywhere a motion tool must be, and the
words route to the right pose in every tier.

Routing (the reBot owner's convention): "home", "go home", "pose 0", "zero",
"rest", "park", "fold" and "so I can turn it off" mean the folded rest pose
(`move_to_rest`); "initial / ready / start pose" means the raised pose
(`move_home`). The LLM tier routes by the tool descriptions, the reflex tier
(dashboard chat, the offline brain) by the grammar in agent/reflex.py; both
are pinned here so they cannot drift apart.
"""

from __future__ import annotations

import importlib.util
import inspect
import string
import sys
from pathlib import Path

import pytest

from cascade.agent.reflex import parse_command
from cascade.skills.runtime import _MOTION_SKILLS, TOOL_SPECS, SkillRuntime

REPO = Path(__file__).resolve().parents[1]
TOOL = "move_to_rest"


def _specs() -> dict:
    return {t["name"]: t for t in TOOL_SPECS}


def _openshell():
    spec = importlib.util.spec_from_file_location("openshell_rig", REPO / "scripts" / "openshell_rig.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── registration ────────────────────────────────────────────────────────────


def test_it_is_a_registered_motion_skill_with_a_spec():
    assert TOOL in _MOTION_SKILLS
    assert callable(getattr(SkillRuntime, f"skill_{TOOL}", None))
    props = _specs()[TOOL]["parameters"]["properties"]
    assert "arm" in props                                  # injected over _MOTION_SKILLS
    assert props["keep_torque"]["type"] == "boolean"
    assert _specs()[TOOL]["parameters"]["required"] == []


def test_the_motion_lock_denies_it_and_never_a_stop():
    rig = _openshell()
    assert TOOL in rig.motion_tools()
    deny = set(rig.deny_set(observe_only=True))
    assert TOOL in deny
    assert not deny & {"emergency_stop", "halt_motion"}
    assert TOOL not in set(rig.deny_set(observe_only=False))


def test_the_structural_registration_check_sees_it():
    sys.path.insert(0, str(REPO / "tests"))
    import test_motion_skill_registration as reg

    assert reg._MOVES.search(inspect.getsource(getattr(SkillRuntime, f"skill_{TOOL}")))


def test_it_is_never_withheld_and_drops_a_grasp_verdict():
    """It needs no perception (never withheld by the capability matrix), and
    it closes the jaws before the motors go off, so a verdict about a hold
    must not survive it (it is absent from CARRIES_GRASP_VERDICT)."""
    from cascade.apps.capabilities import TOOL_REQUIREMENTS
    from cascade.apps.wrist_narration import CARRIES_GRASP_VERDICT

    assert TOOL not in TOOL_REQUIREMENTS
    assert TOOL not in CARRIES_GRASP_VERDICT


def test_it_has_a_plausibility_question_over_its_own_arguments():
    from cascade.agent.milestones import _PLAUSIBILITY_QUESTIONS

    question = _PLAUSIBILITY_QUESTIONS[TOOL]
    fields = {f for _, f, _, _ in string.Formatter().parse(question) if f}
    assert fields <= set(_specs()[TOOL]["parameters"]["properties"]), fields
    assert "EMPTY" in question


# ── the LLM tier: descriptions ──────────────────────────────────────────────


def test_the_rest_description_claims_home_pose_zero_and_powering_off():
    desc = _specs()[TOOL]["description"]
    for phrase in ("'home'", "'go home'", "'pose 0'", "'zero pose'", "'rest'", "'park'",
                   "'fold", "turn it off", "safe_to_power_off", "move_home",
                   "put it down", "keep_torque"):
        assert phrase in desc, phrase
    assert "NOT the raised" in desc


def test_move_home_is_the_initial_pose_and_sends_home_requests_to_rest():
    desc = _specs()["move_home"]["description"]
    for phrase in ("INITIAL", "'initial pose'", "'ready pose'", "raised", "motors on"):
        assert phrase in desc, phrase
    assert "move_to_rest" in desc
    for phrase in ("'home'", "'go home'", "'pose 0'", "'rest'"):
        assert phrase in desc, phrase
    head = desc.split("move_to_rest")[0]
    assert "NOT" in head, "move_home must say the home/pose 0/rest words are NOT for it"


def test_relative_motion_names_both_poses_for_an_ambiguous_return():
    from cascade.skills import relative_motion as rm
    from cascade.types import SkillError

    with pytest.raises(SkillError) as exc:
        rm.normalize_direction("return")
    assert "move_home" in str(exc.value) and "move_to_rest" in str(exc.value)


# ── the reflex tier: grammar ────────────────────────────────────────────────


@pytest.mark.parametrize("text", [
    "home", "go home", "move home", "back home", "go back home", "return home",
    "return to the home position", "go to the home pose",
    "park", "park the arm", "rest", "go to rest", "rest pose", "go to the rest position",
    "pose 0", "pose zero", "zero pose", "zero position", "back to zero", "return to zero",
    "go to pose 0", "fold", "fold the arm", "fold up",
    "vuelve a home", "ve a home", "vuelve a casa", "a casa", "pose cero",
    "vuelve a la pose 0", "posición cero", "posicion de reposo", "reposo", "vuelve al reposo",
    "aparca", "aparca el brazo", "pliega el brazo", "vuelve al origen",
    "please go home", "can you go to pose 0?",
])
def test_home_pose0_and_rest_phrases_are_rest_reflexes(text):
    plan = parse_command(text)
    assert plan is not None, text
    assert plan.intent == "rest" and plan.calls == [(TOOL, {})], (text, plan)


@pytest.mark.parametrize("text", [
    "initial pose", "go to the initial pose", "back to the initial position",
    "ready pose", "go to the ready position", "start pose", "starting position",
    "vuelve a la pose inicial", "pose inicial", "posición inicial", "ve a la posicion inicial",
    "vuelve al inicio", "regresa al inicio",
])
def test_initial_pose_phrases_stay_move_home(text):
    plan = parse_command(text)
    assert plan is not None, text
    assert plan.intent == "initial_pose" and plan.calls == [("move_home", {})], (text, plan)


def test_reset_keeps_its_existing_meaning():
    assert parse_command("reset").intent == "reset_scene"
