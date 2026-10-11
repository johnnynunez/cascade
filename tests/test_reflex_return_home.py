"""Reflex grammar: "return home" phrasings are reflexes -- and "home" is the
REST pose.

Ported (in part) from Seeed WRC b70f3a0, where "back to zero" / "return to
home" / "reset" fell through to the LLM loop and burned up to 30 steps on a
one-call command.

Which pose they reach follows the reBot owner's words: "home", "back home",
"back to zero", "pose 0" mean the folded rest pose the robot can be switched
off from (`move_to_rest`); the raised ready pose is the "initial pose"
(`move_home`). Before `move_to_rest` existed, "home" ran `move_home` and the
zero phrasings were deliberately left unmapped (zero is park_q, a different
pose from home_q -- mapping them to move_home would have moved somewhere the
user did not ask for). Now both families land on park_q, and only the
initial / ready / start phrasings reach home_q. "reset" still means
reset_scene (a sim-prop/world reset); stop clearing (reset_stop) is
staff-only and never a reflex.
"""

from __future__ import annotations

import pytest

from cascade.agent.reflex import parse_command


@pytest.mark.parametrize("text", [
    "back home", "go back home", "come back home", "return home",
    "return to home", "back to home", "go back to home", "return to the home position",
])
def test_return_home_phrasings_are_rest_reflexes(text):
    plan = parse_command(text)
    assert plan is not None, text
    assert plan.intent == "rest" and plan.calls == [("move_to_rest", {})]


@pytest.mark.parametrize("text", ["back to zero", "zero position", "return to zero"])
def test_zero_is_the_rest_pose_never_the_raised_home_q(text):
    plan = parse_command(text)
    assert plan is not None and plan.calls == [("move_to_rest", {})], text


def test_the_raised_pose_is_the_initial_pose():
    plan = parse_command("go back to the initial position")
    assert plan is not None and plan.calls == [("move_home", {})]


def test_reset_keeps_its_existing_meaning():
    assert parse_command("reset").intent == "reset_scene"
