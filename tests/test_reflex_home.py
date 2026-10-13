"""Explicit home commands retain upstream's deterministic rest dispatch.

Profiles without a folded rest pose, including FR3, delegate move_to_rest to
their saved move_home pose without releasing torque or opening the gripper.
The Franka OpenClaw command router independently selects move_home directly.
"""

import pytest

from cascade.agent.reflex import parse_command


@pytest.mark.parametrize("text", [
    "return home",
    "return to home",
    "return to home position",
    "return to the home position",
    "Please return to home position.",
    "Can you return home?",
    "RETURN  TO   HOME POSITION",
    "go home",
    "move home",
    "home",
    "park the arm",
    "vuelve a casa",
])
def test_return_home_commands_dispatch_profile_aware_rest(text):
    plan = parse_command(text)
    assert plan is not None
    assert plan.intent == "rest"
    assert plan.calls == [("move_to_rest", {})]


@pytest.mark.parametrize("text", [
    "do not return home",
    "how do I return to home position",
    "the robot should return home after the pick",
    "return the cube home",
    "return to home position when finished",
    "return home and then open the gripper",
    "return to home position or stay here",
])
def test_home_mentions_and_conditional_commands_are_not_home_reflexes(text):
    assert parse_command(text) is None
