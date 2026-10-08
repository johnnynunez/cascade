"""Reflex grammar: unambiguous "return home" phrasings go straight to move_home.

Ported (in part) from Seeed WRC b70f3a0, where "back to zero" / "return to
home" / "reset" fell through to the LLM loop and burned up to 30 steps on a
one-call command. Only the phrasings that unambiguously mean the HOME pose
are ported:

- "back to zero" / "zero position" are NOT home: on this arm zero is the
  mechanical stop the shutdown park drives to (park_q), a different pose --
  mapping them to move_home would move somewhere the user did not ask for.
- "reset" already means reset_scene in cascade (a sim-prop/world reset), and
  stop clearing (reset_stop) is staff-only; neither may be shadowed by home.
"""

from __future__ import annotations

import pytest

from cascade.agent.reflex import parse_command


@pytest.mark.parametrize("text", [
    "back home", "go back home", "come back home", "return home",
    "return to home", "back to home", "go back to home", "return to the home position",
])
def test_return_home_phrasings_are_reflexes(text):
    plan = parse_command(text)
    assert plan is not None, text
    assert plan.intent == "home" and plan.calls == [("move_home", {})]


@pytest.mark.parametrize("text", ["back to zero", "zero position", "return to zero"])
def test_zero_is_not_silently_mapped_to_home(text):
    plan = parse_command(text)
    assert plan is None or plan.intent != "home", text


def test_reset_keeps_its_existing_meaning():
    assert parse_command("reset").intent == "reset_scene"
