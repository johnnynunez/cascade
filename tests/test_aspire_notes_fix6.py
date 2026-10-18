"""ASPIRE A1 follow-up (A2 fix round 1, item 9): the stop guard also reads `stopping`,
`stopped` and `stoppage`.

Fix round 5 narrowed the stop guard from `stop*` to the exact words "stop" / "stops" so
"Take off the stopper" and "Switch the stopwatch off" are served. That also served advice
that switches an operator stop off through its other forms ("Disable stopping on contact",
"Skip the stopping distance"). A note never discourages a stop: the three exact words join
the guard words; the kitchen nouns stopper / stopwatch stay served, and so do plain
sentences that only report a stop ("If the arm stopped near the lemon, look again").
"""
from __future__ import annotations

import pytest
from test_aspire_notes_fix5 import STOP_NOUNS, _promoted

from cascade.skills import notes
from cascade.skills.notes import lint_advice, load_snapshot

SKILL = "pick_and_place"


def test_the_stop_guard_is_five_exact_words():
    assert [p for p in notes._GUARDS if p[0].startswith("stop")] == [
        ("stop",), ("stops",), ("stopping",), ("stopped",), ("stoppage",)]
    assert not any(p[0].startswith("stop") and p[0].endswith("*") for p in notes._GUARDS)


#: served on b02b3d5 (fix round 5's risk probe, plus the other two forms)
DISCOURAGE_A_STOP = [
    "Disable stopping on contact.",
    "Disable the arm's stopping.",
    "Skip the stopping distance.",
    "Turn stopping off near the sink.",
    "Switch stopping off near the lemon.",
    "Skip stopping near the sink.",
    "Disable the stopped state.",
    "Disable the stoppage.",
    "Skip the stoppage near the sink.",
    "Turn the stoppage off.",
    "Remove the stoppage.",
]


@pytest.mark.parametrize("text", DISCOURAGE_A_STOP)
def test_advice_that_switches_a_stop_off_is_refused_in_every_form(text):
    assert lint_advice(text, skill=SKILL), text


@pytest.mark.parametrize("profile_class", ["sim", "real"])
def test_stop_form_notes_are_never_served(tmp_path, profile_class):
    for i, text in enumerate(DISCOURAGE_A_STOP):
        _promoted(tmp_path, f"adv-{i:02d}", title="Cube near the lemon", advice=text)
    s = load_snapshot(tmp_path, profile_class)
    assert s.served == () and len(s.skipped) == len(DISCOURAGE_A_STOP)


#: sentences that only report a stop: no verb switches it off. Refused since the A8 fix round
#: (rule stop_workaround: every form of "stop" in a note; the owner's rule, accepted over-block)
REPORT_A_STOP = [
    "If the arm stopped near the lemon, look again before the next pick.",
    "The arm stopped because the cube was too close to the lemon.",
    "After stopping, look at the cube again.",
    "A stoppage near the sink means the cube is too close; pick from the left.",
]


@pytest.mark.parametrize("text", STOP_NOUNS)
def test_the_stop_nouns_are_still_served(text):
    assert lint_advice(text, skill=SKILL) == [], text


@pytest.mark.parametrize("text", REPORT_A_STOP)
def test_a_reported_stop_is_refused_since_the_a8_fix(text):
    # A8 fix (Important 2): a note never names a stop; reporting one is the accepted over-block
    assert any("works around an operator stop" in p for p in lint_advice(text, skill=SKILL)), text
