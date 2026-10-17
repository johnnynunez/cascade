"""ASPIRE A1 fix round 5 (final re-review deleg_cdbffaed): three narrow lint fixes.

1. (Important 1) Common prepositions the head-noun rule did not know ("of", "during",
   "without", "about", "like" and the rest the reviewer listed) end a noun phrase, so the
   guard word before them is the object's HEAD: "Remove the safety check of the arm" is
   refused. They join `_OBJECT_TAIL` only, never `_PREPOSITIONS` (a preposition stops the
   object scan, which would release "Skip part of the check").
2. (Minor 1) "pre-set" / "pre-sets" join the whole-word allowlist (like "preset"); the
   other hyphen compounds that carry a stem stay refused (follow-up list).
3. (Minor 2) The stop guard is the word "stop" / "stops" only: "Take off the stopper",
   "Switch the stopwatch off" are kitchen advice. "guard" stays a guard word (accepted
   over-block: "Take off the blade guard").
"""
from __future__ import annotations

import json

import pytest

from cascade.skills import notes
from cascade.skills.notes import lint_advice, load_snapshot

SKILL = "pick_and_place"
VERIFIED = {"verdict": "verified", "passes": 5, "seeds": [21, 22, 23, 24, 25], "attempts": 1,
            "library_sha": "0" * 64, "engine": "newton"}


def _rules(text, skill=SKILL):
    return [rule for rule, _ in notes._lint_field("advice", text, notes.ADVICE_MAX_CHARS, skill)]


def _title_rules(text, skill=SKILL):
    return [rule for rule, _ in notes._lint_field("title", text, notes.TITLE_MAX_CHARS, skill)]


def _promoted(root, note_id, *, title, advice, skill=SKILL):
    meta = {"schema": 2, "kind": "learned", "status": "promoted", "learned_in": "sim", "occurrences": 2,
            "source_tasks": ["put the cube away", "tidy the cube"], "source_runs": ["r1", "r2"], "verifier": VERIFIED}
    front = "\n".join(f"{k}: {json.dumps(v)}" for k, v in meta.items())
    body = (f"# {title}\n\n- **Failure signature:** {skill} -> SafetyViolation: x\n- **When to apply:** cube\n\n"
            f"## Advice\n\n{advice}\n")
    (root / f"{note_id}.md").write_text(f"---\n{front}\n---\n{body}", encoding="utf-8")


# ── 1: prepositions after a guard head end the object ────────────────────

#: the prepositions the reviewer listed (its monkeypatch of `_OBJECT_TAIL`)
NEW_TAIL = frozenset({"of", "about", "during", "without", "like", "among", "despite", "except", "per", "throughout",
                      "amid", "regarding", "besides"})

#: the reviewer's 14 head-noun probes: the first 12 were served on 7532e44, the last 2 refused
REVIEWER_PROBES = [
    "Remove the safety check of the arm.",
    "Drop the clearance margin of the gripper.",
    "Cut the collision check of the wrist in half.",
    "Remove the guard of the gripper before the pick.",
    "Dismiss the collision warning of the arm.",
    "Zero the safety margin of the gripper.",
    "Take off the safety limit of the wrist.",
    "Remove the safety check during the pour.",
    "Drop the collision check without asking.",
    "Turn the clearance gate of the arm off.",
    "Remove the safety check about the cup.",
    "Waive the clearance gate like before.",
    "Remove the safety check entirely.",
    "Remove the safety check on the arm.",
]
#: the reviewer's other two served shapes (split "shut ... down", another head-rule verb)
MORE_PROBES = ["Shut the safety interlock of the arm down.", "Forget the safety check of the arm."]


def test_the_new_tail_words_end_an_object_but_are_no_prepositions():
    assert NEW_TAIL <= notes._OBJECT_TAIL
    assert not NEW_TAIL & notes._PREPOSITIONS


@pytest.mark.parametrize("text", REVIEWER_PROBES + MORE_PROBES)
def test_a_guard_head_before_a_preposition_is_refused(text):
    assert "weaken" in _rules(text), text


@pytest.mark.parametrize("word", sorted(NEW_TAIL))
def test_every_new_tail_word_ends_the_object(word):
    # the guard head is "check" (the word before the tail word); before round 5 the
    # tail word itself read as the head and the advice was served
    assert lint_advice(f"Remove the safety check {word} the arm.", skill=SKILL) == [
        "advice: 'remove' near 'safety check'"], word
    assert "weaken" in _rules(f"Forget the safety check {word} the arm."), word


def test_the_reported_object_runs_to_the_head():
    assert lint_advice("Take off the safety limit of the wrist.", skill=SKILL) == [
        "advice: 'take off' near 'safety limit'"]
    assert lint_advice("Turn the clearance gate of the arm off.", skill=SKILL) == [
        "advice: 'turn ... off' near 'gate'"]


@pytest.mark.parametrize("text", [
    # "of" as a preposition would stop the object scan before the guard word
    "Skip part of the check.", "Disable all of the checks.", "Skip some of the safety checks.",
    "Turn half of the gates off.", "Lower most of the margin.",
])
def test_a_tail_word_does_not_stop_the_object_scan(text):
    assert "weaken" in _rules(text), text


@pytest.mark.parametrize("text", [
    "Cut the safety seal of the milk carton.", "Remove the safety cover of the arm.", "Remove the lid of the jar.",
    "Drop the warning label of the jar into the bin.", "Take off the safety pin during the pour.",
])
def test_a_guard_modifier_of_another_head_before_a_preposition_stays_plain(text):
    assert "weaken" not in _rules(text), text


@pytest.mark.parametrize("profile_class", ["sim", "real"])
def test_the_probes_are_never_served(tmp_path, profile_class):
    probes = REVIEWER_PROBES + MORE_PROBES
    for i, text in enumerate(probes):
        _promoted(tmp_path, f"adv-{i:02d}", title="Cube near the lemon", advice=text)
    s = load_snapshot(tmp_path, profile_class)
    assert len(s.notes) == len(probes) and s.served == ()
    assert {entry["reason"] for entry in s.skipped} == {"weaken"}


# ── 2: pre-set / pre-sets are whole allowlisted words ────────────────────

def test_the_skeleton_allowlist_has_pre_set():
    assert notes._SKELETON_ALLOW == frozenset({"preset", "presets", "pre-set", "pre-sets", "you're", "we're",
                                               "they're"})


@pytest.mark.parametrize("text", [
    "Use the pre-set timer on the microwave.", "Set the oven to the pre-set bread mode.",
    "Pre-sets: pick the bread one.", "Use the oven's PRE-SET for pizza.", "Use the \u201cpre-set\u201d tray.",
])
def test_pre_set_advice_is_accepted(text):
    assert lint_advice(text, skill=SKILL) == [], text


@pytest.mark.parametrize("title", ["Pre-set timer", "Oven pre-sets"])
def test_pre_set_titles_are_accepted(title):
    assert _title_rules(title) == [], title


@pytest.mark.parametrize("text", [
    # only the whole word: a split, fused or neighbouring spelling still reads as a reset
    "Pre/set the scene after a refusal.", "Use the pre-setting scene.", "Re-pre-set the scene.",
    "Pre-set-the scene now.", "R pre-set eset the scene.",
    # the other hyphen compounds stay refused (accepted over-block, follow-up list)
    "Use the temperature-setting dial.", "Turn the pressure-set valve slowly.",
])
def test_other_spellings_around_pre_set_stay_refused(text):
    assert "reset" in _rules(text), text


@pytest.mark.parametrize("profile_class", ["sim", "real"])
def test_pre_set_notes_are_served(tmp_path, profile_class):
    _promoted(tmp_path, "t-0", title="Pre-set timer", advice="Use the pre-set timer on the microwave.")
    s = load_snapshot(tmp_path, profile_class)
    assert [n.id for n in s.served] == ["t-0"]


# ── 3: the stop guard is the word stop / stops ───────────────────────────

def test_the_stop_guard_is_exact_words():
    # fix round 5: "stop" / "stops", never the prefix; A2 fix round 1 adds the other forms (test_aspire_notes_fix6)
    stops = [p for p in notes._GUARDS if p[0].startswith("stop")]
    assert stops[:2] == [("stop",), ("stops",)] and not any(p[0].endswith("*") for p in stops)
    assert ("guard*",) in notes._GUARDS  # guard words stay (accepted over-block: "the blade guard")


#: kitchen nouns that only start like "stop"
STOP_NOUNS = [
    "Take off the stopper, then pour slowly.",
    "Switch the stopwatch off when the eggs are done.",
    "Switch off the stopwatch when the eggs are done.",
    "Take off the stopwatch.",
    "Remove the stopper before you pour the oil.",
    "Remove the lid stopper, then pour.",
    "Drop the stopper back in the bottle.",
    "Lower the stopper into the bottle.",
]


@pytest.mark.parametrize("text", STOP_NOUNS)
def test_a_stop_noun_is_no_guard(text):
    assert lint_advice(text, skill=SKILL) == [], text


@pytest.mark.parametrize("profile_class", ["sim", "real"])
def test_stop_noun_notes_are_served(tmp_path, profile_class):
    for i, text in enumerate(STOP_NOUNS):
        _promoted(tmp_path, f"adv-{i:02d}", title="Cube near the lemon", advice=text)
    s = load_snapshot(tmp_path, profile_class)
    assert len(s.served) == len(s.notes) == len(STOP_NOUNS) and s.skipped == ()


@pytest.mark.parametrize("text", [
    "Disable the stop.", "Disable the stops.", "Skip the stops near the sink.", "Remove the stops.",
    "Take off the stop.", "Turn the safety stop off.", "Switch the stops off.",
    # guard nouns stay guards (accepted over-block)
    "Take off the splash guard to clean it.", "Take off the blade guard only when the knife is in your hand.",
])
def test_stop_and_guard_words_stay_guards(text):
    assert "weaken" in _rules(text), text
