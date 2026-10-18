"""ASPIRE polish round (A3/A4 re-review deleg_1e3157c0 #0, Minors 1 and 5).

1. A dash (a spaced " - " or a typographic dash) right after a determiner or right after a
   weakening verb does not end that verb's object, and one right after the object's head does
   not part a split phrasal verb from its particle: "Disable the - gate ...", "Skip the - check
   ...", "Lower the - safety margin ...", "Turn the clearance gate - off ..." are refused as
   `weaken` -- also when the renderer's own " - " forms them from a clean title and a clean
   advice. A dash after any other word still ends the object ("Lower the cup - check the
   handle first"), and a sentence break still parts a phrasal verb from its particle.
5. The spec's construction bound of `robot_knowledge`'s `learned_notes` is the line the
   renderer can produce: 10 lines of `- `, the longest note text, ` [`, the longest served
   tag and `]`.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import pytest
from test_aspire_notes_fix7 import SKILL, VERIFIED, N, _note, _rules

from cascade.skills import notes
from cascade.skills.notes import Note, load_snapshot, note_tag

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs/superpowers/specs/2026-10-17-aspire-demo-design.md"

#: (text, the rule that refuses it): a dash right after a determiner or a weakening verb, or
#: between a phrasal verb's object and its particle, keeps the verb and its object together
DASHED = [
    ("Disable the - gate when the cube sits near the lemon.", "weaken"),          # the re-review's four
    ("Skip the - check when the cube sits near the lemon.", "weaken"),
    ("Turn the clearance gate - off near the sink.", "weaken"),
    ("Lower the - safety margin near the sink.", "weaken"),
    ("Disable - the gate when the cube sits near the lemon.", "weaken"),          # right after the verb
    ("Turn off - the clearance gate near the sink.", "weaken"),                   # after a two-word verb
    ("Dismiss the - warning near the sink.", "weaken"),                           # a head-noun verb
    ("Remove the - safety check of the arm near the sink.", "weaken"),            # a removal verb
    ("Skip the light and the - gate near the sink.", "weaken"),                   # a listed object
    ("Turn the - clearance gate off near the sink.", "weaken"),                   # split phrasal verb
    ("Disable the \u2014 gate when the cube sits near the lemon.", "weaken"),     # typographic dashes
    ("Skip the \u2013 check when the cube sits near the lemon.", "weaken"),
    ("Turn the clearance gate \u2014 off near the sink.", "weaken"),
    ("Skip\u2014the safety check near the lemon.", "weaken"),                      # unspaced, after the verb
    ("Restart the - scene before the next pick.", "reset"),                       # the same object scan
]
#: a dash after any other word still ends the object; a sentence break still parts the particle
STILL_CLEAN = [
    "Lower the cup - check the handle first.",
    "Skip the mug - the cup is closer to the arm.",
    "Raise the lid - the steam is hot near the sink.",
    "Restart the pick - the scene is the same.",
    "Shut the gate. Off to the sink with the cup.",
    "If the cup is full, skip. Check the next cup first.",                     # a sentence ends after the verb
    "Shut the safety gate first - off to the sink with the cup.",              # the dash is not after the head
]
#: the renderer joins a clean title and a clean advice with " - " into a refused line
COMPOSED = [
    ("Disable the", "gate when the cube sits near the lemon."),
    ("Skip the", "check when the cube sits near the lemon."),
    ("Turn the clearance gate", "off near the sink."),
    ("Lower the", "safety margin near the sink."),
    ("Disable", "the gate when the cube sits near the lemon."),
    ("Remove the", "safety check of the arm near the sink."),
]


def _field_rules(text: str) -> set[str]:
    return {rule for rule, _ in notes._lint_field("advice", text, notes.ADVICE_MAX_CHARS, SKILL)}


def _undashed(text: str) -> str:
    for dash in (" - ", " \u2013 ", " \u2014 ", "\u2013", "\u2014"):
        text = text.replace(dash, " ")
    return text


@pytest.mark.parametrize("text,rule", DASHED, ids=[f"dashed{i}" for i in range(len(DASHED))])
def test_a_dash_after_a_determiner_or_weakening_verb_keeps_the_object(text, rule):
    assert rule in _field_rules(_undashed(text))                                     # premise: refused undashed
    assert rule in _field_rules(text)


@pytest.mark.parametrize("text", STILL_CLEAN)
def test_a_dash_after_any_other_word_still_ends_the_object(text):
    assert _field_rules(text) == set()
    assert notes.lint_advice(text, skill=SKILL) == []


@pytest.mark.parametrize("title,advice", COMPOSED, ids=[f"pair{i}" for i in range(len(COMPOSED))])
def test_the_separator_never_splits_a_weakening_verb_from_its_object(tmp_path, title, advice):
    assert notes._lint_field("title", title, notes.TITLE_MAX_CHARS, SKILL) == []      # premise: each field clean
    assert notes._lint_field("advice", advice, notes.ADVICE_MAX_CHARS, SKILL) == []
    assert "weaken" in _rules(N(title, advice))
    _note(tmp_path / "lib", "dashed", title, advice)
    _note(tmp_path / "lib", "plain", "Cube near the lemon", "Retry once from home.")
    for klass in ("sim", "real"):
        snap = load_snapshot(tmp_path / "lib", klass)
        assert "dashed" not in {n.id for n in snap.served}
        assert {"id": "dashed", "reason": "weaken"} in [dict(e) for e in snap.skipped]
        assert advice not in snap.knowledge() and advice not in snap.block()


# ── 5: the spec's construction bound ─────────────────────────────────────────


def _longest_served_tag() -> str:
    """The longest provenance tag a served note can carry, over every kind, origin and verifier state."""
    tags = []
    for kind, where, verifier, klass in itertools.product(sorted(notes.KINDS), sorted(notes.LEARNED_IN),
                                                          (VERIFIED, None), ("sim", "real")):
        note = Note(id="n", title="t", when="w", advice="a", kind=kind, status="promoted", learned_in=where,
                    n_tasks=9, occurrences=9, verifier=verifier, grades={}, skill=SKILL, schema=2)
        if notes._serves(note, klass, 2):
            tags.append(note_tag(note, "f" * 64))
    return max(tags, key=len)


def test_the_spec_states_the_knowledge_bound_the_renderer_gives():
    tag = _longest_served_tag()
    line = len(notes._line("T" * notes.TITLE_MAX_CHARS, "a" * notes.ADVICE_MAX_CHARS, tag))
    assert line == 2 + notes.LINE_MAX_CHARS + 2 + len(tag) + 1
    spec = " ".join(SPEC.read_text(encoding="utf-8").split())
    assert f"{notes.KNOWLEDGE_SHOWN} lines of at most {line} characters" in spec
    assert f"longest served tag of {len(tag)} characters" in spec
    assert f"~{notes.KNOWLEDGE_SHOWN * line / 1000:.1f} k tokens at one token per character" in spec
    assert "4.1 k tokens" not in spec
