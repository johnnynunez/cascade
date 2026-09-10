"""Referring expressions: "the second cup from the left", "the biggest block".

VoLo's RoboVoLo makes complex references one of four capability suites
(spatial, ordinal, size, negation). ASPIRE ships the ordering rule as a learned
skill. cascade had the axis map but nothing parsed the phrase, so a booth
visitor's wording reached the detector unused.

These tests pin the semantics, especially the ones that are easy to get
backwards (rightmost, furthest).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from cascade.perception.grounding import _SPATIAL_AXES
from cascade.perception.reference import (
    Reference,
    apply_reference,
    parse_reference,
)
from cascade.types import SkillError


@dataclass
class _Det:
    label: str


@dataclass
class _Cand:
    """Stands in for ObjectFix: position + extent + detection.label."""
    position: np.ndarray
    extent: np.ndarray | None = None
    detection: _Det | None = None


def _c(x, y, z=0.04, size=0.05, label="cup"):
    return _Cand(np.array([x, y, z], float), np.array([size, size, size]), _Det(label))


# ── parsing ──────────────────────────────────────────────────────────────

def test_plain_query_has_no_constraints():
    r = parse_reference("cup")
    assert r.is_plain
    assert r.noun == "cup"


def test_ordinal_and_axis():
    r = parse_reference("the second cup from the left")
    assert r.ordinal == 1
    assert r.spatial == "left"
    assert r.noun == "cup"


def test_size_words():
    assert parse_reference("the biggest block").size is True
    assert parse_reference("the smallest block").size is False


def test_negation_is_extracted_and_stripped():
    r = parse_reference("the cup, not the red one")
    assert r.exclude is not None and "red" in r.exclude
    assert "cup" in r.noun


def test_superlatives_carry_an_axis():
    assert parse_reference("the leftmost cup").ordinal == 0
    assert parse_reference("the rightmost cup").ordinal == -1
    assert parse_reference("the nearest cup").spatial == "near"


# ── ordering semantics ───────────────────────────────────────────────────

def test_second_from_the_left_picks_the_second():
    """+y is left, so left-first means descending y."""
    left, middle, right = _c(0.2, 0.20), _c(0.2, 0.00), _c(0.2, -0.20)
    out = apply_reference([right, left, middle],
                          parse_reference("the second cup from the left"),
                          _SPATIAL_AXES)
    assert out[0] is middle


def test_leftmost_and_rightmost_are_opposite_ends():
    left, middle, right = _c(0.2, 0.20), _c(0.2, 0.00), _c(0.2, -0.20)
    cands = [middle, right, left]
    assert apply_reference(list(cands), parse_reference("the leftmost cup"),
                           _SPATIAL_AXES)[0] is left
    assert apply_reference(list(cands), parse_reference("the rightmost cup"),
                           _SPATIAL_AXES)[0] is right


def test_nearest_and_furthest_are_opposite_ends():
    """+x points away from the robot, so 'near' is min x."""
    near, mid, far = _c(0.15, 0.0), _c(0.30, 0.0), _c(0.45, 0.0)
    cands = [mid, far, near]
    assert apply_reference(list(cands), parse_reference("the nearest cup"),
                           _SPATIAL_AXES)[0] is near
    assert apply_reference(list(cands), parse_reference("the furthest cup"),
                           _SPATIAL_AXES)[0] is far


def test_biggest_and_smallest():
    small, big = _c(0.2, 0.1, size=0.03), _c(0.2, -0.1, size=0.12)
    assert apply_reference([small, big], parse_reference("the biggest block"),
                           _SPATIAL_AXES)[0] is big
    assert apply_reference([small, big], parse_reference("the smallest block"),
                           _SPATIAL_AXES)[0] is small


def test_negation_filters_by_label():
    red, blue = _c(0.2, 0.1, label="red cup"), _c(0.2, -0.1, label="blue cup")
    out = apply_reference([red, blue], parse_reference("the cup, not the red one"),
                          _SPATIAL_AXES)
    assert out[0] is blue


def test_unsatisfiable_negation_refuses_instead_of_ignoring_exclusion():
    """The old no-preference expectation licensed picking a forbidden object."""
    a, b = _c(0.2, 0.1, label="red cup"), _c(0.2, -0.1, label="red cup")
    with pytest.raises(SkillError, match="exclusion.*red.*no candidates"):
        apply_reference([a, b], parse_reference("the cup, not the red one"),
                        _SPATIAL_AXES)


def test_ordinal_beyond_the_list_refuses_instead_of_clamping():
    """The old clamping expectation substituted an object the user did not name."""
    a, b = _c(0.2, 0.1), _c(0.2, -0.1)
    with pytest.raises(SkillError, match="ordinal 5.*out of range.*2 candidate"):
        apply_reference([a, b], parse_reference("the fifth cup from the left"),
                        _SPATIAL_AXES)


def test_negative_ordinal_beyond_the_list_refuses_instead_of_clamping():
    a, b = _c(0.2, 0.1), _c(0.2, -0.1)
    with pytest.raises(SkillError, match="ordinal.*out of range.*2 candidate"):
        apply_reference([a, b], Reference(noun="cup", ordinal=-3), _SPATIAL_AXES)


def test_plain_reference_leaves_order_untouched():
    a, b, c = _c(0.2, 0.1), _c(0.2, 0.0), _c(0.2, -0.1)
    order = [b, c, a]
    assert apply_reference(list(order), parse_reference("cup"), _SPATIAL_AXES) == order


def test_reference_parsing_is_vocabulary_independent():
    """No object class is named anywhere in the parser.

    A booth cannot enumerate what visitors bring, so the reference grammar
    must work for a noun it has never seen.
    """
    r = parse_reference("the second frobnicator from the left")
    assert r.ordinal == 1 and r.spatial == "left"
    assert "frobnicator" in r.noun


@pytest.mark.parametrize("phrase", ["", "   ", "the"])
def test_degenerate_input_does_not_raise(phrase):
    r = parse_reference(phrase)
    assert r.is_plain
