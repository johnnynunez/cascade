"""Object aliases must preserve unambiguous physics verification."""
import json

import pytest

from cascade.sim.truth import TruthPoseReader, _match_label


@pytest.mark.parametrize("label", ["tomato tin", "the tomato tin", "tomato_tin"])
def test_tin_alias_selects_the_can(label):
    assert _match_label(label, {"tomato_can": [0.2, 0.1, 0.04]}) == [0.2, 0.1, 0.04]


def test_exact_object_identity_precedes_aliases():
    poses = {"tomato_can": [0.2, 0.1, 0.04], "tomato_tin": [0.3, 0.1, 0.04]}
    assert _match_label("tomato_tin", poses) == poses["tomato_tin"]
    assert _match_label("the tomato tin", poses) == poses["tomato_tin"]


@pytest.mark.parametrize("label", ["the lemon", "a lemon", " THE LEMON "])
def test_article_preserves_an_exact_one_word_object_name(label):
    poses = {"lemon": [0.3, 0.0, 0.02], "orange": [0.2, 0.1, 0.03]}
    assert _match_label(label, poses) == poses["lemon"]


def test_exact_body_name_with_an_article_takes_precedence():
    poses = {"the_lemon": [0.1, 0.0, 0.02], "lemon": [0.3, 0.0, 0.02]}
    assert _match_label("the lemon", poses) == poses["the_lemon"]


@pytest.mark.parametrize("label", [
    "the second lemon", "a different lemon", "the lemon nearest the can",
    "the lemon not the yellow one", "the yellow lemon",
])
def test_article_removal_keeps_every_object_qualifier(label):
    assert _match_label(label, {"lemon": [0.3, 0.0, 0.02]}) is None


def test_alias_requires_a_unique_best_match():
    poses = {"red_tomato_can": [0.2, 0.1, 0.04], "green_tomato_can": [0.3, 0.1, 0.04]}
    assert _match_label("tomato tin", poses) is None
    assert _match_label("red tomato tin", poses) == poses["red_tomato_can"]


@pytest.mark.parametrize("label,body", [
    ("green tomato tin", "red_tomato_can"),
    ("green tomato tin", "tomato_can"),
    ("turquoise tomato tin", "tomato_can"),
    ("small tomato tin", "large_tomato_can"),
    ("green tomato can", "red_tomato_tin"),
    ("green tomato can", "tomato_tin"),
    ("turquoise tomato can", "tomato_tin"),
    ("small tomato can", "large_tomato_tin"),
])
def test_alias_does_not_discard_an_explicit_query_qualifier(label, body):
    assert _match_label(label, {body: [0.2, 0.1, 0.04]}) is None


@pytest.mark.parametrize("label,body", [
    ("red tomato tin", "red_cube"),
    ("red cube", "red_tin"),
])
def test_rejected_alias_candidate_cannot_return_through_color_fallback(label, body):
    assert _match_label(label, {body: [0.2, 0.1, 0.04]}) is None


@pytest.mark.parametrize("label,body", [
    ("tomato can", "tomato_tin"),
    ("green tomato tin", "green_tomato_can"),
    ("small tomato tin", "small_tomato_can"),
    ("tomato tin", "green_tomato_can"),
])
def test_supported_alias_qualifiers_and_reverse_body_alias_remain_usable(label, body):
    position = [0.2, 0.1, 0.04]
    assert _match_label(label, {body: position}) == position


@pytest.mark.parametrize("label,body", [
    ("green tomato can", "red_tomato_can"),
    ("green tomato can", "tomato_can"),
    ("turquoise tomato can", "tomato_can"),
    ("small tomato can", "large_tomato_can"),
    ("small green cube", "green_cube"),
    ("green tomato can", "green_cube"),
    ("small red object", "red_cube"),
    ("red sphere", "red_cube"),
])
def test_non_alias_does_not_discard_explicit_query_content(label, body):
    assert _match_label(label, {body: [0.2, 0.1, 0.04]}) is None


@pytest.mark.parametrize("label,body", [
    ("the green tomato can", "green_tomato_can"),
    ("the small tomato can", "small_tomato_can"),
    ("the turquoise tomato can", "turquoise_tomato_can"),
    ("cube green small", "small_green_cube"),
])
def test_supported_non_alias_qualifiers_remain_usable(label, body):
    position = [0.2, 0.1, 0.04]
    assert _match_label(label, {body: position}) == position


@pytest.mark.parametrize("label", ["tin", "the can", "lemon tin"])
def test_container_word_alone_does_not_identify_a_prop(label):
    assert _match_label(label, {"tomato_can": [0.2, 0.1, 0.04]}) is None


def test_equal_token_matches_do_not_pick_the_first_pose():
    poses = {"red_cube_left": [0.2, 0.1, 0.04], "red_cube_right": [0.3, 0.1, 0.04]}
    assert _match_label("the red cube", poses) is None
    assert _match_label("the red cube", dict(reversed(list(poses.items())))) is None


def test_isaac_reader_resolves_alias_after_validating_physics_positions():
    class Client:
        def request(self, payload):
            assert payload["op"] == "exec"
            return {"stdout": "CASCADE_TRUTH_POSES " + json.dumps({
                "tomato_can": [0.2, 0.1, 0.04], "red_tomato_can": [float("nan"), 0, 0]
            })}

    reader = TruthPoseReader(Client())
    assert reader("tomato tin") == [0.2, 0.1, 0.04]
    assert reader.rejected == 1
