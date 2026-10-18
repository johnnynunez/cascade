"""People guard: who counts as a person, and the `safety.people` config.

Pure checks of `safety/people_guard.py` (the runtime behaviour is pinned in
tests/test_people_guard_runtime.py and tests/test_people_guard_eval_replay.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from cascade.safety import people_guard as pg


@pytest.mark.parametrize("text, word", [
    # English: people, roles, pronouns
    ("the visitor", "visitor"), ("Visitors", "visitors"), ("the people watching", "people"),
    ("that man", "man"), ("a child", "child"), ("someone", "someone"), ("throw it at me", "me"),
    ("the audience", "audience"), ("toward_viewer", "viewer"), ("towards_me", "me"), ("closer to me", "me"),
    ("the operator", "operator"), ("her", "her"),
    # English: body parts, possessed or bare
    ("my wrist", "wrist"), ("the visitor's hand", "visitor"), ("his face", "face"), ("FINGERS", "fingers"),
    ("your arm", "arm"), ("an eye", "eye"), ("the neck", "neck"), ("my head", "head"),
    # Spanish (accents optional)
    ("el visitante", "visitante"), ("la gente", "gente"), ("las personas", "personas"),
    ("mi muñeca", "muneca"), ("la mano", "mano"), ("la cabeza de la niña", "cabeza"), ("el niño", "nino"),
    ("hacia el público", "publico"), ("tu cara", "cara"), ("los dedos", "dedos"), ("conmigo", "conmigo"),
    # French / German / Italian / Portuguese / Catalan
    ("le visiteur", "visiteur"), ("die Leute", "leute"), ("der Besucher", "besucher"), ("la persona", "persona"),
    ("le persone", "persone"), ("a pessoa", "pessoa"), ("o rosto", "rosto"), ("el visitant", "visitant"),
    ("la tête", "tete"), ("das Gesicht", "gesicht"), ("il polso", "polso"),
])
def test_people_and_body_parts_are_named(text, word):
    assert pg.person_mention(text) == word


@pytest.mark.parametrize("text", [
    # every object and destination the kitchen and mock scenes use, and the
    # bake-off's ground-truth names: none of them is a person
    "red cube", "green cube", "pink cube", "red can", "tomato can", "lemon", "orange", "yellow plate",
    "green square", "open box", "drop zone", "the box", "cubo rosa", "cuadrado verde", "blue bottle",
    "white paper cup", "black remote control", "red plush lobster", "green handle", "black smartphone",
    "nut", "screw", "red object", "unobtainium fastener", "table", "the tabletop", "dollhouse",
    # robot parts, object parts and directions that only look like body words
    "the robotic arm", "white robotic arm", "gripper finger", "the arm of the robot", "brazo robótico",
    "screw head", "the head of the bolt", "bottle neck", "cuello de la botella", "clock face", "table leg",
    "away_from_viewer", "away from me", "lejos de la gente", "left", "back", "forward", "up", "again",
    "toward the camera", "above", "next_to", "", "   ",
])
def test_objects_robot_parts_and_directions_are_not_people(text):
    assert pg.person_mention(text) is None


@pytest.mark.parametrize("value", [None, 3, 0.5, ["visitor"], {"label": "visitor"}, True])
def test_a_non_string_names_nobody(value):
    assert pg.person_mention(value) is None


def test_a_robot_word_elsewhere_does_not_hide_a_person():
    """Only a robot word NEXT TO a shared body part names the robot: "put the
    gripper around my wrist" still names the visitor's wrist."""
    assert pg.person_mention("put the gripper around my wrist") == "wrist"
    assert pg.person_mention("robot, grab the screw with my hand") == "hand"
    assert pg.person_mention("the hand on the table") == "hand"


# ── config ─────────────────────────────────────────────────────────────────


def test_no_block_means_every_side_is_a_visitor_side():
    people = pg.parse(None, 0.03)
    assert people.visitor_sides == frozenset({"forward", "left", "right", "back"})
    assert people.zones == () and people.margin_m == 0.03
    assert pg.parse({}, 0.03).visitor_sides == people.visitor_sides
    assert pg.parse({"visitor_sides": None}, 0.03).visitor_sides == people.visitor_sides


def test_sides_can_be_narrowed_explicitly():
    assert pg.parse({"visitor_sides": ["forward", "left"]}, 0.03).visitor_sides == {"forward", "left"}
    assert pg.parse({"visitor_sides": []}, 0.03).visitor_sides == frozenset()


def test_zones_grow_by_the_clearance_margin():
    people = pg.parse({"visitor_zones": [{"name": "front", "min": [0.6, -0.4, 0.0], "max": [1.0, 0.4, 2.0]}]},
                      0.03)
    (zone,) = people.zones
    assert zone.name == "front"
    np.testing.assert_allclose(zone.lo, [0.57, -0.43, -0.03])
    np.testing.assert_allclose(zone.hi, [1.03, 0.43, 2.03])
    assert zone.contains([0.575, 0.0, 0.5]) and not zone.contains([0.565, 0.0, 0.5])
    unnamed = pg.parse({"visitor_zones": [{"min": [0, 0, 0], "max": [1, 1, 1]}]}, 0.0).zones[0]
    assert unnamed.name == "visitor zone 1"


@pytest.mark.parametrize("block, margin", [
    ({"visitor_sides": ["front"]}, 0.03),
    ({"visitor_sides": "forward"}, 0.03),
    ({"visitor_sides": 3}, 0.03),
    ("people", 0.03),
    ({"visitor_zones": ["front"]}, 0.03),
    ({"visitor_zones": [{"min": [0, 0], "max": [1, 1]}]}, 0.03),
    ({"visitor_zones": [{"min": [0, 0, 0], "max": [1, float("nan"), 1]}]}, 0.03),
    ({"visitor_zones": [{"min": [1, 0, 0], "max": [0, 1, 1]}]}, 0.03),
    ({"visitor_zones": [{"max": [1, 1, 1]}]}, 0.03),
    ({}, float("nan")),
    ({}, -0.01),
])
def test_a_malformed_block_fails_closed(block, margin):
    with pytest.raises(ValueError):
        pg.parse(block, margin)


def test_the_harness_limits_carry_the_parsed_block():
    from cascade.config import load_demo_config
    from cascade.safety.harness import SafetyLimits

    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    limits = SafetyLimits.from_config(cfg.safety)
    assert limits.people == pg.parse(cfg.safety.get("people"), 0.03)
    assert limits.people.visitor_sides == frozenset(pg.SIDES)
    # a configured block reaches the harness, grown by min_clearance_m
    cfg._data["safety"]["people"] = {"visitor_sides": ["back"],
                                     "visitor_zones": [{"name": "aisle", "min": [0.6, -1, 0], "max": [1, 1, 2]}]}
    people = SafetyLimits.from_config(cfg.safety).people
    assert people.visitor_sides == {"back"}
    assert [z.name for z in people.zones] == ["aisle"]
    np.testing.assert_allclose(people.zones[0].lo, [0.57, -1.03, -0.03])
    cfg._data["safety"]["people"] = {"visitor_sides": ["sideways"]}
    with pytest.raises(ValueError, match="unknown side"):
        SafetyLimits.from_config(cfg.safety)
    # a dataclass default (tests and harnesses that build SafetyLimits by hand)
    bare = SafetyLimits(workspace_min=np.zeros(3), workspace_max=np.ones(3))
    assert bare.people.visitor_sides == frozenset(pg.SIDES) and bare.people.zones == ()


def test_the_failure_taxonomy_files_a_people_refusal_first():
    from cascade.memory.envelope import normalize_failure

    # its own words name a halt, a keep-out and a missing object: still people
    err = (f"{pg.REFUSAL_PREFIX}: grasp_object refused: label='halted keep-out not found' names a person "
           "or a body part ('visitor'); no command was sent")
    assert normalize_failure(err) == pg.SAFETY_CLASS
    assert normalize_failure("refused by the safety harness: TCP [0.3, 0.2, 0.1] inside visitor zone 'front' "
                             "(people keep-out, safety.people.visitor_zones grown by 0.030 m). The arm did not "
                             "move.") == pg.SAFETY_CLASS
    assert normalize_failure("TCP inside keep-out zone [0, 0, 0]..[1, 1, 1]") == "geometry:keep_out"
