"""Fixed destination queries cannot silently localize a similarly colored prop."""
from types import SimpleNamespace

import pytest

from cascade.agent.effects import PostconditionChecker, UNVERIFIED
from cascade.config import Cfg
from cascade.skills.runtime import SkillRuntime
from cascade.types import SkillError


DELIVERY_ALIASES = ["green square", "delivery area", "delivery zone", "green delivery area",
                    "green delivery zone", "green zone", "delivery square", "drop zone",
                    "THE GREEN DELIVERY ZONE", "  the delivery area  "]


def configured_runtime():
    rt = SkillRuntime.__new__(SkillRuntime)
    rt.cfg = Cfg({"grasp": {"drop_zone": [.14, -.27], "drop_zone_name": "green square",
                            "open_box": {"center_xy_m": [.3, -.14]}, "home_after_place": False}})
    rt._localize = lambda *_a, **_kw: pytest.fail("Configured zone used object detection")
    rt._object_pose = lambda *_: pytest.fail("Configured zone used physics planning")
    return rt


@pytest.mark.parametrize("label", DELIVERY_ALIASES)
def test_delivery_alias_has_configuration_provenance_and_no_detection_claim(label):
    result = configured_runtime().skill_localize_object(label)
    assert result["label"] == label and result["name"] == "green square"
    assert result["position_xy_m"] == [.14, -.27]
    assert result["source"] == "configuration"
    assert result["visibility"] == "not_established"
    assert not {"detected_as", "n_points", "conf", "extent_m", "objects_visible"} & result.keys()


@pytest.mark.parametrize("label", ["open box", "the beige box", "storage box"])
def test_configured_container_retains_its_distinct_center(label):
    result = configured_runtime().skill_localize_object(label)
    assert result["name"] == "open box" and result["kind"] == "container"
    assert result["position_xy_m"] == [.3, -.14]
    assert result["source"] == "configuration" and result["visibility"] == "not_established"


@pytest.mark.parametrize("label", ["green cube", "green block", "green object", "second green square",
                                    "green square not the delivery zone", "orange"])
def test_other_objects_and_referring_expressions_are_not_rewritten(label):
    rt = configured_runtime()
    def visual(query, **kwargs):
        assert query == label
        raise LookupError("requested visual search")
    rt._localize = visual
    with pytest.raises(LookupError, match="visual search"):
        rt.skill_localize_object(label)


@pytest.mark.parametrize("center", [None, [], [.14], [.14, -.27, 0], [float("nan"), 0],
                                     [0, float("inf")], "center"])
def test_missing_or_invalid_delivery_configuration_never_becomes_a_visual_guess(center):
    rt = configured_runtime()
    rt.cfg._data["grasp"]["drop_zone"] = center
    with pytest.raises(SkillError):
        rt.skill_localize_object("green delivery zone")


@pytest.mark.parametrize("box", ["box", {}, {"center_xy_m": [0]}, {"center_xy_m": [0, float("nan")]}])
def test_invalid_box_configuration_is_explicit(box):
    rt = configured_runtime()
    rt.cfg._data["grasp"]["open_box"] = box
    with pytest.raises(SkillError, match="finite XY"):
        rt.skill_localize_object("open box")


@pytest.mark.parametrize("label", DELIVERY_ALIASES)
def test_placement_alias_and_physics_verifier_use_the_same_configured_destination(label):
    rt = configured_runtime()
    rt.held_object = "lemon"
    rt._reconcile_held = lambda: None
    rt.memory = SimpleNamespace(add=lambda *a, **kw: None)
    calls = []
    def place(x, y):
        calls.append((x, y))
        rt.held_object = None
        return {"placed": "lemon", "at": [x, y, .1]}
    rt.skill_place_at = place
    result = rt.skill_pick_and_place("lemon", destination=label)
    assert calls == [(.14, -.27)]
    assert result["destination"] == "green square"
    assert result["destination_kind"] == "configured_point"
    queried = []
    def pose(name):
        queried.append(name)
        return [.145, -.275, .02] if name == "lemon" else None
    verdict = PostconditionChecker(object_pose=pose).verify("pick_and_place",
        {"object": "lemon", "destination": label}, result,
        {"label": "lemon", "pose": [.32, .005, .02], "channel": "physics"})
    assert verdict.status == UNVERIFIED
    assert "containment and release are not established" in verdict.evidence
    assert queried and set(queried) == {"lemon"}
    assert verdict.measured["destination"] == "green square"
