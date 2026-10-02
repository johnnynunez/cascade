"""Structural declarations never create resources or grant control admission."""
import copy
from dataclasses import FrozenInstanceError

import pytest

from cascade.robotics.contracts import ResourceDescriptor
from cascade.robotics.embodiment import EmbodimentDescriptor, embodiment_metadata
from cascade.robotics.resources import ResourceCatalog


def body_dict():
    return {"version": 1, "robot_id": "fixture", "root_link": "base", "root_mode": "floating",
            "links": ["base", "wheel", "lift", "camera"],
            "joints": [
                {"joint_id": "wheel", "parent": "base", "child": "wheel", "type": "continuous",
                 "axis": [0, 1, 0], "units": {"position": "rad", "velocity": "rad/s", "effort": "N*m"}},
                {"joint_id": "lift", "parent": "base", "child": "lift", "type": "prismatic",
                 "axis": [0, 0, 1], "units": {"position": "m", "velocity": "m/s", "effort": "N"},
                 "limits": {"lower": 0, "upper": .4}, "actuation": "actuated"},
                {"joint_id": "mount", "parent": "lift", "child": "camera", "type": "fixed"}],
            "transmissions": [{"transmission_id": "lead_screw", "actuator_id": "lift_motor",
                "resource_id": "control/lift", "actuator_unit": "rad",
                "couplings": [{"joint_id": "lift", "ratio": .002, "ratio_unit": "m/rad"}]}],
            "effectors": [{"effector_id": "tray", "kind": "platform", "link_id": "lift",
                "resource_id": "control/lift", "required_capabilities": ["linear_motion"]}],
            "sensors": [{"sensor_id": "joints", "link_id": "base", "resource_id": "sensing/joints",
                "frame_id": "fixture/base", "joint_ids": ["wheel", "lift"]}]}


def catalog(**changes):
    control = dict(resource_id="control/lift", kind="linear_stage", robot_id="fixture",
                   capabilities=("linear_motion",), controller_id="fixture:bus", writer_id="fixture:owner")
    control.update(changes)
    return ResourceCatalog([ResourceDescriptor(**control), ResourceDescriptor(
        resource_id="sensing/joints", kind="sensor", robot_id="fixture", capabilities=("joint_state",),
        metadata={"sensor_id": "joints", "frame_id": "fixture/base"})])


def test_structure_roundtrip_immutable_and_separate_from_admission():
    raw = body_dict()
    body = EmbodimentDescriptor.from_dict(raw)
    body.validate_resources(catalog())
    sha = body.sha256
    raw["joints"][1]["limits"]["upper"] = 400
    assert body.sha256 == sha
    assert EmbodimentDescriptor.from_dict(body.as_dict()).sha256 == sha
    with pytest.raises(TypeError):
        body.joints[1]["limits"]["upper"] = 1
    with pytest.raises(FrozenInstanceError):
        body.root_mode = "fixed"
    metadata = embodiment_metadata(body, catalog())
    assert metadata["embodiment_source"] == "declared_profile"
    assert not metadata["embodiment_provides_transforms"]
    assert all(r["admission"] == "unvalidated" for r in catalog().describe())
    assert embodiment_metadata(None) == {}


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(version=True),
    lambda d: d.update(root_mode="planar"),
    lambda d: d.update(root_link="absent"),
    lambda d: d["links"].append("orphan"),
    lambda d: d["links"].append("base"),
    lambda d: d["joints"][0].update(parent="wheel"),
    lambda d: d["joints"][1].update(parent="camera"),
    lambda d: d["joints"][1].update(parent="wheel", child="base"),
    lambda d: d["joints"][1].update(axis=[0, 0, 2]),
    lambda d: d["joints"][1].update(axis=[0, float("nan"), 1]),
    lambda d: d["joints"][1]["units"].update(effort="N*m"),
    lambda d: d["joints"][1]["limits"].update(lower=.4),
    lambda d: d["joints"][0].update(limits={"lower": -1, "upper": 1}),
    lambda d: d["joints"][0].update(type="ball"),
    lambda d: d["joints"][2].update(axis=[0, 0, 1]),
    lambda d: d["joints"][1].update(actuation="passive"),
    lambda d: d.update(transmissions=[]),
    lambda d: d["transmissions"][0]["couplings"][0].update(ratio_unit="rad/rad"),
    lambda d: d["transmissions"][0]["couplings"][0].update(ratio=0),
    lambda d: d["transmissions"][0]["couplings"][0].update(ratio=True),
    lambda d: d["effectors"][0].update(link_id="unknown"),
    lambda d: d["sensors"][0].update(joint_ids=["mount"]),
    lambda d: d.update(admitted=True),
])
def test_invalid_declarations_are_rejected(mutate):
    raw = body_dict()
    mutate(raw)
    with pytest.raises(ValueError):
        EmbodimentDescriptor.from_dict(raw)


def test_explicit_loop_closure_describes_constraint_without_claiming_solver():
    raw = body_dict()
    raw["joints"].append({"joint_id": "closure", "type": "fixed", "parent": "camera",
                          "child": "wheel", "loop_closure": True})
    body = EmbodimentDescriptor.from_dict(raw)
    assert body.joints[-1]["loop_closure"]
    raw["joints"][-1]["loop_closure"] = False
    with pytest.raises(ValueError, match="parent"):
        EmbodimentDescriptor.from_dict(raw)


@pytest.mark.parametrize("changes", [{"robot_id": "other"}, {"writer_id": None, "controller_id": None},
                                      {"capabilities": ()}])
def test_resource_binding_uses_existing_owner_and_capabilities(changes):
    with pytest.raises(ValueError):
        EmbodimentDescriptor.from_dict(body_dict()).validate_resources(catalog(**changes))


def test_descriptor_cannot_hide_conflicting_controller_writers():
    c = catalog()
    from cascade.robotics.contracts import ResourceDescriptor
    with pytest.raises(ValueError, match="writer"):
        ResourceCatalog([c.require("control/lift"), ResourceDescriptor(
            "another/motor", "motor", "fixture", controller_id="fixture:bus", writer_id="another:owner")])


def test_missing_or_misframed_sensor_and_duplicate_transmission_refused():
    for field, value in (("frame_id", "wrong/frame"), ("sensor_id", "wrong"), ("joint_ids", [])):
        raw = body_dict()
        raw["sensors"][0][field] = value
        with pytest.raises(ValueError):
            EmbodimentDescriptor.from_dict(raw).validate_resources(catalog())
    raw = body_dict()
    other = copy.deepcopy(raw["transmissions"][0])
    other.update(transmission_id="another", actuator_id="another")
    raw["transmissions"].append(other)
    with pytest.raises(ValueError, match="exclusively"):
        EmbodimentDescriptor.from_dict(raw)
