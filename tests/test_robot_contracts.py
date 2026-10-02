"""Passive composition must detect real command conflicts before opening IO."""
import math

import pytest

from cascade.robotics.contracts import ResourceDescriptor, ToolDescriptor
from cascade.robotics.resources import ResourceCatalog


def resource(name, writer="whole_body", **kwargs):
    return ResourceDescriptor(name, "actuator", "g1", ("joint_targets",),
                              "dds:eth0:domain0:rt/arm_sdk", writer, **kwargs)


def test_disjoint_joints_do_not_allow_two_writers_on_full_lowcmd():
    left = resource("left_arm", "left", metadata={"joint_names": ["left_shoulder"]})
    right = resource("right_arm", "right", metadata={"joint_names": ["right_shoulder"]})
    with pytest.raises(ValueError, match="incompatible writers"):
        ResourceCatalog([left, right])


def test_one_whole_body_writer_can_expose_multiple_resources():
    catalog = ResourceCatalog([resource("legs"), resource("head")])
    assert catalog.require("head", "joint_targets").writer_id == "whole_body"
    with pytest.raises(ValueError, match="lacks capability"):
        catalog.require("head", "grasp")
    with pytest.raises(ValueError, match="unknown resource"):
        catalog.require("invented_arm")


def test_catalog_is_passive_and_defensively_copies_metadata():
    source = {"joints": ["head_yaw"], "calibration": {"version": 1}}
    item = resource("head", metadata=source)
    source["joints"].append("fake_arm")
    source["calibration"]["version"] = 2
    catalog = ResourceCatalog([item])
    description = catalog.describe()
    assert description[0]["metadata"] == {"joints": ["head_yaw"], "calibration": {"version": 1}}
    description[0]["metadata"]["joints"].clear()
    assert catalog.describe()[0]["metadata"]["joints"] == ["head_yaw"]
    with pytest.raises(TypeError):
        item.metadata["calibration"]["version"] = 9


@pytest.mark.parametrize("extra", [dict(controller_id="bus"), dict(writer_id="driver")])
def test_half_declared_writer_is_rejected(extra):
    with pytest.raises(ValueError, match="declared together"):
        ResourceDescriptor("arm", "actuator", "robot", **extra)


def test_duplicate_resource_identity_is_not_last_writer_wins():
    with pytest.raises(ValueError, match="duplicate resource"):
        ResourceCatalog([resource("head"), resource("head")])


def tool(**kwargs):
    kwargs.setdefault("writes", ("body",))
    return ToolDescriptor("base.walk", "Bounded motion", {
        "type": "object", "properties": {"vx": {"type": "number", "maximum": .3}},
        "required": ["vx"],
    }, "base", "walk", effect="motion", **kwargs)


@pytest.mark.parametrize("args", [{"vx": True}, {"vx": "0.1"}, {"vx": .4},
                                  {"vx": .1, "ignore_safety": True}, {}])
def test_arguments_reject_coercion_or_undeclared_fields(args):
    with pytest.raises(ValueError):
        tool().validate_arguments(args)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_values_never_reach_a_tool(value):
    with pytest.raises(ValueError, match="finite"):
        tool().validate_arguments({"vx": value})


def test_local_schema_validation_does_not_fetch_remote_refs():
    descriptor = ToolDescriptor("sensor.read", "Read", {
        "type": "object", "properties": {"x": {"$ref": "https://example.invalid/schema"}},
    }, "sensor", "read")
    with pytest.raises(ValueError, match="local definitions"):
        descriptor.check_schemas()


def test_metadata_never_encodes_a_driver_object():
    class Driver:
        def __getattr__(self, name):
            raise AssertionError("lazy driver inspected")

    with pytest.raises(ValueError, match="unsupported contract value"):
        resource("head", metadata={"driver": Driver()})


def test_read_only_tool_cannot_claim_actuation():
    with pytest.raises(ValueError, match="read-only"):
        ToolDescriptor("sensor.read", "Read", {"type": "object"}, "sensor", "read",
                       writes=("legs",))


def test_motion_must_declare_command_resources():
    with pytest.raises(ValueError, match="declare actuator writes"):
        tool(writes=())
