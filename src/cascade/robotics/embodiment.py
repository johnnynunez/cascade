"""Declared robot structure, independent of control admission and measured frames.

Units are explicit SI. Transmission ratios describe joint position / actuator
position; this module never applies a transmission or constructs a controller.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType

from .contracts import freeze_json, identifier, plain_json
from .joint_coordinates import MULTI_DOF_JOINTS, validate_coordinates


JOINT_UNITS = MappingProxyType({
    "revolute": ("rad", "rad/s", "N*m"),
    "continuous": ("rad", "rad/s", "N*m"),
    "prismatic": ("m", "m/s", "N"),
})


def _object(value, allowed, required, label):
    if not isinstance(value, Mapping) or set(value) - set(allowed) or set(required) - set(value):
        raise ValueError(f"{label}: missing or unknown fields")
    return dict(value)


def _number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    return float(value)


def _ids(values, label, *, minimum=0, maximum=2048):
    if not isinstance(values, (tuple, list)) or not minimum <= len(values) <= maximum:
        raise ValueError(f"{label} has invalid length")
    result = tuple(identifier(v, label) for v in values)
    if len(result) != len(set(result)):
        raise ValueError(f"duplicate {label}")
    return result


def _records(values, key, parser, label, *, maximum=2048):
    if not isinstance(values, (tuple, list)) or len(values) > maximum:
        raise ValueError(f"{label} must be a bounded array")
    result = tuple(parser(v) for v in values)
    if len({v[key] for v in result}) != len(result):
        raise ValueError(f"duplicate {label}")
    return result


def _joint(raw, *, version=1):
    allowed = {"joint_id", "parent", "child", "type", "axis", "units", "limits", "actuation", "loop_closure"}
    if version == 2:
        allowed.add("coordinates")
    data = _object(raw, allowed,
                   {"joint_id", "parent", "child", "type"}, "joint")
    for key in ("joint_id", "parent", "child"):
        identifier(data[key], key)
    if data["parent"] == data["child"]:
        raise ValueError("joint cannot connect a link to itself")
    kind = data["type"]
    if kind not in (*JOINT_UNITS, "fixed") and not (version == 2 and kind in MULTI_DOF_JOINTS):
        raise ValueError("unsupported joint type")
    data.setdefault("loop_closure", False)
    if type(data["loop_closure"]) is not bool:
        raise ValueError("loop_closure must be boolean")
    data.setdefault("actuation", "passive")
    if data["actuation"] not in {"passive", "actuated"}:
        raise ValueError("joint actuation must be explicit passive or actuated")
    if kind in MULTI_DOF_JOINTS:
        if data["actuation"] != "passive" or any(key in data for key in ("axis", "units", "limits")):
            raise ValueError("multi-DoF joints are observation-only; scalar axes, limits and actuation are unsupported")
        data["coordinates"] = validate_coordinates(kind, data.get("coordinates"))
        return freeze_json(data)
    if "coordinates" in data:
        raise ValueError("scalar and fixed declarations retain their existing coordinate schema")
    if kind == "fixed":
        if any(key in data for key in ("axis", "units", "limits")) or data["actuation"] != "passive":
            raise ValueError("fixed joint has no axis, units, limits or actuation")
        return freeze_json(data)
    if "axis" not in data or not isinstance(data["axis"], (tuple, list)) or len(data["axis"]) != 3:
        raise ValueError("movable joint requires a three-dimensional unit axis")
    data["axis"] = tuple(_number(v, "axis") for v in data["axis"])
    if not math.isclose(sum(v*v for v in data["axis"]), 1., rel_tol=0, abs_tol=1e-8):
        raise ValueError("joint axis must be a unit vector")
    expected = dict(zip(("position", "velocity", "effort"), JOINT_UNITS[kind]))
    if data.get("units") != expected:
        raise ValueError(f"{kind} requires explicit units {expected}")
    limits = _object(data.get("limits", {}), {"lower", "upper", "velocity", "effort"}, (), "joint limits")
    limits = {key: _number(value, "joint limit", positive=key in {"velocity", "effort"}) for key, value in limits.items()}
    if kind == "continuous":
        if "lower" in limits or "upper" in limits:
            raise ValueError("continuous joint cannot declare position bounds")
    elif set(limits) & {"lower", "upper"} != {"lower", "upper"} or limits["lower"] >= limits["upper"]:
        raise ValueError("bounded joint requires lower < upper")
    data["limits"] = limits
    return freeze_json(data)


def _transmission(raw):
    data = _object(raw, {"transmission_id", "actuator_id", "resource_id", "actuator_unit", "couplings"},
                   {"transmission_id", "actuator_id", "resource_id", "actuator_unit", "couplings"}, "transmission")
    for key in ("transmission_id", "actuator_id", "resource_id"):
        identifier(data[key], key)
    if data["actuator_unit"] not in {"rad", "m"}:
        raise ValueError("actuator unit must be rad or m")
    def coupling(raw):
        c = _object(raw, {"joint_id", "ratio", "ratio_unit"}, {"joint_id", "ratio", "ratio_unit"}, "coupling")
        identifier(c["joint_id"], "joint_id")
        c["ratio"] = _number(c["ratio"], "transmission ratio")
        if c["ratio"] == 0:
            raise ValueError("transmission ratio cannot be zero")
        return freeze_json(c)
    data["couplings"] = _records(data["couplings"], "joint_id", coupling, "couplings", maximum=512)
    if not data["couplings"]:
        raise ValueError("transmission requires a coupling")
    return freeze_json(data)


def _attachment(raw, *, sensor):
    key = "sensor_id" if sensor else "effector_id"
    extra = {"frame_id", "joint_ids"} if sensor else {"kind", "required_capabilities"}
    data = _object(raw, {key, "link_id", "resource_id"} | extra,
                   {key, "link_id", "resource_id", "frame_id" if sensor else "kind"}, "attachment")
    for k in (key, "link_id", "resource_id", "frame_id" if sensor else "kind"):
        identifier(data[k], k)
    names = "joint_ids" if sensor else "required_capabilities"
    data[names] = _ids(data.get(names, ()), names, maximum=512)
    return freeze_json(data)


@dataclass(frozen=True)
class EmbodimentDescriptor:
    """Immutable structural declaration. Its digest does not attest physics."""
    robot_id: str
    root_link: str
    root_mode: str
    links: tuple
    joints: tuple = ()
    transmissions: tuple = ()
    effectors: tuple = ()
    sensors: tuple = ()
    version: int = 1

    def __post_init__(self):
        if type(self.version) is not int or self.version not in {1, 2}:
            raise ValueError("embodiment requires version 1 or 2")
        identifier(self.robot_id, "robot_id")
        identifier(self.root_link, "root_link")
        if self.root_mode not in {"fixed", "floating"}:
            raise ValueError("root_mode must be fixed or floating")
        links = _ids(self.links, "links", minimum=1)
        joints = _records(self.joints, "joint_id", lambda value: _joint(value, version=self.version), "joints")
        transmissions = _records(self.transmissions, "transmission_id", _transmission, "transmissions")
        effectors = _records(self.effectors, "effector_id", lambda v: _attachment(v, sensor=False), "effectors")
        sensors = _records(self.sensors, "resource_id", lambda v: _attachment(v, sensor=True), "sensors")
        for key, value in (("links", links), ("joints", joints), ("transmissions", transmissions), ("effectors", effectors), ("sensors", sensors)):
            object.__setattr__(self, key, value)
        if self.root_link not in links:
            raise ValueError("root link is absent")
        # Primary edges form one rooted tree. Explicit closure edges describe
        # additional constraints only; they do not provide a closure solver.
        parents = {}
        for joint in joints:
            if joint["parent"] not in links or joint["child"] not in links:
                raise ValueError("joint references unknown link")
            if not joint["loop_closure"]:
                if joint["child"] in parents or joint["child"] == self.root_link:
                    raise ValueError("primary joint graph has duplicate parent or root parent")
                parents[joint["child"]] = joint["parent"]
        for link in links:
            visited = set()
            current = link
            while current != self.root_link:
                if current in visited or current not in parents:
                    raise ValueError("joint graph is disconnected or cyclic")
                visited.add(current)
                current = parents[current]
        by_joint = {j["joint_id"]: j for j in joints}
        driven = set()
        actuators = set()
        for transmission in transmissions:
            if transmission["actuator_id"] in actuators:
                raise ValueError("actuator declared by multiple transmissions")
            actuators.add(transmission["actuator_id"])
            for coupling in transmission["couplings"]:
                name = coupling["joint_id"]
                joint = by_joint.get(name)
                if joint is None or joint["actuation"] != "actuated" or name in driven:
                    raise ValueError("transmission needs an exclusively declared actuated joint")
                expected = f"{JOINT_UNITS[joint['type']][0]}/{transmission['actuator_unit']}"
                if coupling["ratio_unit"] != expected:
                    raise ValueError("transmission ratio dimensions disagree with joint/actuator units")
                driven.add(name)
        if driven != {j["joint_id"] for j in joints if j["actuation"] == "actuated"}:
            raise ValueError("every actuated joint needs exactly one transmission")
        sensor_resources = set()
        for attachment in (*effectors, *sensors):
            if attachment["link_id"] not in links:
                raise ValueError("attachment references unknown link")
        for sensor in sensors:
            if sensor["resource_id"] in sensor_resources:
                raise ValueError("sensor resource has multiple attachments")
            sensor_resources.add(sensor["resource_id"])
            if any(name not in by_joint or by_joint[name]["type"] == "fixed" for name in sensor["joint_ids"]):
                raise ValueError("sensor joint_ids must reference movable joints")

    @classmethod
    def from_dict(cls, value):
        if isinstance(value, cls):
            return value
        data = _object(value, {"version", "robot_id", "root_link", "root_mode", "links", "joints", "transmissions", "effectors", "sensors"},
                       {"version", "robot_id", "root_link", "root_mode", "links"}, "embodiment")
        return cls(**data)

    def as_dict(self):
        return {key: plain_json(getattr(self, key)) for key in self.__dataclass_fields__}

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def sensor_attachment(self, resource_id):
        matches = [s for s in self.sensors if s["resource_id"] == resource_id]
        if not matches:
            raise ValueError(f"sensor {resource_id!r} has no embodiment attachment")
        return matches[0]

    def validate_resources(self, catalog):
        """Check existing owners/capabilities; never grant a resource or admission."""
        def resource(name):
            value = catalog.require(name)
            if value.robot_id != self.robot_id:
                raise ValueError("embodiment resource belongs to another robot")
            return value
        for transmission in self.transmissions:
            value = resource(transmission["resource_id"])
            if value.writer_id is None or value.controller_id is None:
                raise ValueError("transmission resource must have an existing controller writer")
        for effector in self.effectors:
            value = resource(effector["resource_id"])
            for capability in effector["required_capabilities"]:
                catalog.require(value.resource_id, capability)
        for sensor in self.sensors:
            value = resource(sensor["resource_id"])
            if value.kind != "sensor" or value.writer_id is not None:
                raise ValueError("sensor attachment requires a read-only sensor resource")
            if value.metadata.get("sensor_id") != sensor["sensor_id"] or value.metadata.get("frame_id") != sensor["frame_id"]:
                raise ValueError("sensor identity/frame disagrees with attachment")
            joint_sensor = bool(set(value.capabilities) & {"joint_state", "proprioception", "generalized_joint_state"})
            if joint_sensor != bool(sensor["joint_ids"]):
                raise ValueError("joint sensors require exact joint_ids; other modalities cannot claim them")
        for value in catalog.describe():
            if value["robot_id"] == self.robot_id and value["kind"] == "sensor":
                self.sensor_attachment(value["resource_id"])


def embodiment_metadata(value, catalog=None):
    """Optional wire metadata shared by passive MCP discovery and runtime."""
    if value is None:
        return {}
    body = EmbodimentDescriptor.from_dict(value)
    if catalog is not None:
        body.validate_resources(catalog)
    return {"embodiment": body.as_dict(), "embodiment_sha256": body.sha256,
            "embodiment_source": "declared_profile", "embodiment_provides_transforms": False}
