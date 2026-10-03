"""Generalized observations remain separate from scalar bindings and control."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import time

import pytest

from cascade.apps.robot_runtime import describe_robot
from cascade.config import load_robot_config
from cascade.robotics.contracts import plain_json
from cascade.robotics.embodiment import EmbodimentDescriptor
from cascade.robotics.joint_coordinates import coordinate_convention
from cascade.sensing import (BufferedSensorProvider, GeneralizedJointMeasurement,
    GeneralizedJointStatePayload, JointStatePayload, MeasurementMetadata,
    ObservationEnvelope, SensorDescriptor, SensorError, SensorHub, build_sensor_domain)
from cascade.sensing.embodiment import EmbodimentBoundProvider
from cascade.sensing.models import wire


def configuration():
    cfg = load_robot_config("generalized_joint_sensors", llm="mock")
    return cfg, EmbodimentDescriptor.from_dict(cfg.embodiment.as_dict())


def observation_fixture():
    cfg, body = configuration()
    entry = cfg.domains.sensing.as_dict()["providers"][0]
    payload = GeneralizedJointStatePayload(MeasurementMetadata(entry["frame_id"]),
        entry["values"]["joints"], body.sha256)
    return cfg, body, payload


def producer(body, payload):
    descriptor = SensorDescriptor(sensor_id="joints", robot_id=body.robot_id, source="captured-fixture",
        modality=payload.modality, frame_id=payload.metadata.frame_id, clock_domain="simulation",
        measurement_kind="physics", model_identity_sha256="a"*64, epoch="captured-epoch")
    raw = BufferedSensorProvider(descriptor)
    return raw, EmbodimentBoundProvider(raw, "sensing", body)


def test_version_one_wire_and_digest_match_prechange_capture():
    golden = json.loads((Path(__file__).parent / "fixtures/generalized_joints/v1-wire.json").read_text())
    assert golden["source_head"] == "5f50b8b67600123f02488b6c721e685ac051b10e"
    for name, expected in golden["profiles"].items():
        cfg = load_robot_config(name, llm="mock")
        body = EmbodimentDescriptor.from_dict(cfg.embodiment.as_dict())
        entry = cfg.domains.sensing.as_dict()["providers"][0]
        payload = JointStatePayload(MeasurementMetadata(entry["frame_id"]), entry["values"]["joints"], body.sha256)
        assert body.version == 1
        assert body.as_dict() == expected["embodiment"]
        assert body.sha256 == expected["embodiment_sha256"]
        assert wire(payload) == expected["payload"]


def test_multi_dof_declarations_are_immutable_explicit_and_passive():
    cfg, body, payload = observation_fixture()
    assert body.version == 2 and body.root_mode == "fixed" and not body.transmissions
    assert [(j["coordinates"]["nq"], j["coordinates"]["nv"]) for j in body.joints] == [(3, 3), (4, 3), (7, 6)]
    assert all(j["actuation"] == "passive" for j in body.joints)
    assert [(len(j.q), len(j.v)) for j in payload.joints] == [(3, 3), (4, 3), (7, 6)]
    assert payload.joints[2].coordinates["q_units"] == ("m", "m", "m", "1", "1", "1", "1")
    assert payload.joints[2].coordinates["effort_units"] == ("N", "N", "N", "N*m", "N*m", "N*m")
    assert all(j.effort is None for j in payload.joints)
    sha = body.sha256
    cfg._data["embodiment"]["joints"][0]["coordinates"]["q_order"][0] = "wrong"
    assert body.sha256 == sha and payload.joints[0].coordinates["q_order"][0] == "x"
    with pytest.raises(TypeError):
        payload.joints[0].coordinates["nq"] = 9
    assert EmbodimentDescriptor.from_dict(body.as_dict()).sha256 == sha


@pytest.mark.parametrize("change", [
    lambda d: d.update(version=1),
    lambda d: d.update(version=3),
    lambda d: d["joints"][0].update(actuation="actuated"),
    lambda d: d["joints"][0].update(axis=[0, 0, 1]),
    lambda d: d["joints"][0].update(limits={"lower": -1, "upper": 1}),
    lambda d: d["joints"][0].update(units={"position": "rad", "velocity": "rad/s", "effort": "N*m"}),
    lambda d: d["joints"][0].pop("coordinates"),
    lambda d: d["joints"][0]["coordinates"].update(nv=1),
    lambda d: d["joints"][0]["coordinates"].update(nq=True),
    lambda d: d["joints"][1]["coordinates"].update(rotation="xyzw"),
    lambda d: d["joints"][1]["coordinates"].update(motion_frame="child_joint"),
    lambda d: d["joints"][1]["coordinates"].update(motion_relation="absolute"),
    lambda d: d["joints"][2]["coordinates"].update(motion_reference="parent_joint_origin"),
    lambda d: d["joints"][2]["coordinates"].update(velocity_order=["wx", "wy", "wz", "vx", "vy", "vz"]),
    lambda d: d.update(transmissions=[{"transmission_id": "invented", "actuator_id": "motor",
        "resource_id": "control/body", "actuator_unit": "rad",
        "couplings": [{"joint_id": "gimbal", "ratio": 1, "ratio_unit": "rad/rad"}]}]),
])
def test_unknown_or_scalar_multi_dof_interpretations_are_rejected(change):
    _, body = configuration()
    raw = body.as_dict()
    change(raw)
    with pytest.raises(ValueError):
        EmbodimentDescriptor.from_dict(raw)


@pytest.mark.parametrize("changes", [
    {"q": [1, 0, 0]}, {"v": [0, 0, 0, 0]}, {"effort": [0, 0, 0, 0]},
    {"q": [0, 0, 0, 0]}, {"q": [2, 0, 0, 0]}, {"q": [True, 0, 0, 0]},
    {"q": [float("nan"), 0, 0, 0]}, {"v": [0, float("inf"), 0]}, {"effort": [False, 0, 0]},
])
def test_q_v_and_effort_dimensions_and_quaternion_validity_are_independent(changes):
    _, _, payload = observation_fixture()
    with pytest.raises(ValueError):
        replace(payload.joints[1], **changes)


def test_quaternion_sign_roundoff_and_tangent_velocity_are_preserved():
    _, _, payload = observation_fixture()
    q = [-0.70710677, 0., 0., -0.70710677]  # ordinary float32 unit-quaternion roundoff
    joint = replace(payload.joints[1], q=q, v=[1., 2., 3.], effort=[4., 5., 6.])
    assert joint.q == tuple(q)  # neither renormalized nor sign canonicalized
    assert joint.v == (1., 2., 3.) and joint.effort == (4., 5., 6.)
    assert joint.coordinates["rotation"] == "hamilton_wxyz_child_to_parent"
    assert joint.coordinates["motion_frame"] == "parent_joint"
    assert joint.coordinates["motion_reference"] == "child_joint_origin"


def test_generalized_packet_can_include_scalar_joint_without_reinterpreting_its_units():
    for kind in ("revolute", "continuous", "prismatic"):
        value = GeneralizedJointMeasurement("one", kind, [2], [3], coordinate_convention(kind), [4])
        assert value.q == (2.,) and value.v == (3.,) and value.effort == (4.,)
        assert value.coordinates["q_units"] == (("m",) if kind == "prismatic" else ("rad",))


def test_generalized_binding_accepts_mixed_scalar_and_multi_dof_capture():
    _, original, payload = observation_fixture()
    declaration = original.as_dict()
    declaration["links"].append("lift_link")
    declaration["joints"].append({"joint_id": "lift", "parent": "payload", "child": "lift_link",
        "type": "prismatic", "actuation": "passive", "axis": [0, 0, 1],
        "units": {"position": "m", "velocity": "m/s", "effort": "N"},
        "limits": {"lower": 0, "upper": .5}})
    declaration["sensors"][0]["joint_ids"].append("lift")
    body = EmbodimentDescriptor.from_dict(declaration)
    scalar = GeneralizedJointMeasurement("lift", "prismatic", [.8], [.2], coordinate_convention("prismatic"))
    # An out-of-bounds observation remains visible, not clipped into control admission.
    payload = replace(payload, joints=(*payload.joints, scalar), embodiment_sha256=body.sha256)
    _, bound = producer(body, payload)
    bound.validate_payload(payload)
    assert payload.joints[-1].q == (.8,) and payload.joints[-1].effort is None


@pytest.mark.parametrize("change", [
    lambda p: replace(p, joints=(*p.joints, p.joints[0])),
    lambda p: replace(p, joints=()),
    lambda p: replace(p, embodiment_sha256=None),
    lambda p: replace(p, joints=(replace(p.joints[0], joint_type="ball"),)),
    lambda p: replace(p, joints=(replace(p.joints[0], coordinates={}),)),
])
def test_generalized_payload_rejects_duplicate_missing_or_unknown_semantics(change):
    _, _, payload = observation_fixture()
    with pytest.raises(ValueError):
        change(payload)


def test_generalized_modality_requires_explicit_version_two_even_for_scalar_joints():
    cfg = load_robot_config("wheeled_lift_sensors", llm="mock")
    body = EmbodimentDescriptor.from_dict(cfg.embodiment.as_dict())
    entry = cfg.domains.sensing.as_dict()["providers"][0]
    descriptor = SensorDescriptor(sensor_id="joints", robot_id=body.robot_id, source="legacy-version",
        modality="generalized_joint_state", frame_id=entry["frame_id"], clock_domain="monotonic",
        measurement_kind="synthetic")
    with pytest.raises(ValueError, match="version 2"):
        EmbodimentBoundProvider(BufferedSensorProvider(descriptor), "sensing", body)


@pytest.mark.parametrize("modality", ["proprioception", "joint_state"])
def test_multi_dof_attachment_refuses_legacy_bindings_before_read(modality):
    _, body, payload = observation_fixture()
    descriptor = SensorDescriptor(sensor_id="joints", robot_id=body.robot_id, source="legacy",
        modality=modality, frame_id=payload.metadata.frame_id, clock_domain="monotonic", measurement_kind="synthetic")
    class NeverRead:
        def read(self):
            raise AssertionError("binding opened observation before refusing incompatible coordinates")
    raw = NeverRead()
    raw.descriptor = descriptor
    with pytest.raises(ValueError, match="multi-DoF"):
        EmbodimentBoundProvider(raw, "sensing", body)


@pytest.mark.parametrize("change", [
    lambda p: replace(p, embodiment_sha256="0"*64),
    lambda p: replace(p, joints=tuple(reversed(p.joints))),
    lambda p: replace(p, joints=(*p.joints[:2], replace(p.joints[2], joint_id="other"))),
    lambda p: replace(p, joints=(GeneralizedJointMeasurement("platform", "spherical", [1, 0, 0, 0],
        [0, 0, 0], coordinate_convention("spherical")), *p.joints[1:])),
])
def test_payload_identity_type_and_order_cannot_float_between_declarations(change):
    _, body, payload = observation_fixture()
    _, bound = producer(body, payload)
    with pytest.raises(SensorError):
        bound.validate_payload(change(payload))


def test_binding_keeps_original_capture_age_and_physical_identity():
    _, body, payload = observation_fixture()
    raw, bound = producer(body, payload)
    original = ObservationEnvelope(source="captured-fixture", sensor_id="joints", epoch="captured-epoch",
        sequence=7, clock_domain="simulation", capture_time_s=.035, received_monotonic_s=time.monotonic(),
        producer_age_s=.4, model_identity_sha256="a"*64, measurement_kind="physics", payload=payload)
    raw.publish(original)
    assert bound.read() is original and bound.read().producer_age_s == .4
    assert bound.read().payload is payload
    raw.publish(replace(original, producer_age_s=.6))
    hub = SensorHub()
    hub.register(bound)
    try:
        with pytest.raises(SensorError, match="stale"):
            hub.read("joints")
    finally:
        assert hub.close()["ok"]


@pytest.mark.parametrize("kind", ["planar", "spherical", "floating"])
def test_internal_multi_dof_cannot_hide_under_fixed_root_and_enable_legacy_physical_arm(kind):
    cfg = load_robot_config("fixed_so101_mock", llm="mock")
    raw = cfg._data["embodiment"]
    raw["version"] = 2
    raw["links"].append("uncontrolled_link")
    raw["joints"].append({"joint_id": "internal", "parent": raw["root_link"], "child": "uncontrolled_link",
                          "type": kind, "actuation": "passive", "coordinates": coordinate_convention(kind)})
    cfg._data["domains"]["manipulation"]["resolved"]["arms"][0]["type"] = "so101"
    assert raw["root_mode"] == "fixed"
    with pytest.raises(ValueError, match="dynamic frames"):
        describe_robot(cfg)


def test_synthetic_factory_refuses_wrong_digest_without_overwriting_it():
    cfg, body = configuration()
    profile = copy.deepcopy(cfg.domains.sensing.as_dict())
    profile["providers"][0]["values"]["embodiment_sha256"] = "f"*64
    with pytest.raises(SensorError, match="digest"):
        build_sensor_domain("sensing", profile, embodiment=body)
    assert plain_json(body.joints[0]["coordinates"]) == coordinate_convention("planar")
