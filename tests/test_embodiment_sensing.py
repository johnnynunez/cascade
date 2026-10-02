"""Mixed-unit readings retain producer evidence and bind declared structure."""
from dataclasses import replace
import time

import pytest

from cascade.config import load_robot_config
from cascade.robotics.embodiment import EmbodimentDescriptor
from cascade.sensing import (BufferedSensorProvider,
    MeasurementMetadata, ObservationEnvelope, ProprioceptionPayload, SensorDescriptor,
    SensorError, SensorHub, build_sensor_domain)
from cascade.sensing.embodiment import EmbodimentBoundProvider


def setup():
    cfg = load_robot_config("wheeled_lift_sensors")
    body = EmbodimentDescriptor.from_dict(cfg.embodiment.as_dict())
    profile = cfg.domains.sensing.as_dict()
    domain = build_sensor_domain("sensing", profile, embodiment=body)
    payload = domain.hub.read("joints").payload
    return body, profile, domain, payload


def test_mixed_payload_has_explicit_units_immutable_coordinates_and_unknown_effort():
    body, _, domain, payload = setup()
    try:
        assert payload.embodiment_sha256 == body.sha256
        assert [(j.position_unit, j.effort_unit) for j in payload.joints] == [("rad", "N*m"), ("rad", "N*m"), ("m", "N")]
        assert all(j.effort is None for j in payload.joints)
        assert type(payload.joints) is tuple
        with pytest.raises(ValueError):
            replace(payload.joints[2], effort_unit="N*m")
        with pytest.raises(ValueError):
            replace(payload.joints[2], position=float("nan"))
        with pytest.raises(ValueError):
            replace(payload.joints[2], position=True)
        with pytest.raises(ValueError):
            replace(payload, joints=(payload.joints[0], payload.joints[0]))
    finally:
        domain.close()


def test_binding_preserves_exact_capture_and_out_of_limit_measurement():
    body, _, domain, payload = setup()
    domain.close()
    # Deliberately outside declared .4m stroke: preserve measured violation.
    payload = replace(payload, joints=(*payload.joints[:2], replace(payload.joints[2], position=.8)))
    now = time.monotonic()
    original = ObservationEnvelope(source="fixture", sensor_id="joints", epoch="epoch", sequence=5,
        clock_domain="sim_time", capture_time_s=2., received_monotonic_s=now,
        producer_age_s=.4, model_identity_sha256="a"*64, measurement_kind="physics", payload=payload)
    descriptor = SensorDescriptor(sensor_id="joints", robot_id=body.robot_id, source="fixture",
        modality="joint_state", frame_id=payload.metadata.frame_id, clock_domain="sim_time",
        measurement_kind="physics", model_identity_sha256="a"*64, epoch="epoch")
    producer = BufferedSensorProvider(descriptor)
    producer.publish(original)
    bound = EmbodimentBoundProvider(producer, "sensing", body)
    assert bound.read() is original
    assert bound.read().payload.joints[2].position == .8
    assert bound.read().producer_age_s == .4
    producer.publish(replace(original, producer_age_s=.6))
    hub = SensorHub()
    hub.register(bound)
    with pytest.raises(SensorError, match="stale"):
        hub.read("joints")
    assert hub.close()["ok"]


@pytest.mark.parametrize("change", [
    lambda p: replace(p, embodiment_sha256="0"*64),
    lambda p: replace(p, joints=tuple(reversed(p.joints))),
    lambda p: replace(p, joints=(*p.joints[:2], replace(p.joints[2], joint_id="unknown"))),
    lambda p: replace(p, joints=(replace(p.joints[0], joint_type="revolute"), *p.joints[1:])),
])
def test_structural_mismatch_fails_before_observation_is_accepted(change):
    body, _, domain, payload = setup()
    domain.close()
    provider = type("Provider", (), {"descriptor": SensorDescriptor(
        sensor_id="joints", robot_id=body.robot_id, source="fixture", modality="joint_state",
        frame_id=payload.metadata.frame_id, clock_domain="monotonic", measurement_kind="synthetic")})()
    bound = EmbodimentBoundProvider(provider, "sensing", body)
    with pytest.raises(SensorError):
        bound.validate_payload(change(payload))


def test_legacy_angular_payload_remains_compatible_but_cannot_label_linear_axis():
    body, profile, domain, _ = setup()
    domain.close()
    entry = profile["providers"][0]
    entry.update(modality="proprioception", values=dict(joint_names=["left_wheel", "right_wheel", "lift"],
        position_rad=[0, 0, .1], velocity_rad_s=[0, 0, 0]))
    with pytest.raises(ValueError, match="prismatic"):
        build_sensor_domain("sensing", profile, embodiment=body)
    legacy = ProprioceptionPayload(MeasurementMetadata("joints"), ["joint"], [1], [2])
    assert legacy.position_rad == (1.,) and legacy.modality == "proprioception"


def test_synthetic_profile_requires_binding_and_does_not_overwrite_wrong_digest():
    body, profile, domain, _ = setup()
    domain.close()
    with pytest.raises(ValueError, match="embodiment"):
        build_sensor_domain("sensing", profile)
    profile["providers"][0]["values"]["embodiment_sha256"] = "1"*64
    with pytest.raises(SensorError, match="digest"):
        build_sensor_domain("sensing", profile, embodiment=body)


def test_legacy_angular_identity_mismatch_is_refused_with_matching_units():
    cfg = load_robot_config("fixed_so101_mock")
    body = EmbodimentDescriptor.from_dict(cfg.embodiment.as_dict())
    profile = cfg.domains.sensing.as_dict()
    entry = profile["providers"][0]
    names = list(body.sensors[0]["joint_ids"])
    entry.update(modality="proprioception", values=dict(joint_names=names,
        position_rad=[0]*len(names), velocity_rad_s=[0]*len(names)))
    domain = build_sensor_domain("sensing", profile, embodiment=body)
    assert domain.execute("read_sensor", {"sensor_id": "joints"})["ok"]
    domain.close()
    entry["values"]["joint_names"] = list(reversed(names))
    with pytest.raises(SensorError, match="identities/order"):
        build_sensor_domain("sensing", profile, embodiment=body)
