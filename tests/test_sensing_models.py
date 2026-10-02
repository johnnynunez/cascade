"""Modality semantics are software contracts, not physical sensor acceptance."""
from dataclasses import FrozenInstanceError, replace
import json
import struct

import pytest

from cascade.sensing import (EstimatedTactilePayload, ImuPayload, MeasurementMetadata,
    ObservationEnvelope, ProprioceptionPayload, RgbdPayload, RgbPayload,
    SolvedContactPayload, TactileImagePayload)
from mobile_support_fixture import support


def observation(payload=None, **changes):
    values = dict(source="fixture", sensor_id="imu", epoch="episode", sequence=1,
                  clock_domain="simulation", capture_time_s=.005, received_monotonic_s=10.,
                  producer_age_s=.1, model_identity_sha256="e" * 64, measurement_kind="physics",
                  payload=ImuPayload(MeasurementMetadata("body"), [0, 0, 0]) if payload is None else payload)
    return ObservationEnvelope(**(values | changes))


def test_nested_payload_is_immutable_and_unknown_channels_are_not_zero():
    source = [1, 2, 3]
    payload = ImuPayload(MeasurementMetadata("body"), source)
    source[0] = 100
    value = observation(payload)
    assert value.payload.angular_velocity_rad_s == (1., 2., 3.)
    with pytest.raises(FrozenInstanceError):
        value.payload.metadata.frame_id = "world"
    exported = value.as_dict()
    assert exported["payload"]["linear_acceleration_m_s2"] is None
    assert exported["payload"]["metadata"] == {"frame_id": "body", "calibration_id": None, "saturated": None}
    assert exported["payload"]["units"]["angular_velocity_rad_s"] == "rad/s"
    exported["payload"]["angular_velocity_rad_s"][0] = 500
    assert value.payload.angular_velocity_rad_s[0] == 1
    assert value.age_s(10.4) == pytest.approx(.5)


@pytest.mark.parametrize("changes", [
    {"schema_version": True}, {"schema_version": 2}, {"sequence": -1}, {"sequence": True},
    {"capture_time_s": float("nan")}, {"producer_age_s": -1}, {"model_identity_sha256": None},
    {"model_identity_sha256": "E" * 64}, {"measurement_kind": "accepted"}, {"epoch": ""},
    {"payload": {}},
])
def test_invalid_envelopes_never_coerce_or_invent_identity(changes):
    with pytest.raises(ValueError):
        observation(**changes)


def test_proprioception_pins_order_and_lengths_without_fabricated_effort():
    payload = ProprioceptionPayload(MeasurementMetadata("joints", "calibration-v1", False),
                                   ["joint_b", "joint_a"], [2, 1], [0, 0])
    assert payload.joint_names == ("joint_b", "joint_a")
    assert payload.position_rad == (2, 1)
    assert payload.effort_nm is None
    with pytest.raises(ValueError):
        replace(payload, joint_names=["joint_a", "joint_a"])
    with pytest.raises(ValueError):
        replace(payload, velocity_rad_s=[0])
    with pytest.raises(ValueError):
        replace(payload, effort_nm=[0, float("inf")])


def test_solved_contact_is_same_solve_and_never_an_estimated_tactile_alias():
    record = support(1, .005) | {"epoch": "episode", "model_identity_sha256": "e" * 64}
    solved = SolvedContactPayload(MeasurementMetadata("world"), record)
    record["contacts"][0]["normal_force_n"] = 100
    assert solved.observation.contacts[0].normal_force_n == 3
    value = observation(solved)
    assert value.as_dict()["payload"]["modality"] == "solved_contact"
    for changes in ({"epoch": "other"}, {"sequence": 2}, {"capture_time_s": .006},
                    {"model_identity_sha256": "f" * 64}, {"clock_domain": "monotonic"}):
        with pytest.raises(ValueError, match="differs"):
            replace(value, **changes)
    with pytest.raises(ValueError, match="world"):
        replace(solved, metadata=MeasurementMetadata("body"))
    estimated = EstimatedTactilePayload(MeasurementMetadata("finger", "taxel-cal"),
                                        "sdf-penalty-v1", [[0, 0, 0]], [3], [[0, 0]])
    image = TactileImagePayload(MeasurementMetadata("finger"), 1, 1, "gray8", b"\x10", "render-v1")
    assert {solved.modality, estimated.modality, image.modality} == {
        "solved_contact", "estimated_tactile", "tactile_image"}


def test_unavailable_solved_channel_stays_unavailable_not_known_empty():
    record = dict(version=1, status="unavailable", reason="force mapping unknown", epoch="episode",
                  step=1, sim_time_s=.005, model_identity_sha256="e" * 64, contacts=[])
    value = observation(SolvedContactPayload(MeasurementMetadata("world"), record))
    assert value.as_dict()["payload"]["observation"]["status"] == "unavailable"


def test_registered_rgbd_has_bounded_immutable_depth_and_explicit_pixel_intrinsics():
    payload = RgbdPayload(MeasurementMetadata("camera", "cal-v2"), 2, 1,
                         b"\x01\x02\x03" * 2, struct.pack("<ff", .1, 0),
                         [100, 0, .5, 0, 100, 0, 0, 0, 1])
    encoded = observation(payload).as_dict()["payload"]
    assert encoded["units"]["depth_m_f32le"] == "m"
    assert encoded["depth_m_f32le"]["encoding"] == "base64"
    json.dumps(encoded, allow_nan=False)
    for changes in ({"depth_m_f32le": struct.pack("<ff", -.1, 0)},
                    {"depth_m_f32le": struct.pack("<ff", float("nan"), 0)},
                    {"depth_m_f32le": b""}, {"rgb8": bytearray(6)},
                    {"intrinsics": [100, 0, 0, 0, 100, 0, 1, 0, 1]}):
        with pytest.raises(ValueError):
            replace(payload, **changes)
    with pytest.raises(ValueError):
        RgbPayload(MeasurementMetadata("camera"), 2048, 2048, "rgb8", b"0")
