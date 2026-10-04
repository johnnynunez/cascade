"""Registered camera/base geometry and real domain dispatch; no physical admission."""
from dataclasses import replace
import math
from types import SimpleNamespace

import pytest

from cascade.spatial.frames import SpatialStamp, TransformSample, sha256
from cascade.spatial.registration import CameraBaseRegistration
from test_spatial_cuvslam import rig as rig


def config(**changes):
    record = dict(schema="cascade.rigid-camera-base.v1", robot_id="duck",
        model_identity_sha256="e"*64, camera_calibration_sha256="c"*64,
        camera_frame_id="optical", base_frame_id="base", translation_m=[.2, 0., .1],
        rotation_wxyz=[math.sqrt(.5), math.sqrt(.5), 0., 0.],
        position_error_m=.02, angular_error_rad=.03)
    record.update(changes)
    return {**record, "sha256": sha256(record)}


def registration(value=None):
    return CameraBaseRegistration(config() if value is None else value, robot_id="duck",
        descriptor=SimpleNamespace(model_identity_sha256="e"*64, calibration_id="c"*64, frame_id="optical"),
        map_frame_id="local-map")


def camera(**changes):
    stamp = SpatialStamp("room", "session1", "capture-clock", 1., "sensors/rgbd", "d"*64, "c"*64, "estimated")
    return TransformSample(**(dict(parent="local-map", child="optical", translation_m=(1., 2., 3.),
        rotation_wxyz=(math.sqrt(.5), 0., 0., math.sqrt(.5)), stamp=stamp,
        position_error_m=.01, angular_error_rad=.1) | changes))


def test_rigid_composition_rotates_lever_arm_and_retains_capture_identity():
    bound, pose = registration(), camera()
    base = bound.pose(pose)
    assert base.translation_m == pytest.approx((1., 2.2, 3.1))
    assert base.rotation_wxyz == pytest.approx((.5, .5, .5, .5))
    assert base.parent == pose.parent and base.child == "base" and not base.static
    assert base.stamp == replace(pose.stamp, calibration_id=bound.sha256)
    assert base.position_error_m == pytest.approx(.01+.02+2*math.sqrt(.05)*math.sin(.05))
    assert base.angular_error_rad == pytest.approx(.13)
    # A stationary optical origin is not a stationary base when the rig rotates.
    previous = bound.pose(replace(pose, rotation_wxyz=(1., 0., 0., 0.)))
    assert math.dist(previous.translation_m, base.translation_m) == pytest.approx(math.sqrt(.08))


@pytest.mark.parametrize("camera_change,mount_change", [
    ({"position_error_m": None}, {}), ({"angular_error_rad": None}, {}),
    ({}, {"position_error_m": None}),
])
def test_unknown_uncertainty_cannot_be_replaced_by_mount_bounds(camera_change, mount_change):
    result = registration(config(**mount_change)).pose(camera(**camera_change))
    assert result.position_error_m is None
    if "angular_error_rad" in camera_change:
        assert result.angular_error_rad is None


@pytest.mark.parametrize("change", [
    {"robot_id": "other"}, {"model_identity_sha256": "f"*64}, {"model_identity_sha256": None},
    {"camera_calibration_sha256": "f"*64}, {"camera_frame_id": "other"},
    {"base_frame_id": "optical"}, {"base_frame_id": "local-map"},
    {"rotation_wxyz": [0., 0., 0., 0.]}, {"position_error_m": -1.},
    {"angular_error_rad": math.pi+.01},
])
def test_registration_identity_and_geometry_fail_closed(change):
    with pytest.raises(ValueError):
        registration(config(**change))


def test_hash_and_detached_registration_are_not_mutable_authority():
    value = config(); bound = registration(value)
    value["translation_m"][0] = 99.
    with pytest.raises(ValueError, match="SHA256"):
        registration(value)
    output = bound.as_dict(); output["translation_m"][0] = 88.
    assert bound.pose(camera()).translation_m == pytest.approx((1., 2.2, 3.1))
    assert sha256({k:v for k,v in bound.as_dict().items() if k != "sha256"}) == bound.sha256
    for altered in (camera(static=True), camera(child="foreign"),
                    camera(stamp=replace(camera().stamp, calibration_id="f"*64))):
        with pytest.raises(ValueError, match="matching dynamic"):
            bound.pose(altered)


@pytest.mark.parametrize("rig", [{"base_registration": config()}], indirect=True)
def test_registered_base_uses_same_real_domain_capture_and_freshness(rig):
    from cascade.apps.robot_runtime import DomainAdapter
    from cascade.robotics.runtime import RobotRuntime
    from cascade.sensing.domain import SensorDomain
    sensors = SensorDomain("sensors", rig.hub)
    adapters = {name: DomainAdapter(name, {"kind": kind}, d.resources, d.tool_specs, frozenset(),
                    runtime=d, required_resources=getattr(d, "required_resources", ()))
                for name,kind,d in (("sensors", "sensors", sensors), ("space", "spatial", rig.domain))}
    runtime = RobotRuntime(adapters)
    try:
        args = rig.publish()
        result = runtime.execute("space.track_capture", args)
        assert result["ok"], result
        assert result["pose"]["translation_m"] == [1., 2., 3.]
        assert result["base_pose"]["translation_m"] == pytest.approx([.8, 2., 3.1])
        assert result["base_pose"]["stamp"]["source_sha256"] == args["capture_sha256"]
        assert result["base_pose"]["stamp"]["epoch"] == rig.domain.map_epoch
        assert result["base_pose"]["position_error_m"] is None
        assert result["base_pose"]["angular_error_rad"] is None
        assert not result["physical_admission"]
        assert "registered_base_localization" in rig.domain.resources[0].capabilities
        result["base_pose"]["translation_m"][0] = 88.
        read = runtime.execute("space.get_localization", {})
        assert read["base_pose"]["translation_m"][0] == pytest.approx(.8)
        assert len(rig.sdk.calls) == 1
        rig.now[0] += .501
        stale = runtime.execute("space.get_localization", {})
        assert not stale["ok"] and stale["map_epoch_invalidated"]
        assert rig.domain._latest is None
    finally:
        runtime.close()


@pytest.mark.parametrize("rig", [{"base_registration": config()}], indirect=True)
@pytest.mark.parametrize("failure", ["tracking_lost", "stop", "sensor_epoch", "replay"])
def test_registered_base_does_not_survive_tracking_epoch_or_stop_failure(rig, failure):
    args = rig.publish()
    assert rig.domain.execute("track_capture", args)["ok"]
    if failure == "stop":
        rig.domain.stop()
        rejected = rig.domain.execute("get_localization", {})
    elif failure == "replay":
        rejected = rig.domain.execute("track_capture", args)
    else:
        if failure == "tracking_lost":
            rig.sdk.effect = lambda result: (result[0], None)
            obs = replace(rig.obs, sequence=8, capture_time_s=.4)
            args = rig.publish(obs)
        else:
            args = dict(args, epoch="other")
        rejected = rig.domain.execute("track_capture", args)
    assert not rejected["ok"] and rig.domain._latest is None
    assert not rig.domain.execute("get_localization", {})["ok"]


@pytest.mark.parametrize("rig", [{"base_registration": config()}], indirect=True)
def test_expiry_during_base_composition_cannot_return_a_fresh_pose(rig, monkeypatch):
    original = CameraBaseRegistration.pose
    def delayed(registration, camera_pose):
        result = original(registration, camera_pose)
        rig.now[0] += .501
        return result
    monkeypatch.setattr(CameraBaseRegistration, "pose", delayed)
    rejected = rig.domain.execute("track_capture", rig.publish())
    assert not rejected["ok"] and rejected["map_epoch_invalidated"]
    assert "base_pose" not in rejected and rig.domain._latest is None


@pytest.mark.parametrize("rig", [{"base_registration": config()}], indirect=True)
def test_builder_binds_registration_without_starting_the_sdk(rig, monkeypatch):
    from cascade.spatial import cuvslam
    from cascade.spatial.domain import build_spatial_domain
    monkeypatch.setattr(cuvslam, "CuVslamProcess", lambda *a, **k: pytest.fail("passive builder started SDK"))
    profile = {"kind": "spatial", "robot_id": "duck", "cuvslam": rig.config}
    built = build_spatial_domain("other", profile, sensor_domains={"sensors": SimpleNamespace(hub=rig.hub)})
    try:
        assert built._tracker is None
        resource = built.resources[0]
        assert resource.metadata["base_registration"]["sha256"] == config()["sha256"]
        assert not resource.metadata["physical_admission"]
    finally:
        built.close()
