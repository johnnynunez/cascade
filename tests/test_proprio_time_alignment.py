"""B50: capture-time alignment of camera, IMU and proprioception samples.

ROADMAP Perception row: "validate IMU/proprioception fusion and time alignment
while preserving missing, stale and uncertain observations". The sensing layer
already types IMU/proprioception captures (epoch, sequence, clock domain,
capture time), but nothing PAIRED a camera capture with the IMU/joint sample
taken at its capture time: `read_sensor` returns each sensor's latest capture
independently. B39's link self-mask had the only pairing (nearest joint sample
within `max_skew_s`, never interpolated), private to that module.

CPU only: synthetic sample streams (`BufferedSensorProvider` envelopes on a
simulation clock), the real loopback mobile bridge (state socket + RGB-D frame
cache, as `scripts/isaac_microduck_bridge.py` serves them) and the real MCP
composed-robot path. No GPU, no physics engine, no MuJoCo.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.sensing import (BufferedSensorProvider, ImuPayload, JointStatePayload, MeasurementMetadata,
                             ObservationEnvelope, ProprioceptionPayload, RgbdPayload, SensorDescriptor,
                             SyntheticSensorProvider, build_sensor_domain)

EPOCH = "episode-1"
MODEL = "e" * 64
#: This item's port block (CHILD_RULES): the sensors-only runtime dials none of
#: the sidecars, the variables only make sure nothing reaches a shared port.
PORTS = {"CASCADE_GRASPGENX_PORT": "46301", "CASCADE_OCCUPANCY_PORT": "46302",
         "CASCADE_BRIDGE_PORT": "46303", "CASCADE_HUG_PORT": "46304"}


def _al():
    from cascade.sensing import alignment
    return alignment


# ── synthetic sample streams ────────────────────────────────────────────────


def _rgbd(frame_id="camera:overview"):
    return RgbdPayload(MeasurementMetadata(frame_id), 2, 2, bytes(range(12)),
                       np.array([.5, .6, 0., .7], dtype="<f4").tobytes(), [100., 0., 1., 0., 100., 1., 0., 0., 1.])


def _imu(gyro=(0., 0., .1), saturated=None):
    return ImuPayload(MeasurementMetadata("body", None, saturated), list(gyro))


def _joints(velocity=(0., .5)):
    return ProprioceptionPayload(MeasurementMetadata("joints"), ["j0", "j1"], [.1, .2], list(velocity))


def _env(sensor_id, payload, *, t, seq, epoch=EPOCH, clock="simulation", received=None):
    return ObservationEnvelope(source="fixture", sensor_id=sensor_id, epoch=epoch, sequence=seq,
                               clock_domain=clock, capture_time_s=t,
                               received_monotonic_s=time.monotonic() if received is None else received,
                               producer_age_s=.001, model_identity_sha256=MODEL, measurement_kind="physics",
                               payload=payload)


def _descriptor(sensor_id, modality, frame_id, clock="simulation"):
    return SensorDescriptor(sensor_id=sensor_id, robot_id="duck", source="fixture", modality=modality,
                            frame_id=frame_id, clock_domain=clock, measurement_kind="physics",
                            model_identity_sha256=MODEL, max_age_s=5., read_timeout_s=2.)


_STREAMS = {"overview": ("rgbd", "camera:overview"), "imu": ("imu", "body"), "joints": ("proprioception", "joints")}


def _domain(alignment=None, clocks=None):
    clocks = clocks or {}
    providers = {name: BufferedSensorProvider(_descriptor(name, modality, frame, clocks.get(name, "simulation")))
                 for name, (modality, frame) in _STREAMS.items()}
    profile = {"kind": "sensors", "robot_id": "duck",
               "providers": [{"id": name, "kind": "injected"} for name in _STREAMS]}
    if alignment is not None:
        profile["alignment"] = alignment
    return build_sensor_domain("sensing", profile, providers=dict(providers)), providers


@pytest.fixture
def aligned_domain():
    """A factory, so a refused `alignment:` block fails the test (not its setup)."""
    built = []

    def make():
        built.append(_domain({"max_skew_s": .0075, "max_rotation_rad": .01}))
        return built[-1]

    yield make
    for domain, _ in built:
        assert domain.close()["ok"]


def _admit(domain, providers, name, envelope):
    providers[name].publish(envelope)
    result = domain.execute("read_sensor", {"sensor_id": name})
    assert result["ok"], result


def _aligned(domain, **args):
    return domain.execute("read_aligned", {"reference": "overview", **args})


# ── premises (pass on main by design) ───────────────────────────────────────


def test_premise_read_sensor_returns_each_latest_capture_with_no_pairing():
    """The gap: two independent reads of one simulated instant return captures
    from DIFFERENT steps, and nothing says how far apart they are."""
    domain, providers = _domain()
    try:
        assert [s["name"] for s in domain.tool_specs] == ["list_sensors", "read_sensor"]
        providers["overview"].publish(_env("overview", _rgbd(), t=.050, seq=10))
        providers["imu"].publish(_env("imu", _imu(), t=.055, seq=11))
        refused = domain.execute("read_aligned", {"reference": "overview"})
        assert refused["ok"] is False and "invalid arguments" in refused["error"]
        assert domain.hub.history() == ()                      # nothing was read
        camera = domain.execute("read_sensor", {"sensor_id": "overview"})
        imu = domain.execute("read_sensor", {"sensor_id": "imu"})
        assert set(camera) == set(imu) == {"ok", "observation", "capture_sha256"}
        assert camera["observation"]["capture_time_s"] != imu["observation"]["capture_time_s"]
    finally:
        domain.close()


def test_premise_the_hub_history_retains_several_captures_per_sensor_with_their_clocks():
    """Pairing needs no new retention: admitted captures keep their own capture
    time, sequence and epoch in the hub's bounded history."""
    domain, providers = _domain()
    try:
        for seq, t in ((8, .040), (11, .055)):
            _admit(domain, providers, "imu", _env("imu", _imu(), t=t, seq=seq))
        assert [(o.sequence, o.capture_time_s, o.epoch) for o in domain.hub.history("imu")] == [
            (8, .040, EPOCH), (11, .055, EPOCH)]
    finally:
        domain.close()


@pytest.mark.parametrize("skew, reason", [(.125, "link_mask"), (-.25, "stale_joint_state"), (None, "no_joint_state")])
def test_golden_link_self_mask_time_alignment_is_unchanged(skew, reason):
    """B39's pairing (now on the shared component): inclusive bound, stale and
    missing build no mask, the per-arm record keeps its exact shape."""
    from cascade.perception import link_mask as lm
    mask = lm.LinkSelfMask(dilate_px=0, max_skew_s=.125)
    state = None if skew is None else SimpleNamespace(q=np.zeros(2), t=1000.0 + skew)
    mask.add_arm("arm0", SimpleNamespace(pieces={}), lambda s: [{}], lambda: state)
    frame = SimpleNamespace(t=1000.0, rgb=np.zeros((4, 4, 3), np.uint8), K=np.eye(3), robot_mask=None)
    out = mask.mask_for(frame, np.eye(4))
    assert (out is not None) == (reason == "link_mask")
    assert mask.last["reason"] == reason
    assert mask.last["arms"]["arm0"] == {"reason": reason, "skew_s": skew,
                                         "error": None if skew is not None else
                                         "no joint state (the arm is in standby or not readable)"}


#: Sensors-domain tool specs of every robot profile shipped with a sensors
#: domain, computed on unchanged origin/main be57535. Without `alignment:` the
#: catalog must stay byte-identical.
GOLDEN_SENSOR_TOOLS = "30888f10a4d6a06318985a4a65625d08d857e03466f93f5744ffa656697e1908"
SENSOR_PROFILES = ("conversation_mock", "fixed_so101_mock", "generalized_joint_sensors", "mixed_mock",
                   "wheeled_lift_sensors")


def test_golden_shipped_sensor_domains_keep_their_exact_tool_catalog():
    from cascade.apps.robot_runtime import describe_robot
    from cascade.config import load_robot_config
    rows = []
    for profile in SENSOR_PROFILES:
        for name, adapter in sorted(describe_robot(load_robot_config(profile)).items()):
            if adapter.profile["kind"] == "sensors":
                assert "alignment" not in adapter.profile
                rows.append([profile, name, adapter.tool_specs,
                             [t.as_dict() for t in adapter.tool_descriptors]])
                adapter.runtime.close()
    assert len(rows) == 5
    blob = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(blob.encode()).hexdigest() == GOLDEN_SENSOR_TOOLS


# ── the shared component ────────────────────────────────────────────────────


def test_nearest_never_interpolates_and_ties_keep_the_earlier_sample():
    al = _al()
    buf = al.SampleBuffer(maxlen=3)
    for k, t in enumerate((10., 11., 12., 13.)):
        assert buf.add(t, f"s{k}")
    assert len(buf) == 3                               # 10.0 was evicted
    assert buf.nearest(10.2) == ("s1", pytest.approx(.8))
    assert buf.nearest(12.5) == ("s2", pytest.approx(-.5))   # tie: the earlier sample
    assert buf.nearest(13.4) == ("s3", pytest.approx(-.4))
    assert not buf.add(float("nan"), "x") and not buf.add("soon", "x") and not buf.add(None, "x")
    assert len(buf) == 3
    assert al.SampleBuffer().nearest(1.) is None
    items = [(1., "a"), (3., "b")]
    assert al.nearest(items, 2.) == ((1., "a"), 1. - 2.)
    assert al.nearest(items, 2.6) == ((3., "b"), pytest.approx(.4))
    assert al.nearest([], 2.) is None


def test_link_mask_samples_are_the_shared_buffer_not_a_copy():
    from cascade.perception import link_mask as lm
    al = _al()
    assert issubclass(lm.JointSamples, al.SampleBuffer)
    assert lm.JointSamples.nearest is al.SampleBuffer.nearest


def test_classify_four_states_with_the_measured_skew():
    al = _al()
    assert {al.ALIGNED, al.STALE, al.MISSING, al.UNCERTAIN} == set(al.STATUSES) == {
        "aligned", "stale", "missing", "uncertain"}
    missing = al.classify(5., None, max_skew_s=.01)
    assert (missing.status, missing.skew_s, missing.sample, missing.candidate) == ("missing", None, None, None)
    edge = al.classify(5., ("s", .01), max_skew_s=.01)          # the bound is inclusive
    assert (edge.status, edge.skew_s, edge.sample) == ("aligned", .01, "s")
    stale = al.classify(5., ("s", -.0125), max_skew_s=.01)
    assert (stale.status, stale.skew_s, stale.sample, stale.candidate) == ("stale", -.0125, None, "s")
    assert stale.reasons == ("skew_exceeds_bound",)
    # first-order motion over the skew: rate x |skew|, per displacement unit
    calm = al.classify(5., ("s", -.004), max_skew_s=.01, rates={"rad": .5, "m": .1},
                       tolerances={"rad": .0025, "m": .0005})
    assert calm.status == "aligned" and calm.reasons == ()
    assert calm.motion == {"rad": pytest.approx(.002), "m": pytest.approx(.0004)}
    fast = al.classify(5., ("s", -.006), max_skew_s=.01, rates={"rad": .5}, tolerances={"rad": .0025})
    assert (fast.status, fast.sample, fast.reasons) == ("uncertain", "s", ("motion_exceeds_tolerance",))
    at_tolerance = al.classify(5., ("s", .005), max_skew_s=.01, rates={"rad": .5}, tolerances={"rad": .0025})
    assert at_tolerance.motion == {"rad": .0025} and at_tolerance.status == "aligned"   # "exceeds" is strict
    linear = al.classify(5., ("s", .006), max_skew_s=.01, rates={"rad": 0., "m": .1}, tolerances={"rad": .1, "m": .0005})
    assert linear.status == "uncertain"
    untested = al.classify(5., ("s", .006), max_skew_s=.01, rates={"rad": 50.}, tolerances={"m": .0005})
    assert untested.status == "aligned"                         # no rotation tolerance declared
    saturated = al.classify(5., ("s", 0.), max_skew_s=.01, saturated=True)
    assert (saturated.status, saturated.reasons) == ("uncertain", ("saturated",))
    assert al.classify(5., ("s", 0.), max_skew_s=.01, saturated=None).status == "aligned"
    assert al.classify(5., ("s", 0.), max_skew_s=.01, saturated=False).status == "aligned"
    assert fast.as_dict() == {"status": "uncertain", "skew_s": -.006, "reasons": ["motion_exceeds_tolerance"],
                              "motion": {"rad": pytest.approx(.003)}}


def test_rates_are_measured_magnitudes_and_missing_channels_stay_absent():
    from cascade.robotics.joint_coordinates import coordinate_convention
    from cascade.sensing import GeneralizedJointStatePayload
    al = _al()
    assert al.rates(_imu((3., 0., 4.))) == {"rad": 5.}                      # |angular velocity|
    assert al.rates(_joints((.2, -.7))) == {"rad": .7}
    joint = dict(position_unit="rad", velocity_unit="rad/s", effort_unit="N*m")
    slide = dict(position_unit="m", velocity_unit="m/s", effort_unit="N")
    state = JointStatePayload(MeasurementMetadata("joints"), [
        dict(joint_id="a", joint_type="revolute", position=0., velocity=-.3, **joint),
        dict(joint_id="b", joint_type="continuous", position=0., velocity=.2, **joint, effort=1.),
        dict(joint_id="lift", joint_type="prismatic", position=.1, velocity=-.05, **slide)], "a" * 64)
    assert al.rates(state) == {"rad": .3, "m": .05}
    assert al.absent_channels(state) == ["a.effort", "lift.effort"]
    floating = dict(joint_id="root", joint_type="floating", q=[0, 0, 0, 1, 0, 0, 0], v=[3., 0., 4., 0., .6, .8],
                    coordinates=coordinate_convention("floating"))
    hinge = dict(joint_id="knee", joint_type="revolute", q=[0.], v=[-2.], coordinates=coordinate_convention("revolute"),
                 effort=[1.])
    general = GeneralizedJointStatePayload(MeasurementMetadata("joints"), [floating, hinge], "a" * 64)
    assert al.rates(general) == {"rad": 2., "m": 5.}
    assert al.absent_channels(general) == ["root.effort"]
    assert al.rates(_rgbd()) == {}
    assert al.absent_channels(_imu()) == ["linear_acceleration_m_s2"]
    assert al.absent_channels(_joints()) == ["effort_nm"]
    assert al.absent_channels(ImuPayload(MeasurementMetadata("body"), [0, 0, 0], [0, 0, 9.81])) == []


def test_clocks_are_comparable_only_on_one_clock_instance():
    al = _al()
    ref = _env("overview", _rgbd(), t=1., seq=1)
    assert al.comparable(ref, _env("imu", _imu(), t=1., seq=1))
    assert not al.comparable(ref, _env("imu", _imu(), t=1., seq=1, epoch="episode-0"))   # a reset world
    assert not al.comparable(ref, _env("imu", _imu(), t=1., seq=1, clock="monotonic"))
    local = _env("overview", _rgbd(), t=1., seq=1, epoch="a", clock="monotonic")
    assert al.comparable(local, _env("imu", _imu(), t=1., seq=1, epoch="b", clock="monotonic"))


@pytest.mark.parametrize("block, match", [
    ({}, "max_skew_s"), ({"max_skew_s": 0}, "max_skew_s"), ({"max_skew_s": -.01}, "max_skew_s"),
    ({"max_skew_s": True}, "max_skew_s"), ({"max_skew_s": 1.5}, "max_skew_s"), ({"max_skew_s": float("nan")}, "max_skew_s"),
    ({"max_skew_s": "0.01"}, "max_skew_s"), ({"max_skew_s": .01, "max_rotation_rad": 0}, "max_rotation_rad"),
    ({"max_skew_s": .01, "max_translation_m": -1}, "max_translation_m"),
    ({"max_skew_s": .01, "interpolate": True}, "unknown alignment"), ([.01], "mapping"),
])
def test_the_alignment_block_is_strict(block, match):
    with pytest.raises(ValueError, match=match):
        _domain(block)


def test_policy_reports_its_bounds_and_no_interpolation():
    al = _al()
    policy = al.AlignmentPolicy.from_profile({"max_skew_s": .02, "max_translation_m": .005})
    assert policy.as_dict() == {"max_skew_s": .02, "max_rotation_rad": None, "max_translation_m": .005,
                                "selection": "nearest_admitted_capture", "interpolation": "none"}
    assert policy.tolerances == {"rad": None, "m": .005}
    assert al.AlignmentPolicy.from_profile({"max_skew_s": 1}).max_skew_s == 1.0


# ── the consumer: sensing.read_aligned on synthetic streams ─────────────────


def test_camera_pairs_with_the_nearest_imu_sample_verbatim_and_missing_stays_missing(aligned_domain):
    domain, providers = aligned_domain()
    assert [s["name"] for s in domain.tool_specs] == ["list_sensors", "read_sensor", "read_aligned"]
    _admit(domain, providers, "imu", _env("imu", _imu((0., 0., .1)), t=.040, seq=8))
    providers["imu"].publish(_env("imu", _imu((0., 0., .3)), t=.055, seq=11))
    providers["overview"].publish(_env("overview", _rgbd(), t=.050, seq=10))
    result = _aligned(domain)
    assert result["ok"], result
    ref = result["reference"]
    assert (ref["sensor_id"], ref["sequence"], ref["capture_time_s"], ref["epoch"]) == ("overview", 10, .050, EPOCH)
    assert ref["capture_sha256"] == domain.hub.history("overview")[-1].sha256
    assert ref["observation"]["payload"]["modality"] == "rgbd"
    imu = result["pairings"]["imu"]
    assert imu["status"] == "aligned" and imu["read_error"] is None
    assert imu["skew_s"] == pytest.approx(.005, abs=1e-12) and imu["sequence"] == 11
    # never interpolated: the gyro of ONE admitted capture, not 0.233 rad/s
    assert imu["observation"]["payload"]["angular_velocity_rad_s"] == [0., 0., .3]
    assert imu["capture_sha256"] == domain.hub.history("imu")[-1].sha256
    assert imu["motion"] == {"rad": pytest.approx(.3 * .005)}
    # missing channels and sensors stay missing: no zeros, no older capture
    assert imu["observation"]["payload"]["linear_acceleration_m_s2"] is None
    assert imu["absent"] == ["linear_acceleration_m_s2"]
    joints = result["pairings"]["joints"]
    assert joints["status"] == "missing" and joints["skew_s"] is None
    assert "observation" not in joints and "sequence" not in joints
    assert "no completed capture available" in joints["read_error"]
    assert joints["reasons"] == ["no_admitted_capture"]
    assert result["counts"] == {"aligned": 1, "stale": 0, "missing": 1, "uncertain": 0}
    assert result["policy"]["interpolation"] == "none" and result["policy"]["max_skew_s"] == .0075


def test_a_sample_beyond_the_bound_is_stale_and_its_value_is_withheld(aligned_domain):
    domain, providers = aligned_domain()
    _admit(domain, providers, "imu", _env("imu", _imu(), t=.040, seq=8))
    _admit(domain, providers, "joints", _env("joints", _joints((0., 0.)), t=.050, seq=10))
    providers["overview"].publish(_env("overview", _rgbd(), t=.050, seq=10))
    result = _aligned(domain, sensor_ids=["imu", "joints"])
    imu = result["pairings"]["imu"]
    assert imu["status"] == "stale" and imu["skew_s"] == pytest.approx(-.010, abs=1e-12)
    assert imu["sequence"] == 8 and imu["capture_time_s"] == .040
    assert "observation" not in imu and "absent" not in imu
    assert "replay" in imu["read_error"]                  # the producer had nothing newer
    assert result["pairings"]["joints"]["status"] == "aligned"
    assert result["pairings"]["joints"]["skew_s"] == 0.
    assert result["counts"] == {"aligned": 1, "stale": 1, "missing": 0, "uncertain": 0}


def test_motion_over_the_skew_beyond_tolerance_is_uncertain_but_preserved(aligned_domain):
    domain, providers = aligned_domain()
    providers["imu"].publish(_env("imu", _imu((0., 0., 3.)), t=.045, seq=9))
    providers["joints"].publish(_env("joints", _joints((0., 1.)), t=.045, seq=9))
    providers["overview"].publish(_env("overview", _rgbd(), t=.050, seq=10))
    result = _aligned(domain)
    imu, joints = result["pairings"]["imu"], result["pairings"]["joints"]
    assert imu["status"] == "uncertain" and imu["reasons"] == ["motion_exceeds_tolerance"]
    assert imu["motion"]["rad"] == pytest.approx(.015) and imu["skew_s"] == pytest.approx(-.005, abs=1e-12)
    assert imu["observation"]["payload"]["angular_velocity_rad_s"] == [0., 0., 3.]   # preserved, flagged
    assert joints["status"] == "aligned" and joints["motion"]["rad"] == pytest.approx(.005)
    assert joints["absent"] == ["effort_nm"]
    assert result["counts"]["uncertain"] == 1


def test_a_saturated_sample_is_uncertain_however_close(aligned_domain):
    domain, providers = aligned_domain()
    providers["imu"].publish(_env("imu", _imu((0., 0., 0.), saturated=True), t=.050, seq=10))
    providers["overview"].publish(_env("overview", _rgbd(), t=.050, seq=10))
    imu = _aligned(domain, sensor_ids=["imu"])["pairings"]["imu"]
    assert (imu["status"], imu["skew_s"], imu["reasons"]) == ("uncertain", 0., ["saturated"])
    assert imu["observation"]["payload"]["metadata"]["saturated"] is True


def test_another_clock_instance_is_uncertain_never_aligned():
    clock = {"imu": "monotonic"}
    domain, providers = _domain({"max_skew_s": .0075}, clocks=clock)
    try:
        # a capture from a reset world (another epoch) sits AT the camera's time
        _admit(domain, providers, "joints", _env("joints", _joints(), t=.050, seq=3, epoch="episode-0"))
        providers["imu"].publish(_env("imu", _imu(), t=.050, seq=4, clock="monotonic"))
        providers["overview"].publish(_env("overview", _rgbd(), t=.050, seq=10))
        result = _aligned(domain)
        for name in ("imu", "joints"):
            pair = result["pairings"][name]
            assert pair["status"] == "uncertain" and pair["skew_s"] is None, name
            assert pair["reasons"] == ["clock_not_comparable"] and "observation" in pair
            assert pair["motion"] == {}
        assert result["pairings"]["joints"]["epoch"] == "episode-0"
        assert result["counts"] == {"aligned": 0, "stale": 0, "missing": 0, "uncertain": 2}
    finally:
        domain.close()


def test_an_incomparable_capture_is_judged_by_its_age_and_the_latest_one_is_reported():
    """Without a common clock, the skew is unknowable: the LATEST admitted
    capture is reported as uncertain while it is younger than the sensor's
    max age, stale (value withheld) after."""
    al = _al()
    policy = al.AlignmentPolicy.from_profile({"max_skew_s": .0075})
    reference = _env("overview", _rgbd(), t=.050, seq=10)
    old, latest = (_env("imu", _imu((0., 0., k)), t=.050 + k, seq=k, epoch="episode-0", received=100.)
                   for k in (1, 2))
    young = al.align_capture(reference, [old, latest], policy, now=100.4, max_age_s=.5)
    assert (young.status, young.sample, young.skew_s) == ("uncertain", latest, None)
    assert young.reasons == ("clock_not_comparable",) and young.motion == {}
    aged = al.align_capture(reference, [old, latest], policy, now=100.6, max_age_s=.5)
    assert (aged.status, aged.sample, aged.candidate) == ("stale", None, latest)
    assert aged.reasons == ("clock_not_comparable", "older_than_max_age")
    assert al.align_capture(reference, [], policy, now=100.6, max_age_s=.5).status == "missing"
    # one comparable capture is preferred over any incomparable one
    same = _env("imu", _imu(), t=.0525, seq=3, received=100.)
    chosen = al.align_capture(reference, [old, same, latest], policy, now=100.6, max_age_s=.5)
    assert (chosen.status, chosen.sample) == ("aligned", same)
    assert chosen.skew_s == pytest.approx(.0025, abs=1e-12)


def test_the_consumer_judges_ages_on_the_hub_clock():
    """read_aligned through a hub with its own clock: the same incomparable
    capture is uncertain at 0.4 s old and stale (withheld) at 0.6 s."""
    from cascade.sensing import SensorDomain, SensorHub
    al = _al()
    now = [100.4]
    hub = SensorHub(clock=lambda: now[0])
    camera = BufferedSensorProvider(_descriptor("overview", "rgbd", "camera:overview"))
    joints = BufferedSensorProvider(SensorDescriptor(
        sensor_id="joints", robot_id="duck", source="fixture", modality="proprioception", frame_id="joints",
        clock_domain="simulation", measurement_kind="physics", model_identity_sha256=MODEL, max_age_s=.5))
    hub.register(camera)
    hub.register(joints)
    domain = SensorDomain("sensing", hub, alignment=al.AlignmentPolicy.from_profile({"max_skew_s": .0075}))
    try:
        joints.publish(_env("joints", _joints(), t=.050, seq=3, epoch="episode-0", received=100.))
        camera.publish(_env("overview", _rgbd(), t=.050, seq=10, received=100.35))
        young = domain.execute("read_aligned", {"reference": "overview"})["pairings"]["joints"]
        assert young["status"] == "uncertain" and young["read_error"] is None
        now[0] = 100.6
        camera.publish(_env("overview", _rgbd(), t=.060, seq=12, received=100.55))
        aged = domain.execute("read_aligned", {"reference": "overview"})["pairings"]["joints"]
        assert aged["status"] == "stale" and "observation" not in aged and aged["sequence"] == 3
        assert aged["reasons"] == ["clock_not_comparable", "older_than_max_age"]
        assert "replay" in aged["read_error"]
    finally:
        assert hub.close(2)["ok"]


def test_synthetic_streams_on_the_local_clock_align_across_provider_epochs():
    now = [time.monotonic() - 1.]
    imu = SyntheticSensorProvider("imu", "duck", _imu(), period_s=.02, max_age_s=5., read_timeout_s=2.,
                                  clock=lambda: now[0])
    joints = SyntheticSensorProvider("joints", "duck", _joints(), period_s=.01, max_age_s=5., read_timeout_s=2.,
                                     clock=lambda: now[0])
    assert imu.descriptor.epoch != joints.descriptor.epoch
    profile = {"kind": "sensors", "robot_id": "duck", "alignment": {"max_skew_s": .015},
               "providers": [{"id": "imu", "kind": "injected"}, {"id": "joints", "kind": "injected"}]}
    domain = build_sensor_domain("sensing", profile, providers={"imu": imu, "joints": joints})
    try:
        now[0] += .035          # imu lattice 0.02 s -> 0.020; joints 0.01 s -> 0.030
        result = domain.execute("read_aligned", {"reference": "imu"})
        pair = result["pairings"]["joints"]
        assert pair["status"] == "aligned" and pair["skew_s"] == pytest.approx(.01, abs=1e-9)
        assert pair["sequence"] == 3 and result["reference"]["sequence"] == 1
    finally:
        domain.close()


def test_no_fresh_reference_refuses_instead_of_reusing_old_pixels(aligned_domain):
    domain, providers = aligned_domain()
    providers["imu"].publish(_env("imu", _imu(), t=.050, seq=10))
    refused = _aligned(domain)
    assert refused["ok"] is False and "no completed capture available" in refused["error"]
    providers["overview"].publish(_env("overview", _rgbd(), t=.050, seq=10))
    assert _aligned(domain)["ok"]
    again = _aligned(domain)                                   # the camera produced nothing newer
    assert again["ok"] is False and "replay" in again["error"]


@pytest.mark.parametrize("args, match", [
    ({"reference": "lidar"}, "unknown reference"), ({"reference": "overview", "sensor_ids": ["overview"]}, "reference"),
    ({"reference": "overview", "sensor_ids": ["imu", "imu"]}, "duplicate"),
    ({"reference": "overview", "sensor_ids": ["sonar"]}, "unknown sensor"),
    ({"reference": "overview", "sensor_ids": []}, "sensor_ids"), ({"reference": "overview", "sensor_ids": "imu"}, "sensor_ids"),
    ({"reference": "overview", "interpolate": True}, "invalid arguments"), ({}, "invalid arguments"),
])
def test_read_aligned_arguments_are_strict(aligned_domain, args, match):
    domain, providers = aligned_domain()
    providers["overview"].publish(_env("overview", _rgbd(), t=.050, seq=10))
    result = domain.execute("read_aligned", args)
    assert result["ok"] is False and match in result["error"]
    assert domain.hub.history() == ()                         # refused before any read


# ── the real consumer path: MCP -> RobotRuntime -> loopback mobile bridge ───


@pytest.fixture
def bridge(monkeypatch):
    from cascade.sim.mobile_bridge import MobileBridgeController, MobileBridgeServer
    from cascade.sim.mobile_rgbd import RgbdFrameCache
    from mobile_support_fixture import support_contract
    from test_microduck_stepper import render_times
    from test_sensing_providers import profile
    from test_sensing_rgbd import calibration
    for key, value in PORTS.items():
        monkeypatch.setenv(key, value)
        assert 46300 <= int(value) <= 46399
    c = MobileBridgeController(robot_id="duck", source="isolated-bridge", engine="newton", device="cuda:0",
                               asset_sha256="a" * 64, policy_sha256="b" * 64, model_identity_sha256="e" * 64,
                               support_contract=support_contract(), max_linear_speed=.2, max_angular_speed=.8,
                               max_duration_s=5., lease_s=.3, max_state_age_s=5., max_action_wall_s=2.)
    cache = RgbdFrameCache(c.hello(), calibration=calibration(), max_pixels=1000, max_jpeg_bytes=10000)
    rgb = np.zeros((4, 6, 3), np.uint8)
    depth = np.arange(24, dtype=np.float32).reshape(4, 6) / 100

    def camera(step):
        refs = render_times(step * .005)
        cache.publish(rgb, depth_m=depth, calibration=calibration(), step=step, sim_time_s=step * .005,
                      captured_at=time.monotonic(), render_times=refs, rgbd_render_times={"rgb": refs, "depth": refs})

    operations = []
    server = MobileBridgeServer(c, port=0, frame_callback=cache)
    dispatch = server.dispatch

    def recorded(request):
        operations.append({k: v for k, v in request.items() if not k.startswith("_")})
        return dispatch(request)

    server.dispatch = recorded
    server.start()
    p = profile(server.address[1])
    p["timeout_s"], p["epoch"] = 1., c.hello()["epoch"]
    try:
        yield c, server, camera, p, operations
    finally:
        server.close()
        cache.close()


def _state(c, step, gyro, dq=0.):
    from test_mobile_bridge import publish
    publish(c, step=step, sim_time=step * .005, angular_velocity=list(gyro), q=list(range(14)), dq=[dq] * 14)


def _mcp(monkeypatch, tmp_path, p, *, alignment):
    from cascade.apps.mcp_server import McpSkillServer
    from cascade.config import Cfg
    from cascade.control.isaac_base import IsaacBase
    from cascade.control.lazy_arm import LazyArm
    from cascade.sim.mobile_rgbd import calibration_record
    from test_sensing_rgbd import calibration

    def forbidden(*a, **kw):
        pytest.fail("a passive alignment read constructed an actuator")

    monkeypatch.setattr(IsaacBase, "__init__", forbidden)
    monkeypatch.setattr(LazyArm, "__init__", forbidden)
    monkeypatch.delenv("CASCADE_BASE", raising=False)
    monkeypatch.setenv("CASCADE_ROBOT", "aligned_sensors_fixture")
    monkeypatch.setenv("CASCADE_RUN_DIR", str(tmp_path))
    srv = McpSkillServer()
    sensing = dict(kind="sensors", robot_id="duck", read_timeout_s=2., max_age_s=5., providers=[
        dict(id="overview", kind="mobile_rgbd", profile=p, camera="overview",
             calibration_sha256=calibration_record(calibration())[1]),
        dict(id="imu", kind="mobile_state", profile=p, modality="imu"),
        dict(id="joints", kind="mobile_state", profile=p, modality="proprioception")])
    if alignment is not None:
        sensing["alignment"] = alignment
    srv._mobile_config = Cfg(dict(robot_mode="composed", robot_id="duck", memory={}, domains={"sensing": sensing}))

    def call(name, args):
        return json.loads(srv.call_tool(name, args)["content"][-1]["text"])

    return srv, call


def test_premise_mcp_reads_camera_and_imu_from_different_physics_steps(bridge, monkeypatch, tmp_path):
    c, server, camera, p, operations = bridge
    srv, call = _mcp(monkeypatch, tmp_path, p, alignment=None)
    try:
        assert "sensing.read_aligned" not in {row["name"] for row in srv.list_tools()}
        _state(c, 11, (0., 0., .3))
        camera(10)
        cam = call("sensing.read_sensor", {"sensor_id": "overview"})
        imu = call("sensing.read_sensor", {"sensor_id": "imu"})
        assert cam["ok"] and imu["ok"], (cam, imu)
        assert (cam["observation"]["sequence"], imu["observation"]["sequence"]) == (10, 11)
        assert cam["observation"]["epoch"] == imu["observation"]["epoch"] == p["epoch"]
    finally:
        srv.shutdown()


def test_mcp_pairs_the_bridge_camera_with_imu_and_joints_by_physics_time(bridge, monkeypatch, tmp_path):
    c, server, camera, p, operations = bridge
    srv, call = _mcp(monkeypatch, tmp_path, p, alignment={"max_skew_s": .0075, "max_rotation_rad": .01})
    try:
        assert "sensing.read_aligned" in {row["name"] for row in srv.list_tools()}
        _state(c, 8, (0., 0., .1))
        assert call("sensing.read_sensor", {"sensor_id": "imu"})["ok"]
        assert call("sensing.read_sensor", {"sensor_id": "joints"})["ok"]
        _state(c, 11, (0., 0., .3))
        camera(10)
        first = call("sensing.read_aligned", {"reference": "overview"})
        assert first["ok"], first
        assert (first["reference"]["sequence"], first["reference"]["capture_time_s"]) == (10, .05)
        for name in ("imu", "joints"):
            pair = first["pairings"][name]
            assert pair["status"] == "aligned" and pair["sequence"] == 11, (name, pair)
            assert pair["skew_s"] == pytest.approx(.005, abs=1e-12) and pair["epoch"] == p["epoch"]
        assert first["pairings"]["imu"]["observation"]["payload"]["angular_velocity_rad_s"] == [0., 0., .3]
        assert first["pairings"]["imu"]["absent"] == ["linear_acceleration_m_s2"]
        assert first["pairings"]["joints"]["absent"] == ["effort_nm"]
        # the camera moves on, the state does not: nothing newer than step 11
        camera(14)
        second = call("sensing.read_aligned", {"reference": "overview", "sensor_ids": ["imu"]})
        pair = second["pairings"]["imu"]
        assert (pair["status"], pair["sequence"]) == ("stale", 11) and "observation" not in pair
        assert pair["skew_s"] == pytest.approx(-.015, abs=1e-12) and "replay" in pair["read_error"]
        # fast rotation between the state and the camera capture: uncertain, preserved
        _state(c, 15, (0., 0., 3.))
        camera(16)
        third = call("sensing.read_aligned", {"reference": "overview", "sensor_ids": ["imu"]})
        pair = third["pairings"]["imu"]
        assert (pair["status"], pair["reasons"], pair["sequence"]) == ("uncertain", ["motion_exceeds_tolerance"], 15)
        assert pair["motion"]["rad"] == pytest.approx(.015)
        assert {op["op"] for op in operations} == {"hello", "state", "frame"}
        assert all(op.get("role", "reader") == "reader" for op in operations)
        assert not server._owners and c.hello()["generation"] == 0
    finally:
        srv.shutdown()
