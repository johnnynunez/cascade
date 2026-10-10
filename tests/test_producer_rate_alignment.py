"""B72: producer-rate sampling for `sensing.read_aligned` (B50 follow-up).

ROADMAP Perception row, still open after B50: "producer-rate sampling (the
pairing sees only captures that reads admitted)". B50 pairs one fresh
reference capture with the nearest capture that a READ admitted to the hub, so
the skew depends on when the caller reads: the IMU can have produced a sample
at the camera's capture instant that no read ever saw, and the pairing then
takes a later (or earlier) read and reports it farther, or stale.

Opt-in fix: producers record every completed sample as they publish it, in a
bounded ring (`BufferedSensorProvider(history=N)` in process;
`MobileBridgeController(state_history=N)` on the mobile bridge, served to
reader channels as `state_history`), and `alignment.producer_history: N` gives
each paired sensor a bounded hub ring of those samples, admitted with the same
identity / epoch / ordering / age rules as a read. `read_aligned` pairs among
the read-admitted and the produced captures with B50's classification
unchanged: never interpolated, stale and missing withheld, another epoch or
clock domain never aligned, saturation or rate x skew uncertain.

CPU only: synthetic streams, the real loopback mobile bridge and the real MCP
composed-robot path. Every producer sample is published from the test thread at
an explicit physics step (no free-running producer raced against reads).
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import statistics
import threading
import time

import numpy as np
import pytest

from cascade.sensing import (BufferedSensorProvider, ImuPayload, MeasurementMetadata, ObservationEnvelope,
                             ProprioceptionPayload, RgbdPayload, SensorDescriptor, SensorError,
                             SyntheticSensorProvider, build_sensor_domain)

EPOCH = "episode-1"
MODEL = "e" * 64
#: This item's port block (CHILD_RULES W8: 47400-47499). The sensors-only
#: runtime dials none of the sidecars; the bridge itself binds port 0.
PORTS = {"CASCADE_GRASPGENX_PORT": "47401", "CASCADE_OCCUPANCY_PORT": "47402",
         "CASCADE_BRIDGE_PORT": "47403", "CASCADE_HUG_PORT": "47404"}


# ── synthetic producers on a fixed hub clock ────────────────────────────────


def _rgbd(frame_id="camera:overview"):
    return RgbdPayload(MeasurementMetadata(frame_id), 2, 2, bytes(range(12)),
                       np.array([.5, .6, 0., .7], dtype="<f4").tobytes(), [100., 0., 1., 0., 100., 1., 0., 0., 1.])


def _imu(gyro=(0., 0., .1), saturated=None):
    return ImuPayload(MeasurementMetadata("body", None, saturated), list(gyro))


def _joints(velocity=(0., .5), saturated=None, names=("j0", "j1")):
    return ProprioceptionPayload(MeasurementMetadata("joints", None, saturated), list(names),
                                 [.1] * len(names), list(velocity))


def _env(sensor_id, payload, *, t, seq, epoch=EPOCH, clock="simulation", received=99.9, age=.001):
    return ObservationEnvelope(source="fixture", sensor_id=sensor_id, epoch=epoch, sequence=seq,
                               clock_domain=clock, capture_time_s=t, received_monotonic_s=received,
                               producer_age_s=age, model_identity_sha256=MODEL, measurement_kind="physics",
                               payload=payload)


def _descriptor(sensor_id, modality, frame_id, clock="simulation", *, max_age_s=5., read_timeout_s=2.):
    return SensorDescriptor(sensor_id=sensor_id, robot_id="duck", source="fixture", modality=modality,
                            frame_id=frame_id, clock_domain=clock, measurement_kind="physics",
                            model_identity_sha256=MODEL, max_age_s=max_age_s, read_timeout_s=read_timeout_s)


_STREAMS = {"overview": ("rgbd", "camera:overview"), "imu": ("imu", "body"), "joints": ("proprioception", "joints")}


def _rig(alignment=None, *, history=8, without=(), clocks=None):
    """A sensors domain over buffered producers; the hub's age clock is fixed
    at 100 s and every capture is received at 99.9 s, so outputs are exact.
    `history=None` builds B50's providers exactly (no producer ring)."""
    from cascade.sensing import SensorDomain, SensorHub
    from cascade.sensing.alignment import AlignmentPolicy
    hub = SensorHub(clock=lambda: 100.)
    providers = {}
    for name, (modality, frame) in _STREAMS.items():
        descriptor = _descriptor(name, modality, frame, (clocks or {}).get(name, "simulation"))
        if history is None or name == "overview" or name in without:
            providers[name] = BufferedSensorProvider(descriptor)
        else:
            providers[name] = BufferedSensorProvider(descriptor, history=history)
        hub.register(providers[name])
    policy = None if alignment is None else AlignmentPolicy.from_profile(alignment)
    return SensorDomain("sensing", hub, alignment=policy), providers


def _aligned(domain, **args):
    result = domain.execute("read_aligned", {"reference": "overview", **args})
    assert result["ok"], result
    return result


def _produce(provider, sensor_id, steps, payload=None, **kwargs):
    for seq in steps:
        provider.publish(_env(sensor_id, payload(seq) if callable(payload) else (payload or _imu()),
                              t=round(seq * .005, 6), seq=seq, **kwargs))


# ── premises and goldens (pass on main by design) ───────────────────────────


def test_premise_read_time_sampling_pairs_the_latest_read_not_the_produced_instant():
    """The gap: the IMU produced a sample AT the camera's capture time (seq 10),
    but only the capture a read returns reaches the hub, so the pairing is
    20 ms off and stale while the exact instant had been produced."""
    domain, providers = _rig({"max_skew_s": .0075}, history=None)
    try:
        _produce(providers["imu"], "imu", range(8, 15))            # 200 Hz, t = .040 .. .070
        providers["overview"].publish(_env("overview", _rgbd(), t=.050, seq=10))
        imu = _aligned(domain, sensor_ids=["imu"])["pairings"]["imu"]
        assert (imu["status"], imu["sequence"]) == ("stale", 14)
        assert imu["skew_s"] == pytest.approx(.020, abs=1e-12) and "observation" not in imu
        assert [o.sequence for o in domain.hub.history("imu")] == [14]
        assert "producer_error" not in imu and "from_producer" not in imu
    finally:
        domain.close()


#: sha256 of read_aligned's complete JSON for the scenario below, computed on
#: unchanged origin/main 38f6d08 (B50 behaviour). Without `producer_history`
#: the output must stay byte-identical.
GOLDEN_READ_ALIGNED = "641a62f226167a2c868af7decb9f74f982f0225deb9063561d0149400b812635"


def test_golden_read_aligned_output_is_byte_identical_without_producer_history():
    domain, providers = _rig({"max_skew_s": .0075, "max_rotation_rad": .01}, history=None)
    try:
        providers["imu"].publish(_env("imu", _imu((0., 0., .1)), t=.040, seq=8))
        assert domain.execute("read_sensor", {"sensor_id": "imu"})["ok"]
        providers["imu"].publish(_env("imu", _imu((0., 0., 3.)), t=.045, seq=9))
        providers["overview"].publish(_env("overview", _rgbd(), t=.050, seq=10))
        result = _aligned(domain)
        assert set(result) == {"ok", "reference", "policy", "pairings", "counts"}
        assert result["policy"] == {"max_skew_s": .0075, "max_rotation_rad": .01, "max_translation_m": None,
                                    "selection": "nearest_admitted_capture", "interpolation": "none"}
        assert set(result["pairings"]["imu"]) == {"status", "skew_s", "reasons", "motion", "read_error", "sequence",
                                                  "epoch", "clock_domain", "capture_time_s", "capture_sha256",
                                                  "observation", "absent"}
        assert set(result["pairings"]["joints"]) == {"status", "skew_s", "reasons", "motion", "read_error"}
        blob = json.dumps(result, sort_keys=True, separators=(",", ":"))
        assert hashlib.sha256(blob.encode()).hexdigest() == GOLDEN_READ_ALIGNED
    finally:
        domain.close()


def test_golden_shipped_sensor_catalogs_keep_b50s_digest():
    """B50's golden: every shipped sensors domain's tool catalog, unchanged."""
    from test_proprio_time_alignment import GOLDEN_SENSOR_TOOLS, SENSOR_PROFILES
    from cascade.apps.robot_runtime import describe_robot
    from cascade.config import load_robot_config
    rows = []
    for profile in SENSOR_PROFILES:
        for name, adapter in sorted(describe_robot(load_robot_config(profile)).items()):
            if adapter.profile["kind"] == "sensors":
                assert "alignment" not in adapter.profile
                rows.append([profile, name, adapter.tool_specs, [t.as_dict() for t in adapter.tool_descriptors]])
                adapter.runtime.close()
    blob = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(blob.encode()).hexdigest() == GOLDEN_SENSOR_TOOLS


def _controller(**changes):
    from cascade.sim.mobile_bridge import MobileBridgeController
    from mobile_support_fixture import support_contract
    kwargs = dict(robot_id="duck", source="isolated-bridge", engine="newton", device="cuda:0",
                  asset_sha256="a" * 64, policy_sha256="b" * 64, model_identity_sha256=MODEL,
                  support_contract=support_contract(), max_linear_speed=.2, max_angular_speed=.8,
                  max_duration_s=5., lease_s=.3, max_state_age_s=5., max_action_wall_s=2.)
    kwargs.update(changes)
    return MobileBridgeController(**kwargs)


def _state(c, step, gyro=(0., 0., .1), dq=0.):
    from test_mobile_bridge import publish
    publish(c, step=step, sim_time=round(step * .005, 6), angular_velocity=list(gyro), q=list(range(14)),
            dq=[dq] * 14)


def test_golden_default_bridge_advertises_and_serves_no_state_history():
    """Default off: the hello, the state reply and the reader-op refusal are
    the old ones (the B12a off-GIL reader is configured from READER_OPS)."""
    from cascade.sim.mobile_bridge import MobileBridgeServer
    c = _controller()
    hello = c.hello()
    assert hello["capabilities"] == ["state", "velocity", "stop", "reset_stop"]
    assert "state_history" not in hello
    _state(c, 1)
    assert "state_history" not in c.state()
    assert MobileBridgeServer.READER_OPS == frozenset({"state", "frame"})
    server = MobileBridgeServer(c, port=0)
    assert server.dispatch({"op": "state_history"}) == {"ok": False, "error": "unsupported mobile bridge operation"}
    from types import SimpleNamespace
    bare = MobileBridgeServer(SimpleNamespace(), port=0)                # a controller without the method
    assert bare.dispatch({"op": "state_history"}) == {"ok": False, "error": "unsupported mobile bridge operation"}


# ── the producer rings ──────────────────────────────────────────────────────


@pytest.mark.parametrize("value", [0, -1, 257, True, 1.5, "8", None])
def test_producer_history_is_a_bounded_integer(value):
    profile = {"kind": "sensors", "robot_id": "duck", "alignment": {"max_skew_s": .01, "producer_history": value},
               "providers": [{"id": "imu", "kind": "injected"}]}
    with pytest.raises(ValueError, match=r"alignment\.producer_history must be an integer in 1\.\.256"):
        build_sensor_domain("sensing", profile, providers={
            "imu": BufferedSensorProvider(_descriptor("imu", "imu", "body"))})


def test_policy_reports_the_producer_history_only_when_configured():
    from cascade.sensing.alignment import AlignmentPolicy
    plain = AlignmentPolicy.from_profile({"max_skew_s": .02})
    assert plain.producer_history is None and "producer_history" not in plain.as_dict()
    for size in (1, 256):
        policy = AlignmentPolicy.from_profile({"max_skew_s": .02, "producer_history": size})
        assert policy.producer_history == size
        assert policy.as_dict() == {**plain.as_dict(), "producer_history": size}


def test_a_buffered_producer_records_every_publication_in_a_bounded_ring():
    descriptor = _descriptor("imu", "imu", "body")
    provider = BufferedSensorProvider(descriptor, history=3)
    published = [_env("imu", _imu((0., 0., seq)), t=seq / 100, seq=seq) for seq in range(1, 6)]
    for envelope in published:
        provider.publish(envelope)
    assert provider.read() is published[-1]                          # reads are unchanged: the latest
    assert all(a is b for a, b in zip(provider.read_produced(None), published[2:], strict=True))
    assert all(a is b for a, b in zip(provider.read_produced(3), published[3:], strict=True))
    assert provider.read_produced(5) == ()
    plain = BufferedSensorProvider(descriptor)                       # default: nothing is recorded
    plain.publish(published[0])
    with pytest.raises(SensorError, match="records no produced samples"):
        plain.read_produced(None)
    for bad in (-1, 257, True, 1.5, "3"):
        with pytest.raises(ValueError, match="history"):
            BufferedSensorProvider(descriptor, history=bad)
    assert BufferedSensorProvider(descriptor, history=256) is not None
    provider.close()
    with pytest.raises(SensorError, match="closed"):
        provider.read_produced(None)


def test_closing_a_buffered_producer_releases_its_recorded_samples():
    import gc
    import weakref
    provider = BufferedSensorProvider(_descriptor("imu", "imu", "body"), history=4)
    provider.publish(_env("imu", _imu(), t=.005, seq=1))
    provider.publish(_env("imu", _imu(), t=.010, seq=2))
    first = weakref.ref(provider.read_produced(None)[0])
    gc.collect()
    assert first() is not None                                       # held by the ring only
    provider.close()
    gc.collect()
    assert first() is None


def test_the_embodiment_binding_applies_to_produced_samples():
    from dataclasses import replace
    from cascade.config import load_robot_config
    from cascade.robotics.embodiment import EmbodimentDescriptor
    from cascade.sensing.embodiment import EmbodimentBoundProvider
    cfg = load_robot_config("wheeled_lift_sensors")
    body = EmbodimentDescriptor.from_dict(cfg.embodiment.as_dict())
    domain = build_sensor_domain("sensing", cfg.domains.sensing.as_dict(), embodiment=body)
    try:
        payload = domain.hub.read("joints").payload
    finally:
        domain.close()
    descriptor = SensorDescriptor(sensor_id="joints", robot_id=body.robot_id, source="fixture",
                                  modality="joint_state", frame_id=payload.metadata.frame_id,
                                  clock_domain="simulation", measurement_kind="physics", model_identity_sha256=MODEL)
    producer = BufferedSensorProvider(descriptor, history=4)
    bound = EmbodimentBoundProvider(producer, "sensing", body)
    good = _env("joints", payload, t=.005, seq=1)
    producer.publish(good)
    assert bound.read_produced(None) == (good,) and bound.read_produced(None)[0] is good
    producer.publish(_env("joints", replace(payload, joints=tuple(reversed(payload.joints))), t=.010, seq=2))
    with pytest.raises(SensorError, match="identities/order"):
        bound.read_produced(1)
    bare = EmbodimentBoundProvider(type("Bare", (), {"descriptor": descriptor})(), "sensing", body)
    with pytest.raises(SensorError, match="records no produced samples"):
        bare.read_produced(None)


class _Scripted:
    """A producer whose recorded history is scripted per fetch."""

    def __init__(self, descriptor, batches):
        self.descriptor = descriptor
        self.batches = list(batches)
        self.after = []

    def read(self):
        raise SensorError("no completed capture available")

    def read_produced(self, after=None):
        self.after.append(after)
        return self.batches.pop(0)

    def close(self):
        pass


def _hub(*providers, size=4, clock=100., **kwargs):
    from cascade.sensing import SensorHub
    hub = SensorHub(clock=lambda: clock, **kwargs)
    for provider in providers:
        hub.register(provider)
    hub.enable_produced(size)
    hub.seal()
    return hub


def test_the_hub_ring_is_enabled_once_validated_and_absent_by_default():
    from cascade.sensing import SensorHub
    hub = SensorHub()
    hub.register(BufferedSensorProvider(_descriptor("imu", "imu", "body"), history=4))
    hub.register(SyntheticSensorProvider("syn", "duck", _imu(), period_s=.02))
    with pytest.raises(SensorError, match="producer history is not enabled"):
        hub.read_produced("imu")
    assert hub.produced("imu") == ()
    for bad in (0, -1, 257, True, 1.5, "4", None):
        with pytest.raises(ValueError, match=r"producer history must be an integer in 1\.\.256"):
            hub.enable_produced(bad)
    hub.enable_produced(4)
    with pytest.raises(ValueError, match="already enabled"):
        hub.enable_produced(4)
    with pytest.raises(SensorError, match="records no produced samples"):
        hub.read_produced("syn")                                     # a producer that keeps no history
    with pytest.raises(SensorError, match="unknown sensor"):
        hub.read_produced("lidar")
    hub.seal()
    late = SensorHub()
    late.seal()
    with pytest.raises(SensorError, match="sealed"):
        late.enable_produced(4)
    assert hub.close(2)["ok"]


def test_the_hub_ring_admits_a_batch_like_reads_bounded_by_count():
    imu = BufferedSensorProvider(_descriptor("imu", "imu", "body", max_age_s=.5), history=16)
    hub = _hub(imu)
    try:
        _produce(imu, "imu", range(1, 7))
        admitted = hub.read_produced("imu")
        assert [o.sequence for o in admitted] == [1, 2, 3, 4, 5, 6]
        assert [o.sequence for o in hub.produced("imu")] == [3, 4, 5, 6]   # count bound
        assert hub.history("imu") == ()                                    # the read history is separate
        assert hub.read_produced("imu") == ()                              # nothing new
        # a sample older than the sensor's max age is not admitted, yet it was
        # seen: the watermark moves past it
        imu.publish(_env("imu", _imu(), t=.035, seq=7, age=1.))
        imu.publish(_env("imu", _imu(), t=.040, seq=8))
        assert [o.sequence for o in hub.read_produced("imu")] == [8]
        assert [o.sequence for o in hub.produced("imu")] == [4, 5, 6, 8]
        # the read path still works on its own watermark
        assert hub.read("imu").sequence == 8
    finally:
        assert hub.close(2)["ok"]


@pytest.mark.parametrize("bad, match", [
    ("regression", "replay or capture clock regression"), ("same_time", "replay or capture clock regression"),
    ("same_sequence", "replay or capture clock regression"),
    ("receipt", "receipt clock regressed"), ("frame", "modality/frame/calibration mismatch"),
    ("source", "source mismatch"), ("not_envelope", "immutable observation"), ("list", "tuple"),
    ("future", "future receipt"), ("too_many", "too many"),
])
def test_a_bad_produced_batch_is_refused_whole(bad, match):
    from dataclasses import replace
    good = tuple(_env("imu", _imu(), t=.010 * k, seq=k) for k in (1, 2))
    valid = _env("imu", _imu(), t=.025, seq=3)               # admissible on its own
    second = {"regression": _env("imu", _imu(), t=.030, seq=2),
              "same_time": _env("imu", _imu(), t=.025, seq=4),
              "same_sequence": _env("imu", _imu(), t=.030, seq=3),
              "receipt": _env("imu", _imu(), t=.030, seq=4, received=99.8),
              "frame": _env("imu", ImuPayload(MeasurementMetadata("head"), [0., 0., 0.]), t=.030, seq=4),
              "source": replace(_env("imu", _imu(), t=.030, seq=4), source="other"),
              "not_envelope": {"sequence": 4},
              "future": _env("imu", _imu(), t=.030, seq=4, received=100.5)}
    if bad == "list":
        batch = [valid]
    elif bad == "too_many":
        batch = tuple(_env("imu", _imu(), t=.030 + k / 1e4, seq=3 + k) for k in range(1025))
    else:
        batch = (valid, second[bad])
    producer = _Scripted(_descriptor("imu", "imu", "body"), [good, batch, ()])
    hub = _hub(producer)
    try:
        assert [o.sequence for o in hub.read_produced("imu")] == [1, 2]
        with pytest.raises(SensorError, match=match):
            hub.read_produced("imu")
        assert [o.sequence for o in hub.produced("imu")] == [1, 2]       # nothing of the bad batch
        assert hub.read_produced("imu") == ()
        assert producer.after == [None, 2, 2]                            # its watermark did not move either
    finally:
        assert hub.close(2)["ok"]


def test_a_first_produced_batch_cannot_mix_epochs():
    batch = (_env("imu", _imu(), t=.005, seq=1), _env("imu", _imu(), t=.010, seq=2, epoch="episode-2"))
    hub = _hub(_Scripted(_descriptor("imu", "imu", "body"), [batch]))
    try:
        with pytest.raises(SensorError, match="epoch mismatch"):
            hub.read_produced("imu")
        assert hub.produced("imu") == ()
    finally:
        assert hub.close(2)["ok"]


def test_a_produced_sample_dropped_as_stale_is_never_admitted_later():
    stale = _env("imu", _imu(), t=.035, seq=7, age=1.)
    again = _env("imu", _imu(), t=.035, seq=7, received=99.95)     # the same capture, presented fresher
    producer = _Scripted(_descriptor("imu", "imu", "body", max_age_s=.5), [(stale,), (again,)])
    hub = _hub(producer)
    try:
        assert hub.read_produced("imu") == ()
        with pytest.raises(SensorError, match="replay or capture clock regression"):
            hub.read_produced("imu")
        assert hub.produced("imu") == () and producer.after == [None, 7]
    finally:
        assert hub.close(2)["ok"]


def test_the_hub_ring_is_bounded_by_bytes_and_drops_oversized_samples():
    sample = _env("imu", _imu(), t=.051, seq=11)
    size = len(json.dumps(sample.as_dict(), separators=(",", ":"), allow_nan=False).encode())
    bound = size * 5 // 2
    imu = BufferedSensorProvider(_descriptor("imu", "imu", "body"), history=8)
    joints = BufferedSensorProvider(_descriptor("joints", "proprioception", "joints"), history=8)
    hub = _hub(imu, joints, size=8, max_history_bytes=bound, max_packet_bytes=bound)
    try:
        for seq, t in ((11, .051), (12, .052), (13, .053), (14, .054)):
            imu.publish(_env("imu", _imu(), t=t, seq=seq))
        assert len(hub.read_produced("imu")) == 4
        assert [o.sequence for o in hub.produced("imu")] == [13, 14]          # two fit in the byte bound
        names = tuple(f"joint_{i:03d}" for i in range(60))
        joints.publish(_env("joints", _joints([0.] * 60, names=names), t=.050, seq=1))   # one oversized sample
        joints.publish(_env("joints", _joints(), t=.055, seq=2))
        assert [o.sequence for o in hub.read_produced("joints")] == [2]
        assert [o.sequence for o in hub.produced("joints")] == [2]
        assert [o.sequence for o in hub.produced("imu")] == [13, 14]          # per sensor: no cross-eviction
    finally:
        assert hub.close(2)["ok"]


def test_one_epoch_per_sensor_across_reads_and_produced_samples():
    imu = BufferedSensorProvider(_descriptor("imu", "imu", "body"), history=8)
    hub = _hub(imu)
    try:
        imu.publish(_env("imu", _imu(), t=.005, seq=1))
        assert len(hub.read_produced("imu")) == 1                # the ring pins episode-1
        imu.publish(_env("imu", _imu(), t=.010, seq=2, epoch="episode-2"))
        with pytest.raises(SensorError, match="epoch mismatch"):
            hub.read("imu")
        with pytest.raises(SensorError, match="epoch mismatch"):
            hub.read_produced("imu")
        assert [o.epoch for o in hub.produced("imu")] == [EPOCH] and hub.history("imu") == ()
    finally:
        assert hub.close(2)["ok"]
    imu = BufferedSensorProvider(_descriptor("imu", "imu", "body"), history=8)
    hub = _hub(imu)
    try:
        imu.publish(_env("imu", _imu(), t=.005, seq=1))
        assert hub.read("imu").epoch == EPOCH                    # a read pins episode-1
        imu.publish(_env("imu", _imu(), t=.010, seq=2, epoch="episode-2"))
        with pytest.raises(SensorError, match="epoch mismatch"):
            hub.read_produced("imu")                             # [seq 1 (ok), seq 2 (other epoch)]: refused whole
        assert hub.produced("imu") == ()
    finally:
        assert hub.close(2)["ok"]


def test_a_blocked_producer_history_quarantines_like_a_read():
    release = threading.Event()

    class Blocking(BufferedSensorProvider):
        def read_produced(self, after=None):
            release.wait(10.)
            return ()

    provider = Blocking(_descriptor("imu", "imu", "body", read_timeout_s=.05), history=4)
    hub = _hub(provider)
    try:
        with pytest.raises(SensorError, match="deadline expired; provider quarantined"):
            hub.read_produced("imu")
        with pytest.raises(SensorError, match="quarantined"):
            hub.read("imu")
    finally:
        release.set()
        assert hub.close(2)["ok"]


# ── the consumer: read_aligned with the producer ring ───────────────────────


def test_read_aligned_pairs_the_produced_sample_at_the_capture_instant():
    domain, providers = _rig({"max_skew_s": .0075, "max_rotation_rad": .01, "producer_history": 8})
    try:
        _produce(providers["imu"], "imu", range(8, 15), payload=lambda seq: _imu((0., 0., seq / 10)))
        providers["overview"].publish(_env("overview", _rgbd(), t=.050, seq=10))
        result = _aligned(domain, sensor_ids=["imu"])
        imu = result["pairings"]["imu"]
        assert (imu["status"], imu["sequence"], imu["from_producer"]) == ("aligned", 10, True)
        assert imu["skew_s"] == pytest.approx(0., abs=1e-12) and imu["reasons"] == []
        assert imu["read_error"] is None and imu["producer_error"] is None
        # never interpolated: ONE produced capture, byte for byte
        assert imu["observation"]["payload"]["angular_velocity_rad_s"] == [0., 0., 1.]
        ring = domain.hub.produced("imu")
        assert [o.sequence for o in ring] == list(range(8, 15))
        assert imu["capture_sha256"] == ring[2].sha256 and imu["observation"] == ring[2].as_dict()
        assert [o.sequence for o in domain.hub.history("imu")] == [14]
        assert result["policy"]["producer_history"] == 8
        assert result["counts"] == {"aligned": 1, "stale": 0, "missing": 0, "uncertain": 0}
        # the capture a read admitted is reported as such (and preferred as the same capture)
        providers["overview"].publish(_env("overview", _rgbd(), t=.070, seq=14))
        again = _aligned(domain, sensor_ids=["imu"])["pairings"]["imu"]
        assert (again["status"], again["sequence"], again["from_producer"]) == ("aligned", 14, False)
        assert again["capture_sha256"] == domain.hub.history("imu")[-1].sha256
        assert "replay" in again["read_error"] and again["producer_error"] is None
    finally:
        domain.close()


def test_the_ring_never_upgrades_stale_or_missing():
    domain, providers = _rig({"max_skew_s": .0075, "producer_history": 8})
    try:
        _produce(providers["imu"], "imu", (2, 4, 30))            # nothing within 7.5 ms of t = .050
        providers["overview"].publish(_env("overview", _rgbd(), t=.050, seq=10))
        result = _aligned(domain)
        imu = result["pairings"]["imu"]
        assert (imu["status"], imu["sequence"], imu["from_producer"]) == ("stale", 4, True)
        assert imu["skew_s"] == pytest.approx(-.030, abs=1e-12)
        assert "observation" not in imu and "absent" not in imu
        joints = result["pairings"]["joints"]
        assert (joints["status"], joints["skew_s"], joints["from_producer"]) == ("missing", None, None)
        assert joints["reasons"] == ["no_admitted_capture"] and "sequence" not in joints
        assert "no completed capture available" in joints["read_error"] and joints["producer_error"] is None
        assert result["counts"] == {"aligned": 0, "stale": 1, "missing": 1, "uncertain": 0}
    finally:
        domain.close()


@pytest.mark.parametrize("max_skew_s, status", [(.0075, "stale"), (.0078125, "aligned")])
def test_a_gap_between_produced_samples_is_never_interpolated(max_skew_s, status):
    """Produced 7.8125 ms before and after the capture (exact binary times, an
    exact tie): the EARLIER sample is kept, verbatim, although only the later
    one was read; no value is made up for the instant in between."""
    domain, providers = _rig({"max_skew_s": max_skew_s, "producer_history": 8})
    try:
        for seq, t in ((8, .4921875), (12, .5078125)):
            providers["imu"].publish(_env("imu", _imu((0., 0., seq / 10)), t=t, seq=seq))
        providers["overview"].publish(_env("overview", _rgbd(), t=.5, seq=10))
        imu = _aligned(domain, sensor_ids=["imu"])["pairings"]["imu"]
        assert (imu["status"], imu["sequence"], imu["from_producer"]) == (status, 8, True)
        assert imu["skew_s"] == -.0078125
        assert [o.sequence for o in domain.hub.history("imu")] == [12]
        if status == "aligned":
            assert imu["observation"]["payload"]["angular_velocity_rad_s"] == [0., 0., .8]
    finally:
        domain.close()


def test_motion_and_saturation_still_make_a_produced_sample_uncertain():
    domain, providers = _rig({"max_skew_s": .0075, "max_rotation_rad": .01, "producer_history": 8})
    try:
        _produce(providers["imu"], "imu", (9,), payload=_imu((0., 0., 3.)))
        _produce(providers["imu"], "imu", (20,))
        _produce(providers["joints"], "joints", (10,), payload=_joints((0., 0.), saturated=True))
        _produce(providers["joints"], "joints", (20,), payload=_joints())
        providers["overview"].publish(_env("overview", _rgbd(), t=.050, seq=10))
        result = _aligned(domain)
        imu, joints = result["pairings"]["imu"], result["pairings"]["joints"]
        assert (imu["status"], imu["sequence"], imu["reasons"]) == ("uncertain", 9, ["motion_exceeds_tolerance"])
        assert imu["motion"]["rad"] == pytest.approx(.015) and imu["from_producer"] is True
        assert imu["observation"]["payload"]["angular_velocity_rad_s"] == [0., 0., 3.]   # preserved, flagged
        assert (joints["status"], joints["sequence"], joints["reasons"]) == ("uncertain", 10, ["saturated"])
        assert joints["skew_s"] == pytest.approx(0., abs=1e-12)
        assert result["counts"] == {"aligned": 0, "stale": 0, "missing": 0, "uncertain": 2}
    finally:
        domain.close()


def test_produced_samples_on_another_clock_instance_are_never_aligned():
    domain, providers = _rig({"max_skew_s": .0075, "producer_history": 8}, clocks={"joints": "monotonic"})
    try:
        # a reset world (another epoch) produced samples AT the camera's time
        _produce(providers["imu"], "imu", (10, 12), epoch="episode-0")
        _produce(providers["joints"], "joints", (10, 12), payload=_joints(), clock="monotonic")
        providers["overview"].publish(_env("overview", _rgbd(), t=.050, seq=10))
        result = _aligned(domain)
        for name in ("imu", "joints"):
            pair = result["pairings"][name]
            assert (pair["status"], pair["skew_s"], pair["reasons"]) == ("uncertain", None, ["clock_not_comparable"])
            assert pair["sequence"] == 12 and pair["from_producer"] is False   # the latest, as B50 reports it
            assert pair["motion"] == {}
            assert len(domain.hub.produced(name)) == 2
        assert result["pairings"]["imu"]["epoch"] == "episode-0"
    finally:
        domain.close()


def test_a_sensor_without_producer_history_keeps_read_time_sampling():
    domain, providers = _rig({"max_skew_s": .0075, "producer_history": 8}, without=("joints",))
    try:
        _produce(providers["joints"], "joints", (9, 11), payload=_joints())
        _produce(providers["imu"], "imu", (10, 11))
        providers["overview"].publish(_env("overview", _rgbd(), t=.050, seq=10))
        result = _aligned(domain)
        joints = result["pairings"]["joints"]
        assert (joints["status"], joints["sequence"], joints["from_producer"]) == ("aligned", 11, False)
        assert "records no produced samples" in joints["producer_error"]
        assert result["pairings"]["imu"]["sequence"] == 10
    finally:
        domain.close()


def test_merge_keeps_one_copy_per_capture_in_capture_time_order():
    from cascade.sensing.alignment import merge_captures
    read = [_env("imu", _imu(), t=.070, seq=14, received=99.95)]
    produced = [_env("imu", _imu(), t=round(seq * .005, 6), seq=seq) for seq in (12, 13, 14)]
    merged = merge_captures(read, produced)
    assert [o.sequence for o in merged] == [12, 13, 14]
    assert merged[-1] is read[0]                                     # the read copy of seq 14
    assert merge_captures(read, []) == list(read) and merge_captures([], produced) == produced


# ── the real loopback mobile bridge ─────────────────────────────────────────


def _bridge_cli():
    import importlib
    import sys
    from conftest import REPO
    if str(REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts"))
    return importlib.import_module("isaac_microduck_bridge")


_CLI_REQUIRED = ["--engine", "newton", "--release", "/nonexistent", "--bundle", "/nonexistent",
                 "--bundle-sha256", "a" * 64, "--policy", "/nonexistent", "--policy-sha256", "b" * 64,
                 "--bam-source-root", "/nonexistent", "--bam-profile", "fixture", "--robot-id", "microduck",
                 "--source", "fixture", "--device", "cuda:0", "--port", "0", "--limits", "/nonexistent",
                 "--out", "/nonexistent", "--max-wall-s", "1", "--max-steps", "1"]


def test_the_microduck_bridge_cli_flag_is_absent_by_default_and_bounded():
    cli = _bridge_cli()
    default = cli.parse_args(_CLI_REQUIRED)
    assert not hasattr(default, "state_history")             # recorded arguments of a default run unchanged
    assert cli.state_history(default) == 0
    assert cli.state_history(cli.parse_args(_CLI_REQUIRED + ["--state-history", "64"])) == 64
    assert cli.state_history(cli.parse_args(_CLI_REQUIRED + ["--state-history", "256"])) == 256
    for bad in ("0", "257", "-3"):
        with pytest.raises(ValueError, match=r"--state-history must be an integer in 1\.\.256"):
            cli.state_history(cli.parse_args(_CLI_REQUIRED + ["--state-history", bad]))
        with pytest.raises(ValueError, match=r"--state-history must be an integer in 1\.\.256"):
            cli.admit(cli.parse_args(_CLI_REQUIRED + ["--state-history", bad]))   # before any file is read


@pytest.mark.parametrize("history", [None, 8])
def test_the_microduck_bridge_runner_records_every_completed_step(tmp_path, monkeypatch, capsys, history):
    """The real CLI lifecycle on the CPU software backend: with --state-history
    the stepper's own publications fill the ring at every completed step."""
    from types import SimpleNamespace as NS
    from test_microduck_bridge_cli import software_limits, software_model_identity
    from test_microduck_stepper import SoftwareActuator, SoftwareBackend, SoftwarePolicy, render_times
    from cascade.sim import mobile_identity
    from cascade.sim.microduck_policy_admission import target_contract
    from cascade.sim.mobile_bridge import MobileBridgeServer
    cli = _bridge_cli()
    monkeypatch.setattr(mobile_identity, "build_model_identity", software_model_identity)
    out = tmp_path / "run"
    args = NS(out=out, device="cuda:0", robot_id="microduck", source="software-only", max_wall_s=3., max_steps=9,
              port=0, camera_every=4, max_jpeg_bytes=100000, policy=tmp_path / "fixture.onnx",
              policy_sha256="b" * 64, target_profile="direct-v1", python_extra_path=[], camera_rgbd=False)
    if history is not None:
        args.state_history = history
    admission = dict(target_contract=target_contract("b" * 64, "direct-v1"), asset_sha256="a" * 64,
                     asset_receipt_sha256="c" * 64, bam_params={}, limits=software_limits(),
                     experience_text="software fixture\n")
    created, controlled = [], {}

    class Backend(SoftwareBackend):
        def __init__(self, *unused):
            super().__init__()
            self.bam = SoftwareActuator(self)
            self.receipt = {"software_fixture": True}
            created.append(self)

        def open(self):
            pass

        def shutdown(self, exit_code):
            pass

        def capture(self):
            return dict(rgb=np.zeros((24, 32, 3), np.uint8), step=self.step_count, sim_time_s=self.sim_time,
                        captured_at=0., render_times=render_times(self.sim_time))

        def support_probe(self):
            return dict(passed=True, step=self.step_count, sim_time_s=self.sim_time, max_force_torque_difference=0.)

    class Server(MobileBridgeServer):
        def __init__(self, controller, **kwargs):
            controlled["controller"] = controller
            super().__init__(controller, **kwargs)

    result = cli.run(args, admission, backend_factory=Backend,
                     policy_factory=lambda *a, **k: SoftwarePolicy(created[0]), server_factory=Server)
    assert result["completed"] is True
    controller = controlled["controller"]
    rows = [json.loads(x) for x in (out / "physics.jsonl").read_text().splitlines()]
    startup = json.loads((out / "startup.json").read_text())
    if history is None:
        assert "state_history" not in controller.hello()
        assert "state_history" not in startup["arguments"]
        return
    assert controller.hello()["state_history"] == 8
    states = controller.state_history({})["states"]
    assert len(rows) == 9
    assert [(s["step"], s["sim_time_s"]) for s in states] == [(r["step"], r["sim_time"]) for r in rows[-8:]]
    assert startup["arguments"]["state_history"] == 8


def test_the_bridge_records_each_completed_state_as_it_publishes_it():
    clock = [10.]
    c = _controller(state_history=4, clock=lambda: clock[0])
    hello = c.hello()
    assert hello["capabilities"] == ["state", "velocity", "stop", "reset_stop", "state_history"]
    assert hello["state_history"] == 4 and c.state_history_size == 4
    for step in range(1, 7):
        clock[0] = 10. + step / 100
        _state(c, step, (0., 0., step / 10))
    clock[0] = 11.
    reply = c.state_history({"op": "state_history"})
    assert reply["ok"] is True and reply["epoch"] == hello["epoch"] and reply["state_history"] == 4
    assert (reply["robot_id"], reply["source"], reply["model_identity_sha256"]) == ("duck", "isolated-bridge", MODEL)
    states = reply["states"]
    assert [s["step"] for s in states] == [3, 4, 5, 6]
    assert [s["sim_time_s"] for s in states] == [.015, .02, .025, .03]
    assert [s["received_monotonic_s"] for s in states] == pytest.approx([10.03, 10.04, 10.05, 10.06])
    assert [s["producer_age_s"] for s in states] == pytest.approx([.97, .96, .95, .94])
    assert [s["angular_velocity_body"][2] for s in states] == [.3, .4, .5, .6]
    assert {s["controller_status"] for s in states} == {"ready"} and {s["latched"] for s in states} == {False}
    assert c.state()["state"] == states[-1]                       # what `state` serves for that step, exactly
    assert [s["step"] for s in c.state_history({"after_step": 4})["states"]] == [5, 6]
    assert c.state_history({"after_step": 6})["states"] == []
    for bad in (-1, 1.5, True, "3"):
        with pytest.raises(ValueError, match="after_step"):
            c.state_history({"after_step": bad})
    with pytest.raises(ValueError, match="request must be an object"):
        c.state_history(None)
    c.begin_epoch()
    assert c.state_history({})["states"] == []                      # a reset never carries old samples
    # recorded AFTER the publication took effect: permission as `state` serves it
    from test_mobile_bridge import publish
    fresh = _controller(state_history=4)
    publish(fresh, step=1, sim_time=.005, balance_active=True)
    publish(fresh, step=2, sim_time=.010, balance_active=False)
    generation = fresh.hello()["generation"]
    with pytest.raises(ValueError, match="fallen"):
        publish(fresh, step=3, sim_time=.015, fallen=True)
    recorded = fresh.state_history({})["states"]
    assert [(s["step"], s["controller_status"], s["latched"], s["fallen"]) for s in recorded] == [
        (1, "ready", False, False), (2, "disabled", False, False), (3, "fault", True, True)]
    assert recorded[-1]["generation"] == generation + 1 == fresh.hello()["generation"]
    served = fresh.state()["state"]
    assert {k: v for k, v in recorded[-1].items() if k != "producer_age_s"} == {
        k: v for k, v in served.items() if k != "producer_age_s"}
    for bad in (-1, 257, True, 1.5):
        with pytest.raises(ValueError, match="state_history"):
            _controller(state_history=bad)
    assert _controller(state_history=256).hello()["state_history"] == 256
    plain = _controller()
    with pytest.raises(ValueError, match="unsupported mobile bridge operation"):
        plain.state_history({})


@pytest.fixture
def bridges(monkeypatch):
    """Factory for loopback bridges (state socket + RGB-D frame cache) on port 0."""
    from cascade.sim.mobile_bridge import MobileBridgeServer
    from cascade.sim.mobile_rgbd import RgbdFrameCache
    from test_microduck_stepper import render_times
    from test_sensing_providers import profile
    from test_sensing_rgbd import calibration
    for key, value in PORTS.items():
        monkeypatch.setenv(key, value)
        assert 47400 <= int(value) <= 47499
    built = []

    def make(state_history=0, clock=None):
        changes = {"state_history": state_history} if state_history else {}
        if clock is not None:
            changes["clock"] = clock
        c = _controller(**changes)
        cache = RgbdFrameCache(c.hello(), calibration=calibration(), max_pixels=1000, max_jpeg_bytes=10000)
        rgb = np.zeros((4, 6, 3), np.uint8)
        depth = np.arange(24, dtype=np.float32).reshape(4, 6) / 100

        def camera(step):
            refs = render_times(round(step * .005, 6))
            cache.publish(rgb, depth_m=depth, calibration=calibration(), step=step, sim_time_s=round(step * .005, 6),
                          captured_at=time.monotonic(), render_times=refs,
                          rgbd_render_times={"rgb": refs, "depth": refs})

        operations = []
        server = MobileBridgeServer(c, port=0, frame_callback=cache)
        dispatch = server.dispatch

        def recorded(request):
            operations.append({k: v for k, v in request.items() if not k.startswith("_")})
            return dispatch(request)

        server.dispatch = recorded
        server.start()
        built.append((server, cache))
        p = profile(server.address[1])
        p["timeout_s"], p["epoch"] = 1., c.hello()["epoch"]
        return c, server, camera, p, operations

    yield make
    for server, cache in built:
        server.close()
        cache.close()


def test_readers_fetch_the_produced_states_over_the_loopback_bridge(bridges):
    from cascade.sensing import MobileStateSensorProvider
    from cascade.sim.base_truth import BaseTruthReader
    from cascade.sim.bridge_client import BridgeClient, BridgeError
    c, server, camera, p, operations = bridges(state_history=8)
    for step in range(1, 6):
        _state(c, step, (0., 0., step / 10))
    reader = BaseTruthReader(p)
    try:
        assert reader().step == 5
        states = reader.history(None)
        assert [s.step for s in states] == [1, 2, 3, 4, 5]
        assert all(s.epoch == p["epoch"] and s.model_identity_sha256 == MODEL for s in states)
        assert [s.angular_velocity_body[2] for s in states] == [.1, .2, .3, .4, .5]
        assert [s.step for s in reader.history(3)] == [4, 5]
        assert reader.history(5) == ()
    finally:
        reader.close()
    provider = MobileStateSensorProvider("imu", p, "imu", read_timeout_s=2.)
    try:
        produced = provider.read_produced(2)
        assert [(o.sequence, o.capture_time_s) for o in produced] == [(3, .015), (4, .02), (5, .025)]
        assert [o.payload.angular_velocity_rad_s for o in produced] == [(0., 0., .3), (0., 0., .4), (0., 0., .5)]
        assert all(o.epoch == p["epoch"] and o.clock_domain == "simulation" for o in produced)
    finally:
        provider.close()
    # a control-role channel never gets the reader op; readers only
    client = BridgeClient("127.0.0.1", server.address[1], timeout_s=1.)
    client.connect()
    try:
        client.request({"op": "hello", "role": "control", "owner": "probe"})
        with pytest.raises(BridgeError, match="operation forbidden on channel role"):
            client.request({"op": "state_history"})
    finally:
        client.close()
    assert {op["op"] for op in operations} == {"hello", "state", "state_history"}


def test_produced_state_ages_include_the_round_trip_on_one_receipt(bridges):
    """On a frozen producer clock every remote age is 0: what remains is the
    reader's own round trip, one receipt for the whole reply."""
    from cascade.sim.base_truth import BaseTruthReader
    c, server, camera, p, operations = bridges(state_history=8, clock=lambda: 50.)
    for step in range(1, 4):
        _state(c, step)
    assert [s["producer_age_s"] for s in c.state_history({})["states"]] == [0., 0., 0.]
    reader = BaseTruthReader(p)
    try:
        before = time.monotonic()
        states = reader.history(None)
        after = time.monotonic()
        assert len({s.received_monotonic_s for s in states}) == 1
        assert before <= states[0].received_monotonic_s <= after
        assert len({s.producer_age_s for s in states}) == 1
        assert 0. < states[0].producer_age_s <= after - before
        for bad in (-1, True, 1.5, "2"):
            assert reader.history(bad) is None and "after_step must be a nonnegative integer or None" in \
                reader.last_error                                     # refused here, never sent
    finally:
        reader.close()
    assert [op.get("after_step") for op in operations if op["op"] == "state_history"] == [None]


def test_a_bridge_without_state_history_refuses_the_reader(bridges):
    from cascade.sensing import MobileStateSensorProvider
    from cascade.sim.base_truth import BaseTruthReader
    c, server, camera, p, operations = bridges()
    _state(c, 1)
    reader = BaseTruthReader(p)
    try:
        assert reader().step == 1
        assert reader.history(None) is None and "state_history" in reader.last_error
        assert reader().step == 1                                    # the state channel still works
    finally:
        reader.close()
    provider = MobileStateSensorProvider("imu", p, "imu", read_timeout_s=2.)
    try:
        with pytest.raises(SensorError, match="mobile state history unavailable"):
            provider.read_produced(None)
    finally:
        provider.close()
    assert "state_history" not in {op["op"] for op in operations}
    # a reader channel that asks anyway is refused by the transport, like any unknown op
    from cascade.sim.bridge_client import BridgeClient, BridgeError
    client = BridgeClient("127.0.0.1", server.address[1], timeout_s=1.)
    client.connect()
    try:
        client.request({"op": "hello", "role": "reader"})
        with pytest.raises(BridgeError, match="operation forbidden on channel role"):
            client.request({"op": "state_history"})
    finally:
        client.close()


@pytest.mark.parametrize("tamper, match", [
    ("reversed", "advance"), ("epoch", "epoch"), ("identity", "identity"), ("oversized", "bounded"),
    ("old", "advance"), ("refused", "refused"), ("states", "bounded"), ("advertised", "state_history"),
    ("advertised_big", "state_history"), ("time", "advance"), ("step", "advance"),
])
def test_the_reader_validates_every_produced_state(bridges, tamper, match):
    from cascade.sim.base_truth import BaseTruthReader
    from test_mobile_bridge import publish
    c, server, camera, p, operations = bridges(state_history=4)
    for step in range(1, 5):
        _state(c, step)

    def crafted(step, sim_time):
        """A self-consistent state of the same epoch that this producer never published."""
        other = _controller(state_history=4)
        other._epoch = c.hello()["epoch"]
        publish(other, step=step, sim_time=sim_time)
        return other.state_history({})["states"][0]

    backwards = crafted(5, .0125)                            # a later step whose physics time went BACK
    repeated = crafted(3, .025)                              # a later time whose step did not advance
    real = server.dispatch

    def tampered(request):
        reply = real(request)
        if request.get("op") == "hello" and tamper.startswith("advertised"):
            reply["state_history"] = 0 if tamper == "advertised" else 257
        if request.get("op") == "state_history" and reply.get("ok"):
            states = reply["states"]
            if tamper == "reversed":
                states.reverse()
            elif tamper == "time":
                states[-1] = backwards                       # steps 1, 2, 3, 5 at .005, .010, .015, .0125
            elif tamper == "step":
                states[-1] = repeated                        # steps 1, 2, 3, 3 at .005, .010, .015, .025
            elif tamper == "epoch":
                reply["epoch"] = "f" * 32
            elif tamper == "identity":
                states[1]["robot_id"] = "goose"
            elif tamper == "oversized":
                states.append({**states[-1], "step": 99, "sim_time_s": .5})
            elif tamper == "old":
                reply["states"] = c.state_history({})["states"]          # ignores after_step=2
            elif tamper == "refused":
                reply = {"ok": False, "error": "refused by tamper"}
            elif tamper == "states":
                reply["states"] = {"0": states[0]}
        return reply

    server.dispatch = tampered
    reader = BaseTruthReader(p)
    try:
        assert reader().step == 4
        after = 2 if tamper == "old" else None
        assert reader.history(after) is None
        assert match in reader.last_error
    finally:
        reader.close()


# ── the real consumer path: MCP -> RobotRuntime -> loopback mobile bridge ───


def _mcp(monkeypatch, tmp_path, p, domains):
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
    monkeypatch.setenv("CASCADE_ROBOT", "producer_rate_fixture")
    monkeypatch.setenv("CASCADE_RUN_DIR", str(tmp_path))
    srv = McpSkillServer()
    profiles = {}
    for name, alignment in domains.items():
        profiles[name] = dict(kind="sensors", robot_id="duck", read_timeout_s=2., max_age_s=5., alignment=alignment,
                              providers=[dict(id="overview", kind="mobile_rgbd", profile=p, camera="overview",
                                              calibration_sha256=calibration_record(calibration())[1]),
                                         dict(id="imu", kind="mobile_state", profile=p, modality="imu"),
                                         dict(id="joints", kind="mobile_state", profile=p, modality="proprioception")])
    srv._mobile_config = Cfg(dict(robot_mode="composed", robot_id="duck", memory={}, domains=profiles))

    def call(name, args):
        return json.loads(srv.call_tool(name, args)["content"][-1]["text"])

    return srv, call


#: One camera capture every 6 physics steps (33.3 Hz at dt = 5 ms); the state
#: producer publishes every `state_every` steps; `read_aligned` is polled
#: `k % 6` steps after the k-th capture (unsynchronized caller, each delay 4x).
CAPTURES, CAMERA_EVERY = 24, 6
ALIGNMENT = {"max_skew_s": .0075, "max_rotation_rad": .01}


def _gyro(step):
    return (0., 0., 3. + step / 1000)


def _measure(c, camera, call, *, state_every):
    rows, step = [], 0
    for k in range(1, CAPTURES + 1):
        capture = CAMERA_EVERY * k
        while step < capture + k % CAMERA_EVERY:
            step += 1
            if step % state_every == 0:
                _state(c, step, _gyro(step), dq=.5)
            if step == capture:
                camera(step)
        row = {"capture": capture, "delay_steps": k % CAMERA_EVERY}
        for domain in ("read_time", "producer"):
            result = call(f"{domain}.read_aligned", {"reference": "overview"})
            assert result["ok"], result
            assert result["reference"]["sequence"] == capture
            row[domain] = result["pairings"]
        rows.append(row)
    return rows


def _summary(rows):
    out = {}
    for domain in ("read_time", "producer"):
        for sensor in ("imu", "joints"):
            pairs = [row[domain][sensor] for row in rows]
            skews = [abs(p["skew_s"]) * 1000 for p in pairs]
            out[f"{domain}/{sensor}"] = {
                "counts": dict(sorted(Counter(p["status"] for p in pairs).items())),
                "mean_abs_skew_ms": round(statistics.fmean(skews), 3),
                "median_abs_skew_ms": round(statistics.median(skews), 3),
                "max_abs_skew_ms": round(max(skews), 3)}
    cases = Counter()
    for row in rows:
        for sensor in ("imu", "joints"):
            read, ring = row["read_time"][sensor], row["producer"][sensor]
            closer = abs(ring["skew_s"]) < abs(read["skew_s"]) - 1e-9
            cases["ring_farther"] += abs(ring["skew_s"]) > abs(read["skew_s"]) + 1e-9
            cases["read_farther"] += closer
            cases["read_stale_closer_produced"] += closer and read["status"] == "stale" and ring["status"] != "stale"
            cases["read_uncertain_ring_aligned"] += read["status"] == "uncertain" and ring["status"] == "aligned"
            cases["from_producer"] += ring["from_producer"] is True
    out["cases"] = dict(sorted(cases.items()))
    return out


MEASURED = {
    # state at 200 Hz (every physics step, as the MicroDuck stepper publishes)
    1: {"read_time/imu": {"counts": {"aligned": 4, "stale": 16, "uncertain": 4}, "mean_abs_skew_ms": 9.167,
                          "median_abs_skew_ms": 10.0, "max_abs_skew_ms": 15.0},
        "read_time/joints": {"counts": {"aligned": 8, "stale": 16}, "mean_abs_skew_ms": 9.167,
                             "median_abs_skew_ms": 10.0, "max_abs_skew_ms": 15.0},
        "producer/imu": {"counts": {"aligned": 24}, "mean_abs_skew_ms": 0.0, "median_abs_skew_ms": 0.0,
                         "max_abs_skew_ms": 0.0},
        "producer/joints": {"counts": {"aligned": 24}, "mean_abs_skew_ms": 0.0, "median_abs_skew_ms": 0.0,
                            "max_abs_skew_ms": 0.0},
        "cases": {"from_producer": 40, "read_farther": 40, "read_stale_closer_produced": 32,
                  "read_uncertain_ring_aligned": 4, "ring_farther": 0}},
    # state at 50 Hz (every 4 steps): the ring cannot invent the missing instants
    4: {"read_time/imu": {"counts": {"aligned": 8, "stale": 16}, "mean_abs_skew_ms": 8.333,
                          "median_abs_skew_ms": 10.0, "max_abs_skew_ms": 20.0},
        "read_time/joints": {"counts": {"aligned": 8, "stale": 16}, "mean_abs_skew_ms": 8.333,
                             "median_abs_skew_ms": 10.0, "max_abs_skew_ms": 20.0},
        "producer/imu": {"counts": {"aligned": 12, "stale": 12}, "mean_abs_skew_ms": 5.0,
                         "median_abs_skew_ms": 5.0, "max_abs_skew_ms": 10.0},
        "producer/joints": {"counts": {"aligned": 12, "stale": 12}, "mean_abs_skew_ms": 5.0,
                            "median_abs_skew_ms": 5.0, "max_abs_skew_ms": 10.0},
        "cases": {"from_producer": 16, "read_farther": 8, "read_stale_closer_produced": 8,
                  "read_uncertain_ring_aligned": 0, "ring_farther": 0}},
}


@pytest.mark.parametrize("state_every", [1, 4])
def test_mcp_skew_distribution_with_and_without_the_producer_ring(bridges, monkeypatch, tmp_path, state_every):
    """The measurement: the same bridge, the same producer schedule, read by a
    B50 domain (read-time sampling) and a producer-history domain, through MCP."""
    c, server, camera, p, operations = bridges(state_history=64)
    srv, call = _mcp(monkeypatch, tmp_path, p,
                     {"read_time": dict(ALIGNMENT), "producer": {**ALIGNMENT, "producer_history": 64}})
    try:
        names = {row["name"] for row in srv.list_tools()}
        assert {"read_time.read_aligned", "producer.read_aligned"} <= names
        rows = _measure(c, camera, call, state_every=state_every)
        summary = _summary(rows)
        print(json.dumps(summary, sort_keys=True))
        assert summary == MEASURED[state_every]
        for row in rows:
            for sensor in ("imu", "joints"):
                pair = row["producer"][sensor]
                assert "producer_error" in pair and pair["producer_error"] is None
                assert "producer_error" not in row["read_time"][sensor]
                if pair["status"] in ("aligned", "uncertain"):      # verbatim: the gyro of its own step
                    if sensor == "imu":
                        assert pair["observation"]["payload"]["angular_velocity_rad_s"] == list(
                            _gyro(pair["sequence"]))
                else:
                    assert "observation" not in pair
        assert {op["op"] for op in operations} == {"hello", "state", "frame", "state_history"}
        assert all(op.get("role", "reader") == "reader" for op in operations)
        assert not server._owners and c.hello()["generation"] == 0
    finally:
        srv.shutdown()
