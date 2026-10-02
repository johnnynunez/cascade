from dataclasses import replace
import json
import threading

import pytest

from cascade.sensing import (BufferedSensorProvider, ImuPayload, MeasurementMetadata,
    SensorDescriptor, SensorDomain, SensorError, SensorHub, SyntheticSensorProvider,
    build_sensor_domain)
from test_sensing_models import observation


def descriptor(**changes):
    return SensorDescriptor(**(dict(sensor_id="imu", robot_id="duck", source="fixture", modality="imu",
        frame_id="body", clock_domain="simulation", measurement_kind="physics",
        model_identity_sha256="e" * 64, max_age_s=.5, read_timeout_s=1.) | changes))


@pytest.fixture
def hub():
    value = SensorHub(clock=lambda: 10.2)
    yield value
    assert value.close(2)["ok"]


def test_passive_registry_and_history_do_not_read_or_start_a_worker(hub):
    class Forbidden:
        descriptor = descriptor()
        def read(self):
            pytest.fail("metadata must not read")
        def close(self):
            pass
    before = set(threading.enumerate())
    hub.register(Forbidden())
    domain = SensorDomain("sensing", hub)
    assert set(threading.enumerate()) == before
    assert domain.execute("list_sensors", {})["sensors"][0]["modality"] == "imu"
    assert hub.history() == ()
    assert domain.resources[0].writer_id is None
    assert domain.resources[0].admission == "unvalidated"
    with pytest.raises(SensorError, match="sealed"):
        hub.register(Forbidden())


@pytest.mark.parametrize("changes,reason", [
    ({"epoch": "different"}, "epoch"), ({"sequence": 0}, "replay"),
    ({"capture_time_s": 0}, "regression"), ({"source": "other"}, "source"),
    ({"clock_domain": "monotonic"}, "clock_domain"),
    ({"received_monotonic_s": 9.9}, "receipt clock"),
    ({"model_identity_sha256": "f" * 64}, "model_identity"),
    ({"payload": ImuPayload(MeasurementMetadata("world"), [0, 0, 0])}, "frame"),
])
def test_bound_identity_clocks_frames_and_replay_fail_closed(hub, changes, reason):
    provider = BufferedSensorProvider(descriptor())
    hub.register(provider)
    original = observation()
    provider.publish(original)
    assert hub.read("imu") is original
    provider.publish(replace(original, sequence=2, capture_time_s=.010, **{
        k: v for k, v in changes.items() if k not in ("sequence", "capture_time_s")})
        if not {"sequence", "capture_time_s"} & changes.keys()
        else replace(original, **changes))
    with pytest.raises(SensorError, match=reason):
        hub.read("imu")
    assert hub.history() == (original,)


def test_stale_capture_cannot_be_rejuvenated_even_if_it_was_never_accepted(hub):
    provider = BufferedSensorProvider(descriptor())
    hub.register(provider)
    provider.publish(observation(producer_age_s=.6))
    with pytest.raises(SensorError, match="stale"):
        hub.read("imu")
    provider.publish(observation(producer_age_s=0, received_monotonic_s=10.2))
    with pytest.raises(SensorError, match="replay"):
        hub.read("imu")
    assert hub.history() == ()


def test_future_receipt_and_oversized_packet_fail_before_entering_history():
    for value, options, reason in (
        (observation(received_monotonic_s=11), {}, "future receipt"),
        (observation(), {"max_packet_bytes": 100, "max_history_bytes": 1000}, "byte bound"),
    ):
        hub = SensorHub(clock=lambda: 10.2, **options)
        provider = BufferedSensorProvider(descriptor())
        hub.register(provider)
        provider.publish(value)
        try:
            with pytest.raises(SensorError, match=reason):
                hub.read("imu")
            assert hub.history() == ()
        finally:
            assert hub.close(2)["ok"]


def test_stop_reset_does_not_reset_epoch_or_replay_watermark(hub):
    provider = BufferedSensorProvider(descriptor())
    hub.register(provider)
    domain = SensorDomain("sensing", hub)
    provider.publish(observation())
    assert domain.execute("read_sensor", {"sensor_id": "imu"})["ok"]
    assert domain.stop() == domain.reset_stop() == {"ok": True, "actuation": False}
    assert "replay" in domain.execute("read_sensor", {"sensor_id": "imu"})["error"]


def test_history_has_independent_count_and_byte_bounds():
    size = len(json.dumps(observation().as_dict(), separators=(",", ":")).encode())
    hub = SensorHub(clock=lambda: 10.2, max_history=3, max_packet_bytes=size + 50,
                    max_history_bytes=size * 2 + 20)
    provider = BufferedSensorProvider(descriptor())
    hub.register(provider)
    try:
        for step in range(1, 8):
            provider.publish(observation(sequence=step, capture_time_s=step * .005))
            hub.read("imu")
        assert [value.sequence for value in hub.history()] == [6, 7]
        assert all(value.received_monotonic_s == 10 for value in hub.history())
        with pytest.raises(ValueError, match="duplicate"):
            hub.register(provider)
    finally:
        assert hub.close()["ok"]


def test_blocked_provider_is_one_quarantined_worker_with_honest_close():
    entered, release = threading.Event(), threading.Event()
    calls = []
    class Blocked:
        descriptor = descriptor(read_timeout_s=.05)
        def read(self):
            calls.append("read")
            entered.set()
            release.wait(5)
            return observation()
        def close(self):
            calls.append("close")
    hub = SensorHub(clock=lambda: 10.2)
    hub.register(Blocked())
    try:
        with pytest.raises(SensorError, match="deadline"):
            hub.read("imu")
        assert entered.wait(1)
        for _ in range(10):
            with pytest.raises(SensorError, match="quarantined"):
                hub.read("imu")
        result = hub.close(.01)
        assert result == {"ok": False, "pending_providers": ["imu"], "errors": {}}
        assert hub.history() == ()
        assert calls == ["read"]
    finally:
        release.set()
        assert hub.close(2)["ok"]
    assert calls == ["read", "close"]


def test_inflight_reads_never_multiply_provider_calls_and_close_prevents_admission():
    entered, release = threading.Event(), threading.Event()
    class Blocked:
        descriptor = descriptor(read_timeout_s=2.)
        def read(self):
            entered.set()
            release.wait(3)
            return observation()
        def close(self):
            pass
    hub = SensorHub(clock=lambda: 10.2)
    hub.register(Blocked())
    failures = []
    def reader():
        try:
            hub.read("imu")
        except SensorError as exc:
            failures.append(str(exc))
    thread = threading.Thread(target=reader)
    thread.start()
    try:
        assert entered.wait(2)
        with pytest.raises(SensorError, match="flight"):
            hub.read("imu")
        assert not hub.close(.01)["ok"]
    finally:
        release.set()
        thread.join(3)
        assert not thread.is_alive()
        assert hub.close(2)["ok"]
    assert len(failures) == 1 and "closed" in failures[0]
    assert hub.history() == ()


def test_close_failure_is_reported_not_swallowed():
    class BadClose:
        descriptor = descriptor()
        def read(self):
            return observation()
        def close(self):
            raise RuntimeError("fixture close failed")
    hub = SensorHub()
    hub.register(BadClose())
    result = hub.close(2)
    assert not result["ok"] and "fixture close failed" in result["errors"]["imu"]


def test_synthetic_samples_are_explicit_and_polling_does_not_advance_capture():
    now = [10.]
    clock = lambda: now[0]
    provider = SyntheticSensorProvider("imu", "duck", ImuPayload(MeasurementMetadata("body"), [0, 0, 0]),
                                       clock=clock, period_s=.1, read_timeout_s=1)
    hub = SensorHub(clock=clock)
    hub.register(provider)
    domain = SensorDomain("sensors", hub)
    try:
        first = hub.read("imu")
        assert first.measurement_kind == "synthetic" and first.model_identity_sha256 is None
        assert domain.resources[0].admission == "software_only"
        assert domain.resources[0].synthetic
        now[0] += .05
        with pytest.raises(SensorError, match="replay"):
            hub.read("imu")
        now[0] += .1
        second = hub.read("imu")
        assert second.sequence == first.sequence + 1
        assert second.capture_time_s == pytest.approx(first.capture_time_s + .1)
        assert second.producer_age_s == pytest.approx(.05)
    finally:
        assert domain.close()["ok"]


def test_profile_construction_is_strict_and_software_only():
    profile = {"kind": "sensors", "robot_id": "duck", "providers": [
        {"id": "imu", "kind": "synthetic", "modality": "imu", "frame_id": "body",
         "values": {"angular_velocity_rad_s": [0, 0, 0]}}]}
    domain = build_sensor_domain("sensors", profile)
    assert domain.execute("read_sensor", {"sensor_id": "imu"})["ok"]
    assert domain.close()["ok"]
    with pytest.raises(ValueError, match="unknown"):
        build_sensor_domain("sensors", profile | {"actuator": "locomotion"})
    assert domain.execute("read_sensor", {"sensor_id": "imu"})["ok"] is False
