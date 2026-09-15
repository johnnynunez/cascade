"""Exercise the real pump on scripted frames; no cameras, sockets or GPU."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from cascade.perception import stream as streaming


def frame(stamp, *, source="remote-simulator", camera="cam0", robot="/robot"):
    capture = None if stamp is None else {
        "backend": "isaac", "source": [source, 8611], "camera": camera, "t": stamp,
        "proprioception": {"backend": "isaac", "robot_id": robot, "t": stamp,
                           "time_source": "physics_loop_monotonic"},
    }
    return SimpleNamespace(capture=capture, t=0.0, frame_id=0)


def pump(monkeypatch, frames, received):
    """Run finite deliveries synchronously with an independent receiver clock."""
    clock = SimpleNamespace(now=received[0] - .01)
    def sleep(seconds):
        clock.now += seconds
    monkeypatch.setattr(streaming, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep))
    delivered = []
    pending = iter(zip(frames, received))
    def grab():
        value, clock.now = next(pending)
        value.t = clock.now
        value.frame_id = len(delivered) + 1
        if value.frame_id == len(frames):
            stream._stop = True
        return value
    stream = streaming.CameraStream(SimpleNamespace(get_frame=grab, has_depth=True), rate_hz=1000)
    def published():
        delivered.append({"frame": stream.latest(), "sequence": stream._latest_seq, "fps": stream.fps})
    monkeypatch.setattr(stream._cond, "notify_all", published)
    stream._loop()
    assert [row["frame"] for row in delivered] == frames
    assert [row["sequence"] for row in delivered] == list(range(1, len(frames) + 1))
    assert [value.frame_id for value in frames] == list(range(1, len(frames) + 1))
    assert stream.last_error is None
    return stream, delivered, clock


def test_remote_duplicates_keep_all_deliveries_but_report_distinct_capture_rate(monkeypatch):
    frames = [frame(500 + (index // 10) * .65) for index in range(61)]
    received = [1_000_000 + index * .065 for index in range(61)]
    original = deepcopy([value.capture for value in frames])
    stream, delivered, _ = pump(monkeypatch, frames, received)
    assert stream.fps == pytest.approx(1 / .65)
    assert all(row["fps"] == 0 for row in delivered[:10])
    assert max(row["fps"] for row in delivered) < 2
    assert [value.capture for value in frames] == original


def test_repeated_first_capture_does_not_create_a_rate(monkeypatch):
    _, delivered, _ = pump(monkeypatch, [frame(42) for _ in range(20)],
                           [100 + index * .05 for index in range(20)])
    assert all(row["fps"] == 0 for row in delivered)


def test_producer_interval_is_independent_of_receiver_clock_offset_and_delivery_rate(monkeypatch):
    stream, _, _ = pump(monkeypatch, [frame(42), frame(42.5)], [1_000_000, 1_000_000.02])
    assert stream.fps == pytest.approx(2)


def test_local_cameras_keep_local_delivery_cadence(monkeypatch):
    frames = [frame(None) for _ in range(61)]
    stream, _, clock = pump(monkeypatch, frames, [100 + index * .065 for index in range(61)])
    assert 15 < stream.fps < 16
    previous = stream.fps
    clock.now += 10
    assert stream.fps == previous, "Remote capture reporting must not change local-camera semantics"


@pytest.mark.parametrize("bad", ["missing_proprioception", "missing_source", "nan_clock",
                                 "mismatched_clock", "unknown_clock", "missing_capture", "wrong_backend"])
def test_invalid_producer_metadata_reports_zero_without_dropping_delivery(monkeypatch, bad):
    invalid = frame(11)
    if bad == "missing_proprioception": invalid.capture["proprioception"] = None
    elif bad == "missing_source": invalid.capture.pop("source")
    elif bad == "nan_clock": invalid.capture["t"] = float("nan")
    elif bad == "mismatched_clock": invalid.capture["proprioception"]["t"] = 12
    elif bad == "unknown_clock": invalid.capture["proprioception"]["time_source"] = "wall_clock"
    elif bad == "missing_capture": invalid.capture = None
    elif bad == "wrong_backend": invalid.capture["backend"] = "unverified"
    stream, delivered, _ = pump(monkeypatch, [frame(10), frame(10.5), invalid], [100, 100.5, 100.6])
    assert delivered[1]["fps"] == pytest.approx(2)
    assert stream.fps == 0


@pytest.mark.parametrize("changed", ["source", "camera", "robot"])
def test_new_producer_identity_needs_its_own_capture_interval(monkeypatch, changed):
    other = {changed: "another-producer"}
    frames = [frame(10), frame(10.5), frame(80_000, **other), frame(80_000.5, **other)]
    _, delivered, _ = pump(monkeypatch, frames, [100, 100.5, 100.6, 100.7])
    assert [row["fps"] for row in delivered] == pytest.approx([0, 2, 0, 2])


def test_backwards_producer_clock_resets_its_rate(monkeypatch):
    frames = [frame(10), frame(10.5), frame(1), frame(1.5)]
    _, delivered, _ = pump(monkeypatch, frames, [100, 100.5, 100.6, 100.7])
    assert [row["fps"] for row in delivered] == pytest.approx([0, 2, 0, 2])


def test_repeated_captures_cannot_keep_advertising_the_last_active_rate(monkeypatch):
    frames = [frame(10), frame(10.5), frame(10.5), frame(10.5), frame(10.5)]
    stream, delivered, clock = pump(monkeypatch, frames, [100, 100.5, 100.6, 101, 104])
    assert delivered[1]["fps"] == pytest.approx(2)
    assert 0 <= stream.fps <= 1 / 3.5
    clock.now += 1000
    assert round(stream.fps, 1) == 0
