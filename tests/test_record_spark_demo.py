"""Recorder contract tests. Synthetic pixels here are TEST FIXTURES only."""
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "record_spark_demo.py"


def recorder_module():
    assert SCRIPT.is_file(), "the read-only wall-clock recorder is not implemented"
    spec = importlib.util.spec_from_file_location("record_spark_demo", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ReadOnlyBridgeFixture:
    """Only observational methods exist: motion would fail immediately."""
    def __init__(self, delay=0.14, **kwargs):
        self.delay = delay
        self.calls = []
        self.closed = False

    def connect(self):
        self.calls.append("connect")

    def request(self, payload):
        assert payload == {"op": "ping"}
        self.calls.append("ping")
        return {"ok": True, "engine": "newton", "fixture": True}

    def frame(self, camera):
        self.calls.append(("frame", camera))
        time.sleep(self.delay)
        return np.full((72, 128, 3), 71, np.uint8), None, np.eye(3)

    def state(self):
        self.calls.append("state")
        return {"q": [0.1, -0.2], "dq": [0.0, 0.0], "gripper_pos": 0.8}

    def close(self):
        self.closed = True


class MemoryEncoder:
    def __init__(self, output, size, fps):
        self.frames = []
        self.closed = False

    def write(self, frame):
        self.frames.append(frame.copy())

    def close(self):
        self.closed = True
        return 0


def rows(output):
    return [json.loads(line) for line in output.with_suffix(".jsonl").read_text().splitlines()]


def test_slow_source_holds_real_pixels_without_speeding_wall_time(tmp_path):
    module = recorder_module()
    output = tmp_path / "test-fixture.mp4"
    bridge = ReadOnlyBridgeFixture()
    encoders = []

    def encoder_factory(*args):
        encoder = MemoryEncoder(*args)
        encoders.append(encoder)
        return encoder

    cfg = module.Config(output=output, duration=0.35, fps=20, width=640, height=600)
    summary = module.record(cfg, client_factory=lambda **kw: bridge,
                            encoder_factory=encoder_factory)
    events = rows(output)
    frames = [row for row in events if row["event"] == "frame"]
    acquisitions = {row["acquisition_id"]: row for row in events
                    if row["event"] == "acquisition" and row["ok"]}
    assert summary["wall_duration_s"] >= cfg.duration
    assert summary["frame_count"] == len(frames) == math.ceil(cfg.duration * cfg.fps)
    assert summary["media_duration_s"] == pytest.approx(cfg.duration)
    assert summary["repeated_frame_count"] > 0
    assert len(encoders[0].frames) == len(frames)
    assert encoders[0].closed and bridge.closed
    assert summary["physical_task_verdict"] == "not_evaluated"
    for index, frame in enumerate(frames):
        assert frame["frame_index"] == index
        assert frame["pts_s"] == pytest.approx(index / cfg.fps)
        assert frame["q"] == [0.1, -0.2]
        acquired = acquisitions[frame["acquisition_id"]]
        assert acquired["ready_monotonic_ns"] <= frame["scheduled_monotonic_ns"]
        assert frame["wall_time_ns"] > 0
    assert {row["event"] for row in events} >= {"metadata", "acquisition", "frame", "summary"}


def test_real_encoder_produces_h264_faststart_fully_decodable_mosaic(tmp_path):
    import shutil
    import subprocess
    module = recorder_module()
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("requires external ffmpeg/ffprobe, not a camera or robot")
    output = tmp_path / "synthetic-test-fixture.mp4"
    cfg = module.Config(output=output, duration=0.25, fps=12,
                        camera="arbitrary_primary", secondary="cam0", width=640, height=600)
    bridge = ReadOnlyBridgeFixture(delay=0.001)
    summary = module.record(cfg, client_factory=lambda **kw: bridge)
    info = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format",
        "-of", "json", str(output)]))
    video = info["streams"][0]
    assert video["codec_name"] == "h264"
    assert video["pix_fmt"] == "yuv420p"
    assert (video["width"], video["height"]) == (640, 600)
    assert int(video["nb_read_frames"]) == summary["frame_count"] == 3
    assert float(info["format"]["duration"]) == pytest.approx(0.25, abs=0.001)
    data = output.read_bytes()
    assert data.index(b"moov") < data.index(b"mdat")
    decoded = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(output),
                              "-f", "null", "-"], capture_output=True, check=True)
    assert decoded.stderr == b""
    assert output.with_suffix(".ffmpeg.log").is_file()
    assert ("frame", "arbitrary_primary") in bridge.calls
    assert ("frame", "cam0") in bridge.calls


def test_stop_file_finalizes_owned_encoder_and_records_early_wall_end(tmp_path):
    module = recorder_module()
    stop = tmp_path / "stop"
    output = tmp_path / "fixture-stop.mp4"
    bridge = ReadOnlyBridgeFixture(delay=0.001)
    encoders = []

    class StoppingEncoder(MemoryEncoder):
        def write(self, frame):
            super().write(frame)
            if len(self.frames) == 3:
                stop.touch()

    def factory(*args):
        encoders.append(StoppingEncoder(*args))
        return encoders[-1]

    cfg = module.Config(output=output, duration=5, fps=20, width=640, height=600,
                        stop_file=stop)
    summary = module.record(cfg, client_factory=lambda **kw: bridge, encoder_factory=factory)
    assert summary["stop_reason"] == "stop_file"
    assert summary["frame_count"] == 3
    assert summary["wall_duration_s"] < cfg.duration
    assert abs(summary["media_duration_s"] - summary["wall_duration_s"]) <= 1 / cfg.fps + 0.03
    assert encoders[0].closed and bridge.closed
    assert rows(output)[-1]["stop_reason"] == "stop_file"


@pytest.mark.parametrize("payload,expected_verdict", [
    ({"description": "Observation in progress", "ok": True}, None),
    ({"text": "Physical check failed", "verdict": "REFUTED"}, "REFUTED"),
    ({"text": "Explicit false is not success", "verdict": False}, False),
    ("{broken JSON", None),
])
def test_external_status_is_displayed_and_logged_without_inventing_verdict(
        tmp_path, monkeypatch, payload, expected_verdict):
    module = recorder_module()
    status = tmp_path / "external.json"
    status.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    output = tmp_path / "fixture-status.mp4"
    labels = []
    put_text = module.cv2.putText

    def capture_label(image, text, *args, **kwargs):
        labels.append(text)
        return put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(module.cv2, "putText", capture_label)
    cfg = module.Config(output=output, duration=0.1, fps=10, width=640, height=600,
                        status_file=status)
    result = module.record(cfg, client_factory=lambda **kw: ReadOnlyBridgeFixture(0.001),
                           encoder_factory=MemoryEncoder)
    frame = next(row for row in rows(output) if row["event"] == "frame")
    assert frame["external_status"]["verdict"] == expected_verdict
    assert result["physical_task_verdict"] == "not_evaluated"
    assert module.TITLE in labels
    assert any("NOT EVALUATED" in label for label in labels)
    if isinstance(payload, dict):
        assert frame["external_status"]["raw"] == payload
        assert any(payload.get("text", payload.get("description")) in label for label in labels)
    else:
        assert frame["external_status"]["error"]
    if expected_verdict is None:
        assert any("NOT PROVIDED" in label for label in labels)
    else:
        assert any("External verdict (unvalidated):" in label for label in labels)


def test_cli_is_observational_and_validates_before_connecting(tmp_path):
    import subprocess
    module = recorder_module()
    output = tmp_path / "demo.mp4"
    cfg = module.parse_args(["--output", str(output), "--duration", "6",
                             "--camera", "proof", "--secondary", "cam0",
                             "--status-file", str(tmp_path / "status.json"),
                             "--stop-file", str(tmp_path / "stop")])
    assert cfg.camera == "proof" and cfg.secondary == "cam0" and cfg.fps == 10
    assert cfg.controller_label == "diagnostic observation"
    assert cfg.output == output
    help_run = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                              check=True, text=True, capture_output=True)
    assert "read-only" in help_run.stdout.lower()
    assert "--controller-label" in help_run.stdout
    for invalid in (["--fps", "0"], ["--duration", "nan"], ["--width", "641"],
                    ["--height", "100"], ["--output", str(tmp_path / "bad.avi")]):
        with pytest.raises(SystemExit):
            module.parse_args(["--output", str(output), "--duration", "6", *invalid])


@pytest.mark.parametrize("failure_at", ["write", "close", "engine", "factory"])
def test_errors_leave_failure_evidence_and_close_only_owned_resources(tmp_path, failure_at):
    module = recorder_module()
    output = tmp_path / "failed-fixture.mp4"
    bridge = ReadOnlyBridgeFixture(delay=0.001)
    if failure_at == "engine":
        bridge.request = lambda payload: {"ok": True, "engine": "physx"}
    encoders = []

    class FailingEncoder(MemoryEncoder):
        def write(self, frame):
            if failure_at == "write" and len(self.frames) == 2:
                raise RuntimeError("deliberate encoder fixture failure")
            super().write(frame)

        def close(self):
            super().close()
            if failure_at == "close":
                raise RuntimeError("deliberate mux fixture failure")
            return 0

    def factory(*args):
        encoders.append(FailingEncoder(*args))
        return encoders[-1]

    def make_client(**kwargs):
        if failure_at == "factory":
            raise RuntimeError("deliberate bridge fixture factory failure")
        return bridge

    cfg = module.Config(output=output, duration=0.3, fps=10, width=640, height=600,
                        timeout_s=0.01)
    with pytest.raises(RuntimeError):
        module.record(cfg, client_factory=make_client, encoder_factory=factory)
    summary = rows(output)[-1]
    assert summary["event"] == "summary"
    assert summary["recording_ok"] is False
    assert summary["error"]
    assert summary["physical_task_verdict"] == "not_evaluated"
    assert (bridge.closed or failure_at == "factory") and all(encoder.closed for encoder in encoders)


def test_slow_encoder_reports_overrun_without_pulling_future_images_into_past_slots(tmp_path):
    module = recorder_module()
    output = tmp_path / "backlog-fixture.mp4"

    class SlowEncoder(MemoryEncoder):
        def write(self, frame):
            time.sleep(0.08)
            super().write(frame)

    cfg = module.Config(output=output, duration=0.2, fps=20, width=640, height=600)
    summary = module.record(cfg, client_factory=lambda **kw: ReadOnlyBridgeFixture(0.001),
                            encoder_factory=SlowEncoder)
    assert summary["timing_warning"]
    assert summary["encoding_overrun_s"] > 1 / cfg.fps
    assert summary["media_duration_s"] == cfg.duration
    events = rows(output)
    acquisitions = {row["acquisition_id"]: row for row in events
                    if row["event"] == "acquisition" and row["ok"]}
    for row in events:
        if row["event"] == "frame":
            assert acquisitions[row["acquisition_id"]]["ready_monotonic_ns"] <= row["scheduled_monotonic_ns"]


def test_letterbox_retains_whole_camera_and_aspect_ratio():
    module = recorder_module()
    fixture = np.full((100, 200, 3), 77, np.uint8)
    fixture[0, :] = 99
    fixture[-1, :] = 111
    image = module.letterbox(fixture, 200, 200)
    assert image.shape == (200, 200, 3)
    np.testing.assert_array_equal(image[50:150], fixture)
    assert not image[:50].any() and not image[150:].any()
