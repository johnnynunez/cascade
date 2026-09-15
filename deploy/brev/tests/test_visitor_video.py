from collections import deque
import importlib.util
from io import BytesIO
from pathlib import Path
import shutil
import struct
import subprocess
import threading
import time

import pytest

spec = importlib.util.spec_from_file_location(
    "visitor_video", Path(__file__).resolve().parents[1] / "visitor_video.py")
video = importlib.util.module_from_spec(spec)
spec.loader.exec_module(video)


def test_box_reader_handles_partial_reads_without_merging_fragments():
    class Partial(BytesIO):
        def read(self, count):
            return super().read(min(count, 3))
    expected = [struct.pack(">I4s", 11, kind) + b"abc" for kind in (b"moof", b"mdat")]
    reader = video.boxes(Partial(b"".join(expected)))
    assert next(reader) == (b"moof", expected[0])
    assert next(reader) == (b"mdat", expected[1])
    with pytest.raises(EOFError):
        next(reader)


@pytest.mark.parametrize("size", [0, 1, 7, video.MAX_BOX_BYTES + 1])
def test_invalid_box_sizes_fail_before_allocating_payload(size):
    with pytest.raises(ValueError):
        next(video.boxes(BytesIO(struct.pack(">I4s", size, b"mdat"))))


def test_new_client_starts_at_latest_complete_keyframe_and_stale_stream_fails():
    stream = video.CameraVideo.__new__(video.CameraVideo)
    stream.condition = threading.Condition()
    stream.initialization = b"init"
    stream.fragments = deque([(1, b"old"), (2, b"current")], maxlen=6)
    stream.sequence = 2
    stream.updated = time.monotonic()
    stream.failed = False
    chunks = stream.chunks()
    assert next(chunks) == b"init"
    assert next(chunks) == b"current"
    stream.failed = True
    with pytest.raises(TimeoutError):
        next(chunks)
    with pytest.raises(TimeoutError):
        next(stream.chunks())


def test_clients_share_one_encoder_per_camera_and_unknown_routes_cannot_spawn(monkeypatch):
    created = []
    class Stream:
        failed = False
        updated = time.monotonic()
        def __init__(self, command):
            created.append(command)
    monkeypatch.setattr(video.shutil, "which", lambda value: "/usr/bin/ffmpeg")
    monkeypatch.setattr(video, "CameraVideo", Stream)
    streams = video.VideoStreams("http://127.0.0.1:8091")
    first = streams.get("worktop")
    assert streams.get("worktop") is first
    for name in ("kitchen", "side"):
        streams.get(name)
    with pytest.raises(ValueError):
        streams.get("../private")
    assert len(created) == 3
    assert all(command[command.index("-c:v") + 1] == "h264_nvenc" for command in created)
    assert all(command[command.index("-bf") + 1] == "0" for command in created)


def test_video_dimensions_survive_offline_to_live_camera_recovery():
    executable = shutil.which("ffmpeg")
    if not executable:
        pytest.skip("ffmpeg is needed to exercise a changing camera input")
    import cv2
    import numpy as np

    # A viewer can connect while the 640x360 offline card is being served.
    # The returning camera is 960x540; its pixels must fill the same output.
    parts = []
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for (width, height), color in zip([(640, 360), (960, 540), (640, 360)], colors):
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        ok, jpeg = cv2.imencode(".jpg", frame)
        assert ok
        body = jpeg.tobytes()
        parts.append(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                     + str(len(body)).encode() + b"\r\n\r\n" + body + b"\r\n")
    command = video.encoder_command(executable, "http://127.0.0.1:8091", "worktop")
    # Run the production filter with a raw output, so this regression also
    # runs on machines without NVENC. The live check uses the actual encoder.
    output = subprocess.run([
        executable, "-hide_banner", "-loglevel", "error", "-threads", "1",
        "-filter_threads", "1", "-f", "mpjpeg", "-i", "pipe:0",
        "-vf", command[command.index("-vf") + 1], "-threads", "1",
        "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
    ], input=b"".join(parts) + b"--frame--\r\n", capture_output=True, timeout=15)
    assert output.returncode == 0, output.stderr.decode()
    frames = np.frombuffer(output.stdout, dtype=np.uint8).reshape(-1, 540, 960, 3)
    assert len(frames) == 3
    for frame, color in zip(frames, colors):
        np.testing.assert_allclose(frame[10:-10, 10:-10].mean(axis=(0, 1)), color, atol=3)
