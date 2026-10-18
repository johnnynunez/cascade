"""The Spark camera surface (scripts/spark_cameras.py) keeps a viewer's stream
open across short bridge gaps.

On Newton the bridge's `frame` op refuses a camera for ~0.3 s (p90 0.65-0.8 s)
whenever its 0.5 s fallback capture lands off the render rhythm. The surface
raised there, which closed the MJPEG stream; the extension then backed off 2,
4, 8 s and showed "Not updating" for 20.7 % of the samples (newton-diag Q2).
Now a refused read serves the last good image for up to 5 s after it last
advanced (the extension's own "Not updating" threshold), marked with its age;
past that the surface reports the camera as before. Only this viewer holds:
the bridge and every perception reader of it are unchanged, so a missing frame
stays missing for perception's scene-change logic.

Fake bridges on ephemeral loopback ports; no Isaac, no live port.
"""
from __future__ import annotations

import base64
import http.client
from http.server import ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import re
import socketserver
import threading
import time

import pytest

from owned_server import held_dead_port

ROOT = Path(__file__).resolve().parents[1]
JPEG_1 = b"\xff\xd8first-frame\xff\xd9"
JPEG_2 = b"\xff\xd8second-frame\xff\xd9"


def load():
    spec = importlib.util.spec_from_file_location("spark_cameras", ROOT / "scripts" / "spark_cameras.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeBridge:
    """The bridge's `frame` op: `reply` is ("ok", jpeg, stamp) or ("refuse",)
    (the bridge's own "unknown/not-ready camera" answer)."""

    def __init__(self):
        self.reply = ("ok", JPEG_1, 1.0)
        self.calls = 0
        bridge = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                json.loads(self.rfile.readline())
                bridge.calls += 1
                if bridge.reply[0] == "drop":  # closes the connection, no reply
                    return
                if bridge.reply[0] == "ok":
                    _, jpg, stamp = bridge.reply
                    data = {"ok": True, "t": stamp, "rgb_jpeg_b64": base64.b64encode(jpg).decode()}
                else:
                    data = {"ok": False, "error": "unknown/not-ready camera 'cam0'"}
                self.wfile.write(json.dumps(data).encode() + b"\n")

        self.server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def bridge():
    b = FakeBridge()
    try:
        yield b
    finally:
        b.close()


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


# ── the hold, on a fake clock ──────────────────────────────────────────────


def test_a_gap_under_5_s_serves_the_last_image_with_its_age(bridge):
    """Real clock, shipped settings: a refusal 0.6 s after the last good
    frame (past the 0.5 s cache) serves that frame, 0.6 s old."""
    cam = load()
    frames = cam.Frames(bridge.port)
    assert frames.frame("worktop")["jpeg"] == JPEG_1
    bridge.reply = ("refuse",)
    time.sleep(0.6)
    row = frames.frame("worktop")
    assert bridge.calls == 2, "the bridge was asked and refused"
    assert (row["jpeg"], row["frame_id"]) == (JPEG_1, 1)
    assert 0.6 <= cam.age_s(row) < 1.5
    state = frames.state()["cameras"]["worktop"]
    assert state["online"] is True and state["error"] is None and state["frame_id"] == 1
    assert 0.6 <= state["age_s"] < 1.5


def test_the_age_grows_through_the_hold(bridge):
    cam = load()
    clock = Clock()
    frames = cam.Frames(bridge.port, clock=clock)
    first = frames.frame("worktop")
    assert (first["jpeg"], first["frame_id"], cam.age_s(first, clock())) == (JPEG_1, 1, 0.0)

    bridge.reply = ("refuse",)
    for t, age in ((100.6, 0.6), (101.5, 1.5), (104.9, 4.9)):
        clock.t = t
        row = frames.frame("worktop")
        assert (row["jpeg"], row["frame_id"]) == (JPEG_1, 1)
        assert cam.age_s(row, clock()) == pytest.approx(age)
    state = frames.state()["cameras"]["worktop"]
    assert state["online"] is True and state["error"] is None and state["frame_id"] == 1
    assert state["age_s"] == pytest.approx(4.9)


def test_past_5_s_the_camera_reads_as_before(bridge):
    """No advance for 5 s: the refusal is raised (the stream closes) and
    /state says the camera is reconnecting, exactly as before the hold."""
    cam = load()
    clock = Clock()
    frames = cam.Frames(bridge.port, clock=clock)
    frames.frame("worktop")
    bridge.reply = ("refuse",)
    clock.t = 104.99
    assert frames.frame("worktop")["jpeg"] == JPEG_1
    clock.t = 105.0
    with pytest.raises(ValueError, match="camera not ready"):
        frames.frame("worktop")
    assert frames.state()["cameras"]["worktop"] == {
        "frame_id": None, "online": False, "error": "Camera is reconnecting", "fps": 0}


def test_the_hold_is_the_extensions_5_s_threshold():
    cam = load()
    assert cam.HOLD_S == 5.0 and cam.RETRY_S == 0.1
    assert cam.Frames(1).hold_s == 5.0


def test_the_bridge_is_asked_again_soon_during_a_hold_and_a_new_frame_ends_it(bridge):
    cam = load()
    clock = Clock()
    frames = cam.Frames(bridge.port, clock=clock)
    frames.frame("worktop")
    assert bridge.calls == 1
    clock.t = 100.3  # inside the 0.5 s cache: no bridge read
    frames.frame("worktop")
    assert bridge.calls == 1
    bridge.reply = ("refuse",)
    clock.t = 100.6
    frames.frame("worktop")
    assert bridge.calls == 2
    clock.t = 100.65  # 0.05 s later: still inside the retry interval
    frames.frame("worktop")
    assert bridge.calls == 2
    clock.t = 100.72  # 0.12 s later: retried, not after the 0.5 s cache
    frames.frame("worktop")
    assert bridge.calls == 3
    bridge.reply = ("ok", JPEG_2, 2.0)
    clock.t = 100.85
    row = frames.frame("worktop")
    assert bridge.calls == 4
    assert (row["jpeg"], row["frame_id"], cam.age_s(row, clock())) == (JPEG_2, 2, 0.0)


def test_the_hold_counts_from_the_last_advance_not_the_last_read(bridge):
    """A bridge that answers but never advances (a frozen render) earns no
    hold: once that frame is 5 s old, a refusal is raised at once."""
    cam = load()
    clock = Clock()
    frames = cam.Frames(bridge.port, clock=clock)
    for t in (100.0, 101.0, 102.0, 103.0, 104.0, 105.5):
        clock.t = t
        assert frames.frame("worktop")["frame_id"] == 1  # same stamp every read
    bridge.reply = ("refuse",)
    clock.t = 106.1
    with pytest.raises(ValueError):
        frames.frame("worktop")


def test_without_a_good_image_a_refusal_is_raised_as_before(bridge):
    cam = load()
    bridge.reply = ("refuse",)
    frames = cam.Frames(bridge.port, clock=Clock())
    with pytest.raises(ValueError, match="camera not ready"):
        frames.frame("worktop")
    assert frames.state()["cameras"]["worktop"]["error"] == "Camera is reconnecting"


def test_a_dropped_bridge_connection_is_held_like_a_refusal(bridge):
    """A socket error (bridge restarting; here a held port nobody listens
    on) or a reply cut off mid-line is held like the bridge's own refusal."""
    cam = load()
    clock = Clock()
    frames = cam.Frames(bridge.port, clock=clock)
    frames.frame("worktop")
    live_port = frames.bridge_port
    with held_dead_port() as dead:
        frames.bridge_port = dead
        clock.t = 101.0
        assert frames.frame("worktop")["jpeg"] == JPEG_1
    frames.bridge_port = live_port
    bridge.reply = ("drop",)
    clock.t = 104.0
    assert frames.frame("worktop")["jpeg"] == JPEG_1
    clock.t = 105.0
    with pytest.raises(ValueError):  # the empty reply does not parse
        frames.frame("worktop")


def test_each_camera_holds_its_own_image(bridge):
    cam = load()
    clock = Clock()
    frames = cam.Frames(bridge.port, clock=clock)
    frames.frame("worktop")
    bridge.reply = ("refuse",)
    clock.t = 101.0
    with pytest.raises(ValueError):
        frames.frame("side")  # never had a good image
    assert frames.frame("worktop")["jpeg"] == JPEG_1


# ── over HTTP: the MJPEG stream stays open (scaled real time) ──────────────


def _serve(cam, frames):
    http_server = ThreadingHTTPServer(("127.0.0.1", 0), cam.CameraHandler)
    http_server.daemon_threads = True
    http_server.frames = frames
    threading.Thread(target=http_server.serve_forever, daemon=True).start()
    return http_server


def _parts(response, limit):
    """(jpeg, headers) per MJPEG part until the server closes the stream."""
    out = []
    while len(out) < limit:
        line = response.readline()
        if not line:
            return out, True
        if line.strip() != b"--frame":
            continue
        headers = {}
        while (h := response.readline().strip()):
            key, _, value = h.decode().partition(":")
            headers[key.strip().lower()] = value.strip()
        out.append((response.read(int(headers["content-length"])), headers))
    return out, False


def _stream(port, path="/stream/worktop"):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", path)
    response = conn.getresponse()
    assert response.getheader("Content-Type", "").startswith("multipart/x-mixed-replace")
    return conn, response


def test_the_stream_stays_open_through_a_bridge_gap(bridge):
    """Shipped settings: 1.5 s of refusals (three parts, one per 0.5 s) keep
    the stream open; each part is the last good image with its growing age."""
    cam = load()
    http_server = _serve(cam, cam.Frames(bridge.port))
    conn, response = _stream(http_server.server_address[1])
    try:
        first, _ = _parts(response, 1)
        bridge.reply = ("refuse",)
        held, closed = _parts(response, 3)
    finally:
        response.close()
        conn.close()
        http_server.shutdown()
        http_server.server_close()
    assert not closed and len(held) == 3, (closed, held)
    assert first[0][0] == JPEG_1 and float(first[0][1]["x-frame-age"]) < 0.2
    assert all(jpg == JPEG_1 for jpg, _ in held)
    ages = [float(h["x-frame-age"]) for _, h in held]
    assert ages == sorted(ages) and 0.9 <= ages[-1] < 5.0, ages
    assert bridge.calls >= 3, "the bridge was asked again during the gap"


def test_the_stream_closes_once_the_image_is_older_than_the_hold(bridge):
    """hold_s scaled to 1.2 s; the stream sends a part every 0.5 s."""
    cam = load()
    http_server = _serve(cam, cam.Frames(bridge.port, hold_s=1.2))
    conn, response = _stream(http_server.server_address[1])
    try:
        _parts(response, 1)
        bridge.reply = ("refuse",)
        started = time.monotonic()
        held, closed = _parts(response, 10)
        took = time.monotonic() - started
    finally:
        response.close()
        conn.close()
        http_server.shutdown()
        http_server.server_close()
    assert closed, "the stream must close once the image is older than the hold"
    ages = [float(h["x-frame-age"]) for _, h in held]
    assert 2 <= len(held) <= 3 and max(ages) < 1.2, ages
    assert took < 3.0


def test_state_and_snapshot_carry_the_age_during_a_hold(bridge):
    cam = load()
    http_server = _serve(cam, cam.Frames(bridge.port))
    origin = ("127.0.0.1", http_server.server_address[1])
    try:
        conn = http.client.HTTPConnection(*origin, timeout=5)
        conn.request("GET", "/snapshot/worktop.jpg")
        snap = conn.getresponse()
        assert snap.read() == JPEG_1
        assert float(snap.getheader("X-Frame-Age", "nan")) < 0.2
        conn.close()
        bridge.reply = ("refuse",)
        time.sleep(0.6)
        conn = http.client.HTTPConnection(*origin, timeout=5)
        conn.request("GET", "/state")
        state = json.loads(conn.getresponse().read())["cameras"]["worktop"]
        conn.close()
        assert state["online"] is True and state["error"] is None and state["frame_id"] == 1
        assert 0.5 <= state["age_s"] < 5.0
        conn = http.client.HTTPConnection(*origin, timeout=5)
        conn.request("GET", "/snapshot/worktop.jpg")
        snap = conn.getresponse()
        assert snap.status == 200 and snap.read() == JPEG_1
        assert float(snap.getheader("X-Frame-Age", "nan")) >= 0.5
        conn.close()
    finally:
        http_server.shutdown()
        http_server.server_close()


# ── perception never reads this surface ────────────────────────────────────


def test_no_perception_or_brain_reader_uses_the_viewer_surface():
    """The hold lives in the viewer only. The runtime (perception, brain,
    skills) reads frames from the bridge itself, so a refused frame still
    reaches it as missing; nothing under src/ talks to this surface or its
    port, and the only in-repo user is the launcher, as a process."""
    pattern = re.compile(r"spark_cameras|\b8091\b|/stream/(worktop|kitchen|side)")
    offenders = [str(p.relative_to(ROOT)) for p in sorted((ROOT / "src").rglob("*.py"))
                 if pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == []
    importers = [str(p.relative_to(ROOT)) for folder in ("src", "scripts", "demo", "benchmark")
                 for p in sorted((ROOT / folder).rglob("*.py"))
                 if re.search(r"^\s*(from\s+\S*spark_cameras\s+import|import\s+\S*spark_cameras)",
                              p.read_text(encoding="utf-8"), re.M)]
    assert importers == []
    launcher = (ROOT / "scripts" / "spark_browser.py").read_text(encoding="utf-8")
    assert launcher.count("spark_cameras") == 1 and '"scripts/spark_cameras.py"' in launcher
    # the runtime's sim cameras ask the bridge's frame op directly
    readers = [p for p in (ROOT / "src" / "cascade" / "sim").glob("*.py")
               if re.search(r"""["']op["']\s*:\s*["']frame["']""", p.read_text(encoding="utf-8"))]
    assert readers, "expected the runtime's own bridge frame readers under src/cascade/sim"
