"""The live view serves the nvblox map as one more camera (`nvblox_map`).

apps/stream_server.py is what the camera extension discovers on the real rig
(`127.0.0.1:8090/state`, then `/stream/<name>` MJPEG with a Content-Length per
part). With a map tile attached the server announces `nvblox_map` LAST in
`/state` (so the default view stays the first real camera), marked
`{"virtual": true, "kind": "map"}`, and streams it at the tile's own modest
rate -- only while a client is connected. Everything that lists or opens
REAL cameras (depth/agent views, analyze, the rig itself) never sees it.
"""

from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest
from conftest import loopback_host
from test_nvblox_map_tile import Clock, FakeOcc, settings

from cascade.apps import map_tile as mt
from cascade.apps.stream_server import StreamServer
from cascade.perception.mock_camera import MockCamera
from cascade.perception.stream import CameraRig, CameraStream


def _rig(*names):
    rig = CameraRig([CameraStream(MockCamera(), name=n, rate_hz=30.0) for n in names])
    rig.open()
    return rig


@pytest.fixture
def served():
    """A live view over two mock cameras plus a map tile (fps 4, 480 x 270)."""
    started = []

    def make(tile=None, names=("over", "side"), **kw):
        rig = _rig(*names)
        tile = tile if tile is not None else mt.MapTile(FakeOcc(age_s=0.2), settings(), arm_up_fn=lambda: True)
        server = StreamServer(rig, state_fn=lambda: {"agent_status": "testing"}, port=0, map_tile=tile, **kw)
        server.start()
        started.append((server, rig))
        return server, tile

    yield make
    for server, rig in started:
        server.stop()
        rig.close()


def _base(server):
    return f"http://{loopback_host()}:{server.port}"


def _get_json(server, path="/state"):
    return json.loads(urllib.request.urlopen(_base(server) + path, timeout=5).read())


class Stream:
    """A raw MJPEG client that checks every part like the extension's
    MJPEGParser: boundary, `Content-Type: image/jpeg`, `Content-Length`,
    JPEG magic, and exactly that many bytes."""

    def __init__(self, server, name="nvblox_map"):
        self.conn = http.client.HTTPConnection(loopback_host(), server.port, timeout=10)
        self.conn.request("GET", f"/stream/{name}")
        self.resp = self.conn.getresponse()

    def part(self) -> bytes:
        fp = self.resp.fp
        line = fp.readline()
        while line in (b"\r\n", b"\n"):
            line = fp.readline()
        assert line.strip() == b"--wrcframe", line
        headers = {}
        while True:
            line = fp.readline().strip()
            if not line:
                break
            key, value = line.decode().split(":", 1)
            headers[key.strip().lower()] = value.strip()
        assert headers["content-type"] == "image/jpeg"
        n = int(headers["content-length"])
        data = fp.read(n)
        assert len(data) == n and data[:2] == b"\xff\xd8"
        return data

    def close(self):
        self.resp.close()          # the response holds the socket open too
        self.conn.close()


def test_state_announces_the_map_last_and_marks_it_virtual(served):
    server, _ = served()
    cams = _get_json(server)["cameras"]
    assert list(cams) == ["over", "side", "nvblox_map"]   # default view stays a real camera
    entry = cams["nvblox_map"]
    assert entry["virtual"] is True and entry["kind"] == "map" and entry["label"] == "3D map (nvblox)"
    assert entry["map_state"] == "live" and entry["depth"] is False and entry["error"] is None
    assert all("virtual" not in cams[n] for n in ("over", "side"))


def test_without_a_tile_nothing_changes():
    rig = _rig("over")
    server = StreamServer(rig, port=0)
    server.start()
    try:
        assert list(_get_json(server)["cameras"]) == ["over"]
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(_base(server) + "/stream/nvblox_map", timeout=5)
        assert exc.value.code == 404
    finally:
        server.stop()
        rig.close()


def test_the_map_streams_as_mjpeg_with_a_content_length_per_part(served):
    server, _ = served()
    s = Stream(server)
    try:
        assert s.resp.status == 200
        assert s.resp.getheader("Content-Type") == "multipart/x-mixed-replace; boundary=wrcframe"
        for _ in range(2):
            bgr = cv2.imdecode(np.frombuffer(s.part(), np.uint8), cv2.IMREAD_COLOR)
            assert bgr.shape == (270, 480, 3)
    finally:
        s.close()


def test_nothing_renders_without_a_streaming_client(served):
    server, tile = served()
    for _ in range(10):
        assert _get_json(server)["cameras"]["nvblox_map"]["frame_id"] == 0
        time.sleep(0.03)
    urllib.request.urlopen(_base(server) + "/", timeout=5).read()
    assert tile.frame_id == 0


def test_rendering_follows_the_client_and_stops_when_it_leaves(served):
    server, tile = served()
    s = Stream(server)
    try:
        for _ in range(3):
            s.part()
        entry = _get_json(server)["cameras"]["nvblox_map"]
        assert entry["frame_id"] >= 3 and entry["fps"] > 0
    finally:
        s.close()
    time.sleep(0.75)                      # the handler notices the closed socket
    settled = tile.frame_id
    time.sleep(1.0)                       # four periods at 4 fps: nothing new
    assert tile.frame_id == settled
    assert _get_json(server)["cameras"]["nvblox_map"]["fps"] == 0.0


def test_the_map_streams_at_its_own_modest_rate(served):
    server, _ = served()                 # camera fps 15, map fps 4
    s = Stream(server)
    try:
        s.part()
        t0, n = time.monotonic(), 0
        while time.monotonic() - t0 < 1.5:
            s.part()
            n += 1
    finally:
        s.close()
    assert 3 <= n <= 8                    # ~6 at 4 fps, never the cameras' 15


def test_viewers_share_one_render(served):
    server, tile = served()
    streams = [Stream(server), Stream(server)]
    stop = threading.Event()

    def pump(s):
        while not stop.is_set():
            s.part()

    threads = [threading.Thread(target=pump, args=(s,), daemon=True) for s in streams]
    for t in threads:
        t.start()
    time.sleep(1.5)
    stop.set()
    for t in threads:
        t.join(5)
    for s in streams:
        s.close()
    assert 3 <= tile.frame_id <= 10       # one render per period, not one per viewer


def test_the_snapshot_route_serves_one_map_frame(served):
    server, tile = served()
    jpeg = urllib.request.urlopen(_base(server) + "/snapshot/nvblox_map.jpg", timeout=5).read()
    assert jpeg[:2] == b"\xff\xd8" and tile.frame_id == 1


@pytest.mark.parametrize("route", ["/depth/nvblox_map", "/annotated/nvblox_map"])
def test_camera_only_views_of_the_map_do_not_exist(served, route):
    server, _ = served()
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(_base(server) + route, timeout=5)
    assert exc.value.code == 404


def test_analyze_lists_real_cameras_only(served):
    server, _ = served()
    assert set(server.analyze()["cameras"]) == {"over", "side"}
    assert server.analyze("nvblox_map")["cameras"]["nvblox_map"] == {"error": "unknown camera"}


def test_the_dashboard_shows_the_map_tile_without_camera_view_buttons(served):
    server, _ = served()
    html = urllib.request.urlopen(_base(server) + "/", timeout=5).read().decode()
    assert 'id="img-nvblox_map"' in html and 'src="/stream/nvblox_map"' in html
    assert "3D map (nvblox)" in html
    assert "setView('nvblox_map'" not in html          # no depth/agent buttons for a map
    assert html.index("img-side") < html.index("img-nvblox_map")


def test_a_rig_camera_already_named_like_the_map_is_refused():
    rig = _rig("nvblox_map")
    try:
        with pytest.raises(ValueError, match="nvblox_map"):
            StreamServer(rig, port=0, map_tile=mt.MapTile(None, settings()))
    finally:
        rig.close()


def test_state_stays_fast_while_a_frame_renders(served):
    clock = Clock()
    tile = mt.MapTile(FakeOcc(age_s=0.2), settings(), arm_up_fn=lambda: True, clock=clock)
    entered, release = threading.Event(), threading.Event()
    real = tile._render

    def slow(*a, **k):
        entered.set()
        release.wait(5)
        return real(*a, **k)

    tile._render = slow
    server, _ = served(tile)
    s = Stream(server)
    try:
        assert entered.wait(5)
        t0 = time.monotonic()
        _get_json(server)
        assert time.monotonic() - t0 < 1.0
    finally:
        release.set()
        s.close()
