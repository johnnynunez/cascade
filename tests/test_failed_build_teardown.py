"""A build_runtime that fails late stops everything it had already started.

`build_runtime` opens the camera streams, builds the detector, starts the
WorldWatcher, drift monitors and (eager) live view, and only then resolves
later settings and probes (`stream.nvblox_map`, GraspGen-X). A step that
raised there used to leave the camera streams (device owners) and the
watcher running: on the robot a failed start kept a camera busy, and in the
suite the leaked 30 Hz `CameraStream` loop advanced other tests' fake clocks
(the flakes PR #337 fixed by scoping those clocks).

Contract pinned here, all on the mock stack (no socket to a live port):
* the build's own error propagates unchanged;
* no thread the build started is alive afterwards, every camera is closed;
* the detector is closed once, the occupancy client socket is closed;
* nothing the runtime did not start is stopped (the bridge's map is never
  cleared, an external live view is never closed);
* its own live view is recorded before it binds: a raise right after the
  bind still releases the port;
* the teardown is bounded: a watcher stuck in its tick delays the error by
  the bound only, and is named instead of waited for.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest

from cascade.apps import demo
from cascade.config import load_demo_config
from cascade.perception.occupancy import OccupancyMap


def _cfg(nvblox_map=None):
    cfg = load_demo_config(camera="mock_small", arm="so101_mock", llm="mock")
    cfg._data["stream"]["port"] = 0  # ephemeral, never a shared port
    if nvblox_map is not None:
        cfg._data["stream"]["nvblox_map"] = nvblox_map
    return cfg


@pytest.fixture
def cameras(monkeypatch):
    """Every camera the build makes, with the number of times it was closed."""
    made = []
    real = demo.make_camera

    def spy(ccfg):
        cam = real(ccfg)
        cam.close_calls = 0
        close = cam.close

        def counted_close():
            cam.close_calls += 1
            return close()

        cam.close = counted_close
        made.append(cam)
        return cam

    monkeypatch.setattr(demo, "make_camera", spy)
    return made


@pytest.fixture
def new_threads():
    """Threads started since the fixture began that are still alive."""
    before = set(threading.enumerate())
    return lambda: sorted(t.name for t in threading.enumerate() if t not in before and t.is_alive())


class _ClosableDetector:
    """Detector double with a class-defined close(); records each close."""

    def __init__(self, error=None):
        self.closed, self.error = 0, error

    def detect(self, frame, classes=None):
        return []

    def set_classes(self, classes):
        pass

    def close(self):
        self.closed += 1
        if self.error is not None:
            raise self.error


class _FakeBridge:
    """In-process occupancy bridge (tests/test_nvblox_map_runtime.py), plus close()."""

    def __init__(self):
        self.actions, self.closed = [], 0

    def request(self, payload, timeout_ms=None):
        self.actions.append(payload["action"])
        if payload["action"] == "probe":
            return {"ok": True, "backend": "nvblox", "device": "cuda:0", "voxel": 0.01, "esdf": True}
        return {}

    def probe(self, timeout_ms=300):
        return self.request({"action": "probe"})

    def close(self):
        self.closed += 1


def test_a_misspelled_switch_stops_the_cameras_and_the_watcher(tmp_path, monkeypatch, cameras, new_threads):
    monkeypatch.delenv("CASCADE_STREAM", raising=False)
    with pytest.raises(ValueError, match="nvblox_map"):  # the build's own error, unchanged
        demo.build_runtime(_cfg("yes please"), tmp_path / "run", serve=True)
    assert new_threads() == [], "a failed build left threads running"
    assert cameras and [c.close_calls for c in cameras] == [1] * len(cameras)


def test_a_failed_build_closes_its_occupancy_client_and_never_clears_the_bridge(
        tmp_path, monkeypatch, cameras, new_threads):
    monkeypatch.delenv("CASCADE_STREAM", raising=False)
    bridge = _FakeBridge()

    def from_config(cls, cfg, workspace_min=None, workspace_max=None):
        m = cls(client=bridge, region_min=workspace_min, region_max=workspace_max)
        m.probe()
        return m

    monkeypatch.setattr(OccupancyMap, "from_config", classmethod(from_config))
    with pytest.raises(ValueError, match="nvblox_map"):
        demo.build_runtime(_cfg("yes please"), tmp_path / "run", serve=True)
    assert new_threads() == []
    assert bridge.closed == 1, "the build's own bridge client was left open"
    # The bridge (and its map) belongs to serve_occupancy.sh, not to us.
    assert "clear" not in bridge.actions and "shutdown" not in bridge.actions


@pytest.mark.parametrize("close_error", [None, RuntimeError("close broke too")], ids=["close-ok", "close-raises"])
def test_a_late_probe_failure_stops_the_live_view_watcher_cameras_and_detector(
        tmp_path, monkeypatch, cameras, new_threads, close_error):
    from cascade.apps.stream_server import StreamServer

    monkeypatch.setenv("CASCADE_STREAM", "eager")  # the dashboard binds during the build
    detector = _ClosableDetector(close_error)
    monkeypatch.setattr(demo, "_make_detector", lambda cfg: detector)
    servers = []
    real_start = StreamServer.start

    def start(self):
        real_start(self)
        servers.append(self)

    monkeypatch.setattr(StreamServer, "start", start)

    def graspgenx_down(runtime):  # the last raise site of the build
        assert servers, "the eager live view should be up by now"
        raise RuntimeError("GraspGen-X required at startup: connection refused")

    monkeypatch.setattr(demo, "_probe_grasp_backend", graspgenx_down)
    with pytest.raises(RuntimeError, match="GraspGen-X required"):
        demo.build_runtime(_cfg(), tmp_path / "run", serve=True)
    assert new_threads() == []
    assert [c.close_calls for c in cameras] == [1] * len(cameras)
    assert detector.closed == 1
    port = servers[0].port
    with pytest.raises(OSError):  # the live-view port was released
        socket.create_connection(("127.0.0.1", port), timeout=1).close()


@pytest.mark.parametrize("where", ["inside-start", "after-start"])
def test_a_raise_right_after_the_live_view_binds_still_releases_its_port(
        tmp_path, monkeypatch, cameras, new_threads, where):
    """The live view is recorded BEFORE it opens: an error (a Ctrl+C, a reaper
    thread that cannot start) between the bind and the end of open() must not
    keep the dashboard thread serving or the port bound."""
    from cascade.apps.live_control import LiveViewController
    from cascade.apps.stream_server import StreamServer

    monkeypatch.setenv("CASCADE_STREAM", "eager")  # the dashboard binds during the build
    servers = []
    real_start = StreamServer.start

    def start(self):
        real_start(self)
        servers.append(self)
        if where == "inside-start":
            raise RuntimeError("interrupted right after the bind")

    monkeypatch.setattr(StreamServer, "start", start)
    controllers = []
    real_open = LiveViewController.open

    def recording_open(self, reason=""):
        controllers.append(self)
        return real_open(self, reason)

    monkeypatch.setattr(LiveViewController, "open", recording_open)
    if where == "after-start":
        def reaper_down(self):
            raise RuntimeError("can't start new thread")

        monkeypatch.setattr(LiveViewController, "_start_reaper", reaper_down)
    with pytest.raises(RuntimeError, match="right after the bind|can't start new thread"):
        demo.build_runtime(_cfg(), tmp_path / "run", serve=True)
    assert servers, "the live view never bound"
    assert "stream-server" not in new_threads(), "the dashboard thread outlived the failed build"
    assert new_threads() == []
    assert [c.close_calls for c in cameras] == [1] * len(cameras)
    assert not controllers[0].is_open, "torn down behind the controller's back: it still reports open"
    with pytest.raises(OSError):  # the live-view port was released
        socket.create_connection(("127.0.0.1", servers[0].port), timeout=1).close()


def test_a_failed_build_never_closes_an_external_live_view(tmp_path, monkeypatch, cameras, new_threads):
    from cascade.apps.external_view import ExternalLiveViewController

    monkeypatch.setenv("CASCADE_EXTERNAL_VIEW_URL", "http://100.100.1.1:8090")
    closed = []
    monkeypatch.setattr(ExternalLiveViewController, "close", lambda self, reason="": closed.append(reason))
    monkeypatch.setattr(ExternalLiveViewController, "status", lambda self: {"ok": False, "open": False})

    def graspgenx_down(runtime):
        assert isinstance(runtime.live_view, ExternalLiveViewController)
        raise RuntimeError("GraspGen-X required at startup: connection refused")

    monkeypatch.setattr(demo, "_probe_grasp_backend", graspgenx_down)
    with pytest.raises(RuntimeError, match="GraspGen-X required"):
        demo.build_runtime(_cfg(), tmp_path / "run", serve=True)
    assert closed == [], "the supervised camera service is not ours to close"
    assert new_threads() == []
    assert [c.close_calls for c in cameras] == [1] * len(cameras)


class _StuckDetector(_ClosableDetector):
    """The watcher's first tick never returns until released (a cold CUDA inference)."""

    def __init__(self):
        super().__init__()
        self.entered, self.release = threading.Event(), threading.Event()

    def detect(self, frame, classes=None):
        self.entered.set()
        self.release.wait(30)
        return []


def test_a_stuck_watcher_bounds_the_failed_build_teardown(tmp_path, monkeypatch, cameras, new_threads, capsys):
    from cascade.apps import map_tile

    monkeypatch.delenv("CASCADE_STREAM", raising=False)
    monkeypatch.setattr(demo, "_FAILED_BUILD_JOIN_S", 0.2)
    detector = _StuckDetector()
    monkeypatch.setattr(demo, "_make_detector", lambda cfg: detector)
    real_settings = map_tile.map_tile_settings

    def settings_once_the_watcher_is_stuck(*args, **kwargs):
        assert detector.entered.wait(10), "the watcher never reached the detector"
        return real_settings(*args, **kwargs)

    monkeypatch.setattr(map_tile, "map_tile_settings", settings_once_the_watcher_is_stuck)
    outcome = {}

    def build():
        try:
            demo.build_runtime(_cfg("yes please"), tmp_path / "run", serve=True)
        except BaseException as e:  # noqa: BLE001 - handed to the test thread
            outcome["error"] = e

    caller = threading.Thread(target=build, daemon=True, name="test-build")
    caller.start()
    try:
        caller.join(10)
        returned = not caller.is_alive()
        alive = new_threads()
    finally:
        detector.release.set()
    caller.join(10)
    assert returned, "a watcher stuck in its tick held the failed build's error back"
    assert isinstance(outcome.get("error"), ValueError) and "nvblox_map" in str(outcome["error"])
    assert alive == ["world-watcher"], "only the stuck watcher may outlive the bound"
    assert "watcher thread 'world-watcher' is still running" in capsys.readouterr().err
    # Released, the watcher leaves (its stop flag was set) and the cameras were still closed.
    deadline = time.monotonic() + 10
    while new_threads() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert new_threads() == []
    assert [c.close_calls for c in cameras] == [1] * len(cameras)
    # The watcher was still inside detect() (holding the detector lock) when
    # the teardown ran: closing the detector would have waited for it.
    assert detector.closed == 0


def test_a_hung_grab_bounds_the_failed_build_teardown(tmp_path, monkeypatch, cameras, new_threads, capsys):
    """A camera grab that never returns delays the error by the bound only,
    and the camera is never closed under that grab (native code in flight)."""
    from cascade.apps import map_tile

    monkeypatch.delenv("CASCADE_STREAM", raising=False)
    monkeypatch.setattr(demo, "_FAILED_BUILD_JOIN_S", 0.2)
    hang, grabbing, release = threading.Event(), threading.Event(), threading.Event()
    real_make = demo.make_camera  # the `cameras` spy

    def hanging_camera(ccfg):
        cam = real_make(ccfg)
        get_frame = cam.get_frame

        def maybe_hang():
            if hang.is_set():
                grabbing.set()
                release.wait(30)
            return get_frame()

        cam.get_frame = maybe_hang
        return cam

    monkeypatch.setattr(demo, "make_camera", hanging_camera)
    real_settings = map_tile.map_tile_settings

    def settings_once_a_grab_hangs(*args, **kwargs):
        hang.set()
        assert grabbing.wait(10), "the stream never grabbed again"
        return real_settings(*args, **kwargs)

    monkeypatch.setattr(map_tile, "map_tile_settings", settings_once_a_grab_hangs)
    outcome = {}

    def build():
        try:
            demo.build_runtime(_cfg("yes please"), tmp_path / "run", serve=True)
        except BaseException as e:  # noqa: BLE001 - handed to the test thread
            outcome["error"] = e

    caller = threading.Thread(target=build, daemon=True, name="test-build")
    caller.start()
    try:
        caller.join(10)
        returned = not caller.is_alive()
        alive = new_threads()
        closed_under_grab = [c.close_calls for c in cameras]
    finally:
        release.set()
    caller.join(10)
    assert returned, "a hung grab held the failed build's error back"
    assert isinstance(outcome.get("error"), ValueError) and "nvblox_map" in str(outcome["error"])
    assert alive == ["stream-mock_small"], "only the hung stream may outlive the bound"
    assert closed_under_grab == [0], "the camera was closed under a grab in flight"
    assert "stream-mock_small thread 'stream-mock_small' is still running" in capsys.readouterr().err
    deadline = time.monotonic() + 10
    while new_threads() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert new_threads() == [], "the released stream did not leave (its stop flag was set)"
