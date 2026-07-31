"""Tests for the on-demand live view (headless-first UI).

The contract being pinned here:
- nothing binds a port until someone asks to look (chat is the interface);
- WRC_STREAM=0 stays a hard kill switch that open() must refuse;
- a re-open after a close works (a ThreadingHTTPServer cannot be restarted,
  so the controller must build a FRESH server each time);
- chat wiring survives close/open cycles;
- an idle view auto-closes, and an active viewer keeps it alive;
- the depth / annotated / analyze surfaces render without a runtime.
"""

from __future__ import annotations

import json
import time
import urllib.request

import numpy as np
import pytest

from wrc_demo.apps.live_control import (
    MODE_EAGER,
    MODE_LAZY,
    MODE_OFF,
    LiveViewController,
    resolve_mode,
)


# ── mode resolution ──────────────────────────────────────────────────────


def _env(**kw):
    return lambda key: kw.get(key)


def test_default_is_lazy_headless():
    """Chat is the UI: with no config and no env, nothing should bind."""
    mode, _ = resolve_mode({}, _env())
    assert mode == MODE_LAZY


def test_wrc_stream_zero_is_a_hard_kill_switch():
    for value in ("0", "off", "false", "no"):
        mode, _ = resolve_mode({"mode": "eager"}, _env(WRC_STREAM=value))
        assert mode == MODE_OFF, f"WRC_STREAM={value} must win over config"


def test_env_overrides_config_mode():
    mode, _ = resolve_mode({"mode": "lazy"}, _env(WRC_STREAM="eager"))
    assert mode == MODE_EAGER
    mode, _ = resolve_mode({"mode": "eager"}, _env(WRC_STREAM="lazy"))
    assert mode == MODE_LAZY


def test_legacy_enabled_false_maps_to_off():
    mode, _ = resolve_mode({"enabled": False}, _env())
    assert mode == MODE_OFF


def test_booth_config_can_pin_eager():
    mode, idle = resolve_mode({"mode": "eager", "idle_timeout_s": 60}, _env())
    assert mode == MODE_EAGER and idle == 60


# ── controller lifecycle ─────────────────────────────────────────────────


class _FakeServer:
    """Stands in for StreamServer: records lifecycle, no sockets."""

    instances: list = []

    def __init__(self, fail: bool = False):
        self.started = False
        self.stopped = False
        self.task_fn = None
        self.cancel_fn = None
        self._fail = fail
        _FakeServer.instances.append(self)

    def start(self):
        if self._fail:
            raise OSError("address already in use")
        self.started = True

    def stop(self):
        self.stopped = True

    def set_task_fn(self, fn):
        self.task_fn = fn

    def set_cancel_fn(self, fn):
        self.cancel_fn = fn

    @property
    def url(self):
        return "http://127.0.0.1:9999/"


@pytest.fixture(autouse=True)
def _reset_instances():
    _FakeServer.instances.clear()
    yield
    _FakeServer.instances.clear()


def test_lazy_controller_binds_nothing_until_opened():
    ctl = LiveViewController(_FakeServer, mode=MODE_LAZY)
    assert not ctl.is_open and ctl.url is None
    assert _FakeServer.instances == []  # the factory was never called

    out = ctl.open(reason="human asked")
    assert out["ok"] and out["open"] and out["url"].startswith("http")
    assert ctl.is_open and _FakeServer.instances[0].started


def test_open_is_idempotent():
    ctl = LiveViewController(_FakeServer, mode=MODE_LAZY)
    ctl.open()
    out = ctl.open()
    assert out["ok"] and out.get("note") == "already open"
    assert len(_FakeServer.instances) == 1  # no second server built


def test_close_releases_and_reopen_builds_a_fresh_server():
    """A ThreadingHTTPServer cannot be restarted after server_close()."""
    ctl = LiveViewController(_FakeServer, mode=MODE_LAZY)
    ctl.open()
    first = _FakeServer.instances[0]
    assert ctl.close()["open"] is False
    assert first.stopped and not ctl.is_open

    ctl.open()
    assert len(_FakeServer.instances) == 2
    assert _FakeServer.instances[1] is not first
    assert _FakeServer.instances[1].started


def test_close_is_idempotent():
    ctl = LiveViewController(_FakeServer, mode=MODE_LAZY)
    assert ctl.close()["note"] == "already closed"


def test_off_mode_refuses_to_open():
    ctl = LiveViewController(_FakeServer, mode=MODE_OFF)
    out = ctl.open()
    assert not out["ok"] and not out["open"]
    assert "disabled" in out["error"]
    assert _FakeServer.instances == []  # nothing was ever constructed


def test_bind_failure_is_reported_not_raised():
    ctl = LiveViewController(lambda: _FakeServer(fail=True), mode=MODE_LAZY)
    out = ctl.open()
    assert not out["ok"] and "could not bind" in out["error"]
    assert not ctl.is_open  # a failed open must not leave a half-state


def test_chat_wiring_survives_a_close_open_cycle():
    """Otherwise the second open has a dead chat box."""
    ctl = LiveViewController(_FakeServer, mode=MODE_LAZY)
    task, cancel = (lambda t: None), (lambda: None)
    ctl.set_task_fn(task)
    ctl.set_cancel_fn(cancel)

    ctl.open()
    assert _FakeServer.instances[0].task_fn is task
    ctl.close()
    ctl.open()
    assert _FakeServer.instances[1].task_fn is task
    assert _FakeServer.instances[1].cancel_fn is cancel


def test_set_task_fn_reaches_an_already_open_server():
    ctl = LiveViewController(_FakeServer, mode=MODE_LAZY)
    ctl.open()
    fn = lambda t: None  # noqa: E731
    ctl.set_task_fn(fn)
    assert _FakeServer.instances[0].task_fn is fn


def test_idle_view_auto_closes_and_activity_keeps_it_alive():
    ctl = LiveViewController(_FakeServer, mode=MODE_LAZY, idle_timeout_s=0.05)
    ctl.open()
    # a viewer polling keeps it up
    for _ in range(4):
        time.sleep(0.02)
        ctl.note_poll()
        assert ctl.is_open
    # ...and silence makes it reapable. The reaper thread wakes on a 15 s
    # tick, so assert the condition it evaluates rather than sleeping for it.
    time.sleep(0.06)
    assert (time.monotonic() - ctl._last_poll) > ctl.idle_timeout_s
    ctl.close(reason="test")
    assert not ctl.is_open


def test_eager_mode_has_no_idle_timeout_advertised():
    """A booth dashboard must not vanish between visitors."""
    ctl = LiveViewController(_FakeServer, mode=MODE_EAGER)
    ctl.open()
    assert ctl.status()["idle_timeout_s"] is None


def test_status_shape_when_closed_and_open():
    ctl = LiveViewController(_FakeServer, mode=MODE_LAZY)
    closed = ctl.status()
    assert closed["open"] is False and closed["url"] is None
    ctl.open()
    live = ctl.status()
    assert live["open"] is True and live["url"].startswith("http")
    assert live["open_for_s"] >= 0


# ── the diagnostic surfaces (real HTTP) ──────────────────────────────────


def _rig_with_depth():
    from wrc_demo.perception.mock_camera import MockCamera
    from wrc_demo.perception.stream import CameraRig, CameraStream

    cam = MockCamera(_cfg({"type": "mock", "width": 160, "height": 120}))
    stream = CameraStream(cam, name="over")
    return CameraRig([stream]), stream


def _cfg(d):
    from wrc_demo.config import Cfg

    return Cfg(d)


def test_depth_and_analyze_routes_work_without_a_runtime():
    """The live view must render even before/without a SkillRuntime."""
    from wrc_demo.apps.stream_server import StreamServer

    rig, stream = _rig_with_depth()
    rig.open()
    try:
        deadline = time.time() + 5
        while stream.latest() is None and time.time() < deadline:
            time.sleep(0.05)
        server = StreamServer(rig, port=0)
        server.start()
        try:
            base = f"http://127.0.0.1:{server.port}"
            # analyze reports per-camera depth provenance as JSON
            payload = json.loads(urllib.request.urlopen(f"{base}/analyze", timeout=5).read())
            assert payload["ok"] and "over" in payload["cameras"]
            entry = payload["cameras"]["over"]
            assert "depth_source" in entry and "detections" in entry

            # depth renders a JPEG (or an explicit NO DEPTH card), never a 500
            jpeg = server.depth_jpeg("over")
            assert jpeg is not None and jpeg[:2] == b"\xff\xd8"

            # annotated falls back to the detection view with no runtime
            assert server.annotated_view_jpeg("over") is not None

            # the dashboard exposes all three view switches
            html = urllib.request.urlopen(base + "/", timeout=5).read().decode()
            assert "/depth/over" not in html  # switched via JS, not hardcoded
            assert "setView('over','depth'" in html
            assert "setView('over','agent'" in html
            assert 'id="analysis"' in html
        finally:
            server.stop()
    finally:
        rig.close()


def test_analyze_reports_unknown_camera_without_raising():
    from wrc_demo.apps.stream_server import StreamServer

    rig, stream = _rig_with_depth()
    rig.open()
    try:
        server = StreamServer(rig, port=0)
        out = server.analyze("nope")
        assert out["cameras"]["nope"]["error"] == "unknown camera"
    finally:
        rig.close()


def test_on_poll_is_called_by_http_requests():
    """This is what lets the controller reap an idle dashboard."""
    from wrc_demo.apps.stream_server import StreamServer

    rig, stream = _rig_with_depth()
    rig.open()
    polls = []
    try:
        server = StreamServer(rig, port=0, on_poll=lambda: polls.append(1))
        server.start()
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{server.port}/state", timeout=5).read()
        finally:
            server.stop()
    finally:
        rig.close()
    assert polls, "GET /state must refresh the idle timer"


def test_depth_jpeg_handles_a_frame_without_depth():
    from types import SimpleNamespace

    from wrc_demo.apps.stream_server import StreamServer

    class _Stream:
        name = "flat"
        fps = 10.0

        def latest(self):
            return SimpleNamespace(
                rgb=np.zeros((60, 80, 3), dtype=np.uint8),
                depth_m=None, has_depth=False, depth_source="none",
                frame_id=1,
            )

        def overlay(self):
            return [], "idle"

    class _Rig:
        names = ["flat"]

        def get(self, _):
            return _Stream()

        def stats(self):
            return {}

    server = StreamServer(_Rig(), port=0)
    jpeg = server.depth_jpeg("flat")
    assert jpeg is not None and jpeg[:2] == b"\xff\xd8"  # a NO DEPTH card, not a crash
