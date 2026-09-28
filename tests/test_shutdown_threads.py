"""Shutdown must not leave a worker thread running native code at exit.

The launcher's runtime check aborted with "terminate called without an
active exception" in 3 of 20 kitchen launches (3 of 3 with a cold CUDA kernel
cache). WorldWatcher.stop() joined the watcher with timeout=5 and returned
while the first YOLOE inference was still running. The thread then died
inside torch at interpreter exit (perception/thread_join.py has the gdb
evidence). These tests pin the contract with fakes that block inside a tick:

* stop()/close() do not return while the work in flight is still running;
* shutdown_runtime() returns with no owned thread alive;
* a watcher that nobody stops is still joined at exit, before teardown.

`compressed_join_bounds` divides every finite Thread.join timeout by 100, so a
bounded join (the old `join(timeout=5)`) gives up after 50 ms. That models a
tick that outlasts any bound without making the suite wait 5 s. Test code
waits on Events only, never on a (compressed) join.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cascade.apps.demo import owned_threads, shutdown_runtime
from cascade.apps.live_view import RigViewer
from cascade.perception.stream import CameraRig, CameraStream
from cascade.perception.thread_join import join_thread
from cascade.perception.world import WatchedCamera, WorldWatcher
from cascade.types import Frame


@pytest.fixture
def compressed_join_bounds(monkeypatch):
    real_join = threading.Thread.join

    def join(self, timeout=None):
        return real_join(self, None if timeout is None else timeout / 100.0)

    monkeypatch.setattr(threading.Thread, "join", join)


def _frame(frame_id: int = 1) -> Frame:
    return Frame(rgb=np.zeros((8, 8, 3), dtype=np.uint8), depth_m=None, K=np.eye(3),
                 frame_id=frame_id)


class BlockingDetector:
    """detect() blocks until released: a cold first inference outlasting any bound."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def detect(self, frame, classes=None):
        self.entered.set()
        self.release.wait(30)
        self.finished.set()
        return []


class BlockingCamera:
    """get_frame() blocks until released: a bridge `frame` request in flight."""

    has_depth = False

    def __init__(self):
        self.grabbing = threading.Event()
        self.release = threading.Event()
        self.in_grab = False
        self.closed = False
        self.closed_during_grab = False

    def open(self):
        pass

    def get_frame(self):
        self.in_grab = True
        self.grabbing.set()
        self.release.wait(30)
        self.in_grab = False
        return _frame()

    def close(self):
        self.closed = True
        self.closed_during_grab = self.in_grab


def _watcher(detector) -> WorldWatcher:
    stream = SimpleNamespace(name="worktop", latest=_frame, set_overlay=lambda **_: None)
    cam = WatchedCamera(stream, SimpleNamespace(ensure_depth=lambda f: f), None, fuse=False)
    return WorldWatcher([cam], detector, SimpleNamespace(), rate_hz=50.0)


def _in_background(fn) -> threading.Event:
    done = threading.Event()

    def run():
        fn()
        done.set()

    threading.Thread(target=run, daemon=True, name="test-caller").start()
    return done


def test_watcher_stop_waits_for_the_tick_in_flight(compressed_join_bounds):
    det = BlockingDetector()
    watch = _watcher(det)
    watch.start()
    thread = watch._thread
    try:
        assert det.entered.wait(5), "the watcher never reached the detector"
        returned = _in_background(watch.stop)
        early = returned.wait(0.5)
    finally:
        det.release.set()
    assert not early, "stop() returned while the detector was still running"
    assert returned.wait(5)
    assert det.finished.is_set() and not thread.is_alive()
    assert watch._thread is None


def test_watcher_stop_with_a_bound_keeps_the_handle_when_it_expires():
    # The bounded form exists for the atexit safety net only. If the bound
    # expires, the handle must stay, so a later stop() can still join it.
    det = BlockingDetector()
    watch = _watcher(det)
    watch.start()
    thread = watch._thread
    try:
        assert det.entered.wait(5)
        watch.stop(timeout_s=0.1)
        assert watch._thread is thread and thread.is_alive()
    finally:
        det.release.set()
    watch.stop()
    assert watch._thread is None and not thread.is_alive()


def test_camera_stream_close_waits_for_the_grab_in_flight(compressed_join_bounds):
    cam = BlockingCamera()
    stream = CameraStream(cam, name="side")
    stream.open()
    thread = stream._thread
    try:
        assert cam.grabbing.wait(5)
        returned = _in_background(stream.close)
        early = returned.wait(0.5)
        closed_early = cam.closed
    finally:
        cam.release.set()
    assert not early, "close() returned while a grab was still in flight"
    assert not closed_early, "the camera was closed under a grab in flight"
    assert returned.wait(5)
    assert cam.closed and not cam.closed_during_grab and not thread.is_alive()


def test_rig_viewer_stop_waits_for_its_loop(compressed_join_bounds):
    viewer = RigViewer(rig=None)
    drawing, release = threading.Event(), threading.Event()

    def loop():  # stands in for a cv2 draw in flight
        drawing.set()
        release.wait(30)

    viewer._thread = threading.Thread(target=loop, daemon=True, name="rig-viewer")
    viewer._thread.start()
    thread = viewer._thread
    try:
        assert drawing.wait(5)
        returned = _in_background(viewer.stop)
        early = returned.wait(0.5)
    finally:
        release.set()
    assert not early, "stop() returned while the viewer loop was still running"
    assert returned.wait(5) and not thread.is_alive()


def _runtime(watcher, rig, camera):
    return SimpleNamespace(watcher=watcher, rig=rig, camera=camera, stream_server=None,
                           viewer=None, arm=None, arm_rig=None, beliefs_path=None)


def test_shutdown_runtime_leaves_no_owned_thread_running(compressed_join_bounds):
    det, cam = BlockingDetector(), BlockingCamera()
    watch = _watcher(det)
    stream = CameraStream(cam, name="worktop")
    rig = CameraRig([stream])
    runtime = _runtime(watch, rig, stream)
    arm = SimpleNamespace(disconnect=Mock())
    rig.open()
    watch.start()
    owned = owned_threads(runtime)
    assert sorted(label for label, _ in owned) == ["stream-worktop", "watcher"]
    assert det.entered.wait(5) and cam.grabbing.wait(5)
    alive_at_return: list[str] = []

    def shutdown():
        shutdown_runtime(runtime, arm)
        alive_at_return.extend(t.name for _, t in owned if t.is_alive())

    try:
        returned = _in_background(shutdown)
        early = returned.wait(0.5)
    finally:
        det.release.set()
        cam.release.set()
    assert returned.wait(5)
    assert alive_at_return == [], f"still running when shutdown_runtime returned: {alive_at_return}"
    assert not early
    assert not cam.closed_during_grab
    arm.disconnect.assert_called_once()


def test_shutdown_runtime_names_a_thread_an_owner_left_running(capsys):
    stuck = threading.Event()
    thread = threading.Thread(target=stuck.wait, args=(30,), daemon=True, name="world-watcher")
    thread.start()
    regressed = SimpleNamespace(_thread=thread, stop=lambda: None)  # returns without joining
    runtime = _runtime(regressed, None, SimpleNamespace(close=lambda: None))
    try:
        shutdown_runtime(runtime, SimpleNamespace(disconnect=lambda: None))
    finally:
        stuck.set()
        thread.join(5)
    err = capsys.readouterr().err
    assert "watcher thread 'world-watcher' is still running after shutdown" in err


def test_a_watcher_nobody_stops_is_joined_before_interpreter_teardown(tmp_path):
    # End to end in a fresh interpreter: the script returns with a tick in
    # flight and never calls stop(). Without the atexit join the daemon
    # thread is killed mid-tick at exit (inside torch, that is the abort).
    marks = tmp_path / "marks.txt"
    script = textwrap.dedent("""
        import sys, time
        from types import SimpleNamespace
        import numpy as np
        from cascade.perception.world import WatchedCamera, WorldWatcher
        from cascade.types import Frame

        out = open(sys.argv[1], "w", buffering=1)

        class SlowDetector:
            def detect(self, frame, classes=None):
                out.write("tick-start\\n")
                time.sleep(1.0)
                out.write("tick-end\\n")
                return []

        frame = Frame(rgb=np.zeros((8, 8, 3), np.uint8), depth_m=None, K=np.eye(3), frame_id=1)
        stream = SimpleNamespace(name="cam", latest=lambda: frame, set_overlay=lambda **_: None)
        cam = WatchedCamera(stream, SimpleNamespace(ensure_depth=lambda f: f), None, fuse=False)
        watcher = WorldWatcher([cam], SlowDetector(), SimpleNamespace())
        watcher.start()
        while "tick-start" not in open(sys.argv[1]).read():
            time.sleep(0.01)
    """)
    proc = subprocess.run([sys.executable, "-c", script, str(marks)],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert marks.read_text().splitlines() == ["tick-start", "tick-end"]


def test_join_thread_contract(capsys):
    assert join_thread(None, what="x") is True
    done = threading.Thread(target=lambda: None)
    done.start()
    done.join()
    assert join_thread(done, what="x") is True
    assert join_thread(threading.current_thread(), what="x") is False

    release = threading.Event()
    slow = threading.Thread(target=release.wait, args=(30,), daemon=True, name="slow-tick")
    slow.start()
    try:
        assert join_thread(slow, what="probe", timeout_s=0.2) is False
        err = capsys.readouterr().err
        assert "[probe] thread 'slow-tick' still running after" in err
        assert "in wait" in err  # names where the thread is
        threading.Timer(0.35, release.set).start()
        assert join_thread(slow, what="probe", warn_every_s=0.1) is True
        assert "waiting for thread 'slow-tick' to finish" in capsys.readouterr().err
    finally:
        release.set()
