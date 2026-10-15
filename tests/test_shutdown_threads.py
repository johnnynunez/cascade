"""Shutdown must not leave a worker thread running native code at exit.

The launcher's runtime check aborted with "terminate called without an
active exception" in 3 of 20 kitchen launches (3 of 3 with a cold CUDA kernel
cache). WorldWatcher.stop() joined the watcher with timeout=5 and returned
while the first YOLOE inference was still running. The thread then died
inside torch at interpreter exit (perception/thread_join.py has the gdb
evidence). These tests pin the contract with fakes that block inside a tick:

* stop()/close() do not return while the work in flight is still running;
* shutdown_runtime() returns with no owned thread alive;
* a watcher that nobody stops is still joined at exit, before teardown;
* the brain detector is closed after the watcher: its refresh threads and its
  SAM warm-up are joined (bounded), and a warm-up past its bound is named in
  the receipt's pending threads.

`compressed_join_bounds` divides every finite Thread.join timeout by 100, so a
bounded join (the old `join(timeout=5)`) gives up after 50 ms. That models a
tick that outlasts any bound without making the suite wait 5 s. Test code
waits on Events only, never on a (compressed) join.
"""

from __future__ import annotations

import logging
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


# ── the brain detector: closed after the watcher, its threads joined ──────


class _GatedBrain:
    """OpenAI-compatible client: every request waits for `gate`, then answers `[]`."""

    def __init__(self):
        self.gate, self.entered = threading.Event(), threading.Event()
        self.threads: list[str] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.threads.append(threading.current_thread().name)
        self.entered.set()
        self.gate.wait(10)
        message = SimpleNamespace(content="[]")
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


def _brain_refresh_threads() -> list[str]:
    return [t.name for t in threading.enumerate() if t.name.startswith("brain-refresh") and t.is_alive()]


def _stages(receipt) -> list[str]:
    return [s["stage"] for s in receipt["stages"]]


def test_shutdown_runtime_closes_the_brain_detector_and_joins_its_refreshes(tmp_path, monkeypatch):
    pytest.importorskip("pinocchio")
    from cascade.apps import demo
    from cascade.config import load_demo_config
    from cascade.perception.brain_detector import BrainDetector

    closes: list[str] = []

    class SpyBrain(BrainDetector):
        def close(self):
            closes.append(threading.current_thread().name)
            super().close()

    brain = _GatedBrain()
    # No segmenter, so no SAM warm-up: only the watcher's background refresh.
    detector = SpyBrain(base_url="http://brain/v1", model="qwen3.8-27b", api_key_file=None, client=brain)
    monkeypatch.setattr(demo, "_make_detector", lambda cfg: detector)
    runtime, arm = demo.build_runtime(load_demo_config(camera="mock", arm="mock", llm="mock"),
                                      tmp_path / "run", view=False, serve=False)
    at_return: dict = {}

    def shutdown():
        at_return["receipt"] = shutdown_runtime(runtime, arm)
        at_return["refreshing"] = _brain_refresh_threads()

    returned = None
    try:
        assert brain.entered.wait(5), "the watcher never started a brain refresh"
        assert brain.threads[0].startswith("brain-refresh")
        returned = _in_background(shutdown)
        early = returned.wait(0.5)
    finally:
        brain.gate.set()
        if returned is None:
            shutdown_runtime(runtime, arm)
    assert returned.wait(10)
    assert not early, "shutdown_runtime returned while a brain refresh was in flight"
    assert len(closes) == 1
    assert at_return["refreshing"] == [], "a brain refresh outlived shutdown_runtime"
    receipt = at_return["receipt"]
    stages = _stages(receipt)
    assert stages.index("watcher") < stages.index("detector") < stages.index("cameras")
    detector_stage, = (s for s in receipt["stages"] if s["stage"] == "detector")
    assert detector_stage["ok"] is True and receipt["pending_threads"] == []


def test_a_detector_close_failure_is_logged_and_shutdown_goes_on(tmp_path, monkeypatch, caplog):
    pytest.importorskip("pinocchio")
    from cascade.apps import demo
    from cascade.config import load_demo_config
    from cascade.perception.brain_detector import BrainDetector

    class BrokenClose(BrainDetector):
        def close(self):
            super().close()
            raise RuntimeError("close broke")

    detector = BrokenClose(base_url="http://brain/v1", model="m", api_key_file=None, client=_GatedBrain())
    detector._client.gate.set()
    monkeypatch.setattr(demo, "_make_detector", lambda cfg: detector)
    runtime, arm = demo.build_runtime(load_demo_config(camera="mock", arm="mock", llm="mock"),
                                      tmp_path / "run", view=False, serve=False)
    with caplog.at_level(logging.WARNING, logger=demo.logger.name):
        receipt = shutdown_runtime(runtime, arm)  # never raises
    stage, = (s for s in receipt["stages"] if s["stage"] == "detector")
    assert stage["ok"] is False and stage["errors"][0]["message"] == "close broke"
    assert any("close broke" in m for m in caplog.messages)
    assert _stages(receipt)[-2:] == ["cameras", "arms"]  # the later stages still ran
    assert next(s for s in receipt["stages"] if s["stage"] == "arms")["ok"] is True


class _BlockedWarmUp:
    """Stands in for demo._warm_box_segmenter (SAM load + a CUDA predict):
    blocks until the test releases it."""

    def __init__(self):
        self.started, self.release, self.finished = threading.Event(), threading.Event(), threading.Event()

    def __call__(self, segmenter):
        self.started.set()
        self.release.wait(10)
        self.finished.set()


@pytest.fixture
def brain_runtime(tmp_path, monkeypatch):
    """A mock-rig runtime built by the REAL factory from configs/demo.yaml's
    brain block, with a blocked fake SAM warm-up. No watcher, so nothing asks
    the brain or loads SAM; a brain call would be recorded in `asked`."""
    pytest.importorskip("pinocchio")
    from cascade.apps import demo
    from cascade.config import load_demo_config
    from cascade.perception.brain_detector import BrainDetector

    for name in ("CASCADE_PERCEPTION_URL", "CASCADE_PERCEPTION_MODEL", "CASCADE_BOOTH", "CASCADE_ROBOT"):
        monkeypatch.delenv(name, raising=False)
    key = tmp_path / "brain.key"
    key.write_text("k")
    monkeypatch.setenv("CASCADE_PERCEPTION_KEY_FILE", str(key))
    asked: list[str] = []

    def no_brain(self):
        asked.append(threading.current_thread().name)
        raise ConnectionError("tests never reach the brain")

    monkeypatch.setattr(BrainDetector, "_get_client", no_brain)
    warm = _BlockedWarmUp()
    monkeypatch.setattr(demo, "_warm_box_segmenter", warm)
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    cfg._data["camera"].pop("detector")  # the mock camera pins the mock detector
    cfg._data["perception_loop"] = {"enabled": False}
    runtime, arm = demo.build_runtime(cfg, tmp_path / "run", view=False, serve=False)
    try:
        assert type(runtime.detector._detector) is BrainDetector
        assert warm.started.wait(5), "the factory started no SAM warm-up"
        yield runtime, arm, warm
    finally:
        warm.release.set()
        shutdown_runtime(runtime, arm)
    assert asked == []


def test_owned_threads_names_the_sam_warm_up(brain_runtime):
    runtime, _, _ = brain_runtime
    owned = dict(owned_threads(runtime))
    assert owned["sam-warm-up"].name == "sam-warm-up"
    assert owned["sam-warm-up"] is runtime.detector._detector.warm_up_thread


def test_shutdown_runtime_waits_for_the_sam_warm_up(brain_runtime):
    runtime, arm, warm = brain_runtime
    thread = runtime.detector._detector.warm_up_thread
    at_return: dict = {}

    def shutdown():
        at_return["receipt"] = shutdown_runtime(runtime, arm)
        at_return.update(finished=warm.finished.is_set(), alive=thread.is_alive())

    try:
        returned = _in_background(shutdown)
        early = returned.wait(0.3)
    finally:
        warm.release.set()
    assert returned.wait(10)
    assert not early, "shutdown_runtime returned while the SAM warm-up was still running"
    assert (at_return["finished"], at_return["alive"]) == (True, False)
    receipt = at_return["receipt"]
    assert "detector" in _stages(receipt) and receipt["pending_threads"] == []


def test_shutdown_runtime_reports_a_sam_warm_up_that_outlasts_its_bound(brain_runtime, monkeypatch, capsys):
    from cascade.perception import brain_detector

    runtime, arm, warm = brain_runtime
    monkeypatch.setattr(brain_detector, "_WARM_UP_JOIN_S", 0.2)
    thread = runtime.detector._detector.warm_up_thread
    try:
        receipt = shutdown_runtime(runtime, arm)  # the bound expires: returns, names the thread
        assert thread.is_alive()
    finally:
        warm.release.set()
        thread.join(5)
    assert receipt["pending_threads"] == [{"owner": "sam-warm-up", "thread": "sam-warm-up"}]
    assert receipt["complete"] is False
    assert "sam-warm-up thread 'sam-warm-up' is still running after shutdown" in capsys.readouterr().err


def test_the_mock_runtime_keeps_its_teardown_stages(tmp_path):
    pytest.importorskip("pinocchio")
    from cascade.apps import demo
    from cascade.config import load_demo_config

    runtime, arm = demo.build_runtime(load_demo_config(camera="mock", arm="mock", llm="mock"),
                                      tmp_path / "run", view=False, serve=False)
    receipt = shutdown_runtime(runtime, arm)
    assert "detector" not in _stages(receipt)  # MockDetector has no close()
    assert "sam-warm-up" not in dict(owned_threads(runtime))
