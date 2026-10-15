"""A slow detector must neither starve the motion watchdog nor serialize callers.

The brain detector's inventory call takes ~5 s and its refer call ~2 s.
1. WorldWatcher sends `harness.heartbeat()` once per NEW processed frame,
   ahead of semantic inference: a detector still running cannot hold back
   the liveness signal of the frame it is analysing.
2. `LockedDetector` skips its global RLock for detectors whose class
   declares `concurrent_safe = True` (they do their own locking): a
   foreground refer or a `reserve()` must not queue behind a background
   inventory and trip the 5 s analysed-image age limit. Every other detector
   keeps today's serialization (YOLO predict is not re-entrant).

Every wait here is bounded by an Event; no test sleeps for the detector's
nominal duration.
"""
from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.perception import world
from cascade.perception.mock_camera import MockCamera
from cascade.perception.stream import CameraStream
from cascade.perception.world import LockedDetector, WatchedCamera, WorldWatcher
from cascade.types import Frame


def _frame(frame_id: int = 1) -> Frame:
    return Frame(rgb=np.zeros((8, 8, 3), dtype=np.uint8), depth_m=None, K=np.eye(3),
                 frame_id=frame_id)


class _CountingHarness:
    def __init__(self):
        self.beats = 0

    def heartbeat(self):
        self.beats += 1


class _BlockedDetector:
    """detect() blocks like a 10 s brain call, released by an Event."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def detect(self, frame, classes=None):
        self.entered.set()
        self.release.wait(10.0)
        self.finished.set()
        return []


# ── 1. heartbeat ahead of detection ──────────────────────────────────────


def test_heartbeat_fires_while_the_detector_is_still_running():
    stream = CameraStream(MockCamera(), name="cam", rate_hz=60.0)
    detector, harness = _BlockedDetector(), _CountingHarness()
    cam = WatchedCamera(stream, SimpleNamespace(ensure_depth=lambda f: f), None, fuse=False)
    watcher = WorldWatcher([cam], detector, SimpleNamespace(), rate_hz=30.0, harness=harness)
    stream.open()
    watcher.start()
    try:
        assert detector.entered.wait(1.0), "the watcher never reached detection"
        first = stream.latest().frame_id
        # 0.5 s of new frames while the detector is blocked on the first one.
        time.sleep(0.5)
        assert stream.latest().frame_id > first, "the mock camera stopped producing frames"
        assert not detector.finished.is_set(), "the detector must still be blocked"
        assert harness.beats >= 1, "a running detector starved the motion watchdog"
    finally:
        detector.release.set()
        watcher.stop()
        stream.close()


def test_a_duplicate_frame_id_sends_no_heartbeat():
    current = {"frame": _frame(7)}
    stream = SimpleNamespace(name="cam", latest=lambda: current["frame"],
                             set_overlay=lambda **_: None)
    detected = []
    detector = SimpleNamespace(detect=lambda frame, classes=None: detected.append(frame) or [])
    harness = _CountingHarness()
    cam = WatchedCamera(stream, SimpleNamespace(ensure_depth=lambda f: f), None, fuse=False)
    watcher = WorldWatcher([cam], detector, SimpleNamespace(), harness=harness)

    watcher._tick(cam)
    assert harness.beats == 1 and len(detected) == 1
    watcher._tick(cam)  # same frame_id: not a new frame, not alive-ness
    current["frame"] = _frame(7)  # a re-delivery that reuses the id
    watcher._tick(cam)
    assert harness.beats == 1 and len(detected) == 1
    current["frame"] = _frame(8)
    watcher._tick(cam)
    assert harness.beats == 2 and len(detected) == 2


# ── 2. LockedDetector: concurrent-safe detectors skip the global lock ────


class _InventoryBlocks:
    """An open scan (classes=None) blocks until released; a refer returns at once."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def detect(self, frame, classes=None):
        if classes is None:
            self.entered.set()
            assert self.release.wait(2.0), "test never released the inventory"
            return ["inventory"]
        return [f"refer:{classes[0]}"]

    def set_classes(self, classes):
        pass


class _ConcurrentSafe(_InventoryBlocks):
    concurrent_safe = True


class _FlagFalse(_InventoryBlocks):
    concurrent_safe = False


def _instance_flag_only():
    # A per-instance attribute is not the declared class contract (a Mock
    # answers every attribute with something truthy).
    d = _InventoryBlocks()
    d.concurrent_safe = True
    return d


def _in_background(fn):
    done, out = threading.Event(), {}

    def run():
        out["value"] = fn()
        done.set()

    threading.Thread(target=run, daemon=True, name="test-caller").start()
    return done, out


def _reserve_and_leave(locked):
    with locked.reserve():
        return "reserved"


def _start_inventory(inner, *, inside_reserve=False):
    """Start a blocked open scan in another thread: bare (the watcher) or
    inside a reserve() scope (a foreground capture-then-detect)."""
    locked = LockedDetector(inner)

    def scan():
        if inside_reserve:
            with locked.reserve():
                return locked.detect(_frame())
        return locked.detect(_frame())

    background = threading.Thread(target=scan, daemon=True, name="test-inventory")
    background.start()
    assert inner.entered.wait(1.0), "the inventory never started"
    return locked, background


@pytest.mark.parametrize("inside_reserve", [False, True], ids=["watcher-scan", "reserved-scan"])
def test_concurrent_safe_refer_and_reserve_do_not_wait_for_a_running_inventory(inside_reserve):
    inner = _ConcurrentSafe()
    locked, background = _start_inventory(inner, inside_reserve=inside_reserve)
    try:
        refer_done, refer = _in_background(lambda: locked.detect(_frame(), classes=["cup"]))
        assert refer_done.wait(1.0), "a refer queued behind the background inventory"
        assert refer["value"] == ["refer:cup"]
        reserve_done, reserve = _in_background(lambda: _reserve_and_leave(locked))
        assert reserve_done.wait(1.0), "reserve() waited for the background inventory"
        assert reserve["value"] == "reserved"
        assert background.is_alive(), "the inventory must still be blocked"
    finally:
        inner.release.set()
        background.join(2.0)
    assert not background.is_alive()


@pytest.mark.parametrize("make", [_InventoryBlocks, _FlagFalse, _instance_flag_only],
                         ids=["no-flag", "flag-false", "instance-flag-only"])
def test_other_detectors_still_serialize_detect(make):
    inner = make()
    locked, background = _start_inventory(inner)
    try:
        refer_done, refer = _in_background(lambda: locked.detect(_frame(), classes=["cup"]))
        assert not refer_done.wait(0.1), "a non-concurrent-safe detector ran two detects at once"
    finally:
        inner.release.set()
        background.join(2.0)
    assert refer_done.wait(2.0) and refer["value"] == ["refer:cup"]


def test_reserve_on_other_detectors_still_waits_for_a_running_detect():
    inner = _InventoryBlocks()
    locked, background = _start_inventory(inner)
    try:
        reserve_done, reserve = _in_background(lambda: _reserve_and_leave(locked))
        assert not reserve_done.wait(0.1), "reserve() did not wait for the running detect"
    finally:
        inner.release.set()
        background.join(2.0)
    assert reserve_done.wait(2.0) and reserve["value"] == "reserved"


def test_concurrent_safe_slow_call_is_still_logged_with_zero_lock_wait(monkeypatch, caplog):
    clock = {"t": 100.0}
    monkeypatch.setattr(world, "time", SimpleNamespace(monotonic=lambda: clock["t"]))

    class Slow:
        concurrent_safe = True

        def detect(self, frame, classes=None):
            clock["t"] += 1.5
            return []

    with caplog.at_level(logging.INFO, logger=world.logger.name):
        assert LockedDetector(Slow()).detect(_frame(), classes=["cup"]) == []
    assert ("Detector timing: lock wait 0.000s, inference 1.500s, classes=['cup']"
            in caplog.messages)
