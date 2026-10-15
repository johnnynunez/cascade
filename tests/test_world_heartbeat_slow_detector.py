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
3. One heartbeat per tick means a tick must never block for the 5 s
   watchdog: the watcher calls `detect_nowait` (the brain detector answers
   from its cache and asks the brain on a background thread). Detectors
   without one keep `detect()`.

Every wait here is bounded by an Event; no test sleeps for the detector's
nominal duration.
"""
from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cascade.perception import world
from cascade.perception.brain_detector import BrainDetector
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


# ── 3. the watcher never blocks on the brain (detect_nowait) ─────────────


def _wait_for(predicate, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class _HungBrain:
    """An OpenAI-compatible client whose request hangs until `gate` is set,
    then answers one paper cup; records the thread of every request."""

    REPLY = '[{"label": "paper cup", "bbox_2d": [100, 200, 300, 600]}]'

    def __init__(self):
        self.gate = threading.Event()
        self.entered = threading.Event()
        self.threads: list[str] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.threads.append(threading.current_thread().name)
        self.entered.set()
        self.gate.wait(5.0)
        message = SimpleNamespace(content=self.REPLY)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


class _BoxMasks:
    """Box-prompted segmenter stand-in: one box-shaped mask per box."""

    def masks(self, frame, boxes):
        out = []
        for x0, y0, x1, y1 in np.asarray(boxes).reshape(-1, 4).astype(int):
            mask = np.zeros(frame.rgb.shape[:2], dtype=bool)
            mask[y0:y1, x0:x1] = True
            out.append(mask)
        return out


class _TimedHarness:
    def __init__(self):
        self.times: list[float] = []

    def heartbeat(self):
        self.times.append(time.monotonic())

    @property
    def beats(self) -> int:
        return len(self.times)


def _labels(dets) -> list[str]:
    return [d.label for d in dets]


def test_watcher_keeps_beating_while_the_brain_request_hangs():
    brain = _HungBrain()
    detector = BrainDetector(base_url="http://brain/v1", model="qwen3.8-27b", client=brain,
                             segmenter=_BoxMasks())
    stream = CameraStream(MockCamera(), name="cam", rate_hz=60.0)
    harness = _TimedHarness()
    cam = WatchedCamera(stream, SimpleNamespace(ensure_depth=lambda f: f), None, fuse=False)
    watcher = WorldWatcher([cam], LockedDetector(detector), SimpleNamespace(), rate_hz=30.0,
                           harness=harness)
    stream.open()
    watcher.start()
    try:
        assert brain.entered.wait(1.0), "the watcher never asked the brain"
        start = harness.beats
        # Several more ticks while the (cold, ~5 s in reality) inventory request hangs.
        assert _wait_for(lambda: harness.beats >= start + 5, 1.0), \
            "the watcher stopped beating while the brain request hung"
        assert not brain.gate.is_set()
        gaps = np.diff(list(harness.times))
        assert gaps.max() < 1.0, f"a {gaps.max():.2f} s heartbeat gap (watchdog 5 s)"
        assert watcher.last_dets["cam"] == [], "nothing is known while the brain still answers"
        assert len(brain.threads) == 1, "one request in flight, not one per tick"
        assert brain.threads[0].startswith("brain-refresh"), f"the brain was asked on {brain.threads[0]}"
        brain.gate.set()
        assert _wait_for(lambda: _labels(watcher.last_dets.get("cam", [])) == ["paper cup"], 2.0), \
            "a later tick never returned the brain's objects"
        assert len(brain.threads) == 1  # the answer is tracked, not asked again
    finally:
        brain.gate.set()
        watcher.stop()
        stream.close()
        detector.close()


class _PrepareBlocks:
    """prepare() holds LockedDetector's RLock until released (a model warm-up)."""

    def __init__(self):
        self.preparing = threading.Event()
        self.release = threading.Event()

    def prepare(self, classes):
        self.preparing.set()
        assert self.release.wait(2.0), "test never released prepare()"

    def set_classes(self, classes):
        pass

    def detect(self, frame, classes=None):
        return ["detect"]

    def detect_nowait(self, frame, classes=None):
        return [f"nowait:{classes}"]


class _SafePrepareBlocks(_PrepareBlocks):
    concurrent_safe = True


def _hold_the_lock_in_prepare(inner):
    locked = LockedDetector(inner)
    holder = threading.Thread(target=locked.prepare, args=(None,), daemon=True, name="test-prepare")
    holder.start()
    assert inner.preparing.wait(1.0), "prepare() never started"
    return locked, holder


def test_locked_nowait_on_a_concurrent_safe_detector_takes_no_lock():
    inner = _SafePrepareBlocks()
    locked, holder = _hold_the_lock_in_prepare(inner)
    try:
        done, out = _in_background(lambda: locked.detect_nowait(_frame(), classes=["cup"]))
        assert done.wait(1.0), "detect_nowait waited for the detector lock"
        assert out["value"] == ["nowait:['cup']"]
        assert holder.is_alive(), "prepare() must still hold the lock"
    finally:
        inner.release.set()
        holder.join(2.0)
    assert not holder.is_alive()


def test_locked_nowait_on_other_detectors_runs_under_the_lock():
    inner = _PrepareBlocks()
    locked, holder = _hold_the_lock_in_prepare(inner)
    try:
        done, out = _in_background(lambda: locked.detect_nowait(_frame(), classes=["cup"]))
        assert not done.wait(0.1), "detect_nowait of a non-concurrent-safe detector skipped the lock"
    finally:
        inner.release.set()
        holder.join(2.0)
    assert done.wait(2.0) and out["value"] == ["nowait:['cup']"]


def test_locked_nowait_without_one_falls_back_to_detect_under_the_lock():
    inner = _InventoryBlocks()  # no flag, no detect_nowait
    locked, background = _start_inventory(inner)  # holds the RLock inside detect
    try:
        done, out = _in_background(lambda: locked.detect_nowait(_frame(), classes=["cup"]))
        assert not done.wait(0.1), "the fallback detect skipped the lock"
    finally:
        inner.release.set()
        background.join(2.0)
    assert done.wait(2.0) and out["value"] == ["refer:cup"]


def _instance_nowait_only():
    # Only the class counts: a Mock (or an instance attribute) is not the contract.
    d = SimpleNamespace(detect=lambda frame, classes=None: ["detect"])
    d.detect_nowait = lambda frame, classes=None: ["nowait"]
    return d


@pytest.mark.parametrize("make", [lambda: Mock(detect=Mock(return_value=["detect"])), _instance_nowait_only],
                         ids=["mock", "instance-attribute"])
def test_nowait_is_the_class_contract_only(make):
    assert LockedDetector(make()).detect_nowait(_frame()) == ["detect"]
    stream = SimpleNamespace(name="cam", latest=lambda: _frame(1), set_overlay=lambda **_: None)
    cam = WatchedCamera(stream, SimpleNamespace(ensure_depth=lambda f: f), None, fuse=False)
    watcher = WorldWatcher([cam], make(), SimpleNamespace(), harness=_CountingHarness())
    watcher._tick(cam)
    assert watcher.last_dets["cam"] == ["detect"]


class _Nowait:
    def detect(self, frame, classes=None):
        return ["detect"]

    def detect_nowait(self, frame, classes=None):
        return [f"nowait:{classes}"]


@pytest.mark.parametrize("make", [_Nowait, lambda: LockedDetector(_Nowait())], ids=["bare", "locked"])
def test_the_watcher_calls_detect_nowait_with_its_classes(make):
    stream = SimpleNamespace(name="cam", latest=lambda: _frame(1), set_overlay=lambda **_: None)
    cam = WatchedCamera(stream, SimpleNamespace(ensure_depth=lambda f: f), None, fuse=False)
    watcher = WorldWatcher([cam], make(), SimpleNamespace(), classes=["cup"], harness=_CountingHarness())
    watcher._tick(cam)
    assert watcher.last_dets["cam"] == ["nowait:['cup']"]


# ── 4. Ruling 13: a raising detector still sent this frame's heartbeat ───


class _RaisingDetect:
    def detect(self, frame, classes=None):
        raise RuntimeError("CUDA error: an illegal memory access")


class _RaisingNowait:
    def detect(self, frame, classes=None):
        return []

    def detect_nowait(self, frame, classes=None):
        raise RuntimeError("CUDA error: an illegal memory access")


@pytest.mark.parametrize("make", [_RaisingDetect, _RaisingNowait, lambda: LockedDetector(_RaisingNowait())],
                         ids=["detect", "detect_nowait", "locked-detect_nowait"])
def test_a_raising_detector_still_sent_the_heartbeat(make):
    stream = SimpleNamespace(name="cam", latest=lambda: _frame(1), set_overlay=lambda **_: None)
    cam = WatchedCamera(stream, SimpleNamespace(ensure_depth=lambda f: f), None, fuse=False)
    harness = _CountingHarness()
    watcher = WorldWatcher([cam], make(), SimpleNamespace(), harness=harness)
    with pytest.raises(RuntimeError, match="CUDA error"):
        watcher._tick(cam)
    # The heartbeat goes out before the detector call: a crashed detector does
    # not hold motion through the watchdog (a dead camera still does).
    assert harness.beats == 1
    assert watcher.ticks == 0 and "cam" not in watcher.last_dets
