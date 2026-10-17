"""Background perception never competes with a foreground localization (spec
2026-10-16-perception-two-phase, section 7.2 B2-B5, tests 7-10 of 7.4).

On the Spark sim every foreground refer shared the brain with a watcher
inventory that never finished, and the kitchen's cam0/side cameras shared one
cache key, so each tick restarted the other's inventory. Now:
- B2: a foreground scope (motion skills, preview_grasp, localize_object)
  cancels the streamed background request, starts none while it is held,
  answers the watcher's detect_nowait with [] and no SAM, and starts the
  queued refresh when it ends;
- B3: a frame whose capture names its camera keys the cache by it too, and
  build_runtime warns when two watched cameras still share a key;
- B4: the watcher asks the brain detector nothing for a non-fusing camera;
- B5: background requests have their own timeout (`inventory_timeout_s`).

Scaled real time (`SCALE = 0.1`): the brain's latencies and the 5 s
analysed-image age limit are 10x shorter than live. No network, no GPU.
"""

from __future__ import annotations

import base64
import contextlib
import functools
import json
import logging
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from cascade.perception import brain_detector as brain_mod
from cascade.perception.brain_detector import INVENTORY_PROMPT, REFER_PROMPT, BrainDetector
from cascade.perception.world import LockedDetector, WatchedCamera, WorldWatcher
from cascade.types import Frame
from fake_brain_stream import FakeStream, completion, reply

SCALE = 0.1  # seconds in this test per second live
RATE_HZ = 3.0 / SCALE  # the watcher's 3 Hz, scaled
REFER_S = 3.0 * SCALE  # a refer alone (live 3.0 s)
CONTENTION_S = 3.5 * SCALE  # extra while an inventory decodes beside it (live 3.5 s)
W, H = 320, 180
K = np.array([[500.0, 0, W / 2], [0, 500.0, H / 2], [0, 0, 1]])
CUP = '[{"label": "paper cup", "bbox_2d": [100, 200, 300, 600]}]'


def _wait_for(predicate, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


def _refresh_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith("brain-refresh") and t.is_alive()]


def _image_level(kwargs: dict) -> float:
    """Mean grey level of the JPEG a request carried (tells cameras apart)."""
    url = kwargs["messages"][0]["content"][0]["image_url"]["url"]
    data = base64.b64decode(url.split(",", 1)[1])
    return float(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE).mean())


def _isaac_capture(camera: str, t: float) -> dict:
    """What bridge_client.py puts in Frame.capture (enough for the watcher's
    capture dedupe and the cache key)."""
    return {"backend": "isaac", "source": ("127.0.0.1", 8611), "camera": camera, "t": t,
            "proprioception": {"robot_id": "arm", "backend": "isaac",
                               "time_source": "physics_loop_monotonic", "t": t}}


def _frame(level: int, camera: str | None = None, n: int = 0) -> Frame:
    rgb = np.full((H, W, 3), level, dtype=np.uint8)
    capture = _isaac_capture(camera, float(n)) if camera is not None else None
    return Frame(rgb=rgb, depth_m=None, K=K.copy(), frame_id=n, capture=capture)


class _Segmenter:
    """Counts SAM calls per camera; one box-shaped mask per box."""

    def __init__(self):
        self.calls: list[str | None] = []
        self._lock = threading.Lock()

    def masks(self, frame, boxes):
        with self._lock:
            self.calls.append((frame.capture or {}).get("camera"))
        out = []
        for x0, y0, x1, y1 in np.asarray(boxes).reshape(-1, 4).astype(int):
            mask = np.zeros(frame.rgb.shape[:2], dtype=bool)
            mask[y0:y1, x0:x1] = True
            out.append(mask)
        return out


class _Brain:
    """OpenAI-compatible fake. A background inventory (on a brain-refresh
    thread) keeps decoding until `release` is set when `hold` is True, else
    answers at once; foreground requests answer at once. Records every
    request's kwargs, thread and start time, and every stream closed before
    its answer."""

    def __init__(self, *, hold: bool = False, text: str = CUP):
        self.text = text
        self.hold = hold
        self.release = threading.Event()
        self.requests: list[dict] = []
        self.closed_early: list[float] = []
        self.streams: list[FakeStream] = []
        self._lock = threading.Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        background = threading.current_thread().name.startswith("brain-refresh")
        with self._lock:
            self.requests.append(dict(kwargs, _thread=threading.current_thread().name,
                                      _t=time.monotonic(), _background=background))
        if not (background and self.hold):
            return reply(kwargs, self.text)
        if kwargs.get("stream"):
            stream = FakeStream(self.text, hold=self.release, on_close=self._closed)
            with self._lock:
                self.streams.append(stream)
            return stream
        self.release.wait(10)  # a non-streaming client: nothing can close it
        return completion(self.text)

    def _closed(self, stream):
        if not stream.exhausted:
            with self._lock:
                self.closed_early.append(time.monotonic())

    def background(self) -> list[dict]:
        with self._lock:
            return [r for r in self.requests if r["_background"]]


@pytest.fixture
def make():
    built: list[tuple[BrainDetector, _Brain]] = []

    def _make(*, brain=None, segmenter=None, **kw):
        brain = brain or _Brain()
        det = BrainDetector(base_url="http://brain/v1", model="qwen3.8-27b", client=brain,
                            segmenter=segmenter, **kw)
        built.append((det, brain))
        return det, brain

    yield _make
    for det, brain in built:
        brain.release.set()
        det.close()
    assert _refresh_threads() == [], "a refresh thread outlived close()"


# --- 7: a foreground localization on the real runtime ----------------------------


class _ContendedBrain:
    """The sim's shared brain: a refer takes REFER_S alone and CONTENTION_S
    more while a background inventory is open (one decode batch). The
    watcher's inventory decodes until `release`. Records each refer as
    (busy, start, beats at start, beats at end) and every inventory start."""

    def __init__(self):
        self.box_1000: list[int] | None = None
        self.release = threading.Event()
        self.open_inventories = 0
        self.refers: list[SimpleNamespace] = []
        self.inventory_starts: list[float] = []
        self.closed_early: list[float] = []
        self.beats = lambda: 0
        self._lock = threading.Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _answer(self, prompt: str) -> str:
        if self.box_1000 is None or prompt not in (INVENTORY_PROMPT, REFER_PROMPT.format(q="red cube")):
            return "[]"
        return json.dumps([{"label": "red cube", "bbox_2d": self.box_1000}])

    def _create(self, **kwargs):
        prompt = kwargs["messages"][0]["content"][1]["text"]
        if prompt == INVENTORY_PROMPT and threading.current_thread().name.startswith("brain-refresh"):
            with self._lock:
                self.open_inventories += 1
                self.inventory_starts.append(time.monotonic())
            if kwargs.get("stream"):
                return FakeStream(self._answer(prompt), hold=self.release, on_close=self._closed)
            try:  # a non-streaming request: it decodes until released
                self.release.wait(10)
            finally:
                with self._lock:
                    self.open_inventories -= 1
            return completion(self._answer(prompt))
        if prompt == INVENTORY_PROMPT:  # a synchronous (foreground) inventory
            return reply(kwargs, self._answer(prompt))
        with self._lock:
            busy = self.open_inventories > 0
        refer = SimpleNamespace(busy=busy, start=time.monotonic(), beats_start=self.beats())
        time.sleep(REFER_S + (CONTENTION_S if busy else 0.0))
        refer.beats_end = self.beats()
        with self._lock:
            self.refers.append(refer)
        return reply(kwargs, self._answer(prompt))

    def _closed(self, stream):
        with self._lock:
            self.open_inventories -= 1
            if not stream.exhausted:
                self.closed_early.append(time.monotonic())


@pytest.fixture
def contended(tmp_path, monkeypatch):
    """The mock rig with the REAL watcher (3 Hz scaled) and a brain detector
    on the contended fake brain; the 5 s age limit is scaled to 0.5 s."""
    pytest.importorskip("pinocchio")
    from cascade.apps import demo
    from cascade.config import load_demo_config
    from cascade.perception.detector import MockDetector
    from cascade.skills import runtime as runtime_mod

    brain = _ContendedBrain()
    detector = BrainDetector(base_url="http://brain/v1", model="qwen3.8-27b", api_key_file=None, client=brain,
                             refresh_s=4.0 * SCALE, max_cache_age_s=30.0 * SCALE, idle_refresh_s=30.0 * SCALE)
    # Every foreground() scope, as (entered, leaving, a refresh in flight right
    # after it ended); both times are taken INSIDE the scope.
    scopes: list[list] = []
    if hasattr(BrainDetector, "foreground"):
        original = detector.foreground

        @contextlib.contextmanager
        def recorded():
            with original():
                span = [time.monotonic()]
                scopes.append(span)
                yield
                span.append(time.monotonic())
            span.append(bool(detector.refresh_threads()))

        monkeypatch.setattr(detector, "foreground", recorded)
    monkeypatch.setattr(demo, "_make_detector", lambda cfg: detector)
    monkeypatch.setattr(runtime_mod, "_analyzed_frame_age_s",
                        functools.partial(runtime_mod._analyzed_frame_age_s, max_age_s=5.0 * SCALE))
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    cfg._data["perception_loop"] = {"enabled": True, "rate_hz": RATE_HZ}
    runtime, arm = demo.build_runtime(cfg, tmp_path / "run", view=False, serve=False)
    try:
        harness = runtime.watcher._harness
        beats = []
        beat = harness.heartbeat
        monkeypatch.setattr(harness, "heartbeat", lambda: (beats.append(time.monotonic()), beat())[1])
        brain.beats = lambda: len(beats)
        frame = runtime.observe()
        (cube,) = MockDetector().detect(frame)
        h, w = frame.rgb.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in cube.bbox)
        brain.box_1000 = [round(x1 * 1000 / w), round(y1 * 1000 / h), round(x2 * 1000 / w), round(y2 * 1000 / h)]
        # The watcher's first tick found no entry: its inventory is decoding.
        assert _wait_for(lambda: brain.open_inventories == 1, 2.0), "the watcher started no inventory"
        yield SimpleNamespace(runtime=runtime, arm=arm, brain=brain, detector=detector, scopes=scopes,
                              beats=beats)
    finally:
        brain.release.set()
        demo.shutdown_runtime(runtime, arm)


def test_localize_cancels_the_watcher_inventory_and_queues_it_until_the_scope_ends(contended):
    brain, detector = contended.brain, contended.detector
    result = contended.runtime.execute("localize_object", {"label": "red cube"})
    # On main the refer shared the brain with the inventory: SlowPerceptionError.
    assert result["ok"], result
    assert [r.busy for r in brain.refers] == [False], "the refer decoded beside a background inventory"
    (refer,) = brain.refers
    assert brain.closed_early and brain.closed_early[0] <= refer.start, \
        "the in-flight inventory stream was not closed before the refer"
    (scope,) = contended.scopes
    enter, leave, refreshing_after = scope
    assert not [t for t in brain.inventory_starts if enter <= t <= leave], "a refresh started inside the scope"
    # The watcher kept ticking (and beating) while the refer ran.
    assert refer.beats_end - refer.beats_start >= 3, (refer.beats_start, refer.beats_end)
    # The queued refresh starts as the scope ends (in its exit), not a tick later.
    assert refreshing_after, "the queued refresh did not start when the scope ended"
    assert _wait_for(lambda: any(t > leave for t in brain.inventory_starts), 1.0)
    assert detector._failed_at == {}, "a cancelled refresh counted as a failure"


@pytest.mark.parametrize("skill", ["grasp_object", "preview_grasp"])
def test_grasping_skills_pass_perception_while_the_watcher_has_an_inventory_in_flight(contended, skill):
    runtime, arm, brain = contended.runtime, contended.arm, contended.brain
    n0 = len(arm.commands)
    result = runtime.execute(skill, {"label": "red cube"})
    # On main: SlowPerceptionError before any motion. The mock gripper holds
    # nothing, so grasp_object itself ends as an air grasp after the motion.
    assert "SlowPerception" not in str(result.get("error")), result
    if skill == "grasp_object":
        assert len(arm.commands) > n0, f"the arm never moved: {result}"
    else:
        assert result["ok"], result
    assert brain.refers and not any(r.busy for r in brain.refers), [r.busy for r in brain.refers]
    assert brain.closed_early, "the in-flight inventory stream was never closed"
    assert contended.detector._failed_at == {}


# --- B2: the scope's failure rows ---------------------------------------------


def test_a_cancelled_refresh_is_requeued_without_backoff_or_warning(make, caplog):
    det, brain = make(brain=_Brain(hold=True))
    frame = _frame(40)
    assert det.detect_nowait(frame) == []
    assert _wait_for(lambda: len(brain.streams) == 1, 1.0)
    with caplog.at_level(logging.WARNING, logger="cascade.perception.brain_detector"):
        with det.foreground():
            assert brain.closed_early, "entering the scope did not close the stream"
            assert _refresh_threads() == []
            assert det._failed_at == {}
            for _ in range(3):  # watcher ticks inside the scope: queued, not started
                assert det.detect_nowait(frame) == []
            assert len(brain.background()) == 1
        # The scope ended: the queued refresh starts at once (no backoff).
        assert _wait_for(lambda: len(brain.background()) == 2, 1.0)
    assert not caplog.records, [r.getMessage() for r in caplog.records]
    brain.release.set()
    assert _wait_for(lambda: [d.label for d in det.detect_nowait(frame)] == ["paper cup"], 1.0)


def test_a_scope_left_by_an_exception_starts_the_queued_refresh(make):
    det, brain = make(brain=_Brain(hold=True))
    with pytest.raises(RuntimeError), det.foreground():
        assert det.detect_nowait(_frame(40)) == []
        raise RuntimeError("the skill failed")
    assert _wait_for(lambda: len(brain.background()) == 1, 1.0), "the queued refresh never started"


def test_nested_scopes_start_the_refresh_only_after_the_outermost(make):
    det, brain = make(brain=_Brain(hold=True))
    with det.foreground():
        with LockedDetector(det).foreground():
            det.detect_nowait(_frame(40))
        time.sleep(0.05)
        assert brain.background() == [], "the inner scope's exit started a refresh"
    assert _wait_for(lambda: len(brain.background()) == 1, 1.0)


def test_entering_waits_a_bounded_time_for_a_request_without_chunks(make, monkeypatch):
    """A request still waiting for its first chunk (prefill) cannot see the
    cancel: the scope waits at most _CANCEL_WAIT_S, and the request is
    dropped at its first chunk and requeued."""
    monkeypatch.setattr(brain_mod, "_CANCEL_WAIT_S", 0.1)
    gate, entered = threading.Event(), threading.Event()
    brain = _Brain(hold=True)
    create = brain._create

    def prefill(**kwargs):
        entered.set()
        gate.wait(5)
        return create(**kwargs)

    brain.chat.completions.create = prefill
    det, _ = make(brain=brain)
    det.detect_nowait(_frame(40))
    assert entered.wait(1.0), "the refresh never sent its request"
    started = time.monotonic()
    with det.foreground():
        assert time.monotonic() - started < 0.5
        gate.set()  # the first chunk arrives: the refresh sees the cancel
        assert _wait_for(lambda: _refresh_threads() == [], 1.0)
        assert len(brain.streams) == 1 and brain.streams[0].closed
        assert det._failed_at == {}
    assert _wait_for(lambda: len(brain.background()) == 2, 1.0), "the cancelled request was not requeued"


def test_a_refresh_cancelled_after_its_scope_ended_restarts_at_once(make, monkeypatch):
    """The scope ended while the request still waited for its first chunk:
    when that chunk arrives the refresh is cancelled and requeued, and with
    no scope held it starts again right away (no watcher ask needed)."""
    monkeypatch.setattr(brain_mod, "_CANCEL_WAIT_S", 0.05)
    gate, entered = threading.Event(), threading.Event()
    brain = _Brain(hold=True)
    create = brain._create

    def prefill(**kwargs):
        entered.set()
        gate.wait(5)
        return create(**kwargs)

    brain.chat.completions.create = prefill
    det, _ = make(brain=brain)
    det.detect_nowait(_frame(40))
    assert entered.wait(1.0), "the refresh never sent its request"
    with det.foreground():
        pass  # still in prefill: the request cannot see the cancel yet
    assert len(_refresh_threads()) == 1
    gate.set()
    assert _wait_for(lambda: len(brain.background()) == 2, 1.0), "the requeued refresh waited for an ask"
    assert brain.streams[0].closed and det._failed_at == {}


def test_a_refresh_cancelled_before_its_request_sends_nothing(make, monkeypatch):
    """Cancelled while still encoding its image: no request reaches the brain
    (the server would prefill it only to have it aborted at the first chunk)."""
    monkeypatch.setattr(brain_mod, "_CANCEL_WAIT_S", 0.05)
    gate, encoding = threading.Event(), threading.Event()
    encode = brain_mod._jpeg_data_url

    def slow_encode(frame):
        if threading.current_thread().name.startswith("brain-refresh"):
            encoding.set()
            gate.wait(5)
        return encode(frame)

    monkeypatch.setattr(brain_mod, "_jpeg_data_url", slow_encode)
    det, brain = make(brain=_Brain(hold=True))
    det.detect_nowait(_frame(40))
    assert encoding.wait(1.0)
    with det.foreground():
        gate.set()
        assert _wait_for(lambda: _refresh_threads() == [], 1.0)
        assert brain.requests == [], "a cancelled refresh still sent its request"
    assert _wait_for(lambda: len(brain.background()) == 1, 1.0)  # requeued, sent after the scope


def _in_prefill(brain: _Brain, gate: threading.Event, entered: threading.Event, error: Exception | None = None):
    """brain's create() waits for `gate` before it answers (a request in
    prefill: no chunk yet, so it cannot see a cancel), then answers as before
    or raises `error` (a hung brain's timeout)."""
    create = brain._create

    def prefill(**kwargs):
        entered.set()
        gate.wait(5)
        if error is not None:
            with brain._lock:
                brain.requests.append(dict(kwargs, _thread=threading.current_thread().name, _t=time.monotonic(),
                                           _background=threading.current_thread().name.startswith("brain-refresh")))
            raise error
        return create(**kwargs)

    brain.chat.completions.create = prefill


def test_a_scene_change_during_the_scope_reaches_the_restarted_refresh(make, monkeypatch):
    """The scene changed while the cancelled request was still in prefill: the
    requeued refresh asks about the NEW frame, not the one it started with
    (that answer would be one wasted inventory, 25 s in the kitchen)."""
    monkeypatch.setattr(brain_mod, "_CANCEL_WAIT_S", 0.05)
    gate, entered = threading.Event(), threading.Event()
    brain = _Brain(hold=True)
    _in_prefill(brain, gate, entered)
    det, _ = make(brain=brain)
    det.detect_nowait(_frame(40))
    assert entered.wait(1.0), "the refresh never sent its request"
    with det.foreground():
        assert len(_refresh_threads()) == 1  # still in prefill: it cannot see the cancel yet
        assert det.detect_nowait(_frame(120)) == []  # the watcher's tick: the scene changed
        gate.set()  # the first chunk arrives: the refresh sees the cancel and is requeued
        assert _wait_for(lambda: _refresh_threads() == [], 1.0)
    assert _wait_for(lambda: len(brain.background()) == 2, 1.0), "the cancelled request was not requeued"
    assert [round(_image_level(r)) for r in brain.background()] == [40, 120], "the restart sent the old scene"
    assert det._failed_at == {}


def test_a_cancelled_refresh_that_then_fails_is_no_failure_and_the_next_one_is(make, monkeypatch, caplog):
    """A request cancelled in prefill that then ends with an error (a hung
    brain's timeout) is still a cancelled refresh: requeued, no backoff, no
    warning. The hung brain is detected by the next, uncancelled refresh."""
    monkeypatch.setattr(brain_mod, "_CANCEL_WAIT_S", 0.05)
    gate, entered = threading.Event(), threading.Event()
    brain = _Brain(hold=True)
    _in_prefill(brain, gate, entered, error=TimeoutError("Request timed out."))
    det, _ = make(brain=brain)
    frame = _frame(40)
    with caplog.at_level(logging.WARNING, logger="cascade.perception.brain_detector"):
        det.detect_nowait(frame)
        assert entered.wait(1.0), "the refresh never sent its request"
        with det.foreground():
            gate.set()  # the hung request times out after the scope cancelled it
            assert _wait_for(lambda: _refresh_threads() == [], 1.0)
            assert det._failed_at == {}, "a cancelled refresh was recorded as a failure"
            assert not caplog.records, [r.getMessage() for r in caplog.records]
        # The scope ended: the requeued refresh runs uncancelled and fails.
        assert _wait_for(lambda: brain_mod._cache_key(frame, None) in det._failed_at, 1.0), \
            "the hung brain was never detected"
    assert len(brain.background()) == 2
    assert [r.getMessage() for r in caplog.records if "refresh failed" in r.getMessage()]


def _record_joins(monkeypatch, thread: threading.Thread, on_join=None, cap_s: float = 5.0) -> list:
    """Record the timeout of every join() on `thread` (a foreground entry's
    bounded wait for a cancelled stream); `on_join` runs at each call. The
    real wait is capped at `cap_s`, so a mutant that waits without a bound
    fails on the recorded timeout instead of hanging the run."""
    timeouts = []
    real = thread.join

    def join(timeout=None):
        timeouts.append(timeout)
        if on_join is not None:
            on_join()
        real(cap_s if timeout is None else min(timeout, cap_s))

    monkeypatch.setattr(thread, "join", join)
    return timeouts


def test_every_scope_entry_waits_for_a_cancelled_stream_still_open(make, monkeypatch):
    """The outer scope's bounded wait ran out while the cancelled request was
    in prefill. A nested scope (or another thread's) waits again, bounded,
    so its foreground work does not overlap that stream's last chunk.

    Event-based, no wall-clock window: the stream's first chunk is released
    when the nested entry starts waiting for its thread, and the bound is the
    timeout each entry hands to join(). A runner that stalls a thread for
    0.3 s (sys.setswitchinterval(0.12) plus an injected sleep) cannot fail it.
    """
    monkeypatch.setattr(brain_mod, "_CANCEL_WAIT_S", 0.3)
    gate, entered = threading.Event(), threading.Event()
    brain = _Brain(hold=True)
    _in_prefill(brain, gate, entered)
    det, _ = make(brain=brain)
    det.detect_nowait(_frame(40))
    assert entered.wait(5.0), "the refresh never sent its request"
    (refresh,) = det.refresh_threads()
    with det.foreground():
        assert refresh.is_alive()  # the outer wait ran out: still in prefill
        # Its first chunk arrives once the nested entry waits for it; only the
        # stream's end may end that wait (a bound far above any runner stall).
        joins = _record_joins(monkeypatch, refresh, on_join=gate.set)
        monkeypatch.setattr(brain_mod, "_CANCEL_WAIT_S", 30.0)
        with LockedDetector(det).foreground():
            assert joins, "the nested entry did not wait for the cancelled stream"
            assert not refresh.is_alive(), "the nested entry returned before the cancelled stream closed"
        assert all(t is not None and 0.0 <= t <= 30.0 for t in joins), joins
    # Bounded: a stream that never closes costs a nested entry at most _CANCEL_WAIT_S.
    monkeypatch.setattr(brain_mod, "_CANCEL_WAIT_S", 0.3)
    det_2, brain_2 = make(brain=_Brain(hold=True))
    gate_2, entered_2 = threading.Event(), threading.Event()
    _in_prefill(brain_2, gate_2, entered_2)
    det_2.detect_nowait(_frame(40))
    assert entered_2.wait(5.0)
    (refresh_2,) = det_2.refresh_threads()
    with det_2.foreground():
        joins_2 = _record_joins(monkeypatch, refresh_2)
        with det_2.foreground():
            assert refresh_2.is_alive(), "the stream is still in prefill"
        assert joins_2, "the nested entry did not wait for the stream still open"
        assert all(t is not None and 0.0 <= t <= 0.3 for t in joins_2), joins_2
    gate_2.set()


def test_a_scope_interrupted_right_after_its_count_went_up_still_ends(make):
    """An asynchronous KeyboardInterrupt that lands just after the count went
    up (here: raised by a cancel Event's set()) must not leave the scope held,
    or background work would stop for good."""
    det, brain = make()

    class _Interrupting(threading.Event):
        def set(self):
            raise KeyboardInterrupt

    det._cancels[("not", "a key")] = _Interrupting()
    with pytest.raises(KeyboardInterrupt), det.foreground():
        pass
    del det._cancels[("not", "a key")]
    assert det._foreground == 0, "the interrupted scope leaked its count"
    det.detect_nowait(_frame(40))
    assert _wait_for(lambda: len(brain.background()) == 1, 1.0), "background work stayed stopped"


def test_close_waits_up_to_inventory_timeout_for_a_request_in_prefill(make, monkeypatch):
    """A request still waiting for its first chunk cannot see the cancel:
    close() bounds its join by the BACKGROUND timeout, not timeout_s."""
    monkeypatch.setattr(brain_mod, "_CLOSE_MARGIN_S", 0.0)
    entered = threading.Event()
    brain = _Brain(hold=True)
    create = brain._create

    def prefill(**kwargs):
        entered.set()
        time.sleep(0.4)  # longer than timeout_s, shorter than inventory_timeout_s
        return create(**kwargs)

    brain.chat.completions.create = prefill
    det, _ = make(brain=brain, timeout_s=0.1, inventory_timeout_s=2.0)
    det.detect_nowait(_frame(40))
    assert entered.wait(1.0)
    det.close()
    assert _refresh_threads() == [], "close() returned while the refresh was still in prefill"
    assert brain.streams[0].closed


def test_close_cancels_the_stream_before_its_bounded_join(make):
    det, brain = make(brain=_Brain(hold=True), timeout_s=2.0, inventory_timeout_s=2.0)
    det.detect_nowait(_frame(40))
    assert _wait_for(lambda: len(brain.streams) == 1, 1.0)
    started = time.monotonic()
    det.close()
    assert time.monotonic() - started < 0.5, "close() waited for the stream instead of cancelling it"
    assert brain.streams[0].closed and _refresh_threads() == []
    assert len(brain.background()) == 1, "close() requeued or restarted the cancelled refresh"


def test_background_requests_stream_under_inventory_timeout_and_refers_do_not(make):
    det, brain = make(timeout_s=7.0, inventory_timeout_s=33.0)
    det.detect_nowait(_frame(40))
    assert _wait_for(lambda: len(brain.background()) == 1 and _refresh_threads() == [], 1.0)
    det.detect(_frame(40), ["paper cup"])
    (background,) = brain.background()
    (refer,) = [r for r in brain.requests if not r["_background"]]
    assert (background.get("stream"), background.get("timeout")) == (True, 33.0)
    assert "stream" not in refer and "timeout" not in refer  # the client's timeout_s, one completion


def test_a_background_stream_past_inventory_timeout_fails_and_backs_off(make, caplog):
    det, brain = make(brain=_Brain(hold=True), inventory_timeout_s=0.2)
    frame = _frame(40)
    with caplog.at_level(logging.WARNING, logger="cascade.perception.brain_detector"):
        det.detect_nowait(frame)
        assert _wait_for(lambda: _refresh_threads() == [], 2.0), "the stream outlived inventory_timeout_s"
    assert brain.streams[0].closed
    assert list(det._failed_at) == [brain_mod._cache_key(frame, None)]
    assert any("refresh failed" in r.getMessage() for r in caplog.records)


def test_owned_threads_lists_a_refresh_in_flight(make):
    from cascade.apps.demo import owned_threads

    det, brain = make(brain=_Brain(hold=True))
    det.detect_nowait(_frame(40))
    assert _wait_for(lambda: len(brain.streams) == 1, 1.0)
    owned = owned_threads(SimpleNamespace(detector=LockedDetector(det)))
    assert [(label, t) for label, t in owned] == [("brain-refresh", t) for t in _refresh_threads()]


# --- 8: the cache key names the capture camera ----------------------------------


def test_same_size_and_k_with_different_capture_cameras_are_two_entries(make):
    det, brain = make()
    cam0, side = _frame(50, "cam0", 1), _frame(84, "side", 1)
    for frame in (cam0, side, cam0, side):
        assert [d.label for d in det.detect(frame, None)] == ["paper cup"]
    # On main both shared one entry: every call was a changed scene, 4 requests.
    assert len(brain.requests) == 2
    plain = _frame(50)
    assert BrainDetector.camera_key(plain) == (H, W, 500, W // 2)  # no capture identity: size + K
    assert BrainDetector.camera_key(cam0) == (H, W, 500, W // 2, "('127.0.0.1', 8611)", "cam0")


class _Stream:
    """A watched camera: every latest() is a new capture of the same image."""

    def __init__(self, name: str, level: int, camera: str | None = None):
        self.name, self.level, self.camera = name, level, camera if camera is not None else name
        self.n = 0
        self._lock = threading.Lock()

    def latest(self):
        with self._lock:
            self.n += 1
            return _frame(self.level, self.camera, self.n)

    def set_overlay(self, detections=None, status=None):
        pass


def _watch(det, *cams, harness=None):
    watched = [WatchedCamera(stream, SimpleNamespace(ensure_depth=lambda f: f), None, fuse=fuse)
               for stream, fuse in cams]
    return WorldWatcher(watched, LockedDetector(det), SimpleNamespace(), rate_hz=200.0, harness=harness)


def test_a_watcher_alternating_cam0_and_side_asks_for_two_inventories(make):
    det, brain = make()
    watcher = _watch(det, (_Stream("cam0", 50), True), (_Stream("side", 84), True))
    watcher.start()
    try:
        assert _wait_for(lambda: watcher.ticks >= 40, 5.0), watcher.ticks  # 20 per camera
    finally:
        watcher.stop()
    assert _wait_for(lambda: _refresh_threads() == [], 1.0)
    # On main the two cameras shared a key: an inventory after every completion.
    assert len(brain.requests) == 2, len(brain.requests)
    assert sorted(round(_image_level(r)) for r in brain.requests) == [50, 84]


def test_build_time_check_warns_once_when_two_cameras_share_a_key(make, caplog):
    from cascade.apps.demo import _warn_shared_camera_keys

    det, _ = make()
    plain = [SimpleNamespace(name=n, latest=lambda: _frame(50), wait_newer=None) for n in ("scene", "wrist")]
    with caplog.at_level(logging.WARNING, logger="cascade.apps.demo"):
        assert _warn_shared_camera_keys(LockedDetector(det), plain) == [["scene", "wrist"]]
        named = [_Stream("cam0", 50), _Stream("side", 84)]
        assert _warn_shared_camera_keys(LockedDetector(det), named) == []
    (record,) = caplog.records
    assert "scene and wrist" in record.getMessage()


def test_build_runtime_warns_about_two_mock_cameras_sharing_a_key(tmp_path, monkeypatch, caplog):
    pytest.importorskip("pinocchio")
    from cascade.apps import demo
    from cascade.config import load_demo_config

    detector = BrainDetector(base_url="http://brain/v1", model="qwen3.8-27b", api_key_file=None, client=_Brain())
    monkeypatch.setattr(demo, "_make_detector", lambda cfg: detector)
    cfg = load_demo_config(cameras=["mock", "mock"], arm="mock", llm="mock")
    cfg._data["perception_loop"] = {"enabled": False}
    with caplog.at_level(logging.WARNING, logger="cascade.apps.demo"):
        runtime, arm = demo.build_runtime(cfg, tmp_path / "run", view=False, serve=False)
    try:
        shared = [r.getMessage() for r in caplog.records if "share one brain-detector cache key" in r.getMessage()]
        assert len(shared) == 1 and "mock0 and mock1" in shared[0], shared
    finally:
        demo.shutdown_runtime(runtime, arm)


def test_the_build_time_key_check_never_fails_on_a_bad_camera(make):
    """A diagnostic never fails build_runtime: a camera whose K is NaN, or
    whose stream raises, is left out of the comparison."""
    from cascade.apps.demo import _warn_shared_camera_keys

    det, _ = make()
    nan_k = _frame(50)
    nan_k.K[0, 0] = float("nan")

    def broken():
        raise RuntimeError("camera gone")

    streams = [SimpleNamespace(name="scene", latest=lambda: _frame(50), wait_newer=None),
               SimpleNamespace(name="nan", latest=lambda: nan_k, wait_newer=None),
               SimpleNamespace(name="broken", latest=broken, wait_newer=None),
               SimpleNamespace(name="wrist", latest=lambda: _frame(50), wait_newer=None)]
    assert _warn_shared_camera_keys(LockedDetector(det), streams) == [["scene", "wrist"]]

    class _Nameless:
        @property
        def name(self):
            raise RuntimeError("no name")

    assert _warn_shared_camera_keys(LockedDetector(det), [_Nameless(), _Nameless()]) == []


def test_the_build_time_key_check_logs_every_key_and_compares_only_served_cameras(make, caplog):
    """Every camera's key, and every camera left out, is logged at INFO. A
    camera the watcher asks nothing about (`skip`: non-fusing, not the
    primary) cannot invalidate another camera's entry, so it is not compared."""
    from cascade.apps.demo import _warn_shared_camera_keys

    det, _ = make()
    streams = [SimpleNamespace(name=n, latest=lambda: _frame(50), wait_newer=None) for n in ("scene", "proof")]
    with caplog.at_level(logging.INFO, logger="cascade.apps.demo"):
        assert _warn_shared_camera_keys(LockedDetector(det), streams, skip={"proof"}) == []
    messages = [r.getMessage() for r in caplog.records]
    assert all(r.levelno == logging.INFO for r in caplog.records), messages
    assert [m for m in messages if "scene" in m and str(BrainDetector.camera_key(_frame(50))) in m], messages
    assert [m for m in messages if "proof" in m and "left out" in m], messages
    caplog.clear()
    nan_k = _frame(50)
    nan_k.K[0, 0] = float("nan")
    streams.append(SimpleNamespace(name="nan", latest=lambda: nan_k, wait_newer=None))
    with caplog.at_level(logging.INFO, logger="cascade.apps.demo"):
        _warn_shared_camera_keys(LockedDetector(det), streams)
    assert [r for r in caplog.records if "nan" in r.getMessage() and "left out" in r.getMessage()
            and "ValueError" in r.getMessage() and r.levelno == logging.INFO]


@pytest.mark.parametrize("fuse", [(True, False), (False, True)], ids=["second-does-not-fuse", "primary-does-not-fuse"])
def test_build_runtime_compares_the_keys_of_the_cameras_the_watcher_serves(tmp_path, monkeypatch, caplog, fuse):
    """The watcher serves the primary and every fusing camera: two mock
    cameras share a key, which matters only when both are served."""
    pytest.importorskip("pinocchio")
    from cascade.apps import demo
    from cascade.config import load_demo_config

    detector = BrainDetector(base_url="http://brain/v1", model="qwen3.8-27b", api_key_file=None, client=_Brain())
    monkeypatch.setattr(demo, "_make_detector", lambda cfg: detector)
    cfg = load_demo_config(cameras=["mock", "mock"], arm="mock", llm="mock")
    cfg._data["perception_loop"] = {"enabled": False}
    cfg._data["cameras"] = [dict(c, fuse_beliefs=f) for c, f in zip(cfg._data["cameras"], fuse, strict=True)]
    with caplog.at_level(logging.INFO, logger="cascade.apps.demo"):
        runtime, arm = demo.build_runtime(cfg, tmp_path / "run", view=False, serve=False)
    try:
        shared = [r.getMessage() for r in caplog.records if "share one brain-detector cache key" in r.getMessage()]
        if fuse == (True, False):
            assert shared == []
            assert [r for r in caplog.records if "mock1" in r.getMessage() and "left out" in r.getMessage()]
        else:
            assert len(shared) == 1 and "mock0 and mock1" in shared[0], shared
    finally:
        demo.shutdown_runtime(runtime, arm)


# --- 9: no background work for a non-fusing camera ------------------------------


class _Beats:
    def __init__(self):
        self.n = 0

    def heartbeat(self):
        self.n += 1


def test_a_non_fusing_camera_gets_no_brain_request_and_no_sam_step(make):
    seg = _Segmenter()
    det, brain = make(segmenter=seg)
    beats = _Beats()
    watcher = _watch(det, (_Stream("cam0", 50), True), (_Stream("proof", 120), False), harness=beats)
    watcher.start()
    try:
        # The fusing camera is served (its inventory, then track steps) ...
        assert _wait_for(lambda: seg.calls.count("cam0") >= 5, 5.0), seg.calls
    finally:
        watcher.stop()
    assert _wait_for(lambda: _refresh_threads() == [], 1.0)
    # ... the proof camera never: no inventory of its image, no SAM on its frames.
    assert [round(_image_level(r)) for r in brain.requests] == [50]
    assert "proof" not in seg.calls
    assert watcher.last_dets["proof"] == []
    assert beats.n >= 10  # both cameras' ticks still beat


def test_a_primary_camera_that_does_not_fuse_keeps_its_background_inventory(make):
    """The primary's fuse=False (a missing or rejected hand-eye record) must
    not cost it its cache: describe_scene and localization read that entry,
    and without the watcher's inventory each of them would ask the brain
    synchronously. Other non-fusing cameras still get nothing."""
    seg = _Segmenter()
    det, brain = make(segmenter=seg)
    watcher = _watch(det, (_Stream("cam0", 50), False), (_Stream("proof", 120), False))
    watcher.start()
    try:
        assert _wait_for(lambda: seg.calls.count("cam0") >= 3, 5.0), seg.calls
    finally:
        watcher.stop()
    assert _wait_for(lambda: _refresh_threads() == [], 1.0)
    assert [round(_image_level(r)) for r in brain.requests] == [50]
    assert "proof" not in seg.calls


# --- 10: detect_nowait inside the scope: [] and no SAM -------------------------------


def test_detect_nowait_in_a_foreground_scope_returns_nothing_and_runs_no_sam(make):
    seg = _Segmenter()
    det, brain = make(segmenter=seg)
    frame = _frame(50, "cam0", 1)
    det.detect(frame, None)  # a cached inventory: outside a scope the watcher tracks it with SAM
    assert [d.label for d in det.detect_nowait(frame)] == ["paper cup"]
    calls = len(seg.calls)
    for scope in (det.foreground, LockedDetector(det).foreground):
        with scope():
            assert det.detect_nowait(frame) == []
            assert LockedDetector(det).detect_nowait(frame) == []
    assert len(seg.calls) == calls, "a track step ran inside the foreground scope"
    assert [d.label for d in det.detect_nowait(frame)] == ["paper cup"]


def test_a_detector_without_a_foreground_gets_a_no_op_scope():
    from cascade.perception.detector import MockDetector, foreground

    locked = LockedDetector(MockDetector())
    with locked.foreground(), foreground(locked), foreground(MockDetector()), foreground(object()):
        pass


# --- B5: the background timeout from the config --------------------------------


def test_inventory_timeout_comes_from_the_config(monkeypatch):
    from cascade.apps.demo import _make_detector
    from cascade.config import load_demo_config

    for name in ("CASCADE_PERCEPTION_URL", "CASCADE_PERCEPTION_MODEL", "CASCADE_PERCEPTION_KEY_FILE",
                 "CASCADE_BOOTH", "CASCADE_ROBOT"):
        monkeypatch.delenv(name, raising=False)
    default = _make_detector(load_demo_config(camera="isaac", arm="isaac"), warm_up=False)
    assert (default._timeout_s, default._inventory_timeout_s, default._idle_refresh_s) == (20.0, 20.0, 30.0)
    kitchen = _make_detector(load_demo_config(camera="isaac", arm="isaac_kitchen_gpu"), warm_up=False)
    assert (kitchen._timeout_s, kitchen._inventory_timeout_s, kitchen._idle_refresh_s) == (20.0, 45.0, 120.0)
    cfg = load_demo_config(camera="isaac", arm="isaac")
    cfg._data["detector"].pop("inventory_timeout_s", None)
    cfg._data["detector"]["timeout_s"] = 12
    assert _make_detector(cfg, warm_up=False)._inventory_timeout_s == 12.0  # absent: timeout_s


_PERCEPTION_ENV = ("CASCADE_PERCEPTION_URL", "CASCADE_PERCEPTION_MODEL", "CASCADE_PERCEPTION_KEY_FILE",
                   "CASCADE_BOOTH", "CASCADE_ROBOT", "CASCADE_BASE")


#: How long after the idle refresh is due an unchanged scene's entry must still
#: be served: the watcher registers that refresh on its next tick for the
#: camera (~0.3 s, longer behind another camera's slow tick or a foreground
#: scope), and until it is pending an entry past max_cache_age_s costs a
#: synchronous inventory on the caller's thread.
_IDLE_REFRESH_START_MARGIN_S = 5.0


def test_every_shipped_profile_keeps_its_cache_until_the_idle_refresh_is_due(monkeypatch):
    """max_cache_age_s >= idle_refresh_s + 5 s for every profile that loads a
    brain detector (demo.yaml plus each configs/arms overlay). An unchanged
    scene's entry then ages out only once the watcher's idle refresh is
    pending, so the cached path keeps tracking it (Ruling 35) instead of
    asking the brain synchronously on the caller's thread; the 5 s cover the
    watcher's tick between the refresh falling due and its start."""
    from cascade.apps.demo import _make_detector
    from cascade.config import CONFIG_DIR, load_demo_config

    for name in _PERCEPTION_ENV:
        monkeypatch.delenv(name, raising=False)
    checked, broken = [], []
    for arm in sorted(p.stem for p in (CONFIG_DIR / "arms").glob("*.yaml")):
        det = _make_detector(load_demo_config(camera="isaac", arm=arm), warm_up=False)
        if not isinstance(det, BrainDetector):
            continue
        checked.append(arm)
        if det._max_cache_age_s < det._idle_refresh_s + _IDLE_REFRESH_START_MARGIN_S:
            broken.append((arm, det._max_cache_age_s, det._idle_refresh_s))
        det.close()
    assert {"mock", "isaac", "rebot_rs", "isaac_kitchen_gpu"} <= set(checked), checked
    assert broken == [], ("max_cache_age_s < idle_refresh_s + 5 s: a synchronous inventory on an unchanged "
                          "scene before the idle refresh starts")


def test_the_kitchen_never_asks_synchronously_about_an_unchanged_scene(monkeypatch):
    """isaac_kitchen_gpu (idle_refresh_s 120): the caller's cached inventory
    (describe_scene, _reobserve, the localization vocabulary pass) of an
    unchanged scene costs no brain call on the caller's thread at any age,
    before the watcher's idle refresh (31/60/119 s) or after it (121 s).
    Fake clock; one fresh detector per age."""
    from cascade.apps.demo import _make_detector
    from cascade.config import load_demo_config

    for name in _PERCEPTION_ENV:
        monkeypatch.delenv(name, raising=False)
    cfg = load_demo_config(camera="isaac", arm="isaac_kitchen_gpu")
    rows = []
    for age in (31, 60, 119, 121):
        now = [1000.0]
        det = _make_detector(cfg, warm_up=False)
        assert isinstance(det, BrainDetector)
        brain = _Brain(hold=True)  # a background inventory decodes until released
        det._client, det._clock, det._segmenter = brain, (lambda now=now: now[0]), None
        try:
            det.detect(_frame(40, "cam0", 1), None)  # the cached inventory, taken at t=1000
            seeded = len(brain.requests)
            now[0] += age
            det.detect_nowait(_frame(40, "cam0", 2))  # the watcher's tick: same scene
            assert [d.label for d in det.detect_vocabulary(_frame(40, "cam0", 3), None)] == ["paper cup"]
            on_caller = [r for r in brain.requests[seeded:] if not r["_background"]]
            rows.append((age, len(on_caller)))
        finally:
            brain.release.set()
            det.close()
    assert rows == [(31, 0), (60, 0), (119, 0), (121, 0)], rows


def test_a_synchronous_inventory_waits_inventory_timeout_and_an_agent_refer_timeout_s(make):
    """A cache miss on the caller's thread (no entry, a changed scene, a
    vocabulary pass) asks for as many tokens as a background inventory: it
    gets `inventory_timeout_s` (timeout_s 20 s cannot fit the kitchen's 25 s
    inventory). An agent's own query stays a refer under `timeout_s`."""
    det, brain = make(timeout_s=7.0, inventory_timeout_s=33.0)
    det.detect(_frame(40), None)  # no entry
    det.detect(_frame(120), None)  # the scene changed
    det.detect_vocabulary(_frame(40, "cam0"), ["paper cup"])  # a vocabulary pass, no entry
    det.detect(_frame(40), ["paper cup"])  # the agent's own query
    on_caller = [r for r in brain.requests if not r["_background"]]
    assert [(r.get("stream"), r.get("timeout")) for r in on_caller[:3]] == [(None, 33.0)] * 3
    assert len(on_caller) == 4 and "timeout" not in on_caller[3] and "stream" not in on_caller[3]
