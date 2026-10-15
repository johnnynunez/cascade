"""BrainDetector (perception/brain_detector.py): transport, refer, inventory cache.

Everything runs against fakes: a client object with `chat.completions.create`
returning canned replies (optionally blocking on an Event), a segmenter that
records its box prompts, and an injected clock. No network, no GPU, no openai.
"""

from __future__ import annotations

import base64
import json
import logging
import sys
import threading
import types
from types import SimpleNamespace
from typing import ClassVar

import cv2
import numpy as np
import pytest

from cascade.perception.brain_detector import (
    INVENTORY_PROMPT,
    REFER_PROMPT,
    BrainDetector,
)

W, H = 320, 180  # thumbnail 64x36 is an exact 5x area reduction


class APITimeoutError(Exception):
    """Stand-in for openai.APITimeoutError (CI installs no openai)."""


def _reply(*items) -> str:
    return json.dumps([{"label": label, "bbox_2d": list(box)} for label, box in items])


def _px(box) -> list[float]:
    x1, y1, x2, y2 = box
    return [x1 * W / 1000, y1 * H / 1000, x2 * W / 1000, y2 * H / 1000]


CUP = ("paper cup", (100, 200, 300, 600))
BALL = ("red ball", (500, 100, 700, 400))
SPOOL = ("green filament spool", (600, 500, 900, 900))


class Clock:
    def __init__(self, now: float = 0.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakeClient:
    """chat.completions.create -> canned reply per prompt; inventory can block on `gate`."""

    def __init__(self):
        self.requests: list[dict] = []
        self.inventory: str | None = "[]"
        self.refer: dict[str, str] = {}
        self.error: BaseException | None = None
        self.refresh_error: BaseException | None = None
        self.gate: threading.Event | None = None
        self.finish_reason = "stop"
        self.entered = threading.Event()
        self.blocked = 0
        self._lock = threading.Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        prompt = kwargs["messages"][0]["content"][1]["text"]
        with self._lock:
            self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        if prompt == INVENTORY_PROMPT:
            gate = self.gate
            if gate is not None:
                with self._lock:
                    self.blocked += 1
                self.entered.set()
                gate.wait(10)
                with self._lock:
                    self.blocked -= 1
                if self.refresh_error is not None:
                    raise self.refresh_error
            reply = self.inventory
        else:
            reply = next((r for q, r in self.refer.items() if prompt == REFER_PROMPT.format(q=q)), "[]")
        msg = SimpleNamespace(content=reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=self.finish_reason)])

    def prompts(self) -> list[str]:
        with self._lock:
            return [r["messages"][0]["content"][1]["text"] for r in self.requests]

    def inventory_calls(self) -> int:
        return sum(p == INVENTORY_PROMPT for p in self.prompts())


class FakeSegmenter:
    """Records (frame, boxes) per call; one box-shaped mask per box, None for `drop`."""

    def __init__(self, drop=()):
        self.calls: list[tuple[object, np.ndarray]] = []
        self.returned: list[list] = []
        self.drop = set(drop)
        self.error: BaseException | None = None

    def masks(self, frame, boxes):
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        self.calls.append((frame, boxes.copy()))
        if self.error is not None:
            raise self.error
        out = []
        for i, (x0, y0, x1, y1) in enumerate(boxes.astype(int)):
            if i in self.drop:
                out.append(None)
                continue
            mask = np.zeros(frame.rgb.shape[:2], dtype=bool)
            mask[y0:y1, x0:x1] = True
            out.append(mask)
        self.returned.append(out)
        return out


def _frame(level: int = 0, fx: float = 500.0):
    from cascade.types import Frame

    rgb = np.full((H, W, 3), level, dtype=np.uint8)
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]])
    return Frame(rgb=rgb, depth_m=None, K=K)


def _brain_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith("brain-refresh") and t.is_alive()]


@pytest.fixture
def make():
    """Build detectors with fakes; teardown releases every gate and closes them."""
    built: list[tuple[BrainDetector, FakeClient]] = []

    def _make(*, segmenter="fake", client=None, clock=None, **kw):
        client = client or FakeClient()
        seg = FakeSegmenter() if segmenter == "fake" else segmenter
        det = BrainDetector(base_url="http://brain/v1", model="qwen3.8-27b", client=client,
                            segmenter=seg, clock=clock or Clock(), **kw)
        built.append((det, client))
        return det, client, seg

    yield _make
    for det, client in built:
        if client.gate is not None:
            client.gate.set()
        det.close()
    assert _brain_threads() == [], "a refresh thread outlived close()"


def test_concurrent_safe_flag():
    # world.LockedDetector skips its global RLock for detectors with this flag.
    assert BrainDetector.concurrent_safe is True


# --- refer -----------------------------------------------------------------


def test_refer_sends_the_refer_prompt_with_the_jpeg(make):
    det, client, seg = make()
    client.refer["paper cup"] = _reply(("cup", CUP[1]))
    frame = _frame()
    frame.rgb[60:120, 100:200] = (0, 0, 255)  # red in BGR

    dets = det.detect(frame, ["paper cup"])

    (req,) = client.requests
    assert req["model"] == "qwen3.8-27b"
    assert req["temperature"] == 0
    assert req["max_tokens"] == 1500
    assert req["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    (msg,) = req["messages"]
    assert msg["role"] == "user"
    image, text = msg["content"]
    assert text == {"type": "text", "text": REFER_PROMPT.format(q="paper cup")}
    assert image["type"] == "image_url"
    url = image["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8),
                           cv2.IMREAD_COLOR)
    assert decoded.shape == (H, W, 3)
    b, g, r = (int(v) for v in decoded[90, 150])
    assert r > 200 and g < 60 and b < 60, "colours swapped: the frame is BGR already"

    (d,) = dets
    assert d.label == "paper cup"  # the agent's words, not the brain's label
    assert d.conf == 1.0
    assert d.bbox.dtype == np.float32
    np.testing.assert_allclose(d.bbox, _px(CUP[1]), rtol=1e-6)
    (seg_frame, boxes), = seg.calls
    assert seg_frame is frame
    np.testing.assert_allclose(boxes, [_px(CUP[1])], rtol=1e-6)
    assert d.mask is seg.returned[0][0]


def test_max_tokens_comes_from_the_constructor(make):
    det, client, _ = make(max_tokens=777)
    det.detect(_frame(), ["paper cup"])
    assert client.requests[0]["max_tokens"] == 777


def test_refer_several_classes_one_request_each(make):
    det, client, _ = make()
    client.refer["paper cup"] = _reply(CUP)
    client.refer["red ball"] = _reply(("ball", BALL[1]), ("ball", SPOOL[1]))
    dets = det.detect(_frame(), ["paper cup", "red ball"])
    assert client.prompts() == [REFER_PROMPT.format(q="paper cup"), REFER_PROMPT.format(q="red ball")]
    assert [d.label for d in dets] == ["paper cup", "red ball", "red ball"]


def test_refer_keeps_maskless_boxes(make):
    det, client, _ = make(segmenter=FakeSegmenter(drop={0}))
    client.refer["paper cup"] = _reply(CUP)
    (d,) = det.detect(_frame(), ["paper cup"])
    assert d.mask is None  # today's bbox fallback


def test_refer_bypasses_the_inventory_cache(make):
    det, client, _ = make()
    client.refer["paper cup"] = _reply(CUP)
    client.inventory = _reply(BALL)
    det.detect(_frame(), ["paper cup"])
    # A refer neither reads nor fills the cache: the first inventory is cold.
    assert [d.label for d in det.detect(_frame(), None)] == ["red ball"]
    assert client.inventory_calls() == 1
    det.detect(_frame(), ["paper cup"])
    assert client.prompts()[-1] == REFER_PROMPT.format(q="paper cup")
    det.detect(_frame(), None)
    assert client.inventory_calls() == 1


# --- inventory cache -------------------------------------------------------


def test_inventory_cold_calls_the_brain_once_synchronously(make):
    det, client, seg = make()
    client.inventory = _reply(CUP, BALL)
    frame = _frame()
    dets = det.detect(frame, None)
    assert client.prompts() == [INVENTORY_PROMPT]
    assert [d.label for d in dets] == ["paper cup", "red ball"]
    assert all(d.conf == 1.0 for d in dets)
    assert [d.mask is m for d, m in zip(dets, seg.returned[0])] == [True, True]
    assert _brain_threads() == []


def test_cache_hit_segments_the_cached_boxes_on_the_new_frame(make):
    clock = Clock()
    det, client, seg = make(clock=clock)
    client.inventory = _reply(CUP, BALL)
    det.detect(_frame(), None)
    clock.now = 3.9
    frame = _frame()
    dets = det.detect(frame, None)
    assert client.inventory_calls() == 1
    seg_frame, boxes = seg.calls[-1]
    assert seg_frame is frame
    np.testing.assert_allclose(boxes, [_px(CUP[1]), _px(BALL[1])], rtol=1e-6)
    assert [d.label for d in dets] == ["paper cup", "red ball"]
    assert _brain_threads() == []


def test_empty_classes_behave_as_none(make):
    det, client, _ = make()
    client.inventory = _reply(CUP)
    assert [d.label for d in det.detect(_frame(), [])] == ["paper cup"]
    assert client.prompts() == [INVENTORY_PROMPT]
    det.set_classes(["red ball"])  # holds no vocabulary: None stays open world
    assert [d.label for d in det.detect(_frame(), None)] == ["paper cup"]
    assert client.prompts() == [INVENTORY_PROMPT]


def test_scene_change_threshold(make):
    det, client, _ = make(refresh_s=100.0)
    client.inventory = _reply(CUP)
    det.detect(_frame(level=0), None)
    det.detect(_frame(level=6), None)  # mean abs diff 6.0: same scene
    assert client.inventory_calls() == 1
    client.inventory = _reply(BALL)
    dets = det.detect(_frame(level=7), None)  # 7.0 > 6.0: synchronous brain call
    assert client.inventory_calls() == 2
    assert [d.label for d in dets] == ["red ball"]
    assert _brain_threads() == []


def test_entry_older_than_max_age_is_refetched_synchronously(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    det.detect(_frame(), None)
    clock.now = 30.5
    client.inventory = _reply(BALL)
    assert [d.label for d in det.detect(_frame(), None)] == ["red ball"]
    assert client.inventory_calls() == 2
    assert _brain_threads() == []


def test_aged_entry_ten_calls_start_one_background_refresh(make):
    clock = Clock()
    det, client, seg = make(clock=clock)
    client.inventory = _reply(CUP, BALL)
    det.detect(_frame(), None)
    client.gate = threading.Event()
    clock.now = 5.0
    for i in range(10):
        dets = det.detect(_frame(), None)
        assert [d.label for d in dets] == ["paper cup", "red ball"], i
        np.testing.assert_allclose(seg.calls[-1][1], [_px(CUP[1]), _px(BALL[1])], rtol=1e-6)
    assert client.entered.wait(5), "no background refresh started"
    assert client.inventory_calls() == 2
    assert client.blocked == 1 and len(_brain_threads()) == 1
    client.gate.set()
    det.close()
    assert client.inventory_calls() == 2


def test_refresh_replaces_the_entry_on_success(make):
    clock = Clock()
    det, client, seg = make(clock=clock)
    client.inventory = _reply(CUP)
    det.detect(_frame(), None)
    client.inventory = _reply(BALL)
    clock.now = 5.0
    assert [d.label for d in det.detect(_frame(), None)] == ["paper cup"]
    det.close()  # joins the refresh
    clock.now = 6.0  # the new entry dates from 5.0: fresh
    assert [d.label for d in det.detect(_frame(), None)] == ["red ball"]
    np.testing.assert_allclose(seg.calls[-1][1], [_px(BALL[1])], rtol=1e-6)
    assert client.inventory_calls() == 2


def test_failed_refresh_keeps_the_old_entry(make, caplog):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    det.detect(_frame(), None)
    client.gate = threading.Event()
    client.gate.set()
    client.refresh_error = APITimeoutError("brain timed out")
    clock.now = 5.0
    with caplog.at_level(logging.WARNING, logger="cascade.perception.brain_detector"):
        det.detect(_frame(), None)
        det.close()
    assert client.inventory_calls() == 2
    assert len(caplog.records) == 1
    clock.now = 6.0
    assert [d.label for d in det.detect(_frame(), None)] == ["paper cup"]
    clock.now = 30.5  # it still ages out
    client.refresh_error = None
    client.inventory = _reply(BALL)
    assert [d.label for d in det.detect(_frame(), None)] == ["red ball"]


def test_slow_refresh_never_overwrites_a_newer_entry(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    det.detect(_frame(level=0), None)
    gate = client.gate = threading.Event()
    clock.now = 5.0
    det.detect(_frame(level=0), None)  # refresh of the t=5 frame hangs
    assert client.entered.wait(5)
    client.gate = None
    client.inventory = _reply(BALL)
    clock.now = 6.0
    assert [d.label for d in det.detect(_frame(level=20), None)] == ["red ball"]  # scene changed
    client.inventory = _reply(SPOOL)
    gate.set()  # the t=5 refresh now answers, older than the t=6 entry
    det.close()
    clock.now = 7.0
    assert [d.label for d in det.detect(_frame(level=20), None)] == ["red ball"]
    assert client.inventory_calls() == 3


def test_close_stops_later_refreshes(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    det.detect(_frame(), None)
    det.close()
    clock.now = 5.0
    assert [d.label for d in det.detect(_frame(), None)] == ["paper cup"]
    assert client.inventory_calls() == 1
    assert _brain_threads() == []


def _join_refreshes(timeout_s: float = 5.0) -> None:
    """Wait for the refresh threads WITHOUT close(): they must clear their own bookkeeping."""
    for thread in _brain_threads():
        thread.join(timeout_s)
    assert _brain_threads() == []


def test_a_finished_refresh_lets_a_later_aged_call_start_a_new_one(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    det.detect(_frame(), None)
    clock.now = 5.0
    det.detect(_frame(), None)  # aged: refresh #1, entry now dates from 5.0
    _join_refreshes()
    assert client.inventory_calls() == 2
    client.inventory = _reply(BALL)
    clock.now = 10.0  # aged again
    assert [d.label for d in det.detect(_frame(), None)] == ["paper cup"]
    _join_refreshes()
    assert client.inventory_calls() == 3, "the finished refresh left its camera marked in flight"
    clock.now = 11.0
    assert [d.label for d in det.detect(_frame(), None)] == ["red ball"]


def test_refer_returns_while_a_refresh_is_in_flight(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    client.refer["red ball"] = _reply(BALL)
    det.detect(_frame(), None)
    client.gate = threading.Event()
    clock.now = 5.0
    det.detect(_frame(), None)
    assert client.entered.wait(5)
    assert [d.label for d in det.detect(_frame(), ["red ball"])] == ["red ball"]
    assert client.blocked == 1, "the refer waited for the background inventory"


def test_cache_lock_is_not_held_across_a_brain_call(make):
    det, client, _ = make()
    client.inventory = _reply(SPOOL)
    det.detect(_frame(fx=600.0), None)  # wrist camera primed
    client.gate = threading.Event()
    cold = threading.Thread(target=det.detect, args=(_frame(fx=500.0), None))
    cold.start()
    try:
        assert client.entered.wait(5)
        dets = det.detect(_frame(fx=600.0), None)  # track step while the scene call hangs
        assert [d.label for d in dets] == ["green filament spool"]
        assert client.blocked == 1
    finally:
        client.gate.set()
        cold.join(5)


def test_cameras_with_different_intrinsics_keep_separate_entries(make):
    det, client, _ = make()
    client.inventory = _reply(CUP)
    det.detect(_frame(fx=500.0), None)
    client.inventory = _reply(BALL)
    det.detect(_frame(fx=600.0), None)
    assert client.inventory_calls() == 2
    assert [d.label for d in det.detect(_frame(fx=500.0), None)] == ["paper cup"]
    assert [d.label for d in det.detect(_frame(fx=600.0), None)] == ["red ball"]
    assert client.inventory_calls() == 2


# --- masks -----------------------------------------------------------------


def test_track_step_drops_boxes_without_a_mask(make):
    clock = Clock()
    seg = FakeSegmenter(drop={1})
    det, client, _ = make(segmenter=seg, clock=clock)
    client.inventory = _reply(CUP, BALL)
    cold = det.detect(_frame(), None)
    assert [(d.label, d.mask is None) for d in cold] == [("paper cup", False), ("red ball", True)]
    clock.now = 1.0
    assert [d.label for d in det.detect(_frame(), None)] == ["paper cup"]


def test_without_a_segmenter_boxes_keep_no_mask(make):
    clock = Clock()
    det, client, _ = make(segmenter=None, clock=clock)
    client.inventory = _reply(CUP, BALL)
    det.detect(_frame(), None)
    clock.now = 1.0
    dets = det.detect(_frame(), None)
    assert [(d.label, d.mask) for d in dets] == [("paper cup", None), ("red ball", None)]


def test_segmenter_failure_keeps_boxes_without_masks(make, caplog):
    clock = Clock()
    seg = FakeSegmenter()
    seg.error = RuntimeError("SAM weights missing")
    det, client, _ = make(segmenter=seg, clock=clock)
    client.inventory = _reply(CUP)
    with caplog.at_level(logging.WARNING, logger="cascade.perception.brain_detector"):
        cold = det.detect(_frame(), None)
        clock.now = 1.0
        track = det.detect(_frame(), None)
    assert [(d.label, d.mask) for d in cold] == [("paper cup", None)]
    assert [(d.label, d.mask) for d in track] == [("paper cup", None)]
    assert len(caplog.records) == 1


# --- failures --------------------------------------------------------------


def test_brain_failure_is_empty_and_warns_at_most_every_30s(make, caplog):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.error = APITimeoutError("Request timed out.")
    with caplog.at_level(logging.WARNING, logger="cascade.perception.brain_detector"):
        for t in (0.0, 10.0, 29.9):
            clock.now = t
            assert det.detect(_frame(), None) == []
        assert len(caplog.records) == 1
        clock.now = 30.0  # the other camera: this one backs off since its 29.9 failure
        assert det.detect(_frame(fx=600.0), None) == []
        assert len(caplog.records) == 2
        client.error = RuntimeError("connection refused")
        clock.now = 31.0
        assert det.detect(_frame(), ["paper cup"]) == []
        assert len(caplog.records) == 2
    assert all(r.levelno == logging.WARNING for r in caplog.records)


def test_garbled_or_missing_reply_is_empty_with_a_warning(make, caplog):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = "I cannot see a table here."
    with caplog.at_level(logging.WARNING, logger="cascade.perception.brain_detector"):
        assert det.detect(_frame(), None) == []
        assert len(caplog.records) == 1
        clock.now = 30.0
        client.refer["paper cup"] = None  # message.content may be None
        assert det.detect(_frame(), ["paper cup"]) == []
        assert len(caplog.records) == 2
        assert all("garbled reply" in r.getMessage() for r in caplog.records)
        # A garbled reply is a failed call: nothing was cached.
        client.inventory = "[]"
        clock.now = 60.0
        assert det.detect(_frame(), None) == []
        assert client.inventory_calls() == 2
        # "[]" is a real answer (an empty table), cached without a warning.
        clock.now = 61.0
        assert det.detect(_frame(), None) == []
        assert client.inventory_calls() == 2
        assert len(caplog.records) == 2


def test_truncated_reply_is_garbled_and_not_cached(make, caplog):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.finish_reason = "length"  # cut at max_tokens
    client.inventory = _reply(CUP, BALL)[:-30]
    with caplog.at_level(logging.WARNING, logger="cascade.perception.brain_detector"):
        assert det.detect(_frame(), None) == []
        # A list that closed before the cut is still a cut answer: the brain had more to list.
        client.inventory = _reply(CUP) + "\nAlso a red ball at [5"
        assert det.detect(_frame(fx=600.0), None) == []
    (record,) = caplog.records  # one per 30 s
    assert "truncated" in record.getMessage()
    # Nothing was cached: once the backoff is over both cameras ask again.
    client.finish_reason = "stop"
    client.inventory = _reply(CUP, BALL)
    clock.now = 4.5
    assert [d.label for d in det.detect(_frame(), None)] == ["paper cup", "red ball"]
    assert [d.label for d in det.detect(_frame(fx=600.0), None)] == ["paper cup", "red ball"]
    assert client.inventory_calls() == 4


def test_prose_reply_is_garbled_and_not_cached(make, caplog):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = 'I can see [a paper cup] next to a ball.'
    with caplog.at_level(logging.WARNING, logger="cascade.perception.brain_detector"):
        assert det.detect(_frame(), None) == []
        client.inventory = '["paper cup", "red ball"]'  # names, no boxes
        assert det.detect(_frame(fx=600.0), None) == []
    (record,) = caplog.records
    assert "garbled reply" in record.getMessage()
    client.inventory = _reply(CUP)
    clock.now = 4.5
    assert [d.label for d in det.detect(_frame(), None)] == ["paper cup"]
    assert [d.label for d in det.detect(_frame(fx=600.0), None)] == ["paper cup"]
    assert client.inventory_calls() == 4


def test_bare_empty_list_is_a_cached_empty_table(make, caplog):
    clock = Clock()
    det, client, _ = make(clock=clock)
    replies = {500.0: "[]", 600.0: "```json\n[ ]\n```", 700.0: "<think>nothing there</think>\n[]"}
    with caplog.at_level(logging.WARNING, logger="cascade.perception.brain_detector"):
        for fx, reply in replies.items():
            client.inventory = reply
            assert det.detect(_frame(fx=fx), None) == []
        client.inventory = _reply(CUP)
        clock.now = 1.0
        for fx in replies:
            assert det.detect(_frame(fx=fx), None) == []  # the cached empty table, no request
    assert client.inventory_calls() == 3
    assert caplog.records == []


# --- brain outage: keep tracking what is known, back off -------------------


def test_outage_keeps_tracking_known_objects_past_max_age_with_backoff(make, caplog):
    clock = Clock()
    det, client, seg = make(clock=clock)
    client.inventory = _reply(CUP, BALL)
    det.detect(_frame(), None)
    client.error = APITimeoutError("Request timed out.")
    with caplog.at_level(logging.WARNING, logger="cascade.perception.brain_detector"):
        for t in (31.0, 32.0, 34.9):  # the entry is older than max_cache_age_s; same scene
            clock.now = t
            frame = _frame()
            assert [d.label for d in det.detect(frame, None)] == ["paper cup", "red ball"], t
            seg_frame, boxes = seg.calls[-1]
            assert seg_frame is frame  # a track step on THIS frame
            np.testing.assert_allclose(boxes, [_px(CUP[1]), _px(BALL[1])], rtol=1e-6)
        assert client.inventory_calls() == 2, "no backoff: every call asked the down brain"
        clock.now = 35.0  # refresh_s after the failure: ask again
        assert [d.label for d in det.detect(_frame(), None)] == ["paper cup", "red ball"]
        assert client.inventory_calls() == 3
    assert len(caplog.records) == 1
    assert _brain_threads() == []


def test_outage_on_a_changed_scene_is_empty_and_backs_off(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    det.detect(_frame(level=0), None)
    client.error = RuntimeError("connection refused")
    clock.now = 1.0
    # Old boxes on a changed table would put old labels on new things.
    assert det.detect(_frame(level=20), None) == []
    clock.now = 2.0
    assert det.detect(_frame(level=20), None) == []
    clock.now = 3.0  # back on the cached scene: an ordinary track step
    assert [d.label for d in det.detect(_frame(level=0), None)] == ["paper cup"]
    clock.now = 4.9
    assert det.detect(_frame(level=20), None) == []
    assert client.inventory_calls() == 2
    clock.now = 5.0
    assert det.detect(_frame(level=20), None) == []
    assert client.inventory_calls() == 3


def test_outage_backoff_is_per_camera(make):
    det, client, _ = make()
    client.error = RuntimeError("connection refused")
    assert det.detect(_frame(fx=500.0), None) == []
    assert det.detect(_frame(fx=500.0), None) == []
    assert client.inventory_calls() == 1
    assert det.detect(_frame(fx=600.0), None) == []  # the wrist camera asks for itself
    assert client.inventory_calls() == 2


def test_after_the_backoff_a_successful_call_replaces_the_entry(make):
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    det.detect(_frame(), None)
    client.error = RuntimeError("connection refused")
    clock.now = 31.0
    assert [d.label for d in det.detect(_frame(), None)] == ["paper cup"]
    client.error = None  # the brain is back
    client.inventory = _reply(BALL)
    clock.now = 33.0  # still backing off
    assert [d.label for d in det.detect(_frame(), None)] == ["paper cup"]
    assert client.inventory_calls() == 2
    clock.now = 35.0
    assert [d.label for d in det.detect(_frame(), None)] == ["red ball"]
    assert client.inventory_calls() == 3
    clock.now = 36.0  # the new entry is fresh: a track step, no request
    assert [d.label for d in det.detect(_frame(), None)] == ["red ball"]
    assert client.inventory_calls() == 3


def test_backoff_counts_from_the_failure_not_the_request(make):
    # A hung brain times out after timeout_s: counted from the request, the
    # backoff would be over before the call returned and every tick would block.
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.error = APITimeoutError("Request timed out.")
    create = client.chat.completions.create

    def slow_create(**kwargs):
        clock.now += 20.0  # timeout_s passes inside the call
        return create(**kwargs)

    client.chat.completions.create = slow_create
    assert det.detect(_frame(), None) == []  # asked at 0, failed at 20
    clock.now = 22.0
    assert det.detect(_frame(), None) == []
    assert client.inventory_calls() == 1


def test_a_success_clears_the_backoff(make):
    # Two synchronous calls for one camera overlap: the later one fails, then
    # the earlier one answers. The brain is up, so a scene change asks at once.
    clock = Clock()
    det, client, _ = make(clock=clock)
    client.inventory = _reply(CUP)
    client.gate = threading.Event()
    slow = threading.Thread(target=det.detect, args=(_frame(level=0), None))
    slow.start()
    try:
        assert client.entered.wait(5)
        client.error = RuntimeError("connection reset")
        clock.now = 1.0
        assert det.detect(_frame(level=0), None) == []  # fails: backoff from t=1
        client.error = None
    finally:
        client.gate.set()
        slow.join(5)
    client.gate = None
    client.inventory = _reply(BALL)
    clock.now = 2.0
    assert [d.label for d in det.detect(_frame(level=40), None)] == ["red ball"]
    assert client.inventory_calls() == 3


def test_unparseable_reply_never_raises(make):
    det, client, _ = make()
    client.refer["paper cup"] = '[{"label": "cup", "bbox_2d": [1' + "0" * 309 + ', 0, 500, 500]}]'
    client.inventory = "[" * 100000 + "]" * 100000
    det.detect(_frame(), ["paper cup"])
    assert det.detect(_frame(), None) == []


# --- transport and key hygiene ---------------------------------------------


class _StubOpenAI:
    built: ClassVar[list[dict]] = []
    client: ClassVar[FakeClient | None] = None

    def __new__(cls, **kwargs):
        cls.built.append(kwargs)
        return cls.client


@pytest.fixture
def stub_openai(monkeypatch):
    monkeypatch.delitem(sys.modules, "openai", raising=False)
    mod = types.ModuleType("openai")
    mod.OpenAI = _StubOpenAI
    _StubOpenAI.built = []
    _StubOpenAI.client = FakeClient()

    def install():
        monkeypatch.setitem(sys.modules, "openai", mod)
        return _StubOpenAI

    return install


def test_key_from_file_never_logged_nor_in_repr(tmp_path, stub_openai, caplog):
    key = "sk-brain-5ecret-77"
    (tmp_path / "k").write_text(key + "\n")
    with caplog.at_level(logging.DEBUG, logger="cascade.perception.brain_detector"):
        det = BrainDetector(base_url="http://brain/v1", model="qwen3.8-27b",
                            api_key_file=str(tmp_path / "k"), timeout_s=7.5, clock=Clock())
        assert "openai" not in sys.modules, "openai imported before the first brain call"
        stub = stub_openai()
        stub.client.error = RuntimeError(f"401 Unauthorized: bad key {key}")
        assert det.detect(_frame(), None) == []
        assert det.detect(_frame(), ["paper cup"]) == []
    det.close()
    # max_retries=0: the next watcher tick is the retry; openai's default 2
    # would block a synchronous detect for ~3x timeout_s on a hung brain.
    assert stub.built == [{"base_url": "http://brain/v1", "api_key": key, "timeout": 7.5, "max_retries": 0}]
    assert caplog.records, "the failure was not logged"
    for record in caplog.records:
        assert key not in record.getMessage()
    assert key not in repr(det) and key not in str(det)
    assert "qwen3.8-27b" in repr(det)


def test_missing_key_file_sends_EMPTY_with_one_warning(tmp_path, stub_openai, caplog):
    with caplog.at_level(logging.WARNING, logger="cascade.perception.brain_detector"):
        det = BrainDetector(base_url="http://brain/v1", model="m",
                            api_key_file=str(tmp_path / "missing"), clock=Clock())
    assert len(caplog.records) == 1
    stub = stub_openai()
    stub.client.inventory = _reply(CUP)
    assert [d.label for d in det.detect(_frame(), None)] == ["paper cup"]
    det.close()
    assert [b["api_key"] for b in stub.built] == ["EMPTY"]


def test_injected_client_never_imports_openai(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)  # any import would raise
    client = FakeClient()
    client.inventory = _reply(CUP)
    det = BrainDetector(base_url="http://brain/v1", model="m", client=client, clock=Clock())
    assert [d.label for d in det.detect(_frame(), None)] == ["paper cup"]
    det.close()
