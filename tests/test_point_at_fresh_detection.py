"""point_at after a long plan needs a detection newer than the plan (#406 post-merge review).

1. Important: when more than `pointing.FRESH_PLAN_S` passed between the
   localization and the end of planning, point_at looks at the target again
   before the first command. When the detector missed it there, `_localize`
   fell back to the world model's belief, which the watcher cannot refresh
   while the skill runs (fusion is paused) and whose age is measured against
   the skill's start: the look again "found" the pre-skill belief, unmoved,
   and main moved and closed its jaws at the old pose (the review's probe: ok,
   136 commands). The look again now accepts only a detection on an image
   newer than the plan: a miss (or an image from before the plan) refuses with
   no command.
2. Minor 1: the freshness clock starts at the localization, before
   `Budget()` (whose core class search can take ~1.2 s), and a re-plan's at
   the look again that gave its fix. (The search itself:
   tests/test_point_at_core_speed_search.py.)
3. Minor 3: a halt noticed at the first command (or later) keeps the
   operator's reason; a halt since the call started refuses before the first
   command even outside execute().

#411 review (the look again's image, tested at the camera's DEVICE, which the
stream's pump grabs from):
4. Minor 4: a driver that stalls and keeps handing back its last buffer (the
   synthetic `Frame.t` rewrite this file had is replaced by that stall).
5. Minor 2: the image must be RECEIVED after the look again started, not only
   after the plan started: a camera that stalled 0.5 s into a 3.5 s plan gave
   the look again an image received during the plan, and main pointed.
6. Minor 1: `Frame.t` is the client's receipt time. An Isaac-style bridge
   builds a new Frame for every request, so a re-sent capture from before the
   plan arrived with a new `t` and main pointed (136 commands and a jaw
   close). When the frames carry a producer capture, the look again's must be
   newer than the one the plan was made from (`freshness.newer_capture`); one
   that cannot be compared with it (another source, none any more) refuses.

#413 review (the same device doubles):
7. Minor 2: a sim that stalls mid-plan while its bridge keeps re-sending that
   render hands the look again a capture newer than the plan's, in a new
   Frame: main pointed on an image 3 s old. The capture's FIRST receipt
   (`Frame.observed_t`, `freshness.observed_at`) must be at most
   `pointing.FRESH_OBSERVED_S` before the look started; a sim that renders
   slower than its bridge serves (each render re-sent for ~0.1 s) still points.
8. Minor 3: an image stamped (`Frame.t`) just before the look started and
   published after it (the pump's own work after the stamp) was refused on
   main; it counts now, within `pointing.FRESH_IN_FLIGHT_S`. Older ones (a
   driver that stalled 0.3 s before the look) are still refused.
9. Minor 1: a look again that finds the target on ANOTHER camera than the
   plan's image skips the capture comparison (two cameras' captures cannot be
   compared: main refused a fresh side-camera render) and is judged by that
   camera's own receipts: a fresh render points, a re-sent stalled one does
   not.
"""

from __future__ import annotations

import functools
import math
import re
import threading
import time
from dataclasses import replace

import pytest
from conftest import needs_pin
from test_point_at_planning_followups import (  # noqa: F401  (rig: the fixture, by name)
    _commands,
    _localizations,
    _long_plans,
    rig,
)

# ── 1. the look again needs a detection newer than the plan ────────────────


def _detector_misses_after_the_first_look(monkeypatch):
    """The real detector pass (`runtime.localize_object`) finds the target on
    its first call and misses on every later one: `_localize` itself decides
    what a miss gives (its retries, the other cameras, the belief fallback)."""
    from cascade.skills import runtime as rmod
    from cascade.types import SkillError

    real = rmod.localize_object
    calls = []

    def localize_object(*a, **k):
        calls.append(1)
        if len(calls) > 1:
            raise SkillError("No current visual match for 'red cube' (the detector missed it)")
        return real(*a, **k)

    monkeypatch.setattr(rmod, "localize_object", localize_object)
    return calls


def _real_long_plan(monkeypatch, seconds=3.5):
    """Planning really takes `seconds` of wall time (the first budget check
    sleeps), with the shipped 3 s freshness rule."""
    from cascade.skills import pointing

    monkeypatch.setattr(pointing, "FRESH_PLAN_S", 3.0)
    real = pointing.Budget.check
    slept = set()

    def check(self):
        if id(self) not in slept:
            slept.add(id(self))
            time.sleep(seconds)
        return real(self)

    monkeypatch.setattr(pointing.Budget, "check", check)


@needs_pin
@pytest.mark.parametrize("plan", ["fake_4s", "real_3_5s"])
def test_a_long_plan_never_moves_to_the_old_pose_when_the_target_is_not_seen_again(rig, monkeypatch, plan):  # noqa: F811
    """The review's stale-pose rows: the detector misses on the look again.
    main fell back to the pre-skill belief, read it as unmoved and pointed
    (and closed the jaws) where the target was before the plan."""
    r = rig("mock")
    rt = r.runtime
    if plan == "fake_4s":
        _long_plans(monkeypatch)
    else:
        _real_long_plan(monkeypatch)
    detections = _detector_misses_after_the_first_look(monkeypatch)

    out = rt.execute("point_at", {"label": "red cube"})

    assert len(detections) > 1, "test setup: the target must be looked at again"
    assert out["ok"] is False and _commands(r) == [] and r.jaws == [], out
    assert "it was not seen again after a long plan" in out["error"], out
    assert "No current visual match for 'red cube' (the detector missed it)" in out["error"], out
    assert out["error"].endswith("nothing was moved"), out
    assert not [e for e in rt.memory.events(("note",)) if "world-model fix" in e.text], \
        "the look again must never answer from the belief"


@needs_pin
def test_a_long_real_plan_with_the_target_seen_again_points(rig, monkeypatch):  # noqa: F811
    """Control: the same real 3.5 s plan, the detector sees it again (a
    detection on an image newer than the plan): it points as planned."""
    from cascade.skills import runtime as rmod

    r = rig("mock")
    rt = r.runtime
    _real_long_plan(monkeypatch)
    real = rmod.localize_object
    detections = []
    monkeypatch.setattr(rmod, "localize_object", lambda *a, **k: (detections.append(1), real(*a, **k))[1])

    out = rt.execute("point_at", {"label": "red cube"})

    assert len(detections) == 2, "test setup: one look again"
    assert out["ok"] is True and _commands(r) and r.jaws, out


@needs_pin
@pytest.mark.parametrize("stall", ["at_the_localization", "0.5_s_into_the_plan"])
def test_a_look_again_on_a_stalled_camera_is_refused(rig, monkeypatch, stall):  # noqa: F811
    """The camera's driver stalls and keeps handing back its last buffer (the
    same Frame; the stream delivers it again and again). At the localization:
    the look again gets an image from before the plan (refused on main too;
    this replaces the synthetic `Frame.t` rewrite, #411 review Minor 4). 0.5 s
    into a real 3.5 s plan: an image received during the plan, newer than the
    plan's start; main pointed on it, 3 s old (#411 review Minor 2)."""
    r = rig("mock")
    rt = r.runtime
    _real_long_plan(monkeypatch)
    device = _Device(monkeypatch, rt)
    localized = []

    def first_look(frame):
        localized.append(time.monotonic())
        device.freeze(None if stall == "at_the_localization" else localized[0] + 0.5)

    looks = _looks(monkeypatch, rt, after_first=first_look)

    out = rt.execute("point_at", {"label": "red cube"})

    assert len(looks) == 2, "test setup: the target must be looked at again"
    assert looks[1].t == device.last.t, "test setup: the look again must get the stalled buffer"
    assert (looks[1].t > localized[0]) == (stall != "at_the_localization"), "test setup: when it stalled"
    assert out["ok"] is False and _commands(r) == [] and r.jaws == [], out
    why = ("the image it was found on again was taken before the plan started" if stall == "at_the_localization"
           else "the image it was found on again was received during the plan, before this look started")
    assert re.search(r"it was not seen again after a long plan \(\d+\.\d s\): " + re.escape(why), out["error"]), out
    assert out["error"].endswith("nothing was moved"), out


@needs_pin
def test_a_live_camera_still_points_after_a_long_real_plan(rig, monkeypatch):  # noqa: F811
    """Control for the stalls above: the same device wrapper, never stalled,
    the same real 3.5 s plan: it points."""
    r = rig("mock")
    rt = r.runtime
    _real_long_plan(monkeypatch)
    _Device(monkeypatch, rt)
    looks = _looks(monkeypatch, rt)

    out = rt.execute("point_at", {"label": "red cube"})

    assert len(looks) == 2 and out["ok"] is True and _commands(r) and r.jaws, out


# ── the camera's device: a stalled driver, an Isaac-style bridge ─────────────


def _isaac_capture(t, source=("bridge", 5560), camera="cam0", robot_id="mock") -> dict:
    """An Isaac bridge frame's producer capture (`sim/bridge_client.py`): its
    identity and the physics loop's clock (one per sim: `robot_id`), never
    this host's."""
    return {"backend": "isaac", "source": list(source), "camera": camera, "renderer": "isaac", "t": float(t),
            "proprioception": {"robot_id": robot_id, "backend": "isaac", "time_source": "physics_loop_monotonic",
                               "t": float(t)}}


class _Device:
    """The rig camera's device (`rt.camera._camera.get_frame`, what the
    stream's pump grabs): the real mock render, until a test says otherwise.

    - `stamp`: every render carries an Isaac-style producer capture whose clock
      advances one tick per render (`source` names the bridge); `tick()`: a
      render of another camera of the same sim (one physics loop clock);
    - `freeze(at)`: the driver keeps handing back its last buffer, the same
      Frame: from now on (`at` None; a render in flight then is lost, so the
      buffer is `last` as it was at the call), else from its first render at
      or after monotonic time `at`;
    - `resend(frame)`: a bridge re-sends that capture (same pixels, same
      producer clock) in a NEW Frame per request, so `Frame.t`, the client's
      receipt time, is new on every delivery;
    - `resend_at`: the sim stalls on its first render at or after that
      monotonic time and the bridge keeps re-sending that render (as `resend`);
    - `repeat`: the bridge serves each render that many times (the sim renders
      slower than the bridge serves; each delivery a new Frame);
    - `hold_next()`: the next render is received (`Frame.t` stamped) and then
      held in the stream's pump until the returned release is set: an image
      in flight (`held`: the frames held so far); the next render waits for
      `resume()` (5 s at most), so a grab waiting for that delivery gets the
      held one however late it wakes;
    - `stamp = False` later: the frames carry no capture any more."""

    def __init__(self, monkeypatch, rt, *, stamp=False):
        self._real = rt.camera._camera.get_frame
        self._lock = threading.Lock()
        self.stamp, self.source, self.clock = stamp, ("bridge", 5560), 5000.0
        self.last = None
        self.frozen = False
        self.freeze_at = None
        self.resent = None
        self.resend_at = None
        self.repeat = 1
        self._served = 0
        self._hold = None
        self._resumed = threading.Event()
        self._resumed.set()
        self.held = []
        self.by_stamp = {}
        monkeypatch.setattr(rt.camera._camera, "get_frame", self.get_frame)

    def get_frame(self):
        # after a held image: the stream's waiting grab takes that one
        # (CameraStream.get_frame returns the LATEST delivery)
        self._resumed.wait(5.0)
        with self._lock:
            if self.frozen and self.last is not None:
                return self.last
            if self.resent is not None:
                return replace(self.resent, t=time.monotonic())
            if self.repeat > 1 and self.last is not None and self._served % self.repeat:
                self._served += 1
                return replace(self.last, t=time.monotonic())
        frame = self._real()
        with self._lock:
            if self.frozen and self.last is not None:
                # the driver stalled while this render was in flight: it
                # keeps handing back the buffer it had
                return self.last
            self._served += 1
            if self.stamp:
                self.clock += 1.0 / 30.0
                frame.capture = _isaac_capture(self.clock, self.source)
                self.by_stamp[self.clock] = frame
            self.last = frame
            if self.freeze_at is not None and frame.t >= self.freeze_at:
                self.frozen = True
            if self.resend_at is not None and frame.t >= self.resend_at:
                self.resent, self.resend_at = frame, None
            hold, self._hold = self._hold, None
        if hold is not None:
            self.held.append(frame)
            self._resumed.clear()
            hold[0].set()
            hold[1].wait(5.0)
        return frame

    def hold_next(self):
        """(received, release): `received` is set once the next render was
        received; the pump publishes it when `release` is set, and renders
        again after `resume()`."""
        hold = (threading.Event(), threading.Event())
        with self._lock:
            self._hold = hold
        return hold

    def resume(self):
        self._resumed.set()

    def tick(self) -> float:
        """Another camera of this sim rendered: the physics loop clock's
        stamp of that render (every camera of a sim shares that clock)."""
        with self._lock:
            self.clock += 1.0 / 30.0
            return self.clock

    def freeze(self, at=None):
        with self._lock:
            if at is None:
                self.frozen = True
            else:
                self.freeze_at = at

    def resend(self, frame):
        """Re-send, from now on, the capture `frame` (an analysed image) was made from."""
        with self._lock:
            self.resent = self.by_stamp[frame.capture["t"]]


def _looks(monkeypatch, rt, *, after_first=None, before=None, shifts=None) -> list:
    """`rt._localize` as it is, recording each analysed frame (the list it
    returns); `after_first(frame)` runs when the first localization returned,
    `before(i)` before the i-th call (0 = the first); `shifts[i]` moves the
    i-th fix (the last repeats), as `_localizations` does."""
    from test_point_at_planning_followups import _shifted

    real = rt._localize
    looks = []

    def localize(label, *a, **k):
        if before is not None:
            before(len(looks))
        frame, fix = real(label, *a, **k)
        if shifts is not None:
            fix = _shifted(fix, shifts[min(len(looks), len(shifts) - 1)])
        looks.append(frame)
        if len(looks) == 1 and after_first is not None:
            after_first(frame)
        return frame, fix

    monkeypatch.setattr(rt, "_localize", localize)
    return looks


def _resent_delivered(rt, device, timeout_s=5.0):
    """Wait until the stream delivered a re-sent capture (a grab already in
    flight when the re-send began delivers a fresh one first)."""
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        latest = rt.camera.latest()
        if latest is not None and device.resent is not None and latest is not device.resent \
                and latest.capture == device.resent.capture:
            return
        time.sleep(0.01)
    raise AssertionError("test setup: the stream never delivered the re-sent capture")


@needs_pin
@pytest.mark.parametrize("plan", ["first", "replan"])
def test_a_resent_capture_from_before_the_plan_is_not_a_new_look(rig, monkeypatch, plan):  # noqa: F811
    """#411 review Minor 1, the "re-sent capture" row: an Isaac-style bridge
    re-sends the capture the plan was made from, each time in a new Frame (a
    new receipt `Frame.t`). main compared only `Frame.t` and pointed (136
    commands and a jaw close). `replan`: the target moved 3 cm, so it is
    planned once more from the look again's fix; the next look gets that
    look's capture again (main pointed there too)."""
    r = rig("mock")
    rt = r.runtime
    _long_plans(monkeypatch)
    device = _Device(monkeypatch, rt, stamp=True)
    planned_from = 0 if plan == "first" else 1

    def before(i):
        if i == planned_from + 1:
            device.resend(looks[planned_from])
            _resent_delivered(rt, device)

    looks = _looks(monkeypatch, rt, before=before,
                   shifts=[(0.0, 0.0, 0.0), (0.0, 0.03, 0.0)] if plan == "replan" else None)

    out = rt.execute("point_at", {"label": "red cube"})

    assert len(looks) == planned_from + 2, "test setup: the target must be looked at again"
    resent, made_from = looks[-1], looks[planned_from]
    assert resent.capture == made_from.capture and resent.t > made_from.t, "test setup: a re-sent capture"
    assert out["ok"] is False and _commands(r) == [] and r.jaws == [], out
    assert ("it was not seen again after a long plan (4.0 s): its camera re-sent a capture no newer than the one "
            "the plan was made from") in out["error"], out
    assert out["error"].endswith("nothing was moved"), out


@needs_pin
@pytest.mark.parametrize("change", ["another_source", "no_capture_any_more"])
def test_a_look_again_whose_capture_cannot_be_compared_is_refused(rig, monkeypatch, change):  # noqa: F811
    """The plan was made from a producer capture; the look again's capture has
    another identity (the bridge reconnected elsewhere) or none at all: no
    producer clock says it is newer, so it is refused. main pointed."""
    r = rig("mock")
    rt = r.runtime
    _long_plans(monkeypatch)
    device = _Device(monkeypatch, rt, stamp=True)

    def first_look(frame):
        if change == "another_source":
            device.source = ("bridge", 5561)
        else:
            device.stamp = False

    looks = _looks(monkeypatch, rt, after_first=first_look)

    out = rt.execute("point_at", {"label": "red cube"})

    assert len(looks) == 2 and looks[0].capture is not None, "test setup"
    assert (looks[1].capture is None) == (change == "no_capture_any_more"), "test setup"
    assert out["ok"] is False and _commands(r) == [] and r.jaws == [], out
    assert ("it was not seen again after a long plan (4.0 s): its capture cannot be compared with the one the plan "
            "was made from (Camera capture identity changed across the freshness barrier)") in out["error"], out


@needs_pin
@pytest.mark.parametrize("plan", ["first", "replan"])
def test_fresh_producer_captures_still_point_after_a_long_plan(rig, monkeypatch, plan):  # noqa: F811
    """Control: the same Isaac-style frames, every capture new (the producer
    clock advances): it points, after a replan too."""
    r = rig("mock")
    rt = r.runtime
    _long_plans(monkeypatch)
    _Device(monkeypatch, rt, stamp=True)
    looks = _looks(monkeypatch, rt, shifts=[(0.0, 0.0, 0.0), (0.0, 0.03, 0.0)] if plan == "replan" else None)

    out = rt.execute("point_at", {"label": "red cube"})

    assert len(looks) == (3 if plan == "replan" else 2), len(looks)
    assert all(f.capture is not None for f in looks)
    assert out["ok"] is True and _commands(r) and r.jaws, out


# ── #413 review: a bounded first receipt, an image in flight, another camera ─


def _stamped_delivered(rt, timeout_s=3.0):
    """Wait until the stream delivers the device's stamped renders (a grab in
    flight when the device was wrapped delivers an unstamped frame first)."""
    end = time.monotonic() + timeout_s
    while time.monotonic() < end and getattr(rt.camera.latest(), "capture", None) is None:
        time.sleep(0.01)
    rt.camera.get_frame()
    assert rt.camera.latest().capture is not None, "test setup: the stream never delivered a stamped render"


_RESENT_STALL = (r"it was not seen again after a long plan \(\d+\.\d s\): its camera re-sent a capture it first "
                 r"received \d+\.\d\d s before this look started \(more than 0\.5 s: its renders stalled\)")


@needs_pin
@pytest.mark.parametrize("render", ["stalled_mid_plan", "slower_than_the_bridge"])
def test_a_render_that_stalled_during_the_plan_is_refused_however_often_it_is_re_sent(rig, monkeypatch, render):  # noqa: F811
    """#413 review Minor 2: 0.5 s into a real 3.5 s plan the sim stalls and
    its bridge keeps re-sending that render, each time in a new Frame. Its
    capture is newer than the plan's and its receipt `Frame.t` is new, so main
    pointed (136 commands and a jaw close) on an image 3 s old. Its FIRST
    receipt (`Frame.observed_t`) is bounded now. Control: a sim that renders
    slower than its bridge serves (each render sent 3 times: first received
    ~0.1 s before the look at most) points."""
    from cascade.perception.freshness import observed_at
    from cascade.skills import pointing

    r = rig("mock")
    rt = r.runtime
    _real_long_plan(monkeypatch)
    device = _Device(monkeypatch, rt, stamp=True)
    _stamped_delivered(rt)
    starts = []

    def first_look(frame):
        if render == "stalled_mid_plan":
            device.resend_at = time.monotonic() + 0.5
        else:
            device.repeat = 3

    looks = _looks(monkeypatch, rt, after_first=first_look, before=lambda i: starts.append(time.monotonic()))

    out = rt.execute("point_at", {"label": "red cube"})

    assert len(looks) == 2 and looks[0].capture is not None, "test setup: one look again, on stamped renders"
    assert looks[1].capture["t"] > looks[0].capture["t"], "test setup: a capture newer than the plan's"
    first_received = starts[1] - observed_at(looks[1])
    if render == "stalled_mid_plan":
        assert looks[1].observed_t is not None and first_received > 2.5, "test setup: a render re-sent for ~3 s"
        assert out["ok"] is False and _commands(r) == [] and r.jaws == [], out
        assert re.search(_RESENT_STALL, out["error"]), out
        assert out["error"].endswith("nothing was moved"), out
    else:
        assert first_received < pointing.FRESH_OBSERVED_S, "test setup: renders re-sent for ~0.1 s"
        assert out["ok"] is True and _commands(r) and r.jaws, out


@needs_pin
@pytest.mark.parametrize("first_received_s", [0.3, 0.7])
def test_the_first_receipt_bound_sits_between_a_slow_render_and_a_stalled_one(rig, monkeypatch,  # noqa: F811
                                                                              first_received_s):
    """`pointing.FRESH_OBSERVED_S` (0.5 s) from both sides: the sim renders
    once more just before the look again and its bridge re-sends that render,
    so the look again's image was first received `first_received_s` before
    the look started. 0.3 s (a ~3 Hz sim): points; 0.7 s: refused (main
    pointed on both)."""
    from cascade.perception.freshness import observed_at
    from cascade.skills import runtime as rmod

    r = rig("mock")
    rt = r.runtime
    _long_plans(monkeypatch)
    device = _Device(monkeypatch, rt, stamp=True)
    _stamped_delivered(rt)
    real_still = rmod.SkillRuntime._point_at_still_there
    first = []

    def still(self, *a, **k):
        device.resend_at = time.monotonic()  # the next render is the last one; the bridge re-sends it
        end = time.monotonic() + 5.0
        while device.resent is None and time.monotonic() < end:
            time.sleep(0.005)
        assert device.resent is not None, "test setup: the sim never rendered"
        first.append(device.resent.t)
        time.sleep(max(0.0, first[0] + first_received_s - time.monotonic()))
        return real_still(self, *a, **k)

    monkeypatch.setattr(rmod.SkillRuntime, "_point_at_still_there", still)
    looks = _looks(monkeypatch, rt)

    out = rt.execute("point_at", {"label": "red cube"})

    assert len(looks) == 2 and looks[1].capture == device.resent.capture, "test setup: the re-sent render"
    assert looks[1].observed_t is not None and observed_at(looks[1]) == first[0], "test setup: its first receipt"
    if first_received_s < 0.5:
        assert out["ok"] is True and _commands(r) and r.jaws, out
    else:
        assert out["ok"] is False and _commands(r) == [] and r.jaws == [], out
        assert re.search(_RESENT_STALL, out["error"]), out


class _RuntimeClock:
    """`cascade.skills.runtime`'s `time` module as it is, except that the
    next `monotonic()` on the thread that called `pin(t)` returns `t`, once:
    the instants point_at stamps (its localization, the look again's start)
    are chosen by the test, never by how long a runner took to plan (main's
    x86 CI: 0.105 s from the plan's image to the look, past the 0.1 s the
    row needed). `pinned`: the stamps handed out, in order."""

    def __init__(self, monkeypatch):
        from cascade.skills import runtime as rmod

        self._real, self._pins, self.pinned = rmod.time, {}, []
        monkeypatch.setattr(rmod, "time", self)

    def __getattr__(self, name):
        return getattr(self._real, name)

    def pin(self, t: float) -> None:
        self._pins[threading.get_ident()] = t

    def monotonic(self) -> float:
        t = self._pins.pop(threading.get_ident(), None)
        if t is None:
            return self._real.monotonic()
        self.pinned.append(t)
        return t


def _rendered_after(device, t, timeout_s=5.0):
    """The device's last render, once it is one received after monotonic
    time `t` (a starved pump may still hold one from before)."""
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        last = device.last
        if last is not None and last.t > t:
            return last
        time.sleep(0.005)
    raise AssertionError("test setup: the camera rendered nothing after the plan started")


@needs_pin
@pytest.mark.parametrize("image", ["in_flight", "stalled_0.3_s_before", "the_plans_own"])
def test_an_image_in_flight_when_the_look_started_counts_but_an_older_one_does_not(rig, monkeypatch, image):  # noqa: F811
    """#413 review Minor 3 (the single-frame path): the stream's pump stamps
    `Frame.t` when it receives an image and publishes it after its own work
    (the pose, the depth). An image received 30 ms before the look started
    and published after it was refused on main (\"received during the plan,
    before this look started\"; the review: refused from 25 ms on). It counts
    now. One received 0.3 s before the look (the driver stalled then and keeps
    handing it back) is still refused, and so is one from before the plan
    started however recent (the driver stalled at the localization; that
    image is 0.05 s old at the look, within the in-flight time).

    The instants are fixed, not raced: point_at's localization and the look's
    start come from a pinned runtime clock (`_RuntimeClock`), set from the
    receipt (`Frame.t`) of the very image the look gets, so no planning time
    can move an image across `pointing.FRESH_IN_FLIGHT_S`."""
    from cascade.skills import pointing
    from cascade.skills import runtime as rmod

    r = rig("mock")
    rt = r.runtime
    _long_plans(monkeypatch)
    device = _Device(monkeypatch, rt)
    clock = _RuntimeClock(monkeypatch)
    pending, stamps = [], []
    real_still = rmod.SkillRuntime._point_at_still_there

    def still(self, label, fix, localized, *a, **k):
        since = localized[1]
        if image == "the_plans_own":
            received = device.last.t          # frozen at the localization
            looking = received + 0.05
        else:
            _rendered_after(device, since)    # the camera works again after the plan started
            if image == "in_flight":
                got, release = device.hold_next()
                assert got.wait(5.0), "test setup: the camera never rendered"
                pending.append(release)
                received = device.held[-1].t
                looking = received + 0.03     # received 30 ms before the look, published after it
            else:
                device.freeze()
                received = device.last.t
                looking = received + 0.3      # the driver stalled 0.3 s before the look
        stamps.append((since, received, looking))
        clock.pin(looking)
        return real_still(self, label, fix, localized, *a, **k)

    monkeypatch.setattr(rmod.SkillRuntime, "_point_at_still_there", still)
    real_get = rt.camera.get_frame

    def get_frame(*a, **k):
        # the look's grab waits for the stream's next delivery: the held image
        while pending:
            threading.Timer(0.05, pending.pop().set).start()
        try:
            return real_get(*a, **k)
        finally:
            device.resume()

    monkeypatch.setattr(rt.camera, "get_frame", get_frame)

    def first_look(frame):
        if image == "the_plans_own":
            device.freeze()
            clock.pin(device.last.t + 0.01)   # point_at stamps its localization 10 ms after that image

    looks = _looks(monkeypatch, rt, after_first=first_look)

    out = rt.execute("point_at", {"label": "red cube"})

    assert len(looks) == 2 and len(stamps) == 1, "test setup: one look again"
    ((since, received, looking),) = stamps
    assert clock.pinned == ([since] if image == "the_plans_own" else []) + [looking], \
        "test setup: point_at stamped the pinned instants"
    assert looks[1].t == received < looking, "test setup: the look again got that image, received before it started"
    if image == "the_plans_own":
        assert received < since and looking - received < pointing.FRESH_IN_FLIGHT_S, \
            "test setup: an image from before the plan, within the in-flight time of the look"
    else:
        assert since < received, "test setup: an image received during the plan"
        assert (looking - received < pointing.FRESH_IN_FLIGHT_S) == (image == "in_flight"), "test setup"
    if image == "in_flight":
        assert out["ok"] is True and _commands(r) and r.jaws, out
    else:
        assert out["ok"] is False and _commands(r) == [] and r.jaws == [], out
        why = ("taken before the plan started" if image == "the_plans_own"
               else "received during the plan, before this look started")
        assert ("it was not seen again after a long plan (4.0 s): the image it was found on again was " + why
                ) in out["error"], out


class _SideCamera:
    """A second Isaac camera on its own bridge (an `_other_cams` entry): a real
    CameraStream (it records each capture's first receipt) on a device serving
    the mock scene's latest render under its own producer identity (\"cam1\",
    port 5562), one capture per request. Its captures are stamped on the
    clock of the sim that renders them: the primary's (`device.tick()`, one
    sim, one physics loop) or, with another `robot_id`, another sim's own.
    `stalled`: the sim rendered once and the bridge re-sends that render, in a
    new Frame per request; `reconnect()`: the bridge comes back on another
    port and goes on re-sending it (another capture `source`, the same
    producer clock: the stream keeps that capture's first receipt since #423
    review follow-up 1; main restarted it, so the re-send read as first
    received now)."""

    has_depth = True

    def __init__(self, rt, device, *, stalled=False, robot_id="mock"):
        from cascade.perception.stream import CameraStream

        self._rt, self._device, self._stalled, self._robot = rt, device, stalled, robot_id
        self._clock, self._render, self.port = 9000.0, None, 5562
        self.depth, self.extrinsics, self.fuse = rt.depth, rt.extrinsics, True
        self.stream = CameraStream(self, name="side", rate_hz=30.0)
        self.stream.open()

    def open(self):
        pass

    def close(self):
        pass

    def reconnect(self):
        self.port = 5563

    def _capture(self, t):
        return _isaac_capture(t, ("bridge", self.port), camera="cam1", robot_id=self._robot)

    def get_frame(self):
        time.sleep(1.0 / 30.0)
        if self._stalled and self._render is not None:
            return replace(self._render, t=time.monotonic(), capture=self._capture(self._render.capture["t"]))
        if self._robot == "mock":
            t = self._device.tick()
        else:
            self._clock += 1.0 / 30.0
            t = self._clock
        self._render = replace(self._rt.camera.latest(), t=time.monotonic(), observed_t=None, capture=self._capture(t))
        return self._render


def _delivered_through(stream, port, timeout_s=3.0):
    """Wait until the stream published a delivery that came through bridge
    port `port` (the reconnect took effect)."""
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        latest = stream.latest()
        if latest is not None and (getattr(latest, "capture", None) or {}).get("source", [None, None])[1] == port:
            return
        time.sleep(0.005)
    raise AssertionError(f"test setup: the side stream delivered nothing through port {port}")


_OLDER_ON_THE_SHARED_CLOCK = ("it was not seen again after a long plan (4.0 s): another camera's capture on the same "
                              "producer clock is no newer than the one the plan was made from")


@needs_pin
@pytest.mark.parametrize("side", ["fresh", "stalled", "stalled_then_reconnected", "fresh_then_the_primary",
                                  "fresh_on_another_clock", "stalled_on_another_clock",
                                  "stalled_then_reconnected_on_another_clock"])
def test_a_look_again_found_on_another_camera_is_judged_by_its_clock_and_receipts(rig, monkeypatch, side):  # noqa: F811
    """#413 review Minor 1: the plan was made from the primary camera's
    capture; on the look again the primary misses and the side camera finds
    the target. Its capture identity differs (main refused a FRESH side render:
    \"capture identity changed\"), so it is judged by its own receipts, and,
    since the #418 review (Minors 1-2), on the producer clock it shares with
    the primary: both cameras of one Isaac sim stamp their renders on its
    physics loop. A fresh render points; a render the side bridge has re-sent
    since before the call is refused, also through a new bridge port
    (`stalled_then_reconnected`: the bridge came back on another port right
    before the look; #418's main moved, 136 commands and a jaw close, on that
    pre-call capture). `fresh_then_the_primary`: the side camera's fix moved
    3 cm, so it is planned once more from that image, and the next look finds
    it on the primary again, on a newer render: it points. On another sim's
    clock (`..._on_another_clock`) nothing is compared: a fresh render points,
    a re-sent one is refused by its first receipt, also after a reconnect
    (#423 review follow-up 1: the stream restarted that capture's first
    receipt at the new port, and main moved, 136 commands and a jaw close;
    its record is keyed on the producer clock now)."""
    from cascade.perception.freshness import observed_at
    from cascade.skills import runtime as rmod
    from cascade.types import SkillError

    r = rig("mock")
    rt = r.runtime
    _long_plans(monkeypatch)
    device = _Device(monkeypatch, rt, stamp=True)
    _stamped_delivered(rt)
    stalled = side.startswith("stalled")
    camera = _SideCamera(rt, device, stalled=stalled, robot_id="other-sim" if "another_clock" in side else "mock")
    replan = side == "fresh_then_the_primary"
    try:
        monkeypatch.setattr(rt, "_other_cams", lambda: [camera])
        real = rmod.localize_object

        def localize_object(frame, *a, **k):
            primary = (getattr(frame, "capture", None) or {}).get("camera") == "cam0"
            if primary and (len(looks) == 1 if replan else looks):
                raise SkillError("No current visual match for 'red cube' (the primary camera lost it)")
            return real(frame, *a, **k)

        def before(i):
            if i == 1 and side.startswith("stalled_then_reconnected"):
                camera.reconnect()
                _delivered_through(camera.stream, 5563)
            if i == 2 and replan:
                # the primary renders on after the side camera's image (one physics loop)
                end = time.monotonic() + 3.0
                while rt.camera.latest().capture["t"] <= looks[1].capture["t"] and time.monotonic() < end:
                    time.sleep(0.005)

        monkeypatch.setattr(rmod, "localize_object", localize_object)
        looks = _looks(monkeypatch, rt, before=before,
                       shifts=[(0.0, 0.0, 0.0), (0.0, 0.03, 0.0)] if replan else None)
        if stalled:
            time.sleep(0.8)  # the side camera's bridge re-sends its one render meanwhile

        out = rt.execute("point_at", {"label": "red cube"})
    finally:
        camera.stream.close()

    assert len(looks) == (3 if replan else 2) and looks[0].capture["camera"] == "cam0", "test setup: planned on cam0"
    assert looks[1].capture["camera"] == "cam1", "test setup: found again by the side camera"
    shared = "another_clock" not in side
    assert (looks[1].capture["proprioception"]["robot_id"] == "mock") == shared, "test setup: the side camera's sim"
    if replan:
        assert looks[2].capture["camera"] == "cam0", "test setup: the replan's look found it on the primary"
        assert looks[2].capture["t"] > looks[1].capture["t"], "test setup: a newer primary render"
    if stalled:
        assert (looks[1].capture["t"] < looks[0].capture["t"]) == shared, "test setup: a render from before the call"
        assert out["ok"] is False and _commands(r) == [] and r.jaws == [], out
        if "reconnected" in side:
            assert looks[1].capture["source"] == ["bridge", 5563], "test setup: found through the new bridge port"
            assert observed_at(looks[1]) < looks[0].t, "the reconnect kept the capture's first receipt"
        if shared:
            assert _OLDER_ON_THE_SHARED_CLOCK in out["error"], out
        else:
            assert looks[1].observed_t is not None, "test setup: a re-sent side render"
            assert re.search(_RESENT_STALL, out["error"]), out
    else:
        assert out["ok"] is True and _commands(r) and r.jaws, out


@needs_pin
def test_the_camera_a_localization_came_from_is_the_calling_threads_and_the_last_calls(rig, monkeypatch):  # noqa: F811
    """`_localized_camera()`: the stream `_localize`'s last answer on THIS
    thread came from (the primary's here); None on another thread, and None
    again after a localization that found nothing."""
    from cascade.skills import runtime as rmod
    from cascade.types import SkillError

    r = rig("mock")
    rt = r.runtime
    rt._localize("red cube")
    assert rt._localized_camera() is rt.camera
    other = []
    thread = threading.Thread(target=lambda: other.append(rt._localized_camera()))
    thread.start()
    thread.join()
    assert other == [None]

    def miss(*a, **k):
        raise SkillError("No current visual match for 'red cube'")

    monkeypatch.setattr(rmod, "localize_object", miss)
    with pytest.raises(SkillError):
        rt._localize("red cube", memory=False)
    assert rt._localized_camera() is None


@needs_pin
@pytest.mark.parametrize("planned,found", [("primary", "primary"), (None, "primary"), ("primary", None),
                                           ("primary", "side")])
def test_captures_are_compared_unless_the_look_again_is_known_to_be_another_cameras(rig, monkeypatch,  # noqa: F811
                                                                                   planned, found):
    """The look again's rule matrix, on `_point_at_still_there` itself: the
    plan's image and the look again's carry producer captures of different
    identities. Compared (refused: \"cannot be compared\") when both are one
    camera's, or when either camera is unknown (a belief's or the VLM's
    answer: as main); when the look again is known to come from ANOTHER
    camera than the plan's, only their producer clock is (one sim here, and
    the look's capture is newer on it): the receipts decide (fresh here)."""
    from cascade.skills import pointing
    from cascade.types import SkillError

    r = rig("mock")
    rt = r.runtime
    cameras = {"primary": rt.camera, "side": object(), None: None}
    planned_on, fix = rt._localize("red cube")
    planned_on = replace(planned_on, capture=_isaac_capture(5000.0))
    monkeypatch.setattr(pointing, "FRESH_PLAN_S", 3.0)
    monkeypatch.setattr(pointing, "_wall", lambda: 1004.0)
    localized = (1000.0, time.monotonic(), planned_on, cameras[planned])
    looked = []

    def localize(label, *a, **k):
        looked.append(label)
        rt._note_localized_on(cameras[found])
        return replace(planned_on, t=time.monotonic(), capture=_isaac_capture(5001.0, ("bridge", 5562), "cam1")), fix

    monkeypatch.setattr(rt, "_localize", localize)
    harness = rt.arm.harness
    still = functools.partial(rt._point_at_still_there, "red cube", fix, localized, harness,
                              getattr(harness, "_halt_generation", None), False)
    if found == "side":
        assert still() is None
    else:
        with pytest.raises(SkillError, match=r"its capture cannot be compared with the one the plan was made from"):
            still()
    assert looked == ["red cube"], "test setup: one look again"


def _look_again_on(rig, monkeypatch, frame_of, *, found="side"):  # noqa: F811
    """`_point_at_still_there` itself, after a (fake) 4 s plan, on the mock
    rig: the plan's image carries an Isaac cam0 capture at physics time 5000,
    its camera is the primary; the look again
    returns `frame_of(planned_on, looking)` (`looking`: the look's start, a
    pinned runtime instant 1 s ago) as found by `found` (\"side\": another
    camera), and point_at sees it again at that image's receipt (`Frame.t`,
    pinned too: no runner time between them, #423 review Minor 2). Returns
    the call (raises its SkillError); `call.clock`: the pinned clock."""
    from cascade.skills import pointing

    r = rig("mock")
    rt = r.runtime
    clock = _RuntimeClock(monkeypatch)
    planned_on, fix = rt._localize("red cube")
    planned_on = replace(planned_on, capture=_isaac_capture(5000.0))
    monkeypatch.setattr(pointing, "FRESH_PLAN_S", 3.0)
    monkeypatch.setattr(pointing, "_wall", lambda: 1004.0)
    looking = time.monotonic() - 1.0
    localized = (1000.0, looking - 0.2, planned_on, rt.camera)
    cameras = {"side": object(), "primary": rt.camera}

    def localize(label, *a, **k):
        rt._note_localized_on(cameras[found])
        frame = frame_of(planned_on, looking)
        clock.pin(float(frame.t))      # point_at's `seen_again`: the next runtime instant on this thread
        return frame, fix

    monkeypatch.setattr(rt, "_localize", localize)
    harness = rt.arm.harness

    def call():
        clock.pin(looking)
        return rt._point_at_still_there("red cube", fix, localized, harness,
                                        getattr(harness, "_halt_generation", None), False)

    call.clock = clock
    return call


#: the look again's capture, against the plan's (an Isaac cam0 capture at physics time 5000): None = points
_SHARED_CLOCK_ROWS = {
    "newer_on_the_shared_clock": (lambda: _isaac_capture(5000.1, ("bridge", 5562), "cam1"), None),
    "older_on_the_shared_clock": (lambda: _isaac_capture(4999.9, ("bridge", 5562), "cam1"),
                                  "another camera's capture on the same producer clock is no newer"),
    "the_plans_own_capture_via_another_stream": (lambda: _isaac_capture(5000.0),
                                                 "another camera's capture on the same producer clock is no newer"),
    "another_bridge_port_same_sim": (lambda: _isaac_capture(4999.9, ("bridge", 5563), "cam0"),
                                     "another camera's capture on the same producer clock is no newer"),
    "older_on_another_sims_clock": (lambda: _isaac_capture(10.0, ("bridge", 5562), "cam1", robot_id="other"), None),
    "an_invalid_capture": (lambda: {**_isaac_capture(5000.1, ("bridge", 5562), "cam1"), "proprioception": {}},
                           "its capture cannot be compared with the one the plan was made from"),
    "no_capture": (lambda: None, None),
}


@needs_pin
@pytest.mark.parametrize("row", list(_SHARED_CLOCK_ROWS))
def test_another_cameras_capture_is_ordered_on_the_producer_clock_it_shares(rig, monkeypatch, row):  # noqa: F811
    """#418 review Minors 1-2: a look again attributed to ANOTHER stream than
    the plan's skipped the capture comparison, so the plan's own capture (two
    streams on one Isaac camera, or a wrong attribution), or an older one on
    the same sim's physics clock, passed on its receipts alone. Captures on
    one producer clock (same robot_id and time source, whatever camera or
    bridge port) are compared now; another sim's clock is not, a local frame
    has none, and a capture that cannot be read refuses."""
    from cascade.types import SkillError

    capture, refused = _SHARED_CLOCK_ROWS[row]
    call = _look_again_on(rig, monkeypatch, lambda planned_on, looking: replace(
        planned_on, t=time.monotonic(), observed_t=None, capture=capture()))
    if refused is None:
        assert call() is None
    else:
        with pytest.raises(SkillError, match=re.escape(refused)):
            call()


#: (first receipt relative to the look's start, or a value; the refusal or None = points)
_FIRST_RECEIPTS = {
    "0.3_s_before": (lambda looking, now: looking - 0.3, None),
    "0.51_s_before": (lambda looking, now: looking - 0.51, "first received 0.51 s before this look started (more "
                                                           "than 0.5 s: its renders stalled)"),
    "0.503_s_before": (lambda looking, now: looking - 0.503, "first received 0.51 s before this look started"),
    "0.7_s_before": (lambda looking, now: looking - 0.7, "first received 0.70 s before this look started"),
    "nan": (lambda looking, now: math.nan, "its camera gave no usable first receipt of that capture (nan)"),
    "-inf": (lambda looking, now: -math.inf, "its camera gave no usable first receipt of that capture (-inf)"),
    "+inf": (lambda looking, now: math.inf, "its camera gave no usable first receipt of that capture (inf)"),
    "in_the_future": (lambda looking, now: now + 5.0, "its camera gave no usable first receipt of that capture "
                                                       "(stamped 5.00 s in the future)"),
}


@needs_pin
@pytest.mark.parametrize("first", list(_FIRST_RECEIPTS))
def test_the_first_receipt_must_be_a_time_the_heartbeat_would_take(rig, monkeypatch, first):  # noqa: F811
    """#418 review Minors 3-5: the look again's capture was re-sent (its
    `Frame.observed_t`, the first receipt, is set) by the plan's camera,
    received after the look started. A first receipt the perception
    heartbeat ignores (NaN, +-inf, the future) proves no receipt: refused
    (main moved on +inf and the future, and named NaN/-inf as \"first
    received nan s before\"); a future one is named by how far ahead it is,
    rounded up (never a raw clock reading). A finite one more than 0.5 s before the look is
    refused with its age rounded UP (0.51 s read \"0.5 s ... (more than 0.5
    s\" on main; 0.503 s reads 0.51 s, never 0.50 s); 0.3 s points."""
    from cascade.types import SkillError

    stamp, refused = _FIRST_RECEIPTS[first]
    instants = []

    def frame_of(planned_on, looking):
        received = time.monotonic()   # after the look started
        instants.extend([looking, received])
        return replace(planned_on, t=received, observed_t=stamp(looking, received), capture=_isaac_capture(5000.1))

    call = _look_again_on(rig, monkeypatch, frame_of, found="primary")
    if refused is None:
        assert call() is None
    else:
        with pytest.raises(SkillError, match=re.escape(refused)):
            call()
    # the look started and saw the image again at the instants the test chose (#423 review Minor 2: a runner
    # stall between the image's receipt and `seen_again` read \"4.97 s in the future\")
    assert call.clock.pinned == instants, "test setup: point_at's own instants were pinned"


# ── 2. the freshness clock starts at the localization ──────────────────────


@needs_pin
def test_the_core_class_search_counts_toward_the_look_again(rig, monkeypatch):  # noqa: F811
    """Budget()'s core class search took 3.5 s (a loaded host), the plan
    itself no time: the fix is 3.5 s old at the first command, so the target
    is looked at again. main started the clock after Budget() and never
    looked."""
    from cascade.skills import pointing

    r = rig("mock")
    rt = r.runtime
    now = _long_plans(monkeypatch, seconds=0.0)

    def slow_search():
        now[0] += 3.5
        return {}

    monkeypatch.setattr(pointing, "class_factors", slow_search)
    calls = _localizations(monkeypatch, rt, [(0.0, 0.0, 0.0), (0.0, 0.005, 0.0)])

    out = rt.execute("point_at", {"label": "red cube"})

    assert out["ok"] is True and len(calls) == 2, out


@needs_pin
def test_the_replan_starts_its_clock_at_the_look_again(rig, monkeypatch):  # noqa: F811
    """The first plan took 4 s and the target moved 3 cm; the plan from where
    it is now takes no time. Its fix comes from the look again, so it is not
    looked at once more (a clock left at the first localization would)."""
    from cascade.skills import pointing

    r = rig("mock")
    rt = r.runtime
    now = _long_plans(monkeypatch, seconds=0.0)
    planned = pointing.plan_route
    plans = []

    def route(*a, **k):
        plans.append(1)
        if len(plans) == 1:
            now[0] += 4.0
        return planned(*a, **k)

    monkeypatch.setattr(pointing, "plan_route", route)
    calls = _localizations(monkeypatch, rt, [(0.0, 0.0, 0.0), (0.0, 0.03, 0.0)])

    out = rt.execute("point_at", {"label": "red cube"})

    assert out["ok"] is True and len(plans) == 2 and len(calls) == 2, (out, len(plans), len(calls))


# ── 3. a halt keeps the operator's reason ──────────────────────────────────


def _fire(harness, kind):
    harness.halt("operator: stop")
    if kind == "halt_then_estop":
        harness.estop("operator e-stop")


@needs_pin
@pytest.mark.parametrize("kind", ["halt", "halt_then_estop"])
@pytest.mark.parametrize("when", ["after_planning", "inside_the_look_again", "at_the_first_command"])
def test_a_halt_before_or_at_the_first_command_keeps_the_operators_reason(rig, monkeypatch, when, kind):  # noqa: F811
    """The review's rows: a halt inside the look again or right after it was
    reported as "halt received during route planning or execution" (the
    first command's own check). The operator's reason is kept (an e-stop still
    wins), and nothing is commanded."""
    from cascade.skills import runtime as rmod

    r = rig("mock")
    rt = r.runtime
    harness = rt.arm.harness
    fired = []

    def fire():
        if not fired:
            fired.append((len(_commands(r)), len(r.jaws)))
            _fire(harness, kind)

    if when == "inside_the_look_again":
        _long_plans(monkeypatch)
        real_localize = rt._localize
        looks = []

        def localize(label, *a, **k):
            found = real_localize(label, *a, **k)
            looks.append(1)
            if len(looks) == 2:
                fire()
            return found

        monkeypatch.setattr(rt, "_localize", localize)
    elif when == "after_planning":
        real_still = rmod.SkillRuntime._point_at_still_there

        def still(self, *a, **k):
            real_still(self, *a, **k)
            fire()

        monkeypatch.setattr(rmod.SkillRuntime, "_point_at_still_there", still)
    else:
        for name in ("_set_gripper", "_move_joints"):
            real = getattr(rt, name)
            monkeypatch.setattr(rt, name, lambda *a, _real=real, **k: (fire(), _real(*a, **k))[1])

    out = rt.execute("point_at", {"label": "red cube"})

    assert fired == [(0, 0)], f"{when} never ran"
    assert out["ok"] is False and _commands(r) == [] and r.jaws == [], out
    if kind == "halt":
        assert out["error"] == "MotionHalted: halted: operator: stop", out
    else:
        assert out["error"] == "SafetyViolation: e-stop latched", out


@needs_pin
def test_a_halt_after_planning_refuses_before_the_first_command_outside_execute_too(rig, monkeypatch):  # noqa: F811
    """skill_point_at called directly (no execute(), so no skill halt scope):
    main's first command cleared that halt, pointed (136 commands, the jaws
    closed) and only then reported the halt, without its reason. Its start
    generation now refuses it before the first command, with the reason."""
    from cascade.skills import runtime as rmod
    from cascade.types import MotionHalted

    r = rig("mock")
    rt = r.runtime
    real_still = rmod.SkillRuntime._point_at_still_there

    def still(self, *a, **k):
        real_still(self, *a, **k)
        rt.arm.harness.halt("operator: stop")

    monkeypatch.setattr(rmod.SkillRuntime, "_point_at_still_there", still)
    with pytest.raises(MotionHalted, match="^halted: operator: stop$"):
        rt.skill_point_at("red cube")
    assert _commands(r) == [] and r.jaws == []
