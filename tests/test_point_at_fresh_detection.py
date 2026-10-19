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
   `Budget()` (whose core-speed search can take ~1.2 s), and a re-plan's at
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
"""

from __future__ import annotations

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
    stalls = []

    def first_look(frame):
        timer = threading.Timer(0.0 if stall == "at_the_localization" else 0.5, device.freeze)
        timer.daemon = True
        stalls.append(timer)
        timer.start()

    looks = _looks(monkeypatch, rt, after_first=first_look)

    out = rt.execute("point_at", {"label": "red cube"})
    for timer in stalls:
        timer.cancel()

    assert len(looks) == 2, "test setup: the target must be looked at again"
    assert looks[1].t == device.last.t, "test setup: the look again must get the stalled buffer"
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


def _isaac_capture(t, source=("bridge", 5560)) -> dict:
    """An Isaac bridge frame's producer capture (`sim/bridge_client.py`): its
    identity and the physics loop's clock, never this host's."""
    return {"backend": "isaac", "source": list(source), "camera": "cam0", "renderer": "isaac", "t": float(t),
            "proprioception": {"robot_id": "mock", "backend": "isaac", "time_source": "physics_loop_monotonic",
                               "t": float(t)}}


class _Device:
    """The rig camera's device (`rt.camera._camera.get_frame`, what the
    stream's pump grabs): the real mock render, until a test says otherwise.

    - `stamp`: every render carries an Isaac-style producer capture whose clock
      advances one tick per render (`source` names the bridge);
    - `freeze()`: the driver keeps handing back its last buffer, the same Frame;
    - `resend(frame)`: a bridge re-sends that capture (same pixels, same
      producer clock) in a NEW Frame per request, so `Frame.t`, the client's
      receipt time, is new on every delivery;
    - `stamp = False` later: the frames carry no capture any more."""

    def __init__(self, monkeypatch, rt, *, stamp=False):
        self._real = rt.camera._camera.get_frame
        self._lock = threading.Lock()
        self.stamp, self.source, self.clock = stamp, ("bridge", 5560), 5000.0
        self.last = None
        self.frozen = False
        self.resent = None
        self.by_stamp = {}
        monkeypatch.setattr(rt.camera._camera, "get_frame", self.get_frame)

    def get_frame(self):
        with self._lock:
            if self.frozen and self.last is not None:
                return self.last
            if self.resent is not None:
                return replace(self.resent, t=time.monotonic())
        frame = self._real()
        with self._lock:
            if self.stamp:
                self.clock += 1.0 / 30.0
                frame.capture = _isaac_capture(self.clock, self.source)
                self.by_stamp[self.clock] = frame
            self.last = frame
        return frame

    def freeze(self):
        with self._lock:
            self.frozen = True

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


# ── 2. the freshness clock starts at the localization ──────────────────────


@needs_pin
def test_the_core_speed_search_counts_toward_the_look_again(rig, monkeypatch):  # noqa: F811
    """Budget()'s core-speed search took 3.5 s (a loaded host), the plan
    itself no time: the fix is 3.5 s old at the first command, so the target
    is looked at again. main started the clock after Budget() and never
    looked."""
    from cascade.skills import pointing

    r = rig("mock")
    rt = r.runtime
    now = _long_plans(monkeypatch, seconds=0.0)

    def slow_search():
        now[0] += 3.5
        return 1.0

    monkeypatch.setattr(pointing, "core_factor", slow_search)
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
