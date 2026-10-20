"""A refusal BEFORE a move's first approved command never latches the e-stop (#414 review Important 1).

Two handlers turned ANY exception of their move into `arm.stop()`, and `SafeArm.stop()` latches the
e-stop: the grasp's pre-grasp home move and `_return_to_view_home`. A refusal at the move's start
(the perception watchdog, the e-stop, a route preflight) left the arm where it was, yet ended in an
e-stop only the operator can clear. #414 made that reachable in normal operation: a camera out for
5 s during `grasp_at_pixel`, and fr3's view-home return whose frame was 4.8 s old at its freshness
check plus 0.3 s of planning. The next grasp then read "NoExecutableGrasp ... approach unsafe: e-stop
latched" four times over.

Rule: a refusal (SafetyViolation / SkillError, not an attachment loss, not raised from a backend
fault) before the move's first approved command (`SafetyHarness.motions_begun` unchanged) stops
nothing: the grasp raises it as itself (the e-stop wins), the view-home return is "skipped". A move
that started (a fault mid-stream, did not settle) or a backend fault still latches, and the error
says "e-stop latched". A latched e-stop refuses the next grasp as itself.

The harness is also built stale (#414 review Minor 4): no motion before a first real observation.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from base_pose_oracle import make_runtime, raw_arm  # noqa: F401  (make_runtime: a fixture, by name)
from test_watchdog_frame_age import (MOVE, MOVE_BACK, PIXEL, _call, _harnesses, _jaw_counter, _kill, _mock,
                                     _revive, _short_waits)

from cascade.safety.harness import SafetyHarness, SafetyLimits
from cascade.types import MotionHalted, SafetyViolation, SkillError


def _estopped(rt) -> dict:
    return {name: h.estopped for name, h in _harnesses(rt).items()}


def _after_first(rt, monkeypatch, method, before=None, after=None):
    """Run `before()` / `after()` around the FIRST call of `rt.<method>` only."""
    real = getattr(rt, method)
    calls = []

    def wrapped(*a, **k):
        calls.append(1)
        if len(calls) > 1:
            return real(*a, **k)
        if before is not None:
            before()
        try:
            return real(*a, **k)
        finally:
            if after is not None:
                after()

    monkeypatch.setattr(rt, method, wrapped)
    return calls


def _fail_send_after(raw, monkeypatch, n, exc):
    """From now on the raw arm sends `n` more commands, then its next send raises `exc`."""
    real = raw.send_joint_target
    sent = []

    def send(q):
        if len(sent) >= n:
            raise exc
        sent.append(1)
        return real(q)

    monkeypatch.setattr(raw, "send_joint_target", send)
    return lambda: monkeypatch.setattr(raw, "send_joint_target", real)


# ── the grasp's pre-grasp home move ───────────────────────────────────────


def test_a_camera_dying_after_the_jaw_opening_refuses_the_home_move_with_no_latch(make_runtime, monkeypatch):  # noqa: F811
    """The camera dies right after the pre-grasp jaw opening: the home move's own watchdog check
    refuses it before its first command. #414: the handler stopped the arm, so the e-stop latched."""
    rt = _mock(make_runtime)
    _short_waits(rt, monkeypatch)
    jaws = _jaw_counter(rt)
    rt.observe()
    _after_first(rt, monkeypatch, "_set_gripper", after=lambda: _kill(rt))
    out, sent = _call(rt, "grasp_at_pixel", PIXEL)
    assert out["ok"] is False, out
    assert sent == {name: 0 for name in sent} and jaws == {name: 1 for name in jaws}
    assert _estopped(rt) == {name: False for name in sent}
    assert str(out["error"]).startswith("SafetyViolation: perception watchdog: last observation "), out
    assert "home move failed" not in out["error"]
    _revive(rt)
    out, sent = _call(rt, "grasp_at_pixel", PIXEL)
    assert out["ok"] is True, out
    assert all(n > 0 for n in sent.values()), sent


def test_a_retry_loop_ends_on_the_watchdog_refusal_without_a_latch(make_runtime, monkeypatch):  # noqa: F811
    """The persistence loop (pick_and_place's, handover's): the latch used to end it ("e-stop
    latched; not retrying"); now the watchdog refusal ends it at once, as a stale frame does, with
    no re-home or second attempt and no latch."""
    rt = _mock(make_runtime)
    _short_waits(rt, monkeypatch)
    rt.observe()
    calls = _after_first(rt, monkeypatch, "_set_gripper", after=lambda: _kill(rt))
    homes = []
    monkeypatch.setattr(rt, "skill_move_home", lambda *a, **k: homes.append(1))
    n0 = len(raw_arm(rt.arm).commands)
    out = rt._grasp_with_persistence("red cube")
    assert out["ok"] is False and out["grasp_attempts"] == 1, out
    assert out["error"].startswith("SafetyViolation: perception watchdog: last observation "), out
    assert out["note"] == "the perception watchdog refused the motion; no automatic motion recovery"
    assert homes == [] and calls == [1] and len(raw_arm(rt.arm).commands) == n0
    assert _estopped(rt) == {name: False for name in _estopped(rt)}


def test_a_route_preflight_refusal_of_the_home_move_latches_nothing(make_runtime, monkeypatch):  # noqa: F811
    """The home move's route preflight finds no safe route (`plan_route` raises before any command)."""
    from cascade.safety import trajectory

    rt = _mock(make_runtime)
    jaws = _jaw_counter(rt)
    real_plan = trajectory.plan_route

    def refuse(*_a, **_k):
        raise SkillError("no safe joint route found; direct route: synthetic refusal")

    _after_first(rt, monkeypatch, "_move_planned",
                 before=lambda: monkeypatch.setattr(trajectory, "plan_route", refuse),
                 after=lambda: monkeypatch.setattr(trajectory, "plan_route", real_plan))
    out, sent = _call(rt, "grasp_at_pixel", PIXEL)
    assert out["error"] == "SkillError: no safe joint route found; direct route: synthetic refusal", out
    assert sent == {name: 0 for name in sent} and jaws == {name: 1 for name in jaws}
    assert _estopped(rt) == {name: False for name in sent}
    out, sent = _call(rt, "grasp_at_pixel", PIXEL)
    assert out["ok"] is True, out


def test_an_operator_estop_before_the_home_move_is_reported_as_the_estop_and_never_doubled(make_runtime, monkeypatch):  # noqa: F811
    """The operator's e-stop lands right after the jaw opening: the home move is refused by it, the
    error is the e-stop, and the handler stops nothing on top of it (#414: "home move failed before
    pregrasp: no safe joint route found; direct route: e-stop latched" and a second, own e-stop)."""
    rt = _mock(make_runtime)
    jaws = _jaw_counter(rt)
    h = rt.arm.harness
    stops = []
    monkeypatch.setattr(raw_arm(rt.arm), "stop", lambda: stops.append(1))
    _after_first(rt, monkeypatch, "_set_gripper", after=lambda: h.estop("operator e-stop"))
    out, sent = _call(rt, "grasp_at_pixel", PIXEL)
    assert out["error"] == "SafetyViolation: e-stop latched", out
    assert sent == {name: 0 for name in sent} and jaws == {name: 1 for name in jaws}
    assert h.estopped and stops == []
    assert [v for v in h.violations if v.startswith("ESTOP")] == ["ESTOP: operator e-stop"]


def test_an_operator_halt_before_the_home_move_stays_a_halt(make_runtime, monkeypatch):  # noqa: F811
    rt = _mock(make_runtime)
    jaws = _jaw_counter(rt)
    h = rt.arm.harness
    _after_first(rt, monkeypatch, "_set_gripper", after=lambda: h.halt("operator: stop"))
    out, sent = _call(rt, "grasp_at_pixel", PIXEL)
    assert out["error"] == "MotionHalted: halt received during route planning or execution", out
    assert sent == {name: 0 for name in sent} and jaws == {name: 1 for name in jaws}
    assert not h.estopped


@pytest.mark.parametrize("fault", ["backend", "safety"])
def test_a_fault_mid_home_move_still_latches_and_says_so(make_runtime, monkeypatch, fault):  # noqa: F811
    """The home move started (5 commands went out) and then failed: the arm stopped somewhere on the
    way, so the e-stop latches as before, and now the error says so. The next grasp is refused as
    the e-stop itself (#414: "NoExecutableGrasp: ... approach unsafe: e-stop latched" x4)."""
    rt = _mock(make_runtime)
    jaws = _jaw_counter(rt)
    raw = raw_arm(rt.arm)
    exc = RuntimeError("CAN bus fault") if fault == "backend" else SafetyViolation("joint 3 tracking fault")
    restore = []
    _after_first(rt, monkeypatch, "_move_planned",
                 before=lambda: restore.append(_fail_send_after(raw, monkeypatch, 5, exc)),
                 after=lambda: restore.pop()())
    out, sent = _call(rt, "grasp_at_pixel", PIXEL)
    assert out["error"] == f"SkillError: home move failed before pregrasp: {exc}; e-stop latched", out
    assert sent == {name: 5 for name in sent} and jaws == {name: 1 for name in jaws}
    assert _estopped(rt) == {name: True for name in sent}
    for skill, args in (("grasp_at_pixel", PIXEL), ("grasp_object", {"label": "red cube"})):
        out, sent = _call(rt, skill, args)
        assert out["error"] == "SafetyViolation: e-stop latched", (skill, out)
        assert sent == {name: 0 for name in sent} and jaws == {name: 1 for name in jaws}


def test_a_backend_fault_before_the_first_command_still_latches_and_says_so(make_runtime, monkeypatch):  # noqa: F811
    """A backend fault outside the safety checks (the joint read the route starts from fails) keeps
    the latch, whether or not a command went out."""
    rt = _mock(make_runtime)
    raw = raw_arm(rt.arm)
    real_state = raw.get_state

    def broken(*_a, **_k):
        raise RuntimeError("joint feedback read failed")

    _after_first(rt, monkeypatch, "_move_planned",
                 before=lambda: monkeypatch.setattr(raw, "get_state", broken),
                 after=lambda: monkeypatch.setattr(raw, "get_state", real_state))
    out, sent = _call(rt, "grasp_at_pixel", PIXEL)
    assert out["error"] == ("SkillError: home move failed before pregrasp: joint feedback read failed; "
                            "e-stop latched"), out
    assert sent == {name: 0 for name in sent}
    assert _estopped(rt) == {name: True for name in sent}


def test_a_home_move_that_does_not_settle_still_latches_and_says_so(make_runtime, monkeypatch):  # noqa: F811
    rt = _mock(make_runtime)
    real = rt._move_planned
    calls = []

    def unsettled(*a, **k):
        calls.append(1)
        out = real(*a, **k)
        return False if len(calls) == 1 else out

    monkeypatch.setattr(rt, "_move_planned", unsettled)
    out, sent = _call(rt, "grasp_at_pixel", PIXEL)
    assert out["error"] == "SkillError: did not settle at home pose before pregrasp; e-stop latched", out
    assert all(n > 0 for n in sent.values())
    assert _estopped(rt) == {name: True for name in sent}


# ── fr3's view-home return (`_return_to_view_home`) ───────────────────────


def _fr3(make_runtime, monkeypatch):  # noqa: F811
    """fr3 (`return_home_after_motion`) on a MockArm backend, lazily built and materialized."""
    from cascade.apps import demo
    from cascade.config import load_demo_config
    from cascade.control.mock_arm import MockArm

    monkeypatch.setattr(demo, "make_arm", lambda acfg, kinematics=None: MockArm(acfg, kinematics=kinematics))
    rt = make_runtime(load_demo_config(arm="fr3", camera="mock", llm="mock"), lazy=True)
    for safe in rt.arm_rig.arms.values():
        safe.get_state()
    _short_waits(rt, monkeypatch)
    rt.observe()
    return rt


def test_fr3_view_home_return_refused_by_the_watchdog_is_skipped_not_latched(make_runtime, monkeypatch):  # noqa: F811
    """The r414 row: the camera died; its last frame is 4.8 s old when the return's freshness check
    reads it (it passes the 5 s limit) and the route planning takes 0.3 s, so the move's own
    watchdog check refuses it at 5.1 s, before any command. #414: "return to saved home failed:
    SafetyViolation: perception watchdog: last observation 5.1s old" and a latched e-stop."""
    rt = _fr3(make_runtime, monkeypatch)
    real_observe, real_carry = rt.observe, rt._move_carry
    log = {}

    def observe(*, require_fresh=False):
        if require_fresh and "age" not in log:
            _kill(rt, died_s=0.0)
            f = rt.watcher._cams[0].stream.latest()
            f.t -= 4.8 - (time.monotonic() - f.t) - 0.3  # get_frame's 0.3 s wait comes on top
            for h in _harnesses(rt).values():
                h._last_heartbeat = min(h._last_heartbeat, f.t)
            out = real_observe(require_fresh=True)
            log["age"] = time.monotonic() - out.t
            return out
        return real_observe(require_fresh=require_fresh)

    def carry(*a, **k):
        time.sleep(0.3)  # the route planning a real planner spends before begin_motion
        return real_carry(*a, **k)

    monkeypatch.setattr(rt, "observe", observe)
    monkeypatch.setattr(rt, "_move_carry", carry)
    out, sent = _call(rt, *MOVE)
    assert 4.7 < log["age"] < 5.0, log
    assert out["action_ok"] is True and all(n > 0 for n in sent.values()), out
    home = out["return_home"]
    assert home["status"] == "skipped", home
    assert home["reason"].startswith("home perception unavailable: perception watchdog: last observation 5."), home
    assert out["error"] == f"return to saved home skipped: {home['reason']}"
    assert _estopped(rt) == {name: False for name in sent}
    monkeypatch.setattr(rt, "observe", real_observe)
    monkeypatch.setattr(rt, "_move_carry", real_carry)
    _revive(rt)
    out, sent = _call(rt, *MOVE_BACK)
    assert out["ok"] is True, out


@pytest.mark.parametrize("refusal", ["estop", "preflight"])
def test_fr3_view_home_return_refused_at_its_start_is_skipped_with_its_reason(make_runtime, monkeypatch, refusal):  # noqa: F811
    """The operator's e-stop lands between the return's skip checks and its move (the move's own
    check refuses it), or a preflight refuses the move before any command: "skipped" with that
    reason, and no stop on top. #414: "failed" and an `arm.stop()` (an own e-stop)."""
    rt = _fr3(make_runtime, monkeypatch)
    h = rt.arm.harness
    raw = next(raw_arm(s) for s in rt.arm_rig.arms.values())
    stops = []
    monkeypatch.setattr(raw, "stop", lambda: stops.append(1))
    real_carry = rt._move_carry

    def carry(*a, **k):
        if refusal == "preflight":
            raise SafetyViolation("synthetic preflight refusal; no motion sent")
        h.estop("operator e-stop")
        return real_carry(*a, **k)

    monkeypatch.setattr(rt, "_move_carry", carry)
    out, _sent = _call(rt, *MOVE)
    reason = ("e-stop latched" if refusal == "estop"
              else "refused before any motion: SafetyViolation: synthetic preflight refusal; no motion sent")
    assert out["return_home"] == {"ok": False, "status": "skipped", "reason": reason}, out
    assert stops == [] and h.estopped is (refusal == "estop")
    assert [v for v in h.violations if v.startswith("ESTOP")] == (
        ["ESTOP: operator e-stop"] if refusal == "estop" else [])


def test_fr3_view_home_return_that_fails_mid_route_still_latches_and_says_so(make_runtime, monkeypatch):  # noqa: F811
    rt = _fr3(make_runtime, monkeypatch)
    raw = next(raw_arm(s) for s in rt.arm_rig.arms.values())
    restore = []
    _after_first(rt, monkeypatch, "_move_carry",
                 before=lambda: restore.append(_fail_send_after(raw, monkeypatch, 5, RuntimeError("CAN bus fault"))),
                 after=lambda: restore.pop()())
    out, _sent = _call(rt, *MOVE)
    home = out["return_home"]
    assert home == {"ok": False, "status": "failed", "error": "RuntimeError: CAN bus fault; e-stop latched"}, home
    assert _estopped(rt) == {name: True for name in _estopped(rt)}


# ── #414 review Minor 4: the harness is built stale ───────────────────────


def _harness():
    return SafetyHarness(SafetyLimits(workspace_min=np.full(3, -1.0), workspace_max=np.ones(3)))


def test_a_new_harness_refuses_motion_until_its_first_observation():
    """Main: the harness counted as fresh from its construction, so for `watchdog_s` an arm moved
    with no frame ever delivered."""
    h = _harness()
    q = np.zeros(6)
    with pytest.raises(SafetyViolation, match=r"^perception watchdog: no observation yet$"):
        h.begin_motion()
    for check in (h.check_stream_start, lambda: h.approve(q, q, 0.02), h.check_before_motion):
        with pytest.raises(SafetyViolation, match=r"^perception watchdog: no observation yet$"):
            check()
    h.heartbeat(at=math.nan)  # an unstamped frame is no observation
    with pytest.raises(SafetyViolation, match="no observation yet"):
        h.begin_motion()
    h.heartbeat(at=time.monotonic() - 2.0)
    h.begin_motion()
    h.end_motion()


def test_check_before_motion_refuses_like_begin_motion_without_starting_or_clearing_anything():
    """The pre-grasp jaw opening's check (#414 review Minor 2): begin_motion's refusals in its order
    (e-stop, a halt since the baseline, the watchdog), but it starts no motion and leaves a pending
    halt for that motion's begin_motion to clear."""
    h = _harness()
    h.estop("operator e-stop")  # stale AND e-stopped: the e-stop wins
    with pytest.raises(SafetyViolation, match=r"^e-stop latched$"):
        h.check_before_motion()
    h.reset_estop()
    baseline = h._halt_generation
    h.halt("operator: stop")  # stale AND halted since the baseline: the halt wins
    with pytest.raises(MotionHalted):
        h.check_before_motion(halt_generation=baseline)
    with pytest.raises(SafetyViolation, match="no observation yet"):
        h.check_before_motion()  # a halt from before the caller started is not its refusal
    h.heartbeat()
    h.check_before_motion()
    assert h.halted == "operator: stop" and not h.motion_active and h.motions_begun == 0


def test_no_frame_ever_refuses_the_first_motion_and_the_first_frame_releases_it(make_runtime, monkeypatch):  # noqa: F811
    """The r414 row "no frame ever + fresh harness": the camera never delivered and the runtime was
    built a moment ago (its warm-up wait skipped). Main: rotate_joint sent 100 commands."""
    from cascade.perception import mock_camera
    from cascade.perception.camera_base import CameraError
    from cascade.perception.stream import CameraStream

    real_grab = mock_camera.MockCamera._grab
    alive = {"on": False}

    def grab(self):
        if not alive["on"]:
            raise CameraError("camera not delivering yet")
        return real_grab(self)

    monkeypatch.setattr(mock_camera.MockCamera, "_grab", grab)
    monkeypatch.setattr(CameraStream, "warm_up", lambda self, n=10: None)
    rt = _mock(make_runtime)
    _short_waits(rt, monkeypatch)
    assert rt.camera.latest() is None
    out, sent = _call(rt, *MOVE)
    assert out["ok"] is False, out
    assert "perception watchdog: no observation yet" in str(out.get("error")), out
    assert sent == {name: 0 for name in sent}
    alive["on"] = True
    t0, deadline = rt.watcher.ticks, time.monotonic() + 10.0
    while rt.watcher.ticks < t0 + 2 and time.monotonic() < deadline:
        time.sleep(0.02)
    out, sent = _call(rt, *MOVE)
    assert out["ok"] is True, out
    assert all(n > 0 for n in sent.values()), sent


def test_a_live_camera_moves_on_the_first_call_with_no_watcher(make_runtime):  # noqa: F811
    """Live-camera startup: the build feeds the watchdogs from the warm-up's frames (a real
    observation), so a direct skill call right after the build moves, before any watcher tick and
    with no watcher at all."""
    from cascade.config import load_demo_config

    cfg = load_demo_config(arm="mock", camera="mock", llm="mock")
    cfg._data.setdefault("perception_loop", {})["enabled"] = False
    rt = make_runtime(cfg)
    assert rt.watcher is None
    n0 = len(raw_arm(rt.arm).commands)
    out = rt.skill_rotate_joint(joint=1, degrees=4.0)
    assert out.get("ok", True) is not False, out
    assert len(raw_arm(rt.arm).commands) > n0
