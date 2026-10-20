"""#422 review follow-ups on the pre-motion refusal rule.

1. Minor 1: "refused before any motion" is judged on a counter of APPROVED commands
   (`SafetyHarness.approved_commands`, bumped on every command approve() clears), never on the
   `motions_begun` flag. That flag is armed by begin_motion and cleared by ANY end_motion: a
   concurrent motion's `end_motion()` landing between this move's begin_motion and its first
   approve hid the start, and a fault mid-stream then read as "refused before motion" (the
   review's probe row race_end_motion: 4 commands out, then a fault, and no stop and no latch).
2. Minor 2: a retry loop that stops on a latched e-stop names it once, not
   "...; e-stop latched (e-stop latched; not retrying)".
3. Minor 4 (decision, pinned): a refusal of the pre-grasp home move before its first command that
   is not the watchdog (a workspace / route refusal) does not latch, so pick_and_place re-homes and
   tries again, every motion of the retry vetted as any other.
4. Minor 5: the home route is checked before the pre-grasp jaw opening: a route refusal costs 0 jaw
   commands (main: 1).
"""

from __future__ import annotations

import numpy as np
import pytest
from base_pose_oracle import make_runtime, raw_arm  # noqa: F401  (make_runtime: a fixture, by name)
from test_premotion_refusal_no_estop import _after_first, _estopped, _fail_send_after, _fr3
from test_watchdog_frame_age import MOVE, PIXEL, _call, _jaw_counter, _mock

from cascade.safety.harness import SafetyHarness, SafetyLimits
from cascade.types import SafetyViolation, SkillError


def _race_end_motion(harness, monkeypatch):
    """A concurrent motion on this harness ends right before this move's FIRST approve (two motions
    on one SafeArm: what the MCP teardown park beside a running motion could do)."""
    real = harness.approve
    calls = []

    def approve(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            harness.end_motion()
        return real(*a, **k)

    monkeypatch.setattr(harness, "approve", approve)
    return lambda: monkeypatch.setattr(harness, "approve", real)


# ── 1. the approved-command counter ───────────────────────────────────────


def _harness():
    h = SafetyHarness(SafetyLimits(workspace_min=np.full(3, -1.0), workspace_max=np.ones(3)))
    h.heartbeat()
    return h


def test_every_approved_command_counts_and_nothing_else_does():
    """`approved_commands` grows by one per command approve() clears, whatever the motion state:
    a refused approve, a vetted (preflight) step and begin/end_motion count nothing."""
    h = _harness()
    q0, q1 = np.zeros(6), np.full(6, 0.001)
    assert h.approved_commands == 0
    h.begin_motion()
    h.approve(q0, q1, 0.02)
    h.end_motion()                      # a concurrent motion ends: the next approves still count
    h.approve(q1, q0, 0.02)
    h.approve(q0, q1, 0.02)
    assert h.approved_commands == 3 and h.motions_begun == 1
    assert h.vet_step(q0, q1, 0.02) is None
    h.estop("operator e-stop")
    with pytest.raises(SafetyViolation, match="e-stop latched"):
        h.approve(q1, q0, 0.02)
    assert h.approved_commands == 3


def test_a_concurrent_end_motion_cannot_hide_a_started_home_move(make_runtime, monkeypatch):  # noqa: F811
    """The review's race_end_motion row: this move's begin_motion, then a concurrent motion's
    end_motion, then this move's commands, then a fault at the 6th send. 5 commands went out, so the
    arm stops and latches (main: raised as a refusal, no stop, no latch, arm left mid-route)."""
    rt = _mock(make_runtime)
    jaws = _jaw_counter(rt)
    raw = raw_arm(rt.arm)
    restore = []

    def before():
        restore.append(_race_end_motion(rt.arm.harness, monkeypatch))
        restore.append(_fail_send_after(raw, monkeypatch, 5, SafetyViolation("joint 3 tracking fault")))

    _after_first(rt, monkeypatch, "_move_planned", before=before,
                 after=lambda: [undo() for undo in reversed(restore)])
    out, sent = _call(rt, "grasp_at_pixel", PIXEL)
    assert out["error"] == ("SkillError: home move failed before pregrasp: joint 3 tracking fault; "
                            "e-stop latched"), out
    assert sent == {name: 5 for name in sent} and jaws == {name: 1 for name in jaws}
    assert _estopped(rt) == {name: True for name in sent}


def test_a_concurrent_end_motion_cannot_hide_a_started_view_home_return(make_runtime, monkeypatch):  # noqa: F811
    """Same race on fr3's view-home return: started (5 commands), then a fault: "failed ...; e-stop
    latched" (main: "skipped", the arm left where the fault stopped it, no latch)."""
    rt = _fr3(make_runtime, monkeypatch)
    raw = next(raw_arm(s) for s in rt.arm_rig.arms.values())
    restore = []

    def before():
        restore.append(_race_end_motion(rt.arm.harness, monkeypatch))
        restore.append(_fail_send_after(raw, monkeypatch, 5, SafetyViolation("joint 3 tracking fault")))

    _after_first(rt, monkeypatch, "_move_carry", before=before,
                 after=lambda: [undo() for undo in reversed(restore)])
    out, _sent = _call(rt, *MOVE)
    home = out["return_home"]
    assert home == {"ok": False, "status": "failed",
                    "error": "SafetyViolation: joint 3 tracking fault; e-stop latched"}, home
    assert _estopped(rt) == {name: True for name in _estopped(rt)}


def test_a_harness_double_without_the_counter_never_reads_as_refused_before_motion():
    """An unknown count is never "before" (the conservative side: stop and latch)."""
    from types import SimpleNamespace

    from cascade.skills.runtime import SkillRuntime

    rt = SimpleNamespace(arm=SimpleNamespace(harness=SimpleNamespace(motions_begun=0)))
    assert SkillRuntime._approved_commands(rt) is None
    assert SkillRuntime._refused_before_motion(rt, SafetyViolation("refused"), None) is False


# ── 2. the retry loops name a latched e-stop once ─────────────────────────


@pytest.mark.parametrize("loop", ["persistence", "pick_and_place"])
def test_a_latched_home_move_failure_names_the_latch_once_in_the_retry_loop(make_runtime, monkeypatch, loop):  # noqa: F811
    """#422 review Minor 2: "...: joint 3 tracking fault; e-stop latched (e-stop latched; not
    retrying)" -> "...: joint 3 tracking fault; e-stop latched (not retrying)"."""
    monkeypatch.delenv("CASCADE_OBSERVED_FINGER_GATE", raising=False)
    rt = _mock(make_runtime)
    raw = raw_arm(rt.arm)
    restore = []
    _after_first(rt, monkeypatch, "_move_planned",
                 before=lambda: restore.append(_fail_send_after(raw, monkeypatch, 5,
                                                                SafetyViolation("joint 3 tracking fault"))),
                 after=lambda: restore.pop()())
    if loop == "persistence":
        out = rt._grasp_with_persistence("red cube")
    else:
        out = rt.execute("pick_and_place", {"object": "red cube", "destination": "green square"})
    error = str(out["error"])
    assert error.count("e-stop latched") == 1, error
    assert error.endswith("home move failed before pregrasp: joint 3 tracking fault; e-stop latched "
                          "(not retrying)"), error
    assert out["ok"] is False and out.get("grasp_attempts", 1) == 1, out
    assert _estopped(rt) == {name: True for name in _estopped(rt)}


def test_the_stop_text_keeps_a_latch_the_error_does_not_name_yet():
    from cascade.skills.runtime import _with_stop_reason

    assert _with_stop_reason("SkillError: x", "e-stop latched; not retrying") == \
        "SkillError: x (e-stop latched; not retrying)"
    assert _with_stop_reason("SkillError: x; e-stop latched", "e-stop latched; not retrying") == \
        "SkillError: x; e-stop latched (not retrying)"
    assert _with_stop_reason("SafetyViolation: e-stop latched", "contact detected: y") == \
        "SafetyViolation: e-stop latched (contact detected: y)"


# ── 3. decision: a non-watchdog refusal before the home move retries ─────


def test_pick_and_place_retries_after_a_route_refusal_before_the_home_move(make_runtime, monkeypatch):  # noqa: F811
    """Kept from #422 (named in the PR): the first attempt's home move is refused by its route
    preflight before any command (a workspace refusal): nothing moved, nothing latches, so the loop
    re-homes and grasps again; every motion of the retry goes through the same vetting."""
    from cascade.safety import trajectory

    monkeypatch.delenv("CASCADE_OBSERVED_FINGER_GATE", raising=False)
    rt = _mock(make_runtime)
    real_plan = trajectory.plan_route
    refused = []

    def refuse_once(*a, **k):
        if not refused:
            refused.append(1)
            raise SafetyViolation("TCP [0.61, 0.0, 0.2] outside workspace")
        return real_plan(*a, **k)

    monkeypatch.setattr(trajectory, "plan_route", refuse_once)
    vetted = []
    real_vet = rt.arm.harness.vet_step
    monkeypatch.setattr(rt.arm.harness, "vet_step", lambda *a, **k: vetted.append(1) or real_vet(*a, **k))
    out, sent = _call(rt, "pick_and_place", {"object": "red cube", "destination": "green square"})
    assert refused == [1]
    assert out["ok"] is True, out
    assert all(n > 0 for n in sent.values()) and vetted
    assert _estopped(rt) == {name: False for name in sent}


# ── 4. the home route is checked before the pre-grasp jaw opening ────────


@pytest.mark.parametrize("refusal", [SkillError("no safe joint route found; direct route: keep-out box 'monitor' entered"),
                                     SafetyViolation("TCP [0.61, 0.0, 0.2] outside workspace")],
                         ids=["route", "workspace"])
def test_a_home_route_refusal_costs_no_jaw_command(make_runtime, monkeypatch, refusal):  # noqa: F811
    """#422 review Minor 5: the home move's route preflight refuses (no safe route / workspace): the
    jaws stay as they were too (main: 1 jaw command, the opening), no latch, and the next grasp
    grasps once the route is clear again."""
    from cascade.safety import trajectory

    rt = _mock(make_runtime)
    jaws = _jaw_counter(rt)
    real_plan = trajectory.plan_route

    def refuse(*_a, **_k):
        raise refusal

    monkeypatch.setattr(trajectory, "plan_route", refuse)
    out, sent = _call(rt, "grasp_at_pixel", PIXEL)
    assert out["error"] == f"{type(refusal).__name__}: {refusal}", out
    assert sent == {name: 0 for name in sent} and jaws == {name: 0 for name in jaws}
    assert _estopped(rt) == {name: False for name in sent}
    monkeypatch.setattr(trajectory, "plan_route", real_plan)
    out, sent = _call(rt, "grasp_at_pixel", PIXEL)
    assert out["ok"] is True, out
    assert all(n > 0 for n in jaws.values()) and all(n > 0 for n in sent.values()), (jaws, sent)


def test_the_home_route_is_vetted_before_the_jaw_opening_and_again_at_the_move(make_runtime, monkeypatch):  # noqa: F811
    """Order on a live run: the route check (a plan, no command) precedes the jaw opening, and the
    home move still plans its own route from the measured start (the preflight never authorizes it)."""
    from cascade.safety import trajectory

    rt = _mock(make_runtime)
    events = []
    real_plan = trajectory.plan_route
    raw = raw_arm(rt.arm)
    real_jaw = raw.set_gripper
    monkeypatch.setattr(trajectory, "plan_route", lambda *a, **k: events.append("plan") or real_plan(*a, **k))
    monkeypatch.setattr(raw, "set_gripper", lambda *a, **k: events.append("jaw") or real_jaw(*a, **k))
    out, _sent = _call(rt, "grasp_at_pixel", PIXEL)
    assert out["ok"] is True, out
    assert events[:3] == ["plan", "jaw", "plan"], events


def test_a_backend_fault_in_the_home_route_check_still_latches_and_sends_no_jaw_command(make_runtime, monkeypatch):  # noqa: F811
    """The route check reads the joints (as the home move did first): a feedback fault there is a
    backend fault, not a refusal, so it keeps #422's rule (stop, latch, say so), now before the jaw
    opening instead of after it."""
    rt = _mock(make_runtime)
    jaws = _jaw_counter(rt)
    raw = raw_arm(rt.arm)
    real_state, real_check = raw.get_state, rt.arm.preflight_planned

    def broken(*_a, **_k):
        raise RuntimeError("joint feedback read failed")

    def check(*a, **k):
        monkeypatch.setattr(raw, "get_state", broken)
        try:
            return real_check(*a, **k)
        finally:
            monkeypatch.setattr(raw, "get_state", real_state)

    monkeypatch.setattr(rt.arm, "preflight_planned", check)
    out, sent = _call(rt, "grasp_at_pixel", PIXEL)
    assert out["error"] == ("SkillError: home move failed before pregrasp: joint feedback read failed; "
                            "e-stop latched"), out
    assert sent == {name: 0 for name in sent} and jaws == {name: 0 for name in jaws}
    assert _estopped(rt) == {name: True for name in sent}
