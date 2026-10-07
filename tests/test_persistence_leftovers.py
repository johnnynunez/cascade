"""Persistence-loop leftovers (ROADMAP near-term #2..#7), each driven through
the REAL runtime on the mock stack and each proven by mutation.

  #2  provisional held marker: a physically held object is never logically
      unheld after an exception between close and `held_object = label`
  #3  the persistence budget does not multiply across tiers
  #4  handover / sort_by_color persist like pick_and_place
  #5  a legitimately held THIN object is not mistaken for a slip
  #6  every-candidate-too-wide fails fast instead of burning the budget
  #7  coverage: deadline expiry, epoch fallback, exemption z_min,
      place z-cap (the gaps the 2026-07-18 review listed)
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.apps.demo import build_runtime, shutdown_runtime
from cascade.config import load_demo_config
from cascade.types import SkillError


@pytest.fixture
def rt(tmp_path):
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    runtime, arm = build_runtime(cfg, tmp_path / "run", view=False, serve=False)
    arm.object_stop_frac = 0.5  # the mock jaws jam halfway: an object is in the gripper
    try:
        yield runtime
    finally:
        shutdown_runtime(runtime, arm)


def _jaws(rt, width_frac: float):
    """Pin the reported jaw opening (0 closed .. 1 open) on the mock arm."""
    span = rt._grip_closed - rt._grip_open
    pos = rt._grip_open + (1.0 - width_frac) * span
    st = rt.arm.get_state()
    rt.arm.get_state = lambda st=st, pos=pos: SimpleNamespace(
        q=st.q, gripper_pos=pos, gripper_valid=True, t=getattr(st, "t", 0.0))


# ── #2 provisional held marker ──────────────────────────────────────────────

def test_exception_after_close_leaves_a_provisional_marker_that_reconcile_promotes(rt):
    """Simulate: close succeeded (jaws stalled open on the object), then the
    lift raised before `held_object` was assigned. The next skill's
    `_reconcile_held` must PROMOTE the marker, not open the jaws on it."""
    rt.held_object = None
    rt._held_provisional = ("red cube", "cube", "red")
    _jaws(rt, 0.45)  # stalled on a ~45 % object: clearly not air
    rt._reconcile_held()
    assert rt.held_object == "red cube"
    assert rt._held_det_label == "cube" and rt._held_color == "red"
    assert rt._held_provisional is None
    with pytest.raises(SkillError, match="already holding"):
        rt.skill_grasp_object("blue cube")


def test_provisional_marker_on_air_is_dropped_not_promoted(rt):
    rt.held_object = None
    rt._held_provisional = ("red cube", "cube", "red")
    _jaws(rt, 0.01)  # jaws fully closed: air
    rt._reconcile_held()
    assert rt.held_object is None and rt._held_provisional is None


def test_grasp_object_sets_and_clears_the_marker_around_the_close(rt, monkeypatch):
    """Drive the real skill: the marker must exist DURING the close and be
    cleared once `held_object` is assigned (happy path) -- a marker that
    survives a successful grasp would later be 'promoted' over a new one."""
    seen = {}
    real_close = rt._close_two_stage

    def spy(profile):
        seen["marker_during_close"] = getattr(rt, "_held_provisional", None)
        return real_close(profile)

    monkeypatch.setattr(rt, "_close_two_stage", spy)
    rt.observe()
    res = rt.skill_grasp_object("red cube")
    assert res.get("ok", True) and res["held"], res
    assert seen["marker_during_close"] is not None and seen["marker_during_close"][0] == "red cube"
    assert rt._held_provisional is None


# ── #5 thin objects ────────────────────────────────────────────────────────

def test_thin_object_declared_in_profile_is_not_read_as_a_slip(rt):
    rt.held_object = "card"
    rt._held_det_label = "card"
    _jaws(rt, 0.02)  # 2 % of travel: below air_grasp_frac (4 %)
    # without the profile key the heuristic clears it (a slip)
    rt._reconcile_held()
    assert rt.held_object is None
    # with `gripper.min_object_m` declared, 2 % of the jaw span is a real hold
    rt.held_object = "card"
    g = rt.cfg.arm.get("gripper").as_dict()
    g["min_object_m"] = 0.02 * rt._max_width * 0.5  # thinner than what the jaws report
    rt.cfg.arm._data["gripper"] = g
    _jaws(rt, 0.02)
    rt._reconcile_held()
    assert rt.held_object == "card"


# ── #6 fail fast on over-width ─────────────────────────────────────────────

def test_pick_and_place_gives_up_early_when_every_grasp_is_too_wide(rt, monkeypatch):
    calls = {"n": 0}

    def too_wide(*a, **k):
        calls["n"] += 1
        raise SkillError("no executable grasp: cube: required width 120mm > gripper max 55mm (consider push or regrasp)")

    monkeypatch.setattr(rt, "skill_grasp_object", too_wide)
    rt.cfg.grasp._data["persist_seconds"] = 30.0
    rt.cfg.grasp._data["max_pick_attempts"] = 8
    t0 = time.monotonic()
    res = rt.skill_pick_and_place("red cube")
    assert res["ok"] is False and "wider than the jaws" in res["error"], res
    assert calls["n"] == 1, calls  # ONE attempt, not eight
    assert time.monotonic() - t0 < 10.0


def test_ik_failure_alongside_a_width_reason_is_still_retried(rt, monkeypatch):
    """Only 'every candidate too wide' is terminal. A mixed reason list
    (some too wide, some IK failures) can still be cured by a re-scan."""
    calls = {"n": 0}

    def mixed(*a, **k):
        calls["n"] += 1
        raise SkillError("no executable grasp: cube: required width 60mm > gripper max 55mm; pregrasp IK failed (err 0.02)")

    monkeypatch.setattr(rt, "skill_grasp_object", mixed)
    monkeypatch.setattr(rt, "skill_move_home", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(rt, "_reobserve", lambda *a, **k: None)
    rt.cfg.grasp._data["persist_seconds"] = 30.0
    rt.cfg.grasp._data["max_pick_attempts"] = 3
    res = rt.skill_pick_and_place("red cube")
    assert res["ok"] is False and calls["n"] == 3


# ── #3 task budget across tiers ────────────────────────────────────────────

def test_task_budget_caps_a_second_tier_call(rt, monkeypatch):
    """Tier 1 spends the task budget; tier 2's pick_and_place on the same
    object must NOT get a fresh `persist_seconds`."""
    monkeypatch.setattr(rt, "skill_grasp_object", lambda *a, **k: (_ for _ in ()).throw(SkillError("no detections")))
    monkeypatch.setattr(rt, "skill_move_home", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(rt, "_reobserve", lambda *a, **k: None)
    rt.begin_task_budget(seconds=0.0)  # the task's budget is already gone
    try:
        res = rt.skill_pick_and_place("red cube")
    finally:
        rt.end_task_budget()
    assert res["ok"] is False and "task persistence budget exhausted" in res["error"], res
    # and with no task epoch the per-call budget applies as before
    rt.cfg.grasp._data["max_pick_attempts"] = 1
    res2 = rt.skill_pick_and_place("red cube")
    assert "task persistence budget" not in res2.get("error", "")


def test_orchestrator_opens_and_closes_the_task_budget(rt):
    from cascade.agent.llm import LLMResponse, MockLLM
    from cascade.agent.orchestrator import AgentOrchestrator

    seen = {}
    real_begin = rt.begin_task_budget

    def spy_begin(seconds=None):
        real_begin(seconds)
        seen["deadline_open"] = rt._task_deadline is not None

    rt.begin_task_budget = spy_begin
    orch = AgentOrchestrator(MockLLM([LLMResponse(text="nothing to do")]), rt, max_steps=2)
    orch.run_task("describe the scene")
    assert seen.get("deadline_open") is True
    assert getattr(rt, "_task_deadline", None) is None  # closed after the task


# ── #4 handover / sort persist ─────────────────────────────────────────────

def test_handover_retries_a_grasp_like_pick_and_place(rt, monkeypatch):
    attempts = {"n": 0}
    real = rt.skill_grasp_object

    def flaky(label, material=None, **k):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise SkillError("did not settle at grasp pose")
        return real(label, material=material)

    monkeypatch.setattr(rt, "skill_grasp_object", flaky)
    monkeypatch.setattr(rt, "_reobserve", lambda *a, **k: None)
    rt.cfg.grasp._data["max_pick_attempts"] = 5
    rt.observe()
    res = rt.skill_handover("red cube")
    assert res.get("ok", True) and res.get("offering") == "red cube", res
    assert attempts["n"] == 3


def test_sort_by_color_does_not_give_up_on_the_first_miss(rt, monkeypatch):
    attempts = {"n": 0}
    real = rt.skill_grasp_object

    def flaky(label, material=None, **k):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise SkillError("did not settle at grasp pose")
        return real(label, material=material)

    monkeypatch.setattr(rt, "skill_grasp_object", flaky)
    monkeypatch.setattr(rt, "_reobserve", lambda *a, **k: None)
    rt.observe()
    res = rt.skill_sort_by_color()
    assert res["ok"], res
    assert attempts["n"] >= 2 and len(res.get("moved", [])) >= 1, res


# ── #7 coverage gaps ───────────────────────────────────────────────────────

def test_persistence_deadline_expiry_stops_the_loop_early(rt, monkeypatch):
    calls = {"n": 0}

    def slow_miss(*a, **k):
        calls["n"] += 1
        time.sleep(0.15)
        raise SkillError("did not settle at grasp pose")

    monkeypatch.setattr(rt, "skill_grasp_object", slow_miss)
    monkeypatch.setattr(rt, "skill_move_home", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(rt, "_reobserve", lambda *a, **k: None)
    rt.cfg.grasp._data["persist_seconds"] = 0.2
    rt.cfg.grasp._data["max_pick_attempts"] = 50
    res = rt.skill_pick_and_place("red cube")
    assert res["ok"] is False
    assert 1 <= calls["n"] <= 3, calls  # the deadline, not the attempt cap, ended it


def test_localize_epoch_fallback_uses_motion_start_when_no_rescan(rt):
    """During a motion (watcher paused) the staleness reference is the motion
    epoch, not `now`: a belief from just before the motion stays usable."""
    rt.observe()
    b = rt.beliefs.find("red cube") or rt.beliefs.find("object")
    assert b is not None
    rt._last_reobserve_t = None
    rt._motion_t0 = b.last_seen_t + 0.5  # motion began half a second after the sighting
    rt.cfg._data.setdefault("perception_loop", {})["belief_fallback_age_s"] = 1.0
    # make the live camera useless so only the memory path can answer
    rt.detector.detect = lambda *a, **k: []
    _, fix = rt._localize(b.label if not b.color else f"{b.color} {b.label}")
    assert fix is not None
    # with the epoch far in the future the same belief is stale and refused
    rt._motion_t0 = b.last_seen_t + 50.0
    with pytest.raises(SkillError):
        rt._localize(b.label if not b.color else f"{b.color} {b.label}")


def test_exemption_cylinder_z_min_is_honoured_by_in_cylinder():
    from cascade.safety.harness import SafetyHarness

    exempt = (np.array([0.2, 0.0]), 0.05, -0.06)
    assert SafetyHarness._in_cylinder(np.array([0.21, 0.01, -0.05]), exempt)
    assert not SafetyHarness._in_cylinder(np.array([0.21, 0.01, -0.07]), exempt)  # below z_min
    assert not SafetyHarness._in_cylinder(np.array([0.30, 0.01, 0.0]), exempt)   # outside radius
    assert not SafetyHarness._in_cylinder(np.array([0.2, 0.0, 0.0]), None)


def test_place_at_caps_release_height_to_the_topdown_ceiling(rt, monkeypatch):
    rt.observe()
    assert rt.skill_grasp_object("red cube")["held"]
    seen = {}
    real_ik = rt.kin.ik

    def spy_ik(T, q0, *a, **k):
        seen.setdefault("z", []).append(float(np.asarray(T)[2, 3]))
        return real_ik(T, q0, *a, **k)

    monkeypatch.setattr(rt.kin, "ik", spy_ik)
    rt.cfg.grasp._data["topdown_z_max"] = 0.12
    rt.skill_place_at(0.20, -0.10, z=0.50)  # absurdly high request
    assert seen["z"] and max(seen["z"]) <= 0.12 + 1e-6, seen


# ── #2 (finish): the marker against OPEN jaws ───────────────────────────────

def test_open_gripper_discards_a_provisional_marker(rt):
    """A deliberate `open_gripper` after a crashed grasp is the human saying
    "nothing is held": the marker must not survive it, or the next skill's
    `_reconcile_held` reads the OPEN jaws (1.0 >= air_grasp_frac) as a stall
    and promotes a phantom hold that refuses every later grasp."""
    rt.held_object = None
    rt._held_provisional = ("red cube", "cube", "red")
    rt.skill_open_gripper()
    assert rt._held_provisional is None
    rt._reconcile_held()
    assert rt.held_object is None


def test_provisional_marker_with_jaws_at_the_open_position_is_dropped(rt):
    """The close never landed (refused before its first stage) or the jaws
    were opened since: jaws AT the open position cannot be stalled on
    anything, so the marker is refuted, never promoted."""
    rt.held_object = None
    rt._held_provisional = ("red cube", "cube", "red")
    _jaws(rt, 1.0)
    rt._reconcile_held()
    assert rt.held_object is None and rt._held_provisional is None
    # a wide object that stalls the jaws just inside the air tolerance of
    # fully open is still a hold: never refute what the jaws may be holding
    rt._held_provisional = ("red cube", "cube", "red")
    _jaws(rt, 0.95)
    rt._reconcile_held()
    assert rt.held_object == "red cube"


# ── #5 (finish): the MEASURED held width decides, not a fixed fraction ─────

def test_grasp_records_the_measured_held_width(rt):
    """The width the jaws stalled at after the lift is the known held width;
    it, not a profile constant, decides what a later jaw reading means."""
    rt.observe()
    res = rt.skill_grasp_object("red cube")
    assert res["held"]
    assert rt._held_width_m == pytest.approx(0.5 * rt._max_width, abs=1e-6)
    # the promotion path measures too
    rt.held_object = None
    rt._held_width_m = None
    rt._held_provisional = ("red cube", "cube", "red")
    _jaws(rt, 0.45)
    rt._reconcile_held()
    assert rt.held_object == "red cube"
    assert rt._held_width_m == pytest.approx(0.45 * rt._max_width, abs=1e-6)


def test_known_thin_object_is_never_read_as_a_slip_on_width_alone(rt):
    """A card measured at 2 % of the jaw span when it was grasped stalls the
    jaws BELOW `air_grasp_frac`; position feedback cannot tell that hold from
    air, so width alone must never clear it. A chunky known width keeps the
    slip rule exactly as it was (jaws fully closed -> it slipped)."""
    rt.held_object = "card"
    rt._held_det_label = "card"
    rt._held_width_m = 0.02 * rt._max_width  # measured at grasp: thin
    _jaws(rt, 0.0)  # the jaws now read fully closed
    rt._reconcile_held()
    assert rt.held_object == "card"
    rt.held_object = "red cube"
    rt._held_width_m = 0.5 * rt._max_width
    _jaws(rt, 0.0)
    rt._reconcile_held()
    assert rt.held_object is None and rt._held_width_m is None


# ── #6 (finish): an explicit, structured over-width refusal ─────────────────

def _wide_grasp(width_m, quality, label):
    from cascade.types import Grasp
    return Grasp(position=np.array([0.3, 0.0, 0.05]), rotation=np.eye(3), width_m=width_m,
                 approach=np.array([0.0, 0.0, -1.0]), quality=quality, label=label)


def test_selector_refuses_explicitly_when_every_candidate_exceeds_the_jaw_span():
    """The refusal names the narrowest candidate against the jaw span and is
    structured (`all_too_wide`), so persistence stops on a fact instead of
    parsing a truncated reason list."""
    from test_persistence_loop import StubKin
    from cascade.grasping.selector import NoExecutableGrasp, select_grasp

    wide = [_wide_grasp(0.082, 0.9, "a"), _wide_grasp(0.070, 0.5, "b")]
    with pytest.raises(NoExecutableGrasp) as info:
        select_grasp(wide, StubKin(), np.zeros(6), max_width_m=0.055)
    exc = info.value
    assert exc.all_too_wide is True
    assert exc.narrowest_width_m == pytest.approx(0.070)
    assert exc.jaw_max_width_m == pytest.approx(0.055)
    assert "every candidate exceeds the jaw span" in str(exc)
    assert "70mm" in str(exc) and "55mm" in str(exc)


def test_a_truncated_mixed_reason_list_is_not_an_over_width_refusal(rt):
    """Five too-wide candidates ranked ahead of one vetoed candidate used to
    yield a four-reason message of width reasons only, which read as 'every
    candidate too wide' and ended persistence after ONE attempt. Non-width
    reasons are listed first and the structured flag is False."""
    from test_persistence_loop import StubKin
    from cascade.grasping.selector import NoExecutableGrasp, select_grasp

    grasps = [_wide_grasp(0.070, 0.9 - 0.1 * i, f"wide{i}") for i in range(5)]
    grasps.append(_wide_grasp(0.030, 0.1, "narrow"))  # ranked last
    veto = lambda g, qp, qg: "pregrasp unsafe: elbow would hit the table" if g.label == "narrow" else None
    with pytest.raises(NoExecutableGrasp) as info:
        select_grasp(grasps, StubKin(), np.zeros(6), max_width_m=0.055, validate=veto)
    exc = info.value
    assert exc.all_too_wide is False
    assert "pregrasp unsafe" in str(exc)
    assert rt._grasp_retry_verdict("cube", 1, f"SkillError: {exc}") is None
    # and the structured refusal itself is terminal for persistence
    with pytest.raises(NoExecutableGrasp) as info2:
        select_grasp(grasps[:5], StubKin(), np.zeros(6), max_width_m=0.055)
    assert "wider than the jaws" in rt._grasp_retry_verdict("cube", 1, f"SkillError: {info2.value}")


# ── #7 (finish): place-stage loop, per-task cap on handover, McpClient timeout

def test_place_stage_retries_after_a_failed_place(rt, monkeypatch):
    real_place = rt.skill_place_at
    calls = {"place": 0, "reobserve": 0}

    def flaky_place(*a, **k):
        calls["place"] += 1
        if calls["place"] == 1:
            raise SkillError("did not settle at the place pose")
        return real_place(*a, **k)

    monkeypatch.setattr(rt, "skill_place_at", flaky_place)
    monkeypatch.setattr(rt, "_reobserve",
                        lambda *a, **k: calls.__setitem__("reobserve", calls["reobserve"] + 1))
    rt.observe()
    res = rt.skill_pick_and_place("red cube")
    assert res.get("ok", True), res
    assert res["grasp_attempts"] == 1 and res["place_attempts"] == 2
    assert calls["reobserve"] >= 1  # re-home + re-scan between place attempts
    notes = [e.text for e in rt.memory.events()]
    assert any("place attempt 1 failed" in n for n in notes)
    assert any("still holding" in n and "place attempt 2" in n for n in notes)
    assert rt.held_object is None


def test_place_stage_is_bounded_while_still_holding(rt, monkeypatch):
    calls = {"n": 0}

    def never(*a, **k):
        calls["n"] += 1
        raise SkillError("did not settle at the place pose")

    monkeypatch.setattr(rt, "skill_place_at", never)
    monkeypatch.setattr(rt, "_reobserve", lambda *a, **k: None)
    rt.cfg.grasp._data["max_pick_attempts"] = 3
    rt.observe()
    res = rt.skill_pick_and_place("red cube")
    assert res["ok"] is False and res["stage"] == "place", res
    assert calls["n"] == 3 and "after 3 attempts" in res["error"]
    assert "still holding" in res["note"]
    assert rt.held_object == "red cube"  # nothing released, no regrasp attempted


def test_mid_carry_slip_restarts_the_grasp_stage_within_the_budget(rt, monkeypatch):
    real_place = rt.skill_place_at
    calls = {"n": 0}

    def slip_then_place(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            rt.held_object = None  # the carry's slip check cleared the flag
            rt._held_det_label = None
            raise SkillError("'red cube' slipped out of the gripper during the carry")
        return real_place(*a, **k)

    monkeypatch.setattr(rt, "skill_place_at", slip_then_place)
    monkeypatch.setattr(rt, "_reobserve", lambda *a, **k: None)
    rt.observe()
    res = rt.skill_pick_and_place("red cube")
    assert res.get("ok", True), res
    assert res["grasp_attempts"] == 2 and res["place_attempts"] == 2
    notes = [e.text for e in rt.memory.events()]
    assert any("slipped while I was carrying it -- starting over" in n for n in notes)


def test_handover_persistence_is_capped_by_the_task_budget(rt, monkeypatch):
    """#3 x #4: the task epoch bounds handover's loop exactly as it bounds
    pick_and_place's (pin; the cap landed with `_grasp_with_persistence`)."""
    calls = {"n": 0}

    def miss(*a, **k):
        calls["n"] += 1
        raise SkillError("did not settle at grasp pose")

    monkeypatch.setattr(rt, "skill_grasp_object", miss)
    monkeypatch.setattr(rt, "skill_move_home", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(rt, "_reobserve", lambda *a, **k: None)
    rt.cfg.grasp._data["max_pick_attempts"] = 8
    rt.begin_task_budget(seconds=0.0)
    try:
        res = rt.skill_handover("red cube")
    finally:
        rt.end_task_budget()
    assert res["ok"] is False and calls["n"] == 1, res
    assert "persistence budget exhausted" in res["error"]


def test_mcp_client_recv_times_out_instead_of_hanging():
    """`McpClient.recv(timeout=)` was flagged as dead code by the 2026-07-18
    review; it is live (a queue-pumped stdout) -- exercise the timeout path
    without a server: an empty queue must fail the test, not block it."""
    import queue

    from test_mcp_server import McpClient

    c = McpClient.__new__(McpClient)
    c._out_q = queue.Queue()
    c.proc = SimpleNamespace(pid=0)
    t0 = time.monotonic()
    with pytest.raises(pytest.fail.Exception, match="no response within 0.05s"):
        c.recv(timeout=0.05)
    assert time.monotonic() - t0 < 5.0
    c._out_q.put('{"jsonrpc": "2.0", "id": 1, "result": {}}\n')
    assert c.recv(timeout=0.05)["id"] == 1
