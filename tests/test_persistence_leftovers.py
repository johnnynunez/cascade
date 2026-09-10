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
