"""ASPIRE A6: the sim-only retry ledger (spec §4.5; preflight R6, R9, R10, R28).

Pure bookkeeping, no runtime: after a qualifying refusal the ledger opens ONE
slot per goal; the next same-goal call is THE retry (the slot is used before
anything moves); every later same-goal call is refused before motion until a
clear event or the slot's window ends. A halt or e-stop never opens, uses or
re-arms a slot. Only a sim profile with the switch on ever gets a ledger
(`ledger_for`); real and unknown profiles get None. The ledger never edits
agent text (A7 composes `next_action`): it only adds the `sim_retry` field and
returns the precheck refusal.
"""
from __future__ import annotations

import math
import runpy
import threading
import time
from pathlib import Path

import pytest

from cascade.agent import sim_retry
from cascade.agent.sim_retry import (
    DEFAULT_RESERVE_S,
    DEFAULT_WINDOW_S,
    REFUSAL_CLASSES,
    RETRY_ALLOWED,
    RETRY_EXHAUSTED,
    RETRY_SKILLS,
    RETRY_USED,
    TURN_BUDGET_S,
    Goal,
    RetryCall,
    SimRetryLedger,
    goal_of,
    ledger_for,
    refusal_class,
    same_goal,
    same_object,
)
from cascade.config import load_demo_config
from cascade.sim_profile import profile_class

REPO = Path(__file__).resolve().parents[1]
A = {"object": "green cube"}
G1, G2 = ("grasp-1",), ("grasp-2",)
REFUSED = {"ok": False, "error": "SafetyViolation: the finger would enter the lemon", "arm_left_in_place": True}
#: finish(): nothing held, no latch, no halt during the call, no pending episode
CLEAR = {"held": False, "latched": False, "halted": False, "episode_pending": False}
#: begin(): the rig may start a retry
READY = {"held": False, "latched": False, "episode_pending": False, "motion_blocked": False}


class Clock:
    """A monotonic clock the test moves by hand."""

    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def ledger(clock):
    """A ledger whose bookkeeping must never fail: a swallowed internal error
    (fail closed, every slot dropped) fails the test that caused it."""
    L = SimRetryLedger(clock=clock)
    faults = []
    fail_closed = L._fail_closed

    def recording(where):
        faults.append(where)
        fail_closed(where)
    L._fail_closed = recording
    yield L
    assert faults == [], f"bookkeeping failed in {faults}"


def _begin(L, name="pick_and_place", args=A, **ready):
    return L.begin(name, args, **{**READY, **ready})


def _finish(L, name="pick_and_place", args=A, result=None, *, exc="SafetyViolation", grasps=(G1,), retry=None,
            **state):
    return L.finish(name, args, dict(REFUSED if result is None else result), exc_type=exc,
                    refused_grasps=grasps, retry=retry, **{**CLEAR, **state})


def _refused_first(L, clock, name="pick_and_place", args=A, took=60.0, **kw):
    """One first attempt: begin (not a retry), `took` seconds, then finish."""
    assert _begin(L, name, args) is None
    clock.advance(took)
    return _finish(L, name, args, **kw)


# ── goals ────────────────────────────────────────────────────────────────

def test_retry_skills_are_the_two_grasping_orders():
    assert RETRY_SKILLS == frozenset({"pick_and_place", "grasp_object"})


def test_goal_is_object_words_destination_and_arm_normalized():
    assert goal_of("pick_and_place", {"object": "  Green\tCUBE "}) == Goal("green cube", "", "")
    assert goal_of("pick_and_place", {"object": "cube", "destination": " Red  Plate", "arm": "Left"}) == Goal(
        "cube", "red plate", "left")
    # grasp_object has no destination: a stray one never splits its goal
    assert goal_of("grasp_object", {"label": "Lemon", "arm": "right", "destination": "x"}) == Goal("lemon", "", "right")
    assert str(goal_of("pick_and_place", {"object": "Green  Cube", "destination": "plate"})) == "green cube"


@pytest.mark.parametrize("args", [{}, {"object": ""}, {"object": "   "}, {"object": 3}, {"object": None},
                                  {"object": "cube", "destination": 3}, {"object": "cube", "arm": 0},
                                  {"label": "cube"}, None, "green cube"])
def test_no_goal_without_a_readable_object(args):
    assert goal_of("pick_and_place", args) is None


@pytest.mark.parametrize("name", ["place_at", "move_home", "reset_scene", "grasp_at_pixel", "handover", None, 3])
def test_other_skills_have_no_goal(name):
    assert goal_of(name, {"object": "green cube", "label": "green cube"}) is None


def test_renamed_object_is_the_same_goal(ledger, clock):  # Review Focus 1
    assert same_object("cube", "green cube") and same_object("Green  Cube", "the green cube")
    assert not same_object("pink cube", "green cube") and not same_object("cube", "cubes")
    assert not same_object("", "cube") and not same_object(None, "cube")
    assert same_goal(goal_of("grasp_object", {"label": "cube"}), goal_of("pick_and_place", A))
    assert not same_goal(goal_of("pick_and_place", {"object": "pink cube"}), goal_of("pick_and_place", A))
    _refused_first(ledger, clock)
    # the agent renamed the object on its retry: still THE retry, not a fresh first attempt
    r = _begin(ledger, "grasp_object", {"label": "cube"})
    assert r is not None and r.goal == "green cube"
    clock.advance(30)
    _finish(ledger, "grasp_object", {"label": "cube"}, retry=r)
    assert ledger.precheck("grasp_object", {"label": "cube"}) is not None
    assert ledger.precheck("pick_and_place", {"object": "CUBE"})["error"].startswith(
        "The one simulation retry for green cube is used")
    assert ledger.precheck("pick_and_place", {"object": "pink cube"}) is None


def test_changed_destination_or_arm_is_another_goal(ledger, clock):  # R28
    args = {"object": "green cube", "destination": "plate"}
    base = goal_of("pick_and_place", args)
    assert not same_goal(base, goal_of("pick_and_place", {**args, "destination": "bowl"}))
    assert not same_goal(base, goal_of("pick_and_place", {**args, "arm": "right"}))
    assert not same_goal(base, goal_of("grasp_object", {"label": "green cube"}))  # no destination
    assert not same_goal(base, "green cube") and not same_goal(None, None)
    _refused_first(ledger, clock, args=args)
    assert _begin(ledger, args={**args, "destination": "bowl"}) is None  # not the retry ...
    assert _begin(ledger, args=args) is None  # ... and that other goal cleared the slot


# ── refusal classes (R6: by exception type, never by error text) ─────────

@pytest.mark.parametrize("exc,cls", [
    ("SafetyViolation", "safety_refusal"),
    ("SlowPerceptionError", "perception_refusal"),
    ("SceneChangedError", "perception_refusal"),
    ("ReanchorError", "perception_refusal"),
    ("PlanningBudgetExceeded", "planning_refusal"),
    ("NoExecutableGrasp", "planning_refusal"),
])
def test_refusal_classes(exc, cls):
    # pick_and_place returns perception refusals as str(e): no class prefix in the text
    result = {"ok": False, "error": "Perception took 6.1s on a frame that was fresh at capture"}
    assert refusal_class(exc, result, **CLEAR) == cls


def test_refusal_classes_name_the_real_exception_classes():
    from cascade.grasping.planning_budget import PlanningBudgetExceeded
    from cascade.grasping.selector import NoExecutableGrasp
    from cascade.skills.runtime import (
        ReanchorError,
        SceneChangedError,
        SlowPerceptionError,
    )
    from cascade.types import SafetyViolation

    assert REFUSAL_CLASSES == {
        SafetyViolation.__name__: "safety_refusal",
        SlowPerceptionError.__name__: "perception_refusal",
        SceneChangedError.__name__: "perception_refusal",
        ReanchorError.__name__: "perception_refusal",
        PlanningBudgetExceeded.__name__: "planning_refusal",
        NoExecutableGrasp.__name__: "planning_refusal",
    }


def test_safety_violation_subclasses_never_qualify():
    from cascade.skills.carry_attachment import AttachmentInvalid
    from cascade.types import (
        ContactDetected,
        GripperFault,
        MotionHalted,
        SafetyViolation,
    )

    for cls in (MotionHalted, ContactDetected, GripperFault, AttachmentInvalid):
        assert issubclass(cls, SafetyViolation)
        assert refusal_class(cls.__name__, {"ok": False, "error": f"{cls.__name__}: x"}, **CLEAR) is None
        assert refusal_class(SafetyViolation, {"ok": False, "error": "x"}, **CLEAR) is None  # a type, not a name
        assert refusal_class(["SafetyViolation"], {"ok": False, "error": "x"}, **CLEAR) is None


NON_REFUSALS = [
    (REFUSED, "SafetyViolation", {"held": True}),
    (REFUSED, "SafetyViolation", {"latched": True}),
    (REFUSED, "SafetyViolation", {"halted": True}),
    (REFUSED, "SafetyViolation", {"episode_pending": True}),
    ({**REFUSED, "motion_blocked": True}, "SafetyViolation", {}),
    ({**REFUSED, "contact": {"joint": 2}}, "SafetyViolation", {}),
    ({"ok": False, "outcome": "stuck", "error": "SkillStuck: x"}, "SkillStuck", {}),
    ({"ok": False, "outcome": "stuck", "error": "SafetyViolation: x"}, "SafetyViolation", {}),
    ({"ok": False, "action_ok": True, "error": "postcondition failed: ..."}, "SafetyViolation", {}),
    ({"ok": False, "self_reported_ok": True, "error": "postcondition failed: x"}, "SafetyViolation", {}),
    ({"ok": False, "error": "unexpected RuntimeError: x"}, "RuntimeError", {}),
    ({"ok": False, "error": "SafetyViolation: x"}, None, {}),  # text alone never qualifies
    ({"ok": False, "error": "SlowPerceptionError: x"}, "SkillError", {}),
    ({"ok": False, "error": "SkillError: no grasp"}, "SkillError", {}),
    ({"ok": False, "error": "GraspFailed: air"}, "GraspFailed", {}),
    ({"ok": False, "error": "MotionHalted: halt received during route planning or execution"}, "SafetyViolation", {}),
    ({"ok": False, "error": "x; ContactDetected: joint 2"}, "SafetyViolation", {}),
    ({"ok": False, "error": "GripperFault: jaws blocked while opening"}, "SafetyViolation", {}),
    ({"ok": False, "error": "AttachmentInvalid: carry attachment lost"}, "SafetyViolation", {}),
    ({"ok": False, "error": "SafetyViolation: e-stop latched"}, "SafetyViolation", {}),
    ({"ok": False, "error": "SafetyViolation: unfinished contact episode; explicit reset_scene recovery required"},
     "SafetyViolation", {}),
    ({"ok": False, "error": "SafetyViolation: unfinished release episode; explicit reset_scene recovery required"},
     "SafetyViolation", {}),
    ({"ok": True}, "SafetyViolation", {}),
]


@pytest.mark.parametrize("result,exc,state", NON_REFUSALS)
def test_non_refusals_open_no_slot(ledger, clock, result, exc, state):
    assert refusal_class(exc, result, **{**CLEAR, **state}) is None
    out = _refused_first(ledger, clock, result=result, exc=exc, **state)
    assert "sim_retry" not in out and out == result
    assert _begin(ledger) is None  # the next call is a fresh first attempt, not a retry


def test_a_result_that_is_not_a_dict_is_not_a_refusal(ledger, clock):
    assert refusal_class("SafetyViolation", "SafetyViolation: x", **CLEAR) is None
    assert _begin(ledger) is None
    assert ledger.finish("pick_and_place", A, None, exc_type="SafetyViolation", refused_grasps=(), retry=None,
                         **CLEAR) is None
    assert _begin(ledger) is None


# ── the slot ─────────────────────────────────────────────────────────────

def test_first_refusal_opens_one_slot(ledger, clock):
    out = _refused_first(ledger, clock)
    assert out["sim_retry"] == {"allowed": True, "goal": "green cube", "refusal": "safety_refusal"}
    assert {k: v for k, v in out.items() if k != "sim_retry"} == REFUSED  # agent text untouched here
    assert out.get("next_action") == REFUSED.get("next_action")
    assert ledger.last_event is None  # no retry has run yet


def test_finish_returns_the_same_result_object(ledger, clock):
    _begin(ledger)
    result = dict(REFUSED)
    assert ledger.finish("pick_and_place", A, result, exc_type="SafetyViolation", refused_grasps=(G1,),
                         retry=None, **CLEAR) is result


def test_next_same_goal_call_is_the_retry_and_carries_the_exclusions(ledger, clock):
    _refused_first(ledger, clock, grasps=[G1, G2])
    assert ledger.precheck("pick_and_place", A) is None  # the retry itself is not refused
    r = _begin(ledger)
    assert r == RetryCall("green cube", (G1, G2), True, "safety_refusal")
    assert _begin(ledger) is None  # consumed before anything moves


def test_a_failed_retry_is_final_and_a_third_call_is_refused(ledger, clock):
    _refused_first(ledger, clock)
    r = _begin(ledger)
    clock.advance(60)
    out = _finish(ledger, grasps=(G2,), retry=r)
    assert out["sim_retry"] == {"allowed": False, "used": True}
    assert {k: v for k, v in out.items() if k != "sim_retry"} == REFUSED
    pre = ledger.precheck("pick_and_place", A)
    assert pre == {"ok": False, "error": "The one simulation retry for green cube is used",
                   "sim_retry": {"allowed": False, "used": True}}
    assert pre["error"].startswith("The one simulation retry for green cube is used")
    # the retry's own refusal opened no new slot: a call past the precheck is no retry either
    assert _begin(ledger) is None
    clock.advance(10)
    assert "sim_retry" not in _finish(ledger)
    assert ledger.precheck("pick_and_place", A) is not None


def test_a_retry_the_caller_did_not_pass_back_is_still_final(ledger, clock):
    _refused_first(ledger, clock)
    assert _begin(ledger) is not None
    clock.advance(30)
    out = _finish(ledger, retry=None)  # the ledger remembers that this call was the retry
    assert out["sim_retry"] == {"allowed": False, "used": True}
    assert ledger.precheck("pick_and_place", A) is not None


def test_a_retry_that_outlives_the_window_never_opens_a_new_slot(ledger, clock):
    _refused_first(ledger, clock)
    r = _begin(ledger)
    clock.advance(DEFAULT_WINDOW_S + 5)  # a very long retry
    assert _finish(ledger, retry=r)["sim_retry"] == {"allowed": False, "used": True}
    assert _begin(ledger) is None


def test_finish_without_begin_opens_no_slot(ledger, clock):
    out = ledger.finish("pick_and_place", A, dict(REFUSED), exc_type="SafetyViolation", refused_grasps=(G1,),
                        retry=None, **CLEAR)
    assert "sim_retry" not in out
    assert _begin(ledger) is None


def test_finish_after_another_calls_begin_opens_no_slot(ledger, clock):
    _begin(ledger, args={"object": "lemon"})
    clock.advance(10)
    assert "sim_retry" not in _finish(ledger)
    _begin(ledger, "move_home", {})  # a goal-less call forgets the started goal-bearing one
    assert "sim_retry" not in _finish(ledger)


def test_success_and_other_skills_are_untouched(ledger, clock):
    _begin(ledger)
    ok = {"ok": True, "placed": True}
    assert ledger.finish("pick_and_place", A, dict(ok), exc_type=None, refused_grasps=(), retry=None,
                         **CLEAR) == ok
    _begin(ledger, "place_at", {"object": "green cube"})
    out = _finish(ledger, "place_at", {"object": "green cube"})
    assert out == REFUSED and "sim_retry" not in out
    assert _begin(ledger, "place_at", {"object": "green cube"}) is None
    assert _begin(ledger) is None


# ── clear events ─────────────────────────────────────────────────────────

def test_a_successful_retry_clears_the_slot(ledger, clock):
    _refused_first(ledger, clock)
    r = _begin(ledger)
    clock.advance(40)
    out = ledger.finish("pick_and_place", A, {"ok": True}, exc_type=None, refused_grasps=(), retry=r, **CLEAR)
    assert out == {"ok": True}  # a retry that succeeds carries nothing extra
    assert ledger.last_event["outcome"] == "succeeded"
    assert ledger.precheck("pick_and_place", A) is None
    assert _begin(ledger) is None
    clock.advance(10)
    assert _finish(ledger)["sim_retry"]["allowed"] is True  # a new order for the same object


def test_success_of_the_goal_clears_an_open_slot(ledger, clock):
    _refused_first(ledger, clock)
    ledger.finish("grasp_object", {"label": "cube"}, {"ok": True}, exc_type=None, refused_grasps=(), retry=None,
                  **CLEAR)
    assert _begin(ledger) is None


def test_reset_scene_clears_every_slot(ledger, clock):
    _refused_first(ledger, clock)
    _begin(ledger)
    clock.advance(10)
    _finish(ledger)  # used
    ledger.clear("reset_scene")
    assert ledger.precheck("pick_and_place", A) is None
    assert _begin(ledger) is None


def test_a_motion_call_with_another_goal_clears_the_slot(ledger, clock):
    _refused_first(ledger, clock)
    _begin(ledger)
    clock.advance(10)
    _finish(ledger)  # used
    assert _begin(ledger, "grasp_object", {"label": "lemon"}) is None
    assert ledger.precheck("pick_and_place", A) is None


def test_a_goal_less_motion_never_clears_a_slot(ledger, clock):  # R28
    _refused_first(ledger, clock)
    assert _begin(ledger, "move_relative", {"dx": 0.01}) is None
    ledger.finish("move_relative", {"dx": 0.01}, {"ok": True}, exc_type=None, refused_grasps=(), retry=None,
                  **CLEAR)
    ledger.note_motion("move_relative", True)
    assert ledger.precheck("pick_and_place", {"object": "lemon"}) is None  # a precheck clears nothing
    assert _begin(ledger) is not None


@pytest.mark.parametrize("name,ok,left", [
    ("move_home", True, False), ("move_to_rest", True, False),
    ("move_home", False, True),  # a refused home moved nothing
    ("move_relative", True, True), ("place_at", True, True),
])
def test_move_home_clears_arm_left_in_place(ledger, clock, name, ok, left):
    _refused_first(ledger, clock)
    ledger.note_motion(name, ok)
    assert _begin(ledger).arm_left_in_place is left


def test_arm_left_in_place_comes_from_the_refused_result(ledger, clock):
    _refused_first(ledger, clock, result={"ok": False, "error": "SafetyViolation: x"})
    assert _begin(ledger).arm_left_in_place is False


# ── halt and e-stop: never open, use or re-arm a slot ────────────────────

HALT = {"ok": False, "error": "MotionHalted: halt received during route planning or execution"}


@pytest.mark.parametrize("state", [{"halted": True}, {"latched": True}])
def test_a_halted_first_attempt_opens_no_slot(ledger, clock, state):
    out = _refused_first(ledger, clock, **state)  # the refusal text says SafetyViolation, the call was halted
    assert "sim_retry" not in out
    assert _begin(ledger) is None


@pytest.mark.parametrize("state", [{"halted": True}, {"latched": True}])
def test_a_halt_during_any_motion_clears_an_open_slot(ledger, clock, state):
    _refused_first(ledger, clock)
    _begin(ledger, "move_home", {})
    ledger.finish("move_home", {}, dict(HALT), exc_type="MotionHalted", refused_grasps=(), retry=None,
                  **{**CLEAR, **state})
    assert _begin(ledger) is None  # no retry, no auto-home after the operator's halt
    clock.advance(10)
    assert "sim_retry" in _finish(ledger)  # the next refusal is a first attempt again (nothing consumed)


@pytest.mark.parametrize("flag", ["held", "latched", "episode_pending", "motion_blocked"])
def test_a_retry_cannot_start_unless_the_rig_is_ready(ledger, clock, flag):  # R7
    _refused_first(ledger, clock)
    assert _begin(ledger, **{flag: True}) is None  # runs as on main: no auto-home, no exclusions
    clock.advance(10)
    out = _finish(ledger)  # refused again: a call started under a latch or hold never opens a slot
    assert "sim_retry" not in out
    assert ledger.precheck("pick_and_place", A) is None and _begin(ledger) is None


def test_a_latch_at_the_start_of_any_motion_clears_an_open_slot(ledger, clock):
    _refused_first(ledger, clock)
    # e.g. the operator halted between the calls; the agent's next motion is a move_home
    assert _begin(ledger, "move_home", {}, latched=True) is None
    assert _begin(ledger) is None


def test_a_hold_at_the_start_of_a_goal_less_motion_keeps_the_slot(ledger, clock):
    _refused_first(ledger, clock)
    assert _begin(ledger, "move_relative", {"dx": 0.01}, held=True) is None
    assert _begin(ledger) is not None  # only a halt or e-stop clears slots from a goal-less call


def test_a_halt_during_the_retry_keeps_its_payload_and_the_used_slot(ledger, clock):
    _refused_first(ledger, clock)
    r = _begin(ledger)
    clock.advance(5)
    out = _finish(ledger, result=HALT, exc="MotionHalted", retry=r, halted=True)
    assert out == HALT  # main's halt result, no ledger text
    assert ledger.last_event["outcome"] == "halted"
    pre = ledger.precheck("pick_and_place", A)  # never another motion on the ledger's behalf
    assert pre["ok"] is False and pre["error"].startswith("The one simulation retry for green cube is used")


def test_a_halted_retry_that_reports_ok_is_not_a_success(ledger, clock):
    _refused_first(ledger, clock)
    r = _begin(ledger)
    ledger.finish("pick_and_place", A, {"ok": True}, exc_type=None, refused_grasps=(), retry=r,
                  **{**CLEAR, "halted": True})
    assert ledger.last_event["outcome"] == "halted"
    assert ledger.precheck("pick_and_place", A) is not None


# ── window (R9) and turn budget (R10) ────────────────────────────────────

def test_an_open_slot_lives_for_the_window(ledger, clock):
    _refused_first(ledger, clock)
    clock.advance(DEFAULT_WINDOW_S - 1)
    assert _begin(ledger) is not None


def test_a_slot_expires_exactly_one_window_after_it_opened(ledger, clock):
    _refused_first(ledger, clock)
    clock.advance(DEFAULT_WINDOW_S)
    assert _begin(ledger) is None


def test_an_open_slot_expires_after_the_window(ledger, clock):
    _refused_first(ledger, clock)
    clock.advance(DEFAULT_WINDOW_S + 1)
    assert _begin(ledger) is None  # a fresh first attempt ...
    clock.advance(10)
    assert _finish(ledger)["sim_retry"]["allowed"] is True  # ... that may open its own slot


def test_a_used_slot_expires_after_the_window(ledger, clock):
    _refused_first(ledger, clock)  # the slot opens now
    opened = clock.now
    r = _begin(ledger)
    clock.advance(60)
    _finish(ledger, retry=r)
    clock.now = opened + DEFAULT_WINDOW_S - 1
    assert ledger.precheck("pick_and_place", A) is not None
    clock.now = opened + DEFAULT_WINDOW_S + 1  # the next visitor's first order is not refused
    assert ledger.precheck("pick_and_place", A) is None
    assert _begin(ledger) is None
    clock.advance(10)
    assert _finish(ledger)["sim_retry"]["allowed"] is True


def test_the_window_is_read_from_the_monotonic_clock_by_default():
    assert SimRetryLedger()._clock is time.monotonic


@pytest.mark.parametrize("took,allowed", [(130.0, True), (130.5, False)])
def test_no_permission_that_would_overrun_the_turn(ledger, clock, took, allowed):
    # 330 s = the 360 s native turn minus the agent's exit grace; reserve 200 s for the retry
    out = _refused_first(ledger, clock, took=took)
    assert ("sim_retry" in out) is allowed
    assert (_begin(ledger) is not None) is allowed


@pytest.mark.parametrize("took,allowed", [(230.0, True), (230.5, False)])
def test_the_reserve_is_a_parameter(clock, took, allowed):
    L = SimRetryLedger(reserve_s=100, clock=clock)
    assert ("sim_retry" in _refused_first(L, clock, took=took)) is allowed


def test_window_and_turn_budget_follow_the_native_turn_budget():
    budget = runpy.run_path(str(REPO / "scripts" / "native_turn_budget.py"))
    assert DEFAULT_WINDOW_S == budget["NATIVE_TURN_TIMEOUT_S"]
    assert TURN_BUDGET_S == budget["NATIVE_TURN_TIMEOUT_S"] - budget["AGENT_EXIT_GRACE_S"]
    assert DEFAULT_RESERVE_S == 200


@pytest.mark.parametrize("bad", [359.0, 0, -1, math.nan, math.inf, True, "360", None])
def test_the_window_is_never_shorter_than_one_turn(bad):
    with pytest.raises(ValueError):
        SimRetryLedger(window_s=bad)


@pytest.mark.parametrize("bad", [0, -1, math.nan, math.inf, True, "200", None])
def test_the_reserve_must_be_a_positive_number(bad):
    with pytest.raises(ValueError):
        SimRetryLedger(reserve_s=bad)


def test_longer_windows_and_other_reserves_are_accepted():
    assert SimRetryLedger(window_s=360).window_s == 360
    assert SimRetryLedger(window_s=900.5, reserve_s=150).reserve_s == 150


# ── last_event, texts, robustness ────────────────────────────────────────

def test_last_event_describes_the_finished_retry(ledger, clock):
    _refused_first(ledger, clock, result={**REFUSED, "error": "Perception took 6s"}, exc="SlowPerceptionError")
    r = _begin(ledger)
    _finish(ledger, retry=r)
    event = ledger.last_event
    assert isinstance(event.pop("t"), float)
    assert event == {"kind": "retry", "goal": "green cube", "outcome": "failed", "refusal": "perception_refusal"}


def _spec_text() -> str:
    spec = REPO / "docs" / "superpowers" / "specs" / "2026-10-17-aspire-demo-design.md"
    return " ".join(spec.read_text(encoding="utf-8").split())


def test_texts_are_the_spec_texts_verbatim():
    spec = _spec_text()
    assert RETRY_ALLOWED.replace("{skill}", "<skill>") in spec
    assert RETRY_USED in spec
    assert f'error "{RETRY_EXHAUSTED.replace("{goal}", "<goal>")}"' in spec
    assert RETRY_ALLOWED.format(skill="pick_and_place").startswith(
        "SIMULATION ONLY: you may make ONE more attempt for this order: call pick_and_place again")


def test_bookkeeping_errors_fail_closed(clock):
    L = SimRetryLedger(clock=clock)
    _begin(L)
    clock.advance(10)
    out = _finish(L, grasps=5)  # not a sequence: no slot, no exception, the result untouched
    assert out == REFUSED
    assert _begin(L) is None


def test_bookkeeping_errors_in_precheck_and_begin_fail_closed(clock):
    def broken():
        raise RuntimeError("clock fault")

    L = SimRetryLedger(clock=clock)
    _refused_first(L, clock)
    L._clock = broken
    assert L.precheck("pick_and_place", A) is None
    L._clock = clock
    assert _begin(L) is None  # the precheck fault dropped every slot
    _refused_first(L, clock)
    L._clock = broken
    assert _begin(L) is None
    L._clock = clock
    assert _begin(L) is None


def test_a_clear_waits_for_a_begin_in_progress(clock):
    """Every entry point is atomic: a reset_scene clear from another thread (an MCP
    reader) cannot land between a begin's slot lookup and its use of the slot."""
    L = SimRetryLedger(clock=clock)
    _refused_first(L, clock)
    in_begin, cleared = threading.Event(), threading.Event()

    def pausing_clock():
        in_begin.set()
        cleared.wait(0.3)  # an unguarded clear lands here
        return clock.now

    L._clock = pausing_clock
    got = []
    worker = threading.Thread(target=lambda: got.append(_begin(L)))
    worker.start()
    assert in_begin.wait(5)
    L.clear("reset_scene")  # waits until the begin has finished
    cleared.set()
    worker.join(5)
    assert got == [RetryCall("green cube", (G1,), True, "safety_refusal")]
    L._clock = clock
    assert L.precheck("pick_and_place", A) is None and _begin(L) is None  # ... and then clears


# ── only a sim profile with the switch on gets a ledger ─────────────────

ARM_PROFILES = sorted(p.stem for p in (REPO / "configs" / "arms").glob("*.yaml"))
#: every shipped profile that drives hardware (so101_left/right are mock-typed teleop halves)
HARDWARE = ["rebot_rs", "rebot_rs_mb", "rebot_rs_reach", "so101", "so101_ros2", "fr3", "piper", "ros2_generic",
            "g1", "h1", "h1_2", "h1_sdk", "h1_2_sdk"]


@pytest.fixture
def switched_on(monkeypatch):
    """Everything that could turn the retry on is on: the env, the gate, the profile flag."""
    monkeypatch.setenv("CASCADE_SIM_RETRY", "1")
    monkeypatch.setenv("CASCADE_OBSERVED_FINGER_GATE", "1")
    monkeypatch.delenv("CASCADE_BOOTH", raising=False)

    def load(**kw):
        cfg = load_demo_config(**kw)
        cfg._data.setdefault("agent", {})["sim_retry"] = True
        return cfg
    return load


def test_hardware_profiles_are_shipped():
    assert set(HARDWARE) <= set(ARM_PROFILES)


@pytest.mark.parametrize("arm", ARM_PROFILES)
def test_only_sim_profiles_get_a_ledger(switched_on, arm):
    cfg = switched_on(arm=arm)
    ledger = ledger_for(cfg)
    if profile_class(cfg) == "sim":
        assert isinstance(ledger, SimRetryLedger) and arm not in HARDWARE
    else:
        assert ledger is None


@pytest.mark.parametrize("arm", HARDWARE)
def test_no_setting_gives_a_hardware_profile_a_ledger(switched_on, arm, monkeypatch):
    cfg = switched_on(arm=arm)
    assert profile_class(cfg) == "real"
    assert ledger_for(cfg) is None
    # even if the switch logic were broken, the class check alone refuses
    monkeypatch.setattr(sim_retry, "sim_retry_enabled", lambda cfg: True)
    assert ledger_for(cfg) is None


def test_mixed_rig_gets_no_ledger(switched_on):
    cfg = switched_on(arm="isaac_kitchen")
    cfg._data["arms"] = [dict(cfg.arm._data), {"type": "rebot_rs"}]
    assert ledger_for(cfg) is None


@pytest.mark.parametrize("arm_cfg", [{}, {"type": None}, {"type": "isaac_next"}, {"type": "Isaac"}])
def test_unknown_arm_type_gets_no_ledger(switched_on, arm_cfg):
    cfg = switched_on(arm="isaac_kitchen")
    cfg._data["arm"] = dict(arm_cfg)
    cfg._data["arms"] = [dict(arm_cfg)]
    assert ledger_for(cfg) is None


def test_sim_profile_with_everything_on_gets_a_ledger_with_the_defaults(switched_on):
    ledger = ledger_for(switched_on(arm="isaac_kitchen"))
    assert isinstance(ledger, SimRetryLedger)
    assert (ledger.window_s, ledger.reserve_s) == (DEFAULT_WINDOW_S, DEFAULT_RESERVE_S)


def test_the_switch_check_alone_refuses_too(switched_on, monkeypatch):
    cfg = switched_on(arm="isaac_kitchen")
    cfg._data["agent"]["sim_retry"] = False
    monkeypatch.setattr(sim_retry, "profile_class", lambda cfg: "sim")
    assert ledger_for(cfg) is None


@pytest.mark.parametrize("change", ["flag_off", "flag_absent", "kill_switch", "booth", "gate_off", "gate_unset"])
def test_any_switch_off_means_no_ledger(switched_on, monkeypatch, change):
    cfg = switched_on(arm="isaac_kitchen")
    if change == "flag_off":
        cfg._data["agent"]["sim_retry"] = False
    elif change == "flag_absent":
        cfg._data["agent"].pop("sim_retry")
    elif change == "kill_switch":
        monkeypatch.setenv("CASCADE_SIM_RETRY", "0")
    elif change == "booth":
        monkeypatch.setenv("CASCADE_BOOTH", "1")
    elif change == "gate_off":  # R8: without the observed-finger gate the retry has nothing to exclude
        monkeypatch.setenv("CASCADE_OBSERVED_FINGER_GATE", "0")
    else:
        monkeypatch.delenv("CASCADE_OBSERVED_FINGER_GATE")
    assert ledger_for(cfg) is None


def test_window_and_reserve_come_from_the_agent_config(switched_on):
    cfg = switched_on(arm="isaac_kitchen")
    cfg._data["agent"].update(sim_retry_window_s=600, sim_retry_reserve_s=150.5)
    ledger = ledger_for(cfg)
    assert (ledger.window_s, ledger.reserve_s) == (600, 150.5)


@pytest.mark.parametrize("key,value", [("sim_retry_window_s", 300), ("sim_retry_window_s", "600"),
                                       ("sim_retry_window_s", math.nan), ("sim_retry_reserve_s", 0),
                                       ("sim_retry_reserve_s", True), ("sim_retry_reserve_s", None)])
def test_a_bad_window_or_reserve_means_no_ledger(switched_on, key, value):
    cfg = switched_on(arm="isaac_kitchen")
    cfg._data["agent"][key] = value
    assert ledger_for(cfg) is None


@pytest.mark.parametrize("cfg", [None, object(), "isaac_kitchen"])
def test_an_unreadable_config_gets_no_ledger(switched_on, cfg):
    assert ledger_for(cfg) is None


def test_a_config_that_raises_gets_no_ledger(switched_on, monkeypatch):
    cfg = switched_on(arm="isaac_kitchen")

    def boom(*_a, **_k):
        raise RuntimeError("config fault")
    monkeypatch.setattr(sim_retry, "sim_retry_enabled", boom)
    assert ledger_for(cfg) is None
