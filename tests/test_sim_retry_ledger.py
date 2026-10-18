"""ASPIRE A6: the sim-only retry ledger (spec §4.5; preflight R6, R7, R9, R10, R28; A6-fix rulings).

Pure bookkeeping, no runtime. A goal is the order's object, destination and arm,
normalised so a rephrased order is the same goal. After a qualifying refusal the
ledger opens ONE slot per goal; the next same-goal call is THE retry (the slot is
used before anything moves); every later same-goal call is refused before motion,
by `precheck` and by `begin`, until the goal succeeds, `reset_scene`, or the
slot's window ends. A call with another goal never touches a slot. A halt, latch,
hold or pending episode SPENDS the goal: its calls run as without the ledger (no
retry, no refusal, nothing opens). Only a sim profile with the switch on ever gets
a ledger (`ledger_for`); real and unknown profiles get None. The ledger never
edits agent text (A7 composes `next_action`): it only adds the `sim_retry` field
and returns the used-slot refusal.
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
    MAX_RESERVE_S,
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

REPO = Path(__file__).resolve().parents[1]
A = {"object": "green cube"}
LEMON = {"object": "lemon"}
GOAL = goal_of("pick_and_place", A)
G1, G2 = ("grasp-1",), ("grasp-2",)
REFUSED = {"ok": False, "error": "SafetyViolation: the finger would enter the lemon", "arm_left_in_place": True}
#: the refusal `precheck` and `begin` return once green cube's retry is used
USED = {"ok": False, "error": "The one simulation retry for green cube is used",
        "sim_retry": {"allowed": False, "used": True}}
#: finish(): nothing held, no latch, no halt during the call, no pending episode
CLEAR = {"held": False, "latched": False, "halted": False, "episode_pending": False}
#: begin(): the rig may start a retry
READY = {"held": False, "latched": False, "episode_pending": False, "motion_blocked": False}


def W(*words: str) -> frozenset:
    return frozenset(words)


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
    (fail closed, the ledger turns off) fails the test that caused it."""
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


def _agent(L, clock, orders, *, calls=30, step=10.0):
    """An agent that re-issues `orders` (cycled; each `(key, name, args)`) after
    every refusal, as the review's probes did, with no lock of its own. Per key:
    the motion calls that ran, the retries granted, the calls refused before motion."""
    counts = {}
    for i in range(calls):
        key, name, args = orders[i % len(orders)]
        count = counts.setdefault(key, {"motions": 0, "grants": 0, "refused": 0})
        if L.precheck(name, args) is not None:
            count["refused"] += 1
            continue
        started = L.begin(name, args, **READY)
        if isinstance(started, dict):
            count["refused"] += 1
            continue
        count["motions"] += 1
        count["grants"] += isinstance(started, RetryCall)
        clock.advance(step)
        L.finish(name, args, dict(REFUSED), exc_type="SafetyViolation", refused_grasps=(G1,), retry=started,
                 **CLEAR)
    return counts


# ── goals ────────────────────────────────────────────────────────────────

def test_retry_skills_are_the_two_grasping_orders():
    assert RETRY_SKILLS == frozenset({"pick_and_place", "grasp_object"})


def test_goal_is_object_words_destination_and_arm_normalized():
    assert goal_of("pick_and_place", {"object": "  Green\tCUBE "}) == Goal("green cube", W("green", "cube"))
    assert goal_of("pick_and_place", {"object": "the cube", "destination": " Red  Plate.", "arm": "Left"}) == Goal(
        "the cube", W("cube"), W("red", "plate"), W("left"))
    # grasp_object has no destination: a stray one never splits its goal
    assert goal_of("grasp_object", {"label": "Lemon", "arm": "right", "destination": "x"}) == Goal(
        "lemon", W("lemon"), W(), W("right"))
    # the name is for display only (casefolded, whitespace-collapsed, as the agent said it)
    assert str(goal_of("pick_and_place", {"object": "Green  Cube", "destination": "plate"})) == "green cube"
    assert str(goal_of("pick_and_place", {"object": "the Green_Cube"})) == "the green_cube"


@pytest.mark.parametrize("text,words", [
    ("green_cube", {"green", "cube"}),  # scene names: "_" separates words
    ("green-cube.", {"green", "cube"}),
    ("The open box!", {"open", "box"}),
    ("this, that; these/those a an cube", {"cube"}),
    ("ＧＲＥＥＮ　ＣＵＢＥ", {"green", "cube"}),  # NFKC: full-width letters and space
    ("Straße", {"strasse"}),  # casefold
    ("ﬁsh", {"fish"}),  # NFKC ligature
    ("cube 2", {"cube", "2"}),
    ("cubes", {"cubes"}),  # no stemming
])
def test_goal_words_are_nfkc_casefolded_letters_and_digits_without_determiners(text, words):
    assert goal_of("pick_and_place", {"object": text}).words == frozenset(words)
    assert goal_of("pick_and_place", {"object": "x", "destination": text}).destination == frozenset(words)
    assert goal_of("pick_and_place", {"object": "x", "arm": text}).arm == frozenset(words)


@pytest.mark.parametrize("obj", ["the", "...", "-", "_", "a the", "This.", "  ", ""])
def test_an_object_without_words_is_no_goal(obj):  # P7: it would match every slot
    assert goal_of("pick_and_place", {"object": obj}) is None
    assert goal_of("grasp_object", {"label": obj}) is None


@pytest.mark.parametrize("args", [{}, {"object": ""}, {"object": "   "}, {"object": 3}, {"object": None},
                                  {"object": "cube", "destination": 3}, {"object": "cube", "arm": 0},
                                  {"label": "cube"}, None, "green cube"])
def test_no_goal_without_a_readable_object(args):
    assert goal_of("pick_and_place", args) is None


@pytest.mark.parametrize("name", ["place_at", "move_home", "reset_scene", "grasp_at_pixel", "handover", None, 3])
def test_other_skills_have_no_goal(name):
    assert goal_of(name, {"object": "green cube", "label": "green cube"}) is None


#: (first call, rephrased call) of ONE order (review probes P5 / P7)
SAME_ORDER = [
    (("pick_and_place", A), ("grasp_object", {"label": "cube"})),  # spec: a renamed object
    (("pick_and_place", A), ("pick_and_place", {"object": "GREEN  CUBE"})),
    (("pick_and_place", A), ("pick_and_place", {"object": "green-cube"})),
    (("pick_and_place", A), ("pick_and_place", {"object": "green_cube"})),
    (("pick_and_place", A), ("pick_and_place", {"object": "cube."})),
    (("pick_and_place", A), ("pick_and_place", {"object": "this green cube"})),
    (("pick_and_place", A), ("pick_and_place", {"object": "ＧＲＥＥＮ ＣＵＢＥ"})),
    (("pick_and_place", {**A, "destination": "open box"}), ("pick_and_place", {**A, "destination": "the open box"})),
    (("pick_and_place", {**A, "destination": "open box"}), ("pick_and_place", {**A, "destination": "open box."})),
    (("pick_and_place", {**A, "destination": "open box"}), ("pick_and_place", {**A, "destination": "box"})),
    (("pick_and_place", {**A, "destination": "open box"}), ("pick_and_place", {**A, "destination": "open_box"})),
    (("pick_and_place", {**A, "destination": "open box"}), ("grasp_object", {"label": "green cube"})),
    (("pick_and_place", A), ("pick_and_place", {**A, "arm": "left"})),  # an absent arm matches any arm
    (("pick_and_place", {**A, "arm": "left"}), ("pick_and_place", {**A, "arm": "the Left"})),
    (("pick_and_place", {**A, "arm": "left"}), ("pick_and_place", {**A, "arm": "the"})),  # "the" = no arm
]

#: two different orders (P7: none of these may ever merge)
DIFFERENT_ORDERS = [
    (("pick_and_place", A), ("pick_and_place", {"object": "pink cube"})),
    (("pick_and_place", {"object": "cube"}), ("pick_and_place", {"object": "cubes"})),  # no stemming
    (("pick_and_place", {"object": "cube 2"}), ("pick_and_place", {"object": "cube 3"})),
    (("pick_and_place", {**A, "destination": "plate"}), ("pick_and_place", {**A, "destination": "bowl"})),
    (("pick_and_place", {**A, "destination": "open box"}),
     ("pick_and_place", {**A, "destination": "green square"})),
    (("pick_and_place", {**A, "destination": "red plate"}), ("pick_and_place", {**A, "destination": "blue plate"})),
    (("pick_and_place", {**A, "arm": "left"}), ("pick_and_place", {**A, "arm": "right"})),
    (("pick_and_place", {**A, "arm": "left"}), ("pick_and_place", {**A, "arm": "left arm"})),  # given arms: equal
]


@pytest.mark.parametrize("first,second", SAME_ORDER)
def test_a_rephrased_order_is_the_same_goal(first, second):
    a, b = goal_of(*first), goal_of(*second)
    assert same_goal(a, b) and same_goal(b, a)


@pytest.mark.parametrize("first,second", DIFFERENT_ORDERS)
def test_another_order_is_another_goal(first, second):
    a, b = goal_of(*first), goal_of(*second)
    assert not same_goal(a, b) and not same_goal(b, a)


def test_a_goal_without_object_words_matches_nothing():
    assert not same_goal(Goal("the"), GOAL) and not same_goal(GOAL, Goal("the"))
    assert not same_goal(Goal("the"), Goal("..."))


@pytest.mark.parametrize("a,b", [("cube", "green cube"), (None, None), (GOAL, "green cube"), ("green cube", GOAL),
                                 (GOAL, None), ({"object": "cube"}, GOAL)])
def test_same_goal_takes_goals_only(a, b):  # Minor 6: a plan-style string call fails loudly
    with pytest.raises(TypeError):
        same_goal(a, b)


def test_renamed_object_is_the_same_goal(ledger, clock):  # Review Focus 1
    assert same_object("cube", "green cube") and same_object("Green  Cube", "the green cube")
    assert same_object("green_cube", "green-cube.")
    assert not same_object("pink cube", "green cube") and not same_object("cube", "cubes")
    assert not same_object("", "cube") and not same_object(None, "cube") and not same_object("the", "cube")
    assert same_goal(goal_of("grasp_object", {"label": "cube"}), GOAL)
    assert not same_goal(goal_of("pick_and_place", {"object": "pink cube"}), GOAL)
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


def test_changed_destination_or_arm_is_another_goal(ledger, clock):  # R28, A6-fix I1
    args = {"object": "green cube", "destination": "plate", "arm": "left"}
    base = goal_of("pick_and_place", args)
    assert not same_goal(base, goal_of("pick_and_place", {**args, "destination": "bowl"}))
    assert not same_goal(base, goal_of("pick_and_place", {**args, "arm": "right"}))
    _refused_first(ledger, clock, args=args)
    assert _begin(ledger, args={**args, "destination": "bowl"}) is None  # not the retry: another order ...
    assert isinstance(_begin(ledger, args=args), RetryCall)  # ... which left this order's slot alone


REPHRASINGS = {
    "same args": [A],
    "object renamed (spec)": [A, {"object": "cube"}],
    "object case and spaces": [A, {"object": " GREEN   cube "}],
    "object punctuation, hyphen, underscore": [A, {"object": "green-cube"}, {"object": "cube."},
                                               {"object": "green_cube"}],
    "destination article": [{**A, "destination": "open box"}, {**A, "destination": "the open box"}],
    "destination punctuation": [{**A, "destination": "open box"}, {**A, "destination": "open box."}],
    "destination shortened": [{**A, "destination": "open box"}, {**A, "destination": "box"}],
    "arm given / omitted": [A, {**A, "arm": "left"}],
    "demonstrative": [A, {"object": "this green cube"}, {"object": "that cube"}],
    "full-width": [A, {"object": "ＧＲＥＥＮ ＣＵＢＥ"}],
}


@pytest.mark.parametrize("case", sorted(REPHRASINGS))
def test_one_order_rephrased_gets_two_motion_calls(ledger, clock, case):  # probes P5 / P7
    orders = [("order", "pick_and_place", args) for args in REPHRASINGS[case]]
    assert _agent(ledger, clock, orders) == {"order": {"motions": 2, "grants": 1, "refused": 28}}


#: the kitchen's five objects and two destinations, each said two ways
KITCHEN_OBJECTS = [("green cube", "green_cube"), ("pink cube", "the pink_cube"), ("tomato can", "tomato-can"),
                   ("lemon", "the lemon."), ("orange", "ORANGE")]
KITCHEN_DESTINATIONS = [("open box", "the open_box"), ("green square", "the green square")]


def _kitchen_orders():
    """20 phrased orders: 10 distinct orders (5 objects x 2 destinations), each said two ways."""
    return [((o, d), "pick_and_place", {"object": obj[phrasing], "destination": dest[phrasing]})
            for phrasing in (0, 1)
            for o, obj in enumerate(KITCHEN_OBJECTS)
            for d, dest in enumerate(KITCHEN_DESTINATIONS)]


def test_distinct_kitchen_orders_never_merge():  # probe P7
    orders = _kitchen_orders()
    assert len(orders) == 20 and len({key for key, _, _ in orders}) == 10
    for key_a, name_a, args_a in orders:
        for key_b, name_b, args_b in orders:
            assert same_goal(goal_of(name_a, args_a), goal_of(name_b, args_b)) is (key_a == key_b), (args_a, args_b)


def test_interleaved_kitchen_orders_each_get_one_retry(ledger, clock):
    counts = _agent(ledger, clock, _kitchen_orders(), calls=60, step=2.0)
    assert counts == {key: {"motions": 2, "grants": 1, "refused": 4} for key, _, _ in _kitchen_orders()}


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


#: plain SafetyViolations the harness raises for its own callback faults (Minor 8): a retry fails the same way
HARNESS_FAULTS = ["motion safety callback failed; no unguarded retry",
                  "ambiguous motion preflight callbacks; no motion sent"]

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
    ({"ok": False, "error": f"SafetyViolation: {HARNESS_FAULTS[0]}"}, "SafetyViolation", {}),
    ({"ok": False, "error": HARNESS_FAULTS[1]}, "SafetyViolation", {}),
    ({"ok": True}, "SafetyViolation", {}),
]


@pytest.mark.parametrize("result,exc,state", NON_REFUSALS)
def test_non_refusals_open_no_slot(ledger, clock, result, exc, state):
    assert refusal_class(exc, result, **{**CLEAR, **state}) is None
    out = _refused_first(ledger, clock, result=result, exc=exc, **state)
    assert "sim_retry" not in out and out == result
    assert _begin(ledger) is None  # the next call is no retry


def test_the_harness_fault_texts_are_the_harness_words():  # Minor 8
    source = (REPO / "src" / "cascade" / "safety" / "harness.py").read_text(encoding="utf-8")
    for text in HARNESS_FAULTS:
        assert f'SafetyViolation("{text}"' in source
        assert any(excluded in text for excluded in sim_retry._NEVER_QUALIFY_TEXT)


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
    assert _begin(ledger) == USED  # consumed before anything moves: a second caller is refused


def test_a_failed_retry_is_final_and_a_third_call_is_refused(ledger, clock):
    _refused_first(ledger, clock)
    r = _begin(ledger)
    clock.advance(60)
    out = _finish(ledger, grasps=(G2,), retry=r)
    assert out["sim_retry"] == {"allowed": False, "used": True}
    assert {k: v for k, v in out.items() if k != "sim_retry"} == REFUSED
    pre = ledger.precheck("pick_and_place", A)
    assert pre == USED
    assert pre["error"] == RETRY_EXHAUSTED.format(goal="green cube")
    # a caller that skips the precheck is refused by begin itself; nothing opens again
    assert _begin(ledger) == USED
    clock.advance(10)
    assert "sim_retry" not in _finish(ledger)
    assert ledger.precheck("pick_and_place", A) == USED


def test_begin_refuses_a_used_slot_whatever_the_rig_state(ledger, clock):  # invariant (b)
    _refused_first(ledger, clock)
    r = _begin(ledger)
    clock.advance(30)
    _finish(ledger, retry=r)
    for flag in READY:
        assert _begin(ledger, **{flag: True}) == USED
    assert _begin(ledger, "grasp_object", {"label": "the cube."}) == USED
    assert _begin(ledger, args=LEMON) is None  # another goal is not refused


def test_begin_refuses_a_used_slot_without_the_callers_lock(clock):  # invariant (b), probe P4a
    """Eight unsynchronised callers all pass the precheck while the slot is open,
    then all begin: exactly one gets the retry, every other one is refused (none
    runs as a fresh first attempt)."""
    L = SimRetryLedger(clock=clock)
    _refused_first(L, clock)
    n = 8
    passed = [L.precheck("pick_and_place", A) is None for _ in range(n)]
    gate = threading.Barrier(n)
    got = [None] * n

    def caller(i):
        gate.wait(5)
        got[i] = _begin(L)
    threads = [threading.Thread(target=caller, args=(i,)) for i in range(n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert all(passed)
    assert sum(isinstance(g, RetryCall) for g in got) == 1
    assert [g for g in got if not isinstance(g, RetryCall)] == [USED] * (n - 1)


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
    _begin(ledger, args=LEMON)
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


# ── per goal, whatever the interleaving (invariant (a)) ──────────────────

def test_another_goal_keeps_an_open_slot(ledger, clock):
    _refused_first(ledger, clock)
    assert _begin(ledger, args=LEMON) is None  # a first attempt for the lemon ...
    clock.advance(10)
    assert _finish(ledger, args=LEMON)["sim_retry"]["goal"] == "lemon"  # ... with its own slot
    assert isinstance(_begin(ledger), RetryCall)  # green cube's retry is still there


def test_another_goal_never_rearms_a_used_retry(ledger, clock):
    _refused_first(ledger, clock)
    r = _begin(ledger)
    clock.advance(30)
    _finish(ledger, retry=r)  # used
    _refused_first(ledger, clock, args=LEMON)
    _begin(ledger, "grasp_object", {"label": "orange"})
    clock.advance(10)
    ledger.finish("grasp_object", {"label": "orange"}, {"ok": True}, exc_type=None, refused_grasps=(), retry=None,
                  **CLEAR)  # another goal's success clears nothing of green cube's
    assert ledger.precheck("pick_and_place", A) == USED
    assert _begin(ledger) == USED


def test_interleaving_other_goals_never_grants_a_second_retry(ledger, clock):  # invariant (a)
    orders = [("A", "pick_and_place", A), ("B", "pick_and_place", LEMON),
              ("A", "grasp_object", {"label": "the green-cube"}),
              ("C", "pick_and_place", {"object": "orange", "destination": "open box"}),
              ("A", "pick_and_place", {"object": "cube."}), ("B", "grasp_object", {"label": "Lemon"})]
    assert _agent(ledger, clock, orders) == {"A": {"motions": 2, "grants": 1, "refused": 13},
                                            "B": {"motions": 2, "grants": 1, "refused": 8},
                                            "C": {"motions": 2, "grants": 1, "refused": 3}}


def test_an_ambiguous_call_takes_the_most_restrictive_matching_slot(ledger, clock):
    _refused_first(ledger, clock)  # green cube: open
    _refused_first(ledger, clock, args={"object": "pink cube"})  # pink cube: open
    r = _begin(ledger, args={"object": "cube"})  # matches both: the earliest open slot is the retry
    assert r.goal == "green cube"
    clock.advance(10)
    _finish(ledger, args={"object": "cube"}, retry=r)
    assert _begin(ledger, args={"object": "cube"}) == USED  # green cube's used slot wins over pink's open one
    assert isinstance(_begin(ledger, args={"object": "pink cube"}), RetryCall)


def test_a_used_slot_wins_over_a_spent_one(ledger, clock):
    _refused_first(ledger, clock)
    r = _begin(ledger)
    clock.advance(10)
    _finish(ledger, retry=r)  # green cube: used
    _refused_first(ledger, clock, args={"object": "pink cube"}, halted=True)  # pink cube: spent
    assert ledger.precheck("pick_and_place", {"object": "cube"}) == USED
    assert _begin(ledger, args={"object": "cube"}) == USED


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


def test_a_successful_retry_the_caller_did_not_pass_back_clears_the_slot(ledger, clock):
    _refused_first(ledger, clock)
    assert _begin(ledger) is not None
    clock.advance(40)
    ledger.finish("pick_and_place", A, {"ok": True}, exc_type=None, refused_grasps=(), retry=None, **CLEAR)
    assert ledger.last_event["outcome"] == "succeeded"
    assert ledger.precheck("pick_and_place", A) is None and _begin(ledger) is None


def test_a_successful_retry_clears_its_slot_even_if_the_ledger_lost_the_call(ledger, clock):
    _refused_first(ledger, clock)
    r = _begin(ledger)
    _begin(ledger, "move_home", {})  # another begin in between: the ledger forgets the retry's call
    clock.advance(40)
    ledger.finish("pick_and_place", A, {"ok": True}, exc_type=None, refused_grasps=(), retry=r, **CLEAR)
    assert ledger.last_event["outcome"] == "succeeded"
    assert ledger.precheck("pick_and_place", A) is None and _begin(ledger) is None


def test_a_success_never_ends_another_calls_used_slot(ledger, clock):
    _refused_first(ledger, clock)
    r = _begin(ledger)
    clock.advance(10)
    _finish(ledger, retry=r)  # green cube: used
    # a "cube" success the ledger never saw begin (it cannot have been green cube's retry)
    ledger.finish("pick_and_place", {"object": "cube"}, {"ok": True}, exc_type=None, refused_grasps=(), retry=None,
                  **CLEAR)
    assert ledger.precheck("pick_and_place", A) == USED


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
    _refused_first(ledger, clock, args=LEMON, halted=True)  # lemon: spent
    ledger.clear("reset_scene")
    assert ledger.precheck("pick_and_place", A) is None
    assert _begin(ledger) is None
    clock.advance(10)
    assert "sim_retry" in _finish(ledger)
    _refused_first(ledger, clock, args=LEMON)
    assert isinstance(_begin(ledger, args=LEMON), RetryCall)


def test_a_goal_less_motion_never_clears_a_slot(ledger, clock):  # R28
    _refused_first(ledger, clock)
    assert _begin(ledger, "move_relative", {"dx": 0.01}) is None
    ledger.finish("move_relative", {"dx": 0.01}, {"ok": True}, exc_type=None, refused_grasps=(), retry=None,
                  **CLEAR)
    ledger.note_motion("move_relative", True)
    assert ledger.precheck("pick_and_place", LEMON) is None  # a precheck clears nothing
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


# ── halt, e-stop, hold, episode: the goal is spent (invariant (c)) ───────

HALT = {"ok": False, "error": "MotionHalted: halt received during route planning or execution"}


def _runs_as_without_the_ledger(L, clock, args=A, calls=3):
    """Every call of a spent goal: no refusal, no retry, and its refusal opens nothing."""
    for _ in range(calls):
        assert L.precheck("pick_and_place", args) is None
        assert _begin(L, args=args) is None
        clock.advance(10)
        out = _finish(L, args=args)
        assert "sim_retry" not in out and out == REFUSED
    return True


@pytest.mark.parametrize("state", [{"halted": True}, {"latched": True}])
def test_a_halted_first_attempt_spends_its_goal(ledger, clock, state):
    out = _refused_first(ledger, clock, **state)  # the refusal text says SafetyViolation, the call was halted
    assert "sim_retry" not in out
    assert _runs_as_without_the_ledger(ledger, clock)


@pytest.mark.parametrize("state", [{"halted": True}, {"latched": True}])
def test_a_halt_during_any_motion_spends_an_open_slot(ledger, clock, state):
    _refused_first(ledger, clock)
    _begin(ledger, "move_home", {})
    ledger.finish("move_home", {}, dict(HALT), exc_type="MotionHalted", refused_grasps=(), retry=None,
                  **{**CLEAR, **state})
    assert _runs_as_without_the_ledger(ledger, clock)  # no retry, no auto-home after the operator's halt


@pytest.mark.parametrize("name,args", [("move_home", {}), ("pick_and_place", LEMON)])
def test_a_halt_whose_result_is_not_a_dict_still_spends_open_slots(ledger, clock, name, args):  # Minor 5
    _refused_first(ledger, clock)
    _begin(ledger, name, args)
    assert ledger.finish(name, args, None, exc_type="MotionHalted", refused_grasps=(), retry=None,
                         **{**CLEAR, "halted": True}) is None
    assert _runs_as_without_the_ledger(ledger, clock)
    if args:  # the halted call's own goal is spent too
        assert _runs_as_without_the_ledger(ledger, clock, args=args)


@pytest.mark.parametrize("flag", ["held", "latched", "episode_pending", "motion_blocked"])
def test_a_retry_cannot_start_unless_the_rig_is_ready(ledger, clock, flag):  # R7
    _refused_first(ledger, clock)
    assert _begin(ledger, **{flag: True}) is None  # runs as on main: no auto-home, no exclusions
    clock.advance(10)
    out = _finish(ledger)  # refused again: a call started under a latch or hold never opens a slot
    assert "sim_retry" not in out
    assert _runs_as_without_the_ledger(ledger, clock)  # and the goal is spent: no fresh slot either


@pytest.mark.parametrize("flag", ["held", "latched", "episode_pending", "motion_blocked"])
def test_a_first_attempt_started_unready_spends_its_goal(ledger, clock, flag):
    assert _begin(ledger, **{flag: True}) is None
    clock.advance(10)
    assert "sim_retry" not in _finish(ledger)
    assert _runs_as_without_the_ledger(ledger, clock)


@pytest.mark.parametrize("flag", ["held", "latched", "episode_pending", "motion_blocked"])
def test_a_call_started_unready_opens_nothing_even_if_a_reset_lands_during_it(ledger, clock, flag):
    assert _begin(ledger, **{flag: True}) is None
    ledger.clear("reset_scene")  # e.g. an MCP reader's reset_scene while the call runs
    clock.advance(10)
    assert "sim_retry" not in _finish(ledger)


@pytest.mark.parametrize("flag", ["held", "episode_pending"])
def test_a_failed_call_that_ends_holding_or_with_an_episode_pending_spends_its_goal(ledger, clock, flag):
    assert "sim_retry" not in _refused_first(ledger, clock, **{flag: True})
    assert _runs_as_without_the_ledger(ledger, clock)


def test_a_latch_at_the_start_of_any_motion_spends_an_open_slot(ledger, clock):
    _refused_first(ledger, clock)
    # e.g. the operator halted between the calls; the agent's next motion is a move_home
    assert _begin(ledger, "move_home", {}, latched=True) is None
    assert _runs_as_without_the_ledger(ledger, clock)


def test_a_hold_at_the_start_of_a_goal_less_motion_keeps_the_slot(ledger, clock):
    _refused_first(ledger, clock)
    assert _begin(ledger, "move_relative", {"dx": 0.01}, held=True) is None
    assert _begin(ledger) is not None  # only a halt or e-stop touches slots from a goal-less call


def test_a_spent_goal_leaves_other_goals_alone(ledger, clock):
    _refused_first(ledger, clock, args=LEMON)  # lemon: open
    _refused_first(ledger, clock, held=True)  # green cube: spent by its own hold
    assert isinstance(_begin(ledger, args=LEMON), RetryCall)


@pytest.mark.parametrize("end", ["window", "reset_scene", "success"])
def test_a_spent_goal_runs_as_without_the_ledger_until_a_clear_event(ledger, clock, end):
    _refused_first(ledger, clock)
    assert _begin(ledger, held=True) is None
    clock.advance(10)
    _finish(ledger, held=True)  # a hold touched the goal: spent
    assert _runs_as_without_the_ledger(ledger, clock)
    if end == "window":
        clock.advance(DEFAULT_WINDOW_S)
    elif end == "reset_scene":
        ledger.clear("reset_scene")
    else:
        assert _begin(ledger) is None
        clock.advance(10)
        ledger.finish("pick_and_place", A, {"ok": True}, exc_type=None, refused_grasps=(), retry=None, **CLEAR)
    # the goal is fresh again: a first attempt may open a slot and get its one retry
    assert _begin(ledger) is None
    clock.advance(10)
    assert _finish(ledger)["sim_retry"]["allowed"] is True
    assert isinstance(_begin(ledger), RetryCall)


def test_a_halt_during_the_retry_keeps_its_payload_and_the_used_slot(ledger, clock):
    _refused_first(ledger, clock)
    r = _begin(ledger)
    clock.advance(5)
    out = _finish(ledger, result=HALT, exc="MotionHalted", retry=r, halted=True)
    assert out == HALT  # main's halt result, no ledger text
    assert ledger.last_event["outcome"] == "halted"
    pre = ledger.precheck("pick_and_place", A)  # never another motion on the ledger's behalf
    assert pre == USED and _begin(ledger) == USED


def test_a_halted_retry_that_reports_ok_is_not_a_success(ledger, clock):
    _refused_first(ledger, clock)
    r = _begin(ledger)
    ledger.finish("pick_and_place", A, {"ok": True}, exc_type=None, refused_grasps=(), retry=r,
                  **{**CLEAR, "halted": True})
    assert ledger.last_event["outcome"] == "halted"
    assert ledger.precheck("pick_and_place", A) is not None


@pytest.mark.parametrize("state", [{"halted": True}, {"latched": True}])
def test_a_halted_call_that_reports_ok_does_not_end_a_spent_goal(ledger, clock, state):
    _refused_first(ledger, clock, halted=True)  # spent
    assert _begin(ledger) is None
    clock.advance(10)
    ledger.finish("pick_and_place", A, {"ok": True}, exc_type=None, refused_grasps=(), retry=None,
                  **{**CLEAR, **state})
    assert _runs_as_without_the_ledger(ledger, clock)


#: probe P2: an event around one order the agent re-issues 8 times -> (motions, grants, ledger refusals)
P2 = {
    "no halt": (2, 1, 6),
    "halt during the first attempt": (8, 0, 0),
    "latched at the end of the first attempt": (8, 0, 0),
    "halt during the retry": (2, 1, 6),
    "latched at the retry's begin": (8, 0, 0),
    "held at the retry's begin": (8, 0, 0),
    "episode pending at the retry's begin": (8, 0, 0),
    "halt on a goal-less move between": (8, 0, 0),
    "latched at a goal-less move's begin between": (8, 0, 0),
}


def _p2(L, clock, event, calls=8):
    motions = grants = refused = 0
    for i in range(calls):
        if i == 1 and event.endswith("between"):
            latched = event.startswith("latched")
            L.begin("move_home", {}, **{**READY, "latched": latched})
            clock.advance(5)
            L.finish("move_home", {}, dict(HALT), exc_type="MotionHalted", refused_grasps=(), retry=None,
                     **{**CLEAR, "halted": not latched, "latched": latched})
        if L.precheck("pick_and_place", A) is not None:
            refused += 1
            continue
        at_begin = {}
        if i == 1 and event.endswith("at the retry's begin"):
            at_begin = {{"latched": "latched", "held": "held", "episode": "episode_pending"}[event.split()[0]]: True}
        started = L.begin("pick_and_place", A, **{**READY, **at_begin})
        if isinstance(started, dict):
            refused += 1
            continue
        motions += 1
        grants += isinstance(started, RetryCall)
        clock.advance(10)
        at_end = {}
        if (i == 0 and event == "halt during the first attempt") or (i == 1 and event == "halt during the retry"):
            at_end = {"halted": True}
        elif (i == 0 and event == "latched at the end of the first attempt") or (i == 1 and "latched" in at_begin):
            at_end = {"latched": True}
        L.finish("pick_and_place", A, dict(HALT if at_end else REFUSED),
                 exc_type="MotionHalted" if at_end else "SafetyViolation", refused_grasps=(G1,), retry=started,
                 **{**CLEAR, **at_end})
    return motions, grants, refused


@pytest.mark.parametrize("event", sorted(P2))
def test_no_halt_hold_or_latch_ever_grants_a_second_retry(ledger, clock, event):  # probe P2
    assert _p2(ledger, clock, event) == P2[event]


# ── window (R9) and turn budget (R10) ────────────────────────────────────

def test_an_open_slot_lives_for_the_window(ledger, clock):
    _refused_first(ledger, clock, took=10.0)
    clock.advance(DEFAULT_WINDOW_S - 1)
    assert _begin(ledger) is None  # too late for the turn (R10): no retry ...
    assert ledger.precheck("pick_and_place", A) == USED  # ... but the slot was still there: now used


def test_a_slot_expires_exactly_one_window_after_it_opened(ledger, clock):
    _refused_first(ledger, clock)
    clock.advance(DEFAULT_WINDOW_S)
    assert _begin(ledger) is None
    clock.advance(10)
    assert _finish(ledger)["sim_retry"]["allowed"] is True


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
    assert ledger.precheck("pick_and_place", A) == USED and _begin(ledger) == USED
    clock.now = opened + DEFAULT_WINDOW_S + 1  # the next visitor's first order is not refused
    assert ledger.precheck("pick_and_place", A) is None
    assert _begin(ledger) is None
    clock.advance(10)
    assert _finish(ledger)["sim_retry"]["allowed"] is True


def test_a_spent_slot_expires_one_window_after_the_event_that_spent_it(ledger, clock):
    _refused_first(ledger, clock, halted=True)  # spent now, no slot was open
    spent = clock.now
    clock.now = spent + DEFAULT_WINDOW_S - 1
    assert _runs_as_without_the_ledger(ledger, clock, calls=1)
    clock.now = spent + DEFAULT_WINDOW_S
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


@pytest.mark.parametrize("delay,granted", [(0.0, True), (0.5, False), (100.0, False), (199.0, False)])
def test_a_retry_that_would_start_too_late_is_not_granted(ledger, clock, delay, granted):  # R10 at begin, P3
    _refused_first(ledger, clock, took=130.0)  # 130 + 200 = 330: the slot opens
    clock.advance(delay)
    started = _begin(ledger)
    assert isinstance(started, RetryCall) is granted
    if granted:
        return
    assert started is None  # runs as on main: no auto-home, no exclusions
    clock.advance(60)
    assert "sim_retry" not in _finish(ledger)  # the late call opens nothing ...
    assert ledger.last_event is None  # ... and was no retry
    assert ledger.precheck("pick_and_place", A) == USED and _begin(ledger) == USED  # the slot is used, not dropped


def test_a_late_call_that_succeeds_clears_its_slot(ledger, clock):
    _refused_first(ledger, clock, took=130.0)
    clock.advance(1.0)
    assert _begin(ledger) is None
    clock.advance(30)
    ledger.finish("pick_and_place", A, {"ok": True}, exc_type=None, refused_grasps=(), retry=None, **CLEAR)
    assert ledger.precheck("pick_and_place", A) is None and _begin(ledger) is None


@pytest.mark.parametrize("took,allowed", [(230.0, True), (230.5, False)])
def test_the_reserve_is_a_parameter(clock, took, allowed):
    L = SimRetryLedger(reserve_s=100, clock=clock)
    assert ("sim_retry" in _refused_first(L, clock, took=took)) is allowed
    assert isinstance(_begin(L), RetryCall) is allowed


def test_window_and_turn_budget_follow_the_native_turn_budget():
    budget = runpy.run_path(str(REPO / "scripts" / "native_turn_budget.py"))
    assert DEFAULT_WINDOW_S == budget["NATIVE_TURN_TIMEOUT_S"]
    assert TURN_BUDGET_S == budget["NATIVE_TURN_TIMEOUT_S"] - budget["AGENT_EXIT_GRACE_S"]
    # the largest reserve still leaves the first attempt the window's margin (360 - 330 s)
    assert MAX_RESERVE_S == TURN_BUDGET_S - (DEFAULT_WINDOW_S - TURN_BUDGET_S) == 300
    assert DEFAULT_RESERVE_S == 200


@pytest.mark.parametrize("bad", [359.0, 0, -1, math.nan, math.inf, True, "360", None])
def test_the_window_is_never_shorter_than_one_turn(bad):
    with pytest.raises(ValueError):
        SimRetryLedger(window_s=bad)


@pytest.mark.parametrize("bad", [0, -1, math.nan, math.inf, True, "200", None, MAX_RESERVE_S, TURN_BUDGET_S, 400])
def test_the_reserve_must_be_positive_and_leave_the_first_attempt_time(bad):  # Minor 9
    with pytest.raises(ValueError):
        SimRetryLedger(reserve_s=bad)


def test_longer_windows_and_other_reserves_are_accepted():
    assert SimRetryLedger(window_s=360).window_s == 360
    assert SimRetryLedger(window_s=900.5, reserve_s=150).reserve_s == 150
    assert SimRetryLedger(reserve_s=MAX_RESERVE_S - 0.5).reserve_s == MAX_RESERVE_S - 0.5


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


@pytest.mark.parametrize("where", ["precheck", "begin", "finish"])
def test_a_bookkeeping_fault_turns_the_ledger_off(clock, where):
    """Fail closed: after an internal error the ledger never refuses, grants or
    opens anything again (every call runs as without it), so a fault can never
    re-arm a used retry; the faulting call's result is left untouched."""
    L = SimRetryLedger(clock=clock)
    _refused_first(L, clock)
    r = _begin(L)
    clock.advance(10)
    _finish(L, retry=r)  # green cube: used
    assert L.precheck("pick_and_place", A) == USED
    if where == "finish":
        assert _begin(L, args=LEMON) is None
        clock.advance(10)
        assert _finish(L, args=LEMON, grasps=5) == REFUSED  # not a sequence: the result untouched
    else:
        def broken():
            raise RuntimeError("clock fault")
        L._clock = broken
        assert (L.precheck("pick_and_place", LEMON) if where == "precheck" else _begin(L, args=LEMON)) is None
        L._clock = clock
    assert L.precheck("pick_and_place", A) is None
    assert _begin(L) is None
    clock.advance(10)
    assert "sim_retry" not in _finish(L)
    assert _begin(L) is None
    L.note_motion("move_home", True)
    L.clear("reset_scene")
    assert _refused_first(L, clock) == REFUSED and _begin(L) is None


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
#: spec §4.4's simulated arm types, written out here (not the code under test)
SIM_TYPES = {"isaac", "mujoco", "mock"}


def _arm_types(cfg) -> list:
    """The `type` of every arm the rig builds: cfg.arms when present, else cfg.arm."""
    arms = cfg.get("arms") or [cfg.arm._data]
    return [arm.get("type") for arm in arms]


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
def test_only_sim_profiles_get_a_ledger(switched_on, arm):  # Minor 9: the oracle is the arm type
    cfg = switched_on(arm=arm)
    types = _arm_types(cfg)
    simulated = bool(types) and all(kind in SIM_TYPES for kind in types)
    assert isinstance(ledger_for(cfg), SimRetryLedger) is simulated, types
    assert not (simulated and arm in HARDWARE)


@pytest.mark.parametrize("arm", HARDWARE)
def test_no_setting_gives_a_hardware_profile_a_ledger(switched_on, arm, monkeypatch):
    cfg = switched_on(arm=arm)
    assert not set(_arm_types(cfg)) <= SIM_TYPES
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
                                       ("sim_retry_reserve_s", True), ("sim_retry_reserve_s", None),
                                       ("sim_retry_reserve_s", 300), ("sim_retry_reserve_s", 330)])
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
