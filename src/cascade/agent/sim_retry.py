"""Sim-only bounded retry ledger (ASPIRE spec §4.5; preflight R6-R10, R28; A6-fix rulings).

SIMULATION ONLY. On a sim profile with the retry switched on (`ledger_for`), an
order the robot REFUSED -- a safety gate, slow or changed perception, no
plannable grasp -- may be repeated once by the agent.

A goal (`goal_of`) is the order's object, destination and arm, each normalised
to a set of words: NFKC, casefold, runs of letters and digits (punctuation,
hyphens and "_" separate words), determiners (a, an, the, this, that, these,
those) dropped, no stemming. Two goals are the same (`same_goal`) when one
object's words contain the other's, one destination's words contain the
other's (an absent destination matches any) and the arms are equal or one is
absent. An object without words is no goal. So a rephrased order ("the open
box.", "green_cube", "cube") is the same goal; "pink cube" or another
destination is not.

The ledger keeps one slot per goal, in one of three states:

- open: a qualifying refusal of a goal with no slot opened it (`finish`). The
  next same-goal call is THE retry: `begin` marks the slot used before anything
  moves and hands the call the refused grasps to exclude and whether the refused
  attempt left the arm in place (the runtime homes it first). A retry that would
  start too late for the agent's turn (R10: first call's start + reserve past
  TURN_BUDGET_S) is not granted: the slot is marked used and that call runs as
  without the ledger;
- used: every later same-goal call is refused before any motion, by `precheck`
  and again by `begin` itself (under the same lock as the slot's state, so the
  bound never depends on the caller's locking);
- spent: a halt, an e-stop latch, a hold, a pending contact or release episode
  or a blocked arm touched the goal: its calls run exactly as without the
  ledger -- no retry, no refusal, and nothing opens.

A goal is touched when one of its calls starts held, latched, with an episode
pending or with the arm blocked, or ends failed while held or with an episode
pending, or ends halted or latched; every open slot is touched when any motion
call starts latched or ends halted or latched. A retry the operator halts keeps
its used slot, so nothing moves again on the ledger's behalf.

A slot ends only when its goal succeeds, `reset_scene` succeeds (`clear`) or its
window (one native turn, R9) ends; a used slot ends on success only when the call
that used it succeeds. A call with another goal never touches a slot, so per goal
at most one retry is granted within the window, however the agent interleaves
goals, and an order that is not spent gets at most two motion calls: the first
attempt and its retry. Halts and e-stops never open, use or re-arm a slot.

Refusals are classified by the EXACT class name of the exception that refused
the call (R6), never by error text: subclasses such as MotionHalted or
ContactDetected never qualify, and error text can only exclude a result, never
admit one. The ledger never edits agent text: it adds the `sim_retry` field and
returns the used-slot refusal; the runtime composes `next_action` from the texts
below. Bookkeeping never fails a call: an internal error turns the ledger off
for good (every slot dropped; no refusal, no retry, nothing opens again: every
call runs as without it) and leaves the result untouched.

A real or unknown profile never gets a ledger: `ledger_for` returns one only for
`profile_class(cfg) == "sim"` with `sim_retry_enabled(cfg)` and the
observed-finger gate on (R8), and None for anything it cannot read.
"""
from __future__ import annotations

import logging
import math
import os
import re
import threading
import time
import unicodedata
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass

from ..sim_profile import profile_class, sim_retry_enabled

logger = logging.getLogger(__name__)

#: skills whose refusal can open a slot, and the argument that names their object
RETRY_SKILLS = frozenset({"pick_and_place", "grasp_object"})
_OBJECT_ARG = {"pick_and_place": "object", "grasp_object": "label"}
#: a counted, successful call of these brings the arm out of a refused grasp pose
_HOME_SKILLS = frozenset({"move_home", "move_to_rest"})

#: exact exception class name -> refusal class (R6). Subclasses are absent on
#: purpose: MotionHalted, ContactDetected, GripperFault and AttachmentInvalid
#: are SafetyViolations a retry must never follow.
REFUSAL_CLASSES = {
    "SafetyViolation": "safety_refusal",
    "SlowPerceptionError": "perception_refusal",
    "SceneChangedError": "perception_refusal",
    "ReanchorError": "perception_refusal",
    "PlanningBudgetExceeded": "planning_refusal",
    "NoExecutableGrasp": "planning_refusal",
}
#: error text that EXCLUDES a result whatever its exception: a halt, a contact, a
#: latched stop or an unfinished episode folded into a refusal's text, and the
#: harness's own callback faults (safety/harness.py), which a retry repeats
_NEVER_QUALIFY_TEXT = ("MotionHalted", "ContactDetected", "GripperFault", "AttachmentInvalid",
                       "e-stop latched", "unfinished contact episode", "unfinished release episode",
                       "motion safety callback failed", "ambiguous motion preflight callbacks")

#: scripts/native_turn_budget.py: one agent turn, and the grace its final reply needs
NATIVE_TURN_TIMEOUT_S = 360.0
AGENT_EXIT_GRACE_S = 30.0
#: R9: a slot expires this long after it opened; never shorter than one turn, so
#: a slot cannot expire inside the order it bounds
DEFAULT_WINDOW_S = NATIVE_TURN_TIMEOUT_S
MIN_WINDOW_S = NATIVE_TURN_TIMEOUT_S
#: R10: the retry is granted only if the time since the first attempt's start plus
#: this reserve fits TURN_BUDGET_S (re-ruled from A11's measured p95)
DEFAULT_RESERVE_S = 200.0
TURN_BUDGET_S = NATIVE_TURN_TIMEOUT_S - AGENT_EXIT_GRACE_S
#: a larger reserve leaves the first attempt less than the window's margin
#: (window - turn budget = 30 s), so the ledger would silently never grant
MAX_RESERVE_S = TURN_BUDGET_S - (DEFAULT_WINDOW_S - TURN_BUDGET_S)

#: Agent texts, spec §4.5 verbatim (`{skill}` / `{goal}` formatted by the caller).
RETRY_ALLOWED = (
    "SIMULATION ONLY: you may make ONE more attempt for this order: call {skill} again with the same "
    "arguments. That call first returns the arm home itself (move_home is optional), looks again and uses a "
    "different grasp. If the user said 'then stop', report the failure instead. Never rename the object or "
    "destination or invent coordinates.")
RETRY_USED = "The one simulation retry for this order is used: report the failure; do not call it again."
RETRY_EXHAUSTED = "The one simulation retry for {goal} is used"

#: words a goal never compares (A6-fix I1)
_DETERMINERS = frozenset({"a", "an", "the", "this", "that", "these", "those"})
_WORD = re.compile(r"[^\W_]+")


def _word_set(value) -> frozenset[str] | None:
    """The words of `value` a goal compares; None when `value` is not a string."""
    if not isinstance(value, str):
        return None
    text = unicodedata.normalize("NFKC", value).casefold()
    return frozenset(word for word in _WORD.findall(text) if word not in _DETERMINERS)


def _optional_word_set(value) -> frozenset[str] | None:
    """`_word_set` of an optional argument: empty when absent, None when not a string."""
    return frozenset() if value is None else _word_set(value)


def _nested(a: frozenset, b: frozenset) -> bool:
    return a <= b or b <= a


@dataclass(frozen=True)
class Goal:
    """What an order asks for (R28, A6-fix I1): the object's words, the destination's
    words and the arm's words (empty = not given). `name` is for display only: the
    object as the agent said it, casefolded and whitespace-collapsed."""

    name: str
    words: frozenset = frozenset()
    destination: frozenset = frozenset()
    arm: frozenset = frozenset()

    def __str__(self) -> str:
        return self.name


def goal_of(name, args) -> Goal | None:
    """The goal of a `pick_and_place(object, destination)` or `grasp_object(label)`
    call, or None: other skills, an object without words, a non-string argument."""
    key = _OBJECT_ARG.get(name) if isinstance(name, str) else None
    if key is None or not isinstance(args, Mapping):
        return None
    obj = args.get(key)
    words = _word_set(obj)
    destination = _optional_word_set(args.get("destination")) if name == "pick_and_place" else frozenset()
    arm = _optional_word_set(args.get("arm"))
    if not words or destination is None or arm is None:
        return None
    return Goal(" ".join(obj.casefold().split()), words, destination, arm)


def same_object(a, b) -> bool:
    """One object name's word set contains the other's ("cube" / "the green cube"):
    a renamed retry is the same order; "pink cube" / "green cube" are not."""
    wa, wb = _word_set(a), _word_set(b)
    return bool(wa) and bool(wb) and _nested(wa, wb)


def same_goal(a, b) -> bool:
    """Same order: object words nested, destination words nested, arms equal or
    one absent. Raises TypeError unless both are `Goal`s (from `goal_of`)."""
    if not isinstance(a, Goal) or not isinstance(b, Goal):
        raise TypeError(f"same_goal compares two Goal values, got {type(a).__name__} and {type(b).__name__}")
    return (bool(a.words) and bool(b.words) and _nested(a.words, b.words)
            and _nested(a.destination, b.destination)
            and (not a.arm or not b.arm or a.arm == b.arm))


def refusal_class(exc_type, result, *, held: bool, latched: bool, halted: bool, episode_pending: bool) -> str | None:
    """The refusal class ("safety_refusal" | "perception_refusal" |
    "planning_refusal") of a call the robot refused, else None (no retry).

    `exc_type`: the exact class NAME of the exception that refused the call (the
    runtime records it; error text has no reliable prefix). Qualifies only when
    the call failed, nothing is held, no e-stop or halt is latched, no halt arrived
    during the call, no contact or release episode is pending, the motion was not
    blocked, no contact was measured, the step is not stuck, no completed motion
    had its postcondition refuted, and the error text names no halt, contact,
    latch, episode or harness callback fault."""
    if not isinstance(result, Mapping) or result.get("ok"):
        return None
    if held or latched or halted or episode_pending:
        return None
    if result.get("motion_blocked") or "contact" in result or result.get("outcome") == "stuck":
        return None
    if result.get("action_ok") or result.get("self_reported_ok"):
        return None
    error = str(result.get("error") or "")
    if any(text in error for text in _NEVER_QUALIFY_TEXT):
        return None
    if not isinstance(exc_type, str):
        return None
    return REFUSAL_CLASSES.get(exc_type)


@dataclass(frozen=True)
class RetryCall:
    """Handed to THE retry call by `begin`: what to exclude and whether to home first."""

    goal: str
    refused_grasps: tuple
    arm_left_in_place: bool
    refusal: str


_OPEN, _USED, _SPENT = "open", "used", "spent"
#: several slots can match one goal (word containment): the most restrictive
#: state decides, then the earliest slot
_RESTRICTION = {_USED: 0, _SPENT: 1, _OPEN: 2}


@dataclass
class _Slot:
    goal: Goal
    opened_at: float
    first_started_at: float
    state: str = _OPEN
    refusal: str = ""
    refused_grasps: tuple = ()
    arm_left_in_place: bool = False


@dataclass
class _Call:
    """The goal-bearing call between its `begin` and its `finish`."""

    goal: Goal
    started_at: float
    blocked: bool
    retry: RetryCall | None = None
    slot: _Slot | None = None  # the slot this call used (its retry, or a late call under R10)


def _seconds(label: str, value, *, minimum: float, inclusive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number of seconds, got {value!r}")
    if value < minimum or (not inclusive and value == minimum):
        bound = ">=" if inclusive else ">"
        raise ValueError(f"{label} must be {bound} {minimum:g} s, got {value!r}")
    return value


class SimRetryLedger:
    """Per-runtime retry slots (spec §4.5). Thread-safe; never touches the arm.

    The runtime calls, around each outermost motion call: `precheck` (a refusal to
    return before any motion), `begin` (the RetryCall when the call is the retry,
    the same refusal as `precheck` when the goal's retry is used, else None),
    `finish` (sets `sim_retry` on the result), `note_motion` (a counted home), and
    `clear` (reset_scene). `last_event` describes the last finished retry:
    `{t, kind: "retry", goal, outcome: "succeeded"|"failed"|"halted", refusal}`.
    """

    def __init__(self, *, window_s: float = DEFAULT_WINDOW_S, reserve_s: float = DEFAULT_RESERVE_S,
                 clock: Callable[[], float] = time.monotonic):
        self.window_s = _seconds("sim_retry_window_s", window_s, minimum=MIN_WINDOW_S, inclusive=True)
        self.reserve_s = _seconds("sim_retry_reserve_s", reserve_s, minimum=0.0, inclusive=False)
        if self.reserve_s >= MAX_RESERVE_S:
            raise ValueError(f"sim_retry_reserve_s must be < {MAX_RESERVE_S:g} s (the first attempt needs "
                             f"time in the turn), got {reserve_s!r}")
        self._clock = clock
        self._lock = threading.RLock()
        self._slots: list[_Slot] = []
        self._call: _Call | None = None
        self._off = False
        self.last_event: dict | None = None

    # ── entry points ─────────────────────────────────────────────────────

    def precheck(self, name, args) -> dict | None:
        """A refusal to return before any motion when this goal's retry is used, else None."""
        with self._lock:
            try:
                goal = goal_of(name, args)
                if goal is None:
                    return None
                self._expire(self._clock())
                slot = self._slot_for(goal)
                if slot is None or slot.state != _USED:
                    return None
                return self._used_refusal(slot)
            except Exception:  # noqa: BLE001 -- bookkeeping never fails a call
                self._fail_closed("precheck")
                return None

    def begin(self, name, args, *, held: bool, latched: bool, episode_pending: bool,
              motion_blocked: bool) -> RetryCall | dict | None:
        """Start a motion call. Returns the refusal dict (as `precheck`) when this
        goal's retry is used: the caller returns it before any motion. Returns the
        RetryCall when the call is THE retry of an open slot (the slot is used from
        here on), else None (the call runs as without the ledger). Under a hold, a
        latch, a pending episode or a blocked arm the goal is spent instead, and a
        latch spends every open slot. A retry that would start too late for the
        turn (R10) is not granted: the slot is used and the call runs as without
        the ledger."""
        with self._lock:
            self._call = None
            try:
                now = self._clock()
                self._expire(now)
                if latched:
                    self._spend_open()  # a halt or e-stop pending at any motion call
                goal = goal_of(name, args)
                if goal is None:
                    return None
                slot = self._slot_for(goal)
                if slot is not None and slot.state == _USED:
                    return self._used_refusal(slot)
                blocked = bool(held or latched or episode_pending or motion_blocked)
                call = self._call = _Call(goal, now, blocked)
                if blocked:
                    self._spend(goal, now)
                    return None
                if slot is None or slot.state != _OPEN:
                    return None
                slot.state = _USED
                call.slot = slot
                if now - slot.first_started_at + self.reserve_s > TURN_BUDGET_S:
                    return None  # R10: the retry would overrun the agent's turn
                call.retry = RetryCall(str(slot.goal), slot.refused_grasps, slot.arm_left_in_place, slot.refusal)
                return call.retry
            except Exception:  # noqa: BLE001
                self._fail_closed("begin")
                return None

    def finish(self, name, args, result, *, exc_type, held: bool, latched: bool, halted: bool,
               episode_pending: bool, refused_grasps, retry: RetryCall | None):
        """End a call: returns `result`, with `sim_retry` set when a slot opened
        (`{allowed: true, goal, refusal}`) or a retry failed (`{allowed: false,
        used: true}`). `halted`: a halt arrived during the call; `latched`: an
        e-stop or halt is latched now; `refused_grasps`: the attempted candidates
        of the refused grasp (exclusions for the retry)."""
        with self._lock:
            call, self._call = self._call, None
            if self._off:
                return result  # the only way a slot opens: off means nothing ever opens again
            try:
                return self._finish(name, args, result, call, exc_type=exc_type, held=held, latched=latched,
                                    halted=halted, episode_pending=episode_pending,
                                    refused_grasps=refused_grasps, retry=retry)
            except Exception:  # noqa: BLE001
                self._fail_closed("finish")
                return result

    def note_motion(self, name: str, ok: bool) -> None:
        """A counted, successful move_home / move_to_rest: the arm is no longer
        where a refused attempt left it, so the retry needs no home first."""
        if ok is not True or name not in _HOME_SKILLS:
            return
        with self._lock:
            for slot in self._slots:
                slot.arm_left_in_place = False

    def clear(self, reason: str) -> None:
        """Drop every slot, open, used or spent (reset_scene succeeded, or any other reason)."""
        with self._lock:
            if self._slots:
                logger.info("sim retry: %d slot(s) cleared: %s", len(self._slots), reason)
            self._slots = []

    # ── internals ────────────────────────────────────────────────────────

    def _finish(self, name, args, result, call: _Call | None, *, exc_type, held, latched, halted,
                episode_pending, refused_grasps, retry):
        now = self._clock()
        goal = goal_of(name, args)
        if call is not None and call.goal != goal:
            call = None
        if retry is None and call is not None:
            retry = call.retry
        stopped = bool(halted or latched)
        if stopped:
            # a halt or e-stop at any point, whatever the result: every open slot and
            # this call's goal are spent (a used slot stays used: the halted order
            # never moves again on the ledger's behalf)
            self._spend_open()
            if goal is not None:
                self._spend(goal, now)
        if goal is None or not isinstance(result, MutableMapping):
            return result
        own = call.slot if call is not None else None
        if own is None and retry is not None:
            own = self._slot_for(goal)
        if retry is not None:
            outcome = "halted" if stopped else "succeeded" if result.get("ok") else "failed"
            if outcome == "succeeded":
                self._succeeded(goal, own)
            self.last_event = {"t": time.time(), "kind": "retry", "goal": retry.goal, "outcome": outcome,
                               "refusal": retry.refusal}
            if outcome == "failed":
                result["sim_retry"] = {"allowed": False, "used": True}
            return result
        if result.get("ok") and not stopped:
            self._succeeded(goal, own)
            return result
        if held or episode_pending:
            self._spend(goal, now)  # a hold or a pending episode touched this goal
            return result
        if self._slot_for(goal) is not None or call is None or call.blocked:
            return result  # a slot exists, no begin was seen, or the call started under a latch/hold
        refusal = refusal_class(exc_type, result, held=held, latched=latched, halted=halted,
                                episode_pending=episode_pending)
        if refusal is None:
            return result
        if now - call.started_at + self.reserve_s > TURN_BUDGET_S:
            return result  # the retry would overrun the agent's turn (R10)
        grasps = () if refused_grasps is None else tuple(refused_grasps)
        self._slots.append(_Slot(goal, now, call.started_at, _OPEN, refusal, grasps,
                                 bool(result.get("arm_left_in_place"))))
        result["sim_retry"] = {"allowed": True, "goal": str(goal), "refusal": refusal}
        return result

    def _slot_for(self, goal: Goal) -> _Slot | None:
        """The slot that decides for `goal`: the most restrictive matching one."""
        matching = [slot for slot in self._slots if same_goal(slot.goal, goal)]
        return min(matching, key=lambda slot: (_RESTRICTION[slot.state], slot.opened_at), default=None)

    def _spend(self, goal: Goal, now: float) -> None:
        """A halt, latch, hold or pending episode touched `goal`: its open slots are
        spent, a used one stays used, and a goal without a slot gets a spent one."""
        matching = [slot for slot in self._slots if same_goal(slot.goal, goal)]
        for slot in matching:
            if slot.state == _OPEN:
                slot.state = _SPENT
        if not matching:
            self._slots.append(_Slot(goal, now, now, _SPENT))

    def _spend_open(self) -> None:
        for slot in self._slots:
            if slot.state == _OPEN:
                slot.state = _SPENT

    def _succeeded(self, goal: Goal, own: _Slot | None) -> None:
        """The goal succeeded: drop the slot this call used and every open or spent
        slot of the goal. Another order's used slot stays (only its own call, a
        reset or its window ends it)."""
        self._slots = [slot for slot in self._slots
                       if slot is not own and (slot.state == _USED or not same_goal(slot.goal, goal))]

    def _used_refusal(self, slot: _Slot) -> dict:
        return {"ok": False, "error": RETRY_EXHAUSTED.format(goal=slot.goal),
                "sim_retry": {"allowed": False, "used": True}}

    def _expire(self, now: float) -> None:
        self._slots = [slot for slot in self._slots if now - slot.opened_at < self.window_s]

    def _fail_closed(self, where: str) -> None:
        """Off for good: every slot dropped and `finish` never opens one again, so
        `precheck` and `begin` have nothing left to refuse or grant."""
        logger.exception("sim retry %s failed; the ledger is off (no retry, no refusal)", where)
        self._off = True
        self._slots = []
        self._call = None


def ledger_for(cfg) -> SimRetryLedger | None:
    """The runtime's ledger: only for a sim profile with `agent.sim_retry: true`,
    the kill switch off and the observed-finger gate on; None otherwise, and None
    for any config it cannot read or a bad `agent.sim_retry_window_s` /
    `agent.sim_retry_reserve_s`. A real or unknown profile never gets one."""
    try:
        if profile_class(cfg) != "sim" or not sim_retry_enabled(cfg):
            return None
        if os.environ.get("CASCADE_OBSERVED_FINGER_GATE") != "1":
            return None  # R8: with the gate off a retry has no refused grasps to exclude
        agent = cfg.get("agent")
        return SimRetryLedger(window_s=agent.get("sim_retry_window_s", DEFAULT_WINDOW_S),
                              reserve_s=agent.get("sim_retry_reserve_s", DEFAULT_RESERVE_S))
    except Exception as exc:  # noqa: BLE001 -- fail closed: no ledger, no retry
        logger.warning("sim retry off: %s: %s", type(exc).__name__, exc)
        return None
