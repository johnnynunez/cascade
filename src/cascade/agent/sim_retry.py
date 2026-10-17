"""Sim-only bounded retry ledger (ASPIRE spec §4.5; preflight R6, R7, R9, R10, R28).

SIMULATION ONLY. On a sim profile with the retry switched on (`ledger_for`), an
order the robot REFUSED -- a safety gate, slow or changed perception, no
plannable grasp -- may be repeated once by the agent. The ledger keeps one slot
per goal (object words + destination + arm):

- a qualifying refusal of a goal with no slot opens it (`finish`);
- the next same-goal call is THE retry: `begin` marks the slot used before
  anything moves and hands the call the refused grasps to exclude and whether
  the refused attempt left the arm in place (the runtime homes it first);
- every later same-goal call is refused before any motion (`precheck`) until
  the goal succeeds, `reset_scene` succeeds, a motion call names another goal,
  or the slot's window (one native turn) ends.

So one order gets at most two motion calls: the first attempt and its retry.

Halts and e-stops never open, use or re-arm a slot: a call that ends halted or
latched opens nothing and clears every open slot; a retry cannot start under a
latch, a hold or a pending episode (the call then runs as without the ledger and
its refusal opens nothing); a retry the operator halts keeps its used slot, so
nothing moves again on the ledger's behalf.

Refusals are classified by the EXACT class name of the exception that refused
the call (R6), never by error text: subclasses such as MotionHalted or
ContactDetected never qualify, and error text can only exclude a result, never
admit one. The ledger never edits agent text: it adds the `sim_retry` field and
returns the precheck refusal; the runtime composes `next_action` from the texts
below. Bookkeeping never fails a call: an internal error drops every slot (no
retry) and leaves the result untouched.

A real or unknown profile never gets a ledger: `ledger_for` returns one only for
`profile_class(cfg) == "sim"` with `sim_retry_enabled(cfg)` and the
observed-finger gate on (R8), and None for anything it cannot read.
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
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
#: error text that EXCLUDES a result whatever its exception (a halt, a contact,
#: a latched stop or an unfinished episode folded into a refusal's text)
_NEVER_QUALIFY_TEXT = ("MotionHalted", "ContactDetected", "GripperFault", "AttachmentInvalid",
                       "e-stop latched", "unfinished contact episode", "unfinished release episode")

#: scripts/native_turn_budget.py: one agent turn, and the grace its final reply needs
NATIVE_TURN_TIMEOUT_S = 360.0
AGENT_EXIT_GRACE_S = 30.0
#: R9: a slot expires this long after it opened; never shorter than one turn, so
#: a slot cannot expire inside the order it bounds
DEFAULT_WINDOW_S = NATIVE_TURN_TIMEOUT_S
MIN_WINDOW_S = NATIVE_TURN_TIMEOUT_S
#: R10: the retry is granted only if the first attempt's duration plus this
#: reserve fits TURN_BUDGET_S (re-ruled from A11's measured p95)
DEFAULT_RESERVE_S = 200.0
TURN_BUDGET_S = NATIVE_TURN_TIMEOUT_S - AGENT_EXIT_GRACE_S

#: Agent texts, spec §4.5 verbatim (`{skill}` / `{goal}` formatted by the caller).
RETRY_ALLOWED = (
    "SIMULATION ONLY: you may make ONE more attempt for this order: call {skill} again with the same "
    "arguments. That call first returns the arm home itself (move_home is optional), looks again and uses a "
    "different grasp. If the user said 'then stop', report the failure instead. Never rename the object or "
    "destination or invent coordinates.")
RETRY_USED = "The one simulation retry for this order is used: report the failure; do not call it again."
RETRY_EXHAUSTED = "The one simulation retry for {goal} is used"


def _words(value) -> str | None:
    """Casefolded, whitespace-collapsed text; None when `value` is not a non-blank string."""
    if not isinstance(value, str):
        return None
    return " ".join(value.casefold().split()) or None


def _optional_words(value) -> str | None:
    """`_words` of an optional argument: "" when absent or blank, None when not a string."""
    if value is None:
        return ""
    if not isinstance(value, str):
        return None
    return " ".join(value.casefold().split())


@dataclass(frozen=True)
class Goal:
    """What an order asks for (R28): the object's words, the destination and the arm,
    each casefolded and whitespace-collapsed ("" = not given). Shown as its object words."""

    name: str
    destination: str = ""
    arm: str = ""

    def __str__(self) -> str:
        return self.name


def goal_of(name, args) -> Goal | None:
    """The goal of a `pick_and_place(object, destination)` or `grasp_object(label)`
    call, or None: other skills, a missing or blank object, a non-string argument."""
    key = _OBJECT_ARG.get(name) if isinstance(name, str) else None
    if key is None or not isinstance(args, Mapping):
        return None
    obj = _words(args.get(key))
    destination = _optional_words(args.get("destination")) if name == "pick_and_place" else ""
    arm = _optional_words(args.get("arm"))
    if obj is None or destination is None or arm is None:
        return None
    return Goal(obj, destination, arm)


def same_object(a, b) -> bool:
    """One object name's word set contains the other's ("cube" / "green cube"):
    a renamed retry is the same order; "pink cube" / "green cube" are not."""
    wa, wb = _words(a), _words(b)
    if wa is None or wb is None:
        return False
    sa, sb = frozenset(wa.split()), frozenset(wb.split())
    return sa <= sb or sb <= sa


def same_goal(a, b) -> bool:
    """Same object words (containment), same destination, same arm (R28)."""
    if not isinstance(a, Goal) or not isinstance(b, Goal):
        return False
    return same_object(a.name, b.name) and a.destination == b.destination and a.arm == b.arm


def refusal_class(exc_type, result, *, held: bool, latched: bool, halted: bool, episode_pending: bool) -> str | None:
    """The refusal class ("safety_refusal" | "perception_refusal" |
    "planning_refusal") of a call the robot refused, else None (no retry).

    `exc_type`: the exact class NAME of the exception that refused the call (the
    runtime records it; error text has no reliable prefix). Qualifies only when
    the call failed, nothing is held, no e-stop or halt is latched, no halt arrived
    during the call, no contact or release episode is pending, the motion was not
    blocked, no contact was measured, the step is not stuck, no completed motion
    had its postcondition refuted, and the error text names no halt, contact,
    latch or episode."""
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


@dataclass
class _Slot:
    goal: Goal
    opened_at: float
    refusal: str
    refused_grasps: tuple
    arm_left_in_place: bool
    used: bool = False


@dataclass
class _Call:
    """The goal-bearing call between its `begin` and its `finish`."""

    goal: Goal
    started_at: float
    blocked: bool
    retry: RetryCall | None = None


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
    return before any motion), `begin` (the RetryCall when the call is the retry),
    `finish` (sets `sim_retry` on the result), `note_motion` (a counted home), and
    `clear` (reset_scene). `last_event` describes the last finished retry:
    `{t, kind: "retry", goal, outcome: "succeeded"|"failed"|"halted", refusal}`.
    """

    def __init__(self, *, window_s: float = DEFAULT_WINDOW_S, reserve_s: float = DEFAULT_RESERVE_S,
                 clock: Callable[[], float] = time.monotonic):
        self.window_s = _seconds("sim_retry_window_s", window_s, minimum=MIN_WINDOW_S, inclusive=True)
        self.reserve_s = _seconds("sim_retry_reserve_s", reserve_s, minimum=0.0, inclusive=False)
        self._clock = clock
        self._lock = threading.RLock()
        self._slots: list[_Slot] = []
        self._call: _Call | None = None
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
                slot = self._find(goal)
                if slot is None or not slot.used:
                    return None
                return {"ok": False, "error": RETRY_EXHAUSTED.format(goal=slot.goal),
                        "sim_retry": {"allowed": False, "used": True}}
            except Exception:  # noqa: BLE001 -- bookkeeping never fails a call
                self._fail_closed("precheck")
                return None

    def begin(self, name, args, *, held: bool, latched: bool, episode_pending: bool,
              motion_blocked: bool) -> RetryCall | None:
        """Start a motion call. Returns the RetryCall when it is THE retry of an
        open slot (the slot is used from here on), else None. A goal-bearing call
        clears the slots of other goals; any call under an e-stop or halt latch
        clears every open slot. Under a hold, a latch, a pending episode or a
        blocked arm, the goal's open slot is cleared instead of used and this call
        runs as without the ledger (its refusal opens nothing)."""
        with self._lock:
            try:
                self._call = None
                now = self._clock()
                self._expire(now)
                if latched:
                    # a halt or e-stop pending at any motion call: every open slot is cleared
                    self._slots = [slot for slot in self._slots if slot.used]
                goal = goal_of(name, args)
                if goal is None:
                    return None
                self._slots = [slot for slot in self._slots if same_goal(slot.goal, goal)]
                blocked = bool(held or latched or episode_pending or motion_blocked)
                self._call = _Call(goal, now, blocked)
                slot = self._find(goal)
                if slot is None or slot.used:
                    return None
                if blocked:
                    self._slots.remove(slot)
                    return None
                slot.used = True
                retry = RetryCall(str(slot.goal), slot.refused_grasps, slot.arm_left_in_place, slot.refusal)
                self._call.retry = retry
                return retry
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
        """Drop every slot (reset_scene succeeded, or any other reason)."""
        with self._lock:
            if self._slots:
                logger.info("sim retry: %d slot(s) cleared: %s", len(self._slots), reason)
            self._slots = []

    # ── internals ────────────────────────────────────────────────────────

    def _finish(self, name, args, result, call: _Call | None, *, exc_type, held, latched, halted,
                episode_pending, refused_grasps, retry):
        if not isinstance(result, MutableMapping):
            return result
        stopped = bool(halted or latched)
        if stopped:
            # a halt or e-stop at any point: every open slot is cleared (a used one
            # stays used, so the halted order never moves again on the ledger's behalf)
            self._slots = [slot for slot in self._slots if slot.used]
        goal = goal_of(name, args)
        if goal is None:
            return result
        if call is not None and call.goal != goal:
            call = None
        if retry is None and call is not None:
            retry = call.retry
        slot = self._find(goal)
        if retry is not None:
            outcome = "halted" if stopped else "succeeded" if result.get("ok") else "failed"
            if outcome == "succeeded" and slot is not None:
                self._slots.remove(slot)
            self.last_event = {"t": time.time(), "kind": "retry", "goal": retry.goal, "outcome": outcome,
                               "refusal": retry.refusal}
            if outcome == "failed":
                result["sim_retry"] = {"allowed": False, "used": True}
            return result
        if result.get("ok"):
            if slot is not None:
                self._slots.remove(slot)  # the goal succeeded
            return result
        if slot is not None or call is None or call.blocked:
            return result  # a slot exists, no begin was seen, or the call started under a latch/hold
        refusal = refusal_class(exc_type, result, held=held, latched=latched, halted=halted,
                                episode_pending=episode_pending)
        if refusal is None:
            return result
        now = self._clock()
        if now - call.started_at + self.reserve_s > TURN_BUDGET_S:
            return result  # the retry would overrun the agent's turn (R10)
        grasps = () if refused_grasps is None else tuple(refused_grasps)
        self._slots.append(_Slot(goal, now, refusal, grasps, bool(result.get("arm_left_in_place"))))
        result["sim_retry"] = {"allowed": True, "goal": str(goal), "refusal": refusal}
        return result

    def _find(self, goal: Goal) -> _Slot | None:
        return next((slot for slot in self._slots if same_goal(slot.goal, goal)), None)

    def _expire(self, now: float) -> None:
        self._slots = [slot for slot in self._slots if now - slot.opened_at < self.window_s]

    def _fail_closed(self, where: str) -> None:
        logger.exception("sim retry %s failed; every slot dropped (no retry)", where)
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
