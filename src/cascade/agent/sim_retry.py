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
absent. An arm named "", "default" or "primary" is the runtime's primary arm
(`SkillRuntime._select_arm`), so it counts as absent; the runtime passes the
resolved arm's name. An object without words is no goal. So a rephrased order
("the open box.", "green_cube", "cube") is the same goal; "pink cube" or
another destination is not.

The ledger keeps one slot per goal, in one of four states:

- open: a qualifying refusal of a goal with no slot opened it (`finish`), when
  the refused call's time plus the retry's reserve and the agent's think time
  (RETRY_THINK_S) fit the turn (R10). The next same-goal call is THE retry:
  `begin` marks the slot used before anything moves and hands the call the
  refused grasps to exclude and whether the refused attempt left the arm in
  place (the runtime homes it first). A retry that would start too late for the
  agent's turn (first call's start + reserve past TURN_BUDGET_S) is refused
  before any motion ("the retry window for this turn has passed") and its slot
  is used: a call the agent was told is the retry never runs without the
  promised home. A call that matches two or more open slots (a rename that fits
  several orders) is no retry: every matched slot is used and the call runs as
  without the ledger;
- used: every later same-goal call is refused before any motion, by `precheck`
  and again by `begin` itself (under the same lock as the slot's state, so the
  bound never depends on the caller's locking);
- stopped: the operator stopped the rig (a halt or an e-stop, pending or
  cleared again) after the refusal, before the retry: like a used slot, every
  later same-goal call -- the one the agent's rules would make on the earlier
  permission, and any after it, whatever a note or the agent makes of that
  permission -- is refused before any motion, by `precheck` and by `begin`,
  with a text that grants nothing and says how long the refusal lasts ("the
  operator stopped the arm, so this repeat is refused: report it and wait";
  the agent never resets the scene on its own: a new order on the object
  needs a reset only the operator or the visitor can ask for, or the window's
  end; A8 fix, Critical 1; A8 fix 2);
- spent: a halt, an e-stop latch, a hold, a pending contact or release episode
  or a blocked arm touched the goal: its calls run exactly as without the
  ledger -- no retry, no refusal, and nothing opens.

A goal is touched when one of its calls starts held, latched, with an episode
pending or with the arm blocked, or ends failed while held or with an episode
pending, or ends halted or latched; every open slot is stopped when any motion
call starts latched or ends halted or latched. A retry the operator halts keeps
its used slot, so nothing moves again on the ledger's behalf.

Never after a stop (R7): the runtime hands `begin` and `finish` the rig's stop
token (every arm harness's halt / e-stop count, which a Resume or a reset never
lowers). A call during which it changed ends halted and spends its own goal. An
open slot whose token is not the current one at a `begin`, or that was open when
any call started latched or ended halted or latched, is stopped: a halt or an
e-stop between the refusal and the next call, even one cleared again, grants no
retry and no automatic home, and no same-goal call runs again until the slot ends.
A hold or a pending episode spends a slot without stopping it: that call runs as
on main (no operator stop came between the permission and the call).

A slot ends only when its goal succeeds, `reset_scene` succeeds (`clear`) or its
window (one native turn, R9) ends; a used slot ends on success only when the call
that used it succeeds, and a stopped one only when a call that began before the
stop reports its goal's success (no same-goal call starts while it refuses). A call with another goal never touches a slot, so per goal
at most one retry is granted within the window, however the agent interleaves
goals, and an order that is not spent gets at most two motion calls: the first
attempt and its retry. Halts and e-stops never open, use or re-arm a slot.

Refusals are classified by the EXACT class name of the exception that refused
the call (R6), never by error text: subclasses such as MotionHalted or
ContactDetected never qualify, and error text can only exclude a result, never
admit one. The ledger never edits agent text: it adds the `sim_retry` field and
returns the used-slot and stopped-slot refusals; the runtime composes
`next_action` from the texts below. Bookkeeping never fails a call: an internal error turns the ledger off
for good (every slot dropped; no refusal, no retry, nothing opens again: every
call runs as without it) and leaves the result untouched; `off` then reads True
and `last_event` is `{t, kind: "off", where}`.

A real or unknown profile never gets a ledger: `ledger_for` returns one only for
`profile_class(cfg) == "sim"` with `sim_retry_enabled(cfg)` (which also requires
the observed-finger gate, R8), and None for anything it cannot read.
"""
from __future__ import annotations

import logging
import math
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
#: A6-fix2 Minor 3: a slot opens only if the agent also has this long to read the
#: refusal and call again before `begin`'s R10 check would refuse the retry
RETRY_THINK_S = 30.0
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
#: appended to RETRY_EXHAUSTED when the retry would have started too late (R10)
RETRY_TOO_LATE = "the retry window for this turn has passed"
#: the error of every same-goal call refused because the operator stopped the rig after
#: the refusal (A8 fix, Critical 1; A8 fix 2: until reset_scene or the window's end,
#: `{seconds}` = the whole seconds left in it), and the instruction the runtime puts in
#: `next_action`; neither carries the kitchen rules' retry trigger, and the agent never
#: resets the scene on its own: only the operator or the visitor asks for the reset (A8
#: fix 2 text). Both fit the FAILED line's 240-character quote.
RETRY_STOPPED = ("No retry for {goal}: the operator stopped the arm after the refusal, so this repeat is refused "
                 "until the scene is reset or the retry window ends in {seconds} s")
RETRY_STOPPED_NEXT = ("The operator stopped the arm, so this repeat is refused: report it and wait. Never reset the "
                      "scene on your own: a new order on this object needs a scene reset, which only the operator "
                      "or the visitor can ask for (or the retry window's end).")

#: words a goal never compares (A6-fix I1)
_DETERMINERS = frozenset({"a", "an", "the", "this", "that", "these", "those"})
#: arm names `SkillRuntime._select_arm` resolves to the primary arm: the absent arm (A6-fix2 Minor 1)
_PRIMARY_ARM = (frozenset(), frozenset({"default"}), frozenset({"primary"}))
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
    if arm in _PRIMARY_ARM:
        arm = frozenset()
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


_OPEN, _USED, _STOPPED, _SPENT = "open", "used", "stopped", "spent"
#: several slots can match one goal (word containment): the most restrictive
#: state decides, then the earliest slot
_RESTRICTION = {_USED: 0, _STOPPED: 1, _SPENT: 2, _OPEN: 3}


@dataclass
class _Slot:
    goal: Goal
    opened_at: float
    first_started_at: float
    state: str = _OPEN
    refusal: str = ""
    refused_grasps: tuple = ()
    arm_left_in_place: bool = False
    late: bool = False  # used because the retry would have started too late (R10)
    stop_token: object = None  # the rig's stop token when the slot opened (R7)


@dataclass
class _Call:
    """The goal-bearing call between its `begin` and its `finish`."""

    goal: Goal
    started_at: float
    blocked: bool
    retry: RetryCall | None = None
    stop_token: object = None  # the rig's stop token at `begin`


def _seconds(label: str, value, *, minimum: float, inclusive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number of seconds, got {value!r}")
    if value < minimum or (not inclusive and value == minimum):
        bound = ">=" if inclusive else ">"
        raise ValueError(f"{label} must be {bound} {minimum:g} s, got {value!r}")
    return value


class SimRetryLedger:
    """Per-runtime retry slots (spec §4.5). Thread-safe; never touches the arm.

    The runtime calls, around each outermost motion call: `begin` (the RetryCall
    when the call is the retry, the used-slot refusal to return before any motion
    when the goal's retry is used, else None), `finish` (sets `sim_retry` on the
    result), `note_motion` (a counted home), and `clear` (reset_scene); both
    `begin` and `finish` get the rig's stop token (R7). `precheck` is the
    read-only form of `begin`'s refusal. `last_event` describes the last finished retry,
    `{t, kind: "retry", goal, outcome: "succeeded"|"failed"|"halted", refusal}`,
    or the switch-off, `{t, kind: "off", where}` (then `off` is True).
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

    @property
    def off(self) -> bool:
        """True once a bookkeeping fault switched the ledger off for good (read-only)."""
        return self._off

    # ── entry points ─────────────────────────────────────────────────────

    def precheck(self, name, args) -> dict | None:
        """The refusal `begin` would return before any motion for this goal's used
        or stopped slot (the same persistent rule), else None. Read-only: it sees no
        stop token, so a stop that `begin` has not yet seen is not reported."""
        with self._lock:
            try:
                goal = goal_of(name, args)
                if goal is None:
                    return None
                now = self._clock()
                self._expire(now)
                return self._refusal(self._slot_for(goal), now)
            except Exception:  # noqa: BLE001 -- bookkeeping never fails a call
                self._fail_closed("precheck")
                return None

    def begin(self, name, args, *, held: bool, latched: bool, episode_pending: bool,
              motion_blocked: bool, stop_token=None) -> RetryCall | dict | None:
        """Start a motion call. Returns the refusal dict (as `precheck`) when this
        goal's retry is used, or when this call would be the retry but starts too
        late for the turn (R10: the slot is used): the caller returns it before
        any motion. Returns the RetryCall when the call is THE retry of an open
        slot (the slot is used from here on), else None (the call runs as without
        the ledger). Under a hold, a latch, a pending episode or a blocked arm the
        goal is spent instead. A call that matches two or more open slots is no
        retry: all of them are used.
        `stop_token`: the rig's stop token now (R7); every open slot that opened
        under another one (a halt or e-stop since, even cleared again) is
        stopped, and so is every open slot when this call starts latched. Every
        same-goal call of a stopped slot is refused before any motion with the
        stop text (the first would run on a permission given before the stop),
        until the slot ends (A8 fix 2)."""
        with self._lock:
            self._call = None
            try:
                now = self._clock()
                self._expire(now)
                for slot in self._slots:
                    if slot.state == _OPEN and slot.stop_token != stop_token:
                        slot.state = _STOPPED  # a halt or e-stop since the refusal, even one cleared again
                if latched:
                    self._stop_open()  # a halt or e-stop pending at any motion call
                goal = goal_of(name, args)
                if goal is None:
                    return None
                slot = self._slot_for(goal)
                refusal = self._refusal(slot, now)
                if refusal is not None:
                    return refusal
                blocked = bool(held or latched or episode_pending or motion_blocked)
                call = _Call(goal, now, blocked, stop_token=stop_token)
                if blocked:
                    self._call = call
                    self._spend(goal, now)
                    return None
                if slot is None or slot.state != _OPEN:
                    self._call = call
                    return None
                matched = [other for other in self._slots if same_goal(other.goal, goal)]
                if len(matched) > 1:
                    # a rename that fits several orders: retrying one of them would
                    # be a guess, so none is retried and none is retried later
                    for other in matched:
                        other.state = _USED
                    self._call = call
                    return None
                slot.state = _USED
                if now - slot.first_started_at + self.reserve_s > TURN_BUDGET_S:
                    slot.late = True  # R10: the retry would overrun the agent's turn
                    return self._used_refusal(slot)
                self._call = call
                call.retry = RetryCall(str(slot.goal), slot.refused_grasps, slot.arm_left_in_place, slot.refusal)
                return call.retry
            except Exception:  # noqa: BLE001
                self._fail_closed("begin")
                return None

    def finish(self, name, args, result, *, exc_type, held: bool, latched: bool, halted: bool,
               episode_pending: bool, refused_grasps, retry: RetryCall | None, stop_token=None):
        """End a call: returns `result`, with `sim_retry` set when a slot opened
        (`{allowed: true, goal, refusal}`) or a retry failed (`{allowed: false,
        used: true}`). `halted`: a halt arrived during the call; `latched`: an
        e-stop or halt is latched now; `refused_grasps`: the attempted candidates
        of the refused grasp (exclusions for the retry); `stop_token`: the rig's
        stop token now (one that differs from `begin`'s ends the call halted)."""
        with self._lock:
            call, self._call = self._call, None
            if self._off:
                return result  # the only way a slot opens: off means nothing ever opens again
            try:
                return self._finish(name, args, result, call, exc_type=exc_type, held=held, latched=latched,
                                    halted=halted, episode_pending=episode_pending,
                                    refused_grasps=refused_grasps, retry=retry, stop_token=stop_token)
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
                episode_pending, refused_grasps, retry, stop_token):
        now = self._clock()
        goal = goal_of(name, args)
        if call is not None and call.goal != goal:
            call = None
        if retry is None and call is not None:
            retry = call.retry
        # a halt or an e-stop during the call, even one cleared again before its end (R7)
        stopped = bool(halted or latched or (call is not None and call.stop_token != stop_token))
        if stopped:
            # a halt or e-stop at any point, whatever the result: every open slot is
            # stopped and this call's goal is spent (a used slot stays used: the
            # halted order never moves again on the ledger's behalf)
            self._stop_open()
            if goal is not None:
                self._spend(goal, now)
        if goal is None or not isinstance(result, MutableMapping):
            return result
        # the slot a retry used: the one its goal matches (begin granted it as the only match)
        own = self._slot_for(goal) if retry is not None else None
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
        if now - call.started_at + self.reserve_s + RETRY_THINK_S > TURN_BUDGET_S:
            return result  # the retry would overrun the agent's turn (R10, A6-fix2 Minor 3)
        grasps = () if refused_grasps is None else tuple(refused_grasps)
        self._slots.append(_Slot(goal, now, call.started_at, _OPEN, refusal, grasps,
                                 bool(result.get("arm_left_in_place")), stop_token=stop_token))
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

    def _stop_open(self) -> None:
        """The operator stopped the rig: every open slot is stopped (its same-goal
        calls are refused until the slot ends, `begin`)."""
        for slot in self._slots:
            if slot.state == _OPEN:
                slot.state = _STOPPED

    def _succeeded(self, goal: Goal, own: _Slot | None) -> None:
        """The goal succeeded: drop the slot this call used and every open or spent
        slot of the goal. Another order's used slot stays (only its own call, a
        reset or its window ends it)."""
        self._slots = [slot for slot in self._slots
                       if slot is not own and (slot.state == _USED or not same_goal(slot.goal, goal))]

    def _refusal(self, slot: _Slot | None, now: float) -> dict | None:
        """The refusal a goal's deciding slot gives before any motion: used or
        stopped (both until the slot ends), else None."""
        if slot is None:
            return None
        if slot.state == _USED:
            return self._used_refusal(slot)
        if slot.state == _STOPPED:
            return self._stopped_refusal(slot, now)
        return None

    def _used_refusal(self, slot: _Slot) -> dict:
        error = RETRY_EXHAUSTED.format(goal=slot.goal)
        if slot.late:
            error = f"{error}: {RETRY_TOO_LATE}"
        return {"ok": False, "error": error, "sim_retry": {"allowed": False, "used": True}}

    def _stopped_refusal(self, slot: _Slot, now: float) -> dict:
        left = math.ceil(slot.opened_at + self.window_s - now)  # > 0: an expired slot is gone
        return {"ok": False, "error": RETRY_STOPPED.format(goal=slot.goal, seconds=left),
                "sim_retry": {"allowed": False, "stopped": True}}

    def _expire(self, now: float) -> None:
        self._slots = [slot for slot in self._slots if now - slot.opened_at < self.window_s]

    def _fail_closed(self, where: str) -> None:
        """Off for good: every slot dropped and `finish` never opens one again, so
        `precheck` and `begin` have nothing left to refuse or grant."""
        logger.exception("sim retry %s failed; the ledger is off (no retry, no refusal)", where)
        self._off = True
        self._slots = []
        self._call = None
        self.last_event = {"t": time.time(), "kind": "off", "where": where}


def ledger_for(cfg) -> SimRetryLedger | None:
    """The runtime's ledger: only for a sim profile whose `sim_retry_enabled`
    (`agent.sim_retry: true`, the kill switch off, the observed-finger gate on);
    None otherwise, and None for any config it cannot read or a bad
    `agent.sim_retry_window_s` / `agent.sim_retry_reserve_s`. A real or unknown
    profile never gets one."""
    try:
        if profile_class(cfg) != "sim" or not sim_retry_enabled(cfg):
            return None
        agent = cfg.get("agent")
        return SimRetryLedger(window_s=agent.get("sim_retry_window_s", DEFAULT_WINDOW_S),
                              reserve_s=agent.get("sim_retry_reserve_s", DEFAULT_RESERVE_S))
    except Exception as exc:  # noqa: BLE001 -- fail closed: no ledger, no retry
        logger.warning("sim retry off: %s: %s", type(exc).__name__, exc)
        return None
