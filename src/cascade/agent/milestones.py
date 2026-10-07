"""Milestone verification: turning task decomposition into a progress signal.

Agentic-VLA (arXiv:2605.22896) trains with *Adaptive Reward Synthesis* --
decompose a task into learnable sub-goals and score progress against them.
We cannot backprop into a frozen agent, but the useful half of that idea is
inference-time: if you decompose into **checkable** milestones, you can
actually check them, and a dense progress signal beats a binary
success-at-the-end for both recovery and honesty.

The repo already decomposes (``AgentOrchestrator._decompose``) and already
ships the verification prompt (``prompts.VERIFY_USER``) -- but nothing ever
called it, so milestones were decoration.  This module closes that gap.

Two verification tiers, cheapest first:

1. **Symbolic** -- resolve the milestone against the world model the robot
   already maintains (BeliefStore + held-object state).  "the cube is in the
   bin" is answerable from 3D beliefs with zero tokens and no hallucination
   surface.  This is the tier that should fire most often.
2. **Visual** -- fall back to the VLM with ``VERIFY_USER`` when the milestone
   is not symbolically decidable.  Costs one turn, so it is rate-limited and
   only consulted when the symbolic tier abstains.

The tracker deliberately reports ``UNKNOWN`` rather than guessing: an
unverifiable milestone must not be silently counted as done.  That is the
same discipline the system prompt demands of the agent ("Never claim success
you did not verify").

2026-10-07 -- the same rate-limited critic pattern now also runs BEFORE a
motion is dispatched (ROADMAP follow-up #6, Human-CLAW's pre-execution skill
verifier): ``PlausibilityChecker`` asks the VLM a skill-specific question
about the proposed call -- "is this call, with these args, plausible given
the current view, the beliefs, reachability and what is held?" -- and the
answer rides along on the result and the trace row as ``plausibility``.
Unlike Human-CLAW's verifier it may NOT veto or substitute: the safety
harness is the sole authority that refuses motion (AGENTS.md), so the
verdict is a non-blocking caution to the planner, the same booth rule as
``memory/envelope.py``.  It draws VLM turns from a ``VisualBudget`` -- the
tracker's own per-task limiter, factored out so both critics share one
mechanism -- and anything that would otherwise stall or raise (no vision
model, no frame, budget gone, verifier fault) is recorded as ``skipped``.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable

#: verification outcomes
DONE = "done"
PENDING = "pending"
UNKNOWN = "unknown"

#: pre-motion plausibility verdicts (advisory only -- never a veto)
PLAUSIBLE = "plausible"
IMPLAUSIBLE = "implausible"
UNSURE = "unsure"
SKIPPED = "skipped"

#: containment relations we can decide from 3D beliefs alone
_IN_WORDS = r"(?:in|inside|into|within|en|dentro de)"
_ON_WORDS = r"(?:on|onto|on top of|above|sobre|encima de)"
_HELD_WORDS = r"(?:holding|grasped|gripped|picked up|in the gripper|sujeta|agarrad[oa])"
_GONE_WORDS = r"(?:removed|cleared|off the table|away|retirad[oa])"

#: strip leading imperatives so "place the cube in the bin" and "the cube is
#: in the bin" resolve to the same (subject, relation, target) triple
_IMPERATIVE = re.compile(
    r"^(?:the\s+robot\s+)?(?:should\s+|must\s+)?"
    r"(?:pick(?:\s+up)?|place|put|move|drop|grasp|grab|take|set|lower|release|"
    r"coge|pon|coloca|deja|mete)\s+",
    re.IGNORECASE,
)
_ARTICLE = re.compile(r"\b(?:the|a|an|el|la|los|las|un|una)\b", re.IGNORECASE)


def _clean(text: str) -> str:
    text = _IMPERATIVE.sub("", (text or "").strip().rstrip("."))
    text = _ARTICLE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


@dataclass
class Milestone:
    text: str
    status: str = PENDING
    evidence: str = ""
    tier: str = ""          # "symbolic" | "visual" | ""
    checked_at: float = 0.0

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "status": self.status,
            "evidence": self.evidence,
            "tier": self.tier,
        }


@dataclass
class Progress:
    """Dense progress signal for one task."""

    done: int = 0
    total: int = 0
    unknown: int = 0
    newly_done: list[str] = field(default_factory=list)

    @property
    def fraction(self) -> float:
        return self.done / self.total if self.total else 0.0

    @property
    def stalled(self) -> bool:
        return not self.newly_done

    def as_dict(self) -> dict:
        return {
            "done": self.done,
            "total": self.total,
            "unknown": self.unknown,
            "fraction": round(self.fraction, 2),
            "newly_done": list(self.newly_done),
        }


class VisualBudget:
    """Per-task cap on VLM turns spent on verification.

    One visual check costs a whole model turn (2-15 s on the booth rig), so
    every critic that can ask the VLM draws from a budget that resets with
    the task.  Factored out of ``MilestoneTracker`` so the pre-motion
    plausibility critic rate-limits with the SAME mechanism; pass one
    instance to both to make them share one pool of turns.
    """

    def __init__(self, max_checks: int = 3):
        self.max_checks = int(max_checks)
        self.used = 0

    def reset(self) -> None:
        self.used = 0

    @property
    def exhausted(self) -> bool:
        return self.used >= self.max_checks

    def take(self) -> bool:
        """Consume one turn; False (and nothing consumed) once exhausted."""
        if self.exhausted:
            return False
        self.used += 1
        return True


class MilestoneTracker:
    """Checks decomposed milestones against the world model, then the VLM.

    ``beliefs`` is a ``BeliefStore``; ``held_getter`` returns the label of the
    currently held object (or None).  ``vlm_verify`` is an optional
    ``(milestone, jpeg) -> (bool | None, str)`` callable; returning ``None``
    means "cannot tell", which maps to UNKNOWN.  ``budget`` lets the visual
    tier share its per-task VLM turns with another critic; by default it
    owns a ``VisualBudget(max_visual_checks)``.
    """

    def __init__(
        self,
        beliefs=None,
        held_getter: Callable[[], str | None] | None = None,
        vlm_verify: Callable[[str, bytes], tuple[bool | None, str]] | None = None,
        max_visual_checks: int = 3,
        containment_pad_m: float = 0.02,
        budget: VisualBudget | None = None,
    ):
        self.beliefs = beliefs
        self._held = held_getter or (lambda: None)
        self._vlm_verify = vlm_verify
        self.budget = budget if budget is not None else VisualBudget(max_visual_checks)
        self.pad = float(containment_pad_m)
        self.milestones: list[Milestone] = []

    @property
    def max_visual_checks(self) -> int:
        return self.budget.max_checks

    # ── lifecycle ────────────────────────────────────────────────────────

    def reset(self, milestones: list[str]) -> None:
        self.milestones = [Milestone(text=m) for m in milestones if m and m.strip()]
        self.budget.reset()

    @property
    def active(self) -> bool:
        return bool(self.milestones)

    # ── verification ─────────────────────────────────────────────────────

    def update(self, frame_jpeg: bytes | None = None, allow_visual: bool = True) -> Progress:
        """Re-check every still-pending milestone.  Returns the progress delta."""
        newly = []
        for ms in self.milestones:
            if ms.status == DONE:
                continue
            verdict, evidence, tier = self._verify(ms.text, frame_jpeg, allow_visual)
            ms.checked_at = time.time()
            if verdict is True:
                ms.status, ms.evidence, ms.tier = DONE, evidence, tier
                newly.append(ms.text)
            elif verdict is False:
                ms.status, ms.evidence, ms.tier = PENDING, evidence, tier
            else:
                ms.status, ms.evidence, ms.tier = UNKNOWN, evidence, tier
        return Progress(
            done=sum(1 for m in self.milestones if m.status == DONE),
            total=len(self.milestones),
            unknown=sum(1 for m in self.milestones if m.status == UNKNOWN),
            newly_done=newly,
        )

    def _verify(
        self, text: str, frame_jpeg: bytes | None, allow_visual: bool
    ) -> tuple[bool | None, str, str]:
        verdict, evidence = self.check_symbolic(text)
        if verdict is not None:
            return verdict, evidence, "symbolic"
        if (
            allow_visual
            and self._vlm_verify is not None
            and frame_jpeg
            and self.budget.take()
        ):
            try:
                verdict, evidence = self._vlm_verify(text, frame_jpeg)
            except Exception as e:  # a flaky VLM must never fail the task
                return None, f"visual check failed: {e}", "visual"
            return verdict, evidence, "visual"
        return None, "not decidable from the world model", ""

    # ── tier 1: symbolic, from the belief store ──────────────────────────

    def check_symbolic(self, text: str) -> tuple[bool | None, str]:
        """Decide a milestone from 3D beliefs + held state.  None = abstain."""
        phrase = _clean(text)
        if not phrase:
            return None, ""

        held = self._held()

        m = re.match(rf"^(?P<subj>.+?)\s+(?:is\s+|are\s+)?{_HELD_WORDS}\b", phrase)
        if m:
            subj = m.group("subj").strip()
            if held and self._same(subj, held):
                return True, f"gripper holds {held!r}"
            return False, f"gripper holds {held or 'nothing'}"

        m = re.match(
            rf"^(?P<subj>.+?)\s+(?:is\s+|are\s+)?{_IN_WORDS}\s+(?P<targ>.+)$", phrase
        )
        if m:
            return self._check_containment(m.group("subj"), m.group("targ"), mode="in")

        m = re.match(
            rf"^(?P<subj>.+?)\s+(?:is\s+|are\s+)?{_ON_WORDS}\s+(?P<targ>.+)$", phrase
        )
        if m:
            return self._check_containment(m.group("subj"), m.group("targ"), mode="on")

        m = re.match(rf"^(?P<subj>.+?)\s+(?:is\s+|are\s+)?{_GONE_WORDS}\b", phrase)
        if m:
            b = self._lookup(m.group("subj"))
            if b is None:
                return True, "object no longer in the world model"
            return False, "object still tracked"

        if re.search(r"\b(?:home|rest|neutral)\s+(?:position|pose)\b", phrase):
            return None, ""
        return None, ""

    def _check_containment(self, subj: str, targ: str, mode: str) -> tuple[bool | None, str]:
        sb = self._lookup(subj)
        tb = self._lookup(targ)
        if sb is None or tb is None:
            missing = subj if sb is None else targ
            return None, f"{missing!r} not in the world model"
        try:
            sp = [float(v) for v in sb.position[:3]]
            tp = [float(v) for v in tb.position[:3]]
            te = [float(v) for v in (getattr(tb, "extent", None) or [0.1, 0.1, 0.1])[:3]]
        except (TypeError, ValueError, IndexError):
            return None, "belief geometry unavailable"

        dx, dy = abs(sp[0] - tp[0]), abs(sp[1] - tp[1])
        hx, hy = te[0] / 2 + self.pad, te[1] / 2 + self.pad
        inside_xy = dx <= hx and dy <= hy
        if not inside_xy:
            return False, (
                f"offset ({dx:.3f}, {dy:.3f}) m exceeds target half-extent "
                f"({hx:.3f}, {hy:.3f}) m"
            )
        if mode == "on":
            above = sp[2] >= tp[2] - self.pad
            return (True, f"centred over target and above it (dz={sp[2]-tp[2]:+.3f} m)") if above else (
                False, f"below the target surface (dz={sp[2]-tp[2]:+.3f} m)"
            )
        return True, f"centred inside target footprint (dx={dx:.3f}, dy={dy:.3f} m)"

    # ── belief lookup ────────────────────────────────────────────────────

    def _lookup(self, phrase: str):
        if self.beliefs is None:
            return None
        phrase = _clean(phrase)
        try:
            items = list(self.beliefs.all())
        except Exception:
            return None
        best, best_score = None, 0
        for b in items:
            label = str(getattr(b, "label", "")).lower()
            if not label:
                continue
            score = self._overlap(phrase, label)
            if score > best_score:
                best, best_score = b, score
        return best if best_score > 0 else None

    @staticmethod
    def _overlap(phrase: str, label: str) -> int:
        pw = set(re.findall(r"[a-z]+", phrase))
        lw = set(re.findall(r"[a-z]+", label))
        if not pw or not lw:
            return 0
        if lw <= pw or pw <= lw:
            return 2 + len(pw & lw)
        return len(pw & lw)

    def _same(self, phrase: str, label: str) -> bool:
        return self._overlap(_clean(phrase), label.lower()) > 0

    # ── reporting ────────────────────────────────────────────────────────

    def digest(self) -> str:
        """Milestone board for the agent context."""
        if not self.milestones:
            return ""
        marks = {DONE: "[x]", PENDING: "[ ]", UNKNOWN: "[?]"}
        lines = ["Milestone progress (verified against the world model):"]
        for ms in self.milestones:
            line = f"{marks.get(ms.status, '[ ]')} {ms.text}"
            if ms.evidence:
                line += f"  ({ms.evidence})"
            lines.append(line)
        unknown = [m for m in self.milestones if m.status == UNKNOWN]
        if unknown:
            lines.append(
                "[?] means it could NOT be verified -- do not claim it as done; "
                "re-observe or say so in the final summary."
            )
        return "\n".join(lines)

    def as_list(self) -> list[dict]:
        return [m.as_dict() for m in self.milestones]

    def unverified(self) -> list[str]:
        return [m.text for m in self.milestones if m.status != DONE]


def _yes_no(text: str) -> bool | None:
    """Head-of-answer parse: YES -> True, NO -> False, anything else -> None.

    Answers that do not start with a clear YES/NO become ``None`` rather
    than a coin flip -- shared by the milestone verifier and the pre-motion
    critic so both read a model's answer the same way.
    """
    head = re.sub(r"[^a-z]", "", (text or "")[:6].lower())
    if head.startswith("yes"):
        return True
    if head.startswith("no"):
        return False
    return None


def make_vlm_verifier(llm, prompt_template: str) -> Callable[[str, bytes], tuple[bool | None, str]]:
    """Adapt an ``LLMClient`` into a ``(milestone, jpeg) -> (bool|None, str)``.

    Answers that do not start with a clear YES/NO become ``None`` (UNKNOWN)
    rather than a coin flip.
    """

    def verify(milestone: str, jpeg: bytes) -> tuple[bool | None, str]:
        resp = llm.chat(
            system="You verify robot task milestones from a single camera frame. Be strict.",
            messages=[
                {
                    "role": "user",
                    "content": prompt_template.format(milestone=milestone),
                    "images": [jpeg],
                }
            ],
            max_tokens=80,
        )
        text = (resp.text or "").strip()
        verdict = _yes_no(text)
        if verdict is None:
            return None, text[:160] or "no answer"
        return verdict, text[:160]

    return verify


# ── pre-motion plausibility critic (Human-CLAW, ROADMAP #6) ─────────────────
#
# ADVISORY ONLY. Nothing in this section can refuse, delay beyond its budget
# or rewrite a call: ``check()`` returns a dict and never raises, and the
# orchestrator dispatches the call unchanged whatever the verdict says. The
# safety harness is the sole authority that refuses motion (AGENTS.md).


class _SafeArgs(dict):
    """``str.format_map`` source that renders a missing argument as ``?``
    instead of raising -- a wrong placeholder degrades the wording of the
    question, never the check."""

    def __missing__(self, key):
        return "?"


#: One skill-specific question per motion primitive, written against the
#: TOOL_SPECS argument names (grasp/push/point take `label`, pick_and_place
#: takes `object`, place_at takes x/y). Each asks about the preconditions
#: that the harness cannot see -- what is visible, what is held, whether the
#: target makes sense -- not about limits the harness already enforces.
_PLAUSIBILITY_QUESTIONS: dict[str, str] = {
    "grasp_object": (
        "Is {label!r} actually visible on the table, unoccluded and within the "
        "arm's reach, and is the gripper EMPTY right now?"
    ),
    "grasp_at_pixel": (
        "Does pixel ({u}, {v}) in the image fall on a graspable object on the "
        "table, and is the gripper EMPTY right now?"
    ),
    "pick_and_place": (
        "Is {object!r} visible and reachable, is the gripper EMPTY, and is the "
        "destination {destination!r} a free, reachable spot for it?"
    ),
    "place_at": (
        "Is the gripper HOLDING an object right now, and is ({x}, {y}) a free "
        "spot on the table inside the workspace?"
    ),
    "place_on_object": (
        "Is the gripper HOLDING an object, and is {label!r} visible, stable and "
        "large enough to receive it?"
    ),
    "push_object": (
        "Is {label!r} visible on the table with free space to push it "
        "{direction} by {distance_m} m, and is the gripper EMPTY?"
    ),
    "move_relative": (
        "Does moving the tool {direction} by {distance_m} m keep it above the "
        "table and inside the workspace without hitting anything in view?"
    ),
    "handover": (
        "Is the gripper HOLDING the object to hand over, and is the handover "
        "pose clear of people and obstacles in view?"
    ),
    "open_gripper": (
        "Is opening the gripper here sensible -- if it holds an object, is this "
        "a surface where releasing it is intended?"
    ),
    "close_gripper": (
        "Is there an object between the jaws to close on, or is closing on "
        "nothing the intent?"
    ),
    "move_home": (
        "Is the path back to the home pose clear, and if the gripper holds an "
        "object, is carrying it home intended?"
    ),
    "point_at": (
        "Is {label!r} visible so that pointing at it is meaningful, and is the "
        "pointing pose clear?"
    ),
    "sort_by_color": (
        "Are the objects to sort visible on the table with free destination "
        "zones, and is the gripper EMPTY?"
    ),
    "throw": (
        "Is the gripper HOLDING {label!r}, and is the {direction} throw direction "
        "clear of people and fragile things in view?"
    ),
    "turn_screw": (
        "Is the fastener {label!r} visible, engaged by the tool and oriented so "
        "a wrist rotation turns it {direction}?"
    ),
    "reset_scene": "Is a scene reset intended now (nothing held, no motion in progress)?",
}
_GENERIC_QUESTION = (
    "Given the current view and the world model, is this call with these exact "
    "arguments plausible to succeed and sensible right now?"
)

#: cap on belief lines in the digest -- the prompt must stay small
_DIGEST_MAX_BELIEFS = 12


def _short_call(name: str, args: dict) -> str:
    return f"{name}(" + ", ".join(f"{k}={v}" for k, v in (args or {}).items()) + ")"


class PlausibilityChecker:
    """Human-CLAW-style pre-execution interrogation of ONE proposed motion call.

    ``verifier`` is a ``(prompt_text, jpeg) -> (bool | None, str)`` callable
    (see ``make_plausibility_verifier``); ``None`` means no vision-capable
    model is configured and every check is recorded as ``skipped``.
    ``beliefs`` is a ``BeliefStore`` (its ``summary()`` is the digest),
    ``held_getter`` returns the held label, ``workspace`` is the configured
    reachability box (``{"min": [...], "max": [...]}``) quoted to the critic.

    ``check()`` is the whole contract: it never raises and always returns
    ``{"verdict", "reasons", "source"}``.  Verdicts: ``plausible`` /
    ``implausible`` / ``unsure`` from the model, or ``skipped`` with the
    reason (no model, no frame, budget exhausted, verifier fault).  The
    caller treats every one of them the same way for dispatch -- the call
    runs unchanged -- and only ``implausible`` earns the planner a caution.
    """

    def __init__(
        self,
        verifier: Callable[[str, bytes], tuple[bool | None, str]] | None = None,
        *,
        beliefs=None,
        held_getter: Callable[[], str | None] | None = None,
        budget: VisualBudget | None = None,
        max_checks: int = 3,
        workspace=None,
        skip_reason: str | None = None,
        prompt_template: str | None = None,
    ):
        self._verifier = verifier
        self.beliefs = beliefs
        self._held = held_getter or (lambda: None)
        self.budget = budget if budget is not None else VisualBudget(max_checks)
        self.workspace = workspace
        self._skip_reason = skip_reason or "no vision-capable model configured"
        if prompt_template is None:
            from .prompts import PLAUSIBILITY_USER

            prompt_template = PLAUSIBILITY_USER
        self._template = prompt_template

    def reset(self) -> None:
        """New task, fresh budget (the orchestrator calls this per task)."""
        self.budget.reset()

    # ── the interrogation ────────────────────────────────────────────────

    def check(self, name: str, args: dict, frame_jpeg: bytes | None) -> dict:
        """Judge ``name(**args)`` from the current frame. Never raises."""
        try:
            if self._verifier is None:
                return self._skipped(self._skip_reason)
            if not frame_jpeg:
                return self._skipped("no camera frame to judge from")
            if not self.budget.take():
                return self._skipped(
                    f"visual budget exhausted ({self.budget.max_checks} checks per task)"
                )
            source = f"vlm:{getattr(self._verifier, 'source', None) or 'unknown'}"
            try:
                verdict, text = self._verifier(self.prompt(name, args), frame_jpeg)
            except Exception as e:  # noqa: BLE001 -- a flaky VLM must never block a motion
                return self._skipped(f"verifier failed: {type(e).__name__}: {e}")
            reason = str(text or "").strip()[:200] or "no answer"
            if verdict is True:
                return {"verdict": PLAUSIBLE, "reasons": [reason], "source": source}
            if verdict is False:
                return {"verdict": IMPLAUSIBLE, "reasons": [reason], "source": source}
            return {"verdict": UNSURE, "reasons": [reason], "source": source}
        except Exception as e:  # noqa: BLE001 -- belt and braces: advisory code cannot fail a task
            return self._skipped(f"critic failed: {type(e).__name__}: {e}")

    @staticmethod
    def _skipped(reason: str) -> dict:
        return {"verdict": SKIPPED, "reasons": [reason], "source": "none"}

    # ── prompt pieces (public so tests and tools can inspect them) ───────

    def question(self, name: str, args: dict) -> str:
        template = _PLAUSIBILITY_QUESTIONS.get(name, _GENERIC_QUESTION)
        try:
            return template.format_map(_SafeArgs(args or {}))
        except Exception:  # noqa: BLE001 -- odd arg values: fall back to the generic question
            return _GENERIC_QUESTION

    def belief_digest(self) -> str:
        """World model as the critic sees it: beliefs, held state, reach box."""
        lines: list[str] = []
        try:
            rows = list(self.beliefs.summary()) if self.beliefs is not None else []
        except Exception:  # noqa: BLE001
            rows = []
            lines.append("- beliefs unavailable")
        for row in rows[:_DIGEST_MAX_BELIEFS]:
            try:
                pos = ", ".join(f"{float(v):.3f}" for v in (row.get("position") or [])[:3])
                color = f" ({row['color']})" if row.get("color") else ""
                lines.append(
                    f"- {row.get('label')}{color} at ({pos}) {row.get('state')}, "
                    f"age {row.get('age_s')} s, conf {row.get('conf')}"
                )
            except Exception:  # noqa: BLE001
                lines.append(f"- {row!r}")
        if len(rows) > _DIGEST_MAX_BELIEFS:
            lines.append(f"- ... {len(rows) - _DIGEST_MAX_BELIEFS} more")
        if not rows and not lines:
            lines.append("- no tracked objects")
        try:
            held = self._held()
        except Exception:  # noqa: BLE001
            held = None
        lines.append(f"- gripper holding: {held or 'nothing'}")
        ws = self.workspace
        try:
            if ws is not None:
                lo = [float(v) for v in ws.get("min")]
                hi = [float(v) for v in ws.get("max")]
                lines.append(
                    "- reachable TCP box: x [%.2f, %.2f], y [%.2f, %.2f], z [%.2f, %.2f]"
                    % (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
                )
        except Exception:  # noqa: BLE001 -- a malformed box is simply not quoted
            pass
        return "\n".join(lines)

    def prompt(self, name: str, args: dict) -> str:
        return self._template.format(
            call=_short_call(name, args),
            question=self.question(name, args),
            beliefs=self.belief_digest(),
        )


def make_plausibility_verifier(llm, system: str | None = None) -> Callable[[str, bytes], tuple[bool | None, str]]:
    """Adapt an ``LLMClient`` into a ``(prompt_text, jpeg) -> (bool|None, str)``
    for ``PlausibilityChecker`` -- the pre-motion twin of ``make_vlm_verifier``.

    The returned callable carries a ``source`` attribute naming the model
    (``llm.model`` when the client has one, else the client class) so the
    verdict in the trace says WHO judged.
    """
    if system is None:
        from .prompts import PLAUSIBILITY_SYSTEM

        system = PLAUSIBILITY_SYSTEM

    def verify(prompt_text: str, jpeg: bytes) -> tuple[bool | None, str]:
        resp = llm.chat(
            system=system,
            messages=[{"role": "user", "content": prompt_text, "images": [jpeg]}],
            max_tokens=120,
        )
        text = (resp.text or "").strip()
        return _yes_no(text), text

    verify.source = str(getattr(llm, "model", None) or type(llm).__name__)  # type: ignore[attr-defined]
    return verify
