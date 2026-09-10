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


class MilestoneTracker:
    """Checks decomposed milestones against the world model, then the VLM.

    ``beliefs`` is a ``BeliefStore``; ``held_getter`` returns the label of the
    currently held object (or None).  ``vlm_verify`` is an optional
    ``(milestone, jpeg) -> (bool | None, str)`` callable; returning ``None``
    means "cannot tell", which maps to UNKNOWN.
    """

    def __init__(
        self,
        beliefs=None,
        held_getter: Callable[[], str | None] | None = None,
        vlm_verify: Callable[[str, bytes], tuple[bool | None, str]] | None = None,
        max_visual_checks: int = 3,
        containment_pad_m: float = 0.02,
    ):
        self.beliefs = beliefs
        self._held = held_getter or (lambda: None)
        self._vlm_verify = vlm_verify
        self.max_visual_checks = int(max_visual_checks)
        self.pad = float(containment_pad_m)
        self.milestones: list[Milestone] = []
        self._visual_checks = 0

    # ── lifecycle ────────────────────────────────────────────────────────

    def reset(self, milestones: list[str]) -> None:
        self.milestones = [Milestone(text=m) for m in milestones if m and m.strip()]
        self._visual_checks = 0

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
            and self._visual_checks < self.max_visual_checks
        ):
            self._visual_checks += 1
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
        head = re.sub(r"[^a-z]", "", text[:6].lower())
        if head.startswith("yes"):
            return True, text[:160]
        if head.startswith("no"):
            return False, text[:160]
        return None, text[:160] or "no answer"

    return verify
