"""The agent loop: reflex first, then decompose -> act -> verify -> recover.

Inference-time composition of the two papers' ideas plus the Anthropic
robotics study's latency lesson:
- reflex/habit tier (agent.reflex): routine commands compile straight to
  skill calls (µs) or replay a proven plan from experience memory -- the LLM
  never blocks the hot path;
- ASPIRE: curated skill API + per-call multimodal traces + honest failure
  reporting back into context;
- Agentic-VLA: LLM task decomposition into checkable milestones, and a VLM
  advisor consulted only after failures whose one-sentence spatial suggestion
  is appended to the context for the retry.
"""
#
# 2026-07-31 -- the loop closes. Three ideas that were previously *described*
# here but not implemented now actually run:
#
# * Pigey (arXiv:2607.21725) "orchestration gap": the orchestrator TRACKS AND
#   VERIFIES outcomes from observation. Milestones are re-checked after every
#   motion, and a skill whose self-reported success is contradicted by the
#   world model is escalated instead of believed.
# * Agentic-VLA: decomposition becomes a dense PROGRESS signal, not decoration
#   -- the milestone board is fed back each turn and stalls trigger strategy
#   changes rather than blind retries.
# * ASPIRE: validated repairs from earlier runs are retrieved into context at
#   task start (the "load-into-context loop" the ROADMAP listed as open).

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from ..skills.runtime import TOOL_SPECS, SkillRuntime
from .advisor import Advisor
from .aspire import retrieve as retrieve_skills
from .llm import LLMClient, LLMResponse
from .milestones import MilestoneTracker, make_vlm_verifier
from .prompts import DECOMPOSE_PROMPT, SYSTEM_PROMPT, VERIFY_USER
from .reflex import FastPlanner


def _short_args(args: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in args.items())


def _as_bool(value) -> bool:
    """Schema-lax backends send booleans as strings ('false' is truthy!)."""
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


@dataclass
class TaskReport:
    task: str
    success: bool
    summary: str
    steps: int
    milestones: list[str] = field(default_factory=list)
    tool_log: list[dict] = field(default_factory=list)
    path: str = "llm"  # "reflex" | "experience" | "llm"
    duration_s: float = 0.0
    #: per-milestone verification state (Agentic-VLA progress signal)
    milestone_status: list[dict] = field(default_factory=list)
    #: milestones that could not be verified -- surfaced so a "success"
    #: claim can never quietly outrun the evidence
    unverified: list[str] = field(default_factory=list)


class AgentOrchestrator:
    def __init__(
        self,
        llm: LLMClient,
        runtime: SkillRuntime,
        advisor: Advisor | None = None,
        max_steps: int = 30,
        decompose: bool = True,
        attach_images: bool = True,
        fast_planner: FastPlanner | None = None,
        skill_library=None,
        verify_milestones: bool = True,
    ):
        self.llm = llm
        self.runtime = runtime
        self.advisor = advisor
        self.max_steps = max_steps
        self.decompose = decompose
        self.attach_images = attach_images and llm.supports_vision
        self.fast_planner = fast_planner
        self.skill_library = skill_library
        #: Pigey/Agentic-VLA milestone verification. The symbolic tier reads
        #: the belief store directly (free); the visual tier costs one VLM
        #: turn and is rate-limited inside the tracker.
        self.tracker = (
            MilestoneTracker(
                beliefs=getattr(runtime, "beliefs", None),
                held_getter=lambda: getattr(runtime, "held_object", None),
                vlm_verify=(
                    make_vlm_verifier(llm, VERIFY_USER) if llm.supports_vision else None
                ),
            )
            if verify_milestones
            else None
        )

    def run_task(self, task: str) -> TaskReport:
        report = self._run_task(task)
        # which tier actually served the command -- rendered as the
        # dashboard's "via:" chip, so habit/reflex hits are visibly LLM-free
        self.runtime.last_path = report.path
        return report

    def _run_task(self, task: str) -> TaskReport:
        t_start = time.monotonic()
        fast_note = None
        if self.fast_planner is not None:
            report, fast_note = self._try_fast_path(task, t_start)
            if report is not None:
                return report

        milestones = self._decompose(task) if self.decompose else []
        if self.tracker is not None:
            self.tracker.reset(milestones)
        messages: list[dict] = []
        tool_log: list[dict] = []

        intro = f"Task: {task}\n"
        if milestones:
            intro += "Milestones:\n" + "\n".join(f"{i+1}. {m}" for i, m in enumerate(milestones)) + "\n"
        if fast_note:
            intro += f"\n{fast_note}\n"
        intro += (
            "\nMemory (last 15 s):\n" + self.runtime.memory.digest()
        )
        # VIA-style text demonstration (arXiv 2607.11119) + RPent "READ MEMORY
        # FIRST": surface learned grasp priors so the agent starts from proven
        # strategy instead of rediscovering it. Empty on a cold start.
        try:
            gm_digest = self.runtime.grasp_memory.agent_digest()
            if gm_digest:
                intro += "\n\n" + gm_digest
        except Exception:
            pass
        # Harness-VLA (arXiv:2607.08448): the learned OPERATING RANGE of the
        # fixed primitives, plus how they usually fail on this rig.
        try:
            env = self.runtime.envelope.agent_digest()
            if env:
                intro += "\n\n" + env
        except Exception:
            pass
        # ASPIRE: validated repairs distilled from earlier runs, guard-matched
        # to this task. This is the sim->real / run->run transfer channel.
        if self.skill_library is not None:
            try:
                lib = retrieve_skills(self.skill_library, task)
                if lib:
                    intro += "\n\n" + lib
            except Exception:
                pass
        intro += "\n\nBegin. Observe first, then act. Call one tool now."
        messages.append({"role": "user", "content": intro})

        consecutive_failures = 0
        stalled_turns = 0
        for step in range(1, self.max_steps + 1):
            resp = self.llm.chat(
                system=SYSTEM_PROMPT,
                messages=self._prune_images(messages),
                tools=TOOL_SPECS,
                max_tokens=1024,
            )
            if not resp.tool_calls:
                messages.append({"role": "assistant", "content": resp.text or "(no text)"})
                messages.append(
                    {
                        "role": "user",
                        "content": "Respond with exactly one tool call (use task_done to finish).",
                    }
                )
                continue

            # Execute only the first call, but echo ALL calls and answer each
            # one -- OpenAI-style APIs reject the next request if any
            # tool_call in the assistant message lacks a matching result.
            call = resp.tool_calls[0]
            messages.append(
                {"role": "assistant", "content": resp.text, "tool_calls": list(resp.tool_calls)}
            )
            self.runtime.current_tier = "llm"
            try:
                result = self.runtime.execute(call.name, call.arguments)
            finally:
                self.runtime.current_tier = None
            tool_log.append({"step": step, "tool": call.name, "args": call.arguments, "result": result})

            if call.name == "task_done" and result.get("task_complete"):
                summary = str(call.arguments.get("summary", ""))
                success = _as_bool(call.arguments.get("success", False))
                # Pigey: a success claim is checked against the world model
                # before it is accepted. Unverified milestones downgrade the
                # claim rather than riding along with it.
                status, unverified = self._final_check(success)
                if success and unverified:
                    summary += (
                        "\n[verification] could not confirm: "
                        + "; ".join(unverified)
                    )
                self.runtime.trace.finish(
                    f"task: {task}\nsuccess: {success}\nsteps: {step}\n{summary}"
                )
                return TaskReport(
                    task, success, summary, step, milestones, tool_log,
                    duration_s=round(time.monotonic() - t_start, 2),
                    milestone_status=status, unverified=unverified,
                )

            ok = bool(result.get("ok"))
            consecutive_failures = 0 if ok else consecutive_failures + 1
            if self.advisor is not None:
                self.advisor.note_outcome(ok)

            content = json.dumps(result)
            messages.append(
                {"role": "tool", "tool_call_id": call.id or "call_0", "name": call.name, "content": content}
            )
            for extra in resp.tool_calls[1:]:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": extra.id or "call_x",
                        "name": extra.name,
                        "content": json.dumps(
                            {"ok": False, "error": "skipped: one tool call per turn; re-issue if still needed"}
                        ),
                    }
                )

            # ── Pigey closed loop: re-verify progress after every motion ──
            progress = self._check_progress(call.name)
            if progress is not None:
                stalled_turns = stalled_turns + 1 if progress.stalled else 0
                board = self.tracker.digest()
                if progress.newly_done:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Progress: "
                                + "; ".join(progress.newly_done)
                                + f"  ({progress.done}/{progress.total} milestones verified)\n"
                                + board
                            ),
                        }
                    )
                elif stalled_turns >= 3 and board:
                    # No milestone has advanced in three motions: the plan is
                    # not working even if individual calls returned ok.
                    stalled_turns = 0
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "No milestone has advanced in the last 3 actions "
                                "even though calls reported ok. Re-observe and "
                                "change approach.\n" + board
                            ),
                        }
                    )

            # A skill that reported ok but whose physical effect was refuted
            # is the most dangerous state in the system: escalate it loudly.
            pc = (result.get("postcondition") or {}) if isinstance(result, dict) else {}
            if pc.get("status") == "refuted" and result.get("self_reported_ok"):
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"WARNING: {call.name} reported success but independent "
                            f"observation contradicts it -- {pc.get('evidence', '')}. "
                            "Do not build on this step; re-observe and redo it."
                        ),
                    }
                )

            # Failure escalation: one visual suggestion from the advisor.
            if not ok and self.advisor is not None and self.advisor.should_consult:
                jpeg = self.runtime.frame_jpeg()
                if jpeg:
                    suggestion = self.advisor.suggest(
                        task, jpeg, failure=str(result.get("error", ""))
                    )
                    if suggestion:
                        self.runtime.memory.add("note", f"advisor: {suggestion}")
                        messages.append(
                            {"role": "user", "content": f"Advisor suggestion: {suggestion}"}
                        )

            # Refresh the agent's situational context periodically.
            extra: dict = {}
            if self.attach_images and self.runtime.last_frame is not None:
                extra = {"images": [self.runtime.frame_jpeg()]}
            if not ok and consecutive_failures >= 2:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Two consecutive failures. Change strategy: re-observe, "
                            "try a different approach, or finish with task_done("
                            "success=false, honest summary).\nMemory:\n"
                            + self.runtime.memory.digest(max_lines=10)
                        ),
                        **extra,
                    }
                )

        self.runtime.trace.finish(f"task: {task}\nsuccess: false\nran out of steps ({self.max_steps})")
        status, unverified = self._final_check(False)
        return TaskReport(
            task, False, "step budget exhausted", self.max_steps, milestones, tool_log,
            duration_s=round(time.monotonic() - t_start, 2),
            milestone_status=status, unverified=unverified,
        )

    # ── Pigey: outcome tracking ──────────────────────────────────────────

    def _check_progress(self, skill_name: str):
        """Re-verify milestones after a world-changing call.

        Only runs after motions: verification costs a belief lookup (and, at
        most ``max_visual_checks`` times, one VLM turn), and nothing can have
        changed after a pure observation.
        """
        if self.tracker is None or not self.tracker.active:
            return None
        from ..skills.runtime import _MOTION_SKILLS

        if skill_name not in _MOTION_SKILLS:
            return None
        jpeg = self.runtime.frame_jpeg() if self.attach_images else None
        try:
            return self.tracker.update(jpeg)
        except Exception:
            return None

    def _final_check(self, claimed_success: bool) -> tuple[list[dict], list[str]]:
        """Last verification pass before the report is written.

        Returns (milestone status, milestones that are NOT verified done).
        Only a claimed success is worth spending a visual check on -- a
        self-declared failure needs no contradicting.
        """
        if self.tracker is None or not self.tracker.active:
            return [], []
        try:
            jpeg = self.runtime.frame_jpeg() if (claimed_success and self.attach_images) else None
            self.tracker.update(jpeg, allow_visual=claimed_success)
            return self.tracker.as_list(), (self.tracker.unverified() if claimed_success else [])
        except Exception:
            return [], []

    def _try_fast_path(self, task: str, t_start: float) -> tuple[TaskReport | None, str | None]:
        """Reflex/experience execution; (report, None) on success, or
        (None, note-for-the-LLM) when the fast attempt failed or no fast
        plan exists."""
        plan = self.fast_planner.plan(task)
        if plan is None:
            return None, None
        tool_log: list[dict] = []
        for i, (name, args) in enumerate(plan.calls, start=1):
            self.runtime.current_tier = str(plan.source)  # reflex | experience
            try:
                result = self.runtime.execute(name, args)
            finally:
                self.runtime.current_tier = None
            tool_log.append({"step": i, "tool": name, "args": args, "result": result})
            if not result.get("ok", False):
                self.fast_planner.note_outcome(
                    task, plan.calls, False, time.monotonic() - t_start
                )
                note = (
                    f"(A fast {plan.source} plan was tried first and FAILED at "
                    f"{name}({json.dumps(args)}): {str(result.get('error', ''))[:200]}. "
                    "Diagnose before retrying the same thing.)"
                )
                return None, note
        duration = round(time.monotonic() - t_start, 2)
        self.fast_planner.note_outcome(task, plan.calls, True, duration)
        # Agentic-VLA curriculum: credit each sub-goal separately as well, so a
        # clause proven inside this sequence warm-starts any FUTURE task that
        # contains it -- including a different sequence. Without this the
        # memory only ever learns whole instructions verbatim, and the
        # decomposition buys nothing after the first run.
        if plan.subgoals:
            per = duration / max(len(plan.subgoals), 1)
            for clause, clause_calls in plan.subgoal_spans():
                self.fast_planner.note_subgoal_outcome(
                    clause, clause_calls, True, per
                )
        summary = (
            f"done via {plan.source} path in {duration}s: "
            + "; ".join(f"{n}({_short_args(a)})" for n, a in plan.calls)
        )
        self.runtime.trace.finish(
            f"task: {task}\nsuccess: true\npath: {plan.source}\nduration_s: {duration}\n{summary}"
        )
        return (
            TaskReport(
                task, True, summary, len(plan.calls), [], tool_log,
                path=plan.source, duration_s=duration,
            ),
            None,
        )

    @staticmethod
    def _prune_images(messages: list[dict]) -> list[dict]:
        """Keep only the newest image in context: frames are ~50-100 KB of
        base64 each and every retained one is re-sent on every request."""
        last_with_images = None
        for i, m in enumerate(messages):
            if m.get("images"):
                last_with_images = i
        if last_with_images is None:
            return messages
        pruned = []
        for i, m in enumerate(messages):
            if m.get("images") and i != last_with_images:
                m = {k: v for k, v in m.items() if k != "images"}
            pruned.append(m)
        return pruned

    def _decompose(self, task: str) -> list[str]:
        try:
            resp: LLMResponse = self.llm.chat(
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": DECOMPOSE_PROMPT.format(task=task)}],
                max_tokens=300,
            )
        except Exception:
            return []
        lines = []
        for raw in resp.text.splitlines():
            line = raw.strip().lstrip("0123456789.-* ").strip()
            if line:
                lines.append(line)
        return lines[:6]
