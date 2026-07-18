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

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from ..skills.runtime import TOOL_SPECS, SkillRuntime
from .advisor import Advisor
from .llm import LLMClient, LLMResponse
from .prompts import DECOMPOSE_PROMPT, SYSTEM_PROMPT
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
    ):
        self.llm = llm
        self.runtime = runtime
        self.advisor = advisor
        self.max_steps = max_steps
        self.decompose = decompose
        self.attach_images = attach_images and llm.supports_vision
        self.fast_planner = fast_planner

    def run_task(self, task: str) -> TaskReport:
        t_start = time.monotonic()
        fast_note = None
        if self.fast_planner is not None:
            report, fast_note = self._try_fast_path(task, t_start)
            if report is not None:
                return report

        milestones = self._decompose(task) if self.decompose else []
        messages: list[dict] = []
        tool_log: list[dict] = []

        intro = f"Task: {task}\n"
        if milestones:
            intro += "Milestones:\n" + "\n".join(f"{i+1}. {m}" for i, m in enumerate(milestones)) + "\n"
        if fast_note:
            intro += f"\n{fast_note}\n"
        intro += (
            "\nMemory (last 15 s):\n" + self.runtime.memory.digest()
            + "\n\nBegin. Observe first, then act. Call one tool now."
        )
        messages.append({"role": "user", "content": intro})

        consecutive_failures = 0
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
            result = self.runtime.execute(call.name, call.arguments)
            tool_log.append({"step": step, "tool": call.name, "args": call.arguments, "result": result})

            if call.name == "task_done" and result.get("task_complete"):
                summary = str(call.arguments.get("summary", ""))
                success = _as_bool(call.arguments.get("success", False))
                self.runtime.trace.finish(
                    f"task: {task}\nsuccess: {success}\nsteps: {step}\n{summary}"
                )
                return TaskReport(
                    task, success, summary, step, milestones, tool_log,
                    duration_s=round(time.monotonic() - t_start, 2),
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
        return TaskReport(
            task, False, "step budget exhausted", self.max_steps, milestones, tool_log,
            duration_s=round(time.monotonic() - t_start, 2),
        )

    def _try_fast_path(self, task: str, t_start: float) -> tuple[TaskReport | None, str | None]:
        """Reflex/experience execution; (report, None) on success, or
        (None, note-for-the-LLM) when the fast attempt failed or no fast
        plan exists."""
        plan = self.fast_planner.plan(task)
        if plan is None:
            return None, None
        tool_log: list[dict] = []
        for i, (name, args) in enumerate(plan.calls, start=1):
            result = self.runtime.execute(name, args)
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
