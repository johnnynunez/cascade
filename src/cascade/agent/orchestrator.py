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

from ..conversation.receipts import agent_tool_output
from ..memory import recipes
from ..skills.runtime import TOOL_SPECS, SkillRuntime, _MOTION_SKILLS
from .advisor import Advisor
from .aspire import retrieve as retrieve_skills
from .llm import LLMClient, LLMResponse
from .milestones import IMPLAUSIBLE, SKIPPED, MilestoneTracker, PlausibilityChecker, make_vlm_verifier
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
    #: set when the task ended on a `stuck` step (RPent's third finish
    #: status): {"skill", "args", "ask"} -- the ask is the human-actionable
    #: request, relayed verbatim; the step was NOT retried
    stuck: dict | None = None


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
        memory_frames_k: int = 4,
        plausibility: PlausibilityChecker | None = None,
        programs=None,
    ):
        self.llm = llm
        self.runtime = runtime
        self._composed = getattr(runtime, "robot_mode", None) == "composed"
        self._mobile = getattr(runtime, "robot_mode", None) in {"mobile", "composed"}
        if self._composed:
            self.system_prompt = runtime.system_prompt
            self.tool_specs = runtime.tool_specs
        elif self._mobile:
            from ..skills.mobile_runtime import SYSTEM_PROMPT as MOBILE_PROMPT

            self.system_prompt = getattr(runtime, "system_prompt", MOBILE_PROMPT)
            self.tool_specs = runtime.tool_specs
        else:
            self.system_prompt = SYSTEM_PROMPT
            self.tool_specs = TOOL_SPECS
        self.advisor = advisor
        self.max_steps = max_steps
        self.decompose = decompose
        self.attach_images = attach_images and llm.supports_vision
        #: Vesta memory harness (arXiv:2606.20905 §2.4): how many captioned
        #: PAST frames ride along on every planner turn, on top of the
        #: current view. 0 = text-only history (the configuration Vesta's
        #: ablation scores lowest: the planner over-trusts the text and keeps
        #: "continuing the current task"). Needs a vision model.
        self.memory_frames_k = int(memory_frames_k) if self.attach_images else 0
        #: keyframes a just-executed `recall_step` loaded, shown to the planner
        #: on its next turn only (RPent `view_env_state(step=N)`), then dropped
        self._pending_recall: list[dict] = []
        # Existing reflexes/experience encode arm keyframes and grasps; never
        # consult that library for another morphology.
        self.fast_planner = None if self._mobile else fast_planner
        self.skill_library = None if self._mobile else skill_library
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
            if verify_milestones and not self._mobile
            else None
        )
        #: Human-CLAW pre-motion plausibility critic (ROADMAP follow-up #6),
        #: consulted BEFORE a motion skill is dispatched in the LLM tier.
        #: ADVISORY ONLY: it never refuses, delays beyond its per-task budget
        #: or rewrites a call -- the safety harness is the sole authority
        #: that refuses motion. `None` (agent.premotion_check: false) is the
        #: pre-2026-10-07 path exactly: no runtime attribute is touched and
        #: no key is added to any result. Arm skills only, like the tracker.
        self.plausibility = None if self._mobile else plausibility
        #: ROADMAP #8 programs tier (agent/programs.ProgramTier; design note
        #: docs/PROGRAMS_TIER.md). OPT-IN: `None` (agent.programs: false, the
        #: default) is the pre-2026-10-08 dispatch exactly -- no authoring turn,
        #: no runtime write, no store touched. Arm skills only, like the
        #: fast planner: programs encode manipulation skills.
        self.programs = None if self._mobile else programs

    def run_task(self, task: str) -> TaskReport:
        # One persistence budget for the WHOLE task, across tiers: the reflex
        # tier's pick_and_place, then the LLM tier's retry of the same call,
        # share it instead of each bringing a fresh `persist_seconds`.
        begin_task = getattr(self.runtime, "begin_task", None)
        if begin_task is not None:
            begin_task()
        begin = getattr(self.runtime, "begin_task_budget", None)
        if begin is not None:
            begin()
        try:
            report = self._run_task(task)
        finally:
            end = getattr(self.runtime, "end_task_budget", None)
            if end is not None:
                end()
        # which tier actually served the command -- rendered as the
        # dashboard's "via:" chip, so habit/reflex hits are visibly LLM-free
        self.runtime.last_path = report.path
        return report

    def _run_task(self, task: str) -> TaskReport:
        t_start = time.monotonic()
        fast_note = None
        tool_log: list[dict] = []
        if self.fast_planner is not None:
            report, fast_note = self._try_fast_path(task, t_start, tool_log=tool_log)
            if report is not None:
                return report
        # ROADMAP #8 programs tier (opt-in): consulted only when tiers 1/2 had
        # no plan at all. A habit that FAILED goes straight to deliberation,
        # which can see the failure; a program that stops hands its
        # next_action to the LLM tier the same way.
        if self.programs is not None and fast_note is None:
            report, fast_note = self._try_program_tier(task, t_start, tool_log=tool_log)
            if report is not None:
                return report

        milestones = self._decompose(task) if self.decompose else []
        if self.tracker is not None:
            self.tracker.reset(milestones)
        if self.plausibility is not None:
            self.plausibility.reset()  # the critic's VLM budget is per task
        # New episode, new visual history (Vesta: plan.ResetSession()). The
        # text ring is a rolling 15 s window and is left alone.
        try:
            self.runtime.memory.reset_frames()
        except AttributeError:
            pass
        messages: list[dict] = []

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
        # With a memory embedder (ROADMAP #7, opt-in) the notes are ranked by
        # text-embedding similarity instead; without one the call is the
        # pre-embedder call exactly.
        if self.skill_library is not None:
            try:
                embedder = getattr(getattr(self.runtime, "memory", None), "embedder", None)
                lib = (retrieve_skills(self.skill_library, task) if embedder is None
                       else retrieve_skills(self.skill_library, task, embedder=embedder))
                if lib:
                    intro += "\n\n" + lib
            except Exception:
                pass
        # ROADMAP #7 action<->object consolidation (opt-in): what earlier
        # plans did to the objects this task names, across wordings.
        action_objects = getattr(self.fast_planner, "action_objects", None)
        if action_objects is not None:
            try:
                digest = action_objects.agent_digest(task)
                if digest:
                    intro += "\n\n" + digest
            except Exception:  # noqa: BLE001 -- advisory context never fails a task
                pass
        intro += "\n\nBegin. Observe first, then act. Call one tool now."
        messages.append({"role": "user", "content": intro})

        consecutive_failures = 0
        stalled_turns = 0
        #: Task-Specific Memory: the scene as perceived BEFORE the first
        #: motion, so a successful run can be stored with its coordinates
        #: expressed relative to objects (memory/recipes.py), never raw.
        scene0: list[dict] | None = None
        for step in range(1, self.max_steps + 1):
            resp = self.llm.chat(
                system=self.system_prompt,
                messages=self._with_memory_harness(messages),
                tools=self.tool_specs,
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
            if scene0 is None and call.name in _MOTION_SKILLS:
                scene0 = self._scene_snapshot()
            # Human-CLAW pre-motion critic: judged BEFORE dispatch, acted on
            # by nobody but the planner. Whatever it says, the call below is
            # the planner's call, unchanged -- only the harness refuses motion.
            advisory = self._premotion_check(call.name, call.arguments)
            self.runtime.current_tier = "llm"
            if advisory is not None:
                # Hand-off to execute() so the trace row carries the verdict
                # (the `current_tier` pattern); cleared in the finally below.
                self.runtime.pending_plausibility = advisory
            try:
                result = self.runtime.execute(call.name, call.arguments)
            finally:
                self.runtime.current_tier = None
                if advisory is not None:
                    self.runtime.pending_plausibility = None
            if advisory is not None and isinstance(result, dict) and "plausibility" not in result:
                result["plausibility"] = advisory  # untraced early return: attach here
            tool_log.append({"step": step, "tier": "llm", "tool": call.name,
                             "args": call.arguments, "result": result})

            if call.name == "task_done" and result.get("task_complete"):
                summary = str(call.arguments.get("summary", ""))
                success = _as_bool(call.arguments.get("success", False))
                task_verifier = getattr(self.runtime, "unverified_actions", None)
                if task_verifier is not None:
                    success = success and result.get("success") is True
                report = self._finish_report(task, success, summary, step, milestones, tool_log, t_start)
                if report.success:
                    # Only a VERIFIED success is worth remembering: the
                    # report's success already folds in every unverified
                    # effect obligation, so an unconfirmed pick never
                    # becomes a habit (same rule the fast tier applies).
                    self._remember_recipe(task, tool_log, scene0, report)
                    if self.programs is not None:
                        self._remember_program(task, tool_log, scene0, report)
                return report

            # RPent `finish(status=stuck)`: the skill exhausted what the robot
            # can do on its own and named what the HUMAN should change. The
            # task ends here -- no re-plan, no retry of the step with a fresh
            # budget -- and the ask is relayed verbatim to whoever is reading.
            if isinstance(result, dict) and result.get("outcome") == "stuck":
                return self._finish_stuck(task, call.name, dict(call.arguments), result,
                                          step, milestones, tool_log, t_start)

            # The planner asked to see a past step: show it the recalled
            # keyframes on its NEXT turn, inside the one message that carries
            # images (the memory harness), then forget them.
            if call.name == "recall_step" and result.get("ok"):
                self._pending_recall = list(getattr(self.runtime, "last_recalled_frames", None) or [])

            ok = bool(result.get("ok"))
            consecutive_failures = 0 if ok else consecutive_failures + 1
            if self.advisor is not None:
                self.advisor.note_outcome(ok)

            # Mobile/composed runtimes record the complete receipt in their trace
            # before returning. Their motion receipts carry verifier evidence,
            # samples and state snapshots (a native MicroDuck step is ~1.6 MB), so
            # the model gets a bounded view: exact verdicts, metrics and bindings,
            # named bulky attachments omitted with digests.
            content = agent_tool_output(result, recorded=True) if self._mobile else json.dumps(result)
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

            # Pre-motion critic said "implausible": the planner reads a
            # caution on its next turn, nothing else happens. Same booth
            # rule as envelope notes -- advice, never a veto.
            if advisory is not None and advisory.get("verdict") == IMPLAUSIBLE:
                why = "; ".join(str(r) for r in (advisory.get("reasons") or [])) or "no reason given"
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Caution (advisory only): before {call.name}({_short_args(call.arguments)}) "
                            f"ran, the pre-motion plausibility check judged it IMPLAUSIBLE -- {why} "
                            "The call was executed unchanged; the safety harness alone decides what "
                            "moves. Compare its result against the current view before building on it."
                        ),
                    }
                )
                try:
                    self.runtime.memory.add(
                        "note", f"plausibility: {call.name} judged implausible -- {why[:120]}"
                    )
                except Exception:  # noqa: BLE001 -- narration must never fail a task
                    pass

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

        return self._finish_report(task, False, "step budget exhausted", self.max_steps,
                                   milestones, tool_log, t_start)

    def _finish_report(self, task, success, summary, steps, milestones, tool_log,
                       t_start, *, path="llm", verify_milestones=True, stuck=None):
        """Every normal exit retains the task's unresolved effects and history."""
        status, unverified = self._final_check(success) if verify_milestones else ([], [])
        task_verifier = getattr(self.runtime, "unverified_actions", None)
        if task_verifier is not None:
            unverified = list(dict.fromkeys([*unverified, *task_verifier()]))
            success = success and not unverified
        if unverified:
            summary += "\n[verification] could not confirm: " + "; ".join(unverified)
        if stuck is not None:
            success = False  # a stuck task is never a success, whatever was claimed
        duration = round(time.monotonic() - t_start, 2)
        # summary.txt records the third outcome explicitly: a reader of the
        # run directory sees `outcome: stuck` and the ask verbatim, not just
        # `success: False`.
        stuck_lines = (f"outcome: stuck\nask: {stuck['ask']}\n" if stuck is not None else "")
        self.runtime.trace.finish(
            f"task: {task}\nsuccess: {success}\n{stuck_lines}steps: {steps}\npath: {path}\n"
            f"duration_s: {duration}\n{summary}"
        )
        return TaskReport(
            task, success, summary, steps, milestones, tool_log, path=path,
            duration_s=duration, milestone_status=status, unverified=unverified,
            stuck=stuck,
        )

    def _finish_stuck(self, task, skill, args, result, steps, milestones, tool_log,
                      t_start, *, path="llm"):
        """End the task on a `stuck` step: the ask is relayed VERBATIM (report
        summary, memory note, summary.txt) and the step is not retried by
        any tier. The human changes the scene or the instruction; the next
        task starts fresh."""
        ask = str(result.get("ask") or "")
        stuck = {"skill": skill, "args": args, "ask": ask}
        try:
            self.runtime.memory.add("note", f"stuck at {skill}: {ask}")
        except Exception:  # noqa: BLE001 -- reporting must never break the exit
            pass
        summary = f"stuck at {skill}({_short_args(args)}): {ask}"
        return self._finish_report(task, False, summary, steps, milestones, tool_log,
                                   t_start, path=path, verify_milestones=False, stuck=stuck)

    # ── Pigey: outcome tracking ──────────────────────────────────────────

    def _premotion_check(self, skill_name: str, args: dict) -> dict | None:
        """Human-CLAW pre-execution interrogation of a proposed MOTION call.

        Returns the advisory ``{"verdict", "reasons", "source"}`` or ``None``
        when the critic is off or the call is not a motion.  Motions only,
        for the same reason ``_check_progress`` is: an observation changes
        nothing worth a VLM turn.  Mirrors the `execute()` contract -- when
        no frame exists yet, take one (execute() would do the same a moment
        later) so the critic judges the current view, not a blank.  Never
        raises; a fault here is recorded as ``skipped``, never a stall.
        """
        if self.plausibility is None:
            return None
        from ..skills.runtime import _MOTION_SKILLS

        if skill_name not in _MOTION_SKILLS:
            return None
        try:
            if getattr(self.runtime, "last_frame", None) is None:
                try:
                    self.runtime.observe()
                except Exception:  # noqa: BLE001 -- camera hiccup: judge without a frame (-> skipped)
                    pass
            jpeg = self.runtime.frame_jpeg()
            return self.plausibility.check(skill_name, args, jpeg)
        except Exception as e:  # noqa: BLE001 -- advisory code can never block a motion
            return {
                "verdict": SKIPPED,
                "reasons": [f"critic failed: {type(e).__name__}: {e}"],
                "source": "none",
            }

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

    def _try_fast_path(self, task: str, t_start: float, *, tool_log=None) -> tuple[TaskReport | None, str | None]:
        """Reflex/experience execution; (report, None) on success, or
        (None, note-for-the-LLM) when the fast attempt failed or no fast
        plan exists."""
        plan = self.fast_planner.plan(task)
        if plan is None:
            return None, None
        if tool_log is None:
            tool_log = []
        # Task-Specific Memory (Harness-VLA v4, memory/recipes.py): a
        # recalled recipe carries perception QUERIES where the original run
        # had coordinates. Every one of them is re-grounded through the
        # runtime's own perception NOW, before the first motion. A query
        # that does not resolve aborts the whole replay to the LLM tier --
        # there is no stored coordinate to fall back to, by construction.
        calls = plan.calls
        n_grounded = 0
        if plan.needs_grounding:
            try:
                calls, n_grounded = self._ground_recipe(plan, tool_log)
            except recipes.GroundingError as e:
                note = (
                    f"(A remembered {plan.source} recipe for this command could not be "
                    f"re-grounded on the current scene: {e}. No motion was attempted; "
                    "observe and plan from what is actually on the table.)"
                )
                return None, note
        offset = len(tool_log)
        for i, (name, args) in enumerate(calls, start=1):
            self.runtime.current_tier = str(plan.source)  # reflex | experience
            try:
                result = self.runtime.execute(name, args)
            finally:
                self.runtime.current_tier = None
            tool_log.append({"step": offset + i, "tier": str(plan.source), "tool": name,
                             "args": args, "result": result})
            if isinstance(result, dict) and result.get("outcome") == "stuck":
                # The fast tier's step needs a human. Escalating to the LLM
                # would only buy the same pick a fresh persistence budget --
                # a visitor watching minutes of retries. End here, ask.
                self.fast_planner.note_outcome(task, plan.calls, False, time.monotonic() - t_start)
                self._note_action_objects(task, plan.calls, False, failed_at=i - 1)
                return (
                    self._finish_stuck(task, name, dict(args), result, i, [], tool_log, t_start,
                                       path=str(plan.source)),
                    None,
                )
            task_verifier = getattr(self.runtime, "unverified_actions", None)
            unverified = task_verifier() if task_verifier is not None else []
            if not result.get("ok", False) or unverified:
                # plan.calls, not the grounded `calls`: a recipe's record
                # must keep its queries.
                self.fast_planner.note_outcome(
                    task, plan.calls, False, time.monotonic() - t_start
                )
                self._note_action_objects(task, plan.calls, False, failed_at=i - 1)
                note = (
                    f"(A fast {plan.source} plan was tried first and FAILED at "
                    f"{name}({json.dumps(args)}): {str(result.get('error') or '; '.join(unverified))[:200]}. "
                    "Diagnose before retrying the same thing.)"
                )
                return None, note
        duration = round(time.monotonic() - t_start, 2)
        meta = {"summary": plan.summary} if plan.summary else {}
        self.fast_planner.note_outcome(task, plan.calls, True, duration, **meta)
        # Consolidated ONCE for the executed plan, before the per-sub-goal
        # credits below (which re-credit the same execution under clause
        # text and must not count twice per action and object).
        self._note_action_objects(task, plan.calls, True)
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
            + "; ".join(f"{n}({_short_args(a)})" for n, a in calls)
        )
        if n_grounded:
            summary += (
                f" (recipe: {n_grounded} perception quer{'y' if n_grounded == 1 else 'ies'} "
                "re-grounded before motion)"
            )
        return (
            self._finish_report(task, True, summary, len(calls), [], tool_log, t_start,
                                path=plan.source, verify_milestones=False),
            None,
        )

    def _ground_recipe(self, plan, tool_log: list) -> tuple[list, int]:
        """Resolve a recipe's symbolic targets through `localize_object`.

        Perception only -- `localize_object` is not in `_MOTION_SKILLS` --
        and every lookup is logged under the plan's tier so the trace shows
        the replay re-grounded before it moved. Raises GroundingError.
        """
        def localize(label: str) -> dict:
            self.runtime.current_tier = str(plan.source)
            try:
                result = self.runtime.execute("localize_object", {"label": label})
            finally:
                self.runtime.current_tier = None
            tool_log.append({"step": len(tool_log) + 1, "tier": str(plan.source),
                             "tool": "localize_object", "args": {"label": label},
                             "result": result, "grounding": True})
            return result

        return recipes.ground(plan.calls, localize)

    def _scene_snapshot(self) -> list[dict]:
        """Objects the belief store knows right now, as recipe anchors."""
        beliefs = getattr(self.runtime, "beliefs", None)
        if beliefs is None:
            return []
        try:
            return [
                {"label": b.label, "position": [float(v) for v in b.position]}
                for b in beliefs.all()
            ]
        except Exception:  # noqa: BLE001 -- a snapshot failure only costs the recipe
            return []

    def _remember_recipe(self, task: str, tool_log: list, scene0, report: TaskReport) -> None:
        """Store a verified LLM-tier run as a Task-Specific Memory recipe.

        Coordinates become `localize_object(label) + offset` queries anchored
        on the scene before the first motion; a run whose coordinates cannot
        be anchored is NOT stored (never a raw coordinate), and says so in
        the episodic memory so the omission is visible.
        """
        planner = self.fast_planner
        if planner is None or getattr(planner, "experience", None) is None:
            return
        try:
            scene = recipes.anchor_scene(tool_log, scene0 or [])
            steps = recipes.symbolize_run(tool_log, scene)
        except recipes.RecipeError as e:
            try:
                self.runtime.memory.add("note", f"run not kept as a recipe: {e}")
            except Exception:  # noqa: BLE001
                pass
            return
        except Exception:  # noqa: BLE001 -- memory bookkeeping never fails a finished task
            return
        if not steps:
            return  # nothing moved: an answer, not a recipe
        first_line = (report.summary or "").strip().splitlines()
        summary = (first_line[0] if first_line else f"completed: {task}")[:200]
        run_dir = getattr(getattr(self.runtime, "trace", None), "run_dir", None)
        try:
            planner.note_outcome(
                task, steps, True, report.duration_s,
                summary=summary, source_run=(run_dir.name if run_dir is not None else None),
            )
        except Exception:  # noqa: BLE001
            pass
        self._note_action_objects(task, steps, True)

    def _note_action_objects(self, task: str, calls, success: bool, *,
                             failed_at: int | None = None) -> None:
        """Feed the optional action<->object consolidation (ROADMAP #7) with
        one EXECUTED plan. A no-op without it (the shipped default)."""
        store = getattr(self.fast_planner, "action_objects", None)
        if store is None:
            return
        try:
            store.record_plan(calls, success, instruction=task, failed_at=failed_at)
        except Exception:  # noqa: BLE001 -- memory bookkeeping never fails a task
            pass

    # ── ROADMAP #8: the programs tier (opt-in; docs/PROGRAMS_TIER.md) ───────

    def _try_program_tier(self, task: str, t_start: float, *, tool_log: list) -> tuple[TaskReport | None, str | None]:
        """One authoring turn, then the program step by step; (report, None)
        when it completed with every registered effect confirmed, else
        (None, note-for-the-LLM-tier).

        The brain writes a fresh program or reuses a PROMOTED one (candidates
        are never offered); a reply that is not a well-formed program is
        refused before anything runs. Every step is a top-level
        `runtime.execute()` call -- the harness stays the sole authority that
        refuses motion -- and the first failed / refused / refuted /
        unverified step stops the program with a next_action. A stuck step
        ends the task, exactly as in the other tiers.
        """
        from .programs import COMPLETED, INVALID, NONE as NO_PROGRAM, STUCK, write_receipt

        tier = self.programs
        offered = tier.offered(task)
        try:
            resp = self.llm.chat(
                system=self.system_prompt,
                messages=[{"role": "user", "content": tier.prompt(task, offered, self._world_labels())}],
                max_tokens=1200,
            )
        except Exception as e:  # noqa: BLE001 -- no program; the LLM tier proceeds unchanged
            self._memory_note(f"program authoring failed: {type(e).__name__}: {e}")
            return None, None
        if resp.tool_calls:
            return None, None  # a tool call is not a program: the LLM tier gets the task as usual
        proposal = tier.parse(resp.text, offered)
        if proposal.kind == NO_PROGRAM:
            return None, None
        if proposal.kind == INVALID:
            self._memory_note(f"program refused before it ran: {proposal.reason[:160]}")
            return None, (
                f"(A program was proposed for this task but refused before anything ran: {proposal.reason}. "
                "No motion was attempted; plan the task step by step.)"
            )
        scene0 = self._scene_snapshot()
        run = tier.run(proposal, self.runtime, tool_log=tool_log)
        write_receipt(run, getattr(getattr(self.runtime, "trace", None), "run_dir", None))
        run_id = self._task_run_id()
        if run.status == STUCK:
            tier.account(run, proposal, task=task, run_id=run_id, success=False)
            step = run.steps[-1]
            return (
                self._finish_stuck(task, step.tool, dict(step.args), step.result, len(run.steps), [],
                                   tool_log, t_start, path="program"),
                None,
            )
        task_verifier = getattr(self.runtime, "unverified_actions", None)
        unverified = task_verifier() if task_verifier is not None else []
        if run.status == COMPLETED and not unverified:
            report = self._finish_report(task, True, run.summary(), len(run.steps), [], tool_log, t_start,
                                         path="program", verify_milestones=False)
            tier.account(run, proposal, task=task, run_id=run_id, success=report.success)
            if report.success:
                # Same rule as a verified LLM-tier run: this instruction
                # becomes a tier-2 recipe (queries, never coordinates).
                self._remember_recipe(task, tool_log, scene0, report)
            return report, None
        tier.account(run, proposal, task=task, run_id=run_id, success=False)
        self._memory_note(f"program {run.program.name} {run.status}: {run.reason[:160]}")
        return None, run.note()

    def _remember_program(self, task: str, tool_log: list, scene0, report: TaskReport) -> None:
        """Distil a VERIFIED LLM-tier run into the program library (ROADMAP #8).

        The verdict handed to the library is read from the task-effects
        ledger, not from the report: every registered effect of the task must
        be CONFIRMED, else nothing is stored. Never fails a finished task.
        """
        from .effects import CONFIRMED

        try:
            ledger = getattr(self.runtime, "task_effects", None)
            rows = ledger().get("effects") if ledger is not None else None
            if not rows or not all(r.get("status") == CONFIRMED for r in rows):
                return  # no independent verdict = no evidence: nothing to distil, nothing to say
            scene = recipes.anchor_scene(tool_log, scene0 or [])
            first = (report.summary or "").strip().splitlines()
            status, _ = self.programs.distil(tool_log, scene, task=task, run_id=self._task_run_id(),
                                             verdict=CONFIRMED, summary=(first[0] if first else "")[:200])
            if status != "admitted":
                self._memory_note(f"run {status}")
        except Exception:  # noqa: BLE001 -- memory bookkeeping never fails a finished task
            pass

    def _world_labels(self) -> list[str]:
        """The world model for the authoring turn: labels, colour and state
        only. No positions -- a program must not be written from coordinates."""
        beliefs = getattr(self.runtime, "beliefs", None)
        try:
            rows = list(beliefs.summary()) if beliefs is not None else []
        except Exception:  # noqa: BLE001
            rows = []
        out = []
        for row in rows[:12]:
            if not isinstance(row, dict) or not row.get("label"):
                continue
            color = f" ({row['color']})" if row.get("color") else ""
            out.append(f"{row['label']}{color} {row.get('state') or ''}".strip())
        return out

    def _task_run_id(self) -> str:
        """Identity of THIS task execution (run dir + task-effects id): the
        program library counts each execution once."""
        run_dir = getattr(getattr(self.runtime, "trace", None), "run_dir", None)
        task_id = None
        ledger = getattr(self.runtime, "task_effects", None)
        if ledger is not None:
            try:
                task_id = ledger().get("task_id")
            except Exception:  # noqa: BLE001
                task_id = None
        if not task_id:
            import uuid

            task_id = uuid.uuid4().hex
        return f"{run_dir.name if run_dir is not None else 'run'}#{task_id}"

    def _memory_note(self, text: str) -> None:
        try:
            self.runtime.memory.add("note", text)
        except Exception:  # noqa: BLE001 -- narration must never fail a task
            pass

    def _with_memory_harness(self, messages: list[dict]) -> list[dict]:
        """Messages for THIS planner turn: the conversation with all older
        images stripped, plus one trailing user message carrying the Vesta
        memory harness -- up to K captioned past frames (initial state first,
        newest action last, each with the independent verdict on it) and the
        current view. Rebuilt every turn and never appended to `messages`,
        so images are sent once per request, not accumulated.

        Falls back to the plain pruned conversation when there is nothing to
        show (no vision model, K=0, no frames yet)."""
        pruned = self._prune_images(messages)
        # A `recall_step` executed last turn: its BEFORE/AFTER keyframes ride
        # along ONCE, inside the single message that carries images, placed
        # before the current view so "the LAST image is the current view"
        # stays true. Text-only planners get the JSON result alone.
        recall = self._pending_recall if self.attach_images else []
        self._pending_recall = []
        if self.memory_frames_k <= 0 and not recall:
            return pruned
        try:
            frames = self.runtime.memory.memory_frames(self.memory_frames_k) if self.memory_frames_k > 0 else []
        except AttributeError:
            frames = []
        images = [f["jpeg"] for f in frames]
        lines = [self.runtime.memory.frame_caption(f) for f in frames]
        for fr in recall:
            if fr.get("jpeg"):
                images.append(fr["jpeg"])
                lines.append(f"recalled {fr.get('caption', 'step')}")
        current = self.runtime.frame_jpeg() if self.runtime.last_frame is not None else None
        if current is not None:
            images.append(current)
            lines.append("current view (now)")
        if not images:
            return pruned
        text = (
            "Memory frames, oldest first; the LAST image is the current view. "
            "Before choosing a tool, state briefly: Observation (what the "
            "current view shows), Progress (which steps are done, judged from "
            "the frames and verdicts, not from your intent), Reasoning (what "
            "must happen next and why), then call the tool.\n"
            + "\n".join(f"{i + 1}. {ln}" for i, ln in enumerate(lines))
        )
        if recall:
            text += ("\nThe 'recalled' image(s) are the keyframes recorded around the past "
                     "step you asked about, shown once; they are not the current view.")
        # Strip any image the pruned history still carries: the harness is
        # now the single place images enter the request.
        pruned = [
            ({k: v for k, v in m.items() if k != "images"} if m.get("images") else m)
            for m in pruned
        ]
        return pruned + [{"role": "user", "content": text, "images": images}]

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
                system=self.system_prompt,
                messages=[{"role": "user", "content": (
                    f"List bounded, independently checkable mobile milestones for: {task}"
                    if self._mobile else DECOMPOSE_PROMPT.format(task=task))}],
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
