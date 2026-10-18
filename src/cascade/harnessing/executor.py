# SPDX-License-Identifier: Apache-2.0
"""Rollouts: one execution of one harness on one task, on a fresh CASCADE runtime.

A rollout builds the configured stack (MuJoCo, mock, Isaac bridge...) with every learned store
isolated to the rollout directory (beliefs, grasp memory, envelope, episodic and action-object
stores), runs the built-in :class:`~cascade.agent.orchestrator.AgentOrchestrator` with the harness,
and records:

- the trajectory the improver reads: the agent's reasoning, each tool call and the result the
  agent received, and its own task_done claim;
- the measured outcome: ``TaskReport.success``, i.e. CASCADE's own verifier (postconditions on
  physics/belief/gripper channels, milestone checks, no stuck step). The measured outcome is NEVER
  shown to the improver (Harness²'s answer-free rule); it is recorded for the adoption gate and
  the report only.

No fast planner (habits/experience), programs tier or ASPIRE skill-library notes run inside a
rollout: the harness is the only learned text, so a contrast between rollouts is the harness's.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
import traceback
from typing import Any, Callable, Protocol

from .checks import CheckProbe, TaskCheck
from .surface import AgentHarness

#: per-step text caps in TRAJECTORY.md
REASONING_CHARS = 700
RESULT_CHARS = 700


@dataclass(frozen=True)
class TaskSpec:
    id: str
    task: str
    domain: str
    max_steps: int = 20
    #: the hidden success check (`checks.TaskCheck`); None: CASCADE's own verifier decides
    check: TaskCheck | None = None

    @classmethod
    def from_dict(cls, data: dict, *, domain: str | None = None) -> "TaskSpec":
        if not data.get("id") or not data.get("task"):
            raise ValueError(f"task needs id and task: {data!r}")
        return cls(str(data["id"]), str(data["task"]), str(data.get("domain") or domain or "default"),
                   int(data.get("max_steps", 20)), TaskCheck.from_dict(data.get("check")))


@dataclass
class Rollout:
    """`measured_success`: the task's hidden check when it has one (`check`), else CASCADE's
    verifier (`verifier_success`, i.e. `TaskReport.success`). Neither reaches an improver."""

    label: str
    task_id: str
    harness_sha: str
    measured_success: bool | None
    claimed_success: bool | None
    steps: int
    duration_s: float
    ended_by: str
    agent_summary: str
    verification: str = ""
    error: str | None = None
    trajectory: str = ""
    verifier_success: bool | None = None
    check: dict | None = None

    def as_dict(self) -> dict:
        return asdict(self)


class Executor(Protocol):
    def run(self, harness: AgentHarness, spec: TaskSpec, label: str, out_dir: Path) -> Rollout: ...


class RecordingLLM:
    """Wraps an LLM client and records every planner turn (text + tool calls)."""

    def __init__(self, inner):
        self.inner = inner
        self.turns: list[dict] = []
        self.supports_vision = getattr(inner, "supports_vision", False)

    def chat(self, system, messages, tools=None, max_tokens=1024):
        resp = self.inner.chat(system=system, messages=messages, tools=tools, max_tokens=max_tokens)
        if tools is not None:  # planner turns carry tools; decompose/verify turns do not
            self.turns.append({"text": resp.text or "",
                               "calls": [(c.name, dict(c.arguments or {})) for c in resp.tool_calls]})
        return resp

    def close(self):
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _clip(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _result_line(result: Any) -> str:
    if not isinstance(result, dict):
        return _clip(repr(result), RESULT_CHARS)
    keep = {}
    for key in ("ok", "error", "held", "outcome", "next_action", "ask", "postcondition", "warning",
                "refused", "verified", "objects_visible", "position", "placed"):
        if key in result:
            value = result[key]
            if key == "objects_visible" and isinstance(value, list):
                value = [o.get("label") for o in value if isinstance(o, dict)]
            if key == "postcondition" and isinstance(value, dict):
                value = {k: value.get(k) for k in ("status", "channel", "evidence") if k in value}
            keep[key] = value
    if not keep:
        keep = {k: result[k] for k in list(result)[:6]}
    return _clip(json.dumps(keep, default=str), RESULT_CHARS)


def render_trajectory(label: str, spec: TaskSpec, harness: AgentHarness, turns: list[dict],
                      tool_log: list[dict], *, ended_by: str, agent_summary: str,
                      claimed: bool | None) -> str:
    lines = [f"# Execution {label}", "", f"Task: {spec.task}", f"Harness: {harness.sha()[:12]}", ""]
    log = list(tool_log)
    step = 0
    for turn in turns:
        step += 1
        lines.append(f"## Step {step}")
        if turn["text"].strip():
            lines.append(f"Reasoning: {_clip(turn['text'], REASONING_CHARS)}")
        if not turn["calls"]:
            lines += ["(no tool call)", ""]
            continue
        name, args = turn["calls"][0]
        lines.append(f"Call: {name} {_clip(json.dumps(args, default=str), 300)}")
        entry = log.pop(0) if log and log[0].get("tool") == name else None
        if entry is None:
            entry = next((e for e in log if e.get("tool") == name), None)
            if entry is not None:
                log.remove(entry)
        lines.append(f"Result: {_result_line(entry['result']) if entry else '(not executed)'}")
        lines.append("")
    for entry in log:  # tool calls a non-LLM tier made (grounding, plans)
        step += 1
        lines += [f"## Step {step} ({entry.get('tier', 'runtime')})",
                  f"Call: {entry.get('tool')} {_clip(json.dumps(entry.get('args', {}), default=str), 300)}",
                  f"Result: {_result_line(entry.get('result'))}", ""]
    claim = "no task_done call" if claimed is None else f"task_done(success={claimed})"
    lines += ["## End", f"Ended by: {ended_by}; the agent's own claim: {claim}",
              f"Agent summary: {_clip(agent_summary, 600)}", ""]
    return "\n".join(lines)


def _task_done(tool_log: list[dict]) -> tuple[bool | None, str]:
    for entry in reversed(tool_log):
        if entry.get("tool") == "task_done":
            args = entry.get("args") or {}
            return bool(args.get("success")), str(args.get("summary", ""))
    return None, ""


def isolated_config(cfg, out_dir: Path):
    """The configured stack with every learned store private to this rollout."""
    from ..config import Cfg

    out_dir = Path(out_dir).resolve()  # the note library path must be absolute (skills.notes)
    data = cfg.as_dict()
    memory = data.setdefault("memory", {})
    memory.update(persist_beliefs=False, beliefs_path=str(out_dir / "beliefs.json"),
                  persist_episodic=False, episodic_path=str(out_dir / "episodic.json"),
                  envelope_path=str(out_dir / "envelope.json"), action_objects=False,
                  skill_library_path=str(out_dir / "skills_library"))
    data.setdefault("grasp", {})["memory_path"] = str(out_dir / "grasp_memory.json")
    data.setdefault("agent", {}).update(programs=False, sim_retry=False, harness_path=None)
    return Cfg(data)


@dataclass
class CascadeExecutor:
    """Rollouts on a freshly built CASCADE runtime (one per rollout)."""

    camera: str = "mujoco_scene"
    arm: str = "so101_mujoco"
    llm_profile: str = "mock"
    llm_factory: Callable[[Any, str], Any] | None = None
    decompose: bool = True
    verify_milestones: bool = True
    runtime_hook: Callable[[Any, Any], None] | None = None
    env: dict = field(default_factory=dict)

    def _config(self):
        from ..config import load_demo_config

        return load_demo_config(camera=self.camera, arm=self.arm, llm=self.llm_profile)

    def run(self, harness: AgentHarness, spec: TaskSpec, label: str, out_dir: Path) -> Rollout:
        import os

        from ..agent.llm import make_llm
        from ..agent.orchestrator import AgentOrchestrator
        from ..apps.demo import build_runtime, shutdown_runtime

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved_env = {k: os.environ.get(k) for k in self.env}
        os.environ.update({k: str(v) for k, v in self.env.items()})
        t0 = time.monotonic()
        runtime = arm = None
        llm = None
        try:
            cfg = isolated_config(self._config(), out_dir)
            runtime, arm = build_runtime(cfg, out_dir / "run")
            if self.runtime_hook is not None:
                self.runtime_hook(runtime, arm)
            probe = CheckProbe(spec.check, runtime, arm) if spec.check is not None else None
            inner = self.llm_factory(cfg, label) if self.llm_factory else make_llm(cfg.llm)
            llm = RecordingLLM(inner)
            orch = AgentOrchestrator(llm, runtime, advisor=None, max_steps=spec.max_steps,
                                     decompose=self.decompose, fast_planner=None, skill_library=None,
                                     verify_milestones=self.verify_milestones,
                                     memory_frames_k=int(cfg.memory.get("frames_k", 4)),
                                     harness=harness)
            report = orch.run_task(spec.task)
            claimed, claim_summary = _task_done(report.tool_log)
            ended_by = ("stuck" if report.stuck else "task_done" if claimed is not None
                        else "cancelled" if report.cancelled else "step budget or stall")
            summary = claim_summary or report.summary.split("\n[verification]")[0]
            traj = render_trajectory(label, spec, harness, llm.turns, report.tool_log,
                                     ended_by=ended_by, agent_summary=summary, claimed=claimed)
            verification = "; ".join(report.unverified) if report.unverified else ""
            verifier = bool(report.success)
            check = None
            measured = verifier
            if probe is not None:
                measured, rows = probe.evaluate(runtime, arm)
                check = {"passed": measured, "criteria": rows}
            rollout = Rollout(label, spec.id, harness.sha(), measured, claimed, report.steps,
                              round(time.monotonic() - t0, 1), ended_by, summary, verification,
                              trajectory=str(out_dir / "TRAJECTORY.md"), verifier_success=verifier, check=check)
        except Exception as exc:  # noqa: BLE001 -- a broken rollout is recorded, never fatal
            traj = (f"# Execution {label}\n\nTask: {spec.task}\nHarness: {harness.sha()[:12]}\n\n"
                    f"The execution failed before finishing: {_clip(repr(exc), 500)}\n")
            rollout = Rollout(label, spec.id, harness.sha(), None, None, 0, round(time.monotonic() - t0, 1),
                              "executor error", "", error=traceback.format_exc()[-2000:],
                              trajectory=str(out_dir / "TRAJECTORY.md"))
        finally:
            if llm is not None:
                llm.close()
            if runtime is not None:
                try:
                    shutdown_runtime(runtime, arm)
                except Exception:  # noqa: BLE001
                    pass
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        (out_dir / "TRAJECTORY.md").write_text(traj)
        (out_dir / "rollout.json").write_text(json.dumps(rollout.as_dict(), indent=2, default=str) + "\n")
        return rollout
