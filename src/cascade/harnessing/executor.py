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

import contextlib
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import subprocess
import sys
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
    #: the composed robot profile the task runs on (None: the executor's rig)
    robot: str | None = None

    @classmethod
    def from_dict(cls, data: dict, *, domain: str | None = None) -> "TaskSpec":
        if not data.get("id") or not data.get("task"):
            raise ValueError(f"task needs id and task: {data!r}")
        return cls(str(data["id"]), str(data["task"]), str(data.get("domain") or domain or "default"),
                   int(data.get("max_steps", 20)), TaskCheck.from_dict(data.get("check")),
                   str(data["robot"]) if data.get("robot") else None)

    def as_dict(self) -> dict:
        out = {"id": self.id, "task": self.task, "domain": self.domain, "max_steps": self.max_steps}
        if self.check is not None:
            out["check"] = self.check.as_dict()
        if self.robot is not None:
            out["robot"] = self.robot
        return out


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

    @classmethod
    def from_dict(cls, data: dict) -> "Rollout":
        return cls(**{k: data.get(k) for k in cls.__dataclass_fields__ if k in data})


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
                "refused", "verified", "objects_visible", "position", "placed", "verdict", "stopped"):
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


@contextlib.contextmanager
def _file_lock(path: str | None):
    if not path:
        yield
        return
    import fcntl

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@dataclass
class SubprocessExecutor:
    """Runs each rollout of `base` in its own Python process (`python -m cascade.harnessing.worker`),
    so several rollouts can run at once: a runtime is never shared between threads."""

    base: CascadeExecutor
    timeout_s: float = 1800.0
    python: str = sys.executable
    thread_safe: bool = True

    def run(self, harness: AgentHarness, spec: TaskSpec, label: str, out_dir: Path) -> Rollout:
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        job = out_dir / "job.json"
        job.write_text(json.dumps({"executor": self.base.worker_kwargs(), "harness": harness.as_dict(),
                                   "spec": spec.as_dict(), "label": label, "out_dir": str(out_dir)}))
        src = str(Path(__file__).resolve().parents[2])
        env = dict(os.environ)
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        t0 = time.monotonic()
        try:
            with open(out_dir / "worker.log", "w") as log:
                proc = subprocess.run([self.python, "-m", "cascade.harnessing.worker", str(job)], stdout=log,
                                      stderr=subprocess.STDOUT, env=env, timeout=self.timeout_s)
            error = None if proc.returncode == 0 else f"worker exited {proc.returncode} (see worker.log)"
        except subprocess.TimeoutExpired:
            error = f"worker timed out after {self.timeout_s:.0f} s"
        result = out_dir / "rollout.json"
        if result.is_file():  # the worker's own record, a failed rollout included
            return Rollout.from_dict(json.loads(result.read_text()))
        traj = out_dir / "TRAJECTORY.md"
        if not traj.is_file():
            traj.write_text(f"# Execution {label}\n\nTask: {spec.task}\nHarness: {harness.sha()[:12]}\n\n"
                            f"The execution failed before finishing.\n")
        return Rollout(label, spec.id, harness.sha(), None, None, 0, round(time.monotonic() - t0, 1),
                       "executor error", "", error=error or "worker wrote no rollout.json", trajectory=str(traj))


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
    #: a file every concurrent rollout locks while it builds its runtime: the MuJoCo scene writer
    #: regenerates one shared XML next to the robot model (`sim.demo_scene`)
    build_lock: str | None = None
    #: a composed robot profile (`configs/robots/<robot>.yaml`, e.g. the humanoid profiles);
    #: when set, `camera` and `arm` are not used
    robot: str | None = None
    #: a private configuration directory (e.g. a copy whose fastening profile is pinned to a
    #: prepared model identity and device); None: the repository's configs/
    config_dir: str | None = None
    #: the OPERATOR clears a startup stop latch (`reset_stop`) once after the build, as the
    #: fastening validation does, through the operator path (`execute(..., operator=True)`). The
    #: agent never resets a stop: reset_stop is not on a composed robot's agent tool list, the
    #: runtime refuses it without `operator=True`, and Harness² advice may never name it
    operator_arm: bool = False

    def worker_kwargs(self) -> dict:
        """What a rollout worker process needs to rebuild this executor (no callables)."""
        if self.llm_factory is not None or self.runtime_hook is not None:
            raise ValueError("an executor with an llm_factory or runtime_hook cannot run in a worker process")
        return {"camera": self.camera, "arm": self.arm, "llm_profile": self.llm_profile,
                "decompose": self.decompose, "verify_milestones": self.verify_milestones,
                "env": dict(self.env), "build_lock": self.build_lock, "robot": self.robot,
                "config_dir": self.config_dir, "operator_arm": self.operator_arm}

    def _config(self, robot: str | None = None):
        from ..config import load_demo_config

        robot = robot or self.robot
        config_dir = Path(self.config_dir) if self.config_dir else None
        if robot:
            return load_demo_config(robot=robot, llm=self.llm_profile, config_dir=config_dir)
        return load_demo_config(camera=self.camera, arm=self.arm, llm=self.llm_profile, config_dir=config_dir)

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
            cfg = isolated_config(self._config(spec.robot), out_dir)
            with _file_lock(self.build_lock):
                runtime, arm = build_runtime(cfg, out_dir / "run")
            if self.operator_arm:
                armed = runtime.execute("reset_stop", {}, operator=True)
                if not isinstance(armed, dict) or armed.get("ok") is not True:
                    raise RuntimeError(f"operator reset_stop after the build failed: {armed!r}")
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
                measured, rows = probe.evaluate(runtime, arm, claimed=claimed)
                check = {"passed": measured, "physics": rows.pop("physics_passed")["value"], "criteria": rows}
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
                    if getattr(runtime, "robot_mode", None) == "composed":
                        runtime.close()
                    else:
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
