"""One bounded agent episode per explicitly configured robot.

Run ``python -m cascade.apps.fleet --fleet microduck_mock12 --llm mock
--task 'inspect and move briefly' --run-dir /tmp/fleet-example``.
The mock LLM is a labelled script inside AgentOrchestrator, not model inference.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import nullcontext
import copy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import queue
import threading
import time

from ..agent.llm import LLMClient, LLMResponse, MockLLM, ToolCall, make_llm
from ..agent.orchestrator import AgentOrchestrator
from ..config import CONFIG_DIR, Cfg, _load_yaml, load_profile, load_robot_config
from ..lifecycle import teardown_receipt, teardown_step
from ..robotics.contracts import identifier
from ..robotics.fleet import FleetRuntime
from ..robotics.runtime import RobotRuntime
from .robot_runtime import build_robot_runtime, describe_robot


def load_fleet(name, *, task):
    """Resolve all identities/controllers before constructing any domain runtime."""
    path = Path(name).expanduser()
    if not path.is_file():
        if path.name != str(name):
            raise ValueError("fleet must be an existing YAML file or an exact configured name")
        path = CONFIG_DIR / "fleets" / f"{name}.yaml"
    data = _load_yaml(path)
    if (set(data) - {"version", "robots", "deadline_s", "max_workers", "max_steps"}
            or type(data.get("version")) is not int or data.get("version") != 1):
        raise ValueError("fleet requires version 1 and explicit robot profiles")
    entries = data.get("robots")
    if not isinstance(entries, list) or not 1 <= len(entries) <= 64:
        raise ValueError("fleet requires 1..64 robot entries")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("a nonempty assigned task is required")
    for field, default in (("max_workers", len(entries)), ("max_steps", 30)):
        data[field] = data.get(field, default)
        if type(data[field]) is not int or not 1 <= data[field] <= 64:
            raise ValueError(f"{field} must be an integer in 1..64")
    deadline = data.get("deadline_s", 60.)
    if type(deadline) not in {int, float} or not math.isfinite(deadline) or not 0 < deadline <= 3600:
        raise ValueError("deadline_s must be finite and in (0, 3600]")
    configs, tasks = [], {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - {"profile", "config_dir", "mock_id", "task"}:
            raise ValueError("robot entry requires profile and optional config_dir, mock_id, task")
        directory = Path(entry["config_dir"]).expanduser() if "config_dir" in entry else CONFIG_DIR
        if not directory.is_absolute():
            directory = path.resolve().parent / directory
        cfg = load_robot_config(entry.get("profile"), config_dir=directory)
        if "mock_id" in entry:
            # An explicit software fixture convenience, never remote relabeling.
            robot_id = identifier(entry["mock_id"], "mock_id")
            body = cfg.as_dict()
            if "embodiment" in body or any(d["kind"] != "locomotion" or
                    any(p["type"] != "mock" for p in d["resolved"]["bases"])
                    for d in body["domains"].values()):
                raise ValueError("mock_id is only valid for unmounted mock mobile profiles")
            body["robot_id"] = robot_id
            for domain in body["domains"].values():
                domain["robot_id"] = robot_id
                for index, base in enumerate(domain["resolved"]["bases"]):
                    base.update(robot_id=robot_id, name=f"mock_{len(configs)}_{index}")
                domain["resolved"]["base"] = copy.deepcopy(domain["resolved"]["bases"][0])
            cfg = Cfg(body)
        assigned = entry.get("task", task)
        if not isinstance(assigned, str) or not assigned.strip():
            raise ValueError("robot task must be nonempty text")
        configs.append(cfg)
        tasks[cfg.robot_id] = assigned
    _validate_fleet(configs)
    return {"configs": configs, "tasks": tasks, "deadline_s": float(deadline),
            "max_workers": data["max_workers"], "max_steps": data["max_steps"]}


def _validate_fleet(configs):
    passive = [(cfg.robot_id, RobotRuntime(describe_robot(cfg), cfg=cfg)) for cfg in configs]
    FleetRuntime(passive)  # duplicate IDs/controllers refused before the first build
    for cfg in configs:
        for domain in cfg.domains.as_dict().values():
            if domain["kind"] == "locomotion":
                from .mobile_runtime import make_base
                for base in domain["resolved"]["bases"]:
                    make_base(base)  # passive constructor validates native pins/ports
    manipulation = sum(any(d["kind"] == "manipulation" for d in cfg.domains.as_dict().values()) for cfg in configs)
    if manipulation > 1 and any(os.environ.get(key) for key in
            ("CASCADE_BELIEFS_PATH", "CASCADE_GRASP_MEMORY_PATH", "CASCADE_ENVELOPE_PATH",
             "CASCADE_EPISODIC_PATH")):
        raise ValueError("global memory path overrides would share stores across fleet robots")


def build_fleet(plan, run_dir):
    _validate_fleet(plan["configs"])  # plans are ordinary data; recheck before opening IO
    built = []
    try:
        for index, cfg in enumerate(plan["configs"]):
            # Robot IDs are not filesystem paths. Keep exact IDs in the catalog.
            runtime, _ = build_robot_runtime(cfg, Path(run_dir) / "robots" / f"{index:03}")
            built.append((cfg.robot_id, runtime))
        return FleetRuntime(built)
    except BaseException as exc:
        # Every successfully constructed owner is retained for rollback.
        stages = []
        for name, runtime in built:
            try:
                runtime.queue_shutdown()
                stages.append({"stage": name + ".cancel", "ok": True, "complete": True})
            except Exception as error:
                stages.append({"stage": name + ".cancel", "ok": False, "complete": False, "error": str(error)})
        stages.extend(teardown_step(name, runtime.close) for name, runtime in reversed(built))
        exc.fleet_teardown = teardown_receipt(stages)
        raise


class AgentCancelled(RuntimeError):
    pass


class _AgentRuntime:
    """Use the member's existing episode; never clear debts or reset a stop."""
    def __init__(self, coordinator, robot_id):
        self.coordinator, self.robot_id = coordinator, robot_id
        self.member = coordinator.fleet.robots[robot_id]

    def __getattr__(self, name):
        return getattr(self.member, name)

    def __setattr__(self, name, value):
        if name in {"current_tier", "current_task", "last_path"}:
            setattr(self.member, name, value)
        else:
            object.__setattr__(self, name, value)

    def begin_task(self):
        self.coordinator.check(self.robot_id)

    def execute(self, name, args=None):
        self.coordinator.check(self.robot_id)
        result = self.coordinator.fleet.execute(self.robot_id, name, args,
            expected_generation=self.coordinator.tokens[self.robot_id],
            deadline_monotonic_s=self.coordinator.deadline)
        return result.get("result", result)


class _EpisodeLLM(LLMClient):
    def __init__(self, client, coordinator, robot_id):
        self.client, self.coordinator, self.robot_id = client, coordinator, robot_id
        self.supports_vision = client.supports_vision

    def chat(self, *args, **kwargs):
        self.coordinator.check(self.robot_id)
        response = self.client.chat(*args, **kwargs)
        self.coordinator.check(self.robot_id)  # delayed inference cannot renew authority
        return response


def _mock_client(runtime):
    calls = [ToolCall("list_resources", {})]
    if all(r.synthetic for r in runtime.resources) and "locomotion.walk_velocity" in runtime.tool_descriptors:
        calls.append(ToolCall("locomotion.walk_velocity", {"vx": .03, "vy": 0., "wz": 0., "duration_s": .2}))
    calls.append(ToolCall("task_done", {"success": False,
        "summary": "Scripted mock LLM diagnostic; no physical task success or model inference claimed."}))
    return MockLLM([LLMResponse(tool_calls=[call]) for call in calls])


class FleetAgents:
    """A finite job queue and bounded workers, each running an actual orchestrator."""
    def __init__(self, fleet, tasks, llm_cfg, *, deadline_s=60., max_workers=12, max_steps=30):
        if set(tasks) != set(fleet.robots):
            raise ValueError("one explicit task per exact robot identity required")
        if (type(deadline_s) not in {float, int} or not math.isfinite(deadline_s) or not 0 < deadline_s <= 3600
                or type(max_workers) is not int or not 1 <= max_workers <= 64
                or type(max_steps) is not int or not 1 <= max_steps <= 64):
            raise ValueError("invalid fleet worker, deadline or step budget")
        if any(not isinstance(task, str) or not task.strip() for task in tasks.values()):
            raise ValueError("robot tasks must be nonempty text")
        self.fleet, self.tasks, self.llm_cfg = fleet, dict(tasks), llm_cfg
        self.deadline_s, self.max_workers, self.max_steps = deadline_s, max_workers, max_steps
        self.tokens = {name: rt.cancellation_token for name, rt in fleet.robots.items()}
        self.cancelled = {name: threading.Event() for name in fleet.robots}
        self.reports, self.workers, self.clients, self.client_closures = {}, [], {}, {}
        self.condition = threading.Condition()
        self.jobs = queue.Queue(maxsize=len(tasks))
        for name in tasks:
            self.jobs.put_nowait(name)
        self.deadline = None
        self.start_gate = threading.Event()
        self.stop_receipts = deque(maxlen=128)

    def check(self, robot_id):
        if (self.cancelled[robot_id].is_set() or time.monotonic() >= self.deadline or
                self.fleet.robots[robot_id].cancellation_token != self.tokens[robot_id]):
            raise AgentCancelled("episode cancelled, superseded or deadline expired")

    def stop(self, robot_id=None):
        selected = self.cancelled if robot_id is None else {robot_id: self.cancelled[robot_id]}
        for event in selected.values():
            event.set()
        self.start_gate.set()  # release registered but not yet dispatched workers
        receipt = self.fleet.stop(robot_id)
        with self.condition:
            self.stop_receipts.append({"robot_id": robot_id, "receipt": receipt})
            self.condition.notify_all()
        return receipt

    def _worker(self):
        self.start_gate.wait()
        while True:
            try:
                robot_id = self.jobs.get_nowait()
            except queue.Empty:
                return
            runtime = self.fleet.robots[robot_id]
            client = None
            result = {"state": "failed", "error": "agent did not return"}
            try:
                self.check(robot_id)
                client = _mock_client(runtime) if self.llm_cfg.type == "mock" else make_llm(self.llm_cfg)
                with self.condition:
                    self.clients[robot_id] = client
                proxy = _AgentRuntime(self, robot_id)
                proxy.current_task = self.tasks[robot_id]
                agent = AgentOrchestrator(_EpisodeLLM(client, self, robot_id), proxy,
                    max_steps=self.max_steps, decompose=self.llm_cfg.type != "mock", attach_images=False)
                report = agent.run_task(f"Robot {robot_id}: {self.tasks[robot_id]}")
                result = {"state": "completed", "report": asdict(report)}
            except BaseException as exc:
                self.stop(robot_id)
                result = {"state": "cancelled" if isinstance(exc, AgentCancelled) else "failed",
                          "error": f"{type(exc).__name__}: {exc}"}
            finally:
                runtime.current_task = None
                if client is not None:
                    try:
                        receipt = teardown_step(robot_id + ".llm", client.close)
                    except BaseException as exc:
                        receipt = {"ok": False, "complete": False, "error": f"{type(exc).__name__}: {exc}"}
                    if not receipt["ok"]:
                        result.update(state="failed", client_close_error=receipt)
                    with self.condition:
                        self.client_closures[robot_id] = receipt
            with self.condition:
                self.reports[robot_id] = result
                self.condition.notify_all()

    def run(self, *, signals=None):
        if self.deadline is not None:
            raise ValueError("a FleetAgents episode cannot be replayed")
        self.deadline = time.monotonic() + self.deadline_s
        with signals.defer() if signals is not None else nullcontext():
            for index in range(min(self.max_workers, len(self.tasks))):
                worker = threading.Thread(target=self._worker, name=f"fleet-agent-{index}", daemon=True)
                self.workers.append(worker)
                worker.start()
        if signals is not None:
            signals.checkpoint()
        self.start_gate.set()
        with self.condition:
            complete = self.condition.wait_for(lambda: len(self.reports) == len(self.tasks),
                timeout=max(0, self.deadline - time.monotonic()))
        if not complete:
            self.stop()
        return self.snapshot()

    def snapshot(self):
        with self.condition:
            return {"agent": "AgentOrchestrator", "llm": "scripted_mock" if self.llm_cfg.type == "mock" else self.llm_cfg.type,
                    "assignments": dict(self.tasks), "robots": copy.deepcopy(self.reports),
                    "stop_receipts": copy.deepcopy(list(self.stop_receipts)),
                    "pending_agents": [name for name in self.tasks if name not in self.reports],
                    "unverified": self.fleet.unverified_actions(),
                    "physical_fleet_admission": False}

    def close(self):
        if any(worker.is_alive() for worker in self.workers):
            self.stop()
        deadline = time.monotonic() + 5.
        for worker in self.workers:
            if worker.ident is not None:
                worker.join(max(0, deadline - time.monotonic()))
        result = self.fleet.close()
        pending = [worker.name for worker in self.workers if worker.is_alive()]
        with self.condition:
            clients = {name: copy.deepcopy(self.client_closures.get(name,
                {"ok": False, "complete": False, "pending": True})) for name in self.clients}
        return {"ok": result["ok"] and not pending and all(r["ok"] for r in clients.values()),
                "complete": result["complete"] and not pending and all(r["complete"] for r in clients.values()),
                "fleet": result, "llm_clients": clients, "pending_workers": pending, "physical_rest_verified": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleet", required=True)
    parser.add_argument("--llm", default="mock")
    parser.add_argument("--task", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = load_fleet(args.fleet, task=args.task)
    llm_path = Path(args.llm).expanduser()
    llm_cfg = Cfg(_load_yaml(llm_path)) if llm_path.is_file() else load_profile("llm", args.llm)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    from .signal_stop import SignalRequest, StopSignals
    fleet = agents = None
    error = None
    build_teardown = None
    with StopSignals() as signals:
        try:
            with signals.defer():
                fleet = build_fleet(plan, args.run_dir)
                agents = FleetAgents(fleet, plan["tasks"], llm_cfg, deadline_s=plan["deadline_s"],
                                     max_workers=plan["max_workers"], max_steps=plan["max_steps"])
            signals.checkpoint()
            (args.run_dir / "catalog.json").write_text(json.dumps(fleet.catalog(), indent=2))
            print(f"Fleet ready: {len(fleet.robots)} robots; LLM={llm_cfg.type}; no physical fleet admission", flush=True)
            agents.run(signals=signals)
        except SignalRequest:
            if agents is not None:
                agents.stop()
            elif fleet is not None:
                fleet.stop()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            build_teardown = getattr(exc, "fleet_teardown", None)
            if agents is not None:
                agents.stop()
        finally:
            with signals.defer():
                closure = (agents.close() if agents is not None else fleet.close() if fleet is not None
                           else build_teardown or {"ok": False, "complete": False})
                report = agents.snapshot() if agents is not None else {"physical_fleet_admission": False}
                report.update(shutdown=closure, error=error, signal=signals.signum)
                report["software_complete"] = (not error and signals.signum is None and closure["ok"]
                    and not report.get("pending_agents")
                    and all(r["state"] == "completed" for r in report.get("robots", {}).values()))
                report["task_success"] = (report["software_complete"] and llm_cfg.type != "mock"
                    and all(r["report"]["success"] for r in report.get("robots", {}).values()))
                (args.run_dir / "report.json").write_text(json.dumps(report, indent=2))
        if signals.signum is not None:
            return 128 + signals.signum
    return 0 if report["software_complete"] and (llm_cfg.type == "mock" or report["task_success"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
