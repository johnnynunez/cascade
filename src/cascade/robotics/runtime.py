"""Capability composition over existing domain safety and verification paths."""
from __future__ import annotations

import copy
import re
import threading
import time
import uuid
from collections import deque

from .contracts import ToolDescriptor
from .resources import ResourceCatalog
from .embodiment import embodiment_metadata


SYSTEM_PROMPT = """You operate an explicitly configured robot composition.
Use only listed namespaced tools and exact resource selectors. Listing a capability
is not physical admission. Respect each domain's units, frames and limits.
An execution ACK is not success: independent confirmed evidence is required.
Synthetic observations and motions are software diagnostics, not physical proof.
Stop is global and priority. Never reset a safety stop automatically, replay a
cancelled command, or infer a grasp, camera, navigation map or whole-body controller.
Use task_done(success=false) for refuted or unverified requested outcomes.
"""


def _global_tools():
    def tool(name, description, properties=None, required=(), effect="read"):
        return ToolDescriptor(name=name, description=description,
                              parameters={"type": "object", "properties": properties or {},
                                          "required": list(required), "additionalProperties": False},
                              domain="robot", local_name=name, effect=effect)
    return (
        tool("list_resources", "Read declared resources and capabilities; no device connection or actuation."),
        tool("emergency_stop", "Priority cancellation and latch across all domains. ACK does not prove physical rest.", effect="stop"),
        tool("reset_stop", "Explicitly clear stop permission; no motion replay or world reset.", effect="control"),
        tool("task_done", "Finish with independent action verdicts; synthetic execution never proves a physical task.",
             {"success": {"type": "boolean"}, "summary": {"type": "string"}}, ("success", "summary")),
    )


class RobotRuntime:
    """One ordinary operation, independent priority stop, and passive metadata.

    Domain controllers retain their own generation fences and transport leases.
    This coordinator fences pending dispatch; it does not substitute local locks
    for backend admission or claim whole-body collision/balance coordination.
    """
    robot_mode = "composed"
    system_prompt = SYSTEM_PROMPT

    def __init__(self, domains, *, cfg=None, memory=None, trace=None):
        self.domains = dict(domains)
        self.cfg, self.memory, self.trace = cfg, memory, trace
        self.resources = ResourceCatalog([r for d in self.domains.values() for r in d.resources])
        self._embodiment_metadata = embodiment_metadata(
            cfg.as_dict().get("embodiment") if cfg is not None else None, self.resources)
        self.tool_descriptors = {t.name: t for t in _global_tools()}
        for name, domain in self.domains.items():
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,23}", name) or domain.domain_id != name:
                raise ValueError("invalid or mismatched domain id")
            for descriptor in domain.tool_descriptors:
                if descriptor.domain != name or descriptor.name != f"{name}.{descriptor.local_name}":
                    raise ValueError("domain tool must have its exact namespace")
                if descriptor.name in self.tool_descriptors or len(descriptor.name) > 64:
                    raise ValueError("duplicate or oversized tool name")
                for resource in (*descriptor.requires, *descriptor.writes):
                    self.resources.require(resource)
                owned = {resource.resource_id for resource in domain.resources}
                for resource in descriptor.writes:
                    if resource not in owned or self.resources.require(resource).writer_id is None:
                        raise ValueError("tool writes must be owned command resources of its domain")
                self.tool_descriptors[descriptor.name] = descriptor
        self.tool_specs = [t.as_spec() for t in self.tool_descriptors.values()]
        self.motion_skills = frozenset(n for n, t in self.tool_descriptors.items() if t.effect == "motion")
        self.stop_skills = frozenset(n for n, t in self.tool_descriptors.items() if t.effect == "stop")
        self._gate = threading.Lock()
        self._record_lock = threading.Lock()
        self._stop_condition = threading.Condition(self._gate)
        self._stop_slots = {name: {"pending": None, "active": False, "thread": None,
                                   "generation": -1, "result": None, "emergency": False} for name in self.domains}
        self._workers_closed = False
        self._active = False
        self._resetting = False
        self._closed = False
        self._shutdown_complete = False
        self._closed_domains = {}
        self._close_lock = threading.Lock()
        self._drained = threading.Event()
        self._drained.set()
        self._latched = False
        self._generation = 0
        self._task_id = uuid.uuid4().hex
        self._history = deque(maxlen=256)
        self._unverified = set()
        self.current_task = self.current_tier = self.last_path = None
        self.stream_server = self.viewer = self.last_frame = None
        self.effects = None

    @property
    def cancellation_token(self):
        with self._gate:
            return self._generation

    @property
    def stopped(self):
        with self._gate:
            return self._latched or self._closed

    @staticmethod
    def _arguments(descriptor, args):
        if not isinstance(args, dict):
            raise ValueError("tool arguments must be an object")
        schema = descriptor.parameters
        if set(schema.get("required", ())) - args.keys():
            raise ValueError("missing required tool arguments")
        if set(args) - schema.get("properties", {}).keys():
            raise ValueError("unknown tool arguments")
        descriptor.validate_arguments(args)

    def execute(self, name, args=None):
        started = time.monotonic()
        result = self._execute(name, args)
        if self.trace is not None:
            with self._record_lock:
                self.trace.record(name, args or {}, result, (time.monotonic() - started) * 1000,
                                  tier=self.current_tier, context={"robot_mode": "composed", "task": self.current_task})
        return result

    def _execute(self, name, args=None):
        args = {} if args is None else args
        descriptor = None
        try:
            if not isinstance(name, str):
                raise ValueError("tool name must be a string")
            descriptor = self.tool_descriptors.get(name)
            if descriptor is None:
                raise ValueError(f"unsupported robot tool: {name!r}")
            self._arguments(descriptor, args)
            if descriptor.effect == "stop":
                return self.stop()
            if name == "reset_stop":
                return self.reset_stop()
            if name == "list_resources":
                return {"ok": True, **self.resources.as_dict(), **copy.deepcopy(self._embodiment_metadata),
                        "metadata_source": "configured_profile"}
            with self._gate:
                if self._closed:
                    raise ValueError("robot runtime closed")
                if self._active or self._resetting:
                    raise ValueError("another robot operation is active")
                if descriptor.effect in {"motion", "control"} and self._latched:
                    raise ValueError("robot stopped; explicit reset_stop required")
                self._active = True
                self._drained.clear()
                generation, task_id = self._generation, self._task_id
            try:
                if name == "task_done":
                    if type(args["success"]) is not bool or not isinstance(args["summary"], str):
                        raise ValueError("task_done requires boolean success and string summary")
                    missing = self.unverified_actions()
                    return {"ok": True, "task_complete": True,
                            "success": args["success"] and not missing,
                            "summary": args["summary"], "unverified": missing}
                domain = self.domains[descriptor.domain]
                with self._gate:
                    if generation != self._generation or self._closed:
                        raise ValueError("dispatch invalidated by stop or close")
                # Domain implementations perform their own last-moment backend
                # generation check. Stop never waits for this call to finish.
                result = domain.execute(descriptor.local_name, copy.deepcopy(args))
                if not isinstance(result, dict):
                    raise ValueError("domain result must be an object")
                result = copy.deepcopy(result)
                descriptor.validate_result(result)
                with self._gate:
                    raced = generation != self._generation
                    if descriptor.effect == "motion":
                        synthetic = any(self.resources.require(r).synthetic for r in descriptor.writes)
                        self._history.append({"task_id": task_id, "tool": name,
                                              "result": copy.deepcopy(result),
                                              "synthetic": synthetic})
                        if (raced or synthetic or result.get("ok") is not True or
                                result.get("execution_ok") is False or result.get("delivery_uncertain") is True or
                                result.get("postcondition", {}).get("status") != "confirmed"):
                            self._unverified.add(name)
                if raced and descriptor.effect in {"motion", "control"}:
                    return {"ok": False, "execution_ok": False, "error": "operation superseded by stop",
                            "domain_result": result}
                return result
            finally:
                with self._gate:
                    self._active = False
                    self._drained.set()
        except Exception as exc:
            if descriptor is not None and descriptor.effect == "motion":
                with self._gate:
                    self._unverified.add(name)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _stop_worker(self, name):
        slot = self._stop_slots[name]
        while True:
            with self._stop_condition:
                self._stop_condition.wait_for(lambda: self._workers_closed or slot["pending"] is not None)
                if self._workers_closed:
                    return
                generation, shutdown = slot["pending"]
                slot["pending"] = None
                slot["active"] = True
            try:
                domain = self.domains[name]
                if shutdown and callable(getattr(domain, "request_shutdown", None)):
                    result = domain.request_shutdown()
                else:
                    result = domain.stop()
                if not isinstance(result, dict) or type(result.get("ok")) is not bool:
                    raise ValueError("invalid domain stop receipt")
            except Exception as exc:
                result = {"ok": False, "error": str(exc)}
            with self._stop_condition:
                slot.update(active=False, generation=generation, result=copy.deepcopy(result))
                self._stop_condition.notify_all()

    def stop(self, **_kwargs):
        return self._request_stop(shutdown=False)

    def request_shutdown(self):
        """Fence dispatch while preserving each domain's existing teardown policy."""
        with self._gate:
            self._closed = True
        return self._request_stop(shutdown=True)

    def _request_stop(self, *, shutdown):
        workers = []
        with self._stop_condition:
            self._generation += 1
            self._latched = True
            generation = self._generation
            if self._workers_closed:
                return {"ok": False, "latched": True, "error": "runtime already closed"}
            for name, slot in self._stop_slots.items():
                slot["emergency"] = slot["emergency"] or not shutdown
                # A graceful shutdown may not weaken an explicit e-stop that
                # was pending before this worker could reach the backend.
                slot["pending"] = (generation, shutdown and not slot["emergency"])
                if slot["thread"] is None:
                    slot["thread"] = threading.Thread(target=self._stop_worker, args=(name,),
                                                       name=f"robot-stop-{name}", daemon=True)
                    workers.append(slot["thread"])
            self._stop_condition.notify_all()
        # Every domain receives its own bounded mailbox before waiting on any
        # IO. One stuck transport cannot prevent another domain being stopped.
        for worker in workers:
            worker.start()
        deadline = time.monotonic() + .5
        with self._stop_condition:
            self._stop_condition.wait_for(
                lambda: all(s["generation"] >= generation for s in self._stop_slots.values()),
                timeout=max(0, deadline - time.monotonic()))
            results = {name: copy.deepcopy(slot["result"]) if slot["generation"] >= generation else
                       {"ok": False, "pending": True, "error": "domain stop receipt pending"}
                       for name, slot in self._stop_slots.items()}
        return {"ok": all(r.get("ok") is True for r in results.values()), "latched": True,
                "generation": generation, "domains": results, "physical_stop_verified": False}

    def reset_stop(self):
        with self._gate:
            if (self._active or self._resetting or self._closed or
                    any(s["active"] or s["pending"] is not None for s in self._stop_slots.values())):
                return {"ok": False, "error": "active or closed runtime cannot reset stop"}
            self._resetting = True
            generation = self._generation
        results = {}
        try:
            for name, domain in self.domains.items():
                try:
                    results[name] = domain.reset_stop()
                except Exception as exc:
                    results[name] = {"ok": False, "error": str(exc)}
            with self._gate:
                ok = (generation == self._generation and not self._closed
                      and all(r.get("ok") is True for r in results.values()))
                if ok:
                    self._generation += 1
                    self._latched = False
                    for slot in self._stop_slots.values():
                        slot["emergency"] = False
            if not ok:
                self.stop()  # re-latch domains reset before a failure/racing stop
            return {"ok": ok, "domains": results, "latched": not ok}
        finally:
            with self._gate:
                self._resetting = False

    def begin_task(self):
        with self._gate:
            if self._active or self._resetting or self._closed:
                raise ValueError("cannot begin task during another operation or after close")
            self._resetting = True
            self._task_id = uuid.uuid4().hex
            self._unverified.clear()
        try:
            for domain in self.domains.values():
                domain.begin_task()
        finally:
            with self._gate:
                self._resetting = False

    def unverified_actions(self):
        with self._gate:
            return sorted(self._unverified | ({"robot stopped"} if self._latched else set()))

    def frame_jpeg(self):
        return None  # observations retain their explicit sensor/domain identity

    def close(self):
        with self._close_lock:
            with self._gate:
                if self._shutdown_complete:
                    return {"ok": True, "already_closed": True}
                self._closed = True
                self._generation += 1
                active = self._active
            if active:
                self.request_shutdown()
                if not self._drained.wait(5.0):
                    return {"ok": False, "error": "cancelled operation still owns IO; shutdown pending"}
            with self._stop_condition:
                idle = self._stop_condition.wait_for(
                    lambda: not any(s["active"] or s["pending"] is not None for s in self._stop_slots.values()),
                    timeout=.5)
                if not idle:
                    return {"ok": False, "error": "stop worker still owns IO; shutdown pending"}
            # Idle arm shutdown retains its existing park-before-disconnect
            # policy. Do not latch it solely because an observer is closing.
            results = {}
            for name, domain in reversed(tuple(self.domains.items())):
                if self._closed_domains.get(name, {}).get("ok") is True:
                    results[name] = copy.deepcopy(self._closed_domains[name])
                    continue
                try:
                    results[name] = domain.close()
                except Exception as exc:
                    results[name] = {"ok": False, "error": str(exc)}
            self._closed_domains.update(copy.deepcopy(results))
            with self._gate:
                self._shutdown_complete = all(r.get("ok") is True for r in results.values())
                self._workers_closed = True
                self._stop_condition.notify_all()
            for slot in self._stop_slots.values():
                if slot["thread"] is not None:
                    slot["thread"].join()
            return {"ok": all(r.get("ok") is True for r in results.values()), "domains": results}
