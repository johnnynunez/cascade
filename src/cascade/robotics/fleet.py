"""Explicit routing across independent robots; no shared-world physics admission."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import threading
import time
from types import MappingProxyType

from .contracts import identifier
from .resources import ResourceCatalog
from .runtime import RobotRuntime


class FleetRuntime:
    """Compose already admitted RobotRuntime instances without merging their state.

    Callers may execute concurrently on different robots. Each robot retains its
    ordinary-operation gate, generation, traces, memory and verification debt.
    No shared-frame collision avoidance or native fleet admission is inferred.
    Members and their domains must not be reconfigured after composition.
    """

    def __init__(self, robots):
        pairs = tuple(robots.items() if hasattr(robots, "items") else robots)
        if not pairs:
            raise ValueError("fleet requires at least one robot")
        members, resources, episodes, owners = {}, [], set(), {}

        def exclusive(key, robot_id, label):
            previous = owners.setdefault(key, robot_id)
            if previous != robot_id:
                raise ValueError(f"shared {label} between robots {previous!r} and {robot_id!r}")

        for member_index, (robot_id, runtime) in enumerate(pairs):
            identifier(robot_id, "fleet robot ID")
            if robot_id in members:
                raise ValueError("duplicate fleet robot ID")
            if not isinstance(runtime, RobotRuntime):
                raise TypeError("fleet members must be independent RobotRuntime instances")
            exclusive(("object", id(runtime)), robot_id, "runtime")
            if runtime.episode_id in episodes:
                raise ValueError("duplicate robot episode")
            episodes.add(runtime.episode_id)
            if runtime.cfg is not None and runtime.cfg.robot_id != robot_id:
                raise ValueError("fleet ID must match configured robot identity")
            for owner in (runtime, *runtime.domains.values(),
                          *(getattr(d, "runtime", None) for d in runtime.domains.values())):
                if owner is None:
                    continue
                exclusive(("object", id(owner)), robot_id, "domain runtime")
                paths = [getattr(owner, "beliefs_path", None)]
                for key in ("memory", "beliefs", "grasp_memory", "envelope", "trace"):
                    value = getattr(owner, key, None)
                    if value is not None:
                        exclusive(("object", id(value)), robot_id, key)
                        if key == "trace" and hasattr(value, "run_dir"):
                            exclusive(("trace", Path(value.run_dir).resolve()), robot_id, "trace directory")
                        paths.extend(getattr(value, field, None) for field in ("path", "_path", "_trace_path"))
                for path in paths:
                    if path is not None:
                        exclusive(("file", Path(path).expanduser().resolve()), robot_id, "store file")
            writers = {}
            for resource_index, resource in enumerate(runtime.resources):
                if resource.robot_id != robot_id:
                    raise ValueError("resource robot identity does not match fleet member")
                # Preserve controller identity: renaming robots or dividing
                # joint sets never licenses two writers to one endpoint.
                writer = None
                if resource.writer_id is not None:
                    writer = f"fleet/{member_index}/writer/{writers.setdefault(resource.writer_id, len(writers))}"
                resources.append(replace(resource, resource_id=f"fleet/{member_index}/resource/{resource_index}",
                                         writer_id=writer))
            members[robot_id] = runtime
        # Compact aliases are internal conflict-checking keys, not routing IDs.
        # Concatenating arbitrary valid IDs can collide or exceed field bounds;
        # catalog() retains every original robot/resource/tool identity.
        self.resources = ResourceCatalog(resources)
        self.robots = MappingProxyType(members)
        self._gate = threading.Lock()
        self._stop_serial = {robot_id: 0 for robot_id in members}
        self._blocked = set()
        self._closed = False

    def _member(self, robot_id):
        if not isinstance(robot_id, str) or robot_id not in self.robots:
            raise ValueError(f"unknown fleet robot: {robot_id!r}")
        return self.robots[robot_id]

    def catalog(self):
        """Keep local tool/resource names under an explicit robot selector."""
        return {robot_id: {"episode_id": runtime.episode_id,
                           "resources": runtime.resources.describe(),
                           "tools": copy.deepcopy(runtime.tool_specs)}
                for robot_id, runtime in self.robots.items()}

    def execute(self, robot_id, name, args=None, *, expected_generation=None, deadline_monotonic_s=None):
        runtime = self._member(robot_id)
        descriptor = runtime.tool_descriptors.get(name) if isinstance(name, str) else None
        if descriptor is not None and descriptor.effect == "stop":
            runtime._arguments(descriptor, {} if args is None else args)
            return self.stop(robot_id)
        if name == "reset_stop":
            runtime._arguments(descriptor, {} if args is None else args)
            if expected_generation is not None or deadline_monotonic_s is not None:
                raise ValueError("reset_stop requires an explicit operator request without an episode token")
            return self.reset_stop(robot_id)
        with self._gate:
            if self._closed:
                raise ValueError("fleet closed")
            if robot_id in self._blocked and descriptor is not None and (
                    descriptor.effect in {"motion", "control"} or name == "task_done"):
                return {"robot_id": robot_id, "ok": False, "error": "robot stopped; explicit reset_stop required"}
            generation = runtime.cancellation_token if expected_generation is None else expected_generation
        # Preserve the member's last-moment generation/deadline check. A stop
        # between selecting this token and entering execute invalidates dispatch.
        result = runtime.execute(name, args, expected_generation=generation,
                                 deadline_monotonic_s=deadline_monotonic_s)
        return {"robot_id": robot_id, "ok": result.get("ok") is True, "result": result}

    def stop(self, robot_id=None):
        selected = self.robots if robot_id is None else {robot_id: self._member(robot_id)}
        requests, results = {}, {}
        deadline = time.monotonic() + .5
        with self._gate:
            for name in selected:
                self._blocked.add(name)
                self._stop_serial[name] += 1
            for name, runtime in selected.items():
                try:
                    requests[name] = runtime.request_stop()
                except Exception as exc:
                    results[name] = {"ok": False, "error": str(exc)}
        # All robots are fenced before waiting for any RPC. Existing bounded
        # per-domain workers isolate a stuck endpoint; no thread per stop call.
        for name, generation in requests.items():
            results[name] = selected[name].wait_for_stop(generation, deadline_monotonic_s=deadline)
        return {"ok": all(r.get("ok") is True for r in results.values()), "robots": results,
                "physical_stop_verified": False}

    def reset_stop(self, robot_id, *, expected_generation=None):
        runtime = self._member(robot_id)
        with self._gate:
            if self._closed:
                raise ValueError("fleet closed")
            serial = self._stop_serial[robot_id]
            generation = runtime.cancellation_token if expected_generation is None else expected_generation
        result = runtime.reset_stop(expected_generation=generation)
        with self._gate:
            if serial == self._stop_serial[robot_id] and not self._closed and result.get("ok") is True:
                self._blocked.discard(robot_id)
            elif result.get("ok") is True:
                result = {"ok": False, "error": "reset superseded by fleet stop or close"}
        return {"robot_id": robot_id, "ok": result.get("ok") is True, "result": result}

    def unverified_actions(self):
        with self._gate:
            return {robot_id: sorted(set(runtime.unverified_actions()) |
                                    ({"fleet stop unresolved"} if robot_id in self._blocked else set()))
                    for robot_id, runtime in self.robots.items()}

    def close(self):
        """Close members using their existing teardown and parking policies.

        Teardown retains each robot's timeout; only stop fanout has a shared
        half-second receipt deadline. An incomplete close must be retried.
        """
        results, queue_errors = {}, {}
        with self._gate:
            self._closed = True
            for robot_id, runtime in self.robots.items():
                try:
                    runtime.queue_shutdown()
                except Exception as exc:
                    queue_errors[robot_id] = str(exc)
        for robot_id, runtime in self.robots.items():
            try:
                results[robot_id] = runtime.close()
            except Exception as exc:
                results[robot_id] = {"ok": False, "complete": False, "error": str(exc)}
            if robot_id in queue_errors:
                results[robot_id] = {**results[robot_id], "ok": False, "shutdown_error": queue_errors[robot_id]}
        return {"ok": all(r.get("ok") is True for r in results.values()),
                "complete": all(r.get("complete") is True for r in results.values()), "robots": results}
