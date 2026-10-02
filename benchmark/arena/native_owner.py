"""Bounded benchmark owner; no simulator import or generic arm controller.

Only the native preflight constructs this owner with Arena's ZeroActionPolicy.
Its final consumer fence applies to returned actions too. Stopping prevents
additional admitted steps; it does not integrate braking or establish rest.
"""
from __future__ import annotations

import math
import threading
import time

from cascade.eval.arena import ArenaControllerDomain
from cascade.robotics.contracts import ToolDescriptor


class PreflightController:
    def __init__(self, policy):
        self.policy = policy
        self.step_lock = threading.Lock()
        self.halted = False
        self.closed = False

    def get_action(self, env, observation):
        return self.policy.get_action(env, observation)

    def stop(self):
        # A completed stop acknowledgement follows any already admitted step.
        with self.step_lock:
            self.halted = True
        return True

    def reset_stop(self):
        # A preflight episode is single-use; it must never replay after stop.
        return False

    def reset(self, env_ids=None):
        self.policy.reset(env_ids)

    def close(self):
        with self.step_lock:
            self.halted = True
            if not self.closed:
                self.policy.close()
                self.closed = True


class NativePreflightDomain(ArenaControllerDomain):
    """One <=20-step episode through ordinary RobotRuntime.execute.

    All tensor validation and native observations are supplied by the explicit
    host. The host must bind the real env/policy, with no simulator stand-ins.
    ``after_action`` is an optional test injection at the actual consumer fence.
    """
    def __init__(self, controller, *, command_resources, env, observation,
                 observe, validate_action, deadline_monotonic_s, after_action=None):
        super().__init__(controller, command_resources=command_resources)
        if not isinstance(controller, PreflightController):
            raise ValueError("native preflight requires its final-consumer controller")
        if (type(deadline_monotonic_s) not in {float, int}
                or not math.isfinite(deadline_monotonic_s)):
            raise ValueError("finite local deadline required")
        self.env, self.observation = env, observation
        self.observe, self.validate_action = observe, validate_action
        self.deadline = deadline_monotonic_s
        self.after_action = after_action
        self.adapter = None
        self.used = False
        self.completed_steps = 0
        self.samples = []
        self.actions_produced = 0
        writes = tuple(r.resource_id for r in self.resources)
        self.tool_descriptors = (ToolDescriptor(
            name="arena_policy.run_policy_steps", domain=self.domain_id,
            local_name="run_policy_steps", effect="motion", requires=writes, writes=writes,
            description="Run one bounded native zero-action foundation; no task-success claim.",
            parameters={"type": "object", "properties": {
                "steps": {"type": "integer", "minimum": 1, "maximum": 20}},
                "required": ["steps"], "additionalProperties": False}),)
        self.tool_specs = tuple(t.as_spec() for t in self.tool_descriptors)
        self.motion_skills = frozenset(t.name for t in self.tool_descriptors)

    def bind_adapter(self, adapter):
        if self.adapter is not None or adapter._owner is not self:
            raise ValueError("exact adapter may be bound only once")
        self.adapter = adapter

    def _check_consumer(self, generation):
        if (self.controller.halted or self.controller.closed or self._halted or self._closed
                or self.adapter.runtime.stopped
                or generation != self.adapter.runtime.cancellation_token):
            raise RuntimeError("native consumer cancelled before env.step")
        if time.monotonic() >= self.deadline:
            raise RuntimeError("native consumer deadline expired before env.step")

    def execute(self, name, args):
        if name != "run_policy_steps" or self.adapter is None:
            raise ValueError("unbound or unknown preflight tool")
        steps = args.get("steps")
        if type(steps) is not int or not 1 <= steps <= 20 or set(args) != {"steps"}:
            raise ValueError("preflight steps must be an integer in [1, 20]")
        with self._lock:
            if self.used or self._halted or self._closed:
                raise RuntimeError("preflight episode already used, stopped or closed")
            self.used = True
            self._active += 1  # close must not release resources during env.step
        generation = self.adapter.runtime.cancellation_token
        error = None
        try:
            self._check_consumer(generation)
            self.samples.append(self.observe())
            for number in range(steps):
                action = self.adapter.get_action(self.env, self.observation)
                self.actions_produced += 1
                self.validate_action(action)
                if self.after_action is not None:
                    self.after_action(number)
                with self.controller.step_lock:
                    self._check_consumer(generation)
                    self.observation, _, terminated, truncated, _ = self.env.step(action)
                    self.completed_steps += 1
                    self.samples.append(self.observe())
                    if bool(terminated.any() or truncated.any()):
                        raise RuntimeError("episode terminated; upstream may have auto-reset")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self._active -= 1
        return {"ok": error is None, "execution_ok": error is None,
                "error": error, "completed_steps": self.completed_steps,
                "actions_produced": self.actions_produced,
                "postcondition": {"status": "unverified", "reason":
                    "foundation only; no task, tracking, contact or braking admission"},
                "physical_stop_verified": False}
