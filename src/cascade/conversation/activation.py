"""One explicitly activated fixed-hand episode; passive discovery is not admission."""
from __future__ import annotations

import copy
import json
import math
import threading
import time

from ..lifecycle import retain_teardown_attempt


class BoundedHandEpisode:
    """Deferred domain behind the ordinary RobotRuntime generation/stop gates.

    The constructor may outlive a cancelled request. It remains owned, and its
    late result is closed before this object can report complete cleanup.
    There is no reset of the attempted flag or of the native owner's deadline.
    """
    def __init__(self, descriptor, builder, *, clock=time.monotonic):
        self.descriptor, self.builder, self.clock = descriptor, builder, clock
        self._catalog = self._identity()
        self._condition = threading.Condition(threading.RLock())
        self._close_lock = threading.Lock()
        self.runtime = None
        self._domain = self._controller = None
        self._attempted = self._building = self._closed = False
        self._fault = self._closure = None
        self._domain_retired = False
        self._generation = self._deadline = self._owner_start = self._owner_end = None
        self._active = False

    def _identity(self):
        return json.dumps({"profile": self.descriptor.profile,
                           "resources": [r.as_dict() for r in self.descriptor.resources],
                           "tools": [t.as_dict() for t in self.descriptor.tool_descriptors]},
                          sort_keys=True, allow_nan=False)

    def _check_start(self):
        if (self._closed or self._fault or self.clock() >= self._deadline
                or self.runtime.cancellation_token != self._generation
                or self._identity() != self._catalog):
            raise ValueError("robot activation was cancelled, expired or changed")

    def _start_owner(self, controller):
        with self._condition:
            self._check_start()
            if self._controller is not None or controller.owner_wall_s != 30.:
                raise ValueError("activation requires one original 30-second hand owner")
            self._controller = controller
            controller.start()
            self._owner_start = controller._started
            self._owner_end = self._owner_start + controller.owner_wall_s

    def activate(self, expected_generation, *, deadline):
        with self._condition:
            now = self.clock()
            if (type(expected_generation) is not int or self._attempted or self._closed
                    or self.runtime.cancellation_token != expected_generation
                    or type(deadline) not in (int, float) or not math.isfinite(deadline)
                    or not now < deadline <= now + 10.):
                raise ValueError("robot activation is unavailable or generation changed")
            self._attempted = self._building = True
            self._generation, self._deadline = expected_generation, deadline
        candidate = None
        try:
            with self._condition:
                self._check_start()
            candidate = self.builder(self._start_owner)
            with self._condition:
                self._check_start()
                if candidate.controller is not self._controller or self._owner_end is None:
                    raise ValueError("activation builder returned another owner")
                if self.clock() >= self._owner_end:
                    raise ValueError("hand owner expired during activation")
                self._domain = candidate
            # Start robot session is explicit operator authority for this real
            # reset after stopped readiness. It does not replay a robot command.
            reset = self.runtime.reset_stop(expected_generation=expected_generation,
                deadline_monotonic_s=min(deadline, self._owner_end))
            with self._condition:
                if (reset.get("ok") is not True or self._fault or self._closed
                        or self.runtime.cancellation_token != expected_generation + 1
                        or self.clock() >= min(deadline, self._owner_end)):
                    raise ValueError("robot activation reset was refused or superseded")
                self._active = True
                return {"ok": True, "reset": reset, **self.status()}
        except BaseException as exc:
            with self._condition:
                self._fault = self._fault or type(exc).__name__
                self._active = False
            if candidate is not None:
                self._retain_close(candidate)
            if self._controller is not None and (candidate is None
                    or getattr(candidate, "controller", None) is not self._controller):
                self._retain_close(self._controller)
            raise
        finally:
            with self._condition:
                self._building = False
                self._condition.notify_all()

    def revoke(self):
        with self._condition:
            self._attempted = True
            self._fault = self._fault or "activation_failed"
            self._active = False
        return self.stop()

    def status(self):
        with self._condition:
            return {"lifecycle": "bounded_hand", "attempted": self._attempted,
                    "building": self._building, "active": self._active and not self._fault and not self._closed,
                    "fault": self._fault, "closed": self._closed,
                    "owner_started_monotonic_s": self._owner_start,
                    "owner_deadline_monotonic_s": self._owner_end,
                    "remaining_s": (0. if self._closed else max(0., self._owner_end-self.clock()))
                                   if self._owner_end is not None else None,
                    "owner_lifetime_s": 30., "physical_admission": False}

    def check_live(self):
        with self._condition:
            if not self._active or self._fault or self._closed or self.clock() >= self._owner_end:
                raise ValueError("bounded robot episode is unavailable")
            controller = self._controller
        # Existing read admission retains sample age/identity and sticky faults.
        if not controller.read():
            raise ValueError("bounded robot episode has no live observation")

    def execute(self, name, args):
        self.check_live()
        return self._domain.execute(name, args)

    def stop(self):
        with self._condition:
            if self._building:
                self._fault = self._fault or "activation_cancelled"
            controller = self._controller
        if controller is None:
            return {"ok": True, "owner_started": False, "software_only": True,
                    "physical_stop_verified": False}
        return controller.stop()

    def reset_stop(self):
        with self._condition:
            if self._domain is None or self._fault or self._closed:
                return {"ok": False, "error": "robot episode not available; explicit activation required"}
            domain = self._domain
        return domain.reset_stop()

    def _retain_close(self, domain):
        try:
            result = domain.close()
            if not isinstance(result, dict):
                raise ValueError("hand closure requires a receipt")
        except BaseException as exc:
            result = {"ok": False, "complete": False, "error": type(exc).__name__}
        with self._condition:
            self._closure = retain_teardown_attempt(self._closure, result)
            if domain is self._domain and result.get("complete", result.get("ok")) is True:
                self._domain_retired = True
        return result

    def close(self):
        if not self._close_lock.acquire(timeout=5.):
            return {"ok": False, "complete": False, "close_pending": True,
                    "episode": self.status()}
        try:
            return self._close()
        finally:
            self._close_lock.release()

    def _close(self):
        with self._condition:
            if (self._closed and not self._building and self._closure is not None
                    and (self._domain is None or self._domain_retired)):
                return {**copy.deepcopy(self._closure),
                        "ok": self._closure.get("ok") is True and self._fault is None,
                        "episode": self.status()}
            self._closed = True
            self._active = False
        self.stop()
        with self._condition:
            self._condition.wait_for(lambda: not self._building, timeout=5.)
            if self._building:
                self._closure = retain_teardown_attempt(self._closure,
                    {"ok": False, "complete": False, "constructor_pending": True})
                return {**copy.deepcopy(self._closure), "episode": self.status()}
            domain = self._domain if not self._domain_retired else None
        if domain is not None:
            self._retain_close(domain)
        with self._condition:
            if self._closure is None:
                self._closure = {"ok": self._fault is None, "complete": True,
                                 "owner_started": self._controller is not None,
                                 "physical_stop_verified": False}
            return {**copy.deepcopy(self._closure), "ok": self._closure.get("ok") is True and self._fault is None,
                    "episode": self.status()}


def build_bounded_hand_service(cfg, directory):
    """Describe a single fixed hand without importing or constructing MuJoCo."""
    from ..apps.robot_runtime import describe_robot
    from ..apps.hand_runtime import build_hand_runtime
    from ..agent.trace import TraceLogger
    from ..memory.episodic import EpisodicMemory
    from ..robotics.runtime import RobotRuntime
    domains = describe_robot(cfg)
    if len(domains) != 1 or next(iter(domains.values())).profile["kind"] != "hand":
        raise ValueError("bounded_hand service requires exactly one fixed-hand domain")
    name, descriptor = next(iter(domains.items()))
    if descriptor.profile["model_identity_sha256"] is None:
        raise ValueError("bounded_hand service requires a prepared hand model pin")
    profile = copy.deepcopy(descriptor.profile)
    episode = BoundedHandEpisode(descriptor, lambda start: build_hand_runtime(
        profile, directory / "domains" / name, domain_id=name, start_owner=start))
    descriptor.runtime = episode
    runtime = RobotRuntime(domains, cfg=cfg, memory=EpisodicMemory(), trace=TraceLogger(directory))
    episode.runtime = runtime
    return runtime, episode
