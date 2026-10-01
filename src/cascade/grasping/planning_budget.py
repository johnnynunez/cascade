"""One finite wall deadline for pre-actuation learned-grasp search."""
from __future__ import annotations

import math
import time

from ..types import SkillError


class PlanningBudgetExceeded(SkillError):
    """Planning ended before an actuator was authorized."""


class PlanningBudget:
    def __init__(self, max_batches, timeout_s, *, deadlines=(), cancel=None):
        if type(max_batches) is not int or not 1 <= max_batches <= 8:
            raise SkillError("grasp feasibility_max_batches must be an integer in [1, 8]")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise SkillError("grasp feasibility_timeout_s must be finite and positive")
        self.started = time.monotonic()
        self.deadline = self.started + timeout_s
        for deadline in deadlines:
            if deadline is None:
                continue
            if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
                raise SkillError("grasp planning deadline must be finite")
            self.deadline = min(self.deadline, deadline)
        self.max_batches = max_batches
        self.cancel = cancel
        self.check()

    def check(self):
        if self.cancel is not None:
            self.cancel()
        if time.monotonic() >= self.deadline:
            raise PlanningBudgetExceeded("grasp planning budget exhausted before actuation")

    def remaining(self, cap=None):
        self.check()
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise PlanningBudgetExceeded("grasp planning budget exhausted before actuation")
        if cap is not None:
            if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap <= 0:
                raise SkillError("grasp planning RPC cap must be finite and positive")
            remaining = min(remaining, cap)
        return remaining
