"""Immutable command/safety samples of an original planner curve.

The native evaluator is consumed before any actuator operation. A profile is
data, never execution permission: SafeArm must bind it to fresh feedback and
the live safety/scene/contact/payload state immediately before streaming.
"""
from dataclasses import dataclass

import numpy as np

from ..control.motion_profile import ProfileTarget, curve_profile
from ..types import SafetyViolation


def _frozen(value):
    # Bytes-backed arrays cannot be made writable by another consumer.
    array = np.asarray(value, dtype=float)
    return np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape)


@dataclass(frozen=True)
class TrajectoryProfile:
    start: np.ndarray
    end: np.ndarray
    duration_s: float
    targets: tuple[ProfileTarget, ...]
    plan: object

    @classmethod
    def from_curve(cls, evaluate, duration_s, rate_hz, plan, *, max_steps=10000):
        targets = tuple(ProfileTarget(_frozen(item.q), item.dt,
                        tuple((_frozen(a), _frozen(b), dt) for a, b, dt in item.checks))
                        for item in curve_profile(evaluate, duration_s, rate_hz,
                                                  max_steps=max_steps))
        if not targets:
            raise SafetyViolation("trajectory has no command samples")
        return cls(_frozen(targets[0].checks[0][0]), _frozen(targets[-1].q),
                   float(duration_s), targets, plan)

    def __iter__(self):
        return iter(self.targets)
