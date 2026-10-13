"""Franka hardware readiness independent of ROS message imports.

Joint feedback can keep arriving after a reflex deactivates the hardware.
Motion therefore needs a fresh native robot state as well as joint feedback.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping


ROBOT_MODE_NAMES = {
    0: "Other",
    1: "Idle",
    2: "Move",
    3: "Guiding",
    4: "Reflex",
    5: "UserStopped",
    6: "AutomaticErrorRecovery",
}


def _active_errors(errors) -> tuple[str, ...]:
    """Read boolean error flags, including generated ROS messages with slots."""
    if errors is None:
        return ()
    if isinstance(errors, Mapping):
        values = errors.items()
    else:
        fields = getattr(errors, "get_fields_and_field_types", None)
        if callable(fields):
            names = fields().keys()
        elif hasattr(errors, "__dict__"):
            names = vars(errors).keys()
        else:
            names = getattr(type(errors), "__slots__", ())
        values = ((name.lstrip("_"), getattr(errors, name)) for name in names)
    return tuple(sorted(name for name, value in values
                        if isinstance(value, bool) and value))


class FrankaStateMonitor:
    """Cache native fault/mode feedback and refuse unavailable hardware.

    Receipt times use a monotonic clock, rather than ROS timestamps, so a
    clock reset cannot make stale hardware feedback look current. Historical
    errors remain visible after recovery but do not themselves block motion.
    """

    def __init__(self, timeout_s: float = 1.0,
                 clock: Callable[[], float] = time.monotonic):
        self.timeout_s = float(timeout_s)
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("Franka state timeout_s must be finite and positive")
        self._clock = clock
        self._lock = threading.Lock()
        self._data: tuple[int | None, tuple[str, ...], tuple[str, ...], float] | None = None

    def update(self, msg) -> None:
        """Copy fields from a native FrankaRobotState (or a duck-typed stub)."""
        mode = getattr(msg, "robot_mode", None)
        try:
            mode = int(mode) if mode is not None else None
        except (TypeError, ValueError):
            mode = None
        current = _active_errors(getattr(msg, "current_errors", None))
        previous = _active_errors(getattr(msg, "last_motion_errors", None))
        with self._lock:
            self._data = (mode, current, previous, self._clock())

    def snapshot(self) -> dict:
        """Return a detached, JSON-serializable readiness record."""
        with self._lock:
            if self._data is None:
                return {
                    "motion_ready": False,
                    "robot_mode": "Unknown",
                    "robot_mode_code": None,
                    "current_errors": [],
                    "last_motion_errors": [],
                    "state_age_s": None,
                }
            mode, current, previous, received = self._data
            age = max(0.0, self._clock() - received)
            return {
                "motion_ready": mode in (1, 2) and not current and age <= self.timeout_s,
                "robot_mode": ROBOT_MODE_NAMES.get(mode, f"Unknown({mode})"),
                "robot_mode_code": mode,
                "current_errors": list(current),
                "last_motion_errors": list(previous),
                "state_age_s": age,
            }

    def status(self) -> dict:
        """Alias for callers exposing nonthrowing hardware status."""
        return self.snapshot()

    def ensure_ready(self) -> None:
        """Raise with native fault context instead of a later settle timeout."""
        state = self.snapshot()
        if state["motion_ready"]:
            return
        if state["state_age_s"] is None:
            raise RuntimeError("Franka hardware state unavailable; waiting for native robot state")
        reasons = []
        if state["state_age_s"] > self.timeout_s:
            reasons.append(f"native robot state stale ({state['state_age_s']:.2f}s; "
                           f"limit {self.timeout_s:.2f}s)")
        if state["current_errors"]:
            reasons.append("current_errors: " + ", ".join(state["current_errors"]))
        if state["robot_mode_code"] not in (1, 2):
            reasons.append(f"robot mode {state['robot_mode']} does not permit motion")
        if state["last_motion_errors"]:
            reasons.append("last_motion_errors: " + ", ".join(state["last_motion_errors"]))
        raise RuntimeError("Franka hardware not ready: " + "; ".join(reasons))
