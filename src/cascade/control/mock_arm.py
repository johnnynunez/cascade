"""Kinematic mock arm: perfect tracking with a small first-order lag.

Runs against the same URDF/kinematics as the real arm so IK, workspace and
collision logic are exercised identically. Used by all tests and by
--arm mock dry-runs; it is also the natural attach point for an Isaac Sim
backend later (see docs/ROADMAP.md).
"""

from __future__ import annotations

import time

import numpy as np

from ..config import Cfg
from ..types import RobotState
from .arm_base import ArmBase


class MockArm(ArmBase):
    def __init__(self, cfg: Cfg | None = None, kinematics=None):
        self._cfg = cfg
        self.kin = kinematics
        home = None
        if cfg is not None:
            home = cfg.get("home_q")
        self._q = np.asarray(home if home is not None else [0.0, -0.5, -0.9, 0.0, 0.6, 0.0], dtype=float)
        # DOF follows the profile, not the class default: this mock stands in
        # for every arm the framework supports (5-DoF SO-101, 6-DoF reBot,
        # 7-DoF Panda), and a wrong n_joints silently truncates or broadcasts
        # every commanded pose.
        if cfg is not None and cfg.get("n_joints") is not None:
            self.n_joints = int(cfg.get("n_joints"))
        else:
            self.n_joints = int(self._q.size)
        self._gripper = 0.0
        self._gripper_effort = 1.0
        self._connected = False
        self._stopped = False
        self.commands: list[np.ndarray] = []  # inspection hook for tests
        # Gripper travel convention for the mock: open_pos..closed_pos from
        # config (defaults 0..1). If `object_stop_frac` is set, the jaws jam
        # at that fraction of travel -- emulates closing on an object.
        g = cfg.get("gripper") if cfg is not None else None
        self._grip_open = float(g.get("open_pos", 0.0)) if g else 0.0
        self._grip_closed = float(g.get("closed_pos", 1.0)) if g else 1.0
        self.object_stop_frac: float | None = None
        #: joint -> position it is stuck at (block_joint): commands still
        #: stream, the measured pose stops -- an obstacle the arm runs into
        self._blocked: dict[int, tuple[float, int | None]] = {}

    def connect(self) -> None:
        self._connected = True
        self._stopped = False

    def disconnect(self) -> None:
        self._connected = False

    #: No hardware: nothing can drop.
    disconnect_cuts_torque = False

    def get_state(self) -> RobotState:
        return RobotState(
            q=self._q.copy(),
            dq=np.zeros_like(self._q),
            tau=np.zeros_like(self._q),
            gripper_pos=self._gripper,
            t=time.monotonic(),
        )

    def send_joint_target(self, q: np.ndarray) -> None:
        if not self._connected:
            raise RuntimeError("mock arm not connected")
        if self._stopped:
            return
        q = np.asarray(q, dtype=float).copy()
        measured = q.copy()
        for j, (at, direction) in getattr(self, "_blocked", {}).items():
            if direction is None:
                measured[j] = at      # something stops this joint where it is
            elif direction > 0:
                measured[j] = min(q[j], at)   # a wall: cannot pass `at` going up
            else:
                measured[j] = max(q[j], at)
        self._q = measured
        self.commands.append(q.copy())

    def block_joint(self, j: int, at: float | None = None, direction: int | None = None) -> None:
        """Emulate an obstacle: joint `j` stops following commands at `at`
        (default: where it is now) -- the wrist pressed against clutter.
        `direction` None freezes it both ways; +1 / -1 make it a wall the
        joint cannot pass going up / down but can always back away from.
        Commands are still recorded in `commands`; only the measured pose
        (`get_state`) stops. `unblock_joints()` releases."""
        j = int(j)
        if at is not None and direction is None:
            self._q[j] = float(at)
        at = float(self._q[j]) if at is None else float(at)
        self._blocked[j] = (at, None if direction is None else (1 if direction > 0 else -1))

    def unblock_joints(self) -> None:
        self._blocked.clear()

    def set_gripper(self, pos: float, effort: float = 1.0) -> None:
        if self._stopped:
            return
        pos = float(pos)
        if self.object_stop_frac is not None:
            span = self._grip_closed - self._grip_open
            stop = self._grip_open + span * self.object_stop_frac
            # Jaws cannot travel past the object, whichever way "closed" is.
            if span >= 0:
                pos = min(pos, stop)
            else:
                pos = max(pos, stop)
        self._gripper = pos
        self._gripper_effort = float(effort)

    def stop(self) -> None:
        self._stopped = True

    def resume(self) -> None:
        self._stopped = False

    # Mock streaming doesn't need real-time pacing; override to skip sleeps.
    # Same ticks, approve and contact monitor as ArmBase (logical tick time).
    def stream_to(self, q_target, duration_s, rate_hz=50.0, approve=None,
                  settle_tol=None, settle_timeout_s=2.0, preflight=None,
                  before_stream=None, contact_monitor=None) -> bool:
        from .arm_base import min_jerk, prepare_stream

        q_start, q_target = prepare_stream(self, q_target, duration_s, preflight, before_stream)
        steps = max(2, int(duration_s * rate_hz))
        dt = duration_s / steps
        ticks = (q_start + (q_target - q_start) * min_jerk(i / steps) for i in range(1, steps + 1))
        self._stream_ticks(q_start, ticks, dt, approve, contact_monitor, paced=False)
        return not self._stopped

    def stream_path(self, waypoints, duration_s, rate_hz=50.0, approve=None,
                    settle_tol=None, settle_timeout_s=2.0, preflight=None,
                    before_stream=None, contact_monitor=None) -> bool:
        """ArmBase.stream_path without real-time pacing (same ticks)."""
        from .arm_base import path_ticks, prepare_stream

        waypoints = [np.asarray(w, dtype=float).reshape(-1) for w in waypoints]
        if not waypoints:
            from ..types import SafetyViolation

            raise SafetyViolation("empty joint path; no motion sent")
        q_start, _ = prepare_stream(self, waypoints[-1], duration_s, preflight, before_stream)
        ticks, dt = path_ticks(q_start, waypoints, duration_s, rate_hz)
        self._stream_ticks(q_start, ticks, dt, approve, contact_monitor, paced=False)
        return not self._stopped
