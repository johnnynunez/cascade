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

    def connect(self) -> None:
        self._connected = True
        self._stopped = False

    def disconnect(self) -> None:
        self._connected = False

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
        self._q = np.asarray(q, dtype=float).copy()
        self.commands.append(self._q.copy())

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
    def stream_to(self, q_target, duration_s, rate_hz=50.0, approve=None,
                  settle_tol=None, settle_timeout_s=2.0) -> bool:
        q_start = self.get_state().q.copy()
        q_target = np.asarray(q_target, dtype=float).reshape(-1)
        steps = max(2, int(duration_s * rate_hz))
        dt = duration_s / steps
        q_prev = q_start
        from .arm_base import min_jerk

        for i in range(1, steps + 1):
            s = min_jerk(i / steps)
            q_i = q_start + (q_target - q_start) * s
            if approve is not None:
                approve(q_prev, q_i, dt)
            self.send_joint_target(q_i)
            q_prev = q_i
        return not self._stopped
