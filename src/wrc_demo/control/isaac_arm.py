"""Isaac Sim arm backend: the simulated reBot over the bridge.

Same ArmBase contract as the real RS arm -- min-jerk waypoints streamed
through the safety harness, feedback-based settling -- so anything proven
here transfers to hardware by swapping the profile back.
"""

from __future__ import annotations

import numpy as np

from ..config import Cfg
from ..sim.bridge_client import BridgeClient, BridgeError
from ..types import RobotState
from .arm_base import ArmBase


class IsaacArm(ArmBase):
    def __init__(self, cfg: Cfg, kinematics=None):
        self._cfg = cfg
        self.n_joints = int(cfg.get("n_joints", 6))
        self.settle_tol = float(cfg.get("settle_tol", 0.02))
        self._client = BridgeClient(
            host=str(cfg.get("bridge_host", "127.0.0.1")),
            port=int(cfg.get("bridge_port", 8611)),
        )
        self._stopped = False

    def connect(self) -> None:
        self._client.connect()
        self._client.ping()
        self._stopped = False

    def disconnect(self) -> None:
        self._client.close()

    def get_state(self) -> RobotState:
        s = self._client.state()
        q = np.asarray(s["q"], dtype=float)[: self.n_joints]
        dq = np.asarray(s.get("dq", []), dtype=float)
        return RobotState(
            q=q,
            dq=dq[: self.n_joints] if dq.size else None,
            gripper_pos=float(s.get("gripper_pos", 0.0)),
            gripper_valid="gripper_pos" in s,
        )

    def send_joint_target(self, q: np.ndarray) -> None:
        if self._stopped:
            raise BridgeError("soft-stopped; call resume()")
        self._client.set_joints(np.asarray(q, dtype=float)[: self.n_joints])

    def set_gripper(self, pos: float, effort: float = 1.0) -> None:
        if self._stopped:
            raise BridgeError("soft-stopped; call resume()")
        self._client.gripper(pos, effort)

    def stop(self) -> None:
        self._stopped = True
        try:
            self._client.stop()
        except BridgeError:
            pass  # bridge gone: sim arm holds position on its own

    def resume(self) -> None:
        self._stopped = False
