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
        # Newton's solver bleeds off the last of the tracking error more
        # slowly than PhysX: a 0.17 rad step measured 3.6 s to come inside
        # settle_tol on this rig, so the 2.0 s base default reported "did not
        # settle" on poses the arm was reaching correctly.
        self.settle_timeout_s = float(cfg.get("settle_timeout_s", 5.0))
        # joint_signs map the bridge's ASSET joint convention to the client's
        # LOCAL convention that the kinematics/harness use. The bridge reports
        # and accepts raw DOF (asset) values; the planner/IK work in local.
        # Without this conversion, state.q feeds the harness a sign-flipped
        # pose whose FK puts links below the table (phantom "link would hit
        # the table" rejections) even though the real arm is safely elbow-up.
        _signs = cfg.get("joint_signs")
        self._signs = (np.asarray(_signs, dtype=float)[: self.n_joints]
                       if _signs else np.ones(self.n_joints))
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
        # asset -> local convention
        q = np.asarray(s["q"], dtype=float)[: self.n_joints] * self._signs
        dq = np.asarray(s.get("dq", []), dtype=float)
        return RobotState(
            q=q,
            dq=(dq[: self.n_joints] * self._signs) if dq.size else None,
            gripper_pos=float(s.get("gripper_pos", 0.0)),
            gripper_valid="gripper_pos" in s,
        )

    def send_joint_target(self, q: np.ndarray) -> None:
        if self._stopped:
            raise BridgeError("soft-stopped; call resume()")
        # local -> asset convention for the bridge's raw DOF targets
        q_asset = np.asarray(q, dtype=float)[: self.n_joints] * self._signs
        self._client.set_joints(q_asset)

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
