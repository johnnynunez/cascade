"""Isaac Sim arm backend: the simulated reBot over the bridge.

Same ArmBase contract as the real RS arm -- min-jerk waypoints streamed
through the safety harness, feedback-based settling -- so anything proven
here transfers to hardware by swapping the profile back.
"""

from __future__ import annotations

import copy

import numpy as np

from ..config import Cfg
from ..grasping import evidence as grasp_evidence
from ..sim.bridge_client import BridgeClient, BridgeError
from ..sim.isaac_reset import validate_isaac_reset
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
        # All stream/settle durations below are PHYSICS seconds. This
        # independent wall budget includes state/target RPC and lock waits.
        self.motion_wall_timeout_s = float(cfg.get("motion_wall_timeout_s", 120.0))
        self.motion_rpc_timeout_s = float(cfg.get("motion_rpc_timeout_s", 1.0))
        self.settle_hold_s = float(cfg.get("settle_hold_s", .1))
        self.motion_rate_hz = cfg.get("motion_rate_hz", 30.0)
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
        # Acknowledged targets only; never infer motion from requested poses.
        self._acknowledged_joint_targets = 0

    def connect(self) -> None:
        self._client.connect()
        self._client.ping()
        self._stopped = False

    def disconnect(self) -> None:
        self._client.close()

    def get_state(self, *, timeout_s: float | None = None) -> RobotState:
        sample = self._client.state() if timeout_s is None else self._client.state(timeout_s=timeout_s)
        state = self._decode_state(sample)
        grasp_evidence.event("isaac_feedback", state=state, bridge_t=sample.get("t"))
        return state

    def _decode_state(self, s: dict) -> RobotState:
        # Validate WIRE dimensions before signs can broadcast a scalar or
        # slicing can hide surplus DOFs. Asset -> local only after validation.
        q = np.asarray(s.get("q"))
        if (q.shape != (self.n_joints,) or q.dtype.kind not in "fiu"
                or not np.isfinite(q).all()):
            raise BridgeError("Isaac feedback requires finite, exact-DOF asset q")
        dq = None if "dq" not in s else np.asarray(s["dq"])
        if dq is not None and (dq.shape != q.shape or dq.dtype.kind not in "fiu"
                               or not np.isfinite(dq).all()):
            raise BridgeError("Isaac feedback requires finite, exact-DOF asset dq")
        clock = copy.deepcopy(s.get("physics_clock"))
        if isinstance(clock, dict):
            clock["source"] = self._client._addr  # bind locally, never trust a wire endpoint
        return RobotState(
            q=q.astype(float) * self._signs,
            dq=dq.astype(float) * self._signs if dq is not None else None,
            gripper_pos=float(s.get("gripper_pos", 0.0)),
            gripper_valid="gripper_pos" in s,
            physics_clock=clock,
            gripper_joints=copy.deepcopy(s.get("gripper_joints")),
        )

    def state_from_frame(self, frame) -> RobotState:
        """Require this bridge/robot's capture-time q, then use driver signs.

        Endpoint binding comes from the camera client, not an untrusted wire
        field. An old bridge or a camera for another arm must fail closed.
        No timestamp is treated as exposure time or compared across hosts.
        """
        c = getattr(frame, "capture", None)
        if (not isinstance(c, dict) or c.get("backend") != "isaac"
                or c.get("source") != self._client._addr):
            raise BridgeError("Isaac capture source missing or belongs to another bridge")
        s = c.get("proprioception")
        robot_id = self._cfg.get("bridge_robot_id")
        if (not isinstance(s, dict) or type(s.get("version")) is not int or s["version"] != 1
                or s.get("backend") != "isaac" or not robot_id or s.get("robot_id") != robot_id
                or s.get("joint_convention") != "asset"):
            raise BridgeError("Isaac capture snapshot missing/invalid or robot identity mismatch")
        t = s.get("t")
        if (type(t) not in (int, float) or not np.isfinite(t) or t < 0
                or t != c.get("t") or s.get("time_source") != "physics_loop_monotonic"):
            raise BridgeError("Isaac capture snapshot timestamp/clock mismatch")
        q = np.asarray(s.get("q"))
        if (q.shape != (self.n_joints,) or q.dtype.kind not in "fiu"
                or not np.isfinite(q).all()):
            raise BridgeError("Isaac capture snapshot requires finite, exact-DOF asset joints")
        if self._cfg.get("require_robot_pixel_mask") and getattr(frame, "robot_mask", None) is None:
            raise BridgeError("Isaac render robot pixel mask required")
        return self._decode_state(s)

    def send_joint_target(self, q: np.ndarray, *, timeout_s: float | None = None) -> None:
        if self._stopped:
            raise BridgeError("soft-stopped; call resume()")
        # local -> asset convention for the bridge's raw DOF targets
        q_asset = np.asarray(q, dtype=float)[: self.n_joints] * self._signs
        if timeout_s is None:
            self._client.set_joints(q_asset)
        else:
            self._client.set_joints(q_asset, timeout_s=timeout_s)
        self._acknowledged_joint_targets += 1
        grasp_evidence.event("isaac_joint_target_sent", q_local=q, q_asset=q_asset)

    def stream_to(self, q_target, duration_s, rate_hz=None, approve=None,
                  settle_tol=None, settle_timeout_s=None, preflight=None,
                  before_stream=None, bias_compensate=False, feedback_guard=None) -> bool:
        """Stream in simulator time; `bias_compensate` remains unsupported.

        Accept the existing advisory keyword explicitly so TypeError cannot
        trigger SafeArm's legacy retry after any command has been sent.
        """
        from .simulation_motion import SimulationMotion
        from .motion_profile import resolve_motion_rate

        motion = SimulationMotion(self, approve=approve, before_stream=before_stream,
                                  feedback_guard=feedback_guard)
        return motion.stream(q_target, duration_s, resolve_motion_rate(self, rate_hz),
                             self.settle_tol if settle_tol is None else settle_tol,
                             self.settle_timeout_s if settle_timeout_s is None else settle_timeout_s,
                             preflight)

    def wait_settled(self, q_target, tol, timeout_s) -> bool:
        from .simulation_motion import SimulationMotion

        return SimulationMotion(self).settle(q_target, tol, timeout_s)

    def validate_simulation_clock(self) -> dict:
        """Read-only capability check using the motion executor's validator.

        Diagnostics call this before resetting or moving the scene. It proves
        the source/schema at this read; stream_to still validates every step.
        """
        from .simulation_motion import SimulationMotion

        state = SimulationMotion(self).get_state()
        return copy.deepcopy(state.physics_clock)

    def set_gripper(self, pos: float, effort: float = 1.0) -> None:
        if self._stopped:
            raise BridgeError("soft-stopped; call resume()")
        self._client.gripper(pos, effort)
        grasp_evidence.event("isaac_gripper_target_sent", position=pos, effort=effort)

    def stop(self) -> None:
        self._stopped = True
        try:
            self._client.stop()
        except BridgeError:
            pass  # bridge gone: sim arm holds position on its own

    def reset_props(self) -> dict:
        """Reset the live bridge world and return its physics read-back.

        Settling runs on Kit's main thread; allow its 60 s server budget
        plus transport time rather than the usual short state-query timeout.
        """
        if self._stopped:
            raise BridgeError("soft-stopped; call resume() before reset")
        return validate_isaac_reset(self._client.request({"op": "reset_props"}, timeout_s=65.0))

    def resume(self) -> None:
        self._stopped = False
