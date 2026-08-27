"""reBot DevArm B601 (RobStride) over the motorbridge SDK.

Transport sibling of `rebot_rs_arm.RebotRSArm`. That backend rides
reBotArm_control_py -> motorbridge -> SocketCAN, which pins it to Linux;
this one talks to `motorbridge.Controller` directly, so it also works where
SocketCAN does not exist:

    Linux    can0 -> SocketCAN
    macOS    can0 -> PCAN_USBBUS1 via MacCAN PCBUSB (libPCBUSB.dylib)
    Windows  can0 -> PCAN_USBBUS1 via PCAN-Basic

Dropping the SDK also drops its hardware YAML, so everything the SDK used to
supply implicitly is explicit in the arm profile here: motor ids, host id,
and per-joint MIT gains. `mit_kp`/`mit_kd` are REQUIRED for that reason --
there is no vendor default to fall back on.

Rig-verified on the B601-RS over PCAN, 2026-08-27 (see scripts/diag_rebot_mb.py
and scripts/jog_rebot_mb.py, the bring-up tools that established these):

- `mechPos` (0x7019) is in JOINT radians, 1:1 with the URDF scale and sharing
  its zero: joint 4's hand-swept span measured 3.491 rad against the URDF's
  3.480, and joint 1 came out centred on zero at +-2.6. So no gear-ratio or
  offset correction belongs here. `wire_signs` exists only to flip a mirrored
  joint without a code change; it defaults to no flip.
- Positions come from parameter reads, never the state stream. RS motors emit
  compact type-0x18 report frames that the generic `Motor.get_state()` path
  does not decode, and `robstride_set_active_report(True)` yields a single
  type-2 frame and then freezes.
- Motors are LIMP after a clean bus shutdown -- not holding. `connect()`
  therefore captures the live pose and preloads it as the MIT setpoint
  BEFORE `enable_all()`, or the arm snaps from wherever it rests to whatever
  setpoint the motors happened to hold.
- The gripper's travel was measured at ~0..+6.39 rad, closed at 0. That is
  the OPPOSITE polarity to the DM-derived numbers in `configs/arms/rebot_rs.yaml`
  (open -6.8), which is why this profile ships its own gripper block. Commanding
  the DM value here drives the jaws into the closed hard stop at full effort.
- Hand-moving reaches poses OUTSIDE the URDF limits (joint 3 to 4.22 rad past
  its 3.14 limit; joint 6 is continuous-rotation and span 9.89 rad). Park the
  arm inside limits before connecting or the harness rejects the first pose.

Operational notes:
- The bus is exclusive: host id 0xFD is a single owner, so motorbridge-gateway
  / MotorBridge Studio and any LeRobot process must be stopped first.
- `stop()` is a soft stop that re-asserts the last commanded pose;
  `hard_estop()` is torque-off and a loaded arm free-falls.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from ..config import Cfg
from ..types import RobotState
from .arm_base import ArmBase

MECH_POS = 0x7019
MECH_VEL = 0x701A


class RebotRSMotorBridgeArm(ArmBase):
    #: consecutive feedback failures tolerated before motion must abort
    MAX_READ_FAILURES = 3

    def __init__(self, cfg: Cfg):
        self._cfg = cfg
        self._ctrl = None
        self._motors: dict[int, object] = {}
        self._lock = threading.Lock()
        self._stopped = False
        self.n_joints = int(cfg.get("n_joints", 6))
        self._last_q = np.zeros(self.n_joints)
        self._last_cmd_q: np.ndarray | None = None
        self._read_failures = 0

        self.settle_tol = float(cfg.get("settle_tol", 0.05))
        self.settle_timeout_s = float(cfg.get("settle_timeout_s", 2.0))
        self._channel = str(cfg.get("channel", "can0"))
        self._model = str(cfg.get("motor_model", "rs-00"))
        self._host_id = int(cfg.get("host_id", 0xFD))
        self._read_timeout_ms = int(cfg.get("read_timeout_ms", 200))

        ids = cfg.get("joint_ids") or list(range(1, self.n_joints + 1))
        self._joint_ids = [int(i) for i in ids][: self.n_joints]
        gid = cfg.get("gripper_id", 7)
        self._gripper_id = None if gid is None else int(gid)

        signs = cfg.get("wire_signs") or [1] * self.n_joints
        self._wire = np.asarray([int(s) for s in signs][: self.n_joints], dtype=float)

        kp = cfg.get("mit_kp")
        kd = cfg.get("mit_kd")
        if kp is None or kd is None:
            raise RuntimeError(
                "arm profile must set mit_kp and mit_kd: the motorbridge path has "
                "no SDK hardware YAML to inherit per-joint gains from"
            )
        self._mit_kp = np.asarray(kp, dtype=float).reshape(-1)[: self.n_joints]
        self._mit_kd = np.asarray(kd, dtype=float).reshape(-1)[: self.n_joints]

        g = cfg.get("gripper", Cfg({}))
        self._grip_open = float(g.get("open_pos", 6.2))
        self._grip_closed = float(g.get("closed_pos", 0.0))
        self._grip_kp = float(g.get("kp", 6.0))
        self._grip_kd = float(g.get("kd", 0.4))

    # ── lifecycle ────────────────────────────────────────────────────────

    def connect(self) -> None:
        try:
            from motorbridge import Controller, Mode
        except ImportError as e:
            raise RuntimeError(
                "motorbridge not importable; install it into this venv "
                "(pip install motorbridge). On macOS the CAN runtime also needs "
                "MacCAN PCBUSB, and DYLD_FALLBACK_LIBRARY_PATH must include the "
                "directory holding the bare-name 'PCBUSB' symlink."
            ) from e

        ctrl = Controller(channel=self._channel)
        motors: dict[int, object] = {}
        try:
            all_ids = list(self._joint_ids)
            if self._gripper_id is not None:
                all_ids.append(self._gripper_id)
            for mid in all_ids:
                motors[mid] = ctrl.add_robstride_motor(
                    motor_id=mid, feedback_id=self._host_id, model=self._model
                )
            for m in motors.values():
                m.ensure_mode(Mode.MIT)

            self._ctrl, self._motors = ctrl, motors
            # Capture where the arm actually rests. Motors are limp after a
            # clean shutdown, so this pose -- not the motors' stale setpoint --
            # is what they must be told to hold the instant torque arrives.
            q = self._read_positions()
            grip = self._gripper_pos()
            self._send_mit(q)
            if grip is not None:
                self.set_gripper(grip, effort=0.5)
            ctrl.enable_all()
            self._send_mit(q)

            self._stopped = False
            self._read_failures = 0
            self._last_q = q
            self._last_cmd_q = q.copy()
        except Exception:
            self._ctrl, self._motors = None, {}
            try:
                ctrl.close_bus()
            finally:
                ctrl.close()
            raise

    def disconnect(self) -> None:
        """Release the bus. Does NOT torque off: a loaded arm would fall.
        Park the arm (move_home) before calling this."""
        ctrl, self._ctrl = self._ctrl, None
        self._motors = {}
        if ctrl is not None:
            try:
                ctrl.close_bus()
            finally:
                ctrl.close()

    # ── feedback (param reads, the only reliable RS path) ────────────────

    def _read_positions(self) -> np.ndarray:
        q = np.zeros(self.n_joints)
        with self._lock:
            for i, mid in enumerate(self._joint_ids):
                raw = self._motors[mid].robstride_get_param_f32(
                    MECH_POS, self._read_timeout_ms
                )
                q[i] = float(raw)
        return q * self._wire

    def get_state(self) -> RobotState:
        if self._ctrl is None:
            raise RuntimeError("arm not connected")
        try:
            q = self._read_positions()
            self._last_q = q
            self._read_failures = 0
        except Exception as e:
            # One transient CAN hiccup: serve last known. Repeated failures
            # mean the bus is unhealthy -- fail loudly so motion aborts
            # instead of planning from phantom positions.
            self._read_failures += 1
            if self._read_failures >= self.MAX_READ_FAILURES:
                raise RuntimeError(
                    f"CAN feedback lost ({self._read_failures} consecutive "
                    f"mechPos read failures): {e}"
                ) from e
            q = self._last_q
        gp = self._gripper_pos()
        return RobotState(
            q=q.copy(),
            gripper_pos=0.0 if gp is None else gp,
            gripper_valid=gp is not None,
            t=time.monotonic(),
        )

    def _gripper_pos(self) -> float | None:
        """Gripper motor angle, or None when the read fails (callers must
        treat None as 'unknown', never as a position)."""
        if self._ctrl is None or self._gripper_id is None:
            return None
        try:
            with self._lock:
                return float(
                    self._motors[self._gripper_id].robstride_get_param_f32(
                        MECH_POS, self._read_timeout_ms
                    )
                )
        except Exception:
            return None

    def _gripper_vel(self) -> float | None:
        if self._ctrl is None or self._gripper_id is None:
            return None
        try:
            with self._lock:
                return float(
                    self._motors[self._gripper_id].robstride_get_param_f32(
                        MECH_VEL, self._read_timeout_ms
                    )
                )
        except Exception:
            return None

    # ── commands ─────────────────────────────────────────────────────────

    def _send_mit(self, q: np.ndarray) -> None:
        wire = np.asarray(q, dtype=float).reshape(-1)[: self.n_joints] * self._wire
        with self._lock:
            for i, mid in enumerate(self._joint_ids):
                self._motors[mid].send_mit(
                    float(wire[i]), 0.0, float(self._mit_kp[i]), float(self._mit_kd[i]), 0.0
                )

    def send_joint_target(self, q: np.ndarray) -> None:
        if self._ctrl is None:
            raise RuntimeError("arm not connected")
        if self._stopped:
            return
        q = np.asarray(q, dtype=float).reshape(-1)[: self.n_joints]
        self._send_mit(q)
        self._last_cmd_q = q.copy()

    def set_gripper(self, pos: float, effort: float = 1.0) -> None:
        if self._ctrl is None or self._gripper_id is None or self._stopped:
            return
        kp = self._grip_kp * float(np.clip(effort, 0.05, 1.0))
        with self._lock:
            self._motors[self._gripper_id].send_mit(
                float(pos), 0.0, kp, self._grip_kd, 0.0
            )

    def open_gripper(self) -> None:
        self.set_gripper(self._grip_open, effort=0.8)

    def close_gripper_two_stage(
        self,
        width_frac_stage1: float = 0.5,
        width_frac_stage2: float = 0.85,
        effort: float = 0.6,
        stall_vel: float = 0.05,
        timeout_s: float = 3.0,
    ) -> float | None:
        """ASPIRE-style two-stage close; returns the final gripper position
        (rad) or None if feedback was unavailable.

        A stall (jaws stopped short of target) is the grasp-contact signal.
        To avoid mistaking motor spin-up for a stall: skip a grace period
        after each command and require TWO consecutive slow samples with some
        minimum travel. A stage-1 stall does NOT end the close -- stage 2
        still runs at full effort to seat the grip.
        """
        span = self._grip_closed - self._grip_open
        start_pos = self._gripper_pos()
        for frac, eff in ((width_frac_stage1, effort * 0.7), (width_frac_stage2, effort)):
            target = self._grip_open + span * frac
            self.set_gripper(target, effort=eff)
            time.sleep(0.3)  # spin-up grace period
            slow_samples = 0
            deadline = time.monotonic() + timeout_s / 2
            while time.monotonic() < deadline:
                time.sleep(0.15)
                pos = self._gripper_pos()
                vel = self._gripper_vel()
                if pos is None:
                    continue
                if abs(pos - target) < 0.1:
                    break  # reached this stage's target
                traveled = start_pos is not None and abs(pos - start_pos) > 0.05
                if vel is not None and abs(vel) < stall_vel and traveled:
                    slow_samples += 1
                    if slow_samples >= 2:
                        break  # stalled on the object: proceed to seat stage
                else:
                    slow_samples = 0
        return self._gripper_pos()

    def stop(self) -> None:
        """Soft stop: freeze at the last commanded pose and refuse new
        commands. Does NOT torque-off (that would make a loaded arm
        free-fall); use hard_estop() for that."""
        self._stopped = True
        if self._ctrl is not None and self._last_cmd_q is not None:
            try:
                self._send_mit(self._last_cmd_q)
            except Exception:
                pass

    def hard_estop(self) -> None:
        """Torque-off every motor. The arm WILL fall under gravity."""
        self._stopped = True
        if self._ctrl is not None:
            try:
                with self._lock:
                    self._ctrl.disable_all()
            except Exception:
                pass
