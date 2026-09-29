"""Real reBot DevArm B601 (RobStride build) backend.

Transport rides on reBotArm_control_py's RebotArm/JointGroup (motorbridge ->
SocketCAN can0 @ 1 Mbps), but with the rig-verified fixes this SDK lacks:

- Joint positions are read with RobStride param reads of mechPos (0x7019),
  because RS motors stream compact type-0x18 report frames that motorbridge
  get_state() never decodes; robstride_set_active_report(True) yields a
  single type-2 frame and then freezes. Param reads return the exact f32.
- The hardware YAML is forced to the RS profile; kinematics come from
  cascade.control.Kinematics on the RS URDF (this repo's assets), never
  from the SDK's global config (which silently loads the DM URDF).
- Gripper close is a two-stage, effort-scaled MIT command with stall
  detection via mechVel (0x701A) instead of the SDK's broken open/close.

Operational notes (verified on this rig 2026-07):
- bring the bus up first: sudo ip link set can0 up type can bitrate 1000000
- do NOT run motorbridge-gateway / MotorBridge Studio at the same time; both
  use host id 0xFD and fight over reporting mode.
- gripper open/close angles were re-measured on the RS build (0 -> +6.39 rad,
  closed at 0; see configs/arms/rebot_rs.yaml). The open/close KP/KD are still
  the conservative values and should be bumped after an onsite grasp check.
"""

from __future__ import annotations

import sys
import threading
import time

import numpy as np

from ..config import Cfg
from ..types import RobotState
from .arm_base import ArmBase

MECH_POS = 0x7019
MECH_VEL = 0x701A


class RebotRSArm(ArmBase):
    def __init__(self, cfg: Cfg):
        self._cfg = cfg
        sdk_path = cfg.get("sdk_path")
        if sdk_path and sdk_path not in sys.path:
            sys.path.insert(0, sdk_path)
        self._arm = None
        self._lock = threading.Lock()
        self._stopped = False
        self._last_q = np.zeros(self.n_joints)
        self._mit_kp = None
        self._mit_kd = None
        self.settle_tol = float(cfg.get("settle_tol", 0.05))
        g = cfg.get("gripper", Cfg({}))
        self._grip_open = float(g.get("open_pos", -6.8))
        self._grip_closed = float(g.get("closed_pos", 0.0))
        self._grip_kp = float(g.get("kp", 6.0))
        self._grip_kd = float(g.get("kd", 0.4))
        # Stiffness used AFTER contact, to hold with a light constant force
        # rather than crushing at the full close kp. tau = kp*(target - pos)
        # under MIT, so this number is the holding force in kp units.
        self._grip_hold_kp = float(g.get("hold_kp", 1.0))
        self._grip_contact_pos: float | None = None
        # Gravity-compensation feedforward (Pinocchio g(q)) added to every MIT
        # arm command. Pure PD (tau_ff=0) holds with steady-state error g/kp,
        # which is the "joint 3 drops then recovers" droop seen during the
        # park. `tau_scale` is the SDK's empirical model-vs-hardware correction
        # for the gravity-loaded joints. Default OFF; the RS profile enables it.
        gc = cfg.get("gravity_comp", Cfg({}))
        self._gc_enabled = bool(gc.get("enabled", False))
        self._gc_tau_scale = self._resolve_tau_scale(gc.get("tau_scale"))
        self._gc_kin = None  # Kinematics (model + joint_signs baked), built in connect()

    def _resolve_tau_scale(self, tau_scale) -> np.ndarray:
        """`gravity_comp.tau_scale` -> per-joint (n_joints,) scale; default 1.0."""
        if tau_scale is None:
            return np.ones(self.n_joints)
        arr = np.asarray(tau_scale, dtype=float).reshape(-1)
        if arr.size == 1:
            return np.full(self.n_joints, float(arr[0]))
        if arr.size == self.n_joints:
            return arr.copy()
        raise ValueError(
            f"gravity_comp.tau_scale must be a scalar or {self.n_joints} values"
        )

    # ── lifecycle ────────────────────────────────────────────────────────

    def connect(self) -> None:
        try:
            from reBotArm_control_py.actuator import RebotArm
        except ImportError as e:
            raise RuntimeError(
                "reBotArm_control_py not importable; set arm.sdk_path in the "
                "arm profile (and install motorbridge in this venv)"
            ) from e
        hw_yaml = self._cfg.get("hw_yaml", "rebotarm_rs.yaml")
        arm = RebotArm(hw_yaml)
        arm.connect()
        grp = arm.arm
        kp = self._cfg.get("mit_kp")
        kd = self._cfg.get("mit_kd")
        self._mit_kp = np.asarray(kp, dtype=float) if kp is not None else grp._mit_kp.copy()
        self._mit_kd = np.asarray(kd, dtype=float) if kd is not None else grp._mit_kd.copy()
        # SDK-canonical order: set modes on ALL groups first, then enable.
        # JointGroup.enable() is controller-wide (enable_all), so the gripper
        # motor powers up together with the arm and must already be in MIT.
        grp.mode_mit()
        if arm.has_gripper:
            arm.gripper.mode_mit()
        grp.enable()
        self._arm = arm
        self._stopped = False
        self._read_failures = 0
        self._last_q = self._read_positions()
        self._last_cmd_q: np.ndarray | None = None
        if self._gc_enabled:
            from .kinematics import Kinematics

            # Same model the safety/grasp layers use, with joint_signs baked so
            # gravity_torque() returns motor-convention torque directly.
            self._gc_kin = Kinematics(
                model_path=self._cfg.get("model"),
                ee_frame=self._cfg.get("ee_frame", "gripper_end"),
                n_controlled=self.n_joints,
                joint_signs=self._cfg.get("joint_signs"),
            )

    def disconnect(self) -> None:
        if self._arm is not None:
            try:
                self._arm.disconnect()
            finally:
                self._arm = None

    # ── feedback (param reads, the only reliable RS path) ────────────────

    def _motors(self):
        grp = self._arm.arm
        return [grp._mm[name] for name in grp.joint_names]

    def _read_positions(self) -> np.ndarray:
        q = np.zeros(self.n_joints)
        with self._lock:
            for i, m in enumerate(self._motors()[: self.n_joints]):
                q[i] = m.robstride_get_param_f32(MECH_POS)
        return q

    #: consecutive feedback failures tolerated before motion must abort
    MAX_READ_FAILURES = 3

    def get_state(self) -> RobotState:
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
        if self._arm is None or not self._arm.has_gripper:
            return None
        try:
            with self._lock:
                m = self._arm.gripper._mm[self._arm.gripper.joint_names[0]]
                return float(m.robstride_get_param_f32(MECH_POS))
        except Exception:
            return None

    # ── commands ─────────────────────────────────────────────────────────

    def send_joint_target(self, q: np.ndarray) -> None:
        if self._arm is None:
            raise RuntimeError("arm not connected")
        if self._stopped:
            return
        q = np.asarray(q, dtype=float).reshape(-1)[: self.n_joints]
        with self._lock:
            self._send_mit(q)
        self._last_cmd_q = q.copy()

    def _send_mit(self, q: np.ndarray) -> None:
        """Send one arm MIT command, adding the gravity feedforward when
        enabled. The pure-PD path (no gravity comp) omits `tau` entirely so
        the SDK's zeros default applies and the call is byte-identical to the
        pre-feedforward behavior."""
        if self._gc_kin is not None:
            tau = self._gc_kin.gravity_torque(q) * self._gc_tau_scale
            self._arm.arm.send_mit(q, kp=self._mit_kp, kd=self._mit_kd, tau=tau)
        else:
            self._arm.arm.send_mit(q, kp=self._mit_kp, kd=self._mit_kd)

    def set_gripper(self, pos: float, effort: float = 1.0) -> None:
        if self._arm is None or not self._arm.has_gripper or self._stopped:
            return
        kp = self._grip_kp * float(np.clip(effort, 0.05, 1.0))
        self._set_gripper_kp(pos, kp)

    def _set_gripper_kp(self, pos: float, kp: float) -> None:
        """Send an MIT position command with an explicit kp.

        `set_gripper` expresses stiffness as an `effort` scale (0.05..1.0 over
        `_grip_kp`), so the light-hold path passes its target kp (e.g. 1.0)
        straight through here instead of re-deriving an effort fraction -- and
        stays able to hold below the 0.05-effort floor if a profile ever wants
        that.
        """
        if self._arm is None or not self._arm.has_gripper or self._stopped:
            return
        with self._lock:
            self._arm.gripper.send_mit(
                np.array([pos]), kp=np.array([kp]), kd=np.array([self._grip_kd])
            )

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

    def close_gripper_torque(
        self,
        effort: float = 0.6,
        stall_rad: float = 0.03,
        settle_tol: float = 0.1,
        timeout_s: float = 3.0,
        hold_kp: float | None = None,
    ) -> bool:
        """Close the gripper to its mechanical zero and treat the torque the
        MIT controller applies as the "jaws are closed" signal.

        MIT mode sets tau = kp*(target - pos) + kd*vel (the tau feedforward is
        0 here), and mechPos (0x7019) is the one feedback read reliable on this
        firmware (mechVel 0x701A is not rad/s). So a free close decays that
        torque to ~0 as pos -> target, while a jaw that meets the mechanical
        stop or a held object stalls short of target with the position error --
        and therefore the torque -- held high. That sustained torque is the
        closure: the jaw is pressing against something, not air.

        Once contact is detected the jaw does NOT keep pushing at the full
        close kp: it is re-commanded at `hold_kp` (default `gripper.hold_kp`)
        so it holds the object with a light, roughly constant force instead of
        crushing it. Opening (`open_gripper`) is untouched by this. The contact
        position is left in `self._grip_contact_pos` for verification.

        Returns True once the jaw reaches the closed target (nothing held) or
        stalls short of it with residual torque (pressing); False on timeout.
        """
        if self._arm is None or not self._arm.has_gripper or self._stopped:
            return False
        target = self._grip_closed
        self._grip_contact_pos = None
        self.set_gripper(target, effort=effort)
        time.sleep(0.3)  # spin-up grace: a still-moving jaw is not a stall
        last_pos = self._gripper_pos()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            time.sleep(0.15)
            pos = self._gripper_pos()
            if pos is None:
                continue
            if abs(target - pos) < settle_tol:
                return True  # free close reached the target: torque already decayed
            # Short of target and not advancing => the MIT torque kp*(target-pos)
            # is being applied against the stop / a held object (it is high,
            # not the ~0 a free close would hold at the same instant). Closed.
            if last_pos is not None and abs(pos - last_pos) < stall_rad:
                self._hold_light(pos, hold_kp)
                return True
            last_pos = pos
        return False

    def _hold_light(self, contact_pos: float, hold_kp: float | None) -> None:
        """After contact: keep the closed target but drop the stiffness so the
        jaws hold lightly. tau = kp*(target - pos) under MIT, so cutting kp
        (default 6.0 -> `hold_kp`, e.g. 1.0) cuts the holding force by the
        same ratio; the jaw stays put because the object or the mechanical
        stop is still in the way. The reduced command does not affect opening.
        """
        kp = self._grip_hold_kp if hold_kp is None else float(hold_kp)
        self._grip_contact_pos = contact_pos
        self._set_gripper_kp(self._grip_closed, kp)
        print(
            f"[rebot] grip contact at {contact_pos:.3f} rad "
            f"-> holding at kp={kp:.2f} (close kp {self._grip_kp:.2f})"
        )

    def _gripper_vel(self) -> float | None:
        try:
            with self._lock:
                m = self._arm.gripper._mm[self._arm.gripper.joint_names[0]]
                return float(m.robstride_get_param_f32(MECH_VEL))
        except Exception:
            return None

    def open_gripper(self) -> None:
        self.set_gripper(self._grip_open, effort=0.8)

    def stop(self) -> None:
        """Soft stop: freeze at the last commanded pose and refuse new
        commands. Does NOT torque-off (RebotArm.estop == disable_all would
        make a loaded arm free-fall); use hard_estop() for that."""
        self._stopped = True
        if self._arm is not None and self._last_cmd_q is not None:
            try:
                with self._lock:
                    # Re-assert the hold with the same gravity feedforward, so
                    # the frozen pose does not droop once the stream stops.
                    self._send_mit(self._last_cmd_q)
            except Exception:
                pass

    def hard_estop(self) -> None:
        """Torque-off every motor. The arm WILL fall under gravity."""
        self._stopped = True
        if self._arm is not None:
            try:
                with self._lock:
                    self._arm.estop()
            except Exception:
                pass
