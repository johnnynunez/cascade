"""Unitree humanoid ARMS over the official Arm SDK topics (`type: unitree_arm`).

Unitree's own ROS2 stack (unitree_ros2) does NOT ship ros2_control for the
humanoids -- there is no JointTrajectory controller to point `type: ros2` at.
What the robots expose, over CycloneDDS with ROS2-compatible message types
(`unitree_hg/msg/LowCmd`, `unitree_hg/msg/LowState` for G1 / H1-2 / H2;
`unitree_go/msg/*` for the gen-1 H1), is the **Arm SDK** channel:

    publish   rt/arm_sdk   LowCmd    per-motor q / dq / tau / kp / kd
    subscribe rt/lowstate  LowState  per-motor q / dq / tau_est

Semantics, verified against unitree_sdk2's `g1_arm7_sdk_dds_example.cpp`,
`h1_2_arm_sdk_dds_example.cpp`, `h1_arm_sdk_dds_example.cpp` and
`h2_arm_sdk_dds_example.cpp` (all at unitree_sdk2 main 9754cd15, 2026-09):

- Motor indices are FIXED per robot family (the enum in each example);
  `unitree_joint_index` in the profile lists them in this framework's q
  order. There is no name lookup on the wire, so the profile is the mapping.
- One extra "weight" motor slot (G1: 29, H1-2: 27, H1: 9, H2: 31) carries
  the arm-SDK authority 0..1 in its `q` field. The locomotion controller
  blends `weight * arm_sdk_cmd + (1-weight) * its own arm command`, so the
  SDK ramps it UP over ~2 s at connect and DOWN over ~2 s at disconnect;
  slamming it is what makes a standing humanoid jerk its shoulders.
- Every joint command carries kp/kd (per-joint tables in the profile; the
  examples use kp 60 / kd 1.5 on G1 and a 120/80/50 shoulder/elbow/wrist
  ladder on H1-2). A zero-kp command means "no torque", never "hold".
- The `hg` LowCmd has TWO leading mode bytes (`mode_pr`, `mode_machine`)
  and a CRC32 over the raw C struct that the firmware validates and
  silently DROPS on mismatch. The CRC is computed here in Python exactly as
  unitree_ros2's `motor_crc_hg.cpp` does (struct layout replicated byte for
  byte); `mode_machine` must be copied from the incoming LowState.

WHAT THIS DOES AND DOES NOT DO. cascade drives ONE arm of the humanoid for
tabletop skills; balance, legs and the waist stay with Unitree's own
controller (the robot must already be standing/damped in a mode that
accepts arm_sdk). Stopping (`stop()`) holds the current joint targets with
the profile's kp/kd -- it does NOT drop the weight, because handing the arm
back to the locomotion controller mid-motion is itself a large motion.

Heavy imports (rclpy + the unitree_hg / unitree_go message packages from
unitree_ros2's cyclonedds_ws) stay inside connect(), so the mock stack
imports nothing and CI without ROS2 still collects this module; the unit
tests inject stub modules into sys.modules (same convention as ros2_arm).

!!! UNVERIFIED ON HARDWARE. Wire format and semantics come from Unitree's
published examples, not from a robot in this room.
"""

from __future__ import annotations

import logging
import struct
import threading
import time

import numpy as np

from ..config import Cfg
from ..types import RobotState
from .arm_base import ArmBase

logger = logging.getLogger(__name__)

#: Motor counts and weight-slot index per message family / robot, from the
#: unitree_sdk2 examples. `hg` = G1 / H1-2 / H2 (LowCmd has 35 motor slots);
#: `go` = gen-1 H1 (20 slots, the Go2-era message).
_FAMILY_MOTORS = {"hg": 35, "go": 20}


def crc32_core(words: list[int]) -> int:
    """Unitree's CRC32 (poly 0x04c11db7, init 0xFFFFFFFF, no final xor) over
    32-bit words -- `crc32_core()` in unitree_ros2 `motor_crc*.cpp`."""
    crc = 0xFFFFFFFF
    poly = 0x04C11DB7
    for data in words:
        xbit = 1 << 31
        for _ in range(32):
            if crc & 0x80000000:
                crc = ((crc << 1) & 0xFFFFFFFF) ^ poly
            else:
                crc = (crc << 1) & 0xFFFFFFFF
            if data & xbit:
                crc ^= poly
            xbit >>= 1
    return crc & 0xFFFFFFFF


def hg_lowcmd_crc(mode_pr: int, mode_machine: int, motors: list[dict],
                  reserve: tuple[int, int, int, int] = (0, 0, 0, 0)) -> int:
    """CRC of a `unitree_hg` LowCmd, byte-identical to motor_crc_hg.cpp:

        struct LowCmd { uint8 modePr; uint8 modeMachine; MotorCmd motorCmd[35];
                        uint32 reserve[4]; uint32 crc; }
        struct MotorCmd { uint8 mode; float q, dq, tau, Kp, Kd; uint32 reserve; }

    Natural C alignment: MotorCmd is 28 bytes (mode padded to 4), LowCmd
    starts with 2 bytes + 2 padding, and the CRC covers every 32-bit word
    except the last (the crc field itself)."""
    buf = bytearray()
    buf += struct.pack("<BBxx", mode_pr & 0xFF, mode_machine & 0xFF)
    for m in motors:
        buf += struct.pack("<Bxxx", int(m.get("mode", 0)) & 0xFF)
        buf += struct.pack("<fffff", float(m.get("q", 0.0)), float(m.get("dq", 0.0)),
                           float(m.get("tau", 0.0)), float(m.get("kp", 0.0)),
                           float(m.get("kd", 0.0)))
        buf += struct.pack("<I", int(m.get("reserve", 0)) & 0xFFFFFFFF)
    buf += struct.pack("<4I", *[int(r) & 0xFFFFFFFF for r in reserve])
    assert len(buf) == 4 + 28 * len(motors) + 16, len(buf)
    words = list(struct.unpack(f"<{len(buf)//4}I", bytes(buf)))
    return crc32_core(words)


class UnitreeArm(ArmBase):
    def __init__(self, cfg: Cfg):
        self._cfg = cfg
        self.n_joints = int(cfg.get("n_joints", 7))
        self.settle_tol = float(cfg.get("settle_tol", 0.04))
        self.settle_timeout_s = float(cfg.get("settle_timeout_s", 4.0))

        self._family = str(cfg.get("unitree_msg_family", "hg")).lower()
        if self._family not in _FAMILY_MOTORS:
            raise ValueError(f"unitree_msg_family must be hg|go, got {self._family!r}")
        self._n_motors = _FAMILY_MOTORS[self._family]
        idx = cfg.get("unitree_joint_index")
        if not idx or len(idx) != self.n_joints:
            raise ValueError(
                f"unitree_joint_index must list {self.n_joints} motor indices in "
                f"this profile's q order (the Arm SDK addresses motors by INDEX)"
            )
        self._idx = [int(i) for i in idx]
        if any(i < 0 or i >= self._n_motors for i in self._idx):
            raise ValueError(f"unitree_joint_index out of range for {self._family} "
                             f"({self._n_motors} motors): {self._idx}")
        self._weight_idx = int(cfg.get("unitree_weight_index", 29))
        kp = cfg.get("unitree_kp"); kd = cfg.get("unitree_kd")
        self._kp = [float(v) for v in (kp if isinstance(kp, (list, tuple)) else [kp or 60.0] * self.n_joints)]
        self._kd = [float(v) for v in (kd if isinstance(kd, (list, tuple)) else [kd or 1.5] * self.n_joints)]
        if len(self._kp) != self.n_joints or len(self._kd) != self.n_joints:
            raise ValueError("unitree_kp / unitree_kd must be scalars or n_joints-long lists")
        #: seconds to ramp the arm-SDK weight 0 -> 1 (and back). The SDK
        #: examples use 0.2/s, i.e. 5 s; 2 s is their init sweep length.
        self._weight_ramp_s = float(cfg.get("unitree_weight_ramp_s", 2.0))
        self._state_timeout = float(cfg.get("state_timeout_s", 1.0))
        self._first_state_timeout = float(cfg.get("wait_state_s", 5.0))
        self._cmd_topic = str(cfg.get("unitree_cmd_topic", "rt/arm_sdk"))
        self._state_topic = str(cfg.get("unitree_state_topic", "rt/lowstate"))
        #: LowState publishes at 500 Hz on the real robot; we only need to
        #: keep the latest sample, so a depth-1 best-effort QoS is right.

        self._node = None
        self._executor = None
        self._spin_thread: threading.Thread | None = None
        self._pub = None
        self._msg_mod = None
        self.connected = False
        self._stopped = False
        self._lock = threading.Lock()
        self._q = np.zeros(self._n_motors)
        self._dq = np.zeros(self._n_motors)
        self._tau = np.zeros(self._n_motors)
        self._mode_machine = 0
        self._last_state_t: float | None = None
        self._weight = 0.0
        self._last_cmd_q: np.ndarray | None = None

    # ── lifecycle ────────────────────────────────────────────────────────

    def connect(self) -> None:
        if self.connected:
            return
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from rclpy.qos import qos_profile_sensor_data
        except ImportError as e:
            raise RuntimeError(
                "the unitree_arm backend needs rclpy (source a ROS2 setup.bash)"
            ) from e
        try:
            if self._family == "hg":
                from unitree_hg.msg import LowCmd, LowState, MotorCmd  # type: ignore
            else:
                from unitree_go.msg import LowCmd, LowState, MotorCmd  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                f"unitree_{self._family} message package not found: build and source "
                "unitree_ros2's cyclonedds_ws (github.com/unitreerobotics/unitree_ros2)"
            ) from e
        self._msg_mod = (LowCmd, LowState, MotorCmd)

        if not rclpy.ok():
            rclpy.init()  # never paired with shutdown(): the context is process-global
        name = str(self._cfg.get("name", "unitree_arm")).replace("-", "_")
        self._node = Node(f"cascade_{name}")
        self._node.create_subscription(LowState, self._state_topic, self._on_low_state,
                                       qos_profile_sensor_data)
        self._pub = self._node.create_publisher(LowCmd, self._cmd_topic, 10)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin,
                                             name=f"unitree-spin-{name}", daemon=True)
        self._spin_thread.start()

        deadline = time.monotonic() + self._first_state_timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._last_state_t is not None:
                    break
            time.sleep(0.02)
        else:
            self.disconnect()
            raise RuntimeError(
                f"no LowState on {self._state_topic} within {self._first_state_timeout:.0f}s "
                "-- is the robot on, is CYCLONEDDS_URI pointing at its interface, "
                "and are the unitree_ros2 messages sourced?"
            )
        self.connected = True
        self._stopped = False
        q0 = self.get_state().q.copy()
        self._last_cmd_q = q0
        # Ramp the arm-SDK weight in while HOLDING the current pose, so the
        # locomotion controller hands the arm over without a step.
        self._ramp_weight(1.0, hold_q=q0)
        logger.info("unitree arm up: %s family=%s weight=%.1f q=%s", self._cmd_topic,
                    self._family, self._weight, np.round(q0, 3).tolist())

    def disconnect(self) -> None:
        if self.connected and self._pub is not None:
            try:
                # Hand the arm back gently: weight -> 0 over the ramp time,
                # holding the last commanded pose meanwhile.
                self._ramp_weight(0.0, hold_q=self._last_cmd_q)
            except Exception as e:  # noqa: BLE001
                logger.warning("unitree weight ramp-down failed: %s", e)
        if self._executor is not None:
            try:
                self._executor.shutdown(timeout_sec=1.0)
            except Exception:  # noqa: BLE001
                pass
        if self._spin_thread is not None and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:  # noqa: BLE001
                pass
        self._node = self._executor = self._spin_thread = self._pub = None
        self.connected = False

    # ── feedback ─────────────────────────────────────────────────────────

    def _on_low_state(self, msg) -> None:
        with self._lock:
            ms = msg.motor_state
            n = min(len(ms), self._n_motors)
            for i in range(n):
                self._q[i] = float(ms[i].q)
                self._dq[i] = float(ms[i].dq)
                self._tau[i] = float(getattr(ms[i], "tau_est", 0.0))
            mm = getattr(msg, "mode_machine", None)
            if mm is not None:
                self._mode_machine = int(mm)
            self._last_state_t = time.monotonic()

    def get_state(self) -> RobotState:
        if not self.connected and self._last_state_t is None:
            raise RuntimeError("unitree arm not connected")
        with self._lock:
            last_t = self._last_state_t
            if last_t is None or (time.monotonic() - last_t) > self._state_timeout:
                age = "never" if last_t is None else f"{time.monotonic() - last_t:.1f}s"
                raise RuntimeError(f"LowState on {self._state_topic} is stale ({age})")
            q = self._q[self._idx].copy()
            dq = self._dq[self._idx].copy()
            tau = self._tau[self._idx].copy()
            # No hand on the wire: gripper_valid=False tells the skill layer
            # the jaw state is unknown rather than "open".
            return RobotState(q=q, dq=dq, tau=tau, gripper_pos=0.0,
                              gripper_valid=False, t=last_t)

    # ── commands ─────────────────────────────────────────────────────────

    def _build_cmd(self, q: np.ndarray | None, weight: float):
        LowCmd, _LowState, MotorCmd = self._msg_mod
        msg = LowCmd()
        motors = []
        cmd_list = list(msg.motor_cmd)
        for i in range(self._n_motors):
            motors.append({"mode": 0, "q": 0.0, "dq": 0.0, "tau": 0.0, "kp": 0.0, "kd": 0.0})
        if q is not None:
            for j, mi in enumerate(self._idx):
                motors[mi] = {"mode": 1, "q": float(q[j]), "dq": 0.0, "tau": 0.0,
                              "kp": self._kp[j], "kd": self._kd[j]}
        motors[self._weight_idx]["q"] = float(np.clip(weight, 0.0, 1.0))
        for i, m in enumerate(motors):
            mc = cmd_list[i] if i < len(cmd_list) else MotorCmd()
            mc.mode = int(m["mode"]); mc.q = float(m["q"]); mc.dq = float(m["dq"])
            mc.tau = float(m["tau"]); mc.kp = float(m["kp"]); mc.kd = float(m["kd"])
            if i < len(cmd_list):
                cmd_list[i] = mc
            else:
                cmd_list.append(mc)
        try:
            msg.motor_cmd = cmd_list
        except Exception:  # fixed-size arrays in some bindings
            for i, mc in enumerate(cmd_list[: len(msg.motor_cmd)]):
                msg.motor_cmd[i] = mc
        if self._family == "hg":
            msg.mode_pr = 0
            msg.mode_machine = int(self._mode_machine)
            msg.crc = hg_lowcmd_crc(0, self._mode_machine, motors)
        return msg

    def _publish(self, q: np.ndarray | None, weight: float) -> None:
        if self._pub is None:
            return
        self._pub.publish(self._build_cmd(q, weight))

    def _ramp_weight(self, target: float, hold_q: np.ndarray | None) -> None:
        steps = max(1, int(self._weight_ramp_s / 0.02))
        start = self._weight
        for i in range(1, steps + 1):
            self._weight = start + (target - start) * i / steps
            self._publish(hold_q, self._weight)
            time.sleep(0.02)
        self._weight = float(target)

    def send_joint_target(self, q: np.ndarray) -> None:
        if not self.connected:
            raise RuntimeError("unitree arm not connected")
        if self._stopped:
            return
        q = np.asarray(q, dtype=float).reshape(-1)[: self.n_joints]
        self._publish(q, self._weight)
        self._last_cmd_q = q.copy()

    def set_gripper(self, pos: float, effort: float = 1.0) -> None:
        # Hands (Dex3 / Inspire) are separate DDS services with their own
        # topics; the base humanoid profiles declare max_width_m = 0 so the
        # grasp planner never asks. A hand profile plugs in here.
        return

    def stop(self) -> None:
        """Soft stop: re-assert the LAST commanded pose with the profile's
        kp/kd (a hold), weight unchanged. Dropping the weight would hand the
        arm to the locomotion controller mid-motion, which is a motion."""
        self._stopped = True
        if not self.connected:
            return
        try:
            self._publish(self._last_cmd_q, self._weight)
        except Exception as e:  # noqa: BLE001 - stop must never raise
            logger.error("unitree soft stop failed to publish: %s", e)

    def resume(self) -> None:
        self._stopped = False
