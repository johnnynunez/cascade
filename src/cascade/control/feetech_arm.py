"""SO-101 (and any Feetech STS-servo arm) over a USB serial bus.

!!! NOT VERIFIED ON HARDWARE. The protocol layer in `feetech.py` is unit-tested
against a fake port, but nobody has run THIS file against servos. The register
map, the count<->radian mapping and every `wire_signs` entry are derived from
published documentation and the vendored URDF, not measured. Before the first
motion, use the read-only diagnostic:

    python scripts/diag_so101.py --port /dev/ttyACM0            # read only
    python scripts/diag_so101.py --port /dev/ttyACM0 --jog 1    # one joint

The jog step exists because a wrong `wire_signs` entry drives a joint the wrong
way on the first command, and printed PLA links reach a hard stop before the
servo gives up.

DESIGN NOTES

Commands go out as one SYNC_WRITE frame per waypoint. Reads are per-servo, so
`get_state()` costs N round trips; at USB latency that caps the feedback rate
near 150 Hz on a 5-joint arm, comfortably above the 50 Hz stream but not
unlimited -- do not add per-waypoint reads to this path.

The gripper is a separate servo on the same bus, addressed exactly like a joint
but excluded from `q` (framework convention: `ArmBase.get_state().gripper_pos`).

`stop()` is a SOFT stop: it re-commands the last position with torque still on,
so nothing falls. `hard_estop()` cuts torque and the arm WILL drop -- it exists
for the case where a stuck servo is worse than a fall.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from ..config import Cfg
from ..types import RobotState
from .arm_base import ArmBase
from .feetech import (
    ADDR_ACCELERATION,
    ADDR_GOAL_POSITION,
    ADDR_MODE,
    ADDR_PRESENT_POSITION,
    ADDR_PRESENT_TEMPERATURE,
    ADDR_TORQUE_ENABLE,
    ADDR_TORQUE_LIMIT,
    MODE_POSITION,
    SCS,
    STS,
    TORQUE_LIMIT_MAX,
    FeetechError,
    ServoBus,
)

logger = logging.getLogger(__name__)

#: Consecutive read failures on one joint before we call the arm broken. A
#: single dropped frame on a shared TTL bus is normal; three in a row is not.
_MAX_READ_FAILURES = 3


class FeetechArm(ArmBase):
    def __init__(self, cfg: Cfg):
        self._cfg = cfg
        self.n_joints = int(cfg.get("n_joints", 5))
        self._port_name = str(cfg.get("port", "/dev/ttyACM0"))
        self._baud = int(cfg.get("baudrate", 1_000_000))
        self._ids = [int(i) for i in cfg.get("servo_ids", [])]
        self._grip_id = cfg.get("gripper_id")
        self._grip_id = int(self._grip_id) if self._grip_id is not None else None
        self._timeout_s = float(cfg.get("read_timeout_ms", 100)) / 1000.0
        self._counts_per_rev = float(cfg.get("counts_per_rev", 4096))
        self._center = float(cfg.get("center_counts", 2048))
        self._signs = np.asarray(cfg.get("wire_signs") or [1] * self.n_joints, dtype=float)
        self._torque_limit = float(cfg.get("torque_limit", 0.5))
        self._acceleration = int(np.clip(int(cfg.get("servo_acceleration", 32)), 0, 254))
        self._endian = SCS if str(cfg.get("servo_series", "sts")).lower() == "scs" else STS
        g = cfg.get("gripper") or {}
        self._grip_open = float(g.get("open_pos", 0.0))
        self._grip_closed = float(g.get("closed_pos", 1.0))
        self._grip_sign = float(g.get("wire_sign", 1))

        if len(self._ids) != self.n_joints:
            raise ValueError(
                f"servo_ids has {len(self._ids)} entries but n_joints={self.n_joints}"
            )
        if self._signs.size != self.n_joints:
            raise ValueError(
                f"wire_signs has {self._signs.size} entries but n_joints={self.n_joints}"
            )

        self._port = None
        self._bus: ServoBus | None = None
        self._connected = False
        self._stopped = False
        self._last_cmd_q: np.ndarray | None = None
        self._read_failures = 0
        self._gripper_pos = self._grip_open
        self._gripper_valid = True
        # The bus is one shared half-duplex wire. Two threads interleaving a
        # request and a response turn both into checksum errors.
        self._lock = threading.RLock()

    # ── unit conversion ──────────────────────────────────────────────────

    def _counts_to_rad(self, counts: int, sign: float) -> float:
        return float(sign * (counts - self._center) * 2.0 * np.pi / self._counts_per_rev)

    def _rad_to_counts(self, rad: float, sign: float) -> int:
        counts = self._center + sign * float(rad) * self._counts_per_rev / (2.0 * np.pi)
        # A commanded value outside the encoder range wraps rather than
        # saturating on this bus, i.e. asking for just past the limit sends the
        # joint to the OPPOSITE end. Clamp instead.
        return int(round(min(max(counts, 0.0), self._counts_per_rev - 1.0)))

    # ── lifecycle ────────────────────────────────────────────────────────

    def connect(self) -> None:
        if self._connected:
            return
        try:
            import serial  # lazy: pyserial is an optional extra
        except ImportError as e:
            raise RuntimeError(
                "the so101/feetech backend needs pyserial "
                "(pip install 'cascade[arm-feetech]' or pip install pyserial)"
            ) from e
        try:
            self._port = serial.Serial(
                self._port_name, self._baud, timeout=self._timeout_s,
                write_timeout=self._timeout_s,
            )
        except Exception as e:  # noqa: BLE001 - surfaces as a hardware fault
            raise RuntimeError(f"cannot open {self._port_name}: {e}") from e
        self._bus = ServoBus(self._port, endian=self._endian)

        all_ids = list(self._ids) + ([self._grip_id] if self._grip_id else [])
        missing = [sid for sid in all_ids if not self._bus.ping(sid)]
        if missing:
            self._close_port()
            raise RuntimeError(
                f"no response from servo id(s) {missing} on {self._port_name}. "
                f"Check the bus power (the servos need their own supply, not "
                f"USB), the baud rate ({self._baud}), and that no other process "
                f"(LeRobot, the Feetech debug tool) holds the port."
            )

        with self._lock:
            for sid in all_ids:
                self._bus.write_reg(sid, ADDR_MODE, MODE_POSITION)
                self._bus.write_reg(
                    sid, ADDR_TORQUE_LIMIT,
                    int(np.clip(self._torque_limit, 0.0, 1.0) * TORQUE_LIMIT_MAX),
                )
                # Non-zero acceleration: with 0 the servo steps to the goal as
                # fast as it can, which on a streamed trajectory turns every
                # 20 ms waypoint into a jolt. The framework's min-jerk profile
                # already shapes the path; this just stops the servo from
                # un-shaping it.
                self._bus.write_reg(sid, ADDR_ACCELERATION, self._acceleration)
                self._bus.write_reg(sid, ADDR_TORQUE_ENABLE, 1)

        self._connected = True
        self._stopped = False
        state = self.get_state()
        self._last_cmd_q = state.q.copy()
        temps = self._temperatures()
        logger.info(
            "so101/feetech up on %s: q=%s, servo temps=%s C",
            self._port_name, np.round(state.q, 3).tolist(), temps,
        )
        hot = {sid: t for sid, t in temps.items() if t >= 55}
        if hot:
            logger.warning("servo(s) already hot before any motion: %s", hot)

    def _close_port(self) -> None:
        if self._port is not None:
            try:
                self._port.close()
            except Exception:  # noqa: BLE001
                pass
        self._port = None
        self._bus = None

    def disconnect(self) -> None:
        """Torque OFF. The arm is not self-supporting -- park it first.

        `skill_move_home` before disconnecting, exactly as with the reBot: a
        loaded arm drops when its servos let go.
        """
        if self._bus is not None and self._connected:
            with self._lock:
                for sid in list(self._ids) + ([self._grip_id] if self._grip_id else []):
                    try:
                        self._bus.write_reg(sid, ADDR_TORQUE_ENABLE, 0)
                    except FeetechError as e:
                        logger.warning("could not disable torque on id %d: %s", sid, e)
        self._close_port()
        self._connected = False

    # ── feedback ─────────────────────────────────────────────────────────

    def _temperatures(self) -> dict[int, int]:
        out: dict[int, int] = {}
        with self._lock:
            for sid in list(self._ids) + ([self._grip_id] if self._grip_id else []):
                try:
                    out[sid] = int(self._bus.read_reg(sid, ADDR_PRESENT_TEMPERATURE))
                except FeetechError:
                    pass
        return out

    def get_state(self) -> RobotState:
        if not self._connected or self._bus is None:
            raise RuntimeError("so101 arm not connected")
        q = np.zeros(self.n_joints)
        failed: list[int] = []
        with self._lock:
            for i, (sid, sign) in enumerate(zip(self._ids, self._signs)):
                try:
                    q[i] = self._counts_to_rad(
                        self._bus.read_reg(sid, ADDR_PRESENT_POSITION), sign)
                except FeetechError:
                    failed.append(sid)
                    # Last command is a better guess than zero: zero is a real
                    # pose the arm could be nowhere near, and the safety
                    # harness would compare against it.
                    q[i] = (self._last_cmd_q[i]
                            if self._last_cmd_q is not None else 0.0)
            if self._grip_id is not None:
                try:
                    self._gripper_pos = self._counts_to_rad(
                        self._bus.read_reg(self._grip_id, ADDR_PRESENT_POSITION),
                        self._grip_sign)
                    self._gripper_valid = True
                except FeetechError:
                    # Reported as invalid rather than stale: the grasp
                    # verification path treats unknown jaw width as "cannot
                    # confirm" instead of assuming an air grasp.
                    self._gripper_valid = False

        if failed:
            self._read_failures += 1
            if self._read_failures >= _MAX_READ_FAILURES:
                raise RuntimeError(
                    f"lost feedback from servo id(s) {failed} on "
                    f"{self._port_name} ({self._read_failures} consecutive "
                    f"failures) -- check the bus wiring and power"
                )
            logger.warning("joint read failed for id(s) %s (%d/%d)",
                           failed, self._read_failures, _MAX_READ_FAILURES)
        else:
            self._read_failures = 0

        return RobotState(
            q=q, gripper_pos=self._gripper_pos,
            gripper_valid=self._gripper_valid, t=time.monotonic(),
        )

    # ── commands ─────────────────────────────────────────────────────────

    def send_joint_target(self, q: np.ndarray) -> None:
        if not self._connected or self._bus is None:
            raise RuntimeError("so101 arm not connected")
        if self._stopped:
            return
        q = np.asarray(q, dtype=float).reshape(-1)[: self.n_joints]
        values = {
            sid: self._rad_to_counts(qi, sign)
            for sid, qi, sign in zip(self._ids, q, self._signs)
        }
        with self._lock:
            self._bus.sync_write_reg(ADDR_GOAL_POSITION, values)
        self._last_cmd_q = q.copy()

    def set_gripper(self, pos: float, effort: float = 1.0) -> None:
        if self._grip_id is None or self._bus is None:
            return
        if self._stopped:
            return
        with self._lock:
            # Effort scales the torque limit so `close_gripper` on a fragile
            # object squeezes less. Floored well above zero: a servo at ~0
            # torque limit does not hold what it has already picked up.
            self._bus.write_reg(
                self._grip_id, ADDR_TORQUE_LIMIT,
                int(np.clip(effort, 0.15, 1.0) * self._torque_limit * TORQUE_LIMIT_MAX),
            )
            self._bus.write_reg(
                self._grip_id, ADDR_GOAL_POSITION,
                self._rad_to_counts(float(pos), self._grip_sign),
            )

    def stop(self) -> None:
        """Soft stop: hold the last commanded pose, torque ON."""
        self._stopped = True
        if self._connected and self._bus is not None and self._last_cmd_q is not None:
            try:
                values = {
                    sid: self._rad_to_counts(qi, sign)
                    for sid, qi, sign in zip(self._ids, self._last_cmd_q, self._signs)
                }
                with self._lock:
                    self._bus.sync_write_reg(ADDR_GOAL_POSITION, values)
            except FeetechError as e:
                logger.error("soft stop could not re-assert the hold pose: %s", e)

    def resume(self) -> None:
        self._stopped = False

    def hard_estop(self) -> None:
        """Torque OFF on every servo. THE ARM WILL FALL. Last resort only."""
        self._stopped = True
        if self._bus is None:
            return
        with self._lock:
            for sid in list(self._ids) + ([self._grip_id] if self._grip_id else []):
                try:
                    self._bus.write_reg(sid, ADDR_TORQUE_ENABLE, 0)
                except FeetechError:
                    pass
