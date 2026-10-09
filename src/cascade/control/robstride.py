"""RobStride motor helpers shared by the two reBot RS transports
(`rebot_rs_arm.RebotRSArm` over reBotArm_control_py and
`rebot_rs_mb_arm.RebotRSMotorBridgeArm` over motorbridge directly).

Pure Python over duck-typed motor handles: nothing here imports motorbridge,
so the mock stack and the offline tests never load a CAN runtime.
"""

from __future__ import annotations

import time

#: bounded retries for a late clear_error ACK (see clear_motor_faults)
CLEAR_ERROR_ATTEMPTS = 3
CLEAR_ERROR_RETRY_S = 0.1


def clamp_to_travel(pos: float, open_pos: float, closed_pos: float) -> float:
    """Clamp a gripper position target into the profile's measured travel.

    Ported from Seeed WRC `control/gripper.py` (WrcGripper._send_gripper_mit,
    from rebot_grasp's GraspDriver), which clips every target to the open
    soft limit. The RS jaw runs 0.0 (closed) -> +6.39 rad (open hard end);
    `open_pos` keeps margin off that end, and a target past either end drives
    the jaw into a hard stop at full MIT stiffness. Polarity-aware: the
    interval is [min, max] of the two profile values.
    """
    lo, hi = min(open_pos, closed_pos), max(open_pos, closed_pos)
    return float(min(max(float(pos), lo), hi))


def clear_motor_faults(motors, *, attempts: int = CLEAR_ERROR_ATTEMPTS,
                       retry_s: float = CLEAR_ERROR_RETRY_S) -> None:
    """Clear latched fault bits on every motor, or raise before enable.

    Rig finding (Seeed WRC fork, docs/HARDWARE_VERIFICATION_HANDOVER.md
    Finding #1; fix ported from WRC `control/__init__.py::make_arm`): RobStride
    motors KEEP their fault bits across sessions (`fault_raw=0x00000004`,
    status_code=1 on every motor after an earlier session faulted them), and
    the SDK's `enable_all()` does not clear them. A faulted motor then
    silently refuses every MIT command -- the stream "runs" and nothing moves.
    The official motorbridge CLI drove joint 1 normally right after a
    `clear_error`, which is what isolated the cause.

    In the RobStride private protocol the clear is the type-4 STOP frame with
    the clear-fault flag, so it also leaves the motor in reset (no torque).
    Callers must therefore issue it while the motors are still disabled --
    after the run mode is set and strictly before enable -- never on a
    holding arm.

    `motors` is an iterable of `(name, motor)` pairs. Some RobStride firmwares
    ACK the clear late right after power-up or a bus-off (WRC observed
    `comm_type=4` ack timeouts), so an ack timeout is retried `attempts`
    times `retry_s` apart. Anything else -- or a timeout that persists -- is a
    hardware fault: raise a plain RuntimeError (the error taxonomy's
    crash/hardware class) so connect() aborts before torque is enabled,
    rather than handing the agent an arm that ignores its commands.
    """
    for name, motor in motors:
        for attempt in range(attempts):
            try:
                motor.clear_error()
                break
            except Exception as e:  # noqa: BLE001 -- vendor CallError, re-raised below
                transient = "ack timeout" in str(e).lower()
                if transient and attempt + 1 < attempts:
                    time.sleep(retry_s)
                    continue
                raise RuntimeError(
                    f"RobStride motor {name!r}: clear_error failed after "
                    f"{attempt + 1} attempt(s) ({e}); refusing to enable a motor "
                    "that may still hold a latched fault"
                ) from e
