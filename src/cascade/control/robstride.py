"""RobStride motor helpers shared by the two reBot RS transports
(`rebot_rs_arm.RebotRSArm` over reBotArm_control_py and
`rebot_rs_mb_arm.RebotRSMotorBridgeArm` over motorbridge directly).

Pure Python over duck-typed motor handles: nothing here imports motorbridge,
so the mock stack and the offline tests never load a CAN runtime. The opt-in
contact-relative squeeze cap for the gripper close (B38,
`close_two_stage_capped`) lives here so both transports share one copy.
"""

from __future__ import annotations

import math
import numbers
import time

#: bounded retries for a late clear_error ACK (see clear_motor_faults)
CLEAR_ERROR_ATTEMPTS = 3
CLEAR_ERROR_RETRY_S = 0.1

# Two-stage close timing/tolerances, the values both RS drivers' close already
# uses (grace after a command, poll period, "stage target reached", minimum
# travel before a stall can count).
GRIP_GRACE_S = 0.3
GRIP_POLL_S = 0.15
GRIP_REACH_RAD = 0.1
GRIP_MIN_TRAVEL_RAD = 0.05
#: mechPos advance per poll below which the jaw counts as stopped: the stall
#: rule `close_gripper_torque` uses (0.03 rad per 150 ms). The squeeze cap
#: never reads mechVel (0x701A), which is not rad/s on this firmware
#: (docs/WRC_CONTROL_PORT.md).
GRIP_STALL_RAD = 0.03
_CAP_KEY = "gripper.max_contact_squeeze_rad"


# ── reading guard for anything that commands a position it has read ──────

#: MIT position range of every RobStride model on this arm (rs-00, rs-06):
#: +-4*pi rad. A reading outside it is not a position, it is a bad frame.
MIT_P_MAX = 4.0 * math.pi
#: A reading this far outside the URDF limits is still plausible: a joint
#: resting on its mechanical end reads a few mrad past it (j2/j3 at -0.0009
#: against 0.0), and the wrist read 3.149 against +-3.14 before re-zeroing.
READ_LIMIT_TOL_RAD = 0.1
#: Startup reads retried this many times before a motor counts as unreadable.
READ_ATTEMPTS = 3


class UnsafeReading(RuntimeError):
    """A position reading that cannot be a position. Whoever sees one stops
    moving: it holds the last COMMANDED (already vetted) pose and never
    commands, ramps from, or reports the reading.

    Rig incident 2026-10-10 (scripts/sign_check_rebot_mb.py): a mechPos
    parameter read returned +2.3e18 mid-probe; the stall guard commanded the
    motor to it, MIT clamped that to the motor's +4*pi end, and the shoulder
    drove at full stiffness until the operator cut power.
    """


def plausible_position(x, lo=None, hi=None, *, tol: float = READ_LIMIT_TOL_RAD) -> bool:
    """True if ``x`` can be a joint position: a real finite number inside the
    motor's MIT range and, when given, inside [lo - tol, hi + tol]."""
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        return False
    x = float(x)
    if not math.isfinite(x) or abs(x) > MIT_P_MAX:
        return False
    if lo is not None and x < float(lo) - tol:
        return False
    return hi is None or x <= float(hi) + tol


def read_plausible(read, lo=None, hi=None, *, attempts: int = READ_ATTEMPTS):
    """Call ``read()`` until it returns a plausible position; None if it never
    does within ``attempts`` (a timeout -> None counts as an attempt too)."""
    for _ in range(max(1, int(attempts))):
        x = read()
        if plausible_position(x, lo, hi):
            return float(x)
    return None


#: Continuity bound for consecutive joint readings: far above anything the
#: arm is commanded to do (the harness caps ~0.8 rad/s) and above what a
#: limp arm falls at, so only a reading that cannot be a measurement trips it.
READ_MAX_JOINT_SPEED = 6.0     # rad/s
READ_JUMP_TOL_RAD = 0.15


def joint_reading_problem(q, last=None, elapsed_s=None, *,
                          max_speed: float = READ_MAX_JOINT_SPEED,
                          tol: float = READ_JUMP_TOL_RAD) -> str | None:
    """Why a vector of joint readings cannot be the arm's position, or None.

    Every value must be plausible_position (finite, inside the motor's
    +-4*pi); with ``last`` (the last GOOD reading) and ``elapsed_s`` since
    it, no joint may have moved farther than max_speed * elapsed + tol.
    """
    vals = [float(x) if isinstance(x, numbers.Real) and not isinstance(x, bool) else x
            for x in q]
    for i, x in enumerate(vals):
        if not plausible_position(x):
            return f"joint {i + 1} read {x!r}, which is not a position"
    if last is not None and elapsed_s is not None:
        allowed = max_speed * max(0.0, float(elapsed_s)) + tol
        for i, (x, y) in enumerate(zip(vals, last)):
            if abs(x - float(y)) > allowed:
                return (f"joint {i + 1} read {x:+.4f} rad, {abs(x - float(y)):.3f} rad from "
                        f"{float(y):+.4f} {float(elapsed_s):.2f} s earlier -- farther than the "
                        f"arm can travel")
    return None


class ProbeEnvelope:
    """Every position a bring-up probe may send to the joint it moves.

    Built from VETTED values only (the start reading that passed
    plausible_position, and the planned delta), so whatever the bus returns
    later, the command stays inside [lo, hi]: a hold at a stall reading is
    clamped into it, a non-finite target raises instead of being sent.
    """

    def __init__(self, lo: float, hi: float):
        lo, hi = float(lo), float(hi)
        if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
            raise ValueError(f"bad probe envelope [{lo}, {hi}]")
        self.lo, self.hi = lo, hi

    @classmethod
    def around(cls, center: float, radius: float) -> ProbeEnvelope:
        r = abs(float(radius))
        return cls(float(center) - r, float(center) + r)

    @classmethod
    def spanning(cls, a: float, b: float) -> ProbeEnvelope:
        return cls(min(float(a), float(b)), max(float(a), float(b)))

    def clamp(self, x: float) -> float:
        x = float(x)
        if not math.isfinite(x):
            raise ValueError(f"refusing to command a non-finite position ({x})")
        return min(max(x, self.lo), self.hi)


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


def contact_squeeze_cap(value) -> float | None:
    """Validate the arm profile's `gripper.max_contact_squeeze_rad` (B38).

    None (the shipped default) = no cap: `close_gripper_two_stage` runs the
    fixed-fraction close unchanged. Otherwise it must be a finite number of
    radians ABOVE `GRIP_REACH_RAD`: the cap leaves a rigidly held jaw `cap`
    short of its target, and a cap inside the reach tolerance would read as
    "target reached, nothing held" and restore the full squeeze. Anything else
    is a config error, raised while the driver is built -- before any jaw
    command can be sent.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{_CAP_KEY} must be a number of radians or null (got {value!r})")
    cap = float(value)
    if not math.isfinite(cap) or cap <= GRIP_REACH_RAD:
        raise ValueError(
            f"{_CAP_KEY} must be finite and greater than the {GRIP_REACH_RAD} rad "
            f"stage-reach tolerance (got {value!r}); null disables the cap")
    return cap


def capped_squeeze_target(stage_target: float, contact: float | None, cap: float,
                          open_pos: float, closed_pos: float) -> float:
    """The stage target, but never more than `cap` past `contact` toward closed.

    Under MIT the steady torque on a stopped jaw is kp*(target - pos), so this
    bounds it at kp*cap and leaves every target already within `cap` of the
    contact exactly as it was. No contact yet = the stage target. Polarity-
    aware like `clamp_to_travel` (the RS jaw closes toward smaller angles).
    """
    if contact is None:
        return float(stage_target)
    if closed_pos < open_pos:
        return float(max(stage_target, contact - cap))
    return float(min(stage_target, contact + cap))


def close_two_stage_capped(driver, stages, cap: float, timeout_s: float) -> float | None:
    """Two-stage gripper close with the contact-relative squeeze cap (B38).

    `driver` is either RS transport (duck-typed: `set_gripper`,
    `_gripper_pos`, `_grip_open`, `_grip_closed`, `_grip_contact_pos`);
    `stages` = ((target_rad, effort), ...) exactly as its fixed close computes
    them. Same grace, poll period, reach tolerance and per-stage deadline as
    that close, with three differences:

    - contact is a mechPos stall: two consecutive polls that advance less than
      `GRIP_STALL_RAD`, after `GRIP_MIN_TRAVEL_RAD` of travel, short of the
      target. mechVel is never read (not rad/s on this firmware);
    - from the first contact on, every jaw target is at most `cap` past that
      contact (`capped_squeeze_target`), so the steady squeeze torque is
      min(today's, kp_eff*cap). The contact is the FIRST stall and is never
      re-anchored, so a deformable object cannot ratchet the jaw further in;
    - a jaw that comes within `GRIP_REACH_RAD` of a capped target met nothing
      there (a stick-slip, or a stall misread in free air): the contact is
      dropped and the stage target restored, so in free air the close never
      ends weaker than the fixed close and the runtime's post-lift air-grasp
      check still sees the jaws closed.

    Every command goes through `driver.set_gripper`, which keeps the travel
    clamp and refuses everything after `stop()`. Returns the final jaw angle
    (None when feedback failed); `driver._grip_contact_pos` keeps the contact.
    """
    open_pos, closed_pos = driver._grip_open, driver._grip_closed
    start_pos = driver._gripper_pos()
    contact = None
    driver._grip_contact_pos = None
    for stage_target, effort in stages:
        target = capped_squeeze_target(stage_target, contact, cap, open_pos, closed_pos)
        driver.set_gripper(target, effort=effort)
        time.sleep(GRIP_GRACE_S)  # spin-up grace: a still-moving jaw is not a stall
        still, last = 0, None
        deadline = time.monotonic() + timeout_s / 2
        while time.monotonic() < deadline:
            time.sleep(GRIP_POLL_S)
            pos = driver._gripper_pos()
            if pos is None:
                continue
            if abs(pos - target) < GRIP_REACH_RAD:
                if target == stage_target:
                    break  # reached this stage's target
                print(f"[rebot] jaw reached the capped target {target:.3f} rad with "
                      f"nothing in the way: no contact, back to {stage_target:.3f} rad")
                contact = driver._grip_contact_pos = None
                target = stage_target
                driver.set_gripper(target, effort=effort)
                still, last = 0, pos
                continue
            traveled = start_pos is not None and abs(pos - start_pos) > GRIP_MIN_TRAVEL_RAD
            if last is not None and abs(pos - last) < GRIP_STALL_RAD and traveled:
                still += 1
            else:
                still = 0
            last = pos
            if still < 2:
                continue
            if contact is None:
                contact = driver._grip_contact_pos = pos
                capped = capped_squeeze_target(stage_target, contact, cap, open_pos, closed_pos)
                print(f"[rebot] grip contact at {contact:.3f} rad (squeeze cap {cap:.2f} rad: "
                      f"target {capped:.3f}, stage target {stage_target:.3f})")
                if capped != target:
                    target = capped
                    driver.set_gripper(target, effort=effort)
                    still = 0
                    continue  # watch the capped hold: a free jaw would run on to it
            break  # stopped on the object: next stage
    return driver._gripper_pos()
