"""Guided single-joint jog over motorbridge: phase-2 bring-up for a RS backend.

Phase 1 (`scripts/diag_rebot_mb.py`) established that all seven motors answer
and report `mechPos` by parameter read. This script answers what reads alone
cannot: does a commanded MIT delta move `mechPos` the same way, and does the
joint move the direction the URDF predicts? The second half is what
`joint_signs` encodes, and it needs a human watching the arm.

WHY A JOG AND NOT HAND-MOVING: the delta is known and bounded, and the same
MIT write path the real backend will use gets exercised. Hand-moving proves
nothing about the command path.

SAFETY MODEL -- read this before running:

- The arm is HELD, not jogged loose. Every enabled joint is commanded to its
  own current position first, so the arm cannot sag while one joint moves.
  Disabling motors is never part of the happy path: `disable` is a torque-off
  and a loaded arm free-falls.
- Ctrl+C re-commands the hold pose instead of cutting torque, matching
  RebotRSArm.stop()'s soft-stop contract. Use --disable-after only with the
  arm mechanically supported.
- `--delta` and `--kp` are hard-capped (MAX_DELTA_RAD / MAX_DELTA_YAW_RAD and
  MAX_KP) because the RobStride MIT gain scale is NOT yet characterized on this
  build, so an order-of-magnitude typo must not be expressible.
- Gains default to the JOGGED JOINT's own values from
  configs/arms/rebot_rs_mb.yaml, not to a flat number. An earlier version used
  kp=8 for every joint and reported joints 3-5 as "blocked" when they were
  merely being driven at a fraction of their configured gain -- a validation
  run that does not use production's gains certifies nothing about production.
  Override with --kp/--kd only to explore; fix the profile for anything real.
- Run from a pose that is mechanically stable (the calibrated zero is), with
  the arm supported and nobody within reach.

The bus is exclusive (host id 0xFD): stop motorbridge-gateway and any
LeRobot process first.

Usage:

    python scripts/jog_rebot_mb.py --joint 1 --dry-run     # print the plan only
    python scripts/jog_rebot_mb.py --joint 1               # confirm, then jog
    python scripts/jog_rebot_mb.py --joint 2 --delta -0.05
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

MECH_POS = 0x7019
MECH_VEL = 0x701A
RUN_MODE = 0x7005

ARM_IDS = (1, 2, 3, 4, 5, 6)
GRIPPER_ID = 7
HOST_ID = 0xFD

#: Hard caps. These are deliberately small: the MIT gain scale for this build
#: is uncharacterized, so an order-of-magnitude mistake in --kp must not be
#: expressible. Raise them only after gains are validated on the rig.
MAX_DELTA_RAD = 0.15
MAX_KP = 30.0
MAX_KD = 3.0

#: Base yaw (motor 1) rotates about the vertical axis, so gravity does no work
#: on it at any pose and a large delta cannot make the arm sag or run away --
#: unlike j2/j3, where the same delta swings the whole payload against gravity.
#: Measured on the rig 2026-08-27 at kp=8/kd=0.5: sag 0.9 mrad while holding,
#: and a commanded +0.100 rad produced +0.0947 rad of mechPos (95% tracking).
#: The exemption exists because reading the SIGN of the rotation by eye needs
#: tens of degrees, not the 8.6 deg the global cap allows; j1's own travel is
#: +-2.8 rad, so this stays far inside the joint limit.
MAX_DELTA_YAW_RAD = 0.6
YAW_ID = 1


def _delta_cap(motor_id: int) -> float:
    return MAX_DELTA_YAW_RAD if motor_id == YAW_ID else MAX_DELTA_RAD


def _read(motor, param: int, timeout_ms: int = 300) -> float | None:
    try:
        return float(motor.robstride_get_param_f32(param, timeout_ms))
    except Exception:
        return None


def _profile_gains(motor_id: int) -> tuple[float, float] | None:
    """The profile's own MIT gains for this motor, or None if unavailable.

    Jogging every joint with one flat kp is actively misleading. On the rig
    2026-08-27 a uniform kp=8 moved joint 4 by 0.006 rad of a commanded 0.100
    and joints 3 and 5 by 0.015, which the script reported as "blocked" when it
    only meant the gain was far below the 12/30/10 the profile declares. Worse,
    a near-zero response leaves the joint's SIGN unmeasured: 0.35 deg of travel
    can be compliance and backlash rather than tracking, and it is invisible to
    the eye. Validate with the gains production will use, or certify nothing.
    """
    try:
        from cascade.config import load_demo_config

        arm = load_demo_config(cameras=["mock"], arm="rebot_rs_mb", llm="mock").arm
        if motor_id == GRIPPER_ID:
            g = arm.get("gripper") or {}
            kp, kd = g.get("kp"), g.get("kd")
            return (float(kp), float(kd)) if kp is not None and kd is not None else None
        kps, kds = arm.get("mit_kp"), arm.get("mit_kd")
        if not kps or not kds or motor_id > min(len(kps), len(kds)):
            return None
        return float(kps[motor_id - 1]), float(kds[motor_id - 1])
    except Exception as e:
        print(f"[warn] profile gains unavailable ({type(e).__name__}: {e})")
        return None


def _urdf_local_limits():
    """LOCAL-convention joint limits, or None when Pinocchio or the RS model is
    missing. Same accessor diag_rebot_mb.py uses; the jog treats absence as
    "cannot verify" and refuses to move rather than guessing."""
    try:
        from cascade.config import load_demo_config
        from cascade.control.kinematics import Kinematics

        cfg = load_demo_config(cameras=["mock"], arm="rebot_rs_mb", llm="mock")
        arm = cfg.arm
        kin = Kinematics(
            arm.model,
            arm.get("ee_frame", "gripper_end"),
            n_controlled=int(arm.get("n_joints", 6)),
            joint_signs=arm.get("joint_signs"),
        )
        return kin.joint_limits
    except Exception as e:
        print(f"[warn] URDF limits unavailable ({type(e).__name__}: {e})")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--channel", default="can0")
    ap.add_argument("--model", default="rs-00")
    ap.add_argument("--joint", type=int, required=True, choices=list(ARM_IDS) + [GRIPPER_ID],
                    help="motor id to jog (1..6 arm, 7 gripper)")
    ap.add_argument("--delta", type=float, default=0.05,
                    help=f"radians to move, signed (|delta| <= {MAX_DELTA_RAD}, "
                         f"or {MAX_DELTA_YAW_RAD} for the base yaw)")
    ap.add_argument("--kp", type=float, default=None,
                    help=f"MIT kp (<= {MAX_KP}); default is this joint's value "
                         "from configs/arms/rebot_rs_mb.yaml")
    ap.add_argument("--kd", type=float, default=None,
                    help=f"MIT kd (<= {MAX_KD}); default is this joint's value "
                         "from the profile")
    ap.add_argument("--duration", type=float, default=1.5, help="seconds for the ramp")
    ap.add_argument("--rate-hz", type=float, default=50.0)
    ap.add_argument("--hold-only", action="store_true",
                    help="enable and hold the current pose, jog nothing (gain sanity check)")
    ap.add_argument("--no-return", action="store_true", help="stay at the jogged pose")
    ap.add_argument("--disable-after", action="store_true",
                    help="torque OFF at the end. The arm WILL fall if unsupported.")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, touch no motor")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    cap = _delta_cap(args.joint)
    if abs(args.delta) > cap:
        print(f"[!] |--delta| must be <= {cap} rad for motor {args.joint} "
              f"(got {args.delta})")
        return 2

    kp_src = kd_src = "flag"
    if args.kp is None or args.kd is None:
        gains = _profile_gains(args.joint)
        if gains is None:
            print("[!] no --kp/--kd given and the profile's gains could not be "
                  "read. Refusing to guess a gain for a 48 V motor.")
            return 2
        if args.kp is None:
            args.kp, kp_src = gains[0], "profile"
        if args.kd is None:
            args.kd, kd_src = gains[1], "profile"
    # Only a run using the profile's gains end to end says anything about
    # production; a partial override does not.
    gain_src = "profile" if kp_src == kd_src == "profile" else f"kp:{kp_src} kd:{kd_src}"

    if args.kp > MAX_KP or args.kd > MAX_KD:
        print(f"[!] gains capped at kp<={MAX_KP} kd<={MAX_KD}")
        return 2

    from motorbridge import Controller, Mode

    ctrl = Controller(channel=args.channel)
    motors = {
        mid: ctrl.add_robstride_motor(motor_id=mid, feedback_id=HOST_ID, model=args.model)
        for mid in (*ARM_IDS, GRIPPER_ID)
    }
    # Bound before the try so the Ctrl+C handler can never raise NameError while
    # trying to soft-stop: an interrupt arriving during setup must still take the
    # re-assert-the-hold path, not crash out and leave the arm uncommanded.
    q_hold: dict[int, float] = {}
    gains: dict[int, tuple[float, float]] = {}
    try:
        for mid, m in motors.items():
            p = _read(m, MECH_POS)
            if p is None:
                print(f"[!] motor {mid} did not report mechPos; aborting rather than "
                      f"holding a pose built from an unknown joint")
                return 1
            q_hold[mid] = p

        target_id = args.joint
        start = q_hold[target_id]
        goal = start + args.delta

        # PER-MOTOR gains. The jogged joint gets --kp/--kd; every other motor is
        # held with its OWN profile value.
        #
        # This used to send the jogged joint's gains to all seven motors. It was
        # survivable while the default was a flat kp=8 -- below every configured
        # value -- but became dangerous the moment the default became per-joint:
        # jogging j2 or j3 (kp=30) then commanded kp=30 on the wrist (rated 10
        # and 8) and on the gripper (rated 6), up to 5x their stiffness. On the
        # rig 2026-08-27 that run lost the motors mid-jog: mode readback started
        # timing out and the whole bus went silent. Never let one joint's gain
        # leak into another's.
        gains: dict[int, tuple[float, float]] = {}
        for mid in motors:
            if mid == target_id:
                gains[mid] = (args.kp, args.kd)
                continue
            g = _profile_gains(mid)
            if g is None:
                print(f"[!] no profile gains for motor {mid}, so it cannot be held "
                      f"at its own stiffness. Refusing to move.")
                return 2
            gains[mid] = g

        # The gripper (id 7) has no URDF joint, so it is exempt: its travel was
        # characterized by hand in diag_rebot_mb.py instead.
        if target_id != GRIPPER_ID:
            limits = _urdf_local_limits()
            if limits is None:
                print("[!] cannot read the joint limits, so the goal cannot be "
                      "vetted. Refusing to move.")
                return 1
            lo, hi = limits
            j = target_id - 1
            if not (lo[j] < goal < hi[j]):
                print(f"[!] goal {goal:+.4f} rad is outside joint {target_id}'s "
                      f"limit [{lo[j]:+.3f}, {hi[j]:+.3f}]. Reduce --delta.")
                return 2

            # Vetting only the goal is not enough. On 2026-08-27 joints 2 and 3
            # sat at -0.0009 rad against a lower limit of exactly 0.0: the START
            # was already (just) out of bounds, i.e. the joint was resting on its
            # mechanical end with zero room on that side. A goal computed from
            # such a pose looks legal while the joint has nowhere to go if the
            # axis' sign is the other way round.
            if not (lo[j] <= start <= hi[j]):
                print(f"[!] joint {target_id} STARTS at {start:+.4f} rad, outside "
                      f"its limit [{lo[j]:+.3f}, {hi[j]:+.3f}]. It is resting on a "
                      f"mechanical end. Park the arm inside the limits before "
                      f"jogging -- from here one of the two directions is blocked.")
                return 2
            margin = min(start - lo[j], hi[j] - start)
            if margin < abs(args.delta):
                print(f"[warn] joint {target_id} is {margin:.4f} rad from a limit, "
                      f"less than the {abs(args.delta):.4f} rad delta. If this "
                      f"axis' sign is inverted, the jog drives it into the stop; "
                      f"the stall guard will abort, but prefer a smaller --delta.")

        print("=== plan ===")
        print(f"  channel     {args.channel}")
        print(f"  hold pose   " + "  ".join(f"{k}:{v:+.4f}" for k, v in q_hold.items()))
        print(f"  jog motor   {target_id} ({'gripper' if target_id == GRIPPER_ID else 'arm joint'})")
        print(f"  {start:+.4f} -> {goal:+.4f} rad  (delta {args.delta:+.4f})")
        print(f"  gains       kp={args.kp} kd={args.kd} (from {gain_src})"
              f"   ramp {args.duration}s @ {args.rate_hz}Hz")
        # Every motor's gain, not just the jogged one. Printing only the jogged
        # joint's is what let a 5x over-stiff command reach the wrist and gripper
        # unnoticed on 2026-08-27; the plan must show what each motor will get.
        print("  held at     " + "  ".join(
            f"{'grip' if mid == GRIPPER_ID else f'j{mid}'}:"
            f"{gains[mid][0]:g}/{gains[mid][1]:g}"
            + ("*" if mid == target_id else "")
            for mid in sorted(gains)
        ) + "   (kp/kd, * = the jogged joint)")
        print(f"  return      {'no' if args.no_return else 'yes, back to start'}")
        print(f"  end state   {'TORQUE OFF (arm falls if unsupported)' if args.disable_after else 'holding'}")

        if args.dry_run:
            print("\n[dry-run] nothing was enabled or commanded.")
            return 0

        if not args.yes:
            print("\nThe arm will be ENERGIZED and will hold this pose. Confirm the arm is")
            print("supported and clear of people, then type 'go' to proceed:")
            if input("  > ").strip().lower() != "go":
                print("[+] aborted, nothing was energized.")
                return 0

        for mid, m in motors.items():
            m.ensure_mode(Mode.MIT)
        # Hold the captured pose on every joint BEFORE enabling, so the first
        # thing each motor does once energized is stay where it already is.
        _send_hold(motors, q_hold, gains)
        ctrl.enable_all()
        _send_hold(motors, q_hold, gains)
        time.sleep(0.3)

        held = {mid: _read(m, MECH_POS) for mid, m in motors.items()}
        drift = max(
            abs((held[mid] or q_hold[mid]) - q_hold[mid]) for mid in motors
        )
        print(f"\n[+] energized. max drift from the hold pose: {drift:.4f} rad")
        if drift > 0.05:
            print("[!] the arm sagged more than 50 mrad -- gains are too low for this "
                  "pose. Not jogging. Re-run with a higher --kp, or from a more "
                  "gravity-neutral pose.")
            return 1

        if args.hold_only:
            print("[+] --hold-only: holding. Ctrl+C to finish.")
            _idle(motors, q_hold, gains)
            return 0

        print(f"[+] jogging motor {target_id} ...")
        if not _ramp(motors, q_hold, target_id, start, goal, gains, args):
            print("\n[!] stalled -- the arm is left holding where the joint "
                  "actually stopped, torque ON. Nothing further was commanded.")
            print("    Next: check whether this axis' positive direction runs "
                  "toward its mechanical end from this pose, and try the "
                  "OPPOSITE sign with a small --delta. Park the arm inside the "
                  "URDF limits first (scripts/diag_rebot_mb.py shows where it "
                  "is); a joint sitting exactly at a limit has no room to move "
                  "in one of the two directions.")
            return 1
        time.sleep(0.4)
        end = _read(motors[target_id], MECH_POS)
        moved = None if end is None else end - start

        print("\n=== result ===")
        print(f"  commanded delta  {args.delta:+.4f} rad")
        print(f"  mechPos delta    {'  n/a' if moved is None else f'{moved:+.4f} rad'}")
        if moved is not None:
            if abs(moved) < abs(args.delta) * 0.3:
                print("  -> mechPos barely moved: gains too low, or the joint is blocked.")
                print(f"     This measures NOTHING about the joint's sign -- "
                      f"{abs(np.degrees(moved)):.2f} deg of travel is within "
                      f"compliance and backlash, and is invisible to the eye.")
                if gain_src == "profile":
                    print("     These are the PROFILE's gains, so the profile itself is "
                          "too soft for this pose (or the joint is against a stop). "
                          "Raise it there rather than only passing --kp here, or the "
                          "backend will keep under-tracking in production.")
            elif np.sign(moved) == np.sign(args.delta):
                print("  -> mechPos follows the command sign (command and feedback agree).")
            else:
                print("  -> mechPos moved OPPOSITE to the command. The backend must negate "
                      "one of the two; do NOT jog further until that is settled.")
        print("\n  Now the part only you can see: did the joint move the direction the")
        print("  URDF's positive axis predicts? If not, joint_signs for this joint is -1.")

        if not args.no_return:
            print(f"\n[+] returning motor {target_id} to {start:+.4f}")
            if not _ramp(motors, q_hold, target_id, goal, start, gains, args):
                print("[!] stalled on the way BACK, which means the joint cannot "
                      "reach the pose it started from. Support the arm and "
                      "inspect the joint before commanding anything else.")
                return 1

        if args.disable_after:
            print("[+] torque OFF")
            ctrl.disable_all()
        else:
            print("[+] leaving the arm holding. Park it before any disconnect.")
        return 0
    except KeyboardInterrupt:
        # Soft stop: re-assert the hold pose rather than cutting torque, matching
        # RebotRSArm.stop(). Cutting torque on a loaded arm makes it free-fall.
        if not (q_hold and gains):
            # Interrupted during setup: nothing was enabled by us, and inventing
            # a hold pose from partial reads would command joints to positions
            # never measured.
            print("\n[!] interrupted before the hold pose was established -- "
                  "nothing was commanded. If the arm is energized it is from a "
                  "previous run and still holding whatever it last received.")
            return 130
        print("\n[!] interrupted -- re-asserting the hold pose (torque stays ON)")
        try:
            _send_hold(motors, q_hold, gains)
        except Exception as e:
            print(f"[!] could not re-assert the hold ({type(e).__name__}: {e}). "
                  "The motors may be unreachable; SUPPORT THE ARM and check the "
                  "bus with scripts/diag_rebot_mb.py before re-running.")
        return 130
    finally:
        try:
            ctrl.close_bus()
        finally:
            ctrl.close()


def _send_hold(motors, q_hold, gains) -> None:
    for mid, m in motors.items():
        kp, kd = gains[mid]
        m.send_mit(q_hold[mid], 0.0, kp, kd, 0.0)


#: Abort the ramp once the jogged joint lags its commanded position by more than
#: this. A joint driven into a mechanical stop shows exactly this signature: the
#: command keeps advancing, mechPos does not, and the motor screams while the
#: integral of that error becomes torque into the stop.
#:
#: This exists because of a real incident (rig, 2026-08-27). Joints 3-5 returned
#: 6-15% of a commanded 0.100 rad and the script reported "gains too low, or the
#: joint is blocked". The gains were raised -- the WRONG branch of that "or" --
#: and the next run drove a blocked joint at kp=30 until the operator cut power.
#: The lesson is not "pick the right branch by hand": a stalled joint must abort
#: the motion by itself, before a human has to decide anything.
STALL_LAG_RAD = 0.035

#: Sample mechPos every N waypoints. A param read costs a round trip, so reading
#: on every 50 Hz waypoint would distort the ramp; every 5th is ~10 Hz, fast
#: enough to catch a stall within a few tens of milliseconds of travel.
STALL_CHECK_EVERY = 5


def _ramp(motors, q_hold, target_id, start, goal, gains, args) -> bool:
    """Ramp the target joint from start to goal. Returns False if it STALLED.

    On a stall the ramp stops advancing immediately and the caller is expected
    to stop commanding motion; continuing would keep loading the stop.
    """
    steps = max(2, int(args.duration * args.rate_hz))
    dt = args.duration / steps
    t0 = time.monotonic()
    for i in range(1, steps + 1):
        s = i / steps
        # min-jerk, same profile ArmBase.stream_to uses, so the jog feels like
        # the real motion path rather than a step input.
        s = 10 * s**3 - 15 * s**4 + 6 * s**5
        q = start + (goal - start) * s

        if i % STALL_CHECK_EVERY == 0:
            actual = _read(motors[target_id], MECH_POS, timeout_ms=80)
            if actual is not None and abs(q - actual) > STALL_LAG_RAD:
                print(f"\n[!] STALL: commanded {q:+.4f} but mechPos is {actual:+.4f} "
                      f"({abs(q - actual):.4f} rad of lag). Aborting the ramp.")
                print("    A joint that will not follow is against a mechanical "
                      "stop, or the sign of this axis is inverted so the command "
                      "drives it INTO the stop. Do NOT raise the gains: that is "
                      "how the motor gets damaged.")
                # Stop pushing. Command the joint to where it actually is, so it
                # holds there instead of continuing to load the stop.
                for mid, m in motors.items():
                    kp, kd = gains[mid]
                    m.send_mit(actual if mid == target_id else q_hold[mid],
                               0.0, kp, kd, 0.0)
                return False

        for mid, m in motors.items():
            pos = q if mid == target_id else q_hold[mid]
            kp, kd = gains[mid]
            m.send_mit(pos, 0.0, kp, kd, 0.0)
        sleep_s = t0 + i * dt - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)
    return True


def _idle(motors, q_hold, gains) -> None:
    try:
        while True:
            _send_hold(motors, q_hold, gains)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
