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
- `--delta` is hard-capped (MAX_DELTA_RAD) and `--kp` is hard-capped
  (MAX_KP) because the RobStride MIT gain scale is NOT yet characterized on
  this build: the shipped profile leaves `mit_kp: null` to inherit the SDK's
  per-joint YAML, which does not exist on the motorbridge path. Start at the
  defaults and raise only if the joint visibly fails to track.
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


def _read(motor, param: int, timeout_ms: int = 300) -> float | None:
    try:
        return float(motor.robstride_get_param_f32(param, timeout_ms))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--channel", default="can0")
    ap.add_argument("--model", default="rs-00")
    ap.add_argument("--joint", type=int, required=True, choices=list(ARM_IDS) + [GRIPPER_ID],
                    help="motor id to jog (1..6 arm, 7 gripper)")
    ap.add_argument("--delta", type=float, default=0.05,
                    help=f"radians to move, signed (|delta| <= {MAX_DELTA_RAD})")
    ap.add_argument("--kp", type=float, default=8.0, help=f"MIT kp (<= {MAX_KP})")
    ap.add_argument("--kd", type=float, default=0.5, help=f"MIT kd (<= {MAX_KD})")
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

    if abs(args.delta) > MAX_DELTA_RAD:
        print(f"[!] |--delta| must be <= {MAX_DELTA_RAD} rad (got {args.delta})")
        return 2
    if args.kp > MAX_KP or args.kd > MAX_KD:
        print(f"[!] gains capped at kp<={MAX_KP} kd<={MAX_KD}")
        return 2

    from motorbridge import Controller, Mode

    ctrl = Controller(channel=args.channel)
    motors = {
        mid: ctrl.add_robstride_motor(motor_id=mid, feedback_id=HOST_ID, model=args.model)
        for mid in (*ARM_IDS, GRIPPER_ID)
    }
    try:
        q_hold: dict[int, float] = {}
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

        print("=== plan ===")
        print(f"  channel     {args.channel}")
        print(f"  hold pose   " + "  ".join(f"{k}:{v:+.4f}" for k, v in q_hold.items()))
        print(f"  jog motor   {target_id} ({'gripper' if target_id == GRIPPER_ID else 'arm joint'})")
        print(f"  {start:+.4f} -> {goal:+.4f} rad  (delta {args.delta:+.4f})")
        print(f"  gains       kp={args.kp} kd={args.kd}   ramp {args.duration}s @ {args.rate_hz}Hz")
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
        _send_hold(motors, q_hold, args)
        ctrl.enable_all()
        _send_hold(motors, q_hold, args)
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
            _idle(motors, q_hold, args)
            return 0

        print(f"[+] jogging motor {target_id} ...")
        _ramp(motors, q_hold, target_id, start, goal, args)
        time.sleep(0.4)
        end = _read(motors[target_id], MECH_POS)
        moved = None if end is None else end - start

        print("\n=== result ===")
        print(f"  commanded delta  {args.delta:+.4f} rad")
        print(f"  mechPos delta    {'  n/a' if moved is None else f'{moved:+.4f} rad'}")
        if moved is not None:
            if abs(moved) < abs(args.delta) * 0.3:
                print("  -> mechPos barely moved: gains too low, or the joint is blocked.")
            elif np.sign(moved) == np.sign(args.delta):
                print("  -> mechPos follows the command sign (command and feedback agree).")
            else:
                print("  -> mechPos moved OPPOSITE to the command. The backend must negate "
                      "one of the two; do NOT jog further until that is settled.")
        print("\n  Now the part only you can see: did the joint move the direction the")
        print("  URDF's positive axis predicts? If not, joint_signs for this joint is -1.")

        if not args.no_return:
            print(f"\n[+] returning motor {target_id} to {start:+.4f}")
            _ramp(motors, q_hold, target_id, goal, start, args)

        if args.disable_after:
            print("[+] torque OFF")
            ctrl.disable_all()
        else:
            print("[+] leaving the arm holding. Park it before any disconnect.")
        return 0
    except KeyboardInterrupt:
        # Soft stop: re-assert the hold pose rather than cutting torque.
        print("\n[!] interrupted -- re-asserting the hold pose (torque stays ON)")
        try:
            _send_hold(motors, q_hold, args)
        except Exception:
            pass
        return 130
    finally:
        try:
            ctrl.close_bus()
        finally:
            ctrl.close()


def _send_hold(motors, q_hold, args) -> None:
    for mid, m in motors.items():
        m.send_mit(q_hold[mid], 0.0, args.kp, args.kd, 0.0)


def _ramp(motors, q_hold, target_id, start, goal, args) -> None:
    steps = max(2, int(args.duration * args.rate_hz))
    dt = args.duration / steps
    t0 = time.monotonic()
    for i in range(1, steps + 1):
        s = i / steps
        # min-jerk, same profile ArmBase.stream_to uses, so the jog feels like
        # the real motion path rather than a step input.
        s = 10 * s**3 - 15 * s**4 + 6 * s**5
        q = start + (goal - start) * s
        for mid, m in motors.items():
            pos = q if mid == target_id else q_hold[mid]
            m.send_mit(pos, 0.0, args.kp, args.kd, 0.0)
        sleep_s = t0 + i * dt - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)


def _idle(motors, q_hold, args) -> None:
    try:
        while True:
            _send_hold(motors, q_hold, args)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
