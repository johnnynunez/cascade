"""Determine each joint's wire sign WITHOUT needing a human to watch the arm.

`jog_rebot_mb.py` asks the operator which way the joint turned. That works, but
it needs tens of degrees of travel to be readable by eye, and a joint parked on
its mechanical end cannot deliver that in one of the two directions -- which is
exactly how the 2026-08-27 incident happened: joints 3-5 were commanded toward
their stop, returned 6-15% of the command, were misread as "gains too low", and
got driven at kp=30 into the stop until the operator cut power.

This script infers the sign from geometry instead of eyesight:

  A joint resting at one end of its URDF range must be FREE in the direction
  that leads into the range and BLOCKED in the direction that leads out of it.
  Probe both directions with a small delta and see which one moves. If the free
  direction is the one the URDF says should be blocked, mechPos runs opposite to
  the local convention and wire_signs for that joint is -1.

When a joint sits comfortably mid-range both directions are free; the sign is
then NOT determined by this test, and the script says so rather than guessing.

SAFETY
- Deltas are tiny (DEFAULT_DELTA_RAD) and hard-capped (MAX_DELTA_RAD).
- Stall detection runs DURING every ramp at ~10 Hz and aborts on the first sign
  the joint is not following, commanding it to where it actually is. The
  threshold scales with the delta, so a small probe is not allowed to push.
- Every motor that is not being probed is held at ITS OWN profile gains. A
  single joint's gain must never be broadcast to the others: at kp=30 that is 5x
  the wrist's and the gripper's rated stiffness.
- Torque is left ON at the end (a de-energized loaded arm free-falls). Park the
  arm with move_home before disconnecting.

The bus is exclusive (host id 0xFD): stop motorbridge-gateway and any LeRobot
process first.

Usage:

    python scripts/sign_check_rebot_mb.py --dry-run       # plan only
    python scripts/sign_check_rebot_mb.py                 # all six joints
    python scripts/sign_check_rebot_mb.py --joints 3,4,5  # just the suspects
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from jog_rebot_mb import (  # noqa: E402  (path set above)
    ARM_IDS,
    GRIPPER_ID,
    HOST_ID,
    MECH_POS,
    _profile_gains,
    _read,
    _urdf_local_limits,
)

#: Deliberately small. The probe only has to distinguish "moves" from "does not
#: move", and every extra radian is extra energy into a stop if the sign is
#: wrong.
DEFAULT_DELTA_RAD = 0.06
MAX_DELTA_RAD = 0.12

#: Fraction of the commanded delta the joint must achieve to count as FREE.
#: Chosen well above the 6-15% that a blocked joint returned on the rig, and
#: well below the 84-95% a free joint managed.
FREE_FRACTION = 0.5

#: Abort the ramp when the joint lags the command by more than this fraction of
#: the delta. Proportional, not absolute: an absolute threshold larger than the
#: delta could never fire on a small probe.
STALL_LAG_FRACTION = 0.4

CHECK_EVERY = 4
RATE_HZ = 50.0
RAMP_S = 1.0


def _ramp_one(motors, q_hold, gains, target_id, start, goal, stall_lag):
    """Ramp one joint, aborting on stall. Returns (reached, stalled)."""
    steps = max(4, int(RAMP_S * RATE_HZ))
    dt = RAMP_S / steps
    t0 = time.monotonic()
    for i in range(1, steps + 1):
        s = i / steps
        s = 10 * s**3 - 15 * s**4 + 6 * s**5
        q = start + (goal - start) * s
        if i % CHECK_EVERY == 0:
            actual = _read(motors[target_id], MECH_POS, timeout_ms=80)
            if actual is not None and abs(q - actual) > stall_lag:
                for mid, m in motors.items():
                    kp, kd = gains[mid]
                    m.send_mit(actual if mid == target_id else q_hold[mid],
                               0.0, kp, kd, 0.0)
                return actual, True
        for mid, m in motors.items():
            pos = q if mid == target_id else q_hold[mid]
            kp, kd = gains[mid]
            m.send_mit(pos, 0.0, kp, kd, 0.0)
        sleep_s = t0 + i * dt - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)
    time.sleep(0.25)
    return _read(motors[target_id], MECH_POS), False


def _probe(motors, q_hold, gains, jid, delta, stall_lag):
    """One direction. Returns (fraction_achieved, stalled)."""
    start = q_hold[jid]
    end, stalled = _ramp_one(motors, q_hold, gains, jid, start, start + delta, stall_lag)
    moved = 0.0 if end is None else end - start
    frac = abs(moved) / abs(delta) if delta else 0.0
    # Always come back, and treat a stall on the way back as a hard error: the
    # joint could not reach the pose it started from.
    back, back_stalled = _ramp_one(motors, q_hold, gains, jid,
                                   end if end is not None else start, start, stall_lag)
    if back_stalled:
        print(f"    [!] joint {jid} STALLED returning to {start:+.4f} "
              f"(stopped at {back:+.4f}). Not probing further.")
        return frac, True
    return frac, stalled


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--channel", default="can0")
    ap.add_argument("--model", default="rs-00")
    ap.add_argument("--joints", default="1,2,3,4,5,6",
                    help="comma-separated motor ids to test")
    ap.add_argument("--delta", type=float, default=DEFAULT_DELTA_RAD,
                    help=f"probe size in rad (<= {MAX_DELTA_RAD})")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    if abs(args.delta) > MAX_DELTA_RAD:
        print(f"[!] --delta must be <= {MAX_DELTA_RAD} rad (got {args.delta})")
        return 2
    try:
        jids = [int(x) for x in args.joints.split(",") if x.strip()]
    except ValueError:
        print("[!] --joints must be comma-separated integers")
        return 2
    if any(j not in ARM_IDS for j in jids):
        print(f"[!] --joints must be within {ARM_IDS} (the gripper has no URDF joint)")
        return 2

    limits = _urdf_local_limits()
    if limits is None:
        print("[!] the URDF limits are what this test reasons from. Cannot run.")
        return 1
    lo, hi = limits

    from motorbridge import Controller, Mode

    ctrl = Controller(channel=args.channel)
    motors = {
        mid: ctrl.add_robstride_motor(motor_id=mid, feedback_id=HOST_ID, model=args.model)
        for mid in (*ARM_IDS, GRIPPER_ID)
    }
    q_hold: dict[int, float] = {}
    gains: dict[int, tuple[float, float]] = {}
    try:
        for mid, m in motors.items():
            p = _read(m, MECH_POS)
            if p is None:
                print(f"[!] motor {mid} did not report mechPos; aborting")
                return 1
            q_hold[mid] = p
            g = _profile_gains(mid)
            if g is None:
                print(f"[!] no profile gains for motor {mid}; aborting")
                return 1
            gains[mid] = g

        stall_lag = abs(args.delta) * STALL_LAG_FRACTION

        print("=== plan ===")
        print(f"  channel     {args.channel}")
        print(f"  probing     joints {jids} at +-{args.delta:.3f} rad")
        print(f"  stall abort when lag > {stall_lag:.4f} rad")
        print("  held at     " + "  ".join(
            f"{'grip' if mid == GRIPPER_ID else f'j{mid}'}:"
            f"{gains[mid][0]:g}/{gains[mid][1]:g}" for mid in sorted(gains)))
        print()
        print("  start pose vs URDF limits (LOCAL):")
        for j in jids:
            q = q_hold[j]
            room_lo, room_hi = q - lo[j - 1], hi[j - 1] - q
            flag = ""
            if room_lo < abs(args.delta):
                flag = "  <-- ON/NEAR ITS LOWER END"
            elif room_hi < abs(args.delta):
                flag = "  <-- ON/NEAR ITS UPPER END"
            print(f"    j{j}: {q:+.4f}  room below {room_lo:+.3f}  "
                  f"above {room_hi:+.3f}{flag}")

        if args.dry_run:
            print("\n[dry-run] nothing was enabled or commanded.")
            return 0

        if not args.yes:
            print("\nThe arm will be ENERGIZED. Each joint moves at most "
                  f"{args.delta:.3f} rad ({np.degrees(args.delta):.1f} deg) and "
                  "returns. Confirm it is supported and clear, then type 'go':")
            if input("  > ").strip().lower() != "go":
                print("[+] aborted, nothing was energized.")
                return 0

        for m in motors.values():
            m.ensure_mode(Mode.MIT)
        _hold(motors, q_hold, gains)
        ctrl.enable_all()
        _hold(motors, q_hold, gains)
        time.sleep(0.3)

        drift = max(abs((_read(m, MECH_POS) or q_hold[mid]) - q_hold[mid])
                    for mid, m in motors.items())
        print(f"\n[+] energized. max drift from the hold pose: {drift:.4f} rad")
        if drift > 0.05:
            print("[!] the arm sagged past 50 mrad. Not probing.")
            return 1

        results = {}
        for j in jids:
            print(f"\n[+] joint {j}: probing + then -")
            pos_frac, pos_stall = _probe(motors, q_hold, gains, j, +abs(args.delta), stall_lag)
            print(f"    +{abs(args.delta):.3f}: reached {pos_frac * 100:5.1f}%"
                  f"{'  STALLED' if pos_stall else ''}")
            neg_frac, neg_stall = _probe(motors, q_hold, gains, j, -abs(args.delta), stall_lag)
            print(f"    -{abs(args.delta):.3f}: reached {neg_frac * 100:5.1f}%"
                  f"{'  STALLED' if neg_stall else ''}")
            results[j] = (pos_frac, neg_frac)

        _report(results, q_hold, lo, hi, args)
        print("\n[+] leaving the arm holding. Park it (move_home) before any disconnect.")
        return 0
    except KeyboardInterrupt:
        print("\n[!] interrupted -- re-asserting the hold pose (torque stays ON)")
        if q_hold and gains:
            try:
                _hold(motors, q_hold, gains)
            except Exception as e:
                print(f"[!] could not re-assert the hold ({type(e).__name__}: {e}). "
                      "SUPPORT THE ARM.")
        return 130
    finally:
        try:
            ctrl.close_bus()
        finally:
            ctrl.close()


def _hold(motors, q_hold, gains) -> None:
    for mid, m in motors.items():
        kp, kd = gains[mid]
        m.send_mit(q_hold[mid], 0.0, kp, kd, 0.0)


def _report(results, q_hold, lo, hi, args) -> None:
    print("\n=== veredicto ===")
    signs = {}
    for j, (pos_frac, neg_frac) in sorted(results.items()):
        q = q_hold[j]
        pos_free = pos_frac >= FREE_FRACTION
        neg_free = neg_frac >= FREE_FRACTION
        room_lo, room_hi = q - lo[j - 1], hi[j - 1] - q
        at_lower = room_lo < abs(args.delta)
        at_upper = room_hi < abs(args.delta)

        if pos_free and neg_free:
            verdict = ("both directions free -> mid-range, sign NOT determined "
                       "by this test (needs an eye or an FK check)")
            sign = None
        elif not pos_free and not neg_free:
            verdict = ("neither direction moved -> the joint is jammed, or the "
                       "profile's gains are too soft for this pose. NOT a sign "
                       "result.")
            sign = None
        elif pos_free and at_lower:
            verdict = "free toward + while resting on its LOWER end -> sign +1 (as configured)"
            sign = 1
        elif neg_free and at_lower:
            verdict = ("free toward - while resting on its LOWER end -> mechPos "
                       "runs OPPOSITE to the local convention -> sign -1")
            sign = -1
        elif neg_free and at_upper:
            verdict = "free toward - while resting on its UPPER end -> sign +1 (as configured)"
            sign = 1
        elif pos_free and at_upper:
            verdict = ("free toward + while resting on its UPPER end -> mechPos "
                       "runs OPPOSITE to the local convention -> sign -1")
            sign = -1
        else:
            verdict = ("one direction blocked but the joint is not near a URDF "
                       "end: something mechanical, not a sign result")
            sign = None
        signs[j] = sign
        print(f"  j{j}: {verdict}")

    determined = {j: s for j, s in signs.items() if s is not None}
    if determined:
        print("\n  wire_signs implied by this run (only the determined ones):")
        print("   " + "  ".join(f"j{j}={s:+d}" for j, s in sorted(determined.items())))
        if any(s == -1 for s in determined.values()):
            print("\n  At least one joint is INVERTED. Update wire_signs in "
                  "configs/arms/rebot_rs_mb.yaml before commanding any multi-joint "
                  "motion: an inverted joint sends the end effector to the mirror "
                  "image of its target, and move_home would drive it 1.2 rad the "
                  "wrong way.")
    else:
        print("\n  Nothing determined. Park the arm mid-range and re-run: this "
              "test reasons from a joint resting against a known end.")


if __name__ == "__main__":
    raise SystemExit(main())
