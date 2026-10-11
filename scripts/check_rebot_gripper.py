"""Check the reBot RS gripper on the real arm: turn reference, open, close.

The gripper's motor knows its position only within one turn after power-on,
and the jaw travels more than one turn (control/rs_gripper.py). This script
exercises the turn reference and the guarded opening on the real jaw. Only
the gripper moves:

1. Connect the `rebot_rs` driver. The arm joints are held where they are
   measured, with torque on. The script refuses unless every joint starts
   within REST_TOL of park_q, the folded rest. Then nothing has moved, and
   releasing torque at the end drops nothing.
2. Print the gripper motor's RAW reading, its power-on frame.
3. Reference: close slowly to the closed stop and fix the turn offset.
4. Open to `open_pos` (ramped, watched), close with the torque close,
   open halfway, close again. Each step prints the measured jaw angle.
5. Release torque. The arm is still folded at rest.

Any GripperFault stops the run, with the gripper holding where it stopped.
Ctrl-C does the same.

    python scripts/check_rebot_gripper.py --yes      # operator at the e-stop

Stop the robot server first: it owns the CAN bus
(`ARM_SUPPORTED=1 ~/cascade-demo.sh down` with the arm folded at rest).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from cascade.types import GripperFault

REST_TOL = 0.12          # rad, as scripts/home_rebot_rs.py


def _fmt(q) -> str:
    return "[" + " ".join(f"{v:+.3f}" for v in q) + "]"


def run(arm, park, log=print) -> int:
    q = np.asarray(arm.get_state().q, dtype=float)
    arm.send_joint_target(q)                                   # hold exactly here
    off = float(np.max(np.abs(q - park)))
    log(f"[+] arm q {_fmt(q)} (max |q - park| {off:.3f})")
    if off > REST_TOL:
        log(f"[!] the arm is not folded at rest (> {REST_TOL} rad from park_q). It is "
            "HELD here, torque on. Support it before anything cuts the power.")
        return 2
    guard = arm._grip_guard
    raw = arm._gripper_raw()
    log(f"[+] gripper motor raw reading at connect: {raw:+.4f} rad (power-on frame)")
    try:
        guard.reference()
        turns = round(guard.offset / (2 * np.pi))
        log(f"[+] referenced: offset {turns:+d} turn(s); jaw at {arm._gripper_pos():+.3f} rad "
            f"(closed_pos {guard.closed_pos:+.2f}, open_pos {guard.open_pos:+.2f})")
        steps = (("open", lambda: arm.open_gripper()),
                 ("close", lambda: arm.close_gripper_torque()),
                 ("open half", lambda: arm.set_gripper((guard.open_pos + guard.closed_pos) / 2)),
                 ("close", lambda: arm.close_gripper_torque()))
        for name, step in steps:
            t0 = time.monotonic()
            step()
            time.sleep(0.5)
            log(f"    {name:9s} -> jaw {arm._gripper_pos():+.3f} rad "
                f"({time.monotonic() - t0:.1f} s)")
    except GripperFault as e:
        log(f"[!] GripperFault: {e}")
        return 1
    log("[+] gripper OK: referenced, opened without touching the open end, closed")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true",
                    help="required: operator at the e-stop, arm folded at rest, server stopped")
    args = ap.parse_args(argv)
    if not args.yes:
        print("[!] needs --yes (operator at the e-stop, arm folded at rest, robot server "
              "stopped). Nothing done.")
        return 2
    from cascade.config import load_demo_config
    from cascade.control.rebot_rs_arm import RebotRSArm

    cfg = load_demo_config(camera="mock", arm="rebot_rs", llm="mock")
    park = np.asarray(cfg.arm.get("park_q") or np.zeros(int(cfg.arm.n_joints)), dtype=float)
    arm = RebotRSArm(cfg.arm)
    arm.connect()
    code = 1
    try:
        code = run(arm, park)
    except KeyboardInterrupt:
        print("[!] interrupted: the gripper holds where it is")
    finally:
        if code != 2:
            arm.disconnect()          # torque off: the arm rests on its stops
            print("[+] torque released (the arm is folded at rest)")
        else:
            arm.stop()
    return code


if __name__ == "__main__":
    sys.exit(main())
