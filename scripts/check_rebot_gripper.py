"""Check the reBot RS gripper on the real arm: turn reference, open, close.

The gripper's motor knows its position only within one turn after power-on,
and the jaw travels more than one turn (control/rs_gripper.py). This script
exercises the turn reference and the guarded opening on the real jaw. Only
the gripper moves:

1. Read every arm joint TWICE with NO motor enabled (the read-only path of
   scripts/diag_rebot_mb.py), and log the reading (cascade.control.arm_hold
   `log_rest_reading`: the data scripts/rest_tolerance_report.py tunes the
   rig's tolerance from). The script refuses, with nothing energized, unless
   the two reads agree and every joint is within the rig's rest tolerance of
   park_q, the folded rest (`rest_check_tol_rad`, at most REST_TOL). An arm
   an earlier session's teardown left holding (cascade.control.arm_hold) is
   released from its hold here only within 0.05 rad of park_q, the robot
   server's own torque-release tolerance: there its torque drops nothing.
   Farther off the hold stays, and the connect below refuses.
2. Connect the `rebot_rs` driver. The arm joints are held where they are
   measured, with torque on (checked against park_q again). Then nothing
   has moved, and releasing torque at the end drops nothing. A connect that
   went ahead on the operator's confirmation (a hold the 0.05 rule kept:
   `~/cascade-demo.sh confirm-held`, then `up`) only HOLDS, their hands on
   the arm: nothing moves, the jaw included, until they let go and type
   'go' at this terminal (its controlling terminal, in an interactive
   session, never a pipe nor an agent's shell: cascade.control.arm_hold
   `wait_for_let_go`); anything else stops here with the arm held.
3. Print the gripper motor's RAW reading, its power-on frame.
4. Reference: close slowly to the closed stop and fix the turn offset.
5. Open to `open_pos` (ramped, watched), close with the torque close,
   open halfway, close again. Each step prints the measured jaw angle.
6. Release torque. The arm is still folded at rest.

`--reference-only` is the startup check (`~/cascade-demo.sh up` runs it
before the robot server starts): steps 1-4 and 6. The jaws only close, onto
their closed stop, and stay closed. It fails when the gripper's zero no
longer matches its closed stop (rig 2026-10-12: a rack had jumped teeth on
the pinion), so the demo does not start with a gripper that would fail the
first grasp.

Any GripperFault stops the run, with the gripper holding where it stopped.
Ctrl-C does the same. Exit codes: 0 OK, 1 gripper fault, 2 refused (arm not
folded at rest, unreadable, or no --yes), 3 connected on the operator's
confirmation and stopped before anything moved (no 'go' typed at this
terminal): the arm stays POWERED and HOLDING (remembered).

    python scripts/check_rebot_gripper.py --yes                   # operator at the e-stop
    python scripts/check_rebot_gripper.py --yes --reference-only  # startup check

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
#: Exit code: the connect went ahead on the operator's confirmation and they
#: did not let go and type 'go': nothing moved, the arm stays HOLDING.
STOPPED_HOLDING = 3


def _fmt(q) -> str:
    return "[" + " ".join(f"{v:+.3f}" for v in q) + "]"


def read_rest_pose(channel: str = "can0", timeout_ms: int = 300) -> tuple[list, list]:
    """mechPos of joints 1-6 with no motor enabled, read TWICE (a pass over
    the joints, then a second): diag_rebot_mb.py's read-only path (add motor,
    read param; never enable or send). The rebot_rs driver reads the same
    parameter, so these are its joint values. None for a joint that did not
    answer. The two passes must agree (`rests_unpowered`)."""
    sys.path.insert(0, str(REPO / "scripts"))
    import diag_rebot_mb as diag

    ctrl, motors = diag._open_motors(channel, "rs-00")
    try:
        first = [diag._read(motors[mid], diag.MECH_POS, timeout_ms) for mid in diag.ARM_IDS]
        second = [diag._read(motors[mid], diag.MECH_POS, timeout_ms) for mid in diag.ARM_IDS]
        return first, second
    finally:
        try:
            ctrl.close_bus()
        finally:
            ctrl.close()


def rests_unpowered(q, park, log=print, *, again=None, tol: float = REST_TOL) -> bool:
    """The arm is folded at rest, judged BEFORE any torque: a raised arm must
    not be energized by a check that only means to move the gripper. `again`:
    the second read of every joint (the two must agree); `tol`: the rig's
    rest tolerance."""
    from cascade.control import arm_hold

    reads = [q] + ([again] if again is not None else [])
    if any(len(r) != len(park) or any(v is None for v in r) for r in reads):
        log(f"[!] could not read every arm joint ({q}); is the arm powered and the CAN bus "
            "free? Nothing energized.")
        return False
    problem = arm_hold.rest_problem(q, park, tol, again=again)
    if problem is not None:
        log(f"[!] the arm is not folded at rest: {problem}. Nothing energized. Fold it first (arm "
            "power off, by hand; or start the robot server and ask for 'rest'), then run this again.")
        return False
    off = float(np.max(np.abs(np.asarray(reads, dtype=float) - park)))
    log(f"[+] arm folded at rest before torque: {_fmt(q)} (max |q - park| {off:.3f}, read twice)")
    return True


def run(arm, park, log=print, *, reference_only: bool = False) -> int:
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
        if reference_only:
            log("[+] gripper referenced: closed stop where its zero says; the jaws stay closed")
            return 0
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
    ap.add_argument("--reference-only", action="store_true",
                    help="startup check: reference the gripper (it closes onto its stop) and stop")
    args = ap.parse_args(argv)
    if not args.yes:
        print("[!] needs --yes (operator at the e-stop, arm folded at rest, robot server "
              "stopped). Nothing done.")
        return 2
    from cascade.config import load_demo_config
    from cascade.control import rebot_rs_arm

    from cascade.control import arm_hold

    cfg = load_demo_config(camera="mock", arm="rebot_rs", llm="mock")
    park = np.asarray(cfg.arm.get("park_q") or np.zeros(int(cfg.arm.n_joints)), dtype=float)
    tol = arm_hold.rest_tolerance(cfg.arm)
    key = arm_hold.hold_key("rebot_rs", cfg.arm)
    q, again = read_rest_pose()
    problem = arm_hold.rest_problem(q, park, tol, again=again)
    arm_hold.log_rest_reading(source="startup_check", key=key, q=q, again=again, park=park, tol=tol,
                              marker=arm_hold.read(key) is not None,
                              verdict="rest" if problem is None else problem)
    if not rests_unpowered(q, park, again=again, tol=tol):
        return 2
    # An earlier session's teardown may have left this arm powered and holding
    # (cascade.control.arm_hold): measured within the robot server's own
    # torque-release tolerance of park_q, its torque drops nothing, so the hold
    # is cleared and the connect below may go ahead. Farther off it stays (the
    # server may have refused to release the arm there), and the connect refuses.
    arm_hold.clear_if_at_rest(key, q, park, again=again)
    arm = rebot_rs_arm.RebotRSArm(cfg.arm)
    try:
        arm.connect()
    except arm_hold.ArmHeldError as e:
        # refused before any frame that can cut torque: say so, no traceback
        print(f"[!] the arm was NOT connected: {e}")
        return 2
    code = 1
    try:
        if getattr(arm, "connected_on_confirmation", False) is True and not arm_hold.wait_for_let_go(key):
            # The connect went ahead on the operator's confirmation (a hold the
            # 0.05 rule kept): it only HOLDS, their hands on the arm. Nothing
            # moves, the jaw included, until they let go and type 'go' at
            # this terminal (never a pipe, never an agent's environment).
            print("[!] stopped before the gripper check: the arm stays connected and HOLDING where it is.")
            code = STOPPED_HOLDING
        else:
            code = run(arm, park, reference_only=args.reference_only)
    except KeyboardInterrupt:
        print("[!] interrupted: the gripper holds where it is")
    finally:
        if code not in (2, STOPPED_HOLDING):
            arm.disconnect()          # torque off: the arm rests on its stops
            print("[+] torque released (the arm is folded at rest)")
        else:
            arm.stop()
            # exits with the arm powered and holding: remembered for every
            # later process (#426 review Important 1)
            arm_hold.remember(arm.hold_key, arm="arm", source="check_rebot_gripper.py", log=print,
                              reason="the startup check measured it off its rest after connecting: torque stays ON"
                              if code == 2 else "the startup check connected it on the operator's confirmation; "
                                                "no 'go' was typed at its terminal: torque stays ON")
    return code


if __name__ == "__main__":
    sys.exit(main())
