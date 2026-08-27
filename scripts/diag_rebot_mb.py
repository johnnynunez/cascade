"""Read-only RobStride feedback probe over motorbridge (macOS/Windows/Linux).

Phase-1 tool for bringing up a motorbridge-backed reBot RS backend on a host
where `reBotArm_control_py` + SocketCAN are unavailable (macOS routes `can0`
to PCAN via MacCAN PCBUSB; see docs). It answers the two questions a new
backend cannot guess:

  1. What is motorbridge's `mechPos` sign/offset relative to the LOCAL joint
     convention every constant in this repo uses (home_q, limits, tuned
     poses)? Both shipped RS assets are authored mirrored (q_asset =
     -q_local), and `joint_signs` bakes that flip in at model load -- but
     that says nothing about what the WIRE reports.
  2. What is the gripper motor's actual travel and polarity? The shipped
     profile's open/closed angles were characterized on the DM build and the
     profile itself flags them as unverified for RS.

TORQUE SAFETY: this script only ever calls `add_robstride_motor`,
`robstride_ping` and `robstride_get_param_f32`. It never calls `enable`,
`enable_all`, `ensure_mode` or any `send_*`, so it cannot energize a motor
or command motion. Reads alone are safe -- `motorbridge-cli scan` takes the
same path and leaves the arm untouched.

The arm's holding torque is whatever the LAST bus owner left it as: a
MotorBridge Studio session leaves motors enabled and holding, while a clean
shutdown leaves them limp and the arm WILL droop under gravity. Support the
arm before running this.

Usage (bus must be free -- host id 0xFD is exclusive, so stop
motorbridge-gateway and any LeRobot process first):

    python scripts/diag_rebot_mb.py                     # live table, Ctrl+C for summary
    python scripts/diag_rebot_mb.py --snapshot zero     # label one reading
    python scripts/diag_rebot_mb.py --channel can0@1000000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

#: RobStride parameter ids. Positions come from param reads, not the state
#: stream: RS motors emit compact type-0x18 report frames that the generic
#: get_state() path never decodes, and enabling active report yields a single
#: type-2 frame and then freezes. Param reads return the exact f32.
MECH_POS = 0x7019
MECH_VEL = 0x701A

#: Joints 1..6 then the gripper. The follower's host/feedback id is 0xFD,
#: confirmed by `motorbridge-cli scan` (responder_id=254).
ARM_IDS = (1, 2, 3, 4, 5, 6)
GRIPPER_ID = 7
HOST_ID = 0xFD


def _urdf_local_limits() -> tuple[np.ndarray, np.ndarray] | None:
    """LOCAL-convention joint limits from the RS URDF, or None if the model
    or Pinocchio is unavailable (the probe is still useful without them)."""
    try:
        from wrc_demo.config import load_demo_config
        from wrc_demo.control.kinematics import Kinematics

        cfg = load_demo_config(cameras=["mock"], arm="rebot_rs", llm="mock")
        arm = cfg.arm
        kin = Kinematics(
            arm.model,
            arm.get("ee_frame", "gripper_end"),
            n_controlled=int(arm.get("n_joints", 6)),
            joint_signs=arm.get("joint_signs"),
        )
        return kin.joint_limits
    except Exception as e:  # pragma: no cover - diagnostic convenience
        print(f"[warn] URDF limits unavailable ({type(e).__name__}: {e})")
        return None


def _open_motors(channel: str, model: str):
    from motorbridge import Controller

    ctrl = Controller(channel=channel)
    motors: dict[int, object] = {}
    for mid in (*ARM_IDS, GRIPPER_ID):
        motors[mid] = ctrl.add_robstride_motor(
            motor_id=mid, feedback_id=HOST_ID, model=model
        )
    return ctrl, motors


def _read(motor, param: int, timeout_ms: int) -> float | None:
    try:
        return float(motor.robstride_get_param_f32(param, timeout_ms))
    except Exception:
        return None


def _fmt(v: float | None, deg: bool = False) -> str:
    if v is None:
        return "    --  "
    return f"{np.degrees(v):8.2f}" if deg else f"{v:8.4f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--channel", default="can0", help="CAN channel (macOS: can0 -> PCAN_USBBUS1)")
    ap.add_argument("--model", default="rs-00", help="RobStride model string")
    ap.add_argument("--interval", type=float, default=0.25, help="seconds between passes")
    ap.add_argument("--timeout-ms", type=int, default=300, help="per-param read timeout")
    ap.add_argument("--snapshot", metavar="LABEL", help="record one labeled reading and exit")
    ap.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="stop the monitor after N seconds (0 = until Ctrl+C). Use this "
        "when running unattended so the summary is always printed.",
    )
    ap.add_argument(
        "--out",
        default=str(REPO / "runs" / "rebot_mb_probe.json"),
        help="where snapshots accumulate",
    )
    args = ap.parse_args()

    limits = _urdf_local_limits()
    print(f"[+] opening {args.channel} (read-only; no motor will be energized)")
    ctrl, motors = _open_motors(args.channel, args.model)

    try:
        alive = []
        for mid, m in motors.items():
            try:
                dev, responder = m.robstride_ping()
                alive.append(mid)
                print(f"    id={mid} online (device_id={dev} responder={responder})")
            except Exception as e:
                print(f"    id={mid} NO REPLY ({type(e).__name__})")
        if not alive:
            print("[!] no motors replied. Check 48 V power, the CAN harness, and that "
                  "the DIP switch is on 120R (termination) rather than BOOT.")
            return 1

        if args.snapshot:
            return _snapshot(motors, alive, args)

        _monitor(motors, alive, limits, args)
        return 0
    finally:
        # close_bus/shutdown only release the host's handle; they do not change
        # motor torque state either way.
        try:
            ctrl.close_bus()
        finally:
            ctrl.close()


def _snapshot(motors, alive, args) -> int:
    reading = {
        "label": args.snapshot,
        "saved_at": time.time(),
        "channel": args.channel,
        "mech_pos_rad": {},
        "mech_vel": {},
    }
    for mid in alive:
        reading["mech_pos_rad"][str(mid)] = _read(motors[mid], MECH_POS, args.timeout_ms)
        reading["mech_vel"][str(mid)] = _read(motors[mid], MECH_VEL, args.timeout_ms)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if out.exists():
        try:
            history = json.loads(out.read_text())
        except Exception:
            history = []
    history.append(reading)
    out.write_text(json.dumps(history, indent=2))

    print(f"\n[+] snapshot {args.snapshot!r}:")
    for mid in alive:
        p = reading["mech_pos_rad"][str(mid)]
        tag = "gripper" if mid == GRIPPER_ID else f"joint {mid}"
        print(f"    {tag:<9} mechPos = {_fmt(p)} rad ({_fmt(p, deg=True)} deg)")
    print(f"[+] appended to {out} ({len(history)} total)")
    return 0


def _monitor(motors, alive, limits, args) -> None:
    lo = hi = None
    if limits is not None:
        lo, hi = limits
    seen_min: dict[int, float] = {}
    seen_max: dict[int, float] = {}

    header = "  ".join(f"{'grip' if m == GRIPPER_ID else f'j{m}':>8}" for m in alive)
    print(f"\n[+] live mechPos (rad). Move each joint BY HAND through its range.")
    print("    Ctrl+C prints the observed span next to the URDF's local limits.\n")
    print(f"    {header}")

    deadline = time.monotonic() + args.duration if args.duration > 0 else None
    # Unattended runs get one line per pass: \r-overwritten output is unreadable
    # once it lands in a log file.
    inplace = deadline is None
    try:
        while deadline is None or time.monotonic() < deadline:
            row = []
            for mid in alive:
                p = _read(motors[mid], MECH_POS, args.timeout_ms)
                row.append(_fmt(p))
                if p is not None:
                    seen_min[mid] = min(seen_min.get(mid, p), p)
                    seen_max[mid] = max(seen_max.get(mid, p), p)
            print(f"    {'  '.join(row)}", end="\r" if inplace else "\n", flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    print("\n")

    print("=== observed vs URDF (LOCAL convention) ===")
    for mid in alive:
        if mid not in seen_min:
            continue
        span_lo, span_hi = seen_min[mid], seen_max[mid]
        tag = "gripper" if mid == GRIPPER_ID else f"joint {mid}"
        line = (
            f"  {tag:<9} observed [{span_lo:7.3f}, {span_hi:7.3f}] "
            f"span {span_hi - span_lo:6.3f} rad"
        )
        if lo is not None and mid <= len(lo):
            j = mid - 1
            line += f"   urdf [{lo[j]:7.3f}, {hi[j]:7.3f}] span {hi[j] - lo[j]:6.3f}"
        print(line)
    print(
        "\nInterpreting this: a matching span with a flipped sign means the wire "
        "convention is mirrored (joint_signs = -1); a matching span shifted by a "
        "constant means a zero offset. The gripper's span IS its usable travel, "
        "and its two ends give open_pos/closed_pos with the correct polarity."
    )


if __name__ == "__main__":
    raise SystemExit(main())
