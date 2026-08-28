#!/usr/bin/env python3
"""Feetech servo-bus diagnostic. READ-ONLY unless you ask for a jog.

The SO-101 profile ships with values derived from its URDF, not measured on a
rig (see `configs/arms/so101.yaml`). Two of them cannot be checked any other way
than on the arm, and getting either wrong is destructive:

  wire_signs   a flipped entry drives a joint the WRONG WAY on the first
               command, and printed PLA links reach a hard stop long before an
               STS3215 gives up pushing;
  servo_ids    an id collision (two servos answering as 2, the factory default
               before assembly) makes one joint mirror another.

    python scripts/diag_so101.py --port /dev/ttyACM0            # read only
    python scripts/diag_so101.py --port /dev/ttyACM0 --watch    # live angles
    python scripts/diag_so101.py --port /dev/ttyACM0 --jog 1    # move joint 1

The jog is the sign check: it moves ONE joint by a few degrees, prints the
before/after angle in the profile's own local convention, and tells you which
`wire_signs` entry to flip. Torque is enabled only for that joint and only for
the duration of the jog.

Nothing here imports the cascade package: this has to work when the arm profile
is wrong, which is the situation it exists for.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cascade.control.feetech import (  # noqa: E402
    ADDR_GOAL_POSITION,
    ADDR_MODE,
    ADDR_PRESENT_LOAD,
    ADDR_PRESENT_POSITION,
    ADDR_PRESENT_TEMPERATURE,
    ADDR_PRESENT_VOLTAGE,
    ADDR_TORQUE_ENABLE,
    ADDR_TORQUE_LIMIT,
    MODE_POSITION,
    SCS,
    STS,
    TORQUE_LIMIT_MAX,
    FeetechError,
    ServoBus,
    sign_magnitude,
)

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
#: URDF limits, LOCAL convention (assets/urdf/so101/so101.urdf).
LIMITS = [(-1.91986, 1.91986), (-1.74533, 1.74533), (-1.69, 1.69),
          (-1.65806, 1.65806), (-2.74385, 2.84121)]


def counts_to_rad(counts: int, center: float, cpr: float, sign: float) -> float:
    import math

    return sign * (counts - center) * 2.0 * math.pi / cpr


def rad_to_counts(rad: float, center: float, cpr: float, sign: float) -> int:
    import math

    c = center + sign * rad * cpr / (2.0 * math.pi)
    return int(round(min(max(c, 0.0), cpr - 1.0)))


def open_bus(port_name: str, baud: int, endian, timeout_s: float):
    try:
        import serial
    except ImportError:
        raise SystemExit(
            "pyserial is required: pip install pyserial "
            "(or pip install 'cascade[arm-feetech]')"
        ) from None
    try:
        port = serial.Serial(port_name, baud, timeout=timeout_s, write_timeout=timeout_s)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"cannot open {port_name}: {e}\n"
            f"Hints: is the adapter plugged in? On Linux you may need to be in "
            f"the `dialout` group. Is another process (LeRobot, the Feetech "
            f"debug tool) already holding the port?"
        ) from e
    return port, ServoBus(port, endian=endian)


def scan(bus: ServoBus, upto: int) -> list[int]:
    print(f"scanning ids 1..{upto} ...")
    found = [sid for sid in range(1, upto + 1) if bus.ping(sid)]
    print(f"  responding: {found or 'NOTHING'}")
    if not found:
        print("  -> no servo answered. Check BUS POWER first: the servos need "
              "their own supply,\n     USB alone powers only the adapter. Then "
              "check the baud rate (--baud).")
    return found


def report(bus: ServoBus, ids: list[int], args) -> None:
    print(f"\n{'id':>3s} {'joint':14s} {'counts':>7s} {'rad':>8s} {'deg':>8s} "
          f"{'in limits':>10s} {'V':>5s} {'degC':>5s} {'load':>6s}")
    for i, sid in enumerate(ids):
        name = JOINT_NAMES[i] if i < len(JOINT_NAMES) else "(gripper?)"
        sign = args.signs[i] if i < len(args.signs) else 1
        try:
            counts = bus.read_reg(sid, ADDR_PRESENT_POSITION)
            rad = counts_to_rad(counts, args.center, args.cpr, sign)
            volts = bus.read_reg(sid, ADDR_PRESENT_VOLTAGE) / 10.0
            temp = bus.read_reg(sid, ADDR_PRESENT_TEMPERATURE)
            load = sign_magnitude(bus.read_reg(sid, ADDR_PRESENT_LOAD), bits=10)
        except FeetechError as e:
            print(f"{sid:3d} {name:14s} read failed: {e}")
            continue
        if i < len(LIMITS):
            lo, hi = LIMITS[i]
            inside = "yes" if lo <= rad <= hi else "OUT"
        else:
            inside = "-"
        import math

        print(f"{sid:3d} {name:14s} {counts:7d} {rad:8.3f} "
              f"{math.degrees(rad):8.2f} {inside:>10s} {volts:5.1f} {temp:5d} {load:6d}")
    print("\n`OUT` means the servo is outside the URDF limit this repo enforces. "
          "That is\nusually a calibration offset (center_counts) rather than a "
          "bent arm -- fix it\nbefore any motion, because the safety harness "
          "will refuse to move from there.")


def watch(bus: ServoBus, ids: list[int], args) -> None:
    import math

    print("live angles (deg), Ctrl-C to stop. Back-drive the arm by hand: the "
          "numbers\nshould move in the direction the URDF calls positive.")
    try:
        while True:
            cells = []
            for i, sid in enumerate(ids):
                sign = args.signs[i] if i < len(args.signs) else 1
                try:
                    c = bus.read_reg(sid, ADDR_PRESENT_POSITION)
                    cells.append(f"{sid}:{math.degrees(counts_to_rad(c, args.center, args.cpr, sign)):+7.2f}")
                except FeetechError:
                    cells.append(f"{sid}:  ----")
            print("  " + "  ".join(cells), end="\r", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nstopped.")


def jog(bus: ServoBus, sid: int, args) -> None:
    import math

    idx = args.ids.index(sid) if sid in args.ids else 0
    name = JOINT_NAMES[idx] if idx < len(JOINT_NAMES) else f"id {sid}"
    sign = args.signs[idx] if idx < len(args.signs) else 1
    step = math.radians(args.jog_deg)

    print(f"\nJOG {name} (id {sid}) by {args.jog_deg:+.1f} deg, "
          f"wire_signs[{idx}] = {sign}")
    print("Support the arm. Ctrl-C aborts (torque is released either way).")
    try:
        input("press Enter to move, or Ctrl-C to abort: ")
    except KeyboardInterrupt:
        print("\naborted.")
        return

    before = bus.read_reg(sid, ADDR_PRESENT_POSITION)
    rad_before = counts_to_rad(before, args.center, args.cpr, sign)
    target = rad_to_counts(rad_before + step, args.center, args.cpr, sign)
    try:
        bus.write_reg(sid, ADDR_MODE, MODE_POSITION)
        bus.write_reg(sid, ADDR_TORQUE_LIMIT, int(args.torque * TORQUE_LIMIT_MAX))
        bus.write_reg(sid, ADDR_TORQUE_ENABLE, 1)
        bus.write_reg(sid, ADDR_GOAL_POSITION, target)
        time.sleep(1.2)
        after = bus.read_reg(sid, ADDR_PRESENT_POSITION)
    finally:
        # Torque off no matter what: leaving one joint energised after a
        # diagnostic is how an arm gets left fighting its own hard stop.
        try:
            bus.write_reg(sid, ADDR_TORQUE_ENABLE, 0)
        except FeetechError:
            pass

    rad_after = counts_to_rad(after, args.center, args.cpr, sign)
    moved = rad_after - rad_before
    print(f"  counts {before} -> {after}")
    print(f"  local angle {math.degrees(rad_before):+.2f} -> "
          f"{math.degrees(rad_after):+.2f} deg  (moved {math.degrees(moved):+.2f})")
    if abs(moved) < math.radians(0.5):
        print("  !! did not move. Torque limit too low, a hard stop, or the "
              "servo is unpowered.")
    elif (moved > 0) == (step > 0):
        print(f"  OK: moved the commanded direction -> wire_signs[{idx}] = "
              f"{sign} is correct.")
    else:
        print(f"  WRONG DIRECTION -> flip wire_signs[{idx}] to {-sign} in "
              f"configs/arms/so101.yaml")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True, help="e.g. /dev/ttyACM0, COM3")
    p.add_argument("--baud", type=int, default=1_000_000)
    p.add_argument("--ids", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6],
                   help="expected servo ids (default 1..6, arm then gripper)")
    p.add_argument("--scan-upto", type=int, default=12)
    p.add_argument("--signs", type=int, nargs="*", default=[1, 1, 1, 1, 1],
                   help="wire_signs from the profile, to interpret angles with")
    p.add_argument("--cpr", type=float, default=4096, help="counts per revolution")
    p.add_argument("--center", type=float, default=2048, help="counts at 0 rad")
    p.add_argument("--series", choices=["sts", "scs"], default="sts",
                   help="register byte order (STS little-endian, SCS big-endian)")
    p.add_argument("--timeout-ms", type=float, default=100)
    p.add_argument("--watch", action="store_true", help="stream angles until Ctrl-C")
    p.add_argument("--jog", type=int, metavar="ID", default=None,
                   help="MOVES one servo a few degrees to check its sign")
    p.add_argument("--jog-deg", type=float, default=5.0)
    p.add_argument("--torque", type=float, default=0.3,
                   help="torque limit fraction used for the jog only")
    args = p.parse_args(argv)

    endian = SCS if args.series == "scs" else STS
    port, bus = open_bus(args.port, args.baud, endian, args.timeout_ms / 1000.0)
    try:
        print(f"port {args.port} @ {args.baud} baud, {args.series.upper()} byte order")
        found = scan(bus, args.scan_upto)
        missing = [i for i in args.ids if i not in found]
        extra = [i for i in found if i not in args.ids]
        if missing:
            print(f"  MISSING expected id(s): {missing}")
        if extra:
            print(f"  unexpected id(s) answering: {extra} "
                  f"(unassembled servos default to id 1)")
        if not found:
            return 1
        report(bus, found, args)
        if args.jog is not None:
            if args.jog not in found:
                print(f"\nid {args.jog} is not on the bus; not jogging.")
                return 1
            jog(bus, args.jog, args)
        elif args.watch:
            watch(bus, found, args)
    finally:
        port.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
