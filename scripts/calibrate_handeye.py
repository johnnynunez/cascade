#!/usr/bin/env python3
"""Hand-eye calibration for a cascade camera profile (ArUco, joint SE(3) solve).

Ported from Seeed's WRC fork (scripts/calib_top_orbbec.py for the fixed top
camera, scripts/calib_wrist.py for the wrist camera) as one camera- and
arm-agnostic command. The camera profile's ``extrinsics.mode`` selects the
mounting; every arm motion is vetted by the arm profile's SafetyHarness and
executed through SafeArm. Same CLI as the ``cascade-calib-handeye`` console
script. Full procedure: docs/HANDEYE_CALIBRATION.md.

    # which camera is which (two identical D455Fs: the serial decides)
    python scripts/calibrate_handeye.py --list
    python scripts/calibrate_handeye.py --bind d455f_scene --serial 261422303968

    # no hardware: MockArm + a rendered marker through the same code path
    python scripts/calibrate_handeye.py --dry-run

    # fixed top camera (marker on top of the gripper)
    python scripts/calibrate_handeye.py --camera d455f_scene --arm rebot_rs
    # wrist camera (marker flat on the table)
    python scripts/calibrate_handeye.py --camera d455f_wrist --arm rebot_rs

    # then check it against marker positions measured on the table
    python scripts/calibrate_handeye.py --verify configs/calib/d455f_scene_rebot_rs.handeye.json \\
        --camera d455f_scene --known-points "0.30,0,0;0.25,-0.05,0;0.35,0.05,0"
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from cascade.calibration.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
