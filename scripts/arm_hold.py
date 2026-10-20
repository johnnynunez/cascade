"""Held arms: list them, or confirm you are supporting one (cascade.control.arm_hold).

A teardown whose park did not bring a reBot or SO-101 to rest keeps it POWERED and HOLDING, and the
next connect of that arm refuses until an operator confirms, at that moment, that they support it:

    python scripts/arm_hold.py list
    python scripts/arm_hold.py confirm 'holding the arm'          # any held arm, once, within 120 s
    python scripts/arm_hold.py confirm 'holding the arm' --arm can0

Then retry the motion (or the arm-check / calibrate): its connect cuts the torque while you hold the
arm. The same confirmation, written right before a robot-server stop, lets that teardown cut the
torque of an arm its park left raised.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from cascade.control.arm_hold import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
