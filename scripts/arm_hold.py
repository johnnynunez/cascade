"""Held arms: list them, or confirm you are supporting one (cascade.control.arm_hold).

A teardown (or an arm-check, a calibration, a bring-up script) that leaves a reBot or SO-101
POWERED and HOLDING remembers it, and the next connect of that arm refuses until an operator
confirms, at that moment, at a terminal on the robot host, that they support it:

    python scripts/arm_hold.py list      # exit 0: nothing held; 3: an arm is held, or it cannot tell
    python scripts/arm_hold.py confirm   # you type the phrase at the prompt; valid 120 s, once
    python scripts/arm_hold.py confirm --arm can0
    python scripts/arm_hold.py revoke    # drop every pending confirmation

`confirm` reads the phrase from the controlling terminal and refuses without one, or in an agent's
or the robot server's environment: only the operator confirms. Then retry the motion (or the
arm-check / calibrate): its connect cuts the torque while you hold the arm. The same
confirmation, given right before a robot-server or demo CLI stop, lets that teardown cut the
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
