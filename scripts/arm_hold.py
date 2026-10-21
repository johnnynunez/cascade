"""Held arms: list them, or confirm you are supporting one (cascade.control.arm_hold).

A teardown (or an arm-check, a calibration, a bring-up script) that leaves a reBot or SO-101
POWERED and HOLDING remembers it, and the next connect of that arm refuses until an operator
confirms, at that moment, at a terminal on the robot host, that they support it:

    python scripts/arm_hold.py list      # exit 0: nothing held; 3: an arm is held, or it cannot tell
    python scripts/arm_hold.py confirm   # your sudo password, then the phrase; valid 120 s, once
    python scripts/arm_hold.py confirm --arm can0
    python scripts/arm_hold.py revoke    # drop every pending confirmation

`confirm` asks for YOUR sudo password first (`sudo -k -v`: a sudo that needs no password is
refused), then reads the phrase from the controlling terminal; it refuses without one, or in an
agent's or the robot server's environment: only the operator confirms. Then retry the motion (or
the arm-check / calibrate): its connect cuts the torque while you hold the arm, then HOLDS it and
moves nothing ("connected and holding; let go of the arm and retry"): let go, and the next call
moves. The same confirmation, given right before a robot-server or demo CLI stop, lets that
teardown cut the torque of an arm its park left raised.

The files are the robot user's: any process running as that user can still delete a marker or
write a confirmation directly. The boundary against an agent is the OpenShell sandbox. Your sudo
password is the one factor an agent lacks: never give it to Hermes or any agent on a robot host (no
`SUDO_PASSWORD` in `~/.hermes/.env` or any agent's environment): with it, an agent can confirm.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from cascade.control.arm_hold import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
