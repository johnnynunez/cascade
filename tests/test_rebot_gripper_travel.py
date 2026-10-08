"""Gripper commands on the reBot RS never leave the profile's measured travel.

Ported from Seeed WRC `src/wrc_demo/control/gripper.py` (WrcGripper
`_send_gripper_mit`, itself from rebot_grasp's GraspDriver), which clips every
gripper position target to the open soft limit before it reaches the motor.
On this rig the jaw runs 0.0 (closed) -> +6.39 rad (open hard end) and the
profile's `open_pos: 6.2` keeps margin off that end; a target past either
end drives the jaw into a hard stop at full MIT stiffness (the DM-polarity
`-6.8` did exactly that). Both RS transports therefore clamp the target to
[min(open_pos, closed_pos), max(open_pos, closed_pos)] -- polarity-aware, so
a profile with the opposite sign is clamped the same way.

Offline: fake gripper groups/motors record what would go on the wire.
"""

from __future__ import annotations

import numpy as np
import pytest

from cascade.config import Cfg


class _GripGroup:
    def __init__(self):
        self.sent = []
        self._mm = {"gripper": object()}
        self.joint_names = ["gripper"]

    def send_mit(self, pos, kp=None, kd=None, **kw):
        self.sent.append((float(np.asarray(pos).reshape(-1)[0]),
                          float(np.asarray(kp).reshape(-1)[0])))


class _FakeSdkArm:
    has_gripper = True

    def __init__(self):
        self.gripper = _GripGroup()


def _rs(open_pos=6.2, closed_pos=0.0):
    from cascade.control.rebot_rs_arm import RebotRSArm

    arm = RebotRSArm(Cfg({"gripper": {"open_pos": open_pos, "closed_pos": closed_pos,
                                      "kp": 2.0, "kd": 0.4}}))
    arm._arm = _FakeSdkArm()
    return arm


@pytest.mark.parametrize("cmd, expected", [
    (7.0, 6.2), (6.39, 6.2), (3.1, 3.1), (0.0, 0.0), (-0.5, 0.0), (-6.8, 0.0),
])
def test_rs_set_gripper_clamps_to_profile_travel(cmd, expected):
    arm = _rs()
    arm.set_gripper(cmd, effort=1.0)
    assert arm._arm.gripper.sent[-1][0] == pytest.approx(expected)


def test_rs_light_hold_path_is_clamped_too():
    arm = _rs()
    arm._set_gripper_kp(-1.0, 0.01)
    assert arm._arm.gripper.sent[-1] == (pytest.approx(0.0), pytest.approx(0.01))


def test_rs_clamp_is_polarity_aware():
    """A profile whose jaw opens toward NEGATIVE angles is clamped to its
    own [open, closed] interval, not to a hard-coded positive range."""
    arm = _rs(open_pos=-6.2, closed_pos=0.0)
    arm.set_gripper(-7.0)
    arm.set_gripper(0.4)
    assert [p for p, _ in arm._arm.gripper.sent] == [pytest.approx(-6.2), pytest.approx(0.0)]


class _MbMotor:
    def __init__(self):
        self.sent = []

    def send_mit(self, pos, vel, kp, kd, tau):
        self.sent.append(float(pos))


def test_mb_set_gripper_clamps_to_profile_travel():
    from cascade.control.rebot_rs_mb_arm import RebotRSMotorBridgeArm

    arm = RebotRSMotorBridgeArm(Cfg({
        "n_joints": 6, "gripper_id": 7, "mit_kp": [1.0] * 6, "mit_kd": [0.1] * 6,
        "gripper": {"open_pos": 6.2, "closed_pos": 0.0},
    }))
    motor = _MbMotor()
    arm._ctrl = object()
    arm._motors = {7: motor}
    for cmd in (9.0, 2.0, -3.0):
        arm.set_gripper(cmd)
    assert motor.sent == [pytest.approx(6.2), pytest.approx(2.0), pytest.approx(0.0)]


def test_rs_gripper_defaults_are_the_measured_rs_polarity_not_dm():
    """Seeed WRC e3b0b2a: the RS gripper opens toward POSITIVE angles; the DM
    build's open=-6.8 had been copied over and drove the jaws the wrong way.
    A profile that omits the gripper block must fall back to the measured RS
    travel (the motorbridge sibling already does), never to the DM value."""
    from cascade.control.rebot_rs_arm import RebotRSArm
    from cascade.control.rebot_rs_mb_arm import RebotRSMotorBridgeArm

    rs = RebotRSArm(Cfg({}))
    mb = RebotRSMotorBridgeArm(Cfg({"mit_kp": [1.0] * 6, "mit_kd": [0.1] * 6}))
    assert (rs._grip_open, rs._grip_closed) == (mb._grip_open, mb._grip_closed) == (6.2, 0.0)
