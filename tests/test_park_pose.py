"""The shutdown park can target an exact, profile-defined rest pose.

Two load-bearing behaviors:

1. `_park_pose` prefers the profile's `park_q` (an exact zero for the reBot,
   whose joints 2/3 have their lower limit AT 0 rad) and otherwise falls back
   to the clamped all-zero pose so an arm with no `park_q` still parks inside
   its joint margin.

2. The park relaxes ONLY the joint margin (`joint_margin=0.0`) so it can reach
   that mechanical stop, while every other safety gate (workspace, table
   clearance, velocity, neighbours) still runs at full strength. A waypoint at
   the stop must be REJECTED under the default margin and ACCEPTED under the
   relaxed one -- never the reverse.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from conftest import needs_pin

from cascade.apps.demo import _park_pose
from cascade.safety.harness import SafetyHarness, SafetyLimits, SafetyViolation


class _Limits:
    joint_margin = 0.02


class _Kin:
    def __init__(self, lo, hi):
        self.joint_limits = (np.asarray(lo, dtype=float), np.asarray(hi, dtype=float))


class _Harness:
    def __init__(self, park_q=None, limits=None):
        self.park_q = park_q
        self.limits = limits
        self.kin = None


class _Arm:
    def __init__(self, park_q=None, lo=None, hi=None):
        self.n_joints = 6
        h = _Harness(park_q=np.asarray(park_q) if park_q is not None else None)
        if lo is not None:
            h.kin = _Kin(lo, hi)
        h.limits = _Limits()
        self.harness = h


#: reBot RS joint limits (joints 2 & 3 lower limit == 0 rad)
_LO = [-2.8, 0.0, 0.0, -1.69, -1.57, -3.14]
_HI = [2.8, 3.14, 3.14, 1.79, 1.57, 3.14]


def test_park_pose_prefers_profile_park_q():
    """An explicit `park_q` is returned verbatim (the exact mechanical zero)."""
    assert _park_pose(_Arm(park_q=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])) == [0.0] * 6


def test_park_pose_falls_back_to_clamped_zero():
    """No `park_q`: the all-zero fallback is clamped inside the joint margin,
    so joints 2/3 (lower limit 0) stop at +margin, not on the stop."""
    q = _park_pose(_Arm(lo=_LO, hi=_HI))
    assert q[0] == 0.0
    assert q[1] == pytest.approx(0.02)
    assert q[2] == pytest.approx(0.02)
    assert q[3] == 0.0


def test_park_pose_ignores_mis_sized_park_q():
    """A `park_q` whose length disagrees with the arm falls back, never
    broadcasts/truncates a pose across the wrong DOF."""
    q = _park_pose(_Arm(park_q=[0.0] * 3, lo=_LO, hi=_HI))
    assert len(q) == 6
    assert q[1] == pytest.approx(0.02)


@needs_pin
def test_approve_rejects_the_stop_at_default_margin_and_accepts_relaxed():
    """The mechanical stop (zero on a 0-limit joint) is rejected under the
    configured joint margin and accepted only when the park relaxes it -- with
    the workspace/table/velocity gates still passing for the zero pose."""
    from cascade.config import load_demo_config
    from cascade.control.kinematics import Kinematics

    cfg = load_demo_config(camera="mock", arm="rebot_rs", llm="mock")
    a = cfg.arm
    kin = Kinematics(
        a.model, ee_frame=a.get("ee_frame"),
        n_controlled=int(a.get("n_joints", 6)),
        joint_signs=a.get("joint_signs"),
    )
    h = SafetyHarness(SafetyLimits.from_config(cfg.safety), kinematics=kin)
    h._motion_active = True
    h._last_heartbeat = time.monotonic()

    zero = np.zeros(6)
    # default margin (0.02): joints 2/3 at 0.0 are outside [0.02, ...]
    with pytest.raises(SafetyViolation):
        h.approve(zero, zero, 0.0)
    # relaxed margin: reaches the stop, and every other gate still passes
    h.approve(zero, zero, 0.0, joint_margin=0.0)  # must not raise
