"""The reBot RS leaves its real rest pose for home through the harness, with
no bypass -- and its shutdown park target is reachable through it too.

Rig finding (Seeed WRC fork, docs/HARDWARE_VERIFICATION_HANDOVER.md, Finding
#3): at mechanical zero joint 2 READS +0.004 rad, below the URDF lower limit
(0) plus the harness joint margin (0.02). WRC's harness vetoed every
trajectory starting there, and WRC "fixed" it by skipping the harness check
for the first waypoint (`SafeArm._make_hook` first_seen) -- a bypass cascade
must not copy. cascade's `SafetyHarness.approve` already has a principled
escape rule: a joint at/outside its margin may move STRICTLY back toward the
valid band, while holds and moves deeper stay rejected. These tests prove,
on the real RS kinematics and the real harness, that this rule is enough:

- from the measured rest pose (and with joint 3 a hair negative too, the
  same encoder-offset story) SafeArm reaches `home_q` through both the
  streamed `move_joints` and the route-vetted `move_planned` (which uses
  `vet_step`, i.e. the same escape rules), with nothing recorded as a
  violation;
- the escape is directional, not a skip: holding the rest pose, or sinking
  further, is still refused;
- the profile's `park_q` (the shutdown park, which relaxes ONLY the joint
  margin to 0 so it can reach the stop) lies inside the URDF limits, so the
  park completes instead of being rejected a few mrad before the end. It
  used to carry joint 3 = -0.05, below joint 3's lower limit of 0: the park
  was refused at joint 3 = -0.001 with joint 2 still at ~0.05 rad, the
  gripper park never ran, and teardown reported ParkIncomplete.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from conftest import needs_pin

from cascade.config import Cfg, load_demo_config
from cascade.types import SafetyViolation

RS_PROFILES = ["rebot_rs", "rebot_rs_mb"]

#: what the real RS arm reports at mechanical zero (WRC handover, Finding #3)
MEASURED_REST = [0.0, 0.004, 0.0, 0.0, 0.0, 0.0]
#: same encoder-offset story with joint 3 a hair below its zero
REST_J3_NEGATIVE = [0.0, 0.004, -0.003, 0.0, 0.0, 0.0]
#: cascade's own rig, 2026-08-27 (scripts/jog_rebot_mb.py): joints 2 and 3
#: rested at -0.0009 rad, i.e. just past their 0 lower limit
RIG_0827_REST = [0.0, -0.0009, -0.0009, 0.0, 0.0, 0.0]


def _rig(profile, q0):
    from cascade.control.kinematics import Kinematics
    from cascade.control.mock_arm import MockArm
    from cascade.safety.harness import SafeArm, SafetyHarness, SafetyLimits

    cfg = load_demo_config(arm=profile)
    a = cfg.arm
    kin = Kinematics(a.model, a.ee_frame, n_controlled=int(a.n_joints),
                     joint_signs=a.joint_signs)
    raw = MockArm(Cfg({"home_q": list(q0), "n_joints": int(a.n_joints)}), kin)
    raw.connect()
    harness = SafetyHarness(SafetyLimits.from_config(cfg.safety), kinematics=kin)
    if a.get("park_q") is not None:
        harness.park_q = list(a.park_q)
    return cfg, kin, raw, harness, SafeArm(raw, harness)


@needs_pin
@pytest.mark.parametrize("profile", RS_PROFILES)
@pytest.mark.parametrize("start", ["measured_rest", "rest_j3_negative", "rig_0827_rest", "park_q"])
@pytest.mark.parametrize("method", ["move_joints", "move_planned"])
def test_rest_pose_reaches_home_through_the_harness(profile, start, method):
    cfg = load_demo_config(arm=profile)
    q0 = {"measured_rest": MEASURED_REST, "rest_j3_negative": REST_J3_NEGATIVE,
          "rig_0827_rest": RIG_0827_REST,
          "park_q": cfg.arm.get("park_q") or MEASURED_REST}[start]
    _, kin, raw, harness, arm = _rig(profile, q0)
    lo, _ = kin.joint_limits
    m = harness.limits.joint_margin
    # premise: the start really is inside the margin band the harness guards
    assert np.any(np.asarray(q0) < lo + m), "start pose must sit below a joint margin"

    home = np.asarray(cfg.arm.home_q, dtype=float)
    assert getattr(arm, method)(home, duration_s=3.0) is True
    np.testing.assert_allclose(raw.get_state().q, home, atol=1e-9)
    assert harness.violations == []
    assert not harness.estopped


@needs_pin
@pytest.mark.parametrize("profile", RS_PROFILES)
def test_the_escape_is_directional_not_a_skip(profile):
    """No first-waypoint bypass: holding the out-of-margin rest pose, or
    sinking joint 2 further toward its stop, is still refused."""
    _, _, _, harness, _ = _rig(profile, MEASURED_REST)
    rest = np.asarray(MEASURED_REST)
    with pytest.raises(SafetyViolation, match="joint 2"):
        harness.approve(rest, rest.copy(), 0.02)
    deeper = rest.copy()
    deeper[1] = 0.002
    with pytest.raises(SafetyViolation, match="joint 2"):
        harness.approve(rest, deeper, 0.02)
    up = rest.copy()
    up[1] = 0.005  # joint 2 strictly toward its band ...
    up[2] = 0.001  # ... and joint 3, which also rests below its margin
    harness.approve(rest, up, 0.02)  # every violated joint escaping: allowed
    only_j2 = rest.copy()
    only_j2[1] = 0.005
    with pytest.raises(SafetyViolation):
        # joint 3 holds at its violation while joint 2 escapes: refused
        harness.approve(rest, only_j2, 0.02)


def _profiles_with_park_q():
    from conftest import REPO

    out = []
    for path in sorted((REPO / "configs" / "arms").glob("*.yaml")):
        text = path.read_text()
        if "park_q:" in text and "model:" in text:
            out.append(path.stem)
    return out


@needs_pin
@pytest.mark.parametrize("profile", _profiles_with_park_q())
def test_profile_park_q_lies_inside_the_urdf_limits(profile):
    """The park relaxes only the joint MARGIN (to 0); it never relaxes the
    URDF limits themselves, so a park target outside them is unreachable."""
    from cascade.control.kinematics import Kinematics

    a = load_demo_config(arm=profile).arm
    kin = Kinematics(a.model, a.ee_frame, n_controlled=int(a.n_joints),
                     joint_signs=a.get("joint_signs"))
    lo, hi = kin.joint_limits
    park = np.asarray(a.park_q, dtype=float)
    assert len(park) == int(a.n_joints)
    bad = [j + 1 for j in range(len(park)) if not lo[j] - 1e-9 <= park[j] <= hi[j] + 1e-9]
    assert not bad, (f"{profile}: park_q joints {bad} outside the URDF limits "
                     f"{np.round(lo, 3).tolist()}..{np.round(hi, 3).tolist()}")


@needs_pin
def test_rebot_rs_shutdown_park_completes_from_home():
    """The real teardown path: `_park_arm` drives home -> park_q with only the
    joint margin relaxed, and must finish (receipt complete) at park_q."""
    from cascade.apps.demo import _park_arm

    cfg, _, raw, harness, arm = _rig("rebot_rs", load_demo_config(arm="rebot_rs").arm.home_q)
    receipt = _park_arm(SimpleNamespace(arm=arm, arm_rig=None), duration_s=2.0)
    assert receipt["complete"] is True, receipt
    np.testing.assert_allclose(raw.get_state().q, cfg.arm.park_q, atol=1e-9)
    assert harness.violations == []
