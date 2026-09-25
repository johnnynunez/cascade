"""Gravity-compensation feedforward for the reBot RS arm.

The reBot holds in pure PD (MIT tau_ff=0), so a gravity-loaded joint sags by
g/kp at rest -- the "joint 3 drops then recovers" droop the park exposed. This
adds Pinocchio's g(q) as the MIT `tau` term on every arm command, so the
steady-state error goes to ~0.

Two things are load-bearing and pinned here:

1. The SIGN. The shipped RS URDF is authored in the mirrored convention
   (q_asset = -q_local) and `joint_signs: [-1]*6` re-signs the model's axes to
   the local/motor convention at load time. Gravity torque transforms as a
   covector under that flip (tau_motor = -tau_asset), so
   `Kinematics.gravity_torque(q_local)` must equal -g_raw(q_asset) on the
   un-flipped model. A wrong sign would *add* to the sag instead of cancelling
   it, so the reparameterization identity is asserted exactly.

2. The gating. Gravity compensation is OFF unless the profile sets
   `gravity_comp.enabled: true`, so mock/other backends keep the pure-PD path
   (`tau=None` -> the SDK's zeros default).
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import JOINT_SIGNS, URDF, needs_pin

from cascade.config import Cfg
from cascade.control.rebot_rs_arm import RebotRSArm


class _Kin:
    """Stub Kinematics returning a fixed g(q), so the send path is tested
    without pinocchio or the URDF."""

    def __init__(self, g):
        self._g = np.asarray(g, dtype=float)

    def gravity_torque(self, q):
        return self._g.copy()


class _ArmGroup:
    def __init__(self):
        self.calls = []

    def send_mit(self, *a, **kw):
        self.calls.append(kw)


class _FakeArm:
    has_gripper = False

    def __init__(self):
        self.arm = _ArmGroup()


def _gc_arm(monkeypatch, enabled=True, tau_scale=None):
    gc = {"enabled": enabled}
    if tau_scale is not None:
        gc["tau_scale"] = tau_scale
    arm = RebotRSArm(Cfg({"gravity_comp": gc}))
    arm._arm = _FakeArm()
    arm._mit_kp = np.ones(6)
    arm._mit_kd = np.full(6, 0.4)
    return arm


def test_send_joint_target_passes_gravity_tau_when_enabled(monkeypatch):
    """tau = g(q) * tau_scale is passed to send_mit when gravity comp is on."""
    arm = _gc_arm(monkeypatch, tau_scale=[1.0, 1.1, 1.1, 1.0, 1.0, 1.0])
    arm._gc_kin = _Kin([0.0, 3.0, 6.0, 2.0, 0.0, 0.0])
    arm.send_joint_target(np.array([0.0, 1.2, 1.2, 0.0, 0.0, 0.0]))
    (kw,) = arm._arm.arm.calls
    np.testing.assert_allclose(kw["tau"], [0.0, 3.3, 6.6, 2.0, 0.0, 0.0])


def test_send_joint_target_omits_tau_when_disabled(monkeypatch):
    """Disabled -> pure PD: send_mit is called without a tau kwarg, so the
    SDK's zeros default applies (identical to the pre-feedforward call)."""
    arm = _gc_arm(monkeypatch, enabled=False)
    arm.send_joint_target(np.zeros(6))
    (kw,) = arm._arm.arm.calls
    assert "tau" not in kw


def test_stop_reasserts_hold_with_tau(monkeypatch):
    """A soft stop re-asserts the frozen pose WITH the feedforward, so the
    held pose does not droop once streaming ends."""
    arm = _gc_arm(monkeypatch, tau_scale=1.0)
    arm._gc_kin = _Kin([0.0, 3.0, 6.0, 2.0, 0.0, 0.0])
    arm._last_cmd_q = np.array([0.0, 1.2, 1.2, 0.0, 0.0, 0.0])
    arm.stop()
    (kw,) = arm._arm.arm.calls
    np.testing.assert_allclose(kw["tau"], [0.0, 3.0, 6.0, 2.0, 0.0, 0.0])


def test_tau_scale_resolution():
    """tau_scale defaults to all-ones, broadcasts a scalar, and rejects a
    wrong-length list."""
    assert RebotRSArm(Cfg({}))._gc_enabled is False
    np.testing.assert_allclose(RebotRSArm(Cfg({}))._gc_tau_scale, np.ones(6))
    np.testing.assert_allclose(
        RebotRSArm(Cfg({"gravity_comp": {"tau_scale": 1.5}}))._gc_tau_scale,
        np.full(6, 1.5),
    )
    np.testing.assert_allclose(
        RebotRSArm(Cfg({"gravity_comp": {"tau_scale": [1, 2, 3, 4, 5, 6]}}))._gc_tau_scale,
        [1, 2, 3, 4, 5, 6],
    )
    with pytest.raises(ValueError):
        RebotRSArm(Cfg({"gravity_comp": {"tau_scale": [1, 2, 3]}}))


@needs_pin
def test_gravity_torque_sign_matches_reparameterization():
    """Kinematics.gravity_torque(q_local) == -g_raw(q_asset) on the un-flipped
    model -- the exact covector identity under the joint_signs mirror. A sign
    error here would feed the sag back into the arm instead of cancelling it."""
    import pinocchio as pin

    from cascade.control.kinematics import Kinematics
    from cascade.control.usd_model import apply_joint_signs

    raw_model = pin.buildModelFromXML(URDF.read_text())
    raw_data = raw_model.createData()

    kin = Kinematics(
        str(URDF), ee_frame="gripper_end", n_controlled=6,
        joint_signs=JOINT_SIGNS,
    )
    q = np.array([0.0, 1.2, 1.2, 0.0, 0.0, 0.0])
    g_kin = kin.gravity_torque(q)

    q_asset = np.zeros(raw_model.nq)
    q_asset[:6] = -q
    pin.computeGeneralizedGravity(raw_model, raw_data, q_asset)
    g_asset = np.asarray(raw_data.g)[:6]

    np.testing.assert_allclose(g_kin, -g_asset, atol=1e-6)


@needs_pin
def test_gravity_torque_is_physical():
    """At the raised home pose the gravity-loaded joints carry a real, correct
    load: base rotation (vertical axis) and wrist roll are ~0, while shoulder
    lift / elbow / wrist pitch are substantial and the elbow (joint 3, the one
    that drooped) is the largest."""
    from cascade.control.kinematics import Kinematics

    kin = Kinematics(
        str(URDF), ee_frame="gripper_end", n_controlled=6,
        joint_signs=JOINT_SIGNS,
    )
    g = kin.gravity_torque(np.array([0.0, 1.2, 1.2, 0.0, 0.0, 0.0]))
    assert abs(g[0]) < 1e-4, f"vertical base joint must carry no load: {g[0]}"
    assert abs(g[4]) < 1e-4, f"wrist roll must carry no load: {g[4]}"
    for i in (1, 2, 3):
        assert abs(g[i]) > 1.0, f"joint {i} must be gravity-loaded: {g[i]}"
    assert abs(g[2]) >= abs(g[1]), "the elbow is the heaviest load"
