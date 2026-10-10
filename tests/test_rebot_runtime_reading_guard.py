"""The runtime path never acts on a reading that cannot be a position.

Companion to test_rebot_probe_reading_guard.py (the 2026-10-10 incident:
a mechPos read of +2.3e18 was commanded). The bring-up scripts are not the
only consumers of mechPos: both RS drivers feed it to the planner as the
start of every route (get_state), the motorbridge driver commands it as the
hold pose at connect -- before enable_all -- and the SafetyHarness vets the
result. Pinned here:

* SafetyHarness.approve / vet_pose reject a non-finite joint vector. NaN
  compares False against every limit, so before this a NaN pose passed the
  joint-limit, velocity and workspace checks alike.
* A driver reading that is non-finite or outside the motor's +-4*pi counts
  as a failed read: get_state serves the last good pose and raises after
  MAX_READ_FAILURES, exactly like a CAN timeout -- it never returns it.
* So does a reading that jumps farther than the arm can physically travel
  since the last good one.
* The motorbridge driver refuses to connect -- nothing commanded, nothing
  enabled, bus released -- when the rest pose it would hold is implausible.
* An implausible gripper reading is "unknown" (None), never a position.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from test_rebot_clear_error import (
    _install_fake_motorbridge,
    _install_fake_sdk,
    _Log,
    _mb_arm,
    _Motor,
    _rs_arm,
)
from test_safety import FakeKin, limits

from cascade.safety.harness import SafetyHarness
from cascade.types import SafetyViolation

GARBAGE = 2.3354605539310961e18


# ── harness ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kin", [None, FakeKin()])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_harness_rejects_a_non_finite_target(kin, bad):
    h = SafetyHarness(limits(), kinematics=kin)
    ok = np.array([0.3, 0.0, 0.2, 0, 0, 0])
    q = ok.copy()
    q[1] = bad
    with pytest.raises(SafetyViolation, match="non-finite"):
        h.approve(ok, q, 0.02)
    with pytest.raises(SafetyViolation, match="non-finite"):
        h.approve(q, ok, 0.02)             # nor plan from a non-finite start
    assert "non-finite" in (h.vet_step(ok, q, 0.02) or "")


def test_harness_vet_pose_rejects_a_non_finite_pose():
    h = SafetyHarness(limits(), kinematics=FakeKin())
    q = np.array([0.3, 0.0, 0.2, 0, float("nan"), 0])
    assert "non-finite" in (h.vet_pose(q) or "")
    assert h.vet_pose(np.array([0.3, 0.0, 0.2, 0, 0, 0])) is None


# ── RebotRSArm (reBotArm_control_py SDK) ─────────────────────────────────


@pytest.fixture
def readings(monkeypatch):
    """mechPos per motor name/id, settable by the test."""
    values = {}

    def read(self, param, *a):
        return values.get(self.name, 0.0)
    monkeypatch.setattr(_Motor, "robstride_get_param_f32", read)
    return values


@pytest.fixture
def sent(monkeypatch):
    log = []

    def send_mit(self, *a, **kw):
        log.append((self.name, a, kw))
    monkeypatch.setattr(_Motor, "send_mit", send_mit)
    return log


@pytest.fixture
def no_sleep(monkeypatch):
    from cascade.control import robstride

    monkeypatch.setattr(robstride.time, "sleep", lambda s: None)


def _connected_rs(monkeypatch, readings):
    _install_fake_sdk(monkeypatch, _Log())
    arm = _rs_arm()
    arm.connect()
    return arm


@pytest.mark.parametrize("bad", [GARBAGE, float("nan"), float("inf"), 13.0])
def test_rs_get_state_never_returns_an_implausible_position(monkeypatch, no_sleep,
                                                            readings, bad):
    arm = _connected_rs(monkeypatch, readings)
    readings["joint2"] = 0.1           # (within the continuity bound of the rest read)
    good = arm.get_state().q
    assert good[1] == pytest.approx(0.1)
    readings["joint2"] = bad
    for _ in range(arm.MAX_READ_FAILURES - 1):
        q = arm.get_state().q
        assert np.all(np.isfinite(q)) and np.allclose(q, good)     # last good pose
    with pytest.raises(RuntimeError, match="implausible"):
        arm.get_state()


def test_rs_get_state_rejects_a_physically_impossible_jump(monkeypatch, no_sleep, readings):
    arm = _connected_rs(monkeypatch, readings)
    readings["joint3"] = 0.10
    arm.get_state()
    readings["joint3"] = 2.6           # plausible value, impossible 2.5 rad step in ms
    q = arm.get_state().q
    assert q[2] == pytest.approx(0.10)
    readings["joint3"] = 0.11           # back to sane: accepted, counter reset
    assert arm.get_state().q[2] == pytest.approx(0.11)
    assert arm._read_failures == 0


def test_rs_gripper_reading_that_is_not_a_position_is_unknown(monkeypatch, no_sleep, readings):
    arm = _connected_rs(monkeypatch, readings)
    readings["gripper"] = GARBAGE
    st = arm.get_state()
    assert st.gripper_valid is False and math.isfinite(st.gripper_pos)


# ── RebotRSMotorBridgeArm (motorbridge directly) ─────────────────────────


@pytest.fixture
def mb_readings(monkeypatch):
    values = {}

    def read(self, param, *a):
        return values.get(self.name, 0.0)
    monkeypatch.setattr(_Motor, "robstride_get_param_f32", read)
    return values


@pytest.mark.parametrize("bad", [GARBAGE, float("nan")])
def test_mb_connect_refuses_to_hold_an_implausible_rest_pose(monkeypatch, no_sleep,
                                                             mb_readings, sent, bad):
    log = _Log()
    _install_fake_motorbridge(monkeypatch, log)
    mb_readings[2] = bad
    arm = _mb_arm()
    with pytest.raises(RuntimeError, match="implausible"):
        arm.connect()
    assert not any(ev[0] == "enable_all" for ev in log)
    assert sent == []                          # nothing commanded, garbage least of all
    assert ("close_bus", None) in log and arm._ctrl is None


def test_mb_get_state_never_returns_an_implausible_position(monkeypatch, no_sleep,
                                                            mb_readings, sent):
    _install_fake_motorbridge(monkeypatch, _Log())
    arm = _mb_arm()
    arm.connect()
    good = arm.get_state().q
    mb_readings[5] = GARBAGE
    for _ in range(arm.MAX_READ_FAILURES - 1):
        assert np.allclose(arm.get_state().q, good)
    with pytest.raises(RuntimeError, match="implausible"):
        arm.get_state()
    assert all(np.all(np.abs(np.asarray(a[0], dtype=float)) < 13.0) for _, a, _ in sent if a)


@pytest.mark.parametrize("bad", [GARBAGE, float("nan")])
def test_rs_connect_refuses_before_enable_when_the_rest_pose_is_implausible(
        monkeypatch, no_sleep, readings, bad):
    log = _Log()
    _install_fake_sdk(monkeypatch, log)
    readings["joint2"] = bad
    arm = _rs_arm()
    with pytest.raises(RuntimeError, match="implausible"):
        arm.connect()
    assert not any(ev[0] == "enable" for ev in log)
    assert ("disconnect", None) in log and arm._arm is None


def test_mb_gripper_reading_that_is_not_a_position_is_unknown(monkeypatch, no_sleep,
                                                             mb_readings, sent):
    _install_fake_motorbridge(monkeypatch, _Log())
    arm = _mb_arm()
    arm.connect()
    mb_readings[7] = float("nan")
    st = arm.get_state()
    assert st.gripper_valid is False and math.isfinite(st.gripper_pos)
