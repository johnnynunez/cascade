"""Late regression pins (v3, 2026-07-20) for 2026-07-16 adversarial-review
fixes that shipped without dedicated tests: the IK-vs-harness joint-margin
invariant, bounded stale-CAN feedback masking, and the soft-stop torque hold
(no free-fall). All run against fakes -- no SDK, CAN bus or pinocchio."""

import dataclasses
import inspect

import numpy as np
import pytest

from wrc_demo.config import Cfg, load_demo_config
from wrc_demo.control.kinematics import Kinematics
from wrc_demo.control.rebot_rs_arm import RebotRSArm
from wrc_demo.safety.harness import SafetyLimits


# ── IK-vs-harness margin invariant ─────────────────────────────────────────


def test_ik_margin_strictly_exceeds_harness_joint_margin():
    """Review defect: IK returned solutions inside the URDF limits but inside
    the harness margin band, so approve() rejected poses IK had just declared
    feasible (mid-stream aborts). Fix: ik() clamps limit_margin strictly
    GREATER than the harness joint_margin. Pin the invariant against both the
    SafetyLimits default and the shipped configs/demo.yaml value -- if either
    side is tuned past the other, this fails before the rig does."""
    ik_margin = inspect.signature(Kinematics.ik).parameters["limit_margin"].default
    default_margin = next(
        f.default for f in dataclasses.fields(SafetyLimits) if f.name == "joint_margin"
    )
    assert ik_margin > default_margin

    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    harness_margin = SafetyLimits.from_config(cfg.safety).joint_margin
    assert ik_margin > harness_margin


# ── stale-CAN feedback masking (RebotRSArm.get_state) ──────────────────────


def _bare_rs_arm() -> RebotRSArm:
    """RebotRSArm with the attributes connect() would set, minus hardware."""
    arm = RebotRSArm(Cfg({}))
    arm._read_failures = 0
    arm._last_cmd_q = None
    return arm


def test_stale_can_feedback_masked_then_fails_loudly(monkeypatch):
    """Review defect: mechPos read hiccups were masked indefinitely by
    serving last-known positions, so a dead bus let motion plan from phantom
    joint states. Fix: serve last-known q for at most MAX_READ_FAILURES - 1
    consecutive failures, then raise RuntimeError so motion aborts."""
    arm = _bare_rs_arm()
    known = np.linspace(0.1, 0.6, arm.n_joints)
    arm._last_q = known.copy()

    def boom():
        raise OSError("mechPos read timeout")

    monkeypatch.setattr(arm, "_read_positions", boom)

    for _ in range(RebotRSArm.MAX_READ_FAILURES - 1):
        st = arm.get_state()  # transient hiccup: masked with last-known q
        assert np.allclose(st.q, known)
        assert st.gripper_valid is False  # unknown, never served as a pose

    with pytest.raises(RuntimeError, match="CAN feedback lost"):
        arm.get_state()


def test_can_feedback_failure_counter_resets_on_success(monkeypatch):
    """One successful read re-arms the full masking budget; only CONSECUTIVE
    failures count toward the abort threshold."""
    arm = _bare_rs_arm()
    arm._last_q = np.zeros(arm.n_joints)
    fresh = np.full(arm.n_joints, 0.5)
    failing = {"on": True}

    def read():
        if failing["on"]:
            raise OSError("hiccup")
        return fresh.copy()

    monkeypatch.setattr(arm, "_read_positions", read)

    arm.get_state()  # failure 1 of 3: masked
    failing["on"] = False
    st = arm.get_state()  # success resets the counter
    assert np.allclose(st.q, fresh)
    assert arm._read_failures == 0

    failing["on"] = True
    arm.get_state()  # masked again (1 of 3)
    arm.get_state()  # masked again (2 of 3)
    with pytest.raises(RuntimeError, match="CAN feedback lost"):
        arm.get_state()


# ── soft stop holds pose instead of torque-off (free-fall) ─────────────────


class _FakeGroup:
    def __init__(self):
        self.sent = []
        self.gains = []

    def send_mit(self, q, kp=None, kd=None):
        self.sent.append(np.asarray(q, dtype=float).copy())
        self.gains.append((None if kp is None else np.asarray(kp, dtype=float).copy(),
                           None if kd is None else np.asarray(kd, dtype=float).copy()))


class _FakeSdkArm:
    has_gripper = False

    def __init__(self):
        self.arm = _FakeGroup()
        self.estop_calls = 0

    def estop(self):
        self.estop_calls += 1


def test_soft_stop_holds_pose_instead_of_torque_off():
    """Review defect: stop() went through the SDK's estop (disable_all), so
    any soft stop dropped a loaded arm under gravity. Fix: stop() re-commands
    the last MIT target (holding torque) and refuses new commands; it must
    never touch the SDK estop."""
    arm = _bare_rs_arm()
    fake = _FakeSdkArm()
    arm._arm = fake
    arm._mit_kp = np.full(arm.n_joints, 5.0)
    arm._mit_kd = np.full(arm.n_joints, 0.5)

    q0 = np.linspace(0.1, 0.6, arm.n_joints)
    arm.send_joint_target(q0)
    assert len(fake.arm.sent) == 1

    arm.stop()
    assert fake.estop_calls == 0  # never disable_all on a soft stop
    assert len(fake.arm.sent) == 2  # hold pose re-commanded...
    assert np.allclose(fake.arm.sent[-1], q0)
    kp, kd = fake.arm.gains[-1]  # ...WITH the MIT gains: zero/absent
    assert kp is not None and np.allclose(kp, arm._mit_kp)  # stiffness would
    assert kd is not None and np.allclose(kd, arm._mit_kd)  # sag under gravity

    arm.send_joint_target(np.zeros(arm.n_joints))  # refused while stopped
    assert len(fake.arm.sent) == 2


def test_hard_estop_is_the_explicit_torque_off_path():
    """hard_estop() is the only path that torques off (arm WILL fall)."""
    arm = _bare_rs_arm()
    fake = _FakeSdkArm()
    arm._arm = fake

    arm.hard_estop()
    assert fake.estop_calls == 1
    assert arm._stopped
