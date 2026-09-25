"""The reBot park closes the gripper under torque/stall detection, then
relaxes to a light constant hold so a grasped object is not crushed.

Offline: `close_gripper_torque` is exercised against a fake backend with a
scripted mechPos (0x7019) read -- the one feedback path reliable on RS
firmware -- so no CAN bus, SDK import or hardware is touched. The torque the
MIT controller applies is kp*(target - pos), so "the jaw is pressing" is
exactly "the jaw stalled short of target with the position error (and thus
the torque) held high"; the method must report closed on a free close, closed
on a stall, drop the stiffness to `hold_kp` after contact, and only time out
when the jaw keeps advancing without reaching the target.
"""

from __future__ import annotations

import pytest

from cascade.config import Cfg
from cascade.control import rebot_rs_arm as rs
from cascade.control.rebot_rs_arm import RebotRSArm


class _Fake:
    has_gripper = True


def _arm(monkeypatch, pos_fn, **gripper_cfg):
    arm = RebotRSArm(Cfg({"gripper": {
        "open_pos": 6.2, "closed_pos": 0.0, **gripper_cfg,
    }}))
    arm._arm = _Fake()
    monkeypatch.setattr(arm, "set_gripper", lambda pos, effort=1.0: None)
    monkeypatch.setattr(arm, "_gripper_pos", pos_fn)
    kp_calls = []
    monkeypatch.setattr(arm, "_set_gripper_kp",
                        lambda pos, kp: kp_calls.append((pos, kp)))
    arm._kp_calls = kp_calls
    return arm


class _Clock:
    """Fake monotonic clock: sleep() advances it so the poll loop terminates."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, s):
        self.now += s


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(rs.time, "sleep", c.sleep)
    monkeypatch.setattr(rs.time, "monotonic", c.monotonic)
    return c


def test_free_close_reaches_target(monkeypatch, clock):
    """Empty jaw closes freely to the closed target: reported closed, no hold."""
    seq = iter([6.2, 0.05, 0.0])
    arm = _arm(monkeypatch, lambda: next(seq))
    assert arm.close_gripper_torque() is True
    assert arm._kp_calls == []            # nothing held -> no light-hold re-issue
    assert arm._grip_contact_pos is None


def test_stall_short_of_target_is_closed(monkeypatch, clock):
    """Jaw stops short of the target with residual error => pressing => closed,
    and the stiffness drops to hold_kp so it holds lightly."""
    arm = _arm(monkeypatch, lambda: 1.5)  # constant: never reaches 0.0
    assert arm.close_gripper_torque(stall_rad=0.03, settle_tol=0.1) is True
    # the hold is re-issued at the closed target with the light hold kp (1.0),
    # not the full close kp (6.0)
    assert arm._kp_calls == [(0.0, 1.0)]
    assert arm._grip_contact_pos == 1.5


def test_hold_kp_is_configurable(monkeypatch, clock):
    """`hold_kp` on the gripper profile overrides the default light-hold kp."""
    arm = _arm(monkeypatch, lambda: 2.0, hold_kp=0.5)
    assert arm.close_gripper_torque() is True
    assert arm._kp_calls == [(0.0, 0.5)]


def test_advancing_without_reaching_times_out(monkeypatch, clock):
    """A jaw that keeps moving and never reaches the target times out."""
    calls = {"n": 0}

    def pos_fn():
        calls["n"] += 1
        return 6.2 - 0.05 * calls["n"]  # 0.05/step > stall_rad, never < settle_tol

    arm = _arm(monkeypatch, pos_fn)
    assert arm.close_gripper_torque(timeout_s=1.0) is False
    assert arm._kp_calls == []            # never contacted -> never held


def test_no_gripper_is_not_closed(monkeypatch, clock):
    """A gripperless arm (or one without feedback) reports not-closed."""
    arm = RebotRSArm(Cfg({}))
    arm._arm = _Fake()
    arm._arm.has_gripper = False
    assert arm.close_gripper_torque() is False


def test_park_gripper_dispatches_only_when_backend_supports_it(monkeypatch):
    """_park_gripper reaches the backend method on the reBot, and is a no-op
    on backends (mock/so101) that do not expose close_gripper_torque."""
    from cascade.apps.demo import _park_gripper

    class Raw:
        def close_gripper_torque(self):
            return True

    class Arm:
        raw = Raw()

    _park_gripper(Arm())  # must not raise

    class NoGripperRaw:
        pass

    class NoGripperArm:
        raw = NoGripperRaw()

    _park_gripper(NoGripperArm())  # no-op, must not raise
