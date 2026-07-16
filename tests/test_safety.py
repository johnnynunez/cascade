import numpy as np
import pytest

from wrc_demo.safety.harness import SafetyHarness, SafetyLimits
from wrc_demo.types import SafetyViolation


def limits(**kw):
    base = dict(
        workspace_min=np.array([0.10, -0.30, -0.01]),
        workspace_max=np.array([0.50, 0.30, 0.55]),
        table_z=0.0,
        table_clearance=0.02,
        max_joint_vel=1.0,
        watchdog_s=1e9,
    )
    base.update(kw)
    return SafetyLimits(**base)


def test_velocity_cap():
    h = SafetyHarness(limits())
    q = np.zeros(6)
    h.approve(q, q + 0.01, dt=0.02)  # 0.5 rad/s ok
    with pytest.raises(SafetyViolation, match="velocity"):
        h.approve(q, q + 0.1, dt=0.02)  # 5 rad/s


def test_estop_latches():
    h = SafetyHarness(limits())
    h.estop("test")
    with pytest.raises(SafetyViolation, match="e-stop"):
        h.approve(np.zeros(6), np.zeros(6), 0.02)
    h.reset_estop()
    h.approve(np.zeros(6), np.zeros(6), 0.02)


def test_watchdog():
    h = SafetyHarness(limits(watchdog_s=0.0))
    with pytest.raises(SafetyViolation, match="watchdog"):
        h.approve(np.zeros(6), np.zeros(6), 0.02)
    h2 = SafetyHarness(limits(watchdog_s=60.0))
    h2.heartbeat()
    h2.approve(np.zeros(6), np.zeros(6), 0.02)


class FakeKin:
    """Kinematics stub: TCP = q[:3], joint origins along z of TCP."""

    joint_limits = (np.full(6, -3.0), np.full(6, 3.0))

    def fk(self, q):
        T = np.eye(4)
        T[:3, 3] = q[:3]
        return T

    def link_positions(self, q):
        return np.array([[0, 0, 0.2], [0, 0, 0.2], [q[0], q[1], max(q[2], 0.1)]])


def test_joint_limits():
    h = SafetyHarness(limits(), kinematics=FakeKin())
    bad = np.array([3.5, 0.2, 0.2, 0, 0, 0])
    with pytest.raises(SafetyViolation, match="joint 1"):
        h.approve(bad, bad, 0.02)


def test_workspace_aabb():
    h = SafetyHarness(limits(), kinematics=FakeKin())
    inside = np.array([0.3, 0.0, 0.2, 0, 0, 0])
    h.approve(inside, inside, 0.02)
    outside = np.array([0.9, 0.0, 0.2, 0, 0, 0])
    with pytest.raises(SafetyViolation, match="workspace"):
        h.approve(inside, outside, dt=1e9)  # huge dt so velocity check passes


def test_table_clearance_and_grasp_exemption():
    h = SafetyHarness(limits(), kinematics=FakeKin())
    above = np.array([0.3, 0.0, 0.20, 0, 0, 0])
    low = np.array([0.3, 0.0, 0.01, 0, 0, 0])
    # Descending below the clearance floor from above is rejected...
    with pytest.raises(SafetyViolation, match="table"):
        h.approve(above, low, dt=1e9)
    # ...unless the exemption cylinder over (0.3, 0) is open.
    h.allow_grasp_descent(np.array([0.3, 0.0]), radius_m=0.07, z_min=0.0)
    h.approve(above, low, dt=1e9)
    # Sideways below the floor, outside the cylinder: rejected.
    low_off = np.array([0.45, 0.0, 0.01, 0, 0, 0])
    with pytest.raises(SafetyViolation, match="table"):
        h.approve(low, low_off, dt=1e9)
    h.clear_grasp_exemption()
    with pytest.raises(SafetyViolation, match="table"):
        h.approve(above, low, dt=1e9)


def test_keep_out_zone():
    h = SafetyHarness(
        limits(keep_out=[(np.array([0.2, -0.1, 0.0]), np.array([0.3, 0.1, 0.5]))]),
        kinematics=FakeKin(),
    )
    inside_ko = np.array([0.25, 0.0, 0.3, 0, 0, 0])
    with pytest.raises(SafetyViolation, match="keep-out"):
        h.approve(inside_ko, inside_ko, dt=1e9)
