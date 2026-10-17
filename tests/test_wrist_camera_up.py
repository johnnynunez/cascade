"""The wrist camera rides on TOP of the reBot's gripper and must stay there.

Johnny, watching the real rig: the gripper rolled with the camera facing
down and the camera hit the table; it must always be camera-up. A parallel
jaw is symmetric under a half-turn about its approach axis, so every grasp
has a camera-up twin; nothing ever needs the camera underneath.

The rule lives in the SafetyHarness (opt-in per arm profile via
`safety.wrist_camera_up`): a pose whose camera-up axis (a TCP-frame axis)
points below horizontal is refused by vet_pose and by every step check (the
50 Hz approve, trajectory preflights, planned paths). That covers grasps,
nudges, turn_screw and anything added later, not just the grasp planner.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cascade.safety.harness import SafetyHarness, SafetyLimits
from cascade.types import SafetyViolation


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class RollKin:
    """TCP at q[:3]; tool frame = Ry(q[4]) @ Rx(q[5]): pitch, then roll about
    the approach axis (tool x), as the reBot's j6 rolls the gripper."""

    joint_limits = (np.full(6, -3.2), np.full(6, 3.2))

    def fk(self, q):
        q = np.asarray(q, float)
        T = np.eye(4)
        T[:3, :3] = _ry(q[4]) @ _rx(q[5])
        T[:3, 3] = q[:3]
        return T

    def link_positions(self, q):
        q = np.asarray(q, float)
        return np.array([[0, 0, 0.2], [0, 0, 0.2], [q[0], q[1], max(q[2], 0.1)]])


def limits(**kw):
    base = dict(workspace_min=np.array([0.10, -0.30, -0.01]), workspace_max=np.array([0.50, 0.30, 0.55]),
                table_z=0.0, table_clearance=0.02, max_joint_vel=1e9, watchdog_s=1e9,
                tool_up_axis=(0.0, 0.0, 1.0), min_tool_up_z=-0.1)
    base.update(kw)
    return SafetyLimits(**base)


def pose(roll=0.0, pitch=0.0):
    return np.array([0.3, 0.0, 0.2, 0.0, pitch, roll])


def test_camera_up_is_approved():
    h = SafetyHarness(limits(), kinematics=RollKin())
    h.approve(pose(), pose(0.3), dt=1.0)
    assert h.vet_pose(pose(0.3)) is None


@pytest.mark.parametrize("roll", [np.radians(100), np.pi - 0.05, -np.radians(100), -(np.pi - 0.05)])
def test_camera_below_horizontal_is_refused_everywhere(roll):
    h = SafetyHarness(limits(), kinematics=RollKin())
    with pytest.raises(SafetyViolation, match="wrist camera"):
        h.approve(pose(), pose(roll), dt=1.0)
    assert "wrist camera" in (h.vet_step(pose(), pose(roll), 1.0) or "")
    assert "wrist camera" in (h.vet_pose(pose(roll)) or "")


def test_a_side_roll_at_the_tolerance_is_allowed_and_just_past_it_is_not():
    h = SafetyHarness(limits(min_tool_up_z=-0.1), kinematics=RollKin())
    edge = np.arccos(-0.1)                      # up-axis z component == -0.1
    assert h.vet_pose(pose(edge - 1e-3)) is None
    assert "wrist camera" in (h.vet_pose(pose(edge + 0.02)) or "")


def test_a_vertical_dive_is_camera_sideways_and_allowed():
    """Pitching straight down turns the camera-up axis horizontal (outward):
    the camera is behind the jaws, nowhere near the table."""
    h = SafetyHarness(limits(), kinematics=RollKin())
    assert h.vet_pose(pose(pitch=np.pi / 2)) is None


def test_an_arm_already_camera_down_may_only_roll_back_up_or_hold():
    """Escape rule, like the joint-limit and table rules: an arm turned over
    by hand or by an older build must be able to recover, never get worse."""
    h = SafetyHarness(limits(), kinematics=RollKin())
    down = pose(np.pi - 0.2)
    h.approve(down, down, dt=1.0)                   # hold: no motion, allowed
    h.approve(down, pose(np.pi - 0.4), dt=1.0)      # rolling back toward up
    with pytest.raises(SafetyViolation, match="wrist camera"):
        h.approve(down, pose(np.pi - 0.1), dt=1.0)  # further down
    with pytest.raises(SafetyViolation, match="wrist camera"):
        h.approve(down, np.array([0.35, 0.0, 0.2, 0.0, 0.0, np.pi - 0.2]), dt=1.0)  # sliding while down


@pytest.mark.parametrize("step", [2e-9, 1e-5], ids=["just past atol", "inside numpy's default rtol"])
def test_a_camera_down_hold_is_an_absolute_tolerance_only(step):
    """A camera-down "hold" is q_next == q_prev within an ABSOLUTE 1e-9 rad
    (pre-PR M2). numpy's default rtol (1e-5 x |q|) would call a 1e-5 rad
    turn further down at q ~ 2.9 rad a hold: 0.5 mrad/s of creep at 50 Hz."""
    h = SafetyHarness(limits(), kinematics=RollKin())
    down = pose(np.pi - 0.2)
    h.approve(down, pose(np.pi - 0.2 + 0.5e-9), dt=1.0)          # inside atol: still a hold
    further = pose(np.pi - 0.2 + step)                            # turning further down
    with pytest.raises(SafetyViolation, match="wrist camera"):
        h.approve(down, further, dt=1.0)
    assert "wrist camera" in (h.vet_step(down, further, 1.0) or "")


def test_arms_without_a_wrist_camera_are_unaffected():
    h = SafetyHarness(limits(tool_up_axis=None), kinematics=RollKin())
    h.approve(pose(), pose(np.pi - 0.05), dt=1.0)
    assert h.vet_pose(pose(np.pi - 0.05)) is None


def test_config_parses_and_rejects_nonsense():
    cfg = SimpleNamespace(workspace=SimpleNamespace(min=[0.1, -0.3, 0.0], max=[0.5, 0.3, 0.5]))
    cfg.get = lambda k, d=None: {"wrist_camera_up": {"tcp_axis": [0, 0, 2], "min_up_z": -0.1}}.get(k, d)
    lim = SafetyLimits.from_config(cfg)
    assert np.allclose(lim.tool_up_axis, [0, 0, 1]) and lim.min_tool_up_z == -0.1
    for bad in ({"tcp_axis": [0, 0, 0]}, {"tcp_axis": [0, 0, float("nan")]}, {"tcp_axis": [0, 0, 1], "min_up_z": 1.5},
                {"tcp_axis": [0, 0, 1], "min_up_z": float("nan")}):
        cfg.get = lambda k, d=None, bad=bad: {"wrist_camera_up": bad}.get(k, d)
        with pytest.raises(ValueError, match="wrist_camera_up"):
            SafetyLimits.from_config(cfg)


# ── the real reBot ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def rebot():
    from cascade.config import load_demo_config
    from cascade.control.kinematics import Kinematics

    cfg = load_demo_config(arm="rebot_rs", camera="mock", llm="mock")
    arm = cfg.arm
    kin = Kinematics(arm.model, arm.get("ee_frame", "gripper_end"), 6, joint_signs=arm.get("joint_signs"))
    return SimpleNamespace(cfg=cfg, kin=kin, home=np.asarray(arm.home_q, float),
                           harness=SafetyHarness(SafetyLimits.from_config(cfg.safety), kinematics=kin))


def test_the_rebot_profile_keeps_its_camera_up(rebot):
    lim = rebot.harness.limits
    assert lim.tool_up_axis is not None and np.allclose(lim.tool_up_axis, [0, 0, 1])
    assert rebot.harness.vet_pose(rebot.home) is None
    flipped = rebot.home.copy()
    flipped[5] = np.pi - 0.1                         # j6 rolls the gripper about its approach
    assert "wrist camera" in (rebot.harness.vet_pose(flipped) or "")


def test_the_grasp_selector_takes_the_camera_up_twin_even_when_the_other_is_closer(rebot):
    """Path-dependent, as on the rig: a previous grasp left j6 at -100 deg
    (the wrist rolled, camera sideways). The next grasp is offered rolled
    170 deg (camera underneath, j6 -170: 70 deg away) or as its twin (camera
    up, j6 +10: 110 deg away). The selector executes whichever needs less
    joint travel: camera-down (measured: the camera-up axis at -0.85 z, and
    the same from j6 -60 or -140 with rolls of 120-170 deg). With the harness
    vetting candidates it must take the camera-up twin. (The reBot's j6 sign
    is opposite to the tool roll, hence the negative seed.)"""
    from cascade.grasping.selector import select_grasp
    from cascade.types import Grasp

    R = _ry(np.radians(30)) @ _rx(np.radians(170))   # pitched 30 deg down, rolled over
    g = Grasp(position=np.array([0.35, 0.0, 0.12]), rotation=R, width_m=0.04, approach=R[:, 0],
              quality=0.9, label="cube")
    q_now = rebot.home.copy()
    q_now[5] = np.radians(-100)

    _, q_pre, q_grasp = select_grasp([g], rebot.kin, q_now)          # no harness: the old behaviour
    assert rebot.kin.fk(q_grasp)[2, 2] < 0, "precondition: the nearer twin is camera-down"

    h = rebot.harness
    _, q_pre, q_grasp = select_grasp([g], rebot.kin, q_now,
                                     validate=lambda _g, qp, qg: h.vet_pose(qp) or h.vet_pose(qg))
    for q in (q_pre, q_grasp):
        assert rebot.kin.fk(q)[2, 2] > 0.5, f"camera-up axis z {rebot.kin.fk(q)[2, 2]:.2f} at the chosen pose"
