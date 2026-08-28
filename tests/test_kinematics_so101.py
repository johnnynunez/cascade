"""SO-101 facts the arm profile is built on.

`configs/arms/so101.yaml` states a set of measured claims -- the joint ORDER the
`n_controlled` slice depends on, which tool-frame column is the approach, where
the top-down envelope is, that plain 6-DoF IK is the right formulation. None of
those are enforced by the code; a re-vendored model or a "cleanup" of the
profile could invalidate any of them silently, and the symptom on hardware would
be an arm driving into the table.

Each test therefore pins one claim, with the number the profile quotes.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import SO101_URDF, needs_pin_so101

from cascade.config import load_profile
from cascade.grasping.obb_grasp import _yaw_rotation

DOWN = np.array([0.0, 0.0, -1.0])
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


@pytest.fixture(scope="module")
def kin():
    from cascade.control.kinematics import Kinematics

    return Kinematics(str(SO101_URDF), "gripper_frame_link", n_controlled=5)


@pytest.fixture(scope="module")
def profile():
    return load_profile("arms", "so101")


def topdown_R(yaw: float) -> np.ndarray:
    """Target rotation the planner emits for this arm (`open_down` order)."""
    return _yaw_rotation(yaw, axis_order="open_down")


@needs_pin_so101
def test_arm_chain_occupies_the_first_five_joint_indices():
    """Kinematics commands q[:n_controlled] and zero-pads the rest, so the arm
    chain MUST come before the gripper in Pinocchio's ordering.

    This is not something the URDF guarantees: this file declares its joints in
    the reverse order (gripper first, shoulder_pan last) and only Pinocchio's
    root-down tree traversal puts them right. A model where the gripper sorted
    earlier would make every commanded pose drive the wrong joints.
    """
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(SO101_URDF))
    assert model.nq == 6, "5 arm joints + gripper"
    assert list(model.names)[1:] == [*ARM_JOINTS, "gripper"]


@needs_pin_so101
def test_wrist_roll_is_collinear_with_the_approach(kin):
    """The reason no `ik_task_weights` is needed (profile: "0.00 deg").

    If a re-vendored model broke this collinearity, the 5-DoF chain really would
    be unable to hold a vertical approach with a chosen yaw, and unweighted IK
    would start failing everywhere.
    """
    q = np.array([0.2, -0.6, 0.7, 0.5, 0.0])
    approach0 = kin.fk(q)[:3, 2]
    for droll in (0.3, 0.8, -0.5):
        q2 = q.copy()
        q2[4] += droll
        approach1 = kin.fk(q2)[:3, 2]
        turned = np.degrees(np.arccos(np.clip(approach0 @ approach1, -1, 1)))
        assert turned < 0.1, f"roll {droll} turned the approach {turned:.2f} deg"


@needs_pin_so101
def test_home_pose_is_top_down_and_inside_limits(kin, profile):
    """home_q doubles as the grasp IK seed, so a pose that is not actually
    top-down (an earlier revision of this profile was 90 deg off, pointing
    forward) quietly halves the solve rate rather than failing."""
    home = np.asarray(profile.get("home_q"), dtype=float)
    assert home.size == 5
    lo, hi = kin.joint_limits
    assert np.all(home > lo) and np.all(home < hi)

    T = kin.fk(home)
    tilt = np.degrees(np.arccos(np.clip(T[:3, 2] @ DOWN, -1, 1)))
    assert tilt < 2.0, f"home approach is {tilt:.1f} deg off vertical"
    # Profile quotes TCP [0.200, 0.006, 0.060], r = 0.200.
    assert np.hypot(*T[:2, 3]) == pytest.approx(0.200, abs=0.01)
    assert T[2, 3] == pytest.approx(0.060, abs=0.01)


@needs_pin_so101
def test_handover_keeps_the_vertical_approach(kin, profile):
    """It is home with joint 0 yawed, and joint 0 rotates about world z -- so
    the approach must be unchanged. Pins the claim rather than the pose."""
    home = np.asarray(profile.get("home_q"), dtype=float)
    hand = np.asarray(profile.get("handover_q"), dtype=float)
    assert hand.size == home.size
    lo, hi = kin.joint_limits
    assert np.all(hand > lo) and np.all(hand < hi)
    assert np.allclose(hand[1:], home[1:]), "only the base should differ"
    tilt = np.degrees(np.arccos(np.clip(kin.fk(hand)[:3, 2] @ DOWN, -1, 1)))
    assert tilt < 2.0


@needs_pin_so101
def test_topdown_grasps_solve_without_task_weights(kin, profile):
    """The profile's whole grasp budget rests on this rate (quoted 0.944).

    Deliberately mimics how `select_grasp` calls IK -- one seed (home_q, since
    _plan_grasps re-homes first) and the default restarts -- because a rate
    measured with generous multi-seeding would not be the one the pipeline gets.
    """
    seed = np.asarray(profile.get("home_q"), dtype=float)
    ok = total = 0
    for r in (0.14, 0.18, 0.22, 0.26):
        for az in np.radians((-90, -45, 0, 45, 90)):
            for yaw in np.radians((0, 60, 120)):
                p = np.array([r * np.cos(az), r * np.sin(az), 0.03])
                T = np.eye(4)
                T[:3, :3] = topdown_R(yaw)
                T[:3, 3] = p
                total += 1
                res = kin.ik(T, seed)
                if not res.success:  # the 180-deg jaw flip the planner emits
                    T[:3, :3] = topdown_R(yaw + np.pi)
                    res = kin.ik(T, seed)
                ok += bool(res.success)
    assert ok / total > 0.90, f"top-down solve rate fell to {ok / total:.3f}"


@needs_pin_so101
def test_grasp_budget_fits_under_the_measured_ceiling(kin, profile):
    """topdown_z_max + pregrasp_offset_m must stay under the ~0.09 m height at
    which vertical-approach IK drops to zero. Tuning either alone reintroduces
    the reBot-sized offset that made every pregrasp unreachable."""
    g = profile.get("overrides").get("grasp")
    highest_pregrasp = float(g.get("topdown_z_max")) + float(g.get("pregrasp_offset_m"))
    assert highest_pregrasp <= 0.08, (
        f"pregrasp would reach z={highest_pregrasp:.3f}, above the measured band"
    )

    seed = np.asarray(profile.get("home_q"), dtype=float)
    ok = total = 0
    for r in (0.16, 0.20, 0.24):
        for az in np.radians((-60, 0, 60)):
            T = np.eye(4)
            T[:3, :3] = topdown_R(0.0)
            T[:3, 3] = [r * np.cos(az), r * np.sin(az), highest_pregrasp]
            total += 1
            res = kin.ik(T, seed)
            if not res.success:
                T[:3, :3] = topdown_R(np.pi)
                res = kin.ik(T, seed)
            ok += bool(res.success)
    assert ok / total >= 0.5, f"pregrasp height solves only {ok / total:.2f}"


@needs_pin_so101
def test_vertical_approach_beyond_the_annulus_is_rejected(kin, profile):
    """The workspace AABB cannot express the hole in the middle or the outer
    edge, so IK failure is the gate. If IK started "succeeding" out here, the
    harness would be the only thing left between the agent and a stalled arm."""
    seed = np.asarray(profile.get("home_q"), dtype=float)
    for r in (0.04, 0.40):
        solved = 0
        for yaw in np.radians((0, 45, 90, 135)):
            T = np.eye(4)
            T[:3, :3] = topdown_R(yaw)
            T[:3, 3] = [r, 0.0, 0.03]
            solved += bool(kin.ik(T, seed).success)
        assert solved == 0, f"r={r} reported reachable top-down"


def test_tool_axis_order_puts_the_approach_last(profile):
    """`open_down` means columns [open, third, approach]. The SO-101's fingers
    run along col2 (MEASURED from the Menagerie fingertip geometry), so a
    profile switched to the reBot's `down_open` would rotate every grasp."""
    assert profile.get("tool_axis_order") == "open_down"
    R = _yaw_rotation(0.3, axis_order="open_down")
    assert np.allclose(R[:, 2], DOWN), "col2 must be the (downward) approach"
    assert R[2, 0] == pytest.approx(0.0, abs=1e-9), "opening axis is horizontal"
    assert np.linalg.det(R) == pytest.approx(1.0), "must stay right-handed"


def test_profile_declares_five_joints_and_matching_vector_lengths(profile):
    """A wrong n_joints does not raise, it truncates or broadcasts silently."""
    n = int(profile.get("n_joints"))
    assert n == 5
    for key in ("home_q", "handover_q", "joint_signs", "wire_signs", "servo_ids"):
        assert len(profile.get(key)) == n, f"{key} must have {n} entries"


def test_profile_does_not_set_ik_task_weights(profile):
    """Documented as deliberate (see the profile and kinematics.py): this arm
    does not need weighting, and adding it would let IK accept poses the wrist
    cannot actually hold."""
    assert profile.get("ik_task_weights") is None
