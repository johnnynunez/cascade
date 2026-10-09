"""Hand-eye preset poses: an explicit per-arm registry, measured not assumed.

The reBot sets are WRC's verbatim (calib_top_orbbec.TOP_CALIB_POSES, from
rebot_grasp/scripts/calibrate_top_camera.py; calib_wrist.CALIB_POSES). The
pins below re-measure them on cascade's RS kinematics + harness, because the
WRC numbers were tuned against a different FK stack: if an asset or limit
change makes the sweep unreachable or degenerate, this fails here instead of
on the rig.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from conftest import needs_pin

from cascade.calibration.handeye import (
    EYE_IN_HAND,
    EYE_TO_HAND,
    MIN_ROTATION_SPREAD_DEG,
    rotation_spread_deg,
)
from cascade.calibration.session import (
    MAX_JOINT_STEP_RAD,
    load_poses,
    order_by_joint_distance,
)
from cascade.types import pose_to_transform


@pytest.mark.parametrize("arm", ["rebot_rs", "rebot_rs_mb", "mock"])
def test_rs_profiles_share_the_rebot_sets(arm):
    assert load_poses(arm, EYE_TO_HAND) == load_poses("rebot_rs", EYE_TO_HAND)
    assert load_poses(arm, EYE_IN_HAND) == load_poses("rebot_rs", EYE_IN_HAND)


def test_unregistered_arm_is_refused_not_guessed():
    with pytest.raises(ValueError, match="handeye_poses.yaml"):
        load_poses("so101", EYE_TO_HAND)


def test_wrc_provenance_of_the_eye_to_hand_set():
    """WRC test_auto_mode_construction: 38 presets around (0.30, -0.05, 0.28),
    pitch kept small so a marker on top of the gripper faces the camera."""
    poses = load_poses("rebot_rs", EYE_TO_HAND)
    assert len(poses) == 38
    for x, y, z, _r, p, _yaw in poses:
        assert 0.20 < x < 0.40 and -0.15 < y < 0.10 and 0.15 < z < 0.40
        assert -0.5 < p < 0.5


def test_eye_in_hand_set_is_wrcs_wrist_sweep():
    poses = load_poses("rebot_rs", EYE_IN_HAND)
    assert len(poses) == 25
    assert all(0.45 <= p[4] <= 0.95 for p in poses)   # tool pitched down at the table


def test_poses_file_override(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text(yaml.safe_dump([[0.3, 0, 0.3, 0, 0, 0], [0.3, 0.05, 0.3, 0, 0.2, 0]]))
    assert len(load_poses("anything", EYE_TO_HAND, path=f)) == 2
    f.write_text(yaml.safe_dump({"eye_in_hand": [[0.3, 0, 0.3, 0, 0.6, 0]]}))
    assert load_poses("anything", EYE_IN_HAND, path=f) == [[0.3, 0, 0.3, 0, 0.6, 0]]
    f.write_text(yaml.safe_dump([[0.3, 0, 0.3]]))
    with pytest.raises(ValueError, match="roll, pitch, yaw"):
        load_poses("anything", EYE_TO_HAND, path=f)


@needs_pin
@pytest.mark.parametrize("mode", [EYE_TO_HAND, EYE_IN_HAND])
def test_rebot_sets_are_reachable_vetted_and_not_degenerate(mode):
    from cascade.config import load_demo_config
    from cascade.control.kinematics import Kinematics
    from cascade.safety.harness import SafetyHarness, SafetyLimits

    cfg = load_demo_config(arm="rebot_rs", camera="mock", llm="mock")
    a = cfg.arm
    kin = Kinematics(a.model, a.ee_frame, int(a.n_joints), a.joint_signs)
    harness = SafetyHarness(SafetyLimits.from_config(cfg.safety), kinematics=kin)
    home = np.asarray(a.home_q, dtype=float)
    poses = load_poses("rebot_rs", mode)
    ok_q, ok_R = [], []
    for p in poses:
        T = pose_to_transform(p)
        sol = kin.ik(T, home)
        if sol.success and harness.vet_pose(sol.q) is None:
            ok_q.append(sol.q)
            ok_R.append(T[:3, :3])
    assert len(ok_q) >= 0.9 * len(poses), f"{len(ok_q)}/{len(poses)} presets usable"
    assert rotation_spread_deg(ok_R) >= 1.5 * MIN_ROTATION_SPREAD_DEG
    order = order_by_joint_distance(home, ok_q)
    cur, worst = home, 0.0
    for k in order:
        worst = max(worst, float(np.max(np.abs(ok_q[k] - cur))))
        cur = ok_q[k]
    assert worst < MAX_JOINT_STEP_RAD
