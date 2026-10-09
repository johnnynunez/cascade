"""End-to-end, hardware-free: the rig's real pipeline against a known truth.

The arm is built the standard way (apps/demo._build_arm: kinematics -> own
SafetyHarness from the arm profile's resolved safety view -> SafeArm), with
ONLY the backend swapped for the kinematic MockArm. The camera renders the
real ArUco bitmap where FK + a ground-truth hand-eye put it, with pixel
noise and one bumped-marker pose (an outlier sample). Everything between --
preset vetting, SafeArm motion, detection, PnP, the joint solver, the
schema-v1 record and the Extrinsics loader gate -- is production code.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_pin

from cascade.calibration.aruco import ArucoSession
from cascade.calibration.dataset import MarkerSpec, save_hand_eye
from cascade.calibration.frames import pose_error, se3_inv
from cascade.calibration.handeye import EYE_IN_HAND, EYE_TO_HAND
from cascade.calibration.session import CollectionSession, SessionConfig, load_poses
from cascade.config import Cfg
from cascade.perception.grounding import Extrinsics

pytestmark = needs_pin


def _collect(rig, mode, poses):
    from cascade.calibration.cli import solve_and_record

    marker = MarkerSpec()
    session = CollectionSession(
        safe_arm=rig.safe_arm, kin=rig.kin, camera=rig.camera, aruco=ArucoSession(),
        config=SessionConfig(mode=mode, marker=marker, settle_s=0.0, marker_timeout_s=0.5,
                             stable_frames=2, min_move_s=1.0),
        home_q=rig.home_q, dist_coeffs=rig.camera.dist_coeffs,
        log=lambda *_: None, sleep=lambda _s: None)
    samples = session.run_auto(poses)
    record = solve_and_record(samples, mode=mode, marker=marker, camera="synthetic",
                              camera_serial=rig.camera.serial, arm=rig.arm_name,
                              ee_frame=rig.ee_frame, K=rig.camera.K,
                              D=rig.camera.dist_coeffs, image_size=rig.camera.image_size)
    return samples, record


@pytest.mark.parametrize("mode", [EYE_TO_HAND, EYE_IN_HAND])
def test_recovers_the_true_hand_eye_through_the_whole_pipeline(tmp_path, mode):
    from cascade.calibration.cli import dry_run_rig
    from cascade.control.mock_arm import MockArm

    rig = dry_run_rig("rebot_rs", mode, noise_px=0.5, corrupt_poses={5}, seed=1)
    # The reBot profile's envelope and velocity cap, on a backend that
    # cannot touch the bus.
    assert isinstance(rig.raw_arm, MockArm)
    assert rig.safe_arm.harness.limits.max_joint_vel == pytest.approx(0.8)

    samples, record = _collect(rig, mode, load_poses("rebot_rs", mode))
    assert len(samples) >= 15
    assert record.acceptable, record.rejection_reasons
    err = pose_error(se3_inv(rig.T_hand_eye) @ record.T_hand_eye)
    assert np.linalg.norm(err[:3]) < 0.003, err
    assert np.degrees(np.linalg.norm(err[3:])) < 0.5, err

    # The bumped-marker pose was captured and rejected as an outlier.
    bad = [i for i, s in enumerate(samples)
           if any(np.allclose(s.T_gripper2base, T, atol=1e-6) for T in rig.camera.corrupted_tcps)]
    assert bad, "the injected outlier pose was never captured"
    assert set(bad) <= set(record.outlier_indices)

    path = save_hand_eye(tmp_path / f"{mode}.json", record)
    tcp = rig.kin.fk(rig.safe_arm.get_state().q)
    e = Extrinsics.from_config(Cfg({"mode": mode, "hand_eye_json": str(path)}),
                               fk_tcp2base=lambda: tcp, camera_serial=rig.camera.serial)
    assert e.calibrated
    truth = rig.T_hand_eye if mode == EYE_TO_HAND else tcp @ rig.T_hand_eye
    assert np.allclose(e.cam_to_base()[:3, 3], truth[:3, 3], atol=0.003)


def test_a_degenerate_sweep_is_refused_end_to_end(tmp_path):
    """Yaw-only presets at one position: the residual is tiny, the transform
    is unconstrained along the yaw axis -- the record is rejected and the
    runtime refuses to use it."""
    from cascade.calibration.cli import dry_run_rig

    rig = dry_run_rig("rebot_rs", EYE_TO_HAND, noise_px=0.3, seed=2)
    poses = [[0.30, -0.05, 0.28, 0.0, 0.0, float(y)] for y in np.linspace(-0.6, 0.6, 12)]
    samples, record = _collect(rig, EYE_TO_HAND, poses)
    assert len(samples) >= 10
    assert not record.acceptable
    assert any("rotation spread" in r for r in record.rejection_reasons)
    path = save_hand_eye(tmp_path / "degenerate.json", record)
    e = Extrinsics.from_config(Cfg({"hand_eye_json": str(path)}))
    assert not e.calibrated
