"""Hand-eye calibration CLI (camera- and arm-agnostic).

Ported from WRC ``scripts/calib_top_orbbec.py`` (eye-to-hand) and
``scripts/calib_wrist.py`` (eye-in-hand) as ONE command: the mounting comes
from the camera profile's ``extrinsics.mode``, the camera from
``make_camera`` (RealSense, Orbbec, UVC, ... -- intrinsics from its frames),
the arm from ``apps.demo._build_arm`` (kinematics -> the profile's own
SafetyHarness -> SafeArm, a LazyArm so the motors stay untouched until the
first vetted motion). Entry points: ``python scripts/calibrate_handeye.py``
and ``cascade-calib-handeye``. Operator procedure: docs/HANDEYE_CALIBRATION.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import MarkerSpec, record_from_fit
from .frames import so3_exp
from .handeye import EYE_IN_HAND, EYE_TO_HAND, solve_hand_eye

#: --verify passes when the known-point RMSE is below this (Seeed's wiki
#: section 5.3 / rebot_grasp set.py:320: "RMSE < 10 mm").
VERIFY_MAX_RMSE_M = 0.010
#: Fewer samples than this are not worth solving (the gate wants 8 inliers).
MIN_SAMPLES_TO_SOLVE = 6


def _T(t, R) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _dry_run_truth():
    """Ground truth for --dry-run, shaped like the real rig.

    eye_to_hand: a camera ~0.95 m above the base looking straight down
    (the axes of d455f_scene.yaml's placeholder, tilted 5 deg), marker face
    up 3 cm above the TCP. eye_in_hand: a camera behind the TCP looking
    along the approach axis, 16 deg down (WRC's measured wrist mount,
    rounded), marker flat on the table 0.45 m in front of the base.
    """
    down = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    X_eth = _T([0.30, -0.03, 0.95], down @ so3_exp([0.05, -0.06, 0.0]))
    Y_eth = _T([0.0, 0.0, 0.03], so3_exp([0.0, 0.0, 0.3]))
    along = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    X_eih = _T([-0.065, -0.004, 0.020], along @ so3_exp([-0.28, 0.0, 0.0]))
    Z_eih = _T([0.45, 0.0, 0.0], so3_exp([0.0, 0.0, 0.4]))
    return {EYE_TO_HAND: (X_eth, Y_eth), EYE_IN_HAND: (X_eih, Z_eih)}


@dataclass
class DryRunRig:
    cfg: object
    arm_name: str
    raw_arm: object
    safe_arm: object
    kin: object
    camera: object
    T_hand_eye: np.ndarray
    T_marker: np.ndarray
    home_q: np.ndarray
    ee_frame: str


def dry_run_rig(arm: str, mode: str, *, noise_px: float = 0.5, corrupt_poses=(),
                seed: int = 0, marker: MarkerSpec = MarkerSpec()) -> DryRunRig:
    """The arm profile's kinematics, harness and velocity cap on a MockArm,
    plus a synthetic camera. Never builds a hardware backend: the profile's
    ``type`` is replaced by ``mock`` and the result is checked."""
    from ..apps.demo import _arm_cfgs, _build_arm
    from ..config import load_demo_config
    from ..control.mock_arm import MockArm
    from .synthetic import SyntheticMarkerCamera

    cfg = load_demo_config(camera="mock", arm=arm, llm="mock")
    acfg = _arm_cfgs(cfg)[0]
    acfg._data["type"] = "mock"
    raw, safe_arm, kin = _build_arm(acfg, False, None, cfg)
    if not isinstance(raw, MockArm):  # pragma: no cover - defensive, fail closed
        raw.disconnect()
        raise RuntimeError("dry run built a non-mock arm backend; refusing")
    T_he, T_m = _dry_run_truth()[mode]
    camera = SyntheticMarkerCamera(
        mode, T_he, T_m, lambda: kin.fk(safe_arm.get_state().q), noise_px=noise_px,
        corrupt_poses=corrupt_poses, seed=seed, size_m=marker.size_m,
        marker_id=marker.marker_id, dictionary=marker.dictionary)
    return DryRunRig(cfg=cfg, arm_name=arm, raw_arm=raw, safe_arm=safe_arm, kin=kin,
                     camera=camera, T_hand_eye=T_he, T_marker=T_m,
                     home_q=np.asarray(acfg.home_q, dtype=float),
                     ee_frame=str(acfg.get("ee_frame", "")))


def solve_and_record(samples, *, mode, marker, camera="", camera_serial="", arm="",
                     ee_frame="", K=None, D=None, image_size=None, note=""):
    """Joint solve + schema-v1 record (acceptable or not -- check it)."""
    if len(samples) < MIN_SAMPLES_TO_SOLVE:
        raise ValueError(f"only {len(samples)} samples; need >= {MIN_SAMPLES_TO_SOLVE} to solve "
                         "(check the marker is visible across the sweep)")
    fit = solve_hand_eye(samples, mode)
    return record_from_fit(fit, samples, marker=marker, camera=camera,
                           camera_serial=camera_serial, arm=arm, ee_frame=ee_frame,
                           K=K, D=D, image_size=image_size, note=note)
