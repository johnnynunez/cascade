"""Camera->base extrinsic fitting, and the ways it must refuse to succeed.

A rigid fit ALWAYS returns a matrix. Feed it nonsense and it returns a
confident, wrong transform; every pixel back-projected afterwards then lands
somewhere plausible but false, and the arm reaching a few centimetres off
reads as a grasping bug rather than a calibration one. These tests pin the
refusals as hard as the successes -- the refusals are the feature.
"""

from __future__ import annotations

import numpy as np
import pytest

from cascade.perception.calibration import (
    MAX_ACCEPTABLE_RMSE_M,
    CalibrationResult,
    fit_extrinsic,
    load_extrinsic,
    save_extrinsic,
)
from cascade.types import pose_to_transform


def _cloud(n=12, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform([-0.2, -0.2, 0.3], [0.2, 0.2, 0.7], size=(n, 3))


def _apply(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]


def test_exact_correspondences_recover_the_transform():
    T_true = pose_to_transform([0.4, -0.1, 0.6, 0.2, -0.35, 1.1])
    cam = _cloud()
    res = fit_extrinsic(cam, _apply(T_true, cam))
    assert res.acceptable
    assert np.allclose(res.T_cam2base, T_true, atol=1e-9)
    assert res.rmse_m < 1e-12


def test_the_fit_is_a_rotation_not_a_reflection():
    """Without the SVD sign correction a noisy cloud can yield det = -1: a
    perfect-looking fit that MIRRORS the workspace, swapping left and right."""
    T_true = pose_to_transform([0.1, 0.2, 0.3, 0.5, 0.1, -0.9])
    cam = _cloud(seed=3)
    rng = np.random.default_rng(7)
    res = fit_extrinsic(cam + rng.normal(0, 0.003, cam.shape), _apply(T_true, cam))
    R = res.T_cam2base[:3, :3]
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)


def test_realistic_depth_noise_still_passes():
    """2 mm of depth noise is normal on a real sensor and must not trip the
    gate -- a threshold nobody can pass in practice gets deleted."""
    T_true = pose_to_transform([0.4, -0.1, 0.6, 0.2, -0.35, 1.1])
    cam = _cloud()
    rng = np.random.default_rng(1)
    res = fit_extrinsic(cam + rng.normal(0, 0.002, cam.shape), _apply(T_true, cam))
    assert res.acceptable
    assert res.rmse_m < MAX_ACCEPTABLE_RMSE_M


def test_garbage_correspondences_are_rejected_not_raised():
    """"This calibration is bad" is a RESULT to report, not an exception:
    the caller has to be able to print the residual and tell the operator."""
    rng = np.random.default_rng(2)
    res = fit_extrinsic(_cloud(), rng.uniform(-0.5, 0.5, (12, 3)))
    assert isinstance(res, CalibrationResult)
    assert not res.acceptable
    assert res.rmse_m > MAX_ACCEPTABLE_RMSE_M


def test_coplanar_samples_are_rejected_despite_a_perfect_residual():
    """THE trap this module exists for. Points sampled at one height leave the
    fit unconstrained out of plane: the residual is 0 mm and the transform is
    wrong. Only the spread check catches it."""
    T_true = pose_to_transform([0.4, -0.1, 0.6, 0.2, -0.35, 1.1])
    flat = _cloud()
    flat[:, 2] = 0.5
    res = fit_extrinsic(flat, _apply(T_true, flat))
    assert res.rmse_m < 1e-9        # a deceptively perfect fit ...
    assert not res.acceptable       # ... that must not be trusted


def test_unfittable_input_raises():
    with pytest.raises(ValueError, match="at least 3"):
        fit_extrinsic(np.zeros((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="shape mismatch"):
        fit_extrinsic(np.zeros((4, 3)), np.zeros((5, 3)))
    bad = _cloud()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        fit_extrinsic(bad, _cloud())


def test_saved_calibration_round_trips_with_its_diagnostics(tmp_path):
    T_true = pose_to_transform([0.4, -0.1, 0.6, 0.2, -0.35, 1.1])
    cam = _cloud()
    res = fit_extrinsic(cam, _apply(T_true, cam))
    path = save_extrinsic(tmp_path / "cal.json", res, camera="l515", note="test")

    loaded = load_extrinsic(path)
    assert loaded is not None
    assert np.allclose(loaded, T_true, atol=1e-9)

    import json

    saved = json.loads(path.read_text())
    # Residuals live beside the matrix on purpose: months later they are the
    # only way to tell a trustworthy extrinsic from a confident wrong one.
    assert saved["rmse_m"] == pytest.approx(res.rmse_m)
    assert saved["num_points"] == 12
    assert saved["camera"] == "l515"


def test_a_rejected_calibration_does_not_load_by_default(tmp_path):
    """Silently using a rejected fit is the exact failure being prevented:
    "no extrinsics" degrades visibly (3D fusion off), "wrong extrinsics"
    degrades invisibly (every target a few cm out)."""
    rng = np.random.default_rng(4)
    bad = fit_extrinsic(_cloud(), rng.uniform(-0.5, 0.5, (12, 3)))
    assert not bad.acceptable
    path = save_extrinsic(tmp_path / "bad.json", bad)

    assert load_extrinsic(path) is None
    assert load_extrinsic(path, trust_unacceptable=True) is not None
    assert load_extrinsic(tmp_path / "missing.json") is None


def test_the_fit_feeds_the_extrinsics_class_the_pipeline_uses(tmp_path):
    """End of the chain: a fitted matrix has to be usable by the object that
    actually back-projects pixels, or this module is an island."""
    from cascade.perception.grounding import Extrinsics

    T_true = pose_to_transform([0.4, -0.1, 0.6, 0.2, -0.35, 1.1])
    cam = _cloud()
    res = fit_extrinsic(cam, _apply(T_true, cam))

    ext = Extrinsics(mode="eye_to_hand", T=res.T_cam2base)
    point_cam = np.array([0.05, -0.02, 0.42])
    T = ext.cam_to_base()
    assert np.allclose(T[:3, :3] @ point_cam + T[:3, 3], _apply(T_true, point_cam[None])[0])
