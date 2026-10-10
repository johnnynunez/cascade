"""Joint SE(3) hand-eye solver (ported from WRC tests/test_handeye_solver.py).

One solver serves both mountings, in the canonical form A_i X B_i = Z:

    eye_to_hand  camera fixed, marker on the gripper:
                 G_i^-1 . T_cam2base . M_i = T_marker2gripper
    eye_in_hand  camera on the wrist, marker fixed on the table:
                 G_i . T_cam2gripper . M_i = T_marker2base

with G_i = T_gripper2base (FK) and M_i = T_marker2cam (PnP). The refusals
are pinned as hard as the successes: a hand-eye fit on a pose set that only
rotates about one axis converges to a tiny residual on a WRONG transform, the
same trap the Kabsch fit's coplanar case documents.
"""

from __future__ import annotations

import numpy as np
import pytest

from cascade.calibration.frames import pose_error, se3_exp, se3_inv, so3_exp
from cascade.calibration.handeye import (
    EYE_IN_HAND,
    EYE_TO_HAND,
    HandEyeSample,
    assess_hand_eye,
    rotation_spread_deg,
    solve_hand_eye,
)


def _T(t, w) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = so3_exp(np.asarray(w, dtype=float))
    T[:3, 3] = t
    return T


# A top camera ~0.6 m above the table looking down, and a marker taped on
# the gripper a few cm off the TCP (the WRC rig's geometry, roughly).
X_ETH = _T([0.30, -0.05, 0.85], [np.pi * 0.97, 0.05, 0.10])
Y_ETH = _T([0.005, 0.010, 0.030], [0.05, -0.02, 0.03])
# A wrist camera looking along the approach axis, marker on the table.
X_EIH = _T([-0.065, -0.004, 0.020], [1.2, 0.1, 1.3])
Z_EIH = _T([0.50, 0.02, 0.0], [0.0, 0.0, 0.4])


def _gripper_poses(rng, n, *, rot=0.5):
    """Diverse gripper poses: translations over the workspace plus rotations
    about all three axes (what a real preset sweep provides)."""
    out = []
    for _ in range(n):
        t = rng.uniform([0.22, -0.12, 0.20], [0.38, 0.10, 0.36])
        out.append(_T(t, rng.normal(scale=rot, size=3)))
    return out


def _samples(mode, Gs, *, X, Y, rng=None, t_noise=0.0, r_noise=0.0):
    out = []
    for i, G in enumerate(Gs):
        if mode == EYE_TO_HAND:
            M = se3_inv(X) @ G @ Y           # from G Y = X M
        else:
            M = se3_inv(X) @ se3_inv(G) @ Y  # from G X M = Z
        if rng is not None and (t_noise or r_noise):
            M = M @ se3_exp(np.concatenate([rng.normal(scale=t_noise, size=3),
                                            rng.normal(scale=r_noise, size=3)]))
        out.append(HandEyeSample(T_gripper2base=G, T_marker2cam=M, label=f"p{i}"))
    return out


def _rot_err_deg(A, B):
    return float(np.degrees(np.linalg.norm(pose_error(se3_inv(A) @ B)[3:])))


@pytest.mark.parametrize("mode,X,Y", [(EYE_TO_HAND, X_ETH, Y_ETH),
                                      (EYE_IN_HAND, X_EIH, Z_EIH)])
def test_exact_data_recovers_both_transforms(rng, mode, X, Y):
    fit = solve_hand_eye(_samples(mode, _gripper_poses(rng, 15), X=X, Y=Y), mode)
    assert fit.mode == mode
    assert np.allclose(fit.T_hand_eye, X, atol=1e-6)
    assert np.allclose(fit.T_marker, Y, atol=1e-6)
    assert fit.metrics["translation_rmse_m"] < 1e-6
    assert fit.outliers == ()
    assert assess_hand_eye(fit.metrics) == []


def test_named_accessors_follow_the_mode(rng):
    eth = solve_hand_eye(_samples(EYE_TO_HAND, _gripper_poses(rng, 10), X=X_ETH, Y=Y_ETH),
                         EYE_TO_HAND)
    assert np.allclose(eth.T_cam2base, X_ETH, atol=1e-6)
    with pytest.raises(AttributeError):
        eth.T_cam2gripper
    eih = solve_hand_eye(_samples(EYE_IN_HAND, _gripper_poses(rng, 10), X=X_EIH, Y=Z_EIH),
                         EYE_IN_HAND)
    assert np.allclose(eih.T_cam2gripper, X_EIH, atol=1e-6)
    with pytest.raises(AttributeError):
        eih.T_cam2base


@pytest.mark.parametrize("mode,X,Y", [(EYE_TO_HAND, X_ETH, Y_ETH),
                                      (EYE_IN_HAND, X_EIH, Z_EIH)])
def test_realistic_noise_is_accepted_and_close(rng, mode, X, Y):
    """~1 mm / 0.3 deg per PnP reading is a normal rig; the gate must pass it
    (a threshold nobody can meet gets deleted) and X must land within a few mm."""
    s = _samples(mode, _gripper_poses(rng, 25), X=X, Y=Y, rng=rng,
                 t_noise=0.001, r_noise=np.radians(0.3))
    fit = solve_hand_eye(s, mode)
    assert np.linalg.norm(fit.T_hand_eye[:3, 3] - X[:3, 3]) < 0.004
    assert _rot_err_deg(fit.T_hand_eye, X) < 0.5
    assert 0.0005 < fit.metrics["translation_rmse_m"] < 0.005
    assert assess_hand_eye(fit.metrics) == []


def test_clean_noisy_data_flags_no_outliers(rng):
    """Regression for WRC's MAD rule (`inlier iff norm <= 3*MAD`, measured from
    ZERO instead of from the median): on uniform clean noise it rejected a
    third of the samples -- the shipped real-rig file reports 14/37 outliers."""
    s = _samples(EYE_TO_HAND, _gripper_poses(rng, 30), X=X_ETH, Y=Y_ETH, rng=rng,
                 t_noise=0.001, r_noise=np.radians(0.3))
    fit = solve_hand_eye(s, EYE_TO_HAND)
    assert len(fit.outliers) <= 1
    assert fit.metrics["n_inliers"] >= 29


def test_gross_outliers_are_rejected_and_do_not_drag_the_fit(rng):
    s = _samples(EYE_TO_HAND, _gripper_poses(rng, 20), X=X_ETH, Y=Y_ETH, rng=rng,
                 t_noise=0.0008, r_noise=np.radians(0.2))
    bad = (2, 7, 11, 16)
    for i in bad:  # 20 % of the set: wrong marker / motion blur / PnP flip
        M = s[i].T_marker2cam.copy()
        M[:3, 3] += rng.normal(scale=0.15, size=3)
        s[i] = HandEyeSample(s[i].T_gripper2base, M, label=s[i].label)
    fit = solve_hand_eye(s, EYE_TO_HAND)
    assert set(bad) <= set(fit.outliers)
    assert np.linalg.norm(fit.T_cam2base[:3, 3] - X_ETH[:3, 3]) < 0.005
    # Metrics are on the inliers, so the bad samples do not mask a good fit.
    assert fit.metrics["translation_rmse_m"] < 0.003


def test_a_flipped_marker_reading_is_an_outlier(rng):
    """The planar-marker PnP ambiguity: one reading's normal flipped. WRC
    shipped `flip_hand_eye.py` to repair whole fits after the fact; with a
    joint robust solve it is just one more rejected sample."""
    s = _samples(EYE_TO_HAND, _gripper_poses(rng, 20), X=X_ETH, Y=Y_ETH, rng=rng,
                 t_noise=0.0008, r_noise=np.radians(0.2))
    M = s[5].T_marker2cam.copy()
    M[:3, :3] = M[:3, :3] @ so3_exp([np.pi * 0.9, 0.0, 0.0])
    s[5] = HandEyeSample(s[5].T_gripper2base, M)
    fit = solve_hand_eye(s, EYE_TO_HAND)
    assert 5 in fit.outliers
    assert np.linalg.norm(fit.T_cam2base[:3, 3] - X_ETH[:3, 3]) < 0.005


def test_reported_metrics_match_the_real_samples(rng):
    """Regression (WRC HANDOFF §11 issue 15): restarts were scored on a
    synthetic 'fake' dataset, so a file reported 5e-6 mm RMSE for a fit that
    was >100 mm off. Every number reported must be recomputable from the
    samples it claims to describe."""
    s = _samples(EYE_TO_HAND, _gripper_poses(rng, 12), X=X_ETH, Y=Y_ETH, rng=rng,
                 t_noise=0.0005, r_noise=np.radians(0.1))
    fit = solve_hand_eye(s, EYE_TO_HAND, n_restarts=4)
    res = np.array([pose_error(se3_inv(fit.T_marker) @ se3_inv(x.T_gripper2base)
                               @ fit.T_hand_eye @ x.T_marker2cam) for x in s])
    t = np.linalg.norm(res[list(fit.inliers), :3], axis=1)
    assert np.isclose(fit.metrics["translation_rmse_m"], np.sqrt(np.mean(t**2)), rtol=1e-6)
    assert np.allclose(fit.residuals, res, atol=1e-9)


def test_marker_on_the_tcp_baseline(rng):
    """WRC PLAN §6.3 #4: with the marker exactly at the TCP (Y = I) the fit
    must agree with X = G M^-1 and collapse Y to identity."""
    s = _samples(EYE_TO_HAND, _gripper_poses(rng, 10), X=X_ETH, Y=np.eye(4))
    fit = solve_hand_eye(s, EYE_TO_HAND)
    assert np.allclose(fit.T_cam2base, X_ETH, atol=1e-6)
    assert np.allclose(fit.T_marker, np.eye(4), atol=1e-6)


def test_too_few_samples_raises():
    s = [HandEyeSample(np.eye(4), np.eye(4))] * 3
    with pytest.raises(ValueError, match="at least 4"):
        solve_hand_eye(s, EYE_TO_HAND)


def test_non_rigid_input_raises():
    s = [HandEyeSample(np.eye(4), np.eye(4)) for _ in range(5)]
    s[0] = HandEyeSample(np.eye(4) * 2.0, np.eye(4))
    with pytest.raises(ValueError, match="SE\\(3\\)"):
        solve_hand_eye(s, EYE_TO_HAND)


def test_unknown_mode_raises(rng):
    s = _samples(EYE_TO_HAND, _gripper_poses(rng, 6), X=X_ETH, Y=Y_ETH)
    with pytest.raises(ValueError, match="mode"):
        solve_hand_eye(s, "eye_on_base")


# ── degeneracy: tiny residual, wrong answer ─────────────────────────────


def _translations_only(rng, n):
    w = [0.1, 0.2, 0.3]  # one fixed orientation, the TCP only translates
    return [_T(rng.uniform([0.22, -0.12, 0.20], [0.38, 0.10, 0.36]), w) for _ in range(n)]


def _single_axis(rng, n):
    axis = np.array([0.0, 0.0, 1.0])
    return [_T(rng.uniform([0.22, -0.12, 0.20], [0.38, 0.10, 0.36]), axis * a)
            for a in np.linspace(-0.8, 0.8, n)]


@pytest.mark.parametrize("make", [_translations_only, _single_axis])
def test_degenerate_pose_sets_are_refused_despite_a_tiny_residual(rng, make):
    s = _samples(EYE_TO_HAND, make(rng, 15), X=X_ETH, Y=Y_ETH)
    fit = solve_hand_eye(s, EYE_TO_HAND)
    # The trap: the residual says "perfect" ...
    assert fit.metrics["translation_rmse_m"] < 1e-4
    # ... but the rotations never constrained X, and the gate says so.
    assert fit.metrics["rotation_spread_deg"] < 1.0
    reasons = assess_hand_eye(fit.metrics)
    assert any("rotation spread" in r for r in reasons)


def test_rotation_spread_measures_axis_diversity(rng):
    single = [x[:3, :3] for x in _single_axis(rng, 10)]
    diverse = [x[:3, :3] for x in _gripper_poses(rng, 10)]
    assert rotation_spread_deg(single) < 1e-6
    assert rotation_spread_deg(diverse) > 10.0


def test_assess_names_every_failed_gate():
    reasons = assess_hand_eye({
        "translation_rmse_m": 0.05, "rotation_rmse_deg": 9.0,
        "rotation_spread_deg": 0.5, "n_inliers": 3, "n_samples": 10,
    })
    joined = " | ".join(reasons)
    for word in ("translation", "rotation rmse", "rotation spread", "inliers"):
        assert word in joined, joined


def test_assess_refuses_missing_or_nonfinite_metrics():
    assert assess_hand_eye({})
    assert assess_hand_eye({"translation_rmse_m": float("nan"), "rotation_rmse_deg": 0.1,
                            "rotation_spread_deg": 30.0, "n_inliers": 20, "n_samples": 20})
