"""SE(3) helpers used by hand-eye calibration (ported from WRC's
`calibration/frame_convention.py` + the Lie-group half of `handeye_solver.py`).

These are pure numpy and back every residual the solver minimises, so the
edge cases that bite a real calibration are pinned here: rotations near pi
(where the textbook log map divides by sin(theta) ~ 0) and reflections
(det = -1, which pass a naive orthogonality check and mirror the workspace).
"""

from __future__ import annotations

import numpy as np
import pytest

from cascade.calibration.frames import (
    fk_convention_offset,
    is_se3,
    round_trip_check,
    se3_exp,
    se3_inv,
    se3_log,
    so3_exp,
    so3_log,
)


def _random_rot(rng) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_so3_log_exp_round_trip(rng):
    for _ in range(50):
        R = _random_rot(rng)
        assert np.allclose(so3_exp(so3_log(R)), R, atol=1e-9)


def test_so3_log_is_stable_near_pi():
    """The textbook log divides by sin(theta); at theta ~ pi that is ~0 and
    the axis comes back as garbage. A flipped marker reading lands exactly
    there, so the solver must get a finite, correct rotation vector."""
    axis = np.array([1.0, 2.0, -0.5])
    axis /= np.linalg.norm(axis)
    for theta in (np.pi, np.pi - 1e-7, np.pi - 1e-4):
        w = so3_log(so3_exp(axis * theta))
        assert np.all(np.isfinite(w))
        assert np.isclose(np.linalg.norm(w), theta, atol=1e-6)
        assert np.allclose(so3_exp(w), so3_exp(axis * theta), atol=1e-6)


def test_so3_log_of_identity_is_zero():
    assert np.allclose(so3_log(np.eye(3)), 0.0)


def test_se3_exp_log_round_trip(rng):
    for _ in range(30):
        xi = np.concatenate([rng.uniform(-0.5, 0.5, 3), rng.normal(scale=1.0, size=3)])
        T = se3_exp(xi)
        assert is_se3(T)
        assert np.allclose(se3_exp(se3_log(T)), T, atol=1e-9)


def test_se3_log_zero_for_identity():
    assert np.allclose(se3_log(np.eye(4)), np.zeros(6), atol=1e-12)


def test_se3_inv(rng):
    T = se3_exp(np.concatenate([rng.normal(size=3), rng.normal(size=3)]))
    assert np.allclose(se3_inv(T) @ T, np.eye(4), atol=1e-12)


def test_pose_error_reads_the_raw_metric_offset():
    """The solver's residual must be the marker's metric offset, not the
    twist's V^-1-mapped translation (they differ once there is rotation)."""
    from cascade.calibration.frames import pose_error

    T = se3_exp(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5]))
    T[:3, 3] = [0.003, -0.004, 0.0]
    e = pose_error(T)
    assert np.allclose(e[:3], [0.003, -0.004, 0.0])
    assert np.allclose(e[3:], [0.0, 0.0, 0.5])
    assert not np.allclose(se3_log(T)[:3], e[:3])


def test_is_se3_accepts_rigid_transforms():
    T = np.eye(4)
    assert is_se3(T)
    T[:3, 3] = [0.1, 0.2, 0.3]
    assert is_se3(T)


@pytest.mark.parametrize("bad", ["scaled", "last_row", "shape", "reflection", "nan"])
def test_is_se3_rejects_non_rigid(bad):
    T = np.eye(4)
    if bad == "scaled":
        T[0, 0] = 2.0
    elif bad == "last_row":
        T[3, 0] = 1.0
    elif bad == "shape":
        T = np.eye(3)
    elif bad == "reflection":
        # Orthogonal but det = -1: mirrors the workspace (left/right swap).
        T[0, 0] = -1.0
    elif bad == "nan":
        T[0, 3] = np.nan
    assert not is_se3(T)


def test_round_trip_identity():
    ok, rot_deg, trans_mm = round_trip_check(np.eye(4), np.eye(4))
    assert ok and rot_deg < 1e-6 and trans_mm < 1e-6


def test_round_trip_reports_drift():
    T = se3_exp(np.array([0.002, 0.0, 0.0, 0.0, 0.0, np.radians(1.0)]))
    ok, rot_deg, trans_mm = round_trip_check(T, np.eye(4))
    assert not ok
    assert np.isclose(rot_deg, 1.0, atol=1e-6)
    assert trans_mm > 1.0


def test_fk_convention_offset_recovers_a_constant_offset(rng):
    """Two FK conventions on the same hardware differ by one rigid S."""
    S = se3_exp(np.array([0.01, -0.02, 0.03, 0.0, 0.0, 0.3]))
    A = se3_exp(np.concatenate([rng.uniform(-0.3, 0.3, 3), rng.normal(size=3)]))
    assert np.allclose(fk_convention_offset(A, S @ A), S, atol=1e-10)


def test_fk_convention_offset_refuses_non_rigid_input():
    with pytest.raises(ValueError):
        fk_convention_offset(np.eye(4) * 2.0, np.eye(4))
