"""Rigid-transform helpers for hand-eye calibration.

Ported from WRC's ``calibration/frame_convention.py`` and the Lie-group half
of ``handeye_solver.py`` (Seeed fork of the July-2026 cascade baseline), with
two defects fixed that a real calibration hits:

* ``so3_log`` near pi. The textbook form divides by ``sin(theta)``, which is
  ~0 for a half-turn, and returned a garbage axis there. A planar marker read
  "from behind" (the PnP flip ambiguity) produces exactly such residuals, so
  the robust path below recovers the axis from the symmetric part instead.
* ``is_se3`` accepted NaN-poisoned matrices; they are rejected here (NaNs
  from a failed PnP must never reach the solver as a "valid" pose).
* ``se3_log``/``se3_exp`` did not round-trip (see ``se3_log``); the raw
  residual the solver needs is now its own function, ``pose_error``.

Convention: every 4x4 is ``T_a2b`` = maps points expressed in frame a into
frame b (cascade's ``T_cam2base`` naming). Twists are ``(tx, ty, tz, wx, wy,
wz)`` -- translation first, matching WRC so dataset metrics stay comparable.
Pure numpy; nothing here touches a camera, an arm or an SDK.
"""

from __future__ import annotations

import numpy as np


def _skew(v) -> np.ndarray:
    x, y, z = (float(c) for c in np.asarray(v, dtype=float).reshape(3))
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(omega) -> np.ndarray:
    """Rotation vector -> 3x3 rotation matrix (Rodrigues)."""
    omega = np.asarray(omega, dtype=float).reshape(3)
    theta = float(np.linalg.norm(omega))
    if theta < 1e-12:
        return np.eye(3) + _skew(omega)
    K = _skew(omega / theta)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def so3_log(R) -> np.ndarray:
    """3x3 rotation matrix -> rotation vector, stable over [0, pi]."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    cos_t = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(cos_t))
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-6:
        # First-order: log(R) ~ (R - R^T)/2.
        return 0.5 * vee
    if np.pi - theta > 1e-3:
        return vee * (theta / (2.0 * np.sin(theta)))
    # Near pi the antisymmetric part vanishes; recover the axis from the
    # symmetric part instead: at theta == pi, R = 2 a a^T - I, so
    # (R + I) / 2 = a a^T (symmetrised to drop the residual sin(theta) K).
    B = (R + R.T) / 4.0 + np.eye(3) / 2.0
    k = int(np.argmax(np.diag(B)))
    axis = B[:, k] / np.sqrt(max(B[k, k], 1e-300))
    axis /= np.linalg.norm(axis)
    # Fix the sign with whatever antisymmetric signal is left (it is exact
    # at theta == pi, where +axis and -axis are the same rotation).
    if float(np.dot(axis, vee)) < 0.0:
        axis = -axis
    return axis * theta


def se3_exp(xi) -> np.ndarray:
    """Twist ``(t, omega)`` -> 4x4, the group exponential (V-matrix form)."""
    xi = np.asarray(xi, dtype=float).reshape(6)
    t, omega = xi[:3], xi[3:]
    theta = float(np.linalg.norm(omega))
    W = _skew(omega)
    if theta < 1e-9:
        V = np.eye(3) + 0.5 * W
    else:
        V = (np.eye(3)
             + (1.0 - np.cos(theta)) / theta**2 * W
             + (theta - np.sin(theta)) / theta**3 * (W @ W))
    T = np.eye(4)
    T[:3, :3] = so3_exp(omega)
    T[:3, 3] = V @ t
    return T


def se3_log(T) -> np.ndarray:
    """4x4 -> twist ``(t, omega)``: the true group log, inverse of se3_exp.

    WRC's version returned the raw translation here while its ``se3_exp``
    applied the V-matrix, so the pair did not round-trip. The solver's
    residual wants the raw translation (a metric position error), and that
    is ``pose_error`` below -- two names for two different things.
    """
    T = np.asarray(T, dtype=float).reshape(4, 4)
    omega = so3_log(T[:3, :3])
    theta = float(np.linalg.norm(omega))
    W = _skew(omega)
    if theta < 1e-9:
        V_inv = np.eye(3) - 0.5 * W
    else:
        half = theta / 2.0
        V_inv = (np.eye(3) - 0.5 * W
                 + (1.0 - half / np.tan(half)) / theta**2 * (W @ W))
    return np.concatenate([V_inv @ T[:3, 3], omega])


def pose_error(T) -> np.ndarray:
    """4x4 -> ``(translation, rotation vector)`` of a would-be identity.

    The hand-eye residual: its first half is the metric offset (metres)
    between where FK and the camera put the marker, its second the angle-axis
    disagreement (radians). Thresholds read directly in mm / degrees.
    """
    T = np.asarray(T, dtype=float).reshape(4, 4)
    return np.concatenate([T[:3, 3], so3_log(T[:3, :3])])


def se3_inv(T) -> np.ndarray:
    """Inverse of a rigid transform without a general matrix inverse."""
    T = np.asarray(T, dtype=float).reshape(4, 4)
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def is_se3(T, tol: float = 1e-3) -> bool:
    """True when ``T`` is a finite 4x4 proper rigid transform."""
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        return False
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=tol):
        return False
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=tol):
        return False
    # A reflection is orthogonal too; it mirrors the workspace.
    return bool(np.isclose(np.linalg.det(R), 1.0, atol=tol))


def round_trip_check(T_ab, T_bc, *, atol_rot_deg: float = 0.5,
                     atol_trans_mm: float = 0.5) -> tuple[bool, float, float]:
    """Compose two transforms and report their drift from identity.

    Returns ``(ok, rotation_deg, translation_mm)``. Pass ``T`` and its
    supposed inverse to check a round trip.
    """
    T = np.asarray(T_ab, dtype=float) @ np.asarray(T_bc, dtype=float)
    rot_deg = float(np.degrees(np.linalg.norm(so3_log(T[:3, :3]))))
    trans_mm = float(np.linalg.norm(T[:3, 3])) * 1000.0
    return (rot_deg <= atol_rot_deg and trans_mm <= atol_trans_mm,
            rot_deg, trans_mm)


def fk_convention_offset(T_tcp_a, T_tcp_b) -> np.ndarray:
    """Constant offset ``S`` with ``T_b = S @ T_a`` for one joint vector.

    Two FK conventions on the same hardware (e.g. WRC's cuRobo ``end_link``
    vs cascade's ``gripper_end``, or a different ``joint_signs`` bake) differ
    by a rigid ``S`` only if the outputs are both rigid; anything else means
    the two models do not describe the same robot and must not be bridged.
    """
    if not (is_se3(T_tcp_a) and is_se3(T_tcp_b)):
        raise ValueError("FK outputs must be SE(3) to compute a constant offset")
    return np.asarray(T_tcp_b, dtype=float) @ se3_inv(T_tcp_a)
