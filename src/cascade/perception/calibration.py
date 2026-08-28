"""Fit a camera->base extrinsic from observed point correspondences.

`Extrinsics` (perception/grounding.py) consumes a 4x4 `T_cam2base`, but until
now nothing in this repo PRODUCED one: profiles carried a hand-entered matrix
or an npz from the baseline's tooling. This module is the missing half -- the
math, isolated from any camera or arm so it can be tested offline.

The method is Kabsch/Umeyama: given N>=3 non-collinear points expressed in both
frames, recover the rigid transform (rotation + translation, no scale) that
best maps one onto the other in a least-squares sense. On a robot arm the
correspondences come for free: drive the TCP to a known joint pose (its
base-frame position is FK, exact) and observe where the gripper lands in the
camera (its camera-frame position is depth back-projection). RPent's SO-101
calibrator does the same thing, detecting the jaw by toggling the gripper open
and closed between frames.

WHY RMSE IS RETURNED AND CHECKED, not just the matrix: a rigid fit ALWAYS
succeeds. Feed it garbage correspondences and it returns a confident, wrong
transform, and every pixel the agent back-projects afterwards lands somewhere
plausible but false -- the arm then reaches for targets that are centimetres
off, which reads as a grasping problem, not a calibration one. The residual is
the only cheap signal that separates the two, so `fit_extrinsic` reports it and
`CalibrationResult.acceptable` gates on it.

DEGENERATE GEOMETRY IS THE OTHER TRAP. Points sampled on a plane (a common
mistake: moving the TCP around at one height) leave the out-of-plane direction
unconstrained; the fit still reports a small RMSE while being badly wrong along
z. `fit_extrinsic` measures the point cloud's smallest singular value and
refuses the fit when the spread is degenerate, which is why the calibration
routine must sample several heights.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: A fit whose RMSE exceeds this is treated as failed. ~2 cm is roughly the
#: point where back-projected grasp targets start missing a 55 mm jaw; a good
#: fit on a rigid rig is a few millimetres.
MAX_ACCEPTABLE_RMSE_M = 0.02

#: Minimum extent (metres) the correspondence cloud must span in its THINNEST
#: direction. Below this the geometry is effectively planar and the fit is
#: unconstrained out of plane, however small its residual looks.
#:
#: 0.015 is MEASURED, not chosen: a small tabletop arm has very little
#: vertical room to play with. The SO-101's whole reachable top-down band is
#: ~0.06 m tall, and its best full-workspace sample cloud (46 poses over
#: r = 0.16..0.28 m, z = 0.02..0.08 m) spans 0.019 m in its thinnest
#: direction. A 0.03 threshold is therefore unreachable for that arm and
#: would reject every real calibration -- a gate nobody can pass gets deleted,
#: which is worse than a gate tuned to the hardware. Verified against the
#: degenerate case: a genuinely coplanar cloud still scores 0.000.
MIN_SPREAD_M = 0.015


@dataclass
class CalibrationResult:
    """A fitted extrinsic plus the diagnostics needed to distrust it."""

    T_cam2base: np.ndarray
    rmse_m: float
    max_error_m: float
    num_points: int
    spread_m: float

    @property
    def acceptable(self) -> bool:
        return (
            self.rmse_m <= MAX_ACCEPTABLE_RMSE_M
            and self.spread_m >= MIN_SPREAD_M
            and self.num_points >= 4
        )

    def summary(self) -> str:
        verdict = "OK" if self.acceptable else "REJECTED"
        return (
            f"[{verdict}] {self.num_points} points, rmse {self.rmse_m * 1000:.1f} mm, "
            f"max {self.max_error_m * 1000:.1f} mm, spread {self.spread_m * 100:.1f} cm"
        )


def fit_extrinsic(points_cam, points_base) -> CalibrationResult:
    """Rigid transform mapping camera-frame points onto base-frame points.

    Both arguments are (N, 3). Returns the fit AND its residuals -- callers
    must check `.acceptable` rather than assuming success, because a rigid fit
    never fails outright, it only fails quietly.

    Raises ValueError only for input that cannot be fitted at all (too few
    points, mismatched shapes, non-finite values), never for a poor fit: "this
    calibration is bad" is a result to report, not an exception.
    """
    P = np.asarray(points_cam, dtype=float).reshape(-1, 3)
    Q = np.asarray(points_base, dtype=float).reshape(-1, 3)
    if P.shape != Q.shape:
        raise ValueError(f"shape mismatch: {P.shape} camera vs {Q.shape} base points")
    if P.shape[0] < 3:
        raise ValueError(f"need at least 3 correspondences, got {P.shape[0]}")
    if not (np.all(np.isfinite(P)) and np.all(np.isfinite(Q))):
        raise ValueError("correspondences contain NaN/inf (bad depth reads?)")

    pc, qc = P.mean(axis=0), Q.mean(axis=0)
    Pc, Qc = P - pc, Q - qc

    # Kabsch: SVD of the cross-covariance gives the optimal rotation.
    U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    # The sign correction is what keeps this a ROTATION. Without it a noisy
    # or near-degenerate cloud can yield a reflection (det = -1), which fits
    # the points beautifully and mirrors the workspace -- left/right swapped.
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = qc - R @ pc

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    residuals = np.linalg.norm((P @ R.T + t) - Q, axis=1)
    # Smallest singular value of the centred camera cloud = its extent in the
    # thinnest direction, normalised per point so it reads in metres.
    spread = float(np.linalg.svd(Pc, compute_uv=False)[-1] / np.sqrt(len(P)))
    return CalibrationResult(
        T_cam2base=T,
        rmse_m=float(np.sqrt((residuals**2).mean())),
        max_error_m=float(residuals.max()),
        num_points=int(P.shape[0]),
        spread_m=spread,
    )


def save_extrinsic(path, result: CalibrationResult, *, camera: str = "",
                   note: str = "") -> Path:
    """Persist a fit as JSON, diagnostics included.

    The residuals are stored beside the matrix on purpose: six months later
    the only way to tell a trustworthy extrinsic from a confident-but-wrong
    one is the RMSE it was accepted with.
    """
    import time

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "camera": camera,
        "T_cam2base": np.asarray(result.T_cam2base, dtype=float).tolist(),
        "rmse_m": result.rmse_m,
        "max_error_m": result.max_error_m,
        "num_points": result.num_points,
        "spread_m": result.spread_m,
        "acceptable": result.acceptable,
        # Wall clock: this is a persisted record, not an in-process timestamp.
        "saved_at": time.time(),
        "note": note,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_extrinsic(path, *, trust_unacceptable: bool = False) -> np.ndarray | None:
    """Load a saved 4x4 `T_cam2base`, or None when there is nothing to trust.

    A record saved as unacceptable returns None unless explicitly overridden:
    silently using a rejected calibration is exactly the failure this module
    exists to prevent, and "no extrinsics" degrades visibly (no 3D fusion)
    whereas "wrong extrinsics" degrades invisibly.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    if not data.get("acceptable", False) and not trust_unacceptable:
        return None
    return np.asarray(data["T_cam2base"], dtype=float).reshape(4, 4)
