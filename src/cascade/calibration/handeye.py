"""Joint SE(3) hand-eye solver for both mountings.

Ported from WRC ``calibration/handeye_solver.py`` (joint SE(3) LM, Huber,
3*MAD outlier rejection, restarts). The structure is WRC's; the numerics
were rewritten because the original had defects that a real rig exposed:

* Huber was applied by returning ``0.5 r^2`` *as the residual* to a squared-
  loss least_squares call (a quartic loss, the opposite of robust). Here it
  is iteratively reweighted least squares around Levenberg-Marquardt.
* The outlier rule kept ``norm <= 3*MAD`` -- measured from ZERO, not from the
  median -- so uniform clean noise lost a third of its samples (the shipped
  real-rig result reports 14 of 37 rejected). Here: ``norm > median +
  3 * max(1.4826*MAD, floor)`` and the fit is redone on the inliers.
* Restarts seeded from a synthetic "fake" dataset and were (at first) scored
  on it; the defect reached a real file reporting 5e-6 mm for a fit >100 mm
  off. Seeds here are closed-form solves on the REAL samples (all, and random
  subsets for robustness to outliers in the seed), always scored on them.
* Eye-in-hand reused the eye-to-hand equation, which has no constant
  solution when the camera rides on the wrist. Both mountings are now the
  same canonical problem::

      A_i X B_i = Z
      eye_to_hand: A_i = G_i^-1, X = T_cam2base,    Z = T_marker2gripper
      eye_in_hand: A_i = G_i,    X = T_cam2gripper, Z = T_marker2base

  with G_i = T_gripper2base (FK of the measured joints) and B_i = M_i =
  T_marker2cam (square-marker PnP).

* No scipy: the 12-parameter LM is a few lines of numpy, and scipy is not a
  base dependency (CI's minimal-install job runs without it).

A FIT ALWAYS CONVERGES. The per-sample residual cannot tell a good fit from
one the poses never constrained: rotating the TCP about a single axis (or not
at all) leaves X's translation along that axis free, and the solver happily
reports ~0 mm. ``rotation_spread_deg`` measures exactly that, and
``assess_hand_eye`` refuses it -- the hand-eye twin of the Kabsch fit's
``spread_m`` coplanarity gate in ``perception/calibration.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..perception.calibration import MAX_ACCEPTABLE_RMSE_M
from .frames import is_se3, se3_exp, se3_inv, so3_log

EYE_TO_HAND = "eye_to_hand"
EYE_IN_HAND = "eye_in_hand"
MODES = (EYE_TO_HAND, EYE_IN_HAND)

#: Translation residual gate: the SAME number the Kabsch extrinsic fit uses
#: (~2 cm is where back-projected grasp targets start missing a 55 mm jaw).
#: A good rig is a few mm (WRC's real ETH run: 3.0 mm RMSE).
MAX_TRANSLATION_RMSE_M = MAX_ACCEPTABLE_RMSE_M
#: Rotation residual gate. Single-marker PnP orientation noise is ~0.5-1.5 deg
#: at 0.5-1 m (WRC's real ETH run: 1.09 deg RMSE, 1.67 max); 3 deg is well
#: above that and well below a wrong-branch / flipped solution.
MAX_ROTATION_RMSE_DEG = 3.0
#: Degeneracy gate: second singular value of the relative-rotation cloud.
#: Below this the poses rotated about (at most) one axis and X is not
#: determined, whatever the residual says. WRC's preset sweeps (roll/yaw
#: +-0.5..1.0 rad, pitch 0..0.4 rad) score tens of degrees.
MIN_ROTATION_SPREAD_DEG = 8.0
#: 12 unknowns, 6 equations per sample: 4 is the algebraic minimum, but a
#: fit that only just has enough samples has no redundancy to detect a bad one.
MIN_INLIERS = 8
#: If most samples disagree with the fit, the "inliers" are a coincidence,
#: not a consensus (e.g. half the readings flipped).
MIN_INLIER_FRACTION = 0.5
#: Characteristic length that turns the rotation residual (rad) into metres
#: for the joint cost: 1 deg ~ 1.7 mm at 0.1 m.
ROT_WEIGHT_M = 0.1


@dataclass(frozen=True)
class HandEyeSample:
    """One (FK, PnP) observation. Both transforms are 4x4 SE(3)."""

    T_gripper2base: np.ndarray
    T_marker2cam: np.ndarray
    label: str = ""
    q: tuple | None = None          # measured joints the FK came from (audit)
    reprojection_px: float = 0.0


@dataclass(frozen=True)
class HandEyeFit:
    """Solver output: the two transforms, the per-sample evidence, metrics."""

    mode: str
    T_hand_eye: np.ndarray   # eye_to_hand: T_cam2base; eye_in_hand: T_cam2gripper
    T_marker: np.ndarray     # eye_to_hand: T_marker2gripper; eye_in_hand: T_marker2base
    inliers: tuple
    outliers: tuple
    residuals: np.ndarray    # (N, 6) raw [t (m), rotvec (rad)] per sample
    metrics: dict = field(default_factory=dict)

    def _need(self, mode: str, name: str) -> None:
        if self.mode != mode:
            raise AttributeError(f"{name} is only defined for {mode} fits (this is {self.mode})")

    @property
    def T_cam2base(self) -> np.ndarray:
        self._need(EYE_TO_HAND, "T_cam2base")
        return self.T_hand_eye

    @property
    def T_marker2gripper(self) -> np.ndarray:
        self._need(EYE_TO_HAND, "T_marker2gripper")
        return self.T_marker

    @property
    def T_cam2gripper(self) -> np.ndarray:
        self._need(EYE_IN_HAND, "T_cam2gripper")
        return self.T_hand_eye

    @property
    def T_marker2base(self) -> np.ndarray:
        self._need(EYE_IN_HAND, "T_marker2base")
        return self.T_marker

    @property
    def rejection_reasons(self) -> list[str]:
        return assess_hand_eye(self.metrics)

    @property
    def acceptable(self) -> bool:
        return not self.rejection_reasons

    def summary(self) -> str:
        m = self.metrics
        verdict = "OK" if self.acceptable else "REJECTED"
        return (
            f"[{verdict}] {self.mode}: {m.get('n_inliers', 0)}/{m.get('n_samples', 0)} inliers, "
            f"translation rmse {m.get('translation_rmse_m', float('nan')) * 1000:.1f} mm "
            f"(max {m.get('translation_max_m', float('nan')) * 1000:.1f}), "
            f"rotation rmse {m.get('rotation_rmse_deg', float('nan')):.2f} deg, "
            f"rotation spread {m.get('rotation_spread_deg', float('nan')):.1f} deg"
        )


# ── quality gate ─────────────────────────────────────────────────────────


def assess_hand_eye(metrics: dict) -> list[str]:
    """Every reason this fit must not be trusted; empty when acceptable.

    Missing or non-finite metrics are reasons too: a record that cannot show
    its quality is treated as one that failed it.
    """
    def get(key):
        v = metrics.get(key) if metrics else None
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) else None

    reasons = []
    t = get("translation_rmse_m")
    r = get("rotation_rmse_deg")
    spread = get("rotation_spread_deg")
    n_in = get("n_inliers")
    n = get("n_samples")
    if t is None:
        reasons.append("translation rmse missing/non-finite")
    elif t > MAX_TRANSLATION_RMSE_M:
        reasons.append(f"translation rmse {t * 1000:.1f} mm > {MAX_TRANSLATION_RMSE_M * 1000:.0f} mm")
    if r is None:
        reasons.append("rotation rmse missing/non-finite")
    elif r > MAX_ROTATION_RMSE_DEG:
        reasons.append(f"rotation rmse {r:.2f} deg > {MAX_ROTATION_RMSE_DEG:.1f} deg")
    if spread is None:
        reasons.append("rotation spread missing/non-finite")
    elif spread < MIN_ROTATION_SPREAD_DEG:
        reasons.append(
            f"rotation spread {spread:.1f} deg < {MIN_ROTATION_SPREAD_DEG:.0f} deg: the poses "
            "rotated about at most one axis, so the transform is unconstrained -- add "
            "roll, pitch AND yaw variation")
    if n_in is None or n is None:
        reasons.append("inlier counts missing")
    else:
        if n_in < MIN_INLIERS:
            reasons.append(f"only {int(n_in)} inliers (< {MIN_INLIERS})")
        if n > 0 and n_in / n < MIN_INLIER_FRACTION:
            reasons.append(f"inliers {int(n_in)}/{int(n)} below {MIN_INLIER_FRACTION:.0%}")
    return reasons


def rotation_spread_deg(rotations) -> float:
    """How many independent rotation axes the pose set exercised, in degrees.

    Second singular value (RMS-normalised) of the cloud of pairwise relative
    rotation vectors ``log(R_j R_i^T)``. Rotations about one fixed axis --
    in the base OR the tool frame -- make every relative rotation parallel,
    and this is ~0; diverse rotations make it tens of degrees.
    """
    Rs = [np.asarray(R, dtype=float)[:3, :3] for R in rotations]
    vecs = [so3_log(Rs[j] @ Rs[i].T) for i in range(len(Rs)) for j in range(i + 1, len(Rs))]
    if len(vecs) < 2:
        return 0.0
    s = np.linalg.svd(np.asarray(vecs), compute_uv=False)
    return float(np.degrees(s[1] / math.sqrt(len(vecs))))


# ── batched residuals ────────────────────────────────────────────────────


def _so3_log_batch(R: np.ndarray) -> np.ndarray:
    cos_t = np.clip((np.trace(R, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_t)
    vee = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0],
                    R[:, 1, 0] - R[:, 0, 1]], axis=1)
    out = 0.5 * vee
    mid = (theta >= 1e-6) & (np.pi - theta > 1e-3)
    if np.any(mid):
        out[mid] = vee[mid] * (theta[mid] / (2.0 * np.sin(theta[mid])))[:, None]
    for i in np.nonzero(np.pi - theta <= 1e-3)[0]:
        out[i] = so3_log(R[i])
    return out


def _residuals(A, B, X, Z) -> np.ndarray:
    """(N, 6) raw residuals pose_error(Z^-1 A_i X B_i)."""
    E = se3_inv(Z)[None] @ A @ X[None] @ B
    return np.concatenate([E[:, :3, 3], _so3_log_batch(E[:, :3, :3])], axis=1)


def _weighted(res: np.ndarray, rot_weight: float) -> np.ndarray:
    return np.concatenate([res[:, :3], rot_weight * res[:, 3:]], axis=1)


# ── closed-form seed (Kronecker / Shah-style AX = ZC) ────────────────────


def _project_so3(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def _closed_form(A, B):
    """Linear seed for A_i X B_i = Z (as A_i X = Z C_i with C_i = B_i^-1)."""
    C = np.stack([se3_inv(b) for b in B])
    I3 = np.eye(3)
    rows = [np.hstack([np.kron(I3, a[:3, :3]), -np.kron(c[:3, :3].T, I3)])
            for a, c in zip(A, C)]
    _, _, Vt = np.linalg.svd(np.vstack(rows))
    v = Vt[-1]
    Rx = v[:9].reshape(3, 3, order="F")
    Rz = v[9:].reshape(3, 3, order="F")
    det = np.linalg.det(Rx)
    if not np.isfinite(det) or abs(det) < 1e-12:
        return None
    scale = np.sign(det) / abs(det) ** (1.0 / 3.0)
    Rx, Rz = _project_so3(Rx * scale), _project_so3(Rz * scale)
    # R_A t_X + t_A = R_Z t_C + t_Z  ->  [R_A, -I] [t_X; t_Z] = R_Z t_C - t_A
    M = np.vstack([np.hstack([a[:3, :3], -I3]) for a in A])
    rhs = np.concatenate([Rz @ c[:3, 3] - a[:3, 3] for a, c in zip(A, C)])
    sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = Rx, sol[:3]
    Z = np.eye(4)
    Z[:3, :3], Z[:3, 3] = Rz, sol[3:]
    return X, Z


# ── Levenberg-Marquardt with Huber IRLS ──────────────────────────────────


def _lm(A, B, X, Z, w, rot_weight, max_iter=60):
    """Minimise sum_i w_i |e_i|^2 over right-multiplicative updates of X, Z."""
    sw = np.sqrt(w)[:, None]

    def cost_vec(Xc, Zc):
        return (sw * _weighted(_residuals(A, B, Xc, Zc), rot_weight)).ravel()

    def apply(Xc, Zc, d):
        return Xc @ se3_exp(d[:6]), Zc @ se3_exp(d[6:])

    r = cost_vec(X, Z)
    cost = float(r @ r)
    lam = 1e-3
    h = 1e-7
    iters = 0
    rel, step = 1.0, np.ones(12)
    for iters in range(1, max_iter + 1):
        J = np.empty((r.size, 12))
        for k in range(12):
            d = np.zeros(12)
            d[k] = h
            J[:, k] = (cost_vec(*apply(X, Z, d)) - r) / h
        H = J.T @ J
        g = J.T @ r
        improved = False
        while lam < 1e12:
            step = np.linalg.solve(H + lam * (np.diag(np.diag(H)) + 1e-12 * np.eye(12)), -g)
            Xn, Zn = apply(X, Z, step)
            rn = cost_vec(Xn, Zn)
            cn = float(rn @ rn)
            if cn < cost:
                rel = (cost - cn) / max(cost, 1e-300)
                X, Z, r, cost = Xn, Zn, rn, cn
                lam = max(lam / 3.0, 1e-12)
                improved = True
                break
            lam *= 4.0
        if not improved or rel < 1e-12 or np.linalg.norm(step) < 1e-12:
            break
    return X, Z, iters


def _huber_weights(norms: np.ndarray, delta: float) -> np.ndarray:
    w = np.ones_like(norms)
    big = norms > delta
    w[big] = delta / norms[big]
    return w


def _robust_fit(A, B, X, Z, mask, rot_weight, huber_m, rounds=6):
    w = mask.astype(float)
    total_iters = 0
    for _ in range(rounds):
        X, Z, it = _lm(A, B, X, Z, w, rot_weight)
        total_iters += it
        norms = np.linalg.norm(_weighted(_residuals(A, B, X, Z), rot_weight), axis=1)
        w_new = _huber_weights(norms, huber_m) * mask
        if np.allclose(w_new, w, atol=1e-3):
            break
        w = w_new
    return X, Z, total_iters


# ── public entry point ───────────────────────────────────────────────────


def solve_hand_eye(
    samples,
    mode: str,
    *,
    huber_m: float = 0.003,
    rot_weight_m: float = ROT_WEIGHT_M,
    n_restarts: int = 8,
    seed: int = 0,
    mad_k: float = 3.0,
    outlier_floor_m: float = 0.002,
    outlier_rounds: int = 3,
) -> HandEyeFit:
    """Robust joint fit of both unknown transforms.

    Raises ValueError only for input that cannot be fitted at all (unknown
    mode, < 4 samples, non-rigid matrices). A poor or degenerate fit is a
    RESULT -- check ``fit.acceptable`` / ``assess_hand_eye(fit.metrics)``.
    """
    if mode not in MODES:
        raise ValueError(f"unknown hand-eye mode {mode!r}; expected one of {MODES}")
    samples = list(samples)
    if len(samples) < 4:
        raise ValueError(f"need at least 4 samples for a joint hand-eye solve, got {len(samples)}")
    for s in samples:
        if not (is_se3(s.T_gripper2base) and is_se3(s.T_marker2cam)):
            raise ValueError(f"sample {s.label or '?'} is not a pair of SE(3) matrices")
    G = np.stack([np.asarray(s.T_gripper2base, dtype=float) for s in samples])
    B = np.stack([np.asarray(s.T_marker2cam, dtype=float) for s in samples])
    A = np.stack([se3_inv(g) for g in G]) if mode == EYE_TO_HAND else G
    n = len(samples)
    everyone = np.ones(n, dtype=bool)

    def score(X, Z):
        return float(np.median(np.linalg.norm(
            _weighted(_residuals(A, B, X, Z), rot_weight_m), axis=1)))

    # Seeds: closed form on everything, then on random subsets (an outlier in
    # the linear seed can pull it into the wrong basin; a subset without it
    # cannot). Every candidate is refined and scored on ALL real samples.
    rng = np.random.default_rng(seed)
    seeds = []
    first = _closed_form(A, B)
    if first is not None:
        seeds.append(first)
    k = max(4, int(math.ceil(0.6 * n)))
    for _ in range(max(0, n_restarts)):
        idx = np.sort(rng.choice(n, size=min(k, n), replace=False))
        cand = _closed_form(A[idx], B[idx])
        if cand is not None:
            seeds.append(cand)
    if not seeds:  # every linear solve degenerate: fall back to the first sample
        Z0 = np.eye(4)
        seeds.append((se3_inv(A[0]) @ Z0 @ se3_inv(B[0]), Z0))

    best = None
    lm_iters = 0
    for X0, Z0 in seeds:
        X, Z, it = _robust_fit(A, B, X0, Z0, everyone, rot_weight_m, huber_m)
        lm_iters += it
        sc = score(X, Z)
        if best is None or sc < best[0]:
            best = (sc, X, Z)
    _, X, Z = best

    # 3*MAD outlier rejection around the MEDIAN, then refit on the inliers.
    mask = everyone
    for _ in range(max(1, outlier_rounds)):
        norms = np.linalg.norm(_weighted(_residuals(A, B, X, Z), rot_weight_m), axis=1)
        med = float(np.median(norms))
        mad = float(np.median(np.abs(norms - med)))
        thresh = med + mad_k * max(1.4826 * mad, outlier_floor_m)
        new_mask = norms <= thresh
        if new_mask.sum() < 4:
            break
        if np.array_equal(new_mask, mask) and mask is not everyone:
            break
        mask = new_mask
        X, Z, it = _robust_fit(A, B, X, Z, mask, rot_weight_m, huber_m)
        lm_iters += it

    res = _residuals(A, B, X, Z)
    inliers = tuple(int(i) for i in np.nonzero(mask)[0])
    outliers = tuple(int(i) for i in np.nonzero(~mask)[0])
    metrics = _metrics(res, mask, A, G, X, Z, mode)
    metrics.update(lm_iterations=int(lm_iters), n_seeds=len(seeds))
    return HandEyeFit(mode=mode, T_hand_eye=X, T_marker=Z, inliers=inliers,
                      outliers=outliers, residuals=res, metrics=metrics)


def _metrics(res, mask, A, G, X, Z, mode) -> dict:
    inl = res[mask]
    t = np.linalg.norm(inl[:, :3], axis=1)
    r = np.degrees(np.linalg.norm(inl[:, 3:], axis=1))
    # The points that were actually spread through space: the marker (eye-to-
    # hand, it rides on the gripper) or the camera (eye-in-hand). Diagnostic;
    # the degeneracy that matters is rotational (see rotation_spread_deg).
    pts = np.stack([(g @ Z)[:3, 3] if mode == EYE_TO_HAND else (g @ X)[:3, 3]
                    for g in G[mask]])
    centred = pts - pts.mean(axis=0)
    spread = float(np.linalg.svd(centred, compute_uv=False)[-1] / math.sqrt(len(pts)))
    return {
        "n_samples": int(len(res)),
        "n_inliers": int(mask.sum()),
        "n_outliers": int((~mask).sum()),
        "translation_rmse_m": float(np.sqrt(np.mean(t**2))),
        "translation_max_m": float(t.max()),
        "rotation_rmse_deg": float(np.sqrt(np.mean(r**2))),
        "rotation_max_deg": float(r.max()),
        "rotation_spread_deg": rotation_spread_deg(A[mask]),
        "position_spread_m": spread,
    }
