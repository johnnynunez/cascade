"""Markerless eye-to-hand calibration: the arm IS the target.

Reference method: Hydra (arXiv 2504.20584, "Marker-Free RGB-D Hand-Eye
Calibration"): robust point-to-plane ICP of the robot's mesh, posed by FK,
against depth, solved on the Lie algebra jointly over several joint
configurations. Here, pure numpy:

* MODEL. ``RobotSurface`` (the arm's own URDF meshes, skin only, posed by
  the profile's signed kinematics) at FK of the MEASURED joints of each
  pose. Per pose and per iteration only the points the camera can see at the
  current estimate are used: back faces culled, then a coarse z-buffer
  (approximate hidden-point removal) drops the self-occluded ones.
* DATA ASSOCIATION is projective with a small window: a model point is
  projected into a block-median depth pyramid and matched to the closest
  back-projected point in a (2w+1)^2 neighbourhood whose depth-gradient
  normal agrees with the model normal within 60 deg. The window gives the
  coarse levels a basin of several centimetres per iteration; the normal
  test is what keeps a link's SIDE from snapping onto the table.
* COST. Point-to-plane on the (exact) model normal, Tukey-weighted with a
  MAD scale, under a rejection distance annealed level by level (20 cm ->
  1 cm), so the table and background cannot capture the arm once it is
  close. The 6-DoF update rotates about the data centroid (decoupling
  translation from rotation in the normal matrix).
* INITIALISATION from the camera profile's current T (rough placeholders:
  ~0.6-0.95 m above, looking down) plus multi-start perturbations about the
  camera centre, each scored by fitness (fraction of visible model points
  the depth explains) on a coarse subset; the best two are refined on all
  poses and the best wins.

A FIT ALWAYS CONVERGES somewhere, so the result carries metrics and a gate
(``assess_markerless``), the ICP analogue of the marker gate: inlier
fraction, point-to-plane RMSE, per-pose consistency (where each pose ALONE
would pull the camera from the joint solution), pose diversity (count, TCP
spread, tool rotation spread) and a DEGENERACY check on the 6x6 normal
matrix (minimum normalised eigenvalue, condition number). One pose, a barely
visible arm (one face of one link), or a sweep that only translates the TCP
along a line all fit with a small residual -- and are refused.

Eye-in-hand is out of scope: a wrist camera does not see the arm.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np

from .frames import se3_inv, so3_exp, so3_log
from .handeye import EYE_IN_HAND, EYE_TO_HAND, MODES, rotation_spread_deg

METHOD = "depth_icp_markerless"

# ── gate thresholds ──────────────────────────────────────────────────────

#: Distinct poses with enough associated points. Hydra converges from three
#: configurations; a gate needs redundancy to see a bad one.
MIN_POSES = 6
#: A pose counts only with this many inlier points (a few links' worth).
MIN_POSE_INLIERS = 150
#: Total inlier points over all poses.
MIN_INLIERS = 3000
#: Fraction of the visible model points the depth explains. Below half, the
#: "fit" is mostly unexplained geometry (wrong basin, occluded arm).
MIN_INLIER_FRACTION = 0.5
#: Point-to-plane RMSE over inliers at the evaluation level. Dominated by
#: sensor noise (~0.75 % of range on a D455); 15 mm is far above a good fit
#: and below a wrong-basin one.
MAX_RMSE_M = 0.015
#: Per-pose consistency: how far each pose ALONE would move its own arm
#: surface (RMS) / turn the camera from the joint solution (well-constrained
#: directions only).
MAX_POSE_OFFSET_M = 0.010
MAX_POSE_OFFSET_DEG = 1.0
#: Degeneracy: smallest eigenvalue of the normalised 6x6 normal matrix
#: (translation in m, rotation scaled by the cloud's RMS radius), i.e. the
#: share of the information in the weakest direction; and its conditioning.
MIN_DEGENERACY_EIG = 0.01
MAX_CONDITION = 200.0
#: Diversity: spread of the TCP positions along their 2nd principal axis
#: (one pose / a line scores ~0) and of the tool orientations (the marker
#: gate's rotation spread; a translation-only sweep scores ~0).
MIN_TCP_SPREAD_M = 0.01
MIN_ROTATION_SPREAD_DEG = 5.0

#: Inlier = associated within this distance at the evaluation level.
INLIER_DIST_M = 0.012
_COS_NORMAL = math.cos(math.radians(60.0))


@dataclass(frozen=True)
class DepthSample:
    """One static pose: measured joints, their FK, and the depth seen there."""

    q: tuple
    T_gripper2base: np.ndarray
    depth_m: np.ndarray       # (H, W) float32 metres, 0 = invalid
    K: np.ndarray             # (3, 3) intrinsics of depth_m
    label: str = ""
    t: float = 0.0


@dataclass(frozen=True)
class _Level:
    alpha: float      # angular cell size (rad) -> stride = alpha * fx
    window: int       # +- cells searched around the projection
    dmax: float       # rejection distance (m)
    iters: int
    cap: int          # model points per pose


#: Coarse to fine. Window and rejection distance anneal together.
SCHEDULE = (
    _Level(0.05, 3, 0.20, 8, 400),
    _Level(0.025, 3, 0.10, 10, 800),
    _Level(0.0125, 2, 0.05, 8, 1500),
    _Level(0.006, 1, 0.02, 8, 3000),
    _Level(0.003, 1, 0.01, 6, 4000),
)
_EVAL = _Level(0.006, 1, INLIER_DIST_M, 0, 6000)


# ── depth pyramid ────────────────────────────────────────────────────────


def _block_median(depth: np.ndarray, s: int) -> np.ndarray:
    H, W = depth.shape
    h, w = H // s, W // s
    d = np.asarray(depth, dtype=np.float64)[: h * s, : w * s]
    if s == 1:
        return np.where(d > 0, d, np.nan)
    blk = d.reshape(h, s, w, s).transpose(0, 2, 1, 3).reshape(h, w, s * s)
    blk = np.where(blk > 0, blk, np.nan)
    n_ok = np.isfinite(blk).sum(axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        z = np.nanmedian(blk, axis=-1)
    z[n_ok < (s * s + 1) // 2] = np.nan
    return z


def _level_K(K, s):
    Ks = np.array(K, dtype=float, copy=True)
    Ks[0, 0] /= s
    Ks[1, 1] /= s
    Ks[0, 2] = (K[0, 2] - (s - 1) / 2.0) / s
    Ks[1, 2] = (K[1, 2] - (s - 1) / 2.0) / s
    return Ks


def _point_map(z, Ks):
    h, w = z.shape
    us, vs = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    return np.stack([(us - Ks[0, 2]) / Ks[0, 0] * z, (vs - Ks[1, 2]) / Ks[1, 1] * z, z], -1)


def _normal_map(P):
    h, w, _ = P.shape
    N = np.full_like(P, np.nan)
    if h < 3 or w < 3:
        return N
    c = P[1:-1, 1:-1]
    dx = P[1:-1, 2:] - P[1:-1, :-2]
    dy = P[2:, 1:-1] - P[:-2, 1:-1]
    n = np.cross(dx, dy)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = n / norm
    flip = np.sum(n * c, axis=-1) > 0
    n[flip] *= -1.0
    z = c[..., 2]
    jump = np.maximum.reduce([np.abs(P[1:-1, 2:, 2] - z), np.abs(P[1:-1, :-2, 2] - z),
                              np.abs(P[2:, 1:-1, 2] - z), np.abs(P[:-2, 1:-1, 2] - z)])
    with np.errstate(invalid="ignore"):
        bad = ~(jump < 0.05 * z)
    n[bad] = np.nan
    N[1:-1, 1:-1] = n
    return N


class _Pyramid:
    """Block-median point maps per stride, normals from a ~8 mm grid."""

    def __init__(self, depth, K):
        self.depth = depth
        self.K = np.asarray(K, dtype=float)
        self.size = (depth.shape[1], depth.shape[0])
        fx = self.K[0, 0]
        self.normal_stride = max(1, 2 ** int(round(math.log2(max(0.0115 * fx, 1.0)))))
        self._levels = {}

    def stride(self, alpha: float) -> int:
        return max(1, int(round(alpha * self.K[0, 0])))

    def level(self, s: int):
        if s not in self._levels:
            z = _block_median(self.depth, s)
            Ks = _level_K(self.K, s)
            P = _point_map(z, Ks)
            ns = self.normal_stride
            if s >= ns:
                N = _normal_map(P)
            else:
                Pn, Nn, Kn = self.level(ns)
                # Pixel centre of each fine cell -> the coarse normal cell.
                h, w = z.shape
                rows = np.clip(np.round((np.arange(h) * s + (s - 1) / 2.0 - (ns - 1) / 2.0) / ns)
                               .astype(int), 0, Nn.shape[0] - 1)
                cols = np.clip(np.round((np.arange(w) * s + (s - 1) / 2.0 - (ns - 1) / 2.0) / ns)
                               .astype(int), 0, Nn.shape[1] - 1)
                N = Nn[rows[:, None], cols[None, :]]
            self._levels[s] = (P, N, Ks)
        return self._levels[s]


@dataclass
class _Prepared:
    sample: DepthSample
    P: np.ndarray      # model points, base frame, at FK(q)
    N: np.ndarray      # model normals
    pyr: _Pyramid
    order: np.ndarray  # fixed permutation for per-level subsampling


def _prepare(samples, surface, seed=0) -> list[_Prepared]:
    out = []
    for i, s in enumerate(samples):
        cloud = surface.points_at(np.asarray(s.q, dtype=float))
        order = np.random.default_rng([seed, i]).permutation(len(cloud.points))
        out.append(_Prepared(s, cloud.points, cloud.normals,
                             _Pyramid(np.asarray(s.depth_m), s.K), order))
    return out


# ── visibility, association ──────────────────────────────────────────────


def _visible(pc, nc, K, size, spacing_m):
    """Front-facing, inside the image, and not behind other model points in
    a coarse z-buffer (cells ~2.5x the model's projected sample spacing)."""
    w, h = size
    z = pc[:, 2]
    ok = z > 0.05
    with np.errstate(invalid="ignore", divide="ignore"):
        u = K[0, 0] * pc[:, 0] / z + K[0, 2]
        v = K[1, 1] * pc[:, 1] / z + K[1, 2]
    facing = np.einsum("ij,ij->i", nc, pc) < -0.05 * np.linalg.norm(pc, axis=1)
    cand = ok & facing & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    if not cand.any():
        return cand
    zm = float(np.median(z[cand]))
    cell = max(2.0, 2.5 * spacing_m * K[0, 0] / zm)
    tol = max(0.015, cell * zm / K[0, 0])
    cu = (u[cand] / cell).astype(int)
    cv = (v[cand] / cell).astype(int)
    grid = np.full((cv.max() + 1, cu.max() + 1), np.inf)
    np.minimum.at(grid, (cv, cu), z[cand])
    vis = np.zeros_like(cand)
    vis[cand] = z[cand] <= grid[cv, cu] + tol
    return vis


def _associate(pc, nc, P, N, Ks, window, dmax):
    """Closest compatible data point in a (2w+1)^2 window around each model
    point's projection. Returns (q (M,3), nq (M,3), accepted (M,), dist)."""
    h, w = P.shape[:2]
    with np.errstate(invalid="ignore", divide="ignore"):
        ju = np.rint(Ks[0, 0] * pc[:, 0] / pc[:, 2] + Ks[0, 2]).astype(np.int64)
        jv = np.rint(Ks[1, 1] * pc[:, 1] / pc[:, 2] + Ks[1, 2]).astype(np.int64)
    r = np.arange(-window, window + 1)
    dv, du = np.meshgrid(r, r, indexing="ij")
    JV = jv[:, None] + dv.ravel()[None, :]
    JU = ju[:, None] + du.ravel()[None, :]
    inb = (JV >= 0) & (JV < h) & (JU >= 0) & (JU < w)
    JV, JU = np.clip(JV, 0, h - 1), np.clip(JU, 0, w - 1)
    Q = P[JV, JU]
    NQ = N[JV, JU]
    d2 = np.sum((Q - pc[:, None, :]) ** 2, axis=-1)
    compat = np.sum(NQ * nc[:, None, :], axis=-1) > _COS_NORMAL
    with np.errstate(invalid="ignore"):
        d2 = np.where(inb & compat & np.isfinite(d2), d2, np.inf)
    k = np.argmin(d2, axis=1)
    rows = np.arange(len(pc))
    best = d2[rows, k]
    acc = best <= dmax * dmax
    return Q[rows, k], NQ[rows, k], acc, np.sqrt(best)


def _tukey(r, c):
    w = np.zeros_like(r)
    m = np.abs(r) < c
    w[m] = (1.0 - (r[m] / c) ** 2) ** 2
    return w


# ── the joint ICP ────────────────────────────────────────────────────────


@dataclass
class _Assoc:
    """Per-pose associations at one estimate (base frame)."""

    p: np.ndarray       # model points
    n: np.ndarray       # model normals
    q: np.ndarray       # data points
    r: np.ndarray       # point-to-plane residuals
    dist: np.ndarray    # point distances
    n_visible: int


def _associate_all(prep, T, level, spacing_m):
    R, t = T[:3, :3], T[:3, 3]
    out = []
    for pr in prep:
        pc = (pr.P - t) @ R
        nc = pr.N @ R
        P, N, Ks = pr.pyr.level(pr.pyr.stride(level.alpha))
        vis = _visible(pc, nc, pr.sample.K, pr.pyr.size, spacing_m)
        idx = pr.order[vis[pr.order]][: level.cap]
        n_vis = int(vis.sum())
        if len(idx) == 0:
            e3 = np.empty((0, 3))
            out.append(_Assoc(e3, e3, e3, np.empty(0), np.empty(0), n_vis))
            continue
        qc, _, acc, dist = _associate(pc[idx], nc[idx], P, N, Ks, level.window, level.dmax)
        sel = idx[acc]
        qb = qc[acc] @ R.T + t
        r = np.einsum("ij,ij->i", pr.N[sel], pr.P[sel] - qb)
        out.append(_Assoc(pr.P[sel], pr.N[sel], qb, r, dist[acc], n_vis))
    return out


def _normal_equations(assocs, c, weights=None):
    H = np.zeros((6, 6))
    g = np.zeros(6)
    for k, a in enumerate(assocs):
        if len(a.r) == 0:
            continue
        w = np.ones(len(a.r)) if weights is None else weights[k]
        J = np.hstack([-a.n, -np.cross(a.q - c, a.n)])
        H += (J * w[:, None]).T @ J
        g += (J * w[:, None]).T @ a.r
    return H, g


def _apply(T, xi, c):
    """T <- M T, M rotating by xi[3:] about c and translating by xi[:3]."""
    R = so3_exp(xi[3:])
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = c - R @ c + xi[:3]
    return M @ T


def _solve_step(H, g, L, max_t=0.05, max_r=0.1, rel_floor=1e-6):
    """Eigen-truncated Gauss-Newton step: directions whose (normalised)
    information is below ``rel_floor`` of the strongest get NO update
    rather than a noise-driven one."""
    S = np.diag([1.0, 1.0, 1.0, 1.0 / L, 1.0 / L, 1.0 / L])
    Hs = S.T @ H @ S
    gs = S.T @ g
    lam, V = np.linalg.eigh(Hs)
    if lam[-1] <= 0:
        return np.zeros(6)
    keep = lam > rel_floor * lam[-1]
    step_s = -(V[:, keep] @ ((V[:, keep].T @ gs) / lam[keep]))
    xi = S @ step_s
    nt, nr = np.linalg.norm(xi[:3]), np.linalg.norm(xi[3:])
    if nt > max_t:
        xi[:3] *= max_t / nt
    if nr > max_r:
        xi[3:] *= max_r / nr
    return xi


def _icp(prep, T, schedule, spacing_m, c, L, rel_floor=1e-6):
    T = np.array(T, dtype=float, copy=True)
    for level in schedule:
        for _ in range(level.iters):
            assocs = _associate_all(prep, T, level, spacing_m)
            r = np.concatenate([a.r for a in assocs]) if assocs else np.empty(0)
            if len(r) < 12:
                break
            sigma = 1.4826 * float(np.median(np.abs(r)))
            cut = float(np.clip(4.685 * sigma, 0.004, level.dmax))
            weights = [_tukey(a.r, cut) for a in assocs]
            H, g = _normal_equations(assocs, c, weights)
            xi = _solve_step(H, g, L, rel_floor=rel_floor)
            T = _apply(T, xi, c)
            if np.linalg.norm(xi[:3]) < 5e-5 and np.linalg.norm(xi[3:]) < 5e-5:
                break
    return T


# ── evaluation, metrics ──────────────────────────────────────────────────


@dataclass(frozen=True)
class PoseStats:
    label: str
    n_visible: int
    n_inliers: int
    rmse_m: float
    offset_m: float
    offset_deg: float


def surface_displacement(M, pts) -> float:
    """RMS distance the points move under the base-frame correction M.

    The drift / consistency statistic: what a calibration change does to
    the measured arm surface (where grasps happen), not to the camera
    centre -- a 0.5 deg wobble about the arm's own centroid moves the
    camera centre ~4 mm at 0.7 m but the arm surface ~1 mm."""
    P = np.asarray(pts, dtype=float)
    if len(P) == 0:
        return math.nan
    moved = P @ M[:3, :3].T + M[:3, 3]
    return float(np.sqrt(np.mean(np.sum((moved - P) ** 2, axis=1))))


def _offset(T, xi, c, pts):
    """(surface displacement, rotation deg) of the update xi."""
    M = _apply(np.eye(4), xi, c)
    return surface_displacement(M, pts), float(np.degrees(np.linalg.norm(xi[3:])))


def _evaluate(prep, T, spacing_m, c, L, level=_EVAL):
    assocs = _associate_all(prep, T, level, spacing_m)
    poses, all_r = [], []
    inlier_assocs = []
    for pr, a in zip(prep, assocs):
        m = a.dist <= INLIER_DIST_M
        ai = _Assoc(a.p[m], a.n[m], a.q[m], a.r[m], a.dist[m], a.n_visible)
        inlier_assocs.append(ai)
        all_r.append(ai.r)
        n_in = int(m.sum())
        rmse = float(np.sqrt(np.mean(ai.r ** 2))) if n_in else math.nan
        off_m = off_deg = math.nan
        if n_in >= 12:
            Hi, gi = _normal_equations([ai], c)
            xi = _solve_step(Hi, gi, L, max_t=np.inf, max_r=np.inf)
            # Only the directions this pose constrains well (>= 2 % of its
            # strongest): a single view legitimately leaves some free.
            S = np.diag([1.0, 1.0, 1.0, 1.0 / L, 1.0 / L, 1.0 / L])
            lam, V = np.linalg.eigh(S @ Hi @ S)
            keep = lam > 0.02 * lam[-1]
            xs = np.linalg.solve(S, xi)
            xs = V[:, keep] @ (V[:, keep].T @ xs)
            off_m, off_deg = _offset(T, S @ xs, c, ai.p)
        poses.append(PoseStats(pr.sample.label, a.n_visible, n_in, rmse, off_m, off_deg))
    H, _ = _normal_equations(inlier_assocs, c)
    return poses, np.concatenate(all_r) if all_r else np.empty(0), H


def _fitness(prep, T, spacing_m, level):
    assocs = _associate_all(prep, T, level, spacing_m)
    vis = sum(a.n_visible for a in assocs)
    n_in = sum(int(np.sum(a.dist <= level.dmax)) for a in assocs)
    capped = sum(min(a.n_visible, level.cap) for a in assocs)
    return n_in / max(capped, 1) if vis else 0.0


def _degeneracy(H, n, L):
    S = np.diag([1.0, 1.0, 1.0, 1.0 / L, 1.0 / L, 1.0 / L])
    Hn = S @ H @ S / max(n, 1)
    lam = np.linalg.eigvalsh(Hn)
    lam_min = float(max(lam[0], 0.0))
    # Capped (finite) so the record stays plain JSON; 1e12 = "no
    # information at all in the weakest direction".
    cond = float(min(lam[-1] / lam_min, 1e12)) if lam_min > 0 else 1e12
    return lam_min, cond


def _diversity(samples):
    G = np.stack([np.asarray(s.T_gripper2base, dtype=float) for s in samples])
    pts = G[:, :3, 3]
    if len(pts) >= 2:
        sv = np.linalg.svd(pts - pts.mean(axis=0), compute_uv=False)
        spread = float(sv[1] / math.sqrt(len(pts))) if len(sv) > 1 else 0.0
    else:
        spread = 0.0
    rot = rotation_spread_deg([g[:3, :3] for g in G]) if len(G) >= 2 else 0.0
    return spread, rot


# ── public API ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MarkerlessFit:
    T_cam2base: np.ndarray
    metrics: dict
    poses: tuple = ()
    mode: str = EYE_TO_HAND

    @property
    def T_hand_eye(self) -> np.ndarray:
        return self.T_cam2base

    @property
    def rejection_reasons(self) -> list[str]:
        return assess_markerless(self.metrics)

    @property
    def acceptable(self) -> bool:
        return not self.rejection_reasons

    def summary(self) -> str:
        return markerless_summary(self.metrics, self.acceptable)


def markerless_summary(m: dict, ok: bool, camera: str = "") -> str:
    def f(k, scale=1.0, fmt=".1f"):
        try:
            return format(float(m.get(k, math.nan)) * scale, fmt)
        except (TypeError, ValueError):
            return "nan"
    return (f"[{'OK' if ok else 'REJECTED'}] markerless eye_to_hand"
            f"{' camera=' + camera if camera else ''}: {int(m.get('n_poses_used', 0))}/"
            f"{int(m.get('n_poses', 0))} poses, inliers {f('inlier_fraction', 100)} %, "
            f"rmse {f('rmse_m', 1000)} mm, pose offsets <= {f('pose_offset_max_m', 1000)} mm / "
            f"{f('pose_offset_max_deg', 1, '.2f')} deg, min eig {f('degeneracy_min_eig', 1, '.3f')}"
            f", cond {f('condition_number', 1, '.0f')}, tcp spread {f('tcp_spread_m', 1000)} mm, "
            f"rotation spread {f('rotation_spread_deg')} deg")


def assess_markerless(metrics: dict) -> list[str]:
    """Every reason this markerless fit must not be trusted; empty = OK.
    Missing / non-finite metrics are reasons too."""
    def get(key):
        v = metrics.get(key) if metrics else None
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) else None

    reasons = []
    n_used, n = get("n_poses_used"), get("n_poses")
    if n_used is None or n is None:
        reasons.append("pose counts missing/non-finite")
    elif n_used < MIN_POSES:
        reasons.append(f"only {int(n_used)} usable poses (< {MIN_POSES}; a pose needs >= "
                       f"{MIN_POSE_INLIERS} inlier points): add distinct joint configurations")
    n_in = get("n_inliers")
    if n_in is None:
        reasons.append("inlier count missing/non-finite")
    elif n_in < MIN_INLIERS:
        reasons.append(f"only {int(n_in)} inlier points (< {MIN_INLIERS}): the arm is barely "
                       "visible in depth")
    frac = get("inlier_fraction")
    if frac is None:
        reasons.append("inlier fraction missing/non-finite")
    elif frac < MIN_INLIER_FRACTION:
        reasons.append(f"inlier fraction {frac:.0%} < {MIN_INLIER_FRACTION:.0%}: the depth does "
                       "not explain the posed arm (wrong basin, occlusion, wrong arm model)")
    rmse = get("rmse_m")
    if rmse is None:
        reasons.append("rmse missing/non-finite")
    elif rmse > MAX_RMSE_M:
        reasons.append(f"point-to-plane rmse {rmse * 1000:.1f} mm > {MAX_RMSE_M * 1000:.0f} mm")
    om, od = get("pose_offset_max_m"), get("pose_offset_max_deg")
    if om is None or od is None:
        reasons.append("per-pose consistency missing/non-finite")
    else:
        if om > MAX_POSE_OFFSET_M:
            reasons.append(f"poses disagree: one pose alone moves the arm {om * 1000:.1f} mm "
                           f"(> {MAX_POSE_OFFSET_M * 1000:.0f} mm) -- did the camera or the arm "
                           "model shift mid-sweep?")
        if od > MAX_POSE_OFFSET_DEG:
            reasons.append(f"poses disagree: one pose alone turns the camera {od:.2f} deg "
                           f"(> {MAX_POSE_OFFSET_DEG:.1f} deg)")
    eig, cond = get("degeneracy_min_eig"), get("condition_number")
    if eig is None or cond is None:
        reasons.append("degeneracy metrics missing/non-finite (unconstrained fit)")
    else:
        if eig < MIN_DEGENERACY_EIG:
            reasons.append(f"degenerate: weakest direction carries {eig:.4f} of the information "
                           f"(< {MIN_DEGENERACY_EIG}); the visible geometry does not pin all six "
                           "DOF, whatever the residual says")
        if cond > MAX_CONDITION:
            reasons.append(f"degenerate: normal-matrix condition number {cond:.0f} > "
                           f"{MAX_CONDITION:.0f}")
    spread, rot = get("tcp_spread_m"), get("rotation_spread_deg")
    if spread is None or rot is None:
        reasons.append("pose diversity metrics missing/non-finite")
    else:
        if spread < MIN_TCP_SPREAD_M:
            reasons.append(f"pose diversity: TCP positions span {spread * 1000:.1f} mm along their "
                           f"2nd axis (< {MIN_TCP_SPREAD_M * 1000:.0f} mm): one pose or a line")
        if rot < MIN_ROTATION_SPREAD_DEG:
            reasons.append(f"pose diversity: tool rotation spread {rot:.1f} deg < "
                           f"{MIN_ROTATION_SPREAD_DEG:.0f} deg (the tool only translated)")
    return reasons


def _starts(T_init, n_starts, trans_m, rot_deg):
    """The init plus perturbations of the CAMERA about its own centre."""
    out = [np.asarray(T_init, dtype=float)]
    if n_starts <= 0:
        return out
    a = math.radians(rot_deg)
    cands = []
    for axis in range(3):
        for sgn in (1.0, -1.0):
            w = np.zeros(3)
            w[axis] = sgn * a
            D = np.eye(4)
            D[:3, :3] = so3_exp(w)
            cands.append(D)
    for axis in range(3):
        for sgn in (1.0, -1.0):
            D = np.eye(4)
            D[axis, 3] = sgn * trans_m
            cands.append(D)
    rng = np.random.default_rng(1234)
    while len(cands) < n_starts:
        D = np.eye(4)
        w = rng.normal(size=3)
        D[:3, :3] = so3_exp(w / np.linalg.norm(w) * a)
        t = rng.normal(size=3)
        D[:3, 3] = t / np.linalg.norm(t) * trans_m
        cands.append(D)
    return out + [out[0] @ D for D in cands[:n_starts]]


def solve_markerless(samples, surface, T_init, *, mode: str = EYE_TO_HAND, n_starts: int = 12,
                     start_trans_m: float = 0.08, start_rot_deg: float = 12.0,
                     coarse_poses: int = 6, refine_best: int = 3, seed: int = 0) -> MarkerlessFit:
    """Robust joint point-to-plane ICP of the arm surface against every
    sample's depth. ``T_init`` = the camera profile's current T_cam2base.

    Raises ValueError for input that cannot be fitted at all (wrong mode, no
    samples); a poor or degenerate fit is a RESULT -- check ``acceptable``.
    """
    if mode == EYE_IN_HAND:
        raise ValueError("markerless calibration is eye_to_hand only: an eye-in-hand (wrist) "
                         "camera does not see the arm it rides on; use the marker method")
    if mode not in MODES:
        raise ValueError(f"bad hand-eye mode {mode!r}")
    samples = list(samples)
    if not samples:
        raise ValueError("no depth samples to fit")
    prep = _prepare(samples, surface, seed)
    spacing = 1.0 / math.sqrt(surface.points_per_m2)
    allp = np.concatenate([p.P for p in prep])
    c = allp.mean(axis=0)
    L = float(np.sqrt(np.mean(np.sum((allp - c) ** 2, axis=1))))
    # Coarse stage: every start, coarsest level, on an evenly spread subset
    # of poses; the best few continue one level, the best of those is
    # refined on every pose. Fitness (share of visible model points the
    # depth explains) decides, never the residual of the start itself.
    sub = [prep[i] for i in np.unique(np.linspace(0, len(prep) - 1,
                                                  min(coarse_poses, len(prep))).astype(int))]
    scored = []
    for T0 in _starts(T_init, n_starts, start_trans_m, start_rot_deg):
        T1 = _icp(sub, T0, SCHEDULE[:1], spacing, c, L)
        scored.append((_fitness(sub, T1, spacing, SCHEDULE[1]), T1))
    scored.sort(key=lambda s: -s[0])
    best = None
    for _, T1 in scored[: max(1, refine_best)]:
        T2 = _icp(sub, T1, SCHEDULE[1:2], spacing, c, L)
        f = _fitness(sub, T2, spacing, SCHEDULE[2])
        if best is None or f > best[0]:
            best = (f, T2)
    T = _icp(prep, best[1], SCHEDULE[2:], spacing, c, L)
    fitness = _fitness(prep, T, spacing, _EVAL)
    poses, r, H = _evaluate(prep, T, spacing, c, L)
    used = [p for p in poses if p.n_inliers >= MIN_POSE_INLIERS]
    n_in = int(sum(p.n_inliers for p in poses))
    n_vis = int(sum(min(p.n_visible, _EVAL.cap) for p in poses))
    eig, cond = _degeneracy(H, n_in, L)
    spread, rot = _diversity(samples)
    offs_m = [p.offset_m for p in used if math.isfinite(p.offset_m)]
    offs_d = [p.offset_deg for p in used if math.isfinite(p.offset_deg)]
    metrics = {
        "n_poses": len(samples),
        "n_poses_used": len(used),
        "n_inliers": n_in,
        "n_visible": n_vis,
        "inlier_fraction": n_in / max(n_vis, 1),
        "rmse_m": float(np.sqrt(np.mean(r ** 2))) if len(r) else math.nan,
        "pose_offset_max_m": max(offs_m) if offs_m else math.nan,
        "pose_offset_max_deg": max(offs_d) if offs_d else math.nan,
        "pose_offset_rms_m": float(np.sqrt(np.mean(np.square(offs_m)))) if offs_m else math.nan,
        "degeneracy_min_eig": eig,
        "condition_number": cond,
        "weakest_sigma_m": (float(np.sqrt(np.mean(r ** 2)) / math.sqrt(eig * n_in))
                            if eig > 0 and n_in and len(r) else math.nan),
        "tcp_spread_m": spread,
        "rotation_spread_deg": rot,
        "fitness": float(fitness),
        "n_starts": len(scored),
        "cloud_radius_m": L,
    }
    return MarkerlessFit(T_cam2base=T, metrics=metrics, poses=tuple(poses))


# ── single-pose offset (the drift monitor's measurement) ─────────────────


@dataclass(frozen=True)
class OffsetCheck:
    """Where one static view says the camera is, relative to ``T``.

    ``conclusive`` False = the view cannot say (arm occluded / out of view /
    too few points); ``offset_*`` is then NaN. ``offset_m`` is the RMS
    displacement of the visible arm surface under the correction the short
    ICP finds (``surface_displacement``), ``offset_deg`` its rotation angle,
    ``camera_shift_m`` how far the camera centre moves. ``T_estimate`` is
    the corrected camera pose; only the directions this one view constrains
    are measured (the rest stay at ``T``)."""

    conclusive: bool
    reason: str
    offset_m: float = math.nan
    offset_deg: float = math.nan
    camera_shift_m: float = math.nan
    n_visible: int = 0
    n_inliers_before: int = 0
    n_inliers_after: int = 0
    fraction_before: float = 0.0
    fraction_after: float = 0.0
    fraction_front: float = 0.0
    rmse_m: float = math.nan
    T_estimate: np.ndarray | None = field(default=None, compare=False)


#: A single view constrains some directions far better than others (a
#: forearm seen side-on barely pins rotation about its own axis). The drift
#: check only measures directions carrying >= this share of the strongest.
OFFSET_REL_FLOOR = 0.05


def measure_offset(sample: DepthSample, surface, T, *, min_points: int = 300,
                   schedule=SCHEDULE[2:], rel_floor: float = OFFSET_REL_FLOOR) -> OffsetCheck:
    """Short ICP from ``T`` on ONE static view: how far the camera seems to
    have moved. Degeneracy-aware (eigen-truncated steps), so a view that
    leaves a direction free reports no offset along it rather than noise."""
    prep = _prepare([sample], surface)
    spacing = 1.0 / math.sqrt(surface.points_per_m2)
    T = np.asarray(T, dtype=float)
    pr = prep[0]
    c = pr.P.mean(axis=0)
    L = float(np.sqrt(np.mean(np.sum((pr.P - c) ** 2, axis=1))))
    before = _associate_all(prep, T, _EVAL, spacing)[0]
    n_vis = int(min(before.n_visible, _EVAL.cap))
    if n_vis < min_points:
        return OffsetCheck(False, f"only {n_vis} model points in view (< {min_points})",
                           n_visible=n_vis)
    # Occlusion signature at the current estimate: depth IN FRONT of the
    # model along the ray (a person, a held object) vs behind it (misaligned).
    front = _fraction_in_front(pr, T, spacing)
    in_before = int(np.sum(before.dist <= INLIER_DIST_M))
    T2 = _icp(prep, T, schedule, spacing, c, L, rel_floor=rel_floor)
    poses, r, H = _evaluate(prep, T2, spacing, c, L)
    after = poses[0]
    frac_b = in_before / n_vis
    frac_a = after.n_inliers / max(min(after.n_visible, _EVAL.cap), 1)
    common = dict(n_visible=n_vis, n_inliers_before=in_before, n_inliers_after=after.n_inliers,
                  fraction_before=frac_b, fraction_after=frac_a, fraction_front=front,
                  rmse_m=after.rmse_m, T_estimate=T2)
    if after.n_inliers < min_points or frac_a < MIN_INLIER_FRACTION:
        why = ("occluded" if front > 0.3 else "the depth does not explain the arm near the "
               "current extrinsic")
        return OffsetCheck(False, why, **common)
    M = T2 @ se3_inv(T)          # base-frame correction: old -> new estimate
    cloud = _associate_all(prep, T2, _EVAL, spacing)[0]
    return OffsetCheck(True, "ok", offset_m=surface_displacement(M, cloud.p),
                       offset_deg=float(np.degrees(np.linalg.norm(so3_log(M[:3, :3])))),
                       camera_shift_m=float(np.linalg.norm(T2[:3, 3] - T[:3, 3])), **common)


def _fraction_in_front(pr, T, spacing, tol=0.02):
    R, t = T[:3, :3], T[:3, 3]
    pc = (pr.P - t) @ R
    nc = pr.N @ R
    vis = _visible(pc, nc, pr.sample.K, pr.pyr.size, spacing)
    if not vis.any():
        return 0.0
    P, _, Ks = pr.pyr.level(pr.pyr.stride(_EVAL.alpha))
    p = pc[vis]
    ju = np.rint(Ks[0, 0] * p[:, 0] / p[:, 2] + Ks[0, 2]).astype(int)
    jv = np.rint(Ks[1, 1] * p[:, 1] / p[:, 2] + Ks[1, 2]).astype(int)
    ok = (ju >= 0) & (ju < P.shape[1]) & (jv >= 0) & (jv < P.shape[0])
    z = P[jv[ok], ju[ok], 2]
    good = np.isfinite(z)
    if not good.any():
        return 0.0
    return float(np.mean(z[good] < p[ok][good, 2] - tol))
