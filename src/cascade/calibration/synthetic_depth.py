"""Synthetic depth camera for markerless calibration: the arm, rendered.

Used by ``--dry-run --method markerless`` and the markerless / drift-monitor
tests. Like the marker camera in ``synthetic.py`` it RENDERS from a known
ground truth -- the arm's own meshes at FK of the arm the session drives,
seen from a known ``T_cam2base`` -- so the solver, the session and the
record are exercised on pixels, not on synthesised correspondences.

What it models, because each one is a way a real D455F breaks a naive fit:

* the table plane at ``table_z`` behind the arm (background that can capture
  the model if association is careless);
* the finger slides at an unmeasured opening (``finger_opening_m``) --
  geometry that is in the image but cannot be in the model;
* range-proportional white noise (``noise_frac`` x z, ~0.75 % by default)
  plus a low-frequency correlated component (``correlated_frac`` x z) --
  stereo depth errors are spatially correlated and do NOT average out;
* random dropout and dropout at depth discontinuities (flying pixels are
  what a real stream drops), the min/max range, and 1 mm quantisation.

The arm is a dense point splat with a z-buffer (back faces culled, one
hole-filling pass): at 640x360 the dense skin (~0.9 mm spacing) covers
the visible pixels. Noise-free renders are cached per (q, T), so a session's
repeated grabs at one pose only pay for fresh noise.
"""

from __future__ import annotations

import time

import numpy as np

from .frames import se3_inv
from .robot_surface import RobotSurface

#: Render density: ~0.9 mm between skin samples. At 640x360 (fx = 320) a
#: pixel is ~1.6 mm at 0.5 m; 4e5/m^2 left 10 % of the arm's pixels as
#: holes (the table showed through), this leaves ~1.6 %, mostly real gaps.
RENDER_POINTS_PER_M2 = 1.2e6


def default_K(image_size) -> np.ndarray:
    """~90 deg HFOV pinhole, like the D455 depth stream (fx ~ 0.5 w)."""
    w, h = int(image_size[0]), int(image_size[1])
    return np.array([[0.5 * w, 0.0, (w - 1) / 2.0], [0.0, 0.5 * w, (h - 1) / 2.0],
                     [0.0, 0.0, 1.0]])


def splat_depth(points_cam, K, image_size, *, normals_cam=None, fill_holes=True) -> np.ndarray:
    """(H, W) float32 z-buffer of camera-frame points; 0 = empty."""
    w, h = int(image_size[0]), int(image_size[1])
    p = np.asarray(points_cam, dtype=float)
    keep = p[:, 2] > 1e-3
    if normals_cam is not None:
        keep &= np.einsum("ij,ij->i", np.asarray(normals_cam), p) < 0.0   # back faces
    p = p[keep]
    u = np.round(K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]).astype(np.int64)
    v = np.round(K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]).astype(np.int64)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    zbuf = np.full(h * w, np.inf)
    np.minimum.at(zbuf, v[ok] * w + u[ok], p[ok, 2])
    zbuf = zbuf.reshape(h, w)
    if fill_holes:
        zbuf = _fill_holes(zbuf)
    out = np.where(np.isfinite(zbuf), zbuf, 0.0)
    return out.astype(np.float32)


def _fill_holes(z: np.ndarray) -> np.ndarray:
    """Fill an empty pixel from its nearest 8-neighbours when >= 5 of them
    are filled (an interior sampling gap, not a silhouette)."""
    filled = np.isfinite(z)
    pad = np.pad(np.where(filled, z, np.inf), 1, constant_values=np.inf)
    h, w = z.shape
    nbrs = [pad[1 + dy: 1 + dy + h, 1 + dx: 1 + dx + w]
            for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
    stack = np.stack(nbrs)
    count = np.isfinite(stack).sum(axis=0)
    hole = ~filled & (count >= 5)
    out = z.copy()
    out[hole] = np.median(np.where(np.isfinite(stack), stack, np.nan)[:, hole], axis=0) \
        if hole.any() else out[hole]
    return out


def table_depth(T_cam2base, K, image_size, table_z: float) -> np.ndarray:
    """Depth of the plane z = table_z (base frame) along every pixel ray."""
    w, h = int(image_size[0]), int(image_size[1])
    us, vs = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    rays = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us)], -1)
    R, t = T_cam2base[:3, :3], T_cam2base[:3, 3]
    dz = rays @ R[2]                     # base-frame z component of each ray
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (float(table_z) - t[2]) / dz
    return np.where(np.isfinite(s) & (s > 0), s, 0.0).astype(np.float32)


class SyntheticDepthCamera:
    """An eye-to-hand RGB-D camera that sees the arm (and a table).

    ``q_fn()`` returns the CURRENT measured joints of the arm the session
    drives. ``T_cam2base`` is the ground truth and may be reassigned to
    simulate a bumped camera. Duck-typed like ``SyntheticMarkerCamera``
    (open / close / get_frame / K / serial / dist_coeffs).
    """

    has_depth = True

    def __init__(self, kin, model_path, T_cam2base, q_fn, *, K=None,
                 image_size=(640, 360), table_z: float | None = 0.0,
                 noise_frac: float = 0.0075, correlated_frac: float = 0.002,
                 dropout: float = 0.02, edge_dropout: float = 0.5,
                 edge_jump_m: float = 0.02, finger_opening_m: float = 0.02,
                 min_depth_m: float = 0.3, max_depth_m: float = 4.0, seed: int = 0,
                 serial: str = "SYNTHETIC", clock=time.monotonic,
                 dense_per_m2: float = RENDER_POINTS_PER_M2):
        self.kin = kin
        self.model_path = model_path
        self.T_cam2base = np.asarray(T_cam2base, dtype=float)
        self._q = q_fn
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.K = default_K(self.image_size) if K is None else np.asarray(K, dtype=float)
        self.table_z = table_z
        self.noise_frac = float(noise_frac)
        self.correlated_frac = float(correlated_frac)
        self.dropout = float(dropout)
        self.edge_dropout = float(edge_dropout)
        self.edge_jump_m = float(edge_jump_m)
        self.finger_opening_m = float(finger_opening_m)
        self.min_depth_m, self.max_depth_m = float(min_depth_m), float(max_depth_m)
        self.serial = serial
        self.dist_coeffs = np.zeros(5)
        self._rng = np.random.default_rng(seed)
        self._clock = clock
        # Every mesh, fingers included: the image holds geometry the model
        # deliberately does not (see RobotSurface: unmeasured joints).
        self._surface = RobotSurface.from_kinematics(kin, model_path, links="all",
                                                     points_per_m2=dense_per_m2,
                                                     dense_per_m2=dense_per_m2)
        self._cache_key = None
        self._cache = None
        self.grabs = 0
        self.opened = False

    # ── camera surface ───────────────────────────────────────────────────

    def open(self):
        self.opened = True

    def close(self):
        self.opened = False

    def render_clean(self, q) -> np.ndarray:
        """Noise-free depth of the arm (+ table) at joints ``q``."""
        q = np.asarray(q, dtype=float)
        key = (np.round(q, 7).tobytes(), self.T_cam2base.tobytes())
        if key == self._cache_key:
            return self._cache
        cloud = self._surface.points_at(q, passive=[self.finger_opening_m] * 2)
        Tinv = se3_inv(self.T_cam2base)
        pc = cloud.points @ Tinv[:3, :3].T + Tinv[:3, 3]
        nc = cloud.normals @ Tinv[:3, :3].T
        depth = splat_depth(pc, self.K, self.image_size, normals_cam=nc)
        if self.table_z is not None:
            table = table_depth(self.T_cam2base, self.K, self.image_size, self.table_z)
            arm = depth > 0
            depth = np.where(arm & ((table <= 0) | (depth < table)), depth, table)
        self._cache_key, self._cache = key, depth.astype(np.float32)
        return self._cache

    def _noisy(self, clean: np.ndarray) -> np.ndarray:
        rng = self._rng
        h, w = clean.shape
        valid = clean > 0
        z = clean.astype(np.float64)
        noisy = z + rng.normal(0.0, 1.0, z.shape) * self.noise_frac * z
        if self.correlated_frac > 0:
            import cv2

            coarse = rng.normal(0.0, 1.0, (max(2, h // 40), max(2, w // 40)))
            field = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
            noisy += field * self.correlated_frac * z
        drop = rng.random(z.shape) < self.dropout
        if self.edge_dropout > 0:
            jump = np.zeros_like(valid)
            dz_x = np.abs(np.diff(z, axis=1)) > self.edge_jump_m
            dz_y = np.abs(np.diff(z, axis=0)) > self.edge_jump_m
            jump[:, 1:] |= dz_x
            jump[:, :-1] |= dz_x
            jump[1:, :] |= dz_y
            jump[:-1, :] |= dz_y
            drop |= jump & (rng.random(z.shape) < self.edge_dropout)
        noisy = np.round(noisy * 1000.0) / 1000.0          # z16, 1 mm units
        ok = valid & ~drop & (noisy >= self.min_depth_m) & (noisy <= self.max_depth_m)
        return np.where(ok, noisy, 0.0).astype(np.float32)

    def get_frame(self):
        from ..types import Frame

        q = np.asarray(self._q(), dtype=float)
        depth = self._noisy(self.render_clean(q))
        self.grabs += 1
        shade = np.where(depth > 0, np.clip(255.0 * (1.5 - depth) / 1.5, 0, 255), 0)
        rgb = np.repeat(shade.astype(np.uint8)[:, :, None], 3, axis=2)
        return Frame(rgb=rgb, depth_m=depth, K=self.K.copy(), t=self._clock(),
                     depth_source="sensor")
