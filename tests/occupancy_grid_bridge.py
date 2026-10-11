"""A synthetic occupancy bridge for offline tests (no ZMQ, no GPU).

`GridBridge` answers the OccupancyClient protocol (`probe`, `integrate_depth`,
`integrate_masked_depth`, `clear`, `query`) with an exact distance grid over the
requested region, computed from axis-aligned obstacle boxes the test places
(`bridge.boxes`). `query` also returns the occupied voxel centres, like every
real backend. Boxes can be changed between refreshes to simulate an object that
appears in the camera view while the arm moves.

Not a test module (no `test_` prefix); tests import it from the tests dir.
"""

from __future__ import annotations

import threading

import numpy as np

from cascade.perception.occupancy import OccupancyError, OccupancyMap
from cascade.types import Frame

#: a table plane as the camera maps it: one voxel layer at z = 0
TABLE = ((-1.0, -1.0, -0.01), (1.0, 1.0, 0.0))


def box_distance(points: np.ndarray, lo, hi) -> np.ndarray:
    """Exact Euclidean distance from each point to an axis-aligned box (0 inside)."""
    p = np.asarray(points, dtype=float)
    d = np.maximum(np.maximum(np.asarray(lo, float) - p, 0.0), p - np.asarray(hi, float))
    return np.linalg.norm(d, axis=-1)


def centred_box(center, size):
    c, h = np.asarray(center, float), np.asarray(size, float) / 2.0
    return (tuple(c - h), tuple(c + h))


class GridBridge:
    """Fake bridge: distance grid from `boxes`, optional `unknown` boxes."""

    def __init__(self, boxes=(), *, voxel=0.02, backend="nvblox", masked=True):
        self.boxes = list(boxes)
        self.unknown = []          # boxes whose voxels read as unobserved (inf)
        self.voxel = float(voxel)
        self.backend = backend
        self.masked = masked
        self.actions = []
        self.fail = None           # an exception message: every query raises it
        self.gate = threading.Event()
        self.gate.set()            # clear() it to block `query` until set again
        self.in_query = threading.Event()
        self.queries = 0
        self.last_depth = None
        self.last_active = None

    def probe(self, timeout_ms=300):
        return self.request({"action": "probe"})

    def request(self, payload, timeout_ms=None):
        action = payload["action"]
        self.actions.append(action)
        if action == "probe":
            return {"ok": True, "backend": self.backend, "device": "cuda:0", "voxel": self.voxel,
                    "esdf": True, "masked_depth": self.masked}
        if action in ("integrate_depth", "integrate_masked_depth"):
            self.last_depth = np.asarray(payload["depth"])
            self.last_active = payload.get("active_mask")
            return {}
        if action == "clear":
            return {}
        if action == "query":
            self.in_query.set()
            assert self.gate.wait(10), "the test never released the bridge"
            if self.fail:
                raise OccupancyError(self.fail)
            self.queries += 1
            return self._grid(np.asarray(payload["region_min"], float),
                              np.asarray(payload["region_max"], float))
        raise OccupancyError(f"unknown action {action!r}")

    def _grid(self, lo, hi):
        v = self.voxel
        shape = np.maximum(np.round((hi - lo) / v).astype(int), 1)
        axes = [lo[i] + v * (np.arange(shape[i]) + 0.5) for i in range(3)]
        centres = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        grid = np.full(tuple(shape), np.inf, dtype=np.float32)
        for b_lo, b_hi in self.boxes:
            grid = np.minimum(grid, box_distance(centres, b_lo, b_hi).astype(np.float32))
        occupied = centres[grid <= v / 2.0 + 1e-9]
        for u_lo, u_hi in self.unknown:
            grid[box_distance(centres, u_lo, u_hi) <= 1e-9] = np.inf
        return {"grid": grid, "origin": np.asarray([a[0] for a in axes], np.float32),
                "voxel": v, "points": occupied.astype(np.float32)}


class DepthBridge(GridBridge):
    """Fake bridge that maps what the depth frames show: every valid (and,
    on the native path, active) pixel is unprojected and its voxel marked
    occupied; `query` returns the distance from each grid centre to the
    nearest occupied voxel centre. Memoryless per frame (one fixed camera),
    so a test sees exactly what the last masked frame put in the map.
    `boxes` still add synthetic obstacles on top."""

    def __init__(self, boxes=(), *, voxel=0.02, **kw):
        super().__init__(boxes, voxel=voxel, **kw)
        self.cloud = np.zeros((0, 3))

    def request(self, payload, timeout_ms=None):
        if payload["action"] in ("integrate_depth", "integrate_masked_depth"):
            depth = np.asarray(payload["depth"], float)
            K, T = np.asarray(payload["K"], float), np.asarray(payload["T_base_cam"], float)
            ok = np.isfinite(depth) & (depth > 0)
            if payload.get("active_mask") is not None:
                ok &= np.asarray(payload["active_mask"]).astype(bool)
            v, u = np.nonzero(ok)
            z = depth[v, u]
            cam = np.c_[(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z]
            pts = cam @ T[:3, :3].T + T[:3, 3]
            keys = np.unique(np.floor(pts / self.voxel).astype(np.int64), axis=0)
            self.cloud = (keys + 0.5) * self.voxel
        return super().request(payload, timeout_ms)

    def _grid(self, lo, hi):
        from scipy.spatial import cKDTree

        out = super()._grid(lo, hi)
        v = self.voxel
        inside = np.all((self.cloud >= lo - v) & (self.cloud <= hi + v), axis=1)
        cloud = self.cloud[inside]
        if len(cloud):
            g = out["grid"]
            axes = [out["origin"][i] + v * np.arange(g.shape[i]) for i in range(3)]
            centres = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
            d, _ = cKDTree(cloud).query(centres)
            out["grid"] = np.minimum(g, d.reshape(g.shape).astype(np.float32))
            out["points"] = np.concatenate([out["points"], cloud.astype(np.float32)])
        return out


def render_depth(K, T_base_cam, shape, surfaces=(), table_z=0.0):
    """Depth image of a table plane at `table_z` plus surface points (a
    z-buffer of point splats), as a camera at T_base_cam (camera -> base)
    sees them."""
    h, w = shape
    R, t = np.asarray(T_base_cam, float)[:3, :3], np.asarray(T_base_cam, float)[:3, 3]
    v, u = np.mgrid[0:h, 0:w]
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
    rays_b = rays @ R.T
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (table_z - t[2]) / rays_b[..., 2]              # depth along the optical axis
    depth = np.where(np.isfinite(s) & (s > 0), s, np.inf)
    for pts in surfaces:
        pc = (np.asarray(pts, float).reshape(-1, 3) - t) @ R
        pc = pc[pc[:, 2] > 0.05]
        uu = np.round(K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]).astype(int)
        vv = np.round(K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]).astype(int)
        ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
        np.minimum.at(depth, (vv[ok], uu[ok]), pc[ok, 2])
    return np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)


def box_surface(lo, hi, spacing=0.004):
    """Surface points of an axis-aligned box, at most `spacing` apart."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    axes = [np.linspace(lo[i], hi[i], max(2, int(np.ceil((hi[i] - lo[i]) / spacing)) + 1)) for i in range(3)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    on_face = np.any(np.isclose(g, lo) | np.isclose(g, hi), axis=1)
    return g[on_face]


def triangle_surface(tris, spacing=0.004):
    """Points on every triangle (n x 3 x 3), at most about `spacing` apart."""
    tris = np.asarray(tris, float)
    edge = np.max(np.linalg.norm(tris[:, [1, 2, 0]] - tris, axis=-1), axis=1)
    out = []
    for n in np.unique(np.clip(np.ceil(edge / spacing).astype(int), 1, 40)):
        sel = tris[np.clip(np.ceil(edge / spacing).astype(int), 1, 40) == n]
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = i + j <= n
        a, b = i[keep] / n, j[keep] / n
        bary = np.stack([1 - a - b, a, b], -1)                    # m x 3
        out.append(np.einsum("mk,tkd->tmd", bary, sel).reshape(-1, 3))
    return np.concatenate(out) if out else np.zeros((0, 3))


def depth_frame(t=None, shape=(12, 16)):
    """A small depth frame (2 m everywhere); `t` defaults to now (fresh)."""
    kw = {} if t is None else {"t": t}
    return Frame(rgb=np.zeros((*shape, 3), np.uint8), depth_m=np.full(shape, 2.0, np.float32),
                 K=np.array([[20.0, 0, shape[1] / 2], [0, 20.0, shape[0] / 2], [0, 0, 1]]), **kw)


REGION_MIN = np.array([-0.30, -0.45, -0.02])
REGION_MAX = np.array([0.66, 0.45, 0.60])


def make_map(bridge, *, required=True, **kw) -> OccupancyMap:
    occ = OccupancyMap(client=bridge, region_min=REGION_MIN, region_max=REGION_MAX,
                       required=required, **kw)
    occ.probe()
    return occ
