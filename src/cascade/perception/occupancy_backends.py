"""Occupancy / distance-field backends for the cascade occupancy bridge.

Three backends behind ONE interface, chosen by `make_backend(name)`:

  nvblox  P0 on NVIDIA hardware. Real nvblox (nvblox_torch >= 0.0.10):
          hashed TSDF integrated by ray casting from depth images, ESDF on
          the GPU, free-space carving. CUDA only (compute capability >= 7.5,
          linux x86_64 wheel from the GitHub release; Jetson builds from
          source). Verified against the nvblox_torch public API at
          nvidia-isaac/nvblox `public` (Mapper.add_depth_frame /
          update_esdf / query_layer(QueryType.ESDF)); UNVERIFIED at runtime
          on this dev machine (no NVIDIA GPU here) -- the class is exercised
          only where CUDA exists.

  warp    Hardware-agnostic default. ~200 lines of NVIDIA Warp kernels owned
          by this repo: a DENSE projective TSDF over the workspace AABB
          (KinectFusion-style: every voxel projects into the depth image, so
          voxels in front of the surface are carved FREE) and an exact
          Euclidean distance transform (Felzenszwalb-Huttenlocher separable
          parabola lower envelope, run per axis) that turns the occupied set
          into a distance grid. `warp-lang` ships PyPI wheels for macOS
          arm64 (CPU), linux x86_64 and aarch64 (CPU + CUDA): the SAME
          kernels run on this laptop's CPU and on a Jetson's GPU.

  voxel   Zero-dependency numpy fallback. Occupancy-only voxel hash: no
          carving, no distance field (the query answers with the grid's
          nearest-occupied-voxel distance computed brute force). Kept so the
          protocol stays runnable with only pyzmq + msgpack-numpy.

Common interface (all coordinates in the robot BASE frame, metres):

    integrate_points(points (N,3))                      legacy: unprojected cloud
    integrate_depth(depth (H,W) m, K (3,3), T_base_cam (4,4))
    query(region_min (3,), region_max (3,)) -> dict with
        "grid": float32 (nx,ny,nz) distance to nearest occupied voxel [m]
                (0 inside occupied), "origin": (3,) centre of voxel [0,0,0],
        "voxel": float voxel size, "points": occupied voxel centres (M,3)
                inside the region (legacy consumers)
    describe() -> str   backend name + device for the probe reply

The distance GRID is what makes the client backend-independent: a query
point's clearance is a trilinear read, identical whether nvblox, warp or the
fallback produced the grid. Carving means a removed object disappears from
the grid after a few frames -- the property the old accumulator lacked.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

# ─────────────────────────────────────────────────────────────────────────
# Grid geometry shared by the dense backends
# ─────────────────────────────────────────────────────────────────────────


class GridSpec:
    """Dense grid over an AABB. Voxel (i,j,k) centre = origin + (i,j,k)*voxel."""

    def __init__(self, region_min, region_max, voxel: float):
        self.voxel = float(voxel)
        lo = np.asarray(region_min, dtype=np.float64)
        hi = np.asarray(region_max, dtype=np.float64)
        self.shape = tuple(int(max(2, np.ceil((hi[a] - lo[a]) / self.voxel) + 1)) for a in range(3))
        self.origin = lo.astype(np.float32)

    def centres(self) -> np.ndarray:
        ii, jj, kk = np.meshgrid(*[np.arange(n) for n in self.shape], indexing="ij")
        return (np.stack([ii, jj, kk], -1) * self.voxel + self.origin).astype(np.float32)


def edt_3d(occupied: np.ndarray, voxel: float) -> np.ndarray:
    """Exact Euclidean distance transform of a boolean grid (metres), numpy.
    Used by the voxel fallback and as the reference for the Warp kernels."""
    from scipy import ndimage

    if not occupied.any():
        return np.full(occupied.shape, np.inf, dtype=np.float32)
    return (ndimage.distance_transform_edt(~occupied) * voxel).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────
# voxel: numpy fallback (occupancy hash, no carving)
# ─────────────────────────────────────────────────────────────────────────


class VoxelBackend:
    name = "voxel"

    def __init__(self, voxel: float = 0.02, **_):
        self.voxel = float(voxel)
        self._voxels: set[tuple[int, int, int]] = set()

    def describe(self) -> str:
        return "voxel (numpy occupancy hash; no carving, no ESDF)"

    def integrate_points(self, points: np.ndarray) -> None:
        if points.shape[0] == 0:
            return
        idx = np.round(points / self.voxel).astype(np.int64)
        self._voxels.update(map(tuple, idx))

    def integrate_depth(self, depth: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray) -> None:
        self.integrate_points(unproject(depth, K, T_base_cam, stride=4))

    def query(self, region_min, region_max) -> dict[str, Any]:
        spec = GridSpec(region_min, region_max, self.voxel)
        occ = np.zeros(spec.shape, dtype=bool)
        pts = np.empty((0, 3), dtype=np.float32)
        if self._voxels:
            idx = np.asarray(list(self._voxels), dtype=np.float64)
            centres = idx * self.voxel
            keep = np.all((centres >= spec.origin - self.voxel / 2) & (centres <= np.asarray(region_max) + self.voxel / 2), axis=1)
            pts = centres[keep].astype(np.float32)
            g = np.round((pts - spec.origin) / self.voxel).astype(np.int64)
            g = g[np.all((g >= 0) & (g < np.asarray(spec.shape)), axis=1)]
            occ[g[:, 0], g[:, 1], g[:, 2]] = True
        return {"grid": edt_3d(occ, self.voxel), "origin": spec.origin, "voxel": self.voxel, "points": pts}


def unproject(depth: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray, stride: int = 1) -> np.ndarray:
    d = depth[::stride, ::stride]
    ys, xs = np.nonzero(d > 0)
    if ys.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    zs = d[ys, xs]
    xs = xs * stride
    ys = ys * stride
    pts = np.stack([(xs - K[0, 2]) / K[0, 0] * zs, (ys - K[1, 2]) / K[1, 1] * zs, zs], -1)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], 1)
    return (pts_h @ np.asarray(T_base_cam).T)[:, :3].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────
# warp: dense projective TSDF + exact EDT, CPU or CUDA from one source
# ─────────────────────────────────────────────────────────────────────────


class WarpTsdfBackend:
    """Dense TSDF over a fixed workspace AABB with ray-carving, and an exact
    Euclidean distance grid of the occupied set, in Warp kernels.

    Integration (per depth frame): each voxel centre is transformed into the
    camera, projected with K, and compared with the depth at that pixel.
    sdf = depth(pixel) - z_voxel, truncated to ±trunc. Voxels in FRONT of the
    observed surface (sdf > 0, up to trunc... and, with `carve_far`, any
    voxel between camera and surface) move toward FREE; voxels around the
    surface get the truncated signed distance; voxels behind stay unknown.
    Running weighted average with a cap so a removed object decays in
    ~`max_weight` frames instead of persisting forever.

    Distance field: occupied = observed & tsdf <= 0 (surface and just
    behind). Exact EDT via the separable squared-distance lower-envelope
    algorithm, one Warp kernel per axis pass, so the cost is O(N) per axis
    and identical on CPU and CUDA.
    """

    name = "warp"

    def __init__(self, voxel: float = 0.01, region_min=(-0.5, -0.5, -0.1),
                 region_max=(0.8, 0.5, 0.6), trunc_m: float | None = None,
                 max_weight: float = 8.0, device: str = "auto", **_):
        import warp as wp

        self._wp = wp
        wp.config.quiet = True
        wp.init()
        if device == "auto":
            device = "cuda:0" if wp.is_cuda_available() else "cpu"
        self.device = device
        self.voxel = float(voxel)
        self.spec = GridSpec(region_min, region_max, self.voxel)
        self.trunc = float(trunc_m if trunc_m is not None else 4.0 * self.voxel)
        self.max_weight = float(max_weight)
        n = self.spec.shape
        self._tsdf = wp.zeros(n, dtype=wp.float32, device=device)
        self._weight = wp.zeros(n, dtype=wp.float32, device=device)
        self._dist2 = wp.zeros(n, dtype=wp.float32, device=device)
        self._tmp = wp.zeros(n, dtype=wp.float32, device=device)
        self._origin = wp.vec3(*[float(v) for v in self.spec.origin])
        self._frames = 0
        self.last_integrate_ms = 0.0
        self.last_query_ms = 0.0

    def describe(self) -> str:
        return f"warp TSDF+EDT on {self.device} ({'x'.join(map(str, self.spec.shape))} @ {self.voxel*100:.1f} cm)"

    # -- integration ---------------------------------------------------------

    def integrate_points(self, points: np.ndarray) -> None:
        """Legacy path (no camera model = no carving): stamp points as
        occupied with a strong weight."""
        if points.shape[0] == 0:
            return
        g = np.round((points - self.spec.origin) / self.voxel).astype(np.int64)
        ok = np.all((g >= 0) & (g < np.asarray(self.spec.shape)), axis=1)
        g = g[ok]
        if g.shape[0] == 0:
            return
        tsdf = self._tsdf.numpy()
        w = self._weight.numpy()
        tsdf[g[:, 0], g[:, 1], g[:, 2]] = -0.5 * self.trunc
        w[g[:, 0], g[:, 1], g[:, 2]] = self.max_weight
        self._tsdf = self._wp.array(tsdf, dtype=self._wp.float32, device=self.device)
        self._weight = self._wp.array(w, dtype=self._wp.float32, device=self.device)

    def integrate_depth(self, depth: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray) -> None:
        wp = self._wp
        t0 = time.perf_counter()
        depth = np.ascontiguousarray(depth, dtype=np.float32)
        T_cam_base = np.linalg.inv(np.asarray(T_base_cam, dtype=np.float64))
        d_wp = wp.array(depth, dtype=wp.float32, device=self.device)
        R = wp.mat33(*[float(v) for v in T_cam_base[:3, :3].reshape(-1)])
        t = wp.vec3(*[float(v) for v in T_cam_base[:3, 3]])
        KN = _kernels()
        wp.launch(KN.k_integrate, dim=self.spec.shape,
                  inputs=[d_wp, int(depth.shape[1]), int(depth.shape[0]),
                          float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]),
                          R, t, self._origin, self.voxel, self.trunc, self.max_weight,
                          self._tsdf, self._weight],
                  device=self.device)
        wp.synchronize_device(self.device)
        self._frames += 1
        self.last_integrate_ms = (time.perf_counter() - t0) * 1e3

    # -- query ---------------------------------------------------------------

    def query(self, region_min, region_max) -> dict[str, Any]:
        wp = self._wp
        t0 = time.perf_counter()
        nx, ny, nz = self.spec.shape
        big = float(1e12)
        KN = _kernels()
        wp.launch(KN.k_seed, dim=self.spec.shape,
                  inputs=[self._tsdf, self._weight, big, self._dist2], device=self.device)
        # three separable passes: x (over yz), y (over xz), z (over xy)
        wp.launch(KN.k_edt_x, dim=(ny, nz), inputs=[self._dist2, self._tmp, nx, big], device=self.device)
        wp.launch(KN.k_edt_y, dim=(nx, nz), inputs=[self._tmp, self._dist2, ny, big], device=self.device)
        wp.launch(KN.k_edt_z, dim=(nx, ny), inputs=[self._dist2, self._tmp, nz, big], device=self.device)
        wp.synchronize_device(self.device)
        d2 = self._tmp.numpy()
        grid = np.sqrt(np.minimum(d2, big)).astype(np.float32) * self.voxel
        grid[d2 >= big] = np.inf
        # crop to the requested region
        lo = np.maximum(np.round((np.asarray(region_min) - self.spec.origin) / self.voxel).astype(int), 0)
        hi = np.minimum(np.round((np.asarray(region_max) - self.spec.origin) / self.voxel).astype(int) + 1, np.asarray(self.spec.shape))
        sub = grid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        origin = (self.spec.origin + lo * self.voxel).astype(np.float32)
        occ_idx = np.argwhere(sub <= 0.0)
        pts = (occ_idx * self.voxel + origin).astype(np.float32)
        self.last_query_ms = (time.perf_counter() - t0) * 1e3
        return {"grid": np.ascontiguousarray(sub), "origin": origin, "voxel": self.voxel, "points": pts}

    def occupied_mask(self) -> np.ndarray:
        t = self._tsdf.numpy(); w = self._weight.numpy()
        return (w > 0) & (t <= 0.0)


def _kernels():
    """Lazy import: the kernel module needs warp-lang at import time."""
    from . import _warp_tsdf_kernels as K

    return K


# ─────────────────────────────────────────────────────────────────────────
# nvblox: the real thing, CUDA only
# ─────────────────────────────────────────────────────────────────────────


class NvbloxBackend:
    """nvblox_torch Mapper: TSDF by ray casting + on-GPU ESDF.

    API pinned to nvblox_torch 0.0.10 (nvidia-isaac/nvblox `public`):
        Mapper(voxel_sizes_m=float, mapper_parameters=MapperParams())
        Sensor.from_camera(fu, fv, cu, cv, width, height)
        mapper.add_depth_frame(depth_HxW_float32_cuda, t_w_c_4x4_CPU, sensor)
        mapper.update_esdf()
        mapper.query_layer(QueryType.ESDF, points_Nx3_cuda) -> (N,1) distance;
            unknown == constants.esdf_unknown_distance()
    UNVERIFIED on this dev machine (no CUDA). Any API drift raises at
    construction/first call and the bridge reports it -- never a silent
    fallback to another backend, because the client displays the backend
    name as evidence of what checked the motion.
    """

    name = "nvblox"

    def __init__(self, voxel: float = 0.01, region_min=(-0.5, -0.5, -0.1),
                 region_max=(0.8, 0.5, 0.6), max_integration_m: float = 2.0, **_):
        import torch  # noqa: F401
        from nvblox_torch.mapper import Mapper, QueryType  # type: ignore
        from nvblox_torch.mapper_params import MapperParams, ProjectiveIntegratorParams  # type: ignore
        from nvblox_torch.constants import constants  # type: ignore

        import torch as _torch

        if not _torch.cuda.is_available():
            raise RuntimeError("nvblox backend needs a CUDA device (nvblox_torch is GPU-only)")
        self._torch = _torch
        self._QueryType = QueryType
        self._unknown = float(constants.esdf_unknown_distance())
        pip = ProjectiveIntegratorParams()
        pip.projective_integrator_max_integration_distance_m = float(max_integration_m)
        params = MapperParams()
        params.set_projective_integrator_params(pip)
        self._Sensor = None
        try:
            from nvblox_torch.sensor import Sensor  # type: ignore

            self._Sensor = Sensor
        except ImportError:
            pass
        self.mapper = Mapper(voxel_sizes_m=float(voxel), mapper_parameters=params)
        self.voxel = float(voxel)
        self.spec = GridSpec(region_min, region_max, self.voxel)
        self.last_integrate_ms = 0.0
        self.last_query_ms = 0.0

    def describe(self) -> str:
        return f"nvblox (nvblox_torch TSDF+ESDF on {self._torch.cuda.get_device_name(0)}, {self.voxel*100:.1f} cm)"

    def integrate_points(self, points: np.ndarray) -> None:
        raise RuntimeError("nvblox integrates DEPTH FRAMES (ray casting); use integrate_depth")

    def integrate_depth(self, depth: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray) -> None:
        torch = self._torch
        t0 = time.perf_counter()
        h, w = depth.shape
        sensor = self._Sensor.from_camera(float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), int(w), int(h))
        d = torch.as_tensor(np.ascontiguousarray(depth, dtype=np.float32)).cuda()
        pose = torch.as_tensor(np.asarray(T_base_cam, dtype=np.float32))  # CPU, sensor->world
        self.mapper.add_depth_frame(d, pose, sensor)
        self.mapper.update_esdf()
        torch.cuda.synchronize()
        self.last_integrate_ms = (time.perf_counter() - t0) * 1e3

    def query(self, region_min, region_max) -> dict[str, Any]:
        torch = self._torch
        t0 = time.perf_counter()
        spec = GridSpec(region_min, region_max, self.voxel)
        centres = torch.as_tensor(spec.centres().reshape(-1, 3)).cuda()
        sdf = self.mapper.query_layer(self._QueryType.ESDF, centres).reshape(-1).cpu().numpy()
        grid = sdf.reshape(spec.shape).astype(np.float32)
        unknown = grid == self._unknown
        grid = np.abs(grid)          # clearance is unsigned; inside -> 0
        grid[sdf <= 0.0] = 0.0
        grid[unknown] = np.inf        # unobserved reads as "no obstacle known" (kFree policy)
        occ = np.argwhere(grid <= 0.0)
        pts = (occ * self.voxel + spec.origin).astype(np.float32)
        torch.cuda.synchronize()
        self.last_query_ms = (time.perf_counter() - t0) * 1e3
        return {"grid": grid, "origin": spec.origin, "voxel": self.voxel, "points": pts}


# ─────────────────────────────────────────────────────────────────────────
# factory
# ─────────────────────────────────────────────────────────────────────────

BACKENDS = ("nvblox", "warp", "voxel")


def make_backend(name: str, **kw):
    """`auto` = nvblox if importable AND CUDA is present, else warp if
    warp-lang imports, else voxel. Explicit names never fall through: asking
    for nvblox on a machine without it is an error, not a quiet downgrade."""
    name = (name or "auto").lower()
    if name == "auto":
        try:
            return NvbloxBackend(**kw)
        except Exception:  # noqa: BLE001 -- no nvblox/CUDA: fall through
            pass
        try:
            return WarpTsdfBackend(**kw)
        except Exception:  # noqa: BLE001
            pass
        return VoxelBackend(**kw)
    if name == "nvblox":
        return NvbloxBackend(**kw)
    if name == "warp":
        return WarpTsdfBackend(**kw)
    if name == "voxel":
        return VoxelBackend(**kw)
    raise ValueError(f"unknown occupancy backend {name!r} (nvblox|warp|voxel|auto)")
