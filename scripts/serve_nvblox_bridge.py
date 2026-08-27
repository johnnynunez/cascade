"""Reference ZMQ/msgpack bridge for wrc_demo's occupancy client
(src/wrc_demo/perception/occupancy.py).

Two interchangeable backends behind the SAME wire protocol
(integrate/query, see occupancy.py's docstring) and the SAME `integrate()`/
`query()` interface below, selected with `--backend`:

  "open3d" (default, auto-detected): voxel-downsampled point accumulation
    on an Open3D tensor Device chosen at startup -- CUDA:0 if this process
    can see an NVIDIA GPU, CPU:0 otherwise. Same code path either way; only
    the device string changes, so a booth laptop and the DGX Spark rig run
    the identical backend, one just runs it on the GPU. This is still NOT
    real nvblox (no TSDF/ESDF, no free-space carving -- see below), but it
    is a real, device-portable, GPU-accelerated-when-available occupancy
    accumulator, not a placeholder.

  "numpy": plain voxel-index hash, zero extra dependencies beyond
    pyzmq/msgpack-numpy. Always available; used automatically if Open3D
    isn't installed, or forced with --backend numpy.

Neither backend does what real nvblox does: TSDF/ESDF reconstruction with
ray-cast free-space carving (knowing a voxel is EMPTY, not just that no
point has landed in it yet). Both only answer "has any depth sample ever
landed near this voxel" -- occupancy-only, which is what
SafetyHarness._occupancy_violation's clearance check actually consumes, but
it means a voxel a since-removed object used to occupy stays "occupied"
until upstream integration windows it out. There is no NVIDIA-shipped
ZMQ/msgpack front end for nvblox itself -- swapping in the real thing means
replacing whichever backend class's `integrate`/`query` with calls into
nvblox_torch (or a ROS2 node talking to an Isaac ROS nvblox pipeline via
this process acting as a ROS<->ZMQ shim); the wire protocol and everything
upstream of it (occupancy.py, SafetyHarness) does not need to change.

    ./scripts/serve_nvblox_bridge.py [--port 5557] [--voxel-size 0.02] [--backend auto|open3d|numpy]
"""

from __future__ import annotations

import argparse
import sys

import msgpack
import msgpack_numpy
import numpy as np
import zmq

msgpack_numpy.patch()


class _VoxelGrid:
    """Occupancy-only voxel hash, pure numpy: which voxels have EVER seen a
    depth point. No GPU, no extra dependency -- the universal fallback."""

    def __init__(self, voxel_size_m: float = 0.02):
        self.voxel_size = float(voxel_size_m)
        self._voxels: set[tuple[int, int, int]] = set()

    def integrate(self, points: np.ndarray) -> None:
        if points.shape[0] == 0:
            return
        idx = np.round(points / self.voxel_size).astype(np.int64)
        self._voxels.update(map(tuple, idx))

    def query(self, region_min: np.ndarray, region_max: np.ndarray) -> np.ndarray:
        if not self._voxels:
            return np.empty((0, 3), dtype=np.float32)
        idx = np.asarray(list(self._voxels), dtype=np.float64)
        centers = idx * self.voxel_size
        keep = np.all((centers >= region_min) & (centers <= region_max), axis=1)
        return centers[keep].astype(np.float32)


class _Open3DVoxelGrid:
    """Occupancy accumulator on an Open3D tensor Device -- CUDA:0 when this
    process can see an NVIDIA GPU, CPU:0 otherwise, chosen once at
    construction. `integrate`/`query` never branch on device again: Open3D's
    tensor ops (voxel_down_sample, boolean masking) run identically on
    either, which is the whole point -- one code path, GPU-accelerated
    for free when it's there.

    Accumulation strategy: concatenate each new batch onto the running
    point set and re-voxel-downsample. Fine for a bounded manipulation
    workspace with stride-subsampled depth input (thousands, not millions,
    of points per integrate() call); would need a real spatial hash (i.e.
    an actual TSDF/ESDF structure) to scale to room- or building-sized maps.
    """

    def __init__(self, voxel_size_m: float = 0.02):
        import open3d as o3d
        import open3d.core as o3c

        self._o3d = o3d
        self._o3c = o3c
        self.voxel_size = float(voxel_size_m)
        self.device = o3c.Device("CUDA:0") if o3c.cuda.is_available() else o3c.Device("CPU:0")
        self._points = o3c.Tensor(np.empty((0, 3), dtype=np.float32), device=self.device)

    def integrate(self, points: np.ndarray) -> None:
        if points.shape[0] == 0:
            return
        o3c = self._o3c
        new = o3c.Tensor(points.astype(np.float32), dtype=o3c.float32, device=self.device)
        merged = o3c.concatenate([self._points, new], axis=0)
        pcd = self._o3d.t.geometry.PointCloud(merged)
        self._points = pcd.voxel_down_sample(self.voxel_size).point.positions

    def query(self, region_min: np.ndarray, region_max: np.ndarray) -> np.ndarray:
        if self._points.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)
        o3c = self._o3c
        lo = o3c.Tensor(region_min.astype(np.float32), dtype=o3c.float32, device=self.device)
        hi = o3c.Tensor(region_max.astype(np.float32), dtype=o3c.float32, device=self.device)
        mask = (self._points >= lo).all(dim=1).logical_and((self._points <= hi).all(dim=1))
        return self._points[mask].cpu().numpy().astype(np.float32)


def _make_grid(backend: str, voxel_size_m: float):
    if backend == "numpy":
        return _VoxelGrid(voxel_size_m), "numpy"
    try:
        grid = _Open3DVoxelGrid(voxel_size_m)
        return grid, f"open3d/{grid.device}"
    except ImportError as e:
        if backend == "open3d":
            raise SystemExit(f"--backend open3d requested but unavailable: {e}") from e
        return _VoxelGrid(voxel_size_m), "numpy (open3d not installed)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=5557)
    ap.add_argument("--voxel-size", type=float, default=0.02)
    ap.add_argument("--backend", choices=["auto", "open3d", "numpy"], default="auto")
    args = ap.parse_args()

    grid, backend_desc = _make_grid(args.backend, args.voxel_size)
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"[serve_nvblox_bridge] occupancy bridge on :{args.port} "
          f"backend={backend_desc} voxel_size={args.voxel_size} m", file=sys.stderr)

    while True:
        raw = sock.recv()
        try:
            req = msgpack.unpackb(raw, raw=False)
            action = req.get("action")
            if action == "integrate":
                grid.integrate(np.asarray(req["points"], dtype=np.float64))
                resp = {}
            elif action == "query":
                pts = grid.query(
                    np.asarray(req["region_min"], dtype=np.float64),
                    np.asarray(req["region_max"], dtype=np.float64),
                )
                resp = {"points": pts}
            else:
                resp = {"error": f"unknown action {action!r}"}
        except Exception as e:  # noqa: BLE001 -- must always reply on the REQ/REP socket
            resp = {"error": f"{type(e).__name__}: {e}"}
        sock.send(msgpack.packb(resp, use_bin_type=True))


if __name__ == "__main__":
    main()
