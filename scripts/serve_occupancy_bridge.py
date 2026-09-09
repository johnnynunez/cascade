#!/usr/bin/env python3
"""cascade occupancy bridge: the process that owns the 3-D obstacle map.

Backends (see src/cascade/perception/occupancy_backends.py), one wire:

  nvblox   P0 on NVIDIA GPUs: real nvblox (nvblox_torch) TSDF + ESDF.
  warp     hardware-agnostic default: dense projective TSDF with carving +
           exact EDT in NVIDIA Warp kernels; CPU on macOS/Linux/aarch64,
           CUDA on Jetson/x86 -- same source.
  voxel    numpy occupancy hash, zero deps. No carving, no ESDF.

  --backend auto (default): nvblox if importable AND a CUDA device exists,
  else warp, else voxel. An EXPLICIT --backend never falls through: a
  requested backend that cannot start is a startup error, because the
  client shows the backend name to the operator as evidence of what checked
  the motion.

Wire protocol (ZMQ REQ/REP, msgpack-numpy; all frames in the robot BASE
frame, metres, float32):

  {"action": "probe"}
      -> {"ok": true, "backend": "warp", "device": "cpu", "voxel": 0.01,
          "esdf": true, "carving": true, "describe": "..."}
  {"action": "integrate_depth", "depth": (H,W), "K": (3,3), "T_base_cam": (4,4)}
      -> {"ms": float}
  {"action": "integrate", "points": (N,3)}          # legacy cloud input
      -> {}
  {"action": "query", "region_min": (3,), "region_max": (3,)}
      -> {"grid": (nx,ny,nz) distance-to-nearest-obstacle [m] (0 inside,
          inf = nothing known), "origin": (3,), "voxel": float,
          "points": (M,3) occupied voxel centres (legacy consumers)}

    ./scripts/serve_occupancy_bridge.py [--port 5557] [--voxel-size 0.01]
        [--backend auto|nvblox|warp|voxel] [--region-min x y z] [--region-max x y z]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import msgpack
import msgpack_numpy
import numpy as np
import zmq

# repo-local import without requiring an editable install of the package
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from cascade.perception.occupancy_backends import make_backend  # noqa: E402

msgpack_numpy.patch()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=5557)
    ap.add_argument("--voxel-size", type=float, default=0.01)
    ap.add_argument("--backend", choices=["auto", "nvblox", "warp", "voxel"], default="auto")
    ap.add_argument("--region-min", type=float, nargs=3, default=(-0.5, -0.6, -0.15),
                    help="workspace AABB the dense backends allocate (base frame, m)")
    ap.add_argument("--region-max", type=float, nargs=3, default=(0.9, 0.6, 0.7))
    ap.add_argument("--device", default="auto", help="warp: cpu|cuda:0|auto")
    args = ap.parse_args()

    t0 = time.perf_counter()
    grid = make_backend(args.backend, voxel=args.voxel_size, region_min=args.region_min,
                        region_max=args.region_max, device=args.device)
    describe = grid.describe()
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"[occupancy-bridge] :{args.port} backend={grid.name} {describe} "
          f"(up in {time.perf_counter()-t0:.1f}s)", file=sys.stderr, flush=True)

    probe = {
        "ok": True, "backend": grid.name, "describe": describe, "voxel": float(grid.voxel),
        "device": str(getattr(grid, "device", "cpu")),
        "esdf": grid.name in ("nvblox", "warp"), "carving": grid.name in ("nvblox", "warp"),
    }
    while True:
        raw = sock.recv()
        try:
            req = msgpack.unpackb(raw, raw=False)
            action = req.get("action")
            if action == "probe":
                resp = probe
            elif action == "integrate_depth":
                t1 = time.perf_counter()
                grid.integrate_depth(np.asarray(req["depth"], dtype=np.float32),
                                     np.asarray(req["K"], dtype=np.float64),
                                     np.asarray(req["T_base_cam"], dtype=np.float64))
                resp = {"ms": (time.perf_counter() - t1) * 1e3}
            elif action == "integrate":
                grid.integrate_points(np.asarray(req["points"], dtype=np.float32).reshape(-1, 3))
                resp = {}
            elif action == "query":
                out = grid.query(np.asarray(req["region_min"], dtype=np.float64),
                                 np.asarray(req["region_max"], dtype=np.float64))
                resp = {"grid": np.ascontiguousarray(out["grid"], dtype=np.float32),
                        "origin": np.asarray(out["origin"], dtype=np.float32),
                        "voxel": float(out["voxel"]),
                        "points": np.asarray(out["points"], dtype=np.float32).reshape(-1, 3)}
            else:
                resp = {"error": f"unknown action {action!r}"}
        except Exception as e:  # noqa: BLE001 -- must always reply on the REQ/REP socket
            resp = {"error": f"{type(e).__name__}: {e}"}
        sock.send(msgpack.packb(resp, use_bin_type=True))


if __name__ == "__main__":
    main()
