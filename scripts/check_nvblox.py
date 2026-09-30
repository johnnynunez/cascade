#!/usr/bin/env python3
"""Exercise real CUDA depth integration and ESDF queries, without a simulator."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import nvblox_torch
    import torch

    from cascade.perception.occupancy_backends import NvbloxBackend

    if not torch.cuda.is_available():
        raise RuntimeError("nvblox validation requires CUDA")
    backend = NvbloxBackend(voxel=0.01)
    depth = np.full((120, 160), 0.6, dtype=np.float32)
    intrinsics = np.array([[150, 0, 80], [0, 150, 60], [0, 0, 1]], dtype=np.float32)
    for _ in range(3):
        backend.integrate_depth(depth, intrinsics, np.eye(4))
    result = backend.query([0, 0, 0.4], [0.01, 0.01, 0.6])
    grid = result["grid"]
    if not np.isfinite(grid).all() or abs(float(grid[0, 0, 0]) - 0.2) > 0.025:
        raise RuntimeError(f"Unexpected plane clearance: {grid[0, 0, :].tolist()}")
    if float(grid[0, 0, -1]) > 0.02:
        raise RuntimeError("Plane surface is missing from the ESDF")
    backend.clear()
    cleared = backend.query([0, 0, 0.4], [0.01, 0.01, 0.6])["grid"]
    if not np.isinf(cleared).all():
        raise RuntimeError("Map clear retained old ESDF geometry")
    receipt = {
        "passed": True,
        "backend": backend.name,
        "device": backend.device,
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nvblox": nvblox_torch.__version__,
        "source_sha": nvblox_torch.__git_sha__,
        "numpy": importlib.metadata.version("numpy"),
        "voxel_m": backend.voxel,
        "grid_shape": list(grid.shape),
        "plane_clearance_m": grid[0, 0, :].tolist(),
        "map_clear_removed_previous_geometry": True,
        "last_integration_ms": backend.last_integrate_ms,
        "query_ms": backend.last_query_ms,
        "scope": "Real CUDA mapping and ESDF query; separate camera and manipulation acceptance required.",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
