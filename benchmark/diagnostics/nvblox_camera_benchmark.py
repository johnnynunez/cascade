"""Replay the same measured kitchen depth on a GPU; evaluate cube surface coverage."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--repo", type=Path, required=True)
p.add_argument("--capture", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--voxel", type=float, default=0.01)
a = p.parse_args()
sys.path.insert(0, str(a.repo / "src"))
import nvblox_torch
import torch
from scipy.spatial.transform import Rotation

from cascade.perception.occupancy_backends import NvbloxBackend

meta = json.loads((a.capture / "metadata.json").read_text())
for name, entry in meta["cameras"].items():
    actual = hashlib.sha256((a.capture / (name + ".npz")).read_bytes()).hexdigest()
    if actual != entry["sha256"]:
        raise RuntimeError(f"Capture checksum mismatch: {name}")
scene = json.loads((a.repo / "demo/scene/kitchen_config.json").read_text())
frames = {
    name: dict(np.load(a.capture / (name + ".npz")))
    for name in ["cam0", "side", "proof"]
}
for frame in frames.values():
    d = frame["depth"].copy()
    d[frame["robot_mask"]] = 0
    frame["depth"] = d[::2, ::2].copy()
    frame["K"] = frame["K"].copy()
    frame["K"][:2, :] /= 2


# Samples lie on five accessible cube faces, not the underside/table interface.
# Truth never enters reconstruction: it supplies these evaluation points only.
def cube_samples(name):
    half = np.array(scene["cube_dimensions"][name]) / 2
    truth = meta["truth"][name]
    faces = []
    for axis, sign in [(0, -1), (0, 1), (1, -1), (1, 1), (2, 1)]:
        other = [j for j in range(3) if j != axis]
        uv = (
            np.array(np.meshgrid(np.linspace(-0.8, 0.8, 7), np.linspace(-0.8, 0.8, 7)))
            .reshape(2, -1)
            .T
        )
        pts = np.zeros((len(uv), 3))
        pts[:, axis] = sign * half[axis]
        pts[:, other] = uv * half[other]
        q = np.array(truth["quaternion_wxyz"])
        R = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        pts = pts @ R.T + truth["position_m"]
        faces.append((f"{axis}:{sign}", pts))
    return faces


result = {
    "scope": "Actual idle Isaac depth replay. Two cube surface-distance/coverage metrics; not grasp placement accuracy or live motion safety certification.",
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "capability": torch.cuda.get_device_capability(0),
    "nvblox": nvblox_torch.__version__,
    "nvblox_source": nvblox_torch.__git_sha__,
    "voxel_m": a.voxel,
    "depth_stride": 2,
    "input_metadata_sha256": hashlib.sha256(
        (a.capture / "metadata.json").read_bytes()
    ).hexdigest(),
    "variants": [],
}
for names in [("cam0",), ("cam0", "side"), ("cam0", "side", "proof")]:
    b = NvbloxBackend(voxel=a.voxel)
    times = []
    for repeat in range(5):
        for name in names:
            f = frames[name]
            start = time.perf_counter()
            b.integrate_depth(f["depth"], f["K"], f["T"])
            times.append((time.perf_counter() - start) * 1000)
    report = {
        "cameras": names,
        "integrations": len(times),
        "cold_first_integrate_ms": times[0],
        "warm_integrate_median_ms": float(np.median(times[len(names) :])),
        "cubes": {},
    }
    for name in scene["cube_dimensions"]:
        face_metrics = {}
        all_values = []
        observed_count = 0
        for face, xyz in cube_samples(name):
            spheres = np.zeros((len(xyz), 4), np.float32)
            spheres[:, :3] = xyz
            sdf = (
                b.mapper.query_layer(
                    b._QueryType.ESDF, torch.as_tensor(spheres, device="cuda")
                )
                .flatten()
                .cpu()
                .numpy()
            )
            known = np.isfinite(sdf) & (sdf != b._unknown)
            vals = np.abs(sdf[known])
            all_values.extend(vals.tolist())
            observed_count += int(known.sum())
            face_metrics[face] = {
                "samples": len(xyz),
                "observed": int(known.sum()),
                "within_one_voxel": int((vals <= a.voxel).sum()),
                "abs_distance_mean_m": float(vals.mean()) if len(vals) else None,
            }
        report["cubes"][name] = {
            "surface_samples": 245,
            "observed_samples": observed_count,
            "within_one_voxel": sum(
                x["within_one_voxel"] for x in face_metrics.values()
            ),
            "abs_distance_mean_m": float(np.mean(all_values)) if all_values else None,
            "faces": face_metrics,
        }
    start = time.perf_counter()
    grid = b.query([0.1, -0.3, -0.01], [0.5, 0.3, 0.55])
    report["production_grid_query_ms"] = (time.perf_counter() - start) * 1000
    report["grid_shape"] = list(grid["grid"].shape)
    report["finite_grid_cells"] = int(np.isfinite(grid["grid"]).sum())
    report["occupied_cells"] = len(grid["points"])
    result["variants"].append(report)
    print(json.dumps(report), flush=True)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(result, indent=2) + "\n")
