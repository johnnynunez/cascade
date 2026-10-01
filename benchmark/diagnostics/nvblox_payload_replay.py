"""Query complete measured payload surfaces against frozen real RGB-D captures.

Unlike a few historical failure positions, this covers every visible surface
cell of the held prop. Optional vertical translations are candidate geometry,
not a motion, reachability check, swept-volume proof, or physical acceptance.
No surface or depth is synthesized to fill gaps in the map.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_identity(entry, camera):
    capture = entry.get("capture", {})
    proprio = capture.get("proprioception", {})
    source = capture.get("source")
    stamp = capture.get("t")
    if (capture.get("backend") != "isaac" or capture.get("camera") != camera
            or not isinstance(source, list) or len(source) != 2
            or not isinstance(source[0], str) or not source[0]
            or type(source[1]) is not int or not 1 <= source[1] <= 65535
            or not isinstance(stamp, (int, float)) or not np.isfinite(stamp)
            or proprio.get("backend") != "isaac" or proprio.get("t") != stamp
            or not isinstance(proprio.get("robot_id"), str) or not proprio["robot_id"]
            or proprio.get("time_source") != "physics_loop_monotonic"
            or proprio.get("joint_convention") != "asset"):
        raise ValueError(f"Unbound Isaac capture: {camera}")
    return tuple(source), proprio["robot_id"], proprio["time_source"]


def load_capture(folder, cameras, prop):
    """Load archived depth/K/T/robot/prop_N arrays and verify their receipt."""
    metadata = json.loads((folder / "metadata.json").read_text())
    frames = {}
    identity = None
    for name in cameras:
        entry = metadata["cameras"][name]
        current = capture_identity(entry, name)
        if identity is not None and current != identity:
            raise ValueError("Cameras do not share a source, robot and capture clock")
        identity = current
        path = folder / f"{name}.npz"
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Capture checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            frame = dict(archive)
        # Accept the earlier experimental archives and the checked-in capture tool.
        frame["robot"] = frame.get("robot_mask", frame.get("robot"))
        if frame["depth"].ndim != 2 or frame["depth"].dtype.kind != "f":
            raise ValueError(f"Depth must be a floating-point 2D metre image: {path}")
        keys = [key for key, value in entry["props"].items() if value == prop]
        if len(keys) != 1:
            raise ValueError(f"Expected one exact prop mask for {prop}: {path}")
        frame["prop"] = frame[keys[0]]
        for mask in (frame["robot"], frame["prop"]):
            if mask is None or mask.dtype != np.bool_ or mask.shape != frame["depth"].shape:
                raise ValueError(f"Invalid binary capture mask: {path}")
        if (frame["K"].shape != (3, 3) or frame["T"].shape != (4, 4)
                or not np.isfinite(frame["K"]).all() or not np.isfinite(frame["T"]).all()):
            raise ValueError(f"Invalid capture calibration: {path}")
        K, T = frame["K"], frame["T"]
        if (K[0, 0] <= 0 or K[1, 1] <= 0 or K[0, 1] != 0 or K[1, 0] != 0
                or not np.allclose(K[2], [0, 0, 1], atol=1e-8, rtol=0)
                or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
                or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5, rtol=0)
                or not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-5, rtol=0)):
            raise ValueError(f"Calibration requires a pinhole camera and rigid transform: {path}")
        frames[name] = frame
    provenance = {"directory": str(folder), "metadata_sha256": sha256(folder / "metadata.json"),
                  "cameras": {name: metadata["cameras"][name] for name in cameras}}
    return frames, provenance


def validate_pair(anchor, held, anchor_receipt, held_receipt, prop):
    for name in anchor:
        before, after = anchor_receipt["cameras"][name], held_receipt["cameras"][name]
        if capture_identity(before, name) != capture_identity(after, name):
            raise ValueError(f"Anchor/held source, robot or clock changed: {name}")
        if after["capture"]["t"] <= before["capture"]["t"]:
            raise ValueError(f"Held capture must follow the anchor: {name}")
        if prop in before["capture"].get("contact_paths", []):
            raise ValueError("Anchor must precede attachment of the requested prop")
        if prop not in after["capture"].get("contact_paths", []):
            raise ValueError("Held capture lacks confirmed contact with the requested prop")
        if (anchor[name]["depth"].shape != held[name]["depth"].shape
                or not np.allclose(anchor[name]["K"], held[name]["K"], atol=1e-6, rtol=0)):
            raise ValueError(f"Anchor/held image calibration changed: {name}")


def measured_surface(frame):
    """One actual measurement per 3 mm base-frame cell; no meshing/completion."""
    valid = frame["prop"] & np.isfinite(frame["depth"]) & (frame["depth"] > 0)
    y, x = np.nonzero(valid)
    z, K, T = frame["depth"][y, x], frame["K"], frame["T"]
    camera = np.column_stack(((x - K[0, 2]) * z / K[0, 0],
                              (y - K[1, 2]) * z / K[1, 1], z))
    points = camera @ T[:3, :3].T + T[:3, 3]
    _, index = np.unique(np.floor(points / .003).astype(np.int64), axis=0, return_index=True)
    return points[index]


def summarize(mapping, points):
    distance = mapping.payload_clearance(points)
    if distance is None:
        raise RuntimeError("Replay did not provide a fresh query grid")
    unknown = points[~np.isfinite(distance)]
    return {**mapping.last_payload_query,
            "below_30mm_samples": int((distance < .03).sum()),
            "unobserved_bounds_m": [unknown.min(0).tolist(), unknown.max(0).tolist()] if len(unknown) else None,
            "first_unobserved_point_m": unknown[0].tolist() if len(unknown) else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--held", type=Path, required=True)
    parser.add_argument("--prop", default="/World_Props/tomato_can")
    parser.add_argument("--cameras", nargs="+", default=["cam0", "side", "proof"])
    parser.add_argument("--voxels", nargs="+", type=float, default=[.01, .005])
    parser.add_argument("--strides", nargs="+", type=int, default=[2, 1])
    parser.add_argument("--z-offsets", nargs="+", type=float, default=[0., .01, .02, .03, .04, .06])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.output.exists() or any(not np.isfinite(v) or not .001 <= v <= .05 for v in args.voxels)
            or any(not 1 <= s <= 8 for s in args.strides) or not np.isfinite(args.z_offsets).all()
            or any(abs(z) > 1 for z in args.z_offsets) or len(set(args.cameras)) != len(args.cameras)):
        parser.error("output must be new; voxels 1..50mm, strides 1..8, finite offsets within 1m, unique cameras")
    sys.path.insert(0, str(ROOT / "src"))
    import nvblox_torch
    import torch
    from cascade.perception.occupancy import OccupancyMap
    from cascade.perception.occupancy_backends import GridSpec, NvbloxBackend

    anchor, anchor_receipt = load_capture(args.anchor, args.cameras, args.prop)
    held, held_receipt = load_capture(args.held, args.cameras, args.prop)
    validate_pair(anchor, held, anchor_receipt, held_receipt, args.prop)
    surfaces = {name: measured_surface(frame) for name, frame in held.items()}
    points = np.concatenate(list(surfaces.values()))
    if not len(points):
        raise ValueError("Held prop has no measured depth surface")
    candidates = {str(z): points + [0., 0., z] for z in args.z_offsets}
    all_points = np.concatenate(list(candidates.values()))
    for voxel in args.voxels:
        spec = GridSpec(all_points.min(0) - voxel, all_points.max(0) + voxel, voxel, voxel_centres=True)
        if np.prod(spec.shape, dtype=object) > 5_000_000:
            parser.error("replay query would exceed five million grid cells")
    report = {
        "scope": "Frozen real depth, exact prop masks and complete visible surface-cell coverage. "
                 "Vertical translations are candidate geometry only; no motion, reachability, "
                 "swept path or physical acceptance is established.",
        "sampling": "One measured point per 3mm base-frame cell per camera. Live runtime bins "
                    "in captured TCP coordinates, so these are not its exact sample indices.",
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "cuda": torch.version.cuda, "nvblox": nvblox_torch.__version__,
        "nvblox_source": nvblox_torch.__git_sha__,
        "prop": args.prop, "capture": {"anchor": anchor_receipt, "held": held_receipt},
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (
            Path(__file__), ROOT / "src/cascade/perception/occupancy.py",
            ROOT / "src/cascade/perception/occupancy_backends.py")},
        "surface_samples": len(points), "surfaces_by_camera": {k: len(v) for k, v in surfaces.items()},
        "surface_bounds_m": [points.min(0).tolist(), points.max(0).tolist()],
        "surface_points_sha256": hashlib.sha256(points.tobytes()).hexdigest(), "variants": [],
    }
    for voxel in args.voxels:
        for stride in args.strides:
            backend = NvbloxBackend(voxel=voxel)
            mapping = OccupancyMap(None)
            variant = {"voxel_m": voxel, "depth_stride": stride, "native_mask": True, "stages": []}
            for episode, frames in (("anchor", anchor), ("held", held)):
                for name in args.cameras:
                    frame = frames[name]
                    mask = frame["robot"] | frame["prop"]
                    K = frame["K"].copy()
                    K[:2] /= stride
                    backend.integrate_masked_depth(frame["depth"][::stride, ::stride], K, frame["T"],
                                                   (~mask[::stride, ::stride]).astype(np.uint8))
                grid = backend.query(all_points.min(0) - voxel, all_points.max(0) + voxel)
                mapping._grid, mapping._grid_origin = grid["grid"], grid["origin"]
                mapping._grid_voxel, mapping._last_refresh = grid["voxel"], time.monotonic()
                stage = {"episode": episode, "z_offsets_m": {
                    z: summarize(mapping, candidate) for z, candidate in candidates.items()}}
                variant["stages"].append(stage)
                print(json.dumps({"voxel_m": voxel, "stride": stride, **stage}), flush=True)
            report["variants"].append(variant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
