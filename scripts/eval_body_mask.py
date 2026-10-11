#!/usr/bin/env python3
"""Score the occupancy body masks (capsule vs URDF mesh) on a recorded capture.

WHY: on the real reBot rig (2026-10-11) the capsule body mask left 108
occupied voxels on the gripper and the wrist camera, under the 3 cm
clearance. This tool reruns both masks on a recorded RGB-D capture, offline,
and reports what each leaves in the map, so the mask can be re-validated on
any rig or camera without the robot or a bridge:

    python scripts/eval_body_mask.py CAPTURE.npz --arm rebot_rs --q 0 0 0 0 0 0
        [--gripper RAD] [--stride 2] [--voxel 0.01] [--json report.json]

CAPTURE.npz: `depth` (N,H,W) or (H,W) metres, `K` (N,3,3) or (3,3), and
`T_cam2base` (4,4), or (N,4,4) for an eye-in-hand camera (FK(q) @ X per
frame). The arm's joint state is `--q` (the capture has none): the robot
must not have moved during the capture. `--gripper` is the gripper motor
position (the profile's open_pos/closed_pos map it onto the fingers);
omitted = unknown = the fingers swept, as the runtime does.

NOTHING here touches hardware or a bridge: kinematics come from the
profile's URDF, the masks are the production `robot_mask` (capsules,
`body_mask_radius_m`) and `MeshBodyMask` (the profile's `body_mask`, or the
shipped mesh defaults when the profile keeps capsules), at the production
depth stride.

Metrics (per method):
- unmasked_within_3cm / _5cm: depth points per frame that reach the map and
  lie within 3 / 5 cm of the arm's URDF collision surface, above the table
  (z > --table-z). These are the robot polluting its own map.
- occupied_voxels_within_3cm / _5cm: the same, as distinct --voxel cells over
  all frames (an occupancy hash of every unmasked point: no carving, no
  weights, so an upper bound on what a TSDF keeps).
- min_clearance_m: the nearest occupied voxel (any, in the region) to the
  points the safety harness checks (TCP + joint frames 2..n); motion is
  refused below `min_clearance_m` (3 cm in configs/demo.yaml).
- hidden_points: masked depth points per frame farther than 5 cm from the
  arm, above the table and inside the region: real surfaces the mask hides
  (the cost of the margin).
- mask_ms_median: the mask's own cost per frame (FK + raster for the mesh).
- nearest_to_chain_m: the nearest occupied voxel to the joint polyline the
  capsules thicken (the 2026-10-11 finding: 2.8 cm in the nvblox map).

`--tsdf` also runs each mask through cascade's warp TSDF backend on the CPU
(the production path for a bridge without native masking: masked pixels
zeroed, then integrated; needs `warp-lang`, the `occupancy` extra) and adds
tsdf_occupied_within_3cm/_5cm, tsdf_nearest_to_chain_m and
tsdf_harness_clearance_m (the ESDF sampled by OccupancyMap.clearance at the
harness's points). A TSDF marks voxels BEHIND a visible surface as occupied
too, which the point hash above cannot show: that is how the nvblox map had
voxels 2.8 cm from the chain although the capsules mask 6 cm.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def load_capture(path) -> dict:
    """{depth (N,H,W) float32, K (N,3,3), T (N,4,4)} from a capture .npz."""
    data = np.load(path)
    depth = np.asarray(data["depth"], dtype=np.float32)
    if depth.ndim == 2:
        depth = depth[None]
    n = len(depth)
    K = np.asarray(data["K"], dtype=np.float64)
    K = np.broadcast_to(K, (n, 3, 3)) if K.ndim == 2 else K
    T = np.asarray(data["T_cam2base"], dtype=np.float64)
    T = np.broadcast_to(T, (n, 4, 4)) if T.ndim == 2 else T
    if len(K) != n or len(T) != n:
        raise ValueError(f"{path}: {n} depth frames but {len(K)} K and {len(T)} T_cam2base")
    return {"depth": depth, "K": K, "T": T}


def _surface_points(acfg, kin, state, spacing: float = 0.006) -> np.ndarray:
    """Dense samples of the URDF collision surface at `state` (base frame):
    every triangle split to `spacing`, every finger configuration posed."""
    from cascade.perception import link_mask as lm

    urdf = Path(acfg.model)
    root = ET.parse(urdf).getroot()
    local = {}
    for link in root.iter("link"):
        tris = []
        for col in link.findall("collision"):
            geo = col.find("geometry")
            el = geo[0] if geo is not None and len(geo) else None
            if el is None or el.tag != "mesh":
                continue
            t = lm.load_mesh_triangles(lm._resolve_mesh(el.get("filename", ""), urdf.parent))
            scale = np.asarray([float(v) for v in el.get("scale", "1 1 1").split()])
            t = lm._transform((t * scale).reshape(-1, 3), lm._origin(col.find("origin"))).reshape(-1, 3, 3)
            tris.append(lm._subdivide(t, spacing).reshape(-1, 3))
        if tris:
            local[link.get("name")] = np.unique(np.concatenate(tris), axis=0)
    grip = acfg.get("gripper") or {}
    poses = lm.KinematicsLinkPoses(kin, list(local), gripper_open=grip.get("open_pos"),
                                   gripper_closed=grip.get("closed_pos"))(state)
    return np.concatenate([lm._transform(pts, P[name]) for P in poses for name, pts in local.items()])


def _backproject(depth, K, T):
    ys, xs = np.nonzero(np.isfinite(depth) & (depth > 0))
    z = depth[ys, xs].astype(np.float64)
    pc = np.stack([(xs - K[0, 2]) / K[0, 0] * z, (ys - K[1, 2]) / K[1, 1] * z, z], -1)
    return ys, xs, pc @ T[:3, :3].T + T[:3, 3]


def warp_tsdf_cpu(voxel, region_min, region_max):
    """cascade's warp TSDF + EDT backend on the CPU (never a GPU here)."""
    from cascade.perception.occupancy_backends import WarpTsdfBackend

    return WarpTsdfBackend(voxel=voxel, region_min=region_min, region_max=region_max, device="cpu")


def _chain_distance(points: np.ndarray, chain: np.ndarray) -> float:
    from cascade.perception.robot_mask import _segment_distances

    if not len(points):
        return float("inf")
    return float(np.min([_segment_distances(points, chain[i], chain[i + 1]) for i in range(len(chain) - 1)]))


def _harness_clearance(fetched, checked) -> float:
    """The production read: OccupancyMap.clearance on the fetched ESDF."""
    from types import SimpleNamespace

    from cascade.perception.occupancy import OccupancyMap

    om = OccupancyMap(client=None)
    om._commit_grid(SimpleNamespace(has_depth=False), fetched, None)   # install, as a refresh would
    dist = om.clearance(checked)
    return float("inf") if dist is None else float(np.min(dist))


def evaluate(capture: dict, *, arm: str = "rebot_rs", q=None, gripper_pos=None, stride: int = 2,
             voxel: float = 0.01, table_z: float = 0.015, region=None, inflate_m=None,
             dilate_px=None, tsdf_factory=None) -> dict:
    """Both masks on every frame of `capture` (see the module docstring).
    `tsdf_factory(voxel, region_min, region_max)` -> a backend with
    integrate_depth/query (e.g. `warp_tsdf_cpu`) adds the TSDF metrics."""
    from scipy.spatial import cKDTree

    from cascade.config import Cfg, load_demo_config
    from cascade.control.kinematics import Kinematics
    from cascade.perception.robot_mask import (
        arm_link_points,
        body_mask_settings,
        build_mesh_body_mask,
        robot_mask,
    )
    from cascade.types import RobotState, pose_to_transform

    cfg = load_demo_config(arm=arm, llm="mock")
    acfg = cfg.arm
    n_joints = int(acfg.get("n_joints", 6))
    kin = Kinematics(model_path=acfg.model, ee_frame=acfg.get("ee_frame", "gripper_end"),
                     n_controlled=n_joints, joint_signs=acfg.get("joint_signs"))
    q = np.zeros(n_joints) if q is None else np.asarray(q, dtype=np.float64)
    state = RobotState(q=q, gripper_pos=0.0 if gripper_pos is None else float(gripper_pos),
                       gripper_valid=gripper_pos is not None)
    base_T = None if acfg.get("base_pose") is None else pose_to_transform(acfg.base_pose)

    def to_table(p):
        return p if base_T is None else p @ base_T[:3, :3].T + base_T[:3, 3]

    # the mesh mask: the profile's own block, or the shipped mesh defaults
    raw = acfg.as_dict()
    block: dict = {"mode": "mesh"}
    if body_mask_settings(acfg) is not None and isinstance(raw.get("body_mask"), dict):
        block = dict(raw["body_mask"])
    if inflate_m is not None:
        block["inflate_m"] = float(inflate_m)
    if dilate_px is not None:
        block["dilate_px"] = int(dilate_px)
    mesh = build_mesh_body_mask(Cfg({**raw, "body_mask": block}), kin, lambda: state, name=arm)
    chain = to_table(arm_link_points(kin, q))
    radius = float(acfg.get("body_mask_radius_m", 0.06))
    checked = to_table(np.vstack([kin.fk(q)[:3, 3][None], kin.link_positions(q)[1:]]))  # the harness's points
    surface = cKDTree(to_table(_surface_points(acfg, kin, state)))
    if region is None:
        occ = cfg.get("occupancy") or {}
        ws = acfg.get("resolved").safety.workspace if acfg.get("resolved") else cfg.safety.workspace
        region = (np.asarray(occ.get("region_min") or ws.min, float), np.asarray(occ.get("region_max") or ws.max, float))
    rmin, rmax = (np.asarray(r, dtype=np.float64) for r in region)

    methods = {"capsule": lambda d, K, T: robot_mask(d, K, T, chain, radius_m=radius), "mesh": mesh}
    acc = {m: {"u3": 0, "u5": 0, "hidden": 0, "px": 0, "ms": [], "voxels": set()} for m in methods}
    n = len(capture["depth"])
    for depth_full, K_full, T in zip(capture["depth"], capture["K"], capture["T"], strict=True):
        depth = np.ascontiguousarray(depth_full[::stride, ::stride], dtype=np.float32)
        K = K_full.copy()
        K[:2, :] /= stride
        ys, xs, pb = _backproject(depth, K, T)
        d = surface.query(pb, distance_upper_bound=0.5)[0]
        above = pb[:, 2] > table_z
        inside = np.all((pb >= rmin) & (pb <= rmax), axis=1)
        for name, fn in methods.items():
            t0 = time.perf_counter()
            mask = fn(depth, K, T)
            a = acc[name]
            a["ms"].append((time.perf_counter() - t0) * 1e3)
            kept = ~mask[ys, xs]
            a["u3"] += int((kept & above & (d < 0.03)).sum())
            a["u5"] += int((kept & above & (d < 0.05)).sum())
            a["hidden"] += int((~kept & above & inside & (d > 0.05)).sum())
            a["px"] += int(mask.sum())
            a["voxels"].update(map(tuple, np.floor(pb[kept & inside] / voxel).astype(np.int64)))
    report = {"arm": arm, "frames": n, "stride": stride, "shape": list(capture["depth"].shape[1:]),
              "voxel": voxel, "q": q.tolist(), "gripper_pos": gripper_pos,
              "mesh_settings": {k: v for k, v in block.items() if k != "attachments"},
              "region": [rmin.tolist(), rmax.tolist()]}
    for name, a in acc.items():
        keys = np.array(sorted(a["voxels"]), dtype=np.float64).reshape(-1, 3)
        centres = (keys + 0.5) * voxel
        dv = surface.query(centres, distance_upper_bound=0.5)[0] if len(centres) else np.empty(0)
        up = centres[:, 2] > table_z if len(centres) else np.empty(0, bool)
        clearance = (float(cKDTree(centres).query(checked)[0].min()) if len(centres) else float("inf"))
        report[name] = {
            "unmasked_within_3cm": round(a["u3"] / n, 1), "unmasked_within_5cm": round(a["u5"] / n, 1),
            "occupied_voxels_within_3cm": int((up & (dv < 0.03)).sum()),
            "occupied_voxels_within_5cm": int((up & (dv < 0.05)).sum()),
            "min_clearance_m": round(clearance, 4), "hidden_points": round(a["hidden"] / n, 1),
            "nearest_to_chain_m": round(_chain_distance(centres, chain), 4),
            "mask_px": round(a["px"] / n, 1), "mask_ms_median": round(float(np.median(a["ms"])), 2),
        }
    if tsdf_factory is not None:
        for name, fn in methods.items():
            backend = tsdf_factory(voxel, rmin, rmax)
            for depth_full, K_full, T in zip(capture["depth"], capture["K"], capture["T"], strict=True):
                depth = np.ascontiguousarray(depth_full[::stride, ::stride], dtype=np.float32)
                K = K_full.copy()
                K[:2, :] /= stride
                backend.integrate_depth(np.where(fn(depth, K, T), 0.0, depth).astype(np.float32), K, T)
            out = backend.query(rmin, rmax)
            P = np.asarray(out["points"], dtype=np.float64).reshape(-1, 3)
            dv = surface.query(P, distance_upper_bound=0.5)[0] if len(P) else np.empty(0)
            up = P[:, 2] > table_z
            fetched = (P.astype(np.float32), np.asarray(out["grid"], np.float32),
                       np.asarray(out["origin"], np.float32), float(out["voxel"]))
            report[name].update({
                "tsdf_occupied": len(P),
                "tsdf_occupied_within_3cm": int((up & (dv < 0.03)).sum()),
                "tsdf_occupied_within_5cm": int((up & (dv < 0.05)).sum()),
                "tsdf_nearest_to_chain_m": round(_chain_distance(P, chain), 4),
                "tsdf_harness_clearance_m": round(_harness_clearance(fetched, checked), 4),
            })
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("capture", type=Path)
    ap.add_argument("--arm", default="rebot_rs")
    ap.add_argument("--q", type=float, nargs="+", default=None, help="joint state (local convention, rad)")
    ap.add_argument("--gripper", type=float, default=None, help="gripper motor position; omitted = unknown")
    ap.add_argument("--stride", type=int, default=2, help="occupancy.depth_stride (configs/demo.yaml: 2)")
    ap.add_argument("--voxel", type=float, default=0.01)
    ap.add_argument("--table-z", type=float, default=0.015)
    ap.add_argument("--inflate-m", type=float, default=None, help="override the mesh margin")
    ap.add_argument("--dilate-px", type=int, default=None, help="override the mesh dilation")
    ap.add_argument("--tsdf", action="store_true", help="also run cascade's warp TSDF on the CPU")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    report = evaluate(load_capture(args.capture), arm=args.arm, q=args.q, gripper_pos=args.gripper,
                      stride=args.stride, voxel=args.voxel, table_z=args.table_z,
                      inflate_m=args.inflate_m, dilate_px=args.dilate_px,
                      tsdf_factory=warp_tsdf_cpu if args.tsdf else None)
    h, w = report["shape"]
    print(f"{report['frames']} frames, {w // args.stride}x{h // args.stride} after stride {args.stride}, "
          f"q={report['q']}, gripper={'unknown (swept)' if args.gripper is None else args.gripper}")
    keys = ["unmasked_within_3cm", "unmasked_within_5cm", "occupied_voxels_within_3cm",
            "occupied_voxels_within_5cm", "min_clearance_m", "nearest_to_chain_m", "hidden_points", "mask_px",
            "mask_ms_median"]
    if args.tsdf:
        keys += ["tsdf_occupied", "tsdf_occupied_within_3cm", "tsdf_occupied_within_5cm",
                 "tsdf_nearest_to_chain_m", "tsdf_harness_clearance_m"]
    print(f"{'':28s}{'capsule':>12s}{'mesh':>12s}")
    for k in keys:
        print(f"{k:28s}{report['capsule'][k]:>12}{report['mesh'][k]:>12}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
