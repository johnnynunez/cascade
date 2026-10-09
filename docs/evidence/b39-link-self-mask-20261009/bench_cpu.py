"""B39 offline evidence: link self-mask accuracy and CPU cost on the reBot RS.

CPU only (no GPU, no Isaac, no MuJoCo). For four arm poses and both demo
cameras (configs/cameras/isaac.yaml cam0 and isaac_side.yaml, Isaac optics
1280 x 720), it compares the shipped link mask with an INDEPENDENT truth: the
RS collision STLs read here, posed by Pinocchio directly (fingers at half
travel), filled one triangle per cv2.fillConvexPoly call. It times
`LinkSelfMask.mask_for` per frame (FK + rasterisation + dilation) for the
shipped cell size and the alternatives, with and without the scipy hull
reduction.

    PYTHONPATH=src python docs/evidence/b39-link-self-mask-20261009/bench_cpu.py \
        --json docs/evidence/b39-link-self-mask-20261009/cpu_bench.json
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import struct
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
URDF = REPO / "assets/urdf/00-arm-rs_asm-v3/urdf/00-arm-rs_asm-v3.urdf"
SIGNS = [-1, -1, -1, -1, -1, -1]
POSES = {"home": [0.0, 1.2, 1.2, 0.0, 0.0, 0.0], "handover": [0.5, 1.2, 1.2, 0.0, 0.75, 0.0],
         "reach": [0.3, 0.6, 1.6, 0.2, -0.4, 0.5], "low": [-0.4, 1.6, 0.9, -0.3, 0.6, -1.0]}
W, H = 1280, 720
FX = 18.0 / 20.955 * W
K = np.array([[FX, 0.0, W / 2], [0.0, FX, H / 2], [0.0, 0.0, 1.0]])
REPEATS = 30


def _cam(name: str) -> np.ndarray:
    from cascade.config import load_profile
    return np.asarray(load_profile("cameras", name).extrinsics.T, dtype=float).reshape(4, 4)


def _truth_triangles() -> dict:
    from cascade.types import pose_to_transform
    root = ET.parse(URDF).getroot()
    pkg = URDF.parents[1]
    out = {}
    for link in root.iter("link"):
        tris = []
        for col in link.findall("collision"):
            o = col.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            data = (pkg / col.find("geometry/mesh").get("filename").split("/", 3)[-1]).read_bytes()
            n = struct.unpack("<I", data[80:84])[0]
            rec = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
            T = pose_to_transform(xyz + rpy)
            tris.append(rec["v"].reshape(-1, 3, 3).astype(float) @ T[:3, :3].T + T[:3, 3])
        if tris:
            out[link.get("name")] = np.concatenate(tris)
    return out


def _pin_poses(kin, q, finger_m):
    import pinocchio as pin
    data = kin.model.createData()
    qf = np.zeros(kin.model.nq)
    qf[:6] = q
    qf[6:] = finger_m
    pin.forwardKinematics(kin.model, data, qf)
    pin.updateFramePlacements(kin.model, data)
    return {kin.model.frames[i].name: np.array(data.oMf[i].homogeneous)
            for i in range(len(kin.model.frames)) if kin.model.frames[i].type == pin.FrameType.BODY}


def _fill(tris_by_link, poses, T):
    R, t = T[:3, :3], T[:3, 3]
    m = np.zeros((H, W), np.uint8)
    for name, tris in tris_by_link.items():
        P = poses[name]
        pc = ((tris @ P[:3, :3].T + P[:3, 3]) - t) @ R
        assert (pc[..., 2] > 0.05).all()
        uv = np.stack([K[0, 0] * pc[..., 0] / pc[..., 2] + K[0, 2], K[1, 1] * pc[..., 1] / pc[..., 2] + K[1, 2]], -1)
        for tri in np.round(uv * 16).astype(np.int32):
            cv2.fillConvexPoly(m, tri, 1, cv2.LINE_8, 4)
    return m.astype(bool)


def _metrics(mask, truth):
    n = float(truth.sum())
    return {"coverage": round(float((mask & truth).sum()) / n, 5), "bloat": round(float((mask & ~truth).sum()) / n, 4),
            "iou": round(float((mask & truth).sum()) / float((mask | truth).sum()), 4), "truth_px": int(n)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    from cascade.control.kinematics import Kinematics
    from cascade.perception import link_mask as lm
    from cascade.types import Frame, RobotState

    kin = Kinematics(str(URDF), "gripper_end", 6, SIGNS)
    tris = _truth_triangles()
    cams = {c: _cam(c) for c in ("isaac", "isaac_side")}
    truth = {(p, c): _fill(tris, _pin_poses(kin, q, 0.025), cams[c]) for p, q in POSES.items() for c in cams}
    frame = Frame(rgb=np.zeros((H, W, 3), np.uint8), depth_m=None, K=K.copy(), t=100.0)

    variants = [("cell 0.05 m (shipped)", 0.05, True), ("cell 0.02 m", 0.02, True),
                ("one hull per link", 10.0, True), ("cell 0.05 m, no scipy", 0.05, False)]
    out = {"host": {"cpu": platform.processor() or platform.machine(), "python": platform.python_version(),
                    "opencv": cv2.__version__, "cv2_threads": cv2.getNumThreads()},
           "image": [W, H], "repeats": REPEATS, "variants": []}
    original = lm._hull_vertices
    for label, cell, hulls in variants:
        lm._hull_vertices = original if hulls else (lambda pts: np.unique(np.asarray(pts, dtype=float), axis=0))
        t0 = time.perf_counter()
        geometry = lm.LinkGeometry.from_urdf(URDF, cell_m=cell)
        load_ms = (time.perf_counter() - t0) * 1e3
        lm._hull_vertices = original
        poses = lm.KinematicsLinkPoses(kin, geometry.links, gripper_open=1.0, gripper_closed=0.0)
        rows = []
        for pname, q in POSES.items():
            state = RobotState(q=np.asarray(q, float), t=100.0, gripper_pos=0.5)
            for cname, T in cams.items():
                row = {"pose": pname, "camera": cname}
                for d in (0, 2):
                    link = lm.LinkSelfMask(dilate_px=d)
                    link.add_arm("arm0", geometry, poses, lambda s=state: s)
                    mask = link.mask_for(frame, T)
                    row[f"d{d}"] = _metrics(mask, truth[(pname, cname)])
                link = lm.LinkSelfMask(dilate_px=2)
                link.add_arm("arm0", geometry, poses, lambda s=state: s)
                wall, cpu = [], []
                for _ in range(REPEATS):
                    w0, c0 = time.perf_counter(), time.process_time()
                    link.mask_for(frame, T)
                    wall.append((time.perf_counter() - w0) * 1e3)
                    cpu.append((time.process_time() - c0) * 1e3)
                t_fk = []
                for _ in range(REPEATS):
                    w0 = time.perf_counter()
                    poses(state)
                    t_fk.append((time.perf_counter() - w0) * 1e3)
                row["ms_wall_median"] = round(statistics.median(wall), 2)
                row["ms_wall_p90"] = round(sorted(wall)[int(0.9 * len(wall)) - 1], 2)
                row["ms_cpu_median"] = round(statistics.median(cpu), 2)
                row["ms_fk_median"] = round(statistics.median(t_fk), 3)
                rows.append(row)
        out["variants"].append({"variant": label, "cell_m": cell, "scipy_hulls": hulls, "load_ms": round(load_ms, 1),
                                "pieces": geometry.n_pieces(), "points": geometry.n_points(), "rows": rows})

    print(f"host: {out['host']}")
    for v in out["variants"]:
        rows = v["rows"]
        cov0 = min(r["d0"]["coverage"] for r in rows)
        cov2 = min(r["d2"]["coverage"] for r in rows)
        b0 = (min(r["d0"]["bloat"] for r in rows), max(r["d0"]["bloat"] for r in rows))
        b2 = (min(r["d2"]["bloat"] for r in rows), max(r["d2"]["bloat"] for r in rows))
        iou2 = (min(r["d2"]["iou"] for r in rows), max(r["d2"]["iou"] for r in rows))
        ms = (min(r["ms_wall_median"] for r in rows), max(r["ms_wall_median"] for r in rows))
        print(f"| {v['variant']} | {v['pieces']} pieces / {v['points']} pts | load {v['load_ms']:.0f} ms "
              f"| d0 cov>={cov0} bloat {b0[0]}-{b0[1]} | d2 cov>={cov2} bloat {b2[0]}-{b2[1]} "
              f"IoU {iou2[0]}-{iou2[1]} | {ms[0]}-{ms[1]} ms/frame |")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
