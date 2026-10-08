"""B32 diagnosis: what are the bare-scene "biplane" and "building block" beliefs?

On the live demo path (same runtime, cameras, detector and workspace filter as
WorldWatcher._tick), log every YOLOE detection per camera and round with:
- its robot-pixel overlap (frame.robot_mask minus payload, served by the bridge
  with CASCADE_ISAAC_PIXEL_MASK=1);
- its 3D centre/extents, the workspace verdict and the nearest PhysX prop;
- its colour (detection_color);
- its image overlap with the other detections of the same frame.

Annotated crops are saved for every detection not within HIT_M of a real prop
and for every detection labelled like the phantoms.

    python b32_phantom_diag.py --rounds 20 --out results_b32/diag_N
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

LIVE = Path(__file__).resolve().parent
sys.path.insert(0, str(LIVE))
import b31_live_eval as base  # noqa: E402  (sets the private-port environment)

WATCH = ("biplane", "building block")


def _iou_small(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    small = float(min(a.sum(), b.sum()))
    return inter / small if small else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out
    (out / "crops").mkdir(parents=True, exist_ok=True)

    import cv2
    import cascade
    assert str(base.WT) in cascade.__file__, cascade.__file__
    sys.path.insert(0, str(base.WT / "scripts"))
    from eval_detector import HIT_M, _real_objects
    from cascade.apps.demo import build_runtime, shutdown_runtime
    from cascade.config import load_demo_config
    from cascade.perception.colors import detection_color
    from cascade.perception.grounding import mask_to_points_cam, oriented_bbox
    from cascade.sim.bridge_client import BridgeClient
    from cascade.sim.truth import TruthPoseReader
    from cascade.types import transform_points

    cfg = load_demo_config(cameras=["isaac", "isaac_side"], arm="isaac", llm="mock")
    cfg._data.setdefault("stream", {})["mode"] = "off"
    base._private_ports(cfg._data)
    bad = [(p, v) for p, v in base._ports(cfg._data) if v in base.SHARED_PORTS]
    if bad:
        print(f"[!] refusing: shared ports resolved {bad}", file=sys.stderr)
        return 3
    client = BridgeClient(port=base.BRIDGE_PORT)
    client.connect()
    base.set_twins(client, None)  # default scene: green_cube green, props reset
    run_dir = LIVE / "runs" / f"b32_diag_{int(time.time())}"
    rt, arm = build_runtime(cfg, run_dir, view=False, serve=False)
    rows, saved = [], {}
    try:
        time.sleep(2.0)
        real = _real_objects(TruthPoseReader(client).all_poses())
        w = rt.watcher
        for rnd in range(args.rounds):
            for cam in w._cams:
                name = cam.stream.name
                frame = cam.stream.latest()
                if frame is None:
                    continue
                frame = cam.depth.ensure_depth(frame)
                T = frame.T_base_cam if frame.T_base_cam is not None else cam.extrinsics.cam_to_base()
                dets = w._detector.detect(frame, classes=w._classes)
                h, wid = frame.rgb.shape[:2]
                rm = getattr(frame, "robot_mask", None)
                pm = getattr(frame, "payload_mask", None)
                selfm = None
                if rm is not None:
                    selfm = np.asarray(rm, bool).copy()
                    if pm is not None:
                        selfm &= ~np.asarray(pm, bool)
                masks = []
                for d in dets:
                    if d.mask is not None:
                        m = np.asarray(d.mask, bool)
                    else:
                        m = np.zeros((h, wid), bool)
                        x0, y0, x1, y1 = np.asarray(d.bbox).astype(int)
                        m[max(y0, 0):min(y1, h), max(x0, 0):min(x1, wid)] = True
                    masks.append(m)
                for i, (d, m) in enumerate(zip(dets, masks)):
                    px = int(m.sum())
                    row = {"round": rnd, "camera": name, "i": i, "label": d.label,
                           "conf": round(float(d.conf), 3),
                           "bbox": [int(v) for v in np.asarray(d.bbox).tolist()],
                           "has_mask": d.mask is not None, "mask_px": px,
                           "color": detection_color(frame.rgb, d),
                           "robot_mask_present": selfm is not None}
                    if selfm is not None and selfm.shape == m.shape:
                        rpx = int((m & selfm).sum())
                        row["robot_px"] = rpx
                        row["robot_frac"] = round(rpx / px, 3) if px else None
                    row["overlaps"] = [
                        {"j": j, "label": dets[j].label, "iou_small": round(_iou_small(m, masks[j]), 3)}
                        for j in range(len(dets)) if j != i and _iou_small(m, masks[j]) > 0.05]
                    pts = mask_to_points_cam(frame, m)
                    row["n_points"] = int(pts.shape[0])
                    if pts.shape[0] >= 10:
                        pb = transform_points(T, pts)
                        c, e, _ = oriented_bbox(pb)
                        row["center"] = [round(float(v), 4) for v in c]
                        row["extents"] = [round(float(v), 4) for v in e]
                        row["top_z"] = round(float(pb[:, 2].max()), 4)
                        row["workspace"] = w._workspace.reject(c, e, mask_frac=px / m.size) or "accept"
                        n, dist = min(((k, float(np.linalg.norm(c - t))) for k, t in real),
                                      key=lambda kv: kv[1])
                        row["nearest"] = n
                        row["nearest_m"] = round(dist, 4)
                        row["hit"] = dist <= HIT_M
                    rows.append(row)
                    flag = (d.label in WATCH) or (row.get("workspace") == "accept" and not row.get("hit"))
                    key = (name, d.label)
                    if flag and saved.get(key, 0) < 3:
                        saved[key] = saved.get(key, 0) + 1
                        img = frame.rgb.copy()
                        if selfm is not None and selfm.shape == m.shape:
                            img[selfm] = (0.5 * img[selfm] + 0.5 * np.array([0, 0, 255])).astype(img.dtype)
                        img[m] = (0.4 * img[m] + 0.6 * np.array([0, 255, 0])).astype(img.dtype)
                        x0, y0, x1, y1 = row["bbox"]
                        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 0), 2)
                        cv2.putText(img, f"{d.label} {d.conf:.2f} {row['color']}", (max(x0, 0), max(y0 - 6, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        fn = out / "crops" / f"r{rnd:02d}_{name}_{i}_{d.label.replace(' ', '_')}.png"
                        cv2.imwrite(str(fn), img)
                        row["image"] = fn.name
                if rnd == 0:
                    img = frame.rgb.copy()
                    if selfm is not None:
                        img[selfm] = (0.5 * img[selfm] + 0.5 * np.array([0, 0, 255])).astype(img.dtype)
                    for d in dets:
                        x0, y0, x1, y1 = np.asarray(d.bbox).astype(int)
                        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 0), 2)
                        cv2.putText(img, d.label, (max(x0, 0), max(y0 - 6, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.imwrite(str(out / f"frame0_{name}.png"), img)
            time.sleep(0.5)
        beliefs = [{"label": b.label, "color": b.color, "aliases": sorted(b.aliases),
                    "position": [round(float(v), 4) for v in b.position], "observations": b.observations}
                   for b in rt.beliefs.all()]
    finally:
        shutdown_runtime(rt, arm)
    (out / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (out / "beliefs.json").write_text(json.dumps(beliefs, indent=1))
    # Summary per (camera, label)
    summ = {}
    for r in rows:
        k = f"{r['camera']}|{r['label']}"
        s = summ.setdefault(k, {"n": 0, "accepted": 0, "hits": 0, "robot_frac": [], "nearest_m": [], "z": [],
                                "colors": {}})
        s["n"] += 1
        s["accepted"] += r.get("workspace") == "accept"
        s["hits"] += bool(r.get("hit"))
        if r.get("robot_frac") is not None:
            s["robot_frac"].append(r["robot_frac"])
        if r.get("nearest_m") is not None:
            s["nearest_m"].append(r["nearest_m"])
            s["z"].append(r["center"][2])
        s["colors"][str(r["color"])] = s["colors"].get(str(r["color"]), 0) + 1
    for k, s in sorted(summ.items()):
        rf, nm, z = s.pop("robot_frac"), s.pop("nearest_m"), s.pop("z")
        s["robot_frac_min_med_max"] = [min(rf), float(np.median(rf)), max(rf)] if rf else None
        s["nearest_m_min_max"] = [min(nm), max(nm)] if nm else None
        s["z_min_max"] = [min(z), max(z)] if z else None
        print(k.ljust(34), json.dumps(s))
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print("beliefs:", json.dumps(beliefs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
