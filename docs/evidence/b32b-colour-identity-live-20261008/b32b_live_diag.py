"""B32b live diagnosis on Isaac (PhysX): the neighbour-colour IoU at decision time,
and the adversarial scenes (a yellow prop inside / next to the orange bin).

    python b32b_live_diag.py --scene bare|in_bin_centre|in_bin_corner|next_to_bin \
        --per-camera-colour on|off --frames 12 --json out.json

Same runtime / cameras / detector / scoring / private ports as b32b_live_ab.py.
Every call of `memory.beliefs._box_iou` (only reached on the neighbour-colour
path of `_identity_ok`) is logged with both boxes. For the non-bare scenes
`green_cube` is recoloured YELLOW and placed; it is restored to green at its
default pose (reset_props) afterwards, so the bare A/B is unaffected.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b32b_live_ab as base  # noqa: E402  (sets the private-port env before cascade imports)

YELLOW = (0.90, 0.80, 0.10)
SCENES = {
    "bare": None,
    "in_bin_centre": (0.18, -0.17, 0.04),
    "in_bin_corner": (0.15, -0.20, 0.04),
    "next_to_bin": (0.30, -0.17, 0.04),
}


def _colour(client, rgb):
    resp = client.request({"op": "exec", "code": (
        "from pxr import UsdGeom, Gf\n"
        "_m = UsdGeom.Mesh(stage.GetPrimAtPath('/World_Props/green_cube'))\n"
        f"_m.GetDisplayColorAttr().Set([Gf.Vec3f({rgb[0]}, {rgb[1]}, {rgb[2]})])\n"
        "print('recoloured')\n")}, timeout_s=40)
    assert resp.get("ok"), resp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", choices=sorted(SCENES), required=True)
    ap.add_argument("--per-camera-colour", choices=("on", "off"), required=True)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--iou", type=float, default=None,
                    help="memory.neighbour_colour_iou for this run (1.0 = refuse every neighbour fusion, "
                         "so EVERY side frame is logged as a decision sample)")
    ap.add_argument("--json", type=Path, required=True)
    args = ap.parse_args()
    on = args.per_camera_colour == "on"

    import numpy as np
    import cascade
    assert str(base.WT) in cascade.__file__, cascade.__file__
    import cascade.memory.beliefs as bm
    from eval_detector import HIT_M, _real_objects, evaluate
    from cascade.apps.demo import build_runtime, shutdown_runtime
    from cascade.config import load_demo_config
    from cascade.sim.bridge_client import BridgeClient
    from cascade.sim.truth import TruthPoseReader

    decisions = []
    clouds: list[np.ndarray] = []
    real_iou, real_box = bm._box_iou, bm._cloud_box
    box_src: dict[int, np.ndarray] = {}

    def logged_box(points):
        box = real_box(points)
        if box is not None:
            box_src[id(box)] = np.asarray(points, dtype=np.float32).reshape(-1, 3).copy()
        return box

    def logged_iou(a, b):
        v = real_iou(a, b)
        k = len(clouds)
        clouds.extend([box_src.get(id(a)), box_src.get(id(b))])
        decisions.append({"t": round(time.monotonic(), 3), "iou": round(v, 4), "clouds": [k, k + 1],
                          "obs_box": [np.round(a[0], 4).tolist(), np.round(a[1], 4).tolist()],
                          "belief_box": [np.round(b[0], 4).tolist(), np.round(b[1], 4).tolist()]})
        box_src.clear()
        return v

    bm._box_iou = logged_iou
    bm._cloud_box = logged_box

    assignments = []
    real_update_frame = bm.BeliefStore.update_frame

    def logged_update_frame(self, observations, t=None):
        obs = list(observations)
        out = real_update_frame(self, obs, t=t)
        for o, b in zip(obs, out):
            assignments.append({"t": t, "source": o.source, "color": o.color, "label": o.label,
                                "obs_extent_max": None if o.extent is None else round(float(np.max(o.extent)), 3),
                                "obs_pos": [round(float(v), 3) for v in o.position],
                                "belief": id(b), "belief_color": b.color,
                                "belief_sources": dict(b.source_colors)})
        return out

    bm.BeliefStore.update_frame = logged_update_frame

    cfg = load_demo_config(cameras=["isaac", "isaac_side"], arm="isaac", llm="mock")
    cfg._data.setdefault("stream", {})["mode"] = "off"
    cfg._data.setdefault("memory", {})["per_camera_colour"] = on
    if args.iou is not None:
        cfg._data["memory"]["neighbour_colour_iou"] = args.iou
    ports = sorted(set(base._ports(cfg._data)))
    bad = [(p, v) for p, v in ports if v in base.SHARED_PORTS]
    if bad:
        print(f"[!] refusing: shared ports resolved {bad}", file=sys.stderr)
        return 3
    client = BridgeClient(port=base.BRIDGE_PORT)
    client.connect()
    _colour(client, base.GREEN)
    client.request({"op": "reset_props"}, timeout_s=60)
    pos = SCENES[args.scene]
    if pos is not None:
        _colour(client, YELLOW)
        resp = client.request({"op": "place_prop", "name": "green_cube", "pos": list(pos)}, timeout_s=60)
        assert resp.get("ok"), resp
        time.sleep(1.0)
    run_dir = base.LIVE / "runs" / f"diag_{args.scene}_{args.per_camera_colour}_{int(time.time())}"
    rt, arm = build_runtime(cfg, run_dir, view=False, serve=False)
    try:
        assert rt.beliefs._per_camera_colour is on
        assert rt.watcher._workspace.self_mask
        time.sleep(2.0)
        truth = TruthPoseReader(client)
        report = evaluate(rt, truth, args.frames)
        real = _real_objects(truth.all_poses())
        beliefs = []
        for b in rt.beliefs.all():
            p = [float(v) for v in b.position][:3]
            n, d = min(((k, math.dist(p, [float(x) for x in t])) for k, t in real), key=lambda kv: kv[1])
            box = bm._cloud_box(b.points)
            beliefs.append({"label": b.label, "color": b.color, "source_colors": dict(b.source_colors),
                            "aliases": sorted(b.aliases), "position": [round(v, 4) for v in p],
                            "extent": None if b.extent is None else [round(float(v), 4) for v in b.extent],
                            "box": None if box is None else [np.round(box[0], 4).tolist(), np.round(box[1], 4).tolist()],
                            "observations": b.observations, "nearest": n, "nearest_m": round(d, 4),
                            "hit": d <= HIT_M})
        report.update({"scene": args.scene, "prop_pos": pos, "per_camera_colour": args.per_camera_colour,
                       "ports": ports, "beliefs_final": beliefs, "neighbour_decisions": decisions,
                       "assignments": assignments,
                       "threshold": rt.beliefs._neighbour_colour_iou,
                       "truth": {k: [round(float(x), 4) for x in t] for k, t in real},
                       "source_head": os.popen(f"git -C {base.WT} rev-parse --short HEAD").read().strip()})
    finally:
        shutdown_runtime(rt, arm)
        _colour(client, base.GREEN)
        client.request({"op": "reset_props"}, timeout_s=60)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=1))
    if clouds:
        np.savez_compressed(args.json.with_suffix(".npz"),
                            **{f"c{i}": (c if c is not None else np.zeros((0, 3), np.float32))
                               for i, c in enumerate(clouds)})
    ious = sorted(d["iou"] for d in decisions)
    print(json.dumps({k: report[k] for k in ("scene", "per_camera_colour", "counts_per_frame", "precision",
                                             "recall", "phantom_rate")}))
    print("neighbour IoUs:", ious)
    print(json.dumps([(b["color"], b["source_colors"], b["label"], b["nearest"], b["nearest_m"], b["extent"])
                      for b in beliefs]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
