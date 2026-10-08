"""B32b live A/B on Isaac (PhysX): per-camera colour identity in the belief store.

UNRUN by the child (CPU-only rules): written for Hermes (parent) to run on GPU.
Adapted from ../../b32-fusion-self-mask/live/b32_live_ab.py (same scene, same
scoring, same private ports), on THIS worktree's code:

    python b32b_live_ab.py --per-camera-colour on|off --frames 12 --json out.json

Default 3-prop bare scene (green_cube recoloured green, props reset), YOLOE
prompt-free, cameras isaac + isaac_side, the real build_runtime ->
WorldWatcher + get_observation path, PhysX truth, scored by
`scripts/eval_detector.evaluate`. The arm under test is the ONLY config
difference: `memory.per_camera_colour` (true = B32b, false = the one-name
rule). Target: final belief count 4 -> 3 with precision unchanged (1.0 with
the B32a self-mask gate on, which is the default).

Extra output for B32b: every final belief's per-camera names
(`source_colors`) and, for every pair of beliefs nearest the same prop, the
2-98 % box IoU of their remembered clouds (`memory.beliefs._cloud_box`): if
the bin is still two beliefs with `on`, that number says whether the live
views overlap less than `memory.neighbour_colour_iou` (0.75) or the names
were not neighbours.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

ITEM = Path(__file__).resolve().parent.parent
WT = ITEM / "cascade"
LIVE = ITEM / "live"
MODELS = ITEM.parent / "same-colour-beliefs" / "live" / "models"
BRIDGE_PORT = 18521
os.environ.update({
    "CASCADE_BRIDGE_PORT": str(BRIDGE_PORT),
    "CASCADE_GRASPGENX_PORT": "18522",
    "CASCADE_OCCUPANCY_PORT": "18523",
    "CASCADE_BELIEFS": "0",
    "CASCADE_GRASP_MEMORY_PATH": str(LIVE / "runs" / "grasp_memory.json"),
    "CASCADE_ENVELOPE_PATH": str(LIVE / "runs" / "envelope.json"),
    "CASCADE_DETECTOR_MODEL": str(MODELS / "yoloe-11s-seg.pt"),
    "YOLO_OFFLINE": "True",
    "ULTRALYTICS_OFFLINE": "True",
})
sys.path.insert(0, str(WT / "scripts"))
SHARED_PORTS = {8611, 8080, 5556, 5557, 18789, 18790}
GREEN = (0.10, 0.75, 0.20)


def _ports(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            p = f"{path}.{k}" if path else str(k)
            if str(k).endswith("port") and isinstance(v, (int, str)) and str(v).isdigit():
                yield p, int(v)
            else:
                yield from _ports(v, p)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _ports(v, f"{path}[{i}]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-camera-colour", choices=("on", "off"), required=True)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--json", type=Path, required=True)
    args = ap.parse_args()
    on = args.per_camera_colour == "on"

    import cascade
    assert str(WT) in cascade.__file__, cascade.__file__
    from eval_detector import HIT_M, _real_objects, evaluate
    from cascade.apps.demo import build_runtime, shutdown_runtime
    from cascade.config import load_demo_config
    from cascade.memory.beliefs import _box_iou, _cloud_box
    from cascade.sim.bridge_client import BridgeClient
    from cascade.sim.truth import TruthPoseReader

    cfg = load_demo_config(cameras=["isaac", "isaac_side"], arm="isaac", llm="mock")
    cfg._data.setdefault("stream", {})["mode"] = "off"
    cfg._data.setdefault("memory", {})["per_camera_colour"] = on
    ports = sorted(set(_ports(cfg._data)))  # B34 (main e7b32ac) applies CASCADE_*_PORT
    bad = [(p, v) for p, v in ports if v in SHARED_PORTS]
    if bad:
        print(f"[!] refusing: shared ports resolved {bad}", file=sys.stderr)
        return 3
    client = BridgeClient(port=BRIDGE_PORT)
    client.connect()
    resp = client.request({"op": "exec", "code": (
        "from pxr import UsdGeom, Gf\n"
        "_m = UsdGeom.Mesh(stage.GetPrimAtPath('/World_Props/green_cube'))\n"
        f"_m.GetDisplayColorAttr().Set([Gf.Vec3f({GREEN[0]}, {GREEN[1]}, {GREEN[2]})])\n"
        "print('recoloured')\n")}, timeout_s=40)
    assert resp.get("ok"), resp
    client.request({"op": "reset_props"}, timeout_s=60)
    run_dir = LIVE / "runs" / f"colour_{args.per_camera_colour}_{int(time.time())}"
    rt, arm = build_runtime(cfg, run_dir, view=False, serve=False)
    try:
        assert rt.beliefs._per_camera_colour is on, "the A/B switch did not reach the store"
        assert rt.watcher._workspace.self_mask, "B32a self-mask gate must be on in both arms"
        time.sleep(2.0)  # let the watcher warm both cameras
        truth = TruthPoseReader(client)
        report = evaluate(rt, truth, args.frames)
        real = _real_objects(truth.all_poses())
        final = rt.beliefs.all()
        beliefs = []
        for b in final:
            p = [float(v) for v in b.position][:3]
            n, d = min(((k, math.dist(p, [float(x) for x in t])) for k, t in real), key=lambda kv: kv[1])
            beliefs.append({"label": b.label, "color": b.color, "source_colors": dict(b.source_colors),
                            "aliases": sorted(b.aliases), "position": [round(v, 4) for v in p],
                            "observations": b.observations, "nearest": n, "nearest_m": round(d, 4),
                            "hit": d <= HIT_M})
        pairs = []
        for (i, a), (j, c) in itertools.combinations(enumerate(final), 2):
            if beliefs[i]["nearest"] != beliefs[j]["nearest"]:
                continue
            ba, bc = _cloud_box(a.points), _cloud_box(c.points)
            pairs.append({"prop": beliefs[i]["nearest"], "colors": [a.color, c.color],
                          "box_iou": None if ba is None or bc is None else round(_box_iou(ba, bc), 3)})
        report.update({"per_camera_colour": args.per_camera_colour, "ports": ports,
                       "beliefs_final": beliefs, "same_prop_pairs": pairs,
                       "source_head": os.popen(f"git -C {WT} rev-parse --short HEAD").read().strip(),
                       "dirty": bool(os.popen(f"git -C {WT} status --porcelain -- src configs").read().strip())})
    finally:
        shutdown_runtime(rt, arm)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("per_camera_colour", "counts_per_frame", "precision",
                                             "recall", "phantom_rate", "localization_err_m")}))
    print(json.dumps([(b["color"], b["source_colors"], b["label"], b["nearest"], b["nearest_m"])
                      for b in beliefs]))
    print(json.dumps(pairs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
