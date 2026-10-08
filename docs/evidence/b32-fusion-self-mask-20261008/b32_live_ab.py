"""B32 live A/B on Isaac (PhysX): the render self-mask gate in semantic fusion.

One run = one fresh process = one fresh runtime/belief store on the B32
worktree's code (`main` + the gate):

    python b32_live_ab.py --self-mask on|off --frames 12 --json out.json

Default 3-prop bare scene (green_cube recoloured back to green, props reset),
YOLOE prompt-free, cameras isaac + isaac_side, the real build_runtime ->
WorldWatcher + get_observation path, PhysX truth, scored by
`scripts/eval_detector.evaluate` (the metric behind
benchmark/results/perception_open_vocab.json). Private ports only (bridge
18521); refuses to run if any resolved port is a shared one.
"""
from __future__ import annotations

import argparse
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


def _private_ports(node, path=""):
    """c5012e7's load_demo_config does not apply CASCADE_BRIDGE_PORT /
    CASCADE_OCCUPANCY_PORT: rewrite every port of the resolved config."""
    if isinstance(node, dict):
        for k, v in list(node.items()):
            p = f"{path}.{k}" if path else str(k)
            if str(k) == "bridge_port":
                node[k] = BRIDGE_PORT
            elif str(k) == "port" and "graspgenx" in path:
                node[k] = 18522
            elif str(k) == "port" and "occupancy" in path:
                node[k] = 18523
            else:
                _private_ports(v, p)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _private_ports(v, f"{path}[{i}]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-mask", choices=("on", "off"), required=True)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--json", type=Path, required=True)
    args = ap.parse_args()

    import cascade
    assert str(WT) in cascade.__file__, cascade.__file__
    from eval_detector import HIT_M, _real_objects, evaluate
    from cascade.apps.demo import build_runtime, shutdown_runtime
    from cascade.config import load_demo_config
    from cascade.sim.bridge_client import BridgeClient
    from cascade.sim.truth import TruthPoseReader

    cfg = load_demo_config(cameras=["isaac", "isaac_side"], arm="isaac", llm="mock")
    cfg._data.setdefault("stream", {})["mode"] = "off"
    cfg._data.setdefault("workspace_filter", {})["self_mask"] = args.self_mask == "on"
    _private_ports(cfg._data)
    ports = sorted(set(_ports(cfg._data)))
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
    run_dir = LIVE / "runs" / f"selfmask_{args.self_mask}_{int(time.time())}"
    rt, arm = build_runtime(cfg, run_dir, view=False, serve=False)
    try:
        gate = rt.watcher._workspace
        assert gate.self_mask is (args.self_mask == "on") and rt._workspace.self_mask is gate.self_mask
        time.sleep(2.0)  # let the watcher warm both cameras
        truth = TruthPoseReader(client)
        report = evaluate(rt, truth, args.frames)
        real = _real_objects(truth.all_poses())
        beliefs = []
        for b in rt.beliefs.all():
            p = [float(v) for v in b.position][:3]
            n, d = min(((k, math.dist(p, [float(x) for x in t])) for k, t in real), key=lambda kv: kv[1])
            beliefs.append({"label": b.label, "color": b.color, "aliases": sorted(b.aliases),
                            "position": [round(v, 4) for v in p], "observations": b.observations,
                            "nearest": n, "nearest_m": round(d, 4), "hit": d <= HIT_M})
        report.update({"self_mask": args.self_mask, "ports": ports, "beliefs_final": beliefs,
                       "source_head": os.popen(f"git -C {WT} rev-parse --short HEAD").read().strip(),
                       "dirty": bool(os.popen(f"git -C {WT} status --porcelain -- src configs").read().strip())})
    finally:
        shutdown_runtime(rt, arm)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("self_mask", "counts_per_frame", "precision", "recall",
                                             "phantom_rate", "localization_err_m")}))
    print(json.dumps([(b["color"], b["label"], b["nearest"], b["nearest_m"]) for b in beliefs]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
