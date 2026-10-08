"""B31 live A/B on Isaac (PhysX): does per-frame instance association change the
open-vocabulary phantom result, and does it keep same-colour twins apart on the
real demo perception stack (YOLOE prompt-free + Isaac RGB-D + WorldWatcher)?

One run = one fresh process = one fresh runtime/belief store:

    python b31_live_eval.py --association instance|legacy [--twin-sep 0.06] \
        --frames 12 --json out.json

`--twin-sep S` first recolours `green_cube` to the pink cube's exact colour and
places it S metres (+y) from `pink_cube`, i.e. two IDENTICAL props. Ground truth
is PhysX (`TruthPoseReader`), scored by `scripts/eval_detector.evaluate` (the
metric behind benchmark/results/perception_open_vocab.json) plus a twin-specific
count. Private ports only (bridge 18521); refuses to run if any resolved port is
a shared one.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

LIVE = Path(__file__).resolve().parent
WT = LIVE.parent / "cascade"
BRIDGE_PORT = 18521
os.environ.update({
    "CASCADE_BRIDGE_PORT": str(BRIDGE_PORT),
    "CASCADE_GRASPGENX_PORT": "18522",
    "CASCADE_OCCUPANCY_PORT": "18523",
    "CASCADE_BELIEFS": "0",                      # no restored beliefs from disk
    "CASCADE_GRASP_MEMORY_PATH": str(LIVE / "runs" / "grasp_memory.json"),
    "CASCADE_ENVELOPE_PATH": str(LIVE / "runs" / "envelope.json"),
    "CASCADE_DETECTOR_MODEL": str(LIVE / "models" / "yoloe-11s-seg.pt"),
    "YOLO_OFFLINE": "True",
    "ULTRALYTICS_OFFLINE": "True",
})
sys.path.insert(0, str(WT / "scripts"))
SHARED_PORTS = {8611, 8080, 5556, 5557, 18789, 18790}
PINK = (0.95, 0.30, 0.70)
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
    """c5012e7's load_demo_config does NOT apply CASCADE_BRIDGE_PORT /
    CASCADE_OCCUPANCY_PORT (only GraspGenXPlanner reads CASCADE_GRASPGENX_PORT),
    so the resolved config still names 8611/5556/5557: rewrite every port in
    the resolved config (incl. each arm's `resolved` view) to the private ones."""
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


def _exec(client, code: str) -> dict:
    resp = client.request({"op": "exec", "code": code}, timeout_s=40)
    if not resp.get("ok", False):
        raise RuntimeError(f"bridge exec failed: {resp}")
    return resp


def set_twins(client, sep: float | None) -> dict:
    """Recolour green_cube (pink, or back to green) and place it."""
    rgb = PINK if sep is not None else GREEN
    _exec(client, (
        "from pxr import UsdGeom, Gf\n"
        "_m = UsdGeom.Mesh(stage.GetPrimAtPath('/World_Props/green_cube'))\n"
        f"_m.GetDisplayColorAttr().Set([Gf.Vec3f({rgb[0]}, {rgb[1]}, {rgb[2]})])\n"
        "print('recoloured')\n"))
    if sep is None:
        return client.request({"op": "reset_props"}, timeout_s=60)
    pos = [0.17, 0.15 + float(sep), 0.04]
    return client.request({"op": "place_prop", "name": "green_cube", "pos": pos}, timeout_s=60)


def twin_counts(runtime, truth_reader, frames: int, settle_s: float = 0.25) -> dict:
    """Per frame: beliefs within 0.10 m of either twin, and each twin's error
    to its nearest belief (PhysX truth)."""
    import numpy as np

    poses = truth_reader.all_poses()
    twins = [np.asarray(poses[k], float)[:3] for k in ("pink_cube", "green_cube")]
    near, errs, labels, dets = [], [], [], []
    for _ in range(frames):
        obs = runtime.execute("get_observation", {})
        # detections of THIS frame (primary camera) that land on the pair:
        # 1 = the detector merged the twins, 2 = it saw two instances
        dets.append(sum(
            1 for o in (obs.get("objects_visible") or [])
            if o.get("position") is not None and min(
                float(np.linalg.norm(np.asarray(o["position"], float)[:2] - t[:2])) for t in twins) <= 0.10))
        bel = [np.asarray(b.position, float)[:3] for b in runtime.beliefs.all()]
        lab = [f"{b.color}/{b.label}" for b in runtime.beliefs.all()]
        close = [i for i, p in enumerate(bel)
                 if min(float(np.linalg.norm(p[:2] - t[:2])) for t in twins) <= 0.10]
        near.append(len(close))
        labels.append([lab[i] for i in close])
        errs.append([round(min((float(np.linalg.norm(bel[i][:2] - t[:2])) for i in close),
                               default=float("nan")), 4) for t in twins])
        time.sleep(settle_s)
    return {"twin_truth_xy": [[round(float(v), 4) for v in t[:2]] for t in twins],
            "detections_on_twins_per_frame": dets,
            "beliefs_near_twins_per_frame": near,
            "twin_xy_err_per_frame": errs,
            "labels_last_frame": labels[-1] if labels else []}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--association", choices=("instance", "legacy"), required=True)
    ap.add_argument("--twin-sep", type=float, default=None)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--restore", action="store_true",
                    help="recolour green_cube back and reset_props before measuring")
    args = ap.parse_args()

    import cascade
    assert str(WT) in cascade.__file__, cascade.__file__
    from eval_detector import evaluate
    from cascade.apps.demo import build_runtime, shutdown_runtime
    from cascade.config import load_demo_config
    from cascade.sim.bridge_client import BridgeClient
    from cascade.sim.truth import TruthPoseReader

    cfg = load_demo_config(cameras=["isaac", "isaac_side"], arm="isaac", llm="mock")
    cfg._data.setdefault("stream", {})["mode"] = "off"
    cfg._data.setdefault("memory", {})["instance_association"] = args.association == "instance"
    _private_ports(cfg._data)
    ports = sorted(set(_ports(cfg._data)))
    bad = [(p, v) for p, v in ports if v in SHARED_PORTS]
    if bad:
        print(f"[!] refusing: shared ports resolved {bad}", file=sys.stderr)
        return 3
    client = BridgeClient(port=BRIDGE_PORT)
    client.connect()
    placed = set_twins(client, args.twin_sep) if (args.twin_sep is not None or args.restore) else None
    run_dir = LIVE / "runs" / f"{args.association}_{args.twin_sep}_{int(time.time())}"
    rt, arm = build_runtime(cfg, run_dir, view=False, serve=False)
    try:
        assert rt.beliefs._instance_association is (args.association == "instance")
        time.sleep(2.0)  # let the watcher warm both cameras
        truth = TruthPoseReader(client)
        report = evaluate(rt, truth, args.frames)
        if args.twin_sep is not None:
            report["twins"] = twin_counts(rt, truth, max(4, args.frames // 2))
        report["association"] = args.association
        report["twin_sep_m"] = args.twin_sep
        report["placed"] = placed
        report["ports"] = ports
        report["beliefs_final"] = [
            {"label": b.label, "color": b.color, "aliases": sorted(b.aliases),
             "position": [round(float(v), 4) for v in b.position], "observations": b.observations}
            for b in rt.beliefs.all()]
    finally:
        shutdown_runtime(rt, arm)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("association", "twin_sep_m", "counts_per_frame",
                                             "precision", "recall", "phantom_rate",
                                             "localization_err_m")}))
    if "twins" in report:
        t = report["twins"]
        print(json.dumps({"dets_on_twins": t["detections_on_twins_per_frame"],
                          "near_twins": t["beliefs_near_twins_per_frame"],
                          "err_last": t["twin_xy_err_per_frame"][-1],
                          "labels": t["labels_last_frame"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
