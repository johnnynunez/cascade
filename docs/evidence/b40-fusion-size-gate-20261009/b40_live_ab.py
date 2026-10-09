"""B40 live A/B on Isaac 6.2 (PhysX): the size gate in the belief store's fusion.

NOT YET RUN. Written by the CPU-only child for Hermes (parent) to run on GPU 0.
Modelled on ../b32b-colour-identity-live-20261008/b32b_live_ab.py and
b32b_live_diag.py (same bare scene, scoring, private-port discipline). Copy it
to <item>/live/ and run from the directory that holds mobileclip_blt.ts:

    python b40_live_ab.py --size-gate on|off --frames 12 --json out.json

Scene (the B32b "in_bin_centre" adversarial scene, where the defect was seen):
the bare reBot scene with `green_cube` (5 x 5 x 8 cm) recoloured YELLOW
(0.90, 0.80, 0.10) and placed INSIDE the orange bin at (0.18, -0.17, 0.04).
The side camera `isaac_side` names the bin AND the cube "yellow".

Arm = `memory.size_gate` ONLY (true = B40, false = the store before it, byte
for byte); `memory.per_camera_colour` stays at its default (true) in both arms
and the B32a self-mask gate must be on. Both are asserted.

Logged per run (JSON):
- every observation -> belief assignment of every fused frame (watcher and
  get_observation paths: `BeliefStore.update_frame` is wrapped at class
  level), with the observation's source camera, colour, label, OBB max and
  robust diameter (`memory.beliefs._view_diameter`, measured in the wrapper so
  both arms are measured the same way), and the belief's first-view diameter;
- `container_views_in_prop_beliefs`: views >= 0.12 m across that went into a
  belief whose first view was < 0.10 m across (the defect; B32b measured 3
  such views per run with the side camera, OBB 0.218-0.222 m, in both colour
  arms);
- `prop_views_in_container_beliefs`: the reverse (< 0.10 m view into a belief
  first seen >= 0.12 m) -- NOT addressed by B40, reported so a change in it
  is visible;
- final beliefs: label, colour, per-camera names, aliases, position, OBB,
  diameter, observations, nearest PhysX prop and distance;
- `scripts/eval_detector.evaluate` over --frames frames (precision, recall,
  phantoms, counts per frame, localisation error vs PhysX truth).

Expected if B40 works live: `on` -> 0 container views in prop beliefs, the bin
belief carries {isaac: orange, isaac_side: yellow}, the yellow cube belief
keeps a cube-sized cloud and no bin alias; `off` -> the B32b behaviour (about
3 per run). Precision/recall must not drop in `on`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ITEM = Path(__file__).resolve().parent.parent          # <item>/live/b40_live_ab.py
WT = ITEM / "cascade"
LIVE = ITEM / "live"
BRIDGE_PORT = 45250                                    # B40 port block 45200-45299
os.environ.update({
    "CASCADE_BRIDGE_PORT": str(BRIDGE_PORT),
    "CASCADE_GRASPGENX_PORT": "45251",
    "CASCADE_OCCUPANCY_PORT": "45252",
    "CASCADE_HUG_PORT": "45253",
    "CASCADE_BELIEFS": "0",
    "CASCADE_GRASP_MEMORY_PATH": str(LIVE / "runs" / "grasp_memory.json"),
    "CASCADE_ENVELOPE_PATH": str(LIVE / "runs" / "envelope.json"),
    "YOLO_OFFLINE": "True",
    "ULTRALYTICS_OFFLINE": "True",
})
sys.path.insert(0, str(WT / "scripts"))
SHARED_PORTS = {8611, 8080, 5556, 5557, 18789, 18790}
GREEN = (0.10, 0.75, 0.20)
YELLOW = (0.90, 0.80, 0.10)
IN_BIN = (0.18, -0.17, 0.04)
CONTAINER_M, PROP_M = 0.12, 0.10


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


def _colour(client, rgb):
    resp = client.request({"op": "exec", "code": (
        "from pxr import UsdGeom, Gf\n"
        "_m = UsdGeom.Mesh(stage.GetPrimAtPath('/World_Props/green_cube'))\n"
        f"_m.GetDisplayColorAttr().Set([Gf.Vec3f({rgb[0]}, {rgb[1]}, {rgb[2]})])\n"
        "print('recoloured')\n")}, timeout_s=40)
    assert resp.get("ok"), resp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size-gate", choices=("on", "off"), required=True)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--models", type=Path,
                    default=Path("/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261007/"
                                 "same-colour-beliefs/live/models"))
    ap.add_argument("--json", type=Path, required=True)
    args = ap.parse_args()
    on = args.size_gate == "on"
    assert (args.models / "yoloe-11s-seg.pt").exists(), args.models
    assert Path("mobileclip_blt.ts").exists(), "run from the directory holding mobileclip_blt.ts"
    os.environ["CASCADE_DETECTOR_MODEL"] = str(args.models / "yoloe-11s-seg.pt")

    import numpy as np
    import cascade
    assert str(WT) in cascade.__file__, cascade.__file__
    import cascade.memory.beliefs as bm
    from eval_detector import HIT_M, _real_objects, evaluate
    from cascade.apps.demo import build_runtime, shutdown_runtime
    from cascade.config import load_demo_config
    from cascade.sim.bridge_client import BridgeClient
    from cascade.sim.truth import TruthPoseReader

    def diameter(points):
        if points is None:
            return None
        d = bm._view_diameter(bm.BeliefStore._remembered_cloud(points))
        return None if d is None else round(float(d), 4)

    assignments, first_view = [], {}
    real_update_frame = bm.BeliefStore.update_frame

    def logged_update_frame(self, observations, t=None):
        obs = list(observations)
        out = real_update_frame(self, obs, t=t)
        for o, b in zip(obs, out):
            d = diameter(o.points)
            first_view.setdefault(id(b), {"label": o.label, "source": o.source, "diameter": d})
            assignments.append({
                "t": None if t is None else round(float(t), 3), "source": o.source, "color": o.color,
                "label": o.label, "obs_obb_max": None if o.extent is None else round(float(np.max(o.extent)), 4),
                "obs_diameter": d, "obs_pos": [round(float(v), 4) for v in o.position],
                "belief": id(b), "belief_label": b.label, "belief_color": b.color,
                "belief_sources": dict(b.source_colors), "belief_first_view": first_view[id(b)],
            })
        return out

    bm.BeliefStore.update_frame = logged_update_frame

    cfg = load_demo_config(cameras=["isaac", "isaac_side"], arm="isaac", llm="mock")
    cfg._data.setdefault("stream", {})["mode"] = "off"
    cfg._data.setdefault("memory", {})["size_gate"] = on
    ports = sorted(set(_ports(cfg._data)))
    bad = [(p, v) for p, v in ports if v in SHARED_PORTS]
    if bad:
        print(f"[!] refusing: shared ports resolved {bad}", file=sys.stderr)
        return 3
    client = BridgeClient(port=BRIDGE_PORT)
    client.connect()
    _colour(client, GREEN)
    client.request({"op": "reset_props"}, timeout_s=60)
    _colour(client, YELLOW)
    resp = client.request({"op": "place_prop", "name": "green_cube", "pos": list(IN_BIN)}, timeout_s=60)
    assert resp.get("ok"), resp
    time.sleep(1.0)
    run_dir = LIVE / "runs" / f"size_gate_{args.size_gate}_{int(time.time())}"
    rt, arm = build_runtime(cfg, run_dir, view=False, serve=False)
    try:
        assert rt.beliefs._size_gate is on, "the A/B switch did not reach the store"
        assert rt.beliefs._per_camera_colour is True, "B32b colour identity must be on in both arms"
        assert rt.watcher._workspace.self_mask, "B32a self-mask gate must be on in both arms"
        time.sleep(2.0)  # let the watcher warm both cameras
        truth = TruthPoseReader(client)
        report = evaluate(rt, truth, args.frames)
        real = _real_objects(truth.all_poses())
        beliefs = []
        for b in rt.beliefs.all():
            p = [float(v) for v in b.position][:3]
            n, d = min(((k, math.dist(p, [float(x) for x in t])) for k, t in real), key=lambda kv: kv[1])
            beliefs.append({"label": b.label, "color": b.color, "source_colors": dict(b.source_colors),
                            "aliases": sorted(b.aliases), "position": [round(v, 4) for v in p],
                            "obb": None if b.extent is None else [round(float(v), 4) for v in b.extent],
                            "diameter": diameter(b.points), "diameter_m": b.diameter_m,
                            "first_view": first_view.get(id(b)), "observations": b.observations,
                            "nearest": n, "nearest_m": round(d, 4), "hit": d <= HIT_M})
        big_into_small = [a for a in assignments if (a["obs_diameter"] or 0) >= CONTAINER_M
                          and (a["belief_first_view"]["diameter"] or 1) < PROP_M]
        small_into_big = [a for a in assignments if a["obs_diameter"] is not None
                          and a["obs_diameter"] < PROP_M
                          and (a["belief_first_view"]["diameter"] or 0) >= CONTAINER_M]
        report.update({
            "size_gate": args.size_gate, "ports": ports, "scene": {"green_cube": "yellow", "pos": IN_BIN},
            "beliefs_final": beliefs, "assignments": assignments,
            "container_views_in_prop_beliefs": len(big_into_small),
            "container_views_in_prop_beliefs_detail": big_into_small,
            "prop_views_in_container_beliefs": len(small_into_big),
            "truth": {k: [round(float(x), 4) for x in t] for k, t in real},
            "source_head": os.popen(f"git -C {WT} rev-parse --short HEAD").read().strip(),
            "dirty": bool(os.popen(f"git -C {WT} status --porcelain -- src configs").read().strip()),
        })
    finally:
        shutdown_runtime(rt, arm)
        _colour(client, GREEN)
        client.request({"op": "reset_props"}, timeout_s=60)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("size_gate", "counts_per_frame", "precision", "recall",
                                             "phantom_rate", "localization_err_m",
                                             "container_views_in_prop_beliefs",
                                             "prop_views_in_container_beliefs")}))
    print(json.dumps([(b["label"], b["color"], b["source_colors"], b["aliases"], b["diameter"], b["nearest"],
                       b["nearest_m"]) for b in beliefs]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
