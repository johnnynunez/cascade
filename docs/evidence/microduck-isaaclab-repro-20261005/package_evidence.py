#!/usr/bin/env python
"""Package the Isaac Lab reproduction for the CASCADE evidence tree.

Writes, into the evidence directory given as argv[1]:
  raw-logs-manifest.json   absolute path, size and SHA-256 of every raw per-step log (not committed: 45 MB),
                           with the episode kinds and step counts each one holds
  trajectories.csv.gz      compact per-step trajectory of every episode (no 61-dim observations / 14 actions)
"""
import csv
import glob
import gzip
import hashlib
import json
import math
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
DST = sys.argv[1]
os.makedirs(DST, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def yaw_xyzw(q):
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def tilt_xyzw(q):
    x, y, z, w = q
    return math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y))))


manifest = {"note": "Raw per-step logs of lab_microduck_walk_test.py (obs 61, actions 14, root pose/velocity, joint q/dq per policy step). Kept outside git (45 MB); identified by SHA-256.", "files": []}
rows = 0
with gzip.open(os.path.join(DST, "trajectories.csv.gz"), "wt", newline="") as gz:
    w = csv.writer(gz)
    w.writerow(["run", "policy", "task", "variant", "run_seed", "kind", "trial", "i", "phase", "t_s", "cmd_vx", "pos_x", "pos_y", "pos_z",
                "yaw_rad", "tilt_rad", "vw_x", "vw_y", "vb_x", "vb_y", "wz_b", "terminated"])
    for f in sorted(glob.glob(os.path.join(SRC, "results", "*.json"))):
        name = os.path.basename(f)
        if "probe" in name:
            continue
        d = json.load(open(f))
        entry = {"file": os.path.abspath(f), "bytes": os.path.getsize(f), "sha256": sha256(f),
                 "policy": d["policy"]["stem"], "task": d["cfg"]["task"], "variant": d["cfg"]["variant"], "seed": d["cfg"]["seed"],
                 "episodes": [{"kind": e["kind"], "trial": e.get("trial"), "steps": len(e["log"])} for e in d["episodes"]]}
        if name.startswith("OLD_"):
            entry["note"] = "first run; stored the xyzw quaternion under the key quat_wxyz, in-run yaw/tilt summaries wrong; superseded by flat_velocity_flat_lab_play_seed0.json; excluded from the tables"
        manifest["files"].append(entry)
        if name.startswith("OLD_"):
            continue
        run = name[:-5]
        for e in d["episodes"]:
            for r in e["log"]:
                q = r.get("quat_xyzw", r.get("quat_wxyz"))
                w.writerow([run, d["policy"]["stem"], "flat" if "Flat" in d["cfg"]["task"] else "rough", d["cfg"]["variant"], d["cfg"]["seed"],
                            e["kind"], e.get("trial"), r["i"], r["phase"], f"{r['t']:.2f}", r["cmd"][0],
                            f"{r['pos'][0]:.5f}", f"{r['pos'][1]:.5f}", f"{r['pos'][2]:.5f}", f"{yaw_xyzw(q):.5f}", f"{tilt_xyzw(q):.5f}",
                            f"{r['lin_vel_w'][0]:.4f}", f"{r['lin_vel_w'][1]:.4f}", f"{r['lin_vel_b'][0]:.4f}", f"{r['lin_vel_b'][1]:.4f}",
                            f"{r['ang_vel_b'][2]:.4f}", int(r["terminated"])])
                rows += 1
json.dump(manifest, open(os.path.join(DST, "raw-logs-manifest.json"), "w"), indent=1)
print("rows", rows, "files", len(manifest["files"]))
