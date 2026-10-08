"""Offline: compare identity-overlap metrics on the LIVE decision clouds
(results_samples/*.npz, logged at decision time by b32b_live_diag.py --iou 1.0).

same  = bare-scene decisions (the bin seen by isaac_side vs the bin belief of isaac)
diff  = adversarial decisions (a yellow prop in / next to the bin vs the orange bin belief)
"""
import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent / "results_samples"


def box(p, q=2.0, zmin=None, zrel=None):
    if zmin is not None:
        p = p[p[:, 2] > zmin]
    if zrel is not None:
        p = p[p[:, 2] > np.percentile(p[:, 2], q) + zrel]
    if p.shape[0] < 10:
        return None
    return np.percentile(p, q, axis=0), np.percentile(p, 100 - q, axis=0)


def iou(a, b, axes=(0, 1, 2)):
    if a is None or b is None:
        return float("nan")
    ax = list(axes)
    lo, hi = np.maximum(a[0], b[0])[ax], np.minimum(a[1], b[1])[ax]
    i = float(np.prod(np.clip(hi - lo, 0, None)))
    va, vb = float(np.prod((a[1] - a[0])[ax])), float(np.prod((b[1] - b[0])[ax]))
    return i / (va + vb - i)


METRICS = {
    "iou3d q2 (shipped)": lambda a, b: iou(box(a), box(b)),
    "iou3d q5": lambda a, b: iou(box(a, 5), box(b, 5)),
    "iou3d q2, z>1cm": lambda a, b: iou(box(a, zmin=0.01), box(b, zmin=0.01)),
    "iou3d q2, z>2cm": lambda a, b: iou(box(a, zmin=0.02), box(b, zmin=0.02)),
    "iouXY q2, z>1cm": lambda a, b: iou(box(a, zmin=0.01), box(b, zmin=0.01), (0, 1)),
    "iou3d q2, z>own low+1cm": lambda a, b: iou(box(a, zrel=0.01), box(b, zrel=0.01)),
    "iou3d q2, z>own low+2cm": lambda a, b: iou(box(a, zrel=0.02), box(b, zrel=0.02)),
}

rows = {"same": [], "diff": []}
for f in sorted(glob.glob(str(HERE / "*.json"))):
    r = json.load(open(f))
    z = np.load(f[:-5] + ".npz")
    for d in r["neighbour_decisions"]:
        a, b = (z[f"c{i}"] for i in d["clouds"])
        if a.shape[0] < 10 or b.shape[0] < 10:
            continue
        # label by GEOMETRY, not by scene: the belief is always the orange bin;
        # an observation whose robust XY box spans > 10 cm is a view of the bin
        # (same object), a <= 10 cm one is the 5 cm prop (different object)
        ob = box(a)
        kind = "same" if float(np.max((ob[1] - ob[0])[:2])) > 0.10 else "diff"
        rows[kind].append({m: fn(a, b) for m, fn in METRICS.items()} | {"scene": r["scene"],
                          "n": (a.shape[0], b.shape[0]),
                          "a_table_frac": float((a[:, 2] <= 0.01).mean()),
                          "b_table_frac": float((b[:, 2] <= 0.01).mean())})

for m in METRICS:
    s = np.array([r[m] for r in rows["same"]])
    d = np.array([r[m] for r in rows["diff"]])
    print(f"{m:22s} same n={len(s)} min {s.min():.3f} p10 {np.percentile(s, 10):.3f} med {np.median(s):.3f} | "
          f"diff n={len(d)} max {d.max():.3f} | gap {s.min() - d.max():+.3f}")
print("table-height fraction (z<=1cm), same: obs med %.2f belief med %.2f" % (
    np.median([r["a_table_frac"] for r in rows["same"]]), np.median([r["b_table_frac"] for r in rows["same"]])))
for r in rows["diff"]:
    print("diff", r["scene"], {k: round(v, 3) for k, v in r.items() if k in METRICS})
