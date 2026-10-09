"""B40 measurement (CPU only): how large are views of ONE object, and how much
larger is a container's view than the belief of a prop inside / next to it?

    python docs/evidence/b40-fusion-size-gate-20261009/measure_sizes.py \
        [--lab <cascade-lab>/HERMES_BACKLOG_20261007/b32b-colour-identity/live] \
        [--json sizes_summary.json]

Size = what the gate measures: `cascade.memory.beliefs._view_diameter` of the
cloud the store remembers (<= 384 points), i.e. the largest 2nd-98th
percentile span over 8 directions in the table plane, on the cloud without
its lowest centimetre. Two alternatives are reported for comparison: the max
x-y span of the robust axis-aligned box (`_cloud_box`) and the max OBB extent
(`extent[0]`, what `_fusion_radius` uses).

Sources:
- live: the B32b fixture (tests/fixtures/b32b_live_bin_clouds, 6 pairs) and,
  with --lab, every logged B32b decision cloud (results_samples/*.npz,
  results_b32c_adv/*.npz; Isaac 6.2 PhysX + YOLOE, both demo cameras), labelled
  by geometry as B32b did (robust x-y span > 10 cm = the bin);
- live OBB sizes of the defect itself, from the B32b assignment log
  (results_b32c_adv/assign_in_bin_centre_*.json: which belief every
  observation went to);
- ray-cast: the bare Isaac scene (tests/test_colour_identity_beliefs.py:
  calibrated poses, bridge optics, 320 x 180, exact masks) -- the bin, the
  bridge's 5 x 5 x 8 cm prop and a 5 cm cube at the live positions, both
  cameras, mask bleed 0-2 px, half / quarter of the mask hidden.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

import cv2  # noqa: E402

import test_colour_identity_beliefs as tcb  # noqa: E402
from cascade.memory.beliefs import (  # noqa: E402
    SIZE_GATE_EXCESS_M, SIZE_GATE_RATIO, BeliefStore, _cloud_box, _view_diameter,
)
from cascade.perception.grounding import oriented_bbox  # noqa: E402

FIX = REPO / "tests" / "fixtures" / "b32b_live_bin_clouds"
LIVE_PROP_XY = (0.1833, -0.1584)


def sizes(points) -> dict | None:
    p = BeliefStore._remembered_cloud(np.asarray(points, dtype=float))
    d = _view_diameter(p)
    if d is None:
        return None
    lo, hi = _cloud_box(p)
    _, e, _ = oriented_bbox(np.asarray(p, dtype=float))
    return {"diameter": d, "aa_xy": float(np.max((hi - lo)[:2])), "obb": float(np.max(e))}


def stats(v) -> dict:
    v = np.asarray(v, dtype=float)
    return {"n": int(v.size), "min": round(float(v.min()), 4), "median": round(float(np.median(v)), 4),
            "max": round(float(v.max()), 4)}


def live_views(lab: Path | None) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {"bin": [], "prop": []}
    rec = json.loads((FIX / "receipt.json").read_text())
    z = np.load(FIX / "clouds.npz")
    seen = set()
    for p in rec["pairs"]:
        i = p["pair"]
        seen.add((p["run"].split("/")[-1].replace(".json", ""), p["decision"]))
        out["bin"].append(sizes(z[f"p{i}_belief"]))
        out["bin" if p["kind"] == "same" else "prop"].append(sizes(z[f"p{i}_obs"]))
    if lab is None:
        return out
    for sub in ("results_samples", "results_b32c_adv"):
        for f in sorted(glob.glob(str(lab / sub / "*.json"))):
            npz = Path(f[:-5] + ".npz")
            if not npz.exists():
                continue
            r = json.loads(Path(f).read_text())
            zz = np.load(npz)
            for k, d in enumerate(r.get("neighbour_decisions", [])):
                if sub == "results_samples" and (Path(f).stem, k) in seen:
                    continue  # already counted from the fixture
                a, b = (zz[f"c{j}"] for j in d["clouds"])
                sa, sb = sizes(a), sizes(b)
                if sa is None or sb is None:
                    continue
                out["bin" if sa["aa_xy"] > 0.10 else "prop"].append(sa)  # B32b's geometric label
                out["bin"].append(sb)  # the belief was always the bin
    return out


def assignments(lab: Path | None) -> list[dict]:
    """The defect, live: the OBB size of every observation and where it went."""
    if lab is None:
        return []
    rows = []
    for f in sorted(glob.glob(str(lab / "results_b32c_adv" / "assign_*.json"))):
        r = json.loads(Path(f).read_text())
        first = {}
        for a in r["assignments"]:
            first.setdefault(a["belief"], a)
        for a in r["assignments"]:
            home = first[a["belief"]]
            rows.append({"run": Path(f).stem, "source": a["source"], "label": a["label"], "color": a["color"],
                         "obs_obb_max": a["obs_extent_max"], "belief_born_as": home["label"],
                         "belief_born_obb_max": home["obs_extent_max"]})
    return rows


def raycast() -> list[dict]:
    rows = []
    prop = [((LIVE_PROP_XY[0], LIVE_PROP_XY[1], 0.04), (0.05, 0.05, 0.08))]
    scenes = {
        "bin + prop in it (live position)": {"bin": tcb.BIN, "prop": prop},
        "bin + prop next to it": {"bin": tcb.BIN, "prop": [((0.30, -0.17, 0.04), (0.05, 0.05, 0.08))]},
        "prop bare (0.17, 0.15)": {"prop": [((0.17, 0.15, 0.04), (0.05, 0.05, 0.08))]},
        "5 cm cube in the bin corner": {"bin": tcb.BIN, "cube5": [((0.145, -0.205, 0.025), (0.05,) * 3)]},
    }
    for scene, props in scenes.items():
        for cam in tcb.CAMS:
            frame, masks = tcb._render(cam, props)
            for name, m0 in masks.items():
                for px in (0, 1, 2):
                    m = m0 if px == 0 else cv2.dilate(
                        m0.astype(np.uint8), np.ones((2 * px + 1,) * 2, np.uint8)).astype(bool)
                    ys, xs = np.nonzero(m)
                    parts = {"full": np.ones(xs.size, bool)}
                    for k, keep in (("half", xs < np.median(xs)), ("half", xs >= np.median(xs)),
                                    ("half", ys < np.median(ys)), ("half", ys >= np.median(ys))):
                        parts.setdefault(k, []).append(keep)
                    parts["quarter"] = [np.logical_and(a, b) for a in (xs < np.median(xs), xs >= np.median(xs))
                                        for b in (ys < np.median(ys), ys >= np.median(ys))]
                    for part, keeps in parts.items():
                        for keep in (keeps if isinstance(keeps, list) else [keeps]):
                            mm = np.zeros_like(m)
                            mm[ys[keep], xs[keep]] = True
                            pts = tcb._cloud(cam, frame, mm)
                            if pts.shape[0] < 10:
                                continue
                            s = sizes(pts)
                            if s is not None:
                                rows.append({"scene": scene, "object": name, "cam": cam, "bleed_px": px,
                                             "part": part, **s})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    out = {"rule": {"ratio": SIZE_GATE_RATIO, "excess_m": SIZE_GATE_EXCESS_M}}

    live = live_views(args.lab)
    out["live"] = {k: {m: stats([s[m] for s in v]) for m in ("diameter", "aa_xy", "obb")} for k, v in live.items()}
    for m in ("diameter", "aa_xy", "obb"):
        b = np.array([s[m] for s in live["bin"]])
        p = np.array([s[m] for s in live["prop"]])
        out["live"].setdefault("ratios", {})[m] = {
            "same_bin_max_over_min": round(float(b.max() / b.min()), 3),
            "same_prop_max_over_min": round(float(p.max() / p.min()), 3),
            "container_min_over_prop_max": round(float(b.min() / p.max()), 3),
            "container_min_minus_prop_max_m": round(float(b.min() - p.max()), 4),
        }
    rows = assignments(args.lab)
    if rows:
        absorbed = [r for r in rows if r["obs_obb_max"] > 0.15 and r["belief_born_obb_max"] < 0.12]
        out["live_defect"] = {
            "runs": sorted({r["run"] for r in rows}),
            "bin_views_absorbed_by_the_prop": len(absorbed),
            "their_obb_max": stats([r["obs_obb_max"] for r in absorbed]) if absorbed else None,
            "prop_belief_born_obb_max": stats([r["belief_born_obb_max"] for r in absorbed]) if absorbed else None,
            "sources": sorted({r["source"] for r in absorbed}),
        }

    rc = raycast()
    table = {}
    for r in rc:
        key = f"{r['object']} | {r['cam']} | bleed {r['bleed_px']}px | {r['part']}"
        table.setdefault(key, []).append(r["diameter"])
    out["raycast_diameter"] = {k: stats(v) for k, v in sorted(table.items())}
    worst = {}
    for scene in sorted({r["scene"] for r in rc}):
        for obj in sorted({r["object"] for r in rc if r["scene"] == scene}):
            for px in (0, 1):
                sel = [r for r in rc if r["scene"] == scene and r["object"] == obj and r["bleed_px"] <= px]
                full = max(r["diameter"] for r in sel if r["part"] == "full")
                for part in ("half", "quarter"):
                    sub = [r["diameter"] for r in sel if r["part"] == part]
                    if sub:
                        worst[f"{scene} | {obj} | bleed <= {px}px | full / smallest {part}"] = round(full / min(sub), 3)
                bled = [r["diameter"] for r in sel if r["part"] == "full"]
                worst[f"{scene} | {obj} | bleed <= {px}px | largest full / smallest full"] = round(max(bled) / min(bled), 3)
    out["raycast_ratios"] = worst
    text = json.dumps(out, indent=1)
    if args.json:
        args.json.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
