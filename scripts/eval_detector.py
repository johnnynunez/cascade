#!/usr/bin/env python3
"""Score the detector against Isaac physics ground truth.

Step 1 of the synthetic-augmentation plan (docs/SYNTHETIC_RGBD_PIPELINE.md):
before improving a detector you need a number that says whether you improved
it. This is that number.

The sim gives us something a real rig cannot: the exact pose of every prop,
from PhysX, independent of perception. So precision/recall here are measured,
not estimated from a held-out split the detector may have memorised.

Two failure modes are scored separately because they have different causes:

  phantoms  a detection with no real prop within HIT_M -- a false positive.
            Persistent phantoms are worse than noisy ones: they survive into
            the world model and the agent plans around objects that are not
            there.
  flicker   the same static scene yielding a different object count between
            frames. Even when every detection is real, an unstable count
            makes "how many cubes are there" unanswerable.

Usage:
    cd models && PYTHONPATH=../src python ../scripts/eval_detector.py
    ... --frames 20 --cameras isaac,isaac_side --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

#: A detection this close (m) to a real prop counts as a hit. 6 cm is roughly
#: one cube width -- tight enough to catch a wrong object, loose enough to
#: forgive the surface-vs-centroid offset a depth probe inherently has.
HIT_M = 0.06

#: Props that are scenery, not grasp targets. The bin is 4 wall prims; it is
#: scored as ONE object at their centroid rather than four.
_SCENERY = ("table", "ground", "plane", "light")


def _real_objects(poses: dict) -> list[tuple[str, np.ndarray]]:
    """Collapse the bin's walls into one object; drop scenery."""
    walls = [np.asarray(v, dtype=float) for k, v in poses.items() if "wall" in k]
    out = [
        (k, np.asarray(v, dtype=float))
        for k, v in poses.items()
        if "wall" not in k and not any(s in k.lower() for s in _SCENERY)
    ]
    if walls:
        out.append(("bin", np.mean(walls, axis=0)))
    return out


def evaluate(runtime, truth_reader, frames: int, settle_s: float = 0.25) -> dict:
    real = _real_objects(truth_reader.all_poses())
    counts: list[int] = []
    hits = phantoms = total = 0
    missed_per_frame: list[int] = []
    worst_err = 0.0
    errs: list[float] = []

    for _ in range(frames):
        runtime.execute("get_observation", {})
        seen = [o for o in runtime.beliefs.all() if getattr(o, "visible", True)]
        counts.append(len(seen))

        matched: set[str] = set()
        for o in seen:
            total += 1
            p = np.asarray(o.position, dtype=float)[:3]
            if not real:
                phantoms += 1
                continue
            name, d = min(
                ((n, float(np.linalg.norm(p - t))) for n, t in real),
                key=lambda kv: kv[1],
            )
            if d <= HIT_M:
                hits += 1
                matched.add(name)
                errs.append(d)
                worst_err = max(worst_err, d)
            else:
                phantoms += 1
        missed_per_frame.append(len(real) - len(matched))
        time.sleep(settle_s)

    n = max(total, 1)
    exp = max(len(real) * frames, 1)
    return {
        "frames": frames,
        "ground_truth": {n_: [round(float(x), 4) for x in t] for n_, t in real},
        "objects_expected": len(real),
        "counts_per_frame": counts,
        "count_stable": len(set(counts)) == 1,
        "count_mode_correct": bool(counts and max(set(counts), key=counts.count) == len(real)),
        "detections": total,
        "hits": hits,
        "phantoms": phantoms,
        "precision": round(hits / n, 4),
        "recall": round((exp - sum(missed_per_frame)) / exp, 4),
        "phantom_rate": round(phantoms / n, 4),
        "localization_err_m": {
            "mean": round(float(np.mean(errs)), 4) if errs else None,
            "max": round(worst_err, 4) if errs else None,
        },
    }


def _print(r: dict) -> None:
    print(f"\nGROUND TRUTH ({r['objects_expected']} objects, from Isaac physics):")
    for k, v in r["ground_truth"].items():
        print(f"   {k:14} {v}")
    print(f"\nOVER {r['frames']} FRAMES, SCENE STATIC:")
    print(f"   count per frame : {r['counts_per_frame']}")
    stable = "STABLE" if r["count_stable"] else "FLICKER"
    print(f"   stability       : {stable}")
    print(f"   modal count     : {'correct' if r['count_mode_correct'] else 'WRONG'}")
    print(f"\n   detections      : {r['detections']}")
    print(f"   precision       : {r['precision']:.1%}")
    print(f"   recall          : {r['recall']:.1%}")
    print(f"   phantom rate    : {r['phantom_rate']:.1%}  ({r['phantoms']} detections)")
    e = r["localization_err_m"]
    if e["mean"] is not None:
        print(f"   localization    : mean {e['mean']*100:.1f} cm, worst {e['max']*100:.1f} cm")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="Score the detector against physics truth.")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--cameras", default="isaac,isaac_side")
    ap.add_argument("--arm", default="isaac")
    ap.add_argument("--bridge-port", type=int, default=8611)
    ap.add_argument("--json", type=Path, help="also write the report here")
    args = ap.parse_args()

    from cascade.apps.demo import build_runtime, shutdown_runtime
    from cascade.config import load_demo_config
    from cascade.sim.bridge_client import BridgeClient
    from cascade.sim.truth import TruthPoseReader

    cfg = load_demo_config(
        cameras=args.cameras.split(","), arm=args.arm, llm="mock"
    )
    cfg._data.setdefault("stream", {})["mode"] = "off"   # never bind a port to measure

    client = BridgeClient(port=args.bridge_port)
    try:
        client.connect()
    except Exception as e:
        print(f"[!] no Isaac bridge on :{args.bridge_port} -- physics truth is "
              f"required to score honestly ({e})", file=sys.stderr)
        return 2

    rt, arm = build_runtime(cfg, Path("/tmp/eval_detector"), view=False, serve=False)
    try:
        report = evaluate(rt, TruthPoseReader(client), args.frames)
    finally:
        shutdown_runtime(rt, arm)

    _print(report)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"[+] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
