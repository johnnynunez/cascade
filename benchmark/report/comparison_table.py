#!/usr/bin/env python3
"""Build the comparison table: our measurements vs the published literature.

Hard rule enforced here: MEASURED and PUBLISHED never share a column without
being labelled. Published numbers come from different benchmarks, different
perception stacks and different policies -- printing them next to ours as if
they were rivals would be dishonest. They are context, not competition.
"""

import sys as _sys
from pathlib import Path as _Path

_BENCH = _Path(__file__).resolve().parent.parent
if str(_BENCH) not in _sys.path:
    _sys.path.insert(0, str(_BENCH))
import paths  # noqa: E402  (benchmark/paths.py)

paths.add_paths()

import json
import math
from pathlib import Path

R = Path(str(paths.RESULTS_DIR) + "/")
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
GAPF = {"libero_spatial": "gap_spatial", "libero_object": "gap_libero_object",
        "libero_goal": "gap_libero_goal", "libero_10": "gap_libero_10"}
PUBLISHED_OPENVLA = {"libero_spatial": 84.7, "libero_object": 88.4,
                     "libero_goal": 79.2, "libero_10": 53.7}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    rows, tot, N = {}, {k: 0 for k in
                       ("noop", "bare", "loop", "verified", "openvla")}, 0
    for s in SUITES:
        g = json.loads((R / f"{GAPF[s]}.json").read_text())["results"]
        nv = json.loads((R / f"noop_{s}_ep5.json").read_text())
        ov = json.loads((R / f"openvla_{s}_ep5.json").read_text())
        n = g["bare"]["n"]
        rows[s] = {"noop": nv["successes_total"], "bare": g["bare"]["succ"],
                   "loop": g["loop"]["succ"], "verified": g["verified"]["succ"],
                   "openvla": ov["successes_total"], "n": n,
                   "cap": ov["max_steps"]}
        for k in tot:
            tot[k] += rows[s][k]
        N += n

    print("=" * 78)
    print("PER SUITE  (n=50 episodes per cell: 10 tasks x 5 episodes)")
    print("=" * 78)
    hdr = f"{'suite':<15}{'no-op':>7}{'bare':>7}{'loop':>7}{'verified':>10}{'OpenVLA':>9}{'pub.':>8}"
    print(hdr)
    print("-" * 78)
    for s in SUITES:
        r = rows[s]
        n = r["n"]
        print(f"{s:<15}{r['noop']/n:>7.0%}{r['bare']/n:>7.0%}{r['loop']/n:>7.0%}"
              f"{r['verified']/n:>10.0%}{r['openvla']/n:>9.0%}"
              f"{PUBLISHED_OPENVLA[s]:>7.1f}%")
    print("-" * 78)
    print(f"{'TOTAL':<15}{tot['noop']/N:>7.0%}{tot['bare']/N:>7.0%}"
          f"{tot['loop']/N:>7.0%}{tot['verified']/N:>10.0%}"
          f"{tot['openvla']/N:>9.0%}{'':>8}")
    print()
    print(f"n = {N} episodes per condition")
    print()
    print("95% Wilson intervals on the totals:")
    for k in ("noop", "bare", "loop", "verified", "openvla"):
        lo, hi = wilson(tot[k], N)
        print(f"  {k:9} {tot[k]:3}/{N} = {tot[k]/N:6.1%}  [{lo:5.1%}, {hi:5.1%}]")

    b, v, l = tot["bare"] / N, tot["verified"] / N, tot["loop"] / N
    print()
    print("Deltas:")
    print(f"  bare -> verified   {(v-b)*100:+5.1f} pts   (x{v/b:.2f})")
    print(f"  bare -> loop       {(l-b)*100:+5.1f} pts   (x{l/b:.2f})")
    sp = rows["libero_spatial"]
    print(f"  spatial only       {sp['bare']/50:.0%} -> {sp['verified']/50:.0%}"
          f"   ({(sp['verified']-sp['bare'])*2:+.0f} pts, x{sp['verified']/max(1,sp['bare']):.1f})")

    lo_b, hi_b = wilson(tot["bare"], N)
    lo_v, hi_v = wilson(tot["verified"], N)
    overlap = lo_v <= hi_b
    print()
    print(f"  bare  [{lo_b:.1%}, {hi_b:.1%}]")
    print(f"  verif [{lo_v:.1%}, {hi_v:.1%}]  -> intervals "
          f"{'OVERLAP: aggregate gap NOT significant' if overlap else 'DISJOINT: significant'}")


if __name__ == "__main__":
    main()
