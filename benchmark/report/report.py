#!/usr/bin/env python3
"""Aggregate the LIBERO sweep into one comparable table.

Reports a Wilson 95% interval per cell, not a bare percentage: with 50
episodes the normal approximation is bad near 0% and 100%, which is exactly
where the no-op floor and a good policy live.
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

RESULTS = Path(str(paths.RESULTS_DIR) + "/")

#: Published OpenVLA fine-tuned numbers (50 eps/task) for sanity-checking the
#: harness. A cell far outside its interval means a setup bug, not a finding.
PUBLISHED = {
    "libero_spatial": 84.7,
    "libero_object": 88.4,
    "libero_goal": 79.2,
    "libero_10": 53.7,
}

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(policy: str, suite: str, eps: int = 5) -> dict | None:
    f = RESULTS / f"{policy}_{suite}_ep{eps}.json"
    return json.loads(f.read_text()) if f.is_file() else None


def main() -> None:
    print()
    print(f"{'suite':<16} {'no-op':>12} {'OpenVLA-7B':>22} {'pub.':>7} {'cap':>5}")
    print("-" * 68)
    rows = []
    for s in SUITES:
        n, v = load("noop", s), load("openvla", s)
        n_txt = "-"
        if n:
            n_txt = f"{n['success_rate']:.0%} ({n['successes_total']}/{n['episodes_total']})"
        v_txt = "-"
        if v:
            k, N = v["successes_total"], v["episodes_total"]
            lo, hi = wilson(k, N)
            v_txt = f"{k/N:.0%} [{lo:.0%}-{hi:.0%}] ({k}/{N})"
            rows.append((s, k, N, lo, hi))
        cap = (v or n or {}).get("max_steps", "-")
        print(f"{s:<16} {n_txt:>12} {v_txt:>22} {PUBLISHED[s]:>6.1f}% {cap:>5}")

    if rows:
        K = sum(r[1] for r in rows)
        NN = sum(r[2] for r in rows)
        lo, hi = wilson(K, NN)
        print("-" * 68)
        print(f"{'TOTAL':<16} {'':>12} {K/NN:>7.0%} [{lo:.0%}-{hi:.0%}] ({K}/{NN})")

        print("\nsanity check vs published (50 eps/task):")
        for s, k, N, lo, hi in rows:
            pub = PUBLISHED[s] / 100
            ok = lo <= pub <= hi
            print(f"  {s:<16} {'OK  ' if ok else 'OFF '} published {pub:.0%} "
                  f"{'inside' if ok else 'OUTSIDE'} [{lo:.0%}, {hi:.0%}]")

    print("\nper-task detail (OpenVLA):")
    for s in SUITES:
        v = load("openvla", s)
        if not v:
            continue
        print(f"\n  {s}")
        for t in v["tasks"]:
            bar = "#" * int(t["success_rate"] * 20)
            print(f"    [{t['task_id']}] {t['success_rate']:5.0%} {bar:<20} "
                  f"{t['mean_steps']:5.0f} steps  {t['language'][:46]}")


if __name__ == "__main__":
    main()
