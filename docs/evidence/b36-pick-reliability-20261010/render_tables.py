#!/usr/bin/env python3
"""B36: render the evidence tables (per-arm tally + per-run rows) from runs.json.

usage: render_tables.py <runs.json>    -> markdown on stdout
"""
import collections
import json
import statistics
import sys

rows = json.load(open(sys.argv[1]))
ORDER = ["A/B1", "E1", "A/B2", "A/B3", "A/B4", "A/B6", "smoke", "W8", "W8-smoke"]


def key(r):
    return (ORDER.index(r["series"]), r["engine"], r["arm"], r["tag"])


print("### Per arm\n")
print("| series | engine | planner | arm (code) | confirmed | failures by signature |")
print("| --- | --- | --- | --- | --- | --- |")
agg = collections.OrderedDict()
for r in sorted(rows, key=key):
    agg.setdefault((r["series"], r["engine"], r["planner"], f"{r['arm']} ({r['code']})"), []).append(r["outcome"])
for (series, engine, planner, arm), outs in agg.items():
    fails = collections.Counter(o for o in outs if o != "confirmed")
    f = ", ".join(f"{k} {v}" for k, v in sorted(fails.items())) or "--"
    print(f"| {series} | {engine} | {planner} | {arm} | {outs.count('confirmed')}/{len(outs)} | {f} |")

print("\n### Lift wrist-roll error (first lift of each run, |q6 - target|, settle_tol 0.045)\n")
print("| engine | close | lifts | median rad | max rad | >= 0.045 |")
print("| --- | --- | --- | --- | --- | --- |")
m = collections.defaultdict(list)
for r in rows:
    if r["lift_q6_err"] is None:
        continue
    close = "hold applied" if r["hold"] == "applied" else (
        "fragile 0.60" if "fragile" in r["arm"] else "stage-2 0.85, no hold")
    m[(r["engine"], close)].append(r["lift_q6_err"])
for (engine, close), v in sorted(m.items()):
    print(f"| {engine} | {close} | {len(v)} | {statistics.median(v):.4f} | {max(v):.4f} | "
          f"{sum(x >= 0.045 for x in v)} |")

print("\n### Every run\n")
print("| run | series | engine | arm | planner | outcome | grip verified | hold | contact -> hold open | "
      "width after lift | lift q6 err | recovered | final xyz (m) | err to target (m) |")
print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for r in sorted(rows, key=key):
    hold = r["hold"] or "--"
    co = (f"{r['contact_open_frac']:.3f} -> {r['hold_open_frac']:.3f}"
          if r.get("hold_open_frac") is not None else "--")
    w = f"{r['width_after_lift']:.3f}" if r["width_after_lift"] is not None else "--"
    q = f"{r['lift_q6_err']:.6f}" if r["lift_q6_err"] is not None else "--"
    fin = "[" + ", ".join(f"{x:.3f}" for x in r["final_xyz"]) + "]" if r["final_xyz"] else "--"
    te = f"{r['target_err_m']:.3f}" if r["target_err_m"] is not None else "--"
    print(f"| {r['tag']} | {r['series']} | {r['engine']} | {r['arm']} ({r['code']}) | {r['planner']} | "
          f"{r['outcome']} | {r['grip_verified']} | {hold} | {co} | {w} | {q} | "
          f"{'yes' if r['recovered_after_failed_attempt'] else '--'} | {fin} | {te} |")
