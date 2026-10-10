#!/usr/bin/env python3
"""live-w6: B46 per-run records from the probe JSONs + the server traces (no re-interpretation:
fields are copied; the attempt-failure messages are read from the grasp-evidence receipts)."""
import glob
import hashlib
import json
import os
import re
import sys

LV = "/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261009/live-w6"
ORDER = ["on1", "off1", "on2", "off2", "on3", "off3", "on4", "off4", "stop1"]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def attempt_failures(run):
    out = []
    for f in sorted(glob.glob(f"{run}/grasp_evidence/*.json")):
        d = json.load(open(f))
        exc = [e["data"].get("message") for e in d.get("events", []) if e.get("kind") == "attempt_exception"]
        out.append(exc[0] if exc else None)
    return out


def main():
    runs = []
    for i, tag in enumerate(ORDER):
        p = f"{LV}/b46/{tag}.json"
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            runs.append({"tag": tag, "order": i, "missing": True})
            continue
        d = json.load(open(p))
        run = f"{LV}/b46/run_{tag}"
        trace = f"{run}/trace.jsonl"
        rows = [json.loads(x) for x in open(trace) if x.strip()]
        picks = [r for r in rows if r.get("skill") == "pick_and_place"]
        bridge = open(f"{LV}/logs/bridge_{tag}.log", errors="replace").read()
        att = re.search(r'\[bridge\] GPU attestation: (\{.*\})', bridge)
        rec = {"tag": tag, "order": i, "lane": d["lane"], "motion": d["motion"], "motion_s": d["motion_s"],
               "motion_result": d["motion_result"], "motion_postcondition": d["motion_postcondition"],
               "reads": d["reads"], "after_motion_world_state_marked": d["after_motion_world_state_marked"],
               "stop": d.get("stop"),
               "trace_pick_rows": len(picks),
               "trace_pick_postcondition": [(r["result"].get("postcondition") or {}).get("status") for r in picks],
               "grasp_attempt_failures": attempt_failures(run),
               "bridge_gpu": json.loads(att.group(1)) if att else None,
               "sources": {"probe_json": f"b46/{tag}.json", "probe_json_sha256": sha(p),
                           "trace": f"b46/run_{tag}/trace.jsonl", "trace_sha256": sha(trace)}}
        runs.append(rec)
    json.dump({"runs": runs}, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
