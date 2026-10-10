#!/usr/bin/env python3
"""B36 W8: one line per finished W8 run (outcome, hold, error)."""
import glob
import json
import os
import sys

W8 = sys.argv[1]
for f in sorted(glob.glob(f"{W8}/runs/*.json")):
    tag = os.path.basename(f)[:-5]
    try:
        r = json.load(open(f))
    except ValueError:
        print(tag, "unreadable/partial")
        continue
    p = next((c["result"] for c in r["calls"] if c["tool"] == "pick_and_place"), {})
    pc = p.get("postcondition") or {}
    holds = []
    for g in sorted(glob.glob(f"{W8}/runs/run_{tag}/grasp_evidence/*.json")):
        ev = json.load(open(g)).get("events", [])
        holds += [(e["data"].get("applied"), round(e["data"].get("contact_open_frac") or 0, 3)) for e in ev
                  if e["kind"] == "close_hold"]
    rs = [c for c in r["calls"] if c["tool"] == "reset_scene"]
    print(f"{tag:8s} ok={p.get('ok')} outcome={p.get('outcome')} pc={pc.get('status')} "
          f"err_m={(pc.get('measured') or {}).get('target_err_m')} gv={p.get('grip_verified')} "
          f"att={p.get('grasp_attempts')} recovered={'grasp_recovered_after' in p} hold={holds} "
          f"reset_ok={rs[0]['isError'] is False if rs else None} err={str(p.get('error'))[:90]}")
