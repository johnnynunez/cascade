#!/usr/bin/env python3
"""B36: build the per-run evidence table from the recorded live runs.

usage: build_evidence.py <item>/live <out_dir>

Reads every run's trace.jsonl (pick_and_place result + physics postcondition),
its grasp-evidence receipt (close_hold, grip_verification, lift wrist roll,
attempt exceptions) and the server log (grasp planner), classifies the outcome
by signature and writes <out_dir>/runs.json + manifest.json (sha256 of every
input trace and receipt JSON).
"""
import glob
import hashlib
import json
import os
import re
import sys

LV, OUT = sys.argv[1], sys.argv[2]

# prefix -> (series, engine, arm, code); read from live/*.sh and the diffed rig trees
SERIES = [
    ("main", "A/B1", "physx", "main", "133876c"),
    ("rec", "A/B1", "physx", "recovery+contact_lift=pregrasp", "wip-8oct"),
    ("b36", "A/B1", "physx", "recovery+contact_lift=preserve_rotation", "wip-8oct"),
    ("smoke", "smoke", "physx", "recovery+hold", "wip-8oct"),
    ("e1fragile", "E1", "physx", "main material=fragile", "133876c"),
    ("e1rigid", "E1", "physx", "main material=rigid", "133876c"),
    ("ab2main", "A/B2", "physx", "main", "133876c"),
    ("ab2hold", "A/B2", "physx", "recovery+hold(all engines)", "wip-8oct"),
    ("nwmain", "A/B3", "newton", "main", "133876c"),
    ("nwhold", "A/B3", "newton", "recovery+hold(all engines)", "wip-8oct"),
    ("ab4pm", "A/B4", "physx", "main", "4e896c3"),
    ("ab4pf", "A/B4", "physx", "branch", "45c8945"),
    ("ab6pm", "A/B6", "physx", "main", "4e896c3"),
    ("ab6pf", "A/B6", "physx", "branch", "45c8945"),
    ("ab6nm", "A/B6", "newton", "main", "4e896c3"),
    ("ab6nf", "A/B6", "newton", "branch", "45c8945"),
]
#: 10 Oct (W8, live/w8/runs): main 38f6d08 vs the merged branch (rig = archive of ce0d1f8)
SERIES_W8 = [
    ("smoke", "W8-smoke", "physx", "branch", "ce0d1f8"),
    ("pm", "W8", "physx", "main", "38f6d08"),
    ("ph", "W8", "physx", "branch", "ce0d1f8"),
    ("nh", "W8", "newton", "branch", "ce0d1f8"),
]


def series_of(tag, w8=False):
    for prefix, series, engine, arm, code in sorted(SERIES_W8 if w8 else SERIES, key=lambda s: -len(s[0])):
        rest = tag[len(prefix):]
        if tag.startswith(prefix) and rest.isdigit():
            return dict(series=series, engine=engine, arm=arm, code=code)
    return None


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def receipt_facts(run):
    """Facts of the run's FIRST grasp attempt receipt (the one every series
    started with): hold, jaw widths, the lift's final wrist-roll error
    |q6 - target| (as scratch/q6_all_receipts.py), attempt exceptions."""
    facts = dict(hold=None, contact_open_frac=None, hold_open_frac=None, width_after_lift=None,
                 lift_q6_err=None, exceptions=[], receipts=[])
    files = sorted(glob.glob(f"{run}/grasp_evidence/*.json"))
    for i, f in enumerate(files):
        try:
            ev = json.load(open(f)).get("events", [])
        except (OSError, ValueError):
            continue
        facts["receipts"].append(f)
        tgt, last = None, None
        for e in ev:
            k, d = e.get("kind"), e.get("data") or {}
            if k == "attempt_exception":
                facts["exceptions"].append(str(d.get("message", ""))[:80])
            if i:
                continue
            if k == "close_hold" and facts["hold"] is None:
                facts["hold"] = "applied" if d.get("applied") else f"not applied: {d.get('reason')}"
                facts["contact_open_frac"] = d.get("contact_open_frac")
                facts["hold_open_frac"] = d.get("hold_open_frac")
            elif k == "grip_verification" and facts["width_after_lift"] is None:
                facts["width_after_lift"] = d.get("width_after_lift")
            elif k == "move_target" and e.get("phase") == "lift":
                tgt = d["q"][5]
            elif k == "isaac_feedback" and e.get("phase") == "lift" and tgt is not None:
                last = abs(d["state"]["q"][5] - tgt)
        if not i and last is not None:
            facts["lift_q6_err"] = round(last, 6)
    return facts


def planner(run, tag):
    for f in (f"{run}/server.log", f"{LV}/mcp_server_{tag}.log", f"{LV}/w8/logs/mcp_{tag}.log"):
        if os.path.exists(f):
            m = re.search(r"grasp_planner=([a-z_]+)", open(f, errors="replace").read())
            if m:
                return m.group(1)
    return None


rows, manifest = [], {}
run_dirs = sorted(glob.glob(f"{LV}/run_*")) + sorted(glob.glob(f"{LV}/w8/runs/run_*"))
for run in run_dirs:
    tag = os.path.basename(run)[4:]
    meta = series_of(tag, w8="/w8/runs/" in run)
    tr = f"{run}/trace.jsonl"
    if meta is None or not os.path.exists(tr):
        continue
    pick = None
    for line in open(tr):
        r = json.loads(line)
        if r.get("skill") == "pick_and_place":
            pick = r
    if pick is None:
        continue  # cut before the pick (ab4pm4)
    res = pick.get("result") or {}
    pc = res.get("postcondition") or {}
    meas = pc.get("measured") or {}
    err = res.get("error") or ""
    status = pc.get("status")
    moved = meas.get("moved_m") or 0.0
    if status == "confirmed":
        sig = "confirmed"
    elif res.get("stage") == "grasp" and "already holding" in err:
        sig = "S1"
    elif res.get("stage") == "place" and "did not settle above the place target" in err:
        sig = "S2"
    elif status == "refuted" and res.get("self_reported_ok") and res.get("grip_verified") and moved >= 0.05:
        sig = "S3"
    else:
        sig = "other"
    facts = receipt_facts(run)
    recovered = bool(res.get("grasp_recovered_after")) or (
        sig == "S2" and any("did not settle at grasp lift pose" in x for x in facts["exceptions"])
        and "already holding" not in err)
    rel = ("w8/runs/" if "/w8/runs/" in run else "") + f"run_{tag}"
    manifest[f"{rel}/trace.jsonl"] = sha(tr)
    for f in facts.pop("receipts"):
        manifest[f"{rel}/grasp_evidence/{os.path.basename(f)}"] = sha(f)
    rows.append(dict(
        tag=tag, **meta, planner=planner(run, tag), outcome=sig, physics=status,
        grip_verified=res.get("grip_verified"), grasp_attempts=res.get("grasp_attempts"),
        error=err[:120] or None, recovered_after_failed_attempt=recovered,
        final_xyz=meas.get("final"), target_err_m=meas.get("target_err_m"), moved_m=meas.get("moved_m"),
        duration_s=round(pick.get("duration_ms", 0) / 1000, 1), **facts,
    ))

os.makedirs(OUT, exist_ok=True)
json.dump(rows, open(f"{OUT}/runs.json", "w"), indent=1)
json.dump({"inputs_sha256": manifest}, open(f"{OUT}/manifest.json", "w"), indent=1, sort_keys=True)
print(f"{len(rows)} runs -> {OUT}/runs.json")
