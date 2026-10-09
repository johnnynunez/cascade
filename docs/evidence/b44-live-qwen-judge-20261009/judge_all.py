"""B44-live: local Qwen (:8080) as the outcome judge over recorded Isaac picks.

Every run dir under live/runs is a COPY (trace.jsonl + keyframes) of a live
Isaac run (B36 pick-reliability A/B series, B35 NemoClaw runtime), whose
pick_and_place rows carry a physics-channel postcondition. judge_run.py scores
each pick's BEFORE/AFTER keyframes with the GRM prompt; this script collects the
judge-vs-physics confusion matrix over all of them.

run: python live/judge_all.py   (writes live/results.json + live/results.txt)
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parent / "cascade"
PY = "/home/johnny/Projects/demo/cascade/.venv/bin/python"


def main() -> int:
    runs = sorted(p for p in (HERE / "runs").iterdir() if (p / "trace.jsonl").exists())
    out, t0 = [], time.time()
    for i, run in enumerate(runs, 1):
        jr = run / "judge.json"
        if not jr.exists():
            p = subprocess.run([PY, str(WT / "scripts" / "judge_run.py"), str(run), "--config",
                                str(HERE / "judge-qwen.json"), "--skills", "pick_and_place"],
                               env={"PYTHONPATH": str(WT / "src"), "PATH": "/usr/bin:/bin",
                                    "CUDA_VISIBLE_DEVICES": "-1", "HOME": str(Path.home())},
                               capture_output=True, text=True, timeout=900)
            if p.returncode != 0 or not jr.exists():
                out.append({"run": run.name, "error": (p.stderr or p.stdout)[-300:]})
                print(f"[{i}/{len(runs)}] {run.name}: ERROR", flush=True)
                continue
        v = json.loads(jr.read_text())
        for s in v["steps"]:
            out.append({"run": run.name, "physics": s.get("physics"), "hop": s.get("hop"),
                        "error": s.get("error"), "evidence": None,
                        "completion_tokens": ((s.get("response_metadata") or {}).get("usage") or {}).get("completion_tokens")})
        print(f"[{i}/{len(runs)}] {run.name}: {[(s.get('physics'), s.get('hop')) for s in v['steps']]} "
              f"({time.time() - t0:.0f} s)", flush=True)
    c = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n_steps": 0, "n_scored": 0, "n_with_physics": 0,
         "unverified_scored": 0, "errors": 0}
    for r in out:
        if "hop" not in r:
            c["errors"] += 1
            continue
        c["n_steps"] += 1
        if r["hop"] is None:
            c["errors"] += 1
            continue
        c["n_scored"] += 1
        if r["physics"] not in ("confirmed", "refuted"):
            c["unverified_scored"] += 1
            continue
        c["n_with_physics"] += 1
        pos, conf = r["hop"] > 0, r["physics"] == "confirmed"
        c["tp" if pos and conf else "fn" if conf else "fp" if pos else "tn"] += 1
    n = c["n_with_physics"]
    c["agreement"] = (c["tp"] + c["tn"]) / n if n else None
    c["precision"] = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else None
    c["recall"] = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else None
    (HERE / "results.json").write_text(json.dumps({"confusion": c, "steps": out,
                                                   "judge_config": json.loads((HERE / "judge-qwen.json").read_text()),
                                                   "elapsed_s": round(time.time() - t0, 1)}, indent=1))
    line = (f"judge=vlm:Qwen/Qwen3.8-27B runs={len(runs)} steps={c['n_steps']} scored={c['n_scored']} "
            f"physics-graded={n} tp={c['tp']} tn={c['tn']} fp={c['fp']} fn={c['fn']} "
            f"agreement={c['agreement']} precision={c['precision']} recall={c['recall']} "
            f"unverified_scored={c['unverified_scored']} errors={c['errors']}")
    (HERE / "results.txt").write_text(line + "\n")
    print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
