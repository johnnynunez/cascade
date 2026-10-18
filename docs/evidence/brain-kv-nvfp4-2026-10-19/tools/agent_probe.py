#!/usr/bin/env python3
"""The served brain as CASCADE's agent: MuJoCo SO-101 tasks scored by the hidden physics checks.

    agent_probe.py --profile bench_qwen_8080 --label L --out f.jsonl [--reps 2] [--workers 4]

Runs the one-cube tasks (benchmark/harness2/so101_mujoco_tasks.json) and tc-blue-only (two cubes) with the empty
harness, `--reps` times each, every rollout on a fresh runtime in its own process (cascade.harnessing).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

REPO = Path("/home/johnny/Projects/cascade-wt/nvfp4-kv")
sys.path.insert(0, str(REPO / "src"))

from cascade.harnessing import AgentHarness, CascadeExecutor, SubprocessExecutor, TaskSpec  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    one = json.loads((REPO / "benchmark/harness2/so101_mujoco_tasks.json").read_text())
    two = [t for t in json.loads((REPO / "benchmark/harness2/so101_mujoco_two_cubes_tasks.json").read_text())
           if t["id"] == "tc-blue-only"]
    root = args.out.parent / f"rollouts-{args.label}"
    jobs = []
    for camera, tasks in (("mujoco_scene", one), ("mujoco_scene_two", two)):
        ex = SubprocessExecutor(CascadeExecutor(camera=camera, arm="so101_mujoco", llm_profile=args.profile,
                                                build_lock=str(root / ".build.lock")), timeout_s=1500)
        for t in tasks:
            for r in range(1, args.reps + 1):
                jobs.append((ex, TaskSpec.from_dict(t), f"{t['id']}-r{r}"))

    def run(job):
        ex, spec, label = job
        return ex.run(AgentHarness(), spec, label, root / label)

    with ThreadPoolExecutor(args.workers) as pool:
        rollouts = list(pool.map(run, jobs))
    with args.out.open("a") as f:
        for r in rollouts:
            check = r.check or {}
            row = {"label": args.label, "probe": "agent", "task": r.task_id, "rollout": r.label,
                   "passed": r.measured_success, "physics": check.get("physics"), "claimed": r.claimed_success,
                   "steps": r.steps, "s": r.duration_s, "ended_by": r.ended_by, "error": (r.error or "")[-300:]}
            f.write(json.dumps(row) + "\n")
    passed = sum(1 for r in rollouts if r.measured_success)
    physics = sum(1 for r in rollouts if (r.check or {}).get("physics"))
    print(f"{args.label} agent: {passed}/{len(rollouts)} passed (physics {physics}/{len(rollouts)})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
