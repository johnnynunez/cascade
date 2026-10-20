#!/usr/bin/env python3
"""Improve CASCADE's agent harness with Harness²-style recursive harnessing.

For each task in a task file: run the current harness, propose INV and DOM edits, probe each on a
fresh runtime, compose, select blind, gate on CASCADE's verifier, reflect, and append the adopted
harness and lesson to the per-family library. Tasks of one family run in order and each starts from
the harness the previous one adopted. No model weights, safety values or configuration change.

Example (MuJoCo SO-101, GPT-6-Astra through the Codex CLI as the agent, Claude Code as improver):

    CASCADE_OCCUPANCY=0 MUJOCO_GL=egl .venv/bin/python scripts/harness2_improve.py \
        --tasks benchmark/harness2/so101_mujoco_tasks.json --arm so101_mujoco --camera mujoco_scene \
        --llm codex_astra --improver claude --k 1 --mode parallel

Then serve the result: `agent.harness_path: <out>/promoted/harness.json` (or
CASCADE_AGENT_HARNESS=<path>) makes `python -m cascade.apps.demo` read it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import functools
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
print = functools.partial(print, flush=True)  # noqa: A001 -- progress lines reach a log under systemd
sys.path.insert(0, str(REPO / "src"))

from cascade.config import load_demo_config  # noqa: E402
from cascade.harnessing import (  # noqa: E402
    AgentHarness, CascadeExecutor, Harnessing, Library, SubprocessExecutor, TaskSpec, make_improver, scope_key,
)
from cascade.skills.runtime import tool_specs_for_config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", type=Path, required=True, help="JSON list of {id, task, domain?, max_steps?}")
    ap.add_argument("--arm", default="so101_mujoco")
    ap.add_argument("--camera", default="mujoco_scene")
    ap.add_argument("--robot", default=None,
                    help="composed robot profile (configs/robots/<robot>.yaml, e.g. g1_arena_locomanip); "
                         "replaces --arm/--camera")
    ap.add_argument("--llm", default="codex_astra", help="LLM profile of the agent being improved")
    ap.add_argument("--improver", default="claude", choices=("claude", "codex", "llm"))
    ap.add_argument("--improver-model", default=None)
    ap.add_argument("--improver-llm", default=None, help="LLM profile for --improver llm (e.g. local_qwen)")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--mode", choices=("parallel", "sequential"), default="parallel")
    ap.add_argument("--gate", choices=("verified", "none"), default="verified")
    ap.add_argument("--start", default="library",
                    help="seed of a family with no adopted harness yet in this scope: 'empty' (or 'library') "
                         "or a harness.json path; later tasks always continue from the library")
    ap.add_argument("--out", type=Path, default=None, help="run directory (default runs/harness2/<stamp>)")
    ap.add_argument("--library", type=Path, default=Path("~/.cascade/harness2/library").expanduser())
    ap.add_argument("--workers", type=int, default=2, help="concurrent improver sessions")
    ap.add_argument("--rollout-workers", type=int, default=1,
                    help="concurrent rollouts, each in its own process (1: in this process, one at a time)")
    ap.add_argument("--no-decompose", action="store_true", help="skip the milestone decomposition turn")
    ap.add_argument("--dry-run", action="store_true", help="validate tasks and the base harness; run nothing")
    args = ap.parse_args(argv)

    tasks_raw = json.loads(args.tasks.read_text())
    specs = [TaskSpec.from_dict(t) for t in tasks_raw]
    if len({s.id for s in specs}) != len(specs):
        raise SystemExit("task ids must be unique")
    # A task file may span several composed robots (each task names its `robot`); one run
    # improves the tasks of the selected robot only.
    other = [s.id for s in specs if s.robot is not None and s.robot != args.robot]
    if other:
        print(f"[harness2] skipped (other robot than {args.robot!r}): {', '.join(other)}")
        specs = [s for s in specs if s.robot is None or s.robot == args.robot]
    if not specs:
        raise SystemExit("no task for this rig")
    if args.robot:
        from cascade.apps.robot_runtime import robot_tool_descriptors

        cfg = load_demo_config(robot=args.robot, llm=args.llm)
        tool_specs = [t.as_spec() for t in robot_tool_descriptors(cfg).values()]
    else:
        cfg = load_demo_config(camera=args.camera, arm=args.arm, llm=args.llm)
        tool_specs = tool_specs_for_config(cfg)
    tool_names = {s["name"] for s in tool_specs}
    model = str(cfg.llm.get("model", "")) if "llm" in cfg else ""
    scope = (scope_key(arm=f"robot-{args.robot}", camera="", llm_profile=args.llm, model=model) if args.robot
             else scope_key(arm=args.arm, camera=args.camera, llm_profile=args.llm, model=model))
    out = args.out or REPO / "runs" / "harness2" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"[harness2] scope {scope}; {len(specs)} task(s); mode {args.mode} k={args.k}; out {out}")

    def seed() -> AgentHarness:
        if args.start in ("empty", "library"):
            return AgentHarness()
        return AgentHarness.read(Path(args.start))

    def base_for(domain: str) -> AgentHarness:
        # Harness² task order: a family continues from the harness its last task adopted; the
        # --start harness seeds a family whose library is still empty in this scope.
        return Library(args.library, scope, domain).base_or(seed())

    for domain in sorted({s.domain for s in specs}):
        problems = base_for(domain).problems(tool_names)
        if problems:
            raise SystemExit(f"base harness for {domain} refused: {problems}")
    if args.dry_run:
        for s in specs:
            print(f"  {s.domain}/{s.id}: {s.task!r} (max {s.max_steps} steps)")
        print(f"[harness2] dry run: {4 * args.k} rollouts and {3 * args.k} improver sessions per task, "
              f"plus one selection and one reflection")
        return 0

    improver_llm = None
    if args.improver == "llm":
        from cascade.agent.llm import make_llm

        if not args.improver_llm:
            raise SystemExit("--improver llm needs --improver-llm <profile>")
        improver_llm = make_llm((load_demo_config(llm=args.improver_llm, _ignore_robot_environment=True) if args.robot
                                 else load_demo_config(camera=args.camera, arm=args.arm, llm=args.improver_llm)).llm)
    improver = make_improver(args.improver, model=args.improver_model, llm=improver_llm)
    out.mkdir(parents=True, exist_ok=True)
    executor = CascadeExecutor(camera=args.camera, arm=args.arm, llm_profile=args.llm, robot=args.robot,
                               decompose=not args.no_decompose, build_lock=str(out.resolve() / ".build.lock"))
    if args.rollout_workers > 1:
        executor = SubprocessExecutor(executor)
    (out / "config.json").write_text(json.dumps({**vars(args), "tasks": str(args.tasks), "out": str(out),
                                                 "library": str(args.library), "scope": scope,
                                                 "env": {k: v for k, v in os.environ.items()
                                                         if k.startswith(("CASCADE_", "MUJOCO_"))}},
                                                default=str, indent=2) + "\n")
    summary = []
    for spec in specs:
        if (out / spec.id / "result.json").is_file():
            print(f"[harness2] {spec.id}: already done, skipped (resume)")
            summary.append(json.loads((out / spec.id / "result.json").read_text()))
            continue
        library = Library(args.library, scope, spec.domain)
        engine = Harnessing(executor, improver, work_dir=out, tool_specs=tool_specs, library=library,
                            workers=args.workers, gate=args.gate, rollout_workers=args.rollout_workers)
        result = engine.improve(spec, base_for(spec.domain), k=args.k, mode=args.mode)
        summary.append(result.as_dict())
        print(f"[harness2] {spec.id}: adopted {result.adopted_sha[:12]} from {result.adopted_from}; "
              f"gate: {result.gate.get('decision')}")
    promoted = out / "promoted"
    last = specs[-1].domain
    final = Library(args.library, scope, last).latest_adopted() or AgentHarness()
    final.write(promoted)
    lines = ["# Harness² run", "", f"Scope `{scope}`, mode {args.mode}, k={args.k}, gate {args.gate}", "",
             "Measured = the task's hidden check (simulator ground truth) when it has one, else CASCADE's verifier.", "",
             "| task | base | adopted | from | measured: base | measured: every execution |", "|---|---|---|---|---|---|"]
    for r in summary:
        meas = {lbl: ro["measured_success"] for lbl, ro in r["rollouts"].items()}
        base_m = [v for lbl, v in meas.items() if lbl.startswith("base")]
        every = ", ".join(f"{lbl} {v}" for lbl, v in meas.items())
        lines.append(f"| {r['task_id']} | `{r['base_sha'][:12]}` | `{r['adopted_sha'][:12]}` | {r['adopted_from']} | "
                     f"{base_m} | {every} |")
    lines += ["", f"Promoted harness: `{promoted / 'harness.json'}` (`{final.sha()[:12]}`)", ""]
    (out / "SUMMARY.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
