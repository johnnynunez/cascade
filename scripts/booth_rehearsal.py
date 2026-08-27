#!/usr/bin/env python3
"""Booth rehearsal: prove an LLM profile's tool-calling under the runbook's
attendee prompts BEFORE the show (docs/BOOTH_RUNBOOK.md §1.4).

Runs each prompt through the real AgentOrchestrator on the MOCK stack (no
hardware, no fast-path tiers -- every prompt must exercise the model) and
reports per prompt: success, steps, which tools were called, duration.
Grasp/experience memory are isolated to the run dir so a rehearsal never
contaminates the booth's learned state. The CASCADE_BOOTH tuning overlay is on
by default (realistic timing budgets).

    .demo/bin/python scripts/booth_rehearsal.py --llm local_qwen
    .demo/bin/python scripts/booth_rehearsal.py --llm anthropic --max-steps 20
    .demo/bin/python scripts/booth_rehearsal.py --prompts my_prompts.txt

Exit code 1 if any prompt produced ZERO tool calls (total protocol failure);
grasp failures on the mock arm are expected (jaws close on air) and are
themselves part of what is being rehearsed -- watch HOW the model handles
the error result, not whether the pick lands.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# The mock detector labels everything "red cube" -- these mirror the
# runbook §5 beats, renamed to the mock scene so success is achievable.
DEFAULT_PROMPTS = [
    "Wave hello to the group, then tell us what's on the table.",
    "Preview a grasp on the red cube. Have you tried this object before?",
    "Pick up the red cube and put it in the box.",
    "Where is the red cube?",
    "Grab the red cube and throw it into the bin on the left!",
    "Wave goodbye, then go home.",
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--llm", default="local_qwen",
                   help="LLM profile to rehearse (local_qwen|anthropic|openai)")
    p.add_argument("--prompts", default=None,
                   help="file with one prompt per line (default: runbook beats)")
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--no-booth", action="store_true",
                   help="skip the CASCADE_BOOTH tuning overlay")
    args = p.parse_args()

    # explicit either way: an ambient CASCADE_BOOTH from the shell must not
    # silently invert what the flags promise
    os.environ["CASCADE_BOOTH"] = "0" if args.no_booth else "1"

    from cascade.agent.llm import MockLLM, make_llm
    from cascade.agent.orchestrator import AgentOrchestrator
    from cascade.apps.demo import build_runtime, shutdown_runtime
    from cascade.config import PACKAGE_ROOT, load_demo_config

    run_dir = PACKAGE_ROOT / "runs" / time.strftime("rehearsal_%Y%m%d_%H%M%S")
    cfg = load_demo_config(camera="mock", arm="mock", llm=args.llm)
    # rehearsal must never write into the booth's learned grasp priors
    cfg._data["grasp"]["memory_path"] = str(run_dir / "grasp_memory.json")

    prompts = DEFAULT_PROMPTS
    if args.prompts:
        prompts = [ln.strip() for ln in Path(args.prompts).read_text().splitlines()
                   if ln.strip()]

    llm = make_llm(cfg.llm)
    if isinstance(llm, MockLLM):
        print("[!] profile resolved to the scripted mock LLM -- "
              "this rehearses nothing; pass --llm local_qwen etc.")
        return 2

    runtime, arm = build_runtime(cfg, run_dir, view=False, serve=False)
    # no advisor, no fast tiers: the deliberation tier is what's on trial
    agent = AgentOrchestrator(llm, runtime, advisor=None,
                              max_steps=args.max_steps, decompose=True,
                              fast_planner=None)
    results: list[dict] = []
    try:
        for prompt in prompts:
            print(f"\n=== {prompt!r}", flush=True)
            t0 = time.monotonic()
            try:
                r = agent.run_task(prompt)
                row = {
                    "prompt": prompt, "success": r.success, "path": r.path,
                    "steps": r.steps, "duration_s": r.duration_s,
                    "tools": [e.get("tool") for e in (r.tool_log or [])],
                    "summary": (r.summary or "")[:200],
                }
            except Exception as e:  # a crash is itself a rehearsal finding
                row = {"prompt": prompt, "success": False, "tools": [],
                       "error": f"{type(e).__name__}: {e}",
                       "duration_s": round(time.monotonic() - t0, 1)}
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        shutdown_runtime(runtime, arm)

    out = run_dir / "rehearsal.json"
    out.write_text(json.dumps({
        "meta": {
            "llm_profile": args.llm,
            "model": cfg.llm.get("model", "?"),
            "max_steps": args.max_steps,
            "booth_overlay": not args.no_booth,
            "ran_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "results": results,
    }, indent=2, ensure_ascii=False))
    # crashes are reported separately: a crashed run may well have made
    # tool calls before dying, so it is not a "no tool calls" verdict
    crashed = [r for r in results if r.get("error")]
    no_tools = [r for r in results if not r["tools"] and not r.get("error")]
    succeeded = sum(1 for r in results if r.get("success"))
    print(f"\n[rehearsal] {succeeded}/{len(results)} prompts succeeded; "
          f"{len(no_tools)} produced no tool calls; {len(crashed)} crashed; "
          f"report -> {out}")
    for r in no_tools:
        print(f"[rehearsal] NO TOOL CALLS: {r['prompt']!r}")
    for r in crashed:
        print(f"[rehearsal] CRASHED: {r['prompt']!r} -- {r['error']}")
    return 1 if (no_tools or crashed) else 0


if __name__ == "__main__":
    sys.exit(main())
