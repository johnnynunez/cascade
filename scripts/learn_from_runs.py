#!/usr/bin/env python3
"""Offline learner: turn yesterday's runs into today's priors.

This is the OUTER loop of the system. The inner loop (the agent) acts and
leaves ASPIRE traces; this script reads those traces while nothing is running
and folds them into the two persistent models the next session starts from:

1. ``OperatingEnvelope`` (Harness-VLA) -- per-primitive outcome statistics
   and failure signatures from the selected traces.
2. ``SkillLibrary`` (ASPIRE) -- matching, measured-confirmed retry
   associations distilled into scoped markdown guidance, not causal repairs.

Nothing here touches the robot, so it is safe to run from cron, from a Hermes
scheduled job, or by hand between demo sessions:

    # inspect after a session, without writing notes or envelope updates
    python scripts/learn_from_runs.py --dry-run --report

    # nightly, from Hermes
    python scripts/learn_from_runs.py --json

This runs as a separate process so harvesting does not change guidance
mid-session. Better future performance still requires independent trials.
See docs/DREAM_RSI_ADAPTATION.md for admission rules, legacy traces and the
separate envelope path. --report alone writes updates; --dry-run may create
an empty library directory, and --export-md always writes its requested file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cascade.agent.aspire import harvest, report  # noqa: E402
from cascade.memory.envelope import OperatingEnvelope  # noqa: E402
from cascade.skills.library import PROMOTION_MIN_TASKS, SkillLibrary  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Fold run traces into learned priors.")
    ap.add_argument("--runs", default=str(REPO / "runs"),
                    help="directory holding <run>/trace.jsonl (default: repo runs/)")
    ap.add_argument("--library", default=str(REPO / "skills_library"),
                    help="ASPIRE skill library directory")
    ap.add_argument("--min-tasks", type=int, default=PROMOTION_MIN_TASKS,
                    help="distinct tasks a note must recur in before it is promoted "
                         f"(default {PROMOTION_MIN_TASKS}, upstream ASPIRE's rule; 1 = retrieve "
                         "after one confirmed retry, explicit relaxation)")
    ap.add_argument("--envelope", default="~/.cascade/envelope.json",
                    help="persisted operating-envelope model")
    ap.add_argument("--limit", type=int, default=200, help="most recent N runs")
    ap.add_argument("--report", action="store_true", help="print the diagnosis board")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--dry-run", action="store_true", help="diagnose without writing")
    ap.add_argument("--export-md", default="", help="also write MEMORY.md-style envelope export here")
    args = ap.parse_args()
    if args.min_tasks < 1:
        ap.error("--min-tasks must be >= 1 (1 is the explicit single-observation mode)")

    runs_dir = Path(args.runs).expanduser()
    if not runs_dir.exists():
        print(f"no runs directory at {runs_dir}", file=sys.stderr)
        return 1

    envelope = OperatingEnvelope(path=None if args.dry_run else args.envelope)
    ingested = envelope.ingest_runs(runs_dir, limit=args.limit)

    library = SkillLibrary(args.library, min_tasks=args.min_tasks)
    learned = (
        {"learned": 0, "entries": [], "note": "dry run", "library": library.summary()}
        if args.dry_run
        else harvest(runs_dir, library, limit=args.limit)
    )

    out = {
        "runs_dir": str(runs_dir),
        "traces_ingested": ingested,
        "skills": learned,
        "envelope": envelope.stats(),
    }

    if args.export_md:
        Path(args.export_md).expanduser().write_text(envelope.export_markdown())
        out["exported"] = args.export_md

    if args.json:
        print(json.dumps(out, indent=2))
        return 0

    print(f"runs dir      : {runs_dir}")
    print(f"traces folded : {ingested['traces']} traces, {ingested['records']} calls")
    print(f"skills learned: {learned.get('learned', 0)} {learned.get('entries', [])}")
    lib = learned.get("library") or {}
    if lib:
        print(f"skill library : {lib['entries']} notes, {lib['promoted']} promoted, "
              f"{lib['candidates']} candidates (promotion needs >= {lib['min_tasks']} distinct tasks)")
    hist = learned.get("failure_histogram") or {}
    if hist:
        print("\nfailure histogram:")
        for sig, n in hist.items():
            print(f"  {sig:34} {n}")
    env_digest = envelope.envelope_digest()
    if env_digest:
        print("\nlearned operating envelopes:")
        print(env_digest)
    fail_digest = envelope.failure_digest()
    if fail_digest:
        print("\nfailure models:")
        print(fail_digest)
    if args.report:
        print("\n" + report(runs_dir, limit=args.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
