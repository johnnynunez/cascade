#!/usr/bin/env python3
"""Offline learner: turn yesterday's runs into today's priors.

This is the OUTER loop of the system. The inner loop (the agent) acts and
leaves ASPIRE traces; this script reads those traces while nothing is running
and folds them into the two persistent models the next session starts from:

1. ``OperatingEnvelope``  (Harness-VLA) -- where each primitive is proven to
   work and how it usually fails, learned from every recorded outcome.
2. ``SkillLibrary``       (ASPIRE)      -- validated repairs distilled into
   retrievable markdown guidance.

Nothing here touches the robot, so it is safe to run from cron, from a Hermes
scheduled job, or by hand between demo sessions:

    # after a session
    python scripts/learn_from_runs.py --report

    # nightly, from Hermes
    python scripts/learn_from_runs.py --json

Design note: this deliberately runs as a separate process rather than inside
the agent. Learning mid-demo would change behaviour under the audience's feet
and burn booth seconds; learning between sessions makes each morning's robot
strictly better than last night's without a single retrained weight.
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
from cascade.skills.library import SkillLibrary  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Fold run traces into learned priors.")
    ap.add_argument("--runs", default=str(REPO / "runs"),
                    help="directory holding <run>/trace.jsonl (default: repo runs/)")
    ap.add_argument("--library", default=str(REPO / "skills_library"),
                    help="ASPIRE skill library directory")
    ap.add_argument("--envelope", default="~/.cascade/envelope.json",
                    help="persisted operating-envelope model")
    ap.add_argument("--limit", type=int, default=200, help="most recent N runs")
    ap.add_argument("--report", action="store_true", help="print the diagnosis board")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--dry-run", action="store_true", help="diagnose without writing")
    ap.add_argument("--export-md", default="", help="also write MEMORY.md-style envelope export here")
    args = ap.parse_args()

    runs_dir = Path(args.runs).expanduser()
    if not runs_dir.exists():
        print(f"no runs directory at {runs_dir}", file=sys.stderr)
        return 1

    envelope = OperatingEnvelope(path=None if args.dry_run else args.envelope)
    ingested = envelope.ingest_runs(runs_dir, limit=args.limit)

    library = SkillLibrary(args.library)
    learned = (
        {"learned": 0, "entries": [], "note": "dry run"}
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
