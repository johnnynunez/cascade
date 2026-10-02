#!/usr/bin/env python3
"""Inspect a pinned VAB trial without simulator imports or actuation."""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cascade.eval.vab import VAB_REVISION, inspect_trial  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("task", type=Path)
    parser.add_argument("--init-index", type=int, default=0)
    parser.add_argument("--source-revision", default=VAB_REVISION)
    args = parser.parse_args()
    trial = inspect_trial(args.checkout, args.task, init_index=args.init_index,
                          source_revision=args.source_revision)
    print(json.dumps({"status": "inspected", "physical_admission": "unverified",
                      "trial": asdict(trial)}, default=str, indent=2))


if __name__ == "__main__":
    main()
