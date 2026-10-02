#!/usr/bin/env python3
"""Read an actual Arena result; absent observer bindings remain unverified."""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cascade.eval.arena import ARENA_REVISION, import_experiment  # noqa: E402
from cascade.eval.trials import verify_episode  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--source-revision", default=ARENA_REVISION)
    args = parser.parse_args()
    episodes = import_experiment(args.result, source_revision=args.source_revision)
    print(json.dumps([{"episode": asdict(episode), "verification": asdict(verify_episode(
        episode, required_checks=frozenset({"physical_effect"})))} for episode in episodes], indent=2))


if __name__ == "__main__":
    main()
