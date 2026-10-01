"""Export a cuMotion candidate without opening a robot or simulator connection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

from ..planning import PlanningError, make_motion_planner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Planner YAML; model paths relative to this file")
    parser.add_argument("--start", type=float, nargs="+", required=True, help="Local joint positions in configured order, radians/metres")
    parser.add_argument("--goal", type=float, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file (existing files are never overwritten)")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists")
    try:
        config = yaml.safe_load(args.config.read_text())
        if not isinstance(config, dict):
            raise PlanningError("planner configuration must be a YAML mapping")
        for key in ("urdf", "xrdf"):
            path = Path(config[key])
            config[key] = str(path if path.is_absolute() else args.config.parent / path)
        with make_motion_planner(config) as planner:
            result = planner.plan(args.start, args.goal).as_dict()
        # Serialize before creating the destination; a rejected plan leaves no
        # success-shaped output. Exclusive creation also closes the path race.
        encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
        with args.output.open("x") as stream:
            stream.write(encoded)
    except (PlanningError, KeyError, TypeError, ValueError, OSError, yaml.YAMLError) as exc:
        print(f"Motion planning failed: {exc}", file=sys.stderr)
        return 1
    print(f"Saved cuMotion candidate to {args.output}; execution_authorized=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
