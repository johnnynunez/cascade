#!/usr/bin/env python3
"""Read the current kitchen placement without moving or resetting anything."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", required=True)
    parser.add_argument("--destination", required=True, choices=("green square", "open box"))
    parser.add_argument("--port", type=int, default=8611)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from cascade.config import load_demo_config
    from cascade.sim.placement import read_placement

    cfg = load_demo_config(arm="isaac_kitchen_gpu", camera="isaac", llm="mock")
    verdict = read_placement((cfg.arm.get("bridge_host", "127.0.0.1"), args.port),
        cfg.arm.get("bridge_robot_id"), args.object, args.destination, evidence_dir=args.output)
    print(json.dumps(verdict, indent=2, allow_nan=False))
    return {"confirmed": 0, "refuted": 1, "unverified": 2}[verdict["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
