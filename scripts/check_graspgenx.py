#!/usr/bin/env python3
"""Require a real model and finite diffusion grasps, not just an open ZMQ port.

This is an inference readiness check. Physical pick/place acceptance is separate.
Only the lightweight Cascade client dependencies are imported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cascade.grasping.graspgenx_backend import GraspGenXClient, GraspGenXError


def check(host="127.0.0.1", port=5556, timeout_ms=120000):
    client = GraspGenXClient(host=host, port=port, timeout_ms=timeout_ms)
    try:
        health = client.probe(timeout_ms=timeout_ms)
        if health.get("stub"):
            raise GraspGenXError("analytic protocol stub cannot satisfy learned-model readiness")
        metadata = client.request({"action": "metadata"})
        if not metadata.get("model", {}).get("generator_backbone"):
            raise GraspGenXError("server did not identify its learned generator")
        # Deterministic box surface, in meters. No simulator or ground truth.
        rng = np.random.default_rng(2026)
        points = rng.uniform(-0.025, 0.025, (1024, 3)).astype(np.float32)
        faces = np.arange(len(points)) % 6
        points[np.arange(len(points)), faces // 2] = np.where(faces % 2, 0.025, -0.025)
        points += np.array([0.25, 0.0, 0.04], dtype=np.float32)
        payload = {
            "action": "infer_object", "point_cloud": points, "planner": "diffusion",
            "num_grasps": 100,
            "sweep_volume_params": {
                "extents_open": [0.09, 0.02, 0.045], "offset_open": [0, 0, .0755],
                "extents_mid": [0.045, 0.02, 0.045], "offset_mid": [0, 0, .0755],
                "gripper_type": 0, "fingertip_depth": .098,
            },
        }
        started = time.monotonic()
        response = client.request(payload)
        poses = np.asarray(response.get("grasps", []))
        scores = np.asarray(response.get("confidences", []))
        tags = response.get("branch_tags", [])
        if (poses.ndim != 3 or poses.shape[1:] != (4, 4) or not len(poses)
                or scores.shape != (len(poses),) or len(tags) != len(poses)
                or any(tag != "diff" for tag in tags)
                or not np.all(np.isfinite(poses)) or not np.all(np.isfinite(scores))):
            raise GraspGenXError("model did not return finite diffusion grasps with provenance")
        return {"ok": True, "learned": True, "health": health, "metadata": metadata,
                "diffusion_grasps": len(poses), "score_max": float(scores.max()),
                "wall_seconds": round(time.monotonic() - started, 3),
                "timing": response.get("timing", {})}
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--timeout-ms", type=int, default=120000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(args.host, args.port, args.timeout_ms)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
