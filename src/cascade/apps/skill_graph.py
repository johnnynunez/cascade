"""Execute a reviewed JSON skill graph through an explicit robot profile."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_robot_config
from ..robotics.graph import SkillGraph, run_skill_graph
from .robot_runtime import build_robot_runtime, robot_tool_descriptors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    graph = SkillGraph(json.loads(args.graph.read_text()))
    cfg = load_robot_config(args.robot)
    # Validate before constructing a runtime (and before lazy drivers can open).
    from ..robotics.resources import ResourceCatalog
    from .robot_runtime import describe_robot
    from types import SimpleNamespace
    domains = describe_robot(cfg)
    graph.validate(SimpleNamespace(tool_descriptors=robot_tool_descriptors(cfg),
                                   resources=ResourceCatalog([r for d in domains.values() for r in d.resources])))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    runtime, _ = build_robot_runtime(cfg, args.run_dir)
    try:
        result = run_skill_graph(graph, runtime)
    except KeyboardInterrupt:
        result = {"ok": False, "interrupted": "keyboard", "stop_receipt": runtime.stop(),
                  "physical_admission": False}
    finally:
        close_receipt = runtime.close()
    result["close_receipt"] = close_receipt
    if close_receipt.get("ok") is not True:
        result["ok"] = False
    output = args.run_dir / "graph-result.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"ok": result["ok"], "receipt": str(output), "physical_admission": False}))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
