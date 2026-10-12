# SPDX-License-Identifier: Apache-2.0
"""Recompute the approximate equipment clearance of a finished episode and store it in episode.json.
    python scripts/microduck_agentic/clearance.py --run runs/backstage
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from cascade.sim.microduck_clearance import apply_to_verification, clearance_report, obstacle_boxes


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--scenario", type=Path, default=None,
                   help="take the verify spec (no_contact) from this config instead of the run's own scenario.yaml")
    a = p.parse_args()
    scen = json.loads((a.run / "scenario_resolved.json").read_text())
    rec = np.load(a.run / "recording.npz")
    pos = rec["body_pose"][:: a.stride, :, :, :3].astype(np.float64)
    report = clearance_report(pos, [str(n) for n in rec["robot_names"]], obstacle_boxes(scen["scene_path"]))
    ep_path = a.run / "episode.json"
    ep = json.loads(ep_path.read_text())
    ep["clearance"] = report
    ep.setdefault("verification_at_run", ep.get("verification"))
    spec_file = a.scenario or (a.run / "scenario.yaml")
    verify_spec = yaml.safe_load(spec_file.read_text()).get("verify", {})
    ep["verification"] = apply_to_verification(ep.get("verification", {}), verify_spec, report)
    ep_path.write_text(json.dumps(ep, indent=1, default=str) + "\n")
    print(json.dumps({r: {k: v for k, v in x.items() if k != "per_obstacle_m"} for r, x in report.items()}, indent=1))
    print(json.dumps(ep["verification"], indent=1))


if __name__ == "__main__":
    main()
