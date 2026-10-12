# SPDX-License-Identifier: Apache-2.0
"""Re-run the scenario verifier on a finished episode from its RECORDED final physics state (no simulation).

Used when a verifier definition changed after a run: the old verdict is kept under ``verification_at_run`` and the
new one is labelled with the config it came from.
    python scripts/microduck_agentic/reverify.py --run runs/stage_001 --scenario configs/microduck_agentic/stage_edge.yaml
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from cascade.apps.microduck_agentic import load_scenario, verify


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--scenario", type=Path, required=True)
    a = p.parse_args()
    rec = np.load(a.run / "recording.npz")
    # verify() only needs the verify spec; scenarios with an external scene skip the scene resolution
    scen = load_scenario(a.scenario) if "scene_env" not in a.scenario.read_text() else __import__("yaml").safe_load(
        a.scenario.read_text())
    names = [str(n) for n in rec["robot_names"]]
    finals = {}
    for i, n in enumerate(names):
        root = rec["body_pose"][-1, i, 0]
        g = rec["proj_grav"][-1, i]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -float(g[2])))))
        finals[n] = {"x": float(root[0]), "y": float(root[1]), "z": float(root[2]),
                     "fallen": bool(tilt > 55.0 or root[2] < 0.06)}
    ep_path = a.run / "episode.json"
    ep = json.loads(ep_path.read_text())
    ep.setdefault("verification_at_run", ep.get("verification"))
    ep["verification"] = verify(scen, finals)
    ep["verification_source"] = f"reverify.py with {a.scenario.name} on the recorded final state"
    ep_path.write_text(json.dumps(ep, indent=1, default=str) + "\n")
    print(json.dumps(ep["verification"], indent=1))


if __name__ == "__main__":
    main()
