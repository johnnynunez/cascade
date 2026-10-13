# SPDX-License-Identifier: Apache-2.0
"""Skate world smoke test: push / carve / coast / brake with the skating actor; prints speeds per phase.
    python scripts/microduck_agentic/skate_smoke.py --output runs/skate_smoke --num 1 [--course demo/.../skate_plaza.usda]
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
for extra in filter(None, os.environ.get("CASCADE_APPEND_SYSPATH", "").split(os.pathsep)):
    sys.path.append(extra)


def main():
    from isaaclab.app import add_launcher_args, launch_simulation

    home = Path.home()
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num", type=int, default=1)
    p.add_argument("--course", type=Path, default=None)
    p.add_argument("--policy", default="natural_roller")
    p.add_argument("--sweep", type=float, nargs="*", default=None,
                   help="start-from-standstill sweep: one robot per throttle value, 6 s push, report speeds")
    p.add_argument("--gtceu_root", type=Path, default=home / "Projects/microducks/train/cascade/gtceu_main")
    p.add_argument("--roller_usd", type=Path, default=home / "Projects/microducks/train/skillsft/assets/microduck_rollers.usd")
    add_launcher_args(p)
    p.set_defaults(visualizer=[])
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    from cascade.sim.microduck_skate_world import SkatePolicy, SkateSpawn, build_skate_cfg, create_skate_world

    onnx = {"natural_roller": "skating_natural/natural_roller.onnx", "oval_roller": "skating_race/oval_roller.onnx"}[a.policy]
    pol = SkatePolicy(a.policy, a.gtceu_root / "policies/skills" / onnx)
    if a.sweep:
        a.num = len(a.sweep)
    spawns = [SkateSpawn(f"duck_{i}", 0.0, -0.8 * i, 0.0) for i in range(a.num)]
    cfg = build_skate_cfg(gtceu_root=a.gtceu_root, roller_usd=a.roller_usd, course_usd=a.course, spawns=spawns,
                          device="cuda:0", seed=7)
    t0 = time.time()
    with launch_simulation(cfg, a):
        w = create_skate_world(cfg, gtceu_root=a.gtceu_root, spawns=spawns, policy=pol, roller_usd=a.roller_usd)
        if a.sweep:
            w.step(50)
            for i, thr in enumerate(a.sweep):
                w.set_control(i, "skate", (thr, 0.0, 0.0))
            trace = []
            for k in range(300):
                w.step(1)
                if k % 25 == 24:
                    trace.append([round(w.base_state(i)["speed"], 3) for i in range(w.n)])
            res = {"policy": a.policy, "throttles": a.sweep, "speed_every_0.5s": trace,
                   "distance_m": [round(math.hypot(w.base_state(i)["x"], w.base_state(i)["y"] + 0.8 * i), 2) for i in range(w.n)],
                   "fallen": [w.base_state(i)["fallen"] for i in range(w.n)]}
            print("SWEEP", json.dumps(res), flush=True)
            (a.output / "sweep.json").write_text(json.dumps(res, indent=1) + "\n")
            w.env.close()
            return
        phases = [("settle_coast", "coast", 0.0, 0.0, 1.0), ("push_0.5", "skate", 0.5, 0.0, 5.0),
                  ("carve_left_60deg", "skate", 0.5, math.radians(60), 4.0), ("coast", "coast", 0.0, None, 3.0),
                  ("brake_-0.4", "brake", -0.4, None, 3.0), ("push_0.8", "skate", 0.8, math.radians(60), 4.0),
                  ("brake_-0.5", "brake", -0.5, None, 4.0)]
        out = []
        for name, mode, thr, head, dur in phases:
            for i in range(w.n):
                st = w.base_state(i)
                w.set_control(i, mode, (thr, 0.0, head if head is not None else st["target_yaw_rad"]))
            speeds = []
            for _ in range(int(dur / 0.02)):
                w.step(1)
                speeds.append([w.base_state(i)["speed"] for i in range(w.n)])
            st = [w.base_state(i) for i in range(w.n)]
            row = {"phase": name, "end_speed": [round(s["speed"], 3) for s in st],
                   "mean_speed": [round(float(sum(sp[i] for sp in speeds) / len(speeds)), 3) for i in range(w.n)],
                   "yaw_deg": [round(math.degrees(s["yaw_rad"]), 1) for s in st],
                   "xy": [(round(s["x"], 2), round(s["y"], 2)) for s in st], "fallen": [s["fallen"] for s in st]}
            out.append(row)
            print("PHASE", json.dumps(row), flush=True)
        w.save(a.output / "recording.npz")
        (a.output / "smoke.json").write_text(json.dumps(out, indent=1) + "\n")
        print(f"SKATE SMOKE OK {w.step_index} steps {time.time() - t0:.1f} s", flush=True)
        w.env.close()


if __name__ == "__main__":
    main()
