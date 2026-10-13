# SPDX-License-Identifier: Apache-2.0
"""Honest skate onboard smoke (no LLM): push straight, carve, brake on a course; print the robot's own ToF summary
and pose estimate next to truth (truth printed for this check only).
    python scripts/microduck_agentic/skate_honest_smoke.py --course demo/microduck_agentic/skate_plaza.usda --output runs/x
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys

for extra in filter(None, os.environ.get("CASCADE_APPEND_SYSPATH", "").split(os.pathsep)):
    sys.path.append(extra)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main():
    from isaaclab.app import add_launcher_args, launch_simulation

    home = Path.home()
    p = argparse.ArgumentParser()
    p.add_argument("--course", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gtceu_root", type=Path, default=home / "Projects/microducks/train/cascade/gtceu_main")
    p.add_argument("--roller_usd", type=Path, default=home / "Projects/microducks/train/skillsft/assets/microduck_rollers.usd")
    add_launcher_args(p)
    p.set_defaults(visualizer=[])
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    from cascade.memory.robot_map import summarize_frame
    from cascade.sim.microduck_honest import HonestWorld, OnboardRig, SkateAdapter
    from cascade.sim.microduck_skate_world import CONTROL_DT_S, SkatePolicy, SkateSpawn, build_skate_cfg, create_skate_world

    pol = SkatePolicy("natural_roller", a.gtceu_root / "policies/skills/skating_natural/natural_roller.onnx")
    spawns = [SkateSpawn("duck_a", 0.0, 0.0, 0.0)]
    cfg = build_skate_cfg(gtceu_root=a.gtceu_root, roller_usd=a.roller_usd, course_usd=a.course.resolve(), spawns=spawns,
                          device="cuda:0", seed=7)
    with launch_simulation(cfg, a):
        w = create_skate_world(cfg, gtceu_root=a.gtceu_root, spawns=spawns, policy=pol, roller_usd=a.roller_usd)
        rig = OnboardRig(SkateAdapter(w), ["duck_a"], course_usd=a.course.resolve(), robot_usd=a.roller_usd,
                         dt=CONTROL_DT_S, log_path=a.output / "sensors.log", video_dir=a.output)
        for _ in range(50):
            w.step(1)
            rig.imu_tick()
        rig.start(w.time_s, [0.0])
        w.heading_source = lambda i: rig.onboard[i].yaw_f
        hw = HonestWorld(w, rig)
        s0 = w.base_state(0)
        rows = []
        for name, mode, thr, head, dur in [("push", "skate", 0.5, 0.0, 4.0), ("carve right", "skate", 0.5, -0.5, 3.0),
                                           ("brake", "brake", -0.4, None, 2.0)]:
            st = hw.base_state(0)
            hw.set_control(0, mode, (thr, 0.0, head if head is not None else st["target_yaw_rad"]))
            for k in range(int(dur / CONTROL_DT_S)):
                hw.step(1)
                if k % 25 == 24:
                    est = hw.base_state(0)
                    f = hw.tof_frame(0)
                    r = summarize_frame(f["distance_mm"], f["status"], f["sensor_pos_w"], f["beam_dir_w"], est["yaw_rad"],
                                        floor_z=0.0)
                    tr = w.base_state(0)
                    ob = rig.onboard[0]
                    Te = ob.T_E_cam()
                    bp = rig.a.body_pose()
                    Tt = rig.a.lens_T(bp, 0)
                    cam_dbg = {"cam_z_est": round(float(Te[2, 3]), 3), "cam_z_true": round(float(Tt[2, 3]), 3),
                               "pitch_est": round(math.degrees(math.asin(-Te[2, 2])), 1),
                               "pitch_true": round(math.degrees(math.asin(-Tt[2, 2])), 1),
                               "h_cur": round(float(ob.h_cur or 0), 3), "trunk_z_true": round(float(bp[0, rig.a.trunk, 2]), 3)}
                    row = {"phase": name, "t": round(w.time_s, 2), "est": [round(est["x"], 2), round(est["y"], 2),
                           round(math.degrees(est["yaw_rad"]), 1)], "truth": [round(tr["x"] - s0["x"], 2),
                           round(tr["y"] - s0["y"], 2), round(math.degrees(tr["yaw_rad"]), 1)],
                           "tilt_est": round(est["tilt_deg"], 1), "tilt_true": round(tr["tilt_deg"], 1),
                           "tof": r.text()[:90], **cam_dbg}
                    rows.append(row)
                    print("ROW", json.dumps(row), flush=True)
        rig.close()
        (a.output / "skate_honest_smoke.json").write_text(json.dumps(rows, indent=1) + "\n")
        print("SKATE HONEST SMOKE OK", rig.cam_frames, flush=True)
        w.env.close()


if __name__ == "__main__":
    main()
