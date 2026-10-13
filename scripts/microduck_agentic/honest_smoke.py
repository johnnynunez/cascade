# SPDX-License-Identifier: Apache-2.0
"""Honest onboard stack smoke (no agent, no LLM): one walker on a course; a fixed command script; prints the robot's
OWN estimate (cuVSLAM RGBD on the OVRTX head camera + ToF) next to simulator truth mapped into its start frame.
    python scripts/microduck_agentic/honest_smoke.py --course demo/microduck_agentic/backstage_obstacles.usda --output runs/h
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

for extra in filter(None, os.environ.get("CASCADE_APPEND_SYSPATH", "").split(os.pathsep)):
    sys.path.append(extra)

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

from cascade.sim.microduck_lab_world import CONTROL_DT_S, PolicyFiles, RobotSpawn, build_env_cfg, create_world  # noqa: E402

H = Path.home() / "Projects/microducks/deliveries/johnny_handoff"
p = argparse.ArgumentParser()
p.add_argument("--gtceu_root", type=Path, default=Path.home() / "Projects/microducks/microduck_gtceu-final")
p.add_argument("--course", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--turn_rate", type=float, default=0.5)
add_launcher_args(p)
p.set_defaults(visualizer=[])
args = p.parse_args()
policies = PolicyFiles(
    walker_onnx=H / "microduck_johnny_navigation_v1/policies/walking_velocity_flat_existing/velocity_flat_existing.onnx",
    hero_pt=H / "microduck_johnny_perception_v1/policies/onboard_hero/onboard_hero.pt",
    getup_onnx=H / "microduck_johnny_balance_falls_v1/policies/getup_kitfloor_ft_model_1100/getup_kitfloor_ft_model_1100.onnx")
spawns = [RobotSpawn("duck_a", 0.0, 0.0, 0.0)]
args.output.mkdir(parents=True, exist_ok=True)
device = getattr(args, "device", None) or "cuda:0"
cfg = build_env_cfg(gtceu_root=args.gtceu_root, course_usd=args.course.resolve(), num_robots=1, device=device, seed=7,
                    nav_camera=False)
t0 = time.time()
with launch_simulation(cfg, args):
    from cascade.sim.microduck_honest import HonestWorld, OnboardRig, WalkerAdapter

    world = create_world(cfg, gtceu_root=args.gtceu_root, spawns=spawns, policies=policies, device=device,
                         course_usd=args.course.resolve(), meta={"purpose": "honest smoke"})
    rig = OnboardRig(WalkerAdapter(world), ["duck_a"], course_usd=args.course.resolve(),
                     robot_usd=Path.home() / "Projects/microducks/microduck_gtceu/assets/MicroDuck.usd", dt=CONTROL_DT_S,
                     log_path=args.output / "sensors.log", video_dir=args.output)
    for _ in range(50):
        world.step(1)
        rig.imu_tick()
    rig.start(world.time_s, [0.0])
    s0 = world.base_state(0)
    hw = HonestWorld(world, rig)
    rows = []

    def report(tag):
        est = hw.base_state(0)
        tr = world.base_state(0)
        dx, dy = tr["x"] - s0["x"], tr["y"] - s0["y"]
        c, s = math.cos(s0["yaw_rad"]), math.sin(s0["yaw_rad"])
        tx, ty = c * dx + s * dy, -s * dx + c * dy
        tyaw = math.degrees((tr["yaw_rad"] - s0["yaw_rad"] + math.pi) % (2 * math.pi) - math.pi)
        row = {"tag": tag, "t": round(world.time_s, 2),
               "est": [round(est["x"], 3), round(est["y"], 3), round(math.degrees(est["yaw_rad"]), 1)],
               "truth": [round(tx, 3), round(ty, 3), round(tyaw, 1)],
               "err_m": round(math.hypot(est["x"] - tx, est["y"] - ty), 3), "loc": est["localization"],
               "features": est["features"], "tilt": round(est["tilt_deg"], 1), "z_est": round(est["z"], 3)}
        rows.append(row)
        print("EST", json.dumps(row), flush=True)

    report("start")
    script = [("walk", (0.2, 0.0, 0.0), 6.0), ("walk", (0.0, 0.0, args.turn_rate), 3.0), ("walk", (0.2, 0.0, 0.0), 5.0),
              ("walk", (0.0, 0.0, -args.turn_rate), 3.0), ("walk", (0.15, 0.0, 0.0), 4.0)]
    for mode, cmd, dur in script:
        hw.set_control(0, mode, cmd)
        for k in range(int(dur / CONTROL_DT_S)):
            hw.step(1)
            if k % 50 == 49:
                report(f"{mode} {cmd}")
    rig.close()
    (args.output / "honest_smoke.json").write_text(json.dumps({"rows": rows, "camera_frames": rig.cam_frames,
                                                               "calibration": rig.calibration}, indent=1) + "\n")
    print(f"HONEST SMOKE OK {rig.cam_frames} camera frames, {time.time() - t0:.1f} s wall", flush=True)
    world.env.close()
