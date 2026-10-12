# SPDX-License-Identifier: Apache-2.0
"""Smoke test of the Isaac Lab MicroDuck world: every controller for a few seconds, no agent, no LLM."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.append(str(Path.home() / "Projects/microducks/train/cascade/pydeps"))

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

from cascade.sim.microduck_lab_world import PolicyFiles, RobotSpawn, build_env_cfg, create_world  # noqa: E402

H = Path.home() / "Projects/microducks/deliveries/johnny_handoff"
p = argparse.ArgumentParser()
p.add_argument("--gtceu_root", type=Path, default=Path.home() / "Projects/microducks/microduck_gtceu-final")
p.add_argument("--course", type=Path, default=None)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--nav", action="store_true")
add_launcher_args(p)
p.set_defaults(visualizer=[])
args = p.parse_args()

policies = PolicyFiles(
    walker_onnx=H / "microduck_johnny_navigation_v1/policies/walking_velocity_flat_existing/velocity_flat_existing.onnx",
    hero_pt=H / "microduck_johnny_perception_v1/policies/onboard_hero/onboard_hero.pt",
    getup_onnx=H / "microduck_johnny_balance_falls_v1/policies/getup_kitfloor_ft_model_1100/getup_kitfloor_ft_model_1100.onnx",
    nav_onnx=(H / "microduck_johnny_navigation_v1/policies/depth_navigation_v4_iter300/depth_navigation_v4.onnx")
    if args.nav else None,
)
spawns = [RobotSpawn("duck_a", 0.0, 0.0, 0.0), RobotSpawn("duck_b", 0.0, 0.6, 0.0),
          RobotSpawn("duck_c", 0.0, -0.6, 0.0, fallen="back")]
args.output.mkdir(parents=True, exist_ok=True)
device = getattr(args, "device", None) or "cuda:0"
cfg = build_env_cfg(gtceu_root=args.gtceu_root, course_usd=args.course, num_robots=len(spawns), device=device, seed=7,
                    nav_camera=args.nav)
t0 = time.time()
with launch_simulation(cfg, args):
    world = create_world(cfg, gtceu_root=args.gtceu_root, spawns=spawns, policies=policies, device=device, course_usd=args.course,
                         meta={"purpose": "smoke"})
    log = []

    def snap(tag):
        row = {"tag": tag, "t": world.time_s}
        for i, n in enumerate(world.names):
            s = world.base_state(i)
            row[n] = {k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()}
        log.append(row)
        print(json.dumps(row), flush=True)

    snap("start")
    world.step(50)
    snap("settled")
    world.set_control(0, "walk", (0.2, 0.0, 0.0))
    world.set_control(1, "onboard_hero")
    world.set_control(2, "getup")
    world.step(150)
    snap("walk3s/hero3s/getup3s")
    world.set_control(0, "walk", (0.0, 0.0, 0.8))
    world.step(100)
    snap("turn2s")
    if args.nav:
        world.set_control(0, "depth_nav", goal_xy=(1.5, 0.5))
        world.step(250)
        snap("nav5s")
    f = world.tof_frame(0)
    from PIL import Image
    for i in range(world.n):
        Image.fromarray(world.policy_rgb(i)).resize((240, 180), Image.NEAREST).save(args.output / f"rgb_{i}.png")
    print("tof mm\n", f["distance_mm"], "\nstatus\n", f["status"], flush=True)
    world.save(args.output / "smoke.npz", {"log": log, "wall_s": time.time() - t0})
    print("SMOKE OK", world.step_index, "steps", round(time.time() - t0, 1), "s", flush=True)
    world.env.close()
