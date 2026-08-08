"""Verify pixel addressing against physics truth on a real LIBERO frame.

The grid probe found object-sized regions without any class name. That is only
meaningful if the region actually corresponds to a real object, so this checks
the recovered 3D position against the simulator's own body pose.

Method: build the true camera extrinsics from the sim, lift a pixel on the
bowl to 3D, and compare against sim.data.body_xpos for the bowl. Ground truth
is used ONLY as the measuring instrument here, never as an input to the
perception path, which is exactly the eval_detector.py pattern.
"""

import os
import sys

sys.path.insert(0, os.path.expanduser("~/Projects/demo/wrc_demo/src"))
sys.path.insert(0, os.path.expanduser("~/bench/LIBERO"))

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from wrc_demo.perception.pixel_target import fix_from_pixel
from wrc_demo.types import Frame

bm = benchmark.get_benchmark_dict()["libero_spatial"]()
task = bm.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder,
                    task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, controller="JOINT_POSITION",
                         camera_heights=256, camera_widths=256,
                         camera_depths=True)
env.seed(0)
obs = env.reset()
for _ in range(10):
    obs, *_ = env.step(np.zeros(env.env.action_dim))

sim = env.env.sim
depth_raw = obs["agentview_depth"]
extent = sim.model.stat.extent
near = sim.model.vis.map.znear * extent
far = sim.model.vis.map.zfar * extent
depth_m = (near / (1.0 - depth_raw * (1.0 - near / far))).squeeze().astype(np.float32)
rgb = obs["agentview_image"]
h, w = depth_m.shape

cam_id = sim.model.camera_name2id("agentview")
fovy = sim.model.cam_fovy[cam_id]
f = 0.5 * h / np.tan(0.5 * np.deg2rad(fovy))
K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])

# MuJoCo camera: looks down -Z with +Y up. OpenCV: +Z forward, +Y down.
cam_pos = np.array(sim.data.cam_xpos[cam_id])
cam_rot = np.array(sim.data.cam_xmat[cam_id]).reshape(3, 3)
flip = np.diag([1.0, -1.0, -1.0])
T = np.eye(4)
T[:3, :3] = cam_rot @ flip
T[:3, 3] = cam_pos

# Rendered images are vertically flipped relative to the camera convention.
depth_m = depth_m[::-1].copy()
rgb = rgb[::-1].copy()

frame = Frame(rgb=np.ascontiguousarray(rgb[:, :, ::-1]), depth_m=depth_m,
              K=K, t=0.0, depth_source="sensor")

truth = {}
for name in ("akita_black_bowl_2_main", "plate_1_main"):
    try:
        truth[name] = np.array(sim.data.body_xpos[sim.model.body_name2id(name)])
    except Exception:
        pass

print("instruction:", task.language)
for k, v in truth.items():
    print(f"  truth {k:26} {np.round(v, 4)}")

print("\nsearching pixels for object-sized regions, then scoring vs truth:")
best = {}
for v_px in range(40, 220, 6):
    for u_px in range(40, 220, 6):
        try:
            fix = fix_from_pixel(frame, T, u=u_px, v=v_px)
        except Exception:
            continue
        size = float(max(fix.extent))
        if not (0.02 < size < 0.30):
            continue
        for name, xyz in truth.items():
            d = float(np.linalg.norm(fix.position - xyz))
            if name not in best or d < best[name][0]:
                best[name] = (d, (u_px, v_px), size, fix.position)

for name, (d, px, size, pos) in sorted(best.items()):
    verdict = "HIT" if d < 0.06 else ("near" if d < 0.15 else "miss")
    print(f"  {name:26} best pixel {px}  err {d*100:6.1f} cm  "
          f"size {size*100:4.1f} cm  [{verdict}]")
    print(f"      recovered {np.round(pos, 4)}   truth {np.round(truth[name], 4)}")

env.close()
