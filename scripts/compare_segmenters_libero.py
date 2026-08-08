"""SAM3 vs SAM2.1 on the same LIBERO frame, same pixels, same scoring.

SAM2.1 already passes the 6 cm grasp tolerance (bowl 3.0 cm, plate 0.9 cm).
The question SAM3 has to answer is not "does it work" but "is it enough better
to justify 3.29 GB instead of 74 MB on a booth machine".

Both go through `perception.segmenter.fix_at_pixel`, the same function
`grasp_at_pixel` calls, so a wiring difference cannot masquerade as a model
difference. Ground truth picks the pixel and grades the answer; it never
enters the perception path.
"""

import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/Projects/demo/wrc_demo/src"))
sys.path.insert(0, os.path.expanduser("~/bench/LIBERO"))

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from wrc_demo.perception.segmenter import PointSegmenter, fix_at_pixel
from wrc_demo.types import Frame

MODELS = {
    "SAM2.1-t (74 MB)": os.path.expanduser(
        "~/Projects/demo/wrc_demo/models/sam2.1_t.pt"),
    "SAM3 (3.29 GB)": os.path.expanduser(
        "~/Projects/demo/wrc_demo/models/sam3.pt"),
}
TARGETS = ("akita_black_bowl_2_main", "plate_1_main")

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
raw = obs["agentview_depth"].squeeze()
extent = sim.model.stat.extent
near = sim.model.vis.map.znear * extent
far = sim.model.vis.map.zfar * extent
depth = (near / (1.0 - raw * (1.0 - near / far))).astype(np.float32)[::-1].copy()
rgb = obs["agentview_image"][::-1].copy()
h, w = depth.shape

cam_id = sim.model.camera_name2id("agentview")
f = 0.5 * h / np.tan(0.5 * np.deg2rad(sim.model.cam_fovy[cam_id]))
K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])
T = np.eye(4)
T[:3, :3] = (np.array(sim.data.cam_xmat[cam_id]).reshape(3, 3)
             @ np.diag([1.0, -1.0, -1.0]))
T[:3, 3] = np.array(sim.data.cam_xpos[cam_id])
T_inv = np.linalg.inv(T)

frame = Frame(rgb=np.ascontiguousarray(rgb[:, :, ::-1]), depth_m=depth,
              K=K, t=0.0, depth_source="sensor")

pixels = {}
for name in TARGETS:
    xyz = np.array(sim.data.body_xpos[sim.model.body_name2id(name)])
    p = (T_inv @ np.append(xyz, 1.0))[:3]
    pixels[name] = (int(round(K[0, 0] * p[0] / p[2] + K[0, 2])),
                    int(round(K[1, 1] * p[1] / p[2] + K[1, 2])), xyz)

print(f"task: {task.language}\n")
print(f"{'model':20} {'object':26} {'error':>9} {'extent':>9} {'ms':>7}")
print("-" * 78)

for mname, path in MODELS.items():
    if not os.path.exists(path):
        print(f"{mname:20} MISSING {path}")
        continue
    seg = PointSegmenter(model_path=path)
    for name, (u, v, xyz) in pixels.items():
        try:
            fix_at_pixel(frame, T, u, v, segmenter=seg)   # warm up
            t0 = time.perf_counter()
            fix = fix_at_pixel(frame, T, u, v, segmenter=seg)
            ms = (time.perf_counter() - t0) * 1000.0
            err = float(np.linalg.norm(fix.position - xyz))
            print(f"{mname:20} {name:26} {err*100:7.1f} cm "
                  f"{max(fix.extent)*100:7.1f} cm {ms:7.0f}")
        except Exception as e:
            print(f"{mname:20} {name:26}   FAILED  {type(e).__name__}: {e}")

env.close()
