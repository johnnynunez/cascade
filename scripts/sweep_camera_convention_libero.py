"""Which camera convention is correct? Test all of them against truth.

The first attempt recovered z within 3 cm but x,y off by 14-17 cm, which is
the signature of a wrong rotation convention rather than bad depth. Rather than
guess, enumerate the plausible conventions (flip matrices x image flip) and let
ground truth pick the winner. If none lands under a few cm, the problem is not
the convention and this reports that honestly instead of shipping the least-bad
option.
"""

import itertools
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
raw = obs["agentview_depth"].squeeze()
extent = sim.model.stat.extent
near = sim.model.vis.map.znear * extent
far = sim.model.vis.map.zfar * extent
depth_lin = (near / (1.0 - raw * (1.0 - near / far))).astype(np.float32)
rgb0 = obs["agentview_image"]
h, w = depth_lin.shape

cam_id = sim.model.camera_name2id("agentview")
fovy = sim.model.cam_fovy[cam_id]
f = 0.5 * h / np.tan(0.5 * np.deg2rad(fovy))
K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])
cam_pos = np.array(sim.data.cam_xpos[cam_id])
cam_rot = np.array(sim.data.cam_xmat[cam_id]).reshape(3, 3)

bowl = np.array(sim.data.body_xpos[sim.model.body_name2id(
    "akita_black_bowl_2_main")])
plate = np.array(sim.data.body_xpos[sim.model.body_name2id("plate_1_main")])
truth = {"bowl": bowl, "plate": plate}
print("truth bowl ", np.round(bowl, 4))
print("truth plate", np.round(plate, 4))

flips = {
    "identity": np.eye(3),
    "flip_yz": np.diag([1.0, -1.0, -1.0]),
    "flip_xy": np.diag([-1.0, -1.0, 1.0]),
    "flip_xz": np.diag([-1.0, 1.0, -1.0]),
}

results = []
for (fname, F), vflip in itertools.product(flips.items(), (False, True)):
    d = depth_lin[::-1].copy() if vflip else depth_lin.copy()
    rgb = rgb0[::-1].copy() if vflip else rgb0.copy()
    T = np.eye(4)
    T[:3, :3] = cam_rot @ F
    T[:3, 3] = cam_pos
    frame = Frame(rgb=np.ascontiguousarray(rgb[:, :, ::-1]), depth_m=d,
                  K=K, t=0.0, depth_source="sensor")

    best = {}
    for v_px in range(40, 220, 8):
        for u_px in range(40, 220, 8):
            try:
                fix = fix_from_pixel(frame, T, u=u_px, v=v_px)
            except Exception:
                continue
            if not (0.02 < float(max(fix.extent)) < 0.30):
                continue
            for name, xyz in truth.items():
                dd = float(np.linalg.norm(fix.position - xyz))
                if name not in best or dd < best[name]:
                    best[name] = dd
    if best:
        score = min(best.values())
        results.append((score, fname, vflip, best))

results.sort()
print("\nconvention                       vflip   best bowl   best plate")
for score, fname, vflip, best in results:
    print(f"  {fname:12}  {str(vflip):5}   "
          f"{best.get('bowl', float('nan'))*100:8.1f} cm  "
          f"{best.get('plate', float('nan'))*100:8.1f} cm")

if results and results[0][0] < 0.06:
    print(f"\nWINNER: {results[0][1]} vflip={results[0][2]} "
          f"({results[0][0]*100:.1f} cm)")
else:
    print("\nNO convention lands under 6 cm: the error is NOT the convention.")

env.close()
