"""End-to-end check of the shipped code path on a real LIBERO frame.

The earlier comparison called ultralytics directly. This one goes through
`perception.segmenter.fix_at_pixel`, i.e. the exact function `grasp_at_pixel`
calls, so a wiring mistake between the measurement and the product cannot hide.

Ground truth chooses the pixel (standing in for an agent that can see the
object) and grades the answer. It never enters the perception path.

Run from the LIBERO venv:
    PYTHONPATH=~/bench/LIBERO:$PWD/src \
      ~/.venvs/libero/bin/python scripts/verify_segmenter_libero.py
"""

import os
import sys

sys.path.insert(0, os.path.expanduser("~/Projects/demo/cascade/src"))
sys.path.insert(0, os.path.expanduser("~/bench/LIBERO"))

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from cascade.perception.segmenter import PointSegmenter, fix_at_pixel
from cascade.types import Frame

MODEL = os.path.expanduser("~/Projects/demo/cascade/models/sam2.1_t.pt")
SUITE = "libero_spatial"
TARGETS = ("akita_black_bowl_2_main", "plate_1_main")


def main() -> int:
    bm = benchmark.get_benchmark_dict()[SUITE]()
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
    seg = PointSegmenter(model_path=MODEL)

    print(f"task: {task.language}")
    print(f"{'object':26} {'pixel':>12} {'error':>10} {'extent':>9}")
    print("-" * 62)

    worst = 0.0
    for name in TARGETS:
        xyz = np.array(sim.data.body_xpos[sim.model.body_name2id(name)])
        p_cam = (T_inv @ np.append(xyz, 1.0))[:3]
        u = int(round(K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]))
        v = int(round(K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]))
        try:
            fix = fix_at_pixel(frame, T, u, v, segmenter=seg)
            err = float(np.linalg.norm(fix.position - xyz))
            worst = max(worst, err)
            print(f"{name:26} {f'({u},{v})':>12} {err*100:8.1f} cm "
                  f"{max(fix.extent)*100:7.1f} cm")
        except Exception as e:
            print(f"{name:26} {f'({u},{v})':>12}      FAIL  {e}")
            worst = 99.0

    env.close()
    ok = worst < 0.06
    print(f"\nworst error {worst*100:.1f} cm -> {'PASS' if ok else 'FAIL'} "
          "(target < 6 cm, the grasp tolerance)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
