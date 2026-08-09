"""What ARE the gross place failures?

Median aim error is 1.8 cm, but the tail is 15-30 cm and it did not move when
the held-object offset was compensated. Three episodes also never released at
all. Those two facts together suggest the tail is not "placement was
imprecise" but "something else went wrong entirely".

Rather than guess, classify. For each task, record where the failure happened
and what the skill said, so the tail resolves into named modes instead of one
number.

Run:
    WRC_BENCH_LIBERO=~/bench/LIBERO-PRO \
      PYTHONPATH=~/bench/LIBERO-PRO:$PWD/src \
      ~/.venvs/libero/bin/python scripts/classify_place_failures.py [suite] [n]
"""

import os
import sys
import threading

sys.path.insert(0, os.path.expanduser("~/Projects/demo/wrc_demo/src"))
sys.path.insert(0, os.path.expanduser("~/Projects/demo/wrc_demo/benchmark"))
sys.path.insert(0, os.path.expanduser("~/Projects/demo/wrc_demo/benchmark/libero"))

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

import run_wrc

SUITE = sys.argv[1] if len(sys.argv) > 1 else "libero_spatial"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10


def episode(tid):
    bm = benchmark.get_benchmark_dict()[SUITE]()
    task = bm.get_task(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder,
                        task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, controller="JOINT_POSITION",
                             camera_heights=256, camera_widths=256,
                             camera_depths=True)
    env.seed(0)
    obs = env.reset()
    inner = env.env
    for _ in range(10):
        obs, *_ = env.step(np.zeros(inner.action_dim))

    rt, arm, kin = run_wrc.build_wrc_runtime(env, task.language, "oracle")
    if getattr(arm, "camera", None) is not None:
        arm.camera.publish(obs)

    obj_body, dest_body = run_wrc.resolve_task_objects(inner, task.language, bddl)
    if not obj_body or not dest_body:
        env.close()
        return None

    def xyz(n):
        return np.array(inner.sim.data.body_xpos[inner.sim.model.body_name2id(n)])

    def seed():
        for n in (obj_body, dest_body):
            p = xyz(n)
            rt.beliefs.update(label=n, position=p, conf=0.99,
                              extent=np.array([0.06, 0.06, 0.06]),
                              top_z=float(p[2]) + 0.03)

    rt.attach_verifier(object_pose=lambda n: xyz(n) if n else None)
    rt.effects = None
    seed()
    stop = threading.Event()

    def pub():
        while not stop.is_set():
            try:
                seed()
            except Exception:
                pass
            stop.wait(0.5)

    threading.Thread(target=pub, daemon=True).start()

    obj0 = xyz(obj_body)
    lifted = {"max_z": float(obj0[2])}
    real_step = type(arm)._step

    def spy_step(self, a):
        out = real_step(self, a)
        try:
            lifted["max_z"] = max(lifted["max_z"], float(xyz(obj_body)[2]))
        except Exception:
            pass
        return out

    type(arm)._step = spy_step
    try:
        r = rt.execute("pick_and_place",
                       {"object": obj_body, "destination": dest_body})
    finally:
        type(arm)._step = real_step
    stop.set()

    for _ in range(40):
        arm._step(np.zeros(8))

    obj1, dest1 = xyz(obj_body), xyz(dest_body)
    aim = r.get("placed_at") or r.get("at")
    err = r.get("error") or ""
    stage = r.get("stage") or ("ok" if r.get("ok") else "?")

    # Classify.
    moved = float(np.linalg.norm(obj1 - obj0))
    lift = lifted["max_z"] - float(obj0[2])
    gap = float(np.linalg.norm(obj1[:2] - dest1[:2]))
    if "grasp" in str(stage):
        mode = "GRASP FAILED"
    elif lift < 0.02:
        mode = "NEVER LIFTED"
    elif aim is None:
        mode = "NO RELEASE (aborted mid-place)"
    else:
        aim_err = float(np.linalg.norm(obj1[:2] - np.asarray(aim, float)[:2]))
        if aim_err > 0.10:
            mode = "DROPPED IN TRANSIT"
        elif gap > 0.10:
            mode = "AIMED WRONG"
        elif gap > 0.03:
            mode = "NEAR MISS"
        else:
            mode = "ON TARGET"
    out = {"task": tid, "mode": mode, "lift_cm": lift * 100,
           "moved_cm": moved * 100, "gap_cm": gap * 100,
           "stage": stage, "error": str(err)[:70]}
    env.close()
    return out


rows = []
print(f"suite: {SUITE}\n")
print(f"{'task':>4} {'mode':<30} {'lift':>7} {'moved':>8} {'gap':>8}")
for tid in range(N):
    try:
        r = episode(tid)
    except Exception as e:
        print(f"{tid:4d} EPISODE ERROR {type(e).__name__}: {str(e)[:50]}")
        continue
    if r is None:
        continue
    rows.append(r)
    print(f"{r['task']:4d} {r['mode']:<30} {r['lift_cm']:6.1f}cm "
          f"{r['moved_cm']:7.1f}cm {r['gap_cm']:7.1f}cm", flush=True)
    if r["error"]:
        print(f"     ! {r['error']}")

print(f"\n{'mode':<32} {'count':>6}")
from collections import Counter
for mode, n in Counter(r["mode"] for r in rows).most_common():
    print(f"{mode:<32} {n:6d}")
