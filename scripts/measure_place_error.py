"""Distribution of place error, not a single episode.

The layer-attribution run identified place precision as the blocking defect:
6.2 cm against LIBERO's 3 cm predicate. That figure came from ONE episode, so
it is an anecdote until it has a spread.

Also measures how far the arm shoves the destination while placing, since a
place that moves its own target invalidates the belief it aimed at.

Ground truth is the measuring instrument only; the skill runs with oracle
beliefs exactly as the benchmark does.

Run:
    PYTHONPATH=~/bench/LIBERO:$PWD/src \
      ~/.venvs/libero/bin/python scripts/measure_place_error.py [n_tasks]
"""

import os
import sys
import threading

sys.path.insert(0, os.path.expanduser("~/Projects/demo/wrc_demo/src"))
sys.path.insert(0, os.path.expanduser("~/bench/LIBERO"))
sys.path.insert(0, os.path.expanduser("~/Projects/demo/wrc_demo/benchmark"))
sys.path.insert(0, os.path.expanduser("~/Projects/demo/wrc_demo/benchmark/libero"))

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

import run_wrc

N_TASKS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
TOL = 0.03          # LIBERO's On() horizontal tolerance


def episode(tid):
    bm = benchmark.get_benchmark_dict()["libero_spatial"]()
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
    rt.effects = None          # bare path: we want the motion, not the veto

    obj_body, dest_body = run_wrc.resolve_task_objects(inner, task.language)
    if not obj_body or not dest_body:
        env.close()
        return None

    def xyz(name):
        return np.array(inner.sim.data.body_xpos[inner.sim.model.body_name2id(name)])

    def seed():
        for n in (obj_body, dest_body):
            p = xyz(n)
            rt.beliefs.update(label=n, position=p, conf=0.99,
                              extent=np.array([0.06, 0.06, 0.06]),
                              top_z=float(p[2]) + 0.03)

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

    dest_before = xyz(dest_body)
    r = rt.execute("pick_and_place",
                   {"object": obj_body, "destination": dest_body})
    stop.set()

    for _ in range(40):        # let physics settle before measuring
        arm._step(np.zeros(8))

    obj_after = xyz(obj_body)
    dest_after = xyz(dest_body)
    aim = r.get("placed_at") or r.get("at")
    out = {
        "task": tid,
        "ok": bool(r.get("ok")),
        "aim_err": (float(np.linalg.norm(obj_after[:2] - np.asarray(aim, float)[:2]))
                    if aim else None),
        "dest_shove": float(np.linalg.norm(dest_after[:2] - dest_before[:2])),
        "final_gap": float(np.linalg.norm(obj_after[:2] - dest_after[:2])),
        "z_above": float(obj_after[2] - dest_after[2]),
    }
    env.close()
    return out


rows = []
for tid in range(N_TASKS):
    try:
        r = episode(tid)
    except Exception as e:
        print(f"task {tid}: FAILED {type(e).__name__}: {e}", flush=True)
        continue
    if r is None:
        continue
    rows.append(r)
    print(f"task {r['task']}: aim_err "
          f"{(r['aim_err'] or float('nan'))*100:5.1f} cm   dest_shove "
          f"{r['dest_shove']*100:5.1f} cm   final_gap {r['final_gap']*100:5.1f} cm"
          f"   z {r['z_above']*100:+5.1f} cm", flush=True)

if rows:
    def col(k):
        return np.array([r[k] for r in rows if r[k] is not None], float)

    print(f"\nn = {len(rows)}   LIBERO tolerance = {TOL*100:.0f} cm horizontal")
    for k, label in (("aim_err", "placement error vs own aim"),
                     ("dest_shove", "destination displaced by arm"),
                     ("final_gap", "final object-to-destination")):
        v = col(k)
        if len(v):
            print(f"  {label:32} median {np.median(v)*100:5.1f} cm   "
                  f"min {v.min()*100:5.1f}   max {v.max()*100:5.1f}")
    gaps = col("final_gap")
    print(f"\n  episodes within tolerance: {(gaps < TOL).sum()}/{len(gaps)}")
