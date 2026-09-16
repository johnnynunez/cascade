"""_check_success is True even though the bowl was never touched. Why?

After resetting `_last_term` AND reading `_check_success()` live, the harness
still scores done=True on episodes where the grasp failed in 1.3 s with zero
detections. But the standalone probe found the flag False at t=0 for 12/12
episodes. So something between t=0 and the scoring turns it True.

Difference between the two: the standalone probe stepped the raw env with
`np.zeros(7)`. The harness steps through LiberoArm._step with `np.zeros(8)`
and, crucially, runs a SKILL first -- which drives the arm through IK,
gripper commands, and 8 grasp attempts.

Candidate: the action's gripper channel. LIBERO's OSC controller takes 7 dims
(6 pose + 1 gripper); an 8-dim action, or a gripper sign convention, may make
the arm sweep the table and knock the bowl onto the plate -- which IS the task
success condition. "The robot achieved the goal by accident" would score
exactly like this: skill fails, predicate true.

Measure it: track the bowl and plate positions through one episode and see
whether the bowl actually ends up on the plate.
"""
import os
import sys

sys.path.insert(0, os.path.expanduser(os.environ.get("CASCADE_BENCH_LIBERO", "~/bench/LIBERO")))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def unwrap(env):
    e = env
    for _ in range(6):
        if getattr(e, "_check_success", None):
            return e
        e = getattr(e, "env", None)
        if e is None:
            break
    return None


bm = benchmark.get_benchmark_dict()["libero_spatial"]()
task = bm.get_task(0)
env = OffScreenRenderEnv(
    bddl_file_name=os.path.join(get_libero_path("bddl_files"),
                                task.problem_folder, task.bddl_file),
    camera_heights=128, camera_widths=128,
)
inits = bm.get_task_init_states(0)
env.reset()
env.set_init_state(inits[0])

inner = unwrap(env)
sim = inner.sim
names = [sim.model.body_id2name(i) for i in range(sim.model.nbody)]
bowl = [n for n in names if "akita" in (n or "").lower()]
plate = [n for n in names if "plate" in (n or "").lower()]
print("bowl bodies :", bowl)
print("plate bodies:", plate)


def pos(n):
    return np.array(sim.data.body_xpos[sim.model.body_name2id(n)])


b0 = pos(bowl[0]) if bowl else None
p0 = pos(plate[0]) if plate else None
print(f"\nt=0   bowl {np.round(b0,3)}  plate {np.round(p0,3)}  "
      f"success={inner._check_success()}")

# settle exactly as the harness does
for _ in range(10):
    env.step(np.zeros(7))
b1 = pos(bowl[0])
print(f"settle bowl {np.round(b1,3)}  moved {np.linalg.norm(b1-b0)*100:.2f} cm  "
      f"success={inner._check_success()}")

# now step a while with zero action, as the scoring loop does
for i in range(1, 41):
    env.step(np.zeros(7))
    if i % 10 == 0:
        b = pos(bowl[0])
        print(f"  +{i:3d} steps bowl {np.round(b,3)} "
              f"moved {np.linalg.norm(b-b0)*100:5.2f} cm "
              f"success={inner._check_success()}")

print(f"\nfinal success predicate: {inner._check_success()}")
print("if this is True with the bowl unmoved, the predicate itself is the bug")
env.close()
