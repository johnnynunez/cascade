"""POSITIVE CONTROL: can _task_success ever return True?

The corrected sweep reports 0/38 across all four suites. Before believing that,
the instrument needs a positive control. I already proved the predicate is
False at t=0 on 12/12 episodes -- but a predicate that is ALWAYS False would
give exactly the same result, and would mean I replaced "always succeeds" with
"always fails": equally broken, opposite sign.

So: solve the task by hand and check the predicate flips.

`libero_spatial` task 0 is "pick up the black bowl between the plate and the
ramekin and place it on the plate". The goal predicate is the bowl being On()
the plate. Rather than driving the arm, teleport the bowl onto the plate in
MuJoCo qpos, settle, and read _check_success().

If it goes True -> the predicate works, and 0/38 is an honest measurement of a
skill that genuinely cannot do these tasks.
If it stays False -> my fix is broken and the whole sweep is void.
"""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/bench/LIBERO"))
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
            return None
    return None


bm = benchmark.get_benchmark_dict()["libero_spatial"]()
task = bm.get_task(0)
print("task:", task.language)

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

print("success at t=0:", inner._check_success())

# locate the bowl and the plate
names = [sim.model.body_id2name(i) for i in range(sim.model.nbody)]
bowl = next(n for n in names if "akita_black_bowl_1" in (n or ""))
plate = next(n for n in names if n and n.startswith("plate_1"))


def bpos(n):
    return np.array(sim.data.body_xpos[sim.model.body_name2id(n)])


b, p = bpos(bowl), bpos(plate)
print(f"bowl  {np.round(b,3)}")
print(f"plate {np.round(p,3)}")

# find the bowl's free joint in qpos by matching its position
jid = None
for j in range(sim.model.njnt):
    if sim.model.jnt_type[j] != 0:            # 0 = free joint
        continue
    adr = sim.model.jnt_qposadr[j]
    q = sim.data.qpos[adr:adr + 3]
    if np.linalg.norm(q - b) < 0.05:
        jid, qadr = j, adr
        break

if jid is None:
    print("could not locate the bowl free joint; aborting")
    sys.exit(1)

print(f"bowl free joint {jid}, qpos[{qadr}:{qadr+7}]")

# teleport the bowl just above the plate and let it settle onto it
target = np.array([p[0], p[1], p[2] + 0.06])
sim.data.qpos[qadr:qadr + 3] = target
sim.data.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]
sim.data.qvel[sim.model.jnt_dofadr[jid]:sim.model.jnt_dofadr[jid] + 6] = 0
sim.forward()

for i in range(120):
    env.step(np.zeros(7))

b2 = bpos(bowl)
ok = inner._check_success()
print(f"\nafter teleporting the bowl onto the plate:")
print(f"  bowl  {np.round(b2,3)}   plate {np.round(bpos(plate),3)}")
print(f"  _check_success() = {ok}")

print()
if ok:
    print("POSITIVE CONTROL PASSED — the predicate does fire when the task is")
    print("actually solved. The 0/38 sweep is an honest measurement.")
else:
    print("POSITIVE CONTROL FAILED — the predicate never returns True.")
    print("The sweep is VOID and the fix needs rework.")
env.close()
