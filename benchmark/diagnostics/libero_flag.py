"""Is LIBERO's success flag already set before the robot does anything?

run_wrc.py clears `arm._terminated` at episode start (line 310) and then steps
10 zero actions to settle. If LIBERO's own success check is TRUE at that point,
every episode of that task is a free success and the primitive columns are
meaningless.

Check it directly across several tasks and initial states: reset, set_init_state,
settle, then read the env's success predicate BEFORE running any skill.

`OffScreenRenderEnv` wraps a robosuite env; the underlying task exposes
`_check_success()`. Read it through whatever layer is present.
"""
import os
import sys

REPO = "/home/johnny/Projects/demo/wrc_demo"
sys.path.insert(0, REPO + "/src")
# LIBERO lives as a source checkout, not an installed package. Point at it
# directly. Do NOT add REPO/benchmark: it has a `libero/` dir that shadows it.
sys.path.insert(0, os.path.expanduser("~/bench/LIBERO"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def success_now(env):
    """Read the task's own success predicate, whatever the wrapper depth."""
    e = env
    for _ in range(5):
        inner = getattr(e, "env", None)
        if inner is None:
            break
        e = inner
    fn = getattr(e, "_check_success", None)
    return bool(fn()) if callable(fn) else None


SUITE = "libero_spatial"
bm = benchmark.get_benchmark_dict()[SUITE]()

print(f"suite: {SUITE}")
print(f"{'task':<52} {'init':>5} {'success at t=0':>15}")

bad = total = 0
for tid in range(4):
    task = bm.get_task(tid)
    env = OffScreenRenderEnv(
        bddl_file_name=os.path.join(get_libero_path("bddl_files"),
                                    task.problem_folder, task.bddl_file),
        camera_heights=128, camera_widths=128,
    )
    inits = bm.get_task_init_states(tid)
    for ep in range(3):
        env.reset()
        obs = env.set_init_state(inits[ep % len(inits)])
        for _ in range(10):                      # same settle as the harness
            obs, _, _, _ = env.step(np.zeros(7))
        s = success_now(env)
        total += 1
        bad += int(bool(s))
        flag = "  <<< ALREADY DONE" if s else ""
        print(f"{task.language[:50]:<52} {ep:>5} {str(s):>15}{flag}")
    env.close()

print(f"\n{bad}/{total} episodes are already 'successful' before the robot moves")
if bad:
    print("-> the primitive columns in the LIBERO table are inflated by this")
else:
    print("-> the flag is clean at t=0; the inflation is elsewhere")
