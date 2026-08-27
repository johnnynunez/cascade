"""Does _last_term leak a previous episode's success into the next one?

The success flag at t=0 is clean (0/12), so the inflation is not the init
state. Reading the harness:

    run_wrc.py:310   arm._terminated = False     <- cleared each episode
    run_wrc.py:346   done = arm._last_term       <- but THIS decides success
    backend.py:93    self._last_term = bool(term)

`_last_term` is never reset between episodes. And backend.py:79-80 short
circuits: once `_terminated` is set, every later _step returns `(last_obs,
True)` WITHOUT stepping the sim -- so a skill that runs after a termination
gets a fake True.

The failure mode that implies: episode N succeeds (or hits LIBERO's step
limit), sets _last_term=True; episode N+1 starts, clears _terminated but not
_last_term, and if its skill fails fast -- as it does when the mock detector
finds nothing, 1.3 s -- line 346 reads the STALE True and scores it a success.

That is exactly the "grasp never happened, done=True" case I reproduced.

Test: drive the backend directly. Set a termination, start a "new episode" the
way run_wrc does, and read what line 346 would read.
"""
import os
import sys

REPO = "/home/johnny/Projects/demo/cascade"
sys.path.insert(0, REPO + "/src")
sys.path.insert(0, os.path.expanduser("~/bench/LIBERO"))
os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, REPO + "/benchmark/libero")
import backend as B  # noqa: E402


class FakeEnv:
    """Minimal env: terminates on demand, otherwise runs forever."""

    def __init__(self):
        self.term_next = False
        self.steps = 0

    def step(self, a):
        self.steps += 1
        return {"obs": 1}, 0.0, self.term_next, False, {}


arm = B.LiberoArm.__new__(B.LiberoArm)      # bypass __init__ wiring
arm.env = FakeEnv()
arm.last_obs = {"obs": 0}
arm._terminated = False
arm._last_term = False
arm.camera = None

# --- episode 1: terminates
arm.env.term_next = True
obs, term = arm._step([0] * 8)
print(f"episode 1 step -> term={term}  _last_term={arm._last_term}  "
      f"_terminated={arm._terminated}")

# --- episode 2: exactly what run_wrc.py does at line 310
arm._terminated = False              # the harness clears ONLY this
arm.env.term_next = False            # this episode will NOT terminate
arm.env.steps = 0

stale = getattr(arm, "_last_term", False)
print(f"\nepisode 2 begins. run_wrc.py:346 would read _last_term = {stale}")
if stale:
    print("  -> scored a SUCCESS before the robot has taken a single step")
    print("  -> CONFIRMED: _last_term leaks across episodes")
else:
    print("  -> clean")

# and the short-circuit
arm._terminated = True
n_before = arm.env.steps
obs, term = arm._step([0] * 8)
print(f"\nwith _terminated=True: _step returned term={term} after "
      f"{arm.env.steps - n_before} real env steps (0 = short-circuited)")
