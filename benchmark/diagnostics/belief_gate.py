"""Why is the seeded belief rejected? Instrument the actual decision.

run_wrc.py seeds the belief store with the target's true pose before each
episode, and `_localize`'s docstring says a detector miss falls back to a fresh
belief. Yet every episode dies with "no detections", so the fallback is not
firing.

The gate is runtime.py:570-576:

    mem_ok = belief is not None
             and (ref_t - belief.last_seen_t) <= max_age
             and <exact colour match>

with max_age from perception_loop.belief_fallback_age_s (default 3.0 s) and
ref_t taken from _last_reobserve_t, else _motion_t0, else now.

Three ways this rejects a perfectly good seeded belief, and they need telling
apart before anything is changed:

  a) the belief is not found at all (label mismatch between what seed_beliefs
     writes and what _resolve_query looks up)
  b) it is found but judged STALE -- a failed re-scan inside the grasp retry
     loop sets _last_reobserve_t to now, and 8 attempts over 12 s will push
     ref_t well past a 3 s window
  c) it is found and fresh but rejected on the COLOUR requirement -- the seeded
     belief has colour None, and the comment says grasping from memory demands
     an exact colour match

Wrap the real methods and record which branch fires. No behaviour change.
"""
import os
import sys
import time

REPO = "/home/johnny/Projects/demo/cascade"
sys.path.insert(0, REPO + "/src")
sys.path.insert(0, os.path.expanduser("~/bench/LIBERO"))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from cascade.memory import BeliefStore

LOG = []

# --- instrument BeliefStore.find: is the label even resolvable?
_orig_find = BeliefStore.find


def find_traced(self, query, *a, **k):
    out = _orig_find(self, query, *a, **k)
    LOG.append(("find", str(query)[:40], out is not None))
    return out


BeliefStore.find = find_traced

# --- instrument the staleness inputs on the runtime
from cascade.skills import runtime as RT  # noqa: E402

_orig_loc = RT.SkillRuntime._localize


def loc_traced(self, query, spatial_hint=None):
    b = self.beliefs.find(query)
    ref = (self._last_reobserve_t if self._last_reobserve_t is not None
           else self._motion_t0 if self._motion_t0 is not None
           else time.monotonic())
    age = (ref - b.last_seen_t) if b is not None else None
    LOG.append(("localize", str(query)[:40],
                f"belief={'YES' if b else 'NO'} "
                f"age={age if age is None else round(age, 2)}s "
                f"colour={getattr(b, 'color', None) if b else '-'}"))
    return _orig_loc(self, query, spatial_hint)


RT.SkillRuntime._localize = loc_traced

max_age_default = 3.0
print(f"belief_fallback_age_s default: {max_age_default}\n")

# run one episode through the real harness
import runpy  # noqa: E402

sys.argv = ["run_wrc.py", "--suite", "libero_spatial",
            "--tasks", "1", "--episodes", "1", "--conditions", "skill_only"]
try:
    runpy.run_path(REPO + "/benchmark/libero/run_wrc.py", run_name="__main__")
except SystemExit:
    pass
except Exception as e:
    print(f"run raised {type(e).__name__}: {str(e)[:120]}")

print("\n=== trace ===")
for kind, q, info in LOG[:25]:
    print(f"  {kind:9} {q:42} {info}")

finds = [x for x in LOG if x[0] == "find"]
hits = sum(1 for x in finds if x[2])
print(f"\nbelief lookups: {len(finds)}, hits: {hits}")
locs = [x for x in LOG if x[0] == "localize"]
if locs:
    print("localize calls and what the belief looked like:")
    for _, q, info in locs[:8]:
        print(f"   {q}  {info}")
