"""Instrument mem_ok itself: age is 0.0 s, so why is the fallback not taken?

The trace showed the belief IS found (21/21) and the first localize sees
age=0.0 s against a 3.0 s window, with r["color"] presumably None -- so
mem_ok should be True and `_fix_from_belief` should run.

It does not. So either:
  * the code never reaches the mem_ok block (an exception earlier in
    _localize, e.g. the frame grab, escapes before the fallback), or
  * _fix_from_belief itself raises and the error is reported as the original
    detector miss, or
  * r["color"] is NOT None for this query, failing the colour clause

Wrap _fix_from_belief and the frame grab to see which. Read-only.
"""
import os
import sys
import time
import traceback
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO + "/src")
sys.path.insert(0, os.path.expanduser(os.environ.get("CASCADE_BENCH_LIBERO", "~/bench/LIBERO")))
os.environ.setdefault("MUJOCO_GL", "egl")

from cascade.skills import runtime as RT

EVENTS = []

_orig_resolve = RT.SkillRuntime._resolve_query


def resolve_traced(self, query):
    r = _orig_resolve(self, query)
    EVENTS.append(("resolve", f"color={r.get('color')!r} "
                              f"belief={'YES' if r.get('belief') else 'NO'}"))
    return r


RT.SkillRuntime._resolve_query = resolve_traced

_orig_fix = RT.SkillRuntime._fix_from_belief.__func__ \
    if hasattr(RT.SkillRuntime._fix_from_belief, "__func__") \
    else RT.SkillRuntime._fix_from_belief


def fix_traced(query, belief):
    try:
        out = _orig_fix(query, belief)
        EVENTS.append(("fix_from_belief", f"OK pos={getattr(out,'position',None)}"))
        return out
    except Exception as e:
        EVENTS.append(("fix_from_belief", f"RAISED {type(e).__name__}: {str(e)[:80]}"))
        raise


RT.SkillRuntime._fix_from_belief = staticmethod(fix_traced)

_orig_loc = RT.SkillRuntime._localize


def loc_traced(self, query, spatial_hint=None):
    try:
        return _orig_loc(self, query, spatial_hint)
    except Exception as e:
        tb = traceback.format_exc()
        # which line of _localize did we die on?
        line = [l.strip() for l in tb.splitlines() if "runtime.py" in l]
        EVENTS.append(("localize_raise",
                       f"{type(e).__name__} at {line[-1][:90] if line else '?'}"))
        raise


RT.SkillRuntime._localize = loc_traced

import runpy  # noqa: E402

sys.argv = ["run_wrc.py", "--suite", "libero_spatial",
            "--tasks", "1", "--episodes", "1", "--conditions", "skill_only"]
try:
    runpy.run_path(REPO + "/benchmark/libero/run_wrc.py", run_name="__main__")
except SystemExit:
    pass
except Exception as e:
    print(f"harness raised {type(e).__name__}: {str(e)[:100]}")

print("\n=== events ===")
for k, v in EVENTS[:20]:
    print(f"  {k:16} {v}")

kinds = {}
for k, _ in EVENTS:
    kinds[k] = kinds.get(k, 0) + 1
print("\ncounts:", kinds)
if not any(k == "fix_from_belief" for k, _ in EVENTS):
    print("\n_fix_from_belief was NEVER CALLED -> mem_ok evaluated False,")
    print("or _localize died before reaching it.")
