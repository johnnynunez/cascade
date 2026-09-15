"""Count _recentre_by_size calls during a real LIBERO run, via a wrapper.

run_wrc.py is a CLI, so instead of importing its internals, patch the function
from inside the same interpreter before handing control to its main().
"""
import os
import runpy
import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO + "/src")
sys.path.insert(0, REPO + "/benchmark")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

import cascade.perception.grounding as G

CALLS = {"n": 0}
_orig = G._recentre_by_size


def counted(*a, **k):
    CALLS["n"] += 1
    return _orig(*a, **k)


G._recentre_by_size = counted

sys.argv = ["run_wrc.py", "--suite", "libero_spatial",
            "--tasks", "1", "--episodes", "1", "--conditions", "verified"]

try:
    runpy.run_path(REPO + "/benchmark/libero/run_wrc.py", run_name="__main__")
except SystemExit:
    pass
except Exception as e:
    print(f"run raised: {type(e).__name__}: {str(e)[:150]}")

print(f"\n=== _recentre_by_size calls: {CALLS['n']} ===")
if CALLS["n"] == 0:
    print("LIBERO does NOT use the corrected path -> its published numbers")
    print("are unaffected by the perception fix.")
else:
    print("LIBERO DOES use it -> those numbers are stale.")
