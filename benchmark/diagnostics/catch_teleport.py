"""Catch whatever teleports the cube back to spawn, in the act.

`repro_retry_reset.py` proved the jump happens with NO skill running, ~9 s
after an off-spawn placement. That rules out the retry loop and points at
something inside the bridge process itself.

Strategy: monkey-patch RigidPrim.set_world_poses INSIDE the sim process to
log a stack trace on every call, then place the cube off-spawn and wait for
the jump. Whoever calls it will be named by the traceback.
"""
import sys
import time

import numpy as np

from cascade.sim.bridge_client import BridgeClient
from cascade.sim.truth import TruthPoseReader

c = BridgeClient(port=8611)
c.connect()
truth = TruthPoseReader(c)

INSTRUMENT = r"""
import traceback
from isaacsim.core.experimental.prims import RigidPrim

if not globals().get("_POSE_HOOK_INSTALLED"):
    _orig_swp = RigidPrim.set_world_poses

    def _logged_swp(self, *a, **k):
        try:
            path = str(getattr(self, "paths", ""))[:80]
        except Exception:
            path = "?"
        stack = "".join(traceback.format_stack()[-6:-1])
        globals().setdefault("_POSE_CALLS", []).append((path, stack))
        return _orig_swp(self, *a, **k)

    RigidPrim.set_world_poses = _logged_swp
    globals()["_POSE_HOOK_INSTALLED"] = True
    globals()["_POSE_CALLS"] = []
    print("hook installed")
else:
    print("hook already installed")
"""

print("=== installing pose hook in the sim process ===")
r = c.request({"op": "exec", "code": INSTRUMENT})
print(" ", r.get("stdout", "").strip())

print("\n=== clearing log, placing cube off-spawn ===")
c.request({"op": "exec", "code": "globals()['_POSE_CALLS'] = []\nprint('cleared')"})

place = (
    "from isaacsim.core.prims import RigidPrim as RP2\n"
    "import torch\n"
    "RP2('/World_Props/pink_cube').set_world_poses("
    "positions=torch.tensor([[0.175,0.130,0.045]], dtype=torch.float32))\n"
    "print('placed')\n"
)
c.request({"op": "exec", "code": place})
time.sleep(1.5)
p0 = np.asarray(truth.pose("pink_cube"), float)
print(f"  cube at {np.round(p0, 4)}")

print("\n=== waiting for the jump (up to 25 s) ===")
SPAWN = np.array([0.17, 0.15, 0.04])
for i in range(25):
    time.sleep(1.0)
    p = truth.pose("pink_cube")
    if p is None:
        continue
    p = np.asarray(p, float)
    if float(np.linalg.norm(p[:2] - SPAWN[:2])) < 0.012:
        print(f"  [{i:2}s] jumped to {np.round(p, 4)}")
        break
else:
    print("  no jump observed")

print("\n=== who called set_world_poses? ===")
r = c.request({"op": "exec", "code": (
    "calls = globals().get('_POSE_CALLS', [])\n"
    "print('total calls:', len(calls))\n"
    "seen = set()\n"
    "for path, stack in calls[-12:]:\n"
    "    key = stack[-300:]\n"
    "    if key in seen:\n"
    "        continue\n"
    "    seen.add(key)\n"
    "    print('---', path)\n"
    "    print(stack)\n"
)})
print(r.get("stdout", "")[:3000])
