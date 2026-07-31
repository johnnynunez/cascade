"""Reproduce the retry-resets-the-scene bug in isolation.

The ablation showed three verify_retry episodes ending at the cube's DEFAULT
SPAWN pose [0.17, 0.15, 0.04] after moving only 1.7-3.0 cm, and being
CONFIRMED as successes. This script asks one question directly:

    does anything in the wrc_demo path move a prop back to its spawn
    WITHOUT the caller asking for it?

Method: place the cube somewhere non-default, poll the physics pose while
running the same skill sequence the retry path runs, and print any jump back
to spawn.
"""
import sys
import time

import numpy as np

from wrc_demo.sim.bridge_client import BridgeClient
from wrc_demo.sim.truth import TruthPoseReader

SPAWN = np.array([0.17, 0.15, 0.04])
PLACED = (0.175, 0.130)

c = BridgeClient(port=8611)
c.connect()
truth = TruthPoseReader(c)


def place(x, y, z=0.045):
    code = (
        "from isaacsim.core.prims import RigidPrim\n"
        "import torch\n"
        "p = RigidPrim('/World_Props/pink_cube')\n"
        f"p.set_world_poses(positions=torch.tensor([[{x},{y},{z}]], dtype=torch.float32))\n"
    )
    c.request({"op": "exec", "code": code})
    time.sleep(1.2)


def pose():
    p = truth.pose("pink_cube")
    return None if p is None else np.asarray(p, float)


print("=== 1. place the cube off-spawn and watch it for 20 s, untouched ===")
place(*PLACED)
p0 = pose()
print(f"  placed at {np.round(p0, 4)}")
drift = False
for i in range(20):
    time.sleep(1.0)
    p = pose()
    if p is None:
        continue
    if float(np.linalg.norm(p - SPAWN)) < 0.01 and float(np.linalg.norm(p0 - SPAWN)) > 0.02:
        print(f"  [{i:2}s] JUMPED BACK TO SPAWN {np.round(p, 4)}  <-- reproduced")
        drift = True
        break
    if float(np.linalg.norm(p - p0)) > 0.02:
        print(f"  [{i:2}s] moved on its own to {np.round(p, 4)}")
        drift = True
        break
if not drift:
    print(f"  stable at {np.round(pose(), 4)} -- no spontaneous reset")

print("\n=== 2. does anything in the demo runtime reset props? ===")
r = c.request({"op": "exec", "code": (
    "import inspect, sys\n"
    "hits = []\n"
    "for name, mod in list(sys.modules.items()):\n"
    "    if not name.startswith('wrc_demo'):\n"
    "        continue\n"
    "    try:\n"
    "        src = inspect.getsource(mod)\n"
    "    except Exception:\n"
    "        continue\n"
    "    if 'reset_props' in src or '_settle_props' in src:\n"
    "        hits.append(name)\n"
    "print('modules referencing a prop reset:', hits)\n"
)})
print(" ", r.get("stdout", "").strip())
