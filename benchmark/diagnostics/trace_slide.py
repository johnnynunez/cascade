"""Is the cube TELEPORTED back to spawn, or does it SLIDE there?

catch_teleport.py logged ZERO set_world_poses calls while the cube moved, so
nothing is teleporting it. That leaves physics. Two candidates:

  1. the cube is spawned intersecting the table (note z settled to 0.025 when
     placed at 0.045) and PhysX pushes it out of penetration;
  2. something in the scene (a slope, a stray force, the gripper) drives it.

A dense pose trace answers it: a teleport is a single-frame discontinuity, a
physical slide is a smooth trajectory with intermediate positions.
"""
import sys
import time

import numpy as np

from wrc_demo.sim.bridge_client import BridgeClient
from wrc_demo.sim.truth import TruthPoseReader

c = BridgeClient(port=8611)
c.connect()
truth = TruthPoseReader(c)
SPAWN = np.array([0.17, 0.15])


def place(x, y, z):
    c.request({"op": "exec", "code": (
        "from isaacsim.core.prims import RigidPrim\n"
        "import torch\n"
        "RigidPrim('/World_Props/pink_cube').set_world_poses("
        f"positions=torch.tensor([[{x},{y},{z}]], dtype=torch.float32))\n"
    )})


for z0 in (0.045, 0.041):
    print(f"\n=== placed at (0.175, 0.130, {z0}) — dense trace ===")
    place(0.175, 0.130, z0)
    time.sleep(0.4)
    prev = None
    for i in range(40):
        p = truth.pose("pink_cube")
        if p is None:
            time.sleep(0.25)
            continue
        p = np.asarray(p, float)
        d_spawn = float(np.linalg.norm(p[:2] - SPAWN))
        step = 0.0 if prev is None else float(np.linalg.norm(p - prev))
        if i < 6 or step > 0.003 or d_spawn < 0.015:
            print(f"  t={i*0.25:5.2f}s  pos={np.round(p, 4)}  "
                  f"step={step*1000:6.1f}mm  d_spawn={d_spawn*100:5.1f}cm")
        prev = p
        if d_spawn < 0.012:
            print("  -> reached spawn")
            break
        time.sleep(0.25)

print("\n=== velocity while it moves ===")
place(0.175, 0.130, 0.045)
time.sleep(0.5)
for i in range(6):
    r = c.request({"op": "exec", "code": (
        "from isaacsim.core.experimental.prims import RigidPrim\n"
        "rp = RigidPrim('/World_Props/pink_cube')\n"
        "try:\n"
        "    v = rp.get_velocities()\n"
        "    print('vel', [round(float(x), 4) for x in list(v[0])[:6]])\n"
        "except Exception as e:\n"
        "    print('vel err', e)\n"
    )})
    print(f"  [{i}] {r.get('stdout', '').strip()}")
    time.sleep(0.8)
