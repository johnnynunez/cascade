"""Does READING the pose cause the reset?

trace_slide.py polled every 0.25 s and the cube never moved.
repro_retry_reset.py polled every 1.0 s and it jumped at t=9 s.
Same placement, different observer cadence -> the difference is not the cube,
it is what else ran in between.

repro also called `truth.pose()` through TruthPoseReader, which may open a
fresh RigidPrim view per call. Creating a RigidPrim with
`reset_xform_op_properties=True` REWRITES the prim's xform ops -- that is a
documented way to clobber a pose. If the reader does that, then the act of
measuring resets the thing being measured.

Test: place off-spawn, then (A) sit still doing nothing, (B) poll pose
repeatedly, and see which one drifts.
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


def place(x=0.175, y=0.130, z=0.045):
    c.request({"op": "exec", "code": (
        "from isaacsim.core.prims import RigidPrim\n"
        "import torch\n"
        "RigidPrim('/World_Props/pink_cube').set_world_poses("
        f"positions=torch.tensor([[{x},{y},{z}]], dtype=torch.float32))\n"
    )})
    time.sleep(1.0)


def raw_pose():
    """Read WITHOUT TruthPoseReader, via a plain USD attribute query."""
    r = c.request({"op": "exec", "code": (
        "from isaacsim.core.experimental.prims import RigidPrim\n"
        "import numpy as _np\n"
        "rp = RigidPrim('/World_Props/pink_cube')\n"
        "p, _ = rp.get_world_poses()\n"
        "# warp arrays are not item-indexable; go through numpy\n"
        "a = _np.asarray(p.numpy() if hasattr(p, 'numpy') else p).reshape(-1)\n"
        "print(' '.join(str(round(float(v), 5)) for v in a[:3]))\n"
    )})
    try:
        return np.array([float(v) for v in r.get("stdout", "").split()])
    except Exception:
        return None


print("=== A: place, then DO NOT read for 15 s ===")
place()
p0 = raw_pose()
print(f"  after placing: {np.round(p0, 4)}")
time.sleep(15.0)
pA = raw_pose()
print(f"  after 15 s of silence: {np.round(pA, 4)}  "
      f"d_spawn={np.linalg.norm(pA[:2]-SPAWN)*100:.1f}cm")

print("\n=== B: place, then poll TruthPoseReader every 1 s for 15 s ===")
place()
p0 = raw_pose()
print(f"  after placing: {np.round(p0, 4)}")
jumped = None
for i in range(15):
    time.sleep(1.0)
    p = truth.pose("pink_cube")
    if p is None:
        continue
    p = np.asarray(p, float)
    if float(np.linalg.norm(p[:2] - SPAWN)) < 0.012:
        jumped = i
        print(f"  [{i:2}s] TruthPoseReader read -> {np.round(p, 4)}  JUMPED")
        break
if jumped is None:
    print(f"  no jump; final {np.round(np.asarray(truth.pose('pink_cube'), float), 4)}")

print("\n=== verdict ===")
print("  A (no reads) drifted:", float(np.linalg.norm(pA[:2] - SPAWN)) < 0.012)
print("  B (polled) jumped   :", jumped is not None)
