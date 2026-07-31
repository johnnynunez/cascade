"""Run the VERBATIM truth.py probe and watch the cube.

Isolated pieces are all stable:
  A traverse only              stable
  B traverse + Xformable       stable
  C RigidPrim per prop         stable (test_variant_c.py, 12 calls)

So the reset needs the whole probe. This runs `truth.py::_PROBE` exactly as
shipped -- including the `_UsdPhysics.RigidBodyAPI` filter and the
`isaacsim.core.prims` import INSIDE the traversal -- against a cube parked
off-spawn, and reports the first probe call that moves it.
"""
import sys
import time

import numpy as np

from wrc_demo.sim.bridge_client import BridgeClient
from wrc_demo.sim.truth import _PROBE, _PROP_ROOTS, TruthPoseReader

c = BridgeClient(port=8611)
c.connect()
SPAWN = np.array([0.17, 0.15])

PLACE = (
    "from isaacsim.core.prims import RigidPrim\n"
    "import torch\n"
    "RigidPrim('/World_Props/pink_cube').set_world_poses("
    "positions=torch.tensor([[0.175,0.130,0.045]], dtype=torch.float32))\n"
)
READ = (
    "from isaacsim.core.experimental.prims import RigidPrim as _RP\n"
    "import numpy as _np\n"
    "_p, _ = _RP('/World_Props/pink_cube').get_world_poses()\n"
    "_a = _np.asarray(_p.numpy() if hasattr(_p,'numpy') else _p).reshape(-1)\n"
    "print('POSE ' + ' '.join(str(round(float(v),5)) for v in _a[:3]))\n"
)


def read():
    r = c.request({"op": "exec", "code": READ})
    for line in r.get("stdout", "").splitlines():
        if line.startswith("POSE "):
            return np.array([float(v) for v in line[5:].split()])
    return None


print("=== VERBATIM truth.py probe ===")
c.request({"op": "exec", "code": PLACE})
time.sleep(1.2)
print(f"  start: {np.round(read(), 4)}")

probe_code = _PROBE % {"roots": repr(_PROP_ROOTS)}
for i in range(15):
    c.request({"op": "exec", "code": probe_code})
    time.sleep(0.5)
    p = read()
    if p is None:
        continue
    d = float(np.linalg.norm(p[:2] - SPAWN))
    print(f"  [{i:2}] {np.round(p, 4)}  d_spawn={d*100:5.1f}cm"
          f"{'   <-- RESET' if d < 0.012 else ''}")
    if d < 0.012:
        break

print("\n=== same thing through TruthPoseReader (what the checker uses) ===")
c.request({"op": "exec", "code": PLACE})
time.sleep(1.2)
reader = TruthPoseReader(c, ttl_s=0.0)      # no cache, force a probe per call
print(f"  start: {np.round(read(), 4)}")
for i in range(15):
    p = reader.pose("pink cube")
    time.sleep(0.5)
    q = read()
    if q is None:
        continue
    d = float(np.linalg.norm(q[:2] - SPAWN))
    print(f"  [{i:2}] reader={None if p is None else np.round(p,4)}  "
          f"actual={np.round(q,4)}  d_spawn={d*100:5.1f}cm"
          f"{'   <-- RESET' if d < 0.012 else ''}")
    if d < 0.012:
        break
