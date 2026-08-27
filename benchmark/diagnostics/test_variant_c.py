"""Test ONLY variant C, the statement truth.py actually runs.

narrow_probe.py got through A and B (both stable) and then the bridge died
during C -- which is itself a finding: constructing
`isaacsim.core.prims.RigidPrim` (the DEPRECATED, non-experimental API) once
per prop, once per probe, is not free.

truth.py line 90 does exactly that, for every dynamic prop, on EVERY
verification read. This script runs that single pattern in a fresh bridge and
watches both the cube's pose and the bridge's health.
"""
import sys
import time

import numpy as np

from cascade.sim.bridge_client import BridgeClient

SPAWN = np.array([0.17, 0.15])


def connect():
    c = BridgeClient(port=8611)
    c.connect()
    return c


c = connect()

PLACE = (
    "from isaacsim.core.prims import RigidPrim\n"
    "import torch\n"
    "RigidPrim('/World_Props/pink_cube').set_world_poses("
    "positions=torch.tensor([[0.175,0.130,0.045]], dtype=torch.float32))\n"
)

# read through the EXPERIMENTAL api, which confirm_mechanism.py showed stable
READ = (
    "from isaacsim.core.experimental.prims import RigidPrim as _RP\n"
    "import numpy as _np\n"
    "_p, _ = _RP('/World_Props/pink_cube').get_world_poses()\n"
    "_a = _np.asarray(_p.numpy() if hasattr(_p,'numpy') else _p).reshape(-1)\n"
    "print('POSE ' + ' '.join(str(round(float(v),5)) for v in _a[:3]))\n"
)

# the pattern from truth.py:90 -- deprecated API, one view per prop per probe
VARIANT_C = (
    "from isaacsim.core.prims import RigidPrim as _RPc\n"
    "for _nm in ('pink_cube','green_cube'):\n"
    "    try:\n"
    "        _RPc('/World_Props/' + _nm).get_world_poses()\n"
    "    except Exception:\n"
    "        pass\n"
)


def read():
    global c
    try:
        r = c.request({"op": "exec", "code": READ})
    except Exception as e:
        print(f"    bridge error on read: {e}")
        return "DEAD"
    for line in r.get("stdout", "").splitlines():
        if line.startswith("POSE "):
            return np.array([float(v) for v in line[5:].split()])
    return None


print("=== place cube off-spawn, then run truth.py's RigidPrim pattern ===")
c.request({"op": "exec", "code": PLACE})
time.sleep(1.2)
p0 = read()
print(f"  start: {np.round(p0, 4)}  (spawn is {SPAWN})")

for i in range(12):
    try:
        c.request({"op": "exec", "code": VARIANT_C})
    except Exception as e:
        print(f"  [{i:2}] BRIDGE DIED running the probe pattern: {e}")
        break
    time.sleep(0.5)
    p = read()
    if p is None:
        continue
    if isinstance(p, str):
        print(f"  [{i:2}] bridge dead after {i+1} probe calls")
        break
    d = float(np.linalg.norm(p[:2] - SPAWN))
    print(f"  [{i:2}] {np.round(p, 4)}  d_spawn={d*100:5.1f}cm"
          f"{'   <-- RESET' if d < 0.012 else ''}")
    if d < 0.012:
        break
