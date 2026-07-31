"""Does the reset appear only after the bridge has been running a while?

Timeline of evidence:

  * old bridge (hours up, had served the whole 30-episode ablation)
      - cube jumped back to spawn ~9-10 s after an off-spawn placement
      - only while TruthPoseReader was polling; silent otherwise
      - variant C eventually KILLED its TCP thread (:8611 stopped listening
        while the process stayed alive)
  * fresh bridge
      - verbatim probe: 15 calls, stable
      - TruthPoseReader (ttl=0): 15 calls, stable

So the trigger is accumulated state, not a single API call. The prime suspect
is `isaacsim.core.prims.RigidPrim` (the DEPRECATED wrapper): truth.py builds
a NEW view per prop per probe and never releases it. Thousands of live views
over an hour is exactly the kind of leak that ends in a dead TCP thread.

This script hammers the probe and watches three things together:
  - does the cube stay put?
  - does probe latency grow?  (leak signature)
  - does the bridge survive?
"""
import sys
import time

import numpy as np

from wrc_demo.sim.bridge_client import BridgeClient
from wrc_demo.sim.truth import _PROBE, _PROP_ROOTS

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
PROBE = _PROBE % {"roots": repr(_PROP_ROOTS)}


def read():
    try:
        r = c.request({"op": "exec", "code": READ})
    except Exception:
        return "DEAD"
    for line in r.get("stdout", "").splitlines():
        if line.startswith("POSE "):
            return np.array([float(v) for v in line[5:].split()])
    return None


c.request({"op": "exec", "code": PLACE})
time.sleep(1.2)
print(f"start: {np.round(read(), 4)}\n")
print(f"{'probes':>8}{'latency ms':>12}{'cube pos':>28}{'d_spawn':>10}")
print("-" * 60)

t_first = None
for n in range(1, 401):
    t0 = time.monotonic()
    try:
        c.request({"op": "exec", "code": PROBE})
    except Exception as e:
        print(f"{n:>8}  BRIDGE DIED: {e}")
        break
    dt = (time.monotonic() - t0) * 1000
    if t_first is None:
        t_first = dt
    if n % 25 == 0 or n <= 3:
        p = read()
        if isinstance(p, str):
            print(f"{n:>8}  bridge dead on read")
            break
        if p is None:
            continue
        d = float(np.linalg.norm(p[:2] - SPAWN))
        flag = "  <-- RESET" if d < 0.012 else ""
        print(f"{n:>8}{dt:>12.1f}{str(np.round(p, 4)):>28}{d*100:>9.1f}cm{flag}")
        if d < 0.012:
            print(f"\n  reset appeared after {n} probe calls "
                  f"(latency {t_first:.0f} ms -> {dt:.0f} ms)")
            break
