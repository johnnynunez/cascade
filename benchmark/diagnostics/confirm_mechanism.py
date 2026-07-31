"""Confirm the mechanism: constructing a RigidPrim view resets the pose.

observer_effect.py proved reading through TruthPoseReader moves the cube back
to spawn while not reading leaves it alone. truth.py line 90 builds a NEW
`isaacsim.core.prims.RigidPrim(path)` on EVERY probe. That class initialises
the prim's xform ops on construction, which rewrites the transform from the
authored (spawn) values -- so each read teleports the body it is measuring.

This script isolates that single call: place the cube off-spawn, then do
nothing but CONSTRUCT RigidPrim views, and watch the pose.
"""
import sys
import time

import numpy as np

from wrc_demo.sim.bridge_client import BridgeClient

c = BridgeClient(port=8611)
c.connect()
SPAWN = np.array([0.17, 0.15])


def place(x=0.175, y=0.130, z=0.045):
    c.request({"op": "exec", "code": (
        "from isaacsim.core.prims import RigidPrim\n"
        "import torch\n"
        "RigidPrim('/World_Props/pink_cube').set_world_poses("
        f"positions=torch.tensor([[{x},{y},{z}]], dtype=torch.float32))\n"
    )})
    time.sleep(1.0)


def pose_via(kind: str):
    """Read the pose. `kind` selects which API constructs the view."""
    if kind == "experimental":
        code = (
            "from isaacsim.core.experimental.prims import RigidPrim\n"
            "import numpy as _np\n"
            "p, _ = RigidPrim('/World_Props/pink_cube').get_world_poses()\n"
            "a = _np.asarray(p.numpy() if hasattr(p,'numpy') else p).reshape(-1)\n"
            "print(' '.join(str(round(float(v),5)) for v in a[:3]))\n"
        )
    else:                                   # the one truth.py uses
        code = (
            "from isaacsim.core.prims import RigidPrim\n"
            "import numpy as _np\n"
            "p, _ = RigidPrim('/World_Props/pink_cube').get_world_poses()\n"
            "# this API returns a CUDA torch tensor\n"
            "a = _np.asarray(p.cpu().numpy() if hasattr(p,'cpu') else p).reshape(-1)\n"
            "print(' '.join(str(round(float(v),5)) for v in a[:3]))\n"
        )
    r = c.request({"op": "exec", "code": code})
    try:
        return np.array([float(v) for v in r.get("stdout", "").split()])
    except Exception:
        return None


for kind in ("experimental", "core"):
    print(f"\n=== reading via isaacsim.core{'.experimental' if kind=='experimental' else ''}.prims.RigidPrim ===")
    place()
    ok = True
    for i in range(12):
        p = pose_via(kind)
        if p is None or len(p) < 3:
            print(f"  [{i:2}] read failed")
            continue
        d = float(np.linalg.norm(p[:2] - SPAWN))
        flag = "  <-- BACK AT SPAWN" if d < 0.012 else ""
        if i < 3 or flag:
            print(f"  [{i:2}] {np.round(p, 4)}  d_spawn={d*100:5.1f}cm{flag}")
        if flag:
            ok = False
            break
        time.sleep(0.7)
    print(f"  verdict: {'STABLE' if ok else 'RESET BY THE READ'}")
