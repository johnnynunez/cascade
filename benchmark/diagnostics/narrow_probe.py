"""Narrow it to the exact statement inside truth.py's probe.

Single RigidPrim reads are stable (confirm_mechanism.py), yet the full
TruthPoseReader probe moves the cube back to spawn (observer_effect.py). The
probe does more than read one body: it walks the WHOLE stage and, for every
non-rigid prim, calls Xformable.ComputeLocalToWorldTransform.

Candidates, tested one at a time against a cube parked off-spawn:
  A  stage.Traverse() alone
  B  Traverse + ComputeLocalToWorldTransform on non-rigid prims
  C  RigidPrim view for EVERY dynamic prop (not just the cube)
  D  the verbatim probe from truth.py
"""
import sys
import time

import numpy as np

from cascade.sim.bridge_client import BridgeClient
from cascade.sim.truth import _PROBE, _PROP_ROOTS

c = BridgeClient(port=8611)
c.connect()
SPAWN = np.array([0.17, 0.15])

READ = (
    "from isaacsim.core.experimental.prims import RigidPrim as _RP\n"
    "import numpy as _np\n"
    "_p, _ = _RP('/World_Props/pink_cube').get_world_poses()\n"
    "_a = _np.asarray(_p.numpy() if hasattr(_p,'numpy') else _p).reshape(-1)\n"
    "print('POSE ' + ' '.join(str(round(float(v),5)) for v in _a[:3]))\n"
)

VARIANTS = {
    "A_traverse_only": (
        "import omni.usd as _usd\n"
        "_st = _usd.get_context().get_stage()\n"
        "_n = sum(1 for _ in _st.Traverse())\n"
    ),
    "B_traverse_plus_xformable": (
        "import omni.usd as _usd\n"
        "from pxr import UsdGeom as _G, Usd as _U, UsdPhysics as _P\n"
        "_st = _usd.get_context().get_stage()\n"
        "for _pr in _st.Traverse():\n"
        "    if not _pr.GetPath().pathString.startswith('/World'):\n"
        "        continue\n"
        "    if _pr.GetTypeName() not in ('Mesh','Xform'):\n"
        "        continue\n"
        "    if _pr.HasAPI(_P.RigidBodyAPI):\n"
        "        continue\n"
        "    try:\n"
        "        _G.Xformable(_pr).ComputeLocalToWorldTransform(_U.TimeCode.Default())\n"
        "    except Exception:\n"
        "        pass\n"
    ),
    "C_rigidprim_all_props": (
        "from isaacsim.core.prims import RigidPrim as _RPc\n"
        "for _nm in ('pink_cube','green_cube','bin_wall0','bin_wall1',"
        "'bin_wall2','bin_wall3'):\n"
        "    try:\n"
        "        _RPc('/World_Props/' + _nm).get_world_poses()\n"
        "    except Exception:\n"
        "        pass\n"
    ),
    "D_verbatim_probe": _PROBE % {"roots": repr(_PROP_ROOTS)},
}


def place(x=0.175, y=0.130, z=0.045):
    c.request({"op": "exec", "code": (
        "from isaacsim.core.prims import RigidPrim\n"
        "import torch\n"
        "RigidPrim('/World_Props/pink_cube').set_world_poses("
        f"positions=torch.tensor([[{x},{y},{z}]], dtype=torch.float32))\n"
    )})
    time.sleep(1.0)


def read():
    r = c.request({"op": "exec", "code": READ})
    out = r.get("stdout", "")
    for line in out.splitlines():
        if line.startswith("POSE "):
            return np.array([float(v) for v in line[5:].split()])
    return None


for name, code in VARIANTS.items():
    place()
    p0 = read()
    moved_at = None
    for i in range(14):
        c.request({"op": "exec", "code": code})
        time.sleep(0.5)
        p = read()
        if p is None:
            continue
        if float(np.linalg.norm(p[:2] - SPAWN)) < 0.012:
            moved_at = i
            break
    status = (f"RESET after {moved_at+1} calls" if moved_at is not None
              else "stable")
    print(f"  {name:28} start={np.round(p0,3)}  ->  {status}")
