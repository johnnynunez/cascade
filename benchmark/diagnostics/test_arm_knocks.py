"""The missing variable: the ARM was moving during the ablation.

400 probe calls on an idle scene leave the cube alone (stress_probe.py). But
in the ablation the cube reset while a skill was running -- and every skill
ends with `move_home`, which drives the arm right back over the workspace.

Hypothesis: the arm (or the gripper) physically knocks the cube, and the
"reset to spawn" is not a teleport at all -- it is the cube being pushed by a
6-DoF arm sweeping home through the exact region where the cube sits. Note
the ablation's own numbers: the three false-claim episodes moved the cube
1.7-3.0 cm. That is a nudge, not a transport.

And [0.17, 0.15] is not only the spawn; it is where an arm homing from a
grasp attempt would sweep through.

Test: place the cube off-spawn, run ONLY move_home repeatedly (no grasp, no
verification), and watch.
"""
import sys
import time
from pathlib import Path

import numpy as np

from wrc_demo.apps.demo import build_runtime, shutdown_runtime
from wrc_demo.config import load_demo_config
from wrc_demo.sim.bridge_client import BridgeClient

SPAWN = np.array([0.17, 0.15])

c = BridgeClient(port=8611)
c.connect()

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


cfg = load_demo_config(cameras=["isaac"], arm="isaac", llm="mock")
cfg._data.setdefault("stream", {})["mode"] = "off"
rt, arm = build_runtime(cfg, Path("/tmp/wrc_armtest"), view=False, serve=False)

try:
    print("=== move_home only, no grasp, no verification ===")
    c.request({"op": "exec", "code": PLACE})
    time.sleep(1.5)
    p0 = read()
    print(f"  start: {np.round(p0, 4)}")

    for i in range(6):
        rt.execute("move_home", {})
        time.sleep(1.0)
        p = read()
        if p is None:
            continue
        d = float(np.linalg.norm(p[:2] - SPAWN))
        moved = float(np.linalg.norm(p - p0))
        print(f"  [{i}] after move_home: {np.round(p, 4)}  "
              f"moved={moved*100:5.1f}cm  d_spawn={d*100:5.1f}cm"
              f"{'   <-- AT SPAWN' if d < 0.012 else ''}")
        if d < 0.012:
            break

    print("\n=== now a full pick_and_place, tracking the cube throughout ===")
    c.request({"op": "exec", "code": PLACE})
    time.sleep(1.5)
    p0 = read()
    print(f"  start: {np.round(p0, 4)}")
    r = rt.execute("pick_and_place",
                   {"object": "pink cube", "destination": "box"})
    time.sleep(1.0)
    p = read()
    print(f"  after: {np.round(p, 4)}  ok={r.get('ok')} "
          f"stage={r.get('stage')}")
    print(f"  postcondition: {r.get('postcondition')}")
finally:
    shutdown_runtime(rt, arm)
