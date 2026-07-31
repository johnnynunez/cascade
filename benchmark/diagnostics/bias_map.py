"""Is the 1.6 cm perception bias systematic, and is it Newton-specific?

Isolated so far, at one position:

    physics truth   (+0.1686, +0.1510)
    perception      (+0.1810, +0.1410)   -> 1.60 cm error
    gripper landed  (+0.1806, +0.1406)   -> 0.01 cm from the belief

The kinematics are essentially perfect; the arm aims precisely at a wrong
target. The error direction was consistent (+1.24 cm in x, -1.00 cm in y),
which smells like a fixed geometric offset rather than noise.

Two things worth knowing before trying to fix it:

  1. Is the bias the SAME at different cube positions? A constant offset
     points at an extrinsics/frame error (camera pose, base offset). A bias
     that scales with position points at intrinsics or a depth-scale error.
  2. Does it appear under PhysX too? If yes it is a perception bug that was
     always there and Newton merely exposed it (PhysX runs would have been
     grasping with the same 1.6 cm error and getting away with it). If no,
     something engine-specific is shifting the rendered frame.

This measures (1) across four positions on the current engine. Run it again
with the bridge on --engine physx to answer (2).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/johnny/Projects/demo/wrc_demo/src")

import numpy as np

from wrc_demo.apps.demo import build_runtime
from wrc_demo.config import load_demo_config
from wrc_demo.sim.bridge_client import BridgeClient
from wrc_demo.sim.truth import TruthPoseReader

c = BridgeClient(port=8611)
c.connect()
truth = TruthPoseReader(c, ttl_s=0)

cfg = load_demo_config(cameras=["isaac"], arm="isaac", llm="mock")
rt, arm = build_runtime(cfg, Path("/tmp/newton_bias"), view=False)

POSITIONS = [
    (0.170, 0.150),
    (0.200, 0.150),
    (0.170, 0.100),
    (0.220, 0.120),
]

print("  placed        truth            belief           dx      dy    err")
rows = []
for (x, y) in POSITIONS:
    c.request({"op": "reset_props"}, timeout_s=90.0)
    time.sleep(1.0)
    c.request({"op": "place_prop", "name": "pink_cube",
               "pos": [x, y, 0.045]}, timeout_s=60.0)
    time.sleep(1.5)

    t = truth.pose("pink cube")
    if t is None:
        print(f"({x:.3f},{y:.3f})  truth unreadable")
        continue

    rt._reobserve()
    res = rt.execute("localize_object", {"label": "pink cube"})
    p = res.get("position") or res.get("pose") if res.get("ok") else None
    if not p:
        print(f"({x:.3f},{y:.3f})  perception failed: "
              f"{str(res.get('error'))[:50]}")
        continue

    b = np.asarray(p[:3], dtype=float)
    dx, dy = b[0] - t[0], b[1] - t[1]
    err = float(np.hypot(dx, dy))
    rows.append((dx, dy, err))
    print(f"({x:.3f},{y:.3f})  ({t[0]:+.4f},{t[1]:+.4f})  "
          f"({b[0]:+.4f},{b[1]:+.4f})  {dx*100:+5.2f}  {dy*100:+5.2f}  "
          f"{err*100:5.2f}cm")

if rows:
    a = np.array(rows)
    print(f"\nmean bias   dx={a[:,0].mean()*100:+.2f} cm  dy={a[:,1].mean()*100:+.2f} cm")
    print(f"std         dx={a[:,0].std()*100: .2f} cm  dy={a[:,1].std()*100: .2f} cm")
    print(f"mean error  {a[:,2].mean()*100:.2f} cm")
    if a[:, 2].std() < 0.004:
        print("-> CONSTANT offset: extrinsics/frame error, correctable by calibration")
    else:
        print("-> VARIES with position: intrinsics or depth scale")
