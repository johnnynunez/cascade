"""Is the 1.8 cm approach error perception or kinematics?

The grasp fails because the gripper descends 1.8 cm off-centre on a cube whose
half-width is 2.5 cm -- a finger catches the edge and shoves it away, then the
jaws close on air.

Two candidate sources, and they need different fixes:

  A. PERCEPTION -- the detector/depth pipeline reports the cube in the wrong
     place, so the arm is aiming correctly at a wrong target.
  B. KINEMATICS -- the target is right and the arm does not reach it (IK
     tolerance, drive tracking, or a settle timeout cutting the motion short).

Distinguish by comparing three numbers for the same cube:
  1. physics truth (where the cube actually is)
  2. what the perception stack believes
  3. where the gripper ends up when told to go there

truth vs belief isolates (A); belief vs achieved isolates (B).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from cascade.apps.demo import build_runtime
from cascade.config import load_demo_config
from cascade.sim.bridge_client import BridgeClient
from cascade.sim.truth import TruthPoseReader

c = BridgeClient(port=8611)
c.connect()
truth = TruthPoseReader(c, ttl_s=0)

c.request({"op": "reset_props"}, timeout_s=90.0)
time.sleep(1.0)
START = [0.170, 0.150, 0.045]
c.request({"op": "place_prop", "name": "pink_cube", "pos": START}, timeout_s=60.0)
time.sleep(1.5)

t = truth.pose("pink cube")
print(f"1. PHYSICS TRUTH   ({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f})")

cfg = load_demo_config(cameras=["isaac"], arm="isaac", llm="mock")
rt, arm = build_runtime(cfg, Path("/tmp/newton_aim"), view=False)

rt._reobserve()
res = rt.execute("localize_object", {"label": "pink cube"})
if res.get("ok"):
    p = res.get("position") or res.get("pose")
    if p:
        b = np.asarray(p[:3], dtype=float)
        print(f"2. PERCEPTION      ({b[0]:+.4f}, {b[1]:+.4f}, {b[2]:+.4f})")
        err = float(np.linalg.norm(b[:2] - np.asarray(t[:2])))
        print(f"   truth vs belief : {err*100:.2f} cm")
    else:
        print("2. PERCEPTION      no position in result:", str(res)[:120])
else:
    print("2. PERCEPTION      failed:", str(res.get("error"))[:100])

# 3. ask the arm to hover exactly over the truth pose and see where it lands
GRIP = r"""
import numpy as np
from isaacsim.core.experimental.prims import RigidPrim
v = RigidPrim("/tn__00armrs_asmv3_hJ6D/Geometry/base_link/link1/link2/link3/link4/link5/link6/gripper_end").get_world_poses()[0]
v = v.numpy() if hasattr(v, "numpy") else v
v = np.asarray(v).reshape(-1)[:3]
print("GRIP %.4f %.4f %.4f" % (v[0], v[1], v[2]))
"""
r = rt.execute("point_at", {"label": "pink cube"})
time.sleep(2.0)
out = c.request({"op": "exec", "code": GRIP}).get("stdout", "")
ln = [l for l in out.splitlines() if l.startswith("GRIP")]
if ln:
    g = [float(x) for x in ln[0].split()[1:]]
    print(f"3. GRIPPER LANDED  ({g[0]:+.4f}, {g[1]:+.4f}, {g[2]:+.4f})")
    lat = float(np.linalg.norm(np.array(g[:2]) - np.asarray(t[:2])))
    print(f"   lateral vs truth: {lat*100:.2f} cm")
