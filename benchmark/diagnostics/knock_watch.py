"""The arm KNOCKS the cube away before it can close on it.

The swing test never got to test the swing: the grasp reported "air grasp:
gripper closed fully, object not held", and the cube had moved from where it
was placed to 7.9 cm away:

    placed at (0.170, 0.150)
    after the grasp attempt (0.241, 0.186)

So the failure is not transport and not the swing. The arm displaces the cube
during the approach, then closes on empty space. That also explains the sweep
signature perfectly: every final pose scattered a few cm from the start, none
anywhere near the bin, and "moved" values of 10-18 cm accumulated over the
retry attempts.

And it explains why the standalone E2E sometimes succeeded: it started from
wherever the previous run left the cube, occasionally a spot the approach
happens not to clip.

Question this measures: WHEN in the approach does the cube move? Track the
cube continuously through a single grasp attempt and report the first sample
where it shifts more than 5 mm, alongside the gripper position at that moment.
If the gripper is still above the cube, the descent is clipping it; if the
gripper is beside it, the lateral approach is.
"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, "/home/johnny/Projects/demo/cascade/src")

import numpy as np

from cascade.apps.demo import build_runtime
from cascade.config import load_demo_config
from cascade.sim.bridge_client import BridgeClient

c = BridgeClient(port=8611)
c.connect()

PROBE = r"""
import numpy as np
from isaacsim.core.experimental.prims import RigidPrim
def _p(path):
    v = RigidPrim(path).get_world_poses()[0]
    v = v.numpy() if hasattr(v, "numpy") else v
    return np.asarray(v).reshape(-1)[:3]
cu = _p("/World_Props/pink_cube")
gr = _p("/tn__00armrs_asmv3_hJ6D/Geometry/base_link/link1/link2/link3/link4/link5/link6/gripper_end")
print("S %.4f %.4f %.4f %.4f %.4f %.4f" % (cu[0],cu[1],cu[2],gr[0],gr[1],gr[2]))
"""

c.request({"op": "reset_props"}, timeout_s=90.0)
time.sleep(1.0)
START = [0.170, 0.150, 0.045]
c.request({"op": "place_prop", "name": "pink_cube", "pos": START}, timeout_s=60.0)
time.sleep(1.0)

samples = []
stop = threading.Event()


def watcher():
    while not stop.is_set():
        try:
            out = c.request({"op": "exec", "code": PROBE}).get("stdout", "")
            ln = [l for l in out.splitlines() if l.startswith("S ")]
            if ln:
                samples.append((time.monotonic(),
                                [float(x) for x in ln[0].split()[1:]]))
        except Exception:
            pass
        time.sleep(0.10)


th = threading.Thread(target=watcher, daemon=True)
th.start()

cfg = load_demo_config(cameras=["isaac"], arm="isaac", llm="mock")
rt, arm = build_runtime(cfg, Path("/tmp/newton_knock"), view=False)
t0 = time.monotonic()
res = rt.execute("grasp_object", {"label": "pink cube"})
stop.set()
time.sleep(0.3)

print(f"grasp ok={res.get('ok')}  {str(res.get('error',''))[:70]}")
print(f"\n{len(samples)} samples")

origin = np.array(START[:2])
first = None
for (t, v) in samples:
    d = float(np.linalg.norm(np.array(v[:2]) - origin))
    if d > 0.005:
        first = (t, v, d)
        break

if first is None:
    print("the cube never moved during the grasp")
else:
    t, v, d = first
    print(f"\nFIRST DISPLACEMENT at t+{t-t0:.1f}s, {d*100:.1f} cm")
    print(f"  cube    ({v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f})")
    print(f"  gripper ({v[3]:+.3f},{v[4]:+.3f},{v[5]:+.3f})")
    dz = v[5] - v[2]
    lat = float(np.linalg.norm(np.array(v[3:5]) - np.array(v[:2])))
    print(f"  gripper is {dz*100:+.1f} cm above and {lat*100:.1f} cm lateral")
    print("  -> " + ("DESCENT clipped it" if lat < 0.03 else
                     "LATERAL approach swept it"))

    i = samples.index((t, v))
    print("\n  preceding samples:")
    for (t2, v2) in samples[max(0, i-6):i+3]:
        dd = float(np.linalg.norm(np.array(v2[:2]) - origin))
        print(f"    t+{t2-t0:5.1f}  cube({v2[0]:+.3f},{v2[1]:+.3f},{v2[2]:+.3f}) "
              f"grip({v2[3]:+.3f},{v2[4]:+.3f},{v2[5]:+.3f})  moved {dd*100:5.2f}cm")
