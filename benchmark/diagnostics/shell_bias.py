"""Why is the pipeline's error (1.6 cm) smaller than my naive round-trip (3.2 cm)?

Because it does not deproject one pixel. `_localize` builds a POINT CLOUD from
the detection mask (`mask_to_points_cam`), transforms it to base frame, and
takes the centre of an oriented bounding box over those points.

That cloud only covers the cube's VISIBLE faces -- the two or three the camera
can see. The oriented box over a partial shell is centred on the shell, not on
the solid, so its centre sits toward the camera by roughly half the cube's
unseen depth. That is a smaller error than a single-pixel surface deprojection
(which is off by the full half-diagonal), which is exactly the ordering
observed: 1.6 cm vs 3.2 cm.

So the correct fix is NOT "+2.5 cm along the ray" -- that was a diagnostic to
prove the mechanism, and applying it blindly would overshoot, since the point
cloud already recovers part of the offset.

The principled correction: the oriented box reports `extents`. If the cube's
true size is known (5 cm) and the fitted extent along the viewing direction is
smaller, the deficit is the unseen part, and the centre should move away from
the camera by half of it.

Measure the inputs that fix needs:
  - fitted extents vs the true 5 cm
  - the residual error vector, expressed along the camera's viewing direction
    (if the error is mostly ALONG the view ray, the shell explanation holds;
     if it has a large perpendicular component, something else is also wrong)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/johnny/Projects/demo/cascade/src")

import numpy as np
import yaml

from cascade.apps.demo import build_runtime
from cascade.config import load_demo_config
from cascade.sim.bridge_client import BridgeClient
from cascade.sim.truth import TruthPoseReader

c = BridgeClient(port=8611)
c.connect()
truth = TruthPoseReader(c, ttl_s=0)

T = np.array(yaml.safe_load(
    open("/home/johnny/Projects/demo/cascade/configs/cameras/isaac.yaml")
)["extrinsics"]["T"], dtype=float)
CAM_POS = T[:3, 3]
print(f"camera position in base frame: {np.round(CAM_POS, 3)}")

cfg = load_demo_config(cameras=["isaac"], arm="isaac", llm="mock")
rt, arm = build_runtime(cfg, Path("/tmp/newton_shell"), view=False)

print(f"\n{'placed':16} {'extent_m':22} {'err':>6} {'along view':>11} {'perp':>7}")
for (x, y) in [(0.190, 0.130), (0.170, 0.150), (0.210, 0.120)]:
    c.request({"op": "reset_props"}, timeout_s=90.0)
    time.sleep(1.0)
    c.request({"op": "place_prop", "name": "pink_cube",
               "pos": [x, y, 0.045]}, timeout_s=60.0)
    time.sleep(1.5)

    t = np.asarray(truth.pose("pink cube"), dtype=float)
    rt._reobserve()
    res = rt.execute("localize_object", {"label": "pink cube"})
    if not res.get("ok"):
        print(f"({x:.3f},{y:.3f})  localize failed")
        continue

    b = np.asarray(res["position"], dtype=float)
    ext = np.asarray(res.get("extent_m", [0, 0, 0]), dtype=float)

    err_vec = b - t
    view = t - CAM_POS
    view /= np.linalg.norm(view)
    along = float(np.dot(err_vec, view))          # +: belief is FURTHER
    perp = float(np.linalg.norm(err_vec - along * view))

    print(f"({x:.3f},{y:.3f})  {str(np.round(ext,3)):22} "
          f"{np.linalg.norm(err_vec)*100:5.2f}  {along*100:+10.2f}  {perp*100:6.2f}")

print("\nunits: cm.  'along view' negative means the belief sits TOWARD the camera")
print("true cube size is 0.05 m on every axis")
