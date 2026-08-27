"""Does the detection mask leak onto the table?

The fitted oriented box measures 8.8 x 7.2 x 4.4 cm for a 5 cm cube: +76% in
x, +44% in y, -12% in z. That REFUTES my partial-shell hypothesis (a shell
would fit SMALLER, not larger) and points somewhere else: the mask includes
pixels that are not the cube, and those extra points drag the box centre.

The residual splits into -1.38 cm toward the camera and 0.91 cm perpendicular,
so there are two components, not one clean depth offset.

Save the frame with the detection mask and box drawn, plus a histogram of the
point cloud's height above the table. If the mask is bleeding onto the table
surface, the cloud will show a population at z ~ 0 (table) as well as the cube
body, and the picture will show the box overhanging the cube.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/johnny/Projects/demo/cascade/src")

import cv2
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
c.request({"op": "place_prop", "name": "pink_cube",
           "pos": [0.190, 0.130, 0.045]}, timeout_s=60.0)
time.sleep(1.5)

t = np.asarray(truth.pose("pink cube"), dtype=float)
print("truth:", np.round(t, 4))

cfg = load_demo_config(cameras=["isaac"], arm="isaac", llm="mock")
rt, arm = build_runtime(cfg, Path("/tmp/newton_mask"), view=False)
rt._reobserve()

frame, fix = rt._localize("pink cube")
pts = fix.points
print(f"\npoint cloud: {pts.shape[0]} points")
print(f"  x range {pts[:,0].min():.4f} .. {pts[:,0].max():.4f}")
print(f"  y range {pts[:,1].min():.4f} .. {pts[:,1].max():.4f}")
print(f"  z range {pts[:,2].min():.4f} .. {pts[:,2].max():.4f}")

# height histogram: table is at z=0, cube spans 0.015..0.065 around centre 0.04
z = pts[:, 2]
for lo, hi in [(-0.02, 0.005), (0.005, 0.02), (0.02, 0.035),
               (0.035, 0.05), (0.05, 0.07), (0.07, 0.2)]:
    n = int(((z >= lo) & (z < hi)).sum())
    bar = "#" * min(60, n // 20)
    print(f"  z [{lo:+.3f},{hi:+.3f}) {n:6d} {bar}")

near_table = int((z < 0.015).sum())
print(f"\npoints below 1.5 cm (i.e. table, not cube): {near_table} "
      f"({100.0*near_table/len(z):.1f}%)")

# lateral spread against the true cube footprint
inside = ((np.abs(pts[:, 0] - t[0]) <= 0.026) &
          (np.abs(pts[:, 1] - t[1]) <= 0.026))
print(f"points within the true 5 cm footprint: {int(inside.sum())} "
       f"({100.0*inside.sum()/len(pts):.1f}%)")

# draw the mask
det = fix.detection
img = frame.rgb.copy()
if det.mask is not None:
    m = det.mask.astype(bool)
    img[m] = (0.5 * img[m] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
x0, y0, x1, y1 = [int(v) for v in det.bbox]
cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
cv2.imwrite("/tmp/mask_check.png", img)
print("\nwrote /tmp/mask_check.png")
