"""If the mask leaks onto the table, does filtering those points fix the bias?

Evidence for mask contamination rather than surface-vs-centre:

    fitted extent   8.8 x 7.2 x 4.4 cm   for a 5 cm cube  (LARGER, not smaller)
    cloud z span    0.000 .. 0.080 m     8 cm tall for a 5 cm object
    below 1.5 cm    20.4% of points      -> table, not cube
    overlay         mask shifted down: pink uncovered on top, green on table

A partial shell would fit SMALLER than the true size. This fits LARGER, so the
shell story is refuted by its own sign, and the table points -- nearer the
camera and lower -- are what drag the centre.

Test the targeted fix WITHOUT touching the shipped code: pull the same point
cloud the pipeline uses, drop the points that sit at table height, refit the
box, and compare the centre against physics truth.

If the error collapses, the fix is a height filter in the cloud, and it is
cheap and local. If it does not, the mask problem is 2D and needs the detector
addressed instead.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/johnny/Projects/demo/cascade/src")

import numpy as np

from cascade.apps.demo import build_runtime
from cascade.config import load_demo_config
from cascade.perception.grounding import oriented_bbox
from cascade.sim.bridge_client import BridgeClient
from cascade.sim.truth import TruthPoseReader

c = BridgeClient(port=8611)
c.connect()
truth = TruthPoseReader(c, ttl_s=0)

cfg = load_demo_config(cameras=["isaac"], arm="isaac", llm="mock")
rt, arm = build_runtime(cfg, Path("/tmp/newton_filter"), view=False)

print(f"{'placed':16} {'raw err':>8} {'filtered err':>13} "
      f"{'raw extent':>22} {'filt extent':>22}")
raws, filts = [], []
for (x, y) in [(0.190, 0.130), (0.170, 0.150), (0.210, 0.120), (0.180, 0.160)]:
    c.request({"op": "reset_props"}, timeout_s=90.0)
    time.sleep(1.0)
    c.request({"op": "place_prop", "name": "pink_cube",
               "pos": [x, y, 0.045]}, timeout_s=60.0)
    time.sleep(1.5)

    t = np.asarray(truth.pose("pink cube"), dtype=float)
    rt._reobserve()
    try:
        frame, fix = rt._localize("pink cube")
    except Exception as e:
        print(f"({x:.3f},{y:.3f})  localize failed: {str(e)[:40]}")
        continue

    pts = fix.points
    c_raw, e_raw, _ = oriented_bbox(pts)
    err_raw = float(np.linalg.norm(np.asarray(c_raw)[:2] - t[:2]))

    # drop table-height points; keep a margin above the table plane
    keep = pts[pts[:, 2] > 0.015]
    if keep.shape[0] < 50:
        print(f"({x:.3f},{y:.3f})  too few points after filtering")
        continue
    c_f, e_f, _ = oriented_bbox(keep)
    err_f = float(np.linalg.norm(np.asarray(c_f)[:2] - t[:2]))

    raws.append(err_raw)
    filts.append(err_f)
    print(f"({x:.3f},{y:.3f})  {err_raw*100:6.2f}cm  {err_f*100:11.2f}cm  "
          f"{str(np.round(e_raw,3)):22} {str(np.round(e_f,3)):22}")

if raws:
    print(f"\nmean raw      {np.mean(raws)*100:.2f} cm")
    print(f"mean filtered {np.mean(filts)*100:.2f} cm")
    print(f"kept fraction of the error: {np.mean(filts)/np.mean(raws):.2f}")
    if np.mean(filts) < np.mean(raws) * 0.6:
        print("-> the table points ARE the main contaminant")
    else:
        print("-> filtering height is not enough; the mask is wrong in 2D")
