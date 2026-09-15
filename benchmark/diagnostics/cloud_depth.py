"""The depth component: does the cloud sit on the cube's SURFACE?

Breakdown so far for the 1.63 cm error:
    0.69 cm (42%)  mask centroid sits 8.1 px low (shadow inflates the bbox
                   downward: v 143..264 vs a true 156..253)
    0.94 cm (58%)  unexplained, and the residual is -1.38 cm ALONG THE VIEW
                   RAY, which an image-plane shift cannot produce

A depth-direction error on a point CLOUD has an obvious candidate that my
earlier single-pixel test got right in spirit but tested wrongly: every point
in the cloud lies on the cube's visible SURFACE. Fitting a box to a shell of
surface points and taking its centre gives a centre biased toward the camera
by roughly half the object's unseen depth -- unless the cloud wraps far enough
around to see both sides.

Test it directly on the real cloud: measure how far each point is from the
camera, and compare the cloud's centroid range against the true centre's
range. If the cloud is systematically nearer, that is the missing component,
and the size of the gap says how much.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import yaml

from cascade.apps.demo import build_runtime
from cascade.config import load_demo_config
from cascade.perception.grounding import oriented_bbox
from cascade.sim.bridge_client import BridgeClient
from cascade.sim.truth import TruthPoseReader

c = BridgeClient(port=8611)
c.connect()
truth = TruthPoseReader(c, ttl_s=0)

T = np.array(yaml.safe_load(
    open(Path(__file__).resolve().parents[2] / "configs/cameras/isaac.yaml")
)["extrinsics"]["T"], dtype=float)
CAM = T[:3, 3]

c.request({"op": "reset_props"}, timeout_s=90.0)
time.sleep(1.0)
c.request({"op": "place_prop", "name": "pink_cube",
           "pos": [0.190, 0.130, 0.045]}, timeout_s=60.0)
time.sleep(1.5)

t = np.asarray(truth.pose("pink cube"), dtype=float)
cfg = load_demo_config(cameras=["isaac"], arm="isaac", llm="mock")
rt, arm = build_runtime(cfg, Path("/tmp/newton_depthcmp"), view=False)
rt._reobserve()
frame, fix = rt._localize("pink cube")

pts = fix.points
pts = pts[pts[:, 2] > 0.015]                       # drop the table skirt
rng = np.linalg.norm(pts - CAM, axis=1)
true_rng = float(np.linalg.norm(t - CAM))

print(f"true centre range from camera : {true_rng:.4f} m")
print(f"cloud range  min {rng.min():.4f}  median {np.median(rng):.4f}  "
      f"max {rng.max():.4f}")
print(f"\nmedian cloud range - true centre range = "
      f"{(np.median(rng) - true_rng)*1000:+.1f} mm")

ctr, ext, _ = oriented_bbox(pts)
ctr = np.asarray(ctr)
ctr_rng = float(np.linalg.norm(ctr - CAM))
print(f"fitted box centre range = {ctr_rng:.4f} m  "
      f"({(ctr_rng - true_rng)*1000:+.1f} mm vs truth)")
print(f"fitted extent = {np.round(ext, 3)}  (true cube 0.05 each)")

# how much of the cube's far side is missing?
print(f"\ncloud depth span along the ray: {(rng.max()-rng.min())*1000:.1f} mm")
print("(a fully wrapped 5 cm cube would span about 50-70 mm)")

err = ctr - t
view = (t - CAM) / np.linalg.norm(t - CAM)
along = float(np.dot(err, view))
print(f"\ncentre error {np.linalg.norm(err[:2])*100:.2f} cm, "
      f"of which {along*100:+.2f} cm along the view ray")
if along < -0.005:
    print("-> the fitted centre sits TOWARD the camera: surface-shell bias,")
    print("   on top of the mask shift. Two independent contributions.")
