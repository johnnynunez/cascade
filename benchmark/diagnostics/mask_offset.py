"""Where is the mask, in pixels, versus where the cube actually projects?

Height filtering barely helped (1.63 -> 1.57 cm) and the filtered extent is
still 7.3 x 7.5 cm for a 5 cm cube. So the table points are a symptom, not the
cause: the mask is wrong in the IMAGE, and everything downstream inherits it.

Compare, in pixel space:
  * the polygon the true cube projects to (8 corners via K and T)
  * the detector's bbox and mask centroid

If the mask sits systematically below/right of the true projection, that is a
2D detection offset -- and the fact that it is identical on both engines and
at every table position means it is deterministic, not noise.

This prints the offset in pixels and converts it to the world error it would
cause at this depth, which tells us whether the 2D shift fully explains the
1.6 cm.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import cv2
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
    open(Path(__file__).resolve().parents[2] / "configs/cameras/isaac.yaml")
)["extrinsics"]["T"], dtype=float)
T_inv = np.linalg.inv(T)

c.request({"op": "reset_props"}, timeout_s=90.0)
time.sleep(1.0)
c.request({"op": "place_prop", "name": "pink_cube",
           "pos": [0.190, 0.130, 0.045]}, timeout_s=60.0)
time.sleep(1.5)

t = np.asarray(truth.pose("pink cube"), dtype=float)
cfg = load_demo_config(cameras=["isaac"], arm="isaac", llm="mock")
rt, arm = build_runtime(cfg, Path("/tmp/newton_px"), view=False)
rt._reobserve()
frame, fix = rt._localize("pink cube")
K = np.asarray(frame.K, dtype=float)


def project(p):
    q = T_inv @ np.array([p[0], p[1], p[2], 1.0])
    return (K[0, 0] * q[0] / q[2] + K[0, 2], K[1, 1] * q[1] / q[2] + K[1, 2])


# true cube corners, 5 cm side centred on the truth pose
h = 0.025
corners = [t + np.array([sx * h, sy * h, sz * h])
           for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
proj = np.array([project(p) for p in corners])
u_true = proj[:, 0].mean()
v_true = proj[:, 1].mean()
print(f"true cube projects to u {proj[:,0].min():.0f}..{proj[:,0].max():.0f}, "
      f"v {proj[:,1].min():.0f}..{proj[:,1].max():.0f}")
print(f"  true centroid pixel: ({u_true:.1f}, {v_true:.1f})")

det = fix.detection
x0, y0, x1, y1 = [float(v) for v in det.bbox]
print(f"detector bbox: u {x0:.0f}..{x1:.0f}, v {y0:.0f}..{y1:.0f}")
print(f"  bbox centre: ({(x0+x1)/2:.1f}, {(y0+y1)/2:.1f})")

if det.mask is not None:
    m = det.mask.astype(bool)
    vs, us = np.nonzero(m)
    print(f"mask centroid: ({us.mean():.1f}, {vs.mean():.1f})  "
          f"({m.sum()} px)")
    du = us.mean() - u_true
    dv = vs.mean() - v_true
    print(f"\nMASK OFFSET: du {du:+.1f} px   dv {dv:+.1f} px")

    # what world error does that pixel offset cause at this range?
    q = T_inv @ np.array([t[0], t[1], t[2], 1.0])
    z = q[2]
    wx = du * z / K[0, 0]
    wy = dv * z / K[1, 1]
    print(f"at range {z:.3f} m that is {wx*100:+.2f} cm x {wy*100:+.2f} cm "
          f"in the camera plane ({np.hypot(wx,wy)*100:.2f} cm)")
    print("(pipeline error measured earlier: 1.63 cm)")
