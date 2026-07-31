"""Confirm the surface-vs-centre hypothesis by correcting for it.

The depth sensor sees the cube's FRONT FACE; the truth pose is its CENTRE.
Measured at one position:

    true range to centre   0.9336 m
    measured surface range 0.8950 m
    difference             38.6 mm

For a 5 cm cube that is about half a diagonal -- exactly what you expect when
the ray hits a corner-ish face and the reference is the centre.

The consequence is not just a depth error: the pipeline deprojects along the
ray, so a short range places the point closer to the camera AND laterally
offset in base frame, which is the constant dx=+1.30 / dy=-1.00 cm.

Prove it: take the true pixel, deproject with the measured (surface) depth and
with a depth corrected by pushing along the ray by half the cube, and compare
both against truth. If the corrected one lands on the centre, the diagnosis is
confirmed and the fix is well-defined.
"""
import sys
import time

sys.path.insert(0, "/home/johnny/Projects/demo/wrc_demo/src")

import numpy as np
import yaml

from wrc_demo.perception.probe import deproject
from wrc_demo.sim.bridge_client import BridgeClient
from wrc_demo.sim.truth import TruthPoseReader

c = BridgeClient(port=8611)
c.connect()
truth = TruthPoseReader(c, ttl_s=0)

T = np.array(yaml.safe_load(
    open("/home/johnny/Projects/demo/wrc_demo/configs/cameras/isaac.yaml")
)["extrinsics"]["T"], dtype=float)
T_inv = np.linalg.inv(T)

print(f"{'placed':18} {'raw err':>9} {'corrected err':>14}")
raws, cors = [], []
for (x, y) in [(0.190, 0.130), (0.170, 0.150), (0.210, 0.120), (0.180, 0.160)]:
    c.request({"op": "reset_props"}, timeout_s=90.0)
    time.sleep(1.0)
    c.request({"op": "place_prop", "name": "pink_cube",
               "pos": [x, y, 0.045]}, timeout_s=60.0)
    time.sleep(1.5)

    t = np.asarray(truth.pose("pink cube"), dtype=float)
    rgb, depth, K = c.frame("cam0")
    K = np.asarray(K, dtype=float)

    p_cam = T_inv @ np.array([t[0], t[1], t[2], 1.0])
    u = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
    v = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]

    ui, vi = int(round(u)), int(round(v))
    win = depth[max(0, vi-2):vi+3, max(0, ui-2):ui+3]
    win = win[np.isfinite(win) & (win > 0)]
    if not win.size:
        print(f"({x:.3f},{y:.3f})   no depth")
        continue
    z_surf = float(np.median(win))

    # raw: deproject with the surface depth
    p_raw = T @ np.append(deproject(u, v, z_surf, K), 1.0)
    e_raw = float(np.linalg.norm(p_raw[:2] - t[:2]))

    # corrected: push along the ray by half the cube's depth extent.
    # the ray direction in camera frame is the unit vector to the point.
    d = deproject(u, v, 1.0, K)
    d = d / np.linalg.norm(d)
    p_cor_cam = d * (z_surf / d[2] + 0.025)      # +2.5 cm along the ray
    p_cor = T @ np.append(p_cor_cam, 1.0)
    e_cor = float(np.linalg.norm(p_cor[:2] - t[:2]))

    raws.append(e_raw)
    cors.append(e_cor)
    print(f"({x:.3f},{y:.3f})   {e_raw*100:7.2f}cm   {e_cor*100:11.2f}cm")

if raws:
    print(f"\nmean raw error       {np.mean(raws)*100:.2f} cm")
    print(f"mean corrected error {np.mean(cors)*100:.2f} cm")
    if np.mean(cors) < np.mean(raws) / 2:
        print("-> CONFIRMED: the bias is surface-vs-centre depth")
    else:
        print("-> correction did not help; the bias is something else")
