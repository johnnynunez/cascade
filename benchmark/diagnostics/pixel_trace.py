"""Where does the 1.6 cm come from: the pixel, the depth, or the transform?

Established and reproducible to 0.01 cm:
    bias dx=+1.30, dy=-1.00 cm at every table position, both engines,
    at every repetition. Extrinsics match the bridge exactly. Kinematics
    land 0.01 cm from the belief.

The pipeline is: detect -> pixel (u,v) -> depth z -> deproject -> cam->base.
Check each link against ground truth for a single cube:

  1. project the TRUE cube centre into the image using K and the extrinsics.
     That gives the pixel the detector SHOULD report.
  2. read the pixel the detector actually reports.
  3. read the depth at that pixel and compare with the true range.
  4. deproject the TRUE pixel with the TRUE depth and transform back; if that
     round-trips to the true position, K and T are fine and the error is in
     detection (step 2). If it does NOT round-trip, the geometry is wrong.

That isolates the bug to one link instead of guessing.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/johnny/Projects/demo/cascade/src")

import numpy as np
import yaml

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
print(f"true cube centre (base): {np.round(t, 4)}")

cfgp = Path("/home/johnny/Projects/demo/cascade/configs/cameras/isaac.yaml")
y = yaml.safe_load(cfgp.read_text())
T = np.array(y["extrinsics"]["T"], dtype=float)      # cam -> base
print(f"\nT (cam->base):\n{np.round(T, 4)}")

rgb, depth, K = c.frame("cam0")
K = np.asarray(K, dtype=float)
print(f"\nK:\n{np.round(K, 2)}")

# 1. project the true centre into the image
T_inv = np.linalg.inv(T)
p_cam = T_inv @ np.array([t[0], t[1], t[2], 1.0])
xc, yc, zc = p_cam[:3]
u_true = K[0, 0] * xc / zc + K[0, 2]
v_true = K[1, 1] * yc / zc + K[1, 2]
print(f"\ncube in CAMERA frame: {np.round([xc, yc, zc], 4)}")
print(f"projected pixel of the TRUE centre: ({u_true:.1f}, {v_true:.1f})")

# 3. depth the sensor reports there
if depth is not None:
    ui, vi = int(round(u_true)), int(round(v_true))
    H, W = depth.shape[:2]
    if 0 <= ui < W and 0 <= vi < H:
        win = depth[max(0, vi-2):vi+3, max(0, ui-2):ui+3]
        win = win[np.isfinite(win) & (win > 0)]
        z_meas = float(np.median(win)) if win.size else float("nan")
        print(f"depth at that pixel: {z_meas:.4f} m   true zc: {zc:.4f} m   "
              f"diff {(z_meas - zc)*1000:+.1f} mm")

        # 4. round-trip: true pixel + measured depth -> base
        from cascade.perception.probe import deproject
        p = deproject(u_true, v_true, z_meas, K)
        back = T @ np.array([p[0], p[1], p[2], 1.0])
        print(f"\nround-trip back to base: {np.round(back[:3], 4)}")
        err = float(np.linalg.norm(back[:2] - t[:2]))
        print(f"round-trip error: {err*100:.2f} cm")
        if err < 0.005:
            print("-> K and T are CONSISTENT: the 1.6 cm is in DETECTION")
            print("   (the detector reports a pixel that is not the centre)")
        else:
            print("-> geometry itself does not round-trip: K/T mismatch")
    else:
        print("projected pixel is outside the image")
