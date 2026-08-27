"""Compare three candidate corrections on the SAME captured data.

Apples-to-apples matters here: capture one cloud per position, then apply every
correction to that identical input. Running three separate live sweeps would
confound the corrections with per-run scene variation.

The two measured contributions to the 1.6 cm bias:
  (a) the mask sits ~8 px low because the cube's shadow inflates the bbox
      downward  -> 0.69 cm
  (b) the fitted box centre is pulled 18.3 mm toward the camera by point
      density (near faces get more pixels per unit area)

Candidates:

  A. SHADOW FILTER -- drop mask pixels that are much darker than the object's
     core colour (shadow and table), then refit. Attacks (a) at its source.

  B. KNOWN-SIZE ANCHOR -- ignore the fitted centre. Take the near surface along
     the view ray and push half the object's true size away from the camera.
     Attacks (b) without trusting point density.

  C. CONSTANT OFFSET -- subtract the measured dx=+1.25, dy=-1.00 cm. Cheapest,
     but it only holds for this camera pose and this object.

Report each against physics truth at four positions.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/johnny/Projects/demo/cascade/src")

import cv2
import numpy as np
import yaml

from cascade.apps.demo import build_runtime
from cascade.config import load_demo_config
from cascade.perception.grounding import (mask_to_points_cam, oriented_bbox,
                                           transform_points)
from cascade.sim.bridge_client import BridgeClient
from cascade.sim.truth import TruthPoseReader

CUBE_SIZE = 0.05
BIAS = np.array([0.0125, -0.0100, 0.0])       # measured constant offset

c = BridgeClient(port=8611)
c.connect()
truth = TruthPoseReader(c, ttl_s=0)

T = np.array(yaml.safe_load(
    open("/home/johnny/Projects/demo/cascade/configs/cameras/isaac.yaml")
)["extrinsics"]["T"], dtype=float)
CAM = T[:3, 3]

cfg = load_demo_config(cameras=["isaac"], arm="isaac", llm="mock")
rt, arm = build_runtime(cfg, Path("/tmp/three_way"), view=False)


def shadow_filtered_mask(frame, mask):
    """Drop mask pixels markedly darker than the object's core."""
    hsv = cv2.cvtColor(frame.rgb, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(float)
    s = hsv[:, :, 1].astype(float)
    m = mask.astype(bool)
    if m.sum() < 50:
        return mask
    # the object's core: the brightest, most saturated half of the mask
    v_core = float(np.median(v[m]))
    s_core = float(np.median(s[m]))
    keep = m & (v > 0.65 * v_core) & (s > 0.5 * s_core)
    return keep if keep.sum() > 50 else mask


def centre_from_near_face(pts, cam, size):
    """Anchor on the nearest surface, push half the true size along the ray."""
    rng = np.linalg.norm(pts - cam, axis=1)
    near_idx = rng <= np.percentile(rng, 10)
    near_pt = pts[near_idx].mean(axis=0)
    d = (near_pt - cam)
    d /= np.linalg.norm(d)
    return near_pt + d * (size / 2.0)


rows = []
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
        print(f"({x:.3f},{y:.3f}) localize failed: {str(e)[:40]}")
        continue

    det = fix.detection
    mask = det.mask if det.mask is not None else None
    if mask is None:
        print(f"({x:.3f},{y:.3f}) no mask")
        continue

    base = np.asarray(fix.position, dtype=float)
    e_base = np.linalg.norm(base[:2] - t[:2])

    # A -- shadow filter, refit
    m2 = shadow_filtered_mask(frame, mask)
    p2 = transform_points(T, mask_to_points_cam(frame, m2))
    a_ctr = np.asarray(oriented_bbox(p2)[0], dtype=float)
    e_a = np.linalg.norm(a_ctr[:2] - t[:2])

    # B -- known-size anchor on the original cloud
    b_ctr = centre_from_near_face(fix.points, CAM, CUBE_SIZE)
    e_b = np.linalg.norm(b_ctr[:2] - t[:2])

    # C -- constant offset
    c_ctr = base - BIAS
    e_c = np.linalg.norm(c_ctr[:2] - t[:2])

    # A+B combined
    ab_ctr = centre_from_near_face(p2, CAM, CUBE_SIZE)
    e_ab = np.linalg.norm(ab_ctr[:2] - t[:2])

    rows.append((e_base, e_a, e_b, e_c, e_ab, mask.sum(), m2.sum()))
    print(f"({x:.3f},{y:.3f})  base {e_base*100:5.2f}  A {e_a*100:5.2f}  "
          f"B {e_b*100:5.2f}  C {e_c*100:5.2f}  A+B {e_ab*100:5.2f}   "
          f"mask {int(mask.sum())}->{int(m2.sum())} px")

if rows:
    a = np.array([r[:5] for r in rows])
    names = ["baseline", "A shadow filter", "B size anchor",
             "C const offset", "A+B combined"]
    print("\n  method              mean err   max err")
    for i, n in enumerate(names):
        print(f"  {n:20} {a[:,i].mean()*100:6.2f}cm  {a[:,i].max()*100:6.2f}cm")
    best = int(np.argmin(a.mean(axis=0)))
    print(f"\nbest: {names[best]}  ({a[:,best].mean()*100:.2f} cm mean)")
    print(f"cube half-width is 2.5 cm -- anything under ~1 cm should grasp")
