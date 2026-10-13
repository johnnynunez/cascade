# SPDX-License-Identifier: Apache-2.0
"""Offline replay of a robot's onboard estimator on a recorded honest episode (diagnostics only).

Feeds the recorded trunk motion (IMU model), the robot's own recorded head-camera video and ToF frames into a fresh
``Onboard`` estimator and prints its error against simulator truth. Used to tune the estimator BETWEEN episodes;
nothing here runs inside an episode. The video is H.264-compressed, so cuVSLAM sees slightly different pixels than
it did live.
    labpy.sh <wt> tools/onboard_replay.py --run runs/skateH3 [--set attr=value ...]
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

for extra in filter(None, os.environ.get("CASCADE_APPEND_SYSPATH", "").split(os.pathsep)):
    sys.path.append(extra)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--wt", type=Path, default=Path.home() / "Projects/cascade-wt/feat-microduck-agentic-skating")
    p.add_argument("--set", action="append", default=[], help="Onboard attribute override, e.g. vo_gap_fill=1")
    p.add_argument("--every", type=float, default=2.0)
    p.add_argument("--trace", default="", help="lo,hi: print every camera frame in this sim-time window")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    sys.path.insert(0, str(a.wt / "src"))
    from cascade.sim import microduck_onboard as mo
    from cascade.sim.microduck_honest import WalkerAdapter

    z = np.load(a.run / "recording.npz")
    meta = json.loads((a.run / "recording.json").read_text())
    names = [str(b) for b in z["body_names"]]
    robot = str(z["robot_names"][0])
    bp_all = z["body_pose"].astype(np.float64)
    times = z["time_s"]
    onb = np.load(a.run / "onboard.npz")
    cam_t = onb["cam_times"]
    tofs = onb[f"tof_{robot}"]
    skate = meta.get("robot_kind") == "skates"
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(a.run / f"headcam_{robot}.mp4"), "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, mo.CAM_H, mo.CAM_W, 3)

    class Ad:
        trunk = names.index("trunk_base")
        head = names.index("jaw_soft")
        contacts = ([k for k, n in enumerate(names) if n.startswith("tire")] if skate else
                    [names.index("ankle_left"), names.index("ankle_right")])
        lens_T = WalkerAdapter.lens_T
        trunk_T = WalkerAdapter.trunk_T
        contacts_b = WalkerAdapter.contacts_b

    ad = Ad()
    dt = float(meta.get("control_dt_s", 0.02))
    ob = mo.Onboard(robot, dt, seed=a.seed, tof_from_depth=skate)
    tr = [float(x) for x in a.trace.split(",")] if a.trace else None
    for kv in a.set:
        k, v = kv.split("=")
        setattr(ob, k, type(getattr(ob, k))(float(v)) if hasattr(ob, k) else float(v))
    t0 = float(cam_t[0])
    k_start = int(np.searchsorted(times, t0 - 1e-6))
    prev_p = None
    ci = 0
    rows = []
    s0 = None
    for k in range(len(times)):
        bp = bp_all[k:k + 1, 0] if bp_all.ndim == 4 else bp_all[k:k + 1]
        Tt = ad.trunk_T(bp, 0)
        v = np.zeros(3) if prev_p is None else (Tt[:3, 3] - prev_p) / dt
        prev_p = Tt[:3, 3].copy()
        ob.imu_step(Tt[:3, :3], v, np.zeros(3))
        ob.contact_step(ad.contacts_b(bp, 0))
        if not skate:
            ob.leg_step(ad.contacts_b(bp, 0))
        if k == k_start and not ob.started:
            ob.start(mo.T_inv(Tt) @ ad.lens_T(bp, 0), float(Tt[2, 3]))
            s0 = (Tt[0, 3], Tt[1, 3], math.atan2(Tt[1, 0], Tt[0, 0]))
        while ob.started and ci < len(cam_t) and cam_t[ci] <= times[k] + 1e-6 and ci < len(frames):
            ob.update_camera(float(cam_t[ci]), frames[ci], None, mo.T_inv(Tt) @ ad.lens_T(bp, 0),
                             (tofs[ci][:64], tofs[ci][64:]))
            if tr and tr[0] <= cam_t[ci] <= tr[1]:
                e = ob.estimate()
                print(f"  cam t={cam_t[ci]:.3f} x={e['x']:.3f} y={e['y']:.3f} yaw={e['yaw_rad']:.3f} nobs={ob.loc.n_obs} "
                      f"acc={ob.vo_accepted} rej={ob.vo_rejected}", flush=True)
            ci += 1
        if ob.started and (k - k_start) % int(a.every / dt) == 0:
            e = ob.estimate()
            c, s = math.cos(s0[2]), math.sin(s0[2])
            dx, dy = Tt[0, 3] - s0[0], Tt[1, 3] - s0[1]
            tx, ty = c * dx + s * dy, -s * dx + c * dy
            rows.append((float(times[k]), e["x"], e["y"], tx, ty))
            print(f"t={times[k]:6.2f} est=({e['x']:5.2f},{e['y']:5.2f}) truth=({tx:5.2f},{ty:5.2f}) "
                  f"err={math.hypot(e['x'] - tx, e['y'] - ty):.2f} vo acc/rej/lost={ob.vo_accepted}/{ob.vo_rejected}/{ob.vo_lost} flow={ob.flow_used} {getattr(ob.flow, 'last', '')}",
                  flush=True)
    r = np.array(rows)
    err = np.hypot(r[:, 1] - r[:, 3], r[:, 2] - r[:, 4])
    path = float(np.sum(np.hypot(np.diff(r[:, 3]), np.diff(r[:, 4]))))
    print("REPLAY", json.dumps({"final_err_m": round(float(err[-1]), 3), "max_err_m": round(float(err.max()), 3),
                                "path_m": round(path, 2), "set": a.set}), flush=True)


if __name__ == "__main__":
    main()
