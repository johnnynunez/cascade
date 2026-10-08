"""B18: wrist-camera extrinsics validated mid-descent against physics truth.

Every wrist frame the bridge serves carries the cam->base matrix `T_base_cam`
taken from the state history matched to that render's rpFabricTime. This probe
drives a slow top-down descent over a resting cube (no contact) and, for every
wrist frame, projects the cube's physics-truth top-face centre through the
frame's own K and T_base_cam and compares it with (a) the cube's colour-mask
centroid in the frame's RGB and (b) the frame's depth at that pixel. A time or
frame-convention error in the dynamic extrinsics shows up as error that grows
with arm speed (moving frames vs static frames).
usage: wrist_reproj.py <port> <engine> <out.json>
"""
import base64
import importlib.util
import json
import math
import sys
import threading
import time
import zlib

import numpy as np

port, engine, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
sys.argv = ["physics_probe"]
spec = importlib.util.spec_from_file_location("pp", "scripts/physics_probe.py")
pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pp)
import cv2  # noqa: E402  (opencv-python-headless is in the shared venv)

from cascade.sim.bridge_client import BridgeClient  # noqa: E402

p = pp.Probe(port, engine)
p.ensure_playing()
p.discover_objects()
CUBE = "green_cube"
HALF = np.array([0.025, 0.025, 0.04])   # the bridge authors 0.05 x 0.05 x 0.08 m boxes (PROPS width, height)
spot = pp.OBJECTS["pink_cube"][0]          # the near spot: hover..+0.03 all reachable
p.home()
p.teleport("pink_cube", (0.20, -0.20, spot[2]), settle_s=1.0)
p.teleport(CUBE, spot, settle_s=1.5)
truth0 = np.array(p.poses([CUBE])[CUBE])
top = truth0 + np.array([0.0, 0.0, HALF[2]])   # visible top-face centre, base frame
_q = p.ex(
    "from isaacsim.core.experimental.prims import RigidPrim\n"
    f"rp = RigidPrim('/World_Props/{CUBE}')\n"
    "_, q = rp.get_world_poses()\n"
    "q = q.numpy() if hasattr(q, 'numpy') else q\n"
    "print([float(x) for x in q[0]], flush=True)\n")
_w, _x, _y, _z = json.loads(_q.strip().splitlines()[-1])
R_cube = np.array([[1 - 2 * (_y * _y + _z * _z), 2 * (_x * _y - _w * _z), 2 * (_x * _z + _w * _y)],
                   [2 * (_x * _y + _w * _z), 1 - 2 * (_x * _x + _z * _z), 2 * (_y * _z - _w * _x)],
                   [2 * (_x * _z - _w * _y), 2 * (_y * _z + _w * _x), 1 - 2 * (_x * _x + _y * _y)]])
corners = np.array([truth0 + R_cube @ (HALF * np.array([sx, sy, sz], dtype=float))
                    for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
refusals = {}
tcp_z = pp.OBJECTS[CUBE][1]
R = pp._yaw_rotation(math.atan2(spot[1], spot[0]))   # the probe's radial jaw yaw (reachable)

frames, stop = [], threading.Event()
cam_cli = BridgeClient(port=port, timeout_s=60.0)
cam_cli.connect()


def grab():
    last = None
    while not stop.is_set():
        try:
            r = cam_cli.request({"op": "frame", "camera": "wrist"})
        except Exception as exc:  # noqa: BLE001 - the bridge refuses unbound frames; count, keep polling
            key_r = (phase[0], str(exc)[:80])
            refusals[key_r] = refusals.get(key_r, 0) + 1
            time.sleep(0.02)
            continue
        ref = r.get("render_reference", {})
        key = (ref.get("producer_epoch"), ref.get("history_physics_step"))
        if r.get("ok") and key != last and r.get("T_base_cam") is not None:
            last = key
            st = p.cli.request({"op": "state"})
            frames.append({"frame": r, "phase": phase[0], "dq": st.get("dq"), "wall": time.monotonic()})
        time.sleep(0.01)


def analyse(rec):
    r = rec["frame"]
    K = np.array(r["K"], dtype=float)
    T = np.array(r["T_base_cam"], dtype=float)
    rgb = cv2.imdecode(np.frombuffer(base64.b64decode(r["rgb_jpeg_b64"]), np.uint8), cv2.IMREAD_COLOR)
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (40, 80, 40), (85, 255, 255))          # green cube
    n = int(mask.sum() // 255)
    p_cam = np.linalg.inv(T) @ np.append(top, 1.0)
    proj = K @ p_cam[:3]
    u, v = proj[0] / proj[2], proj[1] / proj[2]
    res = {"phase": rec["phase"], "speed_rad_s": float(np.linalg.norm(rec["dq"] or [0])),
           "pred_px": [round(u, 1), round(v, 1)], "pred_depth_m": round(float(p_cam[2]), 4), "mask_px": n,
           "render_sim_time": r["render_reference"].get("render_simulation_time"),
           "history_sim_time": r["render_reference"].get("history_simulation_time")}
    pc = (np.linalg.inv(T) @ np.c_[corners, np.ones(8)].T)[:3]
    if np.all(pc[2] > 0.01):
        uv = (K @ pc)[:2] / (K @ pc)[2]
        hull = cv2.convexHull(uv.T.astype(np.float32)).astype(np.int32)
        sil = np.zeros_like(mask)
        cv2.fillConvexPoly(sil, hull, 255)
        inter = int(np.logical_and(sil > 0, mask > 0).sum())
        union = int(np.logical_or(sil > 0, mask > 0).sum())
        res["silhouette_iou"] = round(inter / union, 3) if union else None
        if n > 50 and sil.any():
            ys, xs = np.nonzero(mask)
            sy, sx = np.nonzero(sil)
            res["err_px"] = round(math.hypot(xs.mean() - sx.mean(), ys.mean() - sy.mean()), 2)
            res["mask_centroid_px"] = [round(float(xs.mean()), 1), round(float(ys.mean()), 1)]
            res["sil_centroid_px"] = [round(float(sx.mean()), 1), round(float(sy.mean()), 1)]
    if n > 50:
        cu, cv_ = u, v   # depth is checked AT the projected top-face centre
        if r.get("depth_z_b64"):
            depth = np.frombuffer(zlib.decompress(base64.b64decode(r["depth_z_b64"])), np.float32).reshape(h, w)
            iu, iv = int(round(cu)), int(round(cv_))
            if 0 <= iu < w and 0 <= iv < h:
                patch = depth[max(iv - 2, 0):iv + 3, max(iu - 2, 0):iu + 3]
                res["depth_at_centroid_m"] = round(float(np.median(patch)), 4)
                res["depth_err_m"] = round(res["depth_at_centroid_m"] - float(p_cam[2]), 4)
    return res


phase = ["static_hover"]
p.arm.set_gripper(p.grip_open, effort=1.0)
assert p.goto(R, (spot[0], spot[1], tcp_z + 0.10), 2.5), "hover IK"
time.sleep(1.0)
t = threading.Thread(target=grab, daemon=True)
t.start()
time.sleep(4.0)                                        # static frames at the hover
for leg, (z_from, z_to, dur) in enumerate(((0.10, 0.03, 3.0), (0.03, 0.10, 3.0), (0.10, 0.03, 1.5))):
    phase[0] = f"moving_leg{leg}_{dur}s"
    ik = p.kin.ik(pp.make_transform(R, np.array([spot[0], spot[1], tcp_z + z_to])),
                  np.asarray(p.arm.get_state().q, float))
    assert ik.success, f"IK leg {leg}"
    p.arm.stream_to(ik.q, duration_s=dur)
    p.arm.wait_settled(ik.q, tol=0.03, timeout_s=8.0)
    phase[0] = f"static_after_leg{leg}"
    time.sleep(2.0)
stop.set()
t.join(timeout=5)
truth1 = np.array(p.poses([CUBE])[CUBE])
p.home()
p.restore()

rows = [analyse(f) for f in frames]
summary = {}
for kind in ("static", "moving"):
    errs = [r["err_px"] for r in rows if r["phase"].startswith(kind) and "err_px" in r]
    derr = [abs(r["depth_err_m"]) for r in rows if r["phase"].startswith(kind) and "depth_err_m" in r]
    ious = [r["silhouette_iou"] for r in rows if r["phase"].startswith(kind) and r.get("silhouette_iou") is not None]
    if errs:
        summary[kind] = {"frames": len(errs), "err_px_median": round(float(np.median(errs)), 2),
                         "err_px_p95": round(float(np.percentile(errs, 95)), 2), "err_px_max": round(max(errs), 2),
                         "iou_median": round(float(np.median(ious)), 3) if ious else None,
                         "iou_min": round(min(ious), 3) if ious else None,
                         "depth_abs_err_m_median": round(float(np.median(derr)), 4) if derr else None}
report = {"refusals": {f"{k[0]} | {k[1]}": v for k, v in refusals.items()}, "engine": engine, "cube": CUBE, "cube_moved_m": round(float(np.linalg.norm(truth1 - truth0)), 4),
          "image": [rows[0]["pred_px"]] if rows else None, "summary": summary, "rows": rows}
json.dump(report, open(out, "w"), indent=1)
print(json.dumps({"engine": engine, "frames": len(rows), "by_phase": {ph: sum(1 for r in rows if r["phase"] == ph) for ph in dict.fromkeys(r["phase"] for r in rows)}, "refusals": report["refusals"], "cube_moved_m": report["cube_moved_m"], "summary": summary}))
if rows:
    import cv2 as _cv
    f0 = frames[0]["frame"]
    img = _cv.imdecode(np.frombuffer(base64.b64decode(f0["rgb_jpeg_b64"]), np.uint8), _cv.IMREAD_COLOR)
    T0, K0 = np.array(f0["T_base_cam"]), np.array(f0["K"])
    pc0 = (np.linalg.inv(T0) @ np.c_[corners, np.ones(8)].T)[:3]
    for uvp in ((K0 @ pc0)[:2] / (K0 @ pc0)[2]).T:
        _cv.circle(img, (int(uvp[0]), int(uvp[1])), 3, (0, 0, 255), -1)
    _cv.imwrite(out.replace(".json", "_frame0.jpg"), img)
