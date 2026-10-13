# SPDX-License-Identifier: Apache-2.0
"""Onboard sensing and localization of a simulated MicroDuck, with nothing privileged in the robot/agent loop.

Sensors (real MicroDuck rig, simulated):
* head RGB camera 640x360 @ 15 Hz on the jaw_soft lens (12 mm / 20.955 mm -> 82.3 deg HFOV), rendered by OVRTX in a
  separate SENSOR PROCESS from the live Newton body poses of every robot in the scene (robots see each other);
* ToF 8x8 (VL53L8CX): the walker world uses the gtceu model; worlds without it (skates) use :class:`TofFromDepth`,
  the same zone geometry (45 x 45 deg, 4 m, status 5 valid / 255 no target, < 0.1 m unusable, 1 % + 5 mm noise)
  applied to the rendered depth of the head camera;
* IMU in the trunk: gyro + accelerometer from the trunk state at the control rate, white noise + constant bias;
* joint encoders (forward kinematics gives the camera / ToF extrinsics relative to the trunk).

Localization (:class:`OnboardLocalizer`): cuVSLAM RGBD odometry on the head RGB with a depth image built from the
robot's OWN ToF zones. Pose frame = the robot's start frame E (x forward, y left at the start, z up from the IMU
gravity, z = 0 on the floor it stands on). Tracking loss is reported, never corrected from simulator truth.
Simulator truth is only read for (a) rendering the sensors, (b) forward kinematics of the head (a function of the joint
angles), (c) one start calibration constant (standing trunk height), and (d) offline evaluation.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

CAM_W, CAM_H = 640, 360
CAM_FX = CAM_W * 12.0 / 20.955
CAM_HZ = 15.0
TOF_HALF_FOV = math.radians(22.5)
TOF_MAX_M = 4.0
IMU_NOISE = dict(gyroscope_noise_density=2.4e-4, gyroscope_random_walk=2.0e-5,
                 accelerometer_noise_density=1.9e-3, accelerometer_random_walk=3.0e-3)
GRAVITY = 9.81


# ---------------------------------------------------------------------------------------------- small math
def quat_to_R(q):
    q = np.asarray(q, np.float64)
    x, y, z, w = (float(v) for v in q / max(np.linalg.norm(q), 1e-12))
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def T_from(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def T_inv(T):
    R, p = T[:3, :3], T[:3, 3]
    return T_from(R.T, -R.T @ p)


def optical_from_lens(lens_R):
    """Lens frame (Isaac Lab 'world' camera convention: +X forward, +Z up) -> optical (+X right, +Y down, +Z fwd)."""
    f, up = lens_R[:, 0] / np.linalg.norm(lens_R[:, 0]), lens_R[:, 2]
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    return np.stack([r, d / np.linalg.norm(d), f], 1)


def yaw_of_R(R):
    return math.atan2(R[1, 0], R[0, 0])


# ---------------------------------------------------------------------------------------------- ToF from depth
def tof_zone_rays():
    """Unit rays of the 64 zone centres in the OPTICAL camera frame (row 0 top, column 0 = robot left)."""
    cell = 2 * TOF_HALF_FOV / 8
    rays = np.zeros((64, 3))
    for z in range(64):
        r, c = divmod(z, 8)
        az = TOF_HALF_FOV - (c + 0.5) * cell  # + left
        el = TOF_HALF_FOV - (r + 0.5) * cell  # + up
        d = np.array([-math.tan(az), -math.tan(el), 1.0])
        rays[z] = d / np.linalg.norm(d)
    return rays


class TofFromDepth:
    """VL53L8CX zone model on the head camera's rendered depth (ToF ~2 cm beside the lens: treated as coincident)."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        cell = 2 * TOF_HALF_FOV / 8
        self.boxes = []
        for z in range(64):
            r, c = divmod(z, 8)
            az0, az1 = TOF_HALF_FOV - c * cell, TOF_HALF_FOV - (c + 1) * cell
            el0, el1 = TOF_HALF_FOV - r * cell, TOF_HALF_FOV - (r + 1) * cell
            u0, u1 = sorted((CAM_W / 2 - CAM_FX * math.tan(az0), CAM_W / 2 - CAM_FX * math.tan(az1)))
            v0, v1 = sorted((CAM_H / 2 - CAM_FX * math.tan(el0), CAM_H / 2 - CAM_FX * math.tan(el1)))
            self.boxes.append((int(v0), int(math.ceil(v1)), int(u0), int(math.ceil(u1))))
        self.rays = tof_zone_rays()

    def measure(self, depth_m):
        """depth_m: (H, W) distance to the image plane, 0 = none. Returns (mm int32 (64,), status uint8 (64,))."""
        mm = np.zeros(64, np.int32)
        st = np.full(64, 255, np.uint8)
        for z, (v0, v1, u0, u1) in enumerate(self.boxes):
            patch = depth_m[v0:v1, u0:u1]
            vals = patch[(patch > 0.0) & np.isfinite(patch)]
            if vals.size < 0.2 * patch.size:
                continue
            zd = float(np.percentile(vals, 30))  # strongest near target dominates the histogram
            rng_m = zd / self.rays[z][2]
            if rng_m > TOF_MAX_M:
                continue
            rng_m += self.rng.normal(0.0, 0.005 + 0.01 * rng_m)
            mm[z] = int(round(rng_m * 1000))
            st[z] = 5 if rng_m >= 0.1 else 0
        return mm, st


def tof_depth_image(mm, status):
    """Splat the ToF zones into the camera's depth image (uint16 millimetres along the optical axis) for cuVSLAM."""
    depth = np.zeros((CAM_H, CAM_W), np.uint16)
    cell = 2 * TOF_HALF_FOV / 8
    rays = tof_zone_rays()
    for z in range(64):
        if int(status[z]) not in (5, 9) or int(mm[z]) < 100:
            continue
        r, c = divmod(z, 8)
        az0, az1 = TOF_HALF_FOV - c * cell, TOF_HALF_FOV - (c + 1) * cell
        el0, el1 = TOF_HALF_FOV - r * cell, TOF_HALF_FOV - (r + 1) * cell
        u0, u1 = sorted((CAM_W / 2 - CAM_FX * math.tan(az0), CAM_W / 2 - CAM_FX * math.tan(az1)))
        v0, v1 = sorted((CAM_H / 2 - CAM_FX * math.tan(el0), CAM_H / 2 - CAM_FX * math.tan(el1)))
        depth[int(max(v0, 0)):int(min(v1, CAM_H)), int(max(u0, 0)):int(min(u1, CAM_W))] = int(mm[z] * rays[z][2])
    return depth


_RAYS_CAM = None


def tof_floor_depth_image(mm, status, T_E_cam, zone_kind):
    """Depth image for cuVSLAM from the robot's own ToF: pixels of zones the ToF classified as FLOOR get the depth of
    the floor plane (z = 0 in the robot's frame E) along their own pixel ray, using the estimated camera pose; zones
    classified as obstacles keep their ToF range; everything else has no depth. uint16 millimetres."""
    global _RAYS_CAM
    if _RAYS_CAM is None:
        u, v = np.meshgrid(np.arange(CAM_W), np.arange(CAM_H))
        _RAYS_CAM = np.stack([(u - (CAM_W - 1) / 2) / CAM_FX, (v - (CAM_H - 1) / 2) / CAM_FX, np.ones_like(u, float)], -1)
    depth = tof_depth_image(mm, np.where(np.asarray(zone_kind).reshape(64) == 2, status, 255))  # obstacles only
    rz = _RAYS_CAM @ T_E_cam[2, :3]  # z component in E of each pixel ray (camera z = 1)
    h = T_E_cam[2, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(rz < -1e-3, -h / rz, 0.0)
    cell = 2 * TOF_HALF_FOV / 8
    floor = np.zeros((CAM_H, CAM_W), bool)
    for z in range(64):
        if int(np.asarray(zone_kind).reshape(64)[z]) != 1:
            continue
        r, c = divmod(z, 8)
        az0, az1 = TOF_HALF_FOV - c * cell, TOF_HALF_FOV - (c + 1) * cell
        el0, el1 = TOF_HALF_FOV - r * cell, TOF_HALF_FOV - (r + 1) * cell
        u0, u1 = sorted((CAM_W / 2 - CAM_FX * math.tan(az0), CAM_W / 2 - CAM_FX * math.tan(az1)))
        v0, v1 = sorted((CAM_H / 2 - CAM_FX * math.tan(el0), CAM_H / 2 - CAM_FX * math.tan(el1)))
        floor[int(max(v0, 0)):int(min(v1, CAM_H)), int(max(u0, 0)):int(min(u1, CAM_W))] = True
    sel = floor & (t > 0.05) & (t < TOF_MAX_M)
    depth[sel] = np.round(t[sel] * 1000).astype(np.uint16)
    return depth


def _so3_log(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    th = math.acos(c)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-6:
        return 0.5 * w
    return th / (2 * math.sin(th)) * w


def _so3_exp(v):
    th = float(np.linalg.norm(v))
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    if th < 1e-9:
        return np.eye(3) + K
    return np.eye(3) + math.sin(th) / th * K + (1 - math.cos(th)) / th ** 2 * (K @ K)


# ---------------------------------------------------------------------------------------------- IMU
class ImuModel:
    """Trunk IMU: gyro (rad/s, body) and specific force (m/s^2, body) with white noise and a constant bias.

    The gyro is reported like a high-rate IMU that integrates internally (coning-compensated delta angle over the
    control period divided by the period): sampling the instantaneous rate at 50 Hz aliases the fast trunk
    oscillations of skating pushes into heading drift, which a real 1 kHz IMU does not suffer."""

    def __init__(self, dt, seed=0):
        self.dt = float(dt)
        self.rng = np.random.default_rng(seed)
        self.gyro_bias = self.rng.normal(0, 0.002, 3)
        self.acc_bias = self.rng.normal(0, 0.02, 3)
        self.prev_v = None
        self.prev_R = None

    def sample(self, R_wb, v_w, w_b):
        a_w = np.zeros(3) if self.prev_v is None else (v_w - self.prev_v) / self.dt
        self.prev_v = np.array(v_w, dtype=np.float64)
        f = R_wb.T @ (a_w + np.array([0.0, 0.0, GRAVITY]))
        s = 1.0 / math.sqrt(self.dt)
        rate = np.asarray(w_b, np.float64)
        if self.prev_R is not None:
            rate = _so3_log(self.prev_R.T @ R_wb) / self.dt  # delta angle over the period (body frame)
        self.prev_R = np.array(R_wb, np.float64)
        gyro = rate + self.gyro_bias + self.rng.normal(0, IMU_NOISE["gyroscope_noise_density"] * s, 3)
        acc = f + self.acc_bias + self.rng.normal(0, IMU_NOISE["accelerometer_noise_density"] * s, 3)
        return gyro, acc


# ---------------------------------------------------------------------------------------------- sensor process
class OvrtxSensorClient:
    """Head cameras of all robots, rendered by OVRTX in a child process from the live Newton body poses."""

    def __init__(self, *, course_usd, robot_usd, robots, body_names, light_scale=1.0, device=0, log_path=None):
        from multiprocessing.connection import Listener

        self.listener = Listener(("127.0.0.1", 0), authkey=b"cascade-sensors")
        port = self.listener.address[1]
        env = dict(os.environ)
        cmd = [sys.executable, "-m", "cascade.sim.microduck_onboard", "serve", str(port)]
        self.log = open(log_path, "ab") if log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(cmd, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        self.conn = self.listener.accept()
        self.conn.send({"course_usd": str(course_usd) if course_usd else None, "robot_usd": str(robot_usd),
                        "robots": list(robots), "body_names": list(body_names), "light_scale": light_scale,
                        "device": device})
        ok = self.conn.recv()
        if not (isinstance(ok, dict) and ok.get("ready")):
            raise RuntimeError(f"sensor process failed to start: {ok}")
        self.info = ok
        self.n_frames = 0

    def render(self, t_s, body_pose, cam_T):
        """body_pose (R, B, 7) xyzw world; cam_T: list of 4x4 OPTICAL camera-to-world. Returns [(rgb, depth)]."""
        self.conn.send({"t": float(t_s), "body_pose": np.asarray(body_pose, np.float32),
                        "cam_T": [np.asarray(T, np.float64) for T in cam_T]})
        out = self.conn.recv()
        if isinstance(out, dict) and out.get("error"):
            raise RuntimeError("sensor process: " + out["error"])
        self.n_frames += 1
        return out

    def close(self):
        try:
            self.conn.send(None)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.proc.wait(20)
        except Exception:  # noqa: BLE001
            self.proc.kill()


def _serve(port):
    from multiprocessing.connection import Client

    for extra in filter(None, os.environ.get("CASCADE_APPEND_SYSPATH", "").split(os.pathsep)):
        sys.path.append(extra)
    from .microduck_ovrtx_replay import ReplayPoses, build_render_scene
    from .ovrtx_renderer import CameraSpec, OvrtxRenderer, SceneSnapshot

    conn = Client(("127.0.0.1", port), authkey=b"cascade-sensors")
    cfg = conn.recv()
    try:
        import tempfile

        tmp = Path(tempfile.mkdtemp(prefix="cascade_sensors_"))
        layout = build_render_scene(tmp / "scene.usdc", course_usd=cfg["course_usd"], microduck_usd=cfg["robot_usd"],
                                    robots=cfg["robots"], floor_size=24.0, light_scale=cfg["light_scale"])
        replay = ReplayPoses(layout, cfg["body_names"], cfg["robots"])
        cams = [CameraSpec(f"head_{i}", CAM_W, CAM_H, CAM_FX, CAM_FX, np.eye(4), near_m=0.01, far_m=30.0)
                for i in range(len(cfg["robots"]))]
        renderer = OvrtxRenderer(tmp / "scene.usdc", cams, dynamic_paths=replay.paths, device=cfg["device"],
                                 render_dt=1.0 / CAM_HZ,
                                 sdk_versions=(("ovrtx", "0.5.1.385782"), ("ovstage", "0.2.1.385922")))
        renderer.__enter__()
        conn.send({"ready": True, "robot_variant": layout.get("robot_variant"), "robot_usd_sha256": layout["microduck_usd_sha256"]})
    except Exception as exc:  # noqa: BLE001
        conn.send({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return
    seq = 0
    while True:
        msg = conn.recv()
        if msg is None:
            break
        try:
            seq += 1
            snap = SceneSnapshot("cascade.microduck.onboard_sensors", "head", seq, msg["t"], time.monotonic(),
                                 replay.transforms(msg["body_pose"]),
                                 {f"head_{i}": T for i, T in enumerate(msg["cam_T"])})
            frames = renderer.render(snap)
            out = []
            for i in range(len(msg["cam_T"])):
                f = frames[f"head_{i}"]
                out.append((np.ascontiguousarray(f.rgb[..., ::-1]), np.asarray(f.depth_m, np.float32)))
            conn.send(out)
        except Exception as exc:  # noqa: BLE001
            conn.send({"error": f"{type(exc).__name__}: {exc}"})
    renderer.__exit__(None, None, None)


# ---------------------------------------------------------------------------------------------- floor flow
class FloorFlow:
    """Planar visual odometry on the floor (the classic ground-robot method): corners on the floor are tracked with
    pyramidal Lucas-Kanade between consecutive head-camera frames and intersected with the floor plane using each
    frame's ESTIMATED camera pose (AHRS attitude, head forward kinematics, contact-FK height). Back-projecting the
    new pixels from the camera placed at the PREVIOUS base position shifts every floor point by minus the base's
    horizontal motion, so the motion is the robust median of the point shifts. Pixels of ToF obstacle zones are
    excluded; outliers (things above the floor) are rejected by their distance to the median."""

    def __init__(self, min_points=12, near_m=0.2, far_m=2.0):
        import cv2

        self.cv2 = cv2
        self.prev = None  # (gray, T_E_cam at that frame)
        self.min_points = min_points
        self.near_m, self.far_m = near_m, far_m
        u, v = np.meshgrid(np.arange(CAM_W, dtype=np.float64), np.arange(CAM_H, dtype=np.float64))
        self.rays = np.stack([(u - (CAM_W - 1) / 2) / CAM_FX, (v - (CAM_H - 1) / 2) / CAM_FX, np.ones_like(u)], -1)
        self.last = {}

    def _floor_hit(self, pts, T):
        r = np.stack([(pts[:, 0] - (CAM_W - 1) / 2) / CAM_FX, (pts[:, 1] - (CAM_H - 1) / 2) / CAM_FX,
                      np.ones(len(pts))], 1) @ T[:3, :3].T
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(r[:, 2] < -1e-3, -T[2, 3] / r[:, 2], np.inf)
        X = T[:3, 3][None] + r * t[:, None]
        dist = np.hypot(X[:, 0] - T[0, 3], X[:, 1] - T[1, 3])
        return X[:, :2], (t > 0) & (dist > self.near_m) & (dist < self.far_m)

    def step(self, rgb, T_now_prevpos, obstacle_mask=None):
        """rgb: current frame; T_now_prevpos: current camera pose with the base at its PREVIOUS position.
        Returns the base's horizontal motion (2,) in E since the previous frame, or None."""
        cv2 = self.cv2
        gray = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
        out = None
        self.last = {"points": 0, "inliers": 0}
        if self.prev is not None:
            g0, T0 = self.prev
            r = self.rays @ T0[:3, :3].T
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(r[..., 2] < -1e-3, -T0[2, 3] / r[..., 2], np.inf)
            mask = ((t * np.hypot(r[..., 0], r[..., 1]) > self.near_m)
                    & (t * np.hypot(r[..., 0], r[..., 1]) < self.far_m)).astype(np.uint8) * 255
            if obstacle_mask is not None:
                mask[obstacle_mask] = 0
            p0 = cv2.goodFeaturesToTrack(g0, maxCorners=300, qualityLevel=0.01, minDistance=7, mask=mask)
            if p0 is not None and len(p0) >= self.min_points:
                p1, st1, _ = cv2.calcOpticalFlowPyrLK(g0, gray, p0, None, winSize=(21, 21), maxLevel=3)
                pb, st2, _ = cv2.calcOpticalFlowPyrLK(gray, g0, p1, None, winSize=(21, 21), maxLevel=3)
                ok = (st1[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm((pb - p0)[:, 0], axis=1) < 1.0)
                a, b = p0[ok, 0].astype(np.float64), p1[ok, 0].astype(np.float64)
                X0, v0 = self._floor_hit(a, T0)
                X1, v1 = self._floor_hit(b, T_now_prevpos)
                sel = v0 & v1
                self.last["points"] = int(sel.sum())
                if sel.sum() >= self.min_points:
                    d = X0[sel] - X1[sel]
                    med = np.median(d, axis=0)
                    res = np.linalg.norm(d - med, axis=1)
                    inl = res < max(0.01, 2.5 * np.median(res))
                    self.last["inliers"] = int(inl.sum())
                    if inl.sum() >= self.min_points:
                        out = np.median(d[inl], axis=0)
        self.prev = (gray, None)
        return out

    def set_prev_pose(self, T):
        if self.prev is not None:
            self.prev = (self.prev[0], np.array(T))


# ---------------------------------------------------------------------------------------------- localization
class OnboardLocalizer:
    """cuVSLAM RGBD odometry (head RGB + depth from the robot's own ToF) expressed in the robot's start frame E."""

    def __init__(self, name):
        import cuvslam

        self.cuvslam = cuvslam
        self.name = name
        cam = cuvslam.Camera()
        cam.size, cam.focal, cam.principal = (CAM_W, CAM_H), (CAM_FX, CAM_FX), ((CAM_W - 1) / 2, (CAM_H - 1) / 2)
        rig = cuvslam.Rig()
        rig.cameras = [cam]
        cfg = cuvslam.Tracker.OdometryConfig(
            odometry_mode=cuvslam.Tracker.OdometryMode.RGBD, async_sba=False, enable_observations_export=True,
            rgbd_settings=cuvslam.Tracker.OdometryRGBDSettings(depth_scale_factor=1000.0, depth_camera_id=0,
                                                              enable_depth_stereo_tracking=False))
        self.tracker = cuvslam.Tracker(rig, cfg)
        self.T_E_odom = None
        self.T_E_cam = None
        self.lost_frames = 0
        self.frames = 0
        self.last_ok_t = None
        self.n_obs = 0
        self.version = cuvslam.get_version() if hasattr(cuvslam, "get_version") else "17.0.0"

    def start(self, T_E_cam0):
        self.T_E_cam = np.array(T_E_cam0)

    def track(self, t_s, rgb, tof_mm, tof_status, depth_image=None):
        if depth_image is None:
            depth_image = tof_depth_image(tof_mm, tof_status)
        pe, _ = self.tracker.track(int(round(t_s * 1e9)), [rgb], depths=[depth_image])
        self.frames += 1
        try:
            self.n_obs = len(self.tracker.get_last_observations(0))
        except Exception:  # noqa: BLE001
            self.n_obs = -1
        if pe.world_from_rig is None:
            self.lost_frames += 1
            return False
        p = pe.world_from_rig.pose
        T_odom_cam = T_from(quat_to_R(list(p.rotation)), np.array(list(p.translation), np.float64))
        if self.T_E_odom is None:
            self.T_E_odom = self.T_E_cam @ T_inv(T_odom_cam)
        self.T_E_cam = self.T_E_odom @ T_odom_cam
        self.last_ok_t = t_s
        return True


def _Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Onboard:
    """Everything one robot knows about itself: its sensors' latest readings and its own pose estimate.

    Loosely coupled odometry in the start frame E: heading = yaw of an AHRS attitude (gyro on SO(3) + accelerometer tilt
    correction); position = leg odometry from encoder forward kinematics (walkers) blended with cuVSLAM RGBD translation
    increments (rotated into E with the AHRS heading), gated: an increment is rejected when it is faster than
    ``max_speed_m_s`` or its heading change disagrees with the gyro by more than ``max_yaw_disagree_deg`` (cuVSLAM can
    jump without reporting a loss). Height from contact-body forward kinematics."""

    def __init__(self, name, dt, seed=0, tof_from_depth=False, max_speed_m_s=1.2, max_yaw_disagree_deg=12.0):
        self.name = name
        self.dt = float(dt)
        self.imu = ImuModel(dt, seed)
        self.loc = OnboardLocalizer(name)
        self.tof_model = TofFromDepth(seed) if tof_from_depth else None
        # skaters (no leg odometry): floor-flow odometry is the primary translation source, cuVSLAM the fallback
        self.flow = FloorFlow() if tof_from_depth else None
        self.flow_used = 0
        self.rgb = None
        self.depth = None
        self.tof = None  # (mm, status)
        self.gyro = np.zeros(3)
        self.acc = np.array([0.0, 0.0, GRAVITY])
        self.acc_lp = np.array([0.0, 0.0, GRAVITY])
        self.up_est = np.array([0.0, 0.0, 1.0])  # attitude filter: world "up" in body coordinates
        self.R_att = np.eye(3)  # AHRS attitude, body -> start frame E
        self.att_gain = 0.5  # 1/s accelerometer correction (time constant 2 s), gated on |a| ~ g
        self.T_cam_base = None
        self.T_base_cam = None
        self.T_cam_tof = np.eye(4)
        self.tof_rays_sensor = tof_zone_rays()
        self.h0 = None
        self.started = False
        self.yaw_f = 0.0
        self.p_f = np.zeros(2)
        self.vo_prev = None  # (x, y, yaw) of the cuVSLAM base pose at the previous accepted/seen frame
        self.yaw_at_prev = 0.0
        self.max_speed, self.max_yaw_dis = float(max_speed_m_s), math.radians(max_yaw_disagree_deg)
        self.vo_accepted = 0
        self.vo_rejected = 0
        self.vo_lost = 0
        self.prev_xy = None
        self.speed = 0.0
        self.track_ok = True
        self.last_camera_t = None
        self.contact_offset = None  # contact point below the lowest contact body origin (calibrated at start)
        self.h_cur = None
        self.leg_prev = None  # (stance index, stance foot position in E-rotation frame)
        self.leg_since_frame = np.zeros(2)
        self.has_leg_odom = False
        # Leg odometry scale: ONE calibration constant measured once on a straight calibration walk
        # (runs/honest_smoke3: 0.632 m true vs 0.531 m estimated), like a lab calibration; never adapted in episodes.
        self.leg_scale = 1.19
        self.zone_kind = None
        self.vel_E = None  # last accepted VO velocity in E (skates)
        self.coast_s = 0.0
        self.max_coast_s = 0.6
        self.rej_reasons = {"fast": 0, "yaw": 0}
        self.vo_weight = 1.0  # fraction of the VO-minus-leg difference applied at each camera frame

    # ------------------------------------------------------------------ IMU
    def _up_b(self):
        return self.up_est

    def imu_step(self, R_wb, v_w, w_b):
        """AHRS (like the IMU's onboard fusion): the full attitude is integrated with the gyro on SO(3) and its tilt is
        pulled toward the accelerometer's gravity direction (Mahony) only while the specific force is close to 1 g.
        Heading = yaw of the integrated attitude (integrating only the vertical gyro component drifts under the
        rolling/pitching of skating pushes: coning)."""
        self.gyro, self.acc = self.imu.sample(R_wb, v_w, w_b)
        self.acc_lp = 0.9 * self.acc_lp + 0.1 * self.acc
        w = np.array(self.gyro, np.float64)
        an = float(np.linalg.norm(self.acc))
        if abs(an - GRAVITY) < 1.5:
            kp = self.att_gain if self.started else 10.0
            w = w + kp * np.cross(self.acc / an, self.R_att[2, :])
        self.R_att = self.R_att @ _so3_exp(w * self.dt)
        u, _, vt = np.linalg.svd(self.R_att)
        self.R_att = u @ vt
        if not self.started:  # heading is defined by the start frame: keep yaw at 0 until then
            self.R_att = _Rz(-math.atan2(self.R_att[1, 0], self.R_att[0, 0])) @ self.R_att
        self.up_est = self.R_att[2, :].copy()
        self.yaw_f = math.atan2(self.R_att[1, 0], self.R_att[0, 0])

    def contact_step(self, contacts_b):
        """Base height above the floor from forward kinematics: depth of the lowest contact body (foot / wheel) below
        the base in the levelled frame, plus the calibrated contact offset."""
        z = (self._R_level_b() @ np.asarray(contacts_b, np.float64).T)[2]
        depth = float(-z.min())
        if self.contact_offset is None:
            self._contact_depth = depth
            return
        self.h_cur = depth + self.contact_offset

    def leg_step(self, feet_b):
        """Leg odometry from joint encoders (forward kinematics): the lower foot is the stance foot and does not move
        in the world, so the base moved by minus the change of that foot's position (rotated into E)."""
        if not self.started:
            return
        self.has_leg_odom = True
        R = _Rz(self.yaw_f) @ self._R_level_b()
        feet_E = (R @ np.asarray(feet_b, np.float64).T).T
        stance = int(np.argmin(feet_E[:, 2]))
        if self.leg_prev is not None and self.leg_prev[0] == stance:
            d = -(feet_E[stance] - self.leg_prev[1])[:2]
            if np.linalg.norm(d) < 0.03:  # a slip / impact spike larger than 1.5 m/s is ignored
                d = d * self.leg_scale
                self.p_f = self.p_f + d
                self.leg_since_frame = self.leg_since_frame + d
        self.leg_prev = (stance, feet_E[stance])

    def tilt_deg(self):
        g = self._up_b()
        return math.degrees(math.acos(max(-1.0, min(1.0, float(g[2])))))

    def _R_level_b(self):
        return _Rz(-self.yaw_f) @ self.R_att  # body -> level frame (heading = yaw of the body x axis)

    def start(self, T_base_cam, h0, init_pose=None):
        """Start frame E from the IMU gravity (roll/pitch), yaw 0, base at height h0 above the floor. ``init_pose``
        (x, y, yaw) = an operator-provided start pose in a shared map frame (like setting the initial pose of a
        navigation stack); E is then that map frame."""
        self.T_base_cam = np.array(T_base_cam)
        self.T_cam_base = T_inv(self.T_base_cam)
        self.h0 = float(h0)
        if init_pose is not None:
            x0, y0, yaw0 = (float(v) for v in init_pose)
            self.R_att = _Rz(yaw0) @ self.R_att
            self.yaw_f = math.atan2(self.R_att[1, 0], self.R_att[0, 0])
            self.p_f = np.array([x0, y0])
        if getattr(self, "_contact_depth", None) is not None:
            self.contact_offset = self.h0 - self._contact_depth
            self.h_cur = self.h0
        self.started = True
        self.loc.start(self.T_E_base() @ self.T_base_cam)

    # ------------------------------------------------------------------ camera frames
    def update_camera(self, t_s, rgb, depth, T_base_cam, tof=None):
        self.rgb, self.depth = rgb, depth
        self.T_base_cam = np.array(T_base_cam)  # forward kinematics of the head at the current joint angles
        self.T_cam_base = T_inv(self.T_base_cam)
        if tof is None and self.tof_model is not None:
            tof = self.tof_model.measure(depth)
        self.tof = tof
        from ..memory.robot_map import classify_frame

        pos, beams = self.tof_in_E()
        kind, _, _ = classify_frame(tof[0], tof[1], pos, beams, floor_z=0.0)
        self.zone_kind = kind
        depth_img = tof_floor_depth_image(tof[0], tof[1], self.T_E_cam(), kind)
        d_flow = None
        if self.flow is not None and self.started:
            obst = tof_depth_image(tof[0], np.where(np.asarray(kind).reshape(64) == 2, tof[1], 255)) > 0
            d_flow = self.flow.step(rgb, self.T_E_cam(), obst)  # base still at its previous position here
        ok = self.loc.track(t_s, rgb, tof[0], tof[1], depth_image=depth_img)
        self.track_ok = ok
        dt_cam = (t_s - self.last_camera_t) if self.last_camera_t is not None else 1.0 / CAM_HZ
        self.last_camera_t = t_s
        if not ok:
            self.vo_lost += 1
            if d_flow is not None and np.linalg.norm(d_flow) < self.max_speed * dt_cam + 0.02:
                self.p_f = self.p_f + d_flow  # camera odometry lost: floor flow alone
                self.flow_used += 1
        else:
            T = self.loc.T_E_cam @ self.T_cam_base
            cur = (float(T[0, 3]), float(T[1, 3]), yaw_of_R(T[:3, :3]))
            if self.vo_prev is not None:
                dl = _Rz(-self.vo_prev[2])[:2, :2] @ np.array([cur[0] - self.vo_prev[0], cur[1] - self.vo_prev[1]])
                dyaw_vo = _wrap(cur[2] - self.vo_prev[2])
                dyaw_g = _wrap(self.yaw_f - self.yaw_at_prev)
                d_vo = _Rz(self.yaw_at_prev)[:2, :2] @ dl  # increment rotated into E with the gyro heading
                fast = np.linalg.norm(dl) > self.max_speed * dt_cam + 0.02
                yaw_bad = abs(_wrap(dyaw_vo - dyaw_g)) > self.max_yaw_dis
                self.rej_reasons["fast"] += int(fast)
                self.rej_reasons["yaw"] += int(yaw_bad)
                jump = fast or yaw_bad
                if self.has_leg_odom and not jump:
                    jump = np.linalg.norm(d_vo - self.leg_since_frame) > 0.06  # disagrees with leg odometry
                if d_flow is not None and np.linalg.norm(d_flow) < self.max_speed * dt_cam + 0.02:
                    self.p_f = self.p_f + d_flow  # floor flow wins on skates; cuVSLAM kept as the fallback
                    self.vel_E = d_flow / max(dt_cam, 1e-3)
                    self.coast_s = 0.0
                    self.flow_used += 1
                elif jump:
                    self.vo_rejected += 1
                    if not self.has_leg_odom and self.vel_E is not None and self.coast_s < self.max_coast_s:
                        # no leg odometry (skates): carry the last accepted velocity through a short rejected stretch,
                        # instead of freezing the position (constant-velocity prediction, as in VO front ends)
                        self.p_f = self.p_f + self.vel_E * dt_cam
                        self.coast_s += dt_cam
                elif self.has_leg_odom:
                    self.p_f = self.p_f + self.vo_weight * (d_vo - self.leg_since_frame)
                    self.vo_accepted += 1
                else:
                    self.p_f = self.p_f + d_vo
                    self.vo_accepted += 1
                    self.vel_E = d_vo / max(dt_cam, 1e-3)
                    self.coast_s = 0.0
            self.vo_prev = cur
        if self.flow is not None:
            self.flow.set_prev_pose(self.T_E_cam())
        self.yaw_at_prev = self.yaw_f
        self.leg_since_frame = np.zeros(2)
        st = self.estimate()
        if self.prev_xy is not None and t_s > self.prev_xy[0]:
            inst = math.hypot(st["x"] - self.prev_xy[1], st["y"] - self.prev_xy[2]) / (t_s - self.prev_xy[0])
            self.speed = 0.6 * self.speed + 0.4 * inst
        self.prev_xy = (t_s, st["x"], st["y"])

    # ------------------------------------------------------------------ outputs
    def T_E_base(self):
        h = self.h_cur if self.h_cur is not None else (self.h0 or 0.0)
        return T_from(_Rz(self.yaw_f) @ self._R_level_b(), np.array([self.p_f[0], self.p_f[1], h]))

    def T_E_cam(self):
        return self.T_E_base() @ self.T_base_cam

    def estimate(self):
        tilt = self.tilt_deg()
        recent = self.vo_accepted + self.vo_rejected + self.vo_lost
        status = "tracking" if self.track_ok else f"LOST ({self.vo_lost} frames so far)"
        return {"x": float(self.p_f[0]), "y": float(self.p_f[1]),
                "z": float(self.h_cur if self.h_cur is not None else (self.h0 or 0.0)), "yaw_rad": self.yaw_f,
                "tilt_deg": tilt, "fallen": bool(tilt > 60.0), "speed": float(self.speed),
                "localization": status, "features": self.loc.n_obs,
                "vo_frames": {"accepted": self.vo_accepted, "rejected_jumps": self.vo_rejected, "lost": self.vo_lost,
                              "total": recent},
                "source": ("own estimate: gyro heading + leg odometry (encoders) corrected by cuVSLAM RGBD (head RGB + ToF)"
                           if self.has_leg_odom else "own estimate: gyro heading + cuVSLAM RGBD (head RGB + ToF) translation")}

    def tof_in_E(self):
        """ToF sensor position and zone beam directions in the robot's own frame E (from the pose estimate)."""
        T = self.T_E_cam() @ self.T_cam_tof
        return T[:3, 3].copy(), (T[:3, :3] @ self.tof_rays_sensor.T).T


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "serve":
        _serve(int(sys.argv[2]))


if __name__ == "__main__":
    main()
