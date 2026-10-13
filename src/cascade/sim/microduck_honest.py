# SPDX-License-Identifier: Apache-2.0
"""Honest robot/agent loop for the MicroDuck demos: the agent side only ever sees the robot's own sensors and its own
pose estimate.

* :class:`OnboardRig` ticks every robot's onboard stack after each control step: the trunk IMU at the control rate;
  at 15 Hz the OVRTX head cameras of ALL robots (one render of the live Newton body poses, so robots see each other),
  the ToF frame, and cuVSLAM.
* :class:`HonestWorld` is what the WorldPort, the agent runtime and its tools are given. It is a closed proxy (no
  attribute forwarding): ``base_state`` = the robot's own estimate (start frame E), ``tof_frame`` = its own ToF frame
  with beams placed by that estimate, ``camera_rgb`` = its own camera, ``set_control`` = its own command interface.
  Simulator truth stays inside the world object and the offline verifiers.
* :class:`OnboardDetector`: YOLOE (Cascade's open-vocabulary detector, prompt-free) on the robot's own 640x360 camera
  image; range from its own ToF zone when the object is inside the ToF field of view, otherwise from the ground-plane
  intersection of the box bottom with the estimated camera pose.
"""
from __future__ import annotations

import math
import subprocess
import threading

import numpy as np

from .microduck_onboard import (CAM_FX, CAM_H, CAM_HZ, CAM_W, TOF_HALF_FOV, Onboard, OvrtxSensorClient, T_from, T_inv,
                                optical_from_lens, quat_to_R)

LENS_OFFSET = np.array([0.0155, -0.000091378, -0.0823])
LENS_ROT = quat_to_R((0.0, 2 ** -0.5, 0.0, 2 ** -0.5))


class WalkerAdapter:
    """Reads the live state of :class:`cascade.sim.microduck_lab_world.MicroduckLabWorld` for SENSOR SIMULATION."""

    tof_from_depth = False

    def __init__(self, world):
        self.w = world
        self.trunk = world.body_names.index("trunk_base")
        self.head = world.body_names.index("jaw_soft")
        self.origins = world.base.scene.env_origins.detach().cpu().numpy().astype(np.float64)
        self.body_names = world.body_names
        self.contacts = [world.body_names.index("ankle_left"), world.body_names.index("ankle_right")]

    leg_odometry = True

    def contacts_b(self, bp, i):
        """Contact body (feet / wheels) positions in the trunk frame: forward kinematics of the joint encoders."""
        Tt = self.trunk_T(bp, i)
        Ri, p = Tt[:3, :3].T, Tt[:3, 3]
        return np.stack([Ri @ (bp[i, f, :3] - p) for f in self.contacts])

    def body_pose(self):
        bp = self.w.robot.data.body_link_pose_w.torch.detach().cpu().numpy().astype(np.float64)
        bp[..., :3] -= self.origins[:, None, :]
        return bp

    def lens_T(self, bp, i):
        h = bp[i, self.head]
        Rh = quat_to_R(h[3:7])
        return T_from(optical_from_lens(Rh @ LENS_ROT), h[:3] + Rh @ LENS_OFFSET)

    def trunk_T(self, bp, i):
        t = bp[i, self.trunk]
        return T_from(quat_to_R(t[3:7]), t[:3])

    def trunk_motion(self, bp, i):
        d = self.w.robot.data
        return (quat_to_R(bp[i, self.trunk, 3:7]), d.root_lin_vel_w.torch[i].detach().cpu().numpy().astype(np.float64),
                d.root_ang_vel_b.torch[i].detach().cpu().numpy().astype(np.float64))

    def tof(self, i):
        m = self.w.tof_model
        return (m.held_mm[i].detach().cpu().numpy().astype(np.int32).ravel(),
                m.held_status[i].detach().cpu().numpy().astype(np.int32).ravel())

    def mode(self, i):
        return self.w.controls[i].mode

    def control_state(self, i):
        return {"mode": self.w.controls[i].mode}


class SkateAdapter:
    """Live state of :class:`cascade.sim.microduck_skate_world.MicroduckSkateWorld` for SENSOR SIMULATION. No leg
    odometry (the feet roll on passive wheels); ToF from the rendered depth."""

    tof_from_depth = True

    def __init__(self, world):
        self.w = world
        self.trunk = world.body_names.index("trunk_base")
        self.head = world.body_names.index("jaw_soft")
        self.origin = np.asarray(world.origin, np.float64)
        self.body_names = world.body_names
        self.contacts = [k for k, n in enumerate(world.body_names) if n.startswith("tire")]

    leg_odometry = False
    contacts_b = WalkerAdapter.contacts_b

    def body_pose(self):
        bp = np.stack([r.data.body_link_pose_w.torch[0].detach().cpu().numpy() for r in self.w.robots]).astype(np.float64)
        bp[..., :3] -= self.origin
        return bp

    lens_T = WalkerAdapter.lens_T
    trunk_T = WalkerAdapter.trunk_T

    def trunk_motion(self, bp, i):
        d = self.w.robots[i].data
        return (quat_to_R(bp[i, self.trunk, 3:7]), d.root_lin_vel_w.torch[0].detach().cpu().numpy().astype(np.float64),
                d.root_ang_vel_b.torch[0].detach().cpu().numpy().astype(np.float64))

    def tof(self, i):
        return None

    def mode(self, i):
        return self.w.mode[i]

    def control_state(self, i):
        return {"mode": self.w.mode[i], "throttle": self.w.throttle[i], "target_yaw_rad": self.w.target_yaw[i]}


class OnboardRig:
    def __init__(self, adapter, names, *, course_usd, robot_usd, dt, light_scale=1.0, log_path=None, video_dir=None):
        self.a = adapter
        self.names = list(names)
        self.dt = float(dt)
        self.onboard = [Onboard(n, dt, seed=17 + i, tof_from_depth=adapter.tof_from_depth) for i, n in enumerate(names)]
        self.sensors = OvrtxSensorClient(course_usd=course_usd, robot_usd=robot_usd, robots=names,
                                         body_names=adapter.body_names, light_scale=light_scale, log_path=log_path)
        self.started = False
        self.vo_weight = 0.3
        self.next_cam_t = 0.0
        self.lock = threading.RLock()
        self.cam_frames = 0
        self.video = {}
        if video_dir is not None:
            for n in names:
                self.video[n] = subprocess.Popen(
                    ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{CAM_W}x{CAM_H}",
                     "-r", str(int(CAM_HZ)), "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                     str(video_dir / f"headcam_{n}.mp4")], stdin=subprocess.PIPE)
        self.cam_times = []
        self.est_log = {n: [] for n in names}  # (t, x, y, yaw, tracking, features)
        self.tof_log = {n: [] for n in names}  # (64 mm, 64 status) per camera frame

    def imu_tick(self):
        bp = self.a.body_pose()
        for i, ob in enumerate(self.onboard):
            ob.imu_step(*self.a.trunk_motion(bp, i))
            ob.contact_step(self.a.contacts_b(bp, i))
            if getattr(self.a, "leg_odometry", False):
                ob.vo_weight = self.vo_weight
                ob.leg_step(self.a.contacts_b(bp, i))
        return bp

    def start(self, t_s, floor_z_world):
        """Start frame E for every robot: IMU gravity + standing trunk height (calibration constant)."""
        bp = self.a.body_pose()
        self.calibration = {}
        for i, ob in enumerate(self.onboard):
            T_base_cam = T_inv(self.a.trunk_T(bp, i)) @ self.a.lens_T(bp, i)  # forward kinematics of the head
            h0 = float(bp[i, self.a.trunk, 2] - floor_z_world[i])
            ob.start(T_base_cam, h0)
            self.calibration[self.names[i]] = {"standing_trunk_height_m": round(h0, 4)}
        self.started = True
        self.next_cam_t = t_s
        self.after_step(t_s, force_camera=True)

    def after_step(self, t_s, force_camera=False):
        with self.lock:
            bp = self.imu_tick()
            if not self.started or (t_s + 1e-9 < self.next_cam_t and not force_camera):
                return
            self.next_cam_t += 1.0 / CAM_HZ
            cams = [self.a.lens_T(bp, i) for i in range(len(self.names))]
            out = self.sensors.render(t_s, bp, cams)
            self.cam_frames += 1
            self.cam_times.append(t_s)
            for i, ob in enumerate(self.onboard):
                rgb, depth = out[i]
                T_base_cam = T_inv(self.a.trunk_T(bp, i)) @ cams[i]
                tof = None if self.a.tof_from_depth else self.a.tof(i)
                ob.update_camera(t_s, rgb, depth, T_base_cam, tof)
                est = ob.estimate()
                self.est_log[self.names[i]].append((t_s, est["x"], est["y"], est["yaw_rad"], bool(ob.track_ok),
                                                    ob.loc.n_obs))
                self.tof_log[self.names[i]].append(np.concatenate([np.asarray(ob.tof[0], np.int32).ravel(),
                                                                   np.asarray(ob.tof[1], np.int32).ravel()]))
                if self.names[i] in self.video:
                    try:
                        self.video[self.names[i]].stdin.write(np.ascontiguousarray(rgb).tobytes())
                    except Exception:  # noqa: BLE001
                        pass

    def close(self):
        for p in self.video.values():
            try:
                p.stdin.close()
                p.wait(30)
            except Exception:  # noqa: BLE001
                pass
        self.sensors.close()


class HonestWorld:
    """Closed proxy given to the WorldPort and the agent runtimes. Exposes only onboard information."""

    __slots__ = ("_w", "_rig", "names", "_ctrl")

    def __init__(self, world, rig: OnboardRig):
        object.__setattr__(self, "_w", world)
        object.__setattr__(self, "_rig", rig)
        object.__setattr__(self, "names", list(world.names))
        object.__setattr__(self, "_ctrl", rig.a.control_state)

    @property
    def time_s(self):
        return self._w.time_s

    @property
    def step_index(self):
        return self._w.step_index

    def step(self, n=1):
        for _ in range(int(n)):
            self._w.step(1)
            self._rig.after_step(self._w.time_s)

    def set_control(self, i, mode, command=(0.0, 0.0, 0.0), goal_xy=None):
        self._w.set_control(i, mode, command, None)

    def base_state(self, i):
        est = self._rig.onboard[i].estimate()
        est.update(self._ctrl(i))  # the robot's own controller state (what it commanded)
        est["sim_time_s"] = self._w.time_s
        return est

    def tof_frame(self, i):
        ob = self._rig.onboard[i]
        mm, st = ob.tof
        pos, beams = ob.tof_in_E()
        return {"distance_mm": np.asarray(mm).reshape(8, 8), "status": np.asarray(st).reshape(8, 8),
                "sensor_pos_w": pos, "beam_dir_w": beams.reshape(8, 8, 3), "sim_time_s": self._w.time_s,
                "frame": "robot start frame E (own pose estimate)"}

    def policy_rgb(self, i):
        return self._w.policy_rgb(i)

    def camera_rgb(self, i):
        return self._rig.onboard[i].rgb

    def onboard(self, i):
        return self._rig.onboard[i]


SCENE_LABELS = ("gym", "sky", "evening", "floor", "room", "rink", "playroom", "hall", "stage", "court", "arena",
                "interior", "studio", "wall", "ceiling", "ground", "field")


class OnboardDetector:
    """YOLOE prompt-free (Cascade's detector model) on the robot's own camera frame. ``detect`` keeps the walker
    runtime's call signature but ignores the pose arguments: everything comes from the onboard stack."""

    SOURCE = "YOLOE-11s prompt-free on the head camera image (640x360); range from own ToF or ground plane"

    def __init__(self, world: HonestWorld, index: int, model, *, conf=0.2):
        self.world, self.index, self.model, self.conf = world, index, model, conf
        self.max_range_m = 4.0
        self.last_vis = None

    def detect(self, x=None, y=None, yaw=None):
        from ..perception.colors import mask_color

        ob = self.world.onboard(self.index)
        rgb = ob.rgb
        if rgb is None:
            return {"source": self.SOURCE, "detections": [], "note": "no camera frame yet"}
        T_E_cam = ob.T_E_cam()
        tof_mm, tof_st = ob.tof
        r = self.model.predict(rgb[..., ::-1], conf=self.conf, verbose=False)[0]
        dets = []
        base = ob.estimate()
        for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
            label = r.names[int(c)]
            x0, y0, x1, y1 = b
            area = (x1 - x0) * (y1 - y0) / (CAM_W * CAM_H)
            if area > 0.35 or any(k in label for k in SCENE_LABELS):
                continue
            box = np.zeros(rgb.shape[:2], bool)
            box[int(y0):int(y1), int(x0):int(x1)] = True
            color = mask_color(rgb[..., ::-1], box) or "unknown"
            u, v = (x0 + x1) / 2, (y0 + y1) / 2
            ray_c = np.array([(u - (CAM_W - 1) / 2) / CAM_FX, (v - (CAM_H - 1) / 2) / CAM_FX, 1.0])
            rng, src = None, None
            az, el = math.atan(-ray_c[0]), math.atan(-ray_c[1])
            if abs(az) < TOF_HALF_FOV and abs(el) < TOF_HALF_FOV:
                cell = 2 * TOF_HALF_FOV / 8
                col, row = int((TOF_HALF_FOV - az) / cell), int((TOF_HALF_FOV - el) / cell)
                z = row * 8 + col
                if int(tof_st[z]) in (5, 9) and int(tof_mm[z]) >= 100:
                    rng, src = tof_mm[z] / 1000.0, "own ToF zone"
            ray_b = np.array([(u - (CAM_W - 1) / 2) / CAM_FX, (y1 - (CAM_H - 1) / 2) / CAM_FX, 1.0])
            d_E = T_E_cam[:3, :3] @ (ray_b / np.linalg.norm(ray_b))
            o_E = T_E_cam[:3, 3]
            if rng is None and d_E[2] < -0.02:
                tg = -o_E[2] / d_E[2]
                if tg < 6.0:
                    p = o_E + tg * d_E
                    rng, src = float(np.linalg.norm(p - o_E)), "ground plane (box bottom)"
            if rng is None:
                pos, dist, bearing = None, None, None
            else:
                dc = T_E_cam[:3, :3] @ (ray_c / np.linalg.norm(ray_c))
                p = o_E + rng * dc
                pos = [round(float(p[0]), 2), round(float(p[1]), 2)]
                dist = round(math.hypot(p[0] - base["x"], p[1] - base["y"]), 2)
                bearing = round(math.degrees(math.atan2(math.sin(math.atan2(p[1] - base["y"], p[0] - base["x"]) - base["yaw_rad"]),
                                                        math.cos(math.atan2(p[1] - base["y"], p[0] - base["x"]) - base["yaw_rad"]))), 1)
            dets.append({"label": f"{color} {label}", "confidence": round(float(s), 2), "bbox_px": [int(v) for v in b],
                         "bearing_deg": bearing, "distance_m": dist, "world_xy": pos, "range_source": src})
        dets.sort(key=lambda d: (d["distance_m"] is None, d["distance_m"] or 0))
        self.last_vis = {"t": self.world.time_s, "dets": dets}
        return {"source": self.SOURCE, "frame": "positions in your own start frame (from your pose estimate)",
                "detections": dets[:10]}


def load_yoloe(repo_root):
    import os

    os.environ.setdefault("YOLO_OFFLINE", "1")
    from ultralytics import YOLO

    return YOLO(str(repo_root / "models/yoloe-11s-seg-pf.pt"))


__all__ = ["SkateAdapter", "WalkerAdapter", "OnboardRig", "HonestWorld", "OnboardDetector", "load_yoloe"]
