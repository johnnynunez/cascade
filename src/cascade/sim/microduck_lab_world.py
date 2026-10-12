# SPDX-License-Identifier: Apache-2.0
"""Isaac Lab 3.0 + Newton world for agentic MicroDuck demos (optional; Isaac Lab venv only).

One environment per robot, all sharing one scene origin and one course USD. Each robot's controller is selected per
control step by the agent layer: the learned walker (velocity commands), the onboard ToF+RGB perception actor, the
learned get-up policy, or the learned depth navigation policy driving the walker. Nothing here invents a gait: every
joint target comes from one of the unchanged policies listed in ``POLICY_FILES``.

Honesty boundaries, kept in the recording metadata:
* every environment is its own Newton world (Isaac Lab cloning). Robots therefore never collide with each other and
  never appear in each other's onboard sensors; the shared course and ground are copies in every world;
* poses written into the recording are completed physics states, sampled once per 50 Hz control step;
* the 48x36 head RGB and the ToF 8x8 frames recorded are exactly the tensors the perception actor would receive.

Import is cheap: Isaac Lab, torch and the gtceu_microduck package are imported inside :func:`open_world`, which must be
called inside ``isaaclab.app.launch_simulation``.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

CONTROL_DT_S = 0.02
WALK_LIMITS = {"vx": (-0.4, 0.4), "vy": (-0.3, 0.3), "wz": (-1.0, 1.0)}
MODES = ("stand", "walk", "onboard_hero", "getup", "depth_nav")
LENS_OFFSET = (0.0155, -0.000091378, -0.0823)
LENS_ROTATION_XYZW = (0.0, 2**-0.5, 0.0, 2**-0.5)
CANONICAL_BODIES = 16


@dataclass(frozen=True)
class RobotSpawn:
    name: str
    x: float
    y: float
    yaw_deg: float = 0.0
    fallen: str | None = None  # None | "back" | "front" | "side"
    z: float = 0.0  # support height under the spawn point (platforms, tables)


@dataclass
class PolicyFiles:
    walker_onnx: Path
    hero_pt: Path
    getup_onnx: Path | None = None
    nav_onnx: Path | None = None

    def digests(self) -> dict:
        out = {}
        for key, path in vars(self).items():
            if path is not None:
                out[key] = {"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        return out


@dataclass
class _Control:
    mode: str = "stand"
    command: tuple = (0.0, 0.0, 0.0)
    goal_xy: tuple | None = None


def yaw_quat_xyzw(yaw_rad: float, roll_rad: float = 0.0, pitch_rad: float = 0.0) -> tuple:
    cy, sy = math.cos(yaw_rad / 2), math.sin(yaw_rad / 2)
    cp, sp = math.cos(pitch_rad / 2), math.sin(pitch_rad / 2)
    cr, sr = math.cos(roll_rad / 2), math.sin(roll_rad / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


def quat_xyzw_yaw(q) -> float:
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class MicroduckLabWorld:
    """Live Isaac Lab world; construct with :func:`open_world` inside ``launch_simulation``."""

    def __init__(self, env, robot_names, policies: PolicyFiles, actor_hero, walker, getup, nav, torch_mod, tof_mod,
                 meta: dict):
        self.env = env
        self.base = env.unwrapped
        self.names = list(robot_names)
        self.n = len(self.names)
        self.policies = policies
        self._torch = torch_mod
        self._tof = tof_mod
        self.hero, self.walker, self.getup, self.nav = actor_hero, walker, getup, nav
        self.robot = self.base.scene["robot"]
        self.sensor = self.base.scene["head_tof"]
        self.tof_term = self.base.observation_manager._group_obs_term_cfgs["tof"][0].func
        self.tof_model = self.tof_term.model
        self.body_names = list(self.robot.body_names)
        self.joint_names = list(self.robot.joint_names)
        self.jaw = self.body_names.index("jaw_soft")
        self.controls = [_Control() for _ in range(self.n)]
        self.step_index = 0
        self.obs = env.get_observations()
        self.meta = dict(meta)
        self.rec = {k: [] for k in ("time_s", "body_pose", "root_lin_vel_b", "proj_grav", "mode", "command",
                                    "tof_mm", "tof_status", "rgb", "lens_pose", "action")}
        self.nav_state = {}
        self._nav_prev = np.zeros((self.n, 3), np.float32)
        self._nav_cmd = np.zeros((self.n, 3), np.float32)

    # ---------------------------------------------------------------- reading
    @property
    def time_s(self) -> float:
        return self.step_index * CONTROL_DT_S

    def robot_index(self, name: str) -> int:
        return self.names.index(name)

    def base_state(self, i: int) -> dict:
        d = self.robot.data
        pos = d.root_link_pos_w.torch[i].cpu().numpy().astype(float)
        quat = d.root_link_quat_w.torch[i].cpu().numpy().astype(float)  # xyzw
        grav = d.projected_gravity_b.torch[i].cpu().numpy().astype(float)
        vel_b = d.root_lin_vel_b.torch[i].cpu().numpy().astype(float)
        wz = float(d.root_ang_vel_b.torch[i, 2])
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -grav[2]))))
        return {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]), "yaw_rad": quat_xyzw_yaw(quat),
                "tilt_deg": tilt, "fallen": bool(tilt > 55.0 or pos[2] < 0.06), "vx_body": float(vel_b[0]),
                "vy_body": float(vel_b[1]), "wz": wz, "mode": self.controls[i].mode, "sim_time_s": self.time_s}

    def tof_frame(self, i: int) -> dict:
        """Latest hardware-model ToF frame of robot ``i`` (what the perception actor holds), world-frame beams."""
        mm = self.tof_model.held_mm[i].cpu().numpy().astype(np.int32).reshape(8, 8)
        st = self.tof_model.held_status[i].cpu().numpy().astype(np.int32).reshape(8, 8)
        start = self.sensor.ray_starts_w.torch[i, 0].cpu().numpy().astype(float)
        dirs = self.sensor.ray_directions_w.torch[i, :64].cpu().numpy().astype(float).reshape(8, 8, 3)
        return {"distance_mm": mm, "status": st, "sensor_pos_w": start, "beam_dir_w": dirs, "sim_time_s": self.time_s}

    def policy_rgb(self, i: int) -> np.ndarray:
        """(36, 48, 3) uint8: the held 15 Hz head RGB frame, exactly the perception actor's input."""
        x = self.obs["rgb"][i].permute(1, 2, 0).cpu().numpy()
        return (np.clip(x, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    # ---------------------------------------------------------------- control
    def set_control(self, i: int, mode: str, command=(0.0, 0.0, 0.0), goal_xy=None) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        if mode == "getup" and self.getup is None:
            raise ValueError("no get-up policy loaded")
        if mode == "depth_nav" and self.nav is None:
            raise ValueError("no navigation policy loaded")
        vx, vy, wz = (float(c) for c in command)
        vx = min(max(vx, WALK_LIMITS["vx"][0]), WALK_LIMITS["vx"][1])
        vy = min(max(vy, WALK_LIMITS["vy"][0]), WALK_LIMITS["vy"][1])
        wz = min(max(wz, WALK_LIMITS["wz"][0]), WALK_LIMITS["wz"][1])
        c = self.controls[i]
        if mode == "depth_nav" and c.mode != "depth_nav":
            self._nav_prev[i] = 0.0
            self._nav_cmd[i] = 0.0
        c.mode, c.command, c.goal_xy = mode, (vx, vy, wz), goal_xy

    def _nav_update(self):
        """10 Hz learned navigation: goal + depth -> slew-limited walker command (the A15/A16 contract)."""
        torch = self._torch
        idx = [i for i, c in enumerate(self.controls) if c.mode == "depth_nav"]
        if not idx or self.step_index % 5:
            return
        d = self.robot.data
        depth = self.base.scene["nav_camera"].data.output["distance_to_image_plane"].torch[..., 0]
        depth = torch.nan_to_num(depth, nan=3.0, posinf=3.0, neginf=3.0).clamp(0.0, 3.0) / 3.0
        depth = torch.where(depth <= 0.0, torch.ones_like(depth), depth)
        for i in idx:
            gx, gy = self.controls[i].goal_xy
            pos = d.root_link_pos_w.torch[i].cpu().numpy()
            yaw = quat_xyzw_yaw(d.root_link_quat_w.torch[i].cpu().numpy())
            dx, dy = gx - pos[0], gy - pos[1]
            dist = math.hypot(dx, dy)
            bearing = math.atan2(dy, dx) - yaw
            grav = d.projected_gravity_b.torch[i].cpu().numpy()
            angv = d.root_ang_vel_b.torch[i].cpu().numpy()
            pol = np.concatenate([[min(dist, 5.0), math.cos(bearing), math.sin(bearing)], self._nav_prev[i],
                                  self._nav_cmd[i], grav, angv]).astype(np.float32)[None]
            img = depth[i].cpu().numpy().astype(np.float32)[None, None]
            act = np.clip(self.nav.run(None, {"policy": pol, "depth": img})[0][0], -1.0, 1.0)
            self._nav_prev[i] = act
            self.nav_state[i] = {"goal_distance_m": dist, "bearing_rad": bearing}

    def _nav_slew(self):
        scales = np.array([0.4, 0.3, 1.0], np.float32)
        slew = np.array([1.0, 1.0, 4.0], np.float32) * CONTROL_DT_S
        for i, c in enumerate(self.controls):
            if c.mode == "depth_nav":
                target = self._nav_prev[i] * scales
                self._nav_cmd[i] = self._nav_cmd[i] + np.clip(target - self._nav_cmd[i], -slew, slew)

    @staticmethod
    def _with_command(policy_obs, cmd):
        x = policy_obs.clone()
        x[:, 48:61] = 0.0
        x[:, 48:51] = cmd
        return x

    def step(self, steps: int = 1) -> None:
        torch = self._torch
        for _ in range(int(steps)):
            self._nav_update()
            self._nav_slew()
            obs = self.obs
            modes = [c.mode for c in self.controls]
            cmd = torch.zeros(self.n, 3, device=self.base.device)
            for i, c in enumerate(self.controls):
                if c.mode == "walk":
                    cmd[i] = torch.tensor(c.command, device=self.base.device)
                elif c.mode == "depth_nav":
                    cmd[i] = torch.from_numpy(self._nav_cmd[i]).to(self.base.device)
            with torch.no_grad():
                action = self.walker(self._with_command(obs["policy"], cmd))
                if "onboard_hero" in modes:
                    hero = self.hero(obs)
                    sel = torch.tensor([m == "onboard_hero" for m in modes], device=action.device)
                    action = torch.where(sel[:, None], hero, action)
                if "getup" in modes:
                    up = self.getup(self._with_command(obs["policy"], torch.zeros_like(cmd)))
                    sel = torch.tensor([m == "getup" for m in modes], device=action.device)
                    action = torch.where(sel[:, None], up, action)
            self._record(action, cmd)
            self.obs, _, dones, _ = self.env.step(action)
            if bool(dones.any()):
                raise RuntimeError(f"unexpected environment reset at step {self.step_index}")
            self.step_index += 1

    def _record(self, action, cmd):
        torch = self._torch
        d = self.robot.data
        head = d.body_link_pose_w.torch[:, self.jaw]
        from isaaclab.utils.math import combine_frame_transforms

        lp, lq = combine_frame_transforms(head[:, :3], head[:, 3:], head.new_tensor(LENS_OFFSET).expand(self.n, -1),
                                          head.new_tensor(LENS_ROTATION_XYZW).expand(self.n, -1))
        r = self.rec
        r["time_s"].append(self.time_s)
        r["body_pose"].append(d.body_link_pose_w.torch.cpu().numpy().astype(np.float32))
        r["root_lin_vel_b"].append(d.root_lin_vel_b.torch.cpu().numpy().astype(np.float32))
        r["proj_grav"].append(d.projected_gravity_b.torch.cpu().numpy().astype(np.float32))
        r["mode"].append([MODES.index(c.mode) for c in self.controls])
        r["command"].append(cmd.cpu().numpy().astype(np.float32))
        r["tof_mm"].append(self.tof_model.held_mm.cpu().numpy().astype(np.int16))
        r["tof_status"].append(self.tof_model.held_status.cpu().numpy().astype(np.uint8))
        r["rgb"].append((self.obs["rgb"].permute(0, 2, 3, 1).cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8))
        r["lens_pose"].append(torch.cat([lp, lq], dim=-1).cpu().numpy().astype(np.float32))
        r["action"].append(action.cpu().numpy().astype(np.float32))

    def save(self, path: Path, extra: dict | None = None) -> Path:
        path = Path(path)
        arrays = {k: np.asarray(v) for k, v in self.rec.items()}
        np.savez_compressed(path, body_names=np.asarray(self.body_names), joint_names=np.asarray(self.joint_names),
                            robot_names=np.asarray(self.names), modes=np.asarray(MODES), **arrays)
        meta = {**self.meta, **(extra or {}), "frames": len(self.rec["time_s"]), "control_dt_s": CONTROL_DT_S}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
        return path


def build_env_cfg(*, gtceu_root: Path, course_usd: Path | None, num_robots: int, device: str, seed: int,
                  nav_camera: bool, extra_floor: bool = True):
    gtceu_root = Path(gtceu_root).resolve()
    if str(gtceu_root) not in sys.path:
        sys.path.insert(0, str(gtceu_root))
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    import gtceu_microduck  # noqa: F401  (registers MicroDuck-GTCEU-Walk)
    from gtceu_microduck.sensors.onboard_env import onboard_env

    cfg = onboard_env([Path(course_usd).resolve()] if course_usd else [], num_robots, rgb_randomize=False)
    if course_usd is None:
        cfg.scene.course = None
    cfg.sim.physics.solver_cfg.nconmax = 256
    cfg.scene.robot.spawn.activate_contact_sensors = True
    cfg.scene.env_spacing = 0.0
    for name in list(vars(cfg.terminations)):
        if not name.startswith("_"):
            setattr(cfg.terminations, name, None)
    if hasattr(cfg.rewards, "failure"):
        cfg.rewards.failure = None
    cfg.episode_length_s = 1.0e6
    cfg.sim.device = device
    cfg.seed = seed
    if nav_camera:
        cfg.scene.nav_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Sensors/NavCamera",
            spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, horizontal_aperture=20.955, clipping_range=(0.02, 3.0)),
            width=64, height=48, update_period=0.1, data_types=["distance_to_image_plane"],
            renderer_cfg=NewtonWarpRendererCfg(depth_clipping_behavior="max"),
        )
    return cfg


def _follow_nav_camera(base, jaw):
    from isaaclab.utils.math import combine_frame_transforms

    head = base.scene["robot"].data.body_link_pose_w.torch[:, jaw]
    n = head.shape[0]
    pos, quat = combine_frame_transforms(head[:, :3], head[:, 3:], head.new_tensor(LENS_OFFSET).expand(n, -1),
                                         head.new_tensor(LENS_ROTATION_XYZW).expand(n, -1))
    base.scene["nav_camera"].set_world_poses(pos, quat, convention="world")


def create_world(cfg, *, gtceu_root: Path, spawns: list[RobotSpawn], policies: PolicyFiles, device: str,
                 meta: dict | None = None, course_usd: Path | None = None) -> MicroduckLabWorld:
    import gymnasium as gym
    import torch
    from rsl_rl.runners import DistillationRunner

    from isaaclab.utils import to_dict
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from gtceu_microduck.policy import WalkingPolicy
    from gtceu_microduck.sensors import tof_vl53l8cx as tof
    from gtceu_microduck.sensors.onboard_env import distillation_runner_cfg, prepare_models

    env = RslRlVecEnvWrapper(gym.make("MicroDuck-GTCEU-Walk", cfg=cfg))
    runner = DistillationRunner(env, to_dict(distillation_runner_cfg("hero", device, 1)), log_dir=None, device=device)
    prepare_models(runner.alg, "hero", device)
    ck = torch.load(policies.hero_pt, map_location=device, weights_only=False)
    runner.alg._raw_student.load_state_dict(ck.get("student_state_dict", ck.get("actor_state_dict")), strict=True)
    hero = runner.alg._raw_student.eval()
    walker = WalkingPolicy(str(policies.walker_onnx)).to(device).eval().requires_grad_(False)
    getup = None
    if policies.getup_onnx is not None:
        getup = WalkingPolicy(str(policies.getup_onnx)).to(device).eval().requires_grad_(False)
    nav = None
    if policies.nav_onnx is not None:
        import onnxruntime as ort

        nav = ort.InferenceSession(str(policies.nav_onnx), providers=["CPUExecutionProvider"])
    world = MicroduckLabWorld(env, [s.name for s in spawns], policies, hero, walker, getup, nav, torch, tof,
                              {**(meta or {}), "policies": policies.digests(), "tof_sensor": tof.LABEL,
                               "spawns": [vars(s) for s in spawns], "hero_iteration": int(ck.get("iter", -1))})
    if "nav_camera" in world.base.scene.keys():
        jaw = world.jaw
        orig_step = env.step

        def step_with_nav_camera(action):
            _follow_nav_camera(world.base, jaw)
            return orig_step(action)

        env.step = step_with_nav_camera
    place_robots(world, spawns)
    return world


def place_robots(world: MicroduckLabWorld, spawns: list[RobotSpawn]) -> None:
    torch = world._torch
    robot = world.robot
    dev = world.base.device
    n = world.n
    default = robot.data.default_root_pose.torch.clone() if hasattr(robot.data, "default_root_pose") else None
    z0 = float(default[0, 2]) if default is not None else float(robot.data.root_link_pos_w.torch[0, 2])
    pos = torch.zeros(n, 3, device=dev)
    quat = torch.zeros(n, 4, device=dev)
    for i, s in enumerate(spawns):
        roll = pitch = 0.0
        z = z0
        if s.fallen == "back":
            pitch, z = -math.pi / 2, 0.09
        elif s.fallen == "front":
            pitch, z = math.pi / 2, 0.09
        elif s.fallen == "side":
            roll, z = math.pi / 2, 0.09
        pos[i] = torch.tensor((s.x, s.y, z + s.z), device=dev)
        quat[i] = torch.tensor(yaw_quat_xyzw(math.radians(s.yaw_deg), roll, pitch), device=dev)
    ids = torch.arange(n, device=dev)
    robot.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=-1), env_ids=ids)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=ids)
    q = robot.data.default_joint_pos.torch.clone()
    robot.write_joint_state_to_sim_index(position=q, velocity=torch.zeros_like(q), env_ids=ids)
    world.base.scene.write_data_to_sim()
    world.meta["spawn_height_m"] = z0


__all__ = ["MicroduckLabWorld", "PolicyFiles", "RobotSpawn", "build_env_cfg", "create_world", "CONTROL_DT_S",
           "MODES", "time"]
