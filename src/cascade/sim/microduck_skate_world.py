# SPDX-License-Identifier: Apache-2.0
"""Isaac Lab 3.0 + Newton world of MicroDucks on roller skates, driven only through the skating policies' command channel.

All robots of an episode are articulations of the skate-wheel robot variant (``microduck_rollers.usd``: the MicroDuck
with four passive wheels under the feet; it differs from the canonical walking ``MicroDuck.usd`` by the skates) in ONE
Isaac Lab environment, i.e. ONE shared Newton world: skaters can collide with each other and with the course.

The scene recipe follows the validated multi-racer setup of microduck_gtceu ``scripts/skills/skating_race.py``
(``Pollen-MicroDuck-Roller-Eval`` task, one 61-D observation group and one 14-D joint-position action term per robot,
domain randomization terms bound per robot with identity parameters, passive wheel friction 0.0015 N m). It is
rebuilt here, not copied, and reads the task module from a read-only export of that repository.

Per 50 Hz control step every robot runs an unchanged skating actor (ONNX, 61 -> 14) whose ONLY command input is the
official ``twist = [throttle, 0, heading_error]`` (throttle: 0 coast, + push, - brake; heading error clamped). The agent
chooses throttle and a WORLD heading target; this module turns the target into the heading error from the simulated
trunk yaw each step (the same quantity the policies were trained on). Nothing writes poses, velocities or joint
targets; there is no get-up policy for the skate robot, so a fall ends that robot's run.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

CONTROL_DT_S = 0.02
OBS_DIM, ACT_DIM = 61, 14
CMD = slice(48, 51)
SKATE_MODES = ("coast", "skate", "brake")
HEADING_MAX_RAD = 0.35  # carve range of the natural roller's training (+-0.35 rad); race pilot used 0.3
WHEEL_FRICTION_NM = 0.0015
ROLLER_BODIES = 19


@dataclass
class SkatePolicy:
    name: str
    onnx: Path

    def digest(self) -> str:
        return hashlib.sha256(Path(self.onnx).read_bytes()).hexdigest()


@dataclass
class SkateSpawn:
    name: str
    x: float
    y: float
    yaw_deg: float = 0.0


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _yaw(q):
    x, y, z, w = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class MicroduckSkateWorld:
    """Surface used by :class:`cascade.sim.microduck_world_port.WorldPort` (names, time_s, step_index, step,
    set_control, base_state). Control modes: ``skate`` (throttle > 0), ``brake`` (throttle < 0), ``coast`` (0)."""

    def __init__(self, env, names, actor_session, policy: SkatePolicy, torch, meta: dict):
        self.env, self.names, self.n = env, list(names), len(names)
        self._torch = torch
        self.actor = actor_session
        self.policy = policy
        self.robots = [env.scene[f"racer_{i}"] for i in range(self.n)]
        self.body_names = list(self.robots[0].body_names)
        if len(self.body_names) != ROLLER_BODIES:
            raise RuntimeError(f"skate robot has {len(self.body_names)} bodies, expected {ROLLER_BODIES}")
        self.joint_names = list(self.robots[0].joint_names)
        self.trunk = self.body_names.index("trunk_base")
        self.step_index = 0
        self.meta = dict(meta)
        self.throttle = [0.0] * self.n
        self.target_yaw = [0.0] * self.n
        self.mode = ["coast"] * self.n
        self.last_cmd = np.zeros((self.n, 3), np.float32)
        self.origin = env.scene.env_origins[0].detach().cpu().numpy()
        self.rec = {k: [] for k in ("time_s", "body_pose", "root_lin_vel_b", "proj_grav", "mode", "command",
                                    "target_yaw", "actor_obs", "action")}
        self.obs = None
        # Honest mode: callable(i) -> the robot's OWN heading estimate (start frame). The heading error fed to the
        # policy is then computed from the estimate (as on hardware), never from simulator truth.
        self.heading_source = None

    # ------------------------------------------------------------------ state
    @property
    def time_s(self) -> float:
        return self.step_index * CONTROL_DT_S

    def _pose(self, i):
        p = self.robots[i].data.body_link_pose_w.torch[0, self.trunk].detach().cpu().numpy().astype(np.float64)
        p[:3] -= self.origin
        return p

    def base_state(self, i: int) -> dict:
        p = self._pose(i)
        r = self.robots[i]
        g = r.data.projected_gravity_b.torch[0].detach().cpu().numpy()
        v = r.data.root_lin_vel_b.torch[0].detach().cpu().numpy()
        vw = r.data.root_lin_vel_w.torch[0].detach().cpu().numpy()
        wz = float(r.data.root_ang_vel_b.torch[0, 2])
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -float(g[2])))))
        return {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "yaw_rad": _yaw(p[3:7]), "tilt_deg": tilt,
                "fallen": bool(tilt > 60.0 or p[2] < 0.06), "vx_body": float(v[0]), "vy_body": float(v[1]),
                "speed": float(math.hypot(vw[0], vw[1])), "wz": wz, "mode": self.mode[i],
                "throttle": self.throttle[i], "target_yaw_rad": self.target_yaw[i], "sim_time_s": self.time_s}

    # ------------------------------------------------------------------ control
    def set_control(self, i: int, mode: str, command=(0.0, 0.0, 0.0), goal_xy=None) -> None:
        """``command = (throttle, unused, world_heading_target_rad)``. ``stand`` (port idle) = coast, same heading."""
        throttle = float(command[0]) if mode in ("skate", "brake") else 0.0
        if mode in ("stand", "coast"):
            mode = "coast"
        elif mode not in SKATE_MODES:
            raise ValueError(f"unknown skate mode {mode!r}")
        self.mode[i] = mode
        self.throttle[i] = max(-0.5, min(0.8, throttle))
        if mode != "coast" and command is not None and len(command) > 2 and command[2] is not None:
            self.target_yaw[i] = _wrap(float(command[2]))

    def _commands(self):
        cmd = np.zeros((self.n, 3), np.float32)
        for i in range(self.n):
            yaw = self.heading_source(i) if self.heading_source is not None else _yaw(self._pose(i)[3:7])
            err = _wrap(self.target_yaw[i] - yaw)
            cmd[i] = (self.throttle[i], 0.0, max(-HEADING_MAX_RAD, min(HEADING_MAX_RAD, err)))
        return cmd

    def step(self, steps: int = 1) -> None:
        torch = self._torch
        for _ in range(steps):
            cmd = self._commands()
            rows = []
            for i in range(self.n):
                o = self.obs[f"racer_{i}"][0].detach().cpu().numpy().astype(np.float32).copy()
                o[CMD] = cmd[i]
                o[51:61] = 0.0
                rows.append(np.nan_to_num(o, nan=0.0, posinf=0.0, neginf=0.0))
            batch = np.stack(rows)
            inp = self.actor.get_inputs()[0]
            if inp.shape[0] == 1 and self.n > 1:  # exported with a fixed batch of 1 (oval_roller): one call per robot
                act = np.concatenate([self.actor.run(None, {inp.name: batch[j:j + 1]})[0] for j in range(self.n)])
            else:
                act = self.actor.run(None, {inp.name: batch})[0]
            act = act.astype(np.float32)
            self._record(batch, act, cmd)
            with torch.inference_mode():
                obs, _, term, trunc, _ = self.env.step(torch.as_tensor(act.reshape(1, -1), device=self.env.device))
            if bool((term | trunc).any()):
                raise RuntimeError(f"unexpected environment reset at step {self.step_index}")
            self.obs = obs
            self.last_cmd = cmd
            self.step_index += 1

    def _record(self, batch, act, cmd):
        torch = self._torch
        r = self.rec
        pose = torch.stack([rb.data.body_link_pose_w.torch[0] for rb in self.robots]).detach().cpu().numpy()
        pose[..., :3] -= self.origin
        r["time_s"].append(self.time_s)
        r["body_pose"].append(pose.astype(np.float32))
        r["root_lin_vel_b"].append(np.stack([rb.data.root_lin_vel_b.torch[0].detach().cpu().numpy()
                                             for rb in self.robots]).astype(np.float32))
        r["proj_grav"].append(np.stack([rb.data.projected_gravity_b.torch[0].detach().cpu().numpy()
                                        for rb in self.robots]).astype(np.float32))
        r["mode"].append([SKATE_MODES.index(m) for m in self.mode])
        r["command"].append(cmd.copy())
        r["target_yaw"].append(np.asarray(self.target_yaw, np.float32))
        r["actor_obs"].append(batch.astype(np.float16))
        r["action"].append(act.astype(np.float16))

    def save(self, path: Path, extra: dict | None = None) -> Path:
        path = Path(path)
        arrays = {k: np.asarray(v) for k, v in self.rec.items()}
        np.savez_compressed(path, body_names=np.asarray(self.body_names), joint_names=np.asarray(self.joint_names),
                            robot_names=np.asarray(self.names), modes=np.asarray(SKATE_MODES), **arrays)
        meta = {**self.meta, **(extra or {}), "frames": len(self.rec["time_s"]), "control_dt_s": CONTROL_DT_S}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
        return path


def build_skate_cfg(*, gtceu_root: Path, roller_usd: Path, course_usd: Path | None, spawns: list[SkateSpawn],
                    device: str, seed: int):
    """Scene: N skate robots in ONE environment (shared Newton world) on the evaluation task's ground plane + course."""
    skills = Path(gtceu_root).resolve() / "scripts/skills"
    if str(skills) not in sys.path:
        sys.path.insert(0, str(skills))
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils import clone
    from isaaclab.utils.configclass import configclass
    import torch

    import isaaclab_tasks  # noqa: F401
    import pollen_skill_envs as envs
    from isaaclab_tasks.utils import resolve_task_config

    cfg, _ = resolve_task_config("Pollen-MicroDuck-Roller-Eval", "rsl_rl_cfg_entry_point", play_mode=True, overrides=[])
    num = len(spawns)
    base_policy, base_events, base_action = cfg.observations.policy, cfg.events, cfg.actions.joint_pos
    robot_cfg = cfg.scene.robot
    robot_cfg = robot_cfg.replace(spawn=robot_cfg.spawn.replace(usd_path=str(Path(roller_usd).resolve(strict=True))))

    @configclass
    class Obs:
        pass

    @configclass
    class Act:
        pass

    @configclass
    class Ev:
        pass

    @configclass
    class Cmd:
        pass

    def zeros(env, dim: int):
        return torch.zeros(env.num_envs, dim, device=env.device)

    def imu_term(func, i):
        asset = SceneEntityCfg(f"racer_{i}")

        def term(env):
            return func(env, event_name=f"imu_misalignment_{i}", asset_cfg=asset)

        term.__name__ = f"imu_observation_racer_{i}"
        return term

    cfg.seed = seed
    cfg.sim.device = device
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 0.0
    cfg.scene.robot = None
    cfg.scene.contact_forces = None
    cfg.scene.self_collision = None
    cfg.observations, cfg.actions, cfg.events, cfg.commands = Obs(), Act(), Ev(), Cmd()
    for name in list(vars(cfg.terminations)):
        if name != "time_out" and not name.startswith("_"):
            setattr(cfg.terminations, name, None)
    cfg.episode_length_s = 1.0e5
    solver = cfg.sim.physics.solver_cfg
    solver.njmax = 160 * num + 256
    solver.nconmax = 32 * num + 256
    z_range = base_events.reset_base.params["pose_range"]["z"]
    servos = lambda i: SceneEntityCfg(f"racer_{i}", joint_names=envs.MICRODUCK_JOINT_NAMES, preserve_order=True)  # noqa: E731
    for i, s in enumerate(spawns):
        yaw = math.radians(s.yaw_deg)
        robot = robot_cfg.replace(prim_path=f"{{ENV_REGEX_NS}}/Racer_{i}")
        robot.init_state = robot.init_state.replace(pos=(s.x, s.y, robot.init_state.pos[2]),
                                                    rot=(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)))
        setattr(cfg.scene, f"racer_{i}", robot)
        group = clone(base_policy)
        for term_name in ("base_ang_vel", "projected_gravity"):
            term = getattr(group, term_name)
            term.params = dict(term.params)
            term.params["term_func"] = imu_term(term.params["term_func"], i)
        group.joint_pos.params = {"asset_cfg": servos(i), "biased": True, "action_name": f"joint_pos_{i}"}
        group.joint_vel.params = {"asset_cfg": servos(i)}
        group.actions.params = {"action_name": f"joint_pos_{i}"}
        group.velocity_commands = ObsTerm(func=zeros, params={"dim": 3})  # overwritten with the twist each step
        group.head_pose_commands = ObsTerm(func=zeros, params={"dim": 4})
        group.body_pose_commands = ObsTerm(func=zeros, params={"dim": 6})
        setattr(cfg.observations, f"racer_{i}", group)
        setattr(cfg.actions, f"joint_pos_{i}", base_action.replace(asset_name=f"racer_{i}"))
        # The evaluation's per-robot terms with IDENTITY parameters (no randomization): the observation terms
        # reference the IMU event state; resets place each robot exactly at its spawn pose.
        for name in ("encoder_bias", "imu_misalignment", "reset_base", "reset_robot_joints"):
            term = getattr(base_events, name, None)
            if term is None:
                continue
            params = dict(term.params)
            if name == "encoder_bias":
                params["action_name"] = f"joint_pos_{i}"
                params["bias_range"] = (0.0, 0.0)
            elif name == "imu_misalignment":
                params["max_angle_deg"] = 0.0
            else:
                params["asset_cfg"] = (params.get("asset_cfg") or SceneEntityCfg("robot")).replace(name=f"racer_{i}")
            if name == "reset_base":
                params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0),
                                        "z": (max(z_range), max(z_range))}
                params["velocity_range"] = {}
            setattr(cfg.events, f"{name}_{i}", EventTerm(func=term.func, mode=term.mode, params=params))
    if course_usd is not None:
        cfg.scene.course = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Course",
                                        spawn=sim_utils.UsdFileCfg(usd_path=str(Path(course_usd).resolve(strict=True))))
    return cfg


def create_skate_world(cfg, *, gtceu_root: Path, spawns: list[SkateSpawn], policy: SkatePolicy, roller_usd: Path,
                       meta: dict | None = None) -> MicroduckSkateWorld:
    import onnxruntime as ort
    import torch

    from isaaclab.envs import ManagerBasedRLEnv

    skills = Path(gtceu_root).resolve() / "scripts/skills"
    if str(skills) not in sys.path:
        sys.path.insert(0, str(skills))
    import pollen_skill_envs as envs

    env = ManagerBasedRLEnv(cfg=cfg)
    num = len(spawns)
    terms = list(env.action_manager.active_terms)
    if terms != [f"joint_pos_{i}" for i in range(num)]:
        raise RuntimeError(f"unexpected action term order {terms}")
    for i in range(num):
        names = list(env.action_manager.get_term(f"joint_pos_{i}")._joint_names)
        if names != list(envs.MICRODUCK_JOINT_NAMES):
            raise RuntimeError(f"robot {i} joint order differs from the ONNX contract: {names}")
        dims = [int(np.prod(d)) for d in env.observation_manager.group_obs_term_dim[f"racer_{i}"]]
        if sum(dims) != OBS_DIM or int(np.cumsum([0] + dims)[5]) != 48:
            raise RuntimeError(f"robot {i} observation layout {dims} differs from the 61-D contract")
    sess = ort.InferenceSession(str(policy.onnx), providers=["CPUExecutionProvider"])
    if tuple(sess.get_inputs()[0].shape[-1:]) != (OBS_DIM,) or tuple(sess.get_outputs()[0].shape[-1:]) != (ACT_DIM,):
        raise RuntimeError("skating ONNX must map 61 observations to 14 joint targets")
    racers = [env.scene[f"racer_{i}"] for i in range(num)]
    wheel_ids = [r.find_joints(list(envs.ROLLER_WHEEL_JOINTS), preserve_order=True)[0] for r in racers]

    def set_wheel_friction():
        for r, ids in zip(racers, wheel_ids):
            r.write_joint_friction_coefficient_to_sim_index(
                joint_friction_coeff=torch.full((1, len(ids)), WHEEL_FRICTION_NM, device=env.device), joint_ids=ids)

    set_wheel_friction()
    obs, _ = env.reset()
    set_wheel_friction()
    world = MicroduckSkateWorld(env, [s.name for s in spawns], sess, policy, torch, {
        **(meta or {}), "robot_kind": "skates", "robot_usd": str(Path(roller_usd).resolve()),
        "robot_usd_sha256": hashlib.sha256(Path(roller_usd).read_bytes()).hexdigest(),
        "policy": {"name": policy.name, "onnx": str(policy.onnx), "sha256": policy.digest()},
        "command_channel": "twist = [throttle, 0, heading_error], heading error clamped to +-%.2f rad" % HEADING_MAX_RAD,
        "wheel_friction_nm": WHEEL_FRICTION_NM, "shared_newton_world": True,
        "spawns": [vars(s) for s in spawns], "domain_randomization": "none (identity parameters)"})
    world.obs = obs
    for i, s in enumerate(spawns):
        world.target_yaw[i] = math.radians(s.yaw_deg)
    return world


__all__ = ["MicroduckSkateWorld", "SkatePolicy", "SkateSpawn", "build_skate_cfg", "create_skate_world",
           "CONTROL_DT_S", "SKATE_MODES", "HEADING_MAX_RAD"]
