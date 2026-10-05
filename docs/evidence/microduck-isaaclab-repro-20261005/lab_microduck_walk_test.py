#!/usr/bin/env python
# Driver for the MicroDuck +-30 mm walk-then-zero-twist test, run DIRECTLY in Isaac Lab (fork
# AntoineRichard/IsaacLab antoiner/feat/microduck-rough-velocity @ eafc80d, Newton MJWarp + BAM).
#
# It builds the task environment exactly as `isaaclab play` does (resolve_task_config + play_mode +
# launch_simulation + gym.make), replaces the RL-library policy loader with onnxruntime on Antoine's
# exported velocity_{flat,rough}.onnx (normalization embedded), and drives the command terms by hand:
#   settle 1.0 s @ (0,0,0)  ->  hold (+-0.3,0,0) until the planar displacement along the initial heading
#   reaches 0.025 m (closed loop) or for a fixed 15 steps (open loop)  ->  (0,0,0) for 3.0 s.
# Every policy step is logged (sim time, root pose/velocity, command, 61-dim observation, 14 actions,
# joint positions). A pure 10 s zero-command standing test is also run.
"""MicroDuck +-30 mm walk-then-zero-twist reproduction in Isaac Lab."""

from __future__ import annotations

import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402
from isaaclab.utils import validate  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli  # noqa: E402

USD_DIR = "/home/johnny/Projects/demo/cascade-lab/MICRODUCK/external/isaaclab-microduck-usd-folder/isaaclab-microduck-usd"
POLICY_DIR = "/home/johnny/Projects/demo/cascade-lab/MICRODUCK/external/isaaclab-policies-20260901/isaaclab"
TASKS = {
    "flat": "IsaacContrib-Velocity-Flat-MicroDuck",
    "rough": "IsaacContrib-Velocity-Rough-MicroDuck",
}
POLICY_JOINTS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
SETTLE_STEPS = 50  # 1.0 s at 50 Hz
ZERO_STEPS = 150  # 3.0 s
STAND_STEPS = 500  # 10 s
MAX_COMMAND_STEPS = 150  # 3.0 s safety cap for the closed-loop command phase
OPEN_LOOP_STEPS = 15
DISPLACEMENT_TARGET = 0.025  # 30 mm minus 5 mm tolerance (CASCADE walk_distance rule)
STILL_SPEED = 0.02  # m/s
STILL_STEPS = 10  # 0.2 s


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=list(TASKS), required=True, help="Which MicroDuck task env to build.")
    parser.add_argument("--policy", choices=["velocity_flat", "velocity_rough"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--variant",
        choices=["lab_play", "norand", "cascade_nominal"],
        default="lab_play",
        help="lab_play: task cfg + play_mode() as `isaaclab play` does (randomization kept). "
        "norand: randomization ranges zeroed, Lab BAM deployment kept (vin 6.5-8.2 V sampled, sag 0-0.2, delay 3-6). "
        "cascade_nominal: norand + BAM vin 7.4 V, sag gain 0.1, delay 0, no current limit (CASCADE's nominal profile).",
    )
    parser.add_argument("--episodes", default="stand,fwd_cl,rev_cl,fwd_ol,rev_ol", help="Comma list of episodes to run.")
    parser.add_argument("--trials", type=int, default=1, help="Repetitions of each episode (seed+trial index).")
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument("--rough_flat_tiles", action="store_true", default=True)
    add_launcher_args(parser)
    parser.set_defaults(device=None)
    args_cli, hydra_args = setup_preset_cli(parser, argv)
    sys.argv = [sys.argv[0]] + hydra_args
    return args_cli


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def quat_to_yaw(q: np.ndarray) -> float:
    """Yaw of an (x, y, z, w) quaternion (Isaac Lab 3.0 root_quat_w convention)."""
    x, y, z, w = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def tilt_from_quat(q: np.ndarray) -> float:
    """Angle between the body z axis and world up [rad], for an (x, y, z, w) quaternion."""
    x, y, z, w = (float(v) for v in q)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    return math.acos(max(-1.0, min(1.0, r22)))


def configure_env_cfg(env_cfg, args_cli, policy_name: str):
    """Apply the local USD, the flat-tiles rough variant, the command overrides and the randomization variant."""
    usd_name = os.path.basename(env_cfg.scene.robot.spawn.usd_path)
    env_cfg.scene.robot.spawn.usd_path = os.path.join(USD_DIR, usd_name)
    env_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = 1
    # Long episodes so the 10 s standing test never times out (timeouts only trigger resets).
    env_cfg.episode_length_s = 60.0
    # Commands are written by this script every step; make the task's own resampling inert.
    for name in ("base_velocity", "head_pose", "body_pose"):
        term = getattr(env_cfg.commands, name)
        term.resampling_time_range = (1.0e6, 1.0e6)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.rel_forward_envs = 0.0
    env_cfg.commands.base_velocity.rel_turn_in_place_envs = 0.0
    env_cfg.commands.base_velocity.debug_vis = False
    # Rough task on FLAT terrain. An all-flat MESH terrain cannot be used: MJWarp's MuJoCo conversion rejects a
    # zero-volume terrain mesh ("mesh volume is too small: /World/ground/terrain/mesh_0", tried first). So the rough
    # task is run on the flat task's collision PLANE while keeping every other rough-task setting (100 solver
    # iterations, nconmax 200, njmax 1024, actor/critic/reward terms). The foot rays need a mesh, so they are dropped
    # and the critic/reward foot-height terms fall back to the flat-ground height (actor observations unaffected).
    if args_cli.task == "rough":
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
        env_cfg.scene.left_foot_height = None
        env_cfg.scene.right_foot_height = None
        env_cfg.observations.critic.foot_height.params["height_sensor_names"] = ()
        env_cfg.rewards.foot_clearance.params["height_sensor_names"] = ()
        env_cfg.rewards.foot_swing_height.params["height_sensor_names"] = ()
        env_cfg.curriculum.terrain_levels = None
    if args_cli.variant in ("norand", "cascade_nominal"):
        ev = env_cfg.events
        ev.foot_friction.params["static_friction_range"] = (1.0, 1.0)
        ev.foot_friction.params["dynamic_friction_range"] = (1.0, 1.0)
        ev.encoder_bias.params["bias_range"] = (0.0, 0.0)
        ev.imu_misalignment.params["max_angle_deg"] = 0.0
        ev.mass_inertia.params["mass_distribution_params"] = (1.0, 1.0)
        ev.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)}
        ev.randomize_com.params["com_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
        ev.randomize_head_com.params["com_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
        ev.randomize_joint_friction.params["scale_range"] = (1.0, 1.0)
        ev.randomize_armature.params["armature_distribution_params"] = (1.0, 1.0)
        # observation delays: IMU lag fixed at 0, joint velocity lag stays at the task's constant 1 step
        env_cfg.observations.policy.base_ang_vel.params["max_lag"] = 0
        env_cfg.observations.policy.projected_gravity.params["max_lag"] = 0
    if args_cli.variant == "cascade_nominal":
        servos = env_cfg.scene.robot.actuators["servos"]
        servos.vin = 7.4
        servos.vin_range = None
        servos.vin_drop_gain_range = (0.1, 0.1)
        servos.vin_min = 6.0
        servos.min_delay = 0
        servos.max_delay = 0
        servos.motor.max_current = 0.0  # zero disables current limiting
    return env_cfg


def summarize_cfg(env_cfg, args_cli):
    servos = env_cfg.scene.robot.actuators["servos"]
    sol = env_cfg.sim.physics.solver_cfg
    ev = env_cfg.events

    def ev_params(name):
        term = getattr(ev, name, None)
        if term is None:
            return None
        return {k: v for k, v in term.params.items() if k not in ("asset_cfg",)}

    return {
        "task": TASKS[args_cli.task],
        "variant": args_cli.variant,
        "seed": args_cli.seed,
        "usd_path": env_cfg.scene.robot.spawn.usd_path,
        "usd_sha256": sha256_of(env_cfg.scene.robot.spawn.usd_path),
        "terrain_type": env_cfg.scene.terrain.terrain_type,
        "terrain_sub_terrains": (
            list(env_cfg.scene.terrain.terrain_generator.sub_terrains)
            if env_cfg.scene.terrain.terrain_generator is not None
            else None
        ),
        "terrain_importer_class": str(env_cfg.scene.terrain.class_type),
        "physics_dt": env_cfg.sim.dt,
        "decimation": env_cfg.decimation,
        "policy_dt": env_cfg.sim.dt * env_cfg.decimation,
        "use_newton_actuators": env_cfg.sim.use_newton_actuators,
        "solver": {
            "iterations": sol.iterations,
            "ls_iterations": sol.ls_iterations,
            "nconmax": sol.nconmax,
            "njmax": sol.njmax,
            "cone": sol.cone,
            "impratio": sol.impratio,
            "integrator": sol.integrator,
            "use_mujoco_contacts": sol.use_mujoco_contacts,
        },
        "bam_actuator_cfg": {
            "kp_fw": servos.kp_fw,
            "vin": servos.vin,
            "vin_range": servos.vin_range,
            "vin_drop_gain_range": servos.vin_drop_gain_range,
            "vin_min": servos.vin_min,
            "min_delay": servos.min_delay,
            "max_delay": servos.max_delay,
            "motor_model": servos.motor.model,
            "motor_max_current": servos.motor.max_current,
            "motor_kt": servos.motor.kt,
            "motor_resistance": servos.motor.resistance,
            "friction_base": servos.motor.friction_base,
            "friction_viscous": servos.motor.friction_viscous,
            "stiff_frictionloss": servos.stiff_frictionloss,
        },
        "events": {name: ev_params(name) for name in (
            "foot_friction", "encoder_bias", "imu_misalignment", "mass_inertia", "reset_base", "reset_robot_joints",
            "randomize_com", "randomize_head_com", "randomize_joint_friction", "randomize_armature", "push_robot",
        )},
        "obs_corruption_enabled": env_cfg.observations.policy.enable_corruption,
        "imu_obs_lag": [
            env_cfg.observations.policy.base_ang_vel.params["min_lag"],
            env_cfg.observations.policy.base_ang_vel.params["max_lag"],
        ],
        "joint_vel_obs_lag": [
            env_cfg.observations.policy.joint_vel.delay_min_lag,
            env_cfg.observations.policy.joint_vel.delay_max_lag,
        ],
        "action_scale": env_cfg.actions.joint_pos.scale,
        "action_use_default_offset": env_cfg.actions.joint_pos.use_default_offset,
        "episode_length_s": env_cfg.episode_length_s,
    }


class OnnxPolicy:
    def __init__(self, stem: str):
        import onnxruntime as ort

        path = os.path.join(POLICY_DIR, f"{stem}.onnx")
        meta = json.load(open(os.path.join(POLICY_DIR, f"{stem}.onnx.meta.json")))
        self.sha256 = sha256_of(path)
        self.data_sha256 = sha256_of(path + ".data")
        assert self.sha256 == meta["onnx_sha256"], f"onnx sha mismatch for {stem}"
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        out = self.session.get_outputs()[0]
        assert inp.name == "obs" and list(inp.shape) == [1, 61], inp
        assert out.name == "actions" and list(out.shape) == [1, 14], out
        self.meta = meta
        self.path = path

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.ascontiguousarray(obs.reshape(1, 61).astype(np.float32))
        act = self.session.run(["actions"], {"obs": obs})[0]
        if not np.isfinite(act).all():
            raise RuntimeError("non-finite policy output")
        return act[0].astype(np.float32)


class Runner:
    def __init__(self, env, policy: OnnxPolicy, cfg_summary: dict):
        self.env = env
        self.uenv = env.unwrapped
        self.policy = policy
        self.robot = self.uenv.scene["robot"]
        self.joint_ids, self.joint_names = self.robot.find_joints(POLICY_JOINTS, preserve_order=True)
        assert list(self.joint_names) == POLICY_JOINTS, self.joint_names
        self.vel_term = self.uenv.command_manager.get_term("base_velocity")
        self.head_term = self.uenv.command_manager.get_term("head_pose")
        self.body_term = self.uenv.command_manager.get_term("body_pose")
        self.device = self.uenv.device
        self.dt = self.uenv.step_dt
        self.cfg_summary = cfg_summary
        # observation layout of the policy group
        names = list(self.uenv.observation_manager.active_terms["policy"])
        dims = [int(d[0]) if hasattr(d, "__len__") else int(d) for d in self.uenv.observation_manager.group_obs_term_dim["policy"]]
        offsets = np.cumsum([0] + dims)
        self.obs_layout = {n: (int(offsets[k]), int(offsets[k + 1])) for k, n in enumerate(names)}
        self.cmd_slice = slice(*self.obs_layout["velocity_commands"])
        self.head_slice = slice(*self.obs_layout["head_pose_commands"])
        self.body_slice = slice(*self.obs_layout["body_pose_commands"])
        self.verify_obs_patch = True
        self.obs_patch_max_abs_diff = 0.0

    # ---- helpers -------------------------------------------------------------------------------------------------
    def set_command(self, vx: float, vy: float, wz: float):
        self.vel_term.vel_command_b[:, 0] = vx
        self.vel_term.vel_command_b[:, 1] = vy
        self.vel_term.vel_command_b[:, 2] = wz
        self.vel_term.is_standing_env[:] = False
        if hasattr(self.vel_term, "is_heading_env"):
            self.vel_term.is_heading_env[:] = False
        self.head_term._command[:] = 0.0
        self.body_term._command[:] = 0.0

    def state(self) -> dict:
        d = self.robot.data
        q = d.root_quat_w.torch[0].detach().cpu().numpy().astype(float)
        st = {
            "pos": d.root_pos_w.torch[0].detach().cpu().numpy().astype(float).tolist(),
            "com_pos": d.root_com_pos_w.torch[0].detach().cpu().numpy().astype(float).tolist(),
            "quat_xyzw": q.tolist(),
            "yaw": quat_to_yaw(q),
            "tilt": tilt_from_quat(q),
            "lin_vel_w": d.root_lin_vel_w.torch[0].detach().cpu().numpy().astype(float).tolist(),
            "ang_vel_w": d.root_ang_vel_w.torch[0].detach().cpu().numpy().astype(float).tolist(),
            "lin_vel_b": d.root_lin_vel_b.torch[0].detach().cpu().numpy().astype(float).tolist(),
            "ang_vel_b": d.root_ang_vel_b.torch[0].detach().cpu().numpy().astype(float).tolist(),
            "joint_pos": d.joint_pos.torch[0, self.joint_ids].detach().cpu().numpy().astype(float).tolist(),
            "joint_vel": d.joint_vel.torch[0, self.joint_ids].detach().cpu().numpy().astype(float).tolist(),
        }
        return st

    def bam_readback(self) -> dict:
        from isaaclab.actuators.newton import read_group_parameter

        out = {}
        for attr in ("vin", "sag_gain", "friction_scale", "max_current", "max_effort", "kp_fw"):
            try:
                v = read_group_parameter(self.robot.actuators, "servos", "drive", attr)
                out[attr] = v[0].detach().cpu().numpy().astype(float).tolist()
            except Exception as exc:  # noqa: BLE001
                out[attr] = f"unavailable: {exc}"
        try:
            act = self.robot.actuators["servos"]
            for name in ("min_delay", "max_delay", "delay_update_period", "delay_hold_prob"):
                for holder in (act, getattr(act, "drive", None), getattr(act, "newton_actuator", None)):
                    if holder is not None and hasattr(holder, name):
                        out[name] = getattr(holder, name)
                        break
        except Exception as exc:  # noqa: BLE001
            out["delay_readback_error"] = str(exc)
        return out

    def default_joint_pos(self) -> list[float]:
        return self.robot.data.default_joint_pos.torch[0, self.joint_ids].detach().cpu().numpy().astype(float).tolist()

    # ---- stepping ------------------------------------------------------------------------------------------------
    def reset(self, seed: int) -> np.ndarray:
        torch.manual_seed(seed)
        np.random.seed(seed)
        obs, _ = self.env.reset(seed=seed)
        self.set_command(0.0, 0.0, 0.0)
        # the velocity-command slice of this vector still holds the command the reset sampled; step() overwrites it
        return obs["policy"][0].detach().cpu().numpy().astype(np.float32)

    def step(self, obs: np.ndarray, cmd: tuple[float, float, float], log: list, phase: str, step_index: int):
        self.set_command(*cmd)
        # `obs` came out of the previous env.step (or reset) with the command that was active then. The policy must
        # act on the command that is active NOW, so the velocity-command slice is replaced (the head/body pose
        # command slices are always zero here). Nothing else in the vector depends on the command.
        obs_now = obs.copy()
        obs_now[self.cmd_slice] = np.asarray(cmd, dtype=np.float32)
        obs_now[self.head_slice] = 0.0
        obs_now[self.body_slice] = 0.0
        if self.verify_obs_patch:
            # one-off consistency check: a recompute (without touching delay history) must give the same vector
            rec_obs = self.uenv.observation_manager.compute()["policy"][0].detach().cpu().numpy().astype(np.float32)
            self.obs_patch_max_abs_diff = max(self.obs_patch_max_abs_diff, float(np.max(np.abs(rec_obs - obs_now))))
        action = self.policy(obs_now)
        action_t = torch.as_tensor(action, device=self.device).unsqueeze(0)
        next_obs, _rew, terminated, truncated, _info = self.env.step(action_t)
        st = self.state()
        rec = {
            "i": step_index,
            "phase": phase,
            "t": step_index * self.dt,
            "cmd": list(cmd),
            "obs": obs_now.astype(float).tolist(),
            "action": action.astype(float).tolist(),
            "terminated": bool(terminated[0].item()),
            "truncated": bool(truncated[0].item()),
            **st,
        }
        log.append(rec)
        done = rec["terminated"] or rec["truncated"]
        return next_obs["policy"][0].detach().cpu().numpy().astype(np.float32), done

    # ---- episodes ------------------------------------------------------------------------------------------------
    def run_episode(self, kind: str, seed: int) -> dict:
        """kind in stand | fwd_cl | rev_cl | fwd_ol | rev_ol."""
        t_wall = time.time()
        log: list[dict] = []
        obs = self.reset(seed)
        result = {"kind": kind, "seed": seed, "fell": False, "reset_during_episode": False, "command_reached": None}
        i = 0
        done = False
        # settle
        for _ in range(SETTLE_STEPS):
            obs, done = self.step(obs, (0.0, 0.0, 0.0), log, "settle", i)
            i += 1
            if done:
                break
        if done:
            result["reset_during_episode"] = True
            result["fell"] = bool(log[-1]["terminated"])
            result["log"] = log
            result["note"] = "episode ended during settle"
            return result
        settle_end = self.state()
        result["settle_end_state"] = settle_end
        if kind == "stand":
            n_still = 0
            ever_still = False
            for _ in range(STAND_STEPS):
                obs, done = self.step(obs, (0.0, 0.0, 0.0), log, "stand", i)
                i += 1
                sp = math.hypot(log[-1]["lin_vel_w"][0], log[-1]["lin_vel_w"][1])
                n_still = n_still + 1 if sp < STILL_SPEED else 0
                ever_still = ever_still or n_still >= STILL_STEPS
                if done:
                    break
            end = self.state()
            p0 = np.array(settle_end["pos"][:2])
            p1 = np.array(end["pos"][:2])
            result.update(
                {
                    "stand_steps": i - SETTLE_STEPS,
                    "stand_duration_s": (i - SETTLE_STEPS) * self.dt,
                    "stand_drift_m": float(np.linalg.norm(p1 - p0)),
                    "stand_drift_xy_m": (p1 - p0).tolist(),
                    "stand_yaw_change_rad": wrap(end["yaw"] - settle_end["yaw"]),
                    "stand_ever_still_0p2s": ever_still,
                    "stand_max_planar_speed": max(math.hypot(r["lin_vel_w"][0], r["lin_vel_w"][1]) for r in log[SETTLE_STEPS:]),
                    "stand_max_tilt_rad": max(r["tilt"] for r in log[SETTLE_STEPS:]),
                    "fell": bool(done and log[-1]["terminated"]),
                    "reset_during_episode": bool(done),
                }
            )
            result["log"] = log
            result["wall_s"] = time.time() - t_wall
            return result

        sign = 1.0 if kind.startswith("fwd") else -1.0
        closed_loop = kind.endswith("_cl")
        cmd = (sign * 0.3, 0.0, 0.0)
        p0 = np.array(settle_end["pos"][:2])
        yaw0 = settle_end["yaw"]
        h0 = np.array([math.cos(yaw0), math.sin(yaw0)])
        l0 = np.array([-math.sin(yaw0), math.cos(yaw0)])
        cmd_steps = 0
        reached = False
        if closed_loop:
            max_steps = MAX_COMMAND_STEPS
        elif kind.endswith("_1s"):
            max_steps = 50  # steady-gait braking test: 1.0 s of command, then zero
        else:
            max_steps = OPEN_LOOP_STEPS
        while cmd_steps < max_steps:
            obs, done = self.step(obs, cmd, log, "command", i)
            i += 1
            cmd_steps += 1
            if done:
                break
            p = np.array(log[-1]["pos"][:2])
            along = float(np.dot(p - p0, h0))
            if closed_loop and sign * along >= DISPLACEMENT_TARGET:
                reached = True
                break
        cmd_end = self.state()
        p_cmd_end = np.array(cmd_end["pos"][:2])
        result.update(
            {
                "command": list(cmd),
                "command_steps": cmd_steps,
                "command_duration_s": cmd_steps * self.dt,
                "command_reached": reached if closed_loop else None,
                "during_command_along_heading_m": float(np.dot(p_cmd_end - p0, h0)),
                "during_command_lateral_m": float(np.dot(p_cmd_end - p0, l0)),
                "during_command_yaw_change_rad": wrap(cmd_end["yaw"] - yaw0),
                "during_command_max_tilt_rad": max(r["tilt"] for r in log[SETTLE_STEPS:]),
                "during_command_max_planar_speed": max(
                    math.hypot(r["lin_vel_w"][0], r["lin_vel_w"][1]) for r in log[SETTLE_STEPS:]
                ),
            }
        )
        if done:
            result["reset_during_episode"] = True
            result["fell"] = bool(log[-1]["terminated"])
            result["log"] = log
            result["note"] = "episode ended during command phase"
            result["wall_s"] = time.time() - t_wall
            return result
        # zero-twist hold
        zero_start_index = len(log)
        n_still = 0
        ever_still = False
        first_still_t = None
        for _ in range(ZERO_STEPS):
            obs, done = self.step(obs, (0.0, 0.0, 0.0), log, "zero", i)
            i += 1
            sp = math.hypot(log[-1]["lin_vel_w"][0], log[-1]["lin_vel_w"][1])
            n_still = n_still + 1 if sp < STILL_SPEED else 0
            if n_still >= STILL_STEPS and not ever_still:
                ever_still = True
                first_still_t = (len(log) - zero_start_index) * self.dt
            if done:
                break
        end = self.state()
        p_end = np.array(end["pos"][:2])
        zero_log = log[zero_start_index:]
        # drift in the last second of the zero window (steady-state behaviour)
        last_sec = zero_log[-50:] if len(zero_log) >= 50 else zero_log
        p_ls0 = np.array(last_sec[0]["pos"][:2]) if last_sec else p_end
        result.update(
            {
                "zero_steps": len(zero_log),
                "zero_duration_s": len(zero_log) * self.dt,
                "post_zero_along_heading_m": float(np.dot(p_end - p_cmd_end, h0)),
                "post_zero_lateral_m": float(np.dot(p_end - p_cmd_end, l0)),
                "post_zero_path_length_m": float(
                    sum(
                        math.hypot(b["pos"][0] - a["pos"][0], b["pos"][1] - a["pos"][1])
                        for a, b in zip(log[zero_start_index - 1 : -1], zero_log)
                    )
                ),
                "post_zero_yaw_change_rad": wrap(end["yaw"] - cmd_end["yaw"]),
                "post_zero_max_planar_speed": max(math.hypot(r["lin_vel_w"][0], r["lin_vel_w"][1]) for r in zero_log),
                "post_zero_mean_planar_speed_last_1s": float(
                    np.mean([math.hypot(r["lin_vel_w"][0], r["lin_vel_w"][1]) for r in last_sec])
                ),
                "post_zero_last_1s_displacement_m": float(np.linalg.norm(p_end - p_ls0)),
                "post_zero_max_tilt_rad": max(r["tilt"] for r in zero_log),
                "post_zero_ever_still_0p2s": ever_still,
                "post_zero_first_still_t_s": first_still_t,
                "total_along_heading_m": float(np.dot(p_end - p0, h0)),
                "total_lateral_m": float(np.dot(p_end - p0, l0)),
                "total_yaw_change_rad": wrap(end["yaw"] - yaw0),
                "fell": bool(done and log[-1]["terminated"]),
                "reset_during_episode": bool(done),
            }
        )
        result["log"] = log
        result["wall_s"] = time.time() - t_wall
        return result


def main(argv=None) -> int:
    args_cli = parse_args(argv)
    task = TASKS[args_cli.task]
    env_cfg, _ = resolve_task_config(task, "", play_mode=True)
    env_cfg = configure_env_cfg(env_cfg, args_cli, args_cli.policy)
    try:
        validate(env_cfg)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid environment configuration: {exc}") from None
    cfg_summary = summarize_cfg(env_cfg, args_cli)
    policy = OnnxPolicy(args_cli.policy)
    out = {
        "fork": "AntoineRichard/IsaacLab antoiner/feat/microduck-rough-velocity @ eafc80dfac8c45dc4392e84432d837b47ab54cc4",
        "policy": {"stem": args_cli.policy, "path": policy.path, "onnx_sha256": policy.sha256, "data_sha256": policy.data_sha256,
                   "meta": policy.meta},
        "cfg": cfg_summary,
        "protocol": {
            "settle_steps": SETTLE_STEPS, "zero_steps": ZERO_STEPS, "stand_steps": STAND_STEPS,
            "max_command_steps": MAX_COMMAND_STEPS, "open_loop_steps": OPEN_LOOP_STEPS,
            "displacement_target_m": DISPLACEMENT_TARGET, "still_speed_m_s": STILL_SPEED, "still_steps": STILL_STEPS,
            "command_speed_m_s": 0.3,
        },
        "episodes": [],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        env = gym.make(task, cfg=env_cfg)
        cleanup.callback(env.close)
        runner = Runner(env, policy, cfg_summary)
        out["default_joint_pos_policy_order"] = runner.default_joint_pos()
        out["joint_ids_policy_order"] = list(runner.joint_ids)
        out["all_joint_names_articulation_order"] = list(runner.robot.joint_names)
        env.reset(seed=args_cli.seed)
        out["bam_readback_after_first_reset"] = runner.bam_readback()
        out["policy_obs_terms"] = list(runner.uenv.observation_manager.active_terms["policy"])
        out["policy_obs_layout"] = runner.obs_layout
        kinds = [k.strip() for k in args_cli.episodes.split(",") if k.strip()]
        for trial in range(args_cli.trials):
            for kind in kinds:
                seed = args_cli.seed * 1000 + trial
                print(f"[run] episode {kind} trial {trial} seed {seed}", flush=True)
                res = runner.run_episode(kind, seed)
                res["trial"] = trial
                summary = {k: v for k, v in res.items() if k != "log"}
                print("[result]", json.dumps(summary), flush=True)
                out["episodes"].append(res)
                out["obs_patch_max_abs_diff"] = runner.obs_patch_max_abs_diff
                with open(args_cli.out, "w") as f:
                    json.dump(out, f)
    with open(args_cli.out, "w") as f:
        json.dump(out, f)
    print(f"[done] wrote {args_cli.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
