#!/usr/bin/env python3
"""Isaac Lab-Arena humanoid loco-manipulation bridge for CASCADE's ``humanoid_locomanip`` domain.

Runs INSIDE Isaac Lab-Arena's Python environment (it imports Arena, Isaac Lab and the GR00T
client); CASCADE never imports it and talks to it over loopback TCP, JSON lines
(``cascade.skills.humanoid_bridge`` documents the protocol). It owns one Arena environment and
one remote GR00T policy client, and it never decides anything on its own:

- every control step it computes the policy's proposed action, sends the MEASURED state plus that
  action to CASCADE, and applies the action only after CASCADE's gate answers ``pass``, with the
  finger targets the gate clamped onto their joint range replaced (``apply_clamps``);
- on ``stop`` it switches to the hold command (zero navigation velocity, previous joint/height
  targets; the whole-body balance controller keeps running) and ends the segment;
- Arena's own terminations (success / failure / time-out) end a policy segment WITHOUT resetting
  the scene (the auto-reset is suspended during segments), so the verifier sees the terminal
  state; Arena's success signal is reported separately, never as evidence;
- after the segment it holds for ``rest_steps`` control steps, then reports the final snapshot.

Usage (Arena checkout as cwd, its venv; a GR00T server must already listen):

    .venv/bin/python <cascade>/scripts/arena_locomanip_bridge.py --bridge_port 8765 \
        --bridge_record_dir <dir> --policy_label "GR00T N1.6 fine-tuned (Arena G1 loco-manip)" \
        --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy \
        --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/g1_locomanip_gr00t_closedloop_config.yaml \
        --remote_host 127.0.0.1 --remote_port 5555 --enable_cameras --device cuda:0 --presets newton \
        galileo_g1_locomanip_pick_and_place --object brown_box --embodiment g1_wbc_joint
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import socket
import sys
import threading
import time
import traceback

PROTOCOL = 2  # cascade.skills.humanoid_bridge.PROTOCOL: gate replies carry finger-target clamps
OBSERVER = "cascade_observer_cam"
OBS_W, OBS_H = 1280, 720


def log(*args):
    print("[bridge]", *args, flush=True)


# ---------------------------------------------------------------------------------------------
# TCP: one session at a time; a second client gets {"type": "busy"} and is closed.
class Server:
    def __init__(self, port: int, hello_fn):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", port))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        self.hello_fn = hello_fn
        self.pending: queue.Queue = queue.Queue()
        self.busy = threading.Event()
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            conn, _ = self.sock.accept()
            if self.busy.is_set():
                try:
                    conn.sendall((json.dumps({"type": "busy"}) + "\n").encode())
                finally:
                    conn.close()
                continue
            self.busy.set()
            self.pending.put(conn)


class Conn:
    def __init__(self, sock):
        self.sock = sock
        self.rfile = sock.makefile("r", encoding="utf-8", newline="\n")

    def send(self, msg):
        self.sock.sendall((json.dumps(msg, allow_nan=False) + "\n").encode())

    def recv(self):
        line = self.rfile.readline()
        if not line:
            raise ConnectionError("client closed")
        return json.loads(line)

    def close(self):
        for x in (self.rfile, self.sock):
            try:
                x.close()
            except OSError:
                pass


# ---------------------------------------------------------------------------------------------
def t(x):
    """ProxyArray / tensor -> torch tensor."""
    return x.torch if hasattr(x, "torch") else x


def observer_views():
    """[(scene key, eye, target)]; BRIDGE_CAMS="eye_x,eye_y,eye_z:t_x,t_y,t_z;..." overrides (env frame, m)."""
    spec = os.environ.get("BRIDGE_CAMS", "-1.05,-1.6,2.2:0.0,-0.6,-0.45;-1.0,-0.8,2.1:0.2,-0.4,-0.5")
    views = []
    for k, item in enumerate(v for v in spec.split(";") if v.strip()):
        eye, target = ([float(x) for x in part.split(",")] for part in item.split(":"))
        views.append((OBSERVER if k == 0 else f"{OBSERVER}_{k}", eye, target))
    return views


def look_at_xyzw(eye, target):
    """World-convention camera orientation (forward +X, up +Z) looking from eye to target, as (x, y, z, w)."""
    dx, dy, dz = (t - e for t, e in zip(target, eye))
    yaw = math.atan2(dy, dx)
    pitch = math.atan2(-dz, math.hypot(dx, dy))  # positive tilts the forward axis down
    cy, sy, cp, sp = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    # q = q_yaw(z) * q_pitch(y)
    return (-sy * sp, cy * sp, sy * cp, cy * cp)


def xyzw_to_wxyz(q):
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def rnd(v, n=5):
    return [round(float(x), n) for x in v]


class Bridge:
    def __init__(self, args, env, policy, policy_label, record_dir):
        import torch

        self.torch = torch
        self.args, self.env, self.policy = args, env, policy
        self.unwrapped = env.unwrapped
        self.scene = self.unwrapped.scene
        self.robot = self.scene["robot"]
        self.policy_label = policy_label
        self.record_dir = record_dir
        self.object = args.object
        self.destination = getattr(args, "destination", None) or "blue_sorting_bin"
        self.joint_names = list(self.robot.data.joint_names)
        self.body_names = list(self.robot.body_names)
        self.pelvis_idx = self.body_names.index("pelvis") if "pelvis" in self.body_names else None
        self.hand_idx = [i for i, n in enumerate(self.body_names) if n.startswith(("left_hand", "right_hand"))
                         or n in ("left_wrist_yaw_link", "right_wrist_yaw_link", "left_rubber_hand", "right_rubber_hand")]
        self.ankle_idx = [i for i, n in enumerate(self.body_names) if "ankle_roll" in n]
        self.floor_z = 0.0
        self.aabb = {}
        self.in_segment = False
        self.terminal = None
        self.obs = None
        self.step_count = 0
        self.session = 0
        self.last_action = None
        self._orig_reset_idx = self.unwrapped._reset_idx
        self.unwrapped._reset_idx = self._reset_idx
        self.physics = self._physics_name()
        log(f"robot joints={len(self.joint_names)} bodies={len(self.body_names)} hands={[self.body_names[i] for i in self.hand_idx]}")

    # -- Arena terminations during a segment end it without resetting the scene -------------
    def _reset_idx(self, env_ids):
        if self.in_segment:
            if self.terminal is None:
                tm = self.unwrapped.termination_manager
                terms = {}
                for name in tm.active_terms:
                    try:
                        terms[name] = bool(t(tm.get_term(name))[0].item())
                    except Exception:  # noqa: BLE001
                        terms[name] = None
                self.terminal = {"step": self.step_count, "terms": terms,
                                 "time_out": bool(t(tm.time_outs)[0].item()),
                                 "terminated": bool(t(tm.terminated)[0].item())}
            return None
        return self._orig_reset_idx(env_ids)

    def _physics_name(self):
        cfg = self.unwrapped.cfg.sim
        phys = getattr(cfg, "physics", None)
        name = type(phys).__name__ if phys is not None else "PhysxCfg(default)"
        return {"backend": "newton" if "Newton" in name else "physx", "config": name,
                "device": str(self.unwrapped.device), "dt": float(cfg.dt),
                "decimation": int(self.unwrapped.cfg.decimation)}

    # -- geometry -----------------------------------------------------------------------------
    def _local_aabb(self, name):
        import omni.usd
        from pxr import Gf, Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        pattern = self.scene[name].cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
        path = re.sub(r"env_(\.\*|\[\^/\]\+)", "env_0", pattern)
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"no USD prim for {name} at {path}")
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        rng = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
        m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        scale = [Gf.Vec3d(m.GetRow(i)[0], m.GetRow(i)[1], m.GetRow(i)[2]).GetLength() for i in range(3)]
        lo, hi = rng.GetMin(), rng.GetMax()
        return [[lo[i] * scale[i] for i in range(3)], [hi[i] * scale[i] for i in range(3)]]

    def _rigid(self, name, with_aabb=True):
        data = self.scene[name].data
        vel = getattr(data, "root_lin_vel_w", None)
        out = {"pos": rnd(t(data.root_pos_w)[0]), "quat_wxyz": rnd(xyzw_to_wxyz(t(data.root_quat_w)[0])),
               "lin_vel": rnd(t(vel)[0]) if vel is not None else [0.0, 0.0, 0.0]}
        if with_aabb:
            out["aabb_local"] = self.aabb[name]
        return out

    def snapshot(self):
        d = self.robot.data
        pos = t(d.root_pos_w)[0]
        quat = t(d.root_quat_w)[0]
        bodies = t(d.body_pos_w)[0]
        hands = {self.body_names[i]: rnd(bodies[i], 4) for i in self.hand_idx}
        return {"step": self.step_count, "sim_time_s": round(self.step_count * self.physics["dt"] * self.physics["decimation"], 4),
                "floor_z": round(self.floor_z, 4), "pelvis_pos": rnd(pos), "pelvis_quat_wxyz": rnd(xyzw_to_wxyz(quat)),
                "pelvis_lin_vel": rnd(t(d.root_lin_vel_w)[0]), "joint_pos": rnd(t(d.joint_pos)[0], 4),
                "joint_vel": rnd(t(d.joint_vel)[0], 4), "hands": hands,
                "objects": {self.object: self._rigid(self.object)},
                "destinations": {self.destination: self._rigid(self.destination)}}

    def joints(self):
        d = self.robot.data
        lim = t(d.joint_pos_limits)[0]
        vel = t(d.joint_vel_limits)[0]
        return {"names": self.joint_names, "lower": rnd(lim[:, 0]), "upper": rnd(lim[:, 1]),
                "velocity": rnd(vel)}

    def hello(self):
        return {"type": "hello", "protocol": PROTOCOL, "env": self.args.environment if hasattr(self.args, "environment")
                else "galileo_g1_locomanip_pick_and_place",
                "scene": "Isaac Lab-Arena galileo_g1_locomanip (shelf, table with bin)",
                "physics": self.physics["backend"], "physics_device": self.physics["device"], "physics_detail": self.physics,
                "policy": {"name": type(self.policy).__name__, "label": self.policy_label,
                           "server": f"{self.args.remote_host}:{self.args.remote_port}"},
                "objects": [self.object], "destinations": [self.destination], "control_hz": round(
                    1.0 / (self.physics["dt"] * self.physics["decimation"]), 3), "joints": self.joints()}

    # -- actions ------------------------------------------------------------------------------
    def split(self, action):
        """The action as the channels CASCADE's gate checks. An action with any other width is
        refused (raised, never applied): every value env.step receives is in one of these channels."""
        a = action[0].detach().cpu().tolist()
        n = len(self.joint_names)
        if len(a) != n + 7:
            raise RuntimeError(f"policy action has {len(a)} values; the gate checks exactly {n + 7} "
                               f"({n} joint targets, nav 3, base height 1, torso rpy 3): refused, not applied")
        return {"joint_targets": rnd(a[:n], 4), "nav": rnd(a[n:n + 3], 4), "base_height": round(a[n + 3], 4),
                "torso_rpy": rnd(a[n + 4:n + 7], 4)}

    def apply_clamps(self, action, reply):
        """Replace the joint targets CASCADE's gate clamped (finger targets moved onto their joint
        range). Each one must name a robot joint and lie inside that joint's own limits; anything
        else is a protocol violation, not something to apply. Returns how many were applied."""
        clamps = reply.get("clamp") or []
        if not clamps:
            return 0
        lim = t(self.robot.data.joint_pos_limits)[0]
        n = len(self.joint_names)
        for item in clamps:
            index, value = int(item[0]), float(item[1])
            if not (0 <= index < n and math.isfinite(value)
                    and float(lim[index, 0]) - 1e-4 <= value <= float(lim[index, 1]) + 1e-4):
                raise RuntimeError(f"gate clamp {item!r} is not a target inside a joint range")
            with self.torch.inference_mode():  # the policy's action is an inference-mode tensor
                action[0, index] = value
        return len(clamps)

    def hold_action(self):
        torch = self.torch
        n = len(self.joint_names)
        if self.last_action is not None:
            a = self.last_action.clone()
        else:
            a = torch.zeros((1, n + 7), device=self.unwrapped.device)
            a[0, :n] = t(self.robot.data.joint_pos)[0]
            a[0, n + 3] = 0.75
        a[0, n:n + 3] = 0.0
        return a

    # -- recording ----------------------------------------------------------------------------
    def _frame(self, key):
        try:
            if key == "observer" or key.startswith(OBSERVER):
                rgb = t(self.scene[OBSERVER if key == "observer" else key].data.output["rgb"])
            else:
                rgb = t(self.obs["camera_obs"][key]) if isinstance(self.obs, dict) and "camera_obs" in self.obs else None
                if rgb is None:
                    return None
            import numpy as np

            f = rgb[0, :, :, :3].detach().cpu().numpy()
            if f.dtype != np.uint8:
                f = np.clip(f * (255.0 if f.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
            return f
        except Exception:  # noqa: BLE001
            return None

    def _place_observer(self):
        return  # fixed observer (see with_observer)
        if OBSERVER not in self.scene.keys():  # noqa: unreachable -- kept for a chase camera on PhysX
            return
        import numpy as np
        torch = self.torch

        p = t(self.robot.data.root_pos_w)[0].detach().cpu().numpy().astype(float)
        target = np.array([p[0], p[1], p[2] - 0.3])
        self._cam_t = target if getattr(self, "_cam_t", None) is None else 0.96 * self._cam_t + 0.04 * target
        off = np.array([float(v) for v in os.environ.get("ARENA_CHASE_OFFSET", "-1.0,-1.0,2.4").split(",")])
        dev = self.unwrapped.device
        self.scene[OBSERVER].set_world_poses_from_view(
            eyes=torch.tensor([list(self._cam_t + off)], dtype=torch.float32, device=dev),
            targets=torch.tensor([list(self._cam_t)], dtype=torch.float32, device=dev))

    def record(self, seg_dir, i, every=2):
        if seg_dir is None or i % every:
            return
        from PIL import Image

        keys = [("observer", "observer")] + [(name, name.replace(OBSERVER, "observer")) for name, _, _ in observer_views()[1:]]
        for key, sub in keys + [(self.args.bridge_head_cam, "head")]:
            f = self._frame(key)
            if f is not None:
                d = os.path.join(seg_dir, sub)
                os.makedirs(d, exist_ok=True)
                Image.fromarray(f).save(os.path.join(d, f"{self.step_count:06d}.jpg"), quality=85)

    # -- protocol handlers ----------------------------------------------------------------------
    def reset(self, seed):
        torch = self.torch
        self.in_segment = False
        self.terminal = None
        with torch.inference_mode():
            self.obs, _ = self.env.reset(seed=seed)
        self.policy.reset()
        self.step_count = 0
        self.last_action = None
        self._cam_t = None
        self._place_observer()
        bodies = t(self.robot.data.body_pos_w)[0]
        if self.ankle_idx:
            self.floor_z = float(min(bodies[i][2] for i in self.ankle_idx))
        for name in (self.object, self.destination):
            if name not in self.aabb:
                self.aabb[name] = [rnd(v, 4) for v in self._local_aabb(name)]
        return self.snapshot()

    def env_step(self, action):
        torch = self.torch
        with torch.inference_mode():
            self.obs, _, _, _, _ = self.env.step(action)
        self.step_count += 1
        self._place_observer()

    def segment(self, conn, msg):
        torch = self.torch
        seg_dir = None
        if self.record_dir:
            seg_dir = os.path.join(self.record_dir, f"session{self.session:03d}_seg{int(msg['id']):03d}")
            os.makedirs(seg_dir, exist_ok=True)
        self.policy.reset()
        self.policy.set_task_description(msg["instruction"])
        self.terminal = None
        self.in_segment = True
        reason, stopped = "budget", False
        start_step = self.step_count
        arena = None
        index = []
        try:
            for i in range(int(msg["max_steps"])):
                with torch.inference_mode():
                    action = self.policy.get_action(self.env, self.obs)
                state = self.snapshot()
                conn.send({"type": "step", "i": i, "phase": "policy", "state": state, "action": self.split(action)})
                reply = conn.recv()
                if reply.get("type") != "gate" or reply.get("i") != i:
                    raise RuntimeError(f"expected gate {i}, got {reply}")
                if reply["verdict"] != "pass":
                    reason, stopped = "gate_stop", True
                    break
                clamped = self.apply_clamps(action, reply)
                self.last_action = action.clone()
                self.env_step(action)
                index.append({"step": self.step_count, "i": i, "phase": "policy", "clamped": clamped})
                self.record(seg_dir, i)
                if self.terminal is not None:
                    terms = self.terminal["terms"]
                    arena = dict(self.terminal)
                    if terms.get("success"):
                        reason = "benchmark_success_signal"
                    elif self.terminal["time_out"]:
                        reason = "time_out"
                    else:
                        reason = "benchmark_failure_term:" + ",".join(k for k, v in terms.items() if v)
                    break
            hold = self.hold_action()
            for k in range(int(msg["rest_steps"])):
                state = self.snapshot()
                conn.send({"type": "step", "i": int(msg["max_steps"]) + k, "phase": "hold", "state": state,
                           "action": self.split(hold)})
                reply = conn.recv()  # the hold IS the stop command; it is applied either way
                if reply.get("type") != "gate":
                    raise RuntimeError(f"expected gate during hold, got {reply}")
                clamped = self.apply_clamps(hold, reply)
                self.env_step(hold)
                index.append({"step": self.step_count, "i": k, "phase": "hold", "clamped": clamped})
                self.record(seg_dir, k)
        finally:
            self.in_segment = False
        final = self.snapshot()
        recording = None
        if seg_dir:
            with open(os.path.join(seg_dir, "index.json"), "w") as f:
                json.dump({"instruction": msg["instruction"], "start_step": start_step, "steps": index,
                           "reason": reason, "physics": self.physics, "policy": self.policy_label}, f)
            recording = seg_dir
        conn.send({"type": "segment_end", "reason": reason, "steps": self.step_count - start_step,
                   "sim_time_s": final["sim_time_s"], "final": final, "arena": arena, "stopped": stopped,
                   "recording": recording})

    def _handle(self, conn, msg):
        kind = msg.get("type")
        if kind == "session":
            seed = msg.get("seed")
            seed = int(seed) if seed is not None else int(time.time()) % 100000
            conn.send({"type": "ready", "state": self.reset(seed), "seed": seed})
        elif kind == "observe":
            conn.send({"type": "state", "state": self.snapshot()})
        elif kind == "segment":
            self.segment(conn, msg)
        elif kind != "close":
            conn.send({"type": "error", "error": f"unknown message {kind!r}"})

    def serve(self, server):
        log(f"listening on 127.0.0.1:{server.port}")
        while True:
            sock = server.pending.get()
            conn = Conn(sock)
            self.session += 1
            try:
                conn.send(self.hello())
                while True:
                    msg = conn.recv()
                    # Policy state (chunk scheduler) is built under inference mode, as in Arena's own runner:
                    # every handler runs under it too (an in-place reset outside it raises).
                    with self.torch.inference_mode():
                        self._handle(conn, msg)
                    if msg.get("type") == "close":
                        break
            except (ConnectionError, OSError, json.JSONDecodeError) as exc:
                log(f"session {self.session} ended: {exc}")
            except Exception as exc:  # noqa: BLE001
                log("session error:", traceback.format_exc())
                try:
                    conn.send({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
                except OSError:
                    pass
            finally:
                self.in_segment = False
                conn.close()
                server.busy.clear()


def main():
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("--bridge_port", type=int, default=8765)
    parser.add_argument("--bridge_record_dir", default=None)
    parser.add_argument("--bridge_head_cam", default="robot_head_cam_rgb")
    parser.add_argument("--policy_label", default="GR00T (remote policy server)")
    parser.add_argument("--bridge_preload", action="append", default=[],
                        help="module imported before the environment is built (e.g. a backend port patch)")
    parser.add_argument("--bridge_offline_assets", action="store_true",
                        help="serve already-cached remote assets without a per-file server freshness probe")
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    with SimulationAppContext(args_cli):
        import importlib

        for module in args_cli.bridge_preload:
            importlib.import_module(module)
            log(f"preloaded {module}")
        if args_cli.bridge_offline_assets:
            import tempfile

            import isaaclab.utils.assets as assets

            probe = assets._remote_fingerprint

            def cached_first(url):
                mirrored = assets._mirror_path(url, tempfile.gettempdir())
                if mirrored and os.path.isfile(mirrored):
                    return None  # Isaac Lab's documented offline path: use the local copy
                return probe(url)

            assets._remote_fingerprint = cached_first
            log("offline assets: cached copies are used without a server freshness probe")
        from isaaclab_arena.evaluation.policy_runner import get_policy_cls
        from isaaclab_arena.evaluation.policy_runner_cli import (
            add_policy_cli_args, add_policy_runner_arguments, build_policy_from_cli)
        from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

        add_policy_runner_arguments(parser)
        args_cli, _ = parser.parse_known_args()
        policy_cls = get_policy_cls(args_cli.policy_type)
        parser = get_isaaclab_arena_environments_cli_parser(parser)
        parser = add_policy_cli_args(parser, policy_cls)
        args_cli, hydra = parser.parse_known_args()
        args_cli.enable_cameras = True
        if args_cli.bridge_record_dir:
            import isaaclab_arena.environments.arena_env_builder as builder_mod

            original = builder_mod._configure_arena_visualizer_defaults

            def with_observer(env_cfg):
                original(env_cfg)
                import isaaclab.sim as sim_utils
                from isaaclab.sensors import TiledCameraCfg

                # FIXED observers (not policy inputs): moving a camera at run time did not reach the
                # renderer on the Newton preset (Fabric transform hierarchy), so each view is chosen once.
                for name, eye, target in observer_views():
                    setattr(env_cfg.scene, name, TiledCameraCfg(
                        prim_path="{ENV_REGEX_NS}/Cascade_" + name, update_period=0.0, height=OBS_H, width=OBS_W,
                        data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, clipping_range=(0.05, 40.0)),
                        offset=TiledCameraCfg.OffsetCfg(pos=tuple(eye), rot=look_at_xyzw(eye, target), convention="world")))

            builder_mod._configure_arena_visualizer_defaults = with_observer
        builder = get_arena_builder_from_cli(args_cli, hydra_overrides=hydra)
        env = builder.make_registered()
        policy = build_policy_from_cli(policy_cls, args_cli)
        bridge = Bridge(args_cli, env, policy, args_cli.policy_label, args_cli.bridge_record_dir)
        import torch

        with torch.inference_mode():
            bridge.reset(args_cli.seed if args_cli.seed is not None else 0)
        server = Server(args_cli.bridge_port, bridge.hello)
        print(f"BRIDGE_READY port={server.port}", flush=True)
        bridge.serve(server)


if __name__ == "__main__":
    sys.exit(main())
