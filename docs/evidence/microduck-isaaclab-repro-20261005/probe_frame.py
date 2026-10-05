#!/usr/bin/env python
"""Probe the MicroDuck root frame in the Lab env: which world direction is the root link's +x/+y, where are head and feet."""
import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402
from isaaclab.utils import validate  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lab_microduck_walk_test import TASKS, configure_env_cfg  # noqa: E402


def rot_from_xyzw(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="flat")
    parser.add_argument("--policy", default="velocity_flat")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variant", default="norand")
    parser.add_argument("--out", required=True)
    add_launcher_args(parser)
    parser.set_defaults(device=None)
    args_cli, hydra_args = setup_preset_cli(parser, None)
    sys.argv = [sys.argv[0]] + hydra_args
    task = TASKS[args_cli.task]
    env_cfg, _ = resolve_task_config(task, "", play_mode=True)
    env_cfg = configure_env_cfg(env_cfg, args_cli, args_cli.policy)
    validate(env_cfg)
    out = {}
    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        env = gym.make(task, cfg=env_cfg)
        cleanup.callback(env.close)
        uenv = env.unwrapped
        robot = uenv.scene["robot"]
        env.reset(seed=args_cli.seed)
        # let it settle 1 s with zero actions (hold default pose)
        for _ in range(50):
            env.step(torch.zeros(1, 14, device=uenv.device))
        d = robot.data
        names = list(robot.body_names)
        pos = d.body_link_pos_w.torch[0].detach().cpu().numpy()
        quat = d.body_link_quat_w.torch[0].detach().cpu().numpy()
        root_q = d.root_quat_w.torch[0].detach().cpu().numpy()
        root_p = d.root_pos_w.torch[0].detach().cpu().numpy()
        R = rot_from_xyzw(root_q)
        out["body_names"] = names
        out["root_pos_w"] = root_p.tolist()
        out["root_quat_w_raw"] = root_q.tolist()
        out["R_assuming_xyzw_cols_are_body_axes_in_world"] = R.tolist()
        out["projected_gravity_b"] = d.projected_gravity_b.torch[0].detach().cpu().numpy().tolist()
        out["bodies_rel_root_world"] = {n: (pos[i] - root_p).tolist() for i, n in enumerate(names)}
        out["bodies_rel_root_in_root_frame_xyzw"] = {n: (R.T @ (pos[i] - root_p)).tolist() for i, n in enumerate(names)}
        out["body_quats_raw"] = {n: quat[i].tolist() for i, n in enumerate(names)}
        # quat_apply check with isaaclab math
        from isaaclab.utils import math as m

        qt = torch.as_tensor(root_q).unsqueeze(0)
        ex = torch.tensor([[1.0, 0.0, 0.0]])
        out["isaaclab_quat_apply_root_q_to_ex"] = m.quat_apply(qt, ex)[0].tolist()
        out["isaaclab_quat_apply_inverse_root_q_to_gravity"] = m.quat_apply_inverse(qt, torch.tensor([[0.0, 0.0, -1.0]]))[0].tolist()
        out["root_lin_vel_b"] = d.root_lin_vel_b.torch[0].detach().cpu().numpy().tolist()
        out["root_link_lin_vel_b"] = d.root_link_lin_vel_b.torch[0].detach().cpu().numpy().tolist()
        try:
            import inspect

            src = inspect.getsource(type(d).root_link_quat_w.fget)
            out["root_link_quat_w_source"] = src[:1500]
        except Exception as exc:  # noqa: BLE001
            out["root_link_quat_w_source"] = str(exc)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1)[:6000])


if __name__ == "__main__":
    main()
