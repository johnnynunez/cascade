"""Optional real Arena/PhysX foundation through CASCADE's ordinary runtime.

Run with Isaac Lab's admitted SDK environment, never the lightweight CASCADE
venv. This is not a task benchmark, SafeArm driver or physical-stop test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))
ARENA_PIN = "c8d04e2199b86abbbb301bb22cec0effd83c4e63"
LAB_PIN = "ae37b028ea415c91ea2bc32609efcd759ed2b974"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def source_inventory(roots):
    result = {}
    for label, root in roots.items():
        paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
        if label == "cascade":
            paths += ["benchmark/arena/native_owner.py", "benchmark/arena/native_preflight.py"]
        selected = {}
        for name in sorted(set(paths)):
            path = root / name
            if path.is_file() and path.suffix in {".py", ".toml", ".lock", ".yaml", ".yml"}:
                selected[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        result[label] = {"revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root).decode().strip(), "files": selected}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arena-source", type=Path, required=True)
    parser.add_argument("--lab-source", type=Path, required=True)
    parser.add_argument("--sdk-version-file", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--python-deps", type=Path,
                        help="Explicit private dependency directory, including its trusted .pth files")
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    roots = {"cascade": REPO, "arena": args.arena_source.resolve(), "lab": args.lab_source.resolve()}
    sources = source_inventory(roots)
    if sources["arena"]["revision"] != ARENA_PIN or sources["lab"]["revision"] != LAB_PIN:
        raise ValueError("unreviewed Arena/Lab source revision")
    (out / "source-before.json").write_text(json.dumps(sources, indent=2) + "\n")
    receipt = {"preflight_pass": False, "task_success": False, "physical_stop_verified": False,
               "recipe_variant": "existing Isaac6.1rc26 PhysX; not Arena uv.lock parity",
               "argv": sys.argv, "pid": os.getpid(), "source_inventory_sha256": digest(sources),
               "sdk_version": args.sdk_version_file.read_text().strip(), "episodes": []}
    app = env = runtime = None
    code = 1

    def write():
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")

    try:
        write()
        if not receipt["sdk_version"].startswith("6.1.0-rc.26+"):
            raise ValueError("this reviewed variant requires Isaac6.1rc26; re-review another SDK")
        if args.python_deps is not None:
            import site
            site.addsitedir(str(args.python_deps.resolve()))
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(headless=True, device="cuda:0", multi_gpu=False,
            enable_cameras=args.capture, width=640, height=480,
            kit_args="--/app/settings/persistent=false "
                     "--/exts/isaacsim.core.simulation_manager/default_engine=physx "
                     "--/exts/isaacsim.physics.newton/auto_switch_on_startup=false "
                     "--/plugins/carb.tasking.plugin/threadCount=4 "
                     "--/plugins/omni.tbb.globalcontrol/maxThreadCount=4")
        app = launcher.app
        import importlib.metadata
        import numpy as np
        import torch
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(.20, 0)
        props = torch.cuda.get_device_properties(0)
        receipt["device"] = {"properties": str(props), "uuid": str(props.uuid),
                             "visible_count": torch.cuda.device_count()}
        assert receipt["device"]["visible_count"] == 1
        assert str(props.uuid).removeprefix("GPU-") == args.expected_gpu_uuid.removeprefix("GPU-")
        receipt["versions"] = {name: importlib.metadata.version(name) for name in
            ("torch", "numpy", "gymnasium", "hydra-core", "pydantic", "warp-lang")}
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
        from isaaclab_arena_environments.cube_goal_pose_environment import (
            CubeGoalPoseEnvironment, CubeGoalPoseEnvironmentCfg)
        from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicy, ZeroActionPolicyCfg
        from isaaclab_arena.policy.policy_base import PolicyBase
        from isaaclab_arena.utils.physics_backend import PhysicsBackend
        from cascade.eval.arena import ArenaPolicyAdapter
        from cascade.robotics.contracts import ResourceDescriptor
        from cascade.robotics.runtime import RobotRuntime
        from native_owner import NativePreflightDomain, PreflightController

        recipe = CubeGoalPoseEnvironment().build(CubeGoalPoseEnvironmentCfg())
        builder = ArenaEnvBuilder(recipe, ArenaEnvBuilderCfg(
            num_envs=1, device="cuda:0", seed=42, presets=PhysicsBackend.PHYSX,
            recorder_dataset_export_dir_path=str(out / "upstream-recorder")))
        env_cfg, env_kwargs = builder.compose_manager_cfg()
        if args.capture:
            from isaaclab.sensors import CameraCfg
            import isaaclab.sim as sim_utils
            from scipy.spatial.transform import Rotation
            eye = np.array([.8, -1., .9])
            direction = np.array([-.15, 0, .15]) - eye
            direction /= np.linalg.norm(direction)
            right = np.cross(direction, [0, 0, 1])
            right /= np.linalg.norm(right)
            up = np.cross(right, direction)
            xyzw = Rotation.from_matrix(np.column_stack((right, up, -direction))).as_quat()
            # This pinned Lab version uses XYZW, unlike the legacy Lab API.
            rotation = tuple(xyzw)
            env_cfg.scene.preflight_camera = CameraCfg(
                prim_path="{ENV_REGEX_NS}/PreflightCamera", update_period=0., width=640, height=480,
                data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24., horizontal_aperture=20.955, clipping_range=(.01, 100.)),
                offset=CameraCfg.OffsetCfg(pos=tuple(eye), rot=rotation, convention="opengl"))
            receipt["capture_variant"] = {"pose_xyz": eye.tolist(), "rotation_xyzw": list(rotation),
                "size": [640, 480], "policy_input": False, "physics_mutation": False}
        env = builder.make_registered(env_cfg, env_kwargs)
        base = env.unwrapped
        receipt.update(physics_dt_s=float(base.physics_dt), control_dt_s=float(base.step_dt),
                       physics_class=type(base.cfg.sim.physics).__module__ + "." + type(base.cfg.sim.physics).__name__)
        assert "physx" in receipt["physics_class"].lower()
        receipt["native_rigid_objects"] = list(base.scene.rigid_objects)
        assert "dex_cube" in base.scene.rigid_objects
        cube = base.scene["dex_cube"]
        robot = base.scene["robot"]
        from isaaclab.sim.utils.stage import get_current_stage
        stage_text = get_current_stage().Flatten().ExportToString()
        (out / "model-scene.usda").write_text(stage_text)
        receipt["model_scene_sha256"] = hashlib.sha256(stage_text.encode()).hexdigest()
        write()

        def tensor(value):
            return getattr(value, "torch", value).detach().cpu()

        def observe(epoch):
            values = {"q": tensor(robot.data.joint_pos),
                      "body_pos_w": tensor(robot.data.body_pos_w),
                      "cube_pose_w": tensor(cube.data.root_pose_w),
                      "cube_velocity_w": tensor(cube.data.root_vel_w)}
            assert all(bool(torch.isfinite(value).all()) for value in values.values())
            solves = int(base.sim.get_physics_step_count())
            sample = {"epoch": epoch, "model_scene_sha256": receipt["model_scene_sha256"],
                      "source_inventory_sha256": receipt["source_inventory_sha256"],
                      "control_step": int(base.common_step_counter), "physics_steps": solves,
                      "sim_time_s": solves * float(base.sim.get_physics_dt()),
                      **{name: value.tolist() for name, value in values.items()}}
            sample["snapshot_sha256"] = digest(sample)
            return sample

        def validate_action(action):
            if (tuple(action.shape) != tuple(env.action_space.shape)
                    or not bool(torch.isfinite(action).all()) or bool(action.any())):
                raise ValueError("only exact upstream zero_action accepted by this preflight")

        class NativeCascadePolicy(ArenaPolicyAdapter, PolicyBase):
            def __init__(self, runtime, controller, *, actuation_owner):
                PolicyBase.__init__(self, ZeroActionPolicyCfg())
                ArenaPolicyAdapter.__init__(self, runtime, controller, actuation_owner=actuation_owner)

        for label, cancel_before in (("foundation", None), ("cancel", 8)):
            epoch = uuid.uuid4().hex
            episode_dir = out / label
            episode_dir.mkdir()
            # Progress-tracker buffers created during inference must be reset
            # in the same PyTorch mode; no policy tensor/state is fabricated.
            with torch.inference_mode():
                observation, _ = env.reset()
            policy = ZeroActionPolicy(ZeroActionPolicyCfg())
            policy.reset()
            controller = PreflightController(policy)
            resource = ResourceDescriptor("arena/franka", "actuator", "arena_franka",
                capabilities=("bounded_zero_action_preflight",), controller_id="arena/native/env0",
                writer_id="arena/zero_action", admission="unvalidated",
                metadata={"epoch": epoch, "model_scene_sha256": receipt["model_scene_sha256"]})
            frames = []

            def witness():
                sample = observe(epoch)
                if args.capture:
                    from PIL import Image
                    # Render only; the native counter and actual solved state must not advance.
                    for _ in range(3 if not frames else 1):
                        base.sim.render()
                    image = tensor(base.scene["preflight_camera"].data.output["rgb"])[0, ..., :3].numpy()
                    if image.shape != (480, 640, 3) or image.dtype != np.uint8:
                        raise ValueError("unexpected native RGB layout")
                    if float(image.std()) < 5 or not 10 < float(image.mean()) < 250:
                        raise ValueError("native RGB is blank, dark or saturated")
                    assert sample == observe(epoch), "render mutated solved physics state"
                    name = f"frame-{len(frames):03d}.png"
                    Image.fromarray(image).save(episode_dir / name)
                    frames.append({"path": name, "snapshot_sha256": sample["snapshot_sha256"],
                                   "mean_rgb": float(image.mean()), "std_rgb": float(image.std())})
                with (episode_dir / "observations.jsonl").open("a") as stream:
                    stream.write(json.dumps(sample, allow_nan=False) + "\n")
                return sample

            stop_receipts = []

            def after_action(number):
                if cancel_before is not None and number == cancel_before:
                    stop_receipts.append(runtime.execute("emergency_stop", {}))

            domain = NativePreflightDomain(controller, command_resources=(resource,),
                env=env, observation=observation, observe=witness, validate_action=validate_action,
                deadline_monotonic_s=time.monotonic() + 45, after_action=after_action)
            runtime = RobotRuntime({"arena_policy": domain})
            adapter = NativeCascadePolicy(runtime, controller, actuation_owner="arena_policy")
            assert isinstance(adapter, PolicyBase)
            domain.bind_adapter(adapter)
            catalog = runtime.execute("list_resources", {})
            with torch.inference_mode():
                result = adapter.execute_skill("arena_policy.run_policy_steps", {"steps": 20})
            task_done = runtime.execute("task_done", {"success": True, "summary": "foundation is not task admission"})
            last = observe(epoch)
            denied = adapter.execute_skill("arena_policy.run_policy_steps", {"steps": 1})
            unchanged_after_denial = last == observe(epoch)
            samples = domain.samples
            expected = 20 if cancel_before is None else cancel_before
            assert domain.completed_steps == expected, result
            assert len(samples) == expected + 1
            assert result["ok"] is (cancel_before is None), result
            assert not task_done["success"] and not denied["ok"] and unchanged_after_denial
            assert all(b["physics_steps"] - a["physics_steps"] == base.cfg.decimation
                       for a, b in zip(samples, samples[1:]))
            assert all(b["control_step"] - a["control_step"] == 1
                       for a, b in zip(samples, samples[1:]))
            cube_z = [sample["cube_pose_w"][0][2] for sample in samples]
            gravity_response = min(cube_z) < cube_z[0] - .01
            assert gravity_response, "no independently observed cube gravity response"
            close = runtime.close()
            assert close["ok"], close
            entry = {"label": label, "epoch": epoch, "result": result, "task_done": task_done,
                     "catalog": catalog, "stop_receipts": stop_receipts, "denied_repeat": denied,
                     "unchanged_after_denial": unchanged_after_denial, "runtime_close": close,
                     "completed_steps": expected, "actions_produced": domain.actions_produced,
                     "physics_steps": samples[-1]["physics_steps"] - samples[0]["physics_steps"],
                     "cube_z_initial_m": cube_z[0], "cube_z_minimum_m": min(cube_z),
                     "cube_gravity_response": gravity_response,
                     "joint_names": list(robot.joint_names), "body_names": list(robot.body_names),
                     "frames": frames}
            receipt["episodes"].append(entry)
            runtime = None
            write()
        receipt["preflight_pass"] = True
        code = 0
    except BaseException as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipt["traceback"] = traceback.format_exc()
        print(receipt["traceback"], flush=True)
    finally:
        write()
        try:
            if runtime is not None:
                receipt["failure_runtime_close"] = runtime.close()
            if env is not None:
                env.close()
            receipt["environment_closed"] = True
            final_sources = source_inventory(roots)
            (out / "source-after.json").write_text(json.dumps(final_sources, indent=2) + "\n")
            receipt["source_unchanged"] = sources == final_sources
            if not receipt["source_unchanged"]:
                receipt["preflight_pass"] = False
                code = 1
            receipt["app_close_requested"] = app is not None
            receipt["intended_exit_code"] = code
            write()
            if app is not None:
                app.close(exit_code=code)  # SDK exits the process; outer supervisor verifies closure.
        except BaseException as exc:
            receipt["close_error"] = f"{type(exc).__name__}: {exc}"
            receipt["preflight_pass"] = False
            code = 1
            write()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
