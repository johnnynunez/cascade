"""Real, optional VAB/Panda interface preflight; never a manipulation score.

Run in an isolated process with the pinned upstream checkout on PYTHONPATH.
Only normal ``move_relative`` is exposed. The actor receives public RGB and
proprioception; the separate witness reads solved MuJoCo state after each step.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import uuid

import numpy as np

from cascade.agent.effects import Postcondition, annotate_result
from cascade.agent.trace import TraceLogger
from cascade.config import Cfg, load_demo_config
from cascade.eval.trials import Artifact, EpisodeBinding, IndependentEvidence, digest_json, verify_episode
from cascade.eval.vab import inspect_trial, open_native_environment, run_trial
from cascade.memory.beliefs import BeliefStore
from cascade.memory.episodic import EpisodicMemory
from cascade.perception.depth_provider import DepthProvider
from cascade.perception.grounding import Extrinsics
from cascade.robotics.contracts import ResourceDescriptor, ToolDescriptor
from cascade.robotics.runtime import RobotRuntime
from cascade.safety.harness import SafeArm, SafetyHarness, SafetyLimits
from cascade.skills.runtime import SkillRuntime, TOOL_SPECS
from cascade.types import MotionHalted
from benchmark.libero.backend import LiberoArm, LiberoCamera
from benchmark.libero.kinematics_mj import MujocoKinematics


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_inventory(checkout):
    repo = Path(__file__).resolve().parents[2]
    groups = {"cascade": (repo, [*repo.joinpath("src").rglob("*.py"), *repo.joinpath("configs").rglob("*.yaml"),
                                *repo.joinpath("benchmark/libero").glob("*.py"), *repo.joinpath("benchmark/vab").glob("*.py")]),
              "vab": (checkout, list(checkout.joinpath("libero/libero").rglob("*.py")))}
    for package in ("robosuite", "mujoco"):
        root = Path(importlib.metadata.distribution(package).locate_file(package))
        groups[package] = (root, [*root.rglob("*.py"), *root.rglob("*.so*")])
    return {f"{name}/{path.relative_to(root)}": hashlib.sha256(path.read_bytes()).hexdigest()
            for name, (root, paths) in groups.items() for path in sorted(set(paths)) if path.is_file()}


def verify_move(before, after, direction, distance_m, tolerance_m=0.005):
    """Independent world-site displacement, without actuator/FK success credit."""
    if (before["epoch"] != after["epoch"] or before["model_identity_sha256"] != after["model_identity_sha256"]
            or after["step"] <= before["step"] or after["sim_time_s"] <= before["sim_time_s"]):
        return Postcondition("move_relative", "tcp_displacement", "unverified", "witness did not advance in the same episode", "mujoco_readonly")
    axes = {"up": [0, 0, 1], "down": [0, 0, -1]}
    axis = np.asarray(axes[direction], float)
    delta = np.asarray(after["tcp_xyz_m"]) - before["tcp_xyz_m"]
    error = float(np.linalg.norm(delta - axis * distance_m))
    rotation = np.asarray(after["tcp_rotation"]) @ np.asarray(before["tcp_rotation"]).T
    angle = float(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
    ok = error <= tolerance_m and angle <= 0.05
    return Postcondition("move_relative", "tcp_displacement", "confirmed" if ok else "refuted",
                         "completed-solve TCP displacement and orientation", "mujoco_readonly",
                         {"delta_m": delta.tolist(), "error_m": error, "orientation_error_rad": angle,
                          "tolerance_m": tolerance_m, "first_step": before["step"], "last_step": after["step"]})


class Witness:
    """No commands, FK, qpos writes, or simulator advancement in this reader."""
    def __init__(self, out, source_sha256):
        self.out = out
        self.source_sha256 = source_sha256
        self.epoch = uuid.uuid4().hex
        self.rows = []
        self.model_identity = None

    def capture(self, env, step):
        data, model = env.sim.data, env.sim.model
        site = model.site_name2id("gripper0_grip_site")
        row = {"epoch": self.epoch, "model_identity_sha256": self.model_identity,
               "step": step, "control_step": step, "sim_time_s": float(data.time),
               "solver_step": round(float(data.time) / float(model.opt.timestep)),
               "solver_dt_s": float(model.opt.timestep),
               "qpos": np.asarray(data.qpos).tolist(), "qvel": np.asarray(data.qvel).tolist(),
               "tcp_xyz_m": np.asarray(data.site_xpos[site]).tolist(),
               "tcp_rotation": np.asarray(data.site_xmat[site]).reshape(3, 3).tolist(),
               "object_positions_m": {name: np.asarray(data.body_xpos[index]).tolist()
                                      for name, index in env._obj_body_id.items()},
               "contacts": int(data.ncon)}
        row["snapshot_sha256"] = digest_json(row)
        self.rows.append(row)
        with (self.out / "physics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        return row


class ObservedEnvironment:
    """Advance the real environment once, then read it before returning obs."""
    def __init__(self, native, witness, trial, out, *, record_placement=False):
        import mujoco
        self.native, self.witness, self.steps = native, witness, 0
        self.trial, self.out = trial, out
        self._mujoco = mujoco
        self.after_step = None
        self.record_placement, self.placement = record_placement, None

    def __getattr__(self, name):
        return getattr(self.native, name)

    def reset(self, **kwargs):
        observation = self.native.reset(**kwargs)
        model_path = self.out / "effective-model.mjb"
        self._mujoco.mj_saveModel(self.native.sim.model._model, str(model_path), None)
        recipe = {"vab_revision": self.trial.source_revision,
                  "runtime_sources_sha256": self.witness.source_sha256,
                  "configuration_sha256": self.trial.configuration_sha256,
                  "compiled_model_sha256": Artifact.read(model_path).sha256,
                  "versions": {key: importlib.metadata.version(key) for key in ("mujoco", "robosuite", "numpy")},
                  # `sim` is the runtime handle, already bound by the compiled
                  # model and episode; every other effective controller field
                  # is JSON data (including joint mapping/actuator limits).
                  "controller": {k: v for k, v in self.native.robots[0].controller_config.items() if k != "sim"},
                  "seed": 0, "renderer": "osmesa", "control_freq": self.trial.control_freq}
        from OpenGL import GL
        recipe["gl_renderer"] = GL.glGetString(GL.GL_RENDERER).decode()
        if "llvmpipe" not in recipe["gl_renderer"].lower():
            raise RuntimeError("CPU preflight did not obtain the admitted software renderer")
        recipe = json.loads(json.dumps(recipe, default=lambda x: np.asarray(x).tolist()))
        self.witness.model_identity = digest_json(recipe)
        write_json(self.out / "model-identity.json", {"sha256": self.witness.model_identity, "recipe": recipe})
        self.witness.capture(self.native, 0)
        self._frame(observation)
        if self.record_placement:
            import yaml
            from benchmark.vab.placement_witness import PlacementRecorder
            task = yaml.safe_load(self.trial.task_path.read_text())
            args = task["success"]["args"]
            self.placement = PlacementRecorder(self.native, self.out/"placement",
                model_identity_sha256=self.witness.model_identity, epoch=self.witness.epoch,
                object_name=args["obj"], support_name=args["container"],
                robot_root_body=self.native.robots[0].robot_model.root_body)
        return observation

    def _frame(self, obs):
        # Preserve actual public camera bytes without any oracle overlays.
        for name, rgb in obs["images"].items():
            if not name.endswith("_depth"):
                from PIL import Image
                Image.fromarray(np.asarray(rgb)[::-1]).save(self.out / f"{self.steps:04d}-{name}.png")

    def step(self, action):
        out = self.native.step(action)
        self.steps += 1
        self.witness.capture(self.native, self.steps)
        self._frame(out[0])
        if self.after_step is not None:
            self.after_step(self.steps)
        return out

    def close(self):
        placement = None
        try:
            if self.placement is not None:
                placement = self.placement.close()
        finally:
            self.native.close()
        write_json(self.out / "environment-close.json", {"ok": True, "steps": self.steps,
                   "placement": placement,
                   "semantics": "native VAB close returned; standalone runner subsequently exits"})
        if placement is not None and not placement["ok"]:
            raise RuntimeError("placement recorder did not close completely")


class PandaArm(LiberoArm):
    """OSC conversion is inherited; pacing/approval uses the actual sim clock."""
    settle_tcp_tol_m = 0.002

    def stream_to(self, q_target, duration_s, rate_hz=50.0, approve=None,
                  settle_tol=None, settle_timeout_s=12.0):
        from cascade.control.arm_base import min_jerk
        q_start = self.get_state().q.copy()
        q_target = np.asarray(q_target, float)
        goal = self.kin.fk(q_target)
        dt = 1.0 / self.env.trial.control_freq
        count = max(2, math.ceil(duration_s / dt))
        for index in range(1, count + math.ceil(settle_timeout_s / dt) + 1):
            if self._stopped:
                raise MotionHalted("VAB stopped")
            q_now = self.get_state().q.copy()
            current = self.kin.fk(q_now)
            angle = np.arccos(np.clip((np.trace(goal[:3, :3] @ current[:3, :3].T) - 1) / 2, -1, 1))
            if index > count and np.linalg.norm(current[:3, 3] - goal[:3, 3]) <= self.settle_tcp_tol_m and angle <= 0.05:
                return True
            target = q_start + (q_target - q_start) * min_jerk(min(index / count, 1.0))
            if approve is not None:
                approve(q_now, target, dt)
            old_t = float(self.env.sim.data.time)
            self.send_joint_target(target)
            elapsed = float(self.env.sim.data.time) - old_t
            if not np.isclose(elapsed, dt, rtol=1e-6, atol=1e-9):
                raise RuntimeError("VAB physical step clock changed")
            if approve is not None:
                approve(q_now, self.get_state().q, elapsed)
        return False


class PandaKinematics(MujocoKinematics):
    """Recompute Jacobian prerequisites in scratch data, never live physics."""
    def _set_q(self, q):
        for adr, value in zip(self.qpos_adr, np.asarray(q, float)):
            self.data.qpos[adr] = value
        # mj_kinematics alone leaves subtree COM data used by mj_jacSite stale.
        self._mj.mj_forward(self.model, self.data)

    def ik(self, T_target, q_init, **kwargs):
        result = super().ik(T_target, q_init, **kwargs)
        actual = self.fk(result.q)
        angle = np.arccos(np.clip((np.trace(T_target[:3, :3] @ actual[:3, :3].T) - 1) / 2, -1, 1))
        if angle >= 0.05:
            result.success = False
        return result


class UnavailableDetector:
    def detect(self, *_args, **_kwargs):
        raise RuntimeError("object perception is not configured for this interface preflight")


class PublicCamera(LiberoCamera):
    """RGB-only VAB camera; preserve the calibrated horizontal pixel axis."""
    @property
    def has_depth(self):
        return False

    def publish(self, obs):
        rgb = np.asarray(obs[f"{self.cam}_image"])
        if rgb.shape != (128, 128, 3) or rgb.dtype != np.uint8:
            raise ValueError("unexpected VAB preflight image geometry")
        # MuJoCo readback is bottom-up RGB; Frame uses top-down BGR. Unlike
        # the historical LIBERO capture, do not also reverse horizontal X.
        with self._lock:
            self._latest = (np.ascontiguousarray(rgb[::-1, :, ::-1]), None)


class MoveDomain:
    domain_id = "manipulation"
    motion_skills = frozenset({"move_relative"})

    def __init__(self, runtime, witness):
        self.runtime, self.witness = runtime, witness
        resource = ResourceDescriptor("manipulation/panda", "arm", "vab_panda", ("tcp_relative",),
                                      "vab/osc_pose", "cascade_safe_arm", admission="unvalidated")
        self.resources = (resource,)
        spec = next(t for t in TOOL_SPECS if t["name"] == "move_relative")
        # Narrow the exposed existing skill to the reviewed empty-space probe.
        params = {"type": "object", "properties": {"direction": {"enum": ["up", "down"]},
                  "distance_m": {"type": "number", "minimum": 0.01, "maximum": 0.04}},
                  "required": ["direction", "distance_m"], "additionalProperties": False}
        self.tool_descriptors = (ToolDescriptor("manipulation.move_relative", spec["description"], params,
                                               self.domain_id, "move_relative", "motion",
                                               (resource.resource_id,), (resource.resource_id,)),)

    def execute(self, name, args):
        before = self.witness.rows[-1]
        result = self.runtime.execute(name, args)
        pc = verify_move(before, self.witness.rows[-1], args["direction"], args["distance_m"])
        return annotate_result(result, pc)

    def begin_task(self):
        pass

    def stop(self):
        self.runtime.arm.stop()
        return {"ok": True, "physical_stop_verified": False, "semantics": "no further synchronous environment steps"}

    def reset_stop(self):
        return {"ok": False, "error": "new episode required"}

    def close(self):
        self.runtime.arm.disconnect()
        self.runtime.camera.close()
        return {"ok": True}


def runtime_factory(witness, out, cancel_after_step=None):
    def build(view, trial, trial_id):
        import mujoco
        raw_model = view.sim.model._model
        # FK/IK use private scratch data; never temporarily write physics qpos.
        kin = PandaKinematics(raw_model, mujoco.MjData(raw_model))
        if kin.n != 7 or kin.site_id < 0:
            raise ValueError("Panda joint/site mapping is incomplete")
        cfg = load_demo_config(arm="libero_panda", cameras=["mock"], llm="mock")
        cfg._data["occupancy"] = {"enabled": False}
        cfg._data["workspace_filter"] = {"enabled": False}
        cfg.grasp._data["backend"] = "obb"
        cfg.grasp._data["memory_path"] = str(out / "grasp-memory.json")
        cfg._data.setdefault("memory", {})["envelope_path"] = str(out / "envelope.json")
        cfg._data["camera"] = {"type": "vab_native", "static_scene": False}
        # This preflight deliberately admits only the pinned floor scene.
        import yaml
        task = yaml.safe_load(trial.task_path.read_text())
        if task["arena"] != {"name": "floor"}:
            raise ValueError("preflight geometry is reviewed only for the VAB floor scene")
        if task["camera_width"] != 128 or task["camera_height"] != 128 or trial.camera_depth:
            raise ValueError("preflight requires the pinned 128x128 RGB-only capture")
        limits = SafetyLimits(np.array([-0.65, -0.6, 0.0]), np.array([0.65, 0.6, 0.8]),
                              table_z=0.0, table_clearance=0.02, max_joint_vel=1.2, joint_margin=0.025)
        harness = SafetyHarness(limits, kin)
        raw = PandaArm(view, kin)
        if raw.controller != "OSC_POSE":
            raise ValueError("Panda preflight requires the actual OSC_POSE controller")
        raw.last_obs = view._get_observations()
        raw.connect()
        camera = PublicCamera(view, res=128)
        camera.open()
        camera.publish(raw.last_obs)
        raw.camera = camera
        runtime = SkillRuntime(camera, DepthProvider(Cfg({})), UnavailableDetector(),
                               Extrinsics(T=camera._T), kin, SafeArm(raw, harness), EpisodicMemory(),
                               BeliefStore(), TraceLogger(out / "skills"), cfg)
        domain = MoveDomain(runtime, witness)
        composed = RobotRuntime({"manipulation": domain}, trace=TraceLogger(out / "runtime"))
        if cancel_after_step is not None:
            def cancel(step):
                if step == cancel_after_step:
                    write_json(out / "inflight-stop.json", {"control_step": step, "result": composed.stop()})
            view.native_env.after_step = cancel
        write_json(out / "runtime-contract.json", {"trial_id": trial_id, "resources": composed.resources.as_dict(),
                  "safety": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in asdict(limits).items()},
                  "depth": "absent; no synthesis", "perception": "not configured; no object tools exposed",
                  "scratch_kinematics": True, "task": "interface preflight only"})
        def cleanup():
            # There is no lease or independent clock producer in synchronous
            # VAB. Verify software cancellation, never call this a physical
            # braking proof under an independently advancing world.
            stopped = composed.stop()
            before_t = float(view.sim.data.time)
            before_q = np.asarray(view.sim.data.qpos).copy()
            denied = composed.execute("manipulation.move_relative", {"direction": "up", "distance_m": .01})
            unchanged = before_t == float(view.sim.data.time) and np.array_equal(before_q, view.sim.data.qpos)
            write_json(out / "stop-probe.json", {"stop": stopped, "denied": denied, "state_unchanged": unchanged,
                       "control_step": view.steps, "sim_time_s": before_t,
                       "physical_braking_verified": False, "lease": "not applicable: synchronous local step owner"})
            closed = composed.close()
            write_json(out / "runtime-close.json", closed)
            if stopped.get("ok") is not True or denied.get("ok") is not False or not unchanged:
                raise RuntimeError("stop probe failed")
            return closed
        return composed, cleanup
    return build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--task", default="tasks/libero_object_all_variance/pick_up_the_alphabet_soup_and_place_it_in_the_basket.yaml")
    parser.add_argument("--init-index", type=int, default=0)
    parser.add_argument("--cancel-after-step", type=int)
    parser.add_argument("--record-placement", action="store_true",
                        help="Record every solved contact for independent placement analysis; does not enable grasping")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for key, name in (("CASCADE_GRASP_MEMORY_PATH", "grasp-memory.json"), ("CASCADE_ENVELOPE_PATH", "envelope.json"),
                      ("CASCADE_BELIEFS_PATH", "beliefs.json"), ("CASCADE_EXPERIENCE_PATH", "experience.json")):
        os.environ[key] = str(out / name)
    if os.environ.get("MUJOCO_GL") != "osmesa" or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("this recipe requires explicit CPU/offscreen environment")
    if args.cancel_after_step is not None and not 1 <= args.cancel_after_step <= 10:
        raise ValueError("cancellation preflight requires a step in [1,10]")
    sources = source_inventory(args.checkout.resolve())
    write_json(out / "sources-before.json", sources)
    write_json(out / "launch.json", {"argv": sys.argv, "executable": sys.executable,
               "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "MUJOCO_GL", "LIBGL_ALWAYS_SOFTWARE", "LD_LIBRARY_PATH", "LIBERO_CONFIG_PATH", "PYTHONPATH")},
               "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}})
    np.random.seed(0)
    trial = inspect_trial(args.checkout, args.task, init_index=args.init_index)
    witness = Witness(out, digest_json(sources))
    def binding_reader(trial_id, trial, _view):
        first, last = witness.rows[0], witness.rows[-1]
        return EpisodeBinding(trial_id, trial.configuration_sha256, witness.model_identity,
                              witness.epoch, first["step"], last["step"], first["sim_time_s"], last["sim_time_s"],
                              first["snapshot_sha256"], last["snapshot_sha256"])
    episode = run_trial(trial, runtime_factory=runtime_factory(witness, out, args.cancel_after_step),
                        skill_plan=[("manipulation.move_relative", {"direction": "up", "distance_m": 0.04}),
                                    ("manipulation.move_relative", {"direction": "down", "distance_m": 0.04})],
                        output_path=out / "episode.json", binding_reader=binding_reader,
                        env_factory=lambda t: ObservedEnvironment(open_native_environment(t), witness, t, out,
                                                                 record_placement=args.record_placement))
    tools = [json.loads(row) for row in (out / "runtime/trace.jsonl").read_text().splitlines()
             if json.loads(row)["skill"] == "manipulation.move_relative" and json.loads(row)["args"].get("distance_m") == .04]
    moves_ok = len(tools) == 2 and all(r["result"].get("postcondition", {}).get("status") == "confirmed" and r["result"].get("ok") is True for r in tools)
    # Containment is intentionally not claimed by this movement-only preflight.
    evidence = IndependentEvidence(episode.binding, episode.artifact.sha256, episode.record_sha256,
                                   "mujoco_readonly", Artifact.read(out / "physics.jsonl"),
                                   {"tcp_motion": "confirmed" if moves_ok else "refuted", "released_supported_placement": "unverified"})
    verdict = verify_episode(episode, required_checks=frozenset({"tcp_motion", "released_supported_placement"}), independent_verifier=lambda _: evidence)
    sources_after = source_inventory(args.checkout.resolve())
    write_json(out / "sources-after.json", sources_after)
    cancelled = (args.cancel_after_step is not None and len(tools) == 1
                 and tools[0]["result"].get("ok") is False and witness.rows[-1]["step"] == args.cancel_after_step
                 and (out / "inflight-stop.json").exists())
    summary = {"native_interface_preflight_pass": moves_ok, "benchmark_verdict": asdict(verdict),
               "native_cancellation_preflight_pass": cancelled, "source_unchanged": sources == sources_after,
               "benchmark_success": episode.benchmark_success, "binding": asdict(episode.binding),
               "steps": witness.rows[-1]["step"], "argv": sys.argv,
               "scope": "real VAB + ordinary CASCADE move_relative; no grasp or benchmark score"}
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if (moves_ok or cancelled) and sources == sources_after else 1


if __name__ == "__main__":
    raise SystemExit(main())
