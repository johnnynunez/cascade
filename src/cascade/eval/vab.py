"""Pinned VAB trials and an optional CASCADE-runtime execution boundary.

The checked VAB API is ``load_task().make_env()``, ``reset(init_index)`` and
the four-value robosuite ``step``. Only single-Panda, non-mutating containment
tasks are admitted here. Native execution requires a reviewed runtime factory;
the historical LIBERO builder's table geometry is not assumed to fit VAB.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Callable, Mapping, Sequence
import uuid

import numpy as np
import yaml

from .trials import Artifact, EpisodeBinding, ExternalEpisode, digest_json

VAB_REVISION = "edcc4bf005446839c0c6f43f8a3bf416702af030"


@dataclass(frozen=True)
class VabTrial:
    checkout: Path
    task_path: Path
    task_sha256: str
    source_revision: str
    task_id: str
    init_index: int
    horizon: int
    control_freq: int
    camera_depth: bool
    configuration_sha256: str

    def new_trial_id(self) -> str:
        # Repeating an initial state is still a new physical episode.
        return f"vab:{self.task_id}:{self.init_index}:{uuid.uuid4().hex}"


def inspect_trial(checkout: str | Path, task_path: str | Path, *, init_index: int = 0,
                  source_revision: str = VAB_REVISION, control_freq: int = 20) -> VabTrial:
    """Validate task data without importing or executing upstream Python."""
    checkout = Path(checkout).resolve()
    task_path = Path(task_path)
    task_path = (checkout / task_path).resolve() if not task_path.is_absolute() else task_path.resolve()
    if not task_path.is_relative_to(checkout / "tasks"):
        raise ValueError("task must be inside the pinned checkout's tasks directory")
    head = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                          capture_output=True, check=True, text=True).stdout.strip()
    if head != source_revision:
        raise ValueError("VAB checkout does not match the reviewed source revision")
    body = task_path.read_bytes()
    task_sha = hashlib.sha256(body).hexdigest()
    tracked = subprocess.run(["git", "-C", str(checkout), "show",
                              f"{head}:{task_path.relative_to(checkout).as_posix()}"],
                             capture_output=True, check=True).stdout
    if body != tracked:
        raise ValueError("VAB task differs from its pinned source")
    raw = yaml.safe_load(body)
    if not isinstance(raw, dict) or not isinstance(raw.get("id"), str) or not raw["id"]:
        raise ValueError("VAB task id is required")
    if raw.get("robots") or raw.get("robot") != {"name": "panda", "controller": "OSC_POSE"}:
        raise ValueError("adapter admits only single Panda OSC_POSE tasks")
    if raw.get("success", {}).get("predicate") != "contained_in":
        raise ValueError("adapter excludes mutating packing and partial multistage predicates")
    inits = raw.get("inits")
    if not isinstance(inits, list) or type(init_index) is not int or not 0 <= init_index < len(inits):
        raise ValueError("init_index must select an existing initial state")
    objects = raw.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("VAB object declarations are required")
    ids = [obj.get("id") for obj in objects if isinstance(obj, dict)]
    if len(ids) != len(objects) or any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("VAB object identities must be unique")
    init = inits[init_index]
    if not isinstance(init, dict) or set(init) != set(ids):
        raise ValueError("selected initial state must define every object exactly once")
    for pose in init.values():
        if (not isinstance(pose, list) or len(pose) != 7
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in pose)
                or not math.isclose(sum(v * v for v in pose[3:]), 1.0, rel_tol=1e-4, abs_tol=1e-4)):
            raise ValueError("poses require finite xyz and normalized xyzw quaternion")
    horizon = raw.get("horizon")
    if type(horizon) is not int or horizon < 1 or type(control_freq) is not int or control_freq < 1:
        raise ValueError("horizon and control_freq must be positive integers")
    if type(raw.get("camera_depth", False)) is not bool:
        raise ValueError("camera_depth must be boolean")
    config_sha = digest_json({"source_revision": head, "task_sha256": task_sha,
                             "init_index": init_index, "control_freq": control_freq,
                             "ignore_done": False, "horizon": horizon})
    return VabTrial(checkout, task_path, task_sha, head, raw["id"], init_index,
                    horizon, control_freq, raw.get("camera_depth", False), config_sha)


def open_native_environment(trial: VabTrial):
    """Use the real VAB loader, with heavy imports restricted to this call."""
    checked = inspect_trial(trial.checkout, trial.task_path, init_index=trial.init_index,
                            source_revision=trial.source_revision, control_freq=trial.control_freq)
    if checked != trial:
        raise ValueError("trial configuration changed after inspection")
    # A pin alone does not attest edited Python/assets in a checkout. Native
    # execution needs a complete, unmodified source subtree, unlike inspection.
    subprocess.run(["git", "-C", str(trial.checkout), "diff", "--quiet", "HEAD", "--", "libero"],
                   check=True)
    config_dir = os.environ.get("LIBERO_CONFIG_PATH")
    if not config_dir or not Path(config_dir).is_absolute():
        raise ValueError("native VAB requires an explicit private LIBERO_CONFIG_PATH before imports")
    config_file = Path(config_dir) / "config.yaml"
    config = yaml.safe_load(config_file.read_text())
    if not isinstance(config, dict) or Path(config.get("assets", "")).resolve() != trial.checkout / "libero/libero/assets":
        raise ValueError("private LIBERO config must point to the pinned VAB assets")
    import libero.vab as vab
    expected = trial.checkout / "libero" / "libero" / "vab"
    if not Path(vab.__file__).resolve().is_relative_to(expected):
        raise ValueError("imported libero.vab comes from a different checkout")
    task = vab.load_task(trial.task_path)
    return task.make_env(control_freq=trial.control_freq, ignore_done=False)


def flatten_observation(observation: dict) -> dict:
    """Project VAB's public RGB/proprio fields to the existing LIBERO backend.

No object poses, segmentation or other privileged observables are obtained.
Depth is forwarded only if the task explicitly supplies it; never synthesized.
"""
    if not isinstance(observation, dict) or set(observation) != {"images", "proprio"}:
        raise ValueError("expected VAB's strict images/proprio observation")
    images, proprio = observation["images"], observation["proprio"]
    if not isinstance(images, dict) or not isinstance(proprio, dict):
        raise ValueError("invalid VAB observation dictionaries")
    expected = {"joint_pos": (7,), "joint_vel": (7,), "eef_pos": (3,),
                "eef_quat": (4,), "gripper_qpos": (2,)}
    if set(proprio) != set(expected):
        raise ValueError("single Panda proprioception is incomplete or unexpected")
    result = {}
    for key, shape in expected.items():
        arr = np.asarray(proprio[key])
        if arr.shape != shape or not np.isfinite(arr).all():
            raise ValueError(f"invalid VAB proprioception: {key}")
        result[f"robot0_{key}"] = arr.copy()
    if not images:
        raise ValueError("VAB camera observations are absent")
    for key, value in images.items():
        if not isinstance(key, str) or not key:
            raise ValueError("invalid VAB camera name")
        arr = np.asarray(value)
        if key.endswith("_depth"):
            if arr.ndim not in (2, 3) or not np.isfinite(arr).all():
                raise ValueError("invalid camera depth")
            result[key] = arr.copy()
        else:
            if arr.ndim != 3 or arr.shape[-1] != 3 or arr.dtype != np.uint8:
                raise ValueError("VAB RGB images must be HxWx3 uint8")
            result[f"{key}_image"] = arr.copy()
    return result


class VabLiberoView:
    """Bounded adapter consumed by existing LiberoArm/LiberoCamera backends.

The runtime factory owns kinematics, correct scene geometry, detector setup and
SafetyHarness. Merely constructing this view does not create those contracts.
"""

    def __init__(self, native_env, trial: VabTrial):
        if native_env.action_dim != 7:
            raise ValueError("single Panda OSC_POSE requires seven actions")
        self.native_env = native_env
        self.trial = trial
        self.steps = 0
        self.benchmark_success = None
        self._last = None
        self._terminated = False

    @property
    def env(self):
        return self

    @property
    def sim(self):
        return self.native_env.sim

    @property
    def robots(self):
        return self.native_env.robots

    def reset(self):
        if self._last is not None:
            raise RuntimeError("create a fresh view and runtime for each episode")
        self._last = flatten_observation(self.native_env.reset(init_index=self.trial.init_index))
        return self._last

    def _get_observations(self, **_kwargs):
        if self._last is None:
            raise RuntimeError("VAB must reset before runtime construction")
        return {key: value.copy() for key, value in self._last.items()}

    def step(self, action):
        if self._last is None:
            raise RuntimeError("VAB must reset before stepping")
        if self._terminated or self.steps >= self.trial.horizon:
            raise RuntimeError("VAB episode step budget exhausted")
        action = np.asarray(action, dtype=float)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("VAB action requires seven finite values")
        low, high = self.native_env.action_spec
        if np.any(action < low) or np.any(action > high):
            raise ValueError("action exceeds the native controller's limits")
        observation, reward, done, info = self.native_env.step(action)
        self.steps += 1
        if type(info.get("success")) is not bool:
            raise ValueError("native VAB success must be an explicit boolean")
        self.benchmark_success = info["success"]
        self._last = flatten_observation(observation)
        self._terminated = bool(done) or self.steps >= self.trial.horizon
        return self._get_observations(), reward, self._terminated, dict(info)


def run_trial(
    trial: VabTrial,
    *,
    runtime_factory: Callable,
    skill_plan: Sequence[tuple[str, Mapping]],
    output_path: str | Path,
    env_factory: Callable = open_native_environment,
    binding_reader: Callable[[str, VabTrial, VabLiberoView], EpisodeBinding] | None = None,
) -> ExternalEpisode:
    """Execute namespaced CASCADE tools, never a replacement grasp controller.

``runtime_factory(view, trial, trial_id)`` returns ``(runtime, cleanup)``.
It must build a reviewed geometry-specific runtime around the existing LIBERO
backends. Runs belong in an isolated process (GL context and factory ownership).
The optional binding reader samples an independent observer; it is not derived
from ``info.success``. Verifying effects remains ``trials.verify_episode``.
"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the evidence filename before allocating simulator resources.
    with output_path.open("x", encoding="utf-8") as stream:
        stream.write('{"status":"starting"}\n')
    trial_id = trial.new_trial_id()
    record = {"trial_id": trial_id, "configuration_sha256": trial.configuration_sha256,
              "source_revision": trial.source_revision, "task_sha256": trial.task_sha256,
              "task_id": trial.task_id, "init_index": trial.init_index,
              "status": "running", "steps": 0, "benchmark_success": None, "tools": []}
    native = view = runtime = cleanup = None
    binding = None
    primary_error = None
    try:
        native = env_factory(trial)
        view = VabLiberoView(native, trial)
        view.reset()
        runtime, cleanup = runtime_factory(view, trial, trial_id)
        runtime.begin_task()
        for name, args in skill_plan:
            if name not in runtime.tool_descriptors:
                raise ValueError(f"unavailable CASCADE tool: {name}")
            result = runtime.execute(name, dict(args))
            # Full sensor/tool traces belong to the runtime. This result receipt
            # records dispatch status, not images or a fabricated physical proof.
            ok = isinstance(result, dict) and result.get("ok") is True
            record["tools"].append({"name": name, "execution_ok": ok})
            if not ok:
                record["status"] = "tool_failed"
                break
        else:
            record["status"] = "completed"
        if binding_reader is not None:
            binding = binding_reader(trial_id, trial, view)
            if (not isinstance(binding, EpisodeBinding) or binding.trial_id != trial_id
                    or binding.configuration_sha256 != trial.configuration_sha256):
                raise ValueError("observer binding does not match this VAB trial")
            record["physical_binding"] = asdict(binding)
    except BaseException as exc:
        primary_error = exc
        record["status"] = "failed"
        record["error_type"] = type(exc).__name__
        raise
    finally:
        if view is not None:
            record["steps"] = view.steps
            record["benchmark_success"] = view.benchmark_success
        teardown_errors = []
        for finish in (runtime.stop if runtime is not None else None, cleanup,
                       native.close if native is not None else None):
            if finish is not None:
                try:
                    finished = finish()
                    if isinstance(finished, dict) and finished.get("ok") is False:
                        raise RuntimeError("runtime stop or cleanup reported failure")
                except BaseException as exc:
                    teardown_errors.append(exc)
        if teardown_errors:
            record["status"] = "failed"
            record["teardown_errors"] = [type(exc).__name__ for exc in teardown_errors]
        output_path.write_text(json.dumps(record, allow_nan=False, indent=2) + "\n")
        if teardown_errors and primary_error is None:
            raise RuntimeError("VAB runtime teardown failed") from teardown_errors[0]
    artifact = Artifact.read(output_path)
    # An aborted skill plan does not become successful through a stale upstream bit.
    success = record["benchmark_success"] if record["status"] == "completed" else False
    return ExternalEpisode("vab", trial.source_revision, trial_id, digest_json(record),
                           artifact, success, binding)
