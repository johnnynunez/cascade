"""Optional adapters for the inspected Isaac Lab-Arena result and policy APIs.

No Isaac, torch or policy weights are imported here. A supplied controller owns
action production; CASCADE tools continue through their runtime safety path.
"""
from __future__ import annotations

import json
import hashlib
import threading
from pathlib import Path
from typing import Mapping

from .trials import Artifact, EpisodeBinding, ExternalEpisode, digest_json

ARENA_REVISION = "c8d04e2199b86abbbb301bb22cec0effd83c4e63"


def import_experiment(
    path: str | Path,
    *,
    source_revision: str = ARENA_REVISION,
    bindings: Mapping[str, EpisodeBinding] | None = None,
    bound_artifact_sha256: str | None = None,
) -> tuple[ExternalEpisode, ...]:
    """Read ``ArenaExperimentResult.to_dict()`` without importing the simulator.

The upstream format has no physical epoch/model/snapshot identity. Optional
bindings therefore come from an external observer sidecar bound to this exact
file. A failed run or an empty result cannot become a passing campaign.
"""
    path = Path(path).resolve()
    body = path.read_bytes()
    artifact = Artifact(str(path), hashlib.sha256(body).hexdigest())
    def reject_nonfinite(value):
        raise ValueError(f"nonfinite JSON value: {value}")
    raw = json.loads(body, parse_constant=reject_nonfinite)
    if bindings and bound_artifact_sha256 != artifact.sha256:
        raise ValueError("observer sidecar is not bound to this result artifact")
    runs = raw.get("runs") if isinstance(raw, dict) else None
    if not isinstance(runs, dict) or not runs:
        raise ValueError("Arena result requires nonempty runs")
    episodes = []
    seen = set()
    for run_name, run in runs.items():
        if not isinstance(run_name, str) or not run_name or not isinstance(run, dict):
            raise ValueError("invalid Arena run")
        if run.get("status") != "completed":
            raise ValueError(f"Arena run {run_name!r} did not complete")
        environment = run.get("environment")
        if (not isinstance(environment, dict)
                or any(not isinstance(environment.get(key), str) or not environment[key]
                       for key in ("name", "definition"))
                or not isinstance(run.get("policy_variant"), str) or not run["policy_variant"]):
            raise ValueError("Arena environment and policy identity required")
        rebuilds = run.get("rebuilds")
        if not isinstance(rebuilds, list) or not rebuilds:
            raise ValueError("Arena run has no rebuild records")
        for rebuild in rebuilds:
            if not isinstance(rebuild, dict) or type(rebuild.get("index")) is not int or rebuild["index"] < 0:
                raise ValueError("invalid Arena rebuild index")
            records = rebuild.get("episodes")
            if not isinstance(records, list) or not records:
                raise ValueError("Arena rebuild has no episodes")
            for record in records:
                if not isinstance(record, dict):
                    raise ValueError("invalid Arena episode record")
                for key in ("env_id", "episode_in_env", "episode_length"):
                    if type(record.get(key)) is not int or record[key] < 0:
                        raise ValueError(f"invalid Arena {key}")
                if "success" not in record or (record["success"] is not None and type(record["success"]) is not bool):
                    raise ValueError("Arena success must be bool or null")
                for key in ("job_name", "timestamp"):
                    if not isinstance(record.get(key), str) or not record[key]:
                        raise ValueError(f"Arena {key} required")
                if "seed" not in record or (record["seed"] is not None and type(record["seed"]) is not int):
                    raise ValueError("Arena seed must be integer or null")
                if "language_instruction" not in record or (record["language_instruction"] is not None
                        and not isinstance(record["language_instruction"], str)):
                    raise ValueError("Arena language_instruction must be string or null")
                # JSON encoding avoids delimiter aliasing between run names and indices.
                record_id = json.dumps([run_name, rebuild["index"], record["env_id"],
                                        record["episode_in_env"]], separators=(",", ":"))
                if record_id in seen:
                    raise ValueError("duplicate Arena episode identity")
                seen.add(record_id)
                record_digest = digest_json({"environment": environment, "policy_variant": run["policy_variant"],
                                             "record_id": record_id, "episode": record})
                binding = (bindings or {}).get(record_id)
                if binding is not None and binding.trial_id != record_id:
                    raise ValueError("observer binding has a different trial_id")
                episodes.append(ExternalEpisode("isaaclab-arena", source_revision, record_id,
                                                record_digest, artifact, record["success"], binding))
    if bindings and set(bindings) - seen:
        raise ValueError("observer sidecar contains unknown episodes")
    return tuple(episodes)


class ArenaControllerDomain:
    """Register the external policy in RobotRuntime's priority stop fanout.

This domain exposes no movement tools and makes no resource/safety admission
claim. It owns the supplied policy's lifecycle; the host owns tensor application
to its declared embodiment. Stop/reset methods must be thread-safe and stop must
interrupt a pending get_action. An uncooperative owner cannot acknowledge stop.
"""
    tool_descriptors = ()
    tool_specs = ()
    motion_skills = frozenset()

    def __init__(self, controller, *, command_resources, domain_id="arena_policy"):
        from cascade.robotics.contracts import ResourceDescriptor, identifier
        identifier(domain_id, "Arena action-owner domain")
        self.resources = tuple(command_resources)
        if not self.resources or any(not isinstance(resource, ResourceDescriptor)
                or resource.controller_id is None or resource.writer_id is None for resource in self.resources):
            raise ValueError("Arena requires explicit command resources with controller_id and writer_id")
        if any(not callable(getattr(controller, key, None))
               for key in ("get_action", "reset", "close", "stop", "reset_stop")):
            raise ValueError("external controller requires get_action/reset/close and stop/reset_stop")
        self.domain_id = domain_id
        self.controller = controller
        self._lock = threading.Lock()
        self._active = 0
        self._halted = False
        self._closed = False

    def begin_task(self):
        pass

    def execute(self, name, args):
        return {"ok": False, "error": "Arena action owner exposes no direct movement tools"}

    def get_action(self, env, observation):
        with self._lock:
            if self._halted or self._closed:
                raise RuntimeError("Arena action owner stopped or closed")
            self._active += 1
        try:
            action = self.controller.get_action(env, observation)
            with self._lock:
                if self._halted or self._closed:
                    raise RuntimeError("Arena action invalidated while computing")
            return action
        finally:
            with self._lock:
                self._active -= 1

    def stop(self):
        with self._lock:
            self._halted = True
        ack = self.controller.stop()
        return {"ok": ack is None or ack is True, "physical_stop_verified": False}

    def reset_stop(self):
        with self._lock:
            if self._active or self._closed:
                return {"ok": False, "error": "Arena controller busy or closed"}
        ack = self.controller.reset_stop()
        if ack is True:
            with self._lock:
                self._halted = False
        return {"ok": ack is True}

    def close(self):
        with self._lock:
            if self._closed:
                return {"ok": True}
            self._halted = True
            if self._active:
                return {"ok": False, "pending": True, "error": "Arena action still owns controller"}
        self.controller.close()
        with self._lock:
            self._closed = True
        return {"ok": True}


class ArenaPolicyAdapter:
    """Duck-type Arena's PolicyBase while keeping actuation ownership explicit.

``get_action(env, observation)`` delegates unchanged to an externally supplied
Arena controller. ``execute_skill`` goes through CASCADE's composed runtime;
tool return values are never converted into actuator commands. The controller
must be the selected embodiment's existing low-level policy/safety adapter.
This adapter does not claim that arbitrary action tensors are SafeArm-approved.
"""

    def __init__(self, runtime, controller, *, actuation_owner: str):
        owner = runtime.domains.get(actuation_owner)
        if not isinstance(owner, ArenaControllerDomain) or owner.controller is not controller:
            raise ValueError("exact external controller must be a registered ArenaControllerDomain")
        if type(runtime.cancellation_token) is not int or type(runtime.stopped) is not bool:
            raise ValueError("runtime must expose its cancellation generation and stop latch")
        self.runtime = runtime
        self.controller = controller
        self.actuation_owner = actuation_owner
        self._owner = owner
        self.task_description = None
        self._closed = False
        self._generation = runtime.cancellation_token
        self._halted = runtime.stopped
        self._cleanup_complete = False

    def _halt(self):
        self._halted = True
        if not self.runtime.stopped:
            self.runtime.stop()

    def _check_generation(self):
        if self._halted or self.runtime.stopped or self.runtime.cancellation_token != self._generation:
            self._halt()
            raise RuntimeError("Arena action invalidated by CASCADE stop or generation change")

    def get_action(self, env, observation):
        if self._closed:
            raise RuntimeError("Arena policy adapter is closed")
        self._check_generation()
        try:
            action = self._owner.get_action(env, observation)
        except BaseException:
            self._halt()
            raise
        self._check_generation()
        return action

    def execute_skill(self, name: str, args: dict):
        if self._closed:
            raise RuntimeError("Arena policy adapter is closed")
        if name not in self.runtime.tool_descriptors:
            raise ValueError(f"unavailable CASCADE tool: {name}")
        result = self.runtime.execute(name, args)
        if name == "reset_stop":
            # The registered owner participates in runtime.reset_stop itself.
            # A failed aggregate reset must not leave another owner enabled.
            if result.get("ok") is not True or self.runtime.stopped:
                self._halt()
                return {"ok": False, "error": "external action owner did not acknowledge stop reset"}
            self._generation = self.runtime.cancellation_token
            self._halted = False
        elif self.runtime.stopped or self.runtime.cancellation_token != self._generation:
            self._halt()
        return result

    def reset(self, env_ids=None):
        if self._closed:
            raise RuntimeError("Arena policy adapter is closed")
        self.controller.reset(env_ids)

    def set_task_description(self, task_description):
        self.task_description = task_description
        setter = getattr(self.controller, "set_task_description", None)
        if setter is not None:
            setter(task_description)
        return task_description

    def has_length(self):
        return False

    def length(self):
        return None

    @property
    def is_remote(self):
        return False

    def close(self):
        if not self._cleanup_complete:
            self._closed = True
            try:
                self._halt()
            finally:
                result = self._owner.close()
                if result.get("ok") is not True:
                    raise RuntimeError("Arena controller still owns work; retry close after it returns")
                self._cleanup_complete = True


def make_policy_type(runtime_factory, controller_factory, *, command_resources, actuation_owner="arena_policy"):
    """Build an actual PolicyBase subclass lazily inside an installed Arena.

    controller_factory(config) builds the action owner; runtime_factory(config,
    owner) must register that exact domain in RobotRuntime. Register the policy
    class through Arena's normal registry. Importing CASCADE alone is lightweight.
"""
    from isaaclab_arena.policy.policy_base import PolicyBase

    class CascadeArenaPolicy(ArenaPolicyAdapter, PolicyBase):
        def __init__(self, config):
            PolicyBase.__init__(self, config)
            controller = controller_factory(config)
            owner = None
            runtime = None
            try:
                owner = ArenaControllerDomain(controller, command_resources=command_resources,
                                               domain_id=actuation_owner)
                runtime = runtime_factory(config, owner)
                ArenaPolicyAdapter.__init__(self, runtime, controller, actuation_owner=actuation_owner)
            except BaseException:
                try:
                    if runtime is not None:
                        runtime.stop()
                        runtime.close()
                finally:
                    try:
                        controller.stop() if owner is None else owner.stop()
                    finally:
                        controller.close() if owner is None else owner.close()
                raise

    return CascadeArenaPolicy
