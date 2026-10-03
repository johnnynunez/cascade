"""Explicit composition builders; legacy domain controllers remain unchanged."""
from __future__ import annotations

import copy
import os
from pathlib import Path

from ..config import Cfg
from ..robotics.contracts import ResourceDescriptor, ToolDescriptor
from ..robotics.resources import ResourceCatalog


_GLOBAL = {"emergency_stop", "reset_stop", "task_done"}
_GRIPPER = {"grasp_object", "pick_and_place", "place_at", "place_on_object", "open_gripper",
            "close_gripper", "grasp_at_pixel", "sort_by_color", "handover", "throw", "turn_screw"}


def _controller(profile, domain, name):
    kind = profile["type"]
    if kind == "mock":
        return f"mock:{domain}:{name}"
    if kind == "isaac":
        host = profile.get("bridge_host", "127.0.0.1")
        host = "loopback" if host in {"127.0.0.1", "localhost", "::1"} else host
        return f"isaac:{host}:{profile.get('bridge_port', 8611)}"
    if kind == "unitree_arm":
        # The existing driver uses rclpy's process context, not a profile
        # ros_domain_id. Relative and absolute root topics are aliases.
        return f"ros:{os.environ.get('ROS_DOMAIN_ID', '0')}:/{str(profile.get('unitree_cmd_topic', 'rt/arm_sdk')).lstrip('/')}"
    if kind == "mujoco":
        # Conservatively reserve the input scene even when the backend will
        # derive a prop-specific scene path. Discovery must not generate it.
        return f"mujoco:{Path(profile['mjcf']).resolve()}"
    if kind == "so101":
        return f"serial:{Path(profile.get('port', '/dev/ttyACM0')).resolve()}"
    if kind in {"rebot_rs", "rebot_rs_mb"}:
        # The SDK version resolves CAN from an external hardware YAML.
        # Until that source has an admission adapter reserve CAN broadly;
        # profile-only channel guesses could permit two physical writers.
        return "can:exclusive"
    if kind == "ros2":
        default = ("joint_trajectory_controller/joint_trajectory" if profile.get("ros_command", "trajectory") == "trajectory"
                   else "forward_position_controller/commands")
        topic = str(profile.get("ros_command_topic", default))
        if not topic.startswith("/"):
            topic = "/" + "/".join(v for v in (str(profile.get("ros_namespace", "")).strip("/"), topic) if v)
        return f"ros:{os.environ.get('ROS_DOMAIN_ID', '0')}:{topic}"
    raise ValueError(f"composition has no controller ownership mapping for {kind!r}")


def _describe_domain(domain_id, profile, *, embodiment=None, sensor_domains=None):
    """Return a domain's static resources/tools without constructing actuators."""
    kind = profile["kind"]
    if kind == "spatial":
        from ..spatial.domain import build_spatial_domain
        spatial = build_spatial_domain(domain_id, profile, sensor_domains=sensor_domains)
        return DomainAdapter(domain_id, profile, tuple(spatial.resources), spatial.tool_specs,
                             frozenset(), runtime=spatial,
                             required_resources=getattr(spatial, "required_resources", ()))
    if kind == "sensors":
        from ..sensing.domain import build_sensor_domain
        sensor = build_sensor_domain(domain_id, profile, embodiment=embodiment)
        return DomainAdapter(domain_id, profile, tuple(sensor.resources), sensor.tool_specs,
                             frozenset(), runtime=sensor)
    cfg = profile["resolved"]
    if kind == "manipulation":
        from ..skills.runtime import TOOL_SPECS, _MOTION_SKILLS
        specs, motions, profiles, resource_kind = copy.deepcopy(TOOL_SPECS), _MOTION_SKILLS, cfg["arms"], "arm"
        if all(p.get("gripper", {}).get("max_width_m", 0) <= 0 for p in profiles):
            specs = [s for s in specs if s["name"] not in _GRIPPER]
    else:
        from ..skills.mobile_runtime import MOTION_SKILLS, tool_specs_for_profiles
        specs, motions, profiles, resource_kind = tool_specs_for_profiles(cfg["bases"]), MOTION_SKILLS, cfg["bases"], "base"
    resources = []
    for index, p in enumerate(profiles):
        name = str(p.get("name", f"{resource_kind}{index}"))
        resource_id = f"{domain_id}/{name}"
        caps = ("joint_motion",) if resource_kind == "arm" else tuple(p["capabilities"])
        if resource_kind == "arm" and p.get("gripper", {}).get("max_width_m", 0) > 0:
            caps += ("gripper",)
        resources.append(ResourceDescriptor(resource_id=resource_id, kind=resource_kind,
            robot_id=p.get("robot_id", profile["robot_id"]), capabilities=caps, controller_id=_controller(p, domain_id, name),
            writer_id=resource_id, synthetic=p["type"] == "mock",
            admission="software_only" if p["type"] == "mock" else p.get("admission", "unvalidated"),
            metadata={"profile": name, "backend": p["type"], "domain": domain_id}))
        if p["type"] == "ros2" and p.get("ros_gripper_topic"):
            grip_profile = {**p, "ros_command_topic": p["ros_gripper_topic"]}
            resources.append(ResourceDescriptor(resource_id=resource_id + "/gripper", kind="gripper",
                robot_id=p.get("robot_id", profile["robot_id"]), capabilities=("gripper",),
                controller_id=_controller(grip_profile, domain_id, name), writer_id=resource_id,
                metadata={"profile": name, "backend": "ros2", "domain": domain_id}))
    return DomainAdapter(domain_id, profile, tuple(resources), specs, motions)


class DomainAdapter:
    def __init__(self, domain_id, profile, resources, specs, motion_skills, *, runtime=None, owner=None,
                 required_resources=()):
        self.domain_id, self.profile, self.resources = domain_id, profile, resources
        self.runtime, self.owner = runtime, owner
        self.motion_skills = frozenset(motion_skills)
        self.tool_specs = [copy.deepcopy(s) for s in specs if s["name"] not in _GLOBAL]
        ids = tuple(r.resource_id for r in resources)
        self.tool_descriptors = tuple(ToolDescriptor(
            name=f"{domain_id}.{s['name']}", description=s["description"], parameters=s["parameters"],
            domain=domain_id, local_name=s["name"],
            effect="stop" if s["name"] in {"stop_navigation", "halt_motion"} else
                   "motion" if s["name"] in self.motion_skills else "read",
            requires=(*ids, *required_resources), writes=ids if s["name"] in self.motion_skills else ()) for s in self.tool_specs)

    def execute(self, name, args):
        return self.runtime.execute(name, args)

    def stop(self):
        if self.profile["kind"] != "manipulation":
            return self.runtime.stop()
        results = {}
        for name, arm in self.runtime.arm_rig.arms.items():
            try:
                arm.stop()  # LazyArm.stop does not materialize a backend
                results[name] = {"ok": True}
            except Exception as exc:
                results[name] = {"ok": False, "error": str(exc)}
        return {"ok": all(v["ok"] for v in results.values()), "arms": results,
                "physical_stop_verified": False}

    def reset_stop(self):
        if self.profile["kind"] != "manipulation":
            return self.runtime.reset_stop()
        for arm in self.runtime.arm_rig:
            arm.harness.reset_estop()
            # All composed arms are explicitly lazy; resume is on LazyArm's
            # passive surface and never creates a motor connection.
            arm.raw.resume()
        return {"ok": True}

    def request_shutdown(self):
        if self.profile["kind"] != "manipulation":
            return self.stop()
        # Mirror the arm CLI's graceful shutdown: cancel existing streams
        # without manufacturing an e-stop that would prohibit safe parking
        # before a torque-off hardware disconnect. Existing e-stops survive.
        for arm in self.runtime.arm_rig:
            arm.harness.halt("composed runtime shutdown")
        return {"ok": True, "halted": True, "physical_stop_verified": False}

    def begin_task(self):
        if self.profile["kind"] == "locomotion":
            self.runtime.begin_task()

    def close(self):
        if self.profile["kind"] == "manipulation":
            from .demo import shutdown_runtime
            return shutdown_runtime(self.runtime, self.owner)
        return self.runtime.close()


def describe_robot(cfg):
    """Validate the entire resource graph before opening any domain."""
    from ..robotics.embodiment import embodiment_metadata
    from ..robotics.joint_coordinates import MULTI_DOF_JOINTS
    body = cfg.as_dict().get("embodiment")
    profiles = cfg.domains.as_dict()
    # Passive providers must exist before resolving explicit observed-spatial
    # references. Profile ordering never selects a different hub or latest data.
    domains = {name: _describe_domain(name, profile, embodiment=body)
               for name, profile in profiles.items() if profile["kind"] == "sensors"}
    sensors = {name: domain.runtime for name, domain in domains.items()}
    domains.update({name: _describe_domain(name, profile, embodiment=body, sensor_domains=sensors)
                    for name, profile in profiles.items() if profile["kind"] != "sensors"})
    domains = {name: domains[name] for name in profiles}
    catalog = ResourceCatalog([r for d in domains.values() for r in d.resources])
    embodiment_metadata(body, catalog)
    actuating = [d for d in domains.values() if d.motion_skills]
    dynamic_structure = body is not None and (body["root_mode"] == "floating" or
        any(joint["type"] in MULTI_DOF_JOINTS for joint in body.get("joints", ())))
    if dynamic_structure and any(
            d.profile["kind"] == "manipulation" and any(not r.synthetic for r in d.resources)
            for d in actuating):
        raise ValueError("floating-root or multi-DoF physical manipulation requires validated dynamic frames and shared control")
    if len(actuating) > 1 and any(not r.synthetic for d in actuating for r in d.resources):
        raise ValueError("mixed physical actuation needs validated shared-frame/control admission; only mixed mock domains are supported")
    return domains


def robot_tool_descriptors(cfg):
    from ..robotics.runtime import _global_tools
    return {t.name: t for t in (*_global_tools(), *(t for d in describe_robot(cfg).values() for t in d.tool_descriptors))}


def build_robot_runtime(cfg, run_dir, **_kwargs):
    from ..agent.trace import TraceLogger
    from ..memory.episodic import EpisodicMemory
    from ..robotics.runtime import RobotRuntime
    domains = describe_robot(cfg)
    built = []
    try:
        for name, domain in domains.items():
            directory = Path(run_dir) / "domains" / name
            directory.mkdir(parents=True, exist_ok=True)
            if domain.profile["kind"] in {"sensors", "spatial"}:
                built.append(domain)
                continue
            domain_cfg = Cfg(copy.deepcopy(domain.profile["resolved"]))
            stores = directory / "stores"
            stores.mkdir(exist_ok=True)
            if domain.profile["kind"] == "manipulation":
                domain_cfg._data.setdefault("grasp", {})["memory_path"] = str(stores / "grasp.json")
                domain_cfg._data.setdefault("memory", {}).update(
                    beliefs_path=str(stores / "beliefs.json"), envelope_path=str(stores / "envelope.json"))
                from .demo import build_runtime
                domain.runtime, domain.owner = build_runtime(domain_cfg, directory, lazy_arm=True, view=False, serve=False)
            else:
                from .mobile_runtime import build_mobile_runtime
                domain.runtime, domain.owner = build_mobile_runtime(domain_cfg, directory)
            built.append(domain)
        runtime = RobotRuntime(domains, cfg=cfg, memory=EpisodicMemory(), trace=TraceLogger(run_dir))
        return runtime, runtime
    except BaseException:
        for domain in reversed(built):
            try:
                domain.close()
            except Exception:
                pass
        raise
