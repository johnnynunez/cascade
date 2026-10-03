"""Read-only spatial domain with an explicitly synthetic, source-bound replay."""
from __future__ import annotations

import json

from ..robotics.contracts import ResourceDescriptor, identifier
from .frames import FrameTree, SpatialStamp, TransformSample, sha256
from .grid import GridSnapshot, plan_route
from .memory import LandmarkObservation, SpatialMemory


class SpatialDomain:
    motion_skills = frozenset()

    def __init__(self, domain_id, robot_id, frames, grid, memory, *, replay_sha256):
        if grid.stamp.measurement_kind != "synthetic" or grid.stamp.context != frames.context:
            raise ValueError("replay domain requires a synthetic grid in its frame context")
        self.domain_id, self.frames, self.grid, self.memory = domain_id, frames, grid, memory
        self.replay_sha256 = replay_sha256
        self.resources = tuple(ResourceDescriptor(
            f"{domain_id}/{kind}", kind, robot_id, capabilities=(capability,),
            synthetic=True, admission="software_only",
            metadata={"source": "synthetic_replay", "replay_sha256": replay_sha256,
                      "map_id": frames.context[0], "epoch": frames.context[1], "clock_id": frames.context[2]})
            for kind, capability in (("frames", "transform_lookup"), ("map", "planar_route"),
                                      ("memory", "spatial_retrieval")))
        string = {"type": "string"}
        scalar = {"type": "number", "minimum": 0}
        xy = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
        context = {"epoch": string, "clock_id": string, "time_s": scalar}

        def spec(name, description, properties, required):
            return {"name": name, "description": description,
                    "parameters": {"type": "object", "properties": properties,
                                   "required": required, "additionalProperties": False}}

        self.tool_specs = [
            spec("get_map", "Read immutable synthetic planar map and exact source digest; does not localize or move.", {}, []),
            spec("lookup_transform", "Resolve a transform at capture time in an explicit map epoch and clock.",
                 {**context, "target": string, "source": string, "max_age_s": scalar},
                 [*context, "target", "source"]),
            spec("recall", "Retrieve separate source-bound search hints; positions do not authorize grasps.",
                 {**context, "label": {"type": "string", "maxLength": 256}, "max_age_s": scalar},
                 [*context, "label"]),
            spec("plan_route", "Plan a conservative flat-ground route on an exact map. Unknown cells block. No actuation.",
                 {**context, "start_xy_m": xy, "goal_xy_m": xy, "footprint_radius_m": scalar,
                  "clearance_m": scalar, "position_error_m": scalar, "max_age_s": scalar,
                  "expected_map_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}},
                 [*context, "start_xy_m", "goal_xy_m", "footprint_radius_m", "clearance_m",
                  "position_error_m", "expected_map_sha256"]),
        ]
        self.closed = False

    def execute(self, name, args):
        try:
            if self.closed:
                raise ValueError("spatial domain closed")
            if name == "get_map" and not args:
                return {"ok": True, "map": self.grid.as_dict(), "map_sha256": self.grid.sha256,
                        "replay_sha256": self.replay_sha256}
            if name == "lookup_transform":
                value = self.frames.lookup(**args).as_dict()
            elif name == "recall":
                value = self.memory.query(**args)
            elif name == "plan_route":
                value = plan_route(self.grid, **args)
            else:
                raise ValueError("unknown spatial tool or arguments")
            return {"ok": True, "result": value, "source": "synthetic_replay", "physical_admission": False}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500], "physical_admission": False}

    def stop(self):
        return {"ok": True, "actuation": False}

    def reset_stop(self):
        return {"ok": True, "actuation": False, "map_epoch_unchanged": True}

    def close(self):
        self.closed = True
        return {"ok": True, "actuation": False}


def build_spatial_domain(domain_id, profile, *, sensor_domains=None):
    if isinstance(profile, dict) and profile.get("kind") == "spatial" and "cuvslam" in profile:
        from .cuvslam import build_cuvslam_domain
        return build_cuvslam_domain(domain_id, profile, sensor_domains or {})
    if isinstance(profile, dict) and profile.get("kind") == "spatial" and "rgbd" in profile:
        from .rgbd import build_rgbd_spatial_domain
        return build_rgbd_spatial_domain(domain_id, profile, sensor_domains or {})
    allowed = {"kind", "robot_id", "replay"}
    if not isinstance(profile, dict) or set(profile) != allowed or profile["kind"] != "spatial":
        raise ValueError("spatial domain requires only kind, robot_id, replay")
    identifier(domain_id); identifier(profile["robot_id"])
    replay = profile["replay"]
    required = {"version", "map_id", "epoch", "clock_id", "map_frame", "time_s", "calibration_id",
                "transforms", "grid", "observations"}
    if not isinstance(replay, dict) or set(replay) != required or type(replay["version"]) is not int or replay["version"] != 1:
        raise ValueError("invalid spatial replay schema")
    if len(json.dumps(replay, allow_nan=False)) > 2*1024*1024:
        raise ValueError("spatial replay exceeds 2 MiB")
    source_hash = sha256(replay)

    def stamp(time_s, calibration_id):
        return SpatialStamp(replay["map_id"], replay["epoch"], replay["clock_id"], time_s,
                            "synthetic-replay", source_hash, calibration_id, "synthetic")

    frames = FrameTree(replay["map_id"], replay["epoch"], replay["clock_id"], replay["map_frame"])
    if not isinstance(replay["transforms"], list) or len(replay["transforms"]) > 4096:
        raise ValueError("transform replay exceeds bound")
    for item in replay["transforms"]:
        if not isinstance(item, dict) or set(item) - {"parent", "child", "translation_m", "rotation_wxyz",
                                                      "time_s", "calibration_id", "static", "position_error_m", "angular_error_rad"}:
            raise ValueError("unknown transform replay keys")
        value = dict(item)
        value["stamp"] = stamp(value.pop("time_s"), value.pop("calibration_id"))
        frames.add(TransformSample(**value))
    grid = GridSnapshot(frame_id=replay["map_frame"], stamp=stamp(replay["time_s"], replay["calibration_id"]),
                        **replay["grid"])
    memory = SpatialMemory(frames)
    if not isinstance(replay["observations"], list) or len(replay["observations"]) > 1024:
        raise ValueError("observation replay exceeds bound")
    for item in replay["observations"]:
        if not isinstance(item, dict) or set(item) != {"observation_id", "label", "frame_id", "point_m", "time_s", "calibration_id", "confidence"}:
            raise ValueError("invalid landmark replay keys")
        value = dict(item)
        value["stamp"] = stamp(value.pop("time_s"), value.pop("calibration_id"))
        memory.observe(LandmarkObservation(**value))
    return SpatialDomain(domain_id, profile["robot_id"], frames, grid, memory, replay_sha256=source_hash)
