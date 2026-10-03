"""Annotate measured surface pixels from an explicitly shared SensorHub.

No acquisition, segmentation, free-space inference or actuator access occurs
here. The first capture pins the epoch and static world-camera calibration.
"""
from __future__ import annotations

import struct
import threading
import time

import numpy as np

from ..robotics.contracts import ResourceDescriptor, identifier
from ..sensing.hub import SensorHub
from ..sensing.models import RgbdPayload, digest, integer
from .frames import FrameTree, SpatialStamp, TransformSample
from .memory import LandmarkObservation, SpatialMemory


def _quaternion(rotation):
    """Convert a proper column-vector rotation to a normalized wxyz value."""
    r = rotation
    # Symmetric eigenproblem also covers rotations at pi without division by
    # a near-zero trace. The input payload has already checked rigidity.
    k = np.array([
        [r[0, 0]-r[1, 1]-r[2, 2], r[0, 1]+r[1, 0], r[0, 2]+r[2, 0], r[2, 1]-r[1, 2]],
        [r[0, 1]+r[1, 0], r[1, 1]-r[0, 0]-r[2, 2], r[1, 2]+r[2, 1], r[0, 2]-r[2, 0]],
        [r[0, 2]+r[2, 0], r[1, 2]+r[2, 1], r[2, 2]-r[0, 0]-r[1, 1], r[1, 0]-r[0, 1]],
        [r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1], np.trace(r)],
    ])
    q = np.linalg.eigh(k)[1][:, -1][[3, 0, 1, 2]]
    if q[0] < 0:
        q = -q
    return tuple(float(v) for v in q)


class RgbdSpatialDomain:
    motion_skills = frozenset()

    def __init__(self, domain_id, robot_id, hub, *, sensor_domain, sensor_id,
                 map_id, world_frame_id, capacity=128, clock=time.monotonic):
        self.domain_id = identifier(domain_id)
        self.robot_id = identifier(robot_id)
        self.sensor_domain, self.sensor_id = identifier(sensor_domain), identifier(sensor_id)
        self.map_id, self.world_frame_id = identifier(map_id), identifier(world_frame_id)
        if not isinstance(hub, SensorHub):
            raise ValueError("explicit SensorHub required")
        if type(capacity) is not int or not 1 <= capacity <= 1024:
            raise ValueError("RGB-D landmark capacity must be 1..1024")
        descriptor = next((v for v in hub.descriptors if v.sensor_id == sensor_id), None)
        if descriptor is None or descriptor.robot_id != robot_id or descriptor.modality != "rgbd":
            raise ValueError("spatial source must be this robot's declared RGB-D sensor")
        if digest(descriptor.calibration_id) is None:
            raise ValueError("spatial RGB-D requires an explicit calibration SHA256")
        hub.seal()
        self.hub, self.descriptor = hub, descriptor
        self.capacity, self._clock = capacity, clock
        self.frames = self.memory = self._calibration = None
        self._lock, self.closed = threading.RLock(), False
        self.required_resources = (f"{sensor_domain}/{sensor_id}",)
        self.resources = (ResourceDescriptor(f"{domain_id}/memory", "memory", robot_id,
            capabilities=("rgbd_surface_annotation", "spatial_retrieval"),
            synthetic=descriptor.measurement_kind == "synthetic",
            admission="software_only" if descriptor.measurement_kind == "synthetic" else "unvalidated",
            metadata={"sensor_resource": self.required_resources[0], "source": descriptor.source,
                "model_identity_sha256": descriptor.model_identity_sha256,
                "calibration_sha256": descriptor.calibration_id, "map_id": map_id,
                "world_frame_id": world_frame_id, "geometry": "observed_surface_point",
                "semantic_identity": "caller_annotation", "uncertainty": "not_estimated"}),)
        string = {"type": "string", "minLength": 1, "maxLength": 128}
        properties = {"epoch": string, "sequence": {"type": "integer", "minimum": 0},
            "capture_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "pixel": {"type": "array", "items": {"type": "integer", "minimum": 0},
                      "minItems": 2, "maxItems": 2},
            "observation_id": string, "label": {"type": "string", "minLength": 1, "maxLength": 256}}
        self.tool_specs = [
            {"name": "annotate_pixel", "description":
             "Project an exact retained RGB-D pixel to a surface point. Label is your annotation, not verified object identity. No new capture or motion.",
             "parameters": {"type": "object", "properties": properties,
                            "required": list(properties), "additionalProperties": False}},
            {"name": "recall", "description":
             "Retrieve historical surface annotations with original capture provenance and current age. Search hints never authorize motion.",
             "parameters": {"type": "object", "properties": {"epoch": string, "label": properties["label"]},
                            "required": ["epoch", "label"], "additionalProperties": False}},
        ]

    def _retained(self, args):
        return self.hub.retained(self.sensor_id, **{k: args[k] for k in
                                 ("epoch", "sequence", "capture_sha256")})

    def _annotate(self, args):
        if set(args) != {"epoch", "sequence", "capture_sha256", "pixel", "observation_id", "label"}:
            raise ValueError("exact capture, integer pixel and annotation required")
        identifier(args["observation_id"], "observation_id")
        if not isinstance(args["label"], str) or not 1 <= len(args["label"]) <= 256:
            raise ValueError("bounded annotation label required")
        if not isinstance(args["pixel"], (tuple, list)) or len(args["pixel"]) != 2:
            raise ValueError("pixel must contain integer u,v")
        u, v = (integer(x, "pixel coordinate") for x in args["pixel"])
        obs = self._retained(args)
        p = obs.payload
        if (type(p) is not RgbdPayload or p.world_from_camera is None
                or p.world_frame_id != self.world_frame_id or p.metadata.saturated is True):
            raise ValueError("calibrated unsaturated RGB-D in the exact world frame required")
        if u >= p.width or v >= p.height:
            raise ValueError("pixel outside capture")
        z = struct.unpack_from("<f", p.depth_m_f32le, 4*(v*p.width+u))[0]
        if z <= 0:
            raise ValueError("invalid zero depth pixel")
        offset = p.pixel_center_offset_uv if p.pixel_center_offset_uv is not None else (0., 0.)
        point = np.linalg.solve(np.asarray(p.intrinsics).reshape(3, 3),
                                [u + offset[0], v + offset[1], 1.]) * z
        if not np.isfinite(point).all():
            raise ValueError("invalid calibrated projection")
        calibration = (p.width, p.height, p.intrinsics, p.world_from_camera, p.world_frame_id,
                       p.metadata.frame_id, p.metadata.calibration_id, p.pixel_center_offset_uv)
        kind = "measured" if obs.measurement_kind == "hardware" else obs.measurement_kind
        stamp = SpatialStamp(self.map_id, obs.epoch, obs.clock_domain, obs.capture_time_s,
                             f"{self.sensor_domain}/{self.sensor_id}", args["capture_sha256"],
                             p.metadata.calibration_id, kind)
        frames = FrameTree(self.map_id, obs.epoch, obs.clock_domain, self.world_frame_id)
        t = np.asarray(p.world_from_camera).reshape(4, 4)
        frames.add(TransformSample(self.world_frame_id, p.metadata.frame_id, tuple(t[:3, 3]),
            _quaternion(t[:3, :3]), stamp, static=True, position_error_m=None, angular_error_rad=None))
        if self.frames is not None:
            if self._calibration != calibration or stamp.context != self.frames.context:
                raise ValueError("spatial calibration or epoch changed; rebuild domain")
        memory = self.memory if self.memory is not None else SpatialMemory(frames, capacity=self.capacity)
        provenance = {"capture_sha256": args["capture_sha256"], "sensor_resource": self.required_resources[0],
            "source": obs.source, "model_identity_sha256": obs.model_identity_sha256,
            "epoch": obs.epoch, "sequence": obs.sequence, "clock_domain": obs.clock_domain,
            "capture_time_s": obs.capture_time_s, "received_monotonic_s": obs.received_monotonic_s,
            "producer_age_s": obs.producer_age_s, "calibration_sha256": p.metadata.calibration_id,
            "pixel_uv": [u, v], "pixel_center_offset_uv": list(offset),
            "pixel_center_convention": "legacy_integer_center" if p.pixel_center_offset_uv is None else "explicit",
            "depth_m": z, "intrinsics": list(p.intrinsics),
            "world_from_camera": list(p.world_from_camera), "world_frame_id": self.world_frame_id,
            "geometry": "observed_surface_point", "label_kind": "caller_annotation",
            "uncertainty": "not_estimated"}
        annotation = LandmarkObservation(args["observation_id"], args["label"], p.metadata.frame_id,
                                        tuple(point), stamp, None, provenance)
        # Math/serialization must not make a capture fresh. Recheck the exact
        # retained object before committing the bounded memory entry.
        if self._retained(args) is not obs:
            raise ValueError("retained capture changed")
        memory.observe(annotation, capture_frames=frames)
        self.frames, self.memory, self._calibration = frames, memory, calibration
        entry = next(v for v in memory.query(args["label"], epoch=stamp.epoch, clock_id=stamp.clock_id,
                     time_s=stamp.time_s, max_age_s=0) if v["observation_id"] == args["observation_id"])
        return {"ok": True, "result": entry, "capture_age_s": obs.age_s(self._clock()),
                "physical_admission": False}

    def execute(self, name, args):
        try:
            with self._lock:
                if self.closed or not isinstance(args, dict):
                    raise ValueError("spatial domain closed or invalid arguments")
                if name == "annotate_pixel":
                    return self._annotate(args)
                if name == "recall" and set(args) == {"epoch", "label"}:
                    if self.memory is None:
                        raise ValueError("no RGB-D annotations admitted")
                    if args["epoch"] != self.frames.context[1]:
                        raise ValueError("memory epoch mismatch")
                    # Host age is computed separately, never by guessing a
                    # current simulation time or changing historical stamps.
                    entries = self.memory.history(args["label"], epoch=args["epoch"],
                                                  clock_id=self.frames.context[2])
                    now = self._clock()
                    ages = {}
                    for entry in entries:
                        source = entry["provenance"]
                        if now < source["received_monotonic_s"]:
                            raise ValueError("local receipt clock regression")
                        age = source["producer_age_s"] + now - source["received_monotonic_s"]
                        ages[entry["observation_id"]] = {"capture_age_s": age,
                            "within_sensor_age_bound": age <= self.descriptor.max_age_s}
                    # Temporal query metadata is outside each content-addressed
                    # entry. Recalling it cannot alter its original digest.
                    return {"ok": True, "result": entries, "capture_ages": ages,
                            "historical": True, "physical_admission": False}
                raise ValueError("unknown spatial tool or arguments")
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500], "physical_admission": False}

    def stop(self):
        return {"ok": True, "actuation": False}

    def reset_stop(self):
        return {"ok": True, "actuation": False, "map_epoch_unchanged": True}

    def close(self):
        with self._lock:
            self.closed = True
        # The composing sensor domain owns provider lifetime, not this reader.
        return {"ok": True, "actuation": False}


def build_rgbd_spatial_domain(domain_id, profile, sensor_domains):
    if set(profile) != {"kind", "robot_id", "rgbd"}:
        raise ValueError("spatial RGB-D requires only kind, robot_id, rgbd")
    settings = profile["rgbd"]
    required = {"sensor_domain", "sensor_id", "map_id", "world_frame_id"}
    if not isinstance(settings, dict) or set(settings) - (required | {"capacity"}) or not required <= set(settings):
        raise ValueError("exact RGB-D sensor reference, map and world frame required")
    source = sensor_domains.get(settings["sensor_domain"])
    if source is None:
        raise ValueError("spatial source must reference an explicit sensors domain")
    return RgbdSpatialDomain(domain_id, profile["robot_id"], source.hub, **settings)
