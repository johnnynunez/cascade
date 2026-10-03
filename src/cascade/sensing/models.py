"""Immutable sensor values. A modality is a measurement contract, not proof of a task."""
from __future__ import annotations

import array
import base64
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
import math
import re
import sys
from typing import ClassVar


MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_PIXELS = 1024 * 1024


def token(value, name):
    if (not isinstance(value, str) or not value or value != value.strip() or len(value) > 256
            or any(ord(c) < 32 for c in value)):
        raise ValueError(f"{name} must be a nonempty bounded string")
    return value


def number(value, name, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    value = float(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} below minimum")
    return value


def integer(value, name, *, maximum=None):
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be a bounded nonnegative integer")
    return value


def digest(value):
    if value is not None and (not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None):
        raise ValueError("model identity must be lowercase SHA256 or explicitly unknown")
    return value


def vector(value, length, name):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must have {length} elements")
    return tuple(number(v, name) for v in value)


def wire(value):
    if isinstance(value, Payload):
        result = {"modality": value.modality, "units": dict(value.units),
                  **{f.name: wire(getattr(value, f.name)) for f in fields(value)}}
        if isinstance(value, RgbdPayload) and value.world_from_camera is None:
            # Preserve the original v1 payload byte shape unless the calibrated
            # optical-to-world extension is explicitly present.
            result.pop('world_from_camera')
            result.pop('world_frame_id')
        return result
    if is_dataclass(value):
        return {f.name: wire(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, bytes):
        return {"encoding": "base64", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, tuple):
        return [wire(v) for v in value]
    if isinstance(value, Mapping):
        return {k: wire(v) for k, v in value.items()}
    return value


@dataclass(frozen=True)
class MeasurementMetadata:
    frame_id: str
    calibration_id: str | None = None
    saturated: bool | None = None

    def __post_init__(self):
        token(self.frame_id, "frame_id")
        if self.calibration_id is not None:
            token(self.calibration_id, "calibration_id")
        if self.saturated is not None and type(self.saturated) is not bool:
            raise ValueError("saturation must be boolean or explicitly unknown")


class Payload:
    modality: ClassVar[str]
    units: ClassVar[tuple]

    def _metadata(self):
        if not isinstance(self.metadata, MeasurementMetadata):
            raise ValueError("typed measurement metadata required")


@dataclass(frozen=True)
class ImuPayload(Payload):
    metadata: MeasurementMetadata
    angular_velocity_rad_s: tuple
    linear_acceleration_m_s2: tuple | None = None
    modality: ClassVar[str] = "imu"
    units: ClassVar[tuple] = (("angular_velocity_rad_s", "rad/s"), ("linear_acceleration_m_s2", "m/s^2"))

    def __post_init__(self):
        self._metadata()
        object.__setattr__(self, "angular_velocity_rad_s", vector(self.angular_velocity_rad_s, 3, "gyro"))
        if self.linear_acceleration_m_s2 is not None:
            object.__setattr__(self, "linear_acceleration_m_s2", vector(self.linear_acceleration_m_s2, 3, "acceleration"))


@dataclass(frozen=True)
class ProprioceptionPayload(Payload):
    metadata: MeasurementMetadata
    joint_names: tuple
    position_rad: tuple
    velocity_rad_s: tuple
    effort_nm: tuple | None = None
    modality: ClassVar[str] = "proprioception"
    units: ClassVar[tuple] = (("position_rad", "rad"), ("velocity_rad_s", "rad/s"), ("effort_nm", "N*m"))

    def __post_init__(self):
        self._metadata()
        if not isinstance(self.joint_names, (list, tuple)) or not 1 <= len(self.joint_names) <= 512:
            raise ValueError("joint_names must contain 1..512 names")
        names = tuple(token(v, "joint name") for v in self.joint_names)
        if len(set(names)) != len(names):
            raise ValueError("duplicate joint names")
        object.__setattr__(self, "joint_names", names)
        for key in ("position_rad", "velocity_rad_s", "effort_nm"):
            if getattr(self, key) is not None:
                object.__setattr__(self, key, vector(getattr(self, key), len(names), key))
            elif key != "effort_nm":
                raise ValueError(f"{key} is required")


@dataclass(frozen=True)
class JointMeasurement:
    """One movable coordinate; effort is unknown when the producer omits it."""
    joint_id: str
    joint_type: str
    position: float
    velocity: float
    position_unit: str
    velocity_unit: str
    effort_unit: str
    effort: float | None = None

    def __post_init__(self):
        from ..robotics.embodiment import JOINT_UNITS
        token(self.joint_id, "joint_id")
        expected = JOINT_UNITS.get(self.joint_type)
        if expected is None or (self.position_unit, self.velocity_unit, self.effort_unit) != expected:
            raise ValueError("joint measurement type/units disagree")
        for key in ("position", "velocity", "effort"):
            if key != "effort" or self.effort is not None:
                object.__setattr__(self, key, number(getattr(self, key), key))


@dataclass(frozen=True)
class JointStatePayload(Payload):
    """Mixed angular/linear proprioception bound to a structural declaration.

    The declaration digest is separate from ObservationEnvelope's physical model
    identity. Neither units nor this digest certify hardware or control admission.
    """
    metadata: MeasurementMetadata
    joints: tuple
    embodiment_sha256: str
    modality: ClassVar[str] = "joint_state"
    units: ClassVar[tuple] = (("joints", "explicit_per_coordinate"),)

    def __post_init__(self):
        self._metadata()
        if self.embodiment_sha256 is None:
            raise ValueError("joint state requires an embodiment digest")
        digest(self.embodiment_sha256)
        if not isinstance(self.joints, (tuple, list)) or not 1 <= len(self.joints) <= 512:
            raise ValueError("joint state must contain 1..512 coordinates")
        joints = tuple(JointMeasurement(**j) if isinstance(j, dict) else j for j in self.joints)
        if any(type(j) is not JointMeasurement for j in joints):
            raise ValueError("typed joint measurements required")
        if len({j.joint_id for j in joints}) != len(joints):
            raise ValueError("duplicate joint measurements")
        object.__setattr__(self, "joints", joints)


@dataclass(frozen=True)
class GeneralizedJointMeasurement:
    """Configuration q and tangent velocity v; quaternion rate is never v.

    For multi-DoF joints, v and its dual effort use the parent joint frame at
    the child joint origin. Missing effort is unknown, not a zero reaction.
    """
    joint_id: str
    joint_type: str
    q: tuple
    v: tuple
    coordinates: object
    effort: tuple | None = None

    def __post_init__(self):
        from ..robotics.joint_coordinates import QUATERNION_NORM_SQUARED_TOLERANCE, validate_coordinates
        token(self.joint_id, "joint_id")
        convention = validate_coordinates(self.joint_type, self.coordinates)
        object.__setattr__(self, "coordinates", convention)
        object.__setattr__(self, "q", vector(self.q, convention["nq"], "configuration q"))
        object.__setattr__(self, "v", vector(self.v, convention["nv"], "tangent velocity v"))
        if self.effort is not None:
            object.__setattr__(self, "effort", vector(self.effort, convention["nv"], "generalized effort"))
        if self.joint_type in {"spherical", "floating"}:
            quaternion = self.q if self.joint_type == "spherical" else self.q[3:]
            if not math.isclose(sum(x*x for x in quaternion), 1., rel_tol=0,
                                abs_tol=QUATERNION_NORM_SQUARED_TOLERANCE):
                raise ValueError("configuration quaternion must be unit length; no implicit normalization")


@dataclass(frozen=True)
class GeneralizedJointStatePayload(Payload):
    """Opt-in observation contract, separate from legacy scalar joint packets."""
    metadata: MeasurementMetadata
    joints: tuple
    embodiment_sha256: str
    modality: ClassVar[str] = "generalized_joint_state"
    units: ClassVar[tuple] = (("joints", "explicit_q_v_effort_components"),)

    def __post_init__(self):
        self._metadata()
        if self.embodiment_sha256 is None:
            raise ValueError("generalized joint state requires an embodiment digest")
        digest(self.embodiment_sha256)
        if not isinstance(self.joints, (tuple, list)) or not 1 <= len(self.joints) <= 512:
            raise ValueError("generalized joint state must contain 1..512 joints")
        joints = tuple(GeneralizedJointMeasurement(**j) if isinstance(j, dict) else j for j in self.joints)
        if any(type(j) is not GeneralizedJointMeasurement for j in joints):
            raise ValueError("typed generalized joint measurements required")
        if len({j.joint_id for j in joints}) != len(joints):
            raise ValueError("duplicate generalized joint measurements")
        object.__setattr__(self, "joints", joints)


@dataclass(frozen=True)
class SolvedContactPayload(Payload):
    metadata: MeasurementMetadata
    observation: object
    modality: ClassVar[str] = "solved_contact"
    units: ClassVar[tuple] = (("force", "N"), ("point", "m"), ("normal", "unit_vector"))

    def __post_init__(self):
        from ..control.mobile_support import SupportObservation
        self._metadata()
        if self.metadata.frame_id != "world":
            raise ValueError("solved support vectors are in world frame")
        observation = SupportObservation.from_dict(self.observation)
        if len(observation.reason) > 2048:
            raise ValueError("support reason exceeds sensor payload bound")
        for contact in observation.contacts:
            for path in (contact.shape_a, contact.shape_b):
                if len(path) > 2048:
                    raise ValueError("support shape path exceeds sensor payload bound")
            for index in (contact.shape_a_id, contact.shape_b_id):
                integer(index, "native shape index", maximum=2 ** 63 - 1)
        object.__setattr__(self, "observation", observation)


@dataclass(frozen=True)
class EstimatedTactilePayload(Payload):
    metadata: MeasurementMetadata
    estimator_id: str
    points_m: tuple
    normal_force_n: tuple
    shear_force_n: tuple
    modality: ClassVar[str] = "estimated_tactile"
    units: ClassVar[tuple] = (("points_m", "m"), ("normal_force_n", "N"), ("shear_force_n", "N"))

    def __post_init__(self):
        self._metadata()
        token(self.estimator_id, "estimator_id")
        if not isinstance(self.points_m, (tuple, list)) or not 1 <= len(self.points_m) <= 4096:
            raise ValueError("tactile points must contain 1..4096 entries")
        points = tuple(vector(p, 3, "tactile point") for p in self.points_m)
        normal = vector(self.normal_force_n, len(points), "normal force")
        if any(v < 0 for v in normal) or not isinstance(self.shear_force_n, (list, tuple)) or len(self.shear_force_n) != len(points):
            raise ValueError("invalid tactile normal/shear force")
        object.__setattr__(self, "points_m", points)
        object.__setattr__(self, "normal_force_n", normal)
        object.__setattr__(self, "shear_force_n", tuple(vector(v, 2, "shear force") for v in self.shear_force_n))


def _image(width, height, encoding, data):
    integer(width, "width"); integer(height, "height")
    if width <= 0 or height <= 0 or width * height > MAX_PIXELS:
        raise ValueError("image pixel bound exceeded")
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_IMAGE_BYTES:
        raise ValueError("image requires bounded immutable bytes")
    if encoding in ("rgb8", "gray8"):
        if len(data) != width * height * (3 if encoding == "rgb8" else 1):
            raise ValueError("image byte count disagrees with dimensions")
    elif encoding == "jpeg":
        from ..sim.mobile_frames import _jpeg_dimensions
        if _jpeg_dimensions(data) != (width, height):
            raise ValueError("JPEG dimensions disagree")
    else:
        raise ValueError("unsupported image encoding")


@dataclass(frozen=True)
class RgbPayload(Payload):
    metadata: MeasurementMetadata
    width: int
    height: int
    encoding: str
    data: bytes
    modality: ClassVar[str] = "rgb"
    units: ClassVar[tuple] = (("image", "uint8"),)

    def __post_init__(self):
        self._metadata()
        if self.encoding == "gray8":
            raise ValueError("RGB cannot contain a grayscale image")
        _image(self.width, self.height, self.encoding, self.data)


@dataclass(frozen=True)
class TactileImagePayload(Payload):
    metadata: MeasurementMetadata
    width: int
    height: int
    encoding: str
    data: bytes
    formation_id: str
    modality: ClassVar[str] = "tactile_image"
    units: ClassVar[tuple] = (("image", "uint8"),)

    def __post_init__(self):
        self._metadata()
        token(self.formation_id, "formation_id")
        _image(self.width, self.height, self.encoding, self.data)


@dataclass(frozen=True)
class RgbdPayload(Payload):
    """One registered capture; zero depth explicitly denotes an invalid pixel."""
    metadata: MeasurementMetadata
    width: int
    height: int
    rgb8: bytes
    depth_m_f32le: bytes
    intrinsics: tuple
    world_from_camera: tuple | None = None
    world_frame_id: str | None = None
    modality: ClassVar[str] = "rgbd"
    units: ClassVar[tuple] = (("rgb8", "uint8"), ("depth_m_f32le", "m"), ("intrinsics", "pixel"))

    def __post_init__(self):
        self._metadata()
        _image(self.width, self.height, "rgb8", self.rgb8)
        if not isinstance(self.depth_m_f32le, bytes) or len(self.depth_m_f32le) != 4 * self.width * self.height:
            raise ValueError("depth requires aligned float32 little-endian bytes")
        depth = array.array("f")
        depth.frombytes(self.depth_m_f32le)
        if sys.byteorder != "little":
            depth.byteswap()
        if any(not math.isfinite(v) or v < 0 for v in depth):
            raise ValueError("depth must be finite nonnegative meters; invalid pixels use zero")
        k = vector(self.intrinsics, 9, "intrinsics")
        if k[0] <= 0 or k[4] <= 0 or k[6:] != (0., 0., 1.):
            raise ValueError("invalid pinhole intrinsics")
        object.__setattr__(self, "intrinsics", k)
        if (self.world_from_camera is None) != (self.world_frame_id is None):
            raise ValueError("RGB-D transform and target frame must be supplied together")
        if self.world_from_camera is not None:
            import numpy as np
            token(self.world_frame_id, "world_frame_id")
            values = vector(self.world_from_camera, 16, "world_from_camera")
            t = np.array(values).reshape(4, 4)
            if (not np.array_equal(t[3], [0, 0, 0, 1])
                    or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), rtol=0, atol=1e-7)
                    or not math.isclose(np.linalg.det(t[:3, :3]), 1., abs_tol=1e-7)):
                raise ValueError("RGB-D transform must be rigid")
            object.__setattr__(self, "world_from_camera", values)


PAYLOAD_TYPES = (ImuPayload, ProprioceptionPayload, JointStatePayload, GeneralizedJointStatePayload, SolvedContactPayload, EstimatedTactilePayload,
                 RgbPayload, RgbdPayload, TactileImagePayload)


@dataclass(frozen=True)
class ObservationEnvelope:
    source: str
    sensor_id: str
    epoch: str
    sequence: int
    clock_domain: str
    capture_time_s: float
    received_monotonic_s: float
    producer_age_s: float
    model_identity_sha256: str | None
    measurement_kind: str
    payload: Payload
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported observation schema")
        for key in ("source", "sensor_id", "epoch", "clock_domain"):
            token(getattr(self, key), key)
        integer(self.sequence, "sequence", maximum=2 ** 63 - 1)
        for key in ("capture_time_s", "received_monotonic_s", "producer_age_s"):
            object.__setattr__(self, key, number(getattr(self, key), key, minimum=0))
        digest(self.model_identity_sha256)
        if self.measurement_kind not in ("physics", "hardware", "synthetic"):
            raise ValueError("unknown measurement kind")
        if self.measurement_kind == "physics" and self.model_identity_sha256 is None:
            raise ValueError("physics requires an explicit model identity")
        if type(self.payload) not in PAYLOAD_TYPES:
            raise ValueError("unsupported typed sensor payload")
        if isinstance(self.payload, SolvedContactPayload):
            support = self.payload.observation
            if (support.epoch != self.epoch or support.step != self.sequence
                    or support.sim_time_s != self.capture_time_s
                    or support.model_identity_sha256 != self.model_identity_sha256
                    or self.clock_domain != "simulation"):
                raise ValueError("solved contact identity/clock differs from envelope")

    def age_s(self, now):
        now = number(now, "local monotonic clock", minimum=0)
        if now < self.received_monotonic_s:
            raise ValueError("future receipt or local clock regression")
        return self.producer_age_s + now - self.received_monotonic_s

    def as_dict(self):
        return wire(self)
