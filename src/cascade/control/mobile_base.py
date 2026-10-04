"""Mobile feedback and finite body-twist contracts, independent of arm/Kit APIs.

SI units throughout. Orientations are unit quaternions in wxyz order. A
physical backend MUST read completed physical steps, never integrate a command
into reported feedback. ``kinematic_mock`` deliberately carries no such proof.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
import math
from numbers import Integral, Real
from typing import Mapping

from .mobile_support import SolvedContact, SupportObservation, digest


def _plain_record(record, cls, nested=()):
    """Copy only exact records with plain scalar/tuple leaves, in field order."""
    if type(record) is not cls:
        return None
    result = {}
    scalar_types = (str, int, float, bool, type(None))
    for field in fields(cls):
        value = getattr(record, field.name)
        if field.name in nested or type(value) in scalar_types:
            result[field.name] = value
        elif type(value) is tuple and all(type(item) in scalar_types for item in value):
            result[field.name] = tuple(item for item in value)
        else:
            return None
    return result


def finite_real(value, name: str) -> float:
    """Reject booleans, strings, non-finite values; do not silently coerce JSON."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def nonnegative_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def identifier(value, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty, unpadded string")
    if any(ord(c) < 32 for c in value):
        raise ValueError(f"{name} contains a control character")
    return value


def _vector(value, size: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != size:
        raise ValueError(f"{name} must contain {size} values")
    return tuple(finite_real(x, name) for x in value)


def _names(value, name: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{name} must be a sequence")
    result = tuple(identifier(x, name) for x in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


@dataclass(frozen=True)
class VelocityCommand:
    """Finite command in body axes (+x forward, +y left, +wz CCW)."""

    vx: float
    vy: float
    wz: float
    duration_s: float

    def __post_init__(self):
        for f in fields(self):
            object.__setattr__(self, f.name, finite_real(getattr(self, f.name), f.name))
        if self.duration_s <= 0:
            raise ValueError("duration_s must be positive and finite")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BaseState:
    """Immutable, source-bound snapshot of one completed step.

    ``received_monotonic_s`` is stamped by the CLIENT after receipt; it is not
    a remote monotonic clock. ``producer_age_s`` is the producer's age at send,
    conservatively including transport latency at the adapter. Both are seconds.
    ``contacts`` is legacy body-name telemetry and never proves support.
    ``support`` carries versioned solved shape-pair forces from this exact
    epoch, step, simulation clock and model identity. Missing means unknown.
    ``generation`` is the backend cancellation/admission fence, not a step ID.
    """

    robot_id: str
    source: str
    epoch: str
    step: int
    sim_time_s: float
    received_monotonic_s: float
    producer_age_s: float
    position_world: tuple[float, ...]
    orientation_wxyz: tuple[float, ...]
    linear_velocity_world: tuple[float, ...]
    angular_velocity_body: tuple[float, ...]
    joint_names: tuple[str, ...]
    joint_positions: tuple[float, ...]
    joint_velocities: tuple[float, ...]
    controller_status: str
    generation: int
    contacts: tuple[str, ...]
    fallen: bool
    latched: bool
    measurement_kind: str
    model_identity_sha256: str | None = None
    support: SupportObservation | None = None

    def __post_init__(self):
        for key in ("robot_id", "source", "epoch"):
            identifier(getattr(self, key), key)
        for key in ("step", "generation"):
            object.__setattr__(self, key, nonnegative_int(getattr(self, key), key))
        for key in ("sim_time_s", "received_monotonic_s", "producer_age_s"):
            value = finite_real(getattr(self, key), key)
            if value < 0:
                raise ValueError(f"{key} must be nonnegative")
            object.__setattr__(self, key, value)
        for key, size in (("position_world", 3), ("orientation_wxyz", 4),
                          ("linear_velocity_world", 3), ("angular_velocity_body", 3)):
            object.__setattr__(self, key, _vector(getattr(self, key), size, key))
        if not math.isclose(math.hypot(*self.orientation_wxyz), 1.0, abs_tol=1e-5):
            raise ValueError("orientation_wxyz must be a unit quaternion; no silent normalization")
        object.__setattr__(self, "joint_names", _names(self.joint_names, "joint_names"))
        for key in ("joint_positions", "joint_velocities"):
            object.__setattr__(self, key, _vector(getattr(self, key), len(self.joint_names), key))
        object.__setattr__(self, "contacts", _names(self.contacts, "contacts"))
        for key in ("fallen", "latched"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")
        if self.controller_status not in ("ready", "active", "fault", "disabled"):
            raise ValueError("unknown controller_status")
        if self.measurement_kind not in ("physics", "hardware", "kinematic_mock"):
            raise ValueError("unknown measurement_kind")
        if self.model_identity_sha256 is not None:
            digest(self.model_identity_sha256)
        if self.support is not None:
            support = SupportObservation.from_dict(self.support)
            for key in ("epoch", "step", "sim_time_s", "model_identity_sha256"):
                if getattr(support, key) != getattr(self, key):
                    raise ValueError(f"support {key} differs from completed BaseState")
            object.__setattr__(self, "support", support)

    def as_dict(self) -> dict:
        # Validated plain records need fresh containers, not recursive deepcopy
        # of every solved-contact scalar. Keep asdict's types and field order;
        # subclasses/legacy leaves retain its full copy semantics.
        snapshot = _plain_record(self, BaseState, ('support',))
        if snapshot is not None and self.support is not None:
            support = _plain_record(self.support, SupportObservation, ('contacts',))
            contacts = []
            if support is not None and type(self.support.contacts) is tuple:
                for contact in self.support.contacts:
                    copied = _plain_record(contact, SolvedContact)
                    if copied is None:
                        support = None
                        break
                    contacts.append(copied)
            else:
                support = None
            if support is None:
                snapshot = None
            else:
                support['contacts'] = tuple(contacts)
                snapshot['support'] = support
        if snapshot is None:
            snapshot = asdict(self)
        return {key: list(value) if isinstance(value, tuple) else value
                for key, value in snapshot.items()}

    @classmethod
    def from_dict(cls, data: Mapping) -> BaseState:
        allowed = {f.name for f in fields(cls)}
        required = allowed - {"model_identity_sha256", "support"}
        if not isinstance(data, Mapping) or not required <= set(data) <= allowed:
            raise ValueError("BaseState requires the exact wire schema")
        return cls(**data)


class MobileBase(ABC):
    """Bounded transport with separate observation, command and stop channels.

    Metadata is side-effect-free and contains robot_id/source/measurement_kind.
    connect does not start travel. command_velocity atomically checks the
    expected generation AND the adapter's bound epoch; successful admission
    consumes that generation (ACK generation = expected + 1). Never retry an
    uncertain command. stop always increments generation and clears pending
    commands; latch=False cannot clear a previous latch. reset_stop increments
    generation, clears permission only, never enables a failed controller or
    replays motion. Backends must enforce a local wall lease even during pauses.

    Command/stop/reset ACKs contain ok (bool), robot_id, source, epoch,
    generation (int), latched (bool), and physical endpoints bind
    model_identity_sha256. Command ACKs additionally contain
    accepted (bool), start_sim_time_s and end_sim_time_s (finite seconds).
    An ACK is admission information, NOT physical completion.
    """

    @property
    @abstractmethod
    def metadata(self) -> dict: ...

    @property
    @abstractmethod
    def capabilities(self) -> frozenset[str]: ...

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def get_state(self) -> BaseState: ...

    @abstractmethod
    def command_velocity(self, command: VelocityCommand, *, generation: int) -> dict: ...

    @abstractmethod
    def stop(self, *, latch: bool = True) -> dict: ...

    @abstractmethod
    def reset_stop(self) -> dict: ...
