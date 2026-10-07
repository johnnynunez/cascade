"""Immutable, versioned evidence of complete solved contact reactions.

No estimator lives here. A producer without validated post-solve forces must
emit unavailable, never known with an empty list. World force acts ON shape B;
the unit normal points from A towards B. Pair reversal negates both vectors.
"""
from dataclasses import dataclass, fields
from copy import deepcopy
import math
import re


def digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("model_identity_sha256 must be lowercase SHA-256")
    return value


def plain_contacts_cell(support):
    """The one-slot memo ``[plain contacts or None, computed]`` of a frozen record.

    Kept in the instance ``__dict__`` (not a field: equality, ``asdict`` and the
    identity digest ignore it) so ``rebound`` copies can share one walk over the
    same contact tuple. Filled by ``mobile_base``.
    """
    cell = support.__dict__.get("_plain_contacts_cell")
    if cell is None:
        cell = support.__dict__["_plain_contacts_cell"] = [None, False]
    return cell


def _record(cls, value):
    if type(value) is cls:
        return value
    if not isinstance(value, dict) or set(value) != {f.name for f in fields(cls)}:
        raise ValueError(f"{cls.__name__} requires the exact versioned schema")
    return cls(**value)


@dataclass(frozen=True)
class SolvedContact:
    shape_a_id: int
    shape_b_id: int
    shape_a: str
    shape_b: str
    force_on_b_world_n: tuple
    normal_force_n: float
    normal_a_to_b_world: tuple
    point_world_m: tuple

    def __post_init__(self):
        from .mobile_base import _vector, finite_real, identifier, nonnegative_int
        for key in ("shape_a_id", "shape_b_id"):
            object.__setattr__(self, key, nonnegative_int(getattr(self, key), key))
        for key in ("shape_a", "shape_b"):
            value = identifier(getattr(self, key), key)
            if not value.startswith("/"):
                raise ValueError("contact shapes require absolute paths")
        if self.shape_a == self.shape_b or self.shape_a_id == self.shape_b_id:
            raise ValueError("contact must identify two different shapes")
        for key in ("force_on_b_world_n", "normal_a_to_b_world", "point_world_m"):
            object.__setattr__(self, key, _vector(getattr(self, key), 3, key))
        force = finite_real(self.normal_force_n, "normal_force_n")
        if force < 0:
            raise ValueError("normal_force_n must be nonnegative")
        object.__setattr__(self, "normal_force_n", force)
        if not math.isclose(math.hypot(*self.normal_a_to_b_world), 1., rel_tol=0., abs_tol=1e-5):
            raise ValueError("contact normal must be unit length")
        compression = sum(f * n for f, n in zip(self.force_on_b_world_n, self.normal_a_to_b_world))
        # Float32 solver-basis conversion may round; these are numerical
        # consistency tolerances, never minimum load/physical acceptance gates.
        if compression < 0 or not math.isclose(compression, force, rel_tol=1e-5, abs_tol=1e-7):
            raise ValueError("normal force differs from compressive solved vector projection")


@dataclass(frozen=True)
class SupportObservation:
    version: int
    status: str
    reason: str
    epoch: str
    step: int
    sim_time_s: float
    model_identity_sha256: str
    contacts: tuple

    def __post_init__(self):
        from .mobile_base import finite_real, identifier, nonnegative_int
        if type(self.version) is not int or self.version != 1:
            raise ValueError("unsupported support observation version")
        if self.status not in ("known", "unavailable"):
            raise ValueError("unknown support observation status")
        if not isinstance(self.reason, str) or (self.status == "unavailable" and not self.reason.strip()):
            raise ValueError("unavailable support requires a reason")
        identifier(self.epoch, "support epoch")
        digest(self.model_identity_sha256)
        object.__setattr__(self, "step", nonnegative_int(self.step, "support step"))
        value = finite_real(self.sim_time_s, "support simulation time")
        if value < 0:
            raise ValueError("support time must be nonnegative")
        object.__setattr__(self, "sim_time_s", value)
        if not isinstance(self.contacts, (tuple, list)) or len(self.contacts) > 4096:
            raise ValueError("support contacts must be a bounded sequence")
        contacts = tuple(_record(SolvedContact, c) for c in self.contacts)
        if self.status == "unavailable" and contacts:
            raise ValueError("unavailable support cannot advertise partial contacts")
        # Multiple points for one shape pair are valid; conflicting ID/name maps are not.
        ids, names = {}, {}
        for c in contacts:
            for sid, name in ((c.shape_a_id, c.shape_a), (c.shape_b_id, c.shape_b)):
                if ids.get(sid, name) != name or names.get(name, sid) != sid:
                    raise ValueError("conflicting contact shape identity")
                ids[sid], names[name] = name, sid
        object.__setattr__(self, "contacts", contacts)

    @classmethod
    def from_dict(cls, data):
        return _record(cls, data)

    def rebound(self, *, epoch, model_identity_sha256):
        """The same validated immutable observation under another epoch/model identity.

        A shared-scene owner binds one decoded contact set to twelve robots on
        every read. ``dataclasses.replace`` would re-run ``__post_init__`` and
        re-parse every solved contact each time (about 0.4 ms for ~90 contacts,
        24 times per step); the contacts were validated when this observation
        was built and are immutable, so only the two identifiers are validated
        here. Anything that is not an exact immutable record keeps the full
        ``replace`` re-validation.
        """
        from dataclasses import replace
        if not immutable_support(self):
            return replace(self, epoch=epoch, model_identity_sha256=model_identity_sha256)
        from .mobile_base import identifier
        identifier(epoch, "support epoch")
        digest(model_identity_sha256)
        result = object.__new__(SupportObservation)
        for field in fields(SupportObservation):
            object.__setattr__(result, field.name, getattr(self, field.name))
        object.__setattr__(result, "epoch", epoch)
        object.__setattr__(result, "model_identity_sha256", model_identity_sha256)
        result.__dict__["_plain_contacts_cell"] = plain_contacts_cell(self)  # same contacts tuple
        return result

    def as_observation_dict(self):
        """Detached native observation; keep its existing list types and order."""
        result = dict(version=self.version, status=self.status, reason=self.reason,
            step=self.step, sim_time_s=self.sim_time_s,
            contacts=[dict(shape_a=c.shape_a, shape_b=c.shape_b,
                shape_a_id=c.shape_a_id, shape_b_id=c.shape_b_id,
                force_on_b_world_n=list(c.force_on_b_world_n), normal_force_n=c.normal_force_n,
                point_world_m=list(c.point_world_m), normal_a_to_b_world=list(c.normal_a_to_b_world))
                for c in self.contacts], epoch=self.epoch, model_identity_sha256=self.model_identity_sha256)
        return result if immutable_support(self) else deepcopy(result)


def immutable_support(value):
    """Exact records with plain strings; numeric/vector fields normalize on init.

    A str subclass may carry mutable attributes. Such records remain valid
    legacy values, but cannot enter a shared-reference copy fast path.
    """
    return (type(value) is SupportObservation
        and all(type(getattr(value, k)) is str for k in ('status', 'reason', 'epoch', 'model_identity_sha256'))
        and all(type(c) is SolvedContact and type(c.shape_a) is str and type(c.shape_b) is str
                for c in value.contacts))


def support_contract(value):
    """An explicit registry bound through the canonical model identity digest."""
    from .mobile_base import _names, _vector
    if value is None:
        return None
    keys = {"version", "model_identity_sha256", "robot_shapes", "foot_shapes",
            "ground_shapes", "gravity_world_m_s2"}
    if not isinstance(value, dict) or set(value) != keys or type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("support_contract requires the exact version 1 registry")
    result = {"version": 1, "model_identity_sha256": digest(value["model_identity_sha256"])}
    for key in ("robot_shapes", "foot_shapes", "ground_shapes"):
        result[key] = _names(value[key], key)
        if not result[key] or any(not p.startswith("/") for p in result[key]):
            raise ValueError("support registry requires explicit absolute shape paths")
    if not set(result["foot_shapes"]) <= set(result["robot_shapes"]):
        raise ValueError("every admitted foot must belong to the robot")
    if set(result["robot_shapes"]) & set(result["ground_shapes"]):
        raise ValueError("robot and ground registries must be disjoint")
    result["gravity_world_m_s2"] = _vector(value["gravity_world_m_s2"], 3, "gravity")
    if math.hypot(*result["gravity_world_m_s2"]) == 0:
        raise ValueError("support requires nonzero gravity")
    return result
