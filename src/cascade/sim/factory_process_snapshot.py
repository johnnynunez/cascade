"""Explicit binary schema for exact private Factory readbacks, CPU candidate.

Only these known constructors are used on decode; no pickle crosses the wire.
This schema preserves evidence, including rows later refused by the controller.
It does not replace check_solve, check_seating_solve, freshness or outcome gates.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass, fields

from ..control.fastening import FasteningSolve, FasteningUpload, SolvedPair
from ..control.fastening_seat import SeatingSolve, ShoulderContact
from .factory_observation import _FactoryReadback, _Float32Snapshot, _RawMetadata
from .factory_process_wire import WireFault, WireLimits, decode, encode
from .microduck_contact_support import _ContactSnapshot

_SCALARS = tuple(field.name for field in fields(FasteningSolve) if field.name != "contacts")
_UPLOAD = tuple(field.name for field in fields(FasteningUpload))
_SHOULDER = tuple(field.name for field in fields(ShoulderContact))
_CHANNELS = ("native_ctrl", "actuator_force", "qfrc_actuator", "qfrc_applied",
             "qfrc_constraint", "qfrc_passive", "body_poses_xyzw")
_KEYS = ("solve", "upload", "effort_time", "collision_buffers", "contacts",
         "constraint_count", "constraint_capacity", *_CHANNELS, "native_clock", "collision_interval")
_EFFORT_TIME = "applied during interval (step-1,step); not reevaluated at final pose"
_META_LIMITS = WireLimits(4096, 256, 6, 128)
_COLLISION_NAMES = ("broad_phase", "contacts", "gjk", "reducer", "shape_pairs_mesh",
                    "shape_pairs_mesh_mesh", "shape_pairs_mesh_plane", "shape_pairs_sdf_sdf", "triangle_pairs")


@dataclass(frozen=True, slots=True)
class SnapshotLayout:
    """Must be tied to the effective native model, not inferred per new row."""

    labels: tuple[str, ...]
    joints: int
    actuators: int
    dofs: int
    bodies: int
    collision_capacity: int
    solver_capacity: int
    constraint_capacity: int
    binding_sha256: str
    epoch: str
    collision_buffers: tuple[tuple[str, int], ...]

    def __post_init__(self):
        if sys.byteorder != "little":
            raise ValueError("Factory snapshot v1 requires little-endian native channels")
        if (type(self.labels) is not tuple or not 1 <= len(self.labels) <= 65536
                or any(type(x) is not str or not x or len(x.encode("utf-8")) > 512 for x in self.labels)
                or len(set(self.labels)) != len(self.labels)):
            raise ValueError("invalid bound shape labels")
        for value, upper in ((self.joints, 32), (self.actuators, 256), (self.dofs, 256),
                             (self.bodies, 256), (self.collision_capacity, 2048),
                             (self.solver_capacity, 2048), (self.constraint_capacity, 1048576)):
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError("invalid bound native capacity")
        if (type(self.binding_sha256) is not str or len(self.binding_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.binding_sha256)
                or type(self.epoch) is not str or not self.epoch or len(self.epoch.encode("utf-8")) > 512):
            raise ValueError("invalid snapshot binding/epoch")
        if (type(self.collision_buffers) is not tuple or len(self.collision_buffers) != len(_COLLISION_NAMES)
                or any(type(row) is not tuple or len(row) != 2 or type(row[0]) is not str
                       or type(row[1]) is not int or not 0 <= row[1] <= 2**31-1 for row in self.collision_buffers)
                or {row[0] for row in self.collision_buffers} != set(_COLLISION_NAMES)
                or dict(self.collision_buffers)["contacts"] != self.collision_capacity):
            raise ValueError("complete collision capacities required")

    @property
    def sha256(self):
        values = {field.name: getattr(self, field.name) for field in fields(self)}
        return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @property
    def shapes(self):
        return ((self.actuators,), (self.actuators,), *((self.dofs,),) * 4, (self.bodies, 7))

    @property
    def maximum_frame_bytes(self):
        # Full contact columns: 8 bytes of shape IDs + 4 active index + 8 force
        # + 9*8 vectors = 92 bytes/candidate. Full shoulder witness primitive
        # tuple: 5 header + 9 candidate + 3*(5+3*9) vectors + 9 force = 119.
        # The remainder explicitly bounds metadata, scalar solve fields, field
        # ordering and byte/tuple headers, and all seven raw float32 channels.
        return (211 * self.solver_capacity + 4096 + 2048 + 1024 + 27 * self.joints
                + 4 * (2 * self.actuators + 4 * self.dofs + 7 * self.bodies))

    @property
    def wire_limits(self):
        return WireLimits(self.maximum_frame_bytes, 32 * self.solver_capacity + 2048, 12, 512)


def _metadata(value):
    if type(value) is not _RawMetadata:
        raise WireFault("exact raw metadata required")
    return tuple((key, _metadata(item) if type(item) is _RawMetadata else item)
                 for key, item in value.entries)


def _restore_metadata(entries):
    if type(entries) is not tuple or any(type(x) is not tuple or len(x) != 2 for x in entries):
        raise WireFault("invalid metadata schema")
    return _RawMetadata(tuple((key, _restore_metadata(item) if type(item) is tuple else item)
                              for key, item in entries))


def _validate(record, layout):
    if type(record) is not _FactoryReadback or type(layout) is not SnapshotLayout:
        raise WireFault("exact Factory readback and bound layout required")
    if len(record.entries) != len(_KEYS) or {key for key, _ in record.entries} != set(_KEYS):
        raise WireFault("incomplete or extra Factory snapshot fields")
    # Repeat its existing cross-field validation; physical/freshness checks are
    # still the original controller's job after raw archival.
    record.__post_init__()
    values = dict(record.entries)
    solve, contacts = values["solve"], values["contacts"]
    if (solve.binding_sha256 != layout.binding_sha256 or solve.epoch != layout.epoch
            or solve.collision_capacity != layout.collision_capacity
            or solve.solver_capacity != layout.solver_capacity
            or contacts.labels != layout.labels or contacts.count >= layout.solver_capacity
            or any(len(getattr(solve, name)) != layout.joints for name in
                   ("joint_position_rad", "joint_velocity_rad_s", "joint_effort_nm"))):
        raise WireFault("snapshot differs from its bound native layout")
    if any(type(pair) is not SolvedPair or type(pair.normal_force_n) is not float for pair in solve.contacts):
        raise WireFault("snapshot requires exact native solved-pair types")
    if type(solve) is SeatingSolve and len(solve.shoulder_contacts) > contacts.count:
        raise WireFault("shoulder witness count exceeds candidate count")
    for name, shape in zip(_CHANNELS, layout.shapes, strict=True):
        channel = values[name]
        if type(channel) is not _Float32Snapshot or channel.shape != shape:
            raise WireFault("raw channel differs from bound shape")
        channel.__post_init__()
    if values["effort_time"] != _EFFORT_TIME:
        raise WireFault("raw effort phase differs")
    if (type(values["constraint_count"]) is not int or type(values["constraint_capacity"]) is not int
            or values["constraint_capacity"] != layout.constraint_capacity
            or not 0 <= values["constraint_count"] < layout.constraint_capacity):
        raise WireFault("invalid native constraint schema")
    clock = values["native_clock"]
    interval = values["collision_interval"]
    if type(clock) is not _RawMetadata or type(interval) is not _RawMetadata:
        raise WireFault("exact native clock/interval metadata required")
    clock, interval = dict(clock.entries), dict(interval.entries)
    if (set(clock) != {"step", "time_s", "timestep_s", "interval_clock_s", "cuda_graph"}
            or type(clock["step"]) is not int or clock["step"] != solve.step
            or clock["cuda_graph"] is not False
            or any(type(clock[k]) is not float or not math.isfinite(clock[k]) for k in
                   ("time_s", "timestep_s", "interval_clock_s"))
            or clock["time_s"] < 0 or clock["timestep_s"] <= 0
            or clock["interval_clock_s"] != solve.simulation_time_s
            or set(interval) != {"before_step", "after_step", "generation"}
            or any(type(v) is not int for v in interval.values())
            or interval != {"before_step": values["upload"].before_step,
                            "after_step": solve.step, "generation": solve.generation}):
        raise WireFault("native clock/interval differs from its solved row")
    buffers = values["collision_buffers"]
    if type(buffers) is not _RawMetadata or {k for k, _ in buffers.entries} != set(_COLLISION_NAMES):
        raise WireFault("incomplete collision buffer metadata")
    capacities = dict(layout.collision_buffers)
    for key, item in buffers.entries:
        if type(item) is not _RawMetadata:
            raise WireFault("invalid collision buffer metadata")
        item = dict(item.entries)
        expected_keys = {"count", "capacity", "insert_failures"} if key == "reducer" else {"count", "capacity"}
        # collision_coverage explicitly marks a missing counter/backing pair.
        # A zero-length existing backing instead has the two ordinary fields.
        # Preserve both variants and their original key order, without treating
        # an unknown or active buffer as disabled.
        disabled = "disabled" in item
        if disabled:
            expected_keys.add("disabled")
        if (set(item) != expected_keys
                or any(type(v) is not int for name, v in item.items() if name != "disabled")
                or disabled and (key == "reducer" or item["disabled"] is not True
                                 or item["capacity"] != 0 or item["count"] != 0)
                or item["capacity"] != capacities[key] or item["count"] < 0
                or (item["count"] >= item["capacity"] if item["capacity"] else item["count"] != 0)
                or key == "reducer" and (item["insert_failures"] != 0
                                         or item["count"] * 100 >= item["capacity"] * 80)):
            raise WireFault("invalid collision buffer counts/capacities")
        if key == "contacts" and item["count"] != solve.collision_count:
            raise WireFault("collision metadata differs from solve")
    return values


def encode_snapshot(record: _FactoryReadback, layout: SnapshotLayout) -> bytes:
    values = _validate(record, layout)
    solve, contacts, upload = values["solve"], values["contacts"], values["upload"]
    metadata = encode(tuple(_metadata(values[key]) for key in
                            ("collision_buffers", "native_clock", "collision_interval")), _META_LIMITS)
    shoulders = (tuple(tuple(getattr(item, key) for key in _SHOULDER) for item in solve.shoulder_contacts)
                 if type(solve) is SeatingSolve else None)
    # Solve.contacts are exactly bound to these same raw columns by _validate;
    # encode them once, reconstructing the original SolvedPair objects on read.
    value = (1, layout.sha256, tuple(_KEYS.index(key) for key, _ in record.entries),
             tuple(getattr(solve, key) for key in _SCALARS), shoulders,
             (contacts.count, contacts.shape_ids, contacts.active_indices, contacts.normal_forces, contacts.vectors),
             tuple(getattr(upload, key) for key in _UPLOAD),
             tuple(values[key].data for key in _CHANNELS), metadata,
             values["constraint_count"])
    return encode(value, layout.wire_limits)


def decode_snapshot(payload: bytes, layout: SnapshotLayout) -> _FactoryReadback:
    if type(layout) is not SnapshotLayout:
        raise WireFault("exact bound snapshot layout required")
    value = decode(payload, layout.wire_limits)
    if type(value) is not tuple or len(value) != 10:
        raise WireFault("invalid Factory snapshot envelope")
    version, digest, order, scalars, shoulders, columns, upload, channels, metadata, count = value
    if (type(version) is not int or version != 1 or digest != layout.sha256
            or type(order) is not tuple or len(order) != len(_KEYS)
            or any(type(i) is not int for i in order) or set(order) != set(range(len(_KEYS)))
            or type(scalars) is not tuple or len(scalars) != len(_SCALARS)
            or type(columns) is not tuple or len(columns) != 5
            or type(columns[0]) is not int or not 0 <= columns[0] < layout.solver_capacity
            or any(type(blob) is not bytes for blob in columns[1:])
            or type(upload) is not tuple or len(upload) != len(_UPLOAD)
            or type(channels) is not tuple or len(channels) != len(_CHANNELS)
            or type(metadata) is not bytes):
        raise WireFault("invalid Factory snapshot field schema")
    if shoulders is not None and (type(shoulders) is not tuple or len(shoulders) > columns[0]
            or any(type(row) is not tuple or len(row) != len(_SHOULDER) for row in shoulders)):
        raise WireFault("invalid shoulder snapshot schema")
    contacts = _ContactSnapshot(layout.labels, *columns)
    shapes, indices, forces, _ = contacts._arrays()
    active = 0
    pairs = []
    for index, (a, b) in enumerate(shapes):
        normal = 0.
        if active < len(indices) and indices[active] == index:
            normal = float(forces[active])
            active += 1
        pairs.append(SolvedPair(layout.labels[a], layout.labels[b], normal))
    arguments = dict(zip(_SCALARS, scalars, strict=True), contacts=tuple(pairs))
    solve = (FasteningSolve(**arguments) if shoulders is None else
             SeatingSolve(**arguments, shoulder_contacts=tuple(ShoulderContact(*row) for row in shoulders)))
    meta = decode(metadata, _META_LIMITS)
    if type(meta) is not tuple or len(meta) != 3:
        raise WireFault("invalid native metadata envelope")
    values = {"solve": solve, "upload": FasteningUpload(*upload), "contacts": contacts,
              "collision_buffers": _restore_metadata(meta[0]), "native_clock": _restore_metadata(meta[1]),
              "constraint_count": count, "constraint_capacity": layout.constraint_capacity,
              "effort_time": _EFFORT_TIME, "collision_interval": _restore_metadata(meta[2])}
    for name, shape, data in zip(_CHANNELS, layout.shapes, channels, strict=True):
        values[name] = _Float32Snapshot(shape, data)
    record = _FactoryReadback(tuple((_KEYS[index], values[_KEYS[index]]) for index in order))
    _validate(record, layout)
    return record
