"""Explicit snapshot schema and unchanged physical vetoes; no SDK or models."""
import json
import math
import struct
from dataclasses import fields, replace

import numpy as np
import pytest
from test_fastening_runtime import binding, limits, row

from cascade.control.fastening import (
    FasteningFault,
    FasteningUpload,
    SolvedPair,
    check_solve,
)
from cascade.control.fastening_seat import SeatingSolve, ShoulderContact
from cascade.sim.factory_observation import (
    _FactoryReadback,
    _Float32Snapshot,
    _RawMetadata,
)
from cascade.sim.factory_process_protocol import FaultState, RawOutbox
from cascade.sim.factory_process_snapshot import (
    _COLLISION_NAMES,
    _EFFORT_TIME,
    SnapshotLayout,
    decode_snapshot,
    encode_snapshot,
)
from cascade.sim.factory_process_wire import WireFault, decode, encode
from cascade.sim.microduck_contact_support import _ContactSnapshot


def fixture(*, count=2, capacity=128, seating=False, inactive=True, disabled=True):
    labels = binding().collider_names
    shape_ids = np.tile(np.array([[0, 1]], np.int32), (count, 1))
    indices = np.arange(0, count, 2 if inactive else 1, dtype=np.int32)
    forces = np.ones(len(indices), np.float64)
    vectors = np.zeros((len(indices), 3, 3), np.float64)
    vectors[:, 0, 2] = forces
    vectors[:, 1, 0] = -0.
    vectors[:, 2, 2] = 1.
    snapshot = _ContactSnapshot(labels, count, shape_ids.tobytes(), indices.tobytes(),
                                forces.tobytes(), vectors.tobytes())
    active = set(map(int, indices))
    value = row(1, collision_capacity=capacity, solver_capacity=capacity,
                collision_count=count, solver_count=count, thread_contacts=len(active), tool_contacts=0,
                contacts=tuple(SolvedPair("nut", "bolt", 1. if i in active else 0.) for i in range(count)))
    if seating:
        value = SeatingSolve(**{f.name: getattr(value, f.name) for f in fields(value)},
            shoulder_contacts=tuple(ShoulderContact(i, (-0., 0., .023), (0., 0., 1.), 1., (0., 0., 1.))
                                    for i in sorted(active)))
    capacities = tuple((name, 0 if name == "shape_pairs_mesh_plane" else capacity) for name in _COLLISION_NAMES)
    layout = SnapshotLayout(labels, 1, 2, 3, 2, capacity, capacity, 4096,
                            value.binding_sha256, value.epoch, capacities)
    buffers = {name: {"count": count if name == "contacts" else 0, "capacity": cap}
               for name, cap in reversed(capacities)}
    buffers["reducer"]["insert_failures"] = 0
    if disabled:
        buffers["shape_pairs_mesh_plane"]["disabled"] = True
    data = {"solve": value, "upload": FasteningUpload(0, 0, -0., 9.9, 9.91),
            "effort_time": _EFFORT_TIME, "collision_buffers": buffers, "contacts": snapshot,
            "constraint_count": 3 * len(active), "constraint_capacity": 4096}
    names = ("native_ctrl", "actuator_force", "qfrc_actuator", "qfrc_applied",
             "qfrc_constraint", "qfrc_passive", "body_poses_xyzw")
    for name, shape in zip(names, layout.shapes, strict=True):
        data[name] = _Float32Snapshot.capture(np.full(shape, -0., np.float32))
    data["native_clock"] = {"step": 1, "time_s": float(np.float32(.01)), "timestep_s": float(np.float32(.01)),
                            "interval_clock_s": .01, "cuda_graph": False}
    data["collision_interval"] = {"before_step": 0, "generation": 0, "after_step": 1}
    return layout, _FactoryReadback.capture(data)


def exact(value):
    """Independent tree comparison includes scalar types/order and float bits."""
    if type(value) is float:
        return (float, struct.pack("<d", value))
    if type(value) is dict:
        return (dict, tuple((exact(k), exact(v)) for k, v in value.items()))
    if type(value) in (tuple, list):
        return (type(value), tuple(map(exact, value)))
    return (type(value), value)


@pytest.mark.parametrize("seating", [False, True])
@pytest.mark.parametrize("disabled", [False, True])
def test_snapshot_roundtrip_keeps_complete_public_types_order_bits_and_detaches(seating, disabled):
    layout, original = fixture(seating=seating, disabled=disabled)
    # Insertion order is part of the retained public record, not a canonical sort.
    original = _FactoryReadback(tuple(reversed(original.entries)))
    expected = original.materialize()
    packet = encode_snapshot(original, layout)
    restored = decode_snapshot(packet, layout)
    actual = restored.materialize()
    assert exact(actual) == exact(expected)
    assert json.dumps(actual, allow_nan=False) == json.dumps(expected, allow_nan=False)
    assert encode_snapshot(restored, layout) == packet
    actual["contacts"][0]["force_on_b_world_n"][0] = 900.
    actual["native_ctrl"][0] = 123.
    assert exact(restored.materialize()) == exact(expected)
    assert math.copysign(1., restored.materialize()["native_ctrl"][0]) == -1.


@pytest.mark.parametrize("failure", ["stale", "geometry", "speed", "thread_count"])
def test_raw_roundtrip_preserves_later_veto_and_original_capture_time(failure):
    layout, original = fixture()
    values = dict(original.entries)
    changes = {"stale": {"captured_monotonic_s": 9.7},
               "geometry": {"geometry_max_m": (1.1, .1, .2)},
               "speed": {"joint_velocity_rad_s": (.81,)},
               "thread_count": {"thread_contacts": 0}}[failure]
    values["solve"] = replace(values["solve"], **changes)
    original = _FactoryReadback(tuple(values.items()))
    payload = encode_snapshot(original, layout)
    restored = dict(decode_snapshot(payload, layout).entries)["solve"]
    state = FaultState(boot_id="a"*32, epoch="b"*32, binding="c"*64)
    outbox = RawOutbox(4, 4*layout.maximum_frame_bytes, layout.maximum_frame_bytes, state)
    with pytest.raises(FasteningFault) as expected:
        check_solve(values["solve"], binding(), limits(), 10.)
    with pytest.raises(type(expected.value)) as actual:
        outbox.transact(payload, lambda: check_solve(restored, binding(), limits(), 10.))
    assert str(actual.value) == str(expected.value)
    assert restored.captured_monotonic_s == values["solve"].captured_monotonic_s
    retained, = outbox.prefix(4)
    assert retained[1] == payload and retained[2] == RawOutbox.REJECTED


@pytest.mark.parametrize("failure", ["layout", "order", "columns", "force_nan", "metadata",
                                      "channel", "epoch", "generation", "constraint", "clock"])
def test_malformed_schema_or_cross_binding_cannot_decode(failure):
    layout, original = fixture(seating=True)
    value = list(decode(encode_snapshot(original, layout), layout.wire_limits))
    if failure == "layout": value[1] = "b"*64
    elif failure == "order": value[2] = (0,) * len(value[2])
    elif failure == "columns": value[5] = (*value[5][:1], b"", *value[5][2:])
    elif failure == "force_nan": value[5] = (*value[5][:3], struct.pack("<d", float("nan")), value[5][4])
    elif failure == "metadata": value[8] = b"n"
    elif failure == "channel": value[7] = (b"", *value[7][1:])
    elif failure == "epoch": value[3] = (value[3][0], "different", *value[3][2:])
    elif failure == "generation": value[6] = (value[6][0]+1, *value[6][1:])
    elif failure == "constraint": value[9] = layout.constraint_capacity
    elif failure == "clock":
        values = dict(original.entries)
        clock = dict(values["native_clock"].entries)
        clock["interval_clock_s"] = .02
        values["native_clock"] = _RawMetadata.capture(clock)
        with pytest.raises(WireFault, match="clock"):
            encode_snapshot(_FactoryReadback(tuple(values.items())), layout)
        return
    with pytest.raises((WireFault, ValueError, TypeError)):
        decode_snapshot(encode(tuple(value), layout.wire_limits), layout)


@pytest.mark.parametrize("disabled", [False, 1, "true"])
def test_disabled_flag_requires_exact_true_and_zero_capacity(disabled):
    layout, record = fixture()
    values = dict(record.entries)
    buffers = values["collision_buffers"].materialize()
    buffers["shape_pairs_mesh_plane"]["disabled"] = disabled
    values["collision_buffers"] = _RawMetadata.capture(buffers)
    with pytest.raises(WireFault):
        encode_snapshot(_FactoryReadback(tuple(values.items())), layout)


def test_full_contact_and_witness_frame_fits_explicit_capacity_bound():
    layout, record = fixture(count=2047, capacity=2048, seating=True, inactive=False)
    # Constraint capacity is independent of the contact count. This synthetic
    # schema fixture does not assert that 2047 shoulder witnesses occur natively.
    layout = replace(layout, constraint_capacity=16384)
    values = dict(record.entries)
    values["constraint_capacity"] = 16384
    record = _FactoryReadback(tuple(values.items()))
    payload = encode_snapshot(record, layout)
    assert len(payload) <= layout.maximum_frame_bytes
    assert exact(decode_snapshot(payload, layout).materialize()) == exact(record.materialize())
