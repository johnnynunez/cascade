"""Read solved MJWarp contact forces; never infer support from candidate pairs.

The admitted Newton 1.6 conversion calls MJWarp 3.12 ``contact_force_fn``
after the solve. Its first three ``contacts.force`` entries are world-frame
newtons ON shape0; the wire contract specifies force ON shape1. Contact
geometry belongs to that solve's constraint evaluation, before integration,
and is labelled by the completed solve, not by a later render or read clock.

Primary implementations: newton/_src/solvers/mujoco/kernels.py
``create_convert_mjw_contacts_to_newton_kernel`` and
mujoco_warp/_src/support.py ``contact_force_fn``. Exact installed source
hashes below admit these semantics; a different implementation is unknown.
No simulation, model mutation, force approximation, or optional SDK import
occurs on module import. The diagnostic API only evaluates existing rows.
"""
from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FOOT_SHAPES = (
    '/World/MicroDuck/Geometry/trunk_base/yaw2roll/hip_l/upper_leg_left/leg/ankle_left/left_foot_collision',
    '/World/MicroDuck/Geometry/trunk_base/bearing_roll/hip_l_2/upper_leg_right/leg_2/ankle_right/right_foot_collision',
)
GROUND_SHAPE = '/World/Ground'
SOURCE_SHA256 = {
    'mujoco_warp._src.support': '1a08ef15f149d1cf3c9ec5df6c688669fe86aa237ee2512a7463caf1b74e3d54',
    'newton._src.solvers.mujoco.kernels': 'bd770d39d20c208e980477e0186799f531364b03f16d76968ee69bacdc490a3d',
    'newton._src.solvers.mujoco.solver_mujoco': '981568ab792dbc9e82392d57da3cbb507ef1c0d8c88e3b8b7cafe5065945c3cf',
    'isaacsim.physics.newton.impl.newton_stage': '08352d2a0e767611461b7dfe75a06ab24a36605f0c806ddba03ff211b1bef47a',
}


def support_contract(shape_labels):
    """Concrete collision identities; an ankle body name is never a sole."""
    labels = tuple(shape_labels)
    if (any(not isinstance(x, str) or not x for x in labels)
            or len(labels) != len(set(labels))
            or not set(FOOT_SHAPES + (GROUND_SHAPE,)).issubset(labels)):
        raise ValueError('support requires the exact admitted sole and ground shapes')
    return dict(version=1,
                robot_shapes=sorted(x for x in labels if x.startswith('/World/MicroDuck/')),
                foot_shapes=list(FOOT_SHAPES), ground_shapes=[GROUND_SHAPE],
                gravity_world_m_s2=[0., 0., -9.81])


def _array(value, dtype, tail, name):
    a = value.numpy()
    if (not isinstance(a, np.ndarray) or a.dtype != dtype or a.ndim != 1 + len(tail)
            or a.shape[1:] != tail):
        raise ValueError(f'invalid support channel {name}')
    return a


def _count(value, limit, name):
    a = _array(value, np.int32, (), name)
    if a.shape != (1,) or not 0 <= a[0] < limit:
        raise ValueError(f'invalid support {name}/capacity')
    return int(a[0])


def _contact_frame_validity(adr, pos, frames, normal, force_on_a, ncon):
    """Batch the unchanged checks, without interpreting inactive storage.

    The decoder still reports failures in candidate order, after checking each
    candidate's constraint rows. This only combines the small NumPy operations;
    it performs no native reads and does not discard any solved contact.
    """
    frame64 = np.empty((ncon, 3, 3), dtype=float)
    valid = np.zeros(ncon, dtype=bool)
    finite_frame = np.zeros(ncon, dtype=bool)
    active = np.flatnonzero(adr[:ncon, 0] >= 0)
    finite_frame[active] = np.isfinite(frames[active]).all(axis=(1, 2))
    convertible = active[finite_frame[active]]
    frame64[convertible] = frames[convertible].astype(float)
    finite = (np.isfinite(pos[active]).all(axis=1)
              & np.isfinite(force_on_a[active]).all(axis=1)
              & np.isfinite(normal[active]).all(axis=1)
              & finite_frame[active])
    selected = active[finite]
    # Non-finite active values fail below; inactive/unused slots may contain
    # NaNs. Neither category is passed to matrix arithmetic or determinants.
    if len(selected):
        f = frame64[selected]
        valid[selected] = (
            np.isclose(f @ f.transpose(0, 2, 1), np.eye(3), rtol=0, atol=2e-5).all(axis=(1, 2))
            & np.isclose(np.linalg.det(f), 1., rtol=0, atol=2e-5)
            & np.isclose(f[:, 0], normal[selected], rtol=0, atol=2e-6).all(axis=1))
    return frame64, valid, finite_frame


@dataclass(frozen=True)
class _ContactSnapshot:
    """Private completed-contact columns, owned bytes rather than SDK views.

    Only the in-process decoder creates these values. This is not a transport
    schema or an API for loading caller-provided serialized objects. Array views
    returned internally are backed by immutable bytes, including after pickle.
    """
    labels: tuple
    count: int
    shape_ids: bytes
    active_indices: bytes
    normal_forces: bytes
    vectors: bytes  # active rows: force-on-B, point, normal; each world xyz

    def __post_init__(self):
        if (type(self.labels) is not tuple or type(self.count) is not int or self.count < 0
                or any(type(v) is not bytes for v in
                       (self.shape_ids, self.active_indices, self.normal_forces, self.vectors))
                or len(self.shape_ids) != self.count * 2 * 4
                or len(self.active_indices) % 4):
            raise ValueError('invalid private contact snapshot layout')
        size = len(self.active_indices) // 4
        if len(self.normal_forces) != size * 8 or len(self.vectors) != size * 9 * 8:
            raise ValueError('invalid private active contact layout')
        shapes, indices, forces, vectors = self._arrays()
        if (size > self.count or (shapes < 0).any() or (shapes >= len(self.labels)).any()
                or (shapes[:, 0] == shapes[:, 1]).any()
                or size and (indices[0] < 0 or indices[-1] >= self.count
                             or (indices[1:] <= indices[:-1]).any())
                or not np.isfinite(forces).all() or (forces < 0).any()
                or not np.isfinite(vectors).all()):
            raise ValueError('invalid private contact snapshot values')

    def _arrays(self):
        size = len(self.active_indices) // 4
        return (np.frombuffer(self.shape_ids, dtype=np.int32).reshape(self.count, 2),
                np.frombuffer(self.active_indices, dtype=np.int32),
                np.frombuffer(self.normal_forces, dtype=np.float64),
                np.frombuffer(self.vectors, dtype=np.float64).reshape(size, 3, 3))

    def _solved_record(self, shapes, index, force, vectors):
        a, b = map(int, shapes[index])
        # Preserve the public insertion order and Python scalar/list types.
        return dict(shape_a=self.labels[a], shape_b=self.labels[b], shape_a_id=a, shape_b_id=b,
                    force_on_b_world_n=vectors[0].tolist(), normal_force_n=float(force),
                    point_world_m=vectors[1].tolist(), normal_a_to_b_world=vectors[2].tolist())

    def solved_records(self):
        shapes, indices, forces, vectors = self._arrays()
        return [self._solved_record(shapes, int(index), force, vector)
                for index, force, vector in zip(indices, forces, vectors, strict=True)]

    def factory_records(self):
        shapes, indices, forces, vectors = self._arrays()
        result, active = [], 0
        for index in range(self.count):
            if active < len(indices) and indices[active] == index:
                record = self._solved_record(shapes, index, forces[active], vectors[active])
                record.update(candidate=index, status='solved')
                active += 1
            else:
                a, b = map(int, shapes[index])
                record = dict(candidate=index, status='inactive_candidate',
                    shape_a=self.labels[a], shape_b=self.labels[b], normal_force_n=0.,
                    point_world_m=None, normal_a_to_b_world=None, force_on_b_world_n=None)
            result.append(record)
        return result


def solved_contacts(ns):
    """Public detached rows; private owners can retain validated columns."""
    return _solved_contact_snapshot(ns).solved_records()


def _solved_contact_snapshot(ns):
    """Decode identities and read forces from ONE completed native solve.

    Validate coverage of every contact-constraint row. Zero normal force is a
    known zero, not support; no active row is a known empty observation only
    when every channel and its capacity is available and coherent.
    """
    solver, contacts = ns.solver, ns.contacts
    d = solver.mjw_data
    labels = tuple(ns.model.shape_label)
    if len(labels) != len(set(labels)):
        raise ValueError('duplicate native shape identity')
    njmax, naconmax = int(d.njmax), int(d.naconmax)
    nefc = _count(d.nefc, njmax, 'constraint count')
    ncon = _count(d.nacon, naconmax, 'contact count')
    if _count(contacts.rigid_contact_count, naconmax, 'Newton contact count') != ncon:
        raise ValueError('native contact count mismatch')
    cone = int(solver.mjw_model.opt.cone)
    if cone not in (0, 1):
        raise ValueError('unsupported contact cone')
    pairs = [_array(getattr(contacts, k), np.int32, (), k)
             for k in ('rigid_contact_shape0', 'rigid_contact_shape1')]
    worlds = _array(d.contact.worldid, np.int32, (), 'contact.worldid')
    geoms = _array(d.contact.geom, np.int32, (2,), 'contact.geom')
    dims = _array(d.contact.dim, np.int32, (), 'contact.dim')
    pos = _array(d.contact.pos, np.float32, (3,), 'contact.pos')
    frames = _array(d.contact.frame, np.float32, (3, 3), 'contact.frame')
    normal = _array(contacts.rigid_contact_normal, np.float32, (3,), 'contact normal')
    force_on_a = _array(contacts.force, np.float32, (6,), 'solved force')
    adr = d.contact.efc_address.numpy()
    mapping = solver.mjc_geom_to_newton_shape.numpy()
    types, ids, efc_force = (getattr(d.efc, k).numpy() for k in ('type', 'id', 'force'))
    for name, a in (('type', types), ('id', ids), ('force', efc_force)):
        dtype = np.float32 if name == 'force' else np.int32
        if (not isinstance(a, np.ndarray) or a.dtype != dtype or a.ndim != 2
                or a.shape[0] != 1 or a.shape[1] < nefc or not np.isfinite(a[0, :nefc]).all()):
            raise ValueError(f'invalid support constraint {name}')
    if (not isinstance(adr, np.ndarray) or adr.dtype != np.int32 or adr.ndim != 2
            or adr.shape[0] < ncon or adr.shape[1] < 1
            or not isinstance(mapping, np.ndarray) or mapping.dtype != np.int32
            or mapping.ndim != 2 or mapping.shape[0] != 1):
        raise ValueError('invalid support address/geom mapping layout')
    if any(len(a) < ncon for a in (*pairs, worlds, geoms, dims, pos, frames, normal, force_on_a)):
        raise ValueError('truncated contact channels')
    if ncon and ((worlds[:ncon] != 0).any() or (geoms[:ncon] < 0).any()
                 or (geoms[:ncon] >= mapping.shape[1]).any()):
        raise ValueError('invalid support contact world/geom')
    frames64, valid_frames, finite_frames = _contact_frame_validity(adr, pos, frames, normal, force_on_a, ncon)
    # Addresses describe contiguous rows. Keep one local coverage mask and
    # index vector per observation instead of allocating index arrays, gathered
    # channel copies and Python row lists/sets for every solved contact.
    row_indices = np.arange(nefc)
    covered = np.zeros(nefc, dtype=bool)
    active_indices = np.empty(ncon, dtype=np.int32)
    normal_forces = np.empty(ncon, dtype=np.float64)
    vectors = np.empty((ncon, 3, 3), dtype=np.float64)
    active_count = 0
    for i in range(ncon):
        pair = (int(pairs[0][i]), int(pairs[1][i]))
        if (min(pair) < 0 or max(pair) >= len(labels) or pair[0] == pair[1]
                or tuple(mapping[0, geoms[i]]) != pair):
            raise ValueError('contact geom/shape identity mismatch')
        first = int(adr[i, 0])
        if first == -1:
            continue
        dim = int(dims[i])
        if dim not in (1, 3, 4, 6):
            raise ValueError('unsupported contact dimension')
        nrows = 2 * (dim - 1) if cone == 0 and dim > 1 else dim
        expected_type = 5 if dim == 1 else (6 if cone == 0 else 7)
        end = first + nrows
        rows = slice(first, end)
        if (first < 0 or end > nefc or adr.shape[1] < nrows
                or not np.array_equal(adr[i, :nrows], row_indices[rows])
                or (types[0, rows] != expected_type).any() or (ids[0, rows] != i).any()
                or covered[rows].any()):
            raise ValueError('contact constraint rows are incomplete or ambiguous')
        covered[rows] = True
        if finite_frames[i]:
            frame = frames64[i]
            if not valid_frames[i]:
                raise ValueError('invalid contact frame/normal')
        else:
            # Preserve the original scalar failure/warning order for malformed
            # active frames, including signaling NaNs under strict NumPy error
            # settings. Inactive storage never reaches a float conversion.
            frame = frames[i].astype(float)
            if (not np.isfinite(pos[i]).all() or not np.isfinite(force_on_a[i]).all()
                    or not np.isfinite(normal[i]).all()
                    or not np.allclose(frame @ frame.T, np.eye(3), rtol=0, atol=2e-5)
                    or not np.isclose(np.linalg.det(frame), 1., rtol=0, atol=2e-5)
                    or not np.allclose(frame[0], normal[i], rtol=0, atol=2e-6)):
                raise ValueError('invalid contact frame/normal')
        # The admitted native decoder sums the pyramid edges for the normal;
        # elliptic/frictionless contact stores its normal in the first row.
        # Preserve the original contiguous reduction order even for a strided
        # diagnostic input. Ordinary native buffers already provide this view.
        row_forces = np.ascontiguousarray(efc_force[0, rows])
        normal_force = float(np.sum(row_forces, dtype=np.float64)
                             if cone == 0 and dim > 1 else efc_force[0, first])
        force_b = -force_on_a[i, :3].astype(float)
        if (normal_force < 0 or force_b @ frame[0] < 0
                or (cone == 0 and (row_forces < 0).any())
                or not np.isclose(force_b @ frame[0], normal_force,
                                  rtol=1e-5, atol=1e-7)):
            raise ValueError('solved normal/force disagreement')
        active_indices[active_count] = i
        normal_forces[active_count] = normal_force
        vectors[active_count, 0] = force_b
        vectors[active_count, 1] = pos[i]
        vectors[active_count, 2] = frame[0]
        active_count += 1
    required = np.isin(types[0, :nefc], (5, 6, 7))
    if not np.array_equal(covered, required):
        raise ValueError('contact observation does not cover all solved contact rows')
    return _ContactSnapshot(labels, ncon, np.column_stack((pairs[0][:ncon], pairs[1][:ncon])).tobytes(),
                            active_indices[:active_count].tobytes(), normal_forces[:active_count].tobytes(),
                            vectors[:active_count].tobytes())


def extraction_provenance(*, sdk_recipe=None):
    """Read the actually imported SDK sources; never substitute a version guess."""
    actual = {}
    expected = SOURCE_SHA256
    if sdk_recipe is not None:
        from .microduck_sdk import INTERNAL_SOURCE_SHA256, verify_runtime_recipe
        version = importlib.import_module('newton').__version__
        actual = verify_runtime_recipe(sdk_recipe, newton_version=version)
        expected = INTERNAL_SOURCE_SHA256
    else:
        for name in SOURCE_SHA256:
            module = importlib.import_module(name)
            path = Path(module.__file__)
            actual[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = dict(version=1, source_sha256=actual, source_admitted=actual == expected,
                source='Newton contacts.force via MJWarp contact_force_fn',
                native_force_on='shape0', emitted_force_on='shape1', frame='world',
                force_unit='N', position_unit='m', normal='shape0_to_shape1',
                geometry_time='constraint evaluation of the completed solve, before integration',
                clock='completed manual solve; unavailable after initialization pose writes',
                coverage='all active contact constraint rows, including zero-force contacts')
    if sdk_recipe is not None:
        result['sdk_recipe'] = sdk_recipe
    return result


def read_support(ns, *, last_solved_clock, source_admitted):
    """Return unknown on unavailable/stale force evidence, never a fake empty set."""
    clock = (ns.simulation_step_count, float(ns.sim_time))
    result = dict(version=1, status='unavailable', reason='', step=clock[0],
                  sim_time_s=clock[1], contacts=[])
    try:
        if not source_admitted:
            raise ValueError('contact extraction SDK source is not admitted')
        if last_solved_clock != clock:
            raise ValueError('no matching completed solve after initialization')
        records = solved_contacts(ns)
        if (ns.simulation_step_count, float(ns.sim_time)) != clock:
            raise ValueError('physics advanced during support read')
        result.update(status='known', contacts=records)
    except (AttributeError, IndexError, TypeError, ValueError, RuntimeError, OverflowError) as exc:
        result['reason'] = f'{type(exc).__name__}: {exc}'
    return result


def compare_native_force_api(ns):
    """Optional read-only diagnostic; no simulation/model/target writes.

    Evaluate MJWarp's public contact_force on the current rows and compare its
    force-on-B vectors with Newton's converted force-on-A channel. Root can
    sample this after touchdown to independently check extraction/signs.
    """
    import mujoco_warp
    import warp as wp
    clock = (ns.simulation_step_count, float(ns.sim_time))
    ncon = int(ns.solver.mjw_data.nacon.numpy()[0])
    ids = wp.array(np.arange(ncon, dtype=np.int32), dtype=wp.int32, device=ns.model.device)
    output = wp.zeros(ncon, dtype=wp.spatial_vector, device=ns.model.device)
    if ncon:
        mujoco_warp.contact_force(ns.solver.mjw_model, ns.solver.mjw_data, ids, True, output)
    actual = output.numpy()
    converted = ns.contacts.force.numpy()[:ncon]
    if actual.shape != (ncon, 6) or converted.shape != actual.shape:
        raise ValueError('contact diagnostic force layout mismatch')
    if (ns.simulation_step_count, float(ns.sim_time)) != clock:
        raise ValueError('physics advanced during contact diagnostic')
    error = float(np.max(np.abs(actual + converted))) if ncon else 0.
    return dict(step=clock[0], sim_time_s=clock[1], candidate_count=ncon,
                max_force_torque_difference=error, passed=bool(np.isfinite(error) and error <= 2e-6),
                public_force_on_b_world=actual.tolist(), converted_force_on_a_world=converted.tolist())
