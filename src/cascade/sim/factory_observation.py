"""Pinned Newton/MJWarp per-solve readback for the mounted Factory fixture.

No SDK import, device allocation or simulation occurs at module import. This
adapter reads a completed solve owned by one thread. It does not establish the
yet-unperformed native admission of the runtime/profile. Contact geometry is
from constraint evaluation; body poses and velocities are post-integration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..control.fastening import FasteningFault, FasteningSolve, SolvedPair
from .microduck_contact_support import SOURCE_SHA256, solved_contacts


SDK_SOURCES = {name: digest for name, digest in SOURCE_SHA256.items()
               if not name.startswith("isaacsim.")}
SDK_SOURCES.update({
    "mujoco_warp._src.forward": "7b7ffa3d914b6adf65f228261e207998d5bed1e59d4fb17f5bc6ac796aacf490",
    "newton._src.sim.collide": "dcc29f631d22dbf4c689e7424e1be67a22b5cc147e4b728a0845c92db760a44b",
    "newton._src.geometry.narrow_phase": "c13ac4b76d6a128a9f73cb44d8ed10f5e51224340606b2eb0b90bceb7f2cba22",
    "newton._src.geometry.contact_reduction_global": "09ccc3c5a8b807fc9780900d0ebd95dc7cdb19f51592b0014e8fe4fd01379251",
})


def sdk_sources(sdk_recipe=None):
    """Check actual loaded implementation, not a caller's complete=True flag."""
    from .factory_sdk import selected_pins
    found = {}
    for name, expected in selected_pins(SDK_SOURCES, sdk_recipe).items():
        path = Path(importlib.import_module(name).__file__).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise FasteningFault(f"unadmitted Factory readback SDK: {name}")
        found[name] = {"path": str(path), "sha256": digest}
    return found


def _integer(buffer, name):
    array = buffer.numpy()
    if array.dtype != np.int32 or array.shape != (1,) or array[0] < 0:
        raise FasteningFault(f"invalid native counter: {name}")
    return int(array[0])


def _floats(buffer, shape, name):
    array = buffer.numpy()
    if array.dtype != np.float32 or array.shape != shape or not np.isfinite(array).all():
        raise FasteningFault(f"missing, invalid or non-finite native channel: {name}")
    return array


def collision_coverage(pipeline, contacts):
    """All counters checked by the pinned diagnostic kernel, but fail closed.

    At-capacity is conservatively refused even if not yet an SDK overflow.
    Disabled buffers must have zero count. Hydroelastic is not this recipe.
    """
    np_phase = pipeline.narrow_phase
    if np_phase.hydroelastic_sdf is not None:
        raise FasteningFault("hydroelastic coverage is not implemented")
    result = {}

    def check(name, counter, backing):
        if counter is None and backing is None:
            result[name] = {"count": 0, "capacity": 0, "disabled": True}
            return
        if counter is None:
            raise FasteningFault(f"missing native counter: {name}")
        count = _integer(counter, name)
        capacity = 0 if backing is None else int(backing.shape[0])
        if count and capacity == 0 or capacity and count >= capacity:
            raise FasteningFault(f"collision buffer has no headroom: {name} {count}/{capacity}")
        result[name] = {"count": count, "capacity": capacity}

    check("broad_phase", pipeline.broad_phase_pair_count, pipeline.broad_phase_shape_pairs)
    check("gjk", np_phase.gjk_candidate_pairs_count, np_phase.gjk_candidate_pairs)
    if np_phase.split_gjk_mpr:
        query_count = (pipeline.broad_phase_pair_count if np_phase.sparse_gjk_pairs
                       else np_phase.gjk_candidate_pairs_count)
        check("split_query", query_count, np_phase.split_query_results)
        check("split_gjk", np_phase.split_gjk_work_count, np_phase.split_gjk_work_items)
        check("split_manifold", np_phase.split_manifold_work_count, np_phase.split_manifold_work_items)
    for stem in ("shape_pairs_mesh", "triangle_pairs", "shape_pairs_mesh_plane",
                 "shape_pairs_mesh_mesh", "shape_pairs_sdf_sdf"):
        check(stem, getattr(np_phase, stem + "_count"), getattr(np_phase, stem))
    check("contacts", contacts.rigid_contact_count, contacts.rigid_contact_shape0)
    if int(contacts.rigid_contact_max) != result["contacts"]["capacity"]:
        raise FasteningFault("collision contact capacity differs from backing storage")
    reducer = np_phase.global_contact_reducer
    if reducer is not None:
        capacity = int(reducer.hashtable.capacity)
        slots = reducer.hashtable.active_slots.numpy()
        if capacity <= 0 or slots.dtype != np.int32 or slots.shape != (capacity + 1,):
            raise FasteningFault("invalid contact reducer capacity/counter")
        count = int(slots[capacity])
        failures = _integer(reducer.ht_insert_failures, "reducer insertion failures")
        # Same 80% critical-load threshold as the admitted SDK warning kernel.
        if count < 0 or count * 100 >= capacity * 80 or failures:
            raise FasteningFault("contact reducer coverage is unavailable")
        result["reducer"] = {"count": count, "capacity": capacity, "insert_failures": failures}
    return result


def world_aabb(local_aabb, positions, rotations):
    """MuJoCo geom_aabb = local center + HALF extents, in geom frame.

    Rotating all half extents by abs(R) contains all eight box corners. No
    second factor of one half, shape-origin assumption, or pose write is used.
    """
    a, p, r = (np.asarray(x, dtype=float) for x in (local_aabb, positions, rotations))
    n = len(a)
    if (n == 0 or a.shape != (n, 6) or p.shape != (n, 3) or r.shape != (n, 3, 3)
            or not all(np.isfinite(x).all() for x in (a, p, r)) or (a[:, 3:] < 0).any()
            or not np.allclose(r @ r.transpose(0, 2, 1), np.eye(3), rtol=0, atol=2e-5)
            or not np.allclose(np.linalg.det(r), 1., rtol=0, atol=2e-5)):
        raise FasteningFault("invalid collision geometry transform/AABB")
    centers = p + np.einsum("nij,nj->ni", r, a[:, :3])
    halves = np.einsum("nij,nj->ni", np.abs(r), a[:, 3:])
    return tuple(map(float, (centers-halves).min(0))), tuple(map(float, (centers+halves).max(0)))


@dataclass(frozen=True)
class JointMap:
    name: str
    newton_joint: int
    newton_q: int
    newton_dof: int
    native_joint: int
    native_q: int
    native_dof: int
    native_actuator: int
    control_index: int
    reference_rad: float


def joint_mapping(scene, names):
    """Exactly one unit-gear hinge actuator per controlled DOF, explicit maps."""
    s, m = scene.solver, scene.solver.mj_model
    jmap = s.mjc_jnt_to_newton_jnt.numpy()
    dmap = s.mjc_dof_to_newton_dof.numpy()
    source = s.mjc_actuator_ctrl_source.numpy()
    indices = s.mjc_actuator_to_newton_idx.numpy()
    qstart, dstart = scene.model.joint_q_start.numpy(), scene.model.joint_qd_start.numpy()
    ref = getattr(scene.model.mujoco, "dof_ref", None)
    ref = np.zeros(scene.model.joint_dof_count) if ref is None else ref.numpy()
    if jmap.shape != (1, m.njnt) or dmap.shape != (1, m.nv):
        raise FasteningFault("only a single-world joint map is implemented")
    result = []
    for name in names:
        nj = scene.joints[name]
        matches = np.flatnonzero(jmap[0] == nj)
        if len(matches) != 1:
            raise FasteningFault(f"ambiguous native joint: {name}")
        j = int(matches[0])
        dof, q = int(m.jnt_dofadr[j]), int(m.jnt_qposadr[j])
        actuators = np.flatnonzero(m.actuator_trnid[:, 0] == j)
        if int(m.jnt_type[j]) != 3 or len(actuators) != 1:  # mjJNT_HINGE
            raise FasteningFault(f"not a singly actuated hinge: {name}")
        actuator = int(actuators[0])
        if (int(m.actuator_trntype[actuator]) != 0 or source[actuator] != 1
                or indices[actuator] != scene._actuators[name if name != "socket_spin" else "socket_motor"]
                or not np.array_equal(m.actuator_gear[actuator], [1., 0., 0., 0., 0., 0.])
                or int(dmap[0, dof]) != int(dstart[nj])):
            raise FasteningFault(f"unsupported transmission/control mapping: {name}")
        result.append(JointMap(name, int(nj), int(qstart[nj]), int(dstart[nj]), j, q, dof,
                               actuator, int(indices[actuator]), float(ref[dstart[nj]])))
    if len({r.native_dof for r in result}) != len(result) or len(result) != m.nu:
        raise FasteningFault("unaccounted or duplicate actuator DOF")
    return tuple(result)


def actuator_descriptor(model, joints):
    """Admit stateless position servos and ONE direct gear-1 spindle motor.

    Exact numerical gains and imported joint/actuator ranges are bound in the
    returned model document. They are not replaced by guessed servo constants.
    The spindle alone has torque input; arm input remains a position target.
    """
    if np.any(model.body_gravcomp) or np.any(model.jnt_actgravcomp):
        raise FasteningFault("unadmitted actuator gravity compensation")
    records = []
    for j in joints:
        a = j.native_actuator
        gain, bias = model.actuator_gainprm[a], model.actuator_biasprm[a]
        spindle = j.name == "socket_spin"
        if (int(model.actuator_dyntype[a]) != 0 or int(model.actuator_gaintype[a]) != 0
                or int(model.actuator_actnum[a]) != 0
                or not np.isfinite(gain).all() or not np.isfinite(bias).all()
                or not model.actuator_ctrllimited[a] or not model.actuator_forcelimited[a]):
            raise FasteningFault("unsupported actuator dynamics, gain or unbounded effort")
        if spindle:
            if (int(model.actuator_biastype[a]) != 0 or not np.array_equal(gain, [1.]+[0.]*(len(gain)-1))
                    or np.any(bias) or not np.allclose(model.actuator_ctrlrange[a], [-.05, .05], rtol=0, atol=1e-9)
                    or not np.allclose(model.actuator_forcerange[a], [-.05, .05], rtol=0, atol=1e-9)):
                raise FasteningFault("spindle is not the bounded direct M20 fixture motor")
        elif (int(model.actuator_biastype[a]) != 1 or gain[0] <= 0 or np.any(gain[1:])
              or bias[0] != 0 or bias[1] != -gain[0] or bias[2] > 0 or np.any(bias[3:])):
            raise FasteningFault("arm actuator is not an ordinary bounded position servo")
        records.append({**asdict(j), "kind": "direct_effort" if spindle else "position_servo",
            "gain": gain.tolist(), "bias": bias.tolist(),
            "joint_axis_local": model.jnt_axis[j.native_joint].tolist(),
            "control_range": model.actuator_ctrlrange[a].tolist(),
            "force_range": model.actuator_forcerange[a].tolist(),
            "joint_range": model.jnt_range[j.native_joint].tolist(),
            "joint_limited": bool(model.jnt_limited[j.native_joint]),
            "joint_aggregate_force_limited": bool(model.jnt_actfrclimited[j.native_joint]),
            "joint_aggregate_force_range": model.jnt_actfrcrange[j.native_joint].tolist(),
            "effort_unit": "Nm", "positive_axis": "native hinge local axis",
            "native_effort_channel": "qfrc_actuator[world,native_dof]"})
    return records


class FactoryGeometry:
    """Read-only shadow FK, never writes the native body's state or joint arrays."""

    def __init__(self, scene, joints):
        self.scene, self.joints = scene, joints
        self.model = scene.solver.mj_model
        self.shadow = scene.mujoco.MjData(self.model)
        mapping = scene.solver.mjc_geom_to_newton_shape.numpy()
        if mapping.dtype != np.int32 or mapping.shape != (1, self.model.ngeom):
            raise FasteningFault("invalid geometry mapping")
        mapping = mapping[0]
        flags = scene.model.shape_flags.numpy()
        collision_bit = int(scene.newton.ShapeFlags.COLLIDE_SHAPES)
        bodies = scene.model.shape_body.numpy()
        bodymap = scene.solver.mjc_body_to_newton.numpy()
        if (bodymap.dtype != np.int32 or bodymap.shape != (1, self.model.nbody)
                or (bodymap < -1).any() or (bodymap >= scene.model.body_count).any()):
            raise FasteningFault("invalid body mapping layout/range")
        bound_bodies = bodymap[0, bodymap[0] >= 0].tolist()
        if sorted(bound_bodies) != list(range(scene.model.body_count)):
            raise FasteningFault("body mapping omits or duplicates Newton bodies")
        moving = []
        for geom, shape in enumerate(mapping):
            if shape < 0 or shape >= len(flags):
                raise FasteningFault("unknown compiled geom")
            body = int(self.model.geom_bodyid[geom])
            if int(bodymap[0, body]) != int(bodies[shape]):
                raise FasteningFault("geom/body/shape frame mapping disagrees")
            if flags[shape] & collision_bit and self.model.body_weldid[body] != 0 and bodies[shape] != scene.nut_body:
                moving.append(geom)
        self.geoms = np.asarray(moving, dtype=int)
        expected_collision = set(np.flatnonzero(flags & collision_bit).tolist())
        compiled_collision = [int(i) for i in mapping if flags[i] & collision_bit]
        if (set(compiled_collision) != expected_collision
                or len(compiled_collision) != len(expected_collision) or not moving):
            raise FasteningFault("moving collision geometry is missing or duplicated")
        self.native_body_to_newton = bodymap[0].copy()
        self.descriptor = {"moving_geoms": moving,
            "moving_shapes": [scene.model.shape_label[mapping[g]] for g in moving],
            "aabb_center_halfsize": self.model.geom_aabb[self.geoms].tolist(),
            "geom_bodyid": self.model.geom_bodyid[self.geoms].tolist(),
            "body_mapping": self.native_body_to_newton.tolist(),
            "position_time": "completed solve, after integration",
            "velocity": "world COM linear xyz; world angular xyz, m/s and rad/s"}

    def evaluate(self, qpos, *, target=None, body_poses=None):
        value = np.asarray(qpos, dtype=float)
        if value.shape != (self.model.nq,) or not np.isfinite(value).all():
            raise FasteningFault("invalid native qpos for shadow FK")
        self.shadow.qpos[:] = value
        if target is not None:
            if len(target) != len(self.joints)-1:
                raise FasteningFault("target/FK joint dimension mismatch")
            for row, q in zip(self.joints[:-1], target, strict=True):
                self.shadow.qpos[row.native_q] = q + row.reference_rad
        self.scene.mujoco.mj_kinematics(self.model, self.shadow)
        if body_poses is not None:
            body_poses = np.asarray(body_poses)
            if (body_poses.shape != (self.scene.model.body_count, 7)
                    or not np.isfinite(body_poses).all()
                    or not np.allclose(np.sum(body_poses[:, 3:]**2, axis=1), 1., rtol=0, atol=2e-5)):
                raise FasteningFault("invalid or non-unit Newton body orientation")
            for native_body, newton_body in enumerate(self.native_body_to_newton):
                if newton_body < 0:
                    continue
                p = body_poses[newton_body]
                q = self.shadow.xquat[native_body][[1, 2, 3, 0]]
                if (not np.allclose(p[:3], self.shadow.xpos[native_body], rtol=0, atol=2e-5)
                        or not np.isclose(abs(np.dot(p[3:], q)), 1., rtol=0, atol=2e-5)):
                    raise FasteningFault("native/Newton body FK frame disagreement")
        return world_aabb(self.model.geom_aabb[self.geoms], self.shadow.geom_xpos[self.geoms],
                          self.shadow.geom_xmat[self.geoms].reshape(-1, 3, 3))


class _ZeroLoadPairs:
    """Reuse immutable zero values, never candidate records or observations."""

    def __init__(self):
        self._values = {}

    def pair(self, collider_a, collider_b, normal_force_n):
        if (type(collider_a) is not str or type(collider_b) is not str
                or type(normal_force_n) is not float or normal_force_n != 0.):
            return SolvedPair(collider_a, collider_b, normal_force_n)
        # Preserve ordering and signed zero, including their raw JSON values.
        key = (collider_a, collider_b, math.copysign(1., normal_force_n))
        value = self._values.get(key)
        if value is None:
            value = SolvedPair(collider_a, collider_b, normal_force_n)
            if len(self._values) < 256:
                self._values[key] = value
        return value


def contact_records(scene, output, *, _zero_pairs=None):
    """Decode all solver candidates; inactive candidates have no solved load.

    Active rows retain vector force, actual point and normal. An inactive
    candidate has a known absent constraint, not a fabricated contact point.
    The shared decoder checks identity for every candidate, including inactive.
    """
    solver = scene.solver
    if _integer(solver.mjw_data.overflow, "solver overflow"):
        raise FasteningFault("native constraint/contact overflow")
    solver.update_contacts(output)  # Conversion only; does not call step/forward.
    active = iter(solved_contacts(SimpleNamespace(model=scene.model, solver=solver, contacts=output)))
    ncon = _integer(solver.mjw_data.nacon, "solver contacts")
    addresses = solver.mjw_data.contact.efc_address.numpy()[:ncon, 0]
    shapes = [getattr(output, "rigid_contact_shape" + str(i)).numpy()[:ncon] for i in (0, 1)]
    pair = SolvedPair if _zero_pairs is None else _zero_pairs.pair
    pairs, records = [], []
    for index, address in enumerate(addresses):
        if address >= 0:
            record = next(active)
            record.update(candidate=index, status="solved")
        else:
            record = dict(candidate=index, status="inactive_candidate",
                shape_a=scene.model.shape_label[shapes[0][index]],
                shape_b=scene.model.shape_label[shapes[1][index]], normal_force_n=0.,
                point_world_m=None, normal_a_to_b_world=None, force_on_b_world_n=None)
        pairs.append(pair(record["shape_a"], record["shape_b"], record["normal_force_n"]))
        records.append(record)
    if next(active, None) is not None:
        raise FasteningFault("unconsumed solved contact rows")
    return tuple(pairs), records


class FactoryObserver:
    """Exactly one call after each solver step; caller retains immutable stamp.

    The generation and requested effort come from the actual pre-solve upload,
    never from the guard's possibly newer post-solve generation. No state writes
    between solves are supported; only the owner may access these SDK arrays.
    """

    def __init__(self, scene, binding, limits, geometry, joints, *, sdk_recipe=None):
        self.scene, self.binding, self.limits = scene, binding, limits
        self.geometry, self.joints = geometry, joints
        if tuple(scene.model.shape_label) != binding.collider_names:
            raise FasteningFault("observer collider registry differs from binding")
        self.sources = sdk_sources() if sdk_recipe is None else sdk_sources(sdk_recipe)
        self.output = scene.newton.Contacts(int(scene.solver.mjw_data.naconmax), 0,
            device=scene.model.device, requested_attributes={"force"})
        self._last_step = 0
        self._zero_pairs = _ZeroLoadPairs()

    def read(self, stamp, captured_monotonic_s, collision_receipt):
        s = self.scene
        if s.step_id != self._last_step + 1 or stamp.before_step != self._last_step:
            raise FasteningFault("observer skipped or repeated a solve")
        q = _floats(s.state.joint_q, (s.model.joint_coord_count,), "Newton joint_q")
        qd = _floats(s.state.joint_qd, (s.model.joint_dof_count,), "Newton joint_qd")
        poses = _floats(s.state.body_q, (s.model.body_count, 7), "Newton body_q")
        velocities = _floats(s.state.body_qd, (s.model.body_count, 6), "Newton body_qd")
        d = s.solver.mjw_data
        m = s.solver.mj_model
        nq = _floats(d.qpos, (1, m.nq), "qpos")
        nv = _floats(d.qvel, (1, m.nv), "qvel")
        effort = _floats(d.qfrc_actuator, (1, m.nv), "qfrc_actuator")
        applied = _floats(d.qfrc_applied, (1, m.nv), "qfrc_applied")
        bodyforce = _floats(d.xfrc_applied, (1, m.nbody, 6), "xfrc_applied")
        constraint = _floats(d.qfrc_constraint, (1, m.nv), "qfrc_constraint")
        passive = _floats(d.qfrc_passive, (1, m.nv), "qfrc_passive")
        if np.any(applied) or np.any(bodyforce):
            raise FasteningFault("unadmitted external joint/body wrench")
        for row in self.joints:
            if (not np.isclose(q[row.newton_q], nq[0, row.native_q]-row.reference_rad, atol=2e-6, rtol=1e-6)
                    or not np.isclose(qd[row.newton_dof], nv[0, row.native_dof], atol=2e-6, rtol=1e-6)):
                raise FasteningFault("native/Newton joint readback mapping disagrees")
        bounds = self.geometry.evaluate(nq[0], body_poses=poses)
        pairs, raw_contacts = contact_records(s, self.output, _zero_pairs=self._zero_pairs)
        # Static bolt transform is bound at construction and checked by owner
        # model fingerprint; no fabricated moving fixture body is introduced.
        nut = poses[s.nut_body]
        tool_index = s.bodies["socket_spindle"]
        tool = poses[tool_index]
        nut_v, tool_v = velocities[s.nut_body], velocities[tool_index]
        spindle = self.joints[-1]
        native_ctrl = _floats(d.ctrl, (1, m.nu), "ctrl")
        scalar_forces = _floats(d.actuator_force, (1, m.nu), "actuator_force")
        if (native_ctrl[0, spindle.native_actuator] != np.float32(stamp.effort_nm)
                or scalar_forces[0, spindle.native_actuator] != effort[0, spindle.native_dof]):
            raise FasteningFault("spindle upload/actuator/generalized effort channels disagree")
        threshold = self.limits.contact_load_threshold_n
        loaded = [tuple(sorted((p.collider_a, p.collider_b))) for p in pairs if p.normal_force_n > threshold]
        value = FasteningSolve(self.binding.sha256, s.epoch, stamp.generation,
            s.step_id, s.time_s, captured_monotonic_s,
            tuple(float(q[j.newton_q]) for j in self.joints[:-1]),
            tuple(float(qd[j.newton_dof]) for j in self.joints[:-1]),
            tuple(float(effort[0, j.native_dof]) for j in self.joints[:-1]),
            *bounds, tuple(map(float, nut[:3])), tuple(map(float, nut[3:])), self.binding.fixture_origin_m, (0., 0., 0., 1.),
            tuple(map(float, tool[:3])), tuple(map(float, tool[3:])), float(np.linalg.norm(nut_v[:3])), float(np.linalg.norm(nut_v[3:])),
            float(np.linalg.norm(tool_v[:3])), float(np.linalg.norm(tool_v[3:])),
            float(qd[spindle.newton_dof]), float(effort[0, spindle.native_dof]), stamp.effort_nm,
            pairs, loaded.count(self.binding.thread_contact_pair),
            sum(p in self.binding.tool_contact_pairs for p in loaded),
            collision_receipt["contacts"]["capacity"], collision_receipt["contacts"]["count"],
            int(d.naconmax), len(pairs))
        if self.limits.seating is not None:
            from ..control.fastening_seat import seating_solve
            value = seating_solve(value, raw_contacts, self.binding)
        self._last_step = s.step_id
        return value, {"solve": asdict(value), "upload": asdict(stamp),
            "effort_time": "applied during interval (step-1,step); not reevaluated at final pose",
            "collision_buffers": collision_receipt, "contacts": raw_contacts,
            "constraint_count": _integer(d.nefc, "constraints"), "constraint_capacity": int(d.njmax),
            "native_ctrl": native_ctrl[0].tolist(), "actuator_force": scalar_forces[0].tolist(),
            "qfrc_actuator": effort[0].tolist(), "qfrc_applied": applied[0].tolist(),
            "qfrc_constraint": constraint[0].tolist(), "qfrc_passive": passive[0].tolist(),
            "body_poses_xyzw": poses.tolist()}
