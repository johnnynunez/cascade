"""Optional private witness of every ordinary VAB/MuJoCo solve.

Install after the environment reset. The actor's observations are unchanged;
no object/contact/pose measurements from this recorder enter its control path.
Only an instance's existing step method is wrapped, never a global SDK symbol.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np

from cascade.eval.placement import PlacementPolicy, seal_placement_row


def _descendants(model, root):
    result = {root}
    for body in range(root+1, model.nbody):
        if int(model.body_parentid[body]) in result:
            result.add(body)
    return result


def _unconstrained_bodies(model, mj, groups, roots):
    """Free joints alone do not exclude a hidden weld, spring, or actuator."""
    if any(int(getattr(model, name)) != 0 for name in ("neq", "ntendon", "nplugin")):
        raise ValueError("placement recipe excludes equalities, tendons, and plugins")
    for callback in ("control", "passive", "sensor", "contactfilter", "act_dyn", "act_gain", "act_bias", "time"):
        if getattr(mj, "get_mjcb_"+callback)() is not None:
            raise ValueError("placement recipe excludes native callbacks")
    for actuator in range(model.nu):
        joint = int(model.actuator_trnid[actuator, 0])
        if (int(model.actuator_trntype[actuator]) != int(mj.mjtTrn.mjTRN_JOINT)
                or not 0 <= joint < model.njnt or int(model.jnt_bodyid[joint]) not in groups["robot"]):
            raise ValueError("placement actuators must drive only robot joints")
    for name in ("object", "support"):
        joint = int(model.body_jntadr[roots[name]])
        dof = int(model.jnt_dofadr[joint])
        if (model.jnt_stiffness[joint] != 0 or np.any(model.dof_frictionloss[dof:dof+6] != 0)
                or any(model.body_gravcomp[body] != 0 for body in groups[name])):
            raise ValueError("placement object/support has passive constraints or gravity compensation")


class PlacementRecorder:
    """Complete candidate ledger, constraint-phase poses, and release bounds.

    Euler is required because the contact solve and derived geometry retained
    by mj_step describe its pre-integration state. No mj_forward, mj_step,
    qpos edits, force writes, or geometry changes are issued by this reader.
    """
    def __init__(self, env, output, *, model_identity_sha256, epoch,
                 object_name, support_name, robot_root_body, max_solves=16000,
                 max_bytes=256*1024*1024, record_box_geometry=False):
        import mujoco
        self.mj, self.sim = mujoco, env.sim
        self.model, self.data = env.sim.model._model, env.sim.data._data
        self.out = Path(output)
        if type(max_solves) is not int or max_solves < 1 or type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("placement recorder budgets must be positive integers")
        self.max_solves, self.max_bytes = max_solves, max_bytes
        if type(record_box_geometry) is not bool:
            raise ValueError("explicit box geometry recording flag required")
        self.box_inventory = None
        if self.out.exists():
            raise ValueError("placement recorder requires a new output directory")
        if int(self.model.opt.integrator) != int(mujoco.mjtIntegrator.mjINT_EULER):
            raise ValueError("placement witness requires Euler constraint-phase semantics")
        self.body_ids = {"object": env._obj_body_id[object_name], "support": env._obj_body_id[support_name]}
        groups = {name: _descendants(self.model, root) for name, root in self.body_ids.items()}
        robot = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, robot_root_body)
        if robot < 1:
            raise ValueError("placement robot root is absent or world")
        groups["robot"] = _descendants(self.model, robot)
        if any(groups[a] & groups[b] for a, b in (("object", "support"), ("object", "robot"), ("support", "robot"))):
            raise ValueError("placement body groups overlap")
        for name in ("object", "support"):
            root = self.body_ids[name]
            if root < 1 or self.model.body_jntnum[root] != 1:
                raise ValueError("placement requires free rigid object/support roots")
            joint = int(self.model.body_jntadr[root])
            if int(self.model.jnt_type[joint]) != int(mujoco.mjtJoint.mjJNT_FREE):
                raise ValueError("placement body root is not free-jointed")
            if any(self.model.body_jntnum[body] for body in groups[name]-{root}):
                raise ValueError("articulated objects/support need a separate placement recipe")
        _unconstrained_bodies(self.model, self.mj, groups, self.body_ids)
        self._groups = groups
        explicit_pairs = set(map(int, self.model.pair_geom1)) | set(map(int, self.model.pair_geom2))
        geoms = {name: tuple(i for i in range(self.model.ngeom)
                            if int(self.model.geom_bodyid[i]) in bodies
                            and (self.model.geom_contype[i] or self.model.geom_conaffinity[i] or i in explicit_pairs))
                 for name, bodies in groups.items()}
        self.geom_ids = sum(geoms.values(), ())
        if any(not math.isfinite(float(self.model.geom_rbound[i])) or self.model.geom_rbound[i] <= 0
               for i in self.geom_ids):
            raise ValueError("placement separation needs finite positive geometry bounds")
        self.policy = PlacementPolicy(model_identity_sha256, epoch, geoms["object"], geoms["support"],
            geoms["robot"], float(sum(self.model.body_mass[i] for i in groups["object"])),
            tuple(map(float, self.model.opt.gravity)), float(self.model.opt.timestep),
            int(self.body_ids["object"]), int(self.body_ids["support"]), int(self.model.ngeom))
        self._bound_fields = {name: np.asarray(getattr(self.model, name)).copy() for name in (
            "geom_rbound", "geom_bodyid", "geom_contype", "geom_conaffinity", "body_mass",
            "body_parentid", "body_jntnum", "body_jntadr", "jnt_type", "pair_geom1", "pair_geom2",
            "actuator_trntype", "actuator_trnid", "jnt_bodyid", "jnt_dofadr", "jnt_stiffness",
            "dof_frictionloss", "body_gravcomp")}
        if record_box_geometry:
            from cascade.eval.box_geometry import BoxGeometryInventory
            from cascade.eval.cavity import BoxWall
            boxes = {}
            for name in ("object", "support"):
                boxes[name] = []
                for geom in geoms[name]:
                    if (int(self.model.geom_type[geom]) != int(self.mj.mjtGeom.mjGEOM_BOX)
                            or int(self.model.geom_bodyid[geom]) != self.body_ids[name]):
                        raise ValueError("box geometry recipe requires all colliders to be boxes on their rigid root")
                    rotation = np.empty(9)
                    self.mj.mju_quat2Mat(rotation, self.model.geom_quat[geom])
                    transform = np.eye(4)
                    transform[:3, :3] = rotation.reshape(3, 3)
                    transform[:3, 3] = self.model.geom_pos[geom]
                    boxes[name].append(BoxWall(geom, tuple(transform.flat), tuple(self.model.geom_size[geom])))
            self.box_inventory = BoxGeometryInventory(model_identity_sha256, self.policy.sha256,
                int(self.body_ids["object"]), int(self.body_ids["support"]),
                tuple(boxes["object"]), tuple(boxes["support"]))
            self.box_inventory.bind(self.policy)
            self._box_inventory_sha256 = self.box_inventory.sha256
            self._bound_fields.update({name: np.asarray(getattr(self.model, name)).copy()
                for name in ("geom_type", "geom_size", "geom_pos", "geom_quat")})
        if np.any(self.data.warning.number):
            raise ValueError("placement cannot start after native warnings")
        self.out.mkdir(parents=True)
        (self.out/"policy.json").write_text(json.dumps(asdict(self.policy), indent=2, allow_nan=False)+"\n")
        if self.box_inventory is not None:
            (self.out/"box-geometry.json").write_text(json.dumps(asdict(self.box_inventory), indent=2, allow_nan=False)+"\n")
        self.steps, self.bytes, self.fault = 0, 0, None
        self._warning_counts = np.asarray(self.data.warning.number).copy()
        self._last_time = float(self.data.time)
        self._original_step = self.sim.step
        self._wrapper = self._step
        self._closed = False
        self.sim.step = self._wrapper

    def _check_model(self):
        _unconstrained_bodies(self.model, self.mj, self._groups, self.body_ids)
        if (float(self.model.opt.timestep) != self.policy.solver_dt_s
                or int(self.model.opt.integrator) != int(self.mj.mjtIntegrator.mjINT_EULER)
                or not np.array_equal(self.model.opt.gravity, self.policy.gravity_world_m_s2)
                or any(not np.array_equal(getattr(self.model, key), saved) for key, saved in self._bound_fields.items())):
            raise RuntimeError("placement solver configuration or geometry changed")

    def _step(self, *args, **kwargs):
        if self._closed or self.fault is not None:
            raise RuntimeError("placement recorder is closed or faulted")
        if self.steps >= self.max_solves:
            self.fault = "placement solve budget exhausted"
            raise RuntimeError(self.fault)
        before = float(self.data.time)
        try:
            self._check_model()
            if before != self._last_time:
                raise RuntimeError("placement clock advanced outside its ordinary step owner")
            result = self._original_step(*args, **kwargs)
            self.steps += 1
            self._check_model()
            if not math.isclose(float(self.data.time)-before, self.policy.solver_dt_s, rel_tol=0, abs_tol=1e-9):
                raise RuntimeError("placement solver clock changed")
            row = self._capture(before)
            encoded = (json.dumps(row, separators=(",", ":"), allow_nan=False)+"\n").encode()
            if self.bytes+len(encoded) > self.max_bytes:
                raise RuntimeError("placement byte budget exhausted; final solve is not archived")
            with (self.out/"solves.jsonl").open("ab") as stream:
                stream.write(encoded)
            self.bytes += len(encoded)
            self._last_time = float(self.data.time)
            if row["native_warnings"] or not row["external_forces_zero"]:
                raise RuntimeError("placement retained a rejected warning or external-force sample")
            return result
        except BaseException as exc:
            self.fault = type(exc).__name__+": "+str(exc)
            raise

    def _capture(self, constraint_time):
        m, d, mj = self.model, self.data, self.mj
        bodies = {}
        for name, body in self.body_ids.items():
            velocity = np.zeros(6)
            mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_BODY, body, velocity, 0)
            bodies[name] = {"body_id": int(body), "position_m": d.xpos[body].tolist(),
                            "linear_velocity_m_s": velocity[3:].tolist(),
                            "angular_velocity_rad_s": velocity[:3].tolist()}
        contacts = []
        for index in range(d.ncon):
            contact, wrench = d.contact[index], np.zeros(6)
            if contact.efc_address >= 0:
                mj.mj_contactForce(m, d, index, wrench)
            frame = np.asarray(contact.frame).reshape(3, 3)
            contacts.append({"index": index, "geom_a": int(contact.geom1), "geom_b": int(contact.geom2),
                "efc_address": int(contact.efc_address), "dimension": int(contact.dim),
                "distance_m": float(contact.dist), "position_m": contact.pos.tolist(),
                "frame_rows": frame.tolist(), "wrench_on_b_contact": wrench.tolist(),
                "force_on_b_world_n": (frame.T @ wrench[:3]).tolist()})
        row = {"model_identity_sha256": self.policy.model_identity_sha256,
            "epoch": self.policy.epoch, "policy_sha256": self.policy.sha256,
            "solver_step": self.steps, "constraint_time_s": constraint_time,
            "advanced_time_s": float(d.time), "phase": "euler_constraint_before_integration",
            "coverage": "all_native_contact_candidates", "native_ngeom": int(m.ngeom), "ncon": int(d.ncon),
            "coupling_admission": self.policy.constraint_recipe,
            "nefc": int(d.nefc),
            "native_warnings": np.flatnonzero(np.asarray(d.warning.number) != self._warning_counts).tolist(),
            "external_forces_zero": bool(np.all(d.xfrc_applied == 0) and np.all(d.qfrc_applied == 0)),
            "bodies": bodies, "contacts": contacts,
            "geometries": [{"id": i, "position_m": d.geom_xpos[i].tolist(),
                            "bound_radius_m": float(m.geom_rbound[i])} for i in self.geom_ids]}
        if self.box_inventory is not None:
            row["box_geometry"] = {"inventory_sha256": self._box_inventory_sha256,
                "body_rotations_world": {name: d.xmat[body].tolist() for name, body in self.body_ids.items()},
                "rotations_world": [{"id": box.geometry_id, "rotation": d.geom_xmat[box.geometry_id].tolist()}
                                    for box in self.box_inventory.object_boxes+self.box_inventory.support_boxes]}
        return seal_placement_row(row)

    def close(self):
        if self._closed:
            raise RuntimeError("placement recorder already closed")
        self._closed = True
        unchanged = self.sim.step is self._wrapper
        if unchanged:
            self.sim.step = self._original_step
        receipt = {"ok": self.fault is None and unchanged, "fault": self.fault,
                   "hook_unchanged": unchanged, "solver_steps": self.steps, "bytes": self.bytes,
                   "policy_sha256": self.policy.sha256, "advances_physics": False}
        (self.out/"closure.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
        return receipt
