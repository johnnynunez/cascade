"""Model-backed release clearance for an explicitly named MuJoCo tool subtree.

All FK uses detached MjData; no extra solve or live qpos write is permitted.
Bounds are conservative collision-geometry AABBs, not a grasp certificate.
"""
from __future__ import annotations

import math
import time
from copy import copy
from contextlib import contextmanager

import numpy as np

from ..config import Cfg
from ..control.motion_profile import nominal_profile
from ..safety.trajectory import PLAN_BUDGET_S, motion_duration, plan_route, vet_segment
from ..types import SafetyViolation, SkillError


def prepare(runtime, release_pose, *, carry_goals=None, deadline=None):
    raw = getattr(runtime.arm, "raw", None)
    name = getattr(raw, "_cfg", {}).get("mj_release_tool_body")
    if name is None:
        return None
    return Withdrawal(runtime, str(name), release_pose, carry_goals=carry_goals, deadline=deadline)


class Withdrawal:
    def __init__(self, runtime, tool_body, release_pose, *, carry_goals=None, deadline=None):
        self.runtime, self.arm = runtime, runtime.arm
        self.raw, self.kin, self.harness = self.arm.raw, runtime.kin, self.arm.harness
        self.world = self.raw.world
        if (self.world is None or self.raw._engine.kind != "mjc"
                or self.arm.motion_planner is not None):
            raise SkillError("release geometry requires the native MuJoCo C world and joint-route planner")
        self.model, self.mj = self.world.model, self.world.mj
        self._bound_engine, self._bound_ctrl = self.raw._engine, self.raw._ctrl
        self._bound_data = self.world.data
        self._bound_history = getattr(self.world, "placement_history", None)
        self._bound_epoch = getattr(self._bound_history, "epoch", None)
        if self.model.opt.disableflags & self.mj.mjtDisableBit.mjDSBL_NATIVECCD:
            raise SkillError("release collider distance requires native convex collision detection")
        if self.model.npair:
            raise SkillError("release adapter does not admit explicit contact-pair overrides")
        self.generation = self.harness._halt_generation
        self.cancellation = self.harness._observation_cancel_generation
        self.clearance = float(self.harness.limits.table_clearance)
        if not np.isfinite(self.clearance) or self.clearance <= 0:
            raise SkillError("release clearance must be a positive existing safety distance")
        self.tool_body = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, tool_body)
        if self.tool_body < 1:
            raise SkillError("release tool body missing from physical model")
        self.articulation_root, self.support_geoms, joint_bodies = self._admit_model()
        # Explicit tool root must belong to the controlled articulation.
        if not any(self._descendant(self.tool_body, b) for b in joint_bodies):
            raise SkillError("release tool body is outside the controlled articulation")
        free = [int(self.model.jnt_bodyid[j]) for j in range(self.model.njnt)
                if self.model.jnt_type[j] == self.mj.mjtJoint.mjJNT_FREE]
        self.robot, self.tool, self.objects = [], [], []
        for g in range(self.model.ngeom):
            if not (self.model.geom_contype[g] or self.model.geom_conaffinity[g]):
                continue
            b = int(self.model.geom_bodyid[g])
            if any(self._descendant(b, root) for root in free):
                self.objects.append(g)
            elif self._descendant(b, self.articulation_root):
                self.robot.append(g)
                if self._descendant(b, self.tool_body):
                    self.tool.append(g)
            elif g not in self.support_geoms:
                name = self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_GEOM, g)
                raise SkillError(f"release scene has an unmodeled static collider: {name or g}")
        if not self.tool or not self.objects:
            raise SkillError("release needs tool and object collision geometry")
        convex = {int(getattr(self.mj.mjtGeom, 'mjGEOM_'+name))
                  for name in ('SPHERE', 'CAPSULE', 'ELLIPSOID', 'CYLINDER', 'BOX', 'MESH')}
        if any(int(self.model.geom_type[g]) not in convex for g in self.robot+self.objects):
            raise SkillError("release requires supported native convex collision shapes")
        if (self.raw._grip_qadr is None or not 0 <= self.raw._grip_qadr < self.model.nq
                or not np.isfinite(self.raw._grip_open)):
            raise SkillError("release needs an explicit physical gripper coordinate")
        self.pairs = [(a, b) for a in self.robot for b in self.objects
                      if (self.model.geom_contype[a] & self.model.geom_conaffinity[b]
                          or self.model.geom_contype[b] & self.model.geom_conaffinity[a])]
        if not self.pairs:
            raise SkillError("release has no collidable arm/object pairs")
        self.pair_a, self.pair_b = np.asarray(self.pairs).T
        bounds = self.model.geom_aabb
        if bounds.shape != (self.model.ngeom, 6) or not np.isfinite(bounds).all() or (bounds[:, 3:] < 0).any():
            raise SkillError("release collision bounds unavailable")
        self.deadline = min(time.monotonic() + PLAN_BUDGET_S, deadline if deadline is not None else float("inf"))
        from ..sim.mujoco_placement import model_digest
        self._model_identity = model_digest(self.model, {})
        self._coordinate_binding = self._coordinates()
        self._planned_open_rad = float(self.raw._grip_open)
        self.data = self._snapshot()
        self.q_start = self.data.qpos[self.raw._qadr].copy()
        self._pose(self.q_start)
        from ..sim.truth import _match_label
        names = {self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_BODY, b): b for b in free}
        self.released_body = _match_label(runtime._held_det_label or runtime.held_object, names)
        if self.released_body is None:
            raise SkillError("release object has no unique physical body")
        self.released_geoms = [g for g in self.objects
                               if self._descendant(int(self.model.geom_bodyid[g]), self.released_body)]
        if carry_goals is not None:
            self._preview_carry(carry_goals)
            self.q_start = np.asarray(carry_goals[-1], dtype=float).copy()
            self._pose(self.q_start)
        self._release_envelope()
        self._plan_escape(release_pose)

    def _plan_escape(self, release_pose):
        """The same bounded candidate families for preview and actual release."""
        lower, upper = self._bounds()
        # Separate withdrawal from the *grasp* height ceiling. Actual full-pose
        # IK, workspace and joint/path limits determine whether this is possible.
        delta = max(0., float(upper[self.objects, 2].max()
                              - lower[self.tool, 2].min())) + self.clearance
        self.target = np.asarray(release_pose, dtype=float).copy()
        if self.target.shape != (4, 4) or not np.isfinite(self.target).all():
            raise SkillError("release needs the complete commanded placement pose")
        current_height = float(self.kin.fk(self.q_start)[2, 3])
        self.target[2, 3] = math.ceil((current_height + delta) * 1000.) / 1000.
        self.duration = float(self.runtime.cfg.grasp.get("descend_duration_s", 2.))
        candidates = []
        home = self.runtime._profile_q("home_q", "move home")
        home_radius = np.linalg.norm(self.kin.fk(home)[:2, 3])
        radius = np.linalg.norm(self.target[:2, 3])
        poses = [("full_pose_lift", self.target.copy())]
        if radius > home_radius > 0:
            inward = self.target.copy()
            inward[:2, 3] *= home_radius/radius
            poses.append(("lift_toward_home_radius", inward))
        for name, pose in poses:
            solution = self.kin.ik(pose, self.q_start)
            actual = self.kin.fk(solution.q)
            if (solution.success and np.max(np.abs(solution.q-self.q_start)) <= np.pi
                    and np.max(np.abs(actual[:3, 3]-pose[:3, 3])) <= .001
                    and np.linalg.norm(actual[:3, :3]-pose[:3, :3]) <= .001):
                candidates.append((name, solution.q.copy()))
        # Same joint-first/joint-last corners used by the ordinary route
        # planner. Once the jaws are open, orientation may change, provided
        # the entire modeled escape separates and the home route clears ALL
        # objects. No joint value, range or actuator gain is synthesized.
        for joint in range(len(home)):
            first = self.q_start.copy(); first[joint] = home[joint]
            last = home.copy(); last[joint] = self.q_start[joint]
            candidates.extend(((f"joint_{joint}_first", first), (f"joint_{joint}_last", last)))
        self.rejections = []
        for name, goal in candidates:
            try:
                reason = vet_segment(self.harness, self.q_start, goal, self.duration,
                                     deadline=self.deadline)
                if reason:
                    raise SkillError(reason)
                self._escape_segment(self.q_start, goal, self.duration)
                self._clear()
                self._home_route(goal)
            except SkillError as exc:
                self.rejections.append({"candidate": name, "reason": str(exc)})
                continue
            self.q = goal.copy()
            self.method = name
            self.target = self.kin.fk(self.q)
            break
        else:
            raise SkillError("no collision-clear release escape: " + repr(self.rejections))
        self.guard()

    def _admit_model(self):
        cfg, model, mj = self.raw._cfg, self.model, self.mj
        root_name, plane = cfg.get("mj_release_articulation_root"), cfg.get("mj_release_support_plane")
        if isinstance(plane, Cfg):
            plane = plane.as_dict()
        if not isinstance(root_name, str) or not isinstance(plane, dict):
            raise SkillError("release requires an explicit articulation root and support plane")
        root = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, root_name)
        if (model.nmocap or root < 1 or model.body_parentid[root] != 0 or model.body_jntnum[root] != 0
                or not np.allclose(model.body_pos[root], 0., atol=1e-12, rtol=0)
                or not np.allclose(np.abs(model.body_quat[root]), [1., 0., 0., 0.], atol=1e-12, rtol=0)):
            raise SkillError("release requires the declared fixed articulation root at the kinematic origin")
        names = list(self.raw._joint_names) + [self.raw._grip_joint]
        if any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
            raise SkillError("release controlled joint names are missing or duplicated")
        joints = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name) for name in names]
        actual = {j for j in range(model.njnt) if self._descendant(int(model.jnt_bodyid[j]), root)}
        if (set(joints) != actual or any(j < 0 or model.jnt_type[j] != mj.mjtJoint.mjJNT_HINGE for j in joints)
                or self.raw.n_joints != len(joints)-1
                or [int(model.jnt_qposadr[j]) for j in joints[:-1]] != list(self.raw._qadr)
                or [int(model.jnt_dofadr[j]) for j in joints[:-1]] != list(self.raw._dadr)
                or int(model.jnt_qposadr[joints[-1]]) != self.raw._grip_qadr):
            raise SkillError("release articulation joint inventory or coordinate binding differs")
        actuator_names = list(self.raw._act_names) + [self.raw._grip_act]
        if (len(actuator_names) != len(joints)
                or any(not isinstance(name, str) for name in actuator_names)
                or len(set(actuator_names)) != len(actuator_names)):
            raise SkillError("release actuator names are missing or duplicated")
        actuators = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name)
                     for name in actuator_names]
        if (any(a < 0 for a in actuators)
                or actuators != list(self.raw._aidx) + [self.raw._grip_aidx]):
            raise SkillError("release actuator coordinate binding differs")
        if (not np.isfinite([self.raw._grip_open, self.raw._grip_closed,
                             self.raw._grip_ctrl_scale, self.raw._grip_ctrl_offset]).all()
                or self.runtime._grip_open != self.raw._grip_open
                or self.runtime._grip_closed != self.raw._grip_closed):
            raise SkillError("release jaw observation and command units differ")
        support_names = plane.get("geoms")
        optional_names = plane.get("optional_geoms", [])
        if (not isinstance(support_names, list) or not support_names
                or any(not isinstance(n, str) for n in support_names)
                or not isinstance(optional_names, list)
                or any(not isinstance(n, str) for n in optional_names)
                or len(set(support_names+optional_names)) != len(support_names+optional_names)):
            raise SkillError("release support plane collider names missing or duplicated")
        supports = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, n) for n in support_names]
        supports += [g for n in optional_names
                     if (g := mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, n)) >= 0]
        position, quat = np.asarray(plane.get("position_m"), float), np.asarray(plane.get("quaternion_wxyz"), float)
        if (position.shape != (3,) or quat.shape != (4,) or not np.isfinite(position).all()
                or not np.isfinite(quat).all()
                or not np.allclose(np.abs(quat), [1., 0., 0., 0.], atol=1e-12, rtol=0)
                or position[2] != float(self.harness.limits.table_z)):
            raise SkillError("release support plane must explicitly match the horizontal safety table")
        for support in supports:
            if (support < 0 or model.geom_type[support] != mj.mjtGeom.mjGEOM_PLANE
                    or model.geom_bodyid[support] != 0
                    or not (model.geom_contype[support] or model.geom_conaffinity[support])
                    or not np.allclose(model.geom_pos[support], position, atol=1e-12, rtol=0)
                    or not (np.allclose(model.geom_quat[support], quat, atol=1e-12, rtol=0)
                            or np.allclose(model.geom_quat[support], -quat, atol=1e-12, rtol=0))):
                raise SkillError("release support plane type, body or pose differs from its descriptor")
        self.support_descriptor = plane
        return root, tuple(supports), [int(model.jnt_bodyid[j]) for j in joints[:-1]]

    def _descendant(self, child, parent):
        while child:
            if child == parent:
                return True
            child = int(self.model.body_parentid[child])
        return False

    def _binding_guard(self):
        if (self.runtime.arm is not self.arm or self.arm.raw is not self.raw
                or self.raw.world is not self.world or self.world.model is not self.model
                or self.arm.harness is not self.harness or self.runtime.kin is not self.kin
                or self.raw._engine is not self._bound_engine
                or self.raw._model is not self.model or self.raw._data is not self.world.data
                or self._bound_engine.model is not self.model
                or self._bound_engine.data is not self.world.data
                or self.raw._lock is not self.world.lock or self._bound_engine.lock is not self.world.lock
                or self.raw._ctrl is not self._bound_ctrl
                or self.raw._ctrl.shape != (self.model.nu,)):
            raise SkillError("release geometry binding changed")
        # Cheap mapping guard also runs at every scoped command/sample. Model
        # fingerprinting remains at planning/preflight, not per waypoint.
        if (hasattr(self, "_coordinate_binding")
                and self._coordinates() != self._coordinate_binding):
            raise SkillError("release native model changed after planning: driver coordinates")

    def retain(self):
        """Install debt BEFORE an opening can write or fail ambiguously."""
        self.guard()
        if self.harness._pending_model_withdrawal is not None:
            raise SafetyViolation("another model withdrawal is already retained")
        pending = getattr(self.runtime, "_mujoco_withdrawals", None)
        if pending is None:
            pending = self.runtime._mujoco_withdrawals = {}
        if id(self.arm) in pending:
            raise SafetyViolation("another arm withdrawal is already retained")
        self._authority = self
        self.harness._pending_model_withdrawal = self
        pending[id(self.arm)] = self

    @contextmanager
    def _scope(self, kind, *, target=None, duration=None):
        self._binding_guard()
        authority = getattr(self, "_authority", None)
        local = self.harness._model_withdrawal_scope
        if (authority is None or self.harness._pending_model_withdrawal is not authority
                or getattr(local, "value", None) is not None):
            raise SafetyViolation("model withdrawal scope unavailable or nested")
        local.value = {"owner": self, "kind": kind, "used": False,
                       "target": None if target is None else np.array(target, copy=True),
                       "duration": duration}
        try:
            yield
        finally:
            local.value = None

    def check_actuation(self, scope, *, command=False, gripper=False,
                        target=None, duration=None, grip=None, effort=None,
                        joint_margin=None, planned=False, cleanup=False):
        """Thread-local, single-command authority; no geometry or SDK in core."""
        if (scope is None or getattr(scope["owner"], "_authority", None) is not self
                or self.harness._pending_model_withdrawal is not self):
            raise SafetyViolation("unfinished model withdrawal; scoped recovery required")
        owner = scope["owner"]
        owner._binding_guard()
        kind = scope["kind"]
        if cleanup:
            if kind != "cleanup":
                raise SafetyViolation("model withdrawal cleanup requires its own scope")
            return
        owner.harness._check_halt_generation(owner.generation)
        owner._check_replan_cancellation()
        if owner.harness.estopped or owner.harness.halted is not None:
            raise SafetyViolation("model withdrawal is stopped")
        if gripper and kind != "open":
            raise SafetyViolation("model withdrawal does not authorize gripper movement")
        if not gripper and kind not in ("move", "plan"):
            raise SafetyViolation("model withdrawal does not authorize arm movement")
        if command:
            if scope["used"] or planned or kind == "plan":
                raise SafetyViolation("model withdrawal command authority already used or unavailable")
            if gripper:
                if grip != owner.runtime._grip_open or effort != .6:
                    raise SafetyViolation("model withdrawal only authorizes its exact opening")
            elif (joint_margin is not None or target is None
                  or not np.array_equal(np.asarray(target), scope["target"])
                  or duration != scope["duration"]):
                raise SafetyViolation("model withdrawal only authorizes its exact planned segment")
            scope["used"] = True

    def clear_grasp_exemption(self):
        with self._scope("cleanup"):
            self.harness.clear_grasp_exemption()

    def complete(self, *, generation=None):
        self._binding_guard()
        if generation is None:
            self._check_replan_cancellation()
        self.harness._check_halt_generation(self.generation if generation is None else generation)
        if self.harness.estopped or self.harness.halted is not None:
            raise SafetyViolation("model withdrawal completion cancelled")
        pending = getattr(self.runtime, "_mujoco_withdrawals", {})
        if (self.harness._pending_model_withdrawal is not self
                or pending.get(id(self.arm)) is not self):
            raise SafetyViolation("model withdrawal completion binding changed")
        self.harness._pending_model_withdrawal = None
        del pending[id(self.arm)]

    def guard(self):
        self._binding_guard()
        self.harness._check_halt_generation(self.generation)
        if self.harness.estopped:
            raise SkillError("release withdrawal cancelled")
        self._check_replan_cancellation()
        if time.monotonic() >= self.deadline:
            raise SkillError("release geometry planning deadline expired")

    def _check_replan_cancellation(self):
        if not getattr(self, "_postrelease_bound", False):
            return
        if self.cancellation != self.harness._observation_cancel_generation:
            raise SkillError("postrelease geometry context was cancelled")
        if (self.world.data is not self._bound_data
                or getattr(self.world, "placement_history", None) is not self._bound_history
                or getattr(self._bound_history, "epoch", None) != self._bound_epoch):
            raise SkillError("postrelease data/history epoch binding changed")

    def _coordinates(self):
        raw = self.raw
        return (raw.n_joints, tuple(raw._qadr), tuple(raw._dadr), tuple(raw._aidx),
                tuple(raw._joint_names), tuple(raw._act_names),
                raw._grip_qadr, raw._grip_aidx, raw._grip_joint, raw._grip_act,
                raw._grip_open, raw._grip_closed, raw._grip_ctrl_scale, raw._grip_ctrl_offset,
                self.runtime._grip_open, self.runtime._grip_closed, raw._ctrl.dtype.str)

    def _check_model_identity(self):
        from ..sim.mujoco_placement import model_digest
        if (self._coordinates() != self._coordinate_binding
                or model_digest(self.model, {}) != self._model_identity):
            raise SkillError("release native model changed after planning")

    def _snapshot(self):
        self.guard()
        with self.world.lock:
            data = self.mj.MjData(self.model)
            data.qpos[:] = self.world.data.qpos
        if not np.isfinite(data.qpos).all():
            raise SkillError("release state has non-finite coordinates")
        return data

    def _pose(self, q, *, open_hand=True):
        self.guard()
        self.data.qpos[self.raw._qadr] = q
        if open_hand:
            self.data.qpos[self.raw._grip_qadr] = self._planned_open_rad
        self.mj.mj_kinematics(self.model, self.data)
        if not np.isfinite(self.data.geom_xpos).all() or not np.isfinite(self.data.geom_xmat).all():
            raise SkillError("release FK geometry is non-finite")

    def _preview_carry(self, goals):
        """Preview an attached payload in scratch, never attach it in physics."""
        if not goals:
            raise SkillError("carry preview needs an explicit release endpoint")
        self.data = self._snapshot()
        self.mj.mj_kinematics(self.model, self.data)
        start = self.data.qpos[self.raw._qadr].copy()
        tcp = self.kin.fk(start)
        body = np.eye(4)
        body[:3, :3] = self.data.xmat[self.released_body].reshape(3, 3)
        body[:3, 3] = self.data.xpos[self.released_body]
        attachment = np.linalg.inv(tcp) @ body
        joint = int(self.model.body_jntadr[self.released_body])
        if joint < 0 or self.model.jnt_type[joint] != self.mj.mjtJoint.mjJNT_FREE:
            raise SkillError("carry preview requires an observed free payload")
        qadr = int(self.model.jnt_qposadr[joint])
        others = [g for g in self.objects if g not in self.released_geoms]
        pairs = [(a, b) for a, b in self.pairs if b in others]
        pairs += [(a, b) for a in self.released_geoms for b in others
                  if (self.model.geom_contype[a] & self.model.geom_conaffinity[b]
                      or self.model.geom_contype[b] & self.model.geom_conaffinity[a])]
        for goal in goals:
            duration = motion_duration(self.harness, start, goal,
                                       float(self.runtime.cfg.grasp.get("move_duration_s", 2.5)))
            for target in nominal_profile(start, goal, duration, 50., max_steps=10000):
                for _, q, _ in target.checks:
                    pose = self.kin.fk(q) @ attachment
                    self.data.qpos[qadr:qadr+3] = pose[:3, 3]
                    quat = np.empty(4)
                    self.mj.mju_mat2Quat(quat, pose[:3, :3].reshape(9))
                    self.data.qpos[qadr+3:qadr+7] = quat
                    self._pose(q, open_hand=False)
                    if any(self._distance(a, b) < 0 for a, b in pairs):
                        raise SkillError("planned carry intersects another physical object; keeping the grasp")
            start = goal

    def _bounds(self):
        rotation = self.data.geom_xmat.reshape(-1, 3, 3)
        center = self.data.geom_xpos + np.einsum("gij,gj->gi", rotation, self.model.geom_aabb[:, :3])
        half = np.einsum("gij,gj->gi", np.abs(rotation), self.model.geom_aabb[:, 3:])
        lower, upper = center-half, center+half
        if hasattr(self, "envelope"):
            lower[self.released_geoms] = self.envelope[0]
            upper[self.released_geoms] = self.envelope[1]
        return lower, upper

    def _release_envelope(self):
        lower, upper = self._bounds()
        center = self.data.xpos[self.released_body].copy()
        # Bound arbitrary orientation of the released object around its
        # measured body origin, plus its vertical fall to the table. This
        # is a planning envelope, not a prediction of bounce or dynamics.
        radius = np.linalg.norm(np.maximum(np.abs(lower[self.released_geoms]-center),
                                           np.abs(upper[self.released_geoms]-center)), axis=1).max()
        lo, hi = center-radius, center+radius
        lo[2] = min(lo[2], float(self.runtime.cfg.safety.get("table_z", 0.)))
        self.envelope = lo, hi

    def _escape_segment(self, start, goal, duration):
        self._pose(start)
        a, b = self.pair_a, self.pair_b
        initial = self._distances()
        held = np.isin(a, self.tool) & np.isin(b, self.released_geoms)
        if (initial[~held] < 0).any():
            raise SkillError("another arm/object pair initially intersects")
        duration = motion_duration(self.harness, start, goal, duration)
        for target in nominal_profile(start, goal, duration, 50., max_steps=10000):
            for prev_q, q, _ in target.checks:
                self._pose(prev_q)
                previous = self._distances()
                self._pose(q)
                separation = self._distances()
                # Starting inside the grasp cannot already have the later
                # empty-tool clearance. Require nonpenetration throughout;
                # any pre-existing intended contact may only separate.
                # The endpoint and the entire subsequent home route require
                # the full conservative clearance envelope, separately.
                if (separation < np.minimum(0., previous)-1e-10).any():
                    raise SkillError("release escape intersects native colliders")

    def _distances(self):
        # Native signed collider distance, capped at the requested clearance.
        # mj_kinematics updates only detached scratch geometry; no solve and
        # no contact/actuator assistance is applied to the live world.
        return np.asarray([self._distance(a, b) for a, b in self.pairs])

    def _distance(self, a, b):
        distance = float(self.mj.mj_geomDistance(self.model, self.data, a, b,
                                               self.clearance, None))
        if not math.isfinite(distance):
            raise SkillError("release native collider distance unavailable")
        return distance

    def _clear(self):
        lower, upper = self._bounds()
        a, b = self.pair_a, self.pair_b
        separation = np.maximum(lower[a]-upper[b], lower[b]-upper[a]).max(axis=1)
        bad = np.flatnonzero(separation < self.clearance)
        if len(bad):
            pair = self.pairs[int(bad[0])]
            raise SkillError(f"release/home collision clearance unavailable for geoms {pair}")

    def _home_route(self, start):
        home = self.runtime._profile_q("home_q", "move home")
        owner = self

        class GeometryHarness:
            def __getattr__(self, name):
                return getattr(owner.harness, name)

            def vet_step(self, previous, waypoint, dt):
                reason = owner.harness.vet_step(previous, waypoint, dt)
                if reason:
                    return reason
                owner._pose(waypoint)
                try:
                    owner._clear()
                except SkillError as exc:
                    return str(exc)
                return None

        return plan_route(GeometryHarness(), start, home, 3.)

    def _geometry_segment(self, start, goal, duration):
        duration = motion_duration(self.harness, start, goal, duration)
        for target in nominal_profile(start, goal, duration, 50., max_steps=10000):
            for _, q, _ in target.checks:
                self._pose(q)
                self._clear()

    def after_withdrawal(self):
        self.deadline = time.monotonic()+PLAN_BUDGET_S
        self.data = self._snapshot()
        self._pose(self.data.qpos[self.raw._qadr].copy())
        self._clear()

    def open_hand(self):
        self.guard()
        with self._scope("open"):
            self.arm.set_gripper(self.runtime._grip_open, effort=.6,
                                 _halt_generation=self.generation)

    def _require_open(self):
        fraction = self.runtime._gripper_width_frac()
        if fraction is None or not math.isfinite(fraction) or fraction < .98:
            raise SkillError("release requires actual open-jaw feedback before withdrawal")

    def _replan_after_open(self):
        """Refresh geometry without releasing the original actuation debt.

        The fixture has no continuous solve owner: its world lock prevents a
        driver step during this bounded scratch plan. Stop never needs that
        lock. A changed state before the eventual command is rejected again.
        """
        from ..sim.mujoco_placement import state_digest
        candidate = copy(self)
        candidate.deadline = time.monotonic()+PLAN_BUDGET_S
        candidate._postrelease_bound = True
        previous = self.receipt()
        with self._scope("plan"), self.world.lock:
            candidate.guard()
            if self.runtime.held_object:
                raise SkillError("postrelease withdrawal requires an empty tool")
            candidate._check_model_identity()
            candidate._require_open()
            snapshot = state_digest(self.world.data)
            data_object = self.world.data
            candidate.data = candidate._snapshot()
            candidate._planned_open_rad = float(candidate.data.qpos[self.raw._grip_qadr])
            candidate.q_start = candidate.data.qpos[self.raw._qadr].copy()
            candidate._pose(candidate.q_start)
            del candidate.envelope
            candidate._release_envelope()
            candidate._plan_escape(self.kin.fk(candidate.q_start))
            candidate._check_model_identity()
            candidate.guard()
            if self.world.data is not data_object or state_digest(data_object) != snapshot:
                raise SkillError("physical state changed during postrelease planning")
            candidate._postrelease_state = snapshot
            candidate._postrelease_data = data_object
            candidate.postrelease = {
                "source": "measured final qpos after actual opening; detached FK",
                "state_sha256": snapshot, "model_sha256": self._model_identity,
                "simulation_time_s": float(data_object.time),
                "generation": candidate.generation, "cancellation": candidate.cancellation,
                "history_epoch": candidate._bound_epoch,
                "joints_rad": candidate.q_start.tolist(),
                "measured_gripper_rad": candidate._planned_open_rad,
                "tcp_from_kinematics_m": self.kin.fk(candidate.q_start)[:3, 3].tolist(),
                "previous_plan": previous, "full_escape_and_home_checked": True,
                "physical_task_verdict": False,
            }
            # Keep self as the unique pending authority. Only validated
            # scratch-plan fields are replaced; no generation is renewed.
            for name in ("data", "q_start", "q", "target", "envelope", "method",
                         "rejections", "duration", "deadline", "_planned_open_rad",
                         "_postrelease_bound", "_postrelease_state", "_postrelease_data",
                         "postrelease"):
                setattr(self, name, getattr(candidate, name))
        return self.postrelease

    def withdraw(self):
        self._replan_after_open()
        def preflight(actual, duration):
            from ..sim.mujoco_placement import state_digest
            with self.world.lock:
                self.guard()
                self._check_model_identity()
                if (self.world.data is not self._postrelease_data
                        or state_digest(self.world.data) != self._postrelease_state
                        or not np.array_equal(actual, self.q_start)):
                    raise SkillError("physical state changed before postrelease command")
                self.data = self._snapshot()
                self._require_open()
                reason = vet_segment(self.harness, actual, self.q, duration,
                                     deadline=self.deadline, stretch=False)
                if reason:
                    raise SkillError("unsafe actual release escape: " + reason)
                self._escape_segment(actual, self.q, duration)
                self._clear()
                self.guard()
        with self._scope("move", target=self.q, duration=self.duration):
            return self.arm.move_joints(self.q, duration_s=self.duration,
                                        _preflight=preflight, _halt_generation=self.generation)

    def home(self):
        """Execute the actual checked joint route through ordinary SafeArm."""
        if self.runtime.held_object:
            raise SkillError("pending empty-tool withdrawal cannot transport a newly held object")
        self.deadline = time.monotonic()+PLAN_BUDGET_S
        self.data = self._snapshot()
        start = self.data.qpos[self.raw._qadr].copy()
        with self._scope("plan"):
            route = self._home_route(start)
        for goal in route:
            def preflight(actual, duration, goal=goal):
                self.deadline = time.monotonic()+PLAN_BUDGET_S
                self.data = self._snapshot()
                self._require_open()
                reason = vet_segment(self.harness, actual, goal, duration,
                                     deadline=self.deadline, stretch=False)
                if reason:
                    raise SkillError("unsafe release home route: " + reason)
                self._geometry_segment(actual, goal, duration)
            with self._scope("move", target=goal, duration=3.):
                if not self.arm.move_joints(goal, duration_s=3., _preflight=preflight,
                                            _halt_generation=self.generation):
                    raise SkillError("did not settle at home after release")
        self.after_withdrawal()
        return {"at": "home"}

    def recover_for_reset(self):
        """Explicit reset may replan; normal retries cannot renew a permit."""
        self._binding_guard()
        if self.runtime.held_object:
            raise SkillError("release reset recovery requires an empty tool")
        if self.harness.estopped or self.harness._halt is not None:
            raise SkillError("release reset recovery cannot clear a stop latch")
        recovery = copy(self)
        # Retain only the model's collider identities. Old target/q/scratch
        # geometry grant no authority to this explicitly requested recovery.
        del recovery.q, recovery.q_start, recovery.target
        recovery.generation = self.harness._halt_generation
        recovery.cancellation = self.harness._observation_cancel_generation
        recovery._bound_data = self.world.data
        recovery._bound_history = getattr(self.world, "placement_history", None)
        recovery._bound_epoch = getattr(recovery._bound_history, "epoch", None)
        recovery.clearance = float(self.harness.limits.table_clearance)
        if not math.isfinite(recovery.clearance) or recovery.clearance <= 0:
            raise SkillError("release recovery clearance unavailable")
        recovery.deadline = time.monotonic()+PLAN_BUDGET_S
        recovery.data = recovery._snapshot()
        recovery._require_open()
        recovery._pose(recovery.data.qpos[self.raw._qadr].copy())
        del recovery.envelope
        recovery._release_envelope()
        result = recovery.home()
        return {"generation": recovery.generation, "home": result,
                "pending_retained_until_scene_reset": True}

    def verify_reset(self, names, generation):
        """A reset ACK alone must not erase retained release state."""
        self._binding_guard()
        self.harness._check_halt_generation(generation)
        if self.harness.estopped:
            raise SkillError("release reset verification cancelled")
        expected = self.world.free_body_names()
        if sorted(names) != sorted(expected) or not expected:
            raise SkillError("release reset did not name every physical object")
        with self.world.lock:
            for j in range(self.model.njnt):
                if self.model.jnt_type[j] != self.mj.mjtJoint.mjJNT_FREE:
                    continue
                qa, va = int(self.model.jnt_qposadr[j]), int(self.model.jnt_dofadr[j])
                if (not np.array_equal(self.world.data.qpos[qa:qa+7], self.model.qpos0[qa:qa+7])
                        or not np.array_equal(self.world.data.qvel[va:va+6], np.zeros(6))):
                    raise SkillError("release reset physical state differs from the model spawn")
        self.harness._check_halt_generation(generation)
        return {"channel": "mujoco_physics", "props": expected,
                "at_model_spawn": True, "zero_free_body_velocity": True}

    def receipt(self):
        return {"channel": "mujoco_collision_bounds", "clearance_m": self.clearance,
                "tcp_target_m": self.target[:3, 3].tolist(),
                "tool_body": self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_BODY, self.tool_body),
                "object_geoms": list(self.objects), "physical_task_verdict": False,
                "articulation_root": self.raw._cfg.get("mj_release_articulation_root"),
                "support_plane": self.support_descriptor,
                "effective_support_planes": [
                    {"geom": self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_GEOM, g),
                     "position_m": self.model.geom_pos[g].tolist(),
                     "quaternion_wxyz": self.model.geom_quat[g].tolist()}
                    for g in self.support_geoms],
                "method": self.method, "rejected_candidates": self.rejections,
                "released_envelope_m": [v.tolist() for v in self.envelope],
                **({"postrelease_replan": self.postrelease} if hasattr(self, "postrelease") else {})}
