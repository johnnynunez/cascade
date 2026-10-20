"""Bind the G1 wrist-driver Newton scene to CASCADE's unchanged fastening contracts.

``HumanoidFasteningBackend`` implements the same backend surface as
``factory_owner.FactoryNewtonBackend`` (step, enter_owner, plan, upload,
upload_zero, advance), so the unchanged ``FactorySolveOwner``,
``FasteningController``/``FasteningWriteGuard``, ``SolveJournal`` and
``FasteningDomain`` (rotation before stop, axial advance, observed rest, final
threading) drive and judge the humanoid exactly as they judge the SO-101.

Mapping onto the existing contract:
- ``joint_names``: every position-controlled humanoid joint (arm follower,
  waist, cosmetic fingers, held legs/left arm), checked every solve;
- "spindle": the turning RIGHT WRIST ROLL, a torque-controlled joint. Its
  commanded and solved effort are the spindle effort channels, so a stop is an
  exact-zero wrist torque and rest requires it to be exactly zero;
- tool: the hand-held driver; tool contacts = solved nut/hex-wall pairs only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ..control.fastening import FasteningBinding, FasteningFault, FasteningLimits, FasteningSolve
from .factory_observation import _ZeroLoadPairs, collision_coverage, contact_records, world_aabb

TOOL_ID = 'g1_hand_held_hex_driver'
FASTENER_ID = 'factory_nut_m20_loose'
FIXTURE_ID = 'fixed_factory_bolt_m20_loose'
# Bounds of every moving arm/tool part this task may occupy (world, metres).
WORKSPACE_MIN_M = (-.35, -.45, 0.)
WORKSPACE_MAX_M = (.70, .80, 1.40)
TABLE_Z_M = 0.
# Right-arm bodies whose geometry is bounded every solve (the moving chain).
MOVING_BODIES = ('right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link',
                 'right_elbow_link', 'right_wrist_roll_link', 'right_wrist_pitch_link',
                 'right_wrist_yaw_link', 'right_hand_thumb_0_link', 'right_hand_thumb_1_link',
                 'right_hand_thumb_2_link', 'right_hand_middle_0_link', 'right_hand_middle_1_link',
                 'right_hand_index_0_link', 'right_hand_index_1_link')
SOURCE_FILES = ('src/cascade/sim/newton_humanoid_screw.py', 'src/cascade/sim/humanoid_fastening.py')
REPO = Path(__file__).resolve().parents[3]


def humanoid_limits(scene):
    """Explicit task budgets of this recipe; ThreadContract and rest rules are unchanged.

    Different from the SO-101 candidate (and why): joint speed 2 rad/s (the
    humanoid's cosmetic finger re-grip; G1 joints are rated far faster), wrist
    effort cap 1 N m (the turning joint; actuator rated 25 N m), and a 15 s /
    120 s one-turn command budget, because the 3.9 rad wrist range needs a
    re-grip inside one turn. Rest window/thresholds, observation age, tracking
    error, margin and the 10 rad/s rotational bound are the shared defaults.
    """
    return FasteningLimits(
        joint_lower_rad=tuple(map(float, scene.lower)), joint_upper_rad=tuple(map(float, scene.upper)),
        joint_effort_nm=tuple(map(float, scene.effort_caps)),
        workspace_min_m=WORKSPACE_MIN_M, workspace_max_m=WORKSPACE_MAX_M, table_z_m=TABLE_Z_M,
        max_joint_speed_rad_s=2., spindle_effort_nm=float(scene.motor_limit_nm),
        max_command_sim_s=15., max_command_wall_s=120.)


def model_document(scene, *, sources=None):
    """Everything the physics identity depends on; hashed into model_sha256."""
    sources = sources if sources is not None else {
        name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    return {
        'recipe': 'g1_wrist_driver_m20_v1', 'scene_xml_sha256': hashlib.sha256(scene.scene_xml.encode()).hexdigest(),
        'factory_assets': scene.asset_hashes, 'robot_asset': scene.robot_asset.name,
        'robot_asset_sha256': hashlib.sha256(scene.robot_asset.read_bytes()).hexdigest(),
        'dt_s': scene.dt, 'thread_friction': scene.thread_friction,
        'socket_clearance_m': scene.socket_clearance_m, 'pelvis_world_m': list(scene.pelvis_world_m),
        'wrist': {'kp': scene.roll_kp, 'kd': scene.roll_kd, 'limit_nm': scene.motor_limit_nm,
                  'low': scene.roll_low, 'high': scene.roll_high, 'stroke_rad_s': scene.stroke_speed_rad_s,
                  'return_rad_s': scene.return_speed_rad_s},
        'joint_names': list(scene.joint_names), 'shape_labels': list(scene.model.shape_label),
        'body_labels': list(scene.model.body_label), 'sources': sources,
    }


def humanoid_binding(scene, limits, document):
    labels = tuple(scene.model.shape_label)
    name = {key: labels[index] for key, index in scene.shape.items()}
    walls = tuple((name['nut'], name[f'socket_wall_{i}']) for i in range(6))
    thread = (name['bolt'], name['nut'])
    return FasteningBinding(
        model_sha256=hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        limits_sha256=limits.sha256, robot_id='unitree_g1_29dof_dex3_wrist_driver', fixture_id=FIXTURE_ID,
        fastener_id=FASTENER_ID, tool_id=TOOL_ID, joint_names=scene.joint_names, collider_names=labels,
        allowed_contact_pairs=(thread, *walls, (name['nut'], name['socket_floor'])),
        thread_contact_pair=thread, tool_contact_pairs=walls, dt_s=scene.dt,
        fixture_origin_m=tuple(map(float, scene.fixture_position)), fixture_recipe=scene.fixture_recipe,
        thread_pitch_m=scene.pitch_m)


class HumanoidGeometry:
    """Conservative world AABB of the moving right arm (all its geoms) and the driver colliders."""

    def __init__(self, scene):
        self.scene = scene
        m = scene.ik_model
        bodies = {m.body(name).id for name in MOVING_BODIES}
        self.geoms = np.array([g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bodies])
        if not len(self.geoms):
            raise FasteningFault('no moving arm geometry to bound')
        self.local = np.asarray(m.geom_aabb[self.geoms], dtype=float)
        self.qpos_index = {name: int(m.joint(name).qposadr[0]) for name in (*scene.joint_names, 'right_wrist_roll_joint')}
        # Driver colliders in the driver frame: six hex walls and the socket floor.
        radius = .015 + scene.socket_clearance_m + .003 + .003
        top = .008 + scene.floor_gap_m + .015
        self.tool_local = np.array([[0., 0., (top - .010) / 2, radius, radius, (top + .010) / 2]])

    def evaluate(self, joint_values, roll, tool_pose):
        m, d = self.scene.ik_model, self.scene.ik_data
        d.qpos[:] = 0.
        for name, value in zip(self.scene.joint_names, joint_values, strict=True):
            d.qpos[self.qpos_index[name]] = value
        d.qpos[self.qpos_index['right_wrist_roll_joint']] = roll
        self.scene.mujoco.mj_kinematics(m, d)
        arm = world_aabb(self.local, d.geom_xpos[self.geoms], d.geom_xmat[self.geoms].reshape(-1, 3, 3))
        from .newton_humanoid_screw import rot_xyzw
        tool = world_aabb(self.tool_local, np.asarray(tool_pose[:3])[None], rot_xyzw(tool_pose[3:])[None])
        return (tuple(min(a, b) for a, b in zip(arm[0], tool[0])),
                tuple(max(a, b) for a, b in zip(arm[1], tool[1])))


@dataclass
class HumanoidBoundModel:
    scene: object
    binding: FasteningBinding
    limits: FasteningLimits
    document: dict
    clock: object = time.monotonic


def bind_humanoid_model(scene, *, clock=time.monotonic):
    limits = humanoid_limits(scene)
    document = model_document(scene)
    return HumanoidBoundModel(scene, humanoid_binding(scene, limits, document), limits, document, clock)


class HumanoidFasteningBackend:
    """One actual solve per owner cycle; the wrist program runs only under a permit."""

    synthetic = False
    pose_every = 5  # body poses retained every 5th solve (120 Hz) for replay/video

    def __init__(self, bound_model):
        self.bound_model = bound_model
        self.scene = scene = bound_model.scene
        self.binding, self.limits = bound_model.binding, bound_model.limits
        if tuple(scene.model.shape_label) != self.binding.collider_names:
            raise FasteningFault('collider registry differs from binding')
        if scene.step_id != 0:
            raise FasteningFault('humanoid backend requires a fresh scene')
        self.geometry = HumanoidGeometry(scene)
        self.output = scene.newton.Contacts(int(scene.solver.mjw_data.naconmax), 0,
            device=scene.model.device, requested_attributes={'force'})
        self._resume = True
        self._last_step = 0
        self._zero_pairs = _ZeroLoadPairs()   # immutable zero-load records reused, as the Factory observer does
        labels = scene.model.shape_label
        self.floor_pair = tuple(sorted((labels[scene.shape['nut']], labels[scene.shape['socket_floor']])))

    @property
    def step(self):
        return self.scene.step_id

    def enter_owner(self):
        self.scene.wp.set_device(self.scene.model.device)

    def upload(self, target, effort):
        self.scene.apply_controls(np.asarray(target, dtype=float), float(effort))

    def upload_zero(self, effort):
        if effort != 0.:
            raise FasteningFault('emergency writer only accepts exact zero wrist effort')
        # Keep every approved position target (and therefore the grasp); zero the wrist.
        self.scene.apply_controls(self.scene.targets, 0.)
        if self.scene.program.phase not in ('idle', 'stopped'):
            self.scene.program.halt(self.scene.time_s)
        self._resume = True

    def plan(self, row):
        if row is None:
            raise FasteningFault('no solved baseline for humanoid target planning')
        s = self.scene
        if self._resume:
            # A newly admitted command re-anchors the position targets on the measured
            # joints (the guard limits the first step from them) and resumes the wrist
            # routine from the measured wrist angle.
            s.targets = np.asarray(row.joint_position_rad, dtype=float)
            s.program.start(s.roll_state()[0], s.time_s)
            self._resume = False
        targets, torque = s.plan()
        tool_pose = s.state.body_q.numpy()[s.tool_body]
        bounds = self.geometry.evaluate(targets, s.roll_state()[0], tool_pose)
        return tuple(map(float, targets)), bounds, float(torque)

    def advance(self, final_upload):
        s = self.scene
        if s.step_id != self._last_step:
            raise FasteningFault('humanoid solve count changed outside the owner')
        s.state.clear_forces()
        s.pipeline.collide(s.state, s.contacts)
        coverage = collision_coverage(s.pipeline, s.contacts)
        stamp = final_upload()
        if stamp.before_step != s.step_id:
            raise FasteningFault('upload stamp does not precede this solve')
        s.integrate()
        captured = self.bound_model.clock()
        row, raw = self._read(stamp, captured, coverage)
        self._last_step = s.step_id
        return row, raw

    def _read(self, stamp, captured, coverage):
        s = self.scene
        q = s.state.joint_q.numpy()
        qd = s.state.joint_qd.numpy()
        poses = s.state.body_q.numpy()
        velocities = s.state.body_qd.numpy()
        d = s.solver.mjw_data
        effort = d.qfrc_actuator.numpy()[0]
        applied, bodyforce = d.qfrc_applied.numpy(), d.xfrc_applied.numpy()
        if np.any(applied) or np.any(bodyforce):
            raise FasteningFault('unadmitted external joint/body wrench')
        ctrl = d.ctrl.numpy()[0]
        roll_actuator = s._actuators['right_wrist_roll_joint']
        if ctrl[roll_actuator] != np.float32(stamp.effort_nm):
            raise FasteningFault('wrist upload and native control disagree')
        joints = q[s.q_index]
        bounds = self.geometry.evaluate(joints, float(q[s._roll_q_index]), poses[s.tool_body])
        pairs, records = contact_records(s, self.output, _zero_pairs=self._zero_pairs)
        threshold = self.limits.contact_load_threshold_n
        loaded = [tuple(sorted((p.collider_a, p.collider_b))) for p in pairs if p.normal_force_n > threshold]
        nut, tool = poses[s.nut_body], poses[s.tool_body]
        nut_v, tool_v = velocities[s.nut_body], velocities[s.tool_body]
        row = FasteningSolve(self.binding.sha256, s.epoch, stamp.generation, s.step_id, s.time_s, captured,
            tuple(map(float, joints)), tuple(map(float, qd[s.qd_index])),
            tuple(float(effort[i]) for i in s.native_dofs), *bounds,
            tuple(map(float, nut[:3])), tuple(map(float, nut[3:])), self.binding.fixture_origin_m, (0., 0., 0., 1.),
            tuple(map(float, tool[:3])), tuple(map(float, tool[3:])),
            float(np.linalg.norm(nut_v[:3])), float(np.linalg.norm(nut_v[3:])),
            float(np.linalg.norm(tool_v[:3])), float(np.linalg.norm(tool_v[3:])),
            float(qd[s._roll_qd_index]), float(effort[s.roll_dof]), stamp.effort_nm,
            pairs, loaded.count(self.binding.thread_contact_pair),
            sum(p in self.binding.tool_contact_pairs for p in loaded),
            coverage['contacts']['capacity'], coverage['contacts']['count'], int(d.naconmax), len(pairs))
        # Persisted evidence: the full typed row stays in the journal; on disk the
        # candidate list is summarised as every LOADED pair (short labels) plus counts.
        solve = asdict(row)
        solve['contacts'] = None
        loaded_pairs = [(p.collider_a.rsplit('/', 1)[-1], p.collider_b.rsplit('/', 1)[-1], p.normal_force_n)
                        for p in pairs if p.normal_force_n > threshold]
        raw = {'solve': solve, 'contact_candidates': len(pairs), 'loaded_contacts': loaded_pairs,
               'upload': asdict(stamp), 'collision_buffers': coverage,
               'constraint_count': int(d.nefc.numpy()[0]), 'constraint_capacity': int(d.njmax),
               'program': {'phase': s.program.phase, 'hand': s.program.hand, 'roll_ref': s.program.roll_ref,
                           'strokes': s.program.strokes, 'regrips': s.program.regrips},
               'grasp_weld_enabled': bool(s.grasp_enabled), 'wrist_roll_rad': float(q[s._roll_q_index]),
               'socket_floor_load_n': float(sum(p.normal_force_n for p in pairs
                   if tuple(sorted((p.collider_a, p.collider_b))) == self.floor_pair)),
               'tool_wall_load_n': float(sum(p.normal_force_n for p in pairs
                   if tuple(sorted((p.collider_a, p.collider_b))) in self.binding.tool_contact_pairs)),
               'thread_load_n': float(sum(p.normal_force_n for p in pairs
                   if tuple(sorted((p.collider_a, p.collider_b))) == self.binding.thread_contact_pair)),
               'body_poses_xyzw': poses.tolist() if s.step_id % self.pose_every == 0 else None}
        return row, raw
