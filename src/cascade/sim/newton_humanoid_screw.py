"""Unitree G1 humanoid turns a hand-held hex driver with its wrist (Newton, SDF thread contact).

Optional standalone Newton experiment built on the same Factory M20 nut/bolt
recipe as ``newton_screw_contact.ThreadingScene`` (SDF meshes, contact
material, MJWarp solver settings, fixture at world (0.24, 0, 0) with +Z axis).
The humanoid holds a socket driver whose axis is its forearm (wrist-roll) axis
and tightens the free nut by rolling its right wrist, re-gripping between
strokes because the wrist-roll range (+-1.97 rad) is shorter than one turn.

After initialisation only robot joint actuator controls and the grasp state
change. There is no nut drive, helical joint, external nut wrench, runtime
body-pose write or kinematic tool motion: the nut turns and advances only
through solved tool/nut and nut/bolt contacts driven by the wrist actuator.

Declared simplifications (``SIMPLIFICATIONS``, also written into every summary):
- pelvis fixed to the world: standing balance is not simulated;
- the grasp is a weld equality between the right hand (``right_wrist_yaw_link``)
  and the driver; a re-grip opens the fingers (weld off), rolls the wrist back,
  closes them and re-enables the weld at the CURRENT relative hand/tool pose
  (no pose correction); finger motion is cosmetic. Only the right hand keeps
  its asset collision geometry (excluded against its own driver), so a hand
  contact with the nut, bolt or table is a forbidden pair for the verifier;
- gravity compensation is a feed-forward through the joint actuators
  (MuJoCo ``actuatorgravcomp``) on every joint except the turning wrist roll;
- the driver starts mounted on the pre-engaged nut (as in the SO-101 fixture);
  pickup, initial engagement, seating and preload are not part of this scene.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import uuid
import xml.etree.ElementTree as ET

import numpy as np

G1_MJCF = 'unitree_g1/mjcf/g1_29dof_with_hand_rev_1_0.xml'
ROBOT_ID = 'unitree_g1_29dof_dex3_wrist_driver'
RECIPE = 'g1_wrist_driver_m20_v1'
SIMPLIFICATIONS = (
    'pelvis fixed to the world (standing balance not simulated)',
    'grasp = weld between right hand and driver; re-grip re-welds at the current relative pose',
    'only the right hand keeps collision geometry (excluded against its own driver); finger motion is cosmetic',
    'gravity compensation feed-forward through the joint actuators (not on the turning wrist roll)',
    'driver starts mounted on the pre-engaged nut (no pickup, engagement, seating or preload)',
)
ROLL_JOINT = 'right_wrist_roll_joint'
IK_JOINTS = ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint', 'right_shoulder_pitch_joint',
             'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint')
RIGHT_FINGERS = ('right_hand_thumb_0_joint', 'right_hand_thumb_1_joint', 'right_hand_thumb_2_joint',
                 'right_hand_index_0_joint', 'right_hand_index_1_joint',
                 'right_hand_middle_0_joint', 'right_hand_middle_1_joint')
# Cosmetic Dex3 postures, 0.05 rad inside every joint limit.
FINGERS_OPEN = (0., 0., -.05, .05, .05, .05, .05)
FINGERS_CLOSED = (0., -.35, -.45, .55, .6, .55, .6)
# Held (never re-commanded) posture of the other joints; legs straight, pelvis fixed.
HELD_POSE = {'left_shoulder_pitch_joint': .25, 'left_shoulder_roll_joint': .25, 'left_elbow_joint': .9,
             'left_hand_thumb_1_joint': .2, 'left_hand_thumb_2_joint': .3,
             'left_hand_index_0_joint': -.3, 'left_hand_index_1_joint': -.3,
             'left_hand_middle_0_joint': -.3, 'left_hand_middle_1_joint': -.3,
             'right_wrist_pitch_joint': 0., 'right_wrist_yaw_joint': 0.}
# Right-hand links that keep their asset collision geometry: any loaded contact of
# them with the nut, bolt or table is a forbidden pair for the fastening verifier.
HAND_COLLIDERS = ('right_wrist_yaw_link', 'right_hand_thumb_0_link', 'right_hand_thumb_1_link',
                  'right_hand_thumb_2_link', 'right_hand_middle_0_link', 'right_hand_middle_1_link',
                  'right_hand_index_0_link', 'right_hand_index_1_link')
# IK seed found by a sampled search with a vertical right forearm (docs).
IK_SEED = (.201, .237, .211, -.745, -.981, 2.294, .684)


def _gains(joint):
    if 'hand' in joint:
        return 5., .1
    if 'wrist' in joint:
        return 20., .5
    if 'shoulder' in joint or 'elbow' in joint:
        return 150., 4.
    return 300., 8.


def quat_wxyz(r):
    w = math.sqrt(max(0., 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    if w > 1e-6:
        return np.array([w, (r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w),
                         (r[1, 0] - r[0, 1]) / (4 * w)])
    i = int(np.argmax(np.diag(r)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(0., 1 + r[i, i] - r[j, j] - r[k, k])) * 2
    q = np.zeros(4)
    q[0] = (r[k, j] - r[j, k]) / s
    q[1 + i] = s / 4
    q[1 + j] = (r[j, i] + r[i, j]) / s
    q[1 + k] = (r[k, i] + r[i, k]) / s
    return q


def rot_xyzw(q):
    x, y, z, w = q
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                     [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                     [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])


class WristDriverProgram:
    """Pure wrist routine: stroke (gripped), settle, release, roll back, settle, re-grip.

    Inputs are the MEASURED wrist-roll angle/speed and whether the commanded
    fingers reached their posture; outputs are the wrist reference and the hand
    posture. It never reads or writes the nut, and it has no completion claim:
    the caller (verifier/domain) decides when the turn is done and stops it.
    """

    def __init__(self, *, roll_low, roll_high, stroke_speed, return_speed, settle_speed=.05,
                 settle_error=.02, hold_s=.15, regrip_s=.2):
        if not roll_low < roll_high or min(stroke_speed, return_speed, settle_speed) <= 0:
            raise ValueError('invalid wrist program limits')
        self.roll_low, self.roll_high = roll_low, roll_high
        self.stroke_speed, self.return_speed = stroke_speed, return_speed
        self.settle_speed, self.settle_error = settle_speed, settle_error
        self.hold_s, self.regrip_s = hold_s, regrip_s
        self.phase, self.phase_t, self.roll_ref, self.hand = 'idle', 0., None, 'closed'
        self.strokes = self.regrips = 0
        self.events = []

    def _set(self, phase, time_s):
        self.phase, self.phase_t = phase, 0.
        self.events.append((round(time_s, 6), phase))

    def start(self, roll, time_s):
        """A new command resumes from the measured wrist angle; it never jumps."""
        self.roll_ref = float(roll)
        if self.hand == 'closed':
            self._set('stroke' if roll < self.roll_high - .05 else 'settle_stroke', time_s)
        else:
            self._set('return', time_s)

    def halt(self, time_s):
        if self.phase != 'stopped':
            self._set('stopped', time_s)

    def advance(self, dt, roll, roll_speed, fingers_done, time_s):
        self.phase_t += dt
        settled = abs(roll_speed) <= self.settle_speed and abs(self.roll_ref - roll) <= self.settle_error
        if self.phase == 'stroke':
            self.roll_ref = min(self.roll_high, self.roll_ref + self.stroke_speed * dt)
            if self.roll_ref >= self.roll_high:
                self._set('settle_stroke', time_s)
        elif self.phase == 'settle_stroke':
            if self.phase_t >= self.hold_s and settled:
                self.hand = 'open'
                self._set('release', time_s)
        elif self.phase == 'release':
            if fingers_done:
                self._set('return', time_s)
        elif self.phase == 'return':
            self.roll_ref = max(self.roll_low, self.roll_ref - self.return_speed * dt)
            if self.roll_ref <= self.roll_low:
                self._set('settle_return', time_s)
        elif self.phase == 'settle_return':
            if self.phase_t >= self.hold_s and settled:
                self.hand = 'closed'
                self.regrips += 1
                self._set('regrip', time_s)
        elif self.phase == 'regrip':
            if fingers_done and self.phase_t >= self.regrip_s:
                self.strokes += 1
                self._set('stroke', time_s)
        return self.roll_ref, self.hand


class HumanoidThreadingScene:
    pitch_m = .0025
    frame_dt = 1 / 60
    motor_limit_nm = 1.          # wrist-roll torque cap of this task (G1 actuator: 25 N m)
    stroke_speed_rad_s = 2.
    return_speed_rad_s = 4.
    roll_kp, roll_kd = 6., .15   # host PD of the turning wrist roll, torque-limited
    roll_low, roll_high = -1.62, 1.62
    max_joint_speed_rad_s = 1.5  # rate limit of every position target (fingers, arm follower)
    socket_x_m = .30             # driver socket centre along the hand x axis (wrist_yaw_link)
    fixture_center_xy_m = (.24, 0.)
    pelvis_world_m = (.04, .28, .369)
    initial_nut_z = .069
    floor_gap_m = .001           # socket floor above the nut face at engagement
    fixture_recipe = 'factory_m20_fixed_axis_v1'

    def __init__(self, assets, robot_asset, cache, *, device='cuda:0', drive=True,
                 thread_friction=.1, socket_clearance_m=.0003, substeps=10, follow_nut=True):
        import mujoco
        import newton
        import trimesh
        import warp as wp
        from scipy.optimize import least_squares

        self.wp, self.newton, self.mujoco, self.least_squares = wp, newton, mujoco, least_squares
        self.assets, self.robot_asset = Path(assets), Path(robot_asset)
        if type(substeps) is not int or not 1 <= substeps <= 100:
            raise ValueError('substeps must be an integer in [1, 100]')
        if not math.isfinite(thread_friction) or thread_friction <= 0:
            raise ValueError('thread friction must be finite and positive')
        self.substeps, self.drive, self.follow_nut = substeps, drive, follow_nut
        self.thread_friction, self.socket_clearance_m = float(thread_friction), float(socket_clearance_m)
        self.dt = self.frame_dt / substeps
        manifest = json.loads((self.assets / 'manifest.json').read_text())
        for file, record in manifest['files'].items():
            if hashlib.sha256((self.assets / file).read_bytes()).hexdigest() != record['sha256']:
                raise ValueError(f'Factory asset hash mismatch: {file}')
        self.epoch = uuid.uuid4().hex
        self.time_s, self.step_id = 0., 0
        self.requested_turns = 1.
        self._active = False
        self._started_angle = None
        self._maximum_nut_angular_speed = 0.
        self.peak_nut_speed_by_phase = {}
        self._thread_contacts = self._tool_contacts = self._floor_contacts = 0
        self._thread_force = self._tool_force = 0.
        self._roll_torque = self._roll_command = 0.
        self._center_xy = np.array(self.fixture_center_xy_m)
        self.fixture_position = np.r_[self._center_xy, 0.]
        self.grasp_enabled = True
        self.program = WristDriverProgram(roll_low=self.roll_low, roll_high=self.roll_high,
            stroke_speed=self.stroke_speed_rad_s, return_speed=self.return_speed_rad_s)
        wp.init()
        wp.set_device(device)

        self.robot_xml = self._robot_xml()
        self._setup_ik(self.robot_xml)
        self.entry = self._ik(np.r_[self._center_xy, self.initial_nut_z], np.array(IK_SEED))
        self.bottom = self._ik(np.r_[self._center_xy, self.initial_nut_z - .012], self.entry)
        # Socket yaw at the initial wrist roll fixes the nut yaw (flats facing the walls).
        hand_r = self._fk(self.entry, self.roll_low)[1]
        tool_r = hand_r @ self.tool_in_hand_rotation
        nut_yaw = math.atan2(tool_r[1, 0], tool_r[0, 0])
        self._last_angle = self._unwrapped = nut_yaw

        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_shape_cfg.gap = 0.
        setattr(newton, 'use_coord_layout_targets', True)
        self.scene_xml = self._scene_xml(self.robot_xml, socket_clearance_m)
        builder.add_mjcf(self.scene_xml, ctrl_direct=True, parse_visuals=True, parse_meshes=True,
                         enable_self_collisions=False, collapse_fixed_joints=False,
                         convert_mjc_equality_constraints=False)
        actuators = [node.attrib['name'] for node in ET.fromstring(self.scene_xml).find('actuator')]
        self._actuators = {name: i for i, name in enumerate(actuators)}
        self.joint_names = tuple(name for name in actuators if name != ROLL_JOINT)
        cfg = builder.ShapeConfig(margin=0., mu=self.thread_friction, ke=1e7, kd=1e4,
                                  gap=.005, density=8000., mu_torsional=0., mu_rolling=0.)
        self.asset_hashes = {}
        for name, file in [('bolt', 'factory_bolt_m20_loose.obj'),
                           ('nut', 'factory_nut_m20_loose_subdiv_3x.obj')]:
            path = self.assets / file
            self.asset_hashes[file] = hashlib.sha256(path.read_bytes()).hexdigest()
            data = trimesh.load(path, force='mesh')
            vertices = np.asarray(data.vertices, dtype=np.float32)
            center = (vertices.min(0) + vertices.max(0)) / 2
            mesh = newton.Mesh(vertices - center, np.asarray(data.faces.flatten(), dtype=np.int32))
            mesh.build_sdf(max_resolution=512, narrow_band_range=(-.005, .005),
                           margin=.005, cache_dir=Path(cache))
            rotation = wp.quat_identity() if name == 'bolt' else wp.quat_from_axis_angle(wp.vec3(0, 0, 1), nut_yaw)
            position = self.fixture_position + np.array([0., 0., .041 if name == 'nut' else 0.])
            position += np.asarray(wp.quat_rotate(rotation, wp.vec3(*center)))
            if name == 'bolt':
                body, pose = -1, wp.transform(wp.vec3(*position), rotation)
            else:
                body = builder.add_body(xform=wp.transform(wp.vec3(*position), rotation), label='nut')
                pose = wp.transform_identity()
                self.nut_body = body
            color = (.65, .70, .76) if name == 'bolt' else (.15, .55, .40)
            builder.add_shape_mesh(body, xform=pose, mesh=mesh, cfg=cfg, label=name, color=color)
        # Work table (top at z = 0, static collider) and visual legs/floor.
        self.floor_z = self.pelvis_world_m[2] - .793
        builder.add_shape_box(-1, xform=wp.transform(wp.vec3(.30, -.02, -.015), wp.quat_identity()),
                              hx=.22, hy=.14, hz=.015, label='bench', color=(.45, .33, .22))
        visual = builder.ShapeConfig(has_shape_collision=False, has_particle_collision=False)
        for i, (x, y) in enumerate(((.1, -.14), (.1, .1), (.5, -.14), (.5, .1))):
            builder.add_shape_box(-1, xform=wp.transform(wp.vec3(x, y, (self.floor_z - .03) / 2), wp.quat_identity()),
                                  hx=.015, hy=.015, hz=(-self.floor_z - .03) / 2, cfg=visual,
                                  label=f'bench_leg_{i}', color=(.35, .26, .18))
        builder.add_shape_box(-1, xform=wp.transform(wp.vec3(.2, .1, self.floor_z - .005), wp.quat_identity()),
                              hx=1.2, hy=1.2, hz=.005, cfg=visual, label='floor', color=(.30, .32, .36))
        self.model = builder.finalize(device=device)
        self.bodies = self._names(self.model.body_label)
        self.joints = self._names(self.model.joint_label)
        self.tool_body = self.bodies['driver']
        self.hand_body = self.bodies['right_wrist_yaw_link']
        self._qstart = self.model.joint_q_start.numpy()
        self._qdstart = self.model.joint_qd_start.numpy()
        self.q_index = np.array([self._qstart[self.joints[n]] for n in self.joint_names])
        self.qd_index = np.array([self._qdstart[self.joints[n]] for n in self.joint_names])
        self.ctrl_index = np.array([self._actuators[n] for n in self.joint_names])
        self._roll_q_index = int(self._qstart[self.joints[ROLL_JOINT]])
        self._roll_qd_index = int(self._qdstart[self.joints[ROLL_JOINT]])
        self.targets = self._initial_targets()
        q = self.model.joint_q.numpy()
        q[self.q_index] = self.targets
        q[self._roll_q_index] = self.roll_low
        start = self._qstart[self.joints['driver_free']]
        q[start:start + 7] = self._initial_tool_pose()
        self.model.joint_q.assign(q)  # Initial condition only, before the first solve.
        self.state, self.next = self.model.state(), self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.next)
        self.control = self.model.control()
        self.solver = newton.solvers.SolverMuJoCo(self.model, use_mujoco_contacts=False,
            solver='newton', integrator='implicitfast', cone='elliptic', njmax=4096,
            nconmax=2048, iterations=50, ls_iterations=100, impratio=1., update_data_interval=1)
        self.pipeline = newton.CollisionPipeline(self.model, reduce_contacts=True,
                                                 rigid_contact_max=2048, verify_buffers=True)
        self.contacts = self.pipeline.contacts()
        self.apply_controls(self.targets, 0.)
        shape_map = self.solver.mjc_geom_to_newton_shape.numpy()[0]
        labels = self._names(self.model.shape_label)
        self.shape = {name: labels[name] for name in ('nut', 'bolt', 'socket_floor', 'bench')}
        self.shape.update({f'socket_wall_{i}': labels[f'socket_wall_{i}'] for i in range(6)})
        self.nut_geom = int(np.flatnonzero(shape_map == labels['nut'])[0])
        self.bolt_geom = int(np.flatnonzero(shape_map == labels['bolt'])[0])
        self.socket_geoms = {int(np.flatnonzero(shape_map == labels[f'socket_wall_{i}'])[0]) for i in range(6)}
        self.floor_geom = int(np.flatnonzero(shape_map == labels['socket_floor'])[0])
        dof_map = self.solver.mjc_dof_to_newton_dof.numpy()[0]
        self.roll_dof = int(np.flatnonzero(dof_map == self._roll_qd_index)[0])
        self.native_dofs = np.array([int(np.flatnonzero(dof_map == i)[0]) for i in self.qd_index])
        if self.model.mujoco.equality_constraint_count != 1:
            raise RuntimeError('expected exactly one grasp weld equality')
        self.lower = self.model.joint_limit_lower.numpy()[self.qd_index]
        self.upper = self.model.joint_limit_upper.numpy()[self.qd_index]
        self.effort_caps = np.array([float(self._joint_limit(n)) for n in self.joint_names])

    # ------------------------------------------------------------------ authoring
    @property
    def tool_in_hand_rotation(self):
        # Driver +z (towards the handle) is hand -x; driver x is hand +y.
        return np.array([[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]])

    @staticmethod
    def _names(labels):
        return {label.rsplit('/', 1)[-1]: i for i, label in enumerate(labels)}

    def _joint_limit(self, name):
        node = ET.fromstring(self.robot_xml).find(f".//joint[@name='{name}']")
        return node.get('actuatorfrcrange', '-25 25').split()[1]

    def _robot_xml(self):
        root = ET.parse(self.robot_asset).getroot()
        root.find('compiler').set('meshdir', str(self.robot_asset.parent.parent / 'meshes'))
        for node in list(root):
            if node.tag in ('sensor', 'visual', 'statistic'):
                root.remove(node)
        for extra in root.findall('worldbody')[1:]:   # the menagerie floor/light block
            root.remove(extra)
        for asset in root.findall('asset')[1:]:
            root.remove(asset)
        pelvis = root.find(".//body[@name='pelvis']")
        pelvis.remove(pelvis.find('joint'))  # fixed base: balance is not simulated
        pelvis.set('pos', ' '.join(map(str, self.pelvis_world_m)))
        for body in root.iter('body'):
            body.set('gravcomp', '1')
            for geom in list(body.findall('geom')):
                if geom.get('contype') == '0' and geom.get('conaffinity') == '0':
                    geom.set('group', '1')
                elif body.get('name') not in HAND_COLLIDERS:
                    body.remove(geom)        # only the right hand keeps its collision geometry
        actuator = root.find('actuator')
        for node in list(actuator):
            actuator.remove(node)
        for joint in root.iter('joint'):
            name = joint.get('name')
            limit = joint.get('actuatorfrcrange', '-25 25').split()
            if name == ROLL_JOINT:
                ET.SubElement(actuator, 'motor', name=name, joint=name, gear='1', ctrllimited='true',
                              ctrlrange=f'-{self.motor_limit_nm} {self.motor_limit_nm}', forcelimited='true',
                              forcerange=f'-{self.motor_limit_nm} {self.motor_limit_nm}')
                joint.set('damping', '.02')
                continue
            joint.set('actuatorgravcomp', 'true')
            kp, kv = _gains(name)
            ET.SubElement(actuator, 'position', name=name, joint=name, kp=str(kp), kv=str(kv),
                          ctrllimited='true', ctrlrange=joint.get('range'), forcelimited='true',
                          forcerange=f'{limit[0]} {limit[1]}')
        return ET.tostring(root, encoding='unicode')

    def _scene_xml(self, robot_xml, clearance):
        root = ET.fromstring(robot_xml)
        world = root.find('worldbody')
        tool = ET.SubElement(world, 'body', name='driver', pos='0 0 1')
        ET.SubElement(tool, 'freejoint', name='driver_free')
        ET.SubElement(tool, 'inertial', pos='0 0 .07', mass='.15', diaginertia='.00025 .00025 .00002')
        no_contact = dict(contype='0', conaffinity='0', group='1', mass='0')
        ET.SubElement(tool, 'geom', name='driver_handle', type='cylinder', size='.014 .0375',
                      pos=f'0 0 {self.socket_x_m - .1625}', rgba='.85 .25 .1 1', **no_contact)
        ET.SubElement(tool, 'geom', name='driver_shaft', type='cylinder', size='.005 .03',
                      pos='0 0 .07', rgba='.75 .76 .8 1', **no_contact)
        ET.SubElement(tool, 'geom', name='driver_socket_case', type='cylinder', size='.021 .0075',
                      pos='0 0 .0325', rgba='.55 .58 .62 1', **no_contact)
        for i in range(6):
            a = i * math.pi / 3
            radius = .015 + clearance + .003
            ET.SubElement(tool, 'geom', name=f'socket_wall_{i}', type='box',
                          pos=f'{radius*math.cos(a)} {radius*math.sin(a)} 0',
                          quat=f'{math.cos(a/2)} 0 0 {math.sin(a/2)}', size='.003 .010 .010',
                          rgba='.6 .65 .72 1', contype='1', conaffinity='1',
                          friction='.3 0 0', solref='.003 1', mass='0', group='2')
        # Socket floor just above the nut face; it only carries the released driver
        # axially. Frictionless (MJWarp minimum) so turning torque reaches the nut
        # through the hex walls alone, never through face friction.
        ET.SubElement(tool, 'geom', name='socket_floor', type='cylinder', size='.021 .0075',
                      pos=f'0 0 {.008 + self.floor_gap_m + .0075}', rgba='.55 .58 .62 1', contype='1',
                      conaffinity='1', friction='1e-5 0 0', solref='.003 1', mass='0', group='2')
        contact = root.find('contact')
        if contact is None:
            contact = ET.SubElement(root, 'contact')
        for body in HAND_COLLIDERS:           # the held driver never collides with its own hand
            ET.SubElement(contact, 'exclude', body1=body, body2='driver')
        quat = quat_wxyz(self.tool_in_hand_rotation)
        equality = ET.SubElement(root, 'equality')
        ET.SubElement(equality, 'weld', name='grasp', body1='right_wrist_yaw_link', body2='driver',
                      relpose=' '.join(f'{v:.9g}' for v in (self.socket_x_m, 0., 0., *quat)), solref='.004 1')
        return ET.tostring(root, encoding='unicode')

    def _setup_ik(self, robot_xml):
        """CPU kinematics on the same robot document (no tool), used only for targets/bounds."""
        self.ik_model = self.mujoco.MjModel.from_xml_string(robot_xml)
        self.ik_data = self.mujoco.MjData(self.ik_model)
        self.ik_hand = self.ik_model.body('right_wrist_yaw_link').id
        self.ik_indices = np.array([int(self.ik_model.joint(n).qposadr[0]) for n in IK_JOINTS])
        self.ik_ranges = np.array([self.ik_model.joint(n).range for n in IK_JOINTS])
        self.ik_roll = int(self.ik_model.joint(ROLL_JOINT).qposadr[0])
        self.ik_static = {int(self.ik_model.joint(n).qposadr[0]): v for n, v in HELD_POSE.items()}

    def _fk(self, q, roll=0.):
        d = self.ik_data
        d.qpos[:] = 0.
        for index, value in self.ik_static.items():
            d.qpos[index] = value
        d.qpos[self.ik_indices] = q
        d.qpos[self.ik_roll] = roll
        self.mujoco.mj_kinematics(self.ik_model, d)
        return d.xpos[self.ik_hand].copy(), d.xmat[self.ik_hand].reshape(3, 3).copy()

    def _ik(self, target, seed):
        half = (self.ik_ranges[:, 1] - self.ik_ranges[:, 0]) / 2

        def residual(q):
            p, r = self._fk(q)
            socket = p + r[:, 0] * self.socket_x_m
            return np.r_[(socket - target) * 10, (r[:, 0] - [0., 0., -1.]) * 3,
                         .002 * (q - np.asarray(IK_SEED)) / half]
        result = self.least_squares(residual, seed, bounds=(self.ik_ranges[:, 0] + .05,
                                    self.ik_ranges[:, 1] - .05), max_nfev=400, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        p, r = self._fk(result.x)
        if np.linalg.norm(p + r[:, 0] * self.socket_x_m - target) > 2e-5 or r[2, 0] > -.99999:
            raise ValueError('socket centre is not reachable with a vertical G1 forearm')
        return result.x

    def _initial_targets(self):
        values = dict(zip(IK_JOINTS, map(float, self.entry)))
        values.update(HELD_POSE)
        values.update(zip(RIGHT_FINGERS, FINGERS_CLOSED))
        return np.array([values.get(name, 0.) for name in self.joint_names])

    def _initial_tool_pose(self):
        p, hand_r = self._fk(self.entry, self.roll_low)
        position = p + hand_r[:, 0] * self.socket_x_m
        w, x, y, z = quat_wxyz(hand_r @ self.tool_in_hand_rotation)
        return np.r_[position, x, y, z, w]

    # ------------------------------------------------------------------ control primitives
    def arm_target(self, nut_z):
        """IK follower of the measured nut height (keeps the socket depth); no pose writes."""
        if not self.follow_nut:
            return self.entry
        fraction = float(np.clip((self.initial_nut_z - nut_z) / .012, 0., 1.))
        return self.entry * (1 - fraction) + self.bottom * fraction

    def desired_targets(self, nut_z, hand):
        values = dict(zip(self.joint_names, self.targets))
        values.update(zip(IK_JOINTS, map(float, self.arm_target(nut_z))))
        values.update(zip(RIGHT_FINGERS, FINGERS_CLOSED if hand == 'closed' else FINGERS_OPEN))
        return np.array([values[name] for name in self.joint_names])

    def fingers_at(self, targets, hand):
        index = [self.joint_names.index(n) for n in RIGHT_FINGERS]
        posture = FINGERS_CLOSED if hand == 'closed' else FINGERS_OPEN
        return bool(np.array_equal(np.asarray(targets)[index], np.asarray(posture)))

    def wrist_torque(self, roll_ref, roll, roll_speed):
        torque = self.roll_kp * (roll_ref - roll) - self.roll_kd * roll_speed
        return float(np.clip(torque, -self.motor_limit_nm, self.motor_limit_nm))

    def roll_state(self):
        q = self.state.joint_q.numpy()
        qd = self.state.joint_qd.numpy()
        return float(q[self._roll_q_index]), float(qd[self._roll_qd_index])

    def apply_controls(self, targets, roll_torque):
        ctrl = self.control.mujoco.ctrl.numpy()
        ctrl[self.ctrl_index] = targets
        ctrl[self._actuators[ROLL_JOINT]] = roll_torque
        self.control.mujoco.ctrl.assign(ctrl)
        self.targets = np.array(targets, dtype=float)
        self._roll_command = float(roll_torque)
        # The grasp follows the commanded hand: welded only while the fingers are closed.
        self.set_grasp(self.fingers_at(self.targets, 'closed'))

    def set_grasp(self, on):
        if on == self.grasp_enabled:
            return
        if on:
            # Re-grip where the hand is now: weld relpose := current hand->driver transform.
            body_q = self.state.body_q.numpy()
            hand, tool = body_q[self.hand_body], body_q[self.tool_body]
            rh = rot_xyzw(hand[3:])
            rel_p = rh.T @ (tool[:3] - hand[:3])
            w, x, y, z = quat_wxyz(rh.T @ rot_xyzw(tool[3:]))
            relpose = self.model.mujoco.equality_constraint_relpose.numpy()
            relpose[0] = np.r_[rel_p, x, y, z, w]
            self.model.mujoco.equality_constraint_relpose.assign(relpose)
        enabled = self.model.mujoco.equality_constraint_enabled.numpy()
        enabled[0] = on
        self.model.mujoco.equality_constraint_enabled.assign(enabled)
        self.solver.notify_model_changed(self.newton.ModelFlags.CONSTRAINT_PROPERTIES)
        self.grasp_enabled = on

    def solve(self):
        """Exactly one physics solve of ``self.dt``; no state writes."""
        self.state.clear_forces()
        self.pipeline.collide(self.state, self.contacts)
        self.integrate()

    def integrate(self):
        """Advance one solve on the contacts collided from the current state."""
        self.solver.step(self.state, self.next, self.control, self.contacts, self.dt)
        self.state, self.next = self.next, self.state
        self.step_id += 1
        self.time_s = self.step_id * self.dt
        if np.any(self.solver.mjw_data.overflow.numpy()):
            raise RuntimeError('contact/constraint buffer overflow')
        for field in ('body_q', 'body_qd', 'joint_q', 'joint_qd'):
            if not np.isfinite(getattr(self.state, field).numpy()).all():
                raise RuntimeError('non-finite solved physics state')
        self._read_contacts()
        speed = float(np.linalg.norm(self.state.body_qd.numpy()[self.nut_body, 3:]))
        self._maximum_nut_angular_speed = max(self._maximum_nut_angular_speed, speed)
        peak = self.peak_nut_speed_by_phase.get(self.program.phase)
        if peak is None or speed > peak[0]:
            self.peak_nut_speed_by_phase[self.program.phase] = (speed, self.time_s)
        self._roll_torque = float(self.solver.mjw_data.qfrc_actuator.numpy()[0, self.roll_dof])
        x, y, z, w = self.state.body_q.numpy()[self.nut_body, 3:]
        angle = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        self._unwrapped += math.atan2(math.sin(angle-self._last_angle), math.cos(angle-self._last_angle))
        self._last_angle = angle

    def _read_contacts(self):
        data = self.solver.mjw_data
        count = int(data.nacon.numpy()[0])
        geoms = data.contact.geom.numpy()[:count]
        addresses = data.contact.efc_address.numpy()[:count, 0]
        forces = data.efc.force.numpy()[0]
        thread, tool, floor, thread_force, tool_force = 0, 0, 0, 0., 0.
        for pair, address in zip(geoms, addresses, strict=True):
            if address < 0 or forces[address] <= 1e-6:
                continue
            pair = set(map(int, pair))
            if pair == {self.nut_geom, self.bolt_geom}:
                thread += 1; thread_force += float(forces[address])
            elif self.nut_geom in pair and pair.intersection(self.socket_geoms):
                tool += 1; tool_force += float(forces[address])
            elif pair == {self.nut_geom, self.floor_geom}:
                floor += 1
        self._thread_contacts, self._tool_contacts, self._floor_contacts = thread, tool, floor
        self._thread_force, self._tool_force = thread_force, tool_force

    # ------------------------------------------------------------------ standalone episode loop
    def command(self, direction='tighten', turns=1.):
        if direction != 'tighten' or not math.isfinite(turns) or turns <= 0:
            raise ValueError('this wrist program implements finite positive tightening only')
        self.requested_turns = turns
        self._started_angle = self._unwrapped
        self._active = True
        self.program.start(self.roll_state()[0], self.time_s)

    def stop(self):
        """Latch: hold every position target, exact zero wrist-roll torque from the next solve."""
        self._active = False
        self.program.halt(self.time_s)

    def turns_so_far(self):
        if self._started_angle is None:
            return 0.
        return -(self._unwrapped - self._started_angle) / (2 * math.pi)

    def plan(self):
        """One control decision from measured state; returns (targets, wrist torque)."""
        roll, roll_speed = self.roll_state()
        nut_z = float(self.state.body_q.numpy()[self.nut_body, 2])
        hand = self.program.hand
        fingers_done = self.fingers_at(self.targets, hand)
        roll_ref, hand = self.program.advance(self.dt, roll, roll_speed, fingers_done, self.time_s)
        desired = self.desired_targets(nut_z, hand)
        step = self.max_joint_speed_rad_s * self.dt
        targets = self.targets + np.clip(desired - self.targets, -step, step)
        # Snap exactly onto the posture once within one rate step (exact equality gates the grasp).
        close = np.abs(desired - self.targets) <= step
        targets[close] = desired[close]
        torque = self.wrist_torque(roll_ref, roll, roll_speed) if self.drive else 0.
        return targets, torque

    def step(self):
        for _ in range(self.substeps):
            if self._active:
                targets, torque = self.plan()
            else:
                targets, torque = self.targets, 0.   # stopped/idle: exact zero wrist effort
            self.apply_controls(targets, torque)
            self.solve()

    def observe(self):
        poses = self.state.body_q.numpy()
        nut = poses[self.nut_body]
        roll, roll_speed = self.roll_state()
        return {'epoch': self.epoch, 'step': self.step_id, 'time_s': self.time_s,
                'fastener_id': 'factory_nut_m20_loose', 'fixture_id': 'fixed_factory_bolt_m20_loose',
                'fastener_position_m': nut[:3].tolist(), 'fastener_quaternion_xyzw': nut[3:].tolist(),
                'fixture_position_m': self.fixture_position.tolist(), 'fixture_quaternion_xyzw': [0., 0., 0., 1.],
                'thread_contacts': self._thread_contacts, 'tool_contacts': self._tool_contacts,
                'socket_floor_contacts': self._floor_contacts,
                'angle_rad': self._unwrapped, 'axial_position_m': float(nut[2]), 'pitch_m': self.pitch_m,
                'thread_contact_force_n': self._thread_force, 'tool_contact_force_n': self._tool_force,
                'wrist_roll_rad': roll, 'wrist_roll_speed_rad_s': roll_speed,
                'wrist_roll_reference_rad': self.program.roll_ref, 'wrist_roll_command_nm': self._roll_command,
                'wrist_roll_torque_nm': self._roll_torque, 'grasp_weld_enabled': self.grasp_enabled,
                'phase': self.program.phase, 'strokes': self.program.strokes, 'regrips': self.program.regrips,
                'body_poses_xyzw': poses.tolist(),
                'maximum_nut_angular_speed_rad_s': self._maximum_nut_angular_speed,
                'motor_active': self._active}
