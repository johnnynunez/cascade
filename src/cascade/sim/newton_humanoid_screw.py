"""Unitree G1 humanoid turns a hand-held hex driver about its own axis (Newton, SDF thread contact).

Optional standalone Newton experiment built on the same Factory M20 nut/bolt
recipe as ``newton_screw_contact.ThreadingScene`` (SDF meshes, contact
material, MJWarp solver settings, fixture at world (0.24, 0, 0) with +Z axis).

Grasp (a person holding a screwdriver from above): the driver stands vertical
in the socket on the nut and the right Dex3 hand holds the top of its handle
palm-down, the THUMB on one side and the INDEX and MIDDLE fingers wrapped
around the opposite side. The hand holds the driver ONLY through solved
finger/handle contacts with friction: no weld, attachment, equality
constraint or driver pose write. The handle axis is the Dex3 palm normal
through the grasp centre, 98 mm from the wrist, so no single G1 wrist joint is
coaxial with it; the hand turns about the tool axis by a whole-arm motion
(waist + shoulder + elbow + wrist) that follows an inverse-kinematics table of
the hand rotation ``phi`` about the vertical tool axis. Strokes are limited by
the arm's reachable rotation range, so the hand re-grips between strokes: it
opens, rotates back around the handle and closes again while the driver stands
in the socket.

After initialisation only robot joint actuator controls change. Every joint is
a position actuator except the right wrist roll, which is torque-controlled
(host PD toward its table value) and is the domain's "spindle" channel: a stop
is an exact-zero wrist-roll torque while every other joint holds its target.
The nut turns and advances only through solved finger/handle, socket/nut and
nut/bolt contacts.

Declared simplifications (``SIMPLIFICATIONS``, also written into every summary).
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
GRASP = 'three_finger_contact'
SIMPLIFICATIONS = (
    'pelvis fixed to the world (standing balance not simulated)',
    'only the right hand keeps collision geometry (Dex3 asset meshes); the rest of the robot does not collide',
    'gravity compensation feed-forward through the joint actuators (passive on the wrist roll)',
    ('driver starts engaged on the pre-engaged nut with the fingers just open of the synthesized contact '
     'posture (no pickup, engagement, seating or preload); it stands in the socket while the hand re-grips'),
)
ROLL_JOINT = 'right_wrist_roll_joint'
# Joints of the whole-arm turn (IK table); the wrist roll among them is torque-controlled.
IK_JOINTS = ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint', 'right_shoulder_pitch_joint',
             'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint',
             ROLL_JOINT, 'right_wrist_pitch_joint', 'right_wrist_yaw_joint')
RIGHT_FINGERS = ('right_hand_thumb_0_joint', 'right_hand_thumb_1_joint', 'right_hand_thumb_2_joint',
                 'right_hand_index_0_joint', 'right_hand_index_1_joint',
                 'right_hand_middle_0_joint', 'right_hand_middle_1_joint')
# Closing direction of every right-hand joint (thumb_0 abduction is held at 0).
CLOSING_SIGN = {'right_hand_thumb_0_joint': 0, 'right_hand_thumb_1_joint': -1, 'right_hand_thumb_2_joint': -1,
                'right_hand_index_0_joint': 1, 'right_hand_index_1_joint': 1,
                'right_hand_middle_0_joint': 1, 'right_hand_middle_1_joint': 1}
FINGER_LINKS = {'thumb': ('right_hand_thumb_0_link', 'right_hand_thumb_1_link', 'right_hand_thumb_2_link'),
                'index': ('right_hand_index_0_link', 'right_hand_index_1_link'),
                'middle': ('right_hand_middle_0_link', 'right_hand_middle_1_link'),
                'palm': ('right_wrist_yaw_link',)}
FINGER_JOINTS = {'thumb': ('right_hand_thumb_1_joint', 'right_hand_thumb_2_joint'),
                 'index': ('right_hand_index_0_joint', 'right_hand_index_1_joint'),
                 'middle': ('right_hand_middle_0_joint', 'right_hand_middle_1_joint')}
# Grasp design in the right_wrist_yaw_link (hand) frame: +y is the palm normal,
# the Dex3 index/middle bases sit at z = +-28.5 mm and the thumb at z = 0.
HANDLE_RADIUS_M = .020     # 40 mm handle: index/middle inner faces are 31.2 mm apart, so a
                           # narrower handle meets them only at their inner edges
GRIP_X_M = .102            # tool axis offset along hand x (thumb at x 0.055-0.074, fingers from 0.112)
HANDLE_TOP_M = .021        # handle top 3.7 mm below the palm face (hand y)
HANDLE_LENGTH_M = .10
SOCKET_Y_M = .201          # socket centre along the tool axis (hand y): 80 mm shaft below the handle
GRIP_FRICTION = .8         # elastomer finger pads on a rubberised handle grip (MuJoCo max rule)
GRIP_SQUEEZE_RAD = .07     # closing target at most this far past the MEASURED finger angle
GRIP_REACH_RAD = .25       # closing goal past the synthesized contact (never reached on contact)
OPEN_BACKOFF_RAD = .45     # open posture: contact posture backed off by this much
OPEN_CLEARANCE_M = .004    # minimum hand/handle clearance required of the open posture
INITIAL_BACKOFF_RAD = .015 # fingers start this far open of the synthesized contact posture ...
INITIAL_SQUEEZE_RAD = .015 # ... with targets this far past it (light initial grip; a command re-grips)
FINGER_RATE_RAD_S = 1.     # rate limit of finger targets (finger kv 0.7: following lag <= 0.07 rad)
TOUCH_M = .0003            # synthesis: a link touches when its convex-hull distance <= 0.3 mm
# Robot placement: pelvis fixed at this world position (fixture at (0.24, 0, 0)).
PELVIS_WORLD_M = (-.05, .20, .369)
# Whole-arm turn table: seed solution at PHI_SEED (sampled search, docs) and continuation limits.
PHI_SEED = .356
IK_SEED = (.3538, .2164, .2206, -.2167, .0099, -1.0134, .831, -1.8519, -.0277, -1.0172)
TABLE_STEP_RAD = .02
DESCENT_TABLE_M = .006     # follower table depth (two turns advance the nut 5 mm)
TABLE_MAX_DQ_DPHI = 1.8    # stop the table where any joint would move faster than this per rad of phi
STROKE_MARGIN_RAD = .1     # stroke limits inside the table ends
# Held (never re-commanded) posture of the other joints; legs straight, pelvis fixed.
HELD_POSE = {'left_shoulder_pitch_joint': .25, 'left_shoulder_roll_joint': .25, 'left_elbow_joint': .9,
             'left_hand_thumb_1_joint': .2, 'left_hand_thumb_2_joint': .3,
             'left_hand_index_0_joint': -.3, 'left_hand_index_1_joint': -.3,
             'left_hand_middle_0_joint': -.3, 'left_hand_middle_1_joint': -.3}
# Right-hand links with their asset collision meshes: loaded contacts with the
# driver handle are the grasp; with the nut, bolt or table they are forbidden.
HAND_COLLIDERS = ('right_wrist_yaw_link', 'right_hand_thumb_0_link', 'right_hand_thumb_1_link',
                  'right_hand_thumb_2_link', 'right_hand_middle_0_link', 'right_hand_middle_1_link',
                  'right_hand_index_0_link', 'right_hand_index_1_link')


def _gains(joint):
    """Position-actuator gains (kp N m/rad, kv N m s/rad)."""
    if joint.startswith('right_hand_thumb'):
        return 16., .7   # squeeze 16 * 0.07 = 1.12 N m (actuator cap 1.4 N m)
    if joint.startswith(('right_hand_index', 'right_hand_middle')):
        return 10., .7   # squeeze 10 * 0.07 = 0.7 N m each: the two fingers balance the thumb
    if 'hand' in joint:
        return 5., .1
    if 'wrist' in joint:
        return 20., .5
    if 'shoulder' in joint or 'elbow' in joint:
        return 150., 4.
    return 300., 8.


def grip_target(goal, measured, sign, squeeze=GRIP_SQUEEZE_RAD):
    """Closing target clamped to ``squeeze`` past the measured angle (force-limited closing).

    A free finger is driven toward ``goal``; a finger stopped by contact gets a
    target exactly ``squeeze`` beyond where it is, so it presses with
    kp * squeeze and its following error never exceeds ``squeeze``.
    """
    if sign == 0:
        return float(goal)
    reach = sign * (goal - measured)
    return float(measured + sign * min(reach, squeeze)) if reach > 0 else float(goal)


def hand_rotation(phi):
    """Palm-down hand orientation rotated by phi clockwise (tightening) about world -z."""
    x = np.array([math.cos(-phi), math.sin(-phi), 0.])
    y = np.array([0., 0., -1.])
    return np.column_stack([x, y, np.cross(x, y)])


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
    """Pure turn routine: stroke (gripped), settle, release, rotate back, settle, re-grip.

    ``roll`` is the hand rotation phi about the tool axis (whole-arm turn).
    Inputs are the MEASURED phi/speed and whether the commanded fingers
    reached their posture; outputs are the phi reference and the hand
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
        """A new command resumes from the measured angle; it never jumps.

        Every command starts with a full re-grip from the open hand (release,
        rotate back to the stroke start, close until contact): the guard
        re-anchors every target on the measured joints at admission, and a
        squeeze re-applied from an already-loaded contact snapped the thumb
        (13-14 rad/s measured) where closing from open stayed below 1.4 rad/s.
        """
        self.roll_ref = float(roll)
        self._set('release' if self.hand == 'closed' else 'return', time_s)
        self.hand = 'open'

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
    motor_limit_nm = 1.          # wrist-roll (spindle channel) torque cap of this task (actuator: 25 N m)
    stroke_speed_rad_s = 1.      # hand rotation about the tool axis while gripping
    return_speed_rad_s = 1.5     # open-hand rotation back (joint speeds <= 1.5 * max dq/dphi)
    roll_kp, roll_kd = 10., .3   # host PD of the wrist roll toward its table value, torque-limited
    max_joint_speed_rad_s = 1.5  # rate limit of every position target (fingers, arm follower)
    fixture_center_xy_m = (.24, 0.)
    pelvis_world_m = PELVIS_WORLD_M
    initial_nut_z = .069
    floor_gap_m = .001           # socket floor above the nut face at engagement
    fixture_recipe = 'factory_m20_fixed_axis_v1'

    grip_enabled = True          # False only for the fingers-open negative control

    def __init__(self, assets, robot_asset, cache, *, device='cuda:0', drive=True, grip=None,
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
        grip = self.grip_enabled if grip is None else grip
        self.substeps, self.drive, self.grip, self.follow_nut = substeps, drive, bool(grip), follow_nut
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
        self.grip_state = {'normal_force_n': {k: 0. for k in FINGER_LINKS}, 'contacts': {k: 0 for k in FINGER_LINKS},
                           'torque_nm': 0., 'net_force_n': 0., 'relative_angle_rad': 0.}
        self._relative_last = self._relative_unwrapped = None
        self._phi_last = self._phi_unwrapped = None
        self._center_xy = np.array(self.fixture_center_xy_m)
        self.fixture_position = np.r_[self._center_xy, 0.]
        wp.init()
        wp.set_device(device)

        self.robot_xml = self._robot_xml()
        self._setup_ik(self.robot_xml)
        self._build_turn_tables()
        self.program = WristDriverProgram(roll_low=self.stroke_low, roll_high=self.stroke_high,
            stroke_speed=self.stroke_speed_rad_s, return_speed=self.return_speed_rad_s)
        self.entry = self.arm_target(self.stroke_low, self.initial_nut_z)
        self.grasp = self._synthesize_grasp()
        # Socket yaw at the initial hand pose fixes the nut yaw (flats facing the walls).
        tool_r = self._fk(self.entry)[1] @ self.tool_in_hand_rotation
        nut_yaw = math.atan2(tool_r[1, 0], tool_r[0, 0])
        self._last_angle = self._unwrapped = nut_yaw

        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_shape_cfg.gap = 0.
        newton.use_coord_layout_targets = True
        self.scene_xml = self._scene_xml(self.robot_xml, socket_clearance_m)
        builder.add_mjcf(self.scene_xml, ctrl_direct=True, parse_visuals=True, parse_meshes=True,
                         enable_self_collisions=True, collapse_fixed_joints=False,
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
        self._finger_index = np.array([self.joint_names.index(n) for n in RIGHT_FINGERS])
        self._arm_index = np.array([self.joint_names.index(n) for n in IK_JOINTS if n != ROLL_JOINT])
        self._roll_q_index = int(self._qstart[self.joints[ROLL_JOINT]])
        self._roll_qd_index = int(self._qdstart[self.joints[ROLL_JOINT]])
        self.targets = self._initial_targets()
        q = self.model.joint_q.numpy()
        q[self.q_index] = self._initial_positions()
        q[self._roll_q_index] = self.entry[IK_JOINTS.index(ROLL_JOINT)]
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
        self.shape = {name: labels[name] for name in ('nut', 'bolt', 'socket_floor', 'bench', 'driver_handle')}
        self.shape.update({f'socket_wall_{i}': labels[f'socket_wall_{i}'] for i in range(6)})
        self.nut_geom = int(np.flatnonzero(shape_map == labels['nut'])[0])
        self.bolt_geom = int(np.flatnonzero(shape_map == labels['bolt'])[0])
        self.handle_geom = int(np.flatnonzero(shape_map == labels['driver_handle'])[0])
        self.socket_geoms = {int(np.flatnonzero(shape_map == labels[f'socket_wall_{i}'])[0]) for i in range(6)}
        self.floor_geom = int(np.flatnonzero(shape_map == labels['socket_floor'])[0])
        # Every colliding shape of a right-hand link, by finger: the grasp contacts.
        shape_body = self.model.shape_body.numpy()
        flags = self.model.shape_flags.numpy()
        collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
        self.grip_shapes, self.grip_geoms = {}, {}
        for finger, links in FINGER_LINKS.items():
            bodies = {self.bodies[link] for link in links}
            shapes = [s for s in range(len(shape_body)) if shape_body[s] in bodies and flags[s] & collide]
            self.grip_shapes[finger] = tuple(shapes)
            for s in shapes:
                for g in np.flatnonzero(shape_map == s):
                    self.grip_geoms[int(g)] = finger
        if not all(self.grip_shapes.values()):
            raise RuntimeError('a grasping finger has no collision shape')
        dof_map = self.solver.mjc_dof_to_newton_dof.numpy()[0]
        self.roll_dof = int(np.flatnonzero(dof_map == self._roll_qd_index)[0])
        self.native_dofs = np.array([int(np.flatnonzero(dof_map == i)[0]) for i in self.qd_index])
        if self.model.mujoco.equality_constraint_count != 0:
            raise RuntimeError('the contact grasp has no equality constraint')
        self.lower = self.model.joint_limit_lower.numpy()[self.qd_index]
        self.upper = self.model.joint_limit_upper.numpy()[self.qd_index]
        self.effort_caps = np.array([float(self._joint_limit(n)) for n in self.joint_names])
        self._read_contacts()

    # ------------------------------------------------------------------ authoring
    @property
    def tool_in_hand_rotation(self):
        # Driver +z (towards the handle) is the hand -y (palm normal, up); driver x is hand x.
        x, z = np.array([1., 0., 0.]), np.array([0., -1., 0.])
        return np.column_stack([x, np.cross(z, x), z])

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
                else:                        # Dex3 collision meshes: elastomer pad friction
                    geom.set('friction', f'{GRIP_FRICTION} 0 0')
                    geom.set('solref', '.004 1')
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

    def _handle_z(self):
        """Handle centre along the driver +z axis (the socket centre is the driver origin)."""
        return SOCKET_Y_M - HANDLE_TOP_M - HANDLE_LENGTH_M / 2

    def _scene_xml(self, robot_xml, clearance):
        root = ET.fromstring(robot_xml)
        world = root.find('worldbody')
        tool = ET.SubElement(world, 'body', name='driver', pos='0 0 1')
        ET.SubElement(tool, 'freejoint', name='driver_free')
        ET.SubElement(tool, 'inertial', pos='0 0 .09', mass='.18', diaginertia='.0006 .0006 .00004')
        no_contact = {'contype': '0', 'conaffinity': '0', 'group': '1', 'mass': '0'}
        # Handle: the only driver part the hand touches (rubberised grip).
        ET.SubElement(tool, 'geom', name='driver_handle', type='cylinder',
                      size=f'{HANDLE_RADIUS_M} {HANDLE_LENGTH_M / 2}', pos=f'0 0 {self._handle_z()}',
                      rgba='.95 .78 .1 1', contype='1', conaffinity='1', friction=f'{GRIP_FRICTION} 0 0',
                      solref='.004 1', mass='0')
        shaft_top = self._handle_z() - HANDLE_LENGTH_M / 2
        ET.SubElement(tool, 'geom', name='driver_shaft', type='cylinder', size=f'.005 {(shaft_top - .04) / 2}',
                      pos=f'0 0 {(shaft_top + .04) / 2}', rgba='.75 .76 .8 1', **no_contact)
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
        # Socket floor just above the nut face; it carries the driver axially (also
        # while the hand is open). Frictionless (MJWarp minimum) so turning torque
        # reaches the nut through the hex walls alone, never through face friction.
        ET.SubElement(tool, 'geom', name='socket_floor', type='cylinder', size='.021 .0075',
                      pos=f'0 0 {.008 + self.floor_gap_m + .0075}', rgba='.55 .58 .62 1', contype='1',
                      conaffinity='1', friction='1e-5 0 0', solref='.003 1', mass='0', group='2')
        # Self-collisions stay on so the hand meets the driver handle; the hand links
        # never collide with each other (the rest of the robot has no collision geometry).
        contact = root.find('contact')
        if contact is None:
            contact = ET.SubElement(root, 'contact')
        for i, first in enumerate(HAND_COLLIDERS):
            for second in HAND_COLLIDERS[i + 1:]:
                ET.SubElement(contact, 'exclude', body1=first, body2=second)
        return ET.tostring(root, encoding='unicode')

    def _setup_ik(self, robot_xml):
        """CPU kinematics on the same robot document (no tool), used only for targets/bounds."""
        self.ik_model = self.mujoco.MjModel.from_xml_string(robot_xml)
        self.ik_data = self.mujoco.MjData(self.ik_model)
        self.ik_hand = self.ik_model.body('right_wrist_yaw_link').id
        self.ik_indices = np.array([int(self.ik_model.joint(n).qposadr[0]) for n in IK_JOINTS])
        self.ik_ranges = np.array([self.ik_model.joint(n).range for n in IK_JOINTS])
        self.ik_static = {int(self.ik_model.joint(n).qposadr[0]): v for n, v in HELD_POSE.items()}

    def _fk(self, q):
        d = self.ik_data
        d.qpos[:] = 0.
        for index, value in self.ik_static.items():
            d.qpos[index] = value
        d.qpos[self.ik_indices] = q
        self.mujoco.mj_kinematics(self.ik_model, d)
        return d.xpos[self.ik_hand].copy(), d.xmat[self.ik_hand].reshape(3, 3).copy()

    def _socket(self, q):
        p, r = self._fk(q)
        return p + r @ np.array([GRIP_X_M, SOCKET_Y_M, 0.]), r

    def _ik(self, phi, nut_z, seed, regularization):
        """Arm pose with the palm-down hand rotated by phi and the socket on the nut axis."""
        bounds = (self.ik_ranges[:, 0] + .06, self.ik_ranges[:, 1] - .06)
        target, rotation = np.r_[self._center_xy, nut_z], hand_rotation(phi)

        def residual(q):
            socket, r = self._socket(q)
            return np.r_[(socket - target) * 20, (r - rotation).ravel() * 2., regularization * (q - seed)]
        result = self.least_squares(residual, seed, bounds=bounds, max_nfev=200, xtol=1e-10, ftol=1e-10)
        socket, r = self._socket(result.x)
        error = max(float(np.linalg.norm(socket - target)), .01 * float(np.linalg.norm(r - rotation)))
        return result.x, error

    def _build_turn_tables(self):
        """IK tables q(phi) at the initial nut height and 6 mm lower, by continuation from IK_SEED."""
        seed, error = self._ik(PHI_SEED, self.initial_nut_z, np.array(IK_SEED), .02)
        if error > 2e-5:
            raise ValueError('the palm-down grasp pose is not reachable from the seed')
        rows = {0: seed}
        for direction in (1, -1):
            q, i = seed, 0
            while abs(i) < 400:
                i += direction
                nxt, error = self._ik(PHI_SEED + i * TABLE_STEP_RAD, self.initial_nut_z, q, .3)
                if error > 2e-5 or np.max(np.abs(nxt - q)) > TABLE_MAX_DQ_DPHI * TABLE_STEP_RAD:
                    break
                rows[i] = q = nxt
        keys = sorted(rows)
        self.phi_grid = PHI_SEED + np.array(keys) * TABLE_STEP_RAD
        self.table_top = np.array([rows[k] for k in keys])
        # The nut descends 2.5 mm per turn; the follower interpolates to a table 6 mm lower.
        # Keep the contiguous phi interval (around the seed) where both tables solve.
        bottom, valid = [], []
        for phi, q in zip(self.phi_grid, self.table_top):
            nxt, error = self._ik(phi, self.initial_nut_z - DESCENT_TABLE_M, q, .02)
            bottom.append(nxt)
            valid.append(error <= 2e-5 and np.max(np.abs(nxt - q)) < .2)
        centre = keys.index(0)
        lo = hi = centre
        while lo > 0 and valid[lo - 1]:
            lo -= 1
        while hi < len(valid) - 1 and valid[hi + 1]:
            hi += 1
        if not valid[centre]:
            raise ValueError('the nut descent is not reachable at the seed pose')
        self.phi_grid = self.phi_grid[lo:hi + 1]
        self.table_top = self.table_top[lo:hi + 1]
        self.table_bottom = np.array(bottom[lo:hi + 1])
        self.stroke_low = float(self.phi_grid[0] + STROKE_MARGIN_RAD)
        self.stroke_high = float(self.phi_grid[-1] - STROKE_MARGIN_RAD)
        if self.stroke_high - self.stroke_low < math.pi / 2:
            raise ValueError('turn table shorter than a quarter turn')
        steps = np.abs(np.diff(self.table_top, axis=0)).max(0) / TABLE_STEP_RAD
        self.max_dq_dphi = float(steps.max())
        self.table_sha256 = hashlib.sha256(np.round(np.r_[self.table_top.ravel(), self.table_bottom.ravel()],
                                                    9).tobytes()).hexdigest()

    # ------------------------------------------------------------------ grasp synthesis
    def _synthesis_model(self):
        """CPU copy of the robot with the handle fixed in the hand at its gripped pose."""
        root = ET.fromstring(self.robot_xml)
        hand = root.find(".//body[@name='right_wrist_yaw_link']")
        q = (math.cos(math.pi / 4), -math.sin(math.pi / 4), 0., 0.)   # cylinder z -> hand y
        ET.SubElement(hand, 'geom', name='synthesis_handle', type='cylinder',
                      size=f'{HANDLE_RADIUS_M} {HANDLE_LENGTH_M / 2}',
                      pos=f'{GRIP_X_M} {HANDLE_TOP_M + HANDLE_LENGTH_M / 2} 0',
                      quat=' '.join(map(str, q)), contype='0', conaffinity='0')
        model = self.mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        return model, self.mujoco.MjData(model)

    def _synthesize_grasp(self):
        """Close each finger until it touches the handle (convex-hull distance, CPU MuJoCo).

        Thumb (thumb_1/thumb_2), index and middle: curl both joints together until a
        link touches, then the distal joint alone until the distal link touches.
        Returns the contact posture, the closing goal past contact and an open
        posture with verified clearance, all for ``RIGHT_FINGERS``.
        """
        mj = self.mujoco
        m, d = self._synthesis_model()
        handle = m.geom('synthesis_handle').id
        fromto = np.zeros(6)
        geoms = {link: [g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name == link]
                 for f in FINGER_LINKS for link in FINGER_LINKS[f]}
        ik = [int(m.joint(n).qposadr[0]) for n in IK_JOINTS]

        def distance(values, links):
            d.qpos[:] = 0.
            for index, value in self.ik_static.items():
                d.qpos[index] = value
            d.qpos[ik] = self.entry
            for name, value in values.items():
                d.qpos[m.joint(name).qposadr[0]] = value
            mj.mj_kinematics(m, d)
            mj.mj_collision(m, d)
            return [min(mj.mj_geomDistance(m, d, g, handle, .05, fromto) for g in geoms[link]) for link in links]

        posture = {'right_hand_thumb_0_joint': 0.}
        for finger, (proximal, distal) in FINGER_JOINTS.items():
            sign = CLOSING_SIGN[proximal]
            lo = [m.joint(j).range[0] + .03 for j in (proximal, distal)]
            hi = [m.joint(j).range[1] - .03 for j in (proximal, distal)]
            q = [float(np.clip(0., lo[0], hi[0])), float(np.clip(0., lo[1], hi[1]))]
            links = FINGER_LINKS[finger]
            step = .002
            while min(distance({**posture, proximal: q[0], distal: q[1]}, links)) > TOUCH_M:
                n = [q[0] + sign * step, q[1] + sign * step]
                if not all(lo[k] <= n[k] <= hi[k] for k in range(2)):
                    break
                q = n
            while distance({**posture, proximal: q[0], distal: q[1]}, links)[-1] > TOUCH_M \
                    and lo[1] <= q[1] + sign * step <= hi[1]:
                q[1] += sign * step
            if min(distance({**posture, proximal: q[0], distal: q[1]}, links)) > TOUCH_M:
                raise ValueError(f'{finger} does not reach the handle')
            posture.update({proximal: q[0], distal: q[1]})
        lower = {n: float(m.joint(n).range[0]) + .03 for n in RIGHT_FINGERS}
        upper = {n: float(m.joint(n).range[1]) - .03 for n in RIGHT_FINGERS}
        goal = {n: float(np.clip(posture[n] + CLOSING_SIGN[n] * GRIP_REACH_RAD, lower[n], upper[n]))
                for n in RIGHT_FINGERS}
        opened = {n: float(np.clip(posture[n] - CLOSING_SIGN[n] * OPEN_BACKOFF_RAD, lower[n], upper[n]))
                  for n in RIGHT_FINGERS}
        links = [link for f in ('thumb', 'index', 'middle') for link in FINGER_LINKS[f]]
        clearance = min(distance(opened, links))
        contact = {f: [round(1000 * v, 3) for v in distance(posture, FINGER_LINKS[f])] for f in FINGER_LINKS}
        if clearance < OPEN_CLEARANCE_M or contact['palm'][0] < 1.:
            raise ValueError(f'open hand clears the handle by only {clearance * 1000:.2f} mm')
        return {'contact': {n: round(posture[n], 6) for n in RIGHT_FINGERS},
                'goal': {n: round(goal[n], 6) for n in RIGHT_FINGERS},
                'open': {n: round(opened[n], 6) for n in RIGHT_FINGERS},
                'open_clearance_m': round(float(clearance), 6), 'contact_link_distance_mm': contact}

    def _initial_fingers(self):
        # Same initial condition with or without ``grip``: the fingers-open negative
        # control differs only in what its command does (it never closes the hand).
        return {n: self.grasp['contact'][n] - CLOSING_SIGN[n] * INITIAL_BACKOFF_RAD for n in RIGHT_FINGERS}

    def _initial_targets(self):
        targets = self._initial_positions()
        for name in RIGHT_FINGERS:
            index = self.joint_names.index(name)
            targets[index] = self.grasp['contact'][name] + CLOSING_SIGN[name] * INITIAL_SQUEEZE_RAD
        return targets

    def _initial_positions(self):
        values = dict(zip(IK_JOINTS, map(float, self.entry)))
        values.update(HELD_POSE)
        values.update(self._initial_fingers())
        return np.array([values.get(name, 0.) for name in self.joint_names])

    def _initial_tool_pose(self):
        position, hand_r = self._socket(self.entry)
        w, x, y, z = quat_wxyz(hand_r @ self.tool_in_hand_rotation)
        return np.r_[position, x, y, z, w]

    # ------------------------------------------------------------------ control primitives
    def arm_target(self, phi, nut_z):
        """Table pose for hand rotation phi; follows the measured nut height (no pose writes)."""
        phi = float(np.clip(phi, self.phi_grid[0], self.phi_grid[-1]))
        top = np.array([np.interp(phi, self.phi_grid, self.table_top[:, j]) for j in range(len(IK_JOINTS))])
        if not self.follow_nut:
            return top
        bottom = np.array([np.interp(phi, self.phi_grid, self.table_bottom[:, j]) for j in range(len(IK_JOINTS))])
        fraction = float(np.clip((self.initial_nut_z - nut_z) / DESCENT_TABLE_M, 0., 1.))
        return top * (1 - fraction) + bottom * fraction

    def desired_targets(self, phi_ref, nut_z, hand, measured):
        """Arm table pose + finger posture; a closing finger is clamped past its MEASURED angle."""
        values = dict(zip(self.joint_names, self.targets))
        arm = dict(zip(IK_JOINTS, map(float, self.arm_target(phi_ref, nut_z))))
        arm.pop(ROLL_JOINT)
        values.update(arm)
        current = dict(zip(self.joint_names, measured))
        if hand == 'closed' and self.grip:
            values.update({n: grip_target(self.grasp['goal'][n], current[n], CLOSING_SIGN[n]) for n in RIGHT_FINGERS})
        else:
            values.update(self.grasp['open'])
        return np.array([values[name] for name in self.joint_names])

    def wrist_torque(self, roll_target, roll, roll_speed):
        torque = self.roll_kp * (roll_target - roll) - self.roll_kd * roll_speed
        return float(np.clip(torque, -self.motor_limit_nm, self.motor_limit_nm))

    def roll_state(self):
        q = self.state.joint_q.numpy()
        qd = self.state.joint_qd.numpy()
        return float(q[self._roll_q_index]), float(qd[self._roll_qd_index])

    def phi_state(self):
        """Measured hand rotation about the tool axis (clockwise from above) and its rate."""
        pose = self.state.body_q.numpy()[self.hand_body]
        x = rot_xyzw(pose[3:])[:, 0]
        angle = -math.atan2(x[1], x[0])
        if self._phi_last is None:
            self._phi_unwrapped = angle + 2 * math.pi * round((self.phi_grid.mean() - angle) / (2 * math.pi))
        else:
            self._phi_unwrapped += math.atan2(math.sin(angle - self._phi_last), math.cos(angle - self._phi_last))
        self._phi_last = angle
        rate = -float(self.state.body_qd.numpy()[self.hand_body, 5])
        return float(self._phi_unwrapped), rate

    def measured_joints(self):
        return self.state.joint_q.numpy()[self.q_index].astype(float)

    def apply_controls(self, targets, roll_torque):
        ctrl = self.control.mujoco.ctrl.numpy()
        ctrl[self.ctrl_index] = targets
        ctrl[self._actuators[ROLL_JOINT]] = roll_torque
        self.control.mujoco.ctrl.assign(ctrl)
        self.targets = np.array(targets, dtype=float)
        self._roll_command = float(roll_torque)

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
        poses = self.state.body_q.numpy()
        tool_pose = poses[self.tool_body]
        tool_r = rot_xyzw(tool_pose[3:])
        tighten = -tool_r[:, 2]                 # driver axis pointing to the nut: tightening sense
        grip = {k: 0. for k in FINGER_LINKS}
        grip_n = {k: 0 for k in FINGER_LINKS}
        grip_torque, net = 0., np.zeros(3)
        positions = frames = None
        for index, (pair, address) in enumerate(zip(geoms, addresses, strict=True)):
            if address < 0 or forces[address] <= 1e-6:
                continue
            a, b = int(pair[0]), int(pair[1])
            pair = {a, b}
            if pair == {self.nut_geom, self.bolt_geom}:
                thread += 1; thread_force += float(forces[address])
            elif self.nut_geom in pair and pair.intersection(self.socket_geoms):
                tool += 1; tool_force += float(forces[address])
            elif pair == {self.nut_geom, self.floor_geom}:
                floor += 1
            elif self.handle_geom in pair and (a in self.grip_geoms or b in self.grip_geoms):
                if positions is None:
                    positions = data.contact.pos.numpy()[:count]
                    frames = data.contact.frame.numpy()[:count]
                finger = self.grip_geoms[a if a in self.grip_geoms else b]
                grip[finger] += float(forces[address])
                grip_n[finger] += 1
                # Contact frame row 0 is the normal from geom 0 to geom 1; the solved
                # force (normal, two friction rows of the elliptic cone) acts on geom 1.
                local = forces[address:address + 3]
                force = frames[index].reshape(3, 3).T @ local
                on_handle = force if b == self.handle_geom else -force
                net += on_handle
                arm = positions[index] - tool_pose[:3]
                grip_torque += float(np.cross(arm, on_handle) @ tighten)
        self._thread_contacts, self._tool_contacts, self._floor_contacts = thread, tool, floor
        self._thread_force, self._tool_force = thread_force, tool_force
        # Relative rotation of the driver in the hand about the driver axis (slip/re-grip).
        hand_r = rot_xyzw(poses[self.hand_body][3:])
        reference = hand_r[:, 0]                # hand x is perpendicular to the tool axis
        x_tool = tool_r[:, 0]
        angle = math.atan2(np.cross(reference, x_tool) @ tighten, reference @ x_tool)
        if self._relative_last is None:
            self._relative_unwrapped = angle
        else:
            self._relative_unwrapped += math.atan2(math.sin(angle - self._relative_last),
                                                   math.cos(angle - self._relative_last))
        self._relative_last = angle
        self.grip_state = {'normal_force_n': grip, 'contacts': grip_n, 'torque_nm': grip_torque,
                           'net_force_n': float(np.linalg.norm(net)),
                           'relative_angle_rad': float(self._relative_unwrapped)}

    # ------------------------------------------------------------------ standalone episode loop
    def command(self, direction='tighten', turns=1.):
        if direction != 'tighten' or not math.isfinite(turns) or turns <= 0:
            raise ValueError('this turn program implements finite positive tightening only')
        self.requested_turns = turns
        self._started_angle = self._unwrapped
        self._active = True
        self.program.start(self.phi_state()[0], self.time_s)

    def stop(self):
        """Latch: hold every position target, exact zero wrist-roll torque from the next solve."""
        self._active = False
        self.program.halt(self.time_s)

    def turns_so_far(self):
        if self._started_angle is None:
            return 0.
        return -(self._unwrapped - self._started_angle) / (2 * math.pi)

    def plan(self):
        """One control decision from measured state; returns (targets, wrist-roll torque)."""
        phi, phi_rate = self.phi_state()
        roll, roll_speed = self.roll_state()
        nut_z = float(self.state.body_q.numpy()[self.nut_body, 2])
        measured = self.measured_joints()
        hand = self.program.hand
        before = self.desired_targets(self.program.roll_ref, nut_z, hand, measured)
        fingers_done = bool(np.all(np.abs(before[self._finger_index] - self.targets[self._finger_index]) <= 1e-3))
        phi_ref, hand = self.program.advance(self.dt, phi, phi_rate, fingers_done, self.time_s)
        desired = self.desired_targets(phi_ref, nut_z, hand, measured)
        step = np.full(len(self.joint_names), self.max_joint_speed_rad_s * self.dt)
        step[self._finger_index] = FINGER_RATE_RAD_S * self.dt
        targets = self.targets + np.clip(desired - self.targets, -step, step)
        close = np.abs(desired - self.targets) <= step
        targets[close] = desired[close]
        roll_target = float(self.arm_target(phi_ref, nut_z)[IK_JOINTS.index(ROLL_JOINT)])
        torque = self.wrist_torque(roll_target, roll, roll_speed) if self.drive else 0.
        return targets, torque

    def step(self):
        for _ in range(self.substeps):
            if self._active:
                targets, torque = self.plan()
            else:
                targets, torque = self.targets, 0.   # stopped/idle: exact zero wrist-roll effort
            self.apply_controls(targets, torque)
            self.solve()

    def grip_record(self):
        g = self.grip_state
        return {'grip_normal_force_n': dict(g['normal_force_n']), 'grip_contacts': dict(g['contacts']),
                'grip_torque_nm': g['torque_nm'], 'grip_net_force_n': g['net_force_n'],
                'hand_driver_angle_rad': g['relative_angle_rad'],
                'finger_targets_rad': [float(v) for v in self.targets[self._finger_index]],
                'finger_positions_rad': [float(v) for v in self.measured_joints()[self._finger_index]]}

    def observe(self):
        poses = self.state.body_q.numpy()
        nut = poses[self.nut_body]
        roll, roll_speed = self.roll_state()
        phi, phi_rate = self.phi_state()
        return {'epoch': self.epoch, 'step': self.step_id, 'time_s': self.time_s,
                'fastener_id': 'factory_nut_m20_loose', 'fixture_id': 'fixed_factory_bolt_m20_loose',
                'fastener_position_m': nut[:3].tolist(), 'fastener_quaternion_xyzw': nut[3:].tolist(),
                'fixture_position_m': self.fixture_position.tolist(), 'fixture_quaternion_xyzw': [0., 0., 0., 1.],
                'thread_contacts': self._thread_contacts, 'tool_contacts': self._tool_contacts,
                'socket_floor_contacts': self._floor_contacts,
                'angle_rad': self._unwrapped, 'axial_position_m': float(nut[2]), 'pitch_m': self.pitch_m,
                'thread_contact_force_n': self._thread_force, 'tool_contact_force_n': self._tool_force,
                'wrist_roll_rad': roll, 'wrist_roll_speed_rad_s': roll_speed,
                'hand_phi_rad': phi, 'hand_phi_rate_rad_s': phi_rate,
                'hand_phi_reference_rad': self.program.roll_ref, 'wrist_roll_command_nm': self._roll_command,
                'wrist_roll_torque_nm': self._roll_torque, 'hand': self.program.hand,
                'phase': self.program.phase, 'strokes': self.program.strokes, 'regrips': self.program.regrips,
                **self.grip_record(), 'body_poses_xyzw': poses.tolist(),
                'maximum_nut_angular_speed_rad_s': self._maximum_nut_angular_speed,
                'motor_active': self._active}
