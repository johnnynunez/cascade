"""SO-101 powered socket and free Factory nut with actual SDF thread contacts.

Optional standalone Newton experiment. Only robot joint actuator controls are
changed after initialization. There is no nut drive, helical joint, attachment,
external nut wrench, or runtime body-pose write. Initial engagement is a fixture
setup; this experiment does not claim a pick or autonomous tool acquisition.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import uuid
import xml.etree.ElementTree as ET

import numpy as np


class ThreadingScene:
    pitch_m = 0.0025
    frame_dt = 1 / 60
    substeps = 10
    motor_limit_nm = 0.05
    requested_speed_rad_s = 1.5
    arm_joints = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')
    socket_offset = np.array([0.012, 0.0, -0.110])

    def __init__(self, assets, robot_asset, cache, *, device='cuda:0', drive=True,
                 thread_friction=0.01, socket_clearance_m=0.0003,
                 misaligned=False, substeps=10):
        import mujoco
        import newton
        import trimesh
        import warp as wp
        from scipy.optimize import least_squares

        self.wp, self.newton, self.mujoco = wp, newton, mujoco
        self.least_squares = least_squares
        self.assets, self.robot_asset = Path(assets), Path(robot_asset)
        if type(substeps) is not int or not 1 <= substeps <= 100:
            raise ValueError('substeps must be an integer in [1, 100]')
        self.substeps = substeps
        manifest = json.loads((self.assets / 'manifest.json').read_text())
        for file, record in manifest['files'].items():
            if hashlib.sha256((self.assets / file).read_bytes()).hexdigest() != record['sha256']:
                raise ValueError(f'Factory asset hash mismatch: {file}')
        self.epoch = uuid.uuid4().hex
        self.drive = drive
        self.time_s, self.step_id = 0., 0
        self.requested_turns, self.direction = 1., 'tighten'
        self._active = False
        self._last_motor = 0.
        self._maximum_nut_angular_speed = 0.
        self._thread_contacts = self._tool_contacts = 0
        self._thread_force = self._tool_force = 0.
        self._unwrapped = self._last_angle = 0.
        self._started_angle = None
        self._initial_nut_z = .069
        self._center_xy = np.array([.24, 0.])
        tool_xy = self._center_xy + (np.array([0., .06]) if misaligned else 0.)
        wp.init()
        wp.set_device(device)
        self.ik_model = mujoco.MjModel.from_xml_path(str(self.robot_asset))
        self.ik_data = mujoco.MjData(self.ik_model)
        self.ik_gripper = self.ik_model.body('gripper').id
        self.ik_indices = np.array([int(self.ik_model.joint(n).qposadr[0]) for n in self.arm_joints])
        self.ik_ranges = np.array([self.ik_model.joint(n).range for n in self.arm_joints])
        self.entry = self._ik(np.r_[tool_xy, self._initial_nut_z], [0., 0., 0., 1.6, .17])
        self.bottom = self._ik(np.r_[tool_xy, .045], self.entry)
        self.ik_data.qpos[self.ik_indices] = self.entry
        mujoco.mj_kinematics(self.ik_model, self.ik_data)
        rotation = self.ik_data.xmat[self.ik_gripper].reshape(3, 3)
        nut_yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        self._last_angle = nut_yaw
        self._unwrapped = nut_yaw
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.gap = 0.
        setattr(newton, 'use_coord_layout_targets', True)
        xml = self._robot_xml(socket_clearance_m)
        builder.add_mjcf(xml, ctrl_direct=True, parse_visuals=True, parse_meshes=True,
                         enable_self_collisions=True, collapse_fixed_joints=False)
        self._actuators = {node.attrib['name']:i for i, node in enumerate(ET.fromstring(xml).find('actuator'))}
        cfg = builder.ShapeConfig(margin=0., mu=thread_friction, ke=1e7, kd=1e4,
                                 gap=.005, density=8000., mu_torsional=0., mu_rolling=0.)
        self.fixture_position = np.r_[self._center_xy, 0.]
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
        self._add_fixture(builder, cache)
        builder.add_shape_box(-1, xform=wp.transform(wp.vec3(.15, 0, -.015), wp.quat_identity()),
                              hx=.4, hy=.25, hz=.015, label='bench', color=(.24,.27,.3))
        self.model = builder.finalize(device=device)
        self.bodies = self._names(self.model.body_label)
        self.joints = self._names(self.model.joint_label)
        self._qstart = self.model.joint_q_start.numpy()
        q = self.model.joint_q.numpy()
        for name, value in zip(self.arm_joints, self.entry, strict=True):
            q[self._qstart[self.joints[name]]] = value
        q[self._qstart[self.joints['gripper']]] = .6
        self.model.joint_q.assign(q)  # Initial condition only, before first solve.
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
        shape_map = self.solver.mjc_geom_to_newton_shape.numpy()[0]
        labels = self._names(self.model.shape_label)
        self.nut_geom = int(np.flatnonzero(shape_map == labels['nut'])[0])
        self.bolt_geom = int(np.flatnonzero(shape_map == labels['bolt'])[0])
        self.socket_geoms = {int(np.flatnonzero(shape_map == index)[0]) for name, index in labels.items()
                             if name.startswith('socket_wall_')}
        dof_map = self.solver.mjc_dof_to_newton_dof.numpy()[0]
        dof = self.model.joint_qd_start.numpy()[self.joints['socket_spin']]
        self.motor_dof = int(np.flatnonzero(dof_map == dof)[0])
        self._motor_qd_index = int(dof)

    @staticmethod
    def _names(labels):
        return {label.rsplit('/', 1)[-1]:i for i, label in enumerate(labels)}

    def _ik(self, target, seed):
        def residual(q):
            self.ik_data.qpos[self.ik_indices] = q
            self.mujoco.mj_kinematics(self.ik_model, self.ik_data)
            r = self.ik_data.xmat[self.ik_gripper].reshape(3, 3)
            p = self.ik_data.xpos[self.ik_gripper] + r @ self.socket_offset
            return np.r_[p-target, .15 * (r @ np.array([0., 0., -1.]) - [0., 0., -1.])]
        result = self.least_squares(residual, seed, bounds=(self.ik_ranges[:, 0]+.001,
            self.ik_ranges[:, 1]-.001), max_nfev=250, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if np.linalg.norm(residual(result.x)) > 1e-5:
            raise ValueError('Socket center is not reachable with vertical SO-101 tool axis')
        return result.x

    def _robot_xml(self, clearance):
        root = ET.parse(self.robot_asset).getroot()
        root.find('compiler').set('meshdir', str(self.robot_asset.parent / 'assets'))
        grip = root.find(".//body[@name='gripper']")
        housing = ET.SubElement(grip, 'body', name='socket_housing', pos='.012 0 -.060')
        ET.SubElement(housing, 'inertial', pos='0 0 0', mass='.05', diaginertia='.00002 .00002 .00001')
        ET.SubElement(housing, 'geom', name='socket_case', type='cylinder', size='.013 .015',
                      rgba='.1 .5 .65 1', contype='1', conaffinity='1', group='2')
        spindle = ET.SubElement(housing, 'body', name='socket_spindle', pos='0 0 -.016')
        ET.SubElement(spindle, 'inertial', pos='0 0 -.034', mass='.05',
                      diaginertia='.00002 .00002 .00002')
        ET.SubElement(spindle, 'joint', name='socket_spin', type='hinge', axis='0 0 -1',
                      limited='false', damping='.001', frictionloss='0', armature='.00001')
        for i in range(6):
            a = i * math.pi / 3
            radius = .015 + clearance + .003
            ET.SubElement(spindle, 'geom', name=f'socket_wall_{i}', type='box',
                          pos=f'{radius*math.cos(a)} {radius*math.sin(a)} -.034',
                          quat=f'{math.cos(a/2)} 0 0 {math.sin(a/2)}', size='.003 .010 .010',
                          rgba='.6 .65 .72 1', contype='1', conaffinity='1',
                          friction='.3 0 0', solref='.003 1', mass='0', group='2')
        ET.SubElement(root.find('actuator'), 'motor', name='socket_motor', joint='socket_spin',
                      gear='1', ctrllimited='true', ctrlrange=f'-{self.motor_limit_nm} {self.motor_limit_nm}',
                      forcelimited='true', forcerange=f'-{self.motor_limit_nm} {self.motor_limit_nm}')
        contact = root.find('contact')
        if contact is None:
            contact = ET.SubElement(root, 'contact')
        for body in ('socket_housing', 'socket_spindle'):
            for mount in ('gripper', 'moving_jaw_so101_v1', 'camera_mount'):
                ET.SubElement(contact, 'exclude', body1=body, body2=mount)
        return ET.tostring(root, encoding='unicode')

    def command(self, direction='tighten', turns=1.):
        if direction not in ('tighten', 'loosen') or not math.isfinite(turns) or turns <= 0:
            raise ValueError('finite positive turn request and tighten/loosen direction required')
        self.direction, self.requested_turns = direction, turns
        self._started_angle = self._unwrapped
        self._active = True

    def _read_contacts(self):
        data = self.solver.mjw_data
        count = int(data.nacon.numpy()[0])
        geoms = data.contact.geom.numpy()[:count]
        addresses = data.contact.efc_address.numpy()[:count, 0]
        forces = data.efc.force.numpy()[0]
        thread, tool, thread_force, tool_force = 0, 0, 0., 0.
        for pair, address in zip(geoms, addresses, strict=True):
            if address < 0 or forces[address] <= 1e-6:
                continue
            pair = set(map(int, pair))
            if pair == {self.nut_geom, self.bolt_geom}:
                thread += 1; thread_force += float(forces[address])
            elif self.nut_geom in pair and pair.intersection(self.socket_geoms):
                tool += 1; tool_force += float(forces[address])
        self._thread_contacts, self._tool_contacts = thread, tool
        self._thread_force, self._tool_force = thread_force, tool_force

    def step(self):
        for _ in range(self.substeps):
            pose = self.state.body_q.numpy()[self.nut_body]
            target = self._arm_target(pose)
            ctrl = self.control.mujoco.ctrl.numpy()
            for name, value in zip(self.arm_joints, target, strict=True):
                ctrl[self._actuators[name]] = value
            ctrl[self._actuators['gripper']] = .6
            speed = float(self.state.joint_qd.numpy()[self._motor_qd_index])
            sign = 1 if self.direction == 'tighten' else -1
            requested_speed = sign * self.requested_speed_rad_s if self.drive and self._active else 0.
            # Disabled drive is exactly zero actuator effort. The authored
            # spindle's passive viscous damping remains physical in both cases.
            # A completed enabled drive commands a velocity brake explicitly.
            ctrl[self._actuators['socket_motor']] = (float(np.clip(.02*(requested_speed-speed),
                -self.motor_limit_nm, self.motor_limit_nm)) if self.drive else 0.)
            self.control.mujoco.ctrl.assign(ctrl)
            self.state.clear_forces()
            self.pipeline.collide(self.state, self.contacts)
            self.solver.step(self.state, self.next, self.control, self.contacts, self.frame_dt/self.substeps)
            self.state, self.next = self.next, self.state
            self.time_s += self.frame_dt/self.substeps
            self.step_id += 1
            if np.any(self.solver.mjw_data.overflow.numpy()):
                raise RuntimeError('contact/constraint buffer overflow')
            for field in ('body_q', 'body_qd', 'joint_q', 'joint_qd'):
                if not np.isfinite(getattr(self.state, field).numpy()).all():
                    raise RuntimeError('non-finite solved physics state')
            self._read_contacts()
            angular_speed = np.linalg.norm(self.state.body_qd.numpy()[self.nut_body, 3:])
            self._maximum_nut_angular_speed = max(self._maximum_nut_angular_speed, float(angular_speed))
            self._last_motor = float(self.solver.mjw_data.qfrc_actuator.numpy()[0, self.motor_dof])
            p = self.state.body_q.numpy()[self.nut_body]
            x, y, z, w = p[3:]
            angle = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
            self._unwrapped += math.atan2(math.sin(angle-self._last_angle), math.cos(angle-self._last_angle))
            self._last_angle = angle
            if self._active and -sign*(self._unwrapped-self._started_angle) >= self.requested_turns*2*math.pi:
                self._active = False
            if self._on_substep(p):
                break

    def _add_fixture(self, builder, cache):
        pass

    def _on_substep(self, nut_pose):
        return False

    def _arm_target(self, nut_pose):
        fraction = float(np.clip((self._initial_nut_z-nut_pose[2])/.024, 0., 1.))
        return self.entry*(1-fraction) + self.bottom*fraction

    def observe(self):
        poses = self.state.body_q.numpy()
        nut = poses[self.nut_body]
        return {'epoch':self.epoch, 'step':self.step_id, 'time_s':self.time_s,
                'fastener_id':'factory_nut_m20_loose', 'fixture_id':'fixed_factory_bolt_m20_loose',
                'fastener_position_m':nut[:3].tolist(), 'fastener_quaternion_xyzw':nut[3:].tolist(),
                'fixture_position_m':self.fixture_position.tolist(), 'fixture_quaternion_xyzw':[0.,0.,0.,1.],
                'thread_contacts':self._thread_contacts, 'tool_contacts':self._tool_contacts,
                'angle_rad':self._unwrapped, 'axial_position_m':float(nut[2]), 'pitch_m':self.pitch_m,
                'thread_contact_force_n':self._thread_force, 'tool_contact_force_n':self._tool_force,
                'motor_torque_nm':self._last_motor, 'body_poses_xyzw':poses.tolist(),
                'maximum_nut_angular_speed_rad_s':self._maximum_nut_angular_speed,
                'motor_active':self._active}
