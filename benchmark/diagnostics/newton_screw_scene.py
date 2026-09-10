"""SO-101 + powered screwdriver in Newton 1.6, CPU only.

Scripted control, not an LLM episode or a CASCADE backend. The original SO-101
MJCF and meshes are retained. A mounted spindle transfers equal/opposite WORLD
body torques only when its measured tip is aligned with the screw. The thread
is an ideal helical constraint; real head/fixture contact and Coulomb joint friction
represent seating and self-locking. Parameters illustrate the mechanism, not a
manufacturer's fastening specification. No state is prescribed after setup.
"""
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import mujoco
import newton
from newton.solvers import SolverMuJoCo
import numpy as np
from scipy.optimize import least_squares
import warp as wp

REPO = Path(__file__).resolve().parents[2]
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
TIP_LOCAL = np.array([0.012, 0.0, -0.145])
DOWN = np.array([0.0, 0.0, -1.0])


def transform(xyz):
    return wp.transform(wp.vec3(*xyz), wp.quat_identity())


def names(labels):
    result = {}
    for index, label in enumerate(labels):
        name = label.rsplit("/", 1)[-1]
        if name in result:
            raise RuntimeError(f"Ambiguous model name: {name}")
        result[name] = index
    return result


def smooth(t):
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)


def mapped_index(mapping, target):
    matches = np.flatnonzero(mapping == target)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one solver mapping for {target}, got {matches}")
    return int(matches[0])


class ScrewDemo:
    frame_dt = 1.0 / 30.0
    substeps = 8
    pitch_m = 0.002
    seat_m = 0.004
    motor_limit = 0.020
    reflected_rotor_inertia = 0.00001

    def __init__(self, *, enable_drive=True, disengaged=False, fixture_collisions=True):
        self.enable_drive = bool(enable_drive)
        self._fixture_collisions = bool(fixture_collisions)
        self.sim_time = 0.0
        self.phase = "aproximar"
        self.completed = False
        self._stalled_s = 0.0
        self._drive_off_at = None
        self._rest_s = 0.0
        self._seat_force = 0.0
        self._peak_seat_force = 0.0
        self._tip_force = 0.0
        self._seat_contact_steps = 0
        self._previous_engaged = False
        self._coupling_reference = 0.0
        self._engaged_steps = 0
        self._max_pitch_error = 0.0
        self._peak_motor = 0.0
        self._peak_coupling = 0.0
        self._applied_torque = 0.0
        self._tool_travel = 0.0
        self._retained_min = math.inf
        self._finite_steps = 0
        self._radial_error = math.inf
        self._axial_error = math.inf
        self._axis_error = math.inf
        self._dt = self.frame_dt / self.substeps
        self.asset = REPO / "assets/mjcf/so101/so101.xml"
        if not self.asset.is_file():
            raise RuntimeError("SO-101 assets missing: run scripts/fetch_robot_assets.py so101")
        # This second model is FK/IK ONLY. Never stepped, never rendered as physics.
        self._ik_model = mujoco.MjModel.from_xml_path(str(self.asset))
        self._ik_data = mujoco.MjData(self._ik_model)
        self._ik_gripper = self._ik_model.body("gripper").id
        self._ik_qids = np.array([int(self._ik_model.joint(n).qposadr[0]) for n in ARM_JOINTS])
        self._ik_ranges = np.array([self._ik_model.joint(n).range for n in ARM_JOINTS])
        self._nominal_tip = np.array([0.24, 0.0, 0.035])
        self._entry = self._inverse_kinematics(self._nominal_tip, [0.0, 0.0, 0.0, 1.6, 0.17])
        self._bottom = self._inverse_kinematics(self._nominal_tip + [0, 0, -0.005], self._entry)
        self._home = self._entry + np.array([-0.30, -0.10, -0.12, -0.35, 0.0])
        bolt_origin = self._nominal_tip + ([0.025, 0, 0] if disengaged else np.zeros(3))
        self.camera_target = (0.19, 0.0, 0.10)
        self.screw_target = tuple(bolt_origin)

        setattr(newton, "use_coord_layout_targets", True)
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.gap = 0.0
        arm_xml = self._arm_xml()
        builder.add_mjcf(arm_xml, ctrl_direct=True, parse_visuals=True,
                         parse_meshes=True, enable_self_collisions=True,
                         collapse_fixed_joints=False)
        self._add_fixture(builder, bolt_origin)
        model = builder.finalize(device="cpu")
        self.model = model
        self._joints = names(model.joint_label)
        self._bodies = names(model.body_label)
        self.screw_body = self._bodies["screw"]
        self.spindle_body = self._bodies["driver_spindle"]
        self._gripper_body = self._bodies["gripper"]
        self._qstart = model.joint_q_start.numpy()
        self._dstart = model.joint_qd_start.numpy()
        q = model.joint_q.numpy()
        for name, value in zip(ARM_JOINTS, self._home, strict=True):
            q[self._qstart[self._joints[name]]] = value
        q[self._qstart[self._joints["gripper"]]] = 0.6
        model.joint_q.assign(q)  # INITIAL CONDITION ONLY
        self.state, self._next = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.state)
        newton.eval_fk(model, model.joint_q, model.joint_qd, self._next)
        self.control = model.control()
        self.solver = SolverMuJoCo(model, use_mujoco_cpu=False, use_mujoco_contacts=False,
                                  integrator="implicitfast", solver="newton",
                                  iterations=80, ls_iterations=30, njmax=4096,
                                  nconmax=1024, update_data_interval=1)
        # Newton's mimic entities do not accept the custom MJCF-equality
        # attribute frequency. Configure the sole exported equality in both
        # solver representations before stepping instead.
        if self.solver.mj_model.neq != 1:
            raise RuntimeError("Expected exactly our one helical equality")
        self.solver.mj_model.eq_solref[0] = (0.005, 1.0)
        eq_solref = self.solver.mjw_model.eq_solref.numpy()
        eq_solref[..., 0], eq_solref[..., 1] = 0.005, 1.0
        self.solver.mjw_model.eq_solref.assign(eq_solref)
        # The thread is a kinematic helix, not a torsional rubber spring.
        # Bound soft-constraint wind-up when the head makes seating contact.
        thread_impedance = (0.9999, 0.9999, 0.001, 0.5, 2.0)
        self.solver.mj_model.eq_solimp[0] = thread_impedance
        eq_solimp = self.solver.mjw_model.eq_solimp.numpy()
        eq_solimp[0, 0] = thread_impedance
        self.solver.mjw_model.eq_solimp.assign(eq_solimp)
        self.pipeline = newton.CollisionPipeline(model, rigid_contact_max=1024,
                                                 deterministic=True, verify_buffers=True)
        self.contacts = self.pipeline.contacts()
        # The fast 1.6 converter intentionally omits MuJoCo string names.
        # Address Newton by name, then use its explicit solver mappings.
        actuators = ET.fromstring(arm_xml).find("actuator")
        assert actuators is not None
        self._actuators = {node.attrib["name"]: index for index, node in enumerate(actuators)}
        body_mapping = self.solver.mjc_body_to_newton.numpy()[0]
        dof_mapping = self.solver.mjc_dof_to_newton_dof.numpy()[0]
        self._motor_dof = mapped_index(dof_mapping, self._dstart[self._joints["driver_spin"]])
        self._mj_screw = mapped_index(body_mapping, self.screw_body)
        self._mj_spindle = mapped_index(body_mapping, self.spindle_body)
        shape_labels = [name.rsplit("/", 1)[-1] for name in model.shape_label]
        geom_mapping = self.solver.mjc_geom_to_newton_shape.numpy()[0]
        self._head_geom = mapped_index(geom_mapping, shape_labels.index("screw_head"))
        self._bit_geom = mapped_index(geom_mapping, shape_labels.index("driver_bit"))
        self._fixture_geoms = ({mapped_index(geom_mapping, i) for i, name in enumerate(shape_labels)
                                if name.startswith("fixture_")} if self._fixture_collisions else set())
        screw_dof = mapped_index(dof_mapping, self._dstart[self._joints["screw_spin"]])
        # Default soft friction permits steady creep under sub-breakaway loads.
        # Use stiff impedance for the intended self-locking thread friction.
        friction_impedance = (0.9999, 0.9999, 0.001, 0.5, 2.0)
        self.solver.mj_model.dof_solimp[screw_dof] = friction_impedance
        dof_solimp = self.solver.mjw_model.dof_solimp.numpy()
        dof_solimp[0, screw_dof] = friction_impedance
        self.solver.mjw_model.dof_solimp.assign(dof_solimp)
        if int(self.solver.mj_model.opt.cone) != int(mujoco.mjtCone.mjCONE_ELLIPTIC):
            raise RuntimeError("Contact-force extraction requires the selected elliptic cone")
        self._initial_tip = self._tip_geometry()[0]
        self._last_base_angle = self._base_angle()
        self._unwrapped_base_angle = self._last_base_angle
        self._last_motor = 0.0
        self._seating_torque = 0.0

    def _inverse_kinematics(self, target, seed):
        def residual(q):
            self._ik_data.qpos[self._ik_qids] = q
            self._ik_data.qpos[int(self._ik_model.joint("gripper").qposadr[0])] = 0.1
            mujoco.mj_kinematics(self._ik_model, self._ik_data)
            rotation = self._ik_data.xmat[self._ik_gripper].reshape(3, 3)
            tip = self._ik_data.xpos[self._ik_gripper] + rotation @ TIP_LOCAL
            return np.r_[tip - target, 0.15 * (rotation @ DOWN - DOWN)]
        result = least_squares(residual, seed, bounds=(self._ik_ranges[:, 0] + 0.001,
                                                     self._ik_ranges[:, 1] - 0.001),
                               max_nfev=250, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if np.linalg.norm(residual(result.x)) > 1e-5:
            raise RuntimeError(f"Tool target unreachable within SO-101 limits: {target}")
        return result.x

    def _arm_xml(self):
        root = ET.parse(self.asset).getroot()
        compiler = root.find("compiler")
        assert compiler is not None
        compiler.set("meshdir", str((self.asset.parent / compiler.get("meshdir", "")).resolve()))
        grip = root.find(".//body[@name='gripper']")
        assert grip is not None
        housing = ET.SubElement(grip, "body", name="driver_housing", pos="0.012 0 -0.095")
        ET.SubElement(housing, "inertial", pos="0 0 0", mass="0.065",
                      diaginertia="0.00002 0.00002 0.00001")
        ET.SubElement(housing, "geom", name="driver_case", type="cylinder", size="0.013 0.022",
                      rgba="0.05 0.50 0.65 1", contype="1", conaffinity="1", group="2")
        spindle = ET.SubElement(housing, "body", name="driver_spindle", pos="0 0 -0.016")
        ET.SubElement(spindle, "inertial", pos="0 0 -0.012", mass="0.015",
                      diaginertia="0.000005 0.000005 0.0000002")
        ET.SubElement(spindle, "joint", name="driver_spin", type="hinge", axis="0 0 -1",
                      limited="false", damping="0.0001", frictionloss="0", armature="0.00001")
        ET.SubElement(spindle, "geom", name="driver_bit", type="cylinder", size="0.0025 0.017",
                      pos="0 0 -0.017", rgba="0.7 0.75 0.8 1", contype="1", conaffinity="1", group="2")
        ET.SubElement(spindle, "geom", name="driver_rotation_mark", type="box", size="0.003 0.001 0.002",
                      pos="0.005 0 -0.004", rgba="1 0.55 0.02 1", contype="0", conaffinity="0", group="2")
        actuators = root.find("actuator")
        assert actuators is not None
        ET.SubElement(actuators, "motor", name="driver_motor", joint="driver_spin",
                      gear="1", ctrllimited="true", ctrlrange=f"{-self.motor_limit} {self.motor_limit}",
                      forcelimited="true", forcerange=f"{-self.motor_limit} {self.motor_limit}")
        contact = root.find("contact")
        if contact is None:
            contact = ET.SubElement(root, "contact")
        # This tool is already rigidly mounted, not dynamically grasped. Only
        # exclude its mounting assembly, never the bench, fixture or screw.
        for tool_body in ("driver_housing", "driver_spindle"):
            for mount_body in ("gripper", "moving_jaw_so101_v1", "camera_mount"):
                ET.SubElement(contact, "exclude", body1=tool_body, body2=mount_body)
        return ET.tostring(root, encoding="unicode")

    def _add_fixture(self, builder, origin):
        visual = builder.ShapeConfig(density=0.0, has_shape_collision=False,
                                     has_particle_collision=False, gap=0.0)
        solid = builder.ShapeConfig(density=0.0, gap=0.0, mu=0.4, ke=100000.0, kd=1000.0)
        fixture_cfg = solid if self._fixture_collisions else visual
        builder.add_shape_box(-1, xform=transform([0.15, 0, -0.018]), hx=0.38, hy=0.27, hz=0.018,
                              cfg=solid, color=(0.16, 0.19, 0.23), label="bench")
        # A square bore keeps the visually simplified shaft visible without
        # pretending the bore's mesh contacts implement the thread.
        for sign in (-1, 1):
            for axis in (0, 1):
                center = np.asarray(origin).copy()
                center[axis] += sign * 0.016
                center[2] = 0.0125
                half = [0.011, 0.027, 0.0125] if axis == 0 else [0.005, 0.011, 0.0125]
                builder.add_shape_box(-1, xform=transform(center), hx=half[0], hy=half[1], hz=half[2],
                                      cfg=fixture_cfg, color=(0.42, 0.47, 0.53), label=f"fixture_{axis}_{sign}")
        carriage = builder.add_link(mass=0.001, inertia=wp.mat33(np.eye(3) * 1e-7), label="thread_carriage")
        lead = self.pitch_m / (2 * math.pi)
        # Keep physical axial mass: shifting rotor inertia onto this DOF
        # changes soft-contact weights and can inflate seating loads badly.
        slide = builder.add_joint_prismatic(-1, carriage, parent_xform=transform(origin),
                                            axis=wp.vec3(*DOWN), label="screw_slide",
                                            limit_lower=0.0, limit_upper=0.020,
                                            limit_ke=100000.0, limit_kd=1000.0,
                                            armature=0.0,
                                            target_ke=0.0, target_kd=0.0)
        screw = builder.add_link(mass=0.012, com=wp.vec3(0, 0, -0.010),
                                 inertia=wp.mat33(np.diag([5e-7, 5e-7, 1e-7])), label="screw")
        spin = builder.add_joint_revolute(carriage, screw, axis=wp.vec3(*DOWN), label="screw_spin",
                                          armature=self.reflected_rotor_inertia, friction=0.008, damping=0.0001,
                                          target_ke=0.0, target_kd=0.0)
        builder.add_articulation([slide, spin], label="threaded_fastener")
        # Express the equality in angular units. MuJoCo Warp's joint-equality
        # impedance sums unweighted DOF inverse masses, so x=lead*theta makes
        # a metre/radian screw constraint excessively soft. This is the SAME
        # helix, theta=x/lead, with the well-conditioned follower coordinate.
        builder.add_constraint_mimic(spin, slide, coef1=1.0 / lead,
                                     label="ideal_helical_thread")
        builder.add_shape_cylinder(screw, xform=transform([0, 0, -0.003]), radius=0.009, half_height=0.003,
                                   cfg=solid, color=(0.70, 0.75, 0.82), label="screw_head")
        builder.add_shape_cylinder(screw, xform=transform([0, 0, -0.015]), radius=0.0035, half_height=0.010,
                                   cfg=solid, color=(0.55, 0.59, 0.63), label="screw_shaft")
        builder.add_shape_box(screw, xform=transform([0, 0, 0.0002]), hx=0.007, hy=0.001, hz=0.0003,
                              cfg=visual, color=(0.06, 0.07, 0.09), label="screw_slot")
        theta = np.linspace(0, 8 * 2 * math.pi, 257)
        cross = np.linspace(0, 2 * math.pi, 8, endpoint=False)
        vertices = np.array([[(0.0037 + 0.00035 * np.cos(c)) * np.cos(t),
                              (0.0037 + 0.00035 * np.cos(c)) * np.sin(t),
                              -0.007 - self.pitch_m * t / (2 * math.pi) + 0.00035 * np.sin(c)]
                             for t in theta for c in cross], dtype=np.float32)
        faces = []
        for i in range(len(theta) - 1):
            for j in range(len(cross)):
                a, b = i * len(cross) + j, i * len(cross) + (j + 1) % len(cross)
                faces.extend([a, a + len(cross), b, b, a + len(cross), b + len(cross)])
        builder.add_shape_mesh(screw, mesh=newton.Mesh(vertices, np.asarray(faces, dtype=np.int32)),
                               cfg=visual, color=(0.65, 0.70, 0.77), label="visual_thread_only")

    def _coordinate(self, name, velocity=False):
        index = self._joints[name]
        starts = self._dstart if velocity else self._qstart
        values = self.state.joint_qd if velocity else self.state.joint_q
        return float(values.numpy()[starts[index]])

    def _tip_geometry(self):
        pose = self.state.body_q.numpy()[self._gripper_body]
        tip = np.asarray(wp.transform_point(wp.transform(*pose), wp.vec3(*TIP_LOCAL)), dtype=float)
        axis = np.asarray(wp.quat_rotate(wp.quat(*pose[3:]), wp.vec3(*DOWN)), dtype=float)
        return tip, axis

    def _base_angle(self):
        pose = self.state.body_q.numpy()[self._gripper_body]
        x = np.asarray(wp.quat_rotate(wp.quat(*pose[3:]), wp.vec3(1, 0, 0)))
        return -float(math.atan2(x[1], x[0]))

    def _read_contact_loads(self):
        """Read solved normal forces, not the narrow-phase candidate count."""
        data = self.solver.mjw_data
        count = int(data.nacon.numpy()[0])
        geoms = data.contact.geom.numpy()[:count]
        addresses = data.contact.efc_address.numpy()[:count, 0]
        forces = data.efc.force.numpy()[0]
        seat, tip = 0.0, 0.0
        for pair, address in zip(geoms, addresses, strict=True):
            if address < 0:
                continue
            normal = max(0.0, float(forces[address]))
            pair = set(map(int, pair))
            if self._head_geom in pair and pair.intersection(self._fixture_geoms):
                seat += normal
            if pair == {self._head_geom, self._bit_geom}:
                tip += normal
        self._seat_force, self._tip_force = seat, tip
        self._peak_seat_force = max(self._peak_seat_force, seat)
        if seat > 0.01:
            self._seat_contact_steps += 1

    def step(self):
        for _ in range(self.substeps):
            axial = self._coordinate("screw_slide")
            spin = self._coordinate("screw_spin")
            speed = self._coordinate("driver_spin", velocity=True)
            base_angle = self._base_angle()
            self._unwrapped_base_angle += math.atan2(math.sin(base_angle - self._last_base_angle),
                                                     math.cos(base_angle - self._last_base_angle))
            self._last_base_angle = base_angle
            driver = self._coordinate("driver_spin") + self._unwrapped_base_angle
            fraction = float(np.clip(axial / 0.005, 0, 1))
            target = self._entry * (1 - fraction) + self._bottom * fraction
            approach = smooth(self.sim_time / 3.0)
            target = self._home * (1 - approach) + target * approach
            command = self.control.mujoco.ctrl.numpy()
            for name, value in zip(ARM_JOINTS, target, strict=True):
                command[self._actuators[name]] = value
            command[self._actuators["gripper"]] = 0.6
            driving = self.enable_drive and self.sim_time >= 3.2 and self._drive_off_at is None
            command[self._actuators["driver_motor"]] = (float(np.clip(0.004 * (6.0 - speed),
                                                                      -self.motor_limit, self.motor_limit))
                                                        if driving else 0.0)
            self.control.mujoco.ctrl.assign(command)
            tip, axis = self._tip_geometry()
            bolt = self.state.body_q.numpy()[self.screw_body, :3]
            self._radial_error = float(np.linalg.norm((tip - bolt)[:2]))
            self._axial_error = float(abs(tip[2] - bolt[2]))
            self._axis_error = float(np.linalg.norm(axis - DOWN))
            # Require solved contact to enter. Keep the ideal bit coupling
            # during zero-load contact frames only while the measured gap is
            # below 0.1 mm; disengage on geometric separation, not force noise.
            contact_witness = self._tip_force > 0.001 or (self._previous_engaged and self._axial_error < 0.0001)
            engaged = (self._radial_error < 0.0015 and self._axial_error < 0.0006
                       and self._axis_error < 0.05 and contact_witness)
            self._tool_travel = max(self._tool_travel, float(np.linalg.norm(tip - self._initial_tip)))
            torque = 0.0
            if engaged:
                if not self._previous_engaged:
                    self._coupling_reference = driver - spin
                twist = driver - spin - self._coupling_reference
                relative_speed = speed - self._coordinate("screw_spin", velocity=True)
                torque = float(np.clip(0.025 * twist + 0.0006 * relative_speed, -0.035, 0.035))
                self._engaged_steps += 1
            self._previous_engaged = engaged
            self.state.clear_forces()
            forces = self.state.body_f.numpy()
            # External WORLD wrenches, not two opposing joint efforts. The
            # latter would cancel the motor reaction that the arm must carry.
            forces[self.screw_body, 3:] = torque * DOWN
            forces[self.spindle_body, 3:] = -torque * DOWN
            self.state.body_f.assign(forces)
            self.pipeline.collide(self.state, self.contacts)
            if int(self.contacts.rigid_contact_count.numpy()[0]) > 1024:
                raise RuntimeError("Newton contact buffer overflow")
            self.solver.step(self.state, self._next, self.control, self.contacts, self._dt)
            self.state, self._next = self._next, self.state
            self.sim_time += self._dt
            data = self.solver.mjw_data
            if np.any(data.overflow.numpy()):
                raise RuntimeError("MuJoCo Warp constraint/contact buffer overflow")
            for field in ("body_q", "body_qd", "joint_q", "joint_qd"):
                if not np.isfinite(getattr(self.state, field).numpy()).all():
                    raise RuntimeError(f"Non-finite Newton state: {field}")
            self._finite_steps += 1
            self._read_contact_loads()
            self._last_motor = float(data.qfrc_actuator.numpy()[0, self._motor_dof])
            self._peak_motor = max(self._peak_motor, abs(self._last_motor))
            applied = data.xfrc_applied.numpy()[0]
            self._applied_torque = -float(applied[self._mj_screw, 5])
            if not np.allclose(applied[self._mj_screw] + applied[self._mj_spindle], 0.0, atol=1e-7):
                raise RuntimeError("Tool/screw wrench reaction is not balanced")
            self._peak_coupling = max(self._peak_coupling, abs(self._applied_torque))
            axial = self._coordinate("screw_slide")
            turns = self._coordinate("screw_spin") / (2 * math.pi)
            self._max_pitch_error = max(self._max_pitch_error, abs(axial - self.pitch_m * turns))
            seated = (axial >= self.seat_m - 0.00015 and self._seat_force > 0.01
                      and abs(self._coordinate("screw_spin", velocity=True)) < 0.15)
            loaded = engaged and abs(self._last_motor) >= 0.75 * self.motor_limit
            self._stalled_s = self._stalled_s + self._dt if seated and loaded else 0.0
            if driving and self._stalled_s >= 0.35:
                self._drive_off_at = self.sim_time
                self._seating_torque = abs(self._applied_torque)
            if self._drive_off_at is not None:
                self._retained_min = min(self._retained_min, axial)
                self.phase = "comprobar_retencion_sin_motor"
                at_rest = (abs(self._coordinate("screw_spin", velocity=True)) < 0.02
                           and self._seat_force > 0.01)
                self._rest_s = self._rest_s + self._dt if at_rest else 0.0
                if self.sim_time - self._drive_off_at >= 1.0 and self._rest_s >= 0.5:
                    self.completed = True
                    self.phase = "terminado"
            elif self.sim_time >= 3.2:
                self.phase = "apretar" if engaged else "sin_engranar"
            else:
                self.phase = "aproximar"

    def metrics(self):
        turns = self._coordinate("screw_spin") / (2 * math.pi)
        axial = self._coordinate("screw_slide")
        held = 0.0 if self._drive_off_at is None else self.sim_time - self._drive_off_at
        verified = bool(self.completed and turns > 1.5 and 0.0038 < axial < 0.005
                        and self._max_pitch_error < 0.00015 and self._peak_motor > 0.005
                        and self._peak_motor <= self.motor_limit + 1e-6
                        and self._retained_min > 0.0038 and abs(self._last_motor) < 1e-7
                        and self._rest_s >= 0.5
                        and self._seat_contact_steps > 0 and self._seat_force > 0.01
                        and self._peak_seat_force < 100.0
                        and self._tool_travel > 0.02 and self._engaged_steps > 0)
        return {"phase": self.phase, "time_s": self.sim_time,
                "screw_turns": turns, "axial_mm": axial * 1000,
                "applied_torque_nm": self._applied_torque,
                "motor_torque_nm": self._last_motor,
                "peak_motor_torque_nm": self._peak_motor,
                "peak_coupling_torque_nm": self._peak_coupling,
                "seating_torque_nm": self._seating_torque,
                "held_after_drive_off_s": held,
                "physical_rest_s": self._rest_s,
                "seating_contact_force_n": self._seat_force,
                "peak_seating_contact_force_n": self._peak_seat_force,
                "tip_contact_force_n": self._tip_force,
                "seating_contact_steps": self._seat_contact_steps,
                "retained_min_mm": self._retained_min * 1000 if math.isfinite(self._retained_min) else None,
                "screw_speed_rad_s": self._coordinate("screw_spin", velocity=True),
                "max_pitch_error_mm": self._max_pitch_error * 1000,
                "tool_travel_m": self._tool_travel, "engaged_steps": self._engaged_steps,
                "radial_alignment_mm": self._radial_error * 1000 if math.isfinite(self._radial_error) else None,
                "axial_alignment_mm": self._axial_error * 1000 if math.isfinite(self._axial_error) else None,
                "finite_steps": self._finite_steps, "completed": self.completed, "verified": verified,
                "newton_version": newton.__version__, "device": str(self.model.device),
                "solver": "Newton SolverMuJoCo / MuJoCo Warp CPU",
                "collisions": "Newton CollisionPipeline; thread geometry is visual only",
                "mechanism": "2 mm/rev mimic constraint; real head/fixture seating contact and Coulomb joint friction; tip-contact/alignment-gated equal/opposite external spindle/screw torques",
                "controller": "scripted SO-101 position control and torque-limited spindle speed feedback; no LLM",
                "scope": "Illustrative physical mechanism, not a calibrated real fastening process or CASCADE backend"}
