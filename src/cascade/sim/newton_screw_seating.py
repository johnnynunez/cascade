"""Long-stroke contact experiment with a socket that leaves the nut face exposed.

The geometry supports a 41 mm descent to the bolt shoulder. Contact points,
normals and forces distinguish nut seating from tool/fixture interference.
"""
from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from cascade.sim.newton_screw_contact import ThreadingScene
from cascade.sim.factory_recipe import LEGACY_RECIPE, seating_recipe


class SeatingScene(ThreadingScene):
    requested_speed_rad_s = 3.
    socket_offset = np.array([.012, 0., -.115])
    shoulder_z_m = .023
    nut_half_height_m = .008

    def __init__(self, *args, recipe=LEGACY_RECIPE, **kwargs):
        self._configure_recipe(recipe)
        self._seat_contact_records = []
        self._tool_fixture_records = []
        self._nut_bolt_records = []
        self._other_contact_records = []
        self._interference_seen = False
        self._substep_samples = []
        # Declared experiment parameter, not a calibrated material estimate.
        # This resists passive gravity-driven coasting in the loose threads.
        kwargs.setdefault('thread_friction', .1)
        super().__init__(*args, **kwargs)
        # Millimetre-spaced IK targets follow measured nut height. They do not
        # impose axial motion on the nut or prescribe a pitch to the controller.
        self._prepare_follower()
        shape_map = self.solver.mjc_geom_to_newton_shape.numpy()[0]
        labels = self._names(self.model.shape_label)
        self._all_tool_geoms = {int(np.flatnonzero(shape_map == index)[0])
                               for name, index in labels.items() if name.startswith('socket_')}
        self.seat_geom = int(np.flatnonzero(shape_map == labels['seat_ring'])[0])
        self._insert_q = [int(self._qstart[self.joints[f'socket_insert_slide_{i}']]) for i in range(6)]
        # Assert that imported springs remain passive in the native solver.
        native = self.solver.mj_model
        self._insert_native_dofs = []
        for i in range(6):
            joint = next(native.joint(j) for j in range(native.njnt)
                         if native.joint(j).name.endswith(f'socket_insert_slide_{i}'))
            dof, qpos = int(joint.dofadr[0]), int(joint.qposadr[0])
            self._insert_native_dofs.append(dof)
            if (not np.isclose(joint.stiffness[0], 2000.)
                    or not np.isclose(native.qpos_spring[qpos], -.00035)
                    or not np.isclose(native.dof_damping[dof], 4.)):
                raise RuntimeError('native socket insert passive spring was not imported')
        if native.nu != len(self._actuators):
            raise RuntimeError('unexpected actuator created for passive socket inserts')

    def _configure_recipe(self, name):
        recipe = seating_recipe(name)  # Reject unknown authoring before SDK IO.
        self.fixture_recipe = recipe.name
        self.fixture_center_xy_m = recipe.center_xy_m
        self.ik_margin_rad = recipe.ik_margin_rad
        self.intersect_position_control_range = recipe.intersect_position_control_range

    def _prepare_follower(self):
        self._heights = np.linspace(.027, self._initial_nut_z, 43)
        seed = self.entry.copy()
        targets = []
        for height in reversed(self._heights):
            seed = self._ik(np.r_[self._center_xy, height], seed)
            targets.append(seed)
        self._targets = np.asarray(list(reversed(targets)))

    def _add_fixture(self, builder, cache):
        import trimesh
        mesh = trimesh.creation.annulus(r_min=.011, r_max=.018, height=.003, sections=96)
        shape = self.newton.Mesh(np.asarray(mesh.vertices, dtype=np.float32),
                                np.asarray(mesh.faces.flatten(), dtype=np.int32))
        shape.build_sdf(max_resolution=256, narrow_band_range=(-.005, .005),
                        margin=.005, cache_dir=Path(cache))
        cfg = builder.ShapeConfig(margin=0., mu=.15, ke=1e7, kd=1e4,
                                  gap=.005, density=8000., mu_torsional=0., mu_rolling=0.)
        builder.add_shape_mesh(-1, xform=self.wp.transform(self.wp.vec3(*self._center_xy, .0215),
            self.wp.quat_identity()), mesh=shape, cfg=cfg, label='seat_ring', color=(.6,.4,.15))

    def _robot_xml(self, clearance):
        root = ET.fromstring(super()._robot_xml(clearance))
        # Raise the case 10 mm and extend the socket 5 mm. The lower inserts
        # then clear the original gripper mesh by 4.25 mm, without exclusions.
        root.find(".//body[@name='socket_housing']").set('pos', '.012 0 -.050')
        spindle = root.find(".//body[@name='socket_spindle']")
        spindle.set('pos', '0 0 -.031')
        for i, node in enumerate(spindle.findall('geom')):
            position, rotation = node.attrib['pos'], node.attrib['quat']
            spindle.remove(node)
            insert = ET.SubElement(spindle, 'body', name=f'socket_insert_{i}',
                                   pos=position, quat=rotation)
            ET.SubElement(insert, 'joint', name=f'socket_insert_slide_{i}', type='slide',
                          axis='1 0 0', limited='true', range='-.0006 .0008',
                          stiffness='2000', springref='-.00035', damping='4', frictionloss='0')
            node.set('pos', '0 0 0')
            node.set('quat', '1 0 0 0')
            node.set('size', '.003 .006 .006')
            node.set('solref', '.015 1')
            node.set('mass', '.002')
            insert.append(node)
        # Three outside standoffs connect the driven socket to its spindle.
        # Their entire cross section stays outside the nut and thread.
        for i in range(3):
            angle = i * 2*math.pi/3
            ET.SubElement(spindle, 'geom', name=f'socket_post_{i}', type='box',
                          pos=f'{.019*math.cos(angle)} {.019*math.sin(angle)} -.0065',
                          quat=f'{math.cos(angle/2)} 0 0 {math.sin(angle/2)}',
                          size='.002 .003 .0215', rgba='.6 .65 .72 1', mass='0',
                          contype='1', conaffinity='1', group='2')
        return ET.tostring(root, encoding='unicode')

    def _arm_target(self, nut_pose):
        return np.array([np.interp(nut_pose[2], self._heights, column)
                         for column in self._targets.T])

    def step(self):
        self._substep_samples = []
        super().step()

    def _on_substep(self, pose):
        self._substep_samples.append({'epoch':self.epoch, 'step':self.step_id, 'time_s':self.time_s,
            'fastener_id':'factory_nut_m20_loose', 'fixture_id':'fixed_factory_bolt_m20_loose',
            'fastener_position_m':pose[:3].tolist(), 'fastener_quaternion_xyzw':pose[3:].tolist(),
            'fixture_position_m':self.fixture_position.tolist(), 'fixture_quaternion_xyzw':[0.,0.,0.,1.],
            'thread_contacts':self._thread_contacts, 'tool_contacts':self._tool_contacts})
        if self._interference_seen:
            self.drive = False
        return self._interference_seen

    def _read_contacts(self):
        super()._read_contacts()
        data = self.solver.mjw_data
        count = int(data.nacon.numpy()[0])
        geoms = data.contact.geom.numpy()[:count]
        addresses = data.contact.efc_address.numpy()[:count, 0]
        positions = data.contact.pos.numpy()[:count]
        normals = data.contact.frame.numpy()[:count, 0]
        forces = data.efc.force.numpy()[0]
        seat, interference, nut_bolt, other = [], [], [], []
        for pair, address, position, normal in zip(geoms, addresses, positions, normals, strict=True):
            if address < 0 or forces[address] <= 1e-6:
                continue
            pair = set(map(int, pair))
            record = {'position_m':position.tolist(), 'normal':normal.tolist(),
                      'normal_force_n':float(forces[address]), 'geoms':sorted(pair)}
            if pair == {self.nut_geom, self.bolt_geom}:
                nut_bolt.append(record)
            elif pair == {self.nut_geom, self.seat_geom}:
                radius = np.linalg.norm(position[:2]-self._center_xy)
                if (abs(position[2]-self.shoulder_z_m) <= .0003
                        and radius >= .011 and abs(normal[2]) >= .95):
                    seat.append(record)
            elif pair.intersection({self.bolt_geom,self.seat_geom}) and pair.intersection(self._all_tool_geoms):
                interference.append(record)
            elif not (self.nut_geom in pair and pair.intersection(self.socket_geoms)):
                record['geom_names'] = [self.solver.mj_model.geom(i).name for i in sorted(pair)]
                other.append(record)
                if pair.intersection(self._all_tool_geoms) and not pair.issubset(self._all_tool_geoms):
                    record['interference_kind'] = 'tool_robot_mount'
                    interference.append(record)
        self._seat_contact_records = seat
        self._tool_fixture_records = interference
        self._nut_bolt_records = nut_bolt
        self._other_contact_records = other
        self._interference_seen = self._interference_seen or bool(interference)

    def observe(self):
        row = super().observe()
        velocity = self.state.body_qd.numpy()[self.nut_body]
        row.update(shoulder_contacts=len(self._seat_contact_records),
                   shoulder_id='fixed_annular_seat_3mm',
                   shoulder_contact_force_n=sum(x['normal_force_n'] for x in self._seat_contact_records),
                   shoulder_contact_records=self._seat_contact_records,
                   nut_bolt_contact_records=self._nut_bolt_records,
                   other_contact_records=self._other_contact_records,
                   tool_fixture_contact_records=self._tool_fixture_records,
                   tool_fixture_contacts=len(self._tool_fixture_records),
                   tool_fixture_interference_seen=self._interference_seen,
                   nut_angular_speed_rad_s=float(np.linalg.norm(velocity[3:])),
                   nut_linear_speed_m_s=float(np.linalg.norm(velocity[:3])),
                   nut_bottom_gap_m=float(row['axial_position_m']-self.nut_half_height_m-self.shoulder_z_m))
        row['socket_insert_displacement_m'] = self.state.joint_q.numpy()[self._insert_q].tolist()
        row['socket_insert_passive_force_n'] = self.solver.mjw_data.qfrc_passive.numpy()[0,self._insert_native_dofs].tolist()
        return row
