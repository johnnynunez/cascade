"""Host-buffer adversaries and CPU geometry only; no native Factory admission."""
from copy import deepcopy
from dataclasses import dataclass
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from cascade.control.fastening import FasteningFault
from cascade.sim.factory_observation import (
    FactoryGeometry, FactoryObserver, actuator_descriptor, collision_coverage, contact_records, joint_mapping, world_aabb,
)
from test_fastening_runtime import binding, limits
from test_microduck_contact_support import Buffer, fixture as contact_fixture


class Array(Buffer):
    @property
    def shape(self):
        return self.value.shape


def counter(n=0):
    return Array([n], np.int32)


def coverage_fixture():
    backing = lambda: Array(np.zeros((10, 2)), np.int32)
    phase = NS(hydroelastic_sdf=None, split_gjk_mpr=True, sparse_gjk_pairs=False,
        gjk_candidate_pairs_count=counter(2), gjk_candidate_pairs=backing(),
        split_query_results=backing(), split_gjk_work_count=counter(1), split_gjk_work_items=backing(),
        split_manifold_work_count=counter(1), split_manifold_work_items=backing(),
        global_contact_reducer=NS(hashtable=NS(capacity=10, active_slots=Array([0]*10+[3], np.int32)),
                                  ht_insert_failures=counter()))
    for stem in ("shape_pairs_mesh", "triangle_pairs", "shape_pairs_mesh_plane", "shape_pairs_mesh_mesh"):
        setattr(phase, stem, backing())
        setattr(phase, stem+"_count", counter())
    phase.shape_pairs_sdf_sdf = None
    phase.shape_pairs_sdf_sdf_count = counter()
    pipeline = NS(narrow_phase=phase, broad_phase_pair_count=counter(2), broad_phase_shape_pairs=backing())
    contacts = NS(rigid_contact_count=counter(1), rigid_contact_shape0=Array([0]*10, np.int32), rigid_contact_max=10)
    return pipeline, contacts


def test_collision_records_all_intermediate_capacity_and_reducer_checks():
    p, c = coverage_fixture()
    result = collision_coverage(p, c)
    assert len(result) == 12
    assert result["contacts"] == {"count": 1, "capacity": 10}
    assert result["reducer"] == {"count": 3, "capacity": 10, "insert_failures": 0}
    assert result["shape_pairs_sdf_sdf"] == {"count": 0, "capacity": 0}
    # Returned receipt is detached from mutable SDK buffers.
    p.broad_phase_pair_count.value[0] = 11
    assert result["broad_phase"]["count"] == 2


@pytest.mark.parametrize("name", ["broad_phase", "gjk", "split_query", "split_gjk", "split_manifold",
    "shape_pairs_mesh", "triangle_pairs", "shape_pairs_mesh_plane", "shape_pairs_mesh_mesh",
    "contacts", "reducer_load", "reducer_failure", "sdf_disabled", "hydroelastic", "missing"])
def test_intermediate_overflow_or_missing_counter_is_not_a_complete_empty_contact_set(name):
    p, c = coverage_fixture()
    n = p.narrow_phase
    if name == "broad_phase": p.broad_phase_pair_count.value[0] = 10
    elif name == "gjk": n.gjk_candidate_pairs_count.value[0] = 11
    elif name == "split_query": n.split_query_results.value = np.zeros((1, 2), np.int32)
    elif name in ("split_gjk", "split_manifold"): getattr(n, name+"_work_count").value[0] = 10
    elif name == "contacts": c.rigid_contact_count.value[0] = 10
    elif name == "reducer_load": n.global_contact_reducer.hashtable.active_slots.value[-1] = 8
    elif name == "reducer_failure": n.global_contact_reducer.ht_insert_failures.value[0] = 1
    elif name == "sdf_disabled": n.shape_pairs_sdf_sdf_count.value[0] = 1
    elif name == "hydroelastic": n.hydroelastic_sdf = object()
    elif name == "missing": n.triangle_pairs_count = None
    else: getattr(n, name+"_count").value[0] = 10
    with pytest.raises(FasteningFault):
        collision_coverage(p, c)


def test_disabled_pipeline_buffers_are_explicitly_empty_not_required_fake_allocations():
    p, c = coverage_fixture()
    p.narrow_phase.shape_pairs_mesh = p.narrow_phase.shape_pairs_mesh_count = None
    assert collision_coverage(p, c)["shape_pairs_mesh"]["disabled"] is True


def test_world_aabb_rotated_box_contains_exact_corners_without_second_halving():
    angle = math.pi / 4
    rotation = np.array([[math.cos(angle), -math.sin(angle), 0.],
                         [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
    center, half, origin = np.array([.03, .02, .01]), np.array([.1, .2, .3]), np.array([1., 2., 3.])
    lower, upper = world_aabb([np.r_[center, half]], [origin], [rotation])
    corners = np.array([origin+rotation@(center+half*[x, y, z])
                        for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    np.testing.assert_allclose(lower, corners.min(0))
    np.testing.assert_allclose(upper, corners.max(0))
    assert upper[2]-lower[2] == pytest.approx(.6)


def test_mujoco_compiled_geom_aabb_is_center_and_half_extent_cpu_only():
    mj = pytest.importorskip("mujoco")
    model = mj.MjModel.from_xml_string('''<mujoco><worldbody><body pos="1 2 3">
      <geom name="box" type="box" size=".1 .2 .3" pos=".03 .02 .01"
      quat=".9238795325112867 0 0 .3826834323650898"/>
    </body></worldbody></mujoco>''')
    data = mj.MjData(model)
    mj.mj_kinematics(model, data)  # No physics step.
    np.testing.assert_allclose(model.geom_aabb[0], [0., 0., 0., .1, .2, .3])
    lower, upper = world_aabb(model.geom_aabb, data.geom_xpos, data.geom_xmat.reshape(-1, 3, 3))
    np.testing.assert_allclose(np.asarray(upper)-lower, [.6/math.sqrt(2), .6/math.sqrt(2), .6])


@pytest.mark.parametrize("bad", ["negative_half", "nan", "scaled_rotation"])
def test_aabb_invalid_geometry_refuses(bad):
    a, p, r = np.array([[0., 0., 0., 1., 1., 1.]]), np.zeros((1, 3)), np.eye(3)[None]
    if bad == "negative_half": a[0, -1] = -1
    elif bad == "nan": p[0, 0] = np.nan
    else: r *= 2
    with pytest.raises(FasteningFault): world_aabb(a, p, r)


def geometry_fixture():
    mj = pytest.importorskip("mujoco")
    m = mj.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="table" type="box" pos="0 0 -.1" size="1 1 .1"/>
      <body name="arm" pos="0 0 .5"><joint type="hinge" name="arm_joint" axis="0 0 1"/>
       <geom name="armgeom" type="box" pos=".1 0 0" size=".1 .02 .02"/>
       <body name="fixed_tool" pos=".2 0 0"><geom name="tool" type="box" size=".02 .02 .03"/></body>
      </body>
      <body name="nut" pos="0 0 .1"><freejoint/><geom name="nutgeom" type="box" size=".01 .01 .01"/></body>
    </worldbody></mujoco>''')
    return NS(solver=NS(mj_model=m, mjc_geom_to_newton_shape=Array([[0, 1, 2, 3]], np.int32),
                        mjc_body_to_newton=Array([[-1, 0, 1, 2]], np.int32)),
        model=NS(shape_flags=Array([1]*4, np.int32), shape_body=Array([-1, 0, 1, 2], np.int32),
                 shape_label=["table", "armgeom", "tool", "nutgeom"], body_count=3),
        mujoco=mj, newton=NS(ShapeFlags=NS(COLLIDE_SHAPES=1)), nut_body=2)


def test_geometry_includes_rigid_tool_child_but_excludes_static_table_and_free_nut():
    scene = geometry_fixture()
    geometry = FactoryGeometry(scene, ())
    assert geometry.descriptor["moving_shapes"] == ["armgeom", "tool"]
    q = scene.solver.mj_model.qpos0.copy()
    initial = q.copy()
    lower, upper = geometry.evaluate(q)
    np.testing.assert_allclose(lower, [0., -.02, .47])
    np.testing.assert_allclose(upper, [.22, .02, .53])
    assert np.array_equal(q, initial)  # Never mutate the solved buffer.
    poses = np.c_[geometry.shadow.xpos[1:], geometry.shadow.xquat[1:, [1, 2, 3, 0]]]
    geometry.evaluate(q, body_poses=poses)
    poses[1, 0] += .01
    with pytest.raises(FasteningFault, match="frame disagreement"):
        geometry.evaluate(q, body_poses=poses)


def test_geometry_refuses_omitted_or_duplicate_collision_mesh():
    scene = geometry_fixture()
    scene.solver.mjc_geom_to_newton_shape.value[0, 2] = 1
    with pytest.raises(FasteningFault, match="mapping disagrees|missing or duplicated"):
        FactoryGeometry(scene, ())


def test_body_quaternion_with_correct_dot_but_nonunit_norm_cannot_pass_fk():
    scene = geometry_fixture()
    geometry = FactoryGeometry(scene, ())
    q = scene.solver.mj_model.qpos0.copy()
    geometry.evaluate(q)
    poses = np.c_[geometry.shadow.xpos[1:], geometry.shadow.xquat[1:, [1, 2, 3, 0]]]
    poses[0, 3:] = [1., 0., 0., 1.]  # Dot with identity remains 1; norm is sqrt(2).
    with pytest.raises(FasteningFault, match="non-unit"):
        geometry.evaluate(q, body_poses=poses)


@pytest.mark.parametrize("bad", ["empty", "truncated", "all_unmapped", "duplicate", "wrong_geom_body", "dtype"])
def test_missing_body_mapping_cannot_silently_skip_native_newton_fk_comparison(bad):
    scene = geometry_fixture()
    b = scene.solver.mjc_body_to_newton
    if bad == "empty": b.value = np.array([], np.int32)
    elif bad == "truncated": b.value = b.value[:, :-1]
    elif bad == "all_unmapped": b.value[:] = -1
    elif bad == "duplicate": b.value[0, 2] = 0
    elif bad == "dtype": b.value = b.value.astype(float)
    else: b.value[0, 1:3] = [1, 0]
    with pytest.raises(FasteningFault, match="body.*mapping|mapping.*bod"):
        FactoryGeometry(scene, ())


def test_contact_decoder_keeps_loaded_zero_and_inactive_candidates_distinct():
    s = contact_fixture(cone=1)
    s.solver.mjw_data.overflow = counter()
    s.solver.update_contacts = lambda output: None
    s.solver.mjw_data.nacon.value[0] = s.contacts.rigid_contact_count.value[0] = 2
    pairs, records = contact_records(s, s.contacts)
    assert len(pairs) == 2 and [p.normal_force_n for p in pairs] == [4., 0.]
    assert records[0]["point_world_m"] == pytest.approx([.1, .2, 0.])
    assert records[1]["status"] == "inactive_candidate" and records[1]["point_world_m"] is None
    s.solver.mjw_data.efc.force.value[:] = s.contacts.force.value[:] = 0
    _, records = contact_records(s, s.contacts)
    assert records[0]["status"] == "solved" and records[0]["normal_force_n"] == 0


def test_contact_decoder_native_overflow_cannot_be_decoded_as_empty():
    s = contact_fixture()
    s.solver.mjw_data.overflow = counter(1)
    with pytest.raises(FasteningFault, match="overflow"): contact_records(s, s.contacts)


@pytest.mark.parametrize("value,dtype", [([], np.int32), ([[-1]], np.int32), ([-1], np.int32),
                                       ([0.], np.float32), ([0, 0], np.int32)])
def test_unavailable_overflow_channel_is_not_zero_overflow(value, dtype):
    s = contact_fixture()
    s.solver.mjw_data.overflow = Array(value, dtype)
    with pytest.raises(FasteningFault, match="counter"):
        contact_records(s, s.contacts)


def mapping_fixture():
    m = NS(njnt=2, nv=2, nu=2, jnt_type=np.array([3, 3]), jnt_dofadr=np.array([1, 0]),
        jnt_qposadr=np.array([1, 0]), actuator_trnid=np.array([[1, -1], [0, -1]]),
        actuator_trntype=np.array([0, 0]), actuator_gear=np.array([[1., 0, 0, 0, 0, 0]]*2))
    s = NS(mj_model=m, mjc_jnt_to_newton_jnt=Array([[1, 0]], np.int32),
        mjc_dof_to_newton_dof=Array([[0, 1]], np.int32),
        mjc_actuator_ctrl_source=Array([1, 1], np.int32), mjc_actuator_to_newton_idx=Array([1, 0], np.int32))
    return NS(solver=s, joints={"arm": 0, "socket_spin": 1}, _actuators={"arm": 1, "socket_motor": 0},
        model=NS(joint_q_start=Array([0, 1], np.int32), joint_qd_start=Array([0, 1], np.int32),
                 mujoco=NS(dof_ref=Array([.1, 0.])), joint_dof_count=2))


def test_mapping_uses_native_joint_dof_and_control_maps_not_list_order():
    rows = joint_mapping(mapping_fixture(), ("arm", "socket_spin"))
    assert rows[0].native_joint == 1 and rows[0].native_dof == 0 and rows[0].control_index == 1
    assert rows[1].native_joint == 0 and rows[1].native_dof == 1 and rows[1].control_index == 0
    assert rows[0].reference_rad == pytest.approx(.1)


@pytest.mark.parametrize("bad", ["gear", "second_actuator", "ball", "target_source", "wrong_ctrl", "wrong_dof", "extra"])
def test_mapping_refuses_ambiguous_effort_and_transmission(bad):
    scene = mapping_fixture()
    s, m = scene.solver, scene.solver.mj_model
    if bad == "gear": m.actuator_gear[0, 0] = 2
    elif bad == "second_actuator": m.actuator_trnid[1, 0] = 1
    elif bad == "ball": m.jnt_type[0] = 1
    elif bad == "target_source": s.mjc_actuator_ctrl_source.value[0] = 0
    elif bad == "wrong_ctrl": s.mjc_actuator_to_newton_idx.value[0] = 0
    elif bad == "wrong_dof": s.mjc_dof_to_newton_dof.value[0, 0] = 1
    else: m.nu = 3
    with pytest.raises(FasteningFault): joint_mapping(scene, ("arm", "socket_spin"))


def actuator_fixture():
    scene = mapping_fixture()
    m = scene.solver.mj_model
    m.body_gravcomp, m.jnt_actgravcomp = np.zeros(2), np.zeros(2)
    m.actuator_gainprm = np.zeros((2, 10)); m.actuator_gainprm[:, 0] = [10., 1.]
    m.actuator_biasprm = np.zeros((2, 10)); m.actuator_biasprm[0, 1:3] = [-10., -.2]
    m.actuator_biastype = np.array([1, 0])
    m.actuator_dyntype = m.actuator_gaintype = m.actuator_actnum = np.zeros(2, dtype=int)
    m.actuator_ctrllimited = m.actuator_forcelimited = np.ones(2, dtype=int)
    m.actuator_ctrlrange = np.array([[-1., 1.], [-.05, .05]])
    m.actuator_forcerange = np.array([[-2., 2.], [-.05, .05]])
    m.jnt_axis = np.array([[0., 0., -1.], [0., 1., 0.]])
    m.jnt_range = np.array([[0., 0.], [-1., 1.]])
    m.jnt_limited = np.array([0, 1]); m.jnt_actfrclimited = np.zeros(2, dtype=int)
    m.jnt_actfrcrange = np.zeros((2, 2))
    return m, joint_mapping(scene, ("arm", "socket_spin"))


def test_descriptor_binds_actual_gains_caps_and_effort_axis():
    model, rows = actuator_fixture()
    result = actuator_descriptor(model, rows)
    assert result[0]["kind"] == "position_servo" and result[0]["gain"][0] == 10.
    assert result[1]["kind"] == "direct_effort" and result[1]["joint_axis_local"] == [0., 0., -1.]
    assert result[1]["force_range"] == [-.05, .05]


@pytest.mark.parametrize("bad", ["gravity", "dynamic", "gain", "bias", "cap", "unlimited", "arm_bias"])
def test_descriptor_refuses_motor_model_changes_not_observable_from_net_effort(bad):
    model, rows = actuator_fixture()
    if bad == "gravity": model.body_gravcomp[0] = 1
    elif bad == "dynamic": model.actuator_dyntype[1] = 1
    elif bad == "gain": model.actuator_gainprm[1, 0] = 2
    elif bad == "bias": model.actuator_biastype[1] = 1
    elif bad == "cap": model.actuator_forcerange[1, 1] = .1
    elif bad == "unlimited": model.actuator_forcelimited[1] = 0
    else: model.actuator_biasprm[0, 1] = 0
    with pytest.raises(FasteningFault): actuator_descriptor(model, rows)


@dataclass(frozen=True)
class Upload:
    generation: int = 2
    effort_nm: float = .03
    before_step: int = 0


def observer_fixture():
    s = contact_fixture(cone=1)
    s.model.shape_label = list(binding().collider_names)
    s.model.joint_coord_count = s.model.joint_dof_count = s.model.body_count = 2
    s.solver.mjw_data.overflow = counter()
    s.solver.update_contacts = lambda output: None
    # Candidate0 nut->bolt; force sign keeps the SDK decoder's original basis.
    s.solver.mjc_geom_to_newton_shape = Array([[0, 1, 2, 3, 4]], np.int32)
    s.contacts.rigid_contact_shape0.value[:] = 0
    s.contacts.rigid_contact_shape1.value[:] = 1
    d = s.solver.mjw_data
    d.qpos = Array([[.1, 0.]])
    d.qvel = Array([[0., .5]])
    d.qfrc_actuator = Array([[.2, .03]])
    d.qfrc_applied = Array([[0., 0.]])
    d.xfrc_applied = Array(np.zeros((1, 2, 6)))
    d.qfrc_constraint = Array([[.9, .4]])
    d.qfrc_passive = Array([[.5, .7]])
    d.ctrl = Array([[.1, .03]])
    d.actuator_force = Array([[.2, .03]])
    s.solver.mj_model = NS(nq=2, nv=2, nu=2, nbody=2)
    s.state = NS(joint_q=Array([0., 0.]), joint_qd=Array([0., .5]),
        body_q=Array([[0., 0., .069, 0., 0., 0., 1.], [0., 0., .069, 0., 0., 0., 1.]]),
        body_qd=Array(np.zeros((2, 6))))
    s.nut_body, s.bodies = 0, {"socket_spindle": 1}
    s.epoch, s.step_id, s.time_s = "epoch", 1, .01
    observer = FactoryObserver.__new__(FactoryObserver)
    observer.scene, observer.binding, observer.limits = s, binding(), limits()
    observer.geometry = NS(evaluate=lambda *a, **k: ((-.1, -.1, .03), (.3, .1, .2)))
    observer.joints = joint_mapping(mapping_fixture(), ("arm", "socket_spin"))
    observer.output, observer._last_step = s.contacts, 0
    return observer


def test_observer_retains_uploaded_generation_and_actual_effort_instead_of_postsolve_guard_state():
    observer = observer_fixture()
    observation, raw = observer.read(Upload(), 10., {"contacts": {"count": 1, "capacity": 4}})
    assert observation.generation == 2 and observation.step == 1
    assert observation.joint_effort_nm == pytest.approx((.2,))
    assert observation.spindle_effort_nm == pytest.approx(.03)
    assert observation.commanded_spindle_effort_nm == .03
    assert raw["qfrc_constraint"] == pytest.approx([.9, .4])
    saved = deepcopy(raw)
    observer.scene.state.body_q.value[:] = 999
    assert raw == saved  # No consumer-visible SDK array alias.
    with pytest.raises(FasteningFault, match="skipped or repeated"):
        observer.read(Upload(), 10., {"contacts": {"count": 1, "capacity": 4}})


@pytest.mark.parametrize("bad", ["joint_force", "body_force", "wrong_qmap", "wrong_vmap", "nonfinite", "skipped",
                                "wrong_upload", "wrong_actuator_force"])
def test_observer_refuses_external_efforts_map_drift_and_missing_solves(bad):
    o = observer_fixture()
    d = o.scene.solver.mjw_data
    if bad == "joint_force": d.qfrc_applied.value[0, 0] = .001
    elif bad == "body_force": d.xfrc_applied.value[0, 0, 0] = .001
    elif bad == "wrong_qmap": d.qpos.value[0, 0] += .01
    elif bad == "wrong_vmap": d.qvel.value[0, 0] += .01
    elif bad == "nonfinite": d.qfrc_actuator.value[0, 0] = np.nan
    elif bad == "wrong_upload": d.ctrl.value[0, 1] = .04
    elif bad == "wrong_actuator_force": d.actuator_force.value[0, 1] = .04
    else: o.scene.step_id = 2
    with pytest.raises(FasteningFault):
        o.read(Upload(), 10., {"contacts": {"count": 1, "capacity": 4}})


@pytest.mark.parametrize("name", ["qpos", "qvel", "qfrc_actuator", "qfrc_applied", "xfrc_applied",
                                "qfrc_constraint", "qfrc_passive", "ctrl", "actuator_force",
                                "joint_q", "joint_qd", "body_q", "body_qd"])
@pytest.mark.parametrize("bad", ["empty", "transposed", "float64"])
def test_missing_or_wrong_layout_force_channels_cannot_be_interpreted_as_known_zero(name, bad):
    o = observer_fixture()
    parent = o.scene.state if name in {"joint_q", "joint_qd", "body_q", "body_qd"} else o.scene.solver.mjw_data
    channel = getattr(parent, name)
    if bad == "empty": channel.value = np.array([], np.float32)
    elif bad == "float64": channel.value = channel.value.astype(np.float64)
    elif channel.value.ndim == 1: channel.value = channel.value[:, None]
    else: channel.value = channel.value.T
    with pytest.raises(FasteningFault, match="channel"):
        o.read(Upload(), 10., {"contacts": {"count": 1, "capacity": 4}})
