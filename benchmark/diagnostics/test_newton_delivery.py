"""Real Newton CPU acceptance tests; not CASCADE/Isaac episodes.

Run with the isolated Newton interpreter and run_newton_delivery.py to persist
per-test measurements, including failures. There are no mocks, skips or xfails.
The native pipeline in Newton 1.5.1 is exported as ``CollisionPipeline``;
``NewtonCollisionPipeline`` below is an explicit alias, not another engine.
"""
from pathlib import Path
import hashlib
import os
import xml.etree.ElementTree as ET

import newton
from newton import CollisionPipeline as NewtonCollisionPipeline
from newton.solvers import SolverMuJoCo
import numpy as np
import pytest
import warp as wp


REPO = Path(__file__).resolve().parents[2]
DT = 1.0 / 240.0
newton.use_coord_layout_targets = True
MUTATION = os.environ.get("NEWTON_DELIVERY_MUTATION")


@pytest.fixture
def measurements(request):
    request.node.newton_measurements = {}
    return request.node.newton_measurements


def transform(x, y, z):
    return wp.transform(wp.vec3(x, y, z), wp.quat_identity())


def make_runtime(model, *, contact_source="newton", disable_contacts=False, nconmax=256, njmax=512):
    assert str(model.device) == "cpu", "This suite certifies CPU only"
    native = contact_source == "newton"
    assert contact_source in ("newton", "mujoco")
    disable_contacts = disable_contacts or MUTATION == "disable_contacts"
    solver = SolverMuJoCo(
        model, use_mujoco_cpu=False, use_mujoco_contacts=not native,
        disable_contacts=disable_contacts, solver="newton",
        integrator="implicitfast", iterations=100, ls_iterations=50,
        tolerance=1e-10, ls_tolerance=1e-8, njmax=njmax, nconmax=nconmax,
        update_data_interval=1,
    )
    assert solver.use_mujoco_cpu is False
    assert solver._use_mujoco_contacts is (not native)
    assert bool(solver.mjw_model.opt.run_collision_detection) is (not native)
    state, next_state = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    newton.eval_fk(model, model.joint_q, model.joint_qd, next_state)
    pipeline = NewtonCollisionPipeline(model, rigid_contact_max=nconmax, deterministic=True,
                                      verify_buffers=True, max_triangle_pairs=1_000_000)
    return {
        "model": model, "solver": solver, "state": state,
        "next_state": next_state, "control": model.control(),
        "pipeline": pipeline, "contacts": pipeline.contacts(),
        "config": {
            "solver_class": "newton.solvers.SolverMuJoCo",
            "solver_algorithm": "newton", "execution_backend": "mujoco_warp",
            "device": str(model.device), "use_mujoco_cpu": False,
            "use_mujoco_contacts": not native,
            "collision_pipeline": ("newton.CollisionPipeline" if native else "MuJoCo Warp collision detection"),
            "disable_contacts": disable_contacts, "dt_s": DT,
            "iterations": 100, "ls_iterations": 50,
            "tolerance": 1e-10, "ls_tolerance": 1e-8,
            "integrator": "implicitfast", "njmax": njmax, "nconmax": nconmax,
            "update_data_interval": 1,
            "gravity_m_s2": model.gravity.numpy().tolist(),
            "effective_mujoco_solver_enum": int(solver.mj_model.opt.solver),
            "effective_mujoco_integrator_enum": int(solver.mj_model.opt.integrator),
            "effective_mujoco_cone_enum": int(solver.mj_model.opt.cone),
            "effective_mujoco_iterations": int(solver.mj_model.opt.iterations),
            "effective_mujoco_ls_iterations": int(solver.mj_model.opt.ls_iterations),
            "native_collision_settings": {"deterministic": True, "verify_buffers": True,
                "rigid_contact_max": nconmax, "max_triangle_pairs": 1_000_000} if native else None,
            "contact_count_note": "Candidate/speculative contact counts are not proof of active contact response; geometry and the collision-disabled control provide that proof.",
        },
    }


def check_finite(state):
    for attr in ("joint_q", "joint_qd", "body_q", "body_qd"):
        assert np.isfinite(getattr(state, attr).numpy()).all(), f"non-finite {attr}"


def advance(runtime, steps):
    """Integrate real physics; check every step, not only a final snapshot."""
    samples = []
    max_contacts = 0
    max_solver_contacts = 0
    max_constraints = 0
    max_actuator_force = 0.0
    max_generalized_actuation = 0.0
    shape_pairs = set()
    for step in range(steps):
        state = runtime["state"]
        state.clear_forces()
        if not runtime["config"]["use_mujoco_contacts"]:
            runtime["pipeline"].collide(state, runtime["contacts"])
            count = int(runtime["contacts"].rigid_contact_count.numpy()[0])
            assert count <= runtime["config"]["nconmax"], f"native contact buffer overflow: {count} > {runtime['config']['nconmax']}"
            assert count <= runtime["contacts"].rigid_contact_max
            max_contacts = max(max_contacts, count)
            if count:
                first = runtime["contacts"].rigid_contact_shape0.numpy()[:count]
                second = runtime["contacts"].rigid_contact_shape1.numpy()[:count]
                shape_pairs.update(tuple(sorted((int(a), int(b)))) for a, b in zip(first, second))
        runtime["solver"].step(state, runtime["next_state"], runtime["control"], runtime["contacts"], DT)
        runtime["state"], runtime["next_state"] = runtime["next_state"], state
        data = runtime["solver"].mjw_data
        solver_contacts = int(data.nacon.numpy()[0])
        constraints = int(data.nefc.numpy().max())
        max_solver_contacts = max(max_solver_contacts, solver_contacts)
        max_constraints = max(max_constraints, constraints)
        assert solver_contacts <= runtime["config"]["nconmax"], "MuJoCo Warp contact buffer overflow"
        assert constraints <= runtime["config"]["njmax"], "MuJoCo Warp constraint buffer overflow"
        assert not data.overflow.numpy().any(), "MuJoCo Warp overflow flag set"
        force = data.actuator_force.numpy()
        assert np.isfinite(force).all(), "non-finite actuator forces"
        if force.size:
            max_actuator_force = max(max_actuator_force, float(np.abs(force).max()))
        applied = data.qfrc_actuator.numpy()
        assert np.isfinite(applied).all(), "non-finite generalized actuator force"
        if applied.size:
            max_generalized_actuation = max(max_generalized_actuation, float(np.abs(applied).max()))
        check_finite(runtime["state"])
        if step % 10 == 0 or step == steps - 1:
            samples.append(runtime["state"].body_q.numpy().copy())
    return {"steps": steps, "simulated_s": steps * DT,
            "all_states_finite": True, "max_native_contacts": max_contacts,
            "max_solver_contacts": max_solver_contacts, "max_solver_constraints": max_constraints,
            "max_abs_raw_actuator_force": max_actuator_force,
            "max_abs_applied_generalized_actuation": max_generalized_actuation,
            "buffer_overflow": False,
            "native_contact_shape_pairs": sorted(shape_pairs),
            "body_samples": np.asarray(samples)}


def make_table_scene(spawn_z=0.33):
    builder = newton.ModelBuilder()
    table_shape = builder.add_shape_box(-1, xform=transform(0, 0, 0.1),
                                        hx=0.3, hy=0.3, hz=0.1, label="table")
    box = builder.add_body(xform=transform(0, 0, spawn_z), label="free_prop")
    half = 0.025
    cfg = builder.ShapeConfig(density=0.05 / (2 * half) ** 3)
    box_shape = builder.add_shape_box(box, hx=half, hy=half, hz=half, cfg=cfg, label="prop_box")
    model = builder.finalize(device="cpu")
    return model, box, table_shape, box_shape


def command_joint_targets(runtime, targets):
    model = runtime["model"]
    command = runtime["control"].joint_target_q.numpy()
    mapping = {label: index for index, label in enumerate(model.joint_label)}
    starts = model.joint_target_q_start.numpy()
    for name, value in targets.items():
        command[starts[mapping[name]]] = 0.0 if MUTATION == "zero_controls" else value
    runtime["control"].joint_target_q.assign(command)


def joint_values(runtime, names, velocity=False):
    model = runtime["model"]
    mapping = name_index(model.joint_label)
    starts = (model.joint_qd_start if velocity else model.joint_q_start).numpy()
    values = (runtime["state"].joint_qd if velocity else runtime["state"].joint_q).numpy()
    return np.asarray([values[starts[mapping[name]]] for name in names])


def name_index(labels):
    """Resolve basename names while rejecting ambiguous imported paths."""
    mapping = {}
    for index, label in enumerate(labels):
        name = label.rsplit("/", 1)[-1]
        assert name not in mapping, f"ambiguous name: {name}"
        mapping[name] = index
    return mapping


FINGER_HOME = 0.10
FINGER_COMMAND = 0.09
FINGER_CONTACT_AT = FINGER_HOME - (0.03 + 0.01)
FINGER_NAMES = ["left_slide", "right_slide"]


def make_finger_scene(effort):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    box = builder.add_body(xform=transform(0, 0, 0.5), label="free_grasp_box")
    box_shape = builder.add_shape_box(box, hx=0.03, hy=0.03, hz=0.03,
                                     cfg=builder.ShapeConfig(density=0.05 / 0.06**3), label="grasp_box")
    fingers, finger_shapes = [], []
    for sign, name in zip((-1.0, 1.0), FINGER_NAMES):
        finger = builder.add_link(label=name + "_body")
        shape = builder.add_shape_box(finger, hx=0.01, hy=0.04, hz=0.04,
                                      cfg=builder.ShapeConfig(density=0.2 / (0.02 * 0.08 * 0.08)), label=name + "_box")
        joint = builder.add_joint_prismatic(
            parent=-1, child=finger, label=name,
            parent_xform=transform(sign * FINGER_HOME, 0, 0.5), axis=wp.vec3(-sign, 0, 0),
            actuator_mode=newton.JointTargetMode.POSITION, target_pos=0.0,
            target_ke=200.0, target_kd=4.0, effort_limit=effort,
            limit_lower=-0.02, limit_upper=0.12,
        )
        builder.add_articulation([joint])
        fingers.append(finger)
        finger_shapes.append(shape)
    return builder.finalize(device="cpu"), box, box_shape, fingers, finger_shapes


@pytest.mark.parametrize("effort", [2.0, 10.0], ids=["2N", "10N"])
def test_fingers_move_and_stop_on_contact(effort, measurements):
    model, box, box_shape, fingers, finger_shapes = make_finger_scene(effort)
    runtime = make_runtime(model)
    measurements.update(config=runtime["config"], effort_limit_N=effort,
                        joint_target_ke=200.0, joint_target_kd=4.0,
                        body_masses_kg=model.body_mass.numpy().tolist(),
                        command_m=FINGER_COMMAND, contact_travel_m=FINGER_CONTACT_AT)
    initial_q = joint_values(runtime, FINGER_NAMES)
    command_joint_targets(runtime, dict.fromkeys(FINGER_NAMES, FINGER_COMMAND))
    trajectory = advance(runtime, 960)
    q = joint_values(runtime, FINGER_NAMES)
    qd = joint_values(runtime, FINGER_NAMES, velocity=True)
    body = runtime["state"].body_q.numpy()
    separation = float(abs(body[fingers[0], 0] - body[fingers[1], 0]))
    surface_gaps = np.abs(body[fingers, 0] - body[box, 0]) - 0.04
    trajectory.pop("body_samples")
    measurements.update(trajectory, initial_q_m=initial_q.tolist(), final_q_m=q.tolist(),
                        final_qd_m_s=qd.tolist(), finger_separation_m=separation,
                        surface_gaps_m=surface_gaps.tolist(), box_final_pose=body[box].tolist(),
                        raw_final_actuator_force_N=runtime["solver"].mjw_data.actuator_force.numpy().tolist(),
                        applied_final_generalized_actuation=runtime["solver"].mjw_data.qfrc_actuator.numpy().tolist(),
                        force_measurement_note="Newton effort_limit clamps the total via MuJoCo jnt_actfrcrange; qfrc_actuator is post-clamp, actuator_force is before this joint clamp.")
    # Same Newton model, same target and actuator, but physically disable
    # contact response. This is a real dynamical counterfactual, not a mock.
    no_collision = make_runtime(model, disable_contacts=True)
    command_joint_targets(no_collision, dict.fromkeys(FINGER_NAMES, FINGER_COMMAND))
    counterfactual = advance(no_collision, 960)
    counterfactual.pop("body_samples")
    free_q = joint_values(no_collision, FINGER_NAMES)
    measurements["collision_disabled_control"] = dict(counterfactual,
        config=no_collision["config"], final_q_m=free_q.tolist())
    assert np.all(np.abs(q - initial_q) > 0.04), "both fingers must REALLY move"
    assert trajectory["max_abs_applied_generalized_actuation"] <= effort + 1e-5, "finger effort limit was not enforced"
    assert np.all(np.abs(q - FINGER_CONTACT_AT) < 0.003), "fingers must stop ON box faces"
    assert np.all(FINGER_COMMAND - q > 0.02), "fingers passed through the box"
    assert np.all(np.abs(qd) < 0.002), "fingers have not stopped"
    assert np.all(np.abs(surface_gaps) < 0.003), "wrong final contact geometry"
    assert abs(separation - 0.08) < 0.006
    assert np.linalg.norm(body[box, :3] - [0, 0, 0.5]) < 0.003
    assert trajectory["max_native_contacts"] > 0
    for shape in finger_shapes:
        assert tuple(sorted((box_shape, shape))) in trajectory["native_contact_shape_pairs"]
    assert np.all(np.abs(free_q - FINGER_COMMAND) < 0.001), "collision-disabled fingers must traverse the box"
    assert np.all(free_q - q > 0.02), "contact did not cause the stop"


@pytest.mark.parametrize("contact_source", ["newton", "mujoco"], ids=["native-newton", "mujoco-contact-comparison"])
def test_box_rests_on_table(contact_source, measurements):
    model, box, table_shape, box_shape = make_table_scene()
    runtime = make_runtime(model, contact_source=contact_source)
    measurements.update(config=runtime["config"], prop_mass_kg=float(model.body_mass.numpy()[box]))
    initial_z = float(runtime["state"].body_q.numpy()[box, 2])
    trajectory = advance(runtime, 720)
    final = runtime["state"].body_q.numpy()[box]
    velocity = runtime["state"].body_qd.numpy()[box]
    expected_z = 0.2 + 0.025
    settled = trajectory.pop("body_samples")[-24:, box, 2]
    measurements.update(trajectory, initial_z_m=initial_z, final_z_m=float(final[2]),
                        expected_rest_z_m=expected_z, rest_error_m=float(final[2] - expected_z),
                        settled_z_span_m=float(np.ptp(settled)), final_velocity=velocity.tolist())
    assert initial_z - final[2] > 0.08, "box never fell: vacuous resting test"
    assert abs(final[2] - expected_z) < 0.001, "box floats or penetrates the table"
    assert np.ptp(settled) < 0.0002, "box has not reached durable rest"
    assert np.linalg.norm(velocity) < 0.001, "box still moving"
    if contact_source == "newton":
        assert trajectory["max_native_contacts"] > 0
        assert sorted((table_shape, box_shape)) in [list(p) for p in trajectory["native_contact_shape_pairs"]]


SO101_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def test_real_so101_mjcf_direct_control_tracks_named_joints(measurements):
    asset = REPO / "assets/mjcf/so101/so101.xml"
    assert asset.is_file(), f"Missing real SO-101 MJCF: {asset}"
    tree = ET.parse(asset).getroot()
    compiler = tree.find("compiler")
    assert compiler is not None and compiler.get("meshdir"), "MJCF meshdir missing"
    meshdir = asset.parent / compiler.attrib["meshdir"]
    mesh_files = sorted({meshdir / mesh.attrib["file"] for mesh in tree.findall("./asset/mesh")})
    assert mesh_files and all(path.is_file() for path in mesh_files), "SO-101 meshes missing"
    measurements.update(asset=str(asset.relative_to(REPO)),
                        asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
                        mesh_files=[{"path": str(path.relative_to(REPO)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in mesh_files],
                        mesh_file_count=len(mesh_files), ctrl_direct=True,
                        parse_meshes=True, parse_visuals=True, enable_self_collisions=True,
                        collapse_fixed_joints=False)
    builder = newton.ModelBuilder()
    # Import the original repo MJCF; no toy arm, replacement XML or mesh pruning.
    builder.add_mjcf(str(asset), ctrl_direct=True, parse_meshes=True,
                     parse_visuals=True, enable_self_collisions=True,
                     collapse_fixed_joints=False)
    model = builder.finalize(device="cpu")
    joint_map = name_index(model.joint_label)
    body_map = name_index(model.body_label)
    assert set(SO101_JOINTS).issubset(joint_map)
    qstarts, dstarts = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
    for name in SO101_JOINTS:
        joint = joint_map[name]
        assert qstarts[joint + 1] - qstarts[joint] == 1
        assert dstarts[joint + 1] - dstarts[joint] == 1
        assert model.joint_type.numpy()[joint] == int(newton.JointType.REVOLUTE)
    assert model.joint_coord_count == model.joint_dof_count == len(SO101_JOINTS)
    # CTRL_DIRECT indices follow the original MJCF actuator order, not joint
    # order. Verify each imported target DOF against the named joint address.
    actuator_element = tree.find("actuator")
    assert actuator_element is not None, "MJCF actuators missing"
    actuators = list(actuator_element)
    actuator_map = {element.attrib["joint"]: index for index, element in enumerate(actuators)}
    assert set(actuator_map) == set(SO101_JOINTS) and len(actuators) == len(SO101_JOINTS)
    trnid = model.mujoco.actuator_trnid.numpy()
    for name, index in actuator_map.items():
        assert trnid[index, 0] == dstarts[joint_map[name]], f"wrong actuator mapping: {name}"
    assert np.all(model.mujoco.ctrl_source.numpy() == int(SolverMuJoCo.CtrlSource.CTRL_DIRECT))
    shape_types, shape_flags = model.shape_type.numpy(), model.shape_flags.numpy()
    mesh_mask = shape_types == int(newton.GeoType.MESH)
    collision_mask = (shape_flags & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
    measurements.update(body_count=model.body_count, joint_count=model.joint_count,
                        shape_count=model.shape_count, mesh_shape_count=int(mesh_mask.sum()),
                        collision_mesh_shape_count=int((mesh_mask & collision_mask).sum()),
                        named_joint_map={name: {"joint_index": joint_map[name], "q_index": int(qstarts[joint_map[name]]),
                                               "dof_index": int(dstarts[joint_map[name]]), "control_index": actuator_map[name]} for name in SO101_JOINTS})
    assert (mesh_mask & collision_mask).sum() >= 3, "real gripper collision meshes were discarded"
    runtime = make_runtime(model, nconmax=1024, njmax=8192)
    measurements["config"] = runtime["config"]
    assert np.all(runtime["solver"].mjc_actuator_ctrl_source.numpy() == int(SolverMuJoCo.CtrlSource.CTRL_DIRECT))
    initial = joint_values(runtime, SO101_JOINTS)
    initial_body = runtime["state"].body_q.numpy()[body_map["gripper"], :3].copy()
    measurements["initial_q_rad"] = dict(zip(SO101_JOINTS, initial.tolist()))
    measurements["waypoints"] = []
    targets = [
        dict(zip(SO101_JOINTS, [0.25, -0.3, 0.35, 0.2, -0.4, 0.6])),
        dict(zip(SO101_JOINTS, [-0.2, 0.15, -0.25, -0.15, 0.25, 0.25])),
    ]
    before = initial
    for target in targets:
        control = runtime["control"].mujoco.ctrl.numpy()
        for name, value in target.items():
            index = actuator_map[name]
            lower, upper = model.mujoco.actuator_ctrlrange.numpy()[index]
            assert lower < value < upper
            control[index] = 0.0 if MUTATION == "zero_controls" else value
        runtime["control"].mujoco.ctrl.assign(control)
        trajectory = advance(runtime, 720)
        trajectory.pop("body_samples")
        actual = joint_values(runtime, SO101_JOINTS)
        speed = joint_values(runtime, SO101_JOINTS, velocity=True)
        errors = np.abs(actual - [target[name] for name in SO101_JOINTS])
        body = runtime["state"].body_q.numpy()[body_map["gripper"], :3]
        displacement = float(np.linalg.norm(body - initial_body))
        measurements["waypoints"].append(dict(trajectory, target_rad=target,
            actual_rad=dict(zip(SO101_JOINTS, actual.tolist())),
            actual_qd_rad_s=dict(zip(SO101_JOINTS, speed.tolist())),
            max_tracking_error_rad=float(errors.max()),
            motion_per_joint_rad=np.abs(actual - before).tolist(),
            gripper_body_displacement_from_start_m=displacement))
        assert np.all(np.abs(actual - before) > 0.08), "every named SO-101 joint must REALLY move"
        assert errors.max() < 0.01, "direct controls did not track the named SO-101 targets"
        assert np.abs(speed).max() < 0.02, "SO-101 did not settle"
        assert displacement > 0.015, "joint changes did not move the real gripper body"
        before = actual.copy()


def test_free_prop_reset_is_durable_in_joint_and_body_state(measurements):
    """Newton reset API only; this is not CASCADE's/Isaac's reset route."""
    model, box, table_shape, box_shape = make_table_scene(spawn_z=0.2255)
    runtime = make_runtime(model)
    spawn = model.body_q.numpy()[box, :3].copy()
    joint_ids = np.flatnonzero(model.joint_child.numpy() == box)
    assert len(joint_ids) == 1
    joint = int(joint_ids[0])
    assert model.joint_type.numpy()[joint] == int(newton.JointType.FREE)
    qstart = int(model.joint_q_start.numpy()[joint])
    dstart = int(model.joint_qd_start.numpy()[joint])
    measurements.update(config=runtime["config"], reset_api="SolverMuJoCo.reset + newton.eval_fk",
                        scope="Newton free-prop reset, not CASCADE/Isaac reset_scene",
                        spawn_xyz_m=spawn.tolist(), cycles=[])
    advance(runtime, 240)
    for sign in (-1.0, 1.0):
        state = runtime["state"]
        q, qd = state.joint_q.numpy(), state.joint_qd.numpy()
        # Arrange a displaced/spinning free prop via genuine generalized state,
        # then integrate it before resetting. No fake state object is involved.
        q[qstart:qstart + 3] = spawn + [sign * 0.12, -0.03, 0.08]
        qd[dstart:dstart + 6] = [sign * 0.15, 0.0, 0.05, 0.4, -0.3, 0.2]
        state.joint_q.assign(q)
        state.joint_qd.assign(qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        advance(runtime, 30)
        before = runtime["state"].body_q.numpy()[box, :3].copy()
        before_distance = float(np.linalg.norm(before - spawn))
        state = runtime["state"]
        if MUTATION == "body_only_reset":
            # Classic false acknowledgement: visual/body pose reset without
            # changing the reduced coordinates the next step will restore.
            state.body_q.assign(model.body_q)
        else:
            runtime["solver"].reset(state)
            state.clear_forces()
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        immediate = state.body_q.numpy()[box, :3].copy()
        after_one_second = advance(runtime, 240)
        after_one_second.pop("body_samples")
        after = runtime["state"].body_q.numpy()[box, :3].copy()
        durable = advance(runtime, 720)
        samples = durable.pop("body_samples")[:, box, :3]
        distances = np.linalg.norm(samples - spawn, axis=1)
        final = runtime["state"].body_q.numpy()[box, :3].copy()
        speed = runtime["state"].body_qd.numpy()[box]
        measurements["cycles"].append(dict(durable,
            before_xyz_m=before.tolist(), before_displacement_m=before_distance,
            immediate_xyz_m=immediate.tolist(), after_1s_xyz_m=after.tolist(),
            final_xyz_m=final.tolist(), final_velocity=speed.tolist(),
            final_spawn_distance_m=float(np.linalg.norm(final - spawn)),
            maximum_durable_spawn_distance_m=float(distances.max()),
            durable_window_s=3.0, total_post_reset_s=4.0))
        assert before_distance > 0.10, "reset premise is vacuous: prop was not displaced"
        assert np.linalg.norm(immediate - spawn) < 1e-6, "reset not applied to body pose"
        assert np.linalg.norm(after - spawn) < 0.001, "reset lost on subsequent physics steps"
        assert distances.max() < 0.001, "free-prop reset is not durable"
        assert np.linalg.norm(speed) < 0.001, "reset prop never settled"
        assert tuple(sorted((table_shape, box_shape))) in durable["native_contact_shape_pairs"]
