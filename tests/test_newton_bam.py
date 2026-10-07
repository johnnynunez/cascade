"""Native PR kernels on CPU; buffer contracts are NOT physics simulation."""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def runtime():
    wp = pytest.importorskip("warp", reason="native BAM tests require Warp")
    newton = pytest.importorskip("newton", reason="native BAM tests require Newton >=1.6")
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "run native BAM tests CPU-only"
    return wp, newton


@pytest.fixture(scope="module")
def source_root():
    root = os.environ.get("NEWTON_BAM_SOURCE_ROOT")
    if root is None:
        pytest.skip("set NEWTON_BAM_SOURCE_ROOT to the externally admitted IsaacLab PR source tree")
    return Path(root)


def module():
    return importlib.import_module("cascade.control.newton_bam")


def recipe():
    """Explicit CPU contract recipe when the lab runs the exact Isaac-bundled pre-release.

    Stable Newton keeps the default admission (no recipe); the pre-release is
    never accepted implicitly.
    """
    import newton
    from cascade.sim import microduck_sdk as sdk
    return sdk.CPU_CONTRACT_RECIPE if newton.__version__ == sdk.INTERNAL_NEWTON_VERSION else None


def test_loads_only_pinned_native_modules(runtime, source_root):
    loaded = module().load_pinned_bam(source_root, sdk_recipe=recipe())
    assert loaded.revision == "28aa1fca5843208ff9a67935695a4d5376e44d50"
    assert loaded.DriveBam.__name__ == "DriveBam"
    assert loaded.MjWarpActuatorBridge.__name__ == "MjWarpActuatorBridge"
    assert loaded.DriveBam.__module__.startswith("_cascade_native_bam_")
    assert len(loaded.sha256) == 3
    assert not any(k == "isaaclab" or k.startswith(("isaaclab.", "isaaclab_newton")) for k in sys.modules)
    assert "torch" not in sys.modules


def params(**overrides):
    result = dict(kp_fw=200.0, vin=7.4, max_current=None, vin_drop_gain=0.0,
                vin_min=6.0, min_delay=0, max_delay=0, delay_hold_prob=0.0,
                delay_update_period=0, delay_seed=17, max_effort=0.96,
                joint_effort_limit=0.96, stiff_frictionloss=False, physics_dt=0.005)
    result.update(overrides)
    return result


def make_model(runtime, hinges=16):
    wp, newton = runtime
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    root = builder.add_link(mass=1.0, inertia=wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1))
    joints = [builder.add_joint_free(root)]
    for i in range(hinges):
        body = builder.add_link(mass=0.1, inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01))
        joints.append(builder.add_joint_revolute(root, body, label=f"hinge{i}",
                      effort_limit=0.96, target_ke=0.0, target_kd=0.0,
                      damping=0.0, armature=0.0, friction=0.0,
                      actuator_mode=newton.JointTargetMode.NONE))
    builder.add_articulation(joints)
    return builder.finalize(device="cpu")


def buffer_stage(runtime, hinges=16):
    """Channel test double: real Newton model + real Warp arrays, NO solver.

    Synthetic qfrc inputs exercise native bridge kernels, never a trajectory.
    notify below mimics property transport only; it does NOT integrate physics.
    """
    from types import SimpleNamespace as NS
    import numpy as np

    wp, _ = runtime
    model = make_model(runtime, hinges)
    n = model.joint_dof_count
    v5 = wp.types.vector(length=5, dtype=wp.float32)
    mjm = NS(nu=0, opt=NS(disableflags=0), dof_frictionloss=wp.full((1, n), 0.123, device="cpu"),
             dof_damping=wp.zeros((1, n), device="cpu"), dof_armature=wp.zeros((1, n), device="cpu"),
             dof_solref=wp.full((1, n), wp.vec2(0.02, 1.0), device="cpu"),
             dof_solimp=wp.full((1, n), v5(0.9, 0.95, 0.001, 0.5, 2.0), device="cpu"))
    mjd = NS(qfrc_bias=wp.zeros((1, n), device="cpu"), qfrc_constraint=wp.zeros((1, n), device="cpu"),
             nefc=wp.zeros(1, dtype=wp.int32, device="cpu"),
             efc=NS(type=wp.zeros((1, 32), dtype=wp.int32, device="cpu"),
                    id=wp.zeros((1, 32), dtype=wp.int32, device="cpu"),
                    force=wp.zeros((1, 32), device="cpu")))
    solver = NS(model=model, use_mujoco_cpu=False, mjw_model=mjm, mjw_data=mjd,
                mjc_dof_to_newton_dof=wp.array(np.arange(n, dtype=np.int32)[None], device="cpu"))
    def notify(flags):
        solver.notifications.append(int(flags))
        for src, dst in (("joint_armature", "dof_armature"), ("joint_damping", "dof_damping"),
                         ("joint_friction", "dof_frictionloss")):
            getattr(mjm, dst).assign(getattr(model, src).numpy()[None])
        mjm.dof_solref.assign(model.mujoco.solreffriction.numpy()[None])
        mjm.dof_solimp.assign(model.mujoco.solimpfriction.numpy()[None])
    solver.notifications = []
    solver.notify_model_changed = notify
    stage = NS(model=model, solver=solver, state_0=model.state(), control=model.control(),
               cfg=NS(num_substeps=1), simulation_step_count=0)
    # Deliberately reversed, offset and non-contiguous from free-root q/qd.
    stage.test_dofs = np.arange(6, 20, dtype=np.int64)[::-1].copy()
    stage.test_q = stage.test_dofs + 1
    return stage


def adapter_for(stage, source_root, **overrides):
    p = params()
    p.update(overrides)
    return module().NewtonBamAdapter(stage, source_root=source_root, sdk_recipe=recipe(),
                 q_indices=stage.test_q, dof_indices=stage.test_dofs, params=p)


def test_native_effort_scatter_and_owned_properties(runtime, source_root):
    import numpy as np
    from cascade.control.microduck_actuator import BamXL330M6, M6_PARAMETERS

    stage = buffer_stage(runtime)
    q_before, qd_before = stage.state_0.joint_q.numpy(), stage.state_0.joint_qd.numpy()
    stage.control.joint_f.fill_(0.456)
    adapter = adapter_for(stage, source_root)
    target = np.linspace(-0.4, 0.4, 14, dtype=np.float32)
    adapter.set_targets(target)
    adapter.before_step(0.005)
    reference = BamXL330M6(kp_fw=200, vin=7.4, max_current=None, vin_drop_gain=0,
                          vin_min=6, friction_reference="bam_mjlab").evaluate(
        target, q_before[stage.test_q], qd_before[stage.test_dofs],
        previous_motor_torque=np.zeros(14), previous_external_torque=np.zeros(14),
        previous_commanded_torque=np.zeros(14))
    output = stage.control.joint_f.numpy()
    np.testing.assert_allclose(output[stage.test_dofs], reference.effort_nm, rtol=2e-6, atol=2e-7)
    unowned = np.setdiff1d(np.arange(len(output)), stage.test_dofs)
    np.testing.assert_array_equal(output[unowned], np.float32(0.456))
    np.testing.assert_array_equal(stage.state_0.joint_q.numpy(), q_before)
    np.testing.assert_array_equal(stage.state_0.joint_qd.numpy(), qd_before)
    for name, coefficient in (("joint_damping", "friction_viscous"), ("joint_armature", "armature")):
        actual = getattr(stage.model, name).numpy()
        np.testing.assert_allclose(actual[stage.test_dofs], M6_PARAMETERS[coefficient], rtol=1e-7)
        np.testing.assert_array_equal(actual[unowned], 0.0)
    np.testing.assert_allclose(stage.solver.mjw_model.dof_frictionloss.numpy()[0, stage.test_dofs],
                               reference.frictionloss_nm, rtol=2e-6, atol=2e-7)
    assert stage.solver.notifications


@pytest.mark.parametrize("change, message", [
    (lambda s: setattr(s.cfg, "num_substeps", 2), "substep"),
    (lambda s: setattr(s.solver, "use_mujoco_cpu", True), "MJWarp"),
    (lambda s: setattr(s.solver.mjw_model, "nu", 1), "actuator"),
    (lambda s: s.model.joint_target_ke.fill_(1.0), "drive"),
    (lambda s: s.model.joint_target_kd.fill_(1.0), "drive"),
    (lambda s: s.model.joint_target_mode.fill_(1), "drive"),
    (lambda s: s.model.joint_damping.fill_(0.4), "joint_damping"),
    (lambda s: s.model.joint_armature.fill_(0.4), "joint_armature"),
    (lambda s: s.model.joint_effort_limit.fill_(0.1), "effort"),
    (lambda s: setattr(s.solver, "notify_model_changed", None), "notify"),
    (lambda s: setattr(s.solver, "model", None), "model"),
    (lambda s: setattr(s, "simulation_step_count", -1), "step_count"),
])
def test_rejects_unsafe_capabilities_before_writes(runtime, source_root, change, message):
    import numpy as np
    stage = buffer_stage(runtime)
    change(stage)
    before = stage.model.joint_armature.numpy().copy()
    effort = stage.control.joint_f.numpy().copy()
    with pytest.raises((ValueError, RuntimeError), match=message):
        adapter_for(stage, source_root)
    np.testing.assert_array_equal(stage.model.joint_armature.numpy(), before)
    np.testing.assert_array_equal(stage.control.joint_f.numpy(), effort)


@pytest.mark.parametrize("key, value", [
    ("unknown", 5), ("kp_fw", float("nan")), ("vin", 0), ("vin_min", 8),
    ("max_current", 0), ("max_current", -1), ("max_current", 1e-300),
    ("vin_min", 1e-300), ("physics_dt", 1e-300), ("vin_drop_gain", -0.1),
    ("min_delay", 1.5), ("min_delay", -1), ("max_delay", -1),
    ("delay_hold_prob", 1.1), ("delay_update_period", -1), ("delay_seed", True),
    ("max_effort", 0), ("max_effort", 1.5), ("joint_effort_limit", 0),
    ("physics_dt", 0), ("stiff_frictionloss", "false"),
])
def test_rejects_invalid_explicit_parameters(runtime, source_root, key, value):
    stage = buffer_stage(runtime)
    with pytest.raises(ValueError, match=key if key != "max_delay" else "delay"):
        adapter_for(stage, source_root, **{key: value})


def test_requires_every_parameter(runtime, source_root):
    stage = buffer_stage(runtime)
    p = params()
    del p["joint_effort_limit"]
    with pytest.raises(ValueError, match="joint_effort_limit"):
        module().NewtonBamAdapter(stage, source_root=source_root, sdk_recipe=recipe(), q_indices=stage.test_q,
                                 dof_indices=stage.test_dofs, params=p)


@pytest.mark.parametrize("case", ["fractional", "duplicate", "negative", "wrong_q", "length", "root"])
def test_rejects_non_scalar_or_inconsistent_mapping(runtime, source_root, case):
    import numpy as np
    stage = buffer_stage(runtime)
    q, dofs = stage.test_q.copy(), stage.test_dofs.copy()
    if case == "fractional":
        dofs = dofs.astype(float) + 0.5
    elif case == "duplicate":
        dofs[0] = dofs[1]
    elif case == "negative":
        dofs[0] = -1
    elif case == "wrong_q":
        q = q[::-1].copy()
    elif case == "length":
        dofs = dofs[:-1]
    elif case == "root":
        dofs[0], q[0] = 0, 0
    before = stage.control.joint_f.numpy()
    with pytest.raises(ValueError, match="indices|scalar|revolute"):
        module().NewtonBamAdapter(stage, source_root=source_root, sdk_recipe=recipe(), q_indices=q, dof_indices=dofs, params=params())
    np.testing.assert_array_equal(stage.control.joint_f.numpy(), before)


@pytest.mark.parametrize("case", ["unexpanded", "duplicate_map", "missing_map", "bad_dtype", "bad_efc"])
def test_rejects_unsafe_solver_array_layout(runtime, source_root, case):
    import numpy as np
    wp, _ = runtime
    stage = buffer_stage(runtime)
    solver = stage.solver
    n = stage.model.joint_dof_count
    if case == "unexpanded":
        solver.mjc_dof_to_newton_dof = wp.array(np.vstack([np.arange(n), np.full(n, -1)]), dtype=wp.int32, device="cpu")
    elif case in ("duplicate_map", "missing_map"):
        m = solver.mjc_dof_to_newton_dof.numpy()
        m[0, stage.test_dofs[0]] = stage.test_dofs[1] if case == "duplicate_map" else -1
        solver.mjc_dof_to_newton_dof.assign(m)
    elif case == "bad_dtype":
        solver.mjw_data.qfrc_bias = wp.zeros((1, n), dtype=wp.float64, device="cpu")
    else:
        solver.mjw_data.efc.force = wp.zeros((1, 2), device="cpu")
    with pytest.raises((ValueError, RuntimeError), match="map|expanded|array|shape|dtype"):
        adapter_for(stage, source_root)


def test_reset_discards_delay_sag_load_and_targets(runtime, source_root):
    import numpy as np
    stage = buffer_stage(runtime)
    adapter = adapter_for(stage, source_root, min_delay=3, max_delay=6, vin_drop_gain=0.2,
                          delay_update_period=4)
    for i in range(10):
        stage.simulation_step_count = i
        adapter.set_targets(np.full(14, i * 0.1))
        adapter.before_step(0.005)
    q_before = stage.state_0.joint_q.numpy()
    stage.control.joint_f.assign(np.full(stage.model.joint_dof_count, 0.42, np.float32))
    stage.solver.mjw_data.qfrc_constraint.fill_(7.0)  # Deliberately stale solver history.
    adapter.reset()
    reset = adapter.telemetry()
    assert reset["steps_since_reset"] == 0 and reset["armed"] is False
    for key in ("motor_torque", "effort", "external_torque", "friction_budget", "delay_fill"):
        np.testing.assert_array_equal(reset[key], np.zeros(14))
    np.testing.assert_array_equal(stage.control.joint_f.numpy()[stage.test_dofs], 0.0)
    unowned = np.setdiff1d(np.arange(stage.model.joint_dof_count), stage.test_dofs)
    np.testing.assert_array_equal(stage.control.joint_f.numpy()[unowned], np.float32(0.42))
    np.testing.assert_array_equal(stage.state_0.joint_q.numpy(), q_before)
    with pytest.raises(RuntimeError, match="targets"):
        adapter.before_step(0.005)
    adapter.set_targets(np.full(14, -0.1))
    adapter.before_step(0.005)
    after = adapter.telemetry()
    np.testing.assert_array_equal(after["external_torque"], np.zeros(14))
    np.testing.assert_allclose(after["effective_vin"], 7.4, rtol=1e-7)
    assert np.all(np.array(after["effort"]) < 0)  # Never replays the old positive targets.
    assert after["reset_count"] == 1


@pytest.mark.parametrize("stiff", [False, True])
def test_friction_softness_is_explicit_owned_and_survives_sync(runtime, source_root, stiff):
    import numpy as np
    stage = buffer_stage(runtime)
    model = stage.model
    before = model.mujoco.solreffriction.numpy().copy()
    adapter = adapter_for(stage, source_root, stiff_frictionloss=stiff)
    expected = adapter._sources.MjWarpActuatorBridge.STIFF_SOLREF_FRICTION if stiff else before[stage.test_dofs]
    np.testing.assert_allclose(model.mujoco.solreffriction.numpy()[stage.test_dofs], np.broadcast_to(expected, (14, 2)))
    unowned = np.setdiff1d(np.arange(model.joint_dof_count), stage.test_dofs)
    np.testing.assert_array_equal(model.mujoco.solreffriction.numpy()[unowned], before[unowned])
    stage.solver.notify_model_changed(runtime[1].ModelFlags.JOINT_DOF_PROPERTIES)
    np.testing.assert_allclose(stage.solver.mjw_model.dof_solref.numpy()[0, stage.test_dofs], np.broadcast_to(expected, (14, 2)))
    adapter.set_targets(np.zeros(14))
    adapter.before_step(0.005)
    np.testing.assert_allclose(stage.solver.mjw_model.dof_frictionloss.numpy()[0, stage.test_dofs],
                               adapter.telemetry()["friction_budget"])


@pytest.mark.parametrize("case", ["no_targets", "bad_dt", "duplicate", "skipped", "substeps", "nan_q", "nefc", "unsynced"])
def test_step_contract_fails_before_effort_write(runtime, source_root, case):
    import numpy as np
    stage = buffer_stage(runtime)
    adapter = adapter_for(stage, source_root)
    if case != "no_targets":
        adapter.set_targets(np.full(14, 0.1))
    dt = 0.01 if case == "bad_dt" else 0.005
    if case in ("duplicate", "skipped"):
        adapter.before_step(dt)
        if case == "skipped":
            stage.simulation_step_count += 2
    elif case == "substeps":
        stage.cfg.num_substeps = 2
    elif case == "nan_q":
        stage.state_0.joint_q.fill_(float("nan"))
    elif case == "nefc":
        stage.solver.mjw_data.nefc.fill_(-1)
    elif case == "unsynced":
        stage.solver.mjw_model.dof_damping.zero_()
    before = stage.control.joint_f.numpy().copy()
    with pytest.raises((ValueError, RuntimeError)):
        adapter.before_step(dt)
    np.testing.assert_array_equal(stage.control.joint_f.numpy(), before)


@pytest.mark.parametrize("bad", [[0.1], [[0.0]*14], [float("nan")]*14, [float("inf")]*14])
def test_target_validation_is_fail_closed(runtime, source_root, bad):
    stage = buffer_stage(runtime)
    adapter = adapter_for(stage, source_root)
    adapter.set_targets([0.2]*14)
    with pytest.raises(ValueError, match="targets"):
        adapter.set_targets(bad)
    with pytest.raises(RuntimeError, match="targets"):
        adapter.before_step(0.005)


def test_reads_current_stage_buffer_not_a_cached_state(runtime, source_root):
    import numpy as np
    stage = buffer_stage(runtime)
    adapter = adapter_for(stage, source_root)
    adapter.set_targets(np.full(14, 0.1))
    adapter.before_step(0.005)
    assert np.all(np.array(adapter.telemetry()["effort"]) > 0)
    stage.state_0 = stage.model.state()
    new_q = stage.state_0.joint_q.numpy()
    new_q[stage.test_q] = 0.3
    stage.state_0.joint_q.assign(new_q)
    stage.simulation_step_count += 1
    adapter.before_step(0.005)
    assert np.all(np.array(adapter.telemetry()["effort"]) < 0)
    np.testing.assert_array_equal(stage.state_0.joint_q.numpy(), new_q)


@pytest.mark.parametrize("current", [None, 1.75])
@pytest.mark.parametrize("sag", [0.0, 0.2])
@pytest.mark.parametrize("delay", [(0, 0), (3, 3), (3, 6)])
def test_native_sequence_matches_declared_mjlab_cpu_reference(runtime, source_root, current, sag, delay):
    import json
    import numpy as np
    import mujoco
    from cascade.control.microduck_actuator import BamXL330M6

    stage = buffer_stage(runtime)
    adapter = adapter_for(stage, source_root, max_current=current, vin_drop_gain=sag,
                          min_delay=delay[0], max_delay=delay[1], delay_update_period=4,
                          delay_hold_prob=0.2, max_effort=0.2)
    reference = BamXL330M6(kp_fw=200, vin=7.4, max_current=current, vin_drop_gain=sag,
                          vin_min=6, friction_reference="bam_mjlab")
    prev_motor, prev_applied = np.zeros(14), np.zeros(14)
    commands = []
    errors = {key: 0.0 for key in ("motor_torque", "effort", "effective_vin", "friction_budget", "external_torque")}
    for step in range(48):
        stage.simulation_step_count = step
        q = stage.state_0.joint_q.numpy()
        dq = stage.state_0.joint_qd.numpy()
        q[stage.test_q] = np.linspace(-0.3, 0.4, 14) * np.cos(step * 0.2)
        dq[stage.test_dofs] = np.linspace(-4.0, 4.0, 14) * np.sin(step * 0.3)
        stage.state_0.joint_q.assign(q)
        stage.state_0.joint_qd.assign(dq)
        target = (np.linspace(-2, 2, 14) * np.sin(step * 0.4) + 0.3).astype(np.float32)
        # Keep replay away from the discontinuous |motor| == |external| branch.
        # Exact, binary-representable ties have their own boundary test below.
        ext = (np.linspace(-0.5, 0.5, 14) if step % 3 else np.copysign(2 * np.abs(prev_applied) + 0.03125, prev_applied)).astype(np.float32)
        mjd = stage.solver.mjw_data
        bias = np.zeros(mjd.qfrc_bias.shape, np.float32)
        bias[0, stage.test_dofs] = -ext
        constraints = np.zeros_like(bias)
        constraints[0, stage.test_dofs[0]] = 0.125 + 0.25
        types, ids, forces = mjd.efc.type.numpy(), mjd.efc.id.numpy(), mjd.efc.force.numpy()
        types[0, :2] = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)
        ids[0, :2] = stage.test_dofs[0]
        forces[0, :2] = [0.125, 0.25]  # TWO own rows must both be excluded.
        mjd.qfrc_bias.assign(bias)
        mjd.qfrc_constraint.assign(constraints)
        mjd.efc.type.assign(types)
        mjd.efc.id.assign(ids)
        mjd.efc.force.assign(forces)
        mjd.nefc.fill_(2)
        adapter.set_targets(target)
        adapter.before_step(0.005)
        telemetry = adapter.telemetry()
        lag = np.asarray(telemetry["delay_lag"])
        assert np.all(lag == lag[0])
        assert delay[0] <= lag[0] <= delay[1]
        selected = commands[-min(int(lag[0]), len(commands))] if lag[0] and commands else target
        expected = reference.evaluate(selected, q[stage.test_q], dq[stage.test_dofs],
                         previous_motor_torque=prev_applied,
                         previous_external_torque=ext if step else np.zeros(14),
                         previous_commanded_torque=prev_motor)
        for key, wanted in (("motor_torque", expected.effort_nm),
                            ("effort", np.clip(expected.effort_nm, -0.2, 0.2)),
                            ("effective_vin", np.full(14, expected.effective_vin_v)),
                            ("friction_budget", expected.frictionloss_nm),
                            ("external_torque", ext if step else np.zeros(14))):
            actual = np.asarray(telemetry[key])
            np.testing.assert_allclose(actual, wanted, rtol=3e-6, atol=3e-7, err_msg=f"{key}: step={step}")
            errors[key] = max(errors[key], float(np.max(np.abs(actual-wanted))))
        prev_motor = expected.effort_nm
        prev_applied = np.clip(expected.effort_nm, -0.2, 0.2)
        commands.append(target.copy())
    print("NATIVE_NUMERIC_RECEIPT", json.dumps(dict(current=current, sag=sag, delay=delay,
          steps=len(commands), dofs=14, max_absolute_error=max(errors.values()),
          max_absolute_error_by_channel=errors, reference="BamXL330M6:bam_mjlab",
          provenance="synthetic_inputs_original_Warp_kernels_not_trajectory")))


def test_exact_history_ties_use_mjlab_not_bam_model_variant(runtime, source_root):
    import numpy as np
    from cascade.control.microduck_actuator import BamXL330M6

    stage = buffer_stage(runtime)
    adapter = adapter_for(stage, source_root)
    adapter.set_targets(np.zeros(14))
    adapter.before_step(0.005)
    stage.simulation_step_count += 1
    # Injected *numeric* history boundary, not measured physics.
    previous = np.full(14, 0.125, np.float32)
    external = np.array([0.125, -0.125, 0.25, -0.25, 0.0625, -0.0625, 0]*2, np.float32)
    adapter._state.prev_applied_torque.assign(previous)
    bias = np.zeros(stage.solver.mjw_data.qfrc_bias.shape, np.float32)
    bias[0, stage.test_dofs] = -external
    stage.solver.mjw_data.qfrc_bias.assign(bias)
    adapter.before_step(0.005)
    budgets = {}
    for variant in ("bam_mjlab", "bam_model"):
        budgets[variant] = BamXL330M6(kp_fw=200, vin=7.4, max_current=None,
                   vin_drop_gain=0, vin_min=6, friction_reference=variant).evaluate(
                       np.zeros(14), np.zeros(14), np.zeros(14), previous_motor_torque=previous,
                       previous_external_torque=external, previous_commanded_torque=np.zeros(14)).frictionloss_nm
    np.testing.assert_allclose(adapter.telemetry()["friction_budget"], budgets["bam_mjlab"], rtol=2e-6, atol=2e-7)
    assert np.max(np.abs(budgets["bam_mjlab"]-budgets["bam_model"])) > 1e-4


@pytest.mark.parametrize("stiff", [False, True])
def test_real_mjwarp_cpu_solver_consumes_native_bam(runtime, source_root, stiff):
    """Actual CPU MJWarp stepping, not Isaac Sim or MicroDuck acceptance."""
    from types import SimpleNamespace as NS
    import json
    import numpy as np
    import mujoco
    from cascade.control.microduck_actuator import M6_PARAMETERS

    wp, newton = runtime
    model = make_model(runtime)
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=False, use_mujoco_contacts=True,
                                        nconmax=8, njmax=64)
    stage = NS(model=model, solver=solver, state_0=model.state(), control=model.control(),
               cfg=NS(num_substeps=1), simulation_step_count=0)
    stage.test_dofs = np.arange(6, 20, dtype=np.int64)[::-1].copy()
    stage.test_q = stage.test_dofs + 1
    adapter = adapter_for(stage, source_root, stiff_frictionloss=stiff)
    q0 = stage.state_0.joint_q.numpy().copy()
    adapter.set_targets(np.linspace(-0.1, 0.1, 14, dtype=np.float32))
    friction_type = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)
    peak_rows, peak_load_error = 0, 0.0
    for step in range(12):
        stage.simulation_step_count = step
        expected_load = np.zeros(14)
        if step:
            data = solver.mjw_data
            types, ids, forces = data.efc.type.numpy()[0], data.efc.id.numpy()[0], data.efc.force.numpy()[0]
            valid = np.arange(len(types)) < int(data.nefc.numpy()[0])
            for i, dof in enumerate(stage.test_dofs):
                own = forces[valid & (types == friction_type) & (ids == dof)].sum()
                expected_load[i] = -data.qfrc_bias.numpy()[0, dof] + data.qfrc_constraint.numpy()[0, dof] - own
        adapter.before_step(0.005)
        telemetry = adapter.telemetry()
        peak_load_error = max(peak_load_error, float(np.max(np.abs(expected_load - telemetry["external_torque"]))))
        np.testing.assert_allclose(telemetry["external_torque"], expected_load, atol=2e-7, rtol=2e-6)
        output = model.state()
        solver.step(stage.state_0, output, stage.control, None, 0.005)
        wp.synchronize()
        stage.state_0 = output
        assert np.isfinite(output.joint_q.numpy()).all()
        np.testing.assert_allclose(solver.mjw_data.qfrc_applied.numpy()[0, stage.test_dofs], telemetry["effort"], atol=2e-7)
        peak_rows = max(peak_rows, int(np.count_nonzero(solver.mjw_data.efc.type.numpy()[0, :int(solver.mjw_data.nefc.numpy()[0])] == friction_type)))
    displacement = float(np.max(np.abs(stage.state_0.joint_q.numpy()[stage.test_q] - q0[stage.test_q])))
    assert displacement > 1e-4
    assert peak_rows >= 14
    np.testing.assert_allclose(solver.mjw_model.dof_armature.numpy()[0, stage.test_dofs], M6_PARAMETERS["armature"], rtol=1e-6)
    print("REAL_MJWARP_CPU_RECEIPT", json.dumps(dict(device=str(model.device), solver=type(solver).__name__,
          use_mujoco_cpu=solver.use_mujoco_cpu, stiff=stiff, steps=12,
          max_joint_displacement=displacement, peak_friction_rows=peak_rows, max_load_error=peak_load_error,
          fixture="free_root_16_hinges_not_MicroDuck", kit=False)))
    physical_q = stage.state_0.joint_q.numpy().copy()
    adapter.reset()
    np.testing.assert_array_equal(stage.state_0.joint_q.numpy(), physical_q)
    np.testing.assert_array_equal(stage.control.joint_f.numpy()[stage.test_dofs], 0)
    adapter.set_targets(np.zeros(14))
    adapter.before_step(0.005)
    np.testing.assert_array_equal(adapter.telemetry()["external_torque"], 0)


@pytest.mark.parametrize("filename", ["bam_component.py", "bam_kernels.py", "mjwarp_actuator_bridge.py"])
def test_loader_rechecks_all_hashes_even_after_cache(runtime, source_root, tmp_path, filename):
    loaded = module().load_pinned_bam(source_root, sdk_recipe=recipe())
    for relative in loaded.sha256:
        dst = tmp_path / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        content = (source_root / relative).read_bytes()
        if dst.name == filename:
            content += b"\nraise AssertionError('unverified source executed')\n"
        dst.write_bytes(content)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module().load_pinned_bam(tmp_path, sdk_recipe=recipe())


@pytest.mark.parametrize("version", ["1.5.0", "1.6.0rc1", "bogus"])
def test_old_or_unknown_newton_fails_before_source_execution(runtime, source_root, monkeypatch, version):
    monkeypatch.setattr(runtime[1], "__version__", version)
    with pytest.raises(RuntimeError, match="Newton >=1.6"):
        module().load_pinned_bam(source_root, sdk_recipe=recipe())


def test_plain_import_has_no_newton_warp_isaaclab_or_torch_import():
    import subprocess
    result = subprocess.run([sys.executable, "-c", "import sys; import cascade.control.newton_bam; "
               "assert not any(m in sys.modules for m in ('newton','warp','torch','isaaclab','isaaclab_newton'))"],
               capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_loader_never_runs_external_init_or_unverified_bytecode(runtime, source_root, tmp_path):
    import subprocess
    import py_compile
    import struct
    for relative in module().SOURCE_SHA256:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Make a cached bytecode file that MUST NOT run, then restore valid source.
        destination.write_text("raise AssertionError('unverified bytecode executed')\n")
        cached = Path(py_compile.compile(str(destination), doraise=True))
        destination.write_bytes((source_root / relative).read_bytes())
        poisoned = bytearray(cached.read_bytes())
        poisoned[8:16] = struct.pack("<II", int(destination.stat().st_mtime), destination.stat().st_size)
        cached.write_bytes(poisoned)
        (destination.parent / "__init__.py").write_text("raise AssertionError('external init executed')\n")
    result = subprocess.run([sys.executable, "-c", "from cascade.control.newton_bam import load_pinned_bam; "
               "from cascade.sim import microduck_sdk as sdk; import newton, sys; "
               "r = sdk.CPU_CONTRACT_RECIPE if newton.__version__ == sdk.INTERNAL_NEWTON_VERSION else None; "
               "s=load_pinned_bam(sys.argv[1], sdk_recipe=r); assert s.DriveBam.__name__=='DriveBam'", str(tmp_path)],
               capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_constructor_does_not_fetch(runtime, source_root, monkeypatch):
    import socket
    import urllib.request
    def forbidden(*args, **kwargs):
        pytest.fail("network attempted while constructing BAM")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    adapter_for(buffer_stage(runtime), source_root)


@pytest.mark.parametrize("flag", ["mjDSBL_CONSTRAINT", "mjDSBL_FRICTIONLOSS", "mjDSBL_DAMPER"])
def test_rejects_disabled_bam_solver_channels(runtime, source_root, flag):
    import mujoco
    stage = buffer_stage(runtime)
    stage.solver.mjw_model.opt.disableflags = int(getattr(mujoco.mjtDisableBit, flag))
    with pytest.raises(RuntimeError, match="disabled"):
        adapter_for(stage, source_root)


def test_rejects_hidden_passive_spring(runtime, source_root):
    stage = buffer_stage(runtime)
    stage.model.mujoco.dof_passive_stiffness.fill_(1.0)
    with pytest.raises(RuntimeError, match="spring"):
        adapter_for(stage, source_root)


def test_refuses_changed_friction_tuning_even_after_sync(runtime, source_root):
    stage = buffer_stage(runtime)
    adapter = adapter_for(stage, source_root, stiff_frictionloss=True)
    stage.model.mujoco.solreffriction.fill_(runtime[0].vec2(0.02, 1.0))
    stage.solver.notify_model_changed(runtime[1].ModelFlags.JOINT_DOF_PROPERTIES)
    adapter.set_targets([0.0]*14)
    with pytest.raises(RuntimeError, match="friction.*changed"):
        adapter.before_step(0.005)


@pytest.mark.parametrize("change", ["stage_model", "solver_data", "map_array", "map_values", "nefc_overflow"])
def test_live_rebinding_and_constraint_overflow_are_rejected(runtime, source_root, change):
    from types import SimpleNamespace as NS
    wp, _ = runtime
    stage = buffer_stage(runtime)
    adapter = adapter_for(stage, source_root)
    if change == "stage_model":
        stage.model = make_model(runtime)
    elif change == "solver_data":
        stage.solver.mjw_data = NS()
    elif change == "map_array":
        stage.solver.mjc_dof_to_newton_dof = wp.clone(stage.solver.mjc_dof_to_newton_dof)
    elif change == "map_values":
        mapping = stage.solver.mjc_dof_to_newton_dof.numpy().copy()
        mapping[0, stage.test_dofs[:2]] = mapping[0, stage.test_dofs[:2]][::-1]
        stage.solver.mjc_dof_to_newton_dof.assign(mapping)
    else:
        stage.solver.mjw_data.nefc.fill_(33)
    adapter.set_targets([0.0]*14)
    with pytest.raises(RuntimeError):
        adapter.before_step(0.005)


def test_manifest_matches_static_admission_and_has_no_implicit_profile():
    import json
    path = Path(__file__).resolve().parents[1] / "assets/microduck/newton-bam.json"
    manifest = json.loads(path.read_text())
    assert manifest["revision"] == module().REVISION
    assert {f["path"]: f["sha256"] for f in manifest["files"]} == dict(module().SOURCE_SHA256)
    assert manifest["license"]["spdx"] == "BSD-3-Clause"
    assert "THIS SOFTWARE IS PROVIDED" in manifest["license"]["text"]
    assert manifest["default_profile"] is None
    assert len(manifest["profiles"]) == 4
    for profile in manifest["profiles"].values():
        assert set(profile) == set(params())
        module()._validated_params(profile)
    reference = manifest['profiles']['official_infer_nominal_no_delay']
    assert reference['stiff_frictionloss'] is True
    assert reference['vin_drop_gain'] == .1 and reference['max_current'] is None
    assert reference['min_delay'] == reference['max_delay'] == 0
    assert reference['kp_fw'] == 200. and reference['vin'] == 7.4
    assert reference['max_effort'] == reference['joint_effort_limit'] == 7.4*.36601349688984386/2.8113923539223227
    assert manifest['profiles']['nominal_no_current_limit_no_delay']['stiff_frictionloss'] is False


# --- shared-scene host snapshot: one device read per world array per step ---

def cohort_stage(runtime, source_root):
    """Two adapters on ONE stage, disjoint owned DOFs (the shared-scene shape)."""
    import numpy as np
    stage = buffer_stage(runtime, hinges=28)
    first = adapter_for(stage, source_root)
    second_dofs = np.arange(20, 34, dtype=np.int64)
    second = module().NewtonBamAdapter(stage, source_root=source_root, sdk_recipe=recipe(),
                                       q_indices=second_dofs + 1, dof_indices=second_dofs, params=params())
    return stage, first, second


def counting_numpy(monkeypatch, wp):
    calls = []
    original = wp.array.numpy

    def counted(self, *args, **kwargs):
        calls.append(self)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(wp.array, "numpy", counted)
    return calls


def test_cohort_snapshot_reads_each_world_array_once_with_identical_efforts(runtime, source_root, monkeypatch):
    import numpy as np
    wp, _ = runtime
    private_stage, a1, b1 = cohort_stage(runtime, source_root)
    shared_stage, a2, b2 = cohort_stage(runtime, source_root)
    for adapter in (a1, b1, a2, b2):
        adapter.set_targets(np.full(14, 0.1))
    calls = counting_numpy(monkeypatch, wp)
    efforts = {"private": [], "shared": []}
    reads = {"private": [], "shared": []}
    for step in range(3):
        del calls[:]
        a1.before_step(0.005)
        b1.before_step(0.005)
        reads["private"].append(len(calls))
        efforts["private"].append(private_stage.control.joint_f.numpy().copy())
        del calls[:]
        snapshot = a2.host_snapshot()
        a2.before_step(0.005, snapshot=snapshot)
        b2.before_step(0.005, snapshot=snapshot)
        reads["shared"].append(len(calls))
        efforts["shared"].append(shared_stage.control.joint_f.numpy().copy())
        for stage in (private_stage, shared_stage):
            stage.simulation_step_count += 1
        # The snapshot read each distinct world/model/solver array exactly once;
        # only the adapters' own drive outputs (external torque + four outputs)
        # were read privately. The private path paid for both adapters.
        assert snapshot.device_reads >= 20
        assert reads["shared"][-1] == snapshot.device_reads + 2 * 5
        assert reads["private"][-1] == 2 * (snapshot.device_reads + 5)
    for private, shared in zip(efforts["private"], efforts["shared"], strict=True):
        np.testing.assert_array_equal(private, shared)
        assert np.count_nonzero(private) == 28


@pytest.mark.parametrize("case", ["other_stage", "stale_step", "not_a_snapshot"])
def test_snapshot_must_belong_to_this_stage_and_step(runtime, source_root, case):
    import numpy as np
    stage, a, b = cohort_stage(runtime, source_root)
    a.set_targets(np.full(14, 0.1))
    if case == "other_stage":
        other, _, _ = cohort_stage(runtime, source_root)
        snapshot = module().BamHostSnapshot(other)
    elif case == "stale_step":
        snapshot = a.host_snapshot()
        stage.simulation_step_count += 1
    else:
        snapshot = object()
    with pytest.raises(RuntimeError):
        a.before_step(0.005, snapshot=snapshot)
    assert a.telemetry()["last_simulation_step_count"] is None
    assert not np.any(stage.control.joint_f.numpy())


@pytest.mark.parametrize("fault", ["damping", "nan_q", "dof_map"])
def test_snapshot_path_keeps_every_fail_closed_check(runtime, source_root, fault):
    import numpy as np
    stage, a, b = cohort_stage(runtime, source_root)
    for adapter in (a, b):
        adapter.set_targets(np.full(14, 0.1))
    if fault == "damping":
        values = stage.model.joint_damping.numpy()
        values[b.coordinate_indices[1][0]] = 0.5
        stage.model.joint_damping.assign(values)
    elif fault == "nan_q":
        q = stage.state_0.joint_q.numpy()
        q[b.coordinate_indices[0][3]] = np.nan
        stage.state_0.joint_q.assign(q)
    else:
        mapping = stage.solver.mjc_dof_to_newton_dof.numpy()
        mapping[0, b.coordinate_indices[1][0]] = -1
        stage.solver.mjc_dof_to_newton_dof.assign(mapping)
    snapshot = a.host_snapshot()
    if fault == "damping":
        # The first adapter's DOFs are untouched; it still actuates from the same snapshot.
        a.before_step(0.005, snapshot=snapshot)
    with pytest.raises((RuntimeError, ValueError)):
        b.before_step(0.005, snapshot=snapshot)
    assert b.telemetry()["last_simulation_step_count"] is None
