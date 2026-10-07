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


# --- shared-scene output check: one device-side finiteness gate, ONE host read per step ---

OUTPUT_NAMES = ("external_torque", "effort", "motor_torque", "effective_vin", "friction_budget")


def output_arrays(adapter):
    """The five drive outputs whose finiteness gates the solver step, by name."""
    return {"external_torque": adapter._drive.external_torque, "effort": adapter._forces,
            "motor_torque": adapter._drive.motor_torque, "effective_vin": adapter._drive.effective_vin,
            "friction_budget": adapter._drive.friction_budget}


def poison(adapter, name):
    """Inject a non-finite value where a native fault would surface: in the solver load the
    bridge gathers (external torque, from the second step on) or in a drive output after
    the pinned ``compute`` ran (the kernels themselves are never modified)."""
    import numpy as np
    if name == "external_torque":
        bias = np.zeros(adapter._solver.mjw_data.qfrc_bias.shape, np.float32)
        bias[0, adapter.coordinate_indices[1][2]] = np.nan
        adapter._solver.mjw_data.qfrc_bias.assign(bias)
        return
    original = adapter._drive.compute

    def compute(*args, **kwargs):
        original(*args, **kwargs)
        output_arrays(adapter)[name].fill_(float("nan"))
    adapter._drive.compute = compute


def excite(stage, step):
    """Per-step joint state so efforts, friction and voltages differ between steps."""
    import numpy as np
    qd = stage.state_0.joint_qd.numpy()
    qd[6:] = np.linspace(-3.0, 3.0, len(qd) - 6) * np.sin(0.7 * step + 0.3)
    stage.state_0.joint_qd.assign(qd)


def test_cohort_output_check_reads_the_device_once_with_identical_outputs(runtime, source_root, monkeypatch):
    import numpy as np
    wp, _ = runtime
    private_stage, a1, b1 = cohort_stage(runtime, source_root)
    shared_stage, a2, b2 = cohort_stage(runtime, source_root)
    for adapter in (a1, b1, a2, b2):
        adapter.set_targets(np.full(14, 0.1))
    check = a2.output_check([a2, b2], labels=["duck0", "duck1"])
    assert check.step_count is None and check.labels == ("duck0", "duck1")
    outputs = [array for adapter in (a2, b2) for array in output_arrays(adapter).values()]
    calls = counting_numpy(monkeypatch, wp)
    for step in range(3):
        for stage in (private_stage, shared_stage):
            excite(stage, step)
        del calls[:]
        snapshot = a1.host_snapshot()
        a1.before_step(0.005, snapshot=snapshot)
        b1.before_step(0.005, snapshot=snapshot)
        private_reads = len(calls)
        del calls[:]
        snapshot = a2.host_snapshot()
        check.reset(shared_stage.simulation_step_count)
        a2.before_step(0.005, snapshot=snapshot, output_check=check)
        b2.before_step(0.005, snapshot=snapshot, output_check=check)
        # Inside before_step: only the snapshot's world reads, none of the ten output arrays.
        assert len(calls) == snapshot.device_reads
        assert not any(any(read is array for array in outputs) for read in calls)
        check.verify(shared_stage.simulation_step_count)
        # The whole cohort's finiteness verdict cost exactly ONE device read: the flag array.
        assert len(calls) == snapshot.device_reads + 1 and calls[-1] is check.flags
        assert private_reads == snapshot.device_reads + 2 * 5
        np.testing.assert_array_equal(private_stage.control.joint_f.numpy(), shared_stage.control.joint_f.numpy())
        np.testing.assert_array_equal(private_stage.solver.mjw_model.dof_frictionloss.numpy(),
                                      shared_stage.solver.mjw_model.dof_frictionloss.numpy())
        assert np.count_nonzero(shared_stage.control.joint_f.numpy()) == 28
        for private, cohort in ((a1, a2), (b1, b2)):
            reference, actual = private.telemetry(), cohort.telemetry()
            for key in OUTPUT_NAMES:
                assert actual[key] == reference[key], key
            assert actual["last_simulation_step_count"] == reference["last_simulation_step_count"] == step
        for stage in (private_stage, shared_stage):
            stage.simulation_step_count += 1
    assert check.device_reads == 3


@pytest.mark.parametrize("name", OUTPUT_NAMES)
def test_cohort_output_check_refuses_the_step_and_names_the_adapter_and_array(runtime, source_root, name):
    import numpy as np
    stage, a, b = cohort_stage(runtime, source_root)
    for adapter in (a, b):
        adapter.set_targets(np.full(14, 0.1))
    check = a.output_check([a, b], labels=["duck0", "duck1"])
    snapshot = a.host_snapshot()
    check.reset(0)
    a.before_step(0.005, snapshot=snapshot, output_check=check)
    b.before_step(0.005, snapshot=snapshot, output_check=check)
    check.verify()
    stage.simulation_step_count = 1
    poison(b, name)
    snapshot = a.host_snapshot()
    check.reset(1)
    a.before_step(0.005, snapshot=snapshot, output_check=check)
    b.before_step(0.005, snapshot=snapshot, output_check=check)  # no host read here: the verdict is verify()'s
    expected = ("nonfinite previous external torque" if name == "external_torque"
                else "nonfinite native BAM output; solver must not step")
    with pytest.raises(ValueError, match=expected) as caught:
        check.verify()
    message = str(caught.value)
    assert "duck1" in message and name in message and "duck0" not in message
    # The healthy adapter is not blamed and no other array of the faulty one is.
    assert not any(other in message for other in OUTPUT_NAMES if other != name)


@pytest.mark.parametrize("case", ["never_reset", "stale_step", "other_step", "unregistered", "other_stage",
                                  "not_a_check", "incomplete", "after_verify"])
def test_output_check_is_refused_when_unbound_stale_foreign_or_incomplete(runtime, source_root, case):
    import numpy as np
    stage, a, b = cohort_stage(runtime, source_root)
    for adapter in (a, b):
        adapter.set_targets(np.full(14, 0.1))
    check = a.output_check([a, b])
    snapshot = a.host_snapshot()
    if case == "never_reset":
        with pytest.raises(RuntimeError):
            check.verify()
        with pytest.raises(RuntimeError):
            a.before_step(0.005, snapshot=snapshot, output_check=check)
    elif case == "stale_step":
        check.reset(0)
        a.before_step(0.005, snapshot=snapshot, output_check=check)
        b.before_step(0.005, snapshot=snapshot, output_check=check)
        stage.simulation_step_count += 1
        with pytest.raises(RuntimeError):
            check.verify()
        with pytest.raises(RuntimeError):
            check.verify(0)
    elif case == "other_step":
        check.reset(5)
        with pytest.raises(RuntimeError):
            a.before_step(0.005, snapshot=snapshot, output_check=check)
    elif case == "unregistered":
        check = a.output_check([a])
        check.reset(0)
        with pytest.raises(RuntimeError):
            b.before_step(0.005, snapshot=snapshot, output_check=check)
        with pytest.raises(ValueError):
            b.output_check([a])  # the builder must be a member of its own cohort
    elif case == "other_stage":
        other, c, d = cohort_stage(runtime, source_root)
        check = c.output_check([c, d])
        check.reset(0)
        with pytest.raises(RuntimeError):
            a.before_step(0.005, snapshot=snapshot, output_check=check)
    elif case == "not_a_check":
        with pytest.raises(RuntimeError):
            a.before_step(0.005, snapshot=snapshot, output_check=object())
    elif case == "incomplete":
        check.reset(0)
        a.before_step(0.005, snapshot=snapshot, output_check=check)
        with pytest.raises(RuntimeError, match="incomplete"):
            check.verify()
    else:
        check.reset(0)
        a.before_step(0.005, snapshot=snapshot, output_check=check)
        b.before_step(0.005, snapshot=snapshot, output_check=check)
        check.verify()
        with pytest.raises(RuntimeError):
            check.mark_external_torque(b, b._drive.external_torque)
    if case in ("never_reset", "other_step", "other_stage", "not_a_check"):
        # Refused before any effort write or cadence bookkeeping.
        assert a.telemetry()["last_simulation_step_count"] is None
        assert not np.any(stage.control.joint_f.numpy())
    if case == "unregistered":
        assert b.telemetry()["last_simulation_step_count"] is None
        assert not np.any(stage.control.joint_f.numpy()[list(b.coordinate_indices[1])])


@pytest.mark.parametrize("name", OUTPUT_NAMES)
def test_private_before_step_still_refuses_nonfinite_outputs_before_any_write(runtime, source_root, name):
    """Regression pin of the single-robot path (no output_check): unchanged reads and wording."""
    import numpy as np
    stage = buffer_stage(runtime)
    adapter = adapter_for(stage, source_root)
    adapter.set_targets(np.full(14, 0.1))
    adapter.before_step(0.005)
    stage.simulation_step_count = 1
    poison(adapter, name)
    effort = stage.control.joint_f.numpy().copy()
    friction = stage.solver.mjw_model.dof_frictionloss.numpy().copy()
    expected = ("nonfinite previous external torque" if name == "external_torque"
                else "nonfinite native BAM output; solver must not step")
    with pytest.raises(ValueError, match=expected):
        adapter.before_step(0.005)
    np.testing.assert_array_equal(stage.control.joint_f.numpy(), effort)
    np.testing.assert_array_equal(stage.solver.mjw_model.dof_frictionloss.numpy(), friction)
    assert adapter.telemetry()["last_simulation_step_count"] == 0


# --- shared-scene cohort: ONE pinned drive/bridge over stacked rows, one launch per stage per step ---

def fleet_stage(runtime, source_root, count, **overrides):
    """``count`` adapters on ONE stage, each owning fourteen disjoint hinges (the twelve-duck shape)."""
    import numpy as np
    stage = buffer_stage(runtime, hinges=14 * count)
    adapters = []
    for row in range(count):
        dofs = np.arange(6 + 14 * row, 6 + 14 * (row + 1), dtype=np.int64)
        adapters.append(module().NewtonBamAdapter(stage, source_root=source_root, sdk_recipe=recipe(),
                                                  q_indices=dofs + 1, dof_indices=dofs, params=params(**overrides)))
    return stage, adapters


def counting_launches(monkeypatch, wp):
    """Record every ``wp.launch`` (by kernel name) and ``wp.copy`` issued through the warp module."""
    launches, copies = [], []
    original_launch, original_copy = wp.launch, wp.copy

    def launch(kernel, *args, **kwargs):
        launches.append(getattr(kernel, "key", str(kernel)).rsplit(".", 1)[-1])
        return original_launch(kernel, *args, **kwargs)

    def copy(*args, **kwargs):
        copies.append(1)
        return original_copy(*args, **kwargs)
    monkeypatch.setattr(wp, "launch", launch)
    monkeypatch.setattr(wp, "copy", copy)
    return launches, copies


def fleet_inputs(rng, count):
    """Random per-row joint state, targets and solver loads: every row's arithmetic differs."""
    import numpy as np
    draw = lambda lo, hi: rng.uniform(lo, hi, (count, 14)).astype(np.float32)  # noqa: E731
    return dict(q=draw(-0.5, 0.5), qd=draw(-4.0, 4.0), targets=draw(-1.0, 1.0),
                bias=draw(-0.6, 0.6), constraint=draw(-0.3, 0.3))


def apply_fleet_inputs(stage, adapters, inputs):
    import numpy as np
    q, qd = stage.state_0.joint_q.numpy(), stage.state_0.joint_qd.numpy()
    mjd = stage.solver.mjw_data
    bias = np.zeros(mjd.qfrc_bias.shape, np.float32)
    constraint = np.zeros(mjd.qfrc_constraint.shape, np.float32)
    for row, adapter in enumerate(adapters):
        qi, di = (list(x) for x in adapter.coordinate_indices)
        q[qi], qd[di] = inputs["q"][row], inputs["qd"][row]
        bias[0, di], constraint[0, di] = inputs["bias"][row], inputs["constraint"][row]
        adapter.set_targets(inputs["targets"][row])
    stage.state_0.joint_q.assign(q)
    stage.state_0.joint_qd.assign(qd)
    mjd.qfrc_bias.assign(bias)
    mjd.qfrc_constraint.assign(constraint)


def poison_cohort(cohort, adapter, name):
    """Inject a non-finite value into ONE member's row where a native fault would surface: in the
    solver load the cohort bridge gathers (external torque) or in that row of a drive output after
    the pinned ``compute`` ran (the member's arrays are live views of the cohort rows)."""
    import numpy as np
    if name == "external_torque":
        bias = np.zeros(adapter._solver.mjw_data.qfrc_bias.shape, np.float32)
        bias[0, adapter.coordinate_indices[1][2]] = np.nan
        adapter._solver.mjw_data.qfrc_bias.assign(bias)
        return
    original = cohort._drive.compute

    def compute(*args, **kwargs):
        original(*args, **kwargs)
        output_arrays(adapter)[name].fill_(float("nan"))
    cohort._drive.compute = compute


def test_twelve_adapter_cohort_launches_one_kernel_per_stage_with_identical_rows(runtime, source_root, monkeypatch):
    """The launch-count claim of this slice, counted (not estimated), and per-row exactness.

    Per-adapter path (slice 1): six ``wp.launch`` per adapter per step (gather, external-torque
    mark, motor, friction, outputs mark, friction publish) and nine ``wp.copy`` (effort scatter +
    eight state copies) -- 72 launches / 108 copies for twelve. Cohort path: one launch per stage
    for the whole fleet (6) and nine copies, every row bitwise equal to its private adapter.
    """
    import json
    import numpy as np
    wp, _ = runtime
    count = 12
    overrides = dict(vin_drop_gain=0.2, max_current=1.75)  # exercise the per-block sag sum and the current limiter
    private_stage, private = fleet_stage(runtime, source_root, count, **overrides)
    shared_stage, members = fleet_stage(runtime, source_root, count, **overrides)
    labels = [f"duck{i}" for i in range(count)]
    private_check = private[0].output_check(private, labels=labels)
    check = members[0].output_check(members, labels=labels)
    cohort = members[0].cohort(members, labels=labels, output_check=check)  # paired at bind time, like the owner
    assert isinstance(cohort, module().BamCohort)
    assert cohort.adapters == tuple(members) and cohort.labels == tuple(labels) and cohort.rows == count
    rng = np.random.default_rng(20261007)
    launches, copies = counting_launches(monkeypatch, wp)
    reads = counting_numpy(monkeypatch, wp)
    receipt = {}
    stages = ("_gather_external_torque_kernel", "mark_nonfinite_rows", "_bam_motor_kernel", "_bam_friction_kernel",
              "mark_nonfinite_outputs_rows", "_publish_dof_friction_kernel")
    for step in range(4):
        inputs = fleet_inputs(rng, count)
        for stage, adapters in ((private_stage, private), (shared_stage, members)):
            apply_fleet_inputs(stage, adapters, inputs)
        del launches[:], copies[:], reads[:]
        snapshot = private[0].host_snapshot()
        private_check.reset(step)
        for adapter in private:
            adapter.before_step(0.005, snapshot=snapshot, output_check=private_check)
        private_check.verify(step)
        before = dict(launch=len(launches), copy=len(copies), reads=len(reads))
        del launches[:], copies[:], reads[:]
        snapshot = members[0].host_snapshot()
        check.reset(step)
        cohort.before_step(0.005, snapshot=snapshot, output_check=check)
        # Inside before_step: only the snapshot's world reads, none of the stacked output arrays.
        assert len(reads) == snapshot.device_reads
        check.verify(step)
        after = dict(launch=len(launches), copy=len(copies), reads=len(reads))
        assert after["reads"] == snapshot.device_reads + 1 and reads[-1] is check.flags
        gather = 0 if step == 0 else 1  # the first step after reset zeroes the load instead of gathering it
        assert before == dict(launch=count * (5 + gather), copy=count * 9, reads=snapshot.device_reads + 1)
        assert after == dict(launch=5 + gather, copy=9, reads=snapshot.device_reads + 1)
        # Exactly one launch per stage, in stage order, for the whole fleet.
        assert [name.rsplit("__", 1)[-1] for name in launches] == [
            s for s in stages if gather or s != "_gather_external_torque_kernel"]
        np.testing.assert_array_equal(private_stage.control.joint_f.numpy(), shared_stage.control.joint_f.numpy())
        np.testing.assert_array_equal(private_stage.solver.mjw_model.dof_frictionloss.numpy(),
                                      shared_stage.solver.mjw_model.dof_frictionloss.numpy())
        assert np.count_nonzero(shared_stage.control.joint_f.numpy()) == 14 * count
        for reference, member in zip(private, members):
            expected, actual = reference.telemetry(), member.telemetry()
            for key in OUTPUT_NAMES + ("targets", "delay_lag", "delay_fill"):
                np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-6, err_msg=f"{key} step={step}")
                assert actual[key] == expected[key], key  # the same pinned kernel on the same bytes: bitwise
            for key in ("last_simulation_step_count", "steps_since_reset", "armed", "reset_count"):
                assert actual[key] == expected[key], key
            assert actual["last_simulation_step_count"] == step
        receipt[step] = dict(per_adapter=before, cohort=after)
        for stage in (private_stage, shared_stage):
            stage.simulation_step_count += 1
    assert check.device_reads == private_check.device_reads == 4
    print("COHORT_LAUNCH_RECEIPT", json.dumps(dict(adapters=count, first_step=receipt[0], steady_state=receipt[1],
          counted="wp.launch/wp.copy monkeypatched; per step; private = slice-1 path with BamOutputCheck",
          provenance="synthetic_random_inputs_original_Warp_kernels_not_trajectory")))


@pytest.mark.parametrize("case", ["delay", "phase_period", "mixed_shared", "mixed_dt", "overlap", "duplicate",
                                  "other_stage", "stranger", "already_registered", "labels", "not_member", "empty",
                                  "check_subset", "check_other_stage", "not_a_check"])
def test_cohort_refuses_inexact_or_foreign_memberships(runtime, source_root, case):
    """A stacked drive is used only where it reproduces every private row exactly; everything else fails closed."""
    import numpy as np
    Cohort = module().BamCohort
    stage, (a, b) = fleet_stage(runtime, source_root, 2)
    if case in ("check_subset", "check_other_stage", "not_a_check"):
        # The owner's check is paired at bind time: a member it does not register, a check of
        # another stage or a non-check refuse the cohort before any row is re-pointed.
        if case == "check_subset":
            check = a.output_check([a])
        elif case == "check_other_stage":
            _, others = fleet_stage(runtime, source_root, 2)
            check = others[0].output_check(others)
        else:
            check = object()
        with pytest.raises((RuntimeError, ValueError)):
            a.cohort([a, b], output_check=check)
    elif case in ("delay", "phase_period"):
        # The pinned kernels seed the lag/phase draws by DOF block, so row k of a stacked drive cannot
        # reproduce a private drive's stream: the builder declines (None) and the class refuses.
        overrides = dict(min_delay=3, max_delay=6) if case == "delay" else dict(delay_update_period=4)
        stage, (a, b) = fleet_stage(runtime, source_root, 2, **overrides)
        assert a.cohort([a, b]) is None
        assert "delay" in Cohort.unsupported([a, b])
        with pytest.raises(ValueError, match="delay"):
            Cohort([a, b])
        assert a._cohort is None and b._cohort is None
    elif case in ("mixed_shared", "mixed_dt"):
        dofs = np.asarray(b.coordinate_indices[1])
        other = module().NewtonBamAdapter(stage, source_root=source_root, sdk_recipe=recipe(), q_indices=dofs + 1,
                                          dof_indices=dofs, params=params(**({"vin_min": 5.0} if case == "mixed_shared"
                                                                              else {"physics_dt": 0.01})))
        assert a.cohort([a, other]) is None
        assert ("vin_min" if case == "mixed_shared" else "physics_dt") in Cohort.unsupported([a, other])
        with pytest.raises(ValueError):
            Cohort([a, other])
    elif case == "overlap":
        dofs = np.asarray(b.coordinate_indices[1])
        other = module().NewtonBamAdapter(stage, source_root=source_root, sdk_recipe=recipe(), q_indices=dofs + 1,
                                          dof_indices=dofs, params=params())
        assert Cohort.unsupported([a, b, other]) is None
        with pytest.raises(ValueError, match="disjoint"):
            a.cohort([a, b, other])
    elif case == "duplicate":
        with pytest.raises(ValueError, match="twice"):
            a.cohort([a, b, a])
    elif case == "other_stage":
        _, (c, _) = fleet_stage(runtime, source_root, 2)
        with pytest.raises(ValueError, match="stage"):
            a.cohort([a, c])
    elif case == "stranger":
        with pytest.raises(ValueError):
            a.cohort([a, object()])
    elif case == "already_registered":
        assert isinstance(a.cohort([a, b]), Cohort)
        with pytest.raises(ValueError, match="already"):
            Cohort([a])
    elif case == "labels":
        with pytest.raises(ValueError, match="label"):
            a.cohort([a, b], labels=["same", "same"])
    elif case == "not_member":
        with pytest.raises(ValueError, match="member"):
            b.cohort([a])
    else:
        with pytest.raises(ValueError):
            Cohort([])
    if case != "already_registered":
        # Nothing was re-pointed or written on refusal: both adapters still actuate privately.
        for adapter in (a, b):
            assert adapter._cohort is None
            adapter.set_targets(np.full(14, 0.1))
            adapter.before_step(0.005)
        assert np.count_nonzero(stage.control.joint_f.numpy()) == 28


@pytest.mark.parametrize("fault", ["unarmed", "damping", "nan_q", "dof_map", "cadence", "dt", "friction_tuning"])
def test_cohort_before_step_keeps_every_member_check_and_writes_nothing_on_refusal(runtime, source_root, fault):
    import numpy as np
    stage, adapters = fleet_stage(runtime, source_root, 3)
    cohort = adapters[0].cohort(adapters, labels=["d0", "d1", "d2"])
    for adapter in adapters[: 2 if fault == "unarmed" else 3]:
        adapter.set_targets(np.full(14, 0.1))
    dt = 0.005
    if fault == "damping":
        values = stage.model.joint_damping.numpy()
        values[adapters[1].coordinate_indices[1][0]] = 0.5
        stage.model.joint_damping.assign(values)
    elif fault == "nan_q":
        q = stage.state_0.joint_q.numpy()
        q[adapters[2].coordinate_indices[0][3]] = np.nan
        stage.state_0.joint_q.assign(q)
    elif fault == "dof_map":
        mapping = stage.solver.mjc_dof_to_newton_dof.numpy()
        mapping[0, adapters[1].coordinate_indices[1][0]] = -1
        stage.solver.mjc_dof_to_newton_dof.assign(mapping)
    elif fault == "cadence":
        cohort.before_step(dt)
        stage.simulation_step_count += 2
    elif fault == "dt":
        dt = 0.01
    elif fault == "friction_tuning":
        values = stage.model.mujoco.solreffriction.numpy()
        values[adapters[2].coordinate_indices[1][5]] = (0.03, 1.0)
        stage.model.mujoco.solreffriction.assign(values)
        stage.solver.notify_model_changed(runtime[1].ModelFlags.JOINT_DOF_PROPERTIES)
    effort = stage.control.joint_f.numpy().copy()
    friction = stage.solver.mjw_model.dof_frictionloss.numpy().copy()
    steps = [adapter.telemetry()["last_simulation_step_count"] for adapter in adapters]
    with pytest.raises((RuntimeError, ValueError)):
        cohort.before_step(dt)
    # One member's refusal refuses the whole cohort before any device write or cadence bookkeeping,
    # including for the members whose own checks passed.
    np.testing.assert_array_equal(stage.control.joint_f.numpy(), effort)
    np.testing.assert_array_equal(stage.solver.mjw_model.dof_frictionloss.numpy(), friction)
    assert [adapter.telemetry()["last_simulation_step_count"] for adapter in adapters] == steps
    assert steps == ([0] * 3 if fault == "cadence" else [None] * 3)


@pytest.mark.parametrize("case", ["never_reset", "other_step", "stale_step", "unregistered", "other_stage", "not_a_check",
                                  "incomplete_external", "incomplete_outputs", "after_verify"])
def test_cohort_output_check_is_refused_when_unbound_stale_foreign_or_incomplete(runtime, source_root, case, monkeypatch):
    """The slice-1 refusals hold on the cohort path, and skipping ONE stage's cohort mark fails verify()."""
    import numpy as np
    stage, adapters = fleet_stage(runtime, source_root, 2)
    cohort = adapters[0].cohort(adapters, labels=["duck0", "duck1"])
    for adapter in adapters:
        adapter.set_targets(np.full(14, 0.1))
    check = adapters[0].output_check(adapters, labels=["duck0", "duck1"])
    snapshot = adapters[0].host_snapshot()
    if case == "never_reset":
        with pytest.raises(RuntimeError):
            check.verify()
        with pytest.raises(RuntimeError):
            cohort.before_step(0.005, snapshot=snapshot, output_check=check)
    elif case == "other_step":
        check.reset(5)
        with pytest.raises(RuntimeError):
            cohort.before_step(0.005, snapshot=snapshot, output_check=check)
    elif case == "stale_step":
        check.reset(0)
        cohort.before_step(0.005, snapshot=snapshot, output_check=check)
        stage.simulation_step_count += 1
        with pytest.raises(RuntimeError):
            check.verify()
        with pytest.raises(RuntimeError):
            check.verify(0)
    elif case == "unregistered":
        check = adapters[0].output_check([adapters[0]])
        check.reset(0)
        with pytest.raises(RuntimeError):
            cohort.before_step(0.005, snapshot=snapshot, output_check=check)
    elif case == "other_stage":
        _, others = fleet_stage(runtime, source_root, 2)
        check = others[0].output_check(others)
        check.reset(0)
        with pytest.raises(RuntimeError):
            cohort.before_step(0.005, snapshot=snapshot, output_check=check)
    elif case == "not_a_check":
        with pytest.raises(RuntimeError):
            cohort.before_step(0.005, snapshot=snapshot, output_check=object())
    elif case in ("incomplete_external", "incomplete_outputs"):
        name = "mark_cohort_external_torque" if case == "incomplete_external" else "mark_cohort_outputs"
        monkeypatch.setattr(check, name, lambda *args, **kwargs: None)
        check.reset(0)
        cohort.before_step(0.005, snapshot=snapshot, output_check=check)
        with pytest.raises(RuntimeError, match="incomplete"):
            check.verify()
    else:
        check.reset(0)
        cohort.before_step(0.005, snapshot=snapshot, output_check=check)
        check.verify()
        with pytest.raises(RuntimeError):
            check.mark_cohort_external_torque(cohort, cohort._drive.external_torque)
    if case in ("never_reset", "other_step", "unregistered", "other_stage", "not_a_check"):
        # Refused before any effort write or cadence bookkeeping, for every member.
        assert all(adapter.telemetry()["last_simulation_step_count"] is None for adapter in adapters)
        assert not np.any(stage.control.joint_f.numpy())


@pytest.mark.parametrize("name", OUTPUT_NAMES)
def test_cohort_output_check_names_the_faulty_row_and_array(runtime, source_root, name):
    import numpy as np
    stage, adapters = fleet_stage(runtime, source_root, 3)
    labels = ["d0", "d1", "d2"]
    cohort = adapters[0].cohort(adapters, labels=labels)
    check = adapters[0].output_check(adapters, labels=labels)
    for adapter in adapters:
        adapter.set_targets(np.full(14, 0.1))
    check.reset(0)
    cohort.before_step(0.005, snapshot=adapters[0].host_snapshot(), output_check=check)
    check.verify()
    stage.simulation_step_count = 1
    poison_cohort(cohort, adapters[1], name)
    check.reset(1)
    cohort.before_step(0.005, snapshot=adapters[0].host_snapshot(), output_check=check)  # no host read: the verdict is verify()'s
    expected = ("nonfinite previous external torque" if name == "external_torque"
                else "nonfinite native BAM output; solver must not step")
    with pytest.raises(ValueError, match=expected) as caught:
        check.verify()
    message = str(caught.value)
    assert "d1" in message and name in message and "d0" not in message and "d2" not in message
    assert not any(other in message for other in OUTPUT_NAMES if other != name)


@pytest.mark.parametrize("name", OUTPUT_NAMES)
def test_cohort_without_output_check_refuses_nonfinite_rows_before_any_write(runtime, source_root, name):
    """Without a cohort check the cohort reads its five stacked outputs itself (five syncs for the
    fleet) and refuses before publish/scatter/update, naming the member -- like the private path."""
    import numpy as np
    stage, adapters = fleet_stage(runtime, source_root, 2)
    cohort = adapters[0].cohort(adapters, labels=["duck0", "duck1"])
    for adapter in adapters:
        adapter.set_targets(np.full(14, 0.1))
    cohort.before_step(0.005)
    stage.simulation_step_count = 1
    poison_cohort(cohort, adapters[1], name)
    effort = stage.control.joint_f.numpy().copy()
    friction = stage.solver.mjw_model.dof_frictionloss.numpy().copy()
    expected = ("nonfinite previous external torque" if name == "external_torque"
                else "nonfinite native BAM output; solver must not step")
    with pytest.raises(ValueError, match=expected) as caught:
        cohort.before_step(0.005)
    assert "duck1" in str(caught.value) and "duck0" not in str(caught.value)
    np.testing.assert_array_equal(stage.control.joint_f.numpy(), effort)
    np.testing.assert_array_equal(stage.solver.mjw_model.dof_frictionloss.numpy(), friction)
    assert all(adapter.telemetry()["last_simulation_step_count"] == 0 for adapter in adapters)


def test_cohort_member_refuses_private_actuation_and_resets_only_its_row(runtime, source_root):
    import numpy as np
    stage, adapters = fleet_stage(runtime, source_root, 3)
    cohort = adapters[0].cohort(adapters, labels=["d0", "d1", "d2"])
    rng = np.random.default_rng(7)
    for adapter in adapters:
        adapter.set_targets(np.full(14, 0.1))
    with pytest.raises(RuntimeError, match="cohort"):
        adapters[1].before_step(0.005)
    assert all(adapter.telemetry()["last_simulation_step_count"] is None for adapter in adapters)
    for step in range(3):
        stage.simulation_step_count = step
        apply_fleet_inputs(stage, adapters, fleet_inputs(rng, 3))
        cohort.before_step(0.005)
    others_before = [adapters[i].telemetry() for i in (0, 2)]
    effort = stage.control.joint_f.numpy().copy()
    friction = stage.solver.mjw_model.dof_frictionloss.numpy().copy()
    own = list(adapters[1].coordinate_indices[1])
    rest = np.setdiff1d(np.arange(len(effort)), own)
    adapters[1].reset()
    reset = adapters[1].telemetry()
    assert reset["steps_since_reset"] == 0 and reset["armed"] is False and reset["reset_count"] == 1
    for key in OUTPUT_NAMES + ("targets", "delay_fill"):
        np.testing.assert_array_equal(reset[key], np.zeros(14))
    assert [adapters[i].telemetry() for i in (0, 2)] == others_before  # the other rows are untouched
    np.testing.assert_array_equal(stage.control.joint_f.numpy()[own], 0.0)
    np.testing.assert_array_equal(stage.control.joint_f.numpy()[rest], effort[rest])
    np.testing.assert_array_equal(stage.solver.mjw_model.dof_frictionloss.numpy()[0, own], 0.0)
    np.testing.assert_array_equal(stage.solver.mjw_model.dof_frictionloss.numpy()[0, rest], friction[0, rest])
    stage.simulation_step_count = 3
    with pytest.raises(RuntimeError, match="targets"):  # an unarmed member refuses the whole cohort step
        cohort.before_step(0.005)
    assert [adapter.telemetry()["last_simulation_step_count"] for adapter in adapters] == [2, None, 2]
    # Re-arm from rest so the sign of the new effort follows the new (negative) targets alone.
    q, qd = stage.state_0.joint_q.numpy(), stage.state_0.joint_qd.numpy()
    q[list(adapters[1].coordinate_indices[0])] = 0.0
    qd[list(adapters[1].coordinate_indices[1])] = 0.0
    stage.state_0.joint_q.assign(q)
    stage.state_0.joint_qd.assign(qd)
    adapters[1].set_targets(np.full(14, -0.1))
    cohort.before_step(0.005)
    after = [adapter.telemetry() for adapter in adapters]
    # The reset row skips the stale load once (like a private adapter); the others gathered theirs.
    np.testing.assert_array_equal(after[1]["external_torque"], np.zeros(14))
    assert np.any(np.asarray(after[0]["external_torque"]) != 0) and np.any(np.asarray(after[2]["external_torque"]) != 0)
    assert np.all(np.asarray(after[1]["effort"]) < 0)  # never replays the old positive targets
    assert [t["steps_since_reset"] for t in after] == [4, 1, 4]
    assert [t["last_simulation_step_count"] for t in after] == [3, 3, 3]


def test_cohort_registration_continues_the_members_private_history_exactly(runtime, source_root):
    """Rows are seeded from the adapters' own arrays at registration: the cohort continues any history."""
    import numpy as np
    rng = np.random.default_rng(11)
    private_stage, private = fleet_stage(runtime, source_root, 3, vin_drop_gain=0.2)
    shared_stage, members = fleet_stage(runtime, source_root, 3, vin_drop_gain=0.2)
    cohort = None
    for step in range(6):
        inputs = fleet_inputs(rng, 3)
        for stage, adapters in ((private_stage, private), (shared_stage, members)):
            stage.simulation_step_count = step
            apply_fleet_inputs(stage, adapters, inputs)
        if step == 3:
            cohort = members[0].cohort(members)
        for adapter in private:
            adapter.before_step(0.005)
        if cohort is None:
            for adapter in members:
                adapter.before_step(0.005)
        else:
            cohort.before_step(0.005)
        np.testing.assert_array_equal(private_stage.control.joint_f.numpy(), shared_stage.control.joint_f.numpy())
        np.testing.assert_array_equal(private_stage.solver.mjw_model.dof_frictionloss.numpy(),
                                      shared_stage.solver.mjw_model.dof_frictionloss.numpy())
        for reference, member in zip(private, members):
            assert member.telemetry() == reference.telemetry()
