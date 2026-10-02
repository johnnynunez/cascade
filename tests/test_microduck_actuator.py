"""Numeric BAM law, NOT evidence of any engine's friction implementation."""
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import numpy as np
import pytest


def module():
    name = "cascade.control.microduck_actuator"
    assert importlib.util.find_spec(name) is not None, "BAM numeric adapter missing"
    return importlib.import_module(name)


def adapter(**overrides):
    cfg = dict(kp_fw=200., vin=7.4, max_current=None, vin_drop_gain=0., vin_min=6.,
               friction_reference="bam_model")
    cfg.update(overrides)
    return module().BamXL330M6(**cfg)


def evaluate(act, target=0., q=0., dq=0., motor=0., external=0., commanded=0.):
    return act.evaluate(target, q, dq, previous_motor_torque=motor,
                        previous_external_torque=external, previous_commanded_torque=commanded)


def test_zero_speed_retains_solver_friction_and_armature_without_fake_torque():
    out = evaluate(adapter())
    assert float(out.effort_nm) == 0
    assert float(out.frictionloss_nm) > 0
    assert float(out.damping_nm_s_rad) > 0
    assert float(out.armature_kg_m2) > 0
    assert float(out.voltage_v) == 0


def test_battery_history_is_distinct_from_friction_solver_history():
    act = adapter(vin_drop_gain=.2)
    a = evaluate(act, target=1., motor=0., external=.5, commanded=1.)
    b = evaluate(act, target=1., motor=.9, external=.5, commanded=1.)
    assert a.effective_vin_v == b.effective_vin_v == pytest.approx(7.2)
    assert a.effort_nm == b.effort_nm
    assert a.frictionloss_nm != b.frictionloss_nm
    floor = evaluate(act, target=1., commanded=100.)
    assert floor.effective_vin_v == 6.


def test_current_limit_is_pwm_constraint_not_torque_clip():
    act = adapter(max_current=1.75)
    limited = evaluate(act, target=100)
    unlimited = evaluate(adapter(), target=100)
    assert limited.effort_nm < unlimited.effort_nm
    # At unattainable back-EMF the physical voltage cap wins, not a torque clip.
    fast = evaluate(act, target=100., dq=100.)
    assert abs(fast.voltage_v) <= 7.4
    assert abs(fast.effort_nm) > .36601349688984386 * 1.75


@pytest.mark.parametrize("cfg", [dict(vin=0),dict(kp_fw=-1),dict(max_current=0),
    dict(vin_drop_gain=-1),dict(vin_min=0),dict(vin_min=8),dict(vin=np.inf), dict(kp_fw=True)])
def test_invalid_configuration_is_not_silently_corrected(cfg):
    with pytest.raises(ValueError): adapter(**cfg)


def test_configuration_has_no_silent_current_limit_default():
    with pytest.raises(TypeError):
        module().BamXL330M6(kp_fw=200, vin=7.4, vin_drop_gain=0, vin_min=6,
                           friction_reference="bam_model")


@pytest.mark.parametrize("args", [dict(dq=np.nan),dict(target=np.inf),
    dict(target=[1,2],q=[0]),dict(external=np.nan),dict(commanded=np.inf)])
def test_nonfinite_and_broadcasting_are_rejected(args):
    with pytest.raises(ValueError): evaluate(adapter(), **args)


def test_friction_reference_must_be_explicit():
    with pytest.raises(TypeError):
        module().BamXL330M6(kp_fw=200, vin=7.4, max_current=None, vin_drop_gain=0, vin_min=6)


@pytest.mark.parametrize("motor,external", [(1., .5), (1., 1.), (1., -1.), (-1., -.5)])
def test_friction_reference_distinguishes_cpu_and_mjlab_quadratic(motor, external):
    # bam/model.py:172-192 versus bam/mjlab.py:466-480 at the SAME pinned SHA.
    cpu = evaluate(adapter(friction_reference="bam_model"), motor=motor, external=external)
    train = evaluate(adapter(friction_reference="bam_mjlab"), motor=motor, external=external)
    # At zero speed Stribeck is one. CPU excludes these same-sign/equal-magnitude cases.
    coefficient, load = ((.004902565732332559, abs(external))
                         if abs(motor) > abs(external) else (.009972471242139415, abs(motor)))
    assert float(train.frictionloss_nm - cpu.frictionloss_nm) == pytest.approx(coefficient * load**2)
    np.testing.assert_array_equal(cpu.effort_nm, train.effort_nm)
    with pytest.raises(ValueError): adapter(friction_reference="guessed")


def test_matches_actual_pinned_bam_over_error_speed_load_voltage_and_current():
    m = module()
    root_value = os.environ.get("MICRODUCK_BAM_ROOT")
    if not root_value:
        pytest.skip("set MICRODUCK_BAM_ROOT to the pinned, installed BAM source tree")
    root = Path(root_value).resolve()
    import bam.model
    assert Path(bam.model.__file__).resolve().is_relative_to(root)
    admission = json.loads((Path(__file__).resolve().parents[1] / "assets/microduck/manifest.json").read_text())
    assert admission["sources"]["bam"]["revision"] == "62bd8ce12154340be97e06f7f41a0ca8f116d967"
    for row in admission["files"]:
        if row["source"] == "bam":
            assert hashlib.sha256((root / row["source_path"]).read_bytes()).hexdigest() == row["sha256"]
    ref = bam.model.load_model(str(root / "bam/params/xl330/m6.json"))
    # Test the actual effective constructor before explicitly configuring both references.
    assert ref.actuator.max_current == 1.75
    assert ref.q_offset.value != 0
    rng = np.random.default_rng(882)
    target = np.r_[[-100., -1., 0., 1., 100.], rng.normal(0, 2, 251)]
    q = rng.normal(0,.5,256)
    dq = np.r_[[-100., -3., 0., 3., 100.], rng.normal(0,5,251)]
    motor = rng.normal(0,1,256)
    external = rng.normal(0,1,256)
    commanded = rng.normal(0,.03,256)
    for current in (None, 1.75):
        for vin in (6., 7.4, 8.2):
            for drop in (0., .2):
                act = m.BamXL330M6(kp_fw=200, vin=vin, max_current=current,
                                  vin_drop_gain=drop, vin_min=6., friction_reference="bam_model")
                actual = act.evaluate(target,q,dq, previous_motor_torque=motor,
                    previous_external_torque=external, previous_commanded_torque=commanded)
                ref.actuator.kp = 200
                ref.actuator.vin = max(vin - drop * np.abs(commanded).sum(), 6.)
                ref.actuator.max_current = current
                voltage = ref.actuator.compute_control(target,q,dq,.005)
                torque = ref.actuator.compute_torque(voltage,True,q,dq)
                friction, damping = ref.compute_frictions(motor,external,dq)
                for got, expected in [(actual.voltage_v,voltage), (actual.effort_nm,torque),
                    (actual.frictionloss_nm,friction), (actual.damping_nm_s_rad,damping),
                    (actual.armature_kg_m2,ref.actuator.get_extra_inertia())]:
                    np.testing.assert_allclose(got,expected,rtol=1e-13,atol=1e-14)
