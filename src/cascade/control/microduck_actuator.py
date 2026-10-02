# Copyright 2025 Marc Duclusaud & Grégoire Passault
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Modified for CASCADE: standalone validated NumPy law and solver contract.
"""Numeric XL330/M6 BAM-compatible reference / golden validator ONLY.

This is NOT the preferred production actuator integration. Newton should
reuse the native BAM path being audited in IsaacLab PR #8161 rather than
reimplementing DriveBam here. No solver support is claimed by this module.

Reference: Rhoban/bam@62bd8ce12154340be97e06f7f41a0ca8f116d967,
bam/actuator.py, bam/model.py, bam/dynamixel/actuator.py,
bam/params/xl330/m6.json; see assets/microduck/NOTICE.md.

For EACH physical step, a solver integration must:
* Supply measured motor q/dq and held q_target, with explicit joint mapping.
* Supply PREVIOUS applied actuator effort (motor-side friction load), PREVIOUS
  external effort = -bias + constraint - own DOF-friction constraint, and
  PREVIOUS commanded motor effort (battery load). These are not synonyms.
* Apply effort_nm with no residual XML position drive, implicit effort clamp
  or duplicated electrical damping; the returned effort includes back-EMF.
* Install frictionloss_nm as a Coulomb/stiction BUDGET for the constraint
  solver, damping_nm_s_rad as ONLY mechanical viscosity, and armature_kg_m2
  as apparent rotor inertia. Do not turn this budget into sign(dq) torque:
  at rest it must oppose the required stopping effort, not become zero.

No integration, solver updates, contact handling, backlash, delay, domain
randomization or torque-off policy is supplied. There is no state: the caller
owns previous-step values and clears them on reset. Numeric agreement with
BAM does not establish physical equivalence of PhysX, Newton or hardware.
Training source retains the XL330 default max_current=1.75 A; infer_policy
explicitly uses None. Choose/configure it explicitly, never infer the historic
training recipe from a weight file. Calibration values are NOT silently fixed;
q_offset is an identification testbench offset and is NOT added to robot HOME.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

BAM_REVISION = "62bd8ce12154340be97e06f7f41a0ca8f116d967"
M6_PARAMETERS = MappingProxyType({
    "kt": 0.36601349688984386,
    "R": 2.8113923539223227,
    "armature": 0.0018077432831600838,
    "q_offset": 0.0271132870444849,
    "friction_base": 0.004771183165566,
    "friction_stribeck": 0.004676345799486616,
    "load_friction_motor": 0.2667860954283698,
    "load_friction_external": 8.515871897059342e-06,
    "load_friction_motor_stribeck": 1.0722918395099123e-05,
    "load_friction_external_stribeck": 0.08077928978935671,
    "load_friction_motor_quad": 0.009972471242139415,
    "load_friction_external_quad": 0.004902565732332559,
    "dtheta_stribeck": 2.890372094130307,
    "alpha": 8.683259907618984,
    "friction_viscous": 0.005359668274599504,
})


@dataclass(frozen=True)
class BamResult:
    effort_nm: np.ndarray
    voltage_v: np.ndarray
    effective_vin_v: float
    frictionloss_nm: np.ndarray
    damping_nm_s_rad: np.ndarray
    armature_kg_m2: np.ndarray


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not np.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"invalid {name}")
    return value


class BamXL330M6:
    """One battery group, scalar or 1-D joint vectors; no implicit broadcasting.

    All configuration is explicit. friction_reference must be 'bam_model'
    (CPU Model.compute_frictions) or 'bam_mjlab' (training friction convention).
    At the SAME source pin these differ: CPU gates its quadratic term on
    opposite signs and excludes equal magnitudes; mjlab does not. The latter
    convention matches the source of IsaacLab PR #8161, but this module does
    not execute or validate that PR's solver integration or effort clamp.
    vin is the unloaded voltage; vin_drop_gain
    (V/Nm) may be None/0 to disable drop. vin_min may be None (no voltage floor).
    Invalid/nonpositive resulting voltage fails, rather than silently clamping.
    """

    def __init__(self, *, kp_fw, vin, max_current, vin_drop_gain, vin_min, friction_reference):
        if friction_reference not in ("bam_model", "bam_mjlab"):
            raise ValueError("friction_reference must explicitly select bam_model or bam_mjlab")
        self.friction_reference = friction_reference
        self.kp_fw = _number(kp_fw, "kp_fw")
        self.vin = _number(vin, "vin", positive=True)
        self.max_current = None if max_current is None else _number(max_current, "max_current", positive=True)
        self.vin_drop_gain = 0. if vin_drop_gain is None else _number(vin_drop_gain, "vin_drop_gain")
        self.vin_min = None if vin_min is None else _number(vin_min, "vin_min", positive=True)
        if self.vin_min is not None and self.vin_min > self.vin:
            raise ValueError("vin_min must not exceed unloaded vin")

    def evaluate(self, q_target, q, dq, *, previous_motor_torque,
                 previous_external_torque, previous_commanded_torque) -> BamResult:
        """Return motor effort and solver parameters without applying physics.

        previous_motor_torque is measured generalized actuator force from the
        last solve. previous_commanded_torque is last return's effort_nm used
        for voltage drop, not a substitute for that measured force.
        """
        values = [np.asarray(x, dtype=np.float64) for x in (
            q_target, q, dq, previous_motor_torque,
            previous_external_torque, previous_commanded_torque,
        )]
        shape = values[0].shape
        if values[0].ndim > 1 or values[0].size == 0 or any(
            x.shape != shape or not np.isfinite(x).all() for x in values
        ):
            raise ValueError("all motor inputs must have identical scalar/1-D shape and be finite")
        target, position, speed, motor, external, commanded = values
        p = M6_PARAMETERS
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            vin = self.vin - self.vin_drop_gain * float(np.abs(commanded).sum())
            if self.vin_min is not None:
                vin = max(vin, self.vin_min)
            if not np.isfinite(vin) or vin <= 0:
                raise ValueError("nonpositive/nonfinite effective battery voltage")
            gain = (4096 / (2 * np.pi)) / (256 * 885)
            duty = (target - position) * self.kp_fw * gain
            if self.max_current is not None:
                center = p["kt"] * speed / vin
                span = p["R"] * self.max_current / vin
                duty = np.clip(duty, center - span, center + span)
            voltage = vin * np.clip(duty, -1., 1.)
            effort = p["kt"] * voltage / p["R"] - p["kt"]**2 * speed / p["R"]
            stribeck = np.exp(-(np.abs(speed / p["dtheta_stribeck"]) ** p["alpha"]))
            friction = p["friction_base"] + np.abs(
                external * p["load_friction_external"] - motor * p["load_friction_motor"]
            )
            friction += stribeck * p["friction_stribeck"]
            friction += stribeck * np.abs(
                external * p["load_friction_external_stribeck"] - motor * p["load_friction_motor_stribeck"]
            )
            motor_dominates = np.abs(motor) > np.abs(external)
            if self.friction_reference == "bam_model":
                external_dominates = np.abs(external) > np.abs(motor)
                enabled = np.sign(external) != np.sign(motor)
            else:
                # Audited bam/mjlab.py and IsaacLab PR #8161 use this DIFFERENT
                # convention: no sign gating, ties take the back-driven branch.
                external_dominates = ~motor_dominates
                enabled = True
            friction += stribeck * (
                motor_dominates * p["load_friction_external_quad"] * np.abs(external)**2
                + external_dominates * p["load_friction_motor_quad"] * np.abs(motor)**2
            ) * enabled
        if not all(np.isfinite(x).all() for x in (voltage, effort, friction)):
            raise ValueError("motor law overflow")
        return BamResult(
            np.asarray(effort), np.asarray(voltage), vin, np.asarray(friction),
            np.full(shape, p["friction_viscous"]), np.full(shape, p["armature"]),
        )
