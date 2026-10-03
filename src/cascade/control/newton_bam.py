"""Thin, offline binding to the *original* IsaacLab #8161 BAM Warp drive.

No IsaacLab package import, copied motor kernels, physics step, or pose writes.
Newton/Warp are optional and imported only when the explicit loader is called.
The external code's BSD-3-Clause notice and provenance are in
``assets/microduck/newton-bam.json``. Only the three compiled-in hashes may run.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.machinery
import importlib.util
import linecache
from pathlib import Path
import re
import sys
import threading
from types import MappingProxyType, ModuleType
from typing import Mapping


REVISION = "28aa1fca5843208ff9a67935695a4d5376e44d50"
SOURCE_SHA256 = MappingProxyType({
    "source/isaaclab/isaaclab/actuators/newton/bam_component.py":
        "0313338887f32fa36040b63768007bfb15ce050921c841b0f0a0f82ec8e7ab90",
    "source/isaaclab/isaaclab/actuators/newton/bam_kernels.py":
        "cf0043556f299ee3eeaca81e98a76ecaf658137444f86a48ddce1ba2c0c1a089",
    "source/isaaclab_newton/isaaclab_newton/physics/mjwarp_actuator_bridge.py":
        "ea935557a274aedf280c915f9cf24fb760e1fd75e8616327aa77b4b465a8add7",
})


@dataclass(frozen=True)
class NativeBamSources:
    DriveBam: type
    MjWarpActuatorBridge: type
    revision: str
    sha256: Mapping[str, str]
    source_root: str


_loaded: NativeBamSources | None = None
_load_lock = threading.RLock()


def _runtime(*, sdk_recipe=None):
    try:
        import newton
        import warp as wp
    except ImportError as exc:
        raise RuntimeError("native BAM requires Newton >=1.6 and Warp; no PD fallback") from exc
    version = str(getattr(newton, "__version__", ""))
    if sdk_recipe is not None:
        from cascade.sim.microduck_sdk import verify_runtime_recipe
        verify_runtime_recipe(sdk_recipe, newton_version=version)
        return wp, newton
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:\+[^ ]+)?", version)
    if match is None or tuple(map(int, match.groups())) < (1, 6, 0):
        raise RuntimeError(f"native BAM requires stable Newton >=1.6; found {version!r}")
    return wp, newton


def load_pinned_bam(source_root, *, sdk_recipe=None) -> NativeBamSources:
    """Verify every file before executing only allowlisted source bytes.

    ``source_root`` is an external checkout/archive root containing ``source/``.
    No network, external __init__, cached .pyc, or caller-supplied manifest is
    executed. Warp's inspect-based JIT sees these same verified source bytes.
    The first admitted tree provides the process-wide native registration;
    subsequent calls still verify their requested tree, then reuse that class.
    """
    global _loaded
    _runtime(sdk_recipe=sdk_recipe)
    root = Path(source_root).expanduser().resolve(strict=True)
    sources = {}
    for relative, expected in SOURCE_SHA256.items():
        path = root / relative
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"BAM source SHA-256 mismatch: {relative}")
        sources[Path(relative).stem] = (path, data)
    with _load_lock:
        if _loaded is not None:
            return _loaded
        name = f"_cascade_native_bam_{REVISION}"
        package = ModuleType(name)
        package.__path__ = [str(sources["bam_component"][0].parent)]
        package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = package
        modules = {}
        try:
            for stem in ("bam_kernels", "bam_component", "mjwarp_actuator_bridge"):
                path, data = sources[stem]
                qualified = f"{name}.{stem}"
                spec = importlib.util.spec_from_file_location(qualified, path)
                native = importlib.util.module_from_spec(spec)
                sys.modules[qualified] = native
                # mtime=None makes inspect/linecache use exactly the admitted text.
                linecache.cache[str(path)] = (len(data), None, data.decode("utf-8").splitlines(True), str(path))
                exec(compile(data, str(path), "exec"), native.__dict__)
                setattr(package, stem, native)
                modules[stem] = native
        except BaseException:
            for stem in sources:
                sys.modules.pop(f"{name}.{stem}", None)
            sys.modules.pop(name, None)
            raise
        _loaded = NativeBamSources(
            modules["bam_component"].DriveBam,
            modules["mjwarp_actuator_bridge"].MjWarpActuatorBridge,
            REVISION, SOURCE_SHA256, str(root),
        )
        return _loaded


def _validated_params(params):
    import numbers
    import numpy as np

    keys = {"kp_fw", "vin", "max_current", "vin_drop_gain", "vin_min", "min_delay", "max_delay",
            "delay_hold_prob", "delay_update_period", "delay_seed", "max_effort", "joint_effort_limit",
            "stiff_frictionloss", "physics_dt"}
    if not isinstance(params, dict) or set(params) != keys:
        got = set(params) if isinstance(params, dict) else set()
        raise ValueError(f"explicit BAM params required; missing={sorted(keys-got)}, unknown={sorted(got-keys)}")
    p = dict(params)
    integers = {"min_delay", "max_delay", "delay_update_period", "delay_seed"}
    positive = {"vin", "vin_min", "max_current", "max_effort", "joint_effort_limit", "physics_dt"}
    for key, value in p.items():
        if key == "stiff_frictionloss":
            if type(value) is not bool:
                raise ValueError("stiff_frictionloss must be an explicit bool")
        elif key == "max_current" and value is None:
            continue
        elif key in integers:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or not 0 <= value < 2**31 - 1:
                raise ValueError(f"{key} must be a nonnegative int32")
            p[key] = int(value)
        else:
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
                    or not np.isfinite(value) or value > np.finfo(np.float32).max
                    or (value <= 0 if key in positive else value < 0)):
                raise ValueError(f"{key} must be a finite {'positive' if key in positive else 'nonnegative'} number")
            if key in positive and np.float32(value) <= 0:
                raise ValueError(f"{key} must remain positive in native float32 (no underflow)")
            p[key] = float(value)
    if p["vin_min"] > p["vin"]:
        raise ValueError("vin_min must not exceed vin")
    if not p["min_delay"] <= p["max_delay"] <= 1024:
        raise ValueError("min_delay <= max_delay <= 1024 is required (bounded history allocation)")
    if p["delay_hold_prob"] > 1:
        raise ValueError("delay_hold_prob must be in [0, 1]")
    if p["max_effort"] > p["joint_effort_limit"]:
        raise ValueError("max_effort must not exceed explicit joint_effort_limit")
    return p


def _indices(values, name, limit):
    import numpy as np

    result = np.asarray(values)
    if (result.shape != (14,) or result.dtype.kind not in "iu" or len(np.unique(result)) != 14
            or np.any(result < 0) or np.any(result >= limit)):
        raise ValueError(f"{name} must be 14 unique, in-range integer indices")
    return result.astype(np.int64, copy=True)


class NewtonBamAdapter:
    """One explicit 14-DOF M6 battery group; the caller alone steps physics.

    Call on the simulation thread after model-property synchronization and
    immediately before one solver step. Targets may be held between policy
    ticks. A new adapter is required after a model/solver rebuild.
    """

    def __init__(self, stage, *, source_root, q_indices, dof_indices, params: dict, sdk_recipe=None):
        import numpy as np
        from .microduck_actuator import M6_PARAMETERS

        self._wp, self._newton = _runtime(sdk_recipe=sdk_recipe)
        wp = self._wp
        self._sources = load_pinned_bam(source_root, sdk_recipe=sdk_recipe)
        self._params = _validated_params(params)
        self._stage = stage
        self._model, self._solver = getattr(stage, "model", None), getattr(stage, "solver", None)
        if not isinstance(self._model, self._newton.Model):
            raise RuntimeError("native BAM requires a Newton model; PhysX is unsupported")
        self._device = self._model.device
        self._dofs = _indices(dof_indices, "dof_indices", self._model.joint_dof_count)
        self._qs = _indices(q_indices, "q_indices", self._model.joint_coord_count)
        self._validate_binding()
        self._dof_indices = wp.array(self._dofs, dtype=wp.uint32, device=self._device)
        self._q_indices = wp.array(self._qs, dtype=wp.uint32, device=self._device)
        self._scatter_indices = wp.array(self._dofs, dtype=wp.int32, device=self._device)
        self._target_indices = wp.array(np.arange(14), dtype=wp.uint32, device=self._device)
        self._targets = wp.zeros(14, device=self._device)
        self._forces = wp.zeros(14, device=self._device)
        p = self._params
        coefficients = {k: v for k, v in M6_PARAMETERS.items()
                        if k not in ("R", "q_offset", "armature", "friction_viscous")}
        coefficients.update(resistance=M6_PARAMETERS["R"], error_gain=(4096 / (2 * np.pi)) / (256 * 885),
                            max_pwm=1.0, stribeck=1, load_dependent=1, quadratic=1,
                            kp_scale=1.0, kd_scale=1.0, friction_scale=1.0,
                            sag_gain=p["vin_drop_gain"], max_current=0.0 if p["max_current"] is None else p["max_current"])
        for key in ("kp_fw", "vin", "vin_min", "min_delay", "max_delay", "delay_hold_prob",
                    "delay_update_period", "delay_seed", "max_effort"):
            coefficients[key] = p[key]
        Drive = self._sources.DriveBam
        resolved = Drive.resolve_arguments(coefficients)
        self._drive = Drive(**{k: v if k in Drive.SHARED_PARAMS else wp.full(14, v, device=self._device)
                              for k, v in resolved.items()})
        self._drive.set_env_dof_stride(14)
        self._drive.finalize(self._device, 14)
        self._state = self._drive.state(14, self._device)
        self._drive.external_torque = wp.zeros(14, device=self._device)
        model = self._model

        class BoundModelBridge(self._sources.MjWarpActuatorBridge):
            # The ONLY adaptation to the upstream bridge: no NewtonManager import.
            def _model_friction_solver_attributes(self):
                return model.mujoco.solreffriction, model.mujoco.solimpfriction

        self._bridge = BoundModelBridge(self._solver, self._dof_indices, model.joint_dof_count, self._device)
        for field, coefficient in (("joint_damping", "friction_viscous"), ("joint_armature", "armature")):
            self._scatter(getattr(model, field), wp.full(14, M6_PARAMETERS[coefficient], device=self._device))
        if p["stiff_frictionloss"]:
            self._bridge.stiffen_friction_constraint()
        self._solver.notify_model_changed(self._newton.ModelFlags.JOINT_DOF_PROPERTIES)
        self._check_model_sync()
        self._bound_map = self._solver.mjc_dof_to_newton_dof.numpy().copy()
        self._bound_friction = {field: getattr(model.mujoco, field).numpy()[self._dofs].copy()
                                for field in ("solreffriction", "solimpfriction")}
        self._bound_channels = [(self._solver, name, getattr(self._solver, name))
                                for name in ("mjw_model", "mjw_data", "mjc_dof_to_newton_dof")]
        self._armed = False
        self._last_step = None
        self._steps_since_reset = 0
        self._skip_external_once = True

    def _array(self, obj, name, dtype, shape):
        wp = self._wp
        value = getattr(obj, name, None)
        if (not isinstance(value, wp.array) or not wp.types.types_equal(value.dtype, dtype)
                or value.shape != shape or value.device != self._device or not value.is_contiguous):
            raise RuntimeError(f"{name}: expected contiguous, expanded Warp array shape={shape}, dtype={dtype}, device={self._device}")
        return value

    def _step_count(self):
        import numbers
        count = getattr(self._stage, "simulation_step_count", None)
        if isinstance(count, bool) or not isinstance(count, numbers.Integral) or count < 0:
            raise RuntimeError("stage.simulation_step_count must be a nonnegative integer")
        return int(count)

    def _validate_binding(self):
        import numpy as np
        from .microduck_actuator import M6_PARAMETERS

        wp, model, solver = self._wp, self._model, self._solver
        if getattr(getattr(self._stage, "cfg", None), "num_substeps", None) != 1:
            raise RuntimeError("native BAM requires exactly one physics substep")
        self._step_count()
        if getattr(solver, "use_mujoco_cpu", None) is not False:
            raise RuntimeError("native BAM requires MJWarp, not MuJoCo C/PhysX/another solver")
        if getattr(solver, "model", None) is not model:
            raise RuntimeError("solver.model must be the stage Newton model")
        if not callable(getattr(solver, "notify_model_changed", None)):
            raise RuntimeError("solver.notify_model_changed is required")
        mjm, mjd = getattr(solver, "mjw_model", None), getattr(solver, "mjw_data", None)
        if mjm is None or mjd is None:
            raise RuntimeError("solver must expose MJWarp model and data")
        import mujoco
        flags = getattr(getattr(mjm, "opt", None), "disableflags", None)
        if not isinstance(flags, (int, np.integer)):
            raise RuntimeError("MJWarp opt.disableflags capability is required")
        for name in ("mjDSBL_CONSTRAINT", "mjDSBL_FRICTIONLOSS", "mjDSBL_DAMPER"):
            bit = getattr(mujoco.mjtDisableBit, name, None)
            if bit is None or flags & int(bit):
                raise RuntimeError(f"BAM solver channel disabled or unavailable: {name}")
        # Deliberately narrow: no second actuator pipeline in this stage.
        if getattr(mjm, "nu", None) != 0 or model.actuators:
            raise RuntimeError("native BAM requires no other MJWarp/Newton actuator pipeline")
        n = model.joint_dof_count
        qs = self._array(model, "joint_q_start", wp.int32, (model.joint_count + 1,)).numpy()
        ds = self._array(model, "joint_qd_start", wp.int32, (model.joint_count + 1,)).numpy()
        types = self._array(model, "joint_type", wp.int32, (model.joint_count,)).numpy()
        for q, dof in zip(self._qs, self._dofs):
            joints = np.flatnonzero(ds[:-1] == dof)
            if (len(joints) != 1 or types[joints[0]] != int(self._newton.JointType.REVOLUTE)
                    or qs[joints[0]] != q or ds[joints[0]+1] - dof != 1 or qs[joints[0]+1] - q != 1):
                raise ValueError("q_indices/dof_indices must identify matching scalar revolute joints")
        for field, dtype in (("joint_target_mode", wp.int32), ("joint_target_ke", wp.float32),
                             ("joint_target_kd", wp.float32)):
            if np.any(self._array(model, field, dtype, (n,)).numpy()[self._dofs] != 0):
                raise RuntimeError(f"residual drive in {field}; disable it before binding BAM")
        for field, coefficient in (("joint_damping", "friction_viscous"), ("joint_armature", "armature")):
            values = self._array(model, field, wp.float32, (n,)).numpy()[self._dofs]
            if not np.all((values == 0) | np.isclose(values, M6_PARAMETERS[coefficient], rtol=1e-6, atol=0)):
                raise RuntimeError(f"{field} already has a different coefficient; no silent retuning")
        limits = self._array(model, "joint_effort_limit", wp.float32, (n,)).numpy()[self._dofs]
        if not np.allclose(limits, self._params["joint_effort_limit"], rtol=1e-6, atol=0):
            raise RuntimeError("model joint_effort_limit differs from explicit BAM effort contract")
        self._array(self._stage.state_0, "joint_q", wp.float32, (model.joint_coord_count,))
        self._array(self._stage.state_0, "joint_qd", wp.float32, (n,))
        self._array(self._stage.control, "joint_f", wp.float32, (n,))
        dof_map = getattr(solver, "mjc_dof_to_newton_dof", None)
        if not isinstance(dof_map, wp.array) or dof_map.ndim != 2 or min(dof_map.shape) < 1:
            raise RuntimeError("MJWarp dof map must be a nonempty 2-D array")
        self._array(solver, "mjc_dof_to_newton_dof", wp.int32, dof_map.shape)
        mapping = dof_map.numpy()
        if np.any(mapping < -1) or np.any(mapping >= n):
            raise RuntimeError("MJWarp dof map contains out-of-range indices")
        cells = [np.argwhere(mapping == dof) for dof in self._dofs]
        if any(cell.shape != (1, 2) for cell in cells):
            raise RuntimeError("MJWarp map must contain every owned DOF exactly once")
        if len({int(cell[0, 0]) for cell in cells}) != 1:
            raise RuntimeError("one BAM battery group cannot span MJWarp worlds")
        self._solver_cells = np.concatenate(cells)
        shape = dof_map.shape
        for field in ("dof_frictionloss", "dof_damping", "dof_armature"):
            self._array(mjm, field, wp.float32, shape)
        self._array(mjm, "dof_solref", wp.vec2, shape)
        vec5 = wp.types.vector(length=5, dtype=wp.float32)
        self._array(mjm, "dof_solimp", vec5, shape)
        attrs = getattr(model, "mujoco", None)
        self._array(attrs, "solreffriction", wp.vec2, (n,))
        self._array(attrs, "solimpfriction", vec5, (n,))
        if np.any(self._array(attrs, "dof_passive_stiffness", wp.float32, (n,)).numpy()[self._dofs] != 0):
            raise RuntimeError("owned joints have a residual passive spring")
        for field in ("qfrc_bias", "qfrc_constraint"):
            self._array(mjd, field, wp.float32, shape)
        self._array(mjd, "nefc", wp.int32, (shape[0],))
        efc = getattr(mjd, "efc", None)
        force = getattr(efc, "force", None)
        if not isinstance(force, wp.array) or force.ndim != 2 or force.shape[0] != shape[0]:
            raise RuntimeError("MJWarp efc.force must be an expanded 2-D array")
        for field, dtype in (("force", wp.float32), ("type", wp.int32), ("id", wp.int32)):
            self._array(efc, field, dtype, force.shape)

    def _check_model_sync(self):
        import numpy as np
        from .microduck_actuator import M6_PARAMETERS

        cells = tuple(self._solver_cells.T)
        for field, expected in getattr(self, "_bound_friction", {}).items():
            if not np.array_equal(getattr(self._model.mujoco, field).numpy()[self._dofs], expected):
                raise RuntimeError("bound friction constraint tuning changed; rebind explicitly")
        for model_field, solver_field, coefficient in (
                ("joint_damping", "dof_damping", "friction_viscous"),
                ("joint_armature", "dof_armature", "armature")):
            for values in (getattr(self._model, model_field).numpy()[self._dofs],
                           getattr(self._solver.mjw_model, solver_field).numpy()[cells]):
                if not np.allclose(values, M6_PARAMETERS[coefficient], rtol=1e-6, atol=0):
                    raise RuntimeError(f"{model_field}/{solver_field} not synchronized to M6 before actuation")
        for model_field, solver_field in (("solreffriction", "dof_solref"), ("solimpfriction", "dof_solimp")):
            if not np.array_equal(getattr(self._model.mujoco, model_field).numpy()[self._dofs],
                                  getattr(self._solver.mjw_model, solver_field).numpy()[cells]):
                raise RuntimeError(f"{solver_field} is not synchronized before actuation")

    def _check_live(self):
        import numpy as np
        if self._stage.model is not self._model or self._stage.solver is not self._solver:
            raise RuntimeError("stage model/solver changed; construct a new BAM adapter")
        if any(getattr(obj, name, None) is not reference for obj, name, reference in self._bound_channels):
            raise RuntimeError("bound solver channels replaced; construct a new BAM adapter")
        if not np.array_equal(self._solver.mjc_dof_to_newton_dof.numpy(), self._bound_map):
            raise RuntimeError("bound solver DOF map changed; construct a new BAM adapter")
        self._validate_binding()
        self._check_model_sync()

    @property
    def coordinate_indices(self):
        """Owned policy coordinates/DOFs, without device reads or mutation."""
        return tuple(map(int, self._qs)), tuple(map(int, self._dofs))

    def _scatter(self, destination, values):
        self._wp.copy(self._wp.indexedarray(destination, [self._scatter_indices]), values)

    def set_targets(self, q_targets):
        """Hold fourteen finite radian targets; invalid input disarms this adapter."""
        import numpy as np

        self._armed = False
        raw = np.asarray(q_targets)
        if raw.shape != (14,) or raw.dtype.kind not in "fiu":
            raise ValueError("targets must be a finite numeric vector of shape (14,)")
        with np.errstate(over="ignore", invalid="ignore"):
            targets = raw.astype(np.float32)
        if not np.isfinite(targets).all():
            raise ValueError("targets must be finite float32 values")
        self._targets.assign(targets)
        self._armed = True

    def before_step(self, dt):
        """Prepare effort/friction for ONE external solver step; never step it.

        Host-side checks are intentionally outside CUDA graph capture. A caller
        must abort the physics step on any exception, not reuse old efforts.
        """
        import numbers
        import numpy as np

        if not self._armed:
            raise RuntimeError("set fresh targets before actuation (also after reset)")
        if (isinstance(dt, bool) or not isinstance(dt, numbers.Real) or not np.isfinite(dt)
                or not np.isclose(dt, self._params["physics_dt"], rtol=1e-7, atol=0)):
            raise ValueError("dt must match the explicit physics_dt")
        self._check_live()
        count = self._step_count()
        if self._last_step is not None and count != self._last_step + 1:
            raise RuntimeError("BAM cadence requires exactly one before_step per consecutive simulation_step_count")
        if self._steps_since_reset >= 2**31 - 1:
            raise RuntimeError("BAM native int32 step history exhausted; reset required")
        state = self._stage.state_0
        if not (np.isfinite(state.joint_q.numpy()).all() and np.isfinite(state.joint_qd.numpy()).all()):
            raise ValueError("nonfinite physics state; no BAM actuation")
        mjd = self._solver.mjw_data
        nefc = mjd.nefc.numpy()
        if np.any(nefc < 0) or np.any(nefc > mjd.efc.force.shape[1]):
            raise RuntimeError("MJWarp nefc exceeds the constraint array capacity")
        if self._skip_external_once:
            self._drive.external_torque.zero_()
        else:
            self._bridge.gather_external_torque(self._drive.external_torque)
        if not np.isfinite(self._drive.external_torque.numpy()).all():
            raise ValueError("nonfinite previous external torque")
        self._drive.compute(state.joint_q, state.joint_qd, self._targets, self._targets, None,
                            self._q_indices, self._dof_indices, self._target_indices, self._target_indices,
                            self._forces, self._state, float(dt), self._device)
        for output in (self._forces, self._drive.motor_torque, self._drive.effective_vin, self._drive.friction_budget):
            if not np.isfinite(output.numpy()).all():
                raise ValueError("nonfinite native BAM output; solver must not step")
        self._bridge.publish_dof_friction(self._drive.friction_budget)
        self._scatter(self._stage.control.joint_f, self._forces)
        self._drive.update_state(self._state, self._state)
        self._last_step = count
        self._steps_since_reset += 1
        self._skip_external_once = False

    def reset(self):
        """Clear owned histories/channels, NOT physical q/qd or another DOF."""
        self._check_live()
        if self._params["delay_seed"] + self._state.reset_count + 1 >= 2**31 - 1:
            raise RuntimeError("native BAM reset seed exhausted; rebind with a new seed")
        self._armed = False
        self._state.reset()
        for array in (*self._drive._next_state_arrays.values(), self._targets, self._forces,
                      self._drive.motor_torque, self._drive.effective_vin,
                      self._drive.external_torque, self._drive.friction_budget):
            array.zero_()
        self._bridge.publish_dof_friction(self._drive.friction_budget)
        self._scatter(self._stage.control.joint_f, self._forces)
        self._last_step = None
        self._steps_since_reset = 0
        self._skip_external_once = True

    def telemetry(self) -> dict:
        """Snapshot, not proof of motion or of a real Isaac/physics backend."""
        from .microduck_actuator import M6_PARAMETERS

        result = {
            "implementation": "IsaacLab#8161 native DriveBam + model-bound MjWarpActuatorBridge",
            "revision": REVISION, "source_sha256": dict(SOURCE_SHA256),
            "source_root": self._sources.source_root, "device": str(self._device),
            "newton_version": self._newton.__version__, "params": dict(self._params),
            "q_indices": self._qs.tolist(), "dof_indices": self._dofs.tolist(),
            "armed": self._armed, "steps_since_reset": self._steps_since_reset,
            "last_simulation_step_count": self._last_step, "reset_count": self._state.reset_count,
            "friction_reference": "bam_mjlab", "env_dof_stride": 14,
            "mechanical_damping": M6_PARAMETERS["friction_viscous"], "armature": M6_PARAMETERS["armature"],
            "physics_verified": False,
        }
        for key, array in (("effort", self._forces), ("targets", self._targets),
                           ("motor_torque", self._drive.motor_torque), ("effective_vin", self._drive.effective_vin),
                           ("friction_budget", self._drive.friction_budget), ("external_torque", self._drive.external_torque),
                           ("delay_lag", self._state.delay_lag), ("delay_fill", self._state.delay_fill)):
            result[key] = array.numpy().tolist()
        return result
