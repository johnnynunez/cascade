"""Build the identity-bound optional Factory scene; no implicit SDK startup."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ..control.fastening import FasteningBinding, FasteningFault, FasteningLimits, check_geometry
from .factory_observation import (
    FactoryGeometry, FactoryObserver, actuator_descriptor, joint_mapping, sdk_sources,
)
from .factory_recipe import MARGIN_RECIPE, seating_recipe


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _array_digest(value, *, allow_infinite=False):
    value = np.ascontiguousarray(value)
    if (value.dtype.kind not in "biuf" or np.isnan(value).any()
            or not allow_infinite and not np.isfinite(value).all()):
        raise FasteningFault("invalid immutable model array")
    return {"dtype": str(value.dtype), "shape": list(value.shape),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest()}


# Explicit model inputs consumed by collision/FK. Runtime body/joint state,
# computed collision AABBs and contact work queues are intentionally separate.
STATIC_NEWTON_ARRAYS = (
    "shape_transform", "shape_body", "shape_flags", "shape_scale", "shape_type",
    "shape_margin", "shape_gap", "shape_material_ke", "shape_material_kd",
    "shape_material_kf", "shape_material_mu", "shape_material_mu_torsional",
    "shape_material_mu_rolling", "shape_filter", "shape_collision_group", "shape_world",
    "body_com", "body_inertia", "body_mass", "body_flags",
    "joint_type", "joint_parent", "joint_child", "joint_X_p", "joint_X_c",
    "joint_axis", "joint_armature", "joint_damping", "joint_friction",
    "joint_q_start", "joint_qd_start", "joint_limit_lower", "joint_limit_upper",
)
MAPPING_ARRAYS = (
    "mjc_body_to_newton", "mjc_geom_to_newton_shape", "mjc_jnt_to_newton_jnt",
    "mjc_dof_to_newton_dof", "mjc_actuator_ctrl_source", "mjc_actuator_to_newton_idx",
)
STATIC_MJW_ARRAYS = (
    "qpos0", "qpos_spring", "body_pos", "body_quat", "body_ipos", "body_iquat",
    "body_mass", "body_inertia", "body_gravcomp", "jnt_qposadr", "jnt_dofadr",
    "jnt_bodyid", "jnt_limited", "jnt_actfrclimited", "jnt_actgravcomp",
    "jnt_pos", "jnt_axis", "jnt_stiffness", "jnt_range", "jnt_actfrcrange",
    "dof_armature", "dof_damping", "dof_frictionloss", "geom_bodyid", "geom_pos",
    "geom_quat", "geom_size", "geom_friction", "geom_solref", "geom_solimp",
    "actuator_trntype", "actuator_dyntype", "actuator_gaintype", "actuator_biastype",
    "actuator_trnid", "actuator_dynprm", "actuator_gainprm", "actuator_biasprm",
    "actuator_forcelimited", "actuator_forcerange", "actuator_ctrllimited",
    "actuator_ctrlrange", "actuator_gear",
)
NATIVE_OPTION_ARRAYS = ("gravity", "wind", "density", "viscosity", "tolerance",
                        "ls_tolerance", "impratio_invsqrt")
NATIVE_OPTION_SCALARS = ("integrator", "cone", "solver", "iterations", "ls_iterations",
                         "disableflags", "enableflags", "run_collision_detection", "graph_conditional")


def _collision_route(solver):
    """Require Newton 1.6's stored mode and the effective MJWarp solve branch.

    The pinned solver stores the constructor option in `_use_mujoco_contacts`
    and copies it into `mjw_model.opt.run_collision_detection`. Its step method
    consumes the latter. Neither a missing field nor a contradictory mode is
    evidence that the Newton contact stream is used.
    """
    try:
        route = {"use_mujoco_cpu": solver.use_mujoco_cpu,
                 "_use_mujoco_contacts": solver._use_mujoco_contacts,
                 "run_collision_detection": solver.mjw_model.opt.run_collision_detection}
    except AttributeError as exc:
        raise FasteningFault("pinned collision mode interface is unavailable") from exc
    if any(value is not False for value in route.values()):
        raise FasteningFault("only explicit Newton collision + MJWarp modes are implemented")
    return route


def _buffer_digest(owner, name, *, optional_int32_shape=None, allow_infinite=False):
    try:
        value = getattr(owner, name)
    except AttributeError as exc:
        raise FasteningFault(f"required model interface is unavailable: {name}") from exc
    if value is None:
        if optional_int32_shape is not None:
            return {"present": False, "declared_dtype": "int32",
                    "declared_shape": list(optional_int32_shape)}
        raise FasteningFault(f"required model array is unavailable: {name}")
    array = value.numpy()
    if optional_int32_shape is not None and (
            array.dtype != np.dtype("int32") or array.shape != optional_int32_shape):
        raise FasteningFault(f"invalid optional model array layout: {name}")
    return {"present": True, **_array_digest(array, allow_infinite=allow_infinite)}


def _newton_array_fingerprint(model):
    # Newton 1.6 Model declares shape_filter: int32[shape_count] | None.
    # Its builder does not populate that field. Unlike the required collision
    # group and material arrays, absence is legitimate and must remain explicit
    # in the identity. A later None <-> array change cannot retain the pin.
    if type(model.shape_count) is not int or model.shape_count <= 0:
        raise FasteningFault("Factory model requires positive shape_count")
    return {name: _buffer_digest(model, name,
                optional_int32_shape=(model.shape_count,) if name == "shape_filter" else None,
                allow_infinite=name in {"joint_limit_lower", "joint_limit_upper"})
            for name in STATIC_NEWTON_ARRAYS}


def model_fingerprint(scene):
    """Compiled MuJoCo geometry/actuation plus actual Newton collision/FK arrays.

    Mesh/SDF construction is additionally source/asset-bound at creation. This
    fixture has one private owner and exposes no model/SDF mutation API.
    """
    route = _collision_route(scene.solver)
    mj = scene.mujoco
    binary = np.zeros(mj.mj_sizeModel(scene.solver.mj_model), np.uint8)
    mj.mj_saveModel(scene.solver.mj_model, buffer=binary)
    arrays = _newton_array_fingerprint(scene.model)
    maps = {name: _buffer_digest(scene.solver, name) for name in MAPPING_ARRAYS}
    native = {name: _buffer_digest(scene.solver.mjw_model, name)
              for name in STATIC_MJW_ARRAYS}
    options = {name: _buffer_digest(scene.solver.mjw_model.opt, name)
               for name in NATIVE_OPTION_ARRAYS}
    options.update({name: int(getattr(scene.solver.mjw_model.opt, name))
                    for name in NATIVE_OPTION_SCALARS})
    return {"mujoco_binary_sha256": hashlib.sha256(binary.tobytes()).hexdigest(),
        "collision_route": route,
        "newton_arrays": arrays, "maps": maps, "native_solve_arrays": native, "native_options": options,
        "shape_labels": list(scene.model.shape_label), "body_labels": list(scene.model.body_label),
        "joint_labels": list(scene.model.joint_label)}


def _files(directory):
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(directory).rglob("*")) if p.is_file()}


def authoring_descriptor(scene):
    """Actual authoring inputs and follower table, never a physical admission."""
    recipe = seating_recipe(scene.fixture_recipe)
    if (tuple(scene.fixture_center_xy_m) != recipe.center_xy_m
            or not np.array_equal(scene._center_xy, recipe.center_xy_m)
            or not np.array_equal(scene.fixture_position, (*recipe.center_xy_m, 0.))
            or scene.ik_margin_rad != recipe.ik_margin_rad
            or scene.intersect_position_control_range is not recipe.intersect_position_control_range
            or scene._initial_nut_z != .069
            or not np.array_equal(scene.socket_offset, [.012, 0., -.115])):
        raise FasteningFault("mounted fixture authoring differs from the declared recipe")
    arrays = {"entry_rad": np.asarray(scene.entry), "bottom_rad": np.asarray(scene.bottom),
              "follower_heights_m": np.asarray(scene._heights),
              "follower_targets_rad": np.asarray(scene._targets),
              "ik_ranges_rad": np.asarray(scene.ik_ranges)}
    shapes = ((5,), (5,), (43,), (43, 5), (5, 2))
    for (name, value), shape in zip(arrays.items(), shapes, strict=True):
        if value.shape != shape or not np.isfinite(value).all():
            raise FasteningFault(f"invalid mounted authoring array: {name}")
    if not np.array_equal(arrays["follower_heights_m"], np.linspace(.027, .069, 43)):
        raise FasteningFault("mounted follower height grid differs from the recipe")
    return {**asdict(recipe), "initial_nut_z_m": scene._initial_nut_z,
            "fixture_origin_m": list(scene.fixture_position),
            "socket_offset_m": scene.socket_offset.tolist(),
            **{name: value.tolist() for name, value in arrays.items()}}


def check_authored_joint(name, position, lower, upper, margin):
    if not lower+margin <= position <= upper-margin:
        raise FasteningFault("authored initial joint state is outside the admitted margin: "
            f"joint={name} q_rad={position:.17g} lower_rad={lower:.17g} "
            f"upper_rad={upper:.17g} margin_rad={margin:.17g}")


class FactoryBoundModel:
    """Constructed scene plus immutable identity; not physical task admission."""

    def __init__(self, scene, *, clock=time.monotonic, precompile=None, sdk_recipe=None, seating=None):
        from .factory_sdk import INTERNAL_PINS, validate_sdk_recipe
        validate_sdk_recipe(sdk_recipe)
        seat_limits = None
        if seating is not None:
            from ..control.fastening_seat import SEATING_RECIPE, SeatingLimits
            if seating != SEATING_RECIPE or scene.fixture_recipe != MARGIN_RECIPE:
                raise FasteningFault("seating requires the explicit mounted margin fixture")
            seat_limits = SeatingLimits()
        from .newton_screw_seating import SeatingScene
        if type(scene) is not SeatingScene or scene.step_id != 0 or scene.time_s != 0:
            raise FasteningFault("binding requires the exact fresh mounted SeatingScene")
        self.scene, self.clock = scene, clock
        _collision_route(scene.solver)
        names = (*scene.arm_joints, "gripper", "socket_spin")
        self.joints = joint_mapping(scene, names)
        self.geometry = FactoryGeometry(scene, self.joints)
        actuators = actuator_descriptor(scene.solver.mj_model, self.joints)
        native = scene.solver.mj_model
        lower, upper, caps = [], [], []
        for row in self.joints[:-1]:
            if not native.jnt_limited[row.native_joint]:
                raise FasteningFault("controlled arm joint has no imported hard range")
            jr, cr = native.jnt_range[row.native_joint], native.actuator_ctrlrange[row.native_actuator]
            lower.append(float(max(jr[0], cr[0])-row.reference_rad))
            upper.append(float(min(jr[1], cr[1])-row.reference_rad))
            force = native.actuator_forcerange[row.native_actuator]
            if force[0] >= 0 or force[1] <= 0:
                raise FasteningFault("invalid imported actuator effort cap")
            caps.append(float(min(abs(force[0]), abs(force[1]))))
        self.limits = FasteningLimits(tuple(lower), tuple(upper), tuple(caps),
            (-.15, -.36, -.02), (.4, .36, .4), 0., seating=seat_limits)
        self._authoring = authoring_descriptor(scene)
        if scene.fixture_recipe == MARGIN_RECIPE:
            if any(row.reference_rad != 0 for row in self.joints[:-1]):
                raise FasteningFault("mounted margin recipe requires zero native joint references")
            for target in (scene.entry, scene.bottom, *scene._targets):
                for i, position in enumerate(target):
                    check_authored_joint(self.joints[i].name, float(position), lower[i], upper[i], scene.ik_margin_rad)
        compilation = None
        if precompile is not None:
            from .factory_precompile import precompile_factory
            compilation = precompile_factory(scene, recipe=precompile, **(
                {} if sdk_recipe is None else {"sdk_recipe": sdk_recipe}))
        self._fingerprint = model_fingerprint(scene)
        source_names = ("factory_model.py", "factory_owner.py", "factory_observation.py",
            "factory_recipe.py", "factory_sdk.py",
            "newton_screw_contact.py", "newton_screw_seating.py", "threading_verification.py",
            "microduck_contact_support.py")
        sources = {str(Path(__file__).with_name(n).resolve()): hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                   for n in source_names}
        if sdk_recipe is not None:
            sources[str(INTERNAL_PINS.resolve())] = hashlib.sha256(INTERNAL_PINS.read_bytes()).hexdigest()
        if compilation is not None:
            for name in ("factory_precompile.py", "factory_precompile_pins.json", "factory_mjdata_layout.json"):
                path = Path(__file__).with_name(name).resolve()
                sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        package = Path(__file__).resolve().parents[1]
        for relative in ("control/fastening.py", "skills/fastening_runtime.py", "config.py",
                         "apps/factory_runtime.py", "apps/robot_runtime.py",
                         "control/fastening_seat.py", "skills/seating_runtime.py"):
            p = package/relative
            sources[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        self.document = {"schema_version": 1, "recipe": scene.fixture_recipe,
            "authoring": self._authoring,
            "mounted_tool": True, "preengaged_nut": True, "pickup": False, "seating_claim": False,
            "sources": sources, "sdk_recipe": sdk_recipe,
            "sdk_sources": sdk_sources() if sdk_recipe is None else sdk_sources(sdk_recipe),
            "factory_files": _files(scene.assets), "robot_files": _files(scene.robot_asset.parent),
            "model": self._fingerprint, "geometry": self.geometry.descriptor, "actuators": actuators,
            "limits": asdict(self.limits), "pitch_m": .0025,
            "dt_s": float(np.float32(scene.frame_dt/scene.substeps)),
            "substeps_per_legacy_frame": scene.substeps, "cuda_graph": False,
            "socket_speed_rad_s": scene.requested_speed_rad_s,
            "requested_spindle_cap_nm": .05,
            "native_float32_spindle_cap_nm": float(np.float32(.05)),
            "lifecycle": "single private solve owner; no mutation between solves"}
        if seat_limits is not None:
            self.document["seating_recipe"] = seating
            self.document["seating_task"] = asdict(seat_limits)
        if compilation is not None:
            self.document["precompilation"] = compilation
        labels = scene._names(scene.model.shape_label)
        label = lambda name: scene.model.shape_label[labels[name]]
        thread = (label("nut"), label("bolt"))
        tools = tuple((label("nut"), label("socket_wall_"+str(i))) for i in range(6))
        seat_pair = (label("nut"), label("seat_ring")) if seat_limits is not None else None
        allowed = (thread, *tools) + ((seat_pair,) if seat_pair is not None else ())
        self.binding = FasteningBinding(_digest(self.document), self.limits.sha256,
            "so101_factory_m20", "fixed_factory_bolt_m20_loose", "factory_nut_m20_loose",
            "mounted_spring_socket", names[:-1], tuple(scene.model.shape_label),
            allowed, thread, tools, self.document["dt_s"],
            fixture_origin_m=tuple(float(v) for v in scene.fixture_position), fixture_recipe=scene.fixture_recipe,
            seat_contact_pair=seat_pair)
        # Initial-condition conversion only, BEFORE the first solve. It copies
        # the already-authored state, introduces no extra force or FK pose write.
        scene.solver._update_mjc_data(scene.solver.mjw_data, scene.model, scene.state)
        q = scene.state.joint_q.numpy()
        nq = scene.solver.mjw_data.qpos.numpy()[0]
        bounds = self.geometry.evaluate(nq, body_poses=scene.state.body_q.numpy())
        check_geometry(*bounds, self.limits)
        self.initial_control = scene.control.mujoco.ctrl.numpy().copy()
        if self.initial_control.shape != (len(self.joints),):
            raise FasteningFault("unexpected initial control layout")
        for index, row in enumerate(self.joints[:-1]):
            position = float(q[row.newton_q])
            check_authored_joint(row.name, position, lower[index], upper[index], self.limits.joint_margin_rad)
            self.initial_control[row.control_index] = position + row.reference_rad
        self.initial_control[self.joints[-1].control_index] = 0.
        self.observer = FactoryObserver(scene, self.binding, self.limits, self.geometry, self.joints,
            **({} if sdk_recipe is None else {"sdk_recipe": sdk_recipe}))

    def check_immutable(self):
        if model_fingerprint(self.scene) != self._fingerprint:
            raise FasteningFault("native model/mapping/geometry changed after identity binding")
        if authoring_descriptor(self.scene) != self._authoring:
            raise FasteningFault("mounted authoring changed after identity binding")
