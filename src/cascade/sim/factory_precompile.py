"""Finite compilation plan for one reviewed Factory recipe, with no kernel launch.

This is preparation, never readiness or physical admission. The pinned SDK's
lazy factories only construct kernel definitions. Loading their owned modules
must leave all physical arrays and counters unchanged. Unreviewed branches are
rejected; there is no global ``force_load`` or warm-up solve.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import ctypes
import hashlib
import importlib
import json
from pathlib import Path
import time
from types import ModuleType

import numpy as np

from ..control.fastening import FasteningFault

PRECOMPILE_RECIPE = "factory_nv19_dense_compile_v1"
_PINS = Path(__file__).with_name("factory_precompile_pins.json")


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ModuleVariant:
    label: str
    module: object
    block_dim: int
    parameters: tuple = ()

    def describe(self):
        return {"label": self.label, "module": self.module.name,
                "block_dim": self.block_dim, "parameters": list(self.parameters)}


def admitted_sdk():
    """Finite imports, checked against reviewed sources before any factory call."""
    modules, sources = {}, {}
    for name, expected in json.loads(_PINS.read_text()).items():
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise FasteningFault(f"unreviewed compilation SDK source: {name}")
        modules[name] = module
        sources[name] = actual
    return modules, sources


def _require(condition, message):
    if not condition:
        raise FasteningFault("unsupported finite precompilation recipe: " + message)


def validate_recipe(scene):
    """Reject branch changes instead of pretending this inventory covers them."""
    from .factory_recipe import MARGIN_RECIPE
    solver = scene.solver
    m, d, narrow = solver.mjw_model, solver.mjw_data, scene.pipeline.narrow_phase
    _require(scene.fixture_recipe == MARGIN_RECIPE, "mounted margin v2 required")
    _require(scene.model.device.is_cuda, "CUDA model required")
    _require(int(solver._deterministic) == 0 and solver._deterministic_max_records == 0,
             "compiler determinism recipe")
    _require(scene.step_id == 0 and scene.time_s == 0 and solver._step == 0,
             "construction must precede every solve")
    _require(np.array_equal(d.time.numpy(), np.zeros(1, dtype=np.float32)), "nonzero native time")
    _require(m.nv == 19 and d.nworld == 1 and not m.is_sparse, "nv19/single-world/dense")
    _require(m.nflex == 0 and m.ntendon == 0 and m.nplugin == 0, "flex/tendon/plugin")
    _require(int(m.opt.solver) == 2 and int(m.opt.cone) == 1 and int(m.opt.integrator) == 3,
             "Newton/elliptic/implicitfast")
    _require(int(m.opt.enableflags) == 0 and int(m.opt.disableflags) == 524288,
             "sleep/island/disable flags changed")
    _require(m.opt.iterations == 50 and m.opt.ls_iterations == 100, "solver iteration recipe")
    _require(not solver.use_mujoco_cpu and not solver._use_mujoco_contacts
             and not m.opt.run_collision_detection, "collision route")
    _require(not any(vars(m.callback).values()), "custom physics callbacks")
    _require(scene.pipeline.broad_phase_mode == "nxn", "broad phase")
    _require(narrow.reduce_contacts and narrow.has_meshes and not narrow.has_heightfields
             and not narrow.split_gjk_mpr and not narrow.speculative
             and narrow.hydroelastic_sdf is None and not narrow.deterministic,
             "Newton collision/reduction branch")
    _require(narrow.block_dim == 128 and narrow.tile_size_mesh_mesh == 256
             and narrow.mesh_triangle_block_dim == 32, "Newton launch dimensions")
    _require(d.njmax == 4096 and m.nv_pad == 20, "constraint/tile dimensions")
    expected = {"actuator_velocity": 32, "contact_jac_tiled": 32,
                "small_cholesky": 64, "cholesky_factorize_solve": 32,
                "update_gradient_JTDAJ_dense": 128, "update_gradient_cholesky": 64,
                "linesearch_iterative": 32}
    _require(all(getattr(m.block_dim, name) == value for name, value in expected.items()),
             "MJWarp block dimensions")
    _require(len(m.M_tiles) == 2 and {tile.elemid.size == 0 for tile in m.M_tiles} == {False, True},
             "two reviewed mass-matrix tile branches")
    _require({(tile.size, tile.elemid.size == 0) for tile in m.M_tiles} == {(6, True), (13, False)}
             and all(tile.adr.size > 0 for tile in m.M_tiles), "mass tiles")
    _require(m.M_colind.shape == (90,), "mass matrix storage recipe")
    _require(d.qLD.shape[1] == m.qLD_block_total, "sparse mass factor tail")


def compilation_plan(scene, modules, wp):
    """Materialize only pure cached factories, using the actual model's fields."""
    validate_recipe(scene)
    m, d = scene.solver.mjw_model, scene.solver.mjw_data
    narrow = scene.pipeline.narrow_phase
    variants = []

    def static(name, block):
        variants.append(ModuleVariant(name, wp.get_module(modules[name].__name__), block))

    def kernel(label, value, block, *params):
        _require(isinstance(block, int) and block > 0, "invalid block dimension")
        variants.append(ModuleVariant(label, value.module, block, tuple(params)))

    prefix = "newton._src."
    for name, block in (("sim.collide", 256), ("geometry.broad_phase_nxn", 256),
                        ("geometry.narrow_phase", 256), ("geometry.contact_reduction_global", 256),
                        ("geometry.contact_reduction_global", narrow.block_dim),
                        ("geometry.sdf_contact", 256)):
        static(prefix + name, block)
    kernel("newton.primitive", narrow.primitive_kernel, narrow.block_dim)
    kernel("newton.gjk_mpr", narrow.narrow_phase_kernel, narrow.block_dim)
    kernel("newton.mesh_triangle_reducer",
           modules[prefix + "geometry.narrow_phase"].mesh_triangle_contacts_to_reducer_kernel,
           narrow.mesh_triangle_block_dim)
    precomputed = (scene.model.mesh_edge_centers is not None
                   and scene.model.mesh_edge_halves is not None)
    kernel("newton.mesh_mesh", narrow.mesh_mesh_contacts_kernel_precomputed if precomputed
           else narrow.mesh_mesh_contacts_kernel, narrow.tile_size_mesh_mesh, precomputed)
    kernel("newton.export_reduced", narrow.export_reduced_contacts_kernel,
           modules[prefix + "geometry.contact_reduction_global"].EXPORT_REDUCED_CONTACTS_BLOCK_DIM)

    variants.extend(_mujoco_variants(m, d, modules, wp))
    _require(len(variants) == 39, "inventory count")
    return tuple(variants)


def _mujoco_variants(m, d, modules, wp):
    """Pure definition inventory, also inspectable with CPU model allocations."""
    variants = []
    def static(name, block):
        variants.append(ModuleVariant(name, wp.get_module(modules[name].__name__), block))
    def kernel(label, value, block, *params):
        variants.append(ModuleVariant(label, value.module, block, tuple(params)))
    prefix = "mujoco_warp._src."
    for name, block in (("constraint", 256), ("forward", m.block_dim.actuator_velocity),
                        ("passive", 256), ("forward", 256), ("support", 256),
                        ("solver", 256), ("sensor", 256), ("derivative", 256)):
        static(prefix + name, block)

    def factory(module, name, args, block=256):
        value = getattr(modules[prefix + module], name)(*args)
        params = tuple(a.size if hasattr(a, "elemid") else a for a in args)
        kernel(module + "." + name, value, block, *params)

    # Branches match the pinned consumers. validate_recipe refuses compact,
    # sparse, sleeping, incremental and alternate-integrator paths beforehand.
    factory("constraint", "_friction_dof", (m.is_sparse, True))
    factory("constraint", "_limit_slide_hinge", (m.is_sparse, True))
    factory("constraint", "_efc_contact_init", (m.opt.cone, m.is_sparse, True))
    factory("constraint", "_efc_contact_jac_dense", (m.block_dim.contact_jac_tiled, m.opt.cone),
            m.block_dim.contact_jac_tiled)
    factory("constraint", "_efc_contact_update", (m.opt.cone,))
    factory("forward", "_qfrc_smooth", (False,))
    for tile in m.M_tiles:
        if tile.elemid.size == 0:
            factory("smooth", "_small_cholesky_factorize_solve_block", (tile.size,), m.block_dim.small_cholesky)
        else:
            factory("smooth", "_tile_cholesky_factorize_solve_block", (tile,), m.block_dim.cholesky_factorize_solve)
    types = modules[prefix + "types"]
    factory("solver", "_solve_init_dof", (not (m.opt.disableflags & types.DisableBit.WARMSTART), m.is_sparse))
    factory("solver", "_solve_init_jaref_kernel", (m.is_sparse, m.nv, 50, False))
    factory("support", "mul_m_kernel", (True,))
    factory("solver", "_update_constraint_efc", (False,))
    factory("solver", "_update_constraint_init_qfrc_constraint_dense", (False,))
    factory("solver", "_update_gradient_zero_grad_dot", (False,))
    factory("solver", "_update_gradient_grad", (False,))
    factory("solver", "_update_gradient_JTDAJ_dense_tiled",
            (m.nv_pad, types.TILE_SIZE_JTDAJ_DENSE, d.njmax, m.M_colind.shape[0]),
            m.block_dim.update_gradient_JTDAJ_dense)
    factory("solver", "_update_gradient_cholesky", (m.nv, False), m.block_dim.update_gradient_cholesky)
    factory("solver", "_linesearch_iterative_kernel",
            (m.opt.ls_iterations, m.opt.cone, True, m.is_sparse, False, bool(m.opt.warn_overflow)),
            m.block_dim.linesearch_iterative)
    factory("solver", "_solve_done", (bool(m.opt.warn_overflow),))
    factory("forward", "_next_time_builder", (bool(m.opt.warn_overflow),))
    _require(len(variants) == 28, "MJWarp inventory count")
    return tuple(variants)


def physical_snapshot(scene, wp):
    """Hash every exposed array recursively in the owned physical object graph.

    Includes both Newton states, controls, model, collision scratch/contacts,
    MJWarp model/data, native MuJoCo data/model arrays and solver mappings. No
    whitelist of named dynamic buffers can silently omit a newly added array.
    Compiler handles/device infrastructure are metadata, never traversed.
    """
    if scene.model.device.is_cuda:
        wp.synchronize_device(scene.model.device)
    rows, seen = {}, {}

    def visit(value, path):
        if isinstance(value, (np.ndarray, wp.array)):
            array = np.asarray(value if isinstance(value, np.ndarray) else value.numpy())
            _require(array.dtype.kind != "O", "object array in physical snapshot")
            rows[path] = {"shape": list(array.shape), "dtype": array.dtype.str,
                          "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
            return
        if value is None or isinstance(value, (str, bool, int, float, np.generic)):
            rows[path] = {"scalar": repr(value)}
            return
        if isinstance(value, (ModuleType, type)) or callable(value):
            return
        if id(value) in seen:
            rows[path] = {"alias": seen[id(value)][0]}
            return
        # Keep the object alive as well as its ID: temporary field dictionaries
        # must not reuse an earlier ID and be misclassified as an alias.
        seen[id(value)] = (path, value)
        if isinstance(value, dict):
            for key, item in sorted(value.items(), key=lambda row: str(row[0])):
                visit(item, path + "/" + str(key))
        elif isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                visit(item, f"{path}/{i}")
        elif isinstance(value, (set, frozenset)):
            for i, item in enumerate(sorted(value, key=repr)):
                visit(item, f"{path}/set/{i}")
        elif is_dataclass(value):
            for field in fields(value):
                visit(getattr(value, field.name), path + "/" + field.name)
        elif hasattr(value, "_cls") and hasattr(value._cls, "vars"):
            # Warp StructInstance: inspect its declared fields, not the ctypes
            # backing object or compiler's type graph.
            for name in value._cls.vars:
                visit(getattr(value, name), path + "/" + name)
        elif isinstance(value, ctypes.Array):
            visit(np.ctypeslib.as_array(value), path + "/ctypes_values")
        elif type(value).__module__.startswith("mujoco."):
            # Pybind objects expose their arrays as descriptors, not __dict__.
            for name in sorted(dir(value)):
                if not name.startswith("_"):
                    item = getattr(value, name)
                    if isinstance(item, (np.ndarray, float, int)):
                        visit(item, path + "/" + name)
        elif type(value).__module__.startswith("warp.") and type(value).__name__ in (
                "Device", "Stream", "Event", "Kernel", "Function", "Module", "Graph", "Runtime"):
            rows[path] = {"infrastructure": type(value).__name__}
        elif type(value).__module__ == "warp._src.texture" and type(value).__name__ in (
                "Texture1D", "Texture2D", "Texture3D"):
            _require(not value._is_mipmapped, "mipmapped texture snapshot unavailable")
            host = wp.empty(shape=value._get_shape(), dtype=value.dtype, device="cpu")
            value.copy_to(host)  # Pinned SDK memcpy; never a sampling kernel.
            if value.device.is_cuda:
                wp.synchronize_device(value.device)
            visit(host, path + "/texture_bytes")
            rows[path + "/texture_identity"] = {"handle": int(value._tex_handle)}
        elif type(value).__module__ == "warp._src.types" and type(value).__name__ == "Volume":
            visit(value.array(), path + "/volume_bytes")
            rows[path + "/volume_identity"] = {"handle": int(value.id)}
        elif hasattr(value, "__dict__"):
            for name, item in sorted(vars(value).items()):
                visit(item, path + "/" + name)
        elif isinstance(value, Path):
            rows[path] = {"path": str(value)}
        else:
            raise FasteningFault(f"unsupported physical snapshot value {path}: {type(value).__name__}")

    # The solver owns mappings and additional contact scratch not in its data.
    for name in ("state", "next", "control", "contacts", "model", "pipeline", "solver"):
        visit(getattr(scene, name), name)
    # Authoring/IK and operation state live on the scene itself, not solely in
    # the solver roots. Omit only our prior evidence receipt to keep repeated
    # preparation bounded; modules and callables are excluded by visit().
    visit({name: value for name, value in vars(scene).items()
           if name != "precompile_receipt"}, "scene_fields")
    effective_class_fields = {}
    for cls in reversed(type(scene).__mro__):
        for name, value in vars(cls).items():
            if (not name.startswith("__") and not callable(value)
                    and not isinstance(value, (staticmethod, classmethod, property))):
                effective_class_fields[name] = getattr(scene, name)
    visit(effective_class_fields, "scene_class_fields")
    rows["scene_clock"] = {"step": scene.step_id, "time_s": scene.time_s}
    _require(any("qpos" in key for key in rows) and any("joint_q" in key for key in rows),
             "missing independent physical arrays")
    return rows


def precompile_factory(scene, *, recipe):
    """Precompile within the original solver's option scope, zero physics steps."""
    _require(recipe == PRECOMPILE_RECIPE, "unknown precompile selector")
    import warp as wp
    from .factory_model import model_fingerprint
    modules, sources = admitted_sdk()
    validate_recipe(scene)
    before = physical_snapshot(scene, wp)
    identity = model_fingerprint(scene)
    started = time.monotonic()
    plan, loaded, error = (), [], None
    try:
        with wp.ScopedDevice(scene.model.device), scene.solver._scoped_mujoco_warp_execution():
            plan = compilation_plan(scene, modules, wp)
            for variant in plan:
                begin = time.monotonic()
                wp.load_module(variant.module, device=scene.model.device, recursive=False,
                               block_dim=variant.block_dim, max_workers=0)
                compiled_hash = variant.module.get_module_hash(variant.block_dim)
                executable = variant.module.execs.get((scene.model.device.context, variant.block_dim))
                _require(executable is not None and executable.module_hash == compiled_hash,
                         "loader did not retain the exact device/block executable")
                loaded.append(variant.describe() | {
                    "module_sha256": compiled_hash.hex(),
                    "compiler_options": dict(variant.module.options),
                    "wall_s": time.monotonic() - begin})
    except BaseException as exc:
        error = exc
    after, after_identity, audit_error = None, None, None
    try:
        after = physical_snapshot(scene, wp)
        after_identity = model_fingerprint(scene)
    except Exception as exc:
        audit_error = exc
    unchanged = before == after and identity == after_identity
    device = {"alias": str(scene.model.device), "cuda_arch": scene.model.device.arch}
    receipt = {"recipe": recipe, "sdk_sources": sources, "loaded": loaded,
               "device": device, "warp_version": wp.__version__,
               "wall_s": time.monotonic() - started,
               "physical_before": before, "physical_after": after,
               "physical_before_sha256": _sha(before), "physical_after_sha256": _sha(after) if after else None,
               "physical_unchanged": unchanged, "model_identity_unchanged": identity == after_identity,
               "step_before": before["scene_clock"], "step_after": after["scene_clock"] if after else None,
               "compiler_error": repr(error) if error else None,
               "audit_error": repr(audit_error) if audit_error else None,
               "physics_admission": False, "ok": error is None and unchanged}
    scene.precompile_receipt = receipt
    if not unchanged:
        raise FasteningFault("module compilation changed physical buffers/model/counters") from error
    if error is not None:
        raise error
    # Timing and scratch-buffer bytes are evidence, not reproducible identity.
    return {"recipe": recipe, "sdk_sources": sources, "device": device,
            "warp_version": wp.__version__,
            "variants": [row.describe() for row in plan],
            "module_hashes": [row["module_sha256"] for row in loaded],
            "compiler_options": [row["compiler_options"] for row in loaded],
            "physical_unchanged": True, "physics_solves": 0}
