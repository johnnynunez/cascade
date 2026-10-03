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
import sys
import time
from types import ModuleType

import numpy as np

from ..control.fastening import FasteningFault

PRECOMPILE_RECIPE = "factory_nv19_contact_writer_compile_v3"
_PINS = Path(__file__).with_name("factory_precompile_pins.json")
_MJDATA_LAYOUT = Path(__file__).with_name("factory_mjdata_layout.json")
_MJDATA_LAYOUT_SHA256 = "3e2ae59397218951e7b3da694c7a1938e65de291eac1027078be2cadd08442b1"


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ModuleVariant:
    label: str
    module: object
    block_dim: int
    parameters: tuple = ()
    kernel: object | None = None

    def describe(self):
        return {"label": self.label, "module": self.module.name,
                "block_dim": self.block_dim, "parameters": list(self.parameters),
                "entrypoint": None if self.kernel is None else
                    {"key": self.kernel.key, "signature": self.kernel.sig}}


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


def recipe_admission(scene, modules):
    """Read every finite-branch predicate before deciding; never load a kernel.

    Each row retains observed values (or a read error), expected semantics and
    its verdict. A failed earlier predicate cannot hide later configuration.
    """
    from .factory_recipe import MARGIN_RECIPE
    solver, model, pipeline = (getattr(scene, name, None) for name in ('solver', 'model', 'pipeline'))
    m, d = getattr(solver, 'mjw_model', None), getattr(solver, 'mjw_data', None)
    narrow = getattr(pipeline, 'narrow_phase', None)
    rows = []

    def plain(value):
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            return {'dtype': str(array.dtype), 'shape': list(array.shape),
                    'sha256': hashlib.sha256(array.tobytes()).hexdigest(),
                    **({'values': plain(array.tolist())} if array.size <= 16 else {})}
        if isinstance(value, np.generic):
            return plain(value.item())
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(v) for v in value]
        if value is None or type(value) in (bool, int, str):
            return value
        if type(value) is float and np.isfinite(value):
            return value
        return {'type': type(value).__module__ + '.' + type(value).__qualname__,
                'callable': callable(value), 'unsupported_value': True}

    def check(name, expected, read, predicate):
        row = {'name': name, 'expected': expected, 'passed': False}
        try:
            observed = read()
            row['observed'] = plain(observed)
            row['passed'] = bool(predicate(observed))
        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
        rows.append(row)

    check('fixture_recipe', MARGIN_RECIPE, lambda: scene.fixture_recipe, lambda v: v == MARGIN_RECIPE)
    check('device', 'CUDA', lambda: model.device.is_cuda, lambda v: v is True)
    check('determinism', [0, 0], lambda: [int(solver._deterministic), solver._deterministic_max_records], lambda v: v == [0, 0])
    check('clocks', 'scene step/time and native step zero',
          lambda: [scene.step_id, scene.time_s, solver._step], lambda v: v == [0, 0., 0])
    check('native_time', 'float32[1] zero', lambda: d.time.numpy(),
          lambda v: v.dtype == np.float32 and v.shape == (1,) and np.array_equal(v, np.zeros(1, np.float32)))
    check('dense_world', [19, 1, False], lambda: [m.nv, d.nworld, m.is_sparse], lambda v: v == [19, 1, False])
    check('unsupported_model_features', [0, 0, 0], lambda: [m.nflex, m.ntendon, m.nplugin], lambda v: v == [0, 0, 0])
    check('solver_options', [2, 1, 3], lambda: [int(m.opt.solver), int(m.opt.cone), int(m.opt.integrator)], lambda v: v == [2, 1, 3])
    check('flags', [0, 524288], lambda: [int(m.opt.enableflags), int(m.opt.disableflags)], lambda v: v == [0, 524288])
    check('iterations', [50, 100], lambda: [m.opt.iterations, m.opt.ls_iterations], lambda v: v == [50, 100])
    check('collision_route', [False, False, False],
          lambda: [solver.use_mujoco_cpu, solver._use_mujoco_contacts, m.opt.run_collision_detection],
          lambda v: all(x is False for x in v))
    check('callbacks', 'all absent', lambda: vars(m.callback), lambda v: not any(v.values()))
    check('broad_phase_mode', 'explicit', lambda: pipeline.broad_phase_mode, lambda v: v == 'explicit')
    check('broad_phase_type', 'exact pinned BroadPhaseExplicit class',
          lambda: {'type': type(pipeline.broad_phase).__module__ + '.' + type(pipeline.broad_phase).__qualname__,
                   'exact_type': type(pipeline.broad_phase) is modules['newton._src.geometry.broad_phase_nxn'].BroadPhaseExplicit},
          lambda v: v['exact_type'])

    def pairs():
        pair_array = pipeline.shape_pairs_filtered.numpy()
        return {'pairs': pair_array, 'owned_model_array': pipeline.shape_pairs_filtered is model.shape_contact_pairs,
                'model_shape_count': model.shape_count, 'max_pairs': pipeline.shape_pairs_max,
                'exclusions_absent': pipeline.shape_pairs_excluded is None,
                'excluded_count': pipeline.shape_pairs_excluded_count,
                'candidate_capacity': narrow.max_candidate_pairs}

    def valid_pairs(v):
        a = v['pairs']
        return (a.dtype == np.int32 and a.ndim == 2 and a.shape[1] == 2 and len(a) > 0
                and v['owned_model_array'] and v['exclusions_absent'] and v['excluded_count'] == 0
                and v['max_pairs'] == v['candidate_capacity'] == len(a)
                and np.all(a >= 0) and np.all(a < v['model_shape_count'])
                and np.all(a[:, 0] != a[:, 1])
                and len(np.unique(np.sort(a, axis=1), axis=0)) == len(a))

    check('explicit_pairs', 'own int32[N,2], unique in-range pairs, exact capacity, no exclusions', pairs, valid_pairs)
    check('collision_features', [True, True, False, False, False, True, False],
          lambda: [narrow.reduce_contacts, narrow.has_meshes, narrow.has_heightfields,
                   narrow.split_gjk_mpr, narrow.speculative, narrow.hydroelastic_sdf is None, narrow.deterministic],
          lambda v: v == [True, True, False, False, False, True, False])
    check('newton_launch_dimensions', [128, 256, 32],
          lambda: [narrow.block_dim, narrow.tile_size_mesh_mesh, narrow.mesh_triangle_block_dim], lambda v: v == [128, 256, 32])
    check('constraint_dimensions', [4096, 20], lambda: [d.njmax, m.nv_pad], lambda v: v == [4096, 20])
    expected = {'actuator_velocity': 32, 'contact_jac_tiled': 32, 'small_cholesky': 64,
                'cholesky_factorize_solve': 32, 'update_gradient_JTDAJ_dense': 128,
                'update_gradient_cholesky': 64, 'linesearch_iterative': 32}
    check('mjwarp_launch_dimensions', expected,
          lambda: {name: getattr(m.block_dim, name) for name in expected}, lambda v: v == expected)
    check('mass_tiles', 'two tiles: scalar6 and dense13, nonempty addresses',
          lambda: [{'size': t.size, 'elemid_count': t.elemid.size, 'addresses': t.adr.numpy()} for t in m.M_tiles],
          lambda v: len(v) == 2 and {(t['size'], t['elemid_count'] == 0) for t in v} == {(6, True), (13, False)}
              and all(t['addresses'].size > 0 for t in v))
    check('mass_storage', [90], lambda: list(m.M_colind.shape), lambda v: v == [90])
    check('mass_factor_tail', 'qLD has no extra sparse factor tail',
          lambda: {'qLD_shape': list(d.qLD.shape), 'qLD_block_total': m.qLD_block_total},
          lambda v: len(v['qLD_shape']) == 2 and v['qLD_shape'][1] == v['qLD_block_total'])
    return {'recipe': PRECOMPILE_RECIPE, 'accepted': all(r['passed'] for r in rows),
            'predicates': rows, 'rejected': [r['name'] for r in rows if not r['passed']],
            'physics_admission': False}


def validate_recipe(scene, modules=None):
    if modules is None:
        modules, _ = admitted_sdk()
    report = recipe_admission(scene, modules)
    if not report['accepted']:
        raise FasteningFault('unsupported finite precompilation recipe: ' + ', '.join(report['rejected']))
    return report


def compilation_plan(scene, modules, wp):
    """Materialize only pure cached factories, using the actual model's fields."""
    validate_recipe(scene, modules)
    variants = (*_newton_variants(scene, modules, wp),
                *_mujoco_variants(scene.solver.mjw_model, scene.solver.mjw_data, modules, wp))
    _require(len(variants) == 39, "inventory count")
    return variants


def _newton_variants(scene, modules, wp):
    """Owned explicit-pair path; the SDK's module name also contains NXN code."""
    narrow = scene.pipeline.narrow_phase
    variants = []

    def static(name, block):
        variants.append(ModuleVariant(name, wp.get_module(modules[name].__name__), block))

    def kernel(label, value, block, *params, writer=False):
        _require(isinstance(block, int) and block > 0, "invalid block dimension")
        if writer:
            value = _contact_writer_overload(value, modules, wp)
        _require(value.is_generic is False, "uninstantiated Newton kernel: " + label)
        variants.append(ModuleVariant(label, value.module, block, tuple(params), value))

    prefix = "newton._src."
    for name, block in (("sim.collide", 256),
                        ("geometry.narrow_phase", 256), ("geometry.contact_reduction_global", 256),
                        ("geometry.contact_reduction_global", narrow.block_dim),
                        ("geometry.sdf_contact", 256)):
        static(prefix + name, block)
    kernel("newton.explicit_pairs", modules[prefix + "geometry.broad_phase_nxn"]._nxn_broadphase_precomputed_pairs, 256)
    kernel("newton.primitive", narrow.primitive_kernel, narrow.block_dim, writer=True)
    kernel("newton.gjk_mpr", narrow.narrow_phase_kernel, narrow.block_dim, writer=True)
    kernel("newton.mesh_triangle_reducer",
           modules[prefix + "geometry.narrow_phase"].mesh_triangle_contacts_to_reducer_kernel,
           narrow.mesh_triangle_block_dim)
    precomputed = (scene.model.mesh_edge_centers is not None
                   and scene.model.mesh_edge_halves is not None)
    kernel("newton.mesh_mesh", narrow.mesh_mesh_contacts_kernel_precomputed if precomputed
           else narrow.mesh_mesh_contacts_kernel, narrow.tile_size_mesh_mesh, precomputed)
    kernel("newton.export_reduced", narrow.export_reduced_contacts_kernel,
           modules[prefix + "geometry.contact_reduction_global"].EXPORT_REDUCED_CONTACTS_BLOCK_DIM,
           writer=True)

    _require(len(variants) == 11, "Newton inventory count")
    return tuple(variants)


def _contact_writer_overload(kernel, modules, wp):
    """Instantiate the pinned caller's sole generic type, without any launch.

    CollisionPipeline.collide builds sim.collide.ContactWriterData and passes
    it to these three kernels. A generic module without an overload can load
    successfully while containing no callable kernel at all.
    """
    types = modules["warp._src.types"]
    writer = modules["newton._src.sim.collide"].ContactWriterData
    _require(kernel.is_generic is True and
             [arg.label for arg in kernel.adj.args if types.type_is_generic(arg.type)] == ["writer_data"],
             "collision writer generic signature changed")
    concrete = wp.overload(kernel, {"writer_data": writer})
    _require(concrete.is_generic is False and bool(concrete.sig)
             and concrete.generic_parent is kernel and concrete.module is kernel.module
             and kernel.overloads.get(concrete.sig) is concrete
             and concrete.adj.arg_types["writer_data"] is writer,
             "collision writer overload does not bind the pinned caller type")
    return concrete


def _mujoco_variants(m, d, modules, wp):
    """Pure definition inventory, also inspectable with CPU model allocations."""
    variants = []
    def static(name, block):
        variants.append(ModuleVariant(name, wp.get_module(modules[name].__name__), block))
    def kernel(label, value, block, *params):
        _require(value.is_generic is False, "uninstantiated MJWarp kernel: " + label)
        variants.append(ModuleVariant(label, value.module, block, tuple(params), value))
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


def _mujoco_arena_layout(value):
    """Admit the exact installed ABI before reading any native pointer field.

    This finite recipe already pins its CUDA/Newton/MJWarp SDK. These offsets
    were generated with offsetof against that wheel's own headers, independently
    compared with all exposed numeric pointer views, and bound to its binary.
    Other architectures/builds require a separately audited layout. No compiler
    or native call runs here, and offsets from an unreviewed file are never used.
    """
    encoded = _MJDATA_LAYOUT.read_bytes()
    _require(hashlib.sha256(encoded).hexdigest() == _MJDATA_LAYOUT_SHA256,
             "MuJoCo arena layout changed")
    layout = json.loads(encoded)
    module = importlib.import_module("mujoco._structs")
    _require(type(value) is module.MjData, "exact native MjData owner required")
    binary = Path(module.__file__).resolve()
    _require(hashlib.sha256(binary.read_bytes()).hexdigest() == layout["structs_module_sha256"],
             "unreviewed MuJoCo arena ABI")
    _require(ctypes.sizeof(ctypes.c_void_p) == layout["pointer_bytes"]
             and sys.byteorder == layout["byteorder"], "MuJoCo arena pointer ABI")
    _require(sys.platform == "linux", "MuJoCo arena loaded-library audit requires Linux")
    # The allocator's ABI also matters: DT_NEEDED/RUNPATH permits a different
    # libmujoco via LD_LIBRARY_PATH even when _structs itself matches our pin.
    loaded = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and Path(parts[5]).name.startswith("libmujoco.so"):
            loaded.add(parts[5])
    _require(len(loaded) == 1, "unique loaded MuJoCo allocator library required")
    library = Path(next(iter(loaded)))
    _require(library.name == layout["native_library"]["basename"]
             and hashlib.sha256(library.read_bytes()).hexdigest() == layout["native_library"]["sha256"],
             "unreviewed loaded MuJoCo allocator ABI")
    for name, expected in layout["headers_sha256"].items():
        _require(hashlib.sha256((binary.parent / name).read_bytes()).hexdigest() == expected,
                 "MuJoCo arena header changed")
    _require(type(value._address) is int and value._address > 0, "native MjData address")
    _require(all(type(offset) is int and 0 <= offset <= layout["sizeof_mjData"]-layout["pointer_bytes"]
                 and offset % layout["pointer_bytes"] == 0 for offset in layout["pointer_offsets"].values()),
             "MuJoCo arena field offsets")
    return layout


def _mujoco_arena_row(value, name, array, layout):
    """Represent absence, or hash a proven native view; never hash NULL garbage."""
    offset = layout["pointer_offsets"][name]
    pointer = ctypes.c_void_p.from_address(value._address + offset).value
    row = {"shape": list(array.shape), "dtype": array.dtype.str,
           "native_pointer": pointer}
    if array.size == 0:
        # InitPyArray creates an empty Python array regardless of native pointer.
        row.update(storage="empty_native_arena_descriptor", sha256=hashlib.sha256(b"").hexdigest())
    elif pointer is None:
        _require(array.flags.owndata and array.base is None,
                 "NULL native arena descriptor unexpectedly aliases storage")
        row["storage"] = "absent_native_arena"
    else:
        _require(not array.flags.owndata and array.base is not None
                 and array.ctypes.data == pointer, "native arena descriptor is not a bound view")
        row.update(storage="native_arena", sha256=hashlib.sha256(array.tobytes()).hexdigest())
    return row


def physical_snapshot(scene, wp):
    """Hash every exposed array recursively in the owned physical object graph.

    Includes both Newton states, controls, model, collision scratch/contacts,
    MJWarp model/data, native MuJoCo data/model arrays and solver mappings. No
    whitelist of named dynamic buffers can silently omit a newly added array.
    Native MuJoCo arena pointer absence is recorded, including shape/dtype and
    transitions to allocated storage; its binding's uninitialized NULL-pointer
    return allocations are not physical buffers. The exact ABI is checked first.
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
            layout = (_mujoco_arena_layout(value) if type(value).__module__ == "mujoco._structs"
                      and type(value).__name__ == "MjData" else None)
            if layout is not None:
                rows[path + "/_arena_owner"] = {"address": value._address,
                    "layout_sha256": _MJDATA_LAYOUT_SHA256,
                    "native_library_sha256": layout["native_library"]["sha256"]}
            for name in sorted(dir(value)):
                if not name.startswith("_"):
                    item = getattr(value, name)
                    if layout is not None and name in layout["pointer_offsets"]:
                        _require(isinstance(item, np.ndarray), "MuJoCo arena descriptor type")
                        rows[path + "/" + name] = _mujoco_arena_row(value, name, item, layout)
                    elif isinstance(item, (np.ndarray, float, int)):
                        visit(item, path + "/" + name)
                    elif type(item).__module__.startswith("mujoco."):
                        # Contact/stat lists and option structs expose arrays
                        # and scalars through their own native descriptors.
                        # Retain them in seen just like other owned objects.
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
    admission = recipe_admission(scene, modules)
    scene.precompile_receipt = {"recipe": recipe, "sdk_sources": sources,
        "admission": admission, "loaded": [], "ok": False, "physics_admission": False}
    if not admission["accepted"]:
        raise FasteningFault("unsupported finite precompilation recipe: " + ", ".join(admission["rejected"]))
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
                # Unlike get_kernel_hooks, this lookup neither configures
                # shared memory nor launches a kernel. Require the actual
                # compiled forward symbol, not just a loaded empty module.
                if variant.kernel is not None:
                    _require(variant.kernel.module is variant.module
                             and variant.kernel.is_generic is False
                             and bool(executable._get_forward_cuda_kernel(variant.kernel)),
                             "loader did not retain the concrete kernel entrypoint")
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
    receipt = {"recipe": recipe, "sdk_sources": sources, "admission": admission, "loaded": loaded,
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
    return {"recipe": recipe, "sdk_sources": sources, "admission": admission, "device": device,
            "warp_version": wp.__version__,
            "variants": [row.describe() for row in plan],
            "module_hashes": [row["module_sha256"] for row in loaded],
            "compiler_options": [row["compiler_options"] for row in loaded],
            "physical_unchanged": True, "physics_solves": 0}
