"""Compiler inventory/state fencing controls. No SDK, GPU or physical solve."""
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from cascade.apps.factory_runtime import prepare_factory_model, validate_factory_profile
from cascade.config import load_robot_config
from cascade.control.fastening import FasteningFault
from cascade.sim import factory_precompile as pre
from test_factory_config import no_sdk


class Buffer:
    def __init__(self, data):
        self.data = np.asarray(data)
        self.shape, self.size = self.data.shape, self.data.size
    def numpy(self): return self.data.copy()


class Module:
    __module__ = 'warp._src.context'
    def __init__(self, name): self.name = name; self.options = {"deterministic":False}; self.execs = {}
    def get_module_hash(self, block): return hashlib.sha256(f"{self.name}:{block}".encode()).digest()


class Kernel:
    __module__ = 'warp._src.context'
    def __init__(self, module, *, generic=False):
        self.module, self.key = module, module.name + '_kernel'
        self.is_generic, self.sig, self.generic_parent = generic, '', None
        self.overloads = {}
        self.adj = NS(args=[NS(label='writer_data', type='generic')] if generic else [], arg_types={})


@dataclass
class Tile:
    size: int
    elemid: Buffer
    adr: Buffer


@pytest.fixture
def fixture(monkeypatch):
    calls, loaded, static = [], [], {}
    pins = __import__('json').loads(pre._PINS.read_text())
    modules = {name: NS(__name__=name) for name in pins}
    def factory(module, name):
        def make(*args):
            calls.append((module, name, args))
            return Kernel(Module(module + "." + name))
        return make
    factories = {
        "constraint": ("_friction_dof", "_limit_slide_hinge", "_efc_contact_init", "_efc_contact_jac_dense", "_efc_contact_update"),
        "forward": ("_qfrc_smooth", "_next_time_builder"),
        "smooth": ("_small_cholesky_factorize_solve_block", "_tile_cholesky_factorize_solve_block"),
        "support": ("mul_m_kernel",),
        "solver": ("_solve_init_dof", "_solve_init_jaref_kernel", "_update_constraint_efc",
                   "_update_constraint_init_qfrc_constraint_dense", "_update_gradient_zero_grad_dot",
                   "_update_gradient_grad", "_update_gradient_JTDAJ_dense_tiled", "_update_gradient_cholesky",
                   "_linesearch_iterative_kernel", "_solve_done"),
    }
    for name, names in factories.items():
        for fname in names:
            setattr(modules["mujoco_warp._src." + name], fname, factory(name, fname))
    class BroadPhaseExplicit: pass
    modules['newton._src.geometry.broad_phase_nxn'].BroadPhaseExplicit = BroadPhaseExplicit
    modules['newton._src.geometry.broad_phase_nxn']._nxn_broadphase_precomputed_pairs = Kernel(Module('explicit_pairs'))
    pairs = Buffer(np.array([[0, 1], [1, 2]], dtype=np.int32))
    modules['mujoco_warp._src.types'].DisableBit = NS(WARMSTART=256)
    modules['mujoco_warp._src.types'].TILE_SIZE_JTDAJ_DENSE = 16
    modules['newton._src.geometry.narrow_phase'].mesh_triangle_contacts_to_reducer_kernel = Kernel(Module('triangle'))
    modules['newton._src.sim.collide'].ContactWriterData = NS(key='ContactWriterData')
    modules['warp._src.types'].type_is_generic = lambda value: value == 'generic'
    modules['newton._src.geometry.contact_reduction_global'].EXPORT_REDUCED_CONTACTS_BLOCK_DIM = 32
    block = NS(actuator_velocity=32, contact_jac_tiled=32, small_cholesky=64,
               cholesky_factorize_solve=32, update_gradient_JTDAJ_dense=128,
               update_gradient_cholesky=64, linesearch_iterative=32)
    m = NS(nv=19, nv_pad=20, nflex=0, ntendon=0, nplugin=0, is_sparse=False,
           opt=NS(solver=2, cone=1, integrator=3, enableflags=0, disableflags=524288,
                  iterations=50, ls_iterations=100, run_collision_detection=False, warn_overflow=True),
           callback=NS(control=None, passive=None), block_dim=block, qLD_block_total=169,
           M_tiles=(Tile(6, Buffer([],), Buffer([7, 13])), Tile(13, Buffer([1]), Buffer([0]))),
           M_colind=Buffer(np.zeros(90, np.int32)))
    d = NS(nworld=1, njmax=4096, time=Buffer(np.zeros(1, np.float32)),
           qpos=Buffer(np.zeros((1,21), np.float32)), qLD=Buffer(np.zeros((1,169), np.float32)))
    narrow = NS(reduce_contacts=True, has_meshes=True, has_heightfields=False,
                split_gjk_mpr=False, speculative=False, hydroelastic_sdf=None,
                deterministic=False, block_dim=128, tile_size_mesh_mesh=256,
                mesh_triangle_block_dim=32, max_candidate_pairs=2, counts=Buffer([0]))
    for name in ('primitive_kernel', 'narrow_phase_kernel', 'mesh_mesh_contacts_kernel',
                 'mesh_mesh_contacts_kernel_precomputed', 'export_reduced_contacts_kernel'):
        setattr(narrow, name, Kernel(Module(name), generic=name in
                ('primitive_kernel', 'narrow_phase_kernel', 'export_reduced_contacts_kernel')))
    class Scene(NS):
        frame_dt = 1/60
        socket_offset = np.array([.012, 0., -.115])
    scene = Scene(fixture_recipe='factory_m20_fixed_axis_margin_v2', step_id=0, time_s=0.,
        drive=False, epoch='fixture_epoch', _active=False, _targets=np.zeros((43,5)),
        model=NS(shape_count=3, shape_contact_pairs=pairs, device=NS(is_cuda=True,arch=120,context="fixture_context"), mesh_edge_centers=Buffer([1]), mesh_edge_halves=Buffer([2])),
        state=NS(joint_q=Buffer([0.]), nested=NS(force=Buffer([0.]))),
        next=NS(joint_q=Buffer([0.])), control=NS(ctrl=Buffer([0.])),
        contacts=NS(count=Buffer([0])), pipeline=NS(broad_phase_mode='explicit', broad_phase=BroadPhaseExplicit(), shape_pairs_filtered=pairs, shape_pairs_max=2, shape_pairs_excluded=None, shape_pairs_excluded_count=0, narrow_phase=narrow),
        solver=NS(mjw_model=m, mjw_data=d, _step=0, use_mujoco_cpu=False,
                  _use_mujoco_contacts=False, _deterministic=False, _deterministic_max_records=0, _scoped_mujoco_warp_execution=nullcontext))
    def load(module, **kwargs):
        loaded.append((module,kwargs))
        module.execs[(kwargs["device"].context,kwargs["block_dim"])]=NS(
            module_hash=module.get_module_hash(kwargs["block_dim"]),
            _get_forward_cuda_kernel=lambda kernel: 1 if kernel.module is module and not kernel.is_generic else None)
    def overload(kernel, types):
        assert set(types) == {'writer_data'}
        signature = types['writer_data'].key
        if signature not in kernel.overloads:
            value = Kernel(kernel.module)
            value.key, value.sig, value.generic_parent = kernel.key, signature, kernel
            value.adj.arg_types['writer_data'] = types['writer_data']
            kernel.overloads[signature] = value
        return kernel.overloads[signature]
    wp = NS(__version__="fixture", array=Buffer, synchronize_device=lambda _: None, ScopedDevice=lambda _: nullcontext(),
            get_module=lambda name: static.setdefault(name, Module(name)),
            load_module=load, overload=overload)
    monkeypatch.setitem(sys.modules, 'warp', wp)
    monkeypatch.setattr(pre, 'admitted_sdk', lambda: (modules, {'fixture':'not_native'}))
    monkeypatch.setattr('cascade.sim.factory_model.model_fingerprint', lambda _: {'fixture':1})
    return NS(scene=scene, modules=modules, wp=wp, calls=calls, loaded=loaded)


def test_finite_plan_uses_actual_tiles_owned_handles_and_effective_launch_dims(fixture):
    f = fixture
    plan = pre.compilation_plan(f.scene, f.modules, f.wp)
    assert len(plan) == 39 and len(f.calls) == 20
    assert sum(p.label.startswith('mujoco_warp.') for p in plan) == 8
    assert len(f.loaded) == 0  # Definition construction alone never loads/executes.
    tiled = next(call for call in f.calls if call[1] == '_tile_cholesky_factorize_solve_block')
    assert tiled[2][0] is f.scene.solver.mjw_model.M_tiles[1]
    mesh = next(p for p in plan if p.label == 'newton.mesh_mesh')
    assert mesh.module is f.scene.pipeline.narrow_phase.mesh_mesh_contacts_kernel_precomputed.module
    assert mesh.parameters == (True,) and mesh.block_dim == 256
    assert next(p for p in plan if p.label == 'solver._update_gradient_cholesky').parameters == (19, False)
    explicit = next(p for p in plan if p.label == 'newton.explicit_pairs')
    assert explicit.module is f.modules['newton._src.geometry.broad_phase_nxn']._nxn_broadphase_precomputed_pairs.module
    assert not any(p.label == 'newton._src.geometry.broad_phase_nxn' for p in plan)


def test_all_admission_predicates_survive_multiple_rejections_before_any_load(fixture):
    import json
    f = fixture
    f.scene.fixture_recipe = 'unknown'
    f.scene.pipeline.broad_phase_mode = 'nxn'
    f.scene.solver.mjw_model.M_colind.shape = (89,)
    del f.scene.solver.mjw_model.block_dim.linesearch_iterative
    with pytest.raises(FasteningFault, match='fixture_recipe.*broad_phase_mode.*mjwarp_launch_dimensions.*mass_storage'):
        pre.precompile_factory(f.scene, recipe=pre.PRECOMPILE_RECIPE)
    receipt = f.scene.precompile_receipt
    report = receipt['admission']
    assert len(report['predicates']) == 22
    assert report['rejected'] == ['fixture_recipe', 'broad_phase_mode', 'mjwarp_launch_dimensions', 'mass_storage']
    assert report['predicates'][-1]['passed']  # Reads after every rejection still run.
    assert not report['accepted'] and not receipt['ok']
    assert 'AttributeError' in next(row for row in report['predicates'] if row['name'] == 'mjwarp_launch_dimensions')['error']
    json.dumps(receipt, allow_nan=False)
    assert f.loaded == f.calls == []


@pytest.mark.parametrize('fault', ['mode', 'class', 'copy', 'float', 'shape', 'duplicate', 'out_of_range', 'self', 'capacity', 'exclusion'])
def test_explicit_pair_route_requires_owned_exact_typed_inventory(fixture, fault):
    f = fixture; p = f.scene.pipeline
    if fault == 'mode': p.broad_phase_mode = 'nxn'
    elif fault == 'class': p.broad_phase = NS()
    elif fault == 'copy': p.shape_pairs_filtered = Buffer(p.shape_pairs_filtered.numpy())
    elif fault == 'float': p.shape_pairs_filtered.data = p.shape_pairs_filtered.data.astype(float)
    elif fault == 'shape': p.shape_pairs_filtered.data = p.shape_pairs_filtered.data.reshape(4)
    elif fault == 'duplicate': p.shape_pairs_filtered.data[1] = [1, 0]
    elif fault == 'out_of_range': p.shape_pairs_filtered.data[1] = [1, 3]
    elif fault == 'self': p.shape_pairs_filtered.data[1] = [1, 1]
    elif fault == 'capacity': p.shape_pairs_max += 1
    elif fault == 'exclusion': p.shape_pairs_excluded = Buffer([0])
    with pytest.raises(FasteningFault, match='broad_phase|explicit_pairs'):
        pre.precompile_factory(f.scene, recipe=pre.PRECOMPILE_RECIPE)
    assert not f.scene.precompile_receipt['admission']['accepted'] and not f.loaded


def test_rejected_admission_is_persisted_by_normal_preparation(fixture, monkeypatch, tmp_path):
    import json
    import cascade.sim.factory_model as model
    import cascade.sim.factory_observation as observation
    import cascade.sim.newton_screw_seating as seating
    f = fixture
    f.scene.pipeline.broad_phase_mode = 'nxn'
    monkeypatch.setattr(observation, 'sdk_sources', lambda: {})
    monkeypatch.setattr(seating, 'SeatingScene', lambda *a, **k: f.scene)
    monkeypatch.setattr(model, 'FactoryBoundModel',
        lambda scene, *, precompile: pre.precompile_factory(scene, recipe=precompile))
    profile = load_robot_config('factory_m20_precompile_writer_v3').domains.fastening.as_dict()
    with pytest.raises(FasteningFault, match='broad_phase_mode'):
        prepare_factory_model(profile | {'device': 'cuda:0'}, tmp_path/'sdf')
    saved = json.loads((tmp_path/'precompile.json').read_text())
    assert saved['admission']['rejected'] == ['broad_phase_mode']
    assert len(saved['admission']['predicates']) == 22 and not saved['ok']


@pytest.mark.parametrize('missing', ['sdk_class', 'pair_array', 'narrow_phase', 'solver'])
def test_missing_sdk_attribute_records_rejection_and_keeps_later_predicates(fixture, missing):
    import json
    f = fixture
    if missing == 'sdk_class':
        del f.modules['newton._src.geometry.broad_phase_nxn'].BroadPhaseExplicit
    elif missing == 'pair_array':
        del f.scene.pipeline.shape_pairs_filtered
    elif missing == 'narrow_phase':
        del f.scene.pipeline.narrow_phase
    else:
        del f.scene.solver
    with pytest.raises(FasteningFault, match='unsupported'):
        pre.precompile_factory(f.scene, recipe=pre.PRECOMPILE_RECIPE)
    report = f.scene.precompile_receipt['admission']
    assert len(report['predicates']) == 22
    assert report['predicates'][-1]['name'] == 'mass_factor_tail'
    assert any('AttributeError' in row.get('error', '') for row in report['predicates'])
    assert not report['accepted'] and f.calls == f.loaded == []
    json.dumps(f.scene.precompile_receipt, allow_nan=False)


def test_nonfinite_native_clock_is_rejected_with_serializable_report(fixture):
    import json
    fixture.scene.solver.mjw_data.time.data[0] = np.nan
    with pytest.raises(FasteningFault, match='native_time'):
        pre.precompile_factory(fixture.scene, recipe=pre.PRECOMPILE_RECIPE)
    report = fixture.scene.precompile_receipt['admission']
    assert report['rejected'] == ['native_time'] and len(report['predicates']) == 22
    json.dumps(fixture.scene.precompile_receipt, allow_nan=False)
    assert fixture.calls == fixture.loaded == []


@pytest.mark.parametrize('owner,field,value', [
    ('scene','step_id',1), ('scene','time_s',.01), ('solver','_step',1),
    ('m','nv',20), ('m','is_sparse',True), ('d','nworld',2), ('m','nflex',1),
    ('m','ntendon',1), ('m','nplugin',1), ('opt','solver',1), ('opt','cone',0),
    ('opt','integrator',0), ('opt','enableflags',1), ('opt','disableflags',0),
    ('opt','iterations',51), ('opt','ls_iterations',101), ('d','njmax',4097),
    ('narrow','split_gjk_mpr',True), ('narrow','has_heightfields',True),
    ('narrow','reduce_contacts',False), ('narrow','speculative',True),
    ('narrow','deterministic',True), ('narrow','mesh_triangle_block_dim',64),
])
def test_unreviewed_branch_refuses_before_any_factory_or_load(fixture, owner, field, value):
    f = fixture; s = f.scene
    owners = dict(scene=s, solver=s.solver, m=s.solver.mjw_model, d=s.solver.mjw_data,
                  opt=s.solver.mjw_model.opt, narrow=s.pipeline.narrow_phase)
    setattr(owners[owner], field, value)
    with pytest.raises(FasteningFault, match='unsupported'):
        pre.compilation_plan(s, f.modules, f.wp)
    assert f.calls == f.loaded == []


def test_precompile_loads_only_explicit_plan_and_preserves_every_snapshot(fixture):
    f = fixture
    result = pre.precompile_factory(f.scene, recipe=pre.PRECOMPILE_RECIPE)
    assert len(f.loaded) == 39
    assert all(row[1]['recursive'] is False and row[1]['max_workers'] == 0 for row in f.loaded)
    assert all(row[1]['device'] is f.scene.model.device for row in f.loaded)
    receipt = f.scene.precompile_receipt
    assert receipt['ok'] and receipt['physical_before'] == receipt['physical_after']
    assert receipt['step_after'] == {'step':0, 'time_s':0.}
    assert result['physics_solves'] == 0 and receipt['physics_admission'] is False


@pytest.mark.parametrize('root', ['state','next','control','contacts','model','pipeline','solver'])
def test_nested_unlisted_physical_buffer_mutation_is_detected(fixture, root):
    f = fixture
    obj = getattr(f.scene, root)
    obj.previously_unlisted = NS(deep=Buffer(np.array([3.], dtype=np.float32)))
    original_load = f.wp.load_module
    def corrupt(*args, **kwargs):
        original_load(*args, **kwargs)
        obj.previously_unlisted.deep.data[0] = 4.
    f.wp.load_module = corrupt
    with pytest.raises(FasteningFault, match='changed physical'):
        pre.precompile_factory(f.scene, recipe=pre.PRECOMPILE_RECIPE)
    assert not f.scene.precompile_receipt['ok']


def test_counter_mutation_during_load_is_not_called_preparation(fixture):
    f = fixture
    original_load = f.wp.load_module
    def corrupt(*args, **kwargs):
        original_load(*args, **kwargs)
        f.scene.solver._step += 1
    f.wp.load_module = corrupt
    with pytest.raises(FasteningFault, match='changed physical'):
        pre.precompile_factory(f.scene, recipe=pre.PRECOMPILE_RECIPE)


@pytest.mark.parametrize('field,value', [('drive',True), ('epoch','replacement'),
    ('_active',True), ('direction','loosen'), ('requested_turns',2.), ('_last_motor',.05),
    ('_unwrapped',.1), ('entry',np.ones(5)), ('bottom',np.ones(5)), ('_targets',np.ones((43,5))),
    ('_heights',np.ones(43))])
def test_scene_control_authoring_and_epoch_mutations_cannot_escape_snapshot(fixture, field, value):
    original_load = fixture.wp.load_module
    def load(*args, **kwargs):
        original_load(*args, **kwargs)
        setattr(fixture.scene, field, value)
    fixture.wp.load_module = load
    with pytest.raises(FasteningFault, match='changed physical'):
        pre.precompile_factory(fixture.scene, recipe=pre.PRECOMPILE_RECIPE)


@pytest.mark.parametrize('field,value', [('frame_dt',1/30), ('socket_offset',np.ones(3))])
def test_consumed_class_constants_are_observed_without_materializing_overrides(fixture, field, value):
    assert field not in vars(fixture.scene)
    original_load = fixture.wp.load_module
    def load(*args, **kwargs):
        original_load(*args, **kwargs)
        setattr(type(fixture.scene), field, value)
    fixture.wp.load_module = load
    with pytest.raises(FasteningFault, match='changed physical'):
        pre.precompile_factory(fixture.scene, recipe=pre.PRECOMPILE_RECIPE)


def test_scene_ik_data_is_included_and_prior_evidence_receipt_is_excluded(fixture):
    fixture.scene.ik_data = NS(qpos=Buffer(np.zeros(5)))
    fixture.scene.precompile_receipt = {'stale_evidence':object()}
    before = pre.physical_snapshot(fixture.scene, fixture.wp)
    fixture.scene.precompile_receipt = {'changed_evidence':object()}
    assert pre.physical_snapshot(fixture.scene, fixture.wp) == before
    fixture.scene.ik_data.qpos.data[0] = .1
    assert pre.physical_snapshot(fixture.scene, fixture.wp) != before


def test_failed_compiler_keeps_partial_receipt_and_original_error(fixture):
    f = fixture
    error = RuntimeError('compiler rejected module')
    def fail(*args, **kwargs): raise error
    f.wp.load_module = fail
    with pytest.raises(RuntimeError) as caught:
        pre.precompile_factory(f.scene, recipe=pre.PRECOMPILE_RECIPE)
    assert caught.value is error
    assert f.scene.precompile_receipt['physical_unchanged']
    assert not f.scene.precompile_receipt['ok']


def test_loader_silent_noop_is_not_a_successful_precompile(fixture):
    fixture.wp.load_module = lambda *args, **kwargs: None
    with pytest.raises(FasteningFault, match='exact device/block executable'):
        pre.precompile_factory(fixture.scene, recipe=pre.PRECOMPILE_RECIPE)
    assert not fixture.scene.precompile_receipt['ok']


def test_loader_error_and_failed_after_audit_are_both_retained(fixture, monkeypatch):
    snapshot = pre.physical_snapshot
    def read(scene, wp):
        if getattr(scene, 'broken', False): raise RuntimeError('readback unavailable')
        return snapshot(scene, wp)
    def fail(*args, **kwargs):
        fixture.scene.broken = True
        raise ValueError('compile error')
    monkeypatch.setattr(pre, 'physical_snapshot', read)
    fixture.wp.load_module = fail
    with pytest.raises(FasteningFault, match='changed physical') as error:
        pre.precompile_factory(fixture.scene, recipe=pre.PRECOMPILE_RECIPE)
    assert isinstance(error.value.__cause__, ValueError)
    receipt = fixture.scene.precompile_receipt
    assert 'compile error' in receipt['compiler_error']
    assert 'readback unavailable' in receipt['audit_error']
    assert receipt['physical_after'] is None and not receipt['ok']


def test_immutable_model_change_alone_refuses_identity(fixture, monkeypatch):
    value = {'generation': 0}
    original_load = fixture.wp.load_module
    def load(*args, **kwargs):
        original_load(*args, **kwargs)
        value['generation'] += 1
    fixture.wp.load_module = load
    monkeypatch.setattr('cascade.sim.factory_model.model_fingerprint', lambda _: value.copy())
    with pytest.raises(FasteningFault, match='changed physical'):
        pre.precompile_factory(fixture.scene, recipe=pre.PRECOMPILE_RECIPE)
    assert not fixture.scene.precompile_receipt['model_identity_unchanged']


def test_compiler_scope_is_active_for_factories_and_loads_and_restored(fixture):
    from contextlib import contextmanager
    events = []
    @contextmanager
    def scope():
        events.append('enter')
        yield
        events.append('exit')
    fixture.scene.solver._scoped_mujoco_warp_execution = scope
    original_load = fixture.wp.load_module
    def load(*args, **kwargs):
        assert events == ['enter']
        original_load(*args, **kwargs)
    fixture.wp.load_module = load
    pre.precompile_factory(fixture.scene, recipe=pre.PRECOMPILE_RECIPE)
    assert events == ['enter', 'exit']


def test_profile_is_opt_in_passive_and_null_pinned(monkeypatch):
    no_sdk(monkeypatch)
    cfg = load_robot_config('factory_m20_precompile_writer_v3')
    profile = cfg.domains.fastening.as_dict()
    validate_factory_profile(profile)
    assert profile['precompile'] == pre.PRECOMPILE_RECIPE
    assert profile['device'] is profile['model_identity_sha256'] is None
    from cascade.apps.robot_runtime import describe_robot
    assert describe_robot(cfg)['fastening'].resources[0].admission == 'unvalidated'
    assert 'precompile' not in load_robot_config('factory_m20_mounted_margin_v2').domains.fastening.as_dict()


@pytest.mark.parametrize('selector', [None, True, '', 'all', 'force_load', 'factory_nv19_dense_compile_v1', 'factory_nv19_explicit_compile_v2'])
def test_unknown_profile_selector_cannot_silently_fall_back(selector):
    profile = load_robot_config('factory_m20_precompile_writer_v3').domains.fastening.as_dict()
    with pytest.raises(ValueError, match='precompile'):
        validate_factory_profile(profile | {'precompile':selector})


def test_builder_passes_explicit_compilation_selection(monkeypatch, tmp_path):
    import cascade.sim.factory_model as model
    import cascade.sim.factory_observation as observation
    import cascade.sim.newton_screw_seating as seating
    monkeypatch.setattr(observation, 'sdk_sources', lambda: {})
    scene = NS(model=NS(device=NS(is_cuda=True)), precompile_receipt={"fixture":True})
    monkeypatch.setattr(seating, 'SeatingScene', lambda *args, **kw: scene)
    calls = []
    monkeypatch.setattr(model, 'FactoryBoundModel', lambda *args, **kw: calls.append((args, kw)))
    profile = load_robot_config('factory_m20_precompile_writer_v3').domains.fastening.as_dict()
    prepare_factory_model(profile | {'device':'cuda:0'}, tmp_path)
    assert calls == [((scene,), {'precompile':pre.PRECOMPILE_RECIPE})]


def test_internal_sdk_is_explicit_and_passes_through_both_guards(monkeypatch, tmp_path):
    import cascade.sim.factory_model as model
    import cascade.sim.factory_observation as observation
    import cascade.sim.newton_screw_seating as seating
    from cascade.sim.factory_sdk import INTERNAL_SDK_RECIPE
    profile = load_robot_config('factory_m20_precompile_writer_v3').domains.fastening.as_dict()
    calls = []
    monkeypatch.setattr(observation, 'sdk_sources', lambda **kw: calls.append(('sdk', kw)))
    scene = NS(model=NS(device=NS(is_cuda=True)), precompile_receipt={})
    monkeypatch.setattr(seating, 'SeatingScene', lambda *args, **kw: scene)
    monkeypatch.setattr(model, 'FactoryBoundModel', lambda *args, **kw: calls.append(('model', kw)))
    prepare_factory_model(profile | {'device': 'cuda:0', 'sdk_recipe': INTERNAL_SDK_RECIPE}, tmp_path)
    assert calls == [('sdk', {'sdk_recipe': INTERNAL_SDK_RECIPE}),
                     ('model', {'precompile': pre.PRECOMPILE_RECIPE, 'sdk_recipe': INTERNAL_SDK_RECIPE})]
    with pytest.raises(ValueError, match='SDK recipe'):
        prepare_factory_model(profile | {'device': 'cuda:0', 'sdk_recipe': 'auto'}, tmp_path)
    assert len(calls) == 2


@pytest.mark.parametrize('guard', ['readback', 'compiler'])
def test_sdk_source_mix_refused_before_native_construction(monkeypatch, tmp_path, guard):
    import cascade.sim.factory_sdk as sdk
    import cascade.sim.factory_observation as observation
    first, second = tmp_path / 'first.py', tmp_path / 'second.py'
    first.write_text('old compatible channel')
    second.write_text('new explicit source')
    pins = {'mujoco_warp._src.support': hashlib.sha256(first.read_bytes()).hexdigest(),
            'newton._src.solvers.mujoco.solver_mujoco': hashlib.sha256(second.read_bytes()).hexdigest()}
    pinfile = tmp_path / 'pins.json'
    pinfile.write_text(__import__('json').dumps(pins))
    monkeypatch.setattr(sdk, 'INTERNAL_PINS', pinfile)
    modules = {name: NS(__file__=str(path)) for name, path in zip(pins, (first, second), strict=True)}
    target = observation if guard == 'readback' else pre
    monkeypatch.setattr(target.importlib, 'import_module', modules.__getitem__)
    check = observation.sdk_sources if guard == 'readback' else pre.admitted_sdk
    check(sdk.INTERNAL_SDK_RECIPE)
    second.write_text('legacy solver mixed into new source set')
    with pytest.raises(FasteningFault, match='SDK'):
        check(sdk.INTERNAL_SDK_RECIPE)


def test_persistence_failure_preserves_primary_error_without_python311_notes(monkeypatch, tmp_path, caplog):
    import cascade.apps.factory_runtime as runtime
    import cascade.sim.factory_model as model
    import cascade.sim.factory_observation as observation
    import cascade.sim.newton_screw_seating as seating
    class OriginalFailure(RuntimeError):
        add_note = None  # Python 3.10 exception interface.
    primary = OriginalFailure('compiler failed first')
    scene = NS(model=NS(device=NS(is_cuda=True)), precompile_receipt={'ok':False})
    monkeypatch.setattr(observation, 'sdk_sources', lambda: {})
    monkeypatch.setattr(seating, 'SeatingScene', lambda *args, **kw: scene)
    def construct(*args, **kwargs): raise primary
    def write(*args): raise OSError('receipt sink failed')
    monkeypatch.setattr(model, 'FactoryBoundModel', construct)
    monkeypatch.setattr(runtime, '_write', write)
    profile = load_robot_config('factory_m20_precompile_writer_v3').domains.fastening.as_dict()
    with pytest.raises(OriginalFailure) as caught:
        prepare_factory_model(profile | {'device':'cuda:0'}, tmp_path)
    assert caught.value is primary
    assert 'receipt sink failed' in caplog.text


def test_identity_descriptor_is_repeatable_but_binds_compilation_device(fixture):
    from cascade.sim.factory_model import _digest
    a = pre.precompile_factory(fixture.scene, recipe=pre.PRECOMPILE_RECIPE)
    b = pre.precompile_factory(fixture.scene, recipe=pre.PRECOMPILE_RECIPE)
    assert _digest(a) == _digest(b)  # Timings and uninitialized scratch are not identity.
    assert "wall_s" not in a and a["recipe"] == pre.PRECOMPILE_RECIPE
    fixture.scene.model.device.arch = 121
    c = pre.precompile_factory(fixture.scene, recipe=pre.PRECOMPILE_RECIPE)
    assert _digest(c) != _digest(a)


def test_snapshot_preserves_dtype_shape_nan_bits_and_nested_dataclasses(fixture):
    f = fixture
    original = pre.physical_snapshot(f.scene, f.wp)
    f.scene.state.sample = np.array([np.nan], dtype=np.float32)
    snapshot = pre.physical_snapshot(f.scene, f.wp)
    assert snapshot != original
    assert snapshot == pre.physical_snapshot(f.scene, f.wp)
    f.scene.state.sample = f.scene.state.sample.reshape(1,1)
    assert snapshot != pre.physical_snapshot(f.scene, f.wp)
