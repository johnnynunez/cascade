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
            return NS(module=Module(module + "." + name))
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
    modules['mujoco_warp._src.types'].DisableBit = NS(WARMSTART=256)
    modules['mujoco_warp._src.types'].TILE_SIZE_JTDAJ_DENSE = 16
    modules['newton._src.geometry.narrow_phase'].mesh_triangle_contacts_to_reducer_kernel = NS(module=Module('triangle'))
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
                mesh_triangle_block_dim=32, counts=Buffer([0]))
    for name in ('primitive_kernel', 'narrow_phase_kernel', 'mesh_mesh_contacts_kernel',
                 'mesh_mesh_contacts_kernel_precomputed', 'export_reduced_contacts_kernel'):
        setattr(narrow, name, NS(module=Module(name)))
    class Scene(NS):
        frame_dt = 1/60
        socket_offset = np.array([.012, 0., -.115])
    scene = Scene(fixture_recipe='factory_m20_fixed_axis_margin_v2', step_id=0, time_s=0.,
        drive=False, epoch='fixture_epoch', _active=False, _targets=np.zeros((43,5)),
        model=NS(device=NS(is_cuda=True,arch=120,context="fixture_context"), mesh_edge_centers=Buffer([1]), mesh_edge_halves=Buffer([2])),
        state=NS(joint_q=Buffer([0.]), nested=NS(force=Buffer([0.]))),
        next=NS(joint_q=Buffer([0.])), control=NS(ctrl=Buffer([0.])),
        contacts=NS(count=Buffer([0])), pipeline=NS(broad_phase_mode='nxn', narrow_phase=narrow),
        solver=NS(mjw_model=m, mjw_data=d, _step=0, use_mujoco_cpu=False,
                  _use_mujoco_contacts=False, _deterministic=False, _deterministic_max_records=0, _scoped_mujoco_warp_execution=nullcontext))
    def load(module, **kwargs):
        loaded.append((module,kwargs))
        module.execs[(kwargs["device"].context,kwargs["block_dim"])]=NS(module_hash=module.get_module_hash(kwargs["block_dim"]))
    wp = NS(__version__="fixture", array=Buffer, synchronize_device=lambda _: None, ScopedDevice=lambda _: nullcontext(),
            get_module=lambda name: static.setdefault(name, Module(name)),
            load_module=load)
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
    cfg = load_robot_config('factory_m20_precompile_v1')
    profile = cfg.domains.fastening.as_dict()
    validate_factory_profile(profile)
    assert profile['precompile'] == pre.PRECOMPILE_RECIPE
    assert profile['device'] is profile['model_identity_sha256'] is None
    from cascade.apps.robot_runtime import describe_robot
    assert describe_robot(cfg)['fastening'].resources[0].admission == 'unvalidated'
    assert 'precompile' not in load_robot_config('factory_m20_mounted_margin_v2').domains.fastening.as_dict()


@pytest.mark.parametrize('selector', [None, True, '', 'all', 'force_load'])
def test_unknown_profile_selector_cannot_silently_fall_back(selector):
    profile = load_robot_config('factory_m20_precompile_v1').domains.fastening.as_dict()
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
    profile = load_robot_config('factory_m20_precompile_v1').domains.fastening.as_dict()
    prepare_factory_model(profile | {'device':'cuda:0'}, tmp_path)
    assert calls == [((scene,), {'precompile':pre.PRECOMPILE_RECIPE})]


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
    profile = load_robot_config('factory_m20_precompile_v1').domains.fastening.as_dict()
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
