"""Concrete contact-writer compilation: no launch, solve or GPU in these controls."""
from types import SimpleNamespace as NS

import pytest

from cascade.config import load_robot_config
from cascade.control.fastening import FasteningFault
from cascade.sim import factory_precompile as pre
from test_factory_precompile import fixture as fixture, Kernel, Module  # noqa: F401


WRITERS = {'newton.primitive', 'newton.gjk_mpr', 'newton.export_reduced'}


def test_every_writer_variant_has_the_concrete_caller_type_before_module_loading(fixture):
    f = fixture
    plan = pre.compilation_plan(f.scene, f.modules, f.wp)
    selected = [v for v in plan if v.label in WRITERS]
    assert len(selected) == 3 and len(plan) == 39 and not f.loaded
    writer = f.modules['newton._src.sim.collide'].ContactWriterData
    for variant in selected:
        kernel = getattr(variant, 'kernel', None)
        assert kernel is not None, 'generic module alone does not identify an executable kernel'
        assert not kernel.is_generic and kernel.sig
        assert kernel.adj.arg_types['writer_data'] is writer
        assert kernel.generic_parent.overloads[kernel.sig] is kernel
        assert variant.describe()['entrypoint'] == {'key':kernel.key, 'signature':kernel.sig}
    repeated = pre.compilation_plan(f.scene, f.modules, f.wp)
    assert [v.describe() for v in repeated] == [v.describe() for v in plan]
    assert not f.loaded and f.scene.step_id == 0


@pytest.mark.parametrize('fault', ['not_generic', 'unknown_argument', 'empty_signature',
                                 'still_generic', 'wrong_parent', 'wrong_module',
                                 'wrong_writer', 'unregistered'])
def test_unknown_or_unbound_writer_specialization_is_rejected_before_loading(fixture, fault):
    f = fixture
    source = f.scene.pipeline.narrow_phase.primitive_kernel
    original = f.wp.overload
    if fault == 'not_generic':
        source.is_generic = False
    elif fault == 'unknown_argument':
        source.adj.args.append(NS(label='other', type='generic'))
    else:
        def broken(kernel, types):
            result = original(kernel, types)
            if fault == 'empty_signature': result.sig = ''
            if fault == 'still_generic': result.is_generic = True
            if fault == 'wrong_parent': result.generic_parent = object()
            if fault == 'wrong_module': result.module = Module('foreign')
            if fault == 'wrong_writer': result.adj.arg_types['writer_data'] = object()
            if fault == 'unregistered': kernel.overloads.clear()
            return result
        f.wp.overload = broken
    with pytest.raises(FasteningFault, match='collision writer'):
        pre.compilation_plan(f.scene, f.modules, f.wp)
    assert not f.loaded and f.scene.step_id == 0


def test_loaded_module_hash_without_a_forward_symbol_does_not_pass(fixture):
    f = fixture
    load = f.wp.load_module
    def empty_module(module, **kwargs):
        load(module, **kwargs)
        module.execs[(kwargs['device'].context, kwargs['block_dim'])]._get_forward_cuda_kernel = lambda _: None
    f.wp.load_module = empty_module
    with pytest.raises(FasteningFault, match='concrete kernel entrypoint'):
        pre.precompile_factory(f.scene, recipe=pre.PRECOMPILE_RECIPE)
    receipt = f.scene.precompile_receipt
    assert not receipt['ok'] and receipt['physical_unchanged']
    assert receipt['step_before'] == receipt['step_after'] == {'step':0, 'time_s':0.}
    assert 'entrypoint' in receipt['compiler_error']


def test_unreviewed_generic_in_another_selected_target_is_refused(fixture):
    f = fixture
    f.modules['newton._src.geometry.narrow_phase'].mesh_triangle_contacts_to_reducer_kernel.is_generic = True
    with pytest.raises(FasteningFault, match='uninstantiated Newton'):
        pre.compilation_plan(f.scene, f.modules, f.wp)
    assert not f.loaded


def test_previous_empty_module_recipe_is_withdrawn_without_reinterpretation():
    with pytest.raises(ValueError, match='precompile'):
        load_robot_config('factory_m20_precompile_explicit_v2')
