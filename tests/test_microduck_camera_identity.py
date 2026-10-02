"""Real USD authoring with a sensor double; no Kit/render acceptance claim."""
import hashlib
import json
from types import SimpleNamespace as NS

import pytest

from cascade.sim.microduck_newton import (
    OVERVIEW_CAMERA, OVERVIEW_RENDER_PRODUCT, create_overview_sensor, synchronize_camera_authoring,
)


@pytest.fixture
def scene():
    Usd = pytest.importorskip('pxr.Usd')
    Sdf = pytest.importorskip('pxr.Sdf')
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim('/World', 'Xform')
    stage.DefinePrim(OVERVIEW_CAMERA, 'Camera')
    body = stage.DefinePrim('/World/Robot', 'Xform')
    body.CreateAttribute('physics:mass', Sdf.ValueTypeNames.Float).Set(1.)
    return stage


def sensor_factory(stage, events, *, fallback=False, mutate=None):
    def create(camera, *, resolution, annotators):
        assert camera.paths == [OVERVIEW_CAMERA]
        assert resolution == (480, 640) and annotators == ['rgb']
        # The installed SDK discovers authored RenderProducts outside /Render
        # via their camera relationship, before its hash(self) fallback.
        found = [prim for prim in stage.Traverse()
                 if not str(prim.GetPath()).startswith('/Render')
                 and prim.GetTypeName() == 'RenderProduct'
                 and [str(p) for p in prim.GetRelationship('camera').GetTargets()] == [OVERVIEW_CAMERA]]
        assert len(found) == 1
        product = found[0]
        # Hydra validates the complete RenderProduct, then requires one USD
        # update before attachment. Discovery alone did not catch that failure.
        assert events == ['sync']
        variables = product.GetRelationship('orderedVars').GetTargets()
        assert len(variables) == 1
        var = stage.GetPrimAtPath(variables[0])
        assert var.GetTypeName() == 'RenderVar'
        assert var.GetAttribute('sourceName').Get() == 'LdrColor'
        if fallback:
            product = stage.DefinePrim('/Render/camera_sensor_123456789', 'RenderProduct')
        if mutate:
            mutate(product)
        return NS(render_product=NS(GetPrim=lambda: product), resolution=resolution)
    return create


def make_sensor(stage, **kwargs):
    events = []
    return create_overview_sensor(stage, NS(paths=[OVERVIEW_CAMERA]), sensor_factory(stage, events, **kwargs),
                                  sync_renderer=lambda: events.append('sync'))


def test_two_sensor_instances_produce_identical_authored_scene_without_normalization(scene):
    Usd = pytest.importorskip('pxr.Usd')
    initial = scene.GetRootLayer().ExportToString()
    sensor1 = make_sensor(scene)
    other = Usd.Stage.CreateInMemory()
    other.GetRootLayer().ImportFromString(initial)
    sensor2 = make_sensor(other)
    assert sensor1 is not sensor2
    a, b = scene.GetRootLayer().ExportToString(), other.GetRootLayer().ExportToString()
    assert a == b
    assert 'camera_sensor_' not in a
    assert str(sensor1.render_product.GetPrim().GetPath()) == OVERVIEW_RENDER_PRODUCT
    # The fix preserves meaningful physical data in the same scene digest.
    other.GetPrimAtPath('/World/Robot').GetAttribute('physics:mass').Set(2.)
    assert hashlib.sha256(a.encode()).digest() != hashlib.sha256(other.GetRootLayer().ExportToString().encode()).digest()


@pytest.mark.parametrize('bad', ['fallback', 'wrong_resolution', 'wrong_camera'])
def test_sdk_fallback_or_mismatched_product_fails_instead_of_hiding_identity_change(scene, bad):
    Sdf = pytest.importorskip('pxr.Sdf')
    Gf = pytest.importorskip('pxr.Gf')
    def mutate(product):
        if bad == 'wrong_resolution':
            product.GetAttribute('resolution').Set(Gf.Vec2i(320, 240))
        elif bad == 'wrong_camera':
            product.GetRelationship('camera').SetTargets([Sdf.Path('/World/OtherCamera')])
    with pytest.raises(RuntimeError, match='did not adopt') as exc:
        make_sensor(scene, fallback=bad == 'fallback', mutate=mutate)
    observed = json.loads(str(exc.value).split(': ', 1)[1])
    assert set(observed) == {'path', 'camera_targets', 'render_resolution', 'sensor_resolution', 'ordered_vars'}
    if bad == 'fallback':
        assert observed['path'] == '/Render/camera_sensor_123456789'
        assert observed['render_resolution'] is None
    elif bad == 'wrong_resolution':
        assert observed['render_resolution'] == [320, 240]
    else:
        assert observed['camera_targets'] == ['/World/OtherCamera']


def test_existing_product_is_never_overwritten(scene):
    occupied = scene.DefinePrim(OVERVIEW_RENDER_PRODUCT, 'Xform')
    before = scene.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match='already occupied'):
        make_sensor(scene)
    assert occupied.GetTypeName() == 'Xform'
    assert scene.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize('change', [None, 'native_step', 'native_time', 'manager_step', 'manager_time',
                                     'timeline_time', 'playing', 'initialized'])
def test_authoring_update_cannot_advance_or_initialize_physics(change):
    state = dict(native_step=0, native_time=0., manager_step=0, manager_time=0.,
                 timeline_time=0., playing=False, initialized=False)
    native = NS(simulation_step_count=0, sim_time=0., initialized=False)
    timeline = NS(is_stopped=lambda: not state['playing'], get_current_time=lambda: state['timeline_time'])
    manager = NS(get_num_physics_steps=lambda: state['manager_step'],
                 get_simulation_time=lambda: state['manager_time'])
    updates = []
    def update():
        updates.append(1)
        if change:
            state[change] = True if change in ('playing', 'initialized') else 1
        native.simulation_step_count = state['native_step']
        native.sim_time = state['native_time']
        native.initialized = state['initialized']
    if change:
        with pytest.raises(RuntimeError, match='advanced physics'):
            synchronize_camera_authoring(NS(update=update), timeline, manager, native)
    else:
        receipt = synchronize_camera_authoring(NS(update=update), timeline, manager, native)
        assert receipt['before'] == receipt['after']
        assert receipt['app_updates'] == 1
    assert updates == [1]


def test_authoring_update_refuses_running_timeline_before_update():
    timeline = NS(is_stopped=lambda: False, get_current_time=lambda: 0.)
    manager = NS(get_num_physics_steps=lambda: 0, get_simulation_time=lambda: 0.)
    native = NS(initialized=False, simulation_step_count=0, sim_time=0.)
    def forbidden_update():
        pytest.fail('app update must not run during physics')
    with pytest.raises(RuntimeError, match='requires stopped zero clocks'):
        synchronize_camera_authoring(NS(update=forbidden_update), timeline, manager, native)
