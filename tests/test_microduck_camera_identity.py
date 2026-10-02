"""Real USD authoring with a sensor double; no Kit/render acceptance claim."""
import hashlib
from types import SimpleNamespace as NS

import pytest

from cascade.sim.microduck_newton import (
    OVERVIEW_CAMERA, OVERVIEW_RENDER_PRODUCT, create_overview_sensor,
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


def sensor_factory(stage, *, fallback=False, mutate=None):
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
        if fallback:
            product = stage.DefinePrim('/Render/camera_sensor_123456789', 'RenderProduct')
        if mutate:
            mutate(product)
        return NS(render_product=NS(GetPrim=lambda: product), resolution=resolution)
    return create


def test_two_sensor_instances_produce_identical_authored_scene_without_normalization(scene):
    Usd = pytest.importorskip('pxr.Usd')
    initial = scene.GetRootLayer().ExportToString()
    sensor1 = create_overview_sensor(scene, NS(paths=[OVERVIEW_CAMERA]), sensor_factory(scene))
    other = Usd.Stage.CreateInMemory()
    other.GetRootLayer().ImportFromString(initial)
    sensor2 = create_overview_sensor(other, NS(paths=[OVERVIEW_CAMERA]), sensor_factory(other))
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
    with pytest.raises(RuntimeError, match='did not adopt'):
        create_overview_sensor(scene, NS(paths=[OVERVIEW_CAMERA]),
                               sensor_factory(scene, fallback=bad == 'fallback', mutate=mutate))


def test_existing_product_is_never_overwritten(scene):
    occupied = scene.DefinePrim(OVERVIEW_RENDER_PRODUCT, 'Xform')
    before = scene.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match='already occupied'):
        create_overview_sensor(scene, NS(paths=[OVERVIEW_CAMERA]), sensor_factory(scene))
    assert occupied.GetTypeName() == 'Xform'
    assert scene.GetRootLayer().ExportToString() == before
