"""Explicit convex-hull vertex limits for every admitted MicroDuck collision mesh.

Newton's USD importer resolves only an AUTHORED limit (newton:maxHullVertices,
mjc:maxhullvert, physxConvexHullCollision:hullVertexLimit) and otherwise caps
every hull at 64 vertices; the admitted MJCF conversions author none. The
authoring tests use real USD composition (the Isaac Lab asset instances its
colliders, which fake prims cannot reproduce) and skip without pxr; the native
readback tests use CPU doubles of the Newton model arrays. No Kit, no claim of
locomotion or physical admission.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pytest

from cascade.sim.microduck_newton import (COMPLETE_HULL, HULL_LIMIT_ATTRIBUTES, author_collision_hull_limits,
                                          read_native_collision_hulls)

BOX = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
FACES = [0, 1, 2, 3, 4, 7, 6, 5, 0, 4, 5, 1, 1, 5, 6, 2, 2, 6, 7, 3, 3, 7, 4, 0]

ASSET = '''#usda 1.0
(
    defaultPrim = "microduck"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "microduck"
{
    def Xform "Geometry"
    {
        def Xform "trunk_base" (prepend apiSchemas = ["PhysicsRigidBodyAPI"])
        {
            def Xform "left_foot_collision" (
                instanceable = true
                prepend references = </Flattened_Prototype_1>
            ) {}
            def Xform "right_foot_collision" (
                instanceable = true
                prepend references = </Flattened_Prototype_1>
            ) {}
            def Mesh "shell" (prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"])
            {
                uniform token physics:approximation = "convexHull"
                custom int mjc:maxhullvert = 16
                point3f[] points = %(points)s
                int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
                int[] faceVertexIndices = %(faces)s
            }
            def Mesh "disabled" (prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"])
            {
                bool physics:collisionEnabled = false
                uniform token physics:approximation = "convexHull"
                point3f[] points = %(points)s
                int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
                int[] faceVertexIndices = %(faces)s
            }
            def Mesh "visual"
            {
                point3f[] points = %(points)s
                int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
                int[] faceVertexIndices = %(faces)s
            }
        }
    }
}

def Xform "Flattened_Prototype_1"
{
    def Mesh "sole" (prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"])
    {
        uniform token physics:approximation = "convexHull"
        point3f[] points = %(points)s
        int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
        int[] faceVertexIndices = %(faces)s
    }
}
''' % {'points': str(BOX), 'faces': str(FACES)}

ROOT = '/World/MicroDuck'
TRUNK = ROOT + '/Geometry/trunk_base'
SOLES = (TRUNK + '/left_foot_collision/sole', TRUNK + '/right_foot_collision/sole')


@pytest.fixture
def asset(tmp_path):
    pytest.importorskip('pxr', reason='USD python bindings are optional')
    path = tmp_path / 'microduck_fixture.usda'
    path.write_text(ASSET)
    return path


def runtime_stage(asset):
    """Our owner's anonymous runtime layer referencing the pinned asset, like the backend."""
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World')
    root = UsdGeom.Xform.Define(stage, ROOT).GetPrim()
    root.GetReferences().AddReference(str(asset))
    return stage


def test_fixture_reproduces_the_asset_structure_newton_imports(asset):
    from pxr import Usd, UsdPhysics
    stage = runtime_stage(asset)
    proxies = [p for p in Usd.PrimRange(stage.GetPrimAtPath(ROOT), Usd.TraverseInstanceProxies())
               if p.HasAPI(UsdPhysics.CollisionAPI) and p.IsInstanceProxy()]
    assert {str(p.GetPath()) for p in proxies} == set(SOLES)
    for prim in proxies:
        assert all(not prim.GetAttribute(n) or not prim.GetAttribute(n).HasAuthoredValue()
                   for n in HULL_LIMIT_ATTRIBUTES)


def test_unauthored_limits_become_complete_hulls_and_authored_limits_are_kept(asset):
    from pxr import Usd
    stage = runtime_stage(asset)
    before = asset.read_bytes()
    record = author_collision_hull_limits(stage, root_path=ROOT)
    assert record['collision_meshes'] == 3 and record['default_limit'] == COMPLETE_HULL == -1
    assert sorted(record['authored']) == sorted(SOLES)
    assert record['kept'] == [{'path': TRUNK + '/shell', 'limits': {'mjc:maxhullvert': 16}}]
    assert sorted(record['deinstanced']) == [TRUNK + '/left_foot_collision', TRUNK + '/right_foot_collision']
    # What the importer resolves: an authored value by name, through composition.
    for path in SOLES:
        prim = stage.GetPrimAtPath(path)
        assert prim.IsValid() and not prim.IsInstanceProxy()
        attr = prim.GetAttribute('newton:maxHullVertices')
        assert attr.HasAuthoredValue() and attr.Get() == -1
    assert stage.GetPrimAtPath(TRUNK + '/shell').GetAttribute('mjc:maxhullvert').Get() == 16
    assert not stage.GetPrimAtPath(TRUNK + '/disabled').GetAttribute('newton:maxHullVertices')
    assert asset.read_bytes() == before, 'the pinned asset bytes never change'
    # The edit lives in the exported runtime layer and survives a reopen.
    exported = asset.parent / 'runtime-scene.usda'
    stage.GetRootLayer().Export(str(exported))
    reopened = Usd.Stage.Open(str(exported))
    assert all(reopened.GetPrimAtPath(p).GetAttribute('newton:maxHullVertices').Get() == -1 for p in SOLES)
    assert 'newton:maxHullVertices = -1' in exported.read_text()


def test_authoring_is_idempotent_and_records_the_same_limits_twice(asset):
    stage = runtime_stage(asset)
    first = author_collision_hull_limits(stage, root_path=ROOT)
    second = author_collision_hull_limits(stage, root_path=ROOT)
    assert second['authored'] == [] and second['deinstanced'] == []
    assert sorted(r['path'] for r in second['kept']) == sorted(first['authored'] + [r['path'] for r in first['kept']])
    assert all(r['limits'] in ({'newton:maxHullVertices': -1}, {'mjc:maxhullvert': 16}) for r in second['kept'])


def test_conflicting_or_foreign_authoring_is_refused_before_any_edit(asset):
    from pxr import Sdf
    stage = runtime_stage(asset)
    shell = stage.GetPrimAtPath(TRUNK + '/shell')
    shell.CreateAttribute('physxConvexHullCollision:hullVertexLimit', Sdf.ValueTypeNames.Int).Set(64)
    with pytest.raises(ValueError, match='conflicting collision hull limits'):
        author_collision_hull_limits(stage, root_path=ROOT)
    with pytest.raises(ValueError, match='robot root missing'):
        author_collision_hull_limits(stage, root_path='/World/Nobody')


def test_edits_require_our_anonymous_runtime_layer(asset, tmp_path):
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.CreateNew(str(tmp_path / 'scene.usda'))
    UsdGeom.Xform.Define(stage, ROOT).GetPrim().GetReferences().AddReference(str(asset))
    with pytest.raises(ValueError, match='anonymous runtime layer'):
        author_collision_hull_limits(stage, root_path=ROOT)


def test_non_convex_approximation_is_not_silently_limited(asset):
    stage = runtime_stage(asset)
    stage.GetPrimAtPath(TRUNK + '/shell').GetAttribute('physics:approximation').Set('convexDecomposition')
    with pytest.raises(ValueError, match='unexpected collision approximation'):
        author_collision_hull_limits(stage, root_path=ROOT)


# --- native readback ---------------------------------------------------------------

def model(labels, types, sources, flags=None):
    flags = [2] * len(labels) if flags is None else flags
    return NS(shape_label=list(labels), shape_type=NS(numpy=lambda: np.array(types, np.int32)),
              shape_flags=NS(numpy=lambda: np.array(flags, np.int32)), shape_source=list(sources))


def limits(authored, kept=()):
    return {'root': ROOT, 'authored': list(authored), 'kept': [{'path': p, 'limits': {'mjc:maxhullvert': v}} for p, v in kept]}


def hull(n, limit):
    return NS(vertices=np.zeros((n, 3), np.float32), maxhullvert=limit)


def test_readback_records_the_hull_newton_built_for_every_colliding_mesh():
    m = model([SOLES[0], SOLES[1], TRUNK + '/shell', '/World/Ground', ROOT + '/Geometry/box', ROOT + '/Geometry/visual'],
              [7, 7, 7, 0, 2, 7], [hull(4964, -1), hull(5029, -1), hull(16, 16), None, None, hull(64, 64)],
              flags=[3, 3, 2, 2, 2, 1])
    record = read_native_collision_hulls(m, root_path=ROOT, limits=limits(SOLES, [(TRUNK + '/shell', 16)]),
                                         mesh_types={7, 8}, collide_flag=2)
    assert record['total_hull_vertices'] == 4964 + 5029 + 16
    assert record['shapes'][0] == {'label': SOLES[0], 'maxhullvert': -1, 'hull_vertices': 4964}
    assert record['shapes'][2]['maxhullvert'] == 16
    assert record['non_colliding_meshes'] == 1  # the visual mesh keeps Newton's cap but never collides


@pytest.mark.parametrize('bad', ['engine_cap', 'missing_shape', 'unlimited_shape', 'no_hull_data'])
def test_readback_refutes_limits_that_did_not_reach_the_native_model(bad):
    sources = [hull(64 if bad == 'engine_cap' else 4964, 64 if bad == 'engine_cap' else -1), hull(5029, -1)]
    labels, types = [SOLES[0], SOLES[1]], [7, 7]
    expected = limits(SOLES)
    if bad == 'missing_shape':
        labels, types, sources = labels[:1], types[:1], sources[:1]
    if bad == 'unlimited_shape':
        labels.append(TRUNK + '/extra'); types.append(7); sources.append(hull(64, 64))
    if bad == 'no_hull_data':
        sources[0] = NS(vertices=None, maxhullvert=-1)
    with pytest.raises(ValueError):
        read_native_collision_hulls(model(labels, types, sources), root_path=ROOT, limits=expected,
                                    mesh_types={7, 8}, collide_flag=2)
