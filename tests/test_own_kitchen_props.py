"""Original visual geometry must not alter the task's independent physics."""
from collections import Counter
import importlib.util
import math
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "demo" / "own_kitchen_props.py"
spec = importlib.util.spec_from_file_location("own_kitchen_props", MODULE_PATH)
props = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = props
spec.loader.exec_module(props)


def _volume(surface):
    volume = 0.0
    for face in surface.faces:
        a = surface.points[face[0]]
        for index in range(1, len(face) - 1):
            b, c = surface.points[face[index]], surface.points[face[index + 1]]
            cross = (b[1] * c[2] - b[2] * c[1],
                     b[2] * c[0] - b[0] * c[2],
                     b[0] * c[1] - b[1] * c[0])
            volume += sum(a[k] * cross[k] for k in range(3)) / 6
    return volume


def _assert_closed(surface):
    edges = Counter()
    directed = Counter()
    for face in surface.faces:
        assert len(face) >= 3 and len(set(face)) == len(face)
        for a, b in zip(face, face[1:] + face[:1]):
            assert 0 <= a < len(surface.points) and 0 <= b < len(surface.points)
            edges[tuple(sorted((a, b)))] += 1
            directed[a, b] += 1
    assert set(edges.values()) == {2}
    assert all(directed[a, b] == directed[b, a] for a, b in directed)
    assert _volume(surface) > 0
    assert all(math.isfinite(value) for point in surface.points for value in point)
    assert len(surface.normals) == len(surface.points)
    assert all(sum(value * value for value in normal) == pytest.approx(1)
               for normal in surface.normals)


@pytest.mark.parametrize("make_surface", [props.orange_surface, props.orange_half_surface])
def test_citrus_is_closed_deterministic_and_inside_unchanged_proxy(make_surface):
    surface = make_surface(segments=48, rings=24)
    assert surface == make_surface(segments=48, rings=24)
    assert surface.points != make_surface(seed=82, segments=48, rings=24).points
    _assert_closed(surface)
    for axis, dimension in enumerate(props.ORANGE_DIMENSIONS):
        assert min(point[axis] for point in surface.points) >= -dimension / 2 - 1e-12
        assert max(point[axis] for point in surface.points) <= dimension / 2 + 1e-12
    assert min(point[2] for point in surface.points) == -props.ORANGE_DIMENSIONS[2] / 2
    assert len(surface.colors) == len(surface.roughness) == len(surface.points)
    assert all(0 <= value <= 1 for color in surface.colors for value in color)
    assert all(.4 < value < .6 for value in surface.roughness)


def test_platter_has_real_thickness_and_a_flat_support_foot():
    surface = props.bowl_surface()
    _assert_closed(surface)
    assert min(point[2] for point in surface.points) == 0
    assert max(point[2] for point in surface.points) == pytest.approx(.0275)
    assert min(point[0] for point in surface.points) == pytest.approx(-.15)
    assert max(point[0] for point in surface.points) == pytest.approx(.15)
    foot = [point for point in surface.points if point[2] == 0]
    assert len(foot) >= 128
    assert min(math.hypot(x, y) for x, y, _ in foot) > .04


def test_pulp_segments_are_closed_outward_surfaces():
    for index in range(10):
        surface = props._pulp_segment(.035, .035, 0, index, 81)
        _assert_closed(surface)
        assert _volume(surface) < 1e-7


def test_decorative_fruit_rests_on_bowl_without_intersection():
    rows = props.bowl_fruit_layout()
    assert len(rows) == 6 and sum(row["half"] for row in rows) == 2
    world_points = []
    for row in rows:
        make_surface = props.orange_half_surface if row["half"] else props.orange_surface
        surface = make_surface(row["dimensions"], seed=row["seed"], segments=64, rings=32)
        points = [props._rotate(point, row["rotation"]) for point in surface.points]
        x, y, z = row["position"]
        clearance = [pz + z - props._inner_height(math.hypot(px + x, py + y))
                     for px, py, pz in points]
        assert min(clearance) == pytest.approx(.00004, abs=1e-10)
        world_points.append([(p[0] + x, p[1] + y, p[2] + z) for p in points])
    # Independent separating-plane check on each pair. Slightly separated
    # shells look natural and cannot geometrically intersect one another.
    for index, first in enumerate(rows):
        for other in range(index + 1, len(rows)):
            direction = [rows[other]["position"][k] - first["position"][k] for k in range(3)]
            a = max(sum(point[k] * direction[k] for k in range(3)) for point in world_points[index])
            b = min(sum(point[k] * direction[k] for k in range(3)) for point in world_points[other])
            assert a < b


@pytest.mark.parametrize("dimensions", [(0, 1, 1), (1, math.nan, 1), (1, 1), (-1, 1, 1)])
def test_invalid_dimensions_fail_before_authoring(dimensions):
    with pytest.raises(ValueError):
        props.orange_surface(dimensions)


def test_usd_visuals_have_no_physics_or_external_dependencies():
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    props.author_orange(stage, "/Orange", segments=48, rings=24)
    props.author_bowl_of_oranges(stage, "/Bowl", position=(.7514, .0106, 0), diameter=.3023)
    for prim in stage.Traverse():
        assert not any(name.lower().startswith(("physics", "physx", "newton"))
                       for name in prim.GetAppliedSchemas())
        assert not prim.GetMetadata("references")
        assert not prim.GetMetadata("payload")
        for attribute in prim.GetAttributes():
            assert attribute.GetTypeName() not in (Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray)
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            assert mesh.GetNormalsInterpolation() == UsdGeom.Tokens.vertex
            assert len(mesh.GetNormalsAttr().Get()) == len(mesh.GetPointsAttr().Get())
            assert mesh.GetSubdivisionSchemeAttr().Get() == UsdGeom.Tokens.none
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    orange = bbox.ComputeWorldBound(stage.GetPrimAtPath("/Orange")).ComputeAlignedRange()
    for axis, dimension in enumerate(props.ORANGE_DIMENSIONS):
        assert orange.GetMin()[axis] >= -dimension / 2 - 1e-8
        assert orange.GetMax()[axis] <= dimension / 2 + 1e-8
    bowl = bbox.ComputeWorldBound(stage.GetPrimAtPath("/Bowl")).ComputeAlignedRange()
    assert bowl.GetMin()[0] > .58 and bowl.GetMax()[0] < .94
    assert bowl.GetMin()[2] == 0
    assert len(stage.GetUsedLayers()) == 2  # In-memory root plus its empty session layer.
