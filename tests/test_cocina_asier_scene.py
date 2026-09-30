"""Geometry and material admission for the actual released room, without Isaac."""
from pathlib import Path
import sys

import pytest

pytest.importorskip("pxr")
from pxr import Sdf, Usd, UsdGeom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
from cocina_asier import author_kitchen
from own_kitchen import DEFAULT_COUNTER


@pytest.fixture(scope="module")
def room():
    if not (ROOT / "demo/scene/cocina_asier/cocina_asier.usdc").is_file():
        pytest.skip("Run scripts/kitchen_assets.py to fetch the pinned room")
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    author_kitchen(stage)
    return stage


def test_actual_worktop_matches_all_six_collision_box_faces(room):
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(
        room.GetPrimAtPath("/Kitchen/Architecture/Island/Countertop")).ComputeAlignedRange()
    assert tuple(bounds.GetMin()) == pytest.approx(DEFAULT_COUNTER["min"], abs=1e-7)
    assert tuple(bounds.GetMax()) == pytest.approx(DEFAULT_COUNTER["max"], abs=1e-7)


def test_room_keeps_expanded_task_volume_clear(room):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    lower, upper = (-.2598733, -.4790851, .00001), (.55, .4795905, .9)
    intrusions = []
    for prim in room.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if all(bounds.GetMin()[i] < upper[i] and bounds.GetMax()[i] > lower[i]
                   for i in range(3)):
                intrusions.append(str(prim.GetPath()))
    assert intrusions == []


def test_room_has_only_packaged_wood_maps_and_no_simulation(room):
    assets = set()
    for prim in room.Traverse():
        assert not any(str(name).lower().startswith(("physics", "physx", "newton"))
                       for name in prim.GetAppliedSchemas())
        for attr in prim.GetAttributes():
            if attr.GetTypeName() == Sdf.ValueTypeNames.Asset and attr.Get():
                value = attr.Get()
                assert value.resolvedPath and Path(value.resolvedPath).is_file()
                assert Path(value.resolvedPath).parent == ROOT / "demo/scene/cocina_asier/textures"
                assets.add(Path(value.resolvedPath).name)
    assert assets == {prefix + "_2K-JPG_" + kind + ".jpg"
                      for prefix in ("Wood044", "WoodFloor051") for kind in ("Color", "Roughness")}


def test_platter_is_supported_beyond_task_and_windows_are_closed(room):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    bounds = cache.ComputeWorldBound(room.GetPrimAtPath("/Kitchen/DecorativeBowl/Bowl")).ComputeAlignedRange()
    assert bounds.GetMin()[2] == pytest.approx(0, abs=1e-8)
    assert bounds.GetMin()[0] > .59
    assert .055 < bounds.GetMax()[2] < .058
    for pane, name in (("L", "Plane_1"), ("R", "Plane1")):
        glass = room.GetPrimAtPath("/Kitchen/Architecture/Room/window__Window__window" + pane + "__" + name)
        assert UsdGeom.Imageable(glass).ComputeVisibility() == "invisible"
        assert room.GetPrimAtPath("/Kitchen/WindowTreatment/" + pane + "Backing")
