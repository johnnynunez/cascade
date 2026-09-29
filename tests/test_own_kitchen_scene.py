"""Authored stage geometry/provenance checks; live acceptance is separate."""
from pathlib import Path
import sys

import pytest

pytest.importorskip("pxr")
from pxr import Sdf, Usd, UsdGeom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
from own_kitchen import DEFAULT_COUNTER, author_kitchen


@pytest.fixture(scope="module")
def kitchen_stage():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    author_kitchen(stage)
    return stage


def test_original_room_has_no_external_assets_or_simulation_schemas(kitchen_stage):
    assert set(kitchen_stage.GetUsedLayers()) == {
        kitchen_stage.GetRootLayer(), kitchen_stage.GetSessionLayer()}
    for prim in kitchen_stage.Traverse():
        assert not prim.HasAuthoredReferences()
        assert not prim.HasAuthoredPayloads()
        assert not any(name.lower().startswith(("physics", "physx", "newton"))
                       for name in prim.GetAppliedSchemas())
        for attribute in prim.GetAttributes():
            if attribute.GetTypeName() in (Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray):
                # A dome light schema exposes an unbound texture slot by default.
                assert not attribute.HasAuthoredValueOpinion()


def test_counter_surface_matches_calibrated_static_collider(kitchen_stage):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    bounds = cache.ComputeWorldBound(kitchen_stage.GetPrimAtPath("/Kitchen/Island/Countertop/Slab")).ComputeAlignedRange()
    assert tuple(bounds.GetMin()) == pytest.approx(DEFAULT_COUNTER["min"], abs=1e-7)
    assert tuple(bounds.GetMax()) == pytest.approx(DEFAULT_COUNTER["max"], abs=1e-7)
    assert bounds.GetMax()[2] == 0


def test_original_decoration_leaves_reachable_worktop_clear(kitchen_stage):
    # Keep the full island depth and 71cm of its length free above the surface.
    # The task bodies/robot are installed separately; this checks decoration.
    clear_min = (-.15987323186177282, -.379085082641188, .001)
    clear_max = (.55, .3795904990656434, .55)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    intrusions = []
    for prim in kitchen_stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if all(bounds.GetMin()[i] < clear_max[i] and bounds.GetMax()[i] > clear_min[i]
               for i in range(3)):
            intrusions.append(str(prim.GetPath()))
    assert intrusions == []


def test_bowl_foot_is_on_counter_and_outside_reach(kitchen_stage):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    bowl = cache.ComputeWorldBound(kitchen_stage.GetPrimAtPath("/Kitchen/DecorativeBowl/Bowl")).ComputeAlignedRange()
    assert bowl.GetMin()[2] == pytest.approx(0, abs=1e-9)
    assert bowl.GetMin()[0] > .59


def test_duplicate_authoring_and_nonzero_worktop_are_rejected(kitchen_stage):
    with pytest.raises(ValueError, match="already exists"):
        author_kitchen(kitchen_stage)
    with pytest.raises(ValueError, match="top at z=0"):
        author_kitchen(Usd.Stage.CreateInMemory(), counter={"min": [-.2, -.4, 0], "max": [1, .4, .04]})
