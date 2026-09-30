#!/usr/bin/env python3
"""Package the owner-supplied Cocina_Asier_01 export without its texture pack.

Maintainer tool (usd-core and numpy). Input files are never modified. Only
byte-verified ambientCG wood maps are admitted; all other surfaces are rebuilt.
The output has baked metre vertices, Z up, no physics and no source references.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt


SOURCE_SHA256 = "607ea10b6edea30e066ad82cd1df64604b2877e166487ac74b96bc84ff522bb7"
COUNTER_MIN = np.array((-.15987323186177282, -.379085082641188, -.03))
COUNTER_MAX = np.array((.9867574274915567, .3795904990656434, 0.))
WOOD_MAPS = ("Wood044_2K-JPG_Color.jpg", "Wood044_2K-JPG_Roughness.jpg",
             "WoodFloor051_2K-JPG_Color.jpg", "WoodFloor051_2K-JPG_Roughness.jpg")


def shader(stage, name, color, roughness=.5, metallic=0):
    mat = UsdShade.Material.Define(stage, "/CocinaAsier/Looks/" + name)
    surf = UsdShade.Shader.Define(stage, str(mat.GetPath()) + "/Surface")
    surf.CreateIdAttr("UsdPreviewSurface")
    for key, typ, value in (("diffuseColor", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*color)),
                            ("roughness", Sdf.ValueTypeNames.Float, roughness),
                            ("metallic", Sdf.ValueTypeNames.Float, metallic)):
        surf.CreateInput(key, typ).Set(value)
    surf.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")
    return mat, surf


def wood_texture(stage, mat, surf, prefix):
    path = str(mat.GetPath())
    uv = UsdShade.Shader.Define(stage, path + "/UV")
    uv.CreateIdAttr("UsdPrimvarReader_float2")
    uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    uv.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    for channel, input_name, output, value_type in (
        ("Color", "diffuseColor", "rgb", Sdf.ValueTypeNames.Color3f),
        ("Roughness", "roughness", "r", Sdf.ValueTypeNames.Float),
    ):
        tex = UsdShade.Shader.Define(stage, path + "/" + channel)
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("textures/" + prefix + "_2K-JPG_" + channel + ".jpg"))
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB" if channel == "Color" else "raw")
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), "result")
        tex.CreateOutput(output, value_type)
        surf.GetInput(input_name).ConnectToSource(tex.ConnectableAPI(), output)


def material(stage, source):
    """Reauthor scalar paint/metal values, never copy an unknown shader graph."""
    name = source.GetName()
    src = UsdShade.Material(source).ComputeSurfaceSource()[0]
    color, roughness, metallic = (.55, .55, .55), .48, 0
    if src:
        inp = src.GetInput("diffuseColor")
        if inp and inp.Get() is not None:
            color = tuple(float(v) for v in inp.Get())
        inp = src.GetInput("roughness")
        if inp and inp.Get() is not None:
            roughness = max(.22, float(inp.Get()))
    lower = name.lower()
    if "steel" in lower or "aluminium" in lower or "chrome" in lower:
        color, roughness, metallic = (.48, .50, .52), .3, .86
    elif "glass" in lower:
        color, roughness, metallic = (.018, .024, .027), .18, .22
    elif "plastic_white" in lower or lower.startswith("plastic_ss"):
        color, roughness = (.73, .72, .69), .4
    elif lower == "wall":
        color, roughness = (.73, .715, .68), .82
    elif "display" in lower:
        color, roughness = (.018, .032, .034), .3
    elif "black" in lower or "spiral" in lower:
        color, roughness = (.016, .018, .019), .35
    # Unknown food photographs and artwork are excluded with their mesh.
    # Remaining cutting boards get clean birch; printed labels become plain.
    elif name in {"Material_kitchen_0076", "Material_kitchen_0079"}:
        color, roughness = (.52, .34, .17), .58
    mat, surf = shader(stage, name, color, roughness, metallic)
    if name == "Glass":
        # The source window panes are transparent. The original opaque blinds
        # behind them supply the exterior closure, not a black replacement pane.
        surf.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.)
    if name in {"Wood_worktop", "Wood_Floor"}:
        wood_texture(stage, mat, surf, "Wood044" if name == "Wood_worktop" else "WoodFloor051")
    return mat


def vertices(mesh, xforms):
    matrix = np.array(xforms.GetLocalToWorldTransform(mesh.GetPrim()))
    points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    return points @ matrix[:3, :3] + matrix[3, :3], matrix


def removal_reason(prim, lo, hi):
    name = prim.GetName()
    path = str(prim.GetPath())
    if prim.IsAbstract() or "/_class_" in path:
        return "unused exporter class definition"
    if name == "Corona_lig":
        return "exporter light helper"
    if name.startswith(("Apple", "Pear", "Bread", "_DCake")):
        return "supplied food dressing removed; original task props only"
    if name in {"Object_kitchen_0022", "Object_kitchen_0148", "Object_kitchen_0149", "Object_kitchen_0150"}:
        return "unverified picture or food photograph removed"
    if name not in {"Box12559", "Box12573"} and (lo[0] >= 400 and hi[0] <= 2800
            and lo[1] >= -1150 and hi[1] <= 350 and hi[2] < 1500):
        return "island dressing or stool removed for clear workspace and cameras"
    return None


def build(source_dir, output_dir):
    source_file = source_dir / "Cocina_Asier_01.usd"
    digest = hashlib.sha256(source_file.read_bytes()).hexdigest()
    if SOURCE_SHA256 and digest != SOURCE_SHA256:
        raise ValueError("Unreviewed Cocina_Asier_01 source bytes")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "textures").mkdir(exist_ok=True)
    verified = json.loads((Path(__file__).resolve().parents[1] / "demo/scene/cocina_asier_sources.json").read_text())
    for name in WOOD_MAPS:
        raw = (source_dir / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != verified["textures"][name]["sha256"]:
            raise ValueError("Unverified wood texture: " + name)
        (output_dir / "textures" / name).write_bytes(raw)
    src = Usd.Stage.Open(str(source_file))
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/CocinaAsier").GetPrim()
    stage.SetDefaultPrim(root)
    root.SetCustomDataByKey("source", "Owner-authorized Cocina_Asier_01; see NOTICE.md")
    root.SetCustomDataByKey("sourceSha256", digest)
    mats = {str(p.GetPath()): material(stage, p) for p in src.Traverse() if p.IsA(UsdShade.Material)}
    fallback, _ = shader(stage, "Neutral", (.6, .59, .56))
    xforms = UsdGeom.XformCache()
    island_src = UsdGeom.Mesh(src.GetPrimAtPath("/root/Box12573"))
    island_pts, _ = vertices(island_src, xforms)
    island_lo, island_hi = island_pts.min(axis=0) * .001, island_pts.max(axis=0) * .001
    scale_xy = (COUNTER_MAX - COUNTER_MIN)[:2] / (island_hi - island_lo)[:2]
    audit = []
    for prim in src.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        if not points:
            audit.append({"source_prim": str(prim.GetPath()), "disposition": "empty exporter shape removed"})
            continue
        pts, matrix = vertices(mesh, xforms)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        reason = removal_reason(prim, lo, hi)
        if reason:
            audit.append({"source_prim": str(prim.GetPath()), "disposition": reason})
            continue
        name = prim.GetName()
        pts *= .001
        extra_scale = np.ones(3)
        if name in {"Box12559", "Box12573"}:
            destination = "/CocinaAsier/Island/" + ("Countertop" if name == "Box12573" else "Base")
            pts[:, :2] = (pts[:, :2] - island_lo[:2]) * scale_xy + COUNTER_MIN[:2]
            extra_scale[:2] = scale_xy
            if name == "Box12573":
                extra_scale[2] = .03 / (island_hi[2] - island_lo[2])
                pts[:, 2] = (pts[:, 2] - island_hi[2]) * extra_scale[2]
            else:
                extra_scale[2] = .81 / .8
                pts[:, 2] = pts[:, 2] * extra_scale[2] - .84
        else:
            destination = "/CocinaAsier/Room/" + str(prim.GetPath()).removeprefix("/root/").replace("/", "__")
            pts += np.array((0., .25, -.84))
        new = UsdGeom.Mesh.Define(stage, destination)
        new.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
        new.CreateFaceVertexCountsAttr(mesh.GetFaceVertexCountsAttr().Get())
        new.CreateFaceVertexIndicesAttr(mesh.GetFaceVertexIndicesAttr().Get())
        new.CreateSubdivisionSchemeAttr("none")
        new.CreateDoubleSidedAttr(True)
        new.CreateOrientationAttr(mesh.GetOrientationAttr().Get())
        new.CreateExtentAttr(UsdGeom.PointBased(new).ComputeExtent(new.GetPointsAttr().Get()))
        normal_attr = mesh.GetNormalsAttr()
        normal_pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("normals")
        normal = normal_pv.ComputeFlattened() if normal_pv else normal_attr.Get()
        if normal:
            n = np.array(normal, dtype=np.float64) @ np.linalg.inv(matrix[:3, :3]).T
            n /= extra_scale
            n /= np.maximum(np.linalg.norm(n, axis=1)[:, None], 1e-12)
            new.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(n.astype(np.float32)))
            new.SetNormalsInterpolation(normal_pv.GetInterpolation() if normal_pv else mesh.GetNormalsInterpolation())
        for pv in UsdGeom.PrimvarsAPI(prim).GetPrimvars():
            if pv.GetPrimvarName() == "normals" or not pv.HasValue():
                continue
            target = UsdGeom.PrimvarsAPI(new).CreatePrimvar(pv.GetPrimvarName(), pv.GetTypeName(), pv.GetInterpolation(), pv.GetElementSize())
            target.Set(pv.Get())
            if pv.IsIndexed():
                target.SetIndices(pv.GetIndices())
        bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        UsdShade.MaterialBindingAPI.Apply(new.GetPrim()).Bind(mats.get(str(bound.GetPath()), fallback) if bound else fallback)
        for subset in UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets():
            target = UsdGeom.Subset.Define(stage, destination + "/" + subset.GetPrim().GetName())
            target.CreateElementTypeAttr(subset.GetElementTypeAttr().Get())
            target.CreateFamilyNameAttr("materialBind")
            target.CreateIndicesAttr(subset.GetIndicesAttr().Get())
            bound, _ = UsdShade.MaterialBindingAPI(subset).ComputeBoundMaterial()
            UsdShade.MaterialBindingAPI.Apply(target.GetPrim()).Bind(mats.get(str(bound.GetPath()), fallback) if bound else fallback)
        audit.append({"source_prim": str(prim.GetPath()), "output_prim": destination,
                      "disposition": "owner-authorized geometry, reauthored surface",
                      "min_m": pts.min(axis=0).tolist(), "max_m": pts.max(axis=0).tolist()})
    # Purge unbound materials, including all unused food material names.
    used = set()
    for prim in stage.Traverse():
        for target in prim.GetRelationship("material:binding").GetTargets():
            used.add(str(target))
    for prim in list(stage.GetPrimAtPath("/CocinaAsier/Looks").GetChildren()):
        if str(prim.GetPath()) not in used:
            stage.RemovePrim(prim.GetPath())
    stage.GetRootLayer().Export(str(output_dir / "cocina_asier.usdc"))
    (output_dir / "geometry-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    scene = Path(__file__).resolve().parents[1] / "demo/scene"
    shutil.copyfile(scene / "NOTICE.md", output_dir / "NOTICE.md")
    shutil.copyfile(scene / "cocina_asier_sources.json", output_dir / "sources.json")
    print(json.dumps({"retained_meshes": sum("output_prim" in r for r in audit),
                      "removed_meshes": sum("output_prim" not in r for r in audit),
                      "size_bytes": (output_dir / "cocina_asier.usdc").stat().st_size}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)
