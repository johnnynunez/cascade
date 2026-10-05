#!/usr/bin/env python3
"""Independently validate the authored USD contract; emit JSON and exit 0/1."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdShade


TOLERANCE = .01
# Explicit conflict resolutions, independently specified (never imported from
# the generator). Original requirements for clearance/grounding/height still apply.
SIDE_BOTTOM = 2.0
FOOT_Z = -.47
HEAD_Z = .08


class Validator:
    def __init__(self):
        self.checks = 0
        self.errors = []

    def check(self, condition, message):
        self.checks += 1
        if not condition:
            self.errors.append(message)

    def near(self, actual, expected, message, tolerance=1e-6):
        try:
            a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
            valid = a.shape == b.shape and np.all(np.isfinite(a)) and np.allclose(a, b, rtol=0, atol=tolerance)
        except (ValueError, TypeError):
            valid = False
        self.check(valid, f'{message}: expected {expected}, got {actual}')

    def type(self, stage, path, expected):
        prim = stage.GetPrimAtPath(path)
        self.check(bool(prim), f'Missing prim: {path}')
        if prim:
            self.check(prim.GetTypeName() == expected, f'{path}: expected type {expected}, got {prim.GetTypeName()}')
        return prim if prim and prim.GetTypeName() == expected else None


def bounds(cache, prim):
    result = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if result.IsEmpty():
        return None
    return np.array([result.GetMin(), result.GetMax()], dtype=float)


def packed(box):
    return None if box is None else [[round(float(v), 6) for v in row] for row in box]


def union(boxes):
    boxes = np.asarray(boxes)
    return np.array([boxes[:, 0].min(axis=0), boxes[:, 1].max(axis=0)])


def around(center, half):
    return np.array([np.array(center) - half, np.array(center) + half])


def joint(v, stage, path, position):
    prim = v.type(stage, path, 'Xform')
    if not prim:
        return
    ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
    v.check([op.GetOpName() for op in ops] == ['xformOp:translate', 'xformOp:rotateXYZ'],
            f'{path}: joint ops must be translate, rotateXYZ in exactly that order')
    v.check(not UsdGeom.Xformable(prim).GetResetXformStack(), f'{path}: resetXformStack is forbidden')
    v.near(prim.GetAttribute('xformOp:translate').Get(), position, f'{path} joint translation')
    v.near(prim.GetAttribute('xformOp:rotateXYZ').Get(), (0, 0, 0), f'{path} neutral rotation')


def expected_stage(v, stage):
    expected = {
        '/Stage/Floor': ('Mesh', [[-9, -6, -.06], [9, 7, .0015]]),
        '/Stage/FrontEdge': ('Mesh', [[-9, -6, .0015], [9, -5.92, .03]]),
        '/Stage/LedWall': ('Mesh', [[-8, 7, .3], [8, 7, 5.3]]),
    }
    joint(v, stage, '/Stage', (0, 0, 0))
    for name, x, yaw in [('L', -7.2, 20), ('R', 7.2, -20)]:
        half = [1.25 * math.cos(math.radians(20)), 1.25 * math.sin(math.radians(20)), .8]
        path = '/Stage/SideScreen' + name
        expected[path] = ('Mesh', around((x, 6.2, SIDE_BOTTOM + .8), half))
        prim = stage.GetPrimAtPath(path)
        if prim:
            v.near(prim.GetAttribute('xformOp:rotateXYZ').Get(), (0, 0, yaw), path + ' inward yaw')
            v.near(prim.GetAttribute('xformOp:translate').Get(), (x, 6.2, SIDE_BOTTOM), path + ' translation')
    for name, y in [('Front', -4), ('Middle', 0), ('Back', 4)]:
        expected['/Stage/Truss/' + name] = ('Mesh', [[-9, y - .15, 6.35], [9, y + .15, 6.65]])
    v.type(stage, '/Stage/Truss', 'Xform')
    v.type(stage, '/Stage/Lights', 'Xform')
    for number in range(1, 13):
        x = (-6, -3.6, -1.2, 1.2, 3.6, 6)[(number - 1) % 6]
        y = -2.5 if number <= 6 else 1.5
        path = '/Stage/Lights/Spot' + str(number)
        # BBoxCache transforms the sphere's local bounding box with the light's tilt.
        angle = math.atan2(1, 6.3 - .0015)
        half = [.25, .25 * (math.cos(angle) + math.sin(angle)), .25 * (math.cos(angle) + math.sin(angle))]
        expected[path] = ('SphereLight', around((x, y, 6.3), half))
        prim = v.type(stage, path, 'SphereLight')
        if not prim:
            continue
        light = UsdLux.SphereLight(prim)
        v.near(light.GetRadiusAttr().Get(), .25, path + ' radius')
        v.near(light.GetIntensityAttr().Get(), 60000, path + ' intensity')
        v.near(light.GetExposureAttr().Get(), 0, path + ' exposure')
        v.check(light.GetNormalizeAttr().Get() is False, path + ': normalization must be disabled')
        v.near(light.GetColorAttr().Get(), (1, .95, .9), path + ' color')
        v.check(prim.HasAPI(UsdLux.ShapingAPI), path + ': missing ShapingAPI')
        shape = UsdLux.ShapingAPI(prim)
        v.near(shape.GetShapingConeAngleAttr().Get(), 40, path + ' cone')
        v.near(shape.GetShapingConeSoftnessAttr().Get(), .4, path + ' softness')
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        origin = transform.Transform(Gf.Vec3d(0))
        v.near(origin, (x, y, 6.3), path + ' position')
        aim = transform.TransformDir(Gf.Vec3d(0, 0, -1)).GetNormalized()
        floor_z = .0015
        hits_floor = all(math.isfinite(value) for value in (*origin, *aim)) and origin[2] > floor_z and aim[2] < -1e-9
        v.check(hits_floor, path + ': -Z ray must point down toward the floor')
        if hits_floor:
            hit = origin + aim * ((floor_z - origin[2]) / aim[2])
            target = Gf.Vec3d(x, y + 1, floor_z)
            miss = (hit - target).GetLength()
            v.check(miss <= .3, f'{path}: floor aim misses target {tuple(target)} by {miss:.6f} m (limit 0.3 m)')
    for name, x in [('L', -4), ('R', 4)]:
        path = '/Stage/Lights/Fill' + name
        angle = math.radians(20)
        half = [3, .5 * math.sin(angle), .5 * math.cos(angle)]
        expected[path] = ('RectLight', around((x, -5.5, 3), half))
        prim = v.type(stage, path, 'RectLight')
        if not prim:
            continue
        light = UsdLux.RectLight(prim)
        v.near(light.GetWidthAttr().Get(), 6, path + ' width')
        v.near(light.GetHeightAttr().Get(), 1, path + ' height')
        v.near(light.GetIntensityAttr().Get(), 1500, path + ' intensity')
        v.near(light.GetExposureAttr().Get(), 0, path + ' exposure')
        v.check(light.GetNormalizeAttr().Get() is False, path + ': normalization must be disabled')
        v.near(light.GetColorAttr().Get(), (.9, .93, 1), path + ' color')
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        v.near(transform.Transform(Gf.Vec3d(0)), (x, -5.5, 3), path + ' position')
        v.near(transform.TransformDir(Gf.Vec3d(0, 0, -1)), (0, math.cos(angle), -math.sin(angle)), path + ' facing')
    ambient = v.type(stage, '/Stage/Lights/Ambient', 'DomeLight')
    if ambient:
        dome = UsdLux.DomeLight(ambient)
        v.near(dome.GetIntensityAttr().Get(), 60, 'Ambient intensity')
        texture = dome.GetTextureFileAttr().Get()
        v.check(not texture or not texture.path, 'Ambient must have no texture')
    for name, z in [('Bottom', .3), ('Top', 5.3)]:
        path = '/Stage/Lights/EdgeWash' + name
        expected[path] = ('RectLight', [[-4, 6.9, z - .2], [4, 6.9, z + .2]])
        prim = v.type(stage, path, 'RectLight')
        if not prim:
            continue
        light = UsdLux.RectLight(prim)
        v.near(light.GetWidthAttr().Get(), 8, path + ' width')
        v.near(light.GetHeightAttr().Get(), .4, path + ' height')
        v.near(light.GetIntensityAttr().Get(), 800, path + ' intensity')
        v.near(light.GetColorAttr().Get(), (.46, .73, 0), path + ' color')
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        v.near(transform.TransformDir(Gf.Vec3d(0, 0, -1)), (0, -1, 0), path + ' facing')
    expected_lights = {path for path, (kind, _) in expected.items() if kind in ('SphereLight', 'RectLight')}
    expected_lights.add('/Stage/Lights/Ambient')
    actual_lights = {str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdLux.LightAPI)}
    v.check(actual_lights == expected_lights, 'Stage must have exactly 12 spots, 2 fills, 2 edge washes and 1 dome')
    return expected


def expected_presenter(v, stage):
    joints = {'/Presenter': (0, 0, 0), '/Presenter/Pelvis': (0, 0, .98), '/Presenter/Pelvis/Neck': (0, 0, .6)}
    shapes = {'/Presenter/Pelvis/Torso': ('Capsule', (0, 0, 1.31), .16, .45),
              '/Presenter/Pelvis/Neck/Head': ('Sphere', (0, 0, 1.58 + HEAD_Z), .11, 0)}
    for side, sign in [('L', 1), ('R', -1)]:
        hip = '/Presenter/Hip' + side
        shoulder = '/Presenter/Shoulder' + side
        joints[hip] = (0, sign * .1, .98)
        joints[hip + '/Knee'] = (0, 0, -.48)
        joints[shoulder] = (0, sign * .24, 1.42)
        joints[shoulder + '/Elbow'] = (0, 0, -.33)
        shapes[hip + '/Thigh'] = ('Capsule', (0, sign * .1, .73), .075, .42)
        shapes[hip + '/Knee/Shin'] = ('Capsule', (0, sign * .1, .26), .06, .40)
        shapes[hip + '/Knee/Foot'] = ('Cube', (.06, sign * .1, .50 + FOOT_Z), (.13, .05, .03), 0)
        shapes[shoulder + '/UpperArm'] = ('Capsule', (0, sign * .24, 1.25), .05, .30)
        shapes[shoulder + '/Elbow/Forearm'] = ('Capsule', (0, sign * .24, .93), .045, .28)
    for path, position in joints.items():
        joint(v, stage, path, position)
    expected = {}
    for path, (kind, center, radius, height) in shapes.items():
        prim = v.type(stage, path, kind)
        half = (radius, radius, radius + height / 2) if kind == 'Capsule' else (radius if kind == 'Cube' else [radius] * 3)
        expected[path] = (kind, around(center, half))
        if not prim:
            continue
        if kind == 'Capsule':
            cap = UsdGeom.Capsule(prim)
            v.near(cap.GetRadiusAttr().Get(), radius, path + ' capsule radius')
            v.near(cap.GetHeightAttr().Get(), height, path + ' cylinder height')
            v.check(cap.GetAxisAttr().Get() == 'Z', path + ': capsule must use axis Z')
        elif kind == 'Sphere':
            v.near(UsdGeom.Sphere(prim).GetRadiusAttr().Get(), radius, path + ' sphere radius')
        else:
            v.near(UsdGeom.Cube(prim).GetSizeAttr().Get(), 1, path + ' cube size')
            v.near(prim.GetAttribute('xformOp:scale').Get(), (.26, .1, .06), path + ' foot dimensions')
        # Verify extent metadata too, preventing fabricated extents from masking bad geometry.
        local_half = [.5] * 3 if kind == 'Cube' else half
        v.near(UsdGeom.Boundable(prim).GetExtentAttr().Get(), around((0, 0, 0), local_half), path + ' local extent')
    for path in joints:
        descendants = [box for leaf, (_, box) in expected.items() if leaf.startswith(path + '/')]
        if descendants:
            expected[path] = ('Xform', union(descendants))
    return expected


def mesh_checks(v, prim):
    shape = UsdGeom.Mesh(prim)
    path = str(prim.GetPath())
    points = np.asarray(shape.GetPointsAttr().Get(), dtype=float)
    counts = list(shape.GetFaceVertexCountsAttr().Get() or [])
    indices = list(shape.GetFaceVertexIndicesAttr().Get() or [])
    normals = np.asarray(shape.GetNormalsAttr().Get(), dtype=float)
    v.check(points.ndim == 2 and points.shape[1:] == (3,) and np.all(np.isfinite(points)), path + ': invalid points')
    v.check(shape.GetSubdivisionSchemeAttr().Get() == 'none', path + ': subdivision must be none')
    v.check(shape.GetNormalsInterpolation() == 'uniform', path + ': expected per-face normals')
    v.check(shape.GetOrientationAttr().Get() == 'rightHanded', path + ': incorrect winding convention')
    topology_ok = sum(counts) == len(indices) and all(c >= 3 for c in counts) and all(0 <= i < len(points) for i in indices)
    v.check(topology_ok, path + ': invalid topology')
    v.check(normals.shape == (len(counts), 3), path + ': missing face normals')
    if topology_ok and normals.shape == (len(counts), 3):
        offset = 0
        for face, count in enumerate(counts):
            a, b, c = points[indices[offset:offset + 3]]
            geometric = np.cross(b - a, c - a)
            length = np.linalg.norm(geometric)
            v.check(length > 0, path + ': degenerate face')
            if length > 0:
                v.near(normals[face], geometric / length, path + f' face {face} normal')
            offset += count
    v.near(shape.GetExtentAttr().Get(), [points.min(axis=0), points.max(axis=0)], path + ' extent versus vertices')
    if prim.GetName() in ('LedWall', 'SideScreenL', 'SideScreenR'):
        v.check(len(points) == 4 and counts == [4], path + ': screen must be a flat quad')
        v.check(not shape.GetDoubleSidedAttr().Get(), path + ': screen must be single-sided')
        uv = UsdGeom.PrimvarsAPI(prim).GetPrimvar('st')
        v.check(bool(uv), path + ': missing primvars:st')
        if uv:
            v.check(uv.GetInterpolation() == 'vertex', path + ': expected vertex UVs')
            v.near(uv.ComputeFlattened(), [(0, 0), (1, 0), (1, 1), (0, 1)], path + ' screen UVs')
        v.near(normals, [(0, -1, 0)], path + ' audience-facing normal')
    return len(points)


def material_checks(v, stage, root):
    v.type(stage, root + '/Looks', 'Scope')
    specs = {'Floor': (.05, .45, 0), 'GreenStrip': (0, .3, 0), 'Truss': (.18, .4, .9),
             'LedWall': (0, .5, 0), 'SideScreen': (0, .5, 0)} if root == '/Stage' else {
                 'Clothing': (.28, .7, 0), 'Head': (.55, .8, 0)}
    for name, (color, roughness, metallic) in specs.items():
        path = root + '/Looks/' + name
        mat = v.type(stage, path, 'Material')
        shader_prim = v.type(stage, path + '/Surface', 'Shader')
        if not mat or not shader_prim:
            continue
        shader = UsdShade.Shader(shader_prim)
        v.check(shader.GetIdAttr().Get() == 'UsdPreviewSurface', path + ': expected UsdPreviewSurface')
        surface = UsdShade.Material(mat).GetSurfaceOutput().GetConnectedSource()
        v.check(bool(surface) and str(surface[0].GetPrim().GetPath()) == path + '/Surface' and surface[1] == 'surface',
                path + ': material surface is disconnected')
        for key, value in [('diffuseColor', [color] * 3), ('roughness', roughness), ('metallic', metallic)]:
            v.near(shader.GetInput(key).Get(), value, path + ' ' + key)
        if name == 'Floor':
            v.near(shader.GetInput('useSpecularWorkflow').Get(), 1, path + ' specular workflow')
            v.near(shader.GetInput('specularColor').Get(), [.5] * 3, path + ' specular color')
        if name == 'GreenStrip':
            v.near(shader.GetInput('emissiveColor').Get(), [.69, 1.095, 0], path + ' emission')
        if name in ('LedWall', 'SideScreen'):
            texprim = v.type(stage, path + '/Texture', 'Shader')
            uvprim = v.type(stage, path + '/UVReader', 'Shader')
            if not texprim or not uvprim:
                continue
            tex, uv = UsdShade.Shader(texprim), UsdShade.Shader(uvprim)
            v.check(tex.GetIdAttr().Get() == 'UsdUVTexture', path + ': incorrect texture shader')
            v.check(uv.GetIdAttr().Get() == 'UsdPrimvarReader_float2', path + ': incorrect UV reader')
            v.check(uv.GetInput('varname').Get() == 'st', path + ': incorrect UV primvar')
            asset = tex.GetInput('file').Get()
            expected_file = 'textures/ledwall.png' if name == 'LedWall' else 'textures/side.png'
            v.check(bool(asset) and asset.path == expected_file, path + ': incorrect relative texture path')
            v.check(bool(asset) and bool(asset.resolvedPath) and Path(asset.resolvedPath).is_file(), path + ': unresolved texture')
            v.check(tex.GetInput('sourceColorSpace').Get() == 'sRGB', path + ': texture must be sRGB')
            gain = 1.6 if name == 'LedWall' else 1.2
            v.near(tex.GetInput('scale').Get(), (gain, gain, gain, 1), path + ' texture emission gain')
            for attr, target, output in [(shader.GetInput('emissiveColor'), path + '/Texture', 'rgb'),
                                         (tex.GetInput('st'), path + '/UVReader', 'result')]:
                connection = attr.GetConnectedSource()
                v.check(bool(connection) and str(connection[0].GetPrim().GetPath()) == target and connection[1] == output,
                        path + ': disconnected texture network')
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Gprim):
            continue
        path = str(prim.GetPath())
        if root == '/Stage':
            name = {'Floor': 'Floor', 'FrontEdge': 'GreenStrip', 'LedWall': 'LedWall',
                    'SideScreenL': 'SideScreen', 'SideScreenR': 'SideScreen'}.get(prim.GetName(), 'Truss')
        else:
            name = 'Head' if prim.GetName() == 'Head' else 'Clothing'
        bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        v.check(bool(bound) and str(bound.GetPath()) == root + '/Looks/' + name, path + ': incorrect material binding')


def validate_stage(v, path, root):
    stage = Usd.Stage.Open(str(path))
    v.check(bool(stage), f'Cannot open {path}')
    if not stage:
        return {}
    v.check(UsdGeom.GetStageUpAxis(stage) == 'Z', str(path) + ': expected Z up')
    v.near(UsdGeom.GetStageMetersPerUnit(stage), 1, str(path) + ' metre units')
    default = stage.GetDefaultPrim()
    v.check(bool(default) and str(default.GetPath()) == root and default.GetTypeName() == 'Xform', str(path) + ': invalid default prim')
    v.check(len(stage.GetRootLayer().subLayerPaths) == 0, str(path) + ': unexpected sublayers')
    expected = expected_stage(v, stage) if root == '/Stage' else expected_presenter(v, stage)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=False)
    for prim_path, (kind, expected_box) in expected.items():
        prim = v.type(stage, prim_path, kind)
        if prim:
            v.near(bounds(cache, prim), expected_box, prim_path + ' world bounds', TOLERANCE)
    if root == '/Stage':
        for prim_path in ('/Stage', '/Stage/Truss', '/Stage/Lights'):
            boxes = [box for leaf, (_, box) in expected.items() if leaf.startswith(prim_path + '/')]
            prim = v.type(stage, prim_path, 'Xform')
            if prim:
                v.near(bounds(cache, prim), union(boxes), prim_path + ' aggregate bounds', TOLERANCE)
    vertices = 0
    shapes = {'Mesh': 0, 'Capsule': 0, 'Sphere': 0, 'Cube': 0}
    physics = []
    for prim in stage.TraverseAll():
        prim_path = str(prim.GetPath())
        apis = [str(schema) for schema in prim.GetAppliedSchemas() if 'physics' in str(schema).lower()]
        if apis or 'physics' in prim.GetTypeName().lower():
            physics.append({'prim': prim_path, 'apis': apis, 'type': prim.GetTypeName()})
        v.check(prim.IsActive() and prim.IsDefined(), prim_path + ': unexpected inactive or undefined prim')
        v.check(not prim.HasAuthoredReferences() and not prim.HasAuthoredPayloads(), prim_path + ': unexpected external geometry')
        for attr in prim.GetAttributes():
            v.check(attr.GetNumTimeSamples() == 0, str(attr.GetPath()) + ': assets must have a static bind pose')
        if prim.IsA(UsdGeom.Gprim):
            kind = prim.GetTypeName()
            v.check(prim_path in expected, prim_path + ': unexpected geometry')
            if kind in shapes:
                shapes[kind] += 1
            if kind == 'Mesh':
                vertices += mesh_checks(v, prim)
            box = bounds(cache, prim)
            v.check(box is not None, prim_path + ': empty geometry bound')
            if root == '/Stage' and box is not None:
                v.check(box[0, 2] >= -.060001, prim_path + ': extends below Z=-0.06')
                if prim_path != '/Stage/Floor':
                    overlap = all(box[1, i] > lo + 1e-6 and box[0, i] < hi - 1e-6
                                  for i, (lo, hi) in enumerate([(-8.5, 8.5), (-5.5, 6.5), (.0015, 2)]))
                    v.check(not overlap, prim_path + ': intrudes into the required walking clearance')
    v.check(not physics, f'{path.name}: physics APIs or prims found: {physics}')
    material_checks(v, stage, root)
    root_prim = stage.GetPrimAtPath(root)
    world = bounds(cache, root_prim) if root_prim else None
    result = {'default_prim': root, 'prims': len(list(stage.Traverse())), 'bounds_m': packed(world),
              'geometry': shapes, 'authored_mesh_vertices': vertices, 'physics_apis': len(physics)}
    if root == '/Presenter' and world is not None:
        minimum = float(world[0, 2])
        height = float(world[1, 2] - minimum)
        v.check(abs(minimum) <= .01 + 1e-6, f'Presenter lowest point {minimum:.6f} m must be 0 ± 0.01 m')
        v.check(abs(height - 1.75) <= .03 + 1e-6, f'Presenter height {height:.6f} m must be 1.75 ± 0.03 m')
        for side in ('L', 'R'):
            foot = stage.GetPrimAtPath('/Presenter/Hip' + side + '/Knee/Foot')
            if foot:
                foot_box = bounds(cache, foot)
                v.check(foot_box is not None and abs(foot_box[0, 2]) <= .01 + 1e-6, f'{foot.GetPath()}: foot bottom must be on the floor')
        result.update(lowest_point_m=round(minimum, 6), standing_height_m=round(height, 6))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('assets'))
    args = parser.parse_args()
    v = Validator()
    assets = {}
    for filename, root in [('stage.usda', '/Stage'), ('presenter.usda', '/Presenter')]:
        try:
            assets[filename] = validate_stage(v, args.out / filename, root)
        except Exception as error:
            v.check(False, f'{filename}: {type(error).__name__}: {error}')
    sizes = {}
    for filename, expected in [('ledwall.png', (3840, 1280)), ('side.png', (1280, 800))]:
        try:
            with Image.open(args.out / 'textures' / filename) as im:
                im.load()
                sizes[filename] = list(im.size)
                v.check(im.size == expected and im.format == 'PNG' and im.mode == 'RGB', filename + ': invalid PNG format or dimensions')
                v.check(np.asarray(im).std() > 10, filename + ': texture appears empty')
        except Exception as error:
            v.check(False, f'{filename}: {type(error).__name__}: {error}')
    try:
        readme = (args.out / 'README.md').read_text(encoding='utf-8')
        for text in ['xformOp:translate', 'xformOp:rotateXYZ', '25°', '35', '20°', '2 cm', '0.65', 'twice']:
            v.check(text in readme, 'README.md: missing animation contract detail ' + text)
    except Exception as error:
        v.check(False, f'README.md: {type(error).__name__}: {error}')
    vertices = sum(asset.get('authored_mesh_vertices', 0) for asset in assets.values())
    analytic = sum(sum(asset.get('geometry', {}).get(kind, 0) for kind in ('Capsule', 'Sphere', 'Cube')) for asset in assets.values())
    # USD stores analytic prims without vertices. Budget a conservative 2048 per
    # primitive for normal display tessellation; actual RTX tessellation is renderer-dependent.
    display_budget = vertices + 2048 * analytic
    v.check(display_budget < 50000, f'Geometry display budget {display_budget} exceeds 50000 vertices')
    report = {'version': 2, 'status': 'pass' if not v.errors else 'fail', 'usd_version': '.'.join(map(str, Usd.GetVersion())),
              'checks': v.checks, 'bounds_tolerance_m': TOLERANCE, 'assets': assets, 'textures': sizes,
              'geometry_budget': {'authored_mesh_vertices': vertices, 'analytic_primitives': analytic,
                                  'estimated_display_vertices': display_budget, 'limit': 50000},
              'documented_spec_adjustments': {'side_screen_bottom_z_m': {'from': 1.2, 'to': SIDE_BOTTOM},
                                             'foot_knee_local_z_m': {'from': -.50, 'to': FOOT_Z},
                                             'head_neck_local_z_m': {'from': .10, 'to': HEAD_Z}},
              'errors': v.errors}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if v.errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
