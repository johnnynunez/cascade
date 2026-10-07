#!/usr/bin/env python3
"""Generate deterministic, visual-only keynote and presenter USD assets."""

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt


# Resolve the task's conflicting coordinates in favor of walking clearance and
# a grounded standing pose. Each adjustment is documented in the asset README.
SIDE_BOTTOM = 2.0
FOOT_Z = -0.47
HEAD_Z = 0.08
GREEN = (0.46, 0.73, 0.0)
FONT_DIR = Path('/usr/share/fonts/truetype/dejavu')


def font(size, bold=False):
    return ImageFont.truetype(str(FONT_DIR / ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')), size)


def centered(draw, text, y, face, fill, width):
    box = draw.textbbox((0, 0), text, font=face)
    draw.text(((width - (box[2] - box[0])) / 2 - box[0], y - box[1]), text, font=face, fill=fill)


def textures(out):
    directory = out / 'textures'
    directory.mkdir(parents=True, exist_ok=True)
    width, height = 3840, 1280
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    pixels[:] = (6, 8, 12)
    # A fine, deliberately restrained LED matrix, with no randomness.
    pixels[::12, :] = (9, 12, 17)
    pixels[:, ::12] = (9, 12, 17)
    pixels[5::12, 5::12] = (16, 20, 26)
    led = Image.fromarray(pixels)
    draw = ImageDraw.Draw(led)
    title = 'CASCADE'
    face = font(276, True)
    tracking = 26
    widths = [draw.textlength(letter, font=face) for letter in title]
    x = (width - sum(widths) - tracking * (len(title) - 1)) / 2
    for letter, advance in zip(title, widths):
        box = draw.textbbox((0, 0), letter, font=face)
        draw.text((x, 360 - box[1]), letter, font=face, fill=(245, 247, 250))
        x += advance + tracking
    centered(draw, '12 MicroDucks · Isaac Sim 6.2 · Newton', 685, font(66), (188, 195, 205), width)
    draw.rectangle((620, 842, width - 620, 848), fill='#76b900')
    centered(draw, 'zero-shot local inference · sim → real', 977, font(43), (151, 164, 177), width)
    led.save(directory / 'ledwall.png', compress_level=9)

    side = Image.new('RGB', (1280, 800), '#06080c')
    draw = ImageDraw.Draw(side)
    for x in range(0, 1280, 16):
        draw.line((x, 0, x, 800), fill=(9, 12, 17))
    for y in range(0, 800, 16):
        draw.line((0, y, 1280, y), fill=(9, 12, 17))
    centered(draw, 'MicroDuck', 212, font(144, True), (239, 244, 249), 1280)
    wave = []
    for x in range(150, 1131):
        t = (x - 150) / 980
        envelope = math.sin(math.pi * t) ** 2
        y = 498 - 64 * envelope * math.sin(8 * math.pi * t)
        wave.append((x, y))
    draw.line(wave, fill='#76b900', width=6, joint='curve')
    centered(draw, 'LOCAL INFERENCE', 638, font(29), (148, 161, 174), 1280)
    side.save(directory / 'side.png', compress_level=9)


def new_stage(path, name):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(60)
    stage.SetFramesPerSecond(60)
    root = joint(stage, '/' + name)
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Scope.Define(stage, '/' + name + '/Looks')
    return stage


def joint(stage, path, translation=(0, 0, 0)):
    prim = UsdGeom.Xform.Define(stage, path)
    prim.AddTranslateOp().Set(Gf.Vec3d(*translation))
    prim.AddRotateXYZOp().Set(Gf.Vec3f(0))
    return prim


def transform(prim, translation=(0, 0, 0), rotation=None, scale=None):
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
    if rotation is not None:
        xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotation))
    if scale is not None:
        xform.AddScaleOp().Set(Gf.Vec3f(*scale))


def material(stage, path, color, roughness, metallic=0, emission=None, texture=None, gain=1, specular=None):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + '/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(1.0)
    if specular is not None:
        shader.CreateInput('useSpecularWorkflow', Sdf.ValueTypeNames.Int).Set(1)
        shader.CreateInput('specularColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(specular))
    if emission is not None:
        shader.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emission) * gain)
    if texture:
        reader = UsdShade.Shader.Define(stage, path + '/UVReader')
        reader.CreateIdAttr('UsdPrimvarReader_float2')
        reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
        reader.CreateOutput('result', Sdf.ValueTypeNames.Float2)
        sampler = UsdShade.Shader.Define(stage, path + '/Texture')
        sampler.CreateIdAttr('UsdUVTexture')
        sampler.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture))
        sampler.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
        sampler.CreateInput('wrapS', Sdf.ValueTypeNames.Token).Set('clamp')
        sampler.CreateInput('wrapT', Sdf.ValueTypeNames.Token).Set('clamp')
        sampler.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(gain, gain, gain, 1))
        sampler.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
        sampler.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
        shader.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(sampler.ConnectableAPI(), 'rgb')
    shader.CreateOutput('surface', Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return mat


def bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mat)


def mesh(stage, path, points, faces, normals, mat):
    shape = UsdGeom.Mesh.Define(stage, path)
    shape.CreatePointsAttr(points)
    shape.CreateFaceVertexCountsAttr([len(face) for face in faces])
    shape.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
    shape.CreateNormalsAttr(normals)
    shape.SetNormalsInterpolation(UsdGeom.Tokens.uniform)
    shape.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    shape.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    shape.CreateDoubleSidedAttr(False)
    shape.CreateExtentAttr([Gf.Vec3f(*(min(p[i] for p in points) for i in range(3))),
                            Gf.Vec3f(*(max(p[i] for p in points) for i in range(3)))])
    bind(shape, mat)
    return shape


def box(stage, path, low, high, mat):
    x0, y0, z0 = low
    x1, y1, z1 = high
    points = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
              (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    normals = [(0, 0, -1), (0, 0, 1), (0, -1, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 0)]
    return mesh(stage, path, points, faces, normals, mat)


def quad(stage, path, width, height, position, yaw, mat):
    w = width / 2
    shape = mesh(stage, path, [(-w, 0, 0), (w, 0, 0), (w, 0, height), (-w, 0, height)],
                 [(0, 1, 2, 3)], [(0, -1, 0)], mat)
    uv = UsdGeom.PrimvarsAPI(shape).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    uv.Set(Vt.Vec2fArray([(0, 0), (1, 0), (1, 1), (0, 1)]))
    transform(shape, position, (0, 0, yaw))
    return shape


def keynote(out):
    stage = new_stage(out / 'stage.usda', 'Stage')
    floor = material(stage, '/Stage/Looks/Floor', (.05, .05, .05), .45, specular=.5)
    strip = material(stage, '/Stage/Looks/GreenStrip', (0, 0, 0), .3, emission=GREEN, gain=1.5)
    metal = material(stage, '/Stage/Looks/Truss', (.18, .18, .18), .4, metallic=.9)
    led = material(stage, '/Stage/Looks/LedWall', (0, 0, 0), .5, texture='textures/ledwall.png', gain=1.6)
    side = material(stage, '/Stage/Looks/SideScreen', (0, 0, 0), .5, texture='textures/side.png', gain=1.2)
    box(stage, '/Stage/Floor', (-9, -6, -.06), (9, 7, .0015), floor)
    box(stage, '/Stage/FrontEdge', (-9, -6, .0015), (9, -5.92, .03), strip)
    quad(stage, '/Stage/LedWall', 16, 5, (0, 7, .3), 0, led)
    quad(stage, '/Stage/SideScreenL', 2.5, 1.6, (-7.2, 6.2, SIDE_BOTTOM), 20, side)
    quad(stage, '/Stage/SideScreenR', 2.5, 1.6, (7.2, 6.2, SIDE_BOTTOM), -20, side)
    UsdGeom.Xform.Define(stage, '/Stage/Truss')
    for name, y in [('Front', -4), ('Middle', 0), ('Back', 4)]:
        box(stage, '/Stage/Truss/' + name, (-9, y - .15, 6.35), (9, y + .15, 6.65), metal)
    UsdGeom.Xform.Define(stage, '/Stage/Lights')
    for row, y in enumerate((-2.5, 1.5)):
        for column, x in enumerate((-6, -3.6, -1.2, 1.2, 3.6, 6)):
            light = UsdLux.SphereLight.Define(stage, f'/Stage/Lights/Spot{row * 6 + column + 1}')
            light.CreateRadiusAttr(.25)
            light.CreateIntensityAttr(60000)
            light.CreateExposureAttr(0)
            light.CreateColorAttr(Gf.Vec3f(1, .95, .9))
            light.CreateNormalizeAttr(False)
            shaping = UsdLux.ShapingAPI.Apply(light.GetPrim())
            shaping.CreateShapingConeAngleAttr(40)
            shaping.CreateShapingConeSoftnessAttr(.4)
            xf = UsdGeom.Xformable(light)
            xf.AddTranslateOp().Set(Gf.Vec3d(x, y, 6.3))
            direction = Gf.Vec3d(0, 1, .0015 - 6.3).GetNormalized()
            rotation = Gf.Rotation(Gf.Vec3d(0, 0, -1), direction)
            xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(rotation.GetQuat())
    for name, x in [('L', -4), ('R', 4)]:
        light = UsdLux.RectLight.Define(stage, '/Stage/Lights/Fill' + name)
        light.CreateWidthAttr(6)
        light.CreateHeightAttr(1)
        light.CreateIntensityAttr(1500)
        light.CreateExposureAttr(0)
        light.CreateColorAttr(Gf.Vec3f(.9, .93, 1))
        light.CreateNormalizeAttr(False)
        # RectLight emits along local -Z: rotate +70° around X for +Y, 20° down.
        transform(light, (x, -5.5, 3), (70, 0, 0))
    dome = UsdLux.DomeLight.Define(stage, '/Stage/Lights/Ambient')
    dome.CreateIntensityAttr(60)
    dome.CreateColorAttr(Gf.Vec3f(1))
    for name, z in [('Bottom', .3), ('Top', 5.3)]:
        light = UsdLux.RectLight.Define(stage, '/Stage/Lights/EdgeWash' + name)
        light.CreateWidthAttr(8)
        light.CreateHeightAttr(.4)
        light.CreateIntensityAttr(800)
        light.CreateColorAttr(Gf.Vec3f(*GREEN))
        light.CreateNormalizeAttr(False)
        transform(light, (0, 6.9, z), (-90, 0, 0))
    stage.GetRootLayer().Save()


def capsule(stage, path, radius, height, position, mat):
    shape = UsdGeom.Capsule.Define(stage, path)
    shape.CreateAxisAttr(UsdGeom.Tokens.z)
    shape.CreateRadiusAttr(radius)
    shape.CreateHeightAttr(height)
    shape.CreateExtentAttr([(-radius, -radius, -height / 2 - radius),
                            (radius, radius, height / 2 + radius)])
    transform(shape, position)
    bind(shape, mat)


def presenter(out):
    stage = new_stage(out / 'presenter.usda', 'Presenter')
    clothing = material(stage, '/Presenter/Looks/Clothing', (.28, .28, .28), .7)
    head_mat = material(stage, '/Presenter/Looks/Head', (.55, .55, .55), .8)
    joint(stage, '/Presenter/Pelvis', (0, 0, .98))
    capsule(stage, '/Presenter/Pelvis/Torso', .16, .45, (0, 0, .33), clothing)
    joint(stage, '/Presenter/Pelvis/Neck', (0, 0, .6))
    head = UsdGeom.Sphere.Define(stage, '/Presenter/Pelvis/Neck/Head')
    head.CreateRadiusAttr(.11)
    head.CreateExtentAttr([(-.11, -.11, -.11), (.11, .11, .11)])
    transform(head, (0, 0, HEAD_Z))
    bind(head, head_mat)
    for side, sign in [('L', 1), ('R', -1)]:
        hip = '/Presenter/Hip' + side
        joint(stage, hip, (0, sign * .1, .98))
        capsule(stage, hip + '/Thigh', .075, .42, (0, 0, -.25), clothing)
        joint(stage, hip + '/Knee', (0, 0, -.48))
        capsule(stage, hip + '/Knee/Shin', .06, .4, (0, 0, -.24), clothing)
        foot = UsdGeom.Cube.Define(stage, hip + '/Knee/Foot')
        foot.CreateSizeAttr(1)
        foot.CreateExtentAttr([(-.5, -.5, -.5), (.5, .5, .5)])
        transform(foot, (.06, 0, FOOT_Z), scale=(.26, .1, .06))
        bind(foot, clothing)
        shoulder = '/Presenter/Shoulder' + side
        joint(stage, shoulder, (0, sign * .24, 1.42))
        capsule(stage, shoulder + '/UpperArm', .05, .30, (0, 0, -.17), clothing)
        joint(stage, shoulder + '/Elbow', (0, 0, -.33))
        capsule(stage, shoulder + '/Elbow/Forearm', .045, .28, (0, 0, -.16), clothing)
    stage.GetRootLayer().Save()


def readme(out):
    (out / 'README.md').write_text(f'''# CASCADE keynote assets

Visual-only, metre-scale, Z-up USD assets for Isaac Sim 6.2. No physics schemas or
ground collider are authored. The simulation supplies the physical ground at Z=0.
Audience is at negative Y; the stage extends forward along +Y. Presenter forward
is +X; positive Y is its left side. Both files have an Xform default prim.

## Reproduction

From the repository directory, use the supplied OpenUSD 0.26.8 environment:

```sh
/home/johnny/Projects/demo/cascade-lab/ISAACLAB_MICRODUCK_REPRO_20261005/fork/.venv/bin/python make_assets.py --out assets
/home/johnny/Projects/demo/cascade-lab/ISAACLAB_MICRODUCK_REPRO_20261005/fork/.venv/bin/python validate_assets.py --out assets
```

Textures use the installed DejaVu Sans regular and bold fonts with Pillow; no
network, random values, timestamps, or absolute asset paths enter generated USD.
Keep each USD file beside `textures/` when relocating it. Texture asset paths are
relative. Emission is encoded as linear color multiplied by its intensity.

## V2 lighting and materials

The floor now has a soft sheen (base 0.05, roughness 0.45, specular 0.5), reducing
LED-wall reflections around the small robots. Front-edge emission is 1.5; the
LED wall and side screens use emission gains 1.6 and 1.2. Presenter clothing is
mid charcoal (base 0.28, roughness 0.7), with head base 0.55.

Twelve overhead spots cover two rows, aimed 1 m forward onto the visible floor
plane at Z=0.0015. Two broad audience-side fills face +Y and 20° downward to
light subjects facing the camera. Spot and fill exposure is explicitly 0;
brightness is authored with intensity. The dome remains at 60, and the green
edge washes are reduced to 800. The version 2 validator intersects each spot's
world-space -Z ray with the floor and requires a hit within 0.3 m of its target.
Geometry dimensions, existing prim paths and the animation contract are unchanged.

## Resolution of contradictory task dimensions

The literal side-screen positions obstruct the specified walking volume. The
literal foot centres yield bottoms at -0.03 m; the literal head position yields
a top at 1.79 m and a full figure height of 1.82 m. These cannot simultaneously
satisfy the clearance and standing-pose requirements. Three small, explicit
coordinate changes prioritize those requirements; all other dimensions remain
as specified:

- Side-screen bottoms are raised from 1.2 to **2.0 m**; the 2.5 × 1.6 m panels,
  X/Y centres, and 20° inward angles are preserved. Their bottoms sit at the
  excluded upper boundary of the walking volume.
- Each foot centre is raised 3 cm, from knee-local Z=-0.50 to **-0.47 m**, placing
  its bottom at Z=0 without moving the root, hips, knees, or shins.
- The head centre is lowered 2 cm, from neck-local Z=0.10 to **0.08 m**, preserving
  its 0.11 m radius and producing a **1.77 m** standing height.

The validator checks these explicit revised coordinates at the original 1 cm
bound tolerance, as well as independently checking clearance, grounded feet and
the 1.75 ± 0.03 m height requirement. It does not waive those checks.

## Stage prims

All stage geometry is static. Meshes have explicit face normals, right-handed
winding, no subdivision, and single-sided faces. The three screen quads have ST UVs.

| Prim below `/Stage` | Type | Dimensions and placement (metres) |
| --- | --- | --- |
| `/Stage` | Xform | Default prim, identity |
| `Floor` | Mesh box | X [-9,9], Y [-6,7], Z [-0.06,0.0015]; 18 × 13 × 0.0615 |
| `FrontEdge` | Mesh box | X [-9,9], Y [-6,-5.92], Z [0.0015,0.03]; green emission ×1.5 |
| `LedWall` | Mesh quad | 16 × 5; X [-8,8], Y=7, Z [0.3,5.3]; normal -Y |
| `SideScreenL`, `SideScreenR` | Mesh quads | 2.5 × 1.6; bottom Z={SIDE_BOTTOM:g}; centres X=-7.2/+7.2, Y=6.2; yaw +20°/-20° inward |
| `Truss` | Xform | Container for three beams |
| `Truss/Front`, `Truss/Middle`, `Truss/Back` | Mesh boxes | 18 × 0.3 × 0.3; centre Z=6.5, Y=-4/0/4 |
| `Lights` | Xform | Container for seventeen lights |
| `Lights/Spot1`–`Lights/Spot6` | SphereLight | X=-6/-3.6/-1.2/1.2/3.6/6, Y=-2.5, Z=6.3; radius 0.25 |
| `Lights/Spot7`–`Lights/Spot12` | SphereLight | X=-6/-3.6/-1.2/1.2/3.6/6, Y=1.5, Z=6.3; radius 0.25 |
| `Lights/FillL`, `Lights/FillR` | RectLight | 6 × 1; X=-4/+4, Y=-5.5, Z=3; face +Y, 20° down; color (0.9,0.93,1); intensity 1500 |
| `Lights/Ambient` | DomeLight | Intensity 60; white; no texture |
| `Lights/EdgeWashBottom`, `Lights/EdgeWashTop` | RectLight | 8 × 0.4; X=0, Y=6.9, Z=0.3/5.3; face -Y; green; intensity 800 |
| `Looks` | Scope | Material container |
| `Looks/Floor`, `Looks/GreenStrip`, `Looks/Truss`, `Looks/LedWall`, `Looks/SideScreen` | Material | PreviewSurface networks; `Surface` child in each |

Spots are warm white (1,0.95,0.9), intensity 60000, exposure 0, cone angle 40°,
softness 0.4 via UsdLux.ShapingAPI. Each local -Z light axis points toward
(its X, its Y+1, 0.0015), using translate then orient xformOps. Rect lights use
translate then rotateXYZ: fills (70,0,0), edge washes (-90,0,0).
Floor: base 0.05, roughness 0.45, metallic 0, specular workflow/color 0.5.
Truss: base 0.18, roughness 0.4, metallic 0.9.
Screen materials use `UVReader` (UsdPrimvarReader_float2, st) → `Texture`
(UsdUVTexture, sRGB) → `Surface` (UsdPreviewSurface emissiveColor, LED-wall gain
1.6 and side-screen gain 1.2).
`textures/ledwall.png` is 3840 × 1280; `textures/side.png` is 1280 × 800.

## Presenter prims and animation contract

All positions below are local to the immediate parent. Capsule `height` is the
straight cylinder length; full tip-to-tip length is height + 2 × radius.
Joint capsules intentionally overlap slightly to conceal articulation seams.

| Prim below `/Presenter` | Type | Local translation; dimensions |
| --- | --- | --- |
| `/Presenter` | Xform | (0,0,0); floor origin between feet; forward +X |
| `Pelvis` | Xform | (0,0,0.98) |
| `Pelvis/Torso` | Capsule | (0,0,0.33); Z axis, radius 0.16, height 0.45 |
| `Pelvis/Neck` | Xform | (0,0,0.6) |
| `Pelvis/Neck/Head` | Sphere | (0,0,{HEAD_Z:g}); radius 0.11 |
| `HipL`, `HipR` | Xform | (0,+0.1,0.98), (0,-0.1,0.98) |
| `HipL/Thigh`, `HipR/Thigh` | Capsule | (0,0,-0.25); Z axis, radius 0.075, height 0.42 |
| `HipL/Knee`, `HipR/Knee` | Xform | (0,0,-0.48) |
| `HipL/Knee/Shin`, `HipR/Knee/Shin` | Capsule | (0,0,-0.24); Z axis, radius 0.06, height 0.40 |
| `HipL/Knee/Foot`, `HipR/Knee/Foot` | Cube | (0.06,0,{FOOT_Z:g}); 0.26 × 0.1 × 0.06, toe toward +X |
| `ShoulderL`, `ShoulderR` | Xform | (0,+0.24,1.42), (0,-0.24,1.42) |
| `ShoulderL/UpperArm`, `ShoulderR/UpperArm` | Capsule | (0,0,-0.17); Z axis, radius 0.05, height 0.30 |
| `ShoulderL/Elbow`, `ShoulderR/Elbow` | Xform | (0,0,-0.33) |
| `ShoulderL/Elbow/Forearm`, `ShoulderR/Elbow/Forearm` | Capsule | (0,0,-0.16); Z axis, radius 0.045, height 0.28 |
| `Looks` | Scope | Material container |
| `Looks/Clothing`, `Looks/Head` | Material | `Surface` PreviewSurface child; matte base 0.28/0.55, roughness 0.7/0.8 |

Every joint Xform, including root, Pelvis, Neck, both Hips/Knees and both
Shoulders/Elbows, has **exactly** `xformOp:translate`, `xformOp:rotateXYZ`, in that
order. All rotations start at zero; there are no animation samples. The driver
overwrites root translate + rotateXYZ every frame, and rotateXYZ on the joints.
Preserve each joint's authored translation and the geometry's own transforms.
Yaw the root about Z to steer. Rotations are degrees.

For a walk, let phase = 2π × stride_frequency × time. Swing the hips ±25° about
Y out of phase: left = 25 sin(phase), right = -25 sin(phase). A positive hip Y
angle moves the hanging leg behind the +X-facing figure; flex that knee toward
the back using rotateY = 35 max(0, sin(phase)) on the left and
35 max(0, -sin(phase)) on the right. Counter-swing the shoulders ±20° about Y;
left = -20 sin(phase), right = +20 sin(phase). Optional elbow flex can soften the
pose. Add a 2 cm root bob at twice the stride frequency (for example
0.01 × (1 - cos(2 × phase)), a 0–2 cm lift). Nominal step length is 0.65 m, so
root speed is about 1.3 × stride_frequency metres/second. This is a kinematic
proxy; foot planting/IK belongs to the driver. No physics or skeleton is needed.
''', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('assets'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    textures(args.out)
    # Clear only the two files owned by this generator, permitting repeat runs.
    for filename in ('stage.usda', 'presenter.usda'):
        (args.out / filename).unlink(missing_ok=True)
    keynote(args.out)
    presenter(args.out)
    # OpenUSD adds a blank line after the root prim; keep tracked files tidy.
    for filename in ('stage.usda', 'presenter.usda'):
        path = args.out / filename
        path.write_text(path.read_text(encoding='utf-8').rstrip() + '\n', encoding='utf-8')
    readme(args.out)


if __name__ == '__main__':
    main()
