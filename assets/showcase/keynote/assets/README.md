# CASCADE keynote assets

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
| `SideScreenL`, `SideScreenR` | Mesh quads | 2.5 × 1.6; bottom Z=2; centres X=-7.2/+7.2, Y=6.2; yaw +20°/-20° inward |
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
| `Pelvis/Neck/Head` | Sphere | (0,0,0.08); radius 0.11 |
| `HipL`, `HipR` | Xform | (0,+0.1,0.98), (0,-0.1,0.98) |
| `HipL/Thigh`, `HipR/Thigh` | Capsule | (0,0,-0.25); Z axis, radius 0.075, height 0.42 |
| `HipL/Knee`, `HipR/Knee` | Xform | (0,0,-0.48) |
| `HipL/Knee/Shin`, `HipR/Knee/Shin` | Capsule | (0,0,-0.24); Z axis, radius 0.06, height 0.40 |
| `HipL/Knee/Foot`, `HipR/Knee/Foot` | Cube | (0.06,0,-0.47); 0.26 × 0.1 × 0.06, toe toward +X |
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
