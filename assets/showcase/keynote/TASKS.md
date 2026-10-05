# Task: author two USD assets for Isaac Sim 6.2 (keynote stage + walking-presenter proxy)

Work ONLY inside this directory (a scratch git repo). Do not touch anything else on this machine.

## Tooling
- Python with `pxr` (OpenUSD 0.26.8), Pillow and numpy: `/home/johnny/Projects/demo/cascade-lab/ISAACLAB_MICRODUCK_REPRO_20261005/fork/.venv/bin/python`
  (use it for generation AND validation; do not pip install anything; no network).
- Deliver a single generator script `make_assets.py` (run: `<python> make_assets.py --out assets`) that writes
  `assets/stage.usda`, `assets/presenter.usda`, `assets/textures/*.png`, plus `assets/README.md`
  (what each prim is, dimensions, how to animate the presenter). Keep the generator deterministic.
- Add `validate_assets.py`: opens both stages with pxr, checks every prim path listed below exists with the
  right type, bounds (UsdGeom.BBoxCache) are within 1 cm of the spec, no prim has any `Physics*` API applied,
  and prints a JSON report. Run both scripts; they must exit 0. Commit the result (`git add -A && git commit`).

## Coordinate frame (metres, Z up, Y forward on stage, X across the stage)
- World origin = centre of the stage floor's front half. Audience/camera is at negative Y looking toward +Y.
- Ground collider at z = 0 belongs to the simulation and is NOT part of these assets. The stage floor is a
  VISUAL slab whose TOP face is at z = +0.0015 (1.5 mm above the physical ground) so robots appear to walk
  on it. Do NOT add UsdPhysics APIs anywhere (no CollisionAPI, no RigidBodyAPI, no physics materials).

## stage.usda — `/Stage` (Xform, default prim), a NVIDIA-GTC-keynote-style stage, everything static
- `/Stage/Floor` (Mesh, box): x in [-9, 9], y in [-6, 7], z in [-0.06, 0.0015]. Material: very dark glossy
  (base color ~0.03, roughness 0.12, metallic 0.0, specular high) — a near-black reflective keynote floor.
- `/Stage/FrontEdge` (Mesh): thin emissive strip along the front edge, x in [-9, 9], y in [-6.0, -5.92],
  z in [0.0015, 0.03], emissive color NVIDIA green (0.46, 0.73, 0.0), emissive intensity ~8.
- `/Stage/LedWall` (Mesh, flat quad, single-sided facing -Y): x in [-8, 8], z in [0.3, 5.3], at y = 7.0.
  Emissive material with texture `textures/ledwall.png` (3840x1280). Render the texture with Pillow:
  near-black background (#06080c) with a subtle fine grid of dark LED pixels, a large centred title
  "CASCADE" in white, subtitle "12 MicroDucks · Isaac Sim 6.2 · Newton" in light grey, a thin NVIDIA-green
  (#76b900) horizontal rule, and small footer text "zero-shot local inference · sim → real". No logos of
  companies (text only). Emissive intensity ~3 so it reads as a screen without blowing out.
- `/Stage/SideScreenL`, `/Stage/SideScreenR` (Mesh quads facing -Y, slightly angled toward centre by 20°):
  2.5 m wide x 1.6 m tall, bottom at z = 1.2, centred at x = ∓7.2 (L at negative x), y = 6.2. Texture
  `textures/side.png` (1280x800): dark background with the word "MicroDuck" and a simple green waveform line.
- `/Stage/Truss` (Xform) with 3 horizontal box beams (0.3 m x 0.3 m cross-section) at z = 6.5,
  y = -4, 0, 4, spanning x in [-9, 9], material dark grey metal (base 0.18, metallic 0.9, roughness 0.4).
- `/Stage/Lights` (Xform):
  - 6 `UsdLux.SphereLight` "stage spots": radius 0.15, intensity 30000, color warm white (1.0, 0.95, 0.9),
    positioned at z = 6.3 on the y = -4 and y = 0 beams at x = -6, 0, 6; each with a `UsdLux.ShapingAPI`
    cone angle 35°, softness 0.3, aimed at the floor point below-and-forward (orient with xformOps).
  - 1 `UsdLux.DomeLight` intensity 60 (dim ambient), no texture.
  - 2 `UsdLux.RectLight` as green edge washes behind the LED wall, 8 m x 0.4 m, intensity 2000, color (0.46, 0.73, 0.0), at y = 6.9, z = 0.3 and z = 5.3, facing -Y.
- All meshes must have proper normals and `subdivisionScheme = none`; UV coordinates (`primvars:st`) on the
  textured quads. Materials via UsdPreviewSurface + UsdUVTexture + UsdPrimvarReader_float2.
- Nothing may extend below z = -0.06 or into x in [-8.5, 8.5], y in [-5.5, 6.5] at heights 0.0015 < z < 2.0
  except the floor slab (the robots and presenter walk there).

## presenter.usda — `/Presenter` (Xform, default prim), a stylised walking human proxy, ~1.75 m tall
A clean low-poly mannequin (capsules/boxes/spheres are fine; UsdGeom.Capsule/Sphere/Cube allowed), dark
charcoal matte clothing (base 0.08, roughness 0.8), a slightly lighter head (0.35), and a thin NVIDIA-green
emissive line down each sleeve (optional). Forward = +X of `/Presenter` (the controller yaws the root).
Hierarchy (every joint prim is an Xform with xformOps in this exact order: `xformOp:translate`,
`xformOp:rotateXYZ`; the animation driver will overwrite `rotateXYZ` on the joints and `translate`+`rotateXYZ`
on the root every frame, so author those ops explicitly even if zero):
- `/Presenter` root: translate (0,0,0), rotateXYZ (0,0,0). The root origin is on the floor between the feet.
- `/Presenter/Pelvis` translate (0, 0, 0.98): torso capsule `/Presenter/Pelvis/Torso` (axis Z, radius 0.16,
  height 0.45, centred at z=+0.33 above the pelvis), `/Presenter/Pelvis/Neck/Head` sphere radius 0.11 at
  z = +0.70 above the pelvis (Neck is an Xform at (0,0,0.6)).
- Legs: `/Presenter/HipL` translate (0, 0.1, 0.98), `/Presenter/HipR` translate (0, -0.1, 0.98). Under each:
  `Thigh` capsule (radius 0.075, height 0.42) hanging DOWN from the hip (centre at z = -0.25 relative to
  the hip), then `Knee` Xform at (0,0,-0.48) with `Shin` capsule (radius 0.06, height 0.40, centre z=-0.24)
  and `Foot` cube (0.26 x 0.1 x 0.06) at (0.06, 0, -0.50). The joint we animate is the Hip (swing about Y).
- Arms: `/Presenter/ShoulderL` translate (0, 0.24, 1.42), `/Presenter/ShoulderR` translate (0, -0.24, 1.42):
  `UpperArm` capsule radius 0.05 height 0.30 hanging down (centre z=-0.17), `Elbow` Xform at (0,0,-0.33)
  with `Forearm` capsule radius 0.045 height 0.28 (centre z=-0.16). Swing about Y.
- Standing pose: all joint rotations 0 → a neutral A-pose-ish standing figure whose lowest point is at z=0
  (feet bottoms on the floor) ± 1 cm and total height 1.75 ± 0.03 m. The validator must check both.
- Capsule axis: use `axis = "Z"` capsules. Make sure nothing self-intersects grossly in the standing pose.

## README.md must include
- the prim list with dimensions, the animation contract (which ops the driver writes), and a short
  description of how to make the figure walk: hips swing ±25° about Y out of phase, knees flex 0–35° when
  the leg is behind, arms counter-swing ±20°, root bob 2 cm at twice the stride frequency. Stride length
  ~0.65 m per step at this scale.

Quality bar: this is for a presentation video rendered with RTX; proportions should look deliberate and
clean, not like debug geometry. Keep the total vertex count small (< 50k). Do not spend time on anything
outside this brief. Finish by printing the validator report.
# Task 2: lighting/material revision of the keynote assets (v2), from a real RTX render review

Context: the v1 assets (`assets/stage.usda`, `assets/presenter.usda`) load in Isaac Sim 6.2 and render.
Review of the actual frames (camera at the audience, ~10 m away, scene-level dome light 20 and a weak sun):

1. The glossy floor (roughness 0.12) acts as a mirror: a huge blurred reflection of the LED-wall text covers
   the floor and competes with the subjects. The small robots (0.2 m tall) are invisible against it.
2. The front-edge green strip (emissive intensity 8) is the brightest thing in frame: blown out.
3. The presenter reads as a near-black silhouette (clothing base 0.08; the spots do not light it visibly).
4. The stage itself is almost black apart from emissive surfaces; the 6 SphereLight spots are not visibly
   lighting the floor where the presenter and robots walk (y in [-1, 4], x in [-7, 6]).
5. Everything else is good: geometry, text legibility, truss, proportions, no physics APIs, validator passes.

## Changes for v2 (keep everything else; keep the same prim paths, dimensions and the animation contract)
- Floor material: roughness 0.45, base color 0.05, specular 0.5 — still a dark keynote floor, but the
  reflection becomes a soft sheen, not a mirror.
- FrontEdge emissive intensity 1.5; LedWall emissive intensity 1.6; side screens 1.2.
- Presenter: clothing base color 0.28 (mid charcoal), roughness 0.7; head 0.55; keep the green sleeve lines.
  Add a lighter shirt-like band on the torso? Not required — simplicity first.
- Stage spots: raise to 12 SphereLights (two rows y = -2.5 and y = 1.5 at x = -6, -3.6, -1.2, 1.2, 3.6, 6,
  z = 6.3), radius 0.25, intensity 60000, cone 40°, softness 0.4, each aimed at the floor point directly
  below it offset +1.0 m in y (so the strip y in [-2.5, 3.5] is evenly lit). Verify the aim maths in the
  validator: compute each light's -Z axis in world and check it hits the floor plane within 0.3 m of the
  intended point. Use `UsdLux.ShapingAPI` cone angle/softness and keep `intensity` with `exposure 0`.
- Add 2 soft fill RectLights at the audience side: 6 m x 1 m, intensity 1500, color (0.9, 0.93, 1.0),
  at y = -5.5, z = 3.0, x = -4 and x = 4, facing +Y and slightly down (-20°), so subjects facing the
  camera are not black.
- Keep the stage DomeLight at 60 (unchanged). Keep the edge washes but reduce to intensity 800.
- Update `assets/README.md` with the v2 lighting notes and bump a `version` field in the validator report.

Run `make_assets.py` then `validate_assets.py --out assets` with the same python as before
(`/home/johnny/Projects/demo/cascade-lab/ISAACLAB_MICRODUCK_REPRO_20261005/fork/.venv/bin/python`),
both must exit 0 and the report must be `pass`; commit as "v2 lighting/material revision". Print the report.
Do not touch anything outside this directory.
