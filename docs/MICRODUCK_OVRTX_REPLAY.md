# MicroDuck OVRTX replay

`src/cascade/sim/microduck_ovrtx_replay.py` replays recorded MicroDuck physics (body world poses at 50 Hz) through
Cascade's `OvrtxRenderer`. There is no physics stepping and no pose repair: each frame shows the recorded poses.

## Render scene

`build_render_scene(out, course_usd=..., microduck_usd=..., robots=[...])` writes one flattened, metre-authored, Z-up
USD containing:

* the course (any metre Z-up USD; `guide`-purpose meshes are not rendered by OVRTX);
* a floor;
* a dome light and a key light;
* one copy of the canonical `MicroDuck.usd` per robot. The sha256 is pinned
  (`feca1ff67d2e4140df48469cbe51fad69886c9447929941fc056a0e668b31235`), and the build refuses any other file.

Each of the 16 rigid bodies becomes a flat prim `/World/DuckBodies/<robot>/<body>` that carries the body's own visual
subtree. Every body therefore receives its recorded world pose directly. Course meshes without a material get a
`UsdPreviewSurface`, built from their `displayColor` or, failing that, a neutral blue-grey.

`ReplayPoses(layout, body_names, robot_names).transforms(frame)` returns the `{prim: 4x4}` dict that `SceneSnapshot`
expects. Rotations are re-orthonormalized because float32 recordings drift slightly from rigid.

## SDK pair

`OvrtxRenderer` pins ovrtx 0.5.0.377615 / ovstage 0.2.0.377349, and that pin stays the default. The Isaac Lab 3.0
venv ships 0.5.1.385782 / 0.2.1.385922, so `OvrtxRenderer(..., sdk_versions=...)` accepts a pair. Only pairs listed
in `KNOWN_SDK` are allowed, and an unknown pair raises before any native call.

## Finding: the first write query goes stale (ovrtx 0.5.1 / ovstage 0.2.1)

`OvrtxRenderer.render()` writes `omni:xform` through one single-prim query per dynamic path. It writes them in
`dynamic_paths` order, then advances the write floor and steps. The measured behaviour is:

* The prim listed **first** takes its frame-1 pose.
* Later writes to that first prim are not observed, even though every later prim moves correctly.
* Reordering moves the problem: when the trunk is listed last, the new first prim (a hip link) is the one left
  behind.

The recorded data are sound. Each body's local transform reproduces the USD rest offsets to 0.1 mm on every sampled
frame.

**Workaround in the replay:** an empty Xform `/World/ReplayWriteSentinel` is listed first and written as identity on
every frame. With it, all sampled frames render with every body attached (see the evidence folder). The root cause
inside the SDK was not investigated further. Other `OvrtxRenderer` users on 0.5.1 should check whether their first
dynamic prim moves.

Evidence and probe: `docs/evidence/microduck-ovrtx-replay-20261013/`.
