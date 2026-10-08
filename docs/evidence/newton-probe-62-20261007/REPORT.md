# B16 — real reBot asset on Newton and PhysX, internal Isaac Sim 6.2 build (7 October 2026)

Hermes backlog item B16 (Newton physics probe against the real reBot asset) plus the
sim half of B18 (wrist-camera extrinsics validated mid-descent). Measured by Hermes on
the x86 test rig, GPU 0, 7 October 2026 evening; record written 8 October from the raw
rows listed in [`manifest.json`](manifest.json) (path, bytes, SHA-256 of every raw
file; the raw files stay in `cascade-lab/`).

- Builds: Isaac Sim **6.2.0-alpha.19** (`omni_isaac_sim` develop `48b2d951`), Newton
  1.6.1rc1, Warp 1.17.0; baseline Isaac Sim **6.1.0-rc.26** (develop `2469084b`,
  `~/Projects/isaac/IsaacSim`), Newton 1.5.0, Warp 1.16.0.
- Scene: the bare reBot scene `assets/usd/RS-rebot-dev-arm/RS-rebot-dev-arm.usda`
  with the two 0.05 × 0.05 × 0.08 m boxes the bridge authors (`pink_cube` at
  r = 0.227 m, `green_cube` at r = 0.340 m). Not the kitchen.
- Bridge: `scripts/isaac_bridge.py` on a private port (18511–18513) through
  `scripts/isaac_launch.py`, `CASCADE_ISAAC_PIXEL_MASK=1`, pinned to GPU 0 by UUID:

      env CASCADE_ISAAC_PIXEL_MASK=1 CUDA_VISIBLE_DEVICES=GPU-<uuid> .venv/bin/python scripts/isaac_launch.py \
        --python <build>/_build/linux-x86_64/release/python.sh -- "$PWD/scripts/isaac_bridge.py" \
        --port <port> --usd "$PWD/assets/usd/RS-rebot-dev-arm/RS-rebot-dev-arm.usda" --engine <newton|physx>

## 1. Start-up deadlock on the 6.2 build (fixed in `scripts/isaac_bridge.py`)

`--engine newton --headless` never served on 6.2: ~25k frames of `await_viewport:
waiting for viewport handle`, the Render Async thread at 100 %, no `[bridge]` line,
with and without GPU pinning. The Newton launch selects the full app experience,
whose `isaacsim.app.setup` waits for the viewport's first frame before app-ready;
`SimulationApp(disable_viewport_updates=True)` stops the viewport at construction.
With the fix (keep the viewport on until `is_app_ready()`, then switch its updates
off) the gated bridge logs `[bridge] viewport updates disabled after app-ready` on
Newton 6.2 and Newton 6.1 (`bridge_newton62_gated.log`, `bridge_newton61_gated.log`);
the PhysX 6.2 bridge keeps the construction-time switch (`Viewport updates disabled`
at 5.6 s, `bridge_physx62_gated.log`). Unit tests: `tests/test_isaac_viewport_ready.py`,
`tests/test_isaac_gui_framing.py`.

## 2. Probe battery (`scripts/physics_probe.py --port <port> --engine <engine> --report <json>`)

The July battery brought up to the current bridge (props from the bridge's own
`_PROP_SPAWNS`, teleports through the engine-aware `place_prop` op, a stopped
timeline resumed first, build identity recorded, an unreachable lift reported as
`skipped`). Result: **Newton 6.2 8/8, PhysX 6.2 8/8, Newton 6.1 8/8.**

| test | Newton 6.2 | PhysX 6.2 | Newton 6.1 |
| --- | --- | --- | --- |
| settle `pink_cube` | pass (jitter 0.0 m) | pass (jitter 0.0 m) | pass (jitter 0.0 m) |
| settle `green_cube` | pass (jitter 0.0 m) | pass (jitter 0.0 m) | pass (jitter 0.0 m) |
| drop `pink_cube` | pass (settle 0.38 s, z 0.0399 m) | pass (settle 0.38 s, z 0.04 m) | pass (settle 0.35 s, z 0.0399 m) |
| drop `green_cube` | pass (settle 0.3 s, z 0.0399 m) | pass (settle 0.38 s, z 0.04 m) | pass (settle 0.41 s, z 0.0399 m) |
| grasp `pink_cube` | pass (TCP err 3.7 mm, lift +0.12 m → dz 0.119 m) | pass (TCP err 2.3 mm, lift +0.12 m → dz 0.119 m) | pass (TCP err 3.7 mm, lift +0.12 m → dz 0.119 m) |
| grasp `green_cube` | pass (TCP err 5.4 mm, lift +0.10 m → dz 0.099 m) | pass (TCP err 5.1 mm, lift +0.10 m → dz 0.093 m) | pass (TCP err 5.4 mm, lift +0.10 m → dz 0.099 m) |
| push `pink_cube` | pass (moved 0.0874 m) | pass (moved 0.0066 m) | pass (moved 0.0672 m) |
| push `green_cube` | pass (moved 0.0639 m) | pass (moved 0.0167 m) | pass (moved 0.0661 m) |

## 3. Why the far green box "failed" on Newton before the probe fix

At r = 0.340 m a top-down TCP 0.12 m above the grasp height is outside the IK
envelope on this asset (+0.10 m solves); `goto` returned False without moving and
the old probe scored the untouched box as a failed lift. Before the fix the far-box
grasp failed on PhysX as well (`probe_physx_62_gated.json`), not only on Newton
(`probe_newton_62_gated.json`, `_run1`, `_run2`). Attribution on Newton 6.2
before the fix, with the probe's own grasp recipe:

Swapping the boxes (`harness/swap_grasp.py <port> newton <json>`): the failure
follows the spot, not the box.

| run | lifted | dz (m) | TCP err (mm) |
| --- | --- | --- | --- |
| green at pink's spot (r=0.227 m) | True | 0.119 | 3.7 |
| pink at green's spot (r=0.340 m) | False | 0.0 | 5.4 |
| green at its own spot (control) | False | 0.0 | 5.4 |
| pink at its own spot (control) | True | 0.119 | 3.7 |

Corner versus face contact (`harness/face_grasp.py <port> newton <json> 2`): a
face-aligned jaw (fingers at 0.0252 m, the 5 cm face) fails identically at the far
spot and lifts at the near spot, so neither corner contact nor the box causes it.

| spot | jaw | trial | jaw − cube yaw (°) | fingers (m) | lifted |
| --- | --- | --- | --- | --- | --- |
| near r=0.227 | radial | 0 | 41.4 | 0.0351 | True |
| near r=0.227 | radial | 1 | 41.4 | 0.0351 | True |
| near r=0.227 | face | 0 | 0.0 | 0.0252 | True |
| near r=0.227 | face | 1 | 0.0 | 0.0252 | True |
| far r=0.340 | radial | 0 | 28.1 | 0.0323 | False |
| far r=0.340 | radial | 1 | 28.1 | 0.0322 | False |
| far r=0.340 | face | 0 | 0.0 | 0.0252 | False |
| far r=0.340 | face | 1 | 0.0 | 0.0252 | False |

## 4. Wrist camera validated mid-descent (B18, sim part)

`harness/wrist_reproj.py <port> <engine> <json>`: a slow top-down descent/ascent over
the resting `green_cube` (no contact, `cube_moved_m` = 0.0 on both engines); every
wrist frame's own `K` and `T_base_cam` project the box's physics-truth corners and
top-face centre, compared with the colour-mask silhouette and the frame's depth.

| engine | phase | frames | centroid err median / p95 / max (px) | IoU median (min) | depth err median (m) |
| --- | --- | --- | --- | --- | --- |
| physx | static | 69 | 0.83 / 1.01 / 1.02 | 0.995 (0.994) | 0.0 |
| physx | moving | 198 | 0.69 / 0.96 / 1.07 | 0.997 (0.993) | 0.0 |
| newton | static | 54 | 2.63 / 7.67 / 7.71 | 0.982 (0.969) | 0.0 |
| newton | moving | 104 | 2.73 / 7.59 / 9.38 | 0.984 (0.964) | 0.0 |

Moving error does not exceed static error on either engine, so the per-frame
extrinsics are time-aligned with the render. The bridge refused 246 wrist refreshes
during the Newton run (no matching state history, section 5) and none on PhysX; the
table covers the frames it served. Not covered: the real D435i hand-eye calibration (B25).

## 5. Open: Newton camera cadence on the 6.2 build

`harness/frame_gap_probe.py <port> newton <json>` samples frame publication at ~10 Hz
while the arm streams. Every refused refresh carries `render frame has no unique
matching state history` (fail-closed: the render's time has no recorded state; no
frame was ever served with the wrong state).

| run | idle: refreshes without frame | arm streaming: refreshes without frame |
| --- | --- | --- |
| Newton 6.2, run 1 | 1/77 (1.3 %) | 56/164 (34.1 %) |
| Newton 6.2, run 2 | 2/75 (2.7 %) | 64/161 (39.8 %) |
| Newton 6.2, `useFixedTimeStepping` toggled | 2/76 (2.6 %) | 58/163 (35.6 %) |
| Newton 6.1 (baseline build) | 1/81 (1.2 %) | 4/145 (2.8 %) |

PhysX 6.2 refused no frame during the wrist run (section 4). The Newton experience
advances three physics steps per `app.update()` on both builds, yet 6.1 drops 2.8 %
of streaming refreshes and 6.2 34–40 %, so the step count alone does not explain the
6.2 rate; toggling `/app/player/useFixedTimeStepping` live did not change it. Cause
open; candidate fixes to measure: record a state per physics step so an intermediate
render binds exactly, or one physics step per update on the Newton app loop.

## Not claimed

No physical acceptance, no kitchen-scene result, no DGX Spark (GB10) run, no change of
any limit, gain, contact setting or asset, no real hand-eye calibration, and no fix
for the 6.2 Newton camera cadence.
