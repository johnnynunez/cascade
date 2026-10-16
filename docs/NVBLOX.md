# nvblox and the kitchen cameras

**Next action:** build nvblox into its own prefix and check it on the GPU (Linux x86_64 or Spark
aarch64 with a C++ compiler, `git`, `uv` and an NVIDIA driver; the build takes a while, budget
30–60 min):

```bash
bash scripts/install_nvblox_source.sh
# Optional: --prefix /path/to/private/nvblox --jobs 4

export LD_LIBRARY_PATH="$PWD/.nvblox/cuda-13.2.2/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
.nvblox/venv/bin/python scripts/check_nvblox.py --output runs/nvblox-check.json
.nvblox/venv/bin/python scripts/serve_occupancy_bridge.py --backend nvblox --port 5557
```

The installer extracts NVIDIA's checksummed CUDA redistributions into its prefix, builds the
pinned source and requires real depth integration, ESDF queries and map clearing on the GPU. Use
explicit `--backend nvblox` when evaluating it: `auto` can select another implementation if nvblox
fails to initialize.

Status:

1. **Works:** real nvblox CUDA mapping on RTX PRO 6000 Blackwell and DGX Spark GB10; the reBot runs
   [always-on occupancy](#always-on-occupancy-on-the-rebot) with a mesh body mask.
2. **Off by default** in the Spark presenter profile (the published GraspGen-X acceptance used
   `occupancy=disabled`); see [current acceptance](ROADMAP.md).
3. **Not shown:** five-object or presenter acceptance with the map, more accurate grasping or
   placement. Mapping is a clearance gate, not localization input.

## Build details

Both machines compiled nvblox v0.0.10 source `c457c3fc01003bec6eba3ec1c61e6bf84bc3f51f` with a
private CUDA **13.2.2** (nvcc 13.2.86) and **PyTorch 2.14.1+cu132** (SM120 / SM121), leaving system
CUDA, drivers, Isaac Sim and the app venv in place. The compatibility patch fixes Blackwell
architecture parsing, uses C++20 for the PyTorch wrapper and resolves the ARM64 cuDNN from PyTorch;
upstream metadata caps Torch at 2.9.1, so the wrapper is installed with `--no-deps`. This is a
tested local build, not an upstream support claim (the
[upstream release](https://github.com/nvidia-isaac/nvblox/releases/tag/v0.0.10) has no ARM64 wheel;
the package reports `0.0.10.dev1`).

Hardware validation ([receipts](../benchmark/results/), `nvblox-rtx-*` and `nvblox-spark-*`):

1. Both GPUs passed 15 upstream query/frame-integration tests.
2. Through the real ZMQ bridge a synthetic 5 cm cube's five known clearances differed by at most
   **10 mm** at 10 mm voxels; 20 frames after removing it, its position went from 0 to 40 mm
   clearance (real ray carving).
3. The ESDF query needs **Nx4 `(x, y, z, radius)`** input despite the Nx3 docstring; flat SDF masks
   cannot index a 3-D grid.
4. Bridge requests are serialized: a concurrent probe and refresh could corrupt the shared ZMQ REQ
   socket.

## Payload and release contracts

1. Carry attachment: with `CASCADE_ISAAC_CONTACT_MASK=1` the bridge reuses the
   contacts captured after each completed update. Lost or unknown attachment
   during lift or transport is terminal: no automatic open, retry, home or reset.
2. Release retreat: `skills/release_episode.wait_open` waits for open jaws and
   stable measured joints held for `settle_hold_s` (0.1 s, `configs/arms/isaac.yaml`);
   once open and free of bilateral attachment, regressing either is terminal.
3. Retained anchors: for a confirmed attachment,
   `_reset_camera_frames(scene_changed=False)` invalidates the cached map and
   replays only measured depth history; a prop reset calls `begin_scene_reset()`
   and drops all history.
4. Truth views: `TruthPoseReader` (`sim/truth.py`) builds `RigidPrim` views with
   `reset_xform_properties=False` and `prepare_contact_sensors=False`.
5. Mapper reset: a clear replaces the nvblox mapper
   (`perception/occupancy_backends.py`) and keeps it unavailable until
   construction and CUDA synchronization succeed; queries never see the old map.

The carry guard does not correct mechanical slip and is not enabled by occupancy-off presenter
runs. RTX environment runs on Isaac (1 October onward): environment 12 passed a native orange
release/recovery diagnostic ([receipt](evidence/nvblox-environment-12/release.json)), but its
five-object campaign failed the first orange case on real contact loss during transport
([campaign](evidence/nvblox-environment-12/campaign.json)); environments 13–15 stopped fail-closed
(stale analyzed image, 9.5 s analysis over the 5 s image lifetime, the 1 mrad release-feedback
guard). None establishes campaign or presenter acceptance.

## Integration and limitations

1. `occupancy.required: true` (opt-in; default `false`) needs a reachable bridge at startup;
   missing wire dependencies, `enabled: false` or `CASCADE_OCCUPANCY=0` then fail startup. During
   motion an absent or expired grid or a failed refresh raises `SafetyViolation`, cleared only by a
   successful depth refresh. The age bound must be finite and positive.
2. Queries outside the grid or with unknown interpolation support reject waypoints and pose
   candidates outside the local contact exemption, which cannot override a missing, stale or
   failed map.
3. The ordinary world-map consumer reads unknown cells as no known obstacle and an unavailable cache
   as no data (not proof of free space); payload/recovery modes reject unknown samples and require
   fresh, bound integration commits; robot-mask failures fail closed even after the cache ages out.
4. In the camera profiles `isaac` and `isaac_side` feed fusion and `isaac_proof` has
   `fuse_beliefs: false`: three viewer streams are not three map inputs.
5. The grid gates TCP, sampled arm links and payload samples, and vets configured candidate routes;
   it is not an obstacle-avoidance planner, and an idle-scene benchmark cannot establish collision
   safety around unseen geometry. Upstream references: [nvblox interface](https://nvidia-isaac.github.io/nvblox/v0.0.10/pages/core_library_interface.html),
   [multi-camera tutorial](https://nvidia-isaac-ros.github.io/concepts/scene_reconstruction/nvblox/tutorials/tutorial_multi_realsense.html).

### Does another camera improve accuracy?

Replay of archived idle Isaac RGB-D frames (245 known surface samples on each of two cubes, depth
robot-masked at 640x360, each camera integrated five times; same result on both GPUs):

| Voxel | Cameras | Observed samples / 490 | Mean surface residual on observed samples | Samples within one voxel / 490 |
| --- | --- | --- | --- | --- |
| 10 mm | Main | 443 | 3.612 mm | 423 |
| 10 mm | Main + side | 490 | 4.516 mm | 470 |
| 10 mm | Main + side + proof | 490 | 4.421 mm | 474 |
| 5 mm | Main | 383 | 1.971 mm | 360 |
| 5 mm | Main + side | 458 | 2.314 mm | 428 |
| 5 mm | Main + side + proof | 469 | 2.455 mm | 426 |

The second camera improves coverage substantially; the third changes little. A lower residual with
fewer observed samples is not a better reconstruction. Median warm three-camera integration: 2.93 /
18.91 ms/frame (RTX / Spark) at 10 mm, 29.99 / 98.47 at 5 mm. Replay the
[capture](../benchmark/fixtures/nvblox/kitchen-idle-01/):

```bash
.nvblox/venv/bin/python benchmark/diagnostics/nvblox_camera_benchmark.py \
  --repo . --capture benchmark/fixtures/nvblox/kitchen-idle-01 \
  --voxel 0.01 --output runs/nvblox-cameras.json
```

Capture another idle scene with `benchmark/diagnostics/capture_nvblox_scene.py --port 8611 --output
DIR` (app interpreter, bridge with validated robot pixel masks).

### Record the map live

```bash
scripts/record_live_map.py --seconds 60 --out runs/videos/live_map.mp4
```

It shows the cameras' annotated streams above the occupied voxels (3-D orbit plus top-down height
map). It only watches: cameras come from the live-view server (`/stream/<camera>`) and the bridge
is wrapped in `ReadOnlyBridge` (forwards `probe`/`query`, raises on `clear`/`integrate_*`). A
stopped stream is drawn as missing; with `CASCADE_OCCUPANCY=0` the video shows 0 voxels or a stale
map. On the Spark a 1280 x 720 composite writes at 10 fps in real time.

## Robot body mask on real arms (`body_mask: mesh`)

Every depth frame is masked before integration. Capsules (`body_mask_radius_m`, 6 cm around the
joint polyline) left voxels on the reBot's open jaws and wrist D455F 2.8 cm from the chain (under
`min_clearance_m` 0.03), so motion from rest could be refused by the robot's own body. With
`body_mask: mesh` (on for `rebot_rs`), `MeshBodyMask` rasterizes the URDF collision meshes at the
current joint state into the shipped depth image (same code as fusion's B39 link self-mask,
`perception/link_mask.py`). The [lift-first route fallback](CLEARANCE_LIFT_ROUTES.md) is a
separate fix for limited route search, not for mask leakage.

```yaml
body_mask:
  mode: mesh          # capsule (the default when absent) | mesh
  inflate_m: 0.02     # metric margin on every convex piece, applied once at load
  dilate_px: 2        # pixel dilation of the silhouette (flying pixels on its edge)
  cell_m: 0.05        # mesh piece size (LinkGeometry), the default
  attachments:        # rigid parts the URDF lacks: boxes in a link frame
    - {name: wrist_d455f, link: gripper_end, box: [0.065, 0.14, 0.055],
       xyz: [-0.0925, 0.0, 0.0625], rpy: [0, 0, 0]}
```

1. **Fingers** are posed from the measured gripper position (`gripper.open_pos/closed_pos`); when
   the opening is unknown (failed read, or while an RS arm streams a motion) they are swept over
   their whole travel: more mask, never less.
2. **Margins:** `inflate_m` absorbs hand-eye, mesh and finger-map errors and stays below
   `min_clearance_m` (3 cm); `dilate_px` covers the silhouette edge. The wrist D455F box is an
   envelope from the 2026-10-11 capture plus ~1 cm: measure the mount with calipers and tighten it.
3. **Any camera:** the mask needs only K and T_base_cam (fixed for eye-to-hand, FK(q) @ X for
   eye-in-hand) and does not read depth values, so wrong depth on robot pixels is masked too.
4. **Fail closed:** without joint state (LazyArm standby, failed read, non-finite q) the frame is
   not integrated; the fault latches and clearance is refused with `robot body pose unavailable for
   the <arm> mesh mask: ...` until a masked refresh succeeds. nvblox gets the silhouette via
   `integrate_masked_depth`; older bridges get zeroed depth or filtered points.
5. **No silent capsule fallback:** a missing mesh (e.g. git-lfs not fetched) makes a `body_mask:
   mesh` profile refuse to build, naming the link: run `git lfs pull` or set `body_mask: capsule`
   knowingly. Unknown attachment links, `mesh` on Isaac arms and `mesh` with payload tracking are
   refused too.

Measured on the real capture (24 frames, 640x360 after stride 2, q = 0, fingers swept) with
`scripts/eval_body_mask.py ... --tsdf` (cascade's warp TSDF on CPU over the workspace region):

| | capsule (6 cm) | mesh (2 cm, 2 px) |
|---|---:|---:|
| unmasked depth points within 3 cm of the URDF surface, per frame | 3859 | **0** |
| ... within 5 cm, per frame | 4576 | 84 |
| TSDF occupied voxels within 3 cm / 5 cm of the arm | 77 / 132 | **0** / 3 |
| nearest TSDF voxel to the joint chain | 3.1 cm | 8.0 cm |
| ESDF clearance at the harness points (TCP + joints) | 5.1 cm | 8.7 cm |
| masked points > 5 cm from the URDF surface, per frame | 0 | 197 |
| mask cost per frame (FK + raster) | 22.7 ms | **5.0 ms** |

The 197 far masked points sit on the wrist camera box and an elbow cable (robot parts the URDF
lacks). Margin study (unmasked points within 3 cm, 2 px): 417 at 0 cm, 214 at 0.5, 108 at 1, 37 at
1.5, 0 at 2 cm. Re-validate on any rig without the robot or a bridge:

```bash
uv run python scripts/eval_body_mask.py capture.npz --arm rebot_rs --q 0 0 0 0 0 0 [--gripper RAD] [--tsdf]
```

The capture is a `.npz` with `depth` (N,H,W) in metres, `K` and `T_cam2base` (per frame for a wrist
camera); the arm must not move. Limitations: cables, the mounting plate and anything outside the
URDF and attachments still reach the map; the RS finger angle->width map is unverified; a held
object is not integrated while an RS arm streams; the eye-in-hand path is synthetic-tested only
(the wrist D455F has no hand-eye record yet).

## Always-on occupancy on the reBot

Since a 2026-10-11 rig incident (a side approach pressed the wrist D455F into clutter for ~8 s while
the server ran with `CASCADE_OCCUPANCY=0`), every step of a computed trajectory keeps checking
against the cameras. What runs on `rebot_rs` (and `rebot_rs_reach`, which extends it):

| | when | what |
|---|---|---|
| map refresh | every perception tick (`perception_loop.rate_hz`, 3 Hz), per new frame of each `map_depth` camera, **also while a motion holds `watcher.paused()`** (beliefs pause, the map does not) | depth masked by the URDF mesh silhouette at the arm's current joint state (`body_mask: mesh`), plus the held-payload envelope while holding |
| preflight | before any command: `vet_pose` of goals/pregrasps, `vet_step` over every segment of a route or planned curve | TCP, joint origins, held payload, **tool volume** (wrist camera box + open jaws) against the latest grid |
| every streamed step | `approve()` in the stream loop, 50 Hz (move_joints, move_planned, move_joint_path, move_cartesian, cuMotion profiles, grasp approach/descent, place, spatial tools: all go through SafeArm) | the same samples against the grid **in force at that step**, not the one the route was vetted on |

The streamed check reads the cached grid (0.18 ms per step on the Spark with 408 tool-volume
samples); a required map refreshes outside the lock the 50 Hz reads take
(`OccupancyMap._refresh_plain`). An obstacle appearing mid-motion enters at the next refresh (one
tick plus the ~60 ms bridge round trip) and the next step is refused; the arm holds its last
commanded pose (`tests/test_occupancy_during_motion.py`).

Profile (`configs/arms/rebot_rs.yaml` `overrides.occupancy`):

```yaml
occupancy:
  required: true        # fail closed
  unobserved: free      # the arm's own masked volume is never observed
  max_age_s: 1.5        # ~4 perception ticks; an older grid authorizes nothing
  ready_timeout_s: 3.0  # first motion after a LazyArm wake waits for one masked refresh
```

1. **Fail closed.** No bridge at startup refuses startup; `CASCADE_OCCUPANCY=0` (or `launch.sh
   --occupancy none`) with this profile is a startup error (`required occupancy conflicts with
   CASCADE_OCCUPANCY=off`). A stale grid, failed refresh or body-mask fault refuses the next step
   with the cause (e.g. `required occupancy unavailable: distance grid is stale (1.62 s old,
   max_age_s 1.5)`).
2. **`unobserved` is separate from `required`.** `blocked` (default for a required map) treats
   never-observed voxels as obstacles and so refuses motion from rest; `free` reads them as far
   while observed obstacles, staleness and failures still block.
3. **LazyArm standby:** an unpowered arm has no joint state, so the map latches `robot body pose
   unavailable` by design; the first motion (`LazyArm(on_materialize=...)`) waits up to
   `ready_timeout_s` for one masked refresh and otherwise stays refused with the cause.
4. **Tool volume** (`safety.tool_volume`): wrist camera box from `relative_motion.wrist_camera`
   (`spatial_moves.wrist_camera_dims`), open jaws from `grasp.graspgenx.sweep`, sampled every
   `spacing_m` (2 cm). Voxels within `table_band_m` of the table are the table; the grasp target's
   own voxels inside its exemption cylinder are not obstacles to it; every other voxel blocks. An arm
   already inside the margin may move away or hold, never closer
   (`tests/test_harness_tool_volume.py`). `body_mask.held_payload` masks a held object
   (`tests/test_occupancy_held_payload.py`).
5. **Reported** in `backends.occupancy_state` (MCP `world_state`, live view `/state`) and the
   `occupancy` capability cell: `state` (live/stale/failed/waiting/no bridge), `age_s`,
   `occupied_voxels`, `refreshes`, `required`.

Offline end to end (`tests/test_rebot_occupancy_live.py`): with the capsule mask home is refused
by the arm's own jaws; with the mesh mask rest, home and every step between are admitted, and the
incident's side grasp is refused at preflight. Limitations: only `map_depth` cameras feed the map
(the wrist D455F has `map_depth: false`, so what the scene camera cannot see reads as free); at
3 Hz a new obstacle can cost up to ~0.4 s of travel (the map is not a contact sensor); objects
lower than `table_band_m` (2 cm) are the table for the tool volume; `rebot_rs_mb` keeps the capsule
mask and the optional map.

## The map as a live-view tile (`nvblox_map`)

The live view (`apps/stream_server.py`, `127.0.0.1:8090`) announces one more camera,
`nvblox_map`, rendered by `apps/map_tile.py` with `scripts/record_live_map.py`'s renderer: a 3-D
voxel orbit, a top-down height map, and backend / voxel count / map age / `arm: up` or `arm:
standby` / clock. Default 960 x 540 at 3 fps. `/state` lists it **last**, marked `{"virtual": true,
"kind": "map", "label": "3D map (nvblox)"}` with `map_state`, `voxels` and `map_age_s`;
`/stream/nvblox_map` is MJPEG and `/snapshot/nvblox_map.jpg` one frame (no `/depth/` or
`/annotated/` view; `analyze`, `world_state`, `camera_snapshot` and the capability matrix list only
the camera rig).

1. It draws the runtime's own `OccupancyMap` (last successful query) via
   `OccupancyMap.view_snapshot()`: no lock, nothing sent to the bridge, a non-writeable view; the
   standby flag comes from `LazyArm.connected`, so a viewer never powers the motors.
2. Rendering happens only while a viewer is connected (shared per period; `/state` never renders):
   5.5–12.7 ms per frame on the Spark, 4–6 % of one core at 3 fps, no GPU.
3. A frame never freezes; placeholders carry a clock, and a render failure becomes a red `map view
   failed` frame plus `/state` `error`.

| state         | the tile says                                                      |
|---------------|--------------------------------------------------------------------|
| `off`         | `nvblox off (CASCADE_OCCUPANCY=0)` (or `occupancy.enabled: false`) |
| `bridge_down` | `nvblox bridge not answering` and the error (probe failed, or a refresh timed out) |
| `standby`     | `map starts when the arm powers on` / `(the arm must be masked out)` |
| `waiting`     | `waiting for the first masked depth frame` and why                 |
| `stale`       | the last map, its age (and the last refresh error) in red          |
| `live`        | the map                                                            |

Configuration (`stream.nvblox_map` in `configs/demo.yaml`; `CASCADE_NVBLOX_MAP=0|1|auto` overrides
`enabled`):

| key       | default | meaning |
|-----------|---------|---------|
| `enabled` | `auto`  | `auto`: only when this runtime has an occupancy map; `true`: always (the frame says why there is none); `false`: never |
| `fps`     | `3`     | frames per second while someone watches (0.5-10) |
| `stale_s` | `2.0`   | an older map is drawn with its age in red |
| `quality` | `75`    | JPEG quality |
| `width`, `height` | `960`, `540` | frame size |

On the rig (occupancy on, bridge on `127.0.0.1:5557`, `CASCADE_STREAM=eager`) the view selector
lists `d455f_scene`, `d455f_wrist` and `3D map (nvblox)`. Check
`curl -s 127.0.0.1:8090/state | python3 -m json.tool | grep -A9 nvblox_map`: `map_state` reads
`standby` until the arm is powered, then `live`. `CASCADE_NVBLOX_MAP=1` shows the tile while
occupancy is off.

## Troubleshooting: every motion refused as "not yet validated"

```
SafetyViolation: occupancy unsafe: robot body pose not yet validated by a masked depth refresh
```

With a robot body registered, the map refuses all motion until one refresh succeeds with the robot
masked out. Refresh state changes are logged once (`occupancy refresh failed: ...` / `occupancy
refresh recovered`) and the refusal carries the cause, e.g. `(last refresh failed: occupancy bridge
at 127.0.0.1:5557 timed out (500 ms))`. On a Jetson Thor with a D435i at 1280x720 the numpy `voxel`
fallback took 0.60–0.69 s per refresh (over the default `timeout_ms: 500`), while `warp` on
`cuda:0` took 62–75 ms after two compile-time timeouts. Fix: install the `occupancy` extra
(`warp-lang`) and restart `scripts/serve_occupancy.sh` (its `--backend auto` then selects `warp`);
the demo prints a note when the bridge reports `voxel`.

## The grasp target and its own approach

The pregrasp is vetted before the descent cylinder opens, so the target's own top can block its
approach (a ~10 cm paper cup's map surface sat ~1.5 cm above its rim: `pregrasp unsafe: point 0
clearance 0.010-0.022 m below 0.030 m (occupancy map)`). `grasp.approach_target_exemption` (opt-in
per arm; on for `rebot_rs`) exempts the column above the target: its observed footprint around the
grasp XY (95th percentile + 1 cm, capped by `exempt_radius_m`), from 2 cm under its observed top
upwards, never below the table clearance. The same column applies to the pregrasp check, the
approach-segment check and the executed approach (`SafetyHarness.target_column_exemption`); arm
points outside it, neighbours and the table stay fully checked, and a harness without the method or
a retained release/withdrawal episode keeps the strict check. Occupancy refusals count as
`occupancy_refused` failures with no grasp-height nudge.
