# nvblox and the kitchen cameras

Current integration: PR #27 is merged, including payload/contact and release
recovery, retained anchors, mapper replacement on clear and historical renderer
state. PR #49 adds bounded synchronization of actual jaw opening and newer
camera captures before establishing release geometry. See [current source and acceptance](PROJECT_STATUS_20261001.md) for the
MAIN pin and separate RTX diagnostic/campaign results. The measurements below
retain their original dates and sources; nvblox remains disabled in the normal
Spark presenter profile.

On 1 October, the isolated RTX environment 12 on `7e02de7` passed a native
orange release/recovery diagnostic: a labeled mapper refusal after opening
prevented withdrawal, then one explicit reset resumed the original withdrawal
before returning home and resetting all props. Three newer camera map commits
and all sampled camera ages passed. See the [source-bound receipt](evidence/nvblox-environment-12/release.json).
The subsequent five-object campaign failed on the first orange case before
release: the same observed attachment could not be confirmed. The planned
native reset passed; the remaining four cases were not attempted. Witnesses show real contact loss and a fall during horizontal transport, before
any release command. The mechanical cause remains unresolved. The [campaign receipt](evidence/nvblox-environment-12/campaign.json)
retains the failure, reset and camera measurements. Neither run establishes
five-object or presenter acceptance.

The [retained carry attachment guard](#payload-and-release-contracts) checks the
original post-close attachment in existing Isaac state feedback. Lost or
unavailable evidence stops further transport and preserves a terminal failure;
it does not authorize an automatic open, reset, retry or return home. This
guard does not correct mechanical slip and is not enabled by occupancy-off
presenter runs. [Local RTX validation](LOCAL_RTX_VALIDATION.md) records the
subsequent software checks, administrative closure and separate PC diagnostics.

Environment 13 on `f9cb6b8e` passed passive admission and read-only truth readiness,
then failed its first orange route preflight before any actuator command. No
reset or transport followed. Localization retained a roughly 104-second-old
image while a missing model downloaded; independently fresh camera witnesses
did not make that analyzed image fresh. The subsequent three-second route
budget began after grasp generation. See the
[separate NV result](LOCAL_RTX_VALIDATION.md#separate-nv-result) for the retained
failure, timing limits and independent review.

Environment 14 on `0a27bda2` passed passive admission and native read-only
readiness with local model files already present. Its first orange localization
then exceeded the five-second image lifetime (9.5 seconds of analysis) and
stopped before actuation, without home or reset. Witness cameras remained
fresh; this did not validate the expired analysis image. The environment was
closed administratively with its failure retained. See the
[NV14 result and detector follow-up](LOCAL_RTX_VALIDATION.md#detector-reuse-and-nvblox-environment-14).

Environment 15 on `c9147db8` passed perception and carried the orange to the
box, then failed the unchanged 1 mrad release-feedback guard while waiting
for geometry. Home and reset were not attempted; placement remained
unverified. Camera witnesses passed, and administrative closure preserved
the failure. The [NV15 report](LOCAL_RTX_VALIDATION.md#nvblox-environment-15-release-feedback-movement)
records the measured drift and the separate 352.74-second direct case; it
does not establish presenter deadline or campaign acceptance.

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

## Status (2026-09-30)

**Real nvblox CUDA mapping now runs on both RTX PRO 6000 Blackwell and DGX
Spark GB10. It is still disabled in the validated Spark presenter profile.**
The published PR #23 GraspGen-X acceptance used `occupancy=disabled`.

Both machines compiled nvblox v0.0.10 source commit
`c457c3fc01003bec6eba3ec1c61e6bf84bc3f51f`, using private CUDA **13.2.2**
(nvcc 13.2.86) and **PyTorch 2.14.1+cu132**. Native targets were SM120 and
SM121 respectively. System CUDA, NVIDIA drivers, Isaac Sim and the app's
Python environment were left in place. The release's source identifies its
Python package as `0.0.10.dev1`; receipts include the source SHA.

The source compatibility patch fixes Blackwell architecture parsing, uses
C++20 for the PyTorch wrapper, and resolves the ARM64 cuDNN library supplied
by PyTorch. Upstream Python metadata caps Torch at 2.9.1, so this isolated
build installs the wrapper with `--no-deps` after installing its mapping
dependencies. This is a tested local compatibility build, not an upstream
support claim. The [upstream release](https://github.com/nvidia-isaac/nvblox/releases/tag/v0.0.10)
has no ARM64 wheel; that did not prevent a working Spark source build.

## Reproduce the build

On Linux x86_64 or Spark aarch64, with a C++ compiler, `git`, `uv` and an
NVIDIA driver already installed:

```bash
bash scripts/install_nvblox_source.sh
# Optional: --prefix /path/to/private/nvblox --jobs 4

export LD_LIBRARY_PATH="$PWD/.nvblox/cuda-13.2.2/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
.nvblox/venv/bin/python scripts/check_nvblox.py --output runs/nvblox-check.json
.nvblox/venv/bin/python scripts/serve_occupancy_bridge.py --backend nvblox --port 5557
```

The installer extracts NVIDIA's checksummed CUDA redistributions into its
own prefix, builds the pinned source, then requires actual depth integration,
ESDF queries and map clearing on the GPU. It does not enable the map in the
presenter launcher. Use explicit `--backend nvblox` when evaluating it:
`auto` can select a different implementation if nvblox fails to initialize.

## Hardware validation

- Both GPUs passed 15 upstream query/frame-integration tests.
- The real Cascade ZMQ bridge reconstructed a synthetic 5 cm cube above a
  table. Five known clearances differed by at most **10 mm** with 10 mm voxels.
  After 20 depth frames showing the cube removed, its old position changed
  from 0 to 40 mm clearance. This verifies actual ray carving, not an import
  test or a CPU stub.
- The production adapter needed two fixes: the ESDF query requires **Nx4
  `(x, y, z, radius)`** input despite the Python Nx3 docstring, and flat SDF
  masks cannot index a 3-D grid. The probe now reports `cuda:0` correctly.
- A concurrent startup probe and map refresh could corrupt the shared ZMQ
  REQ socket's send/receive state. Requests are now serialized; a real delayed
  REP peer reproduces and tests that race.

Machine-specific JSON receipts are in [benchmark/results](../benchmark/results/),
with names beginning `nvblox-rtx-` and `nvblox-spark-`. CUDA integration is
measured independently of robot manipulation acceptance.

## Does another camera improve accuracy?

The same archived idle Isaac RGB-D frames were replayed on both GPUs. The
metric samples 245 known surface locations on each of the two cubes, covering
five faces per cube. Ground truth is used only for evaluation. Depth is
robot-masked and subsampled to 640x360; each camera is integrated five times.
Both GPUs produced the same geometric results.

| Voxel | Cameras | Observed samples / 490 | Mean surface residual on observed samples | Samples within one voxel / 490 |
| --- | --- | --- | --- | --- |
| 10 mm | Main | 443 | 3.612 mm | 423 |
| 10 mm | Main + side | 490 | 4.516 mm | 470 |
| 10 mm | Main + side + proof | 490 | 4.421 mm | 474 |
| 5 mm | Main | 383 | 1.971 mm | 360 |
| 5 mm | Main + side | 458 | 2.314 mm | 428 |
| 5 mm | Main + side + proof | 469 | 2.455 mm | 426 |

**The second camera improved coverage substantially. The third camera gave
only a small change on these cubes, and did not consistently reduce residuals.**
A lower residual for one camera does not mean it reconstructed the scene
better: it missed more samples. Likewise, means at 5 and 10 mm use different
observed subsets, so they are not a controlled precision improvement.

Processing cost rises with resolution and camera count. In these runs, the
median warm depth integration for the three-camera sequence was 2.93 ms/frame
on RTX and 18.91 ms/frame on Spark at 10 mm; at 5 mm it was 29.99 and 98.47
ms/frame. The corresponding full workspace grid queries took 4.23 / 6.41 ms
and 19.87 / 31.85 ms. These are component measurements during the recorded
runs, not an end-to-end latency or throughput guarantee.

The input [capture](../benchmark/fixtures/nvblox/kitchen-idle-01/) includes
camera calibration, robot masks and SHA256 receipts. To replay it:

```bash
.nvblox/venv/bin/python benchmark/diagnostics/nvblox_camera_benchmark.py \
  --repo . --capture benchmark/fixtures/nvblox/kitchen-idle-01 \
  --voxel 0.01 --output runs/nvblox-cameras.json
```

To capture another idle scene, use
`benchmark/diagnostics/capture_nvblox_scene.py --port 8611 --output DIR` with
the app interpreter and a bridge providing validated robot pixel masks.

## Integration and limitations

An opt-in `occupancy.required: true` profile requires a reachable bridge at
runtime construction. Missing wire dependencies, `enabled: false` or
`CASCADE_OCCUPANCY=0` then fail startup. During motion, an absent or expired
grid or a failed refresh raises `SafetyViolation`; a later probe alone cannot
clear that failure. A successful depth refresh is needed. Queries outside the
grid or with unknown interpolation support reject arm waypoints and pose
candidates outside the existing local contact exemption. That exemption cannot
override a missing/stale/failed map. The age bound must be finite and positive.

The default remains `required: false`. Payload and body-mask guards retain
their existing stricter contracts in either mode. Requiring an observed map
does not establish complete geometric coverage, calibrated uncertainty or a
successful mapped manipulation campaign; native validation of this opt-in
policy remains pending. It does not enable mapping in the Spark presenter.

In the current camera profiles, `isaac` and `isaac_side` feed geometric
fusion, while `isaac_proof` has `fuse_beliefs: false`. Three viewer streams
therefore do not automatically mean three map inputs. The replay above
explicitly integrates all selected cameras; it does not alter those profiles.

Cascade consumes the distance grid as a clearance gate for TCP and sampled
arm-link positions and attached payload samples. It does **not** refine object
localization or GraspGen-X input geometry. The merged planned-motion path vets
configured candidate routes against the map; this is not an arbitrary obstacle
avoidance planner or evidence that an unseen route is free.
These measurements establish mapping coverage, not more accurate grasping or
placement. No manipulation accuracy improvement has been demonstrated.

The first live Spark manipulation trial exposed an integration issue: the
carried cube remained in the world map and blocked the arm's own carry path.
An experimental follow-up separates payload geometry using measured bilateral
jaw contact and capture-time segmentation. Its interaction with retained map
observations, intentional grasp contact and reset is still under physical
validation. The presenter default remains unchanged until that acceptance
passes.

The ordinary non-payload world-map consumer can report unknown cells as no
known obstacle and an unavailable cache as no data. That policy is not proof of
free space. Payload/recovery modes additionally reject unknown samples and
require fresh, correctly bound integration commits. Known robot-mask failures
fail closed even if the cache has aged out. A mapping
benchmark, especially an idle scene, cannot establish collision safety during
motion or around unseen geometry.

For the upstream model and multi-camera requirements, see the
[nvblox interface](https://nvidia-isaac.github.io/nvblox/v0.0.10/pages/core_library_interface.html)
and NVIDIA's [multi-camera tutorial](https://nvidia-isaac-ros.github.io/concepts/scene_reconstruction/nvblox/tutorials/tutorial_multi_realsense.html).

## Watching the map live

`scripts/record_live_map.py` records what nvblox holds next to what the
cameras see: the cameras' annotated streams on top, the occupied voxels below
(a slow 3-D orbit with the safety workspace, and a top-down height map with
the agent's visible objects). It runs beside the MCP server and only watches:
cameras come from the live-view server (`/stream/<camera>`), and the bridge is
wrapped in `ReadOnlyBridge`, which forwards `probe`/`query` and raises on
`clear`/`integrate_*`. A stopped stream is drawn as missing, never frozen;
the voxel count and map age are on every frame.

```bash
scripts/record_live_map.py --seconds 60 --out runs/videos/live_map.mp4
```

The map shows what the server integrated: with `CASCADE_OCCUPANCY=0` nothing
new arrives, and the video says so (0 voxels or a stale map). On the Spark a
1280 x 720 composite at 10 fps writes in real time, and each map query takes
14-16 ms (92 x 102 x 68 voxels).

## Robot body mask on real arms (`body_mask: mesh`)

Robot self-mask leakage and a limited route search are different faults. The
[bounded lift-first joint-route fallback](CLEARANCE_LIFT_ROUTES.md) can propose
additional fully vetted paths without changing the clearance limit; it neither
removes residual robot points nor makes an invalid map authoritative.

The camera sees the arm, so every depth frame is masked before integration.
Isaac frames carry a render self-mask. Real arms used capsules instead: each
pixel's 3D point was tested against `body_mask_radius_m` (6 cm) around the
joint polyline (`perception/robot_mask.py robot_mask`). A gripper is wide
where that chain is thin. On the real reBot (2026-10-11, calibrated fixed
D455F, arm at rest q = 0), the nvblox map built with the capsule mask kept
voxels on the open jaws (~14 cm wide) and on the wrist D455F above them. The
nearest was 2.8 cm from the chain, under `min_clearance_m` 0.03, so motion
from rest could be refused by the robot's own body.

An arm profile opts in with `body_mask: mesh` (on for `rebot_rs`). Each frame
is then masked by the arm's **silhouette**: `MeshBodyMask` rasterizes the
URDF collision meshes of every link at the current joint state into the depth
image that ships (after `depth_stride`), the same code as fusion's B39 link
self-mask (`perception/link_mask.py`).

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

- **Fingers.** The passive finger joints are posed from the measured gripper
  position through the profile's `gripper.open_pos/closed_pos`. When the
  opening is unknown they are swept over their whole travel, which gives more
  mask and never less. The opening is unknown after a failed gripper read,
  and while an RS arm streams a motion (`body_mask_state()` reads nothing on
  the bus then, for the same 50 Hz timing reason as `body_mask_q`).
- **Margins.** `inflate_m` absorbs metric errors such as the hand-eye
  calibration, the mesh against the printed part, and the finger
  angle->width map. 2 cm is ~10 px at 0.6 m but ~64 px at 0.1 m in a 640x360
  D455F frame, so a pixel dilation alone cannot absorb it at every depth.
  `dilate_px` covers the silhouette's edge. The 2 cm margin stays below
  `min_clearance_m` (3 cm), so the surfaces it hides lie in the band that the
  clearance gate already keeps clear of the arm.
- **Wrist camera and attachments.** The RS URDF does not model the wrist
  D455F. Its box is an envelope sized from the depth points that its body
  left on the 2026-10-11 capture, plus ~1 cm. Those points were measured
  through the markerless hand-eye calibration. Measure the mount with
  calipers and tighten the box.
- **Any camera.** The mask needs only K and T_base_cam: a fixed T for
  eye-to-hand cameras, FK(q) @ X for eye-in-hand. A wrist camera's jaws
  straddle the near plane and are clipped there. Because the mask is a
  silhouette, it does not read the depth values, so a wrong depth on a robot
  pixel (stereo speckle, a close-range mismatch on the jaws) is masked too.
  The capsule test would integrate it.
- **Fail closed.** If there is no joint state (the arm in standby: a LazyArm
  is never woken by looking; a failed read; a non-finite q), the frame is not
  integrated. The fault latches, and clearance is refused with
  `robot body pose unavailable for the <arm> mesh mask: ...` until a masked
  refresh succeeds. Both wire paths mask: nvblox `integrate_masked_depth`
  gets the silhouette as inactive rays, and older bridges get zeroed depth or
  filtered points.
- **No silent capsule fallback.** `LinkGeometry` replaces an unreadable mesh
  with a capsule, which suits fusion's detection gate, but a capsule is the
  mask that left the jaws in the map. If a mesh is missing (git-lfs not
  fetched, for example), a `body_mask: mesh` profile refuses to build and
  names the link. Fetch the meshes (`git lfs pull`) or set
  `body_mask: capsule` knowingly. An attachment on a link that the model
  lacks is refused the same way. `mesh` is also refused on Isaac arms (they
  have the render mask and capture-time joint state) and together with
  payload tracking (its replay re-sends history with render masks only).

Measured on the same real capture (24 frames, 640x360 after stride 2, q = 0,
gripper unknown so the fingers are swept) with
`scripts/eval_body_mask.py ... --tsdf`. The TSDF columns use cascade's warp
TSDF on the CPU, fed each mask's zeroed depth, over the configured workspace
region:

| | capsule (6 cm) | mesh (2 cm, 2 px) |
|---|---:|---:|
| unmasked depth points within 3 cm of the URDF surface, per frame | 3859 | **0** |
| ... within 5 cm, per frame | 4576 | 84 |
| TSDF occupied voxels within 3 cm / 5 cm of the arm | 77 / 132 | **0** / 3 |
| nearest TSDF voxel to the joint chain | 3.1 cm | 8.0 cm |
| ESDF clearance at the harness points (TCP + joints) | 5.1 cm | 8.7 cm |
| masked points > 5 cm from the URDF surface, per frame | 0 | 197 |
| mask cost per frame (FK + raster) | 22.7 ms | **5.0 ms** |

Scored the same way (over its larger region), the original nvblox
capsule-masked map from that day had 407 occupied voxels within 3 cm of the
URDF surface, and its nearest voxel was 2.81 cm from the chain. The 197
masked points per frame that lie more than 5 cm from the URDF surface were
inspected on the last frame. They sit on the wrist camera (masked by its box,
as intended) and on a cable at the elbow: robot parts that the URDF lacks,
not scene surfaces. The margin study on the same capture, as unmasked points
within 3 cm per frame with 2 px dilation, gave 417 at 0 cm, 214 at 0.5 cm,
108 at 1 cm, 37 at 1.5 cm and 0 at 2 cm.

Re-validate on any rig without the robot or a bridge:

```bash
python scripts/eval_body_mask.py capture.npz --arm rebot_rs --q 0 0 0 0 0 0 [--gripper RAD] [--tsdf]
```

The capture is a `.npz` with `depth` (N,H,W) in metres, `K`, and
`T_cam2base` (per frame for a wrist camera). The arm must not move during
the capture.

Limitations: cables, the base's mounting plate and anything else outside the
URDF and the attachments still reach the map. The finger angle->width map
of the RS profile is unverified. While an RS arm streams a motion, the jaw
gap is masked (swept fingers), so a held object is not integrated during the
stream. A wrist camera's T comes from FK at the watcher's read, while the
mask uses the state read a moment later. The eye-in-hand path is
synthetic-tested only, because the wrist D455F has no hand-eye record yet.

## Always-on occupancy on the reBot

Rig incident, 2026-10-11: `grasp_object(green object)` chose a side approach
(object at [0.28, 0.129, 0.043]). The wrist and the padded wrist D455F, which
sits on top of the gripper (8 cm up, 14 cm back from the fingertips, 15 cm
across the jaws), swept into the clutter beside the target and pressed for
~8 s until `did not settle at grasp pose`. The MCP server ran with
`CASCADE_OCCUPANCY=0` because the capsule mask left the jaws in the map, so
the harness had no obstacle model at all. Even with a map it sampled only the
TCP and the joint origins. The requirement since: even with a computed
trajectory, every step keeps checking against the cameras.

What runs now on `rebot_rs` (and `rebot_rs_reach`, which extends it):

| | when | what |
|---|---|---|
| map refresh | every perception tick (`perception_loop.rate_hz`, 3 Hz), per new frame of each `map_depth` camera, **also while a motion holds `watcher.paused()`** (beliefs pause, the map does not) | depth masked by the URDF mesh silhouette at the arm's current joint state (`body_mask: mesh`), plus the held-payload envelope while holding |
| preflight | before any command: `vet_pose` of goals/pregrasps, `vet_step` over every segment of a route or planned curve | TCP, joint origins, held payload, **tool volume** (wrist camera box + open jaws) against the latest grid |
| every streamed step | `approve()` in the stream loop, 50 Hz (move_joints, move_planned, move_joint_path, move_cartesian, cuMotion profiles, grasp approach/descent, place, spatial tools: all go through SafeArm) | the same samples against the grid **in force at that step**, not the one the route was vetted on |

The streamed check reads the cached grid (no bridge round trip); a required
map refreshes outside the lock the 50 Hz reads take
(`OccupancyMap._refresh_plain`). Cost measured on the Spark: 0.18 ms per step
with the 408 tool-volume samples (0.09 ms without). An obstacle that appears
during a motion enters the map at the next refresh (up to one perception tick
plus the ~60 ms bridge round trip) and the next step is refused; the arm holds
its last commanded pose (`tests/test_occupancy_during_motion.py`).

Profile (`configs/arms/rebot_rs.yaml` `overrides.occupancy`):

```yaml
occupancy:
  required: true        # fail closed
  unobserved: free      # the arm's own masked volume is never observed
  max_age_s: 1.5        # ~4 perception ticks; an older grid authorizes nothing
  ready_timeout_s: 3.0  # first motion after a LazyArm wake waits for one masked refresh
```

- **Fail closed.** No bridge at startup refuses startup. `CASCADE_OCCUPANCY=0`
  (or `launch.sh --occupancy none`) with this profile is a startup error
  (`required occupancy conflicts with CASCADE_OCCUPANCY=off`). During a
  motion, a grid older than `max_age_s`, a failed refresh or a body-mask fault
  refuses the next step with the cause, e.g. `required occupancy unavailable:
  distance grid is stale (1.62 s old, max_age_s 1.5)` or `... refresh failed:
  occupancy bridge at 127.0.0.1:5557 timed out (500 ms)`.
- **`unobserved` is separate from `required`.** `blocked` (the default for a
  required map) treats every voxel the camera never observed as an obstacle.
  The camera never observes the arm's own masked volume, which is where the
  arm is, so `blocked` refuses motion from rest. `free` reads unobserved as
  far; observed obstacles, staleness and failures still block.
- **LazyArm standby.** With the arm unpowered there is no joint state to mask
  with, so the map integrates nothing and latches `robot body pose
  unavailable` (by design). When the first motion powers the arm on,
  `LazyArm(on_materialize=...)` waits up to `ready_timeout_s` for one masked
  refresh. It neither refuses for the standby latch nor proceeds unchecked;
  on timeout it prints the cause and the motion stays refused.
- **Tool volume** (`safety.tool_volume`): the wrist camera box comes from
  `relative_motion.wrist_camera` (the box the spatial tools plan with, via
  `spatial_moves.wrist_camera_dims`), the open jaws from
  `grasp.graspgenx.sweep`. Samples every `spacing_m` (2 cm) over faces and
  interior. Obstacles are attributed by where the obstacle is: voxels within
  `table_band_m` of the table plane are the table (the table rule governs
  it), and the grasp target's own voxels inside its exemption cylinder are not
  obstacles to it. Every other voxel, a neighbour of the target included,
  blocks the camera and the jaws. An arm already inside the margin may move
  away or hold, never closer (`tests/test_harness_tool_volume.py`).
- **Held object.** `body_mask.held_payload` (an envelope in the
  `gripper_end` frame) is drawn into the mask while the runtime holds an
  object, so a held object is the robot's payload, not an obstacle at its
  TCP (`tests/test_occupancy_held_payload.py`).
- **Reported.** `backends.occupancy_state` (MCP `world_state`, live view
  `/state`) and the `occupancy` capability cell carry `state`
  (live/stale/failed/waiting/no bridge), `age_s`, `occupied_voxels`,
  `refreshes` and `required`.

End to end offline (`tests/test_rebot_occupancy_live.py`): the reBot's URDF
meshes are rendered into the front camera's depth and go through the mesh
mask, a voxelizing fake bridge and the `_build_arm` harness. With the capsule
mask home is refused by the arm's own jaws; with the mesh mask rest, home and
every step between them are admitted (jaws open and closed), and the
incident's side grasp is refused at the preflight once the blue object beside
the target is in view.

Limitations:

- Only cameras with `map_depth` feed the map. The wrist D455F has no
  hand-eye record yet (`map_depth: false`), so the map is the scene camera's
  view: what it cannot see (behind the arm, under the gripper) reads as free.
- Detection latency: an obstacle that appears mid-motion is caught at the
  next refresh, not instantly. At 3 Hz that is up to ~0.4 s of travel. The
  map is not a contact sensor.
- Objects lower than `table_band_m` (2 cm) are the table for the tool volume
  (the TCP and joints still see them). Cables and anything outside the URDF
  and the attachments still reach the map (see the body mask section).
- `rebot_rs_mb` (motorbridge transport) keeps the capsule mask and the
  optional map.

## The map as a live-view tile (`nvblox_map`)

The demo screen's camera extension can show the map next to the two camera
tiles. The live view (`apps/stream_server.py`, `127.0.0.1:8090`) announces
one more camera, `nvblox_map`, rendered by `apps/map_tile.py` with the same
renderer as `scripts/record_live_map.py` (`apps/live_map_video.map_panels`):

- left: the occupied voxels in a slow 3-D orbit (a turn a minute), coloured
  by height, with the safety workspace in green and
  `N occupied voxels · map 0.4 s old` underneath;
- top right: the top-down height map with the robot base and the agent's
  visible objects;
- bottom right: backend and voxel size, voxel count, map age, `arm: up` /
  `arm: standby`, and a clock.

It is 960 x 540 at 3 fps by default. `/state` lists it **last** (the
extension's default view stays the first real camera), marked
`{"virtual": true, "kind": "map", "label": "3D map (nvblox)"}` with
`map_state`, `voxels` and `map_age_s`. `/stream/nvblox_map` is MJPEG with a
`Content-Length` per part, like the cameras, and `/snapshot/nvblox_map.jpg`
is one frame. There is no `/depth/` or `/annotated/` view of it, and
`analyze`, MCP `world_state`, `camera_snapshot` and the capability matrix
list only the camera rig.

**What it reads, and what it cannot do.** The tile draws the runtime's own
`OccupancyMap`: the cloud of the last successful query, i.e. what the
clearance gate uses, through `OccupancyMap.view_snapshot()`. That read takes
no lock (the perception thread holds the map's locks through a bridge round
trip, and `clearance()` takes them for every 50 Hz waypoint), sends nothing
to the bridge, and hands out a non-writeable view. The standby flag comes
from `LazyArm.connected`, so a viewer never powers the motors. Tests pin all
of this with a bridge that raises on `integrate_*`/`clear` and records every
request (it receives none) while another thread holds both map locks.

**Cost.** Frames are rendered in the HTTP handler of a connected viewer:
no viewer, no rendering, and several viewers share one render per period;
`/state` never renders. Measured on the Spark (median of 20 frames, render
plus JPEG): 5.5 ms wall / 12 ms CPU for 3 k voxels, 12.7 ms / 19 ms for 14 k
voxels, 1.9 ms for a placeholder; at 3 fps that is 4-6 % of one core and
120-200 KiB/s per viewer. No GPU, no bridge time. The renderer is numpy-
vectorized: a thread ticking every 2 ms saw at most 0.5 ms of lateness while
the tile rendered 14 k voxels at 3 fps, against 7.6 ms with the earlier
per-point Python loop (which held the GIL the motion stream needs).

**Honest placeholders.** A frame never freezes; every placeholder carries a
clock:

| state         | the tile says                                                      |
|---------------|--------------------------------------------------------------------|
| `off`         | `nvblox off (CASCADE_OCCUPANCY=0)` (or `occupancy.enabled: false`) |
| `bridge_down` | `nvblox bridge not answering` and the error (probe failed, or a refresh timed out) |
| `standby`     | `map starts when the arm powers on` / `(the arm must be masked out)` |
| `waiting`     | `waiting for the first masked depth frame` and why                 |
| `stale`       | the last map, its age (and the last refresh error) in red          |
| `live`        | the map                                                            |

A map younger than `stale_s` is shown even in standby (a simulated arm is
masked without powering a LazyArm). A render failure becomes a red
`map view failed` frame and `/state` `error`, never a dead stream.

**Configuration** (`stream.nvblox_map` in `configs/demo.yaml`):

| key       | default | meaning |
|-----------|---------|---------|
| `enabled` | `auto`  | `auto`: only when this runtime has an occupancy map; `true`: always (the frame says why there is none); `false`: never |
| `fps`     | `3`     | frames per second while someone watches (0.5-10) |
| `stale_s` | `2.0`   | an older map is drawn with its age in red |
| `quality` | `75`    | JPEG quality |
| `width`, `height` | `960`, `540` | frame size |

`CASCADE_NVBLOX_MAP=0|1|auto` overrides `enabled`. `auto` is the default
because a tile that can only say "off" would sit on every mock and simulator
dashboard; with `CASCADE_OCCUPANCY=0` the live view is exactly the camera rig.

**On the rig.** With occupancy on (`CASCADE_OCCUPANCY` unset or `1`, the
bridge on `127.0.0.1:5557`) and `CASCADE_STREAM=eager`, the extension's view
selector lists `d455f_scene`, `d455f_wrist` and `3D map (nvblox)`. Check
`curl -s 127.0.0.1:8090/state | python3 -m json.tool | grep -A9 nvblox_map`:
`map_state` should read `standby` until the arm is powered, then `live`. To
show the tile while occupancy is still off, set `CASCADE_NVBLOX_MAP=1`.

## Troubleshooting: every motion refused as "not yet validated"

With a robot body registered, the map refuses all motion until one depth
refresh has succeeded with the robot masked out. `refresh()` records its own
failures in `last_error` instead of raising, so a refresh that **never**
succeeds used to leave only the generic refusal:

```
SafetyViolation: occupancy unsafe: robot body pose not yet validated by a masked depth refresh
```

Each change of refresh state is now logged once (`occupancy refresh failed:
...` / `occupancy refresh recovered`), and the refusal carries the cause, for
example `(last refresh failed: occupancy bridge at 127.0.0.1:5557 timed out
(500 ms))`.

Measured on a Jetson Thor with a D435i at 1280x720: the numpy `voxel`
fallback took 0.60–0.69 s per refresh, over the default `timeout_ms: 500`, so
the latch never cleared. The `warp` backend on `cuda:0` took 62–75 ms per
refresh. Its first two refreshes also timed out while its CUDA kernels
compiled; the third succeeded. The demo now prints a note when the bridge
reports the `voxel` backend. Install the `occupancy` extra (`warp-lang`) and
restart `scripts/serve_occupancy.sh`, whose `--backend auto` then selects
`warp`.

## The grasp target and its own approach

The pregrasp is vetted, and approached, before the descent cylinder opens.
With an occupancy map, the target's own top is therefore an obstacle to its
own approach. On the physical reBot (front D435i at ~1.2 m, 1 cm warp voxels),
a ~10 cm paper cup's map surface sat ~1.5 cm above its real rim, so a pregrasp
that cleared the rim by 3.7 cm was refused for every candidate (`pregrasp
unsafe: point 0 clearance 0.010-0.022 m below 0.030 m (occupancy map)`).

`grasp.approach_target_exemption` (opt-in per arm; on for `rebot_rs`) exempts
the column directly
above the target: its observed footprint around the grasp XY (95th percentile
+ 1 cm, capped by `exempt_radius_m`), from 2 cm under its observed top
upwards, never below the table clearance. The same column is applied to the
pregrasp check, the approach-segment check and the executed approach
(`SafetyHarness.target_column_exemption`), so vetting and execution agree. Arm
points outside the column, neighbours and the table stay fully checked; a
harness without the method, or a retained release/withdrawal episode, keeps
the strict check.

Grasp memory no longer turns an occupancy-map refusal into "raise the
grasp": six such refusals had pushed that cup's grasp ~5 cm up, above its rim.
They now count as `occupancy_refused` failures with no height nudge.
