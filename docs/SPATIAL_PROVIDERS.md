# Spatial providers, grounded memory and route replay

**Next action:** start the MCP server on the synthetic spatial replay (seconds; no simulator):

```bash
CASCADE_ROBOT=spatial_replay CASCADE_RUN_DIR=/tmp/cascade-spatial/run \
CASCADE_BELIEFS_PATH=/tmp/cascade-spatial/beliefs.json \
CASCADE_GRASP_MEMORY_PATH=/tmp/cascade-spatial/grasp.json \
CASCADE_ENVELOPE_PATH=/tmp/cascade-spatial/envelope.json \
uv run python -m cascade.apps.mcp_server
```

Then call the [replay tools](#mcp-replay). The optional `spatial` domain supplies capture-time
transforms, source-bound landmark memory and planar route planning; `spatial_replay` exercises the
real composed runtime and MCP with synthetic observations, has no actuator resources and exposes no
navigation execution tool.

Status:

1. **Works (software):** replay, contracts and MCP dispatch (CPU tests); cuVSLAM in a 12-frame
   native synthetic replay and a [live native simulation stream](#live-native-rgb-d-stream) with
   unknown pose uncertainty; [retained RGB-D surface observations](#retained-rgb-d-surface-observations).
2. **CPU tests only:** the opt-in [`go_to` route runner](#opt-in-observed-route-execution), cuVSLAM
   preparation and base registration.
3. **Not shown:** physical localization, a shipped provider meeting the navigation contract, native
   route execution, an admitted mobile-manipulation stack.

## Optional cuVSLAM localization

The `spatial` domain accepts a `cuvslam` block for an optional RGB-D SLAM estimator (cuVSLAM
17.0.0, [reviewed Python API](https://github.com/nvidia-isaac/cuVSLAM/blob/b405f132b8fb1d861a570f3aea64c2c5d4b59525/python/cuvslam2.cpp);
not installed or imported by default).

1. **Profile:** `kind: spatial`, `robot_id` and `cuvslam` with required `sensor_domain`,
   `sensor_id`, `map_id`, `map_epoch`, `map_frame_id`, `binding_sha256` (the installed
   `pycuvslam*.so`) and `max_gap_s` (0–5 s, exclusive of zero); `max_poses` defaults to 256 (cap
   4096); `timeout_s` bounds each SDK call including warmup (default 5 s, max 60). `map_epoch` is a
   label: each domain reports a unique session epoch, so a restart cannot reuse a map origin.
2. **Tools:** `warmup_localization` initializes an isolated SDK process from an exact retained
   capture's calibration (admits no pose or image); `track_capture` consumes fresh `{epoch,
   sequence, capture_sha256}` captures, which the application must feed within `max_gap_s`
   independently of LLM latency; `get_localization` reads the last still-fresh estimate. None
   acquires a frame or steps physics.
3. **Preparation** (`preparation_intrinsics`, schema `cascade.rgbd-localization-intrinsics.v1`:
   `width`, `height`, nine `intrinsics`, `pixel_center_offset_uv` (`null`, `[0, 0]` or `[0.5,
   0.5]`), `frame_id`, `calibration_id`, `model_identity_sha256` equal to the sensor descriptor)
   exposes `prepare_localization({})` to initialize the SDK before acquisition. The first real
   capture must match exactly and be captured after preparation; a changed declaration needs a new
   domain and map epoch.
4. **Estimator limits:** synchronous odometry/SLAM with an in-memory pose graph; an optical-camera
   pose in a new local map frame with unknown uncertainty and possible loop-closure jumps; ignores
   simulated `world_from_camera`; never supplies obstacle geometry or motion admission. Registered
   pinhole RGB-D needs zero skew; depth is rounded to `uint16` millimetres (overflow, zero-rounded
   and all-invalid frames rejected). Calibration change, tracking loss, gaps, stale results or stop
   invalidate the session; rebuild with a new map epoch.
5. **Process isolation:** tracking runs in one owned process (the SDK's GIL cannot block actuator
   threads); IPC deadlines and stop terminate the worker; close has a bounded kill/reap path and
   reports incomplete cleanup. The extension hash and library version are checked.

Optional `cuvslam.base_registration` binds a **rigid** camera mount to the base: `schema:
cascade.rigid-camera-base.v1`, `robot_id`, `model_identity_sha256`, `camera_calibration_sha256`,
`camera_frame_id`, `base_frame_id`, `translation_m`, `rotation_wxyz` (**T_camera_base**, base into
optical frame, from an independent mount calibration), `position_error_m`, `angular_error_rad`
(nonnegative or explicit `null`) and `sha256` (`cascade.spatial.frames.sha256(record)` over the
record without `sha256`; it binds the declaration, not its accuracy). `track_capture` and
`get_localization` then also return `base_pose` and `base_registration_sha256` from the same capture
and session; errors propagate conservatively (lever arm included, unknown stays unknown); both poses
invalidate together. A moving or joint-mounted camera cannot use it. With cuVSLAM's unknown error
bounds, registration alone cannot enable navigation.

## Opt-in observed route execution

`build_mobile_runtime(..., navigation_source=source, navigation_settings=settings)` (composed:
`navigation_bindings={domain_name: {"source": source, "settings": settings}}`) wraps a mobile
controller with `NavigationDomain`, replacing free-motion tools with `go_to(goal_xy_m, map_epoch,
map_sha256)` while keeping the controller, generation fences and independent mobile verifier.
Defaults and the read-only replay are unchanged.

1. **Provider:** bounded `read(deadline_monotonic_s=...)` returning a typed `NavigationSample`
   (registered map-from-base pose with known error bounds, identities, sensor epoch and sequence,
   transport age, exact `GridSnapshot`, collision-volume hash) and `swept_clearance(query,
   deadline_monotonic_s=...)` returning `VolumeClearance` bound to the query and volume hash.
   Unknown space fails closed; point samples cannot clear a swept cylinder. Native providers must
   release the GIL or isolate blocking work; a stuck provider is quarantined.
2. **Settings** bind base/frame, geometry and calibration hashes, whole-body cylinder (payload and
   every permitted articulation), clearance and stopping margins, localization error, goal
   tolerance, map age, route length, command count and wall deadline; the query inflates for tilt,
   localization error, heading/lateral drift, latency and stop drift, and checks both the current
   body and the corridor continuously. Expired captures, map/volume changes, unknown uncertainty and
   localization jumps stop the route; new observations never renew the wall deadline.
3. **Robot volume:** physical navigation also requires `settings.robot_volume`
   (`spatial.robot_volume.RobotVolume.from_dict`; the cylinder-only fixture is mock-only) covering
   every link, collider and payload with error-bounded boxes and origin ranges; `geometry_sha256`
   must equal its digest and the cylinder must enclose every box over the whole reach range.
4. **Per-sample link poses:** each `NavigationSample.robot_volume` carries a `RobotVolumeSample` at
   the base pose's capture time, sequence, epochs and clock; unknown error, stale captures, omitted
   links, static substitutions or uncertainty beyond a reach bound refuse the route, and a fresh
   base cannot renew an older limb capture.
5. **Execution:** `SafeBase` rechecks route authority around feedback reads and before dispatch
   (revocation delivers a latched stop); arrival needs a full fresh settling window; stop and
   failed-route obligations survive until the task boundary; synthetic runs keep `ok: false` even
   with `software_complete: true`.

These are CPU software contracts, not native navigation admission: no shipped provider supplies
the complete registered-base and whole-volume contract (cuVSLAM's pose alone cannot activate the
runner). Dynamic-map replanning, foothold/terrain planning and physical acceptance remain pending.

## Contracts

1. `SpatialStamp`: map, reset epoch, capture clock and time, source and its SHA-256, calibration and
   measurement kind; clock names never imply conversion to host time (the replay's
   `replay-seconds` t=2 capture never becomes fresh physical data).
2. `FrameTree`: bounded history of rigid transforms (`T_parent_child` maps child points into the
   parent; metres, unit wxyz). Resolves only samples at or before the capture time, refuses dynamic
   edges older than the bound; a changed parent, provider, calibration or reset needs a new tree;
   unknown frames never alias to `base` or `world`.
3. `SpatialMemory`: the original capture point, its map point and every transform used; equal labels
   stay distinct observations (not object tracking). Queries reject foreign epochs/clocks and
   old/future observations; recall returns search hints that need a fresh observation before any
   action. In-memory only.
4. `GridSnapshot`: immutable, hashed, cells unknown/free/occupied. `integrate_scan` consumes a
   horizontal planar range capture with its capture-time transform (continuous-endpoint DDA: a corner
   touch never clears a neighbour; `None` means an observed clear ray to max range; occupied
   dominates). A partial scan keeps its oldest contribution time; no per-cell aging or uncertainty.
5. `plan_route`: bounded cardinal A* with unknown blocked, checking map digest, epoch, clock and
   oldest contribution; inflation covers the circumscribed body/payload radius, clearance, position
   error and cell extent. Results are `execution: not_executed`, `physical_admission: false`: a 2D
   path proves no support, footholds, overhead clearance, balance or tracking.

## MCP replay

| Tool | Request |
| --- | --- |
| `spatial.get_map` | `{}`; returns the map and its digest. |
| `spatial.lookup_transform` | `target`, `source`, `time_s`, `epoch`, `clock_id`, optional `max_age_s`. |
| `spatial.recall` | `label`, `time_s`, `epoch`, `clock_id`, optional `max_age_s`. |
| `spatial.plan_route` | `start_xy_m`, `goal_xy_m`, `footprint_radius_m`, `clearance_m`, `position_error_m`, the context and exact `expected_map_sha256`. |

For the example use `time_s: 2`, `epoch: episode-1`, `clock_id: replay-seconds`. Recall `cup`, or
resolve `spatial_replay/camera` into `world/map`. A sample route: start `[0.45, 0.45]`, goal `[2.65,
1.45]`, radius `0.1`, clearance `0.02`, position error `0.01` (metres); it crosses the measured
opening in the synthetic wall. Stop/reset keeps the map epoch and replays no motion.

## Live native RGB-D stream

The [stream receipt](evidence/spatial/cuvslam-live-stream-20261004.json) records
an Isaac/Newton MicroDuck producer (320 solves, zero policy commands) whose first
12 real RGB-D captures cuVSLAM tracked; `emergency_stop` invalidated the map epoch.
The [12-frame synthetic replay](evidence/spatial/cuvslam-native-20261004.json)
ended with 0.01178 m translation error. The [covariance record](evidence/spatial/cuvslam-covariance-native-20261004.json)
replayed that fixture and carried all twelve 36-value odometry covariance matrices
unchanged through the worker IPC; they are diagnostics, not a calibrated error bound.
To rebuild natively, apply
`evidence/spatial/cunls-thrust-tuple.patch` to the SDK's cuNLS sources, then run
`scripts/validate_cuvslam_rgbd.py --sdk-checkout <dir> --output <json>`.

## Camera-only RGB-D domain (`mobile_rgbd`)

Add `--camera-rgbd` to an owned `scripts/isaac_microduck_bridge.py` run: the
static, world-mounted overview camera then feeds the `mobile_rgbd` sensor
provider. That profile builds no arm, controller or map, and stays
`admission: unvalidated`. A camera on a rig body needs a calibrated mount:
`--camera-rgbd --camera-mount /absolute/mount.json --camera-mount-sha256 <SHA-256>`,
only on the `isaac62_48b2d951` recipe (the bridge refuses a file without its SHA).

## Retained RGB-D surface observations

1. Configure `spatial.rgbd` with `sensor_domain`, `sensor_id`, `map_id`,
   `world_frame_id` and `capacity` (it reads the `mobile_rgbd` provider above).
   The provider's calibration needs an explicit SHA-256, or the domain refuses
   to build; a capture in another world frame is refused.
2. Call `sensing.read_sensor` with `{"sensor_id": "overview"}`.
3. Call `spatial.annotate_pixel` with the returned epoch, sequence and digest.
   The result is an observed surface point, not an object class; it never
   admits motion and adds no grid, route planner or `go_to`.

## Validation and next boundary

CPU regressions cover transform direction, moving-base capture time, stale/foreign clocks, map
resets, calibration changes, distinct equal-label observations, bounded history, subcell ray
traversal, partial-map age, blocked unknown space, swept footprint clearance and real stdio MCP
dispatch ([receipts](evidence/spatial/software-validation.json)); the executor's add real
mock-controller routes, source loss, revocation after ACK, reader quarantine, delayed reset and
composed-runtime obligations. Native navigation still needs a measured localization provider, a
live map source, body/terrain admission and a controller following a bounded geometric goal;
neither the replay nor the arm-local nvblox map supplies them.
