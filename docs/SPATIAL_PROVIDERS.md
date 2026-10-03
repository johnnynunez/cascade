# Spatial providers, grounded memory and route replay

The optional `spatial` domain supplies capture-time transforms, source-bound
landmark memory and planar route planning. The `spatial_replay` robot profile
exercises the real composed runtime and MCP with synthetic observations. It has
no actuator resources and exposes no navigation execution tool.

The separate [retained RGB-D path](RGBD_SPATIAL_OBSERVATIONS.md) now connects an
explicit `SensorHub` to these same transform and memory contracts. It records
selected surface pixels as caller annotations, using capture-bound calibration
and provenance. It does not construct a grid or expose route planning.

The replay implements the first increment proposed in
[the HomeBody comparison](HOMEBODY_COMPARISON.md). It is not a HomeBody port,
physical localization source or admitted mobile-manipulation stack.

## Optional cuVSLAM localization

The `spatial` domain also accepts a `cuvslam` configuration for an optional
RGB-D SLAM estimator. Its [reviewed Python API](https://github.com/nvidia-isaac/cuVSLAM/blob/b405f132b8fb1d861a570f3aea64c2c5d4b59525/python/cuvslam2.cpp)
is cuVSLAM 17.0.0; it is not installed or imported by default. A profile contains
`kind: spatial`, `robot_id` and `cuvslam`, whose required settings are
`sensor_domain`, `sensor_id`, `map_id`, `map_epoch`, `map_frame_id`,
`binding_sha256` (the installed `pycuvslam*.so`) and `max_gap_s` (0–5 seconds,
exclusive of zero); `max_poses` defaults to 256 and is capped at 4096. Optional
`timeout_s` bounds each SDK call, including warmup (default 5 seconds, maximum 60).
The configured `map_epoch` is a label; each domain creates and reports a unique
session epoch, so restarting with the same profile cannot reuse a map origin.
Construction only binds the robot's existing calibrated SensorHub.
`warmup_localization` first uses an exact retained capture's calibration to
initialize an isolated SDK process; it admits no pose or image. After warmup,
`track_capture` consumes fresh `{epoch, sequence, capture_sha256}` captures; an application must feed
captures within the configured gap independently of LLM response time.
`get_localization` reads its last still-fresh estimate. Neither tool acquires a
frame or steps physics. CPU contracts and a 12-frame native synthetic RGB-D
replay pass; [the native validation record](CUVSLAM_NATIVE_VALIDATION.md) binds
the build, inputs, estimates and cleanup. Physical localization remains pending.

The estimator uses synchronous odometry/SLAM with an in-memory pose graph and
reports an estimated optical-camera pose in a new local map frame, with unknown
uncertainty and possible loop-closure jumps. It ignores simulated
`world_from_camera`, never supplies obstacle/free-space geometry, and grants no
motion admission. Registered pinhole RGB-D must have zero skew; explicit pixel
center offsets are preserved. The Python binding requires `uint16` depth, so
meters are rounded to millimeters; overflow, positive depths rounding to zero
and all-invalid frames are rejected. Calibration changes, lost tracking,
capture gaps, stale results or stop invalidate the map session; rebuild with a
new map epoch to resume. Native tracking runs in one owned process so the SDK's
GIL does not block actuator threads. IPC deadlines and stop invalidate results
and terminate that worker; close has a bounded kill/reap path and reports
incomplete cleanup if still busy. The installed
extension hash and library version are checked, while the upstream source pin
documents the reviewed API rather than certifying all linked SDK libraries.

## Opt-in observed route execution

`build_mobile_runtime(..., navigation_source=source, navigation_settings=settings)`
can wrap a configured mobile controller with `NavigationDomain`. The composed
builder accepts the same explicit objects as
`navigation_bindings={domain_name: {"source": source, "settings": settings}}`.
This replaces exposed free-motion tools with `go_to(goal_xy_m, map_epoch,
map_sha256)` and keeps the existing controller, generation fences and independent
mobile verifier. Default profiles and the read-only spatial replay are unchanged.

The borrowed provider must implement bounded `read(deadline_monotonic_s=...)`
and `swept_clearance(query, deadline_monotonic_s=...)`. Reads return a typed
`NavigationSample`: a registered map-from-base pose with known error bounds,
robot/model/body/calibration identities, sensor epoch and capture sequence,
transport age, exact `GridSnapshot`, and immutable collision-volume hash.
Clearance returns `VolumeClearance` bound to the entire query and volume hash;
unknown space fails closed. Point samples or free planar cells cannot establish
clearance of the complete swept cylinder. Native providers must release the GIL
or isolate blocking work in a process; the Python watchdog cannot preempt native
code that holds it. A stuck provider is quarantined instead of spawning more readers.
The provider owner retains responsibility for its lifecycle.

Settings require the exact base/frame, geometry/calibration hashes, whole-body
cylinder radius and vertical extents (including payload and every permitted
articulation), clearance and stopping margins, localization error bounds, goal
tolerance, map age, route length, command count and wall deadline. The query
inflates that envelope for admitted tilt, localization error, controller heading
and lateral drift, observation latency and stop drift. It continuously checks
both the current body and the selected corridor. Expired captures, map/volume
revision changes, unknown uncertainty and localization jumps stop the route;
new observations cannot renew its original wall deadline.

`SafeBase` rechecks the passive route authority around feedback reads and before
dispatch; revocation delivers a latched stop. Final arrival requires a complete
fresh settling window, bounded translation and full-orientation speed and drift.
Stop and failed-route obligations survive until the explicit task boundary.
Synthetic runs retain `ok: false`, even when `software_complete: true`.

These are CPU software contracts, not native navigation admission. No shipped
provider currently supplies the complete registered-base and whole-volume
contract. In particular cuVSLAM's optical pose has unknown uncertainty, and its
output alone cannot activate this runner. Dynamic-map replanning, foothold/terrain
planning and physical end-to-end acceptance remain pending.

## Contracts

`SpatialStamp` identifies a map, reset epoch, capture clock, capture time, source,
source SHA-256, calibration and measurement kind. Clock names never imply a
conversion to host time. The replay uses its recorded `replay-seconds` clock;
its t=2 capture does not become fresh physical data when this file is opened.

`FrameTree` stores a bounded history of rigid transforms. `T_parent_child`
maps child points into the parent frame, using meters and a unit wxyz quaternion.
It resolves only samples at or before the requested capture time and refuses
dynamic edges older than the requested bound. Static extrinsics remain bound to
their calibration and epoch. A changed parent, provider, calibration or reset
requires a new tree. Unknown frames do not alias to `base` or `world`.
Relative transforms depend only on their path through the common ancestor;
a stale world pose cannot invalidate an otherwise valid camera-to-torso static
extrinsic, but it still blocks that camera's map transform.

`SpatialMemory` stores the original capture point and its transformed map point,
plus every transform used. Two observations with the same label remain distinct.
The example sees one cup before and after the base moves, and another cup with
the same label. These are observation identities, not automatic object tracking.
Queries reject another map epoch/clock and omit old/future observations. Recall
returns search hints requiring a fresh observation before an action; it cannot
authorize a grasp from a remembered point. This increment is bounded in-memory
replay, not persistent storage with restart-age recovery.

`GridSnapshot` is immutable and hashes geometry and provenance. Cells explicitly
mean unknown, free or occupied. `integrate_scan` consumes a horizontal planar
range capture and the transform at its capture time. Its DDA uses continuous
endpoints: a corner touch never clears an adjacent cell. `None` specifically
means an observed clear ray up to the maximum range; invalid/missing rays must
be omitted. All occupied returns dominate free rays within that capture.
Endpoints outside the declared map and tilted scan planes are refused.

A partial scan retains old capture sources and their oldest time. It cannot
refresh unrelated free geometry. This whole-map age policy is deliberately
conservative: rebuilding a complete snapshot is necessary to discard old
contributions. There is no per-cell aging or dynamic-object forgetting yet.
Transform provenance is retained, but map/transform uncertainty is not estimated
or aggregated into a probabilistic occupancy model.

`plan_route` runs bounded cardinal A* with unknown cells blocked. It checks the
exact map digest, epoch, clock and oldest contribution age. Inflation accounts
for a declared circumscribed body/payload radius, clearance, position error and
the traversed cell extent. Map boundaries block the complete footprint. The
radius must cover the whole projected robot; an inscribed radius is insufficient.
The result is a path proposal, with `execution: not_executed` and
`physical_admission: false`. A 2D clear path does not establish support, footholds,
stairs, overhead clearance, balance, obstacle dynamics or controller tracking.

## MCP replay

Launch the normal server with task-specific state paths:

```bash
CASCADE_ROBOT=spatial_replay CASCADE_RUN_DIR=/tmp/cascade-spatial/run \
CASCADE_BELIEFS_PATH=/tmp/cascade-spatial/beliefs.json \
CASCADE_GRASP_MEMORY_PATH=/tmp/cascade-spatial/grasp.json \
CASCADE_ENVELOPE_PATH=/tmp/cascade-spatial/envelope.json \
python -m cascade.apps.mcp_server
```

The domain exposes four read/plan tools:

| Tool | Request |
| --- | --- |
| `spatial.get_map` | `{}`; returns the map and its digest. |
| `spatial.lookup_transform` | `target`, `source`, `time_s`, `epoch`, `clock_id`, optional `max_age_s`. |
| `spatial.recall` | `label`, `time_s`, `epoch`, `clock_id`, optional `max_age_s`. |
| `spatial.plan_route` | `start_xy_m`, `goal_xy_m`, `footprint_radius_m`, `clearance_m`, `position_error_m`, the context and exact `expected_map_sha256`. |

For the example use `time_s: 2`, `epoch: episode-1`, `clock_id: replay-seconds`.
Recall `cup`, or resolve `spatial_replay/camera` into `world/map`. A sample route
uses start `[0.45, 0.45]`, goal `[2.65, 1.45]`, radius `0.1`, clearance `0.02`
and position error `0.01`, in meters. It crosses the measured opening in the
synthetic wall. Stop/reset preserves the map epoch and does not replay motion.

## Validation and next boundary

Replay CPU regressions cover transform direction/inversion, moving-base capture time,
stale/foreign clocks and map resets, calibration changes, distinct equal-label
observations, bounded history, subcell ray traversal, partial-map age retention,
blocked unknown space, swept footprint clearance and real stdio MCP dispatch.
Review found and corrected both subcell ray over-clearing and partial-refresh
age loss before this delivery. Receipts are in
[the spatial evidence](evidence/spatial/software-validation.json).

Native navigation still needs a measured localization provider, live map source,
body/terrain admission and a controller that follows a bounded geometric goal.
Neither a replay path nor the existing arm-local nvblox collision map supplies
those missing proofs. The opt-in executor retains the existing actuator owner,
priority stop, cancellation generation and independent effect verifier. Its CPU
regressions additionally cover real mock-controller route execution, source
loss during travel, revocation after command ACK, blocked-reader quarantine,
delayed reset, full-orientation rest and composed-runtime task obligations.
