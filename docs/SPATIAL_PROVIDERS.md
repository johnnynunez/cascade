# Spatial providers, grounded memory and route replay

The optional `spatial` domain supplies capture-time transforms, source-bound
landmark memory and planar route planning. The `spatial_replay` robot profile
exercises the real composed runtime and MCP with synthetic observations. It has
no actuator resources and exposes no navigation execution tool.

This is the first implementation of the replay increment proposed in
[the HomeBody comparison](HOMEBODY_COMPARISON.md). It is not a HomeBody port,
SLAM system, physical localization source or admitted mobile-manipulation stack.

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

CPU regressions cover transform direction/inversion, moving-base capture time,
stale/foreign clocks and map resets, calibration changes, distinct equal-label
observations, bounded history, subcell ray traversal, partial-map age retention,
blocked unknown space, swept footprint clearance and real stdio MCP dispatch.
Review found and corrected both subcell ray over-clearing and partial-refresh
age loss before this delivery. Receipts are in
[the spatial evidence](evidence/spatial/software-validation.json).

Native navigation still needs a measured localization provider, live map source,
body/terrain admission and a controller that follows a bounded geometric goal.
Neither a replay path nor the existing arm-local nvblox collision map supplies
those missing proofs. Any future executor must use the existing actuator owner,
its priority stop, immutable episode generation and independent effect verifier.
