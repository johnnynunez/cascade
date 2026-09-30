# nvblox and the kitchen cameras

## Current status (2026-09-30)

**nvblox is not active in the Spark presenter setup or the published PR #23
acceptance runs.** The Spark launcher forces `--occupancy none` and exports
`CASCADE_OCCUPANCY=0`. On the tested GB10, the runtime reported
`occupancy=disabled`, `occupancy_live=false`; port 5557 had no service and the
app environment had no `nvblox_torch` package. Successful GraspGen-X inference
does not imply that occupancy mapping is enabled.

Cascade has an optional nvblox adapter, bridge and clearance consumer. Its
installer currently supports nvblox wheels only on Linux x86_64. The upstream
[v0.0.10 release](https://github.com/nvidia-isaac/nvblox/releases/tag/v0.0.10)
provides CUDA 12/13 x86_64 wheels, but no ARM64 wheel. Spark would need a
compatible source build or a separately validated ARM64 distribution, plus
launcher integration. That has not been tested here. A working Warp backend
would be a different implementation, not evidence that nvblox works.

## Two cameras versus three

These are the actual Isaac camera configurations in this checkout:

| Camera | Depth and calibrated pose | Feeds the map if occupancy is enabled? |
| --- | --- | --- |
| `isaac` / `cam0` | Yes | Yes; fusion defaults on with extrinsics |
| `isaac_side` / `side` | Yes | Yes; `fuse_beliefs: true` |
| `isaac_proof` / `proof` | Yes, inherited depth plus its own pose | No; `fuse_beliefs: false` |

`WorldWatcher` feeds depth to occupancy only for a camera whose `fuse` flag is
true. The same flag currently controls geometric and semantic fusion. Thus
three views in the UI do not mean three views in a map; with the current
presenter profile there is no occupancy map at all.

nvblox integrates depth with camera intrinsics and a camera-to-map transform,
then derives a distance field for collision queries. See its
[library interface](https://nvidia-isaac.github.io/nvblox/v0.0.10/pages/core_library_interface.html)
and NVIDIA's [multi-camera example](https://nvidia-isaac-ros.github.io/concepts/scene_reconstruction/nvblox/tutorials/tutorial_multi_realsense.html).
For this kitchen, a side view could fill depth hidden from `cam0` by the arm,
a prop or the box wall. This is an expected benefit, not a measured improvement
from these runs. A third view could help only where it adds useful coverage;
it also adds frame processing and map updates. No two-versus-three-camera
nvblox latency or coverage benchmark has been run.

Before enabling the third camera for mapping, validate its depth, coordinate
transform, capture timestamps and robot mask, then enable fusion explicitly.
Because fusion also changes object beliefs, rerun manipulation and reset
acceptance. Do not reuse these Isaac extrinsics for physical cameras.

## How Cascade would use it

The existing consumer queries obstacle clearance at the TCP and sampled arm
link positions. With a fresh map it can reject poses or motion commands that
violate `min_clearance_m` (3 cm by default), subject to the grasp exemption.
It does not automatically plan a new path around an obstacle. It does not
change Newton's contacts or generate better GraspGen-X candidates itself.

This adapter treats unobserved cells as infinite clearance. Missing or stale
map data disables the measured-clearance check; the other geometric checks
remain. Therefore a map must be validated for coverage, freshness, robot
self-masking and removal of moved objects before relying on it for an event.

During this audit, `NvbloxBackend.query()` failed when applying a flat ESDF
mask to a three-dimensional grid. PR #23 fixes that conversion and preserves
the existing occupied/unknown policy. The new regression reproduces the old
`IndexError` and passes with the fix. It substitutes the tensor/device boundary;
it is **not** a CUDA integration test. The focused occupancy suite passed
41 tests, including map freshness, camera capture order and launcher opt-out.

For the validated presenter version, keep the current occupancy selection.
A separate Spark integration should first prove explicit `nvblox` selection
with no fallback, then compare two and three cameras on the same obstruction
and object-movement scenarios, recording map coverage, stale obstacles,
clearance decisions and frame-to-map latency while the full demo runs.
