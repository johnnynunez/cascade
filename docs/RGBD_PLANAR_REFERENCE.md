# Independent planar RGB-D reference

`benchmark/rgbd/planar_reference.py` prepares a CPU-validated measurement
reference for the existing RGB-D→SensorHub→spatial annotation path. It is not
installed into the native runner, does not start Kit, and provides no control
tool or physical admission. Native rendering of this reference is still pending.

The visual fixture is an 8×6 checkerboard of 30 mm squares at world origin
`[0.08, 0.01, 0.002]` m. Four colored markers resolve its orientation. The
emissive USD meshes have no collision, rigid-body, mass, drive or articulation
schemas. Their plane is 2 mm above the physical ground and their background is
0.1 mm below the board to avoid coplanar overlap. The measured surface would be
the visual board, not the ground. Its descriptor and exact vertices have separate
SHA256 digests; any native use requires a newly bound composed scene/model.

`reference_from_rgb` detects corners from the captured RGB bytes. A fixed parity
split gives 18 fit corners and 17 held-out corners. Normalized DLT fits the planar
image→world-X/Y homography from declared corner positions, with no camera K,
extrinsics, depth, producer pixel-center offset or raycast input. Colored markers
choose orientation; it is never selected by minimizing the product's errors.
The reference refuses missing/ambiguous markers and excessive held-out error.
Marker uniqueness applies to the entire image: another red/green/blue/yellow
component in the robot or scene can legitimately reject the reference. There is
no post-hoc component selection to obtain a passing fit.
The lower-level `reference_from_corners` is available for recorded detections and
CPU checks; it does not establish that a supplied digest came from a camera.

The held-out detections select integer sample indices by nearest-integer rounding
with ties toward positive infinity. Reference X/Y are evaluated at those exact
integer indices. They are not the unrounded corner's world coordinates, and the
product's `+0.5` convention is not inserted into the oracle. This distinction
prevents the benchmark from concealing the same half-pixel error it tests.

`compare_annotations` joins complete spatial receipts to the original typed
capture, checking RGB bytes, capture SHA, epoch, sequence, source/model/calibration,
clocks, exact selected pixels, annotation content hashes, recorded consumer ages,
and non-admission scope. It neither reads another capture nor renews timestamps.
The age ceiling is 2 s, matching the explicit native02 campaign configuration;
the provider's smaller default was not that campaign's effective setting.
An offline audit does not acquire new fresh-observation credit. Unknown producer
honesty, packet timestamps and dynamic AOV synchronization remain outside this
check.

The frozen diagnostic gates are held-out reference RMS ≤0.15 px and maximum
≤0.35 px; product coordinate error ≤1 mm per axis, with inverse-reference pixel
RMS ≤0.35 px and maximum ≤0.7 px. They are test thresholds, not calibrated
uncertainties. Failure to locate a sufficiently accurate reference must remain
unverified, not lead to an after-the-fact wider tolerance.

CPU tests rasterize the authored world quads without using the product camera
matrix, then feed explicit synthetic RGB-D through the real retained SensorHub
and spatial domain. Corrupt fx/fy, X/Y translation, depth units, axes and the
pixel offset fail the independent reference check. Other controls reject missing
depth, stale/foreign captures, altered calibration/model provenance, content
mutation and duplicate/incomplete receipts. No native render, physical movement,
hardware calibration or general 3D acceptance is implied.

Run the CPU controls with the repository's normal pytest environment:

```sh
python -m pytest tests/test_rgbd_planar_reference.py -q
```

OpenUSD is imported lazily by `author_visual_board`; its authoring test explicitly
skips where `pxr` is unavailable. A separate existing-SDK CPU probe can check
authored geometry and unchanged preexisting stage properties without creating
SimulationApp/Kit. The board's visibility, material rendering, native collision
registry preservation, live processing cost and real rendered reference accuracy
still require a separately reviewed bounded native episode.

Primary algorithm/material references:
[OpenCV homography and corner APIs](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html),
[OpenUSD Preview Surface specification](https://openusd.org/release/spec_usdpreviewsurface.html).
Runtime module versions and binary/source hashes must be recorded separately from
these documentation versions.
# Prepared native entrypoint

`benchmark/rgbd/native_bridge.py` reuses the ordinary MicroDuck bridge admission,
signal owner and bounded runner through its existing `backend_factory` argument.
It requires explicit reference-document and canonical-model SHA256 pins. It adds
the four benchmark sources to the effective-model source manifest; the ordinary
builder rehashes them after opening the backend. Check-only performs asset,
policy, BAM and reference-document admission without opening Kit, sockets or an
output directory.

The subclass authors the visual board immediately before the parent's unchanged
camera setup. Export, play, two bootstrap solves and initial pose assignment stay
in their original order. No extra application updates, solves or motion/reset
clients are added. Existing USD values, metadata, connections, relationships and
time samples must remain equal. List operations are compared by content, and
unlimited USD joint bounds are retained as typed infinity tokens in this
diagnostic snapshot. The new subtree must contain the 53 expected visual meshes
and no physics schema. After opening, all prior native recipe fields, shape/body
labels, masses, solver/BAM properties and calibration must match exactly. An
extra imported visual shape is a rejection, even if it appears harmless.

The actual exported scene and canonical model must acquire new digests. An
unchanged camera calibration digest is legitimate. Old captures are never
relabelled. `benchmark/rgbd/live_reference.py` obtains the exact retained typed
capture after an ordinary MCP sensor read, fits the RGB-only reference and calls
the ordinary `spatial.annotate_pixel` handler for all 17 held-out pixels. It
retains every result and refuses an extra reader RPC. The original two-second
age bound applies to every annotation; fit time does not renew it. Missing,
ambiguous, stale or inaccurate sets remain recorded failures.

The external task-owned campaign prepares an 80-solve episode, camera every 20,
180-second inner and 240-second outer bounds, private cache/stores and a new
nonce-labelled process scope. It requires one complete fresh geometry set for
the combined planar diagnostic and reports producer/consumer/geometry/closure
separately. CPU controls exercise the real in-process MCP handler over explicitly
synthetic raster observations. They establish neither a native render nor a
physical calibration. No native XY episode has been executed at this checkpoint.
