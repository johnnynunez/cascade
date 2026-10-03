# Binary orientation reference (CPU candidate)

This is a new, explicit Ground texture variant. The previous colored-marker
fixture and its native geometry failure remain unchanged. No native run is
authorized or represented by these CPU controls.

The checker remains eight by six squares, with the same 35 internal corners.
Four standard ArUco `DICT_4X4_50` markers, IDs 0, 1, 2 and 3, declare its corner
order. Detection uses exactly one occurrence of every ID; missing, duplicate,
unexpected or ambiguous decoded markers reject the image. Bit correction is
disabled. Rejected undecodable quadrilateral candidates are diagnostic, since
the checker squares themselves produce such candidates. No marker is recovered
from the camera model or selected manually after a failed run.

The tags only orient the checker. The existing normalized homography fits the
same parity-selected 18 checker corners and holds out 17. Its pixel residual
gates and the metric comparison gates are unchanged. The oracle accepts only
RGB and the declared board; no K, extrinsic transform, depth or raycast enters
it. IDs are fixture labels, not semantic object observations or motion authority.

The bitmap retains the 4000×4000, two-meter Ground mapping and exact 0.5 mm texel
edges. Every black/white tag cell, quiet zone and checker boundary must be
exactly representable. Tags are oriented in texture coordinates (top row at
positive world Y). Scene placement may use the recorded camera K/T to select
an explicit field of view with margin, but that design calculation is separate
from the independent image-only measurement. Placement also needs a later
native visibility check; fitting inside the camera does not prove no occlusion.

The explicit schema-3 layout uses 24 mm checker squares, 36 mm tags (six
6 mm cells including the black border), a 6 mm white quiet zone, and origin
`(0.09, -0.13, 0)` meters. IDs 0/1/2/3 denote low-X/low-Y,
high-X/low-Y, low-X/high-Y and high-X/high-Y corners respectively. The new
[PNG](../benchmark/rgbd/assets/ground_binary_reference.png) is generated from
exact metric rectangles; the historical colored PNG remains byte-identical.
An authoring calculation using a retained camera predicts a 37.56 px outer
background margin and minimum projected cell edge of 2.64 px. These are design
predictions, not evidence from a new render, and do not establish visibility.

Every admitted RGB image must have at least two pixels per projected tag cell
edge (including border cells) and 24 px margin around each projected quiet
zone. These support checks use decoded tag quadrilaterals only. A checker
intersection must also exhibit four locally sampled quadrants: diagonal
spread at most 32 RGB8 luminance levels and contrast at least 48 levels. The
samples lie 0.2 local grid spacings from the detected corner. This explicit
visibility check is necessary because the actual OpenCV detector inferred
35 corners and passed the fixed homography residuals in a CPU image with one
corner occluded. The new check rejects that image. It is not a calibrated
confidence score or a proof against every possible occluder.
The observation is retained with its actual OpenCV 5.0 output and hashes.
The regression injects those frozen inferred corners to exercise this guard;
the separate real-detector test also accepts an earlier rejection by a newer
SDK. It does not require future OpenCV versions to repeat the inference.

The authoring path keeps the existing Ground geometry, physics material, ST
mapping and shader contract. The CPU checkpoint required a future entrypoint to bind the new bitmap,
descriptor, source and composed scene into a new model identity and repeat the
complete native physics/label comparison. No old frame or model is relabeled.
The historical native entrypoint still selects the colored fixture. This
checkpoint added no binary native entrypoint or model admission; a future
entrypoint must include `binary_reference.py` and its effective detector
descriptor in the recipe/source binding, in addition to the PNG and actual
composed USD. No missing source can be justified by the old model identity.

CPU validation passed 117 tests, with five explicit OpenUSD-import skips in
the ordinary environment. The separate installed OpenUSD CPU interpreter
passed both legacy and binary authoring cases (two tests): one original
Ground geometry prim, preserved physics material and the same ST mapping.
No Kit, solver, GPU or physical renderer was launched. The controls exercise
actual ArUco decoding, rotations and reversed output order; missing,
duplicate, foreign, misplaced and extra IDs; all 16 individual payload-bit
corruptions; border damage, tag/corner occlusion, excessive blur and frame
margin violations. Mild blur and supported rescaling pass the original
metric gates. These sampled controls do not imply that all distortions of
that class are supported.
After that test-only portability adjustment, all 42 binary controls passed
on the final test source; 1619 inputs and protected stores remained unchanged.
Detector, authoring and consumer code did not change after the broader run.

The existing in-process MCP handler also consumes the binary CPU raster via
SensorHub and produces the 17 retained-capture spatial annotations. Invalid
X/Y focal calibration, staleness and extra reader RPCs still reject. This is
labelled synthetic data and ordinary dispatch, not an external MCP host or
native RGB-D acceptance. The compact [CPU receipt](../benchmark/results/rgbd_binary_reference_cpu_20261003.json)
binds source, bitmap, tests, original capture used only for authoring and
unchanged protected stores. Local full artifacts are in the sibling
`RGBD_BINARY_REFERENCE/` evidence root.

The existing environment was inspected, not modified: OpenCV reports 5.0.0;
both installed wheel metadata entries report 5.0.0.93. `cv2.aruco.ArucoDetector`
and `generateImageMarker` are present. Runtime capability checks reject a
missing API instead of providing a different detector. The resulting contract
records the actual OpenCV version, dictionary bytes and detector parameters.

Primary references: OpenCV's [marker detection tutorial](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
describes dictionary-based decoding and ID/corner output. The inspected
[5.0.0 detector API](https://github.com/opencv/opencv/blob/40738fb16ceddb5fb3fea747585f7ce6abb0605b/modules/objdetect/include/opencv2/objdetect/aruco_detector.hpp)
is pinned to release commit `40738fb16ceddb5fb3fea747585f7ce6abb0605b`;
the [license](https://github.com/opencv/opencv/blob/40738fb16ceddb5fb3fea747585f7ce6abb0605b/LICENSE)
is Apache-2.0. This uses OpenCV's ArUco detector, not a claimed execution of
the separate AprilRobotics implementation.


The separate native-preparation candidate adds
`benchmark/rgbd/binary_native_bridge.py`. Its consumer descriptor is a required
canonical JSON input authenticated by SHA-256 and included in the model sources.
This prepared host uses OpenCV 5.0.0 in the consumer and 4.14.0 in the Isaac SDK.
Both real CPU interpreters report the same dictionary bytes. The SDK authors
only the bitmap; it must reproduce the committed PNG exactly and records its
actual authoring version/package code hashes in the composed USD. It does not
claim to have executed the consumer detector. The consumer descriptor also binds
its actual package code inventory. `reference_from_rgb` checks the active
implementation and all parameters itself before detection, without relying on
the campaign caller to enforce that condition. Foreign versions, parameters,
codebooks, implementation hashes and malformed declarations reject.

The native entrypoint injects explicit author/descriptor dependencies into the
existing guarded Ground factory. Camera/export/bootstrap order is unchanged;
complete Ground/ancestor/material/ST checks run after camera creation and after
bootstrap. Model construction rehashes the new helpers, declaration and PNG;
the composed scene binds consumer and generator metadata. The previous colored
entrypoint and bitmap retain their default behavior. New scene/model identities
are required; no previous model admission transfers to this candidate.

CPU preparation passed 152 tests with six explicit missing-OpenUSD skips in the
ordinary interpreter. Three installed OpenUSD CPU checks separately passed,
including actual metadata/physics binding and stronger ancestor material
rejection. The 1624 inputs and protected stores stayed unchanged. External
campaign controls retain failed checks, timeout and implementation drift without
opening Kit. Full local evidence is in `RGBD_BINARY_NATIVE/`; the portable
[preparation receipt](../benchmark/results/rgbd_binary_native_preparation_20261003.json)
binds those inputs and actual interpreter reports.

The prepared campaign retains 80 recorded solves after the original two
bootstrap solves, 20 policy commits, five captures, 180 s inner/240 s outer
bounds and 2 s capture-age/read bounds. It checks the first recorded epoch/step,
model identity, original height/tilt limits, controller faults and contact-channel
completeness. Known-empty contact rows are retained and never called measured
support or balanced rest. The native standing policy still controls balance;
there is no external locomotion/reset client. A future native run must separately
pass producer/closure, passive live reader and the 17-point planar geometry
consumer. This preparation has launched no Kit, renderer or GPU work and makes
no new physical or native geometry claim.
