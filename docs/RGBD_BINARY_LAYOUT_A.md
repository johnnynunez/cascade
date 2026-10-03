# Ground binary layout A (CPU candidate)

Layout A is a new schema-4 benchmark fixture. It preserves the 35 checker
intersections, 18 fitting corners and 17 held-out corners, while declaring a
different placement, four larger tags and a stricter observed resolution gate.
The historical schema-3 bitmap, descriptor, entrypoint and failed captures remain
unchanged. This candidate has no native render or physical admission.

The board uses 24 mm checker squares on the existing Ground at world Z=0. Its
local-to-world transform has origin `(0.03150801262889796,
0.0705392619922785, 0)` meters and yaw −44 degrees. All four ArUco tags are
84 mm wide, including their black border. IDs 0, 1, 2 and 3 have these centers
in the declared board-local frame:

| ID | Local X (m) | Local Y (m) |
| --- | ---: | ---: |
| 0 | −0.139 | 0.0135 |
| 1 | 0.2805 | 0.011 |
| 2 | −0.116 | 0.1545 |
| 3 | 0.2595 | 0.1525 |

The new [bitmap](../benchmark/rgbd/assets/ground_binary_layout_a.png) covers
two meters in each local axis at 4000×4000 pixels. Every rectangle boundary is
an exact 0.5 mm local texel boundary: tag cells are 28 texels, checker squares
48 texels and the white tag quiet zone one 14 mm cell wide. A changed Ground
ST mapping rotates this local bitmap into the declared world placement. The
descriptor binds the actual float32 `TexCoord2fArray` values; it does not pretend
that USD stores exact real-valued rotation coefficients. CPU reconstruction at
the Ground vertices bounds this storage error below 0.3 micrometers. That is a
storage check, not a rendered measurement or a general calibration guarantee.

The previous native failure showed that a minimum projected *edge* length can
be insufficient under shear: its first tag had more than two pixels per cell
along the edges but only about 1.38 pixels per cell in the least resolved local
direction. Layout A therefore retains the existing edge and quiet-zone tests
and adds a conservative lower bound on the smallest singular value of the
image homography Jacobian over each complete six-by-six-cell tag. Admission
requires that bound to be at least four pixels per cell for all four tags.

The calculation uses only the four observed RGB corners of each decoded tag.
It treats the resulting binary64 homography coefficients as exact rationals,
partitions the tag into 8×8 rectangles, bounds the affine numerator and
denominator extrema on each rectangle, and uses
`|det(J)| / ||J||F <= sigma_min(J)`. Its square-root upper bound uses integer
arithmetic; the final comparison is rational, so rounding cannot admit a bound
just below four. This bound may reject an image whose actual singular value
exceeds four when the conservative certificate is inconclusive. It is a
certificate for the represented corner homography, not for the true optical
system, texture filtering, bit-decoding accuracy or dynamic occlusion.

Authoring used the retained camera model solely to choose the layout. The
offline design certificate predicted a whole-tag lower bound of
4.257975285718269 pixels/cell and a minimum projected quiet-zone margin of
34.851350128262126 pixels. Those predictions are not accepted observations.
The static mesh-bounds check also cannot establish dynamic visibility. The
consumer receives only RGB and the declared board geometry: no K, camera pose,
depth, projected design points or manually selected quadrilaterals enter the
reference fit. The exact original pixel residual and metric gates remain in
force. Four exact IDs, no error correction, a unique orientation and observed
checker contrast remain mandatory.

The explicit [entrypoint](../benchmark/rgbd/layout_a_native_bridge.py) requires
an authenticated consumer-detector JSON input. The producer may author the
same dictionary bits with a different SDK OpenCV version, but must reproduce
the committed PNG and record its actual authoring implementation. It cannot
claim that the consumer ran there. The consumer checks its own full detector
contract before detection. Model source binding includes the new geometry,
resolution helper, entrypoint, consumer declaration and PNG; composed USD also
binds the ST, shader, metadata and source scene. A future native run requires a
new scene/model identity. The old calibration hash may remain the same if the
camera itself is unchanged; an old model admission cannot be reused.

Ground authoring keeps the original collision geometry and physics-material
binding. The existing guarded backend checks full native labels, body/model
properties and BAM configuration without filtering any fields. It also checks
Ground appearance, geometry, metadata, ancestors and inherited material
relationships after camera creation and after bootstrap. CPU doubles verify
these failure paths and camera ordering; actual OpenUSD CPU checks verify the
stored ST/physics binding and rejection of later ST, shader, metadata or strong
ancestor-material changes. These checks do not execute a solver or validate a
native render.

The CPU image controls include all four rotations, reversed detector output
order, reflected or ambiguous orientation, absent/duplicate/foreign IDs,
payload and border corruption, occluded checker corners, blur and resampling,
as well as long-edge/low-singular-value shear. Selected mild-blur and 75% resize
fixtures meet the unchanged gates. A separate exploratory 80% resize was
rejected by the held-out pixel residual gate and is retained as a failure; no
general robustness claim is inferred from the passing samples. The ordinary
environment explicitly skips tests requiring OpenUSD; a separate CPU process
imports the existing OpenUSD installation and runs those checks without Kit.

Related contracts: [binary reference](RGBD_BINARY_REFERENCE.md),
[planar reference](RGBD_PLANAR_REFERENCE.md). The detector is OpenCV ArUco,
using the previously pinned [upstream implementation](https://github.com/opencv/opencv/blob/40738fb16ceddb5fb3fea747585f7ce6abb0605b/modules/objdetect/src/aruco/aruco_detector.cpp);
no separate AprilRobotics implementation or hardware fiducial detection is
claimed.
