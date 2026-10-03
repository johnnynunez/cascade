# Explicit checker accuracy candidate

Schema5 adds an opt-in, image-only checker localizer for the declared layout-A
board. It uses `findChessboardCornersSB` with `NORMALIZE_IMAGE | ACCURACY`
(flags34). Schema4 retains flags2 and its existing geometry, PNG, contrast,
orientation and metrology gates. The new consumer descriptor binds the checker
API, flags, pattern, grayscale conversion and actual OpenCV package, separately
from its ArUco declaration. A changed/missing declaration is refused before
image inspection. Declaring another SDK's consumer does not claim it ran there.

This remains a CPU benchmark candidate. It is not connected to a native producer
by this change, has no new physical model admission, and does not establish that
the earlier native failure is corrected. Neither retained native images nor
camera K/T/depth were inputs to candidate selection or synthetic evaluation.

The [implementation](../benchmark/rgbd/checker_accuracy.py) returns the upstream
coordinates unchanged. OpenCV documents ACCURACY as internal upsampling to
address subpixel aliasing; its pinned implementation doubles the image and
already compensates keypoint scale and pixel-center offset. EXHAUSTIVE increases
search effort instead and was not added. No additional `cornerSubPix`, constant
bias correction or post-filtering of detected coordinates is applied.
[OpenCV5 API](https://github.com/opencv/opencv/blob/40738fb16ceddb5fb3fea747585f7ce6abb0605b/modules/objdetect/include/opencv2/objdetect.hpp#L402-L465),
[flag implementation](https://github.com/opencv/opencv/blob/40738fb16ceddb5fb3fea747585f7ce6abb0605b/modules/objdetect/src/chessboard.cpp#L3839-L3907),
[coordinate conversion](https://github.com/opencv/opencv/blob/40738fb16ceddb5fb3fea747585f7ce6abb0605b/modules/objdetect/src/chessboard.cpp#L595-L665).

## Frozen synthetic evidence

The [corpus specification](../benchmark/rgbd/assets/checker_accuracy_corpus_spec.json)
and [analytic truth](../benchmark/rgbd/assets/checker_accuracy_corpus_truth.json)
were hashed before rasterization or detection. They define48 cases: four
homographies, four subpixel phases and three fixed filter treatments. All use
640×480 RGB and integer OpenCV pixel centers. Their geometry-only checks cover
all tag quads and phases: minimum conservative singular bound4.68157px/cell
and minimum quiet-zone margin39.729px, above the unchanged4px/24px gates.

The initial design proposed an oversized1600×1200 canvas and a reflected
vertical convention. Those design errors and a17.48px margin refusal were
retained, then corrected before any detector outcome. The final specification
has a downward image axis and contracts the third analytic projection. No
homography, flag, split, threshold or case was changed after results.

The analytic raster integrates a fixed16×16 grid per pixel. It accelerates only
uniform interior/exterior footprints; boundary pixels retain every sample.
CPU tests compare this with exhaustive sampling. A separate32×32 integration,
without any localizer, records maximum mean absolute image difference0.006654
RGB8 levels and maximum single-channel difference6. These are descriptive
quadrature diagnostics, not a newly invented acceptance threshold.

Each localizer ran once on every case. All35 corners were retained whenever
available; the fit remains18 points and held-out evaluation17. The table
contains all48 outcomes, with no frame or point selection:

| Outcome | Historical flags2 | Candidate flags34 |
|---|---:|---:|
| Reference gates pass |17|19|
| Checker not detected |16|16|
| Held-out pixel gate rejects |15|13|

Paired outcomes are17 passes in both,29 refusals in both, two gains and zero
losses. All nine fixed adversaries were rejected: reflection, duplicate/foreign/
missing ID, bit corruption, missing checker corner, half-board occlusion,
sigma2 blur and eightfold downscale. Their image hashes and refusal reasons
are retained.

A homography's internal consistency is insufficient to establish absolute
accuracy. Across all32 common detections and all1120 points, RMS relative to
the prefixed analytic truth improves from0.374898 to0.299756px. Per-case truth
RMS improves in30 cases and worsens in two. The worst truth RMS among accepted
references is still0.350258px for the candidate(0.364324px historically).
These frontal subpixel-phase cases expose coherent bias that the held-out
homography check can absorb. No bias was subtracted, no threshold was relaxed,
and the candidate does not support every nominal condition.

## Validation and reproduction

The focused software suite passes185 tests; nine existing OpenUSD tests skip
because `pxr` is absent from the ordinary CPU environment(five layout-A native,
one binary native, one planar and two Ground-texture tests). No SDK or GPU
was launched for this change. The legacy extraction path is compared with a
retained917a98c synthetic result; descriptor mutations, orientation reversal,
image corruption and raster equivalence have dedicated controls.

Run from the repository root in the admitted consumer environment:

```bash
python -m benchmark.rgbd.checker_accuracy_campaign prepare /new/empty/corpus
python -m benchmark.rgbd.checker_accuracy_campaign evaluate /new/empty/corpus
python -m benchmark.rgbd.checker_accuracy_campaign negatives /new/empty/corpus
```

The frozen corpus requires its recorded OpenCV consumer package. A different
package needs a separately declared study; the runner refuses to reinterpret
these inputs. `evaluate` and `negatives` use exclusive result files and do not
overwrite or retry previous evaluations. All failures are records; successful
process exit means the comparison completed, not that all cases passed.

Portable evidence is summarized in
[the CPU receipt](../benchmark/results/rgbd_checker_accuracy_cpu_20261003.json).
Local raw artifacts are outside this checkout under
`RGBD_CHECKER_ACCURACY/{corpus-01,corpus-prepare-01,corpus-evaluation-01,corpus-negatives-01,focused-final-01}`;
`paired-summary.json` and `analyze_saved.py` use saved numerical results only.
The original design and primary-source copies remain under
`RGBD_CHECKER_ACCURACY_RESEARCH`. All1644 snapshotted inputs/protected-store
entries matched before/after each recorded run. This evidence does not close
native XY accuracy, general3D calibration or per-AOV synchronization.
