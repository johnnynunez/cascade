# Explicit checker accuracy candidate

Schema5 adds an opt-in, image-only checker localizer for the declared layout-A
board. It uses `findChessboardCornersSB` with `NORMALIZE_IMAGE | ACCURACY`
(flags34). Schema4 retains flags2 and its existing geometry, PNG, contrast,
orientation and metrology gates. The new consumer descriptor binds the checker
API, flags, pattern, grayscale conversion and actual OpenCV package, separately
from its ArUco declaration. A changed/missing declaration is refused before
image inspection. Declaring another SDK's consumer does not claim it ran there.

This remains a candidate. Explicit schema5 producer selection has now run on
a fresh current-SDK episode, and all five saved captures still fail the original
held-out pixel gate. It has no physical model admission and does not correct
the earlier native failure. Neither retained native images nor camera K/T/depth
were inputs to candidate selection or the frozen synthetic evaluation.

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

The initial CPU study passed185 focused software tests; nine existing OpenUSD tests skip
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

## Explicit producer selection, 2026-10-04

The existing `benchmark/rgbd/layout_a_native_bridge.py` accepts
`--checker-accuracy` together with the authenticated
`benchmark/rgbd/assets/checker_accuracy_consumer.json` and its SHA-256 through
`--consumer-detector` / `--consumer-detector-sha256`. The combined declaration
binds both checker and ArUco implementations. Without that explicit flag the
entrypoint remains schema4; either mode rejects the other mode's consumer path
and fixture. The reader must also select
`board_from_fixture(fixture, consumer_json, accuracy=True)`, which checks its
actual implementation before returning `AccuracyBoard`. The ordinary live
annotation route already dispatches that board to the existing image-only
schema5 localizer.

Schema5 authoring delegates only rectangle geometry and stored ST to layout A.
The PNG and rectangle hashes are unchanged; board, consumer, source and composed
scene identities bind the selected checker. The producer records its actual
bitmap-authoring OpenCV separately and does not claim to execute the consumer.
The complete physics/calibration comparator, Ground guards after camera creation
and bootstrap, 18/17 fitting split and 0.15/0.35 px gates remain unchanged.

The complete RGB-D CPU suite passed 286 tests, including OpenUSD authoring,
physics-material preservation, changed ST/shader/metadata/ancestor rejection,
cross-schema refusal and source rebinding. Real check-only executions under
consumer OpenCV 5.0.0 and SDK author OpenCV 4.14.0 produced equal schema5 descriptors
without creating a native run directory. These are CPU checks, not rendering.

A fresh schema4-only replay on main `e5ac96f` reproduced all five historical
corner arrays exactly and retained all five refusals. Analytically projected
controls have held-out errors below 2.1e-13 px; a uniform half-pixel shift does not
remove the observed residual. The saved images identify repeatable corner
inconsistency, but cannot separate native sampling/filtering from localization
bias. Changing camera calibration or fitting corrections to these failed
held-outs is not justified. The schema5 candidate was not evaluated on them.
Local recovery/check-only artifacts are under
`RGBD_LAYOUT_A_ACCURACY_20261004`; the subsequent native result follows.

## Fresh current-SDK episode: geometry remains FAIL

The [compact native receipt](../benchmark/results/rgbd_checker_accuracy_native_20261004.json)
binds source `75451f1d47baea3b380b29b954847e5333d1f57e` and the explicit
`isaac62_48b2d951` recipe for current SDK source `48b2d951`. A new ordinary
RGB-D baseline was measured on that SDK before schema5: model `fbcdb853`.
Its native episode completed 80 solves, 20 policy evaluations and five captures
and closed naturally. The outer harness remains **FAIL** because its post-close
bookkeeping read `port` from `hello` instead of the containing marker. A separate
CPU audit verified all saved captures, identity, clocks, state health, input
hashes, closed port and ordinary scope closure. That audit admits a reference
for exact comparison; it does not relabel the original receipt or establish
locomotion, balanced rest or geometry accuracy.

The schema5 candidate, model `9ce4373c`, also completed all 80/20/5 records. Its
complete native physics/calibration comparison and both Ground appearance
checks passed against the new baseline. Inputs remained unchanged, both native
and scope exits were zero, and closure required no signals or forced cleanup.
Twelve foreign GPU process births were observed before launch and remained
present afterward. No foreign process was controlled.

Two independent failures are retained. The consumer used a Python environment
missing `jsonschema`: all 56 ordinary read attempts failed before any reader RPC,
so there were zero live geometry evaluations or annotations. Separately, CPU
analysis of **all five saved schema5 RGB images**, with the frozen flags34,
18/17 split and unchanged gates, rejected every reference: RMS
**0.158779–0.164282 px > 0.15 px**, maxima **0.254681–0.299271 px < 0.35 px**.
Offline evaluation is not a fresh sensor read. Because the SDK changed, these
numbers are not a causal paired comparison with the historical five failures.

The future external campaign now exercises actual MCP schema validation,
including `jsonschema`, before creating output or launching a producer. The
missing-dependency environment is rejected; the tested `.demo` environment
passes that CPU preflight. Neither check retries or changes the native result.
Further localizer or rendering work needs a separately declared variant,
independent CPU validation and a fresh native episode; the pixel gates remain
unchanged.

## Explicit local RGB refinement, schema6

The opt-in `--checker-saddle` variant adds a fixed local observation after the
unchanged schema5 seed detector. Each corner uses its complete 7×7 RGB window,
Gaussian weights with sigma 1.5 px, and one quadratic stationary-saddle solve.
The estimator receives neither calibration nor depth, projected corners or a
homography. Degenerate patches, incomplete windows and shifts greater than
1.5 px reject the capture; there is no fallback, padding or extrapolation.
The original tag/contrast checks, 18/17 split and 0.15/0.35 px gates remain.

The separate `checker_saddle_consumer.json` binds OpenCV, the NumPy package and
the fixed refinement policy. Producer selection and
`board_from_fixture(..., saddle=True)` require that explicit schema6 declaration.
Geometry, texture rectangles and ST remain unchanged; source and model identity
bind the new module and consumer. Schema4 and schema5 keep their own selection.
In the descriptor, `checker.returned_coordinates` describes the unchanged
OpenCV **seed** coordinates. `checker.post_refinement` describes the subsequent
local solve; schema6's final coordinates are its refined output. The consumer
JSON also separates `seed` and `refinement`. This clarifies the existing contract
without changing its descriptor or hash.

The [new frozen specification](../benchmark/rgbd/assets/checker_saddle_corpus_spec.json)
declares 48 held-out images: four published analytic projections, four new
subpixel phases and three fixed filters. Three previously observed frontal
regressions are separate known cases. Specification and analytic truth were
hashed before rasterization or detector evaluation. Native captures were not
inputs. Every case and refusal is retained in the
[CPU receipt](../benchmark/results/rgbd_checker_saddle_cpu_20261004.json).

| Held-out outcome, all 48 cases | Schema5 | Schema6 |
|---|---:|---:|
| Reference gates pass | 28 | 29 |
| Missing checker detection | 10 | 10 |
| Ambiguous orientation | 2 | 2 |
| Held-out residual rejection | 8 | 0 |
| Fixed local offset rejection | 0 | 7 |

There is one admission gain and no loss. Across the 29 complete paired cases
(1,015 points), truth RMS improves in all 29: pooled RMS is 0.270962 →
0.040732 px. This comparison excludes rejected incomplete estimates explicitly;
it does not establish all-condition performance. The maximum truth RMS among
schema6's 29 accepted cases is 0.072468 px, with maximum point error 0.216079 px.
All nine predeclared adversaries reject.

**All three known regressions worsen**, although both variants pass their
original reference gates. Truth RMS changes 0.030297 → 0.113693 px,
0.033268 → 0.111653 px and 0.048648 → 0.087488 px. Their analytic corners are at
half-pixel phase. The quadratic estimates have component errors of ±0.080393,
±0.078950 and ±0.061864 px; their signs follow which integer window the seed
selects. Complete windows are 176 px from the nearest image boundary, so this
is local approximation bias, not border extrapolation. No parameter, correction
or gate was changed after observing it. Internal homography consistency remains
insufficient to prove absolute accuracy or universal improvement.

The 16×16 versus 32×32 raster control has maximum mean absolute difference
0.006615 RGB8 levels and maximum channel difference 6; no detector or new
threshold is used for that control. All 312 RGB-D software tests pass, including
OpenUSD authoring and cross-schema/consumer rejection. Independent code and
saved-artifact reviews found no blocking issue. These CPU results preceded the
native episode below. The previous schema4 and schema5 native failures keep
their original verdicts; no physical admission is claimed.

Reproduce in the declared consumer environment with `--variant saddle` added
to each `checker_accuracy_campaign` prepare/evaluate/negatives command above.
Raw files are under the external sibling `RGBD_CHECKER_PATCH_20261004/corpus-02`.
The retained `corpus-01` was never evaluated: preparation was repeated before
the first detector call after adding macOS `.dylib` package hashing and source
formatting, with identical specification, truth and all 51 images.

## Static planar native result, 2026-10-04

The new schema6 episode **passes** on frozen source `5062648` and current SDK
`48b2d951` (`isaac62_48b2d951`). Its new model is `2a489f7e…`, epoch
`974ee571…`. A new baseline on that same source produced model `e331fad1…`;
one of the prior baseline's 24 bridge sources had changed, so its identity was
not reused. The first preparation was refused before Kit because a recorded
foreign GPU process had exited. That refusal remains saved; only the resource
inventory and output/cache paths changed for the subsequent baseline.
[Bound native receipt](../benchmark/results/rgbd_checker_saddle_native_20261004.json).

All five retained RGB images pass the unchanged 18/17 reference gates:
held-out RMS is **0.075429–0.093954 px** against 0.15 px, and maximum error is
**0.182673 px** against 0.35 px. This fixed image-only evaluation covers steps
3, 22, 42, 62 and 82; its post-close analysis is not a new sensor observation.

The ordinary in-process MCP reader admitted three fresh captures (steps 3, 42
and 62) and created **51/51 accepted annotations**, 17 per capture. All three
retained-receipt comparisons pass: maximum axis error is **0.152868 mm** against
1 mm, maximum pixel RMS is 0.047245 against 0.35, and maximum pixel error is
0.088824 against 0.7. Recorded capture age stays at or below 0.862644 s against
2 s. Annotation fan-out issues zero additional reader RPCs. All 12 read attempts
are retained: the three successes are followed by **one EOF rejection and eight
connection-refused rejections**, which earn no annotations.

Both reference and candidate complete the original 80 solves, 20 policy
evaluations and five captures. The candidate takes 129.896 s within the original
180 s native / 240 s outer limits. Full native physics/calibration invariance,
state health, post-camera/bootstrap appearance guards and before/after input
hashes pass. Both native/scope exits are zero; ordinary closure leaves no owned
members or births and uses no signals or forced cleanup. The foreign GPU
occupant remains unchanged. The native balance policy is active, with zero
semantic commands; this was not a locomotion/rest test.

CUDA/Warp/XDG cache paths were isolated, and the native log confirms the Warp
path. RTX shader caches retained the SDK's shared defaults. No private RTX
cache, cold-cache timing, timing isolation or reliability claim follows.

This result covers one static simulated plane, the declared board and these
three live captures. It does not establish general 3D calibration, moving-camera
accuracy, per-AOV synchronization, hardware admission or universal improvement.
The three CPU regressions and all historical native/dependency failures remain
failures. Neither native calibration nor analytic ground truth was supplied to
the local refinement.

The [original step-42 JPEG](../benchmark/results/rgbd_checker_saddle_native_step42_20261004.jpg)
is copied byte-for-byte (SHA-256 `d3b364997c6c7ae949390882ab10e87a48c4acc01aeead7699ee731708696329`).
It illustrates the scene; metric evaluation uses the separate retained lossless
RGB bytes. Raw packets, all reader/annotation receipts, source manifests and
ordinary closure remain under `RGBD_CHECKER_PATCH_20261004/native-01`; its
118-file manifest and the baseline audit are linked by the compact receipt.
