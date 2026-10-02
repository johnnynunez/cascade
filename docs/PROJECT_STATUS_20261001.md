# Project status — updated 2 October 2026

This is the current source and acceptance index. Dated experiment reports remain
historical evidence; their successful cases and failures are not rewritten when
a later source is merged. See [Spark setup](DGX_SPARK_SETUP.md) for installation,
[Spark delivery](SPARK_DELIVERY.md) for native proof and campaign contracts, and
[nvblox](NVBLOX.md) for the separate mapping experiments.

## Source and software validation

Local integration `feat/manipulation-assembly-runtime`, production source
`c07d9225f93ebe7cc1be5d65079ebcc141b60117`, passed **3,797 tests**, with 48
skipped and four deselected in 312.57 seconds. Live OVRTX plus cuMotion completed
one full pick/place/home episode on x86 PhysX GPU. Newton Factory fastening
completed 15.365 turns, measured seating and two seconds of zero-motor retention.
These are source-bound local results, separate from the complete kitchen/Spark
campaign. The [integration report](MANIPULATION_ASSEMBLY_20261002.md) includes
the trial 12 failure, follow-up and remaining limits.
The changes are published in [PR #65](https://github.com/johnnynunez/cascade/pull/65).
Its separate 640×360 follow-up passed orange and green-cube physical placement,
home and reset. Both exceeded the host's 300-second budget (483.72/438.83 s),
so real-host/MCP proof2 and complete campaign acceptance remain incomplete.

Previously recorded merged main was `f7a88233cc80fccc1b02396d963086b1146d2078`, the exact-tree
[merge of documentation PR #63](evidence/isaac-profile10/pr63-merge.json).
All five [PR checks](evidence/isaac-profile10/pr63-ci-passed.json) passed;
all five [merged-source checks](evidence/isaac-profile10/pr63-main-ci-passed.json)
also passed.

Combined candidate `d1d54dcf078b3efefd9542d46947c77bd29afe7e` includes the
release-open stability change, independent RGB/depth encoding (`8935c0e`) and
the reviewed profile 09 documentation. Its
[complete software suite](evidence/isaac-profile10/combined-full.json) passed
[3,757 tests](evidence/isaac-profile10/combined-test-summary.json), with 43
skipped and four deselected in 339.27 seconds. All 1,305 source-bound inputs
remained unchanged. The
[encoding measurements and contract](ISAAC_FRAME_ENCODING.md) separate
100 passive observations on each of `6e1bc81a` and `d1d54dc`, a CPU comparison
on one retained frame and the new implementation's focused checks. The new-source
passive repeat and administrative closure passed. This candidate subsequently
ran [native trial 12](LOCAL_RTX_VALIDATION.md#native-trial-12-and-route-preflight-follow-up):
the green-cube physical audit passed, orange preflight failed, and the full
proof/campaign remained incomplete.

Optional [Isaac bridge Python profiling](ISAAC_BRIDGE_PROFILING.md) at
`3ccdc2e82815cca2a462b49770f58bb6e522789a` passed **3,685 tests**, 43 skipped
and four deselected in 334.79 seconds. All 1,276 source, model and release
files remained unchanged in the [complete local suite](evidence/isaac-profile09/software-full.json).
[PR #62](https://github.com/johnnynunez/cascade/pull/62) retains that software
result separately from the failed physical profile 09 below. Its first macOS
CI run failed because a test replaced the process-wide clock and watcher
threads consumed its fake ticks. Test-only commit `381ad0c4` isolates that
clock within the kitchen installer module; the existing fixture reproduces
the interference deterministically and all 42 module tests pass. All five
[CI checks passed on `381ad0c4`](evidence/isaac-profile09/pr62-ci-passed.json),
including Linux x86_64, Linux aarch64 and macOS. The
[original failure](evidence/isaac-profile09/pr62-ci-first-failed.json) and
[focused test result](evidence/isaac-profile09/pr62-clock-test.json) remain
separate; the complete local suite above belongs to `3ccdc2e8`.
PR #62 merged as `c5b2fd5173641613c4ee27d2f05e5eef8cd09430` with the exact
reviewed tree, as confirmed by the [merge identity](evidence/isaac-profile09/pr62-merge.json).
All five [merged-source CI checks](evidence/isaac-profile09/main62-ci.json)
also passed; their evidence remains separate from the five PR checks.

The separate release-open stability candidate
`6e1bc81a89039eaf585166310af54385a7f5e732` passed **3,741 tests**, 43 skipped
and four deselected in 349.32 seconds. Its
[software result](evidence/isaac-profile09/release-candidate-full.json) and
[test summary](evidence/isaac-profile09/release-candidate-tests.json) bind the
exact tree and 1,277 unchanged inputs. It waits for stable measured joints
after the observed jaw opening within the existing release deadline; geometry
guards and tolerances remain unchanged. This candidate is **unmerged and
physically unvalidated**. Its full-suite result does not belong to source
`3ccdc2e8`, merged main `c5b2fd5`, or the failed physical attempts.

The previous command baseline for physical validation is **`ff8d58be1ae8e000bbecb19cd8db1551ac54be73`**,
from [PR #60](https://github.com/johnnynunez/cascade/pull/60), merged as
`c9147db84356de32d6a98c7a70a1aca3ecefe084` with the exact tested tree.
[Detector model reuse](DETECTOR_MODEL_REUSE.md) keeps two detector modes and up
to eight query vocabularies resident, and prepares localization queries before
selecting their image. Image freshness and physical gates remain unchanged.
Its full suite passed **3,643 tests**, 43 skipped and four deselected in
342.90 seconds; all 1,229 source/release files stayed unchanged. All five PR CI
jobs passed. A local RTX Pro comparison matched all 182 detections across
15 comparisons, with identical labels, confidences, boxes and masks. The first
GPU supervisor's CPU reporting error is retained alongside the successful
offline comparison; no inference was repeated. See the
[validation records](evidence/detector-reuse/retained-inputs.json).
All five [merged-source CI jobs](evidence/detector-reuse/main60-ci.json) also
passed. New physical acceptance is tracked separately.

Documentation [PR #61](https://github.com/johnnynunez/cascade/pull/61) merged as
`5359405a` with the exact reviewed tree after all five PR checks passed.
All five [merged-source CI jobs](evidence/isaac-profile08/main61-ci.json) also passed.
It records detector validation, NV14 and the failed startup of profile 07.
The [CI and merge records](evidence/isaac-profile08/retained-inputs.json) preserve
that documentation revision separately from the runtime pin above.

Documentation [PR #59](https://github.com/johnnynunez/cascade/pull/59) merged as
`70e4f5fe` after all five PR checks passed; all five merged-source checks passed.
It records the failed native attempt 06 and the prior source-bound results.

The earlier reviewed command pin was **`93af4f6e70c9db2319f14e24b9f24fd30c1f074c`**,
from [PR #58](https://github.com/johnnynunez/cascade/pull/58). It checks the same
[observed closing conflict](observed-finger-closing.md#reject-closing-conflicts-before-candidate-route-sampling)
before expensive candidate route sampling, after the initial harness/map
admission. Every accepted candidate still requires all existing checks; ranking,
geometry and deadlines are unchanged. Its focused suite passed 77 tests and
both source reviews passed. Its complete suite passed **3,628 tests**, 43 skipped
and four deselected in 338.11 seconds. All 1,169 source/release files remained
unchanged and were independently rehashed. All five PR CI jobs passed; it merged
as `0a27bda2` with the exact tested tree. All five merged-source CI jobs also
passed. Native acceptance remains incomplete, as recorded below.

It includes candidate **`727aaabc5dec7a72a43e112a7ba076d7c6e5aab3`** from
[PR #57](https://github.com/johnnynunez/cascade/pull/57). That change rejects
[localization results whose image expired during analysis](LOCALIZATION_FRESHNESS.md),
including detector misses, VLM errors and accumulated fallback work. An expired
analysis cannot trigger automatic home or retry. A secondary-view VLM result
retains its actual image and calibration. The focused suite passed 142 tests;
an independent selection passed 70. Its complete suite passed **3,619 tests**,
43 skipped and four deselected in 346.98 seconds. All 1,167 source/release files
remained unchanged and were independently rehashed. All five PR CI jobs passed;
it merged as `6ee6375b` with the exact tested tree. All five merged-source CI jobs
also passed. New physical acceptance remains separate.

This includes [PR #56](https://github.com/johnnynunez/cascade/pull/56), merged
as `ba8f2e83` with the exact tree of tested candidate `33b27382`. Its full suite
passed **3,591 tests**, 43 skipped and four deselected in 338.61 seconds, with
all 1,165 source/release files unchanged. All five PR and all five merged-source
CI jobs passed.
[Isaac startup readiness](ISAAC_STARTUP_READINESS.md) prepares the native
read-only verifier and waits for newer camera captures before proof starts.
It does not extend the native motion deadline or establish physical success.
[Documentation PR #55](https://github.com/johnnynunez/cascade/pull/55) merged as
`261dc1b0`, with all five PR and all five merged-source CI jobs passing.
These results and the failed physical runs remain separate in
[local validation](LOCAL_RTX_VALIDATION.md).

Earlier command pin **`ed29a59eece5af6abd804c03aeb90de47ff88c74`**,
from [PR #54](https://github.com/johnnynunez/cascade/pull/54), merged as `f9cb6b8e`
with an identical tree after all five PR CI jobs passed. It preserves literal
GPU selection in the actual MCP environment. Parent `dc566892` passed the full
**3,556-test suite**, with 43 skipped and four deselected in 349.27 seconds;
all 1,125 source and release files remained unchanged. The follow-up changes
only an apostrophe in a Python comment embedded in a shell heredoc, after
macOS CI rejected the original launch syntax. Its 36 focused tests pass and
all 16 embedded Python ASTs are identical. The original macOS failure and
follow-up checks are retained in [local validation](LOCAL_RTX_VALIDATION.md).
The full-suite count belongs to the parent, not a new local measurement of
the follow-up. Physical acceptance is separate and currently incomplete.

[PR #53](https://github.com/johnnynunez/cascade/pull/53) merged as `41deced8`.
Its tree equals candidate `da0255b`, whose complete suite passed **3,525 tests**,
43 skipped and four deselected in 350.18 seconds. All five PR and all five
merged-source CI jobs passed. The [retained carry contract](NVBLOX_CARRY_ATTACHMENT.md)
rejects further transport after lost or unavailable attachment feedback, without
automatically opening, resetting or retrying. It does not cure mechanical slip.
[PR #52](https://github.com/johnnynunez/cascade/pull/52) previously merged the
source-bound acceptance documentation as `6af34a8`; both CI runs passed all five
jobs. [Retained receipts](evidence/local-rtx-validation-20261001/retained-inputs.json)
bind these reviews, full-suite records and CI results.

Earlier runtime **`27f2b0d7ab76f1c4e3bf246d1b23c636e570496f`** merges
[PR #51](https://github.com/johnnynunez/cascade/pull/51). Its tree
`6ceed3ab9c9e894daf9f427f004c404cd63c8c3c` exactly matches tested candidate
`eb90272`: **3,467 tests passed**, 43 skipped and four deselected, in 340.66 seconds.
All 1,100 tracked files and eight release assets stayed unchanged. All five PR
CI jobs passed; merged-source CI is tracked separately in the
[software receipt](evidence/finger-depth-occlusion-integration/validation.json).
The [endpoint occlusion veto](observed-finger-occlusion.md) rejects finger
endpoints behind observed non-target depth; independently vetted wrist twins
remain available when the original orientation fails. Planning deadlines,
0.1 mm jaw tracking, physical colliders and camera settings are unchanged.
These checks do not certify hidden free space or establish physical acceptance.

Earlier runtime **`7e02de70a24cd5e89a6f774e95189d1225ea2cb5`** merges
[PR #50](https://github.com/johnnynunez/cascade/pull/50), following the release
synchronization fix in PR #49. Its tree `7d52a7fcc778b69a962418c5713cf20fb7baf9a1`
is byte-identical to tested candidate `c5e63d7`: **3,441 tests passed**, 43 skipped
and four deselected, in 346.22 seconds. All 1,092 tracked files and eight release
assets stayed unchanged, and all five PR and all five merged-source CI jobs passed. The
[combined software receipt](evidence/camera-publication-integration/validation.json)
binds the complete suite, two independent source reviews and exact MAIN equality.
Repeated render tokens now leave each camera pending for the next existing
update; publication still requires valid render history and original timestamps.
New physical proof, campaign, release recovery and restart require separate results.

Commit `13ec4830c913f4da3c3c6d07c4a810f95a49f323` adds the optional
[cuMotion planner](CUMOTION.md) in [PR #47](https://github.com/johnnynunez/cascade/pull/47).
Its six new files leave the existing native runtime paths unchanged. The
earlier Spark and nvblox trials described below remain bound to `0e23870`. The focused CPU suite passed
78 tests, including 47 new planner contract tests. Real x86 and Spark aarch64
GPU CLI/factory smokes also passed, with no actuator execution.
All five PR CI jobs and all five merged-source CI jobs passed; their hashes are
recorded in the [planner receipt](../benchmark/results/cumotion-planner-20261001.json).
The 3,365-test local baseline result is not a full-suite measurement of the later
addition. Documentation PR #48 subsequently merged as `00c89f1`, including the
renamed [AGENTS.md](../AGENTS.md) guide and the
[screw manipulation investigation](SCREW_MANIPULATION_RESEARCH.md). Its five PR
and five merged-source CI jobs also passed; see the
[documentation CI receipt](evidence/final-documentation-48/validation.json).

[PR #49](https://github.com/johnnynunez/cascade/pull/49) merged as `8ecf19f`
after five successful CI jobs. Its candidate `c3ea27c` passed **3,436 tests**,
43 skipped and four deselected, in 355.82 seconds. Release now waits within the
existing deadlines for both actual jaws and a genuinely newer, fully open
camera capture before recording release floors. Malformed state, wrong source,
changed epoch and foreign contacts still reject. The [software receipt](evidence/release-opening-integration/validation.json)
does not change environment 11’s failed physical result.

Earlier runtime baseline `0e2387070c2b784ba864981f5c291a1b1e4d117a` merged
[PR #46](https://github.com/johnnynunez/cascade/pull/46) after all five PR CI jobs
passed. All five merged-source CI jobs also passed. Its tree
`de39bba7c65f466e7fce42179ed22c213cd256da` is identical to tested
candidate `4999a754c1c71759ff25dcbda5867bb1b3513fa5`: **3,365 tests passed**,
43 skipped and four deselected, in 342.63 seconds. This run includes the core
suite and the separate Brev visitor/video/portable-bundle Python suites.
All 1,075 tracked files and eight release assets stayed unchanged; the new
physical-run checkout matches every byte. The
[validation receipt](evidence/native-host-budget-integration/validation.json)
binds these checks. The preceding CI run exposed a stale bundle-test expectation
of 300 seconds; the test now expects the implemented 360-second turn budget.
That failed run is retained. The MCP call limit remains 300 seconds, with
60 seconds reserved around it for host routing and response; expiry still stops
motion. Physical proof, recovery and restart require their own receipts.

Earlier baseline `28061a60c3ae8099e8a80ab0d1eb8511adbbab86` merged
[PR #44](https://github.com/johnnynunez/cascade/pull/44) after all five CI jobs
passed; all five jobs also passed on the merged source. Its tree
`e571c787c5a195a1a02288adb6bf313df76a4f38` is identical to the
tested integration `76410fcc01a07aa68307a593d126a558e81e31f0`: **3,295 tests passed**,
43 were skipped and four deselected, in 340.20 seconds. All 1,073 tracked files
and eight pinned kitchen release assets stayed unchanged.
The [integration receipt](evidence/render-geometry-integration/validation.json)
binds the test log, full source manifest, merge and CI. These results cover the
combined geometry and optional renderer changes; they do not establish new
physical acceptance.

[PR #45](https://github.com/johnnynunez/cascade/pull/45) fixes diagnostic
recording of the motion safety callbacks. It merged as `07670e2`, with all five
PR and merged-source CI jobs passing. Its candidate `525ab80` passed
**3,304 tests**, with 43 skipped and four deselected, in 334.95 seconds, plus
81 focused checks. Both recorders now store opaque callback metadata while
forwarding the original arguments unchanged; logging neither calls nor replaces
the safety callbacks. This candidate's software result does not turn the failed
nvblox run below into a recovery pass.

The previous physical-run baseline,
`477c88fe40092fdaad99c0f777a6f1c3b9a41224`, merges the reviewed integration
`391436063c5dd98b0686202db02cefc9bdf294d1` with the same product tree
`ff43ff758ddf39dac8e9661fe083fb5ac7943bcd`. The final suite passed **3,205 tests**,
with 43 skipped and four deselected, in 331.79 seconds. The 1,056 tracked files
were compared with the immutable test-source manifest. These are software
results, not a new physical acceptance or a fresh-install measurement.

The previous integration run retained one failing test: it expected the learned
candidate to remain first in an optional learned/analytic list, although the
legacy selector already sorted that list by quality. The test now checks model
recovery by candidate identity, required-mode isolation and optional-mode quality
order. Production was unchanged by that follow-up. The earlier failure remains
in the validation history.

| Merged change | Current contract and documentation |
| --- | --- |
| [PR #30](https://github.com/johnnynunez/cascade/pull/30) | Read-only [Jev/Kev evaluation](JEV_DECISIONS.md); no active presenter backend or demonstrated advantage over recorded-status rules. |
| [PR #32](https://github.com/johnnynunez/cascade/pull/32) | Optional grasp-attempt evidence retains inputs, model responses and motion phases without changing the normal control path. [Forensic example](SPARK_LAUNCH02_FORENSICS_20261001.md). |
| [PR #33](https://github.com/johnnynunez/cascade/pull/33) | [Strict native confirmation](SPARK_DELIVERY.md#native-confirmation-gate-and-failed-follow-up-1-october-2026) binds saved native cases to placement, home, camera and reset evidence. |
| [PR #34](https://github.com/johnnynunez/cascade/pull/34) | [Isaac physical-time motion](ISAAC_MOTION_CLOCK.md), bounded RPCs and settling; 30 Hz nominal Isaac targets retain the full legacy 50 Hz safety edges. |
| [PR #35](https://github.com/johnnynunez/cascade/pull/35) | Headless viewport updates are disabled. [Render comparison](SPARK_RENDER_COMPARISON_20261001.md) is an idle benchmark, not loaded-motion throughput. |
| [PRs #36–37](https://github.com/johnnynunez/cascade/pull/37) | [Observed finger approach](observed-finger-gate.md), [mask alignment](detector-mask-letterbox.md), full-stroke [closing preflight](observed-finger-closing.md), and Spark defaults. |
| [PR #38](https://github.com/johnnynunez/cascade/pull/38) | [Bounded planning search](grasp-planning-search.md): at most three batches/eight seconds, one saved scene and prior, before motion. |
| [PR #39](https://github.com/johnnynunez/cascade/pull/39) | [Native visitor turn budget](native-visitor-turn-budget.md): visitor/proof alignment, subsequently corrected to include a 60-second host reserve around the 300-second MCP call limit. Cancellation still stops motion. |
| [PR #40](https://github.com/johnnynunez/cascade/pull/40) | [Render-bound frame history](isaac-render-frame-history.md), including exact SDK time encodings observed on ARM. |
| [PR #27](https://github.com/johnnynunez/cascade/pull/27) | Multicamera payload mapping, retained contact/release episodes, scoped recovery, [anchor refresh](NVBLOX_RETAINED_ANCHOR_REFRESH.md), [mapper reset](NVBLOX_MAPPER_RESET.md) and [read-only truth construction](NVBLOX_TRUTH_READONLY.md). |
| [PR #41](https://github.com/johnnynunez/cascade/pull/41) | A deterministic recorder-clock test replaces an assumption about host scheduling overhead; recorder production is unchanged. |
| [PR #42](https://github.com/johnnynunez/cascade/pull/42) | [Preserved grasp-memory ranking](grasp-memory-ranking.md) and [coherent held-object observations](HELD_OBJECT_OBSERVATION.md); a cached aiming estimate cannot authorize slip release. |
| [PR #43](https://github.com/johnnynunez/cascade/pull/43) | Repository documentation reconciled with the installed profiles, software contracts and retained physical failures. |
| [PR #44](https://github.com/johnnynunez/cascade/pull/44) | Optional [OVRTX RGBD rendering](OVRTX_RENDERER.md), the [PhysX envelope](physx-finger-envelope.md) admitted against x86 and ARM exports, and complete portable distribution of their runtime inputs. |
| [PR #45](https://github.com/johnnynunez/cascade/pull/45) | Diagnostic recorders retain opaque callback metadata and forward the actual safety callback unchanged. |
| [PR #46](https://github.com/johnnynunez/cascade/pull/46) | [Native host reserve](native-visitor-turn-budget.md): 360-second visitor/proof pick turns around the unchanged 300-second MCP call limit. |
| [PR #47](https://github.com/johnnynunez/cascade/pull/47) | Optional [cuMotion 1.1.0 planning](CUMOTION.md), static-world candidate export, real x86 and Spark ARM GPU smokes; no integration into native actuator execution. |
| [PR #48](https://github.com/johnnynunez/cascade/pull/48) | Agent guide renamed to [AGENTS.md](../AGENTS.md), delivery documentation reconciled, and [threaded assembly research](SCREW_MANIPULATION_RESEARCH.md) added. |
| [PR #49](https://github.com/johnnynunez/cascade/pull/49) | Wait for both actual jaws and newer release captures within the existing deadlines; partial opening cannot authorize withdrawal. |
| [PR #50](https://github.com/johnnynunez/cascade/pull/50) | Keep incomplete camera publications pending across existing updates; retain duplicate packets and original timestamps until a new bound render is available. |
| [PR #51](https://github.com/johnnynunez/cascade/pull/51) | [Endpoint depth occlusion checks](observed-finger-occlusion.md) and independently vetted symmetric wrist alternatives, with unchanged planning and physical limits. |
| [PR #52](https://github.com/johnnynunez/cascade/pull/52) | Source-bound software, Spark and nvblox acceptance documentation; disconnected Sparks recorded separately from local validation. |
| [PR #53](https://github.com/johnnynunez/cascade/pull/53) | [Retained carry attachment](NVBLOX_CARRY_ATTACHMENT.md): terminal failure on lost or unavailable feedback, without automatic opening or recovery. |
| [PR #54](https://github.com/johnnynunez/cascade/pull/54) | Explicit GPU variables survive MCP registration; existing literal selections and explicit setup overrides are preserved. A comment-only follow-up corrects macOS shell parsing. [Validation](LOCAL_RTX_VALIDATION.md). |

## Merged renderer and geometry integration

The [PhysX finger envelope](physx-finger-envelope.md) retains all eight nominal
components and adds sixteen derived cooking components from each of the x86
and ARM exports: forty components per finger. It changes
observed-surface vetting geometry, not the physical colliders, opening interval,
masks, contact offsets or configured occupancy clearance. Both requests produce
derived cooking representations, not direct dumps of active `PxShape` vertices.
Independent audits admit both complete exports against the final artifact.
The earlier x86-only artifact failed ARM coverage; that negative result remains
in the [geometry review](evidence/physx-finger-envelope/multiexport-review.json).

Geometry candidate `26b68266e5dce52e33706ea14cdd57c36024322b` passed **3,238 tests**,
with 43 skipped and four deselected, in 326.74 seconds. Tracked source and eight
release assets were unchanged. The first run failed because those ignored
kitchen assets were missing; its failure is retained, and the successful rerun
used the same code. The subsequent cross-platform geometry revision `d14b0c1`
passed 74 focused tests and both representation audits. The merged source then
passed the two-object Spark proof recorded below. The joint renderer/geometry
tree passed the separate 3,295-test
run recorded above; it does not inherit its result from the earlier component count.

The optional [OVRTX adapter](OVRTX_RENDERER.md) has real analytic RGBD execution
on Linux x86_64 and Spark aarch64, plus a public static-camera factory capture
on x86. Four packets per analytic run matched the projected object bounds with
zero pixel error and metric Z-depth within 2.50 micrometres. The factory capture
was within 0.5 micrometres. These are synthetic scene measurements, not kitchen,
robot, mapping or manipulation acceptance. Failed startup/calibration attempts
remain in the [renderer receipt](../benchmark/results/ovrtx-renderer-20261001.json).

Neither addition retroactively changes the physical source `477c88f` or the
failures below. Documentation PR #43 preserved that earlier source distinction.

## Current physical acceptance

**Passive observer profile 10 on `6e1bc81a` passed its measurement checks.**
It collected 100 samples with no actuator commands or native task. Image-hash
sections dominated the observed payload cost; code inspection found that an
RGB-only lookup also compressed unused depth. The independent component
candidate reduced median RGB-only encoding from 44.323 to 2.196 ms in a
CPU replay of one retained frame; full packet encoding stayed near 44.1 ms
with identical bytes and capture metadata. These results do not establish a
live task speedup or repair profile 09's failed placement. See the
[profiles 10/11 evidence and limits](ISAAC_FRAME_ENCODING.md).

**Passive profile 11 on combined candidate `d1d54dc` also passed.** Its 100
samples completed in 15.186 seconds, with 50 new and 50 repeated captures and
50 distinct JPEG hashes per camera. Against profile 10, the new-capture payload
median fell from 272.239 to 68.170 ms; repeated captures measured 33.435 and
30.376 ms. Geometry, inventory, camera freshness/identity and physics checks
passed. A 309.668 ms slow sample remains included; its geometry section took
267.636 ms, without an established internal cause. Owned closure passed with
all 1,305 source inputs and 12 protected process identities verified. These
are two passive runs, without native commands or task acceptance.

**Local dual-RTX Pro attempt 09 on `3ccdc2e8` failed during placement.**
The grasp was verified in 164.497 seconds, then the unchanged 300-second MCP
request budget cancelled the motion and latched the stop. The final cube
center was 0.123 m above the support plane, with contacts at both jaws;
release and containment remained unverified. No second case, reset, campaign
or restart followed. All 1,341 samples per camera met the two-second limit
(maximum server/delivery ages 1.155/1.253 s). The observed real-time factor was
0.120 under profiling and shared-machine load, without a controlled speedup
comparison. A 120.460-second technical capture passed; its first offline
matcher failed on the SDK's ` (Python)` name suffix. A strict source-bound
adapter then qualified 118.593 seconds and 968 complete CPU updates. The
largest self elapsed was in `bridge.exec_job.code` (75.398 seconds inclusive,
73.802 seconds self elapsed); its internal cost and possible observer overhead
need separate measurement. These are elapsed durations, not CPU utilization.
Global profile and GPU clock alignment remain unqualified. The
[attempt 09 records](LOCAL_RTX_VALIDATION.md#profiling-attempt-09-python-spans-and-placement-timeout)
retain the failed task and strict check, raw trace, first analysis failure
and successful administrative closure.

**nvblox environment 15 on `c9147db8` failed during post-release retreat.**
Unlike NV14, this detector revision reached actuation, verified the orange
grasp and completed transport. The native case then rejected measured joint
movement while waiting for release geometry. It took 352.74 seconds; home
and reset were not attempted, and the postcondition remained unverified.
All 1,994 witnesses per camera met the two-second age limit (maximum 0.940 s).
This direct diagnostic does not validate the presenter MCP deadline of
300 seconds. Its separate administrative closure passed.
See [NV15](LOCAL_RTX_VALIDATION.md#nvblox-environment-15-release-feedback-movement)
for retained results and the measured drift; no tolerance was relaxed.

**Local dual-RTX Pro attempt 08 on `ff8d58b` failed during placement.**
Read-only startup readiness and all three cameras passed; the grasp was
verified in 171.835 seconds. The unchanged 300-second MCP request limit
then cancelled placement and latched the stop. The cube remained held about
13 cm above the table; proximity to the destination did not establish
containment or release. No reset, second case, campaign or restart followed.
All 1,324 witness samples per camera met the two-second limit (maximum
server age 0.828 s, delivery age 0.855 s). Observed real-time factor was
0.117, under the profiler and existing shared GPU load.
A 120-second Tracy capture decoded real CPU and GPU zones, but included
buffered startup without a precise mapping to the task clock. It cannot
certify a normal-task timing window. [Attempt 08](LOCAL_RTX_VALIDATION.md#profiling-attempt-08-placement-timeout-and-real-trace)
retains the failed task, separate successful closure and profiling limits.

Diagnostic [profiling attempt 07](LOCAL_RTX_VALIDATION.md#profiling-attempt-07-startup-port-mismatch)
on `ff8d58b` stopped during startup because its diagnostic bridge port differed
from the unchanged camera profiles. It produced no manipulation result or
loaded-motion trace; its administrative closure passed.

**nvblox environment 14 on `0a27bda2` failed before actuation.** Models were
present locally and readiness passed, but 9.5-second localization analysis
expired its image. Sampled camera ages passed; no home, retry or reset followed.
Its separate administrative closure passed. [NV14 and detector follow-up](LOCAL_RTX_VALIDATION.md#detector-reuse-and-nvblox-environment-14)
preserve this negative result; the later detector comparison is not physical
acceptance.

**Local dual-RTX Pro attempt 06 on `93af4f6` failed during post-place retreat.**
Grasp completed in 137.410 seconds, but the 300-second MCP deadline cancelled
the operation during retreat; return home was not attempted. The native
postcondition remains unverified. Independent review of 1,391 witness samples
found all three cameras within the two-second limit: maximum server age
1.018476 seconds and delivery age 1.102881 seconds. The final cube was near the
pad with open jaws and no contacts, while the arm had nonzero velocities and
1.339 radians of maximum home error. No reset, second case, campaign or restart
followed. The real-time factor was 0.136; throughput remains unresolved.
The [local report](LOCAL_RTX_VALIDATION.md#local-attempt-06-fresh-cameras-retreat-timeout)
binds the failure, independent review and preserved archive. Prepared attempts
04 and 05 were never launched. They remain tied to their earlier source pins.

**Local dual-RTX Pro attempt 03 on `ed29a59` failed its first native case.**
The normal launcher used a single MCP runtime on the selected GPU. Grasp
completed in 146.856 seconds, but the unchanged 300-second request limit expired
before a completed placement/home result. Witnesses show the green cube on the
destination pad and open jaws, with home incomplete and nonzero final velocities;
they do not establish the exact cancellation stage or a verified postcondition.
No reset, orange case, campaign or acceptance restart followed. The sampled
real-time factor was 0.146. Two early samples per camera exceeded the freshness
limit before the first actuator command, reaching 2.395272 seconds at the server.
Attempt 02's timeout and separate camera failure remain retained.
The [local validation report](LOCAL_RTX_VALIDATION.md)
records the timings, source, retained failure and software corrections.

Separate nvblox environment 13 on `f9cb6b8e` passed passive camera/map admission
and a read-only truth probe, then failed the orange route preflight before any
actuator command. Localization took 104.046 seconds, including a model download
in that caller's working directory, and returned an expired image despite fresh
independent camera witnesses. The later route-budget failure is distinct: its
three-second timer began after grasp generation. There was no reset and no
transport validation of the retained-attachment guard.

The endpoint-occlusion correction in [PR #51](https://github.com/johnnynunez/cascade/pull/51)
passed 3,467 software tests on candidate `eb90272`. Its new Spark run has not
started: Tailscale reported the host offline from 15:24 UTC and SSH timed out
before source transfer. The owner subsequently confirmed the Sparks were
intentionally disconnected and requested local validation on the dual-RTX Pro
PC. The local result above does not certify ARM delivery. The [candidate receipt](evidence/finger-depth-occlusion-integration/validation.json)
records the full suite, equality check and blocked deployment. The separate
environment 12 five-object campaign on `7e02de7` failed its first orange case
before release because the observed attachment could not be confirmed. Its
planned native reset passed; the remaining four objects were not attempted.
The [campaign receipt](evidence/nvblox-environment-12/campaign.json) records
zero of five cases passed, the successful reset and preserved failure. Witness data show bilateral contact loss during horizontal transport, followed
by the orange falling to the table. No jaw command occurred between closure and
the later planned reset. The mechanical cause remains unresolved.

The [carry attachment correction](NVBLOX_CARRY_ATTACHMENT.md) retains the
post-close attachment and rejects subsequent lift/transport targets on observed
loss or unavailable atomic feedback. It preserves a terminal failure instead of
opening, retrying or returning home. This software change is not a successful
repeat of the failed campaign and does not resolve the mechanical cause.

**The latest executed Spark proof, correction-MAIN-07 on `7e02de7`, failed.**
Green completed native placement, return home and reset, with independent
physical review. Orange stopped during descent when the right jaw fell below
the geometrically checked opening interval, before any close or lift. The
same guard refused withdrawal; no campaign or acceptance restart followed.
All three sampled camera server ages stayed within two seconds in both cases.
The [retained proof](evidence/spark-correction-main-07/proof.json) binds both
outcomes and the independently rehashed 186-file archive. Strict confirmation
and READY remained negative. Independent offline analysis found that
the modeled open endpoint intersects the retained pink-cube box behind a visible
depth surface. This identifies a blind spot in point-only vetting; the run did
not measure a pink contact pair or force.

**RTX nvblox environment 12 passed its native release/recovery diagnostic on
`7e02de7`.** Orange completed pickup and placement; the injected mapper refusal
blocked withdrawal after release. One explicit reset then performed the original
retained withdrawal, returned home and reset the scene with three fresh map
commits. All 2,476 observed samples per camera stayed below 0.834 s server age.
The [release receipt](evidence/nvblox-environment-12/release.json) binds the
source, native checks, 27 retained artifacts and independent root review.
Observed capture gaps remain recorded; the five-object campaign is still separate.

Earlier **Spark correction-MAIN-05 passed both native proof cases, strict confirmation
and READY on `0e23870`.** Green cube and orange completed placement, return home
and reset, with independent physical review and unchanged Chromium page identity.
Its five-object campaign stopped after one complete pass: orange placement,
return home and reset passed, but a subsequent camera sample exceeded the
two-second server-age limit. The aggregate campaign remains FAIL and no
acceptance restart followed. RTX nvblox environment 11 reached placement and opening but rejected
release confirmation; that failed run remains unchanged by environment 12’s
new result. The earlier final-MAIN-03 campaign failure remains in the history below.

| Source/run | Measured result and boundary |
| --- | --- |
| MAIN `f9cb6b8e`, nvblox environment 13 | Passive admission and read-only truth readiness passed. Orange route preflight failed before any actuator command; no reset or further cases. Expired localization and later route timeout are separately recorded. [Evidence](LOCAL_RTX_VALIDATION.md#separate-nv-result). |
| Candidate `93af4f6`, local dual-RTX Pro attempt 06 | Cameras passed throughout the sampled case. Grasp verified; MCP cancellation aborted post-place retreat, with HOME unattempted and placement unverified. No reset, campaign or restart. [Evidence](LOCAL_RTX_VALIDATION.md#local-attempt-06-fresh-cameras-retreat-timeout). |
| Candidate `ed29a59`, local dual-RTX Pro attempt 03 | One native MCP session on the selected GPU. Grasp verified; the request expired before completed placement/home verification. Two early stale samples per camera also failed strict checking. No reset, campaign or restart. [Evidence](LOCAL_RTX_VALIDATION.md#local-attempt-03-single-native-session). |
| Candidate `dc566892`, local dual-RTX Pro attempt 02 | Camera/GPU admission passed. Green grasp was verified, but the MCP request timed out during placement and latched e-stop. No reset or further cases followed; native proof and strict confirmation remain FAIL. [Local evidence](LOCAL_RTX_VALIDATION.md#local-attempt-02-native-timeout-during-placement). |
| MAIN `7e02de7`, Spark correction-MAIN-07 | Green placement/home/reset and independent audit passed. Orange aborted before closure when the right jaw reached 49.8942 mm against a 49.9 mm checked lower bound; withdrawal was also refused. No orange target contact or lift was observed. Sampled server camera ages peaked at 1.669 s for green and 1.259 s for orange. Overall proof, strict confirmation and READY remain FAIL; no campaign or restart followed. [Evidence](evidence/spark-correction-main-07/proof.json). |
| MAIN `7e02de7`, nvblox environment 12 | Passive admission and one native release/recovery diagnostic passed. After the labeled post-open mapper refusal, no actuator commands occurred until the explicitly requested original withdrawal. Endpoint error was 0.004094 rad, orange displacement during withdrawal 1.86 nm; home/reset and three newer map commits passed. Sampled camera server/delivery maxima were 0.8332/0.8749 s. [Admission](evidence/nvblox-environment-12/admission.json) and [release](evidence/nvblox-environment-12/release.json). This is direct native skill execution, not visitor-host or five-object acceptance. |
| MAIN `7e02de7`, nvblox environment 12 normal campaign | Orange pickup/lift passed, but placement rejected the missing observed attachment before release. Planned native reset passed; the other four objects were not attempted. Sampled camera server/delivery maxima were 1.0133/1.0433 s. Final non-target displacement was 17.85 mm for lemon and 0.315 mm for green; this measurement alone does not identify the contact mechanism. Overall FAIL, with no restart acceptance. [Campaign](evidence/nvblox-environment-12/campaign.json). |
| MAIN `0e23870`, Spark correction-MAIN-05 proof | Both objects passed native placement/home/reset, strict confirmation, READY and independent physical review. Lift had bilateral contacts in 98/98 green and 190/190 orange samples. Maximum sampled server camera ages were 1.967 s and 1.872 s; client delivery maxima of 2.087 s and 2.077 s are retained separately. Same Chromium page and three advancing 1280 × 720 cameras. [Proof receipt](evidence/spark-correction-main-05/proof.json). This result covers two proof cases, not the separate five-object campaign or restart. |
| MAIN `0e23870`, Spark correction-MAIN-05 campaign | Green passed. Orange placement, home and reset passed, but one subsequent sample on all three cameras aged to 2.07146 s at the server during `world_state`, 31.099 s after the reset order finished. The aggregate audit and campaign remain FAIL: one of two attempted cases passed, three were not attempted, and no acceptance restart followed. [Campaign evidence](evidence/spark-correction-main-05/campaign.json) preserves the physical successes and camera failure separately. |
| MAIN `0e23870`, nvblox environment 11 | Native orange pick, lift, carry and placement reached the acknowledged gripper opening. Release confirmation then rejected incompletely open jaw feedback before the intended mapper-barrier failure, so no recovery or normal five-object campaign followed. No actuator commands occurred after the opening acknowledgement. Witness cameras still carried an intermediate opening until 8 ms after the abort; the exact internal state-versus-camera throw site was not logged. [Retained failure and root review](evidence/nvblox-environment-11/failure.json). |
| MAIN `28061a6`, Spark final-MAIN-03 | Green and orange completed native placement, return home and reset; both independent physical audits and the strict checker passed. Sampled camera server ages remained ≤2 s, and measured neighboring props remained stationary within 0.00013 mm during grasp. The [proof receipt](evidence/spark-final-main-03/proof.json) retains separate delivery ages, capture gaps and earlier helper failures. This is an upgraded existing installation, not a fresh-install measurement. |
| MAIN `28061a6`, Spark final-MAIN-03 campaign | Green cube, orange, pink cube and lemon passed complete placement/home/reset audits. Tomato-can placement and retreat were confirmed, but the 300-second turn deadline cancelled return home. The reset refused e-stop without resetting props. Strict confirmation and the aggregate campaign remain FAIL; no restart followed. The [campaign receipt](evidence/spark-final-main-03/campaign.json) preserves all five verdicts, camera measurements and the independently verified 356-file archive. |
| MAIN `28061a6`, nvblox environment 10 | Passive three-camera admission passed. Initial map warmup took 6.006 s; some later commits aged to 2.642 s, while all three final commits were ≤1.045 s. The first release launcher failed its read-only atomic-truth preflight before any actuator commands. A later native truth probe passed in 88 ms. The second diagnostic completed setup reset, then its recorder rejected a callback as non-JSON data before the first pick motion. No close, injected mapping fault or release episode occurred. Both failures are retained; no recovery or five-object pass is claimed. [Failure receipt](evidence/nvblox-environment-10/failures.json). |
| September `9cf5402` delivery | A fresh-destination GB10 installation, two desktop proofs, a five-object visitor round and restart passed on that source. Download caches were reused. [Historical receipt](../benchmark/results/spark_clean_delivery_20260930.json). Later orange failures remain recorded; this is not the acceptance result for current MAIN. |
| MAIN `6b5dad6`, Spark final-MAIN-01 | Green completed placement, return home and reset under the existing server-camera-age audit. Orange failed with an air grasp and no observed lift. The overall native proof failed; no five-object campaign or restart followed. [Ranking diagnosis](grasp-memory-ranking.md). |
| MAIN `6b5dad6`, nvblox environment 08 | A slip verdict opened the gripper although the last measured sample still showed bilateral contact and a lifted orange. The intended post-release fault was never reached. The exact legacy offset branch was not logged. [Retained failure](HELD_OBJECT_OBSERVATION.md). |
| MAIN `477c88f`, Spark final-MAIN-02 | Green completed placement, return home and reset. Orange stopped during descent when finger feedback left its checked opening interval, before any close command. The overall proof failed; no five-object campaign or restart followed. The [geometry diagnosis](physx-finger-envelope.md) and revised observed-surface envelope are separate from this failed run. Outcome memory is preserved. |
| MAIN `477c88f`, nvblox environment 09 | Passive admission passed: three cameras, no actuator writes; mapping warmup is reported separately. The subsequent release diagnostic stopped during descent, retreated and returned home before any close, fault injection or release episode. Aggregate failure is retained; no five-object campaign followed. Subsequent guarded metadata and derived-geometry queries informed the new envelope; they do not change this result. |

The camera auditor requires each sampled camera's **server snapshot age ≤2 s**
and advancing, correctly bound captures. Client delivery age is also recorded
where available but is not that gate. For example, the successful green case
on `6b5dad6` had one reset delivery at 2.122 s while server age was 1.991 s.
An observed gap between two captures does not prove that no intermediate frame
was produced while a request was blocked. No report here claims continuous
camera availability from sparse samples.

A two-object READY proof, strict native confirmation, five-object campaign,
installation check and same-version restart are distinct results. An overall
failure stays a failure even when its grasp, retreat or placement subchecks pass.

## Profiles, clocks and architectures

Defaults belong to entry points and profiles; do not apply a setting from one
row universally.

| Entry point/profile | Relevant defaults and scope |
| --- | --- |
| Installed Spark desktop/profile | PhysX CUDA; `isaac_kitchen_gpu`; required real GraspGen-X; Qwen Q4 with vision; three cameras; physics `dt=1/120`; camera cadence six bridge iterations; observed-finger gate enabled. Explicit supported environment overrides take precedence. Occupancy is disabled. |
| Isaac arm driver/profile | `motion_rate_hz=30` nominal targets; an explicit call rate overrides the profile. The safety profile retains complete legacy 50 Hz edges and checks actual command edges. Physical pacing and acknowledgements can lower observed target throughput. |
| Optional OVRTX camera | Explicit `ovrtx` extra and camera profile; static USD with `static_scene: true`, or an independently supplied `SceneSnapshot` through the renderer API. Square-pixel centered calibration, metric Z-depth, no simulator steps or default demo switch. |
| Base hardware arm API | Existing 50 Hz default. Isaac timing is not imposed on serial, CAN or ROS2 drivers. |
| Raw `scripts/isaac_bridge.py` | Engine default Newton, timestep `1/60`, camera cadence two bridge iterations unless overridden. The installed Spark launcher deliberately selects different defaults. |
| Isolated RTX nvblox diagnostics | Explicit PhysX/CUDA and mapper configuration, independent ports/ownership, three real camera sources and unchanged clearance/unknown-space rules. These runs do not enable nvblox in the presenter profile. |

Camera cadence counts **bridge iterations**, not physics steps or measured image
Hz. The producer polls at the configured cadence or after 0.5 wall seconds, then
publishes only a render token that resolves to matching recorded physics state.
Repeated renderer output retains the entire previous packet and timestamp.
`rpFabricTime`, captured q/jaws/transforms and producer epoch travel together.
An unmatched or ambiguous history does not get relabeled as fresh.

The RTX workstation is Linux x86_64; DGX Spark GB10 is Linux aarch64. Isaac,
Torch/CUDA and learned-model environments remain separate. x86 simulation or
mapper evidence does not certify ARM deployment, and neither certifies a real
robot. [Newton results](NEWTON_ENGINE.md) are a separate engine/asset matrix.

## Research and separate implementation work

[Jev/Kev decision experiments](JEV_DECISIONS.md) remain offline research. The
captured ten-outcome replay found 10/10 for Kev 4B, Qwen and recorded-status
rules, with no demonstrated decision-quality benefit over the rule baseline.
That comparison has no motion authority and is not integrated into presenter
startup. Official TypeSafe Jev was not evaluated without its API credential.

[NVIDIA ovrtx](OVRTX_RENDERER.md) began as an optional camera and scene renderer,
with standalone execution evidence on both Linux architectures. It was not
included in physical baseline `477c88f`. The local integration now adds an
automatic Isaac physics-snapshot producer and robot/prop masks, with one full
OVRTX/cuMotion manipulation episode validated on x86 PhysX GPU. This does not
establish the same runtime result on Spark or Newton. Cosmos,
mobile-base navigation and new hardware validation retain separate roadmap
scopes. Locally installed development skills are authoring tools, not runtime dependencies or substitutes for source-bound physical tests.

[Screw manipulation research](SCREW_MANIPULATION_RESEARCH.md) identifies five
SimReady USD catalogue assets and the Factory/Isaac Lab/Newton threading paths.
The ordinary `turn_screw` still commands wrist strokes and reports physical
verification as unknown. The separate [Factory implementation](FACTORY_THREAD_CONTACT.md)
now validates contact-driven advancement, actual seating and zero-spindle-motor
retention; its arm servos and mounted socket remain engaged. SimReady's listed
CC BY-NC terms and the original Factory mesh BSD-3-Clause licence are distinct.
The original research installed no simulation; the later implementation and
its pinned assets are recorded separately.

## Operator evidence sequence

Use the normal [desktop launch](DGX_SPARK_SETUP.md#4-start-and-prove-the-demo)
from the intended installed checkout. A successful installation check means
PREPARED; a skip-proof launch means STARTED / UNVERIFIED. After a complete native
proof, run the saved-evidence [strict checker](SPARK_DELIVERY.md#native-confirmation-gate-and-failed-follow-up-1-october-2026).
Run a five-object native campaign in that same verified session, fail fast on a
failed case, and preserve its receipts before considering restart acceptance.
Never delete outcome memory to make a new test look like a clean success.
