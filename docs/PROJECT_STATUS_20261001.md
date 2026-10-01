# Project status — 1 October 2026

This is the current source and acceptance index. Dated experiment reports remain
historical evidence; their successful cases and failures are not rewritten when
a later source is merged. See [Spark setup](DGX_SPARK_SETUP.md) for installation,
[Spark delivery](SPARK_DELIVERY.md) for native proof and campaign contracts, and
[nvblox](NVBLOX.md) for the separate mapping experiments.

## Source and software validation

Runtime baseline `0e2387070c2b784ba864981f5c291a1b1e4d117a` merged
[PR #46](https://github.com/johnnynunez/cascade/pull/46) after all five PR CI jobs
passed. Its tree `de39bba7c65f466e7fce42179ed22c213cd256da` is identical to tested
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

**Spark final-MAIN-03 passed the two-object native proof, strict confirmation
and READY on `28061a6`. Its subsequent five-object campaign failed, with four
complete cases passing.** The tomato can reached confirmed placement, but the
whole-turn deadline cancelled its return home; reset then refused the latched
stop. The existing Chromium page retained its identity and three advancing
1280 × 720 cameras. Normal restart and separate RTX nvblox recovery remain
unaccepted stages. Preparation, software tests and passive observations do not
replace them.

| Source/run | Measured result and boundary |
| --- | --- |
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

[NVIDIA ovrtx](OVRTX_RENDERER.md) is now an optional camera and scene renderer,
with execution evidence on both Linux architectures as described above. It was
not included in physical baseline `477c88f`. No automatic Isaac/Newton snapshot
producer, robot/target masks or manipulation authority is supplied; those
connections require their own implementation and acceptance. Cosmos,
mobile-base navigation and new hardware validation retain separate roadmap
scopes. Locally installed development skills are authoring tools, not runtime dependencies or substitutes for source-bound physical tests.

## Operator evidence sequence

Use the normal [desktop launch](DGX_SPARK_SETUP.md#4-start-and-prove-the-demo)
from the intended installed checkout. A successful installation check means
PREPARED; a skip-proof launch means STARTED / UNVERIFIED. After a complete native
proof, run the saved-evidence [strict checker](SPARK_DELIVERY.md#native-confirmation-gate-and-failed-follow-up-1-october-2026).
Run a five-object native campaign in that same verified session, fail fast on a
failed case, and preserve its receipts before considering restart acceptance.
Never delete outcome memory to make a new test look like a clean success.
