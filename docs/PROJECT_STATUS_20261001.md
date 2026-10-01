# Project status — 1 October 2026

This is the current source and acceptance index. Dated experiment reports remain
historical evidence; their successful cases and failures are not rewritten when
a later source is merged. See [Spark setup](DGX_SPARK_SETUP.md) for installation,
[Spark delivery](SPARK_DELIVERY.md) for native proof and campaign contracts, and
[nvblox](NVBLOX.md) for the separate mapping experiments.

## Source and software validation

Current runtime **`27f2b0d7ab76f1c4e3bf246d1b23c636e570496f`** merges
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

The endpoint-occlusion correction in [PR #51](https://github.com/johnnynunez/cascade/pull/51)
passed 3,467 software tests on candidate `eb90272`. Its new Spark run has not
started: Tailscale reported the host offline from 15:24 UTC and SSH timed out
before source transfer. The owner subsequently confirmed the Sparks were
intentionally disconnected and requested local validation on the dual-RTX Pro
PC. That x86 validation is being prepared and will not certify ARM delivery. The [candidate receipt](evidence/finger-depth-occlusion-integration/validation.json)
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

[NVIDIA ovrtx](OVRTX_RENDERER.md) is now an optional camera and scene renderer,
with execution evidence on both Linux architectures as described above. It was
not included in physical baseline `477c88f`. No automatic Isaac/Newton snapshot
producer, robot/target masks or manipulation authority is supplied; those
connections require their own implementation and acceptance. Cosmos,
mobile-base navigation and new hardware validation retain separate roadmap
scopes. Locally installed development skills are authoring tools, not runtime dependencies or substitutes for source-bound physical tests.

[Screw manipulation research](SCREW_MANIPULATION_RESEARCH.md) identifies five
SimReady USD catalogue assets and the Factory/Isaac Lab/Newton threading paths.
The existing `turn_screw` commands wrist strokes; measured thread advancement,
engagement, seating and tightening torque are not implemented. SimReady's listed
CC BY-NC terms and the original Factory mesh BSD-3-Clause licence are distinct.
No threaded asset or simulation was installed for this research.

## Operator evidence sequence

Use the normal [desktop launch](DGX_SPARK_SETUP.md#4-start-and-prove-the-demo)
from the intended installed checkout. A successful installation check means
PREPARED; a skip-proof launch means STARTED / UNVERIFIED. After a complete native
proof, run the saved-evidence [strict checker](SPARK_DELIVERY.md#native-confirmation-gate-and-failed-follow-up-1-october-2026).
Run a five-object native campaign in that same verified session, fail fast on a
failed case, and preserve its receipts before considering restart acceptance.
Never delete outcome memory to make a new test look like a clean success.
