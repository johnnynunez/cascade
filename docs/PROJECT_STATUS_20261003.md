# Project status — 3 October 2026

CASCADE implements modular robot domains, bounded tool execution, independent
observations and explicit outcome verification. Physical acceptance belongs to
an exact model, source and episode; a declaration, passing software test or
successful simulator startup does not admit a new robot. This index separates
implemented capabilities, measured results and remaining work. The architecture
and [SVG / PNG diagram](ROBOT_MODULARITY.md) now distinguish the existing
single-robot domains from fleet coordination and the pending shared scene. Current
software includes Isaac Sim source-installation detection in
[PR #109](https://github.com/johnnynunez/cascade/pull/109); source updates do not
transfer acceptance from earlier physical episodes. The dated source and CI
records below remain historical evidence, including
[the integrated baseline](evidence/project-status-20261003/integrated-main-ci.json).
The [1–2 October index](PROJECT_STATUS_20261001.md) preserves earlier history.

## Capability and evidence

| Area | Implemented and measured | Remaining boundary |
| --- | --- | --- |
| Manipulation | SafeArm control, observed grasp/release guards, task-effect accounting and measured withdrawal. Historical OVRTX/cuMotion PhysX pick/place/home passed. The original two-object MuJoCo delivery test also passed in one new episode on integrated main `274fa3a9`, including release, support, rest, both home returns and retention of the first object. | Broader task/campaign reliability, hardware and acceptance of later source compositions. |
| Fastening | The explicit optional SDK recipe passed a native ordinary reset/one-turn/rest episode on 2026-10-04: final 0.958650 turns within the existing 0.05-turn tolerance, 2.398416 mm advance, 0.5 simulated seconds of rest, all 2,047 solves retained and owned closure complete. [Bound result](FACTORY_FASTENING_RUNTIME.md#native-mounted-turn-and-rest-2026-10-04). | One shared-GPU episode with only 0.151804 s command deadline margin; repeatability remains unproved. Tool pickup, initial engagement, seating, withdrawal and calibrated preload remain unproved by this configured domain. |
| Locomotion | Optional measured `walk_distance`, cancellation and independent support/rest checks; historical fresh-start ±30 mm episodes passed. | Published candidate profiles have no new model/support admission. The 0.1 m software ceiling is not measured 0.1 m capability or general gait acceptance. |
| Sensing and spatial memory | Passive, bounded sensor providers; identity/epoch/age checks; observed calibrated RGB-D and retained pixel-to-surface annotations without constructing an arm. | Independent metric XY/general-3D accuracy, moving-camera calibration and per-AOV synchronization. No physical SLAM or `go_to` execution is admitted. |
| Speech | Browser/media gateway, real speech-provider path, original-intent deadlines, priority stop and bounded observed-result views. One continuous synthetic-input/native-motion recording exists. | That motion video contains no spoken robot reply. General dialogue/action reliability, microphone/speaker hardware and public hosted service remain unvalidated. |
| Fleet coordination | Concurrent `FleetRuntime` routes tasks to independent robot runtimes with separate ownership and stop state. | The multi-agent application and native shared scene remain in progress. No shared-scene 12-MicroDuck demo has run. |
| Structure and whole-body control | Typed embodiment and passive generalized-joint observations; domains expose only registered resources/tools. | Mixed physical mobile manipulation is refused. No admitted humanoid balance, dexterous-hand or generic whole-body controller. |
| Evaluation | Source-bound VAB and Arena integration/cancellation preflights ran real simulation. | Neither is a completed benchmark task or a predictor of general deployment success. |


## Isaac Sim validation and current follow-up

The selected Isaac Sim installation passed eight native Newton substep/graph
tests and CASCADE's source-installation check after #109. Deployment
configuration selects the concrete dependencies; these checks do not validate
a kitchen task or a MicroDuck policy on that installation.

The first kitchen attempt stopped at the old 6.1-only metadata gate. The second
passed that gate but blocked in remote asset discovery for an empty YCB list;
it was cancelled through its owned supervisor before a robot tool action.
[PR #110](https://github.com/johnnynunez/cascade/pull/110) removes the unused
lookup and dead object-spawning block without changing the locally authored
scene. The subsequent native retry confirmed grasp but reached the unchanged
300-second deadline during release/retreat. Complete placement, rest and home
remain unverified, including the post-release joint-stability behavior from
[PR #107](https://github.com/johnnynunez/cascade/pull/107).

MicroDuck's BAM, support extraction and graph guards bind an explicit Newton
configuration. Compatibility checks are implemented; a single-robot native
episode is still required before shared-scene fleet validation. No model or
source hash is automatically treated as physically admitted.

## Published software corrections

Reproduced software defects and focused fixture fixes have source-bound CPU
checks published in the following PRs. These selections overlap, particularly the three #93 selections;
the counts must not be summed. The selections do not establish physical
admission; the separate integrated-main CI receipt is linked above. Subsequent Factory startup checks are recorded
separately below.

| PR and source | Corrected behavior | CPU evidence |
| --- | --- | --- |
| [#85](https://github.com/johnnynunez/cascade/pull/85), `cce8880` | Final threading verdict includes the unchanged stop/rest interval, so observed unwinding cannot retain a pre-stop success. Late positive motion cannot rescue an earlier failed result. | [402 passed](evidence/project-status-20261003/factory-final-outcome-cpu.json). |
| [#85](https://github.com/johnnynunez/cascade/pull/85), `5c21af4` | Backend exception formatting cannot erase an owner fault or strand queued admissions. Stop precedes formatting, and failed zero upload remains a separate failed outcome. No native result is transferred to this fix. | [393 passed](evidence/project-status-20261003/factory-error-containment-cpu.json). |
| [#87](https://github.com/johnnynunez/cascade/pull/87), `09650fd5` | `walk_distance` retains a veto for excess drift after command completion, including valid late observations. Original positive-credit, support and cancellation gates remain intact. | [566 passed](evidence/project-status-20261003/mobile-distance-final-outcome-cpu.json). |
| [#87](https://github.com/johnnynunez/cascade/pull/87), `1e9c8902` | Two zero-twist balance fixtures isolate unrelated cyclic GC with the existing helper. Real delayed reads still reject at the unchanged 40 ms limit; runtime and physical criteria are unchanged. Controlled GC reproduces the fixture failure but does not establish the original ARM CI cause. | [336 passed](evidence/project-status-20261003/mobile-balance-arm-fixture-cpu.json). |
| [#87](https://github.com/johnnynunez/cascade/pull/87), `40e36d90` | Positive distance fixtures advance the mock through its original small steps without relying on a background worker. A held read still reaches the unchanged wall deadline, triggers independent stop and earns no late credit. This controlled reproduction does not identify the original macOS scheduling cause. | [106 passed](evidence/project-status-20261003/mobile-distance-positive-fixture-cpu.json). |
| [#90](https://github.com/johnnynunez/cascade/pull/90), `3e696d49` | Terminal reports retain unresolved task effects and earlier fast-path history. Admitted motion cancellation records uncertainty before `BaseException` propagates. | [141 passed](evidence/project-status-20261003/task-terminal-obligations-cpu.json). |
| [#90](https://github.com/johnnynunez/cascade/pull/90), `92cc6c11` | An admitted motion's ordinary exception cannot attach its uncertainty to a new task begun after admission release. Same-task debt, early validation refusals and cancellation accounting remain intact. | [78 passed](evidence/project-status-20261003/task-exception-epoch-cpu.json). |
| [#90](https://github.com/johnnynunez/cascade/pull/90), `80035374` | The TCP timeout fixture permits read-only reconnects after a delayed caller resumes. Its real 40 ms limit, negative verdict and production code remain unchanged. | [68 passed](evidence/project-status-20261003/task-tcp-fixture-publication.json). |
| [#93](https://github.com/johnnynunez/cascade/pull/93), `7b765a6` | Withdrawal/home consumers preserve the cancellation token; completion validates original or registered reset context and clears debt atomically. Checks between reset callbacks prevent later work after detected cancellation; completed belief clearing remains reported. | [151 passed: token/context](evidence/project-status-20261003/mujoco-withdrawal-cancellation-cpu.json); [183 passed: callback boundaries](evidence/project-status-20261003/mujoco-reset-callback-cancellation-cpu.json); [122 passed: consumer-token fixture](evidence/project-status-20261003/mujoco-withdrawal-publication.json). |

The #93 head combines `fe0a49ee`, callback fix `7a6f078d` and fixture update
`a86cfca4`. Its checks are cooperative callback boundaries, not rollback of
already-executed reset work. Historical Trial12, Factory readiness and RGB-D
geometry failures below keep their original sources and verdicts.

The [local composition follow-up](evidence/project-status-20261003/terminal-fixes-composition.json)
on `167be813` passed 264 tests with one physical test deselected. The preceding
985-pass/3-fixture-failure result is retained; those three failures were resolved
by the token-aware fixture update. Its 1044-case coverage union spans distinct
source-bound phases, not a full suite or all cases run on the final head.

## Manipulation and trial 12

The earlier [OVRTX/cuMotion integration](MANIPULATION_ASSEMBLY_20261002.md)
completed a PhysX GPU grasp, lift, placement and home in 126.19 s. That result
remains distinct from ordinary MuJoCo delivery, the mounted-socket experiment
and the complete kitchen campaign.

The latest ordinary OpenClaw/Qwen kitchen attempt used source `9b08094`,
cuMotion 1.1.0 and Isaac cameras in epoch `1497ff1f`. Seven planned curves
completed; the eighth was refused before motion because joint feedback changed
between planning and execution. `pick_and_place` failed during post-place retreat
at **273.588 s**, inside the unchanged 300-second MCP limit. Full placement,
rest and home remain unverified. All four owned scopes closed and 41,713 inputs
plus five protected stores stayed unchanged; campaign teardown/rest gates did
not pass. The detailed log prefix dropped 137 events, so the exact final drift
cannot be reconstructed. [Results and audit reference](evidence/project-status-20261003/native-task-status.json).

The original two-pick test has now been revalidated on integrated main
`274fa3a9d48d96a8b1402dfeb2fe79c45e67037e`: **1 passed in 161.17 s**, using the
ordinary runtime with mock LLM, MuJoCo 3.14 and Mesa software rendering on CPU.
This new episode uses epoch `459727548a2c47dc8a9c64f427a0cd2e`; its observed
placement and withdrawal model digests retain their distinct scopes.
[Independent physical audit](evidence/project-status-20261003/mujoco-main-two-pick-review.json),
[unchanged test and verifier](evidence/project-status-20261003/mujoco-main-physical-contract.json).

Both placements pass actual opening, positive solved support, full collision
footprint containment and the original rest gates, with no active arm contact
or other loaded contact in the settling windows. Red supplies **11 samples over
0.600 s**; blue and retained red after the second home each supply **12 over
0.660 s**. Both measured home errors are below **0.000358 rad**, within the
unchanged **0.03 rad** tolerance. Final distances to the original drop point are
**25.776 mm** and **30.657 mm**, below the unchanged **60 mm** criterion.

The offline audit reconstructs all **512** saved state digests, contact support
and counts, velocities, gripper fractions and box footprints from the raw
records and generated scene. Fourteen negative audit controls reject. The
supervisor exits zero with natural closure and no signals; **1,854 source files,
26 assets, 5,687 SDK files and five protected stores** remain unchanged.
This is one episode with observations at existing driver batch boundaries,
not every substep or a reliability/performance benchmark. The audit does not
recompile the model, exercise reset or admit hardware, Isaac, GPU or an LLM-host
campaign. The historical AIM result and Trial12 failures below remain separate.

The original two-pick test on local source `d273dc4f` passed in **157.36 s**
using MuJoCo 3.14 and Mesa software rendering on CPU. Both placements were
independently confirmed after actual opening, withdrawal and home; the second
preserved the first. The red verification window contains 11 observations over
0.600 simulated seconds; blue and the retained red prefix have 12 over 0.660 s. The original 6 cm destination criterion remains intact.
The process exited naturally and source/assets/protected stores were unchanged.
Its [source-bound audit](evidence/project-status-20261003/mujoco-two-pick.json)
does not admit Isaac, GPU or hardware. Earlier occupied-region,
post-opening escape and rotated-attachment targeting failures remain historical
failures; this is one corrected two-object episode.

Scripted stdio MCP kitchen cases passed separately: green pick/place/home in
199.455 s and orange in 241.531 s, followed by separately timed reset and
complete owned closure. They used independent physical/camera witnesses but no
LLM host. [Green receipt](evidence/project-status-20261003/kitchen-mcp-300s.json),
[orange receipt](evidence/project-status-20261003/kitchen-mcp-orange-complete.json).

The earlier real-host attempt on `f674fa64` remains **FAIL** at the unchanged
300 s MCP limit: one green pick, no retry/reset/orange case, and no completed
placement/home result. Return-home timing was inferred from joint readback and
control flow, without an exact home-start command receipt. Its resource closure
and external embedding 429 errors remain in the
[earlier source-bound report](evidence/project-status-20261003/realhost-trial12-observer.json).

**The subsequent instrumented OpenClaw/Qwen trial also remains FAIL.** Source
`ef773697183f676973244ceb513d44f74a177792` used the same 300000 ms MCP deadline;
the first green-cube pick/place agent turn lasted **303.910 s** and returned a
timeout. The grasp reported verified after 125.708 s, but no completed task,
placement/home audit, retry, reset or orange case followed. The last passive
cube position was 2.015 mm from the destination center with open fingers; that
endpoint does not establish placement, rest or whole-task acceptance.
[Retained command audit](evidence/project-status-20261003/realhost-trial12-command-audit.json).

The optional [PR #86](https://github.com/johnnynunez/cascade/pull/86) target and
motion diagnostics remain **off by default**. This episode exercised them,
but coverage is partial: the **16 MiB** budget retained 6019 events
(16777178 compact bytes) and dropped **1102**. Eight stream starts and seven
returns survive, without `skill_end`; the final stream cannot be reconstructed.
The two last retained command IDs lack setter observations; partial logging does
not establish that those commands were lost.
For the two complete 225-waypoint streams, nominal **7.5 s** became **64.571 s**
and **80.238 s** of wall time, each spanning 11.367 simulated seconds. Fresh
post-ACK anchors contributed **3.75 simulated seconds** beyond the nominal
intervals. These are measured costs of the existing no-catch-up contract, not
authorization to remove its progress/freshness checks. ACK median was 0.544 ms;
a setter return still does not prove a subsequent solve. The shared-GPU run
does not isolate an OS, GPU, renderer or kernel cause and demonstrates no speedup.

The audit preserves launcher exit 1 and false campaign lifecycle/normal-shutdown
flags. Separately, both MCP teardown receipts report complete software cleanup;
park was skipped under e-stop and physical rest remains unverified. The four
owned units and four ports closed, and six foreign process births remained
intact. Resource containment does not turn this failed campaign into a pass.
The new audit adds command timing to this episode only; it does not retrospectively
supply receipts for the earlier failed run or replay an action.

## Fastening and locomotion

The earlier Factory startup measurement uses source
`4bda2183a30074f8ad105ce497876301ce8fc0ed`, whose complete Git tree equals merged
main `274fa3a9`, and model
`01595b62c1b4856e41c7e8f83be6ee91b6414bda742aea7e6a7ed0b3e8daa94d`.
After a retained refusal caused by an extra GPU process, a separately authorized
preparation passed with **39 compiler loads, 22 predicates and zero solves**.
It started no owner or motion. [Preparation audit](evidence/project-status-20261003/factory-archive-preparation-review.json).

A new instrumented readiness episode in epoch `6eed234fd7674e7b8da18dcc20039e3b`
then **passed**: steps **59–359** supplied **0.5000000121 simulated seconds** of
continuous quiet with positive thread/tool contacts, within **9.787795156 wall
seconds** of the unchanged **10 s** deadline. The margin was only **0.2122 s**;
this single shared-device episode does not establish repeatability or general
performance. All **361** retained solve rows preserve the initial arm hold and
zero spindle command/effort, and every controller acceptance returned normally.
Both owned processes closed naturally with exit zero; five protected stores and
all source/SDK/assets inventories were unchanged. No turn or reset was requested.
**Readiness passes for this episode; fastening and post-command physical rest
remain unvalidated.** [Independent native audit](evidence/project-status-20261003/factory-archive-readiness-review.json).

The complete 7,988-event phase trace still contains a 417.793 ms generation-2 GC
interval, on a different thread after readiness. The result does not mean GC
was removed or prove a general cause or speedup. Ordinary raw-record decoding
can occur after readiness while the owner is alive; only phase export and the
remaining post-close drain are guaranteed to follow owner join. Earlier failed
episodes below retain their original source, model, clocks and verdicts.

### 2026-10-03 — Subsequent one-turn attempt failed during startup

On the same `4bda2183` source and `01595b62` model, new epoch
`888ba87b2f6145a8ae25e0d6b095c90d` **failed** the unchanged **10 s** readiness
deadline. Quiet began at step **61**; the last gen0 solve accepted before the
deadline was **359**, supplying only **0.496666679 simulated seconds**. Two
further quiet steps would have been needed for the required 0.5 s. No reset or
turn was dispatched. There was no freshness fault or generation-2 GC interval
during this readiness call; those observations do not establish a performance
cause or a validated optimization.

All **360 rows and 98,326 contacts** were retained. Actual exit was **1**, with
natural owned-process closure, no owner error and a zero spindle upload.
Source/SDK/assets inventories and five protected stores were unchanged.
**This is a startup failure, with no fastening-task or post-stop-rest proof.**
It does not alter the earlier readiness PASS or historical failures.
[Independent failure and closure audit](evidence/project-status-20261003/factory-one-turn-startup-failure-review.json).

### 2026-10-03 — New readiness PASS; one-turn task remains UNVERIFIED

Source `96198a95` and model `7de78290` passed preparation with **39 loads, 22 predicates and zero solves**.
Separate epoch `b72d7943` passed readiness in **9.492777 wall seconds**, with **0.5000000121 simulated seconds** of quiet under the original limits.
New epoch `abf26edb` then invoked one ordinary `reset_stop` and `fastening.turn_screw(turns=1.0, direction="tighten")` through the production MCP handler, in-process without stdio or an LLM.
At step **1236**, controller acceptance rejected observation age **0.457537746 s > 0.2 s**; task completion and post-command rest remain **UNVERIFIED**.
The trace measured **449.221773 ms** of generation-2 GC on the owner thread inside `observer.read`, within this rejected observation interval.
This identifies a pause in this episode; it establishes neither a validated fix nor a general speedup or explanation of earlier failures.
Owned processes closed naturally with exit **1**, and source inputs/five protected stores were unchanged; `runtime_closure.complete=false` and `physical_stop_verified=false` remain explicit.
[Compact results and exact local audit hashes](evidence/project-status-20261003/native-task-status.json). Earlier outcomes above are preserved.

The historical mounted-socket experiment measured 15.365 turns, 38.155 mm
advance, seating and two seconds of zero spindle-motor torque. The arm/socket
remained engaged; it was not tool withdrawal or preload calibration. Ordinary
`turn_screw` cannot claim verified tightening without an independent fastener
observer. [Experiment and limits](MANIPULATION_ASSEMBLY_20261002.md#physical-threading-seating-and-retention).

Factory's separate v2 runtime readiness failed its unchanged **10 s** budget:
the first advance took 9.364 s, with 20 completed steps before timeout and one
more drained at closure. No quiet startup or fastening action was admitted.
The v3 compiler recipe binds concrete contact-writer types and loaded symbols;
492 CPU checks passed, with seven exact-ABI checks separately passing. Native v3 preparation subsequently passed on `1f0d194f`, model `e712ac6c`:
39 loads and 22 predicates, with scene/native clocks and steps unchanged at zero.
Both owned processes closed naturally; six foreign process births and the
source/SDK/assets/stores were preserved. The observed 82.512 s on a shared GPU
is not a performance claim.

The subsequent v3 ordinary readiness episode also **failed**, after 309 native
solves and 0.515 simulated seconds. The solve owner rejected a stale/future
state after 8.280 wall seconds, before the unchanged 10-second startup limit.
The retained quiet segment spans steps 55–309, only 0.423 simulated seconds of
the required 0.5; it cannot establish readiness. Both owned processes exited
naturally and six foreign process births remained intact. Exact rejection age
and stage were not recorded, so the cause cannot be assigned to GPU, GC or
scheduling. No fastening action or reset was requested, and physical stop was
not verified. [Retained v3 failure and independent audit](evidence/project-status-20261003/factory-readiness-v3-review.json).
[Readiness failure](evidence/project-status-20261003/factory-readiness-v2-review.json),
[v3 CPU receipt](evidence/project-status-20261003/factory-contact-writer-cpu.json),
[v3 native preparation](evidence/project-status-20261003/factory-native-v3-prepare.json).

The subsequent source `cce88808d4bded97a84c02a562cb3ceb997de954`, with model
`69d9a46aaba9304df44828cbca1460a1a4166b9d58c9625b3b6f6bdacf07ed00`, passed a
separate preparation: 39 compiler loads, 22 predicates and **zero scene/native
solves**, with no owner started. Preparation and readiness used distinct epochs.
The preparation process closed naturally; this admits neither readiness nor a
fastening task. [Preparation audit](evidence/project-status-20261003/factory-current-preparation-review.json).

Its ordinary readiness attempt remains **FAIL**, now with a bound age diagnostic.
At `controller_accept_solve`, step **303** was rejected with capture-to-check age
**0.43165935698 s**, exceeding the unchanged **0.2 s** limit. The retained raw
quiet span before that row is steps **64–302**, **0.3966666763 simulated seconds**,
short of the required **0.5 s**. Rejected row 303 cannot supply readiness credit;
even including its raw values would give only 0.398333343 s. The failure occurred
after 8.938 wall seconds inside the unchanged 10-second readiness budget.
[Readiness audit](evidence/project-status-20261003/factory-current-readiness-review.json),
[bound error and age extraction](evidence/project-status-20261003/factory-current-age-diagnostic.json).

All 303 raw solve records retain the original arm hold and zero spindle command
and observed effort. These records do not establish that every row was accepted
live, and do not verify physical stop. The owner thread closed but its fault and
`ok: false` remain. The scope exited naturally with code 1, all three tracked
owned processes closed, six foreign process births were unchanged, and the
source/SDK/assets/stores inventories matched before and after. No turn, reset or
task was requested. The age diagnostic measures a gap, not its cause: it assigns
no GPU, garbage-collection or scheduling attribution and renews no timestamp.
**That source's ordinary readiness and fastening remain unadmitted.** Compilation success,
zero spindle effort and resource closure do not establish a solved task.

A subsequent **instrumented** readiness episode on the same `cce88808` source
and `69d9a46a` model also ended **FAIL + CLOSED**, in new epoch
`17817f0741da4303bebed4da366f82d5`. Step **309** was rejected at
`controller_accept_solve`: its original capture-to-check age was
**441.3566 ms**, above the unchanged **200 ms** limit. The raw quiet prefix is
steps **63–308**, **0.4083333432 simulated seconds**, below the required **0.5 s**;
including rejected row 309 would still give only 0.4100000099 s. Readiness failed
after 9.17247 wall seconds within its unchanged 10-second budget. No turn, reset
or fastening task was requested.

The complete phase trace records a same-thread **generation-2 cyclic-GC interval
of 432.885 ms wall / 427.607 ms thread CPU**, wholly inside the step-309
`contact_records` interval (439.388 ms). These are nested intervals, not additive
timings. This localizes an observed pause in **this instrumented episode**; it
does not identify retained objects or explain the earlier uninstrumented
failures. GC was neither disabled nor forced, and no extra SDK reads were added.
The timings include instrumentation overhead and do not establish isolated GPU
or whole-system performance. [Phase and terminal audit](evidence/project-status-20261003/factory-phase-diagnostic-review.json),
[bound original error](evidence/project-status-20261003/factory-phase-age-diagnostic.json).

All 309 raw rows preserve the arm hold and zero spindle command/observed effort;
they do not prove live consumer acceptance or physical stop. The owner thread
closed with its original fault and `ok: false`. The process scope exited
naturally with code 1, both tracked owned processes were absent, six foreign
process births were unchanged, and source/SDK/assets/stores matched before and
after. The complete diagnostic trace and resource closure do not promote the
readiness or task verdict.

MicroDuck has four recorded fresh-start forward/reverse **30 mm** passes, a
separate small negative turn and priority cancellation with observed rest.
Other transition/contact and larger-turn failures remain retained. New source
composition requires a new identity: the distance profiles remain unpinned and
reject construction before transport. Same-solve reuse does not create advancing
observations or new travel credit; mocks cannot confirm physical outcomes.
[Historical distance receipt](evidence/project-status-20261003/microduck-distance-candidate.json).

## RGB-D geometry, speech and evaluation

A prior native static-camera RGB-D v2 episode delivered fresh captures over TCP
through the real MCP handler/SensorHub into 20 retained surface annotations.
Its selected ground-Z checks did not establish XY accuracy, general 3D accuracy
or independent timestamps for both AOVs.
[Capture/annotation receipt](evidence/project-status-20261003/rgbd-native02-review.json).

The later layout-A episode remains **FAIL**: all five native captures identified
the four tags, but all five failed the unchanged independent held-out geometry
gate. Four live geometry attempts produced **zero accepted annotations**.
Held-out RMS was 0.1648–0.1776 px against 0.15 px. The separate schema-5 CPU
checker evaluated a frozen 48-image synthetic corpus, excluding those native
failures: 19 accepted, **29 rejected** (16 missing detections, 13 residual
failures), with two gains and no admission losses against the prior detector.
Some admitted cases still had analytical-truth RMS 0.350258 px: a consistent
homography is not proof of absolute accuracy. No pixel gate was relaxed.
[Native result](evidence/project-status-20261003/rgbd-layout-a-native-summary.json),
[CPU comparison](evidence/project-status-20261003/rgbd-checker-accuracy-cpu.json).

The 2026-10-04 CPU follow-up reproduced those five failures exactly and added
explicit schema-5 producer/consumer binding with the same geometry and gates.
All 286 RGB-D tests passed; consumer/SDK check-only descriptors matched. No new
native render or accepted annotation is claimed.
[Producer selection and diagnostic limits](RGBD_CHECKER_ACCURACY.md#explicit-producer-selection-2026-10-04).

The explicit schema6 RGB-only local saddle candidate passes 29/48 new frozen
synthetic cases versus 28/48 for schema5; 19 cases still reject. In the 29
complete pairs, absolute truth RMS improves in each, but all three separately
retained known regressions worsen despite passing the original homography
gates. No parameters or gates changed. All nine adversaries reject and 312
RGB-D tests pass. This is an opt-in CPU candidate with no new native result.
[Full outcomes and limits](RGBD_CHECKER_ACCURACY.md#explicit-local-rgb-refinement-schema6),
[bound CPU receipt](../benchmark/results/rgbd_checker_saddle_cpu_20261004.json).

The delivered **61.056 s** continuous video retains 915 frames. Labelled
synthetic speech entered the actual browser/Whisper-base/Qwen3-1.7B route and
selected `locomotion.walk_distance(0.03)`. Controller displacement was 25.249 mm;
an independent reconstruction with different endpoints measured 25.921 mm.
The supervisor selected stop, and the native audit reported motion and rest
confirmed. **No TTS reply was recorded.** This used conversation/RobotRuntime,
not MCP/OpenClaw or hardware. The original owner failure receipt remains intact;
the later bounded audit/delivery is recorded separately. File/media/source
consistency was independently reviewed without rerunning the physical auditor.
[Delivery](evidence/project-status-20261003/microduck-video-delivery.json),
[independent review](evidence/project-status-20261003/microduck-video-root-review.json).

VAB's real Panda move/cancel and Arena's PhysX zero-action/cancel preflights
validate limited integration. Arena ran 20 control steps/160 solves, and the
interrupted case stopped admission before step nine. No official benchmark task
success or continuous-world braking follows from those checks.
[VAB receipt](evidence/project-status-20261003/vab-native-preflight-20261002.json),
[Arena receipt](evidence/project-status-20261003/arena-native.json).

## Publication and reproducibility

The [GitHub snapshot](evidence/project-status-20261003/github-publication.json)
records the read-only API observation time and exact branch/commit state.
PRs [#79](https://github.com/johnnynunez/cascade/pull/79),
[#80](https://github.com/johnnynunez/cascade/pull/80),
[#81](https://github.com/johnnynunez/cascade/pull/81),
[#83](https://github.com/johnnynunez/cascade/pull/83) and
[#84](https://github.com/johnnynunez/cascade/pull/84) were merged into main.
[#82](https://github.com/johnnynunez/cascade/pull/82) merged into a feature
branch; [#85](https://github.com/johnnynunez/cascade/pull/85) is the separate
Factory integration into main and was still open at observation. PRs
[#86](https://github.com/johnnynunez/cascade/pull/86),
[#87](https://github.com/johnnynunez/cascade/pull/87),
[#88](https://github.com/johnnynunez/cascade/pull/88),
[#89](https://github.com/johnnynunez/cascade/pull/89),
[#90](https://github.com/johnnynunez/cascade/pull/90) and
[#91](https://github.com/johnnynunez/cascade/pull/91) were open. PR state is not
CI or physical acceptance; this documentation change performs no merges.

Historical receipt fields about a publication freeze or reserved video resources
refer to those past runs, not current publication or execution prerequisites.
The [copy manifest](evidence/project-status-20261003/sources.json) binds each
portable receipt to its unchanged original bytes. Original local artifact paths
inside receipts identify retained evidence, not downloadable repository files.
No new unit suite, benchmark, model inference or physical run was performed for
this documentation update.
