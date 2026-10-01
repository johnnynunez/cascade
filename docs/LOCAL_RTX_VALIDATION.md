# Local RTX Pro validation — 1 October 2026

The owner disconnected the Sparks and requested validation on the Linux x86_64
workstation with two RTX Pro GPUs. This continues the project validation on a
different machine. It does not establish new aarch64 acceptance or replace the
source-bound Spark results in [project status](PROJECT_STATUS_20261001.md).

## Source and execution scope

The local native path uses Isaac PhysX CUDA, GraspGen-X and an existing Qwen
endpoint through an isolated OpenClaw profile. Each run has its own checkout,
Python import binding, process ownership, gateway state, memory and evidence.
Qwen is borrowed; unrelated processes retain their identities and ownership.
The presenter configuration uses three 1280 × 720 RGB-D cameras, a 120 Hz
physics timestep and `occupancy=none`. Its camera and physical checks retain
their existing acceptance limits. This is native CLI validation, with no
claim of visitor HTTP, desktop UI or new Spark installation acceptance.

The local validation began with these software changes:

- [PR #53](https://github.com/johnnynunez/cascade/pull/53), candidate `da0255b`,
  retains the original NV attachment through transport. The complete suite
  passed 3,525 tests, with 43 skipped and four deselected in 350.18 seconds;
  all 1,124 source and release files remained unchanged. Its
  [full receipt](evidence/local-rtx-validation-20261001/carry-full.json) and
  [independent review](evidence/local-rtx-validation-20261001/carry-peer.json)
  establish software validation. The [carry contract](NVBLOX_CARRY_ATTACHMENT.md)
  describes its NV-only scope and terminal failure behavior.
  It merged as `41deced8` after all five PR CI jobs passed; the
  [merge review](evidence/local-rtx-validation-20261001/pr53-merged.json)
  confirms equality with the tested tree.
- [PR #54](https://github.com/johnnynunez/cascade/pull/54), candidate `dc56689`,
  preserves explicit GPU selection in MCP registration. It includes PR #53.
  Its [complete suite](evidence/local-rtx-validation-20261001/cuda-environment-full.json)
  passed **3,556 tests**, with 43 skipped and four deselected in 349.27 seconds.
  All 1,125 source and release files remained unchanged and were independently
  rehashed in the [root review](evidence/local-rtx-validation-20261001/cuda-environment-full-root.json).
  The [independent review](evidence/local-rtx-validation-20261001/cuda-environment-peer.json)
  covers literal values, absent and empty variables, local ordinals, UUIDs and
  explicit overrides. [Quickstart](QUICKSTART.md) describes the configuration.
  The first PR CI run passed both Linux architectures, minimal installation and
  UI checks, but macOS rejected the launcher's heredoc syntax. Follow-up
  `ed29a59` removes an apostrophe from one embedded Python comment. Its
  [validation](evidence/local-rtx-validation-20261001/cuda-bash-comment.json)
  records 36 passing focused tests and identical ASTs for all 16 embedded Python
  blocks. The complete local suite above remains bound to `dc56689`; the
  [original CI failure](evidence/local-rtx-validation-20261001/pr54-ci-failed.json)
  is retained separately.
  All five [follow-up CI jobs](evidence/local-rtx-validation-20261001/pr54-ci.json)
  passed, including macOS. PR #54 merged as `f9cb6b8e`; the
  [merge review](evidence/local-rtx-validation-20261001/pr54-merged.json)
  confirms its tree exactly equals reviewed `ed29a59`. All five
  [merged-source CI jobs](evidence/local-rtx-validation-20261001/main54-ci.json)
  also passed.

[Documentation PR #55](https://github.com/johnnynunez/cascade/pull/55) merged as
`261dc1b0` after all five [PR CI jobs](evidence/local-rtx-validation-20261001/pr55-ci.json)
passed. Its [merge review](evidence/local-rtx-validation-20261001/pr55-merged.json)
binds the reviewed documentation tree; it adds no physical acceptance claim.
All five [merged-source CI jobs](evidence/local-rtx-validation-20261001/main55-ci.json)
also passed.

The subsequent [PR #56](https://github.com/johnnynunez/cascade/pull/56) candidate
`33b27382` adds [Isaac verifier startup readiness](ISAAC_STARTUP_READINESS.md).
Its [complete local suite](evidence/local-rtx-validation-20261001/startup-readiness-full.json)
passed **3,591 tests**, with 43 skipped and four deselected in 338.61 seconds.
The [root check](evidence/local-rtx-validation-20261001/startup-readiness-full-root.json)
independently rehashed all 1,165 source and release files.
The [author's 96 focused tests](evidence/local-rtx-validation-20261001/startup-readiness-focused.json)
and [independent 37-test review](evidence/local-rtx-validation-20261001/startup-readiness-peer.json)
cover the probe, clock, camera, cleanup and packaging contracts. All five
[PR CI jobs](evidence/local-rtx-validation-20261001/pr56-ci.json) passed. It merged
as `ba8f2e83`; the [merge review](evidence/local-rtx-validation-20261001/pr56-merged.json)
confirms equality with the tested tree. All five
[merged-source CI jobs](evidence/local-rtx-validation-20261001/main56-ci.json)
also passed. New live acceptance is a separate stage; software tests do not
replace physical acceptance.

[PR #57](https://github.com/johnnynunez/cascade/pull/57), candidate `727aaabc`,
adds [localization freshness checks](LOCALIZATION_FRESHNESS.md). The
[142-test focused run](evidence/local-rtx-validation-20261001/localization-focused.json)
includes 28 new cases, including delayed constructor failure, empty depth and
geometry errors. The
[independent 70-test review](evidence/local-rtx-validation-20261001/localization-peer.json)
and [root review](evidence/local-rtx-validation-20261001/localization-root.json)
confirm terminal expiration and preservation of the actual successful camera
view. The original candidate and its corrected error paths remain in the
[initial focused receipt](evidence/local-rtx-validation-20261001/localization-focused-initial.json).
The [complete suite](evidence/local-rtx-validation-20261001/localization-full.json)
passed **3,619 tests**, with 43 skipped and four deselected in 346.98 seconds.
The [root check](evidence/local-rtx-validation-20261001/localization-full-root.json)
independently rehashed all 1,167 frozen source/release files and the full log.
All five [PR CI jobs](evidence/local-rtx-validation-20261001/pr57-ci.json) passed.
It merged as `6ee6375b`; the
[merge review](evidence/local-rtx-validation-20261001/pr57-merged.json)
confirms equality with tested `727aaabc`. The earlier candidate's superseded
CI run was [cancelled](evidence/local-rtx-validation-20261001/pr57-initial-ci-superseded.json)
after the error-exit correction, without claiming success for that older run.
All five [merged-source CI jobs](evidence/local-rtx-validation-20261001/main57-ci.json)
also passed. The new native run is tracked separately.

[PR #58](https://github.com/johnnynunez/cascade/pull/58), candidate `93af4f6`,
checks an observed closing conflict before sampling a candidate's route, after
the original pregrasp harness/map admission. Every candidate considered for
acceptance still runs all original checks. The
[closing contract](observed-finger-closing.md#reject-closing-conflicts-before-candidate-route-sampling)
describes the unchanged geometry, ranking, cancellation and deadline rules.
Its [77 focused tests](evidence/local-rtx-validation-20261001/closure-first-focused.json),
[independent review](evidence/local-rtx-validation-20261001/closure-first-peer.json)
and [root review](evidence/local-rtx-validation-20261001/closure-first-root.json)
pass. The [complete suite](evidence/local-rtx-validation-20261001/closure-first-full.json)
passed **3,628 tests**, with 43 skipped and four deselected in 338.11 seconds.
The [root check](evidence/local-rtx-validation-20261001/closure-first-full-root.json)
independently rehashed all 1,169 frozen source/release files and the full log.
All five [PR CI jobs](evidence/local-rtx-validation-20261001/pr58-ci.json) passed.
It merged as `0a27bda2`; the
[merge review](evidence/local-rtx-validation-20261001/pr58-merged.json)
confirms equality with tested `93af4f6`. All five
[merged-source CI jobs](evidence/local-rtx-validation-20261001/main58-ci.json)
also passed. New native acceptance is separate. The six historical negative poses
still reject in the [offline replay](evidence/local-rtx-validation-20261001/nv13-veto-early-closure.json).
Its timings do not measure native speed or identify a valid route.

## Retained local attempt 01

Source `27f2b0d` completed native startup, including real GraspGen-X inference
and a Qwen text turn. Robot proof was explicitly skipped; there were no robot
tool calls, motion traces or grasp-memory outcomes.

The first admission stopped because its diagnostic expected the registered
Isaac launcher PID to own the bridge port. The source installation uses a
separate Kit child. The corrected diagnostic bound the private readiness
record, both process birth identities, command, ancestry, listener and GPU.

The [second admission](evidence/local-rtx-validation-20261001/x86-01-gpu-admission.json)
then rejected the actual GPU assignment: Isaac and GraspGen-X used GPU 1, but
the new MCP process loaded its models on GPU 0. Its explicit environment lacked
the selected CUDA variables. This led to PR #54; no physical case was attempted.

Normal owned shutdown returned before Kit had finished exiting, so the immediate
port check recorded a [failure](evidence/local-rtx-validation-20261001/x86-01-close.json).
A separate [read-only follow-up](evidence/local-rtx-validation-20261001/x86-01-close-settled.json)
confirmed all owned services, the Kit interpreter and its launcher shell had
exited, all three ports were closed, and protected processes, source, proof and
absent memory were unchanged. No additional signals or robot commands were sent.
The original failure was retained, alongside the follow-up result.

The [retention record](evidence/local-rtx-validation-20261001/x86-01-retention.json)
binds the original diagnostic files. Startup and administrative closure are
not physical acceptance results.

## Local attempt 02: native timeout during placement

Source `dc566892` passed native startup and read-only camera/GPU admission.
Isaac, GraspGen-X and the actual MCP process used the selected GPU 1. Six
1280 × 720 RGB-D captures had valid calibration, advancing render identities
and ages at client delivery between 0.241 and 0.444 seconds. The
[admission receipt](evidence/local-rtx-validation-20261001/x86-02-admission.json)
and [actual proof-process binding](evidence/local-rtx-validation-20261001/x86-02-gpu-binding.json)
record these checks.

**The native proof failed on its first green-cube case.** Grasp completed in
208.508 seconds with verified grip, but placement was still running when the
unchanged 300-second MCP request limit expired. The gateway cancelled the call
and the MCP server latched the emergency stop. The
[timeout extracts](evidence/local-rtx-validation-20261001/x86-02-timeout.json),
[failed proof](evidence/local-rtx-validation-20261001/x86-02-proof.json) and
[negative strict check](evidence/local-rtx-validation-20261001/x86-02-strict.json)
retain that result. No reset, second proof case, five-object campaign or
acceptance restart followed. The scene and outcome memory were preserved.

The [independent partial-case review](evidence/local-rtx-validation-20261001/x86-02-physical-peer.json)
also found a separate camera failure: one early sample from each of the three
cameras had a server age of 2.268394 seconds, above the unchanged two-second
limit. It occurred before the first motion target. No exclusive cause is
established for either the stale capture or the later timeout. The last witness
sample still had bilateral green contact and was 0.16791 m from the destination
in XY; nonzero measured velocities prevent claiming a settled final state.

The [offline timing analysis](evidence/local-rtx-validation-20261001/x86-02-costs.json)
measures a 1.699-second first planning batch, 109.872 seconds for pregrasp
movement and 31.769 seconds for lift. Across 1,259 witness samples, simulation
advanced 31.783 seconds during 303.958 wall seconds: a real-time factor of
0.105. These measurements identify the movement/simulation cost; they do not
establish a unique cause or justify extending the timeout. Sparse final samples
still show bilateral contact, without confirming completed placement.

The [retention manifest](evidence/local-rtx-validation-20261001/x86-02-retention.json)
binds 75 retained files. Independent root review rehashed all files and the
27,639,002-byte archive, SHA-256
`6d3f399fb3ee947ee7b663764ef9eea7e49911e827573d5bf439ed9bbb63b2bd`.
This integrity check does not convert the failed physical case into a pass.

After preserving the failure, [normal owned shutdown](evidence/local-rtx-validation-20261001/x86-02-close.json)
returned successfully. The bounded follow-up confirmed that all owned process
identities had exited and the three ports were closed. Source, outcome memory,
proof, traces and protected processes remained unchanged. This administrative
closure sent no additional robot commands and did not clear or recover the task.

Process telemetry found two simultaneous owned MCP runtimes: one from the
brain-only startup check and another from the separate proof invocation. Both
kept perception active. Attempt 03 used the normal complete launcher and its
single proof session, removing the extra diagnostic runtime.

## Local attempt 03: single native session

Source `ed29a59` ran one normal launcher proof with the outcome memory retained
from attempt 02. The [runtime binding](evidence/local-rtx-validation-20261001/x86-03-gpu-binding.json)
confirms the actual MCP, Isaac and GraspGen-X processes used GPU 1. The physical
configuration and the 300-second MCP limit were unchanged.

**The first green-cube case still failed.** The
[timing analysis](evidence/local-rtx-validation-20261001/x86-03-costs.json)
records verified grasp in 146.856 seconds, including 80.338 seconds of pregrasp
movement, 19.027 seconds of descent and 20.569 seconds of lift. Across 1,489
witness samples, physics advanced 44.433 seconds during 303.749 wall seconds:
a real-time factor of 0.146. This was faster than attempt 02, but changing host
load and retained outcome memory prevent assigning the improvement exclusively
to removal of the second MCP runtime.

The [failed proof](evidence/local-rtx-validation-20261001/x86-03-proof.json)
again reached the per-call deadline. The trace contains no completed
`pick_and_place` result or verified placement postcondition. The
[independent partial-case review](evidence/local-rtx-validation-20261001/x86-03-physical-peer.json)
shows the green cube within the destination pad at approximately
`[0.137009, -0.276564, 0.030000]` m and open jaws, but home remains incomplete
with 1.119 radians of maximum joint error and nonzero final velocities. These observations do not establish
completed placement or identify the exact skill stage when cancellation arrived.
No reset, second proof case, campaign or acceptance restart followed.

The [strict check](evidence/local-rtx-validation-20261001/x86-03-strict.json)
also rejects two early samples per camera, with maximum server age 2.395272
seconds, before the first actuator target. The
[readiness change](ISAAC_STARTUP_READINESS.md) prepares the native read-only
physics channel before proof and waits for fresh subsequent camera captures;
its effect on live acceptance requires a new run. It does not extend the motion
deadline or repair slow simulation throughput by itself.

The [retention manifest](evidence/local-rtx-validation-20261001/x86-03-retention.json)
binds 56 files. Root review independently rehashed them and the 9,290,800-byte
archive, SHA-256
`ed6199b5a1f82ae9a6ba0747cde59ec9fc0ab5a5cc031f6d9b8b6b47b493e96c`.
After the failed launcher had closed its gateway and MCP,
[normal administrative shutdown](evidence/local-rtx-validation-20261001/x86-03-close.json)
closed only the remaining owned bridge and GraspGen-X processes. All owned ports
closed; source, memory, proof, trace and protected process identities remained
unchanged. It sent no additional robot commands.

## Local attempt 06: fresh cameras, retreat timeout

Source `93af4f6` ran the normal complete launcher with the reviewed startup,
localization and selection changes. Prepared attempts 04 (`33b27382`) and 05
(`727aaabc`) were never launched. Attempt 06 inherited the normal user HOME,
with explicit private OpenClaw state and memory paths, and preserved attempt
03's grasp-outcome memory. Its user-cache context differs from attempt 03.

The [startup readiness report](evidence/local-rtx-validation-20261001/x86-06-readiness.json)
passed in 2.338 seconds, with all three camera age upper bounds at 0.461 seconds.
That startup result is separate from the later independent case measurements.

The [native attempt](evidence/local-rtx-validation-20261001/x86-06-proof.json)
failed its first green-cube case at the unchanged 300-second MCP deadline.
The trace records cancellation during post-place retreat, with return home
not attempted and the placement postcondition still unverified. Grasp completed
in 137.410 seconds. No reset, orange case, five-object campaign or restart
followed. The [strict check](evidence/local-rtx-validation-20261001/x86-06-strict.json)
remains negative.

The [independent physical review](evidence/local-rtx-validation-20261001/x86-06-physical-peer.json)
decoded all 1,391 raw witness replies and matched 627 shared physics steps to
the control feedback. All three cameras stayed within the two-second server
age limit: maximum 1.018476 seconds, with maximum client-delivery age
1.102881 seconds. Final observations show the cube near the pad, open jaws and
no bilateral contact, but nonzero joint velocities and 1.339 radians of maximum
home error. These observations do not replace the incomplete native verdict.

The [timing analysis](evidence/local-rtx-validation-20261001/x86-06-costs.json)
records 1.615 seconds of planning, 68.636 seconds of pregrasp movement,
21.815 seconds of descent and 27.084 seconds of lift. The witness interval's
real-time factor was 0.135792. GPU utilization and process CPU samples describe
the shared host load; they do not isolate a cause or establish a controlled
performance comparison with attempt 03.

The [process provenance](evidence/local-rtx-validation-20261001/x86-06-process-binding.json)
binds the failed MCP to its registered owner and attempt PID. The launcher's
earlier probe MCP had exited before the proof MCP ran. Gateway and proof MCP
closed through the normal failure path; a terminal check therefore could not
attest their live GPU environment. That limitation and the additional binding
error are retained, without inventing a historical observation.

The [retention manifest](evidence/local-rtx-validation-20261001/x86-06-retention.json)
binds 63 files, independently rehashed along with the 9,150,540-byte archive,
SHA-256 `4878dccaccaeae13a11d55eb1854d40fa016e4cd844281c93c3830338ec5e1a7`.
The original administrative archive-path error was retained; its existing
destination check prevented overwriting attempt 03.

After preservation, [normal owned shutdown](evidence/local-rtx-validation-20261001/x86-06-close.json)
closed the remaining bridge and GraspGen-X processes. The
[root confirmation](evidence/local-rtx-validation-20261001/x86-06-close-root.json)
found all four recorded process identities absent and all three owned ports
closed, with evidence and memory unchanged. No additional robot commands were
sent during this administrative closure.

## Separate NV result

NV environment 12 retains its successful release-recovery diagnostic and its
failed normal campaign. The latter stopped after orange lost bilateral support
during transport; the remaining four objects were not attempted. Its planned
reset completed before administrative closure. A fresh read-only capture then
confirmed the recorded home, open jaws and reset object poses. The
[shutdown receipt](evidence/local-rtx-validation-20261001/nv12-administrative-shutdown.json)
records closure of only that run's bridge and mapper, without additional motion.

The [offline orientation review](evidence/local-rtx-validation-20261001/nv12-orientation-review.json)
found abrupt physical loss, while the mechanical trigger remains unresolved.
Preserving the held orientation is feasible for the measured IK endpoints, but
joint interpolation still changes orientation between them. This is a candidate
for further investigation, not evidence that changing orientation prevents slip.
The carry guard stops subsequent commands after lost or unavailable attachment;
it does not repair mechanical stability.

NV environment 13 used frozen source `f9cb6b8e`, with contact tracking, body
masks and three 960 × 540 cameras. Its
[passive admission](evidence/local-rtx-validation-20261001/nv13-passive.json)
passed, followed by a single
[read-only truth probe](evidence/local-rtx-validation-20261001/nv13-readiness.json).
The first orange-to-box attempt then
[failed before any actuator command](evidence/local-rtx-validation-20261001/nv13-native.json):
route validation exhausted its existing three-second budget. There were two
state-feedback reads, zero target or jaw commands, and no reset. This did not
exercise the retained-attachment guard during transport.

The [failure analysis](evidence/local-rtx-validation-20261001/nv13-analysis.json)
records 104.046 seconds in localization. Although the independent camera witness
remained fresh, the particular image used by localization had a server capture
age of about 104.112 seconds when its result returned. Its client-local receipt
age, used by the localization fix, was about 103.861 seconds. A missing MobileCLIP file in the
caller's working directory caused a download; its bytes subsequently matched
the expected model hash. This retrospective match does not change the original
startup inventory. The model already existed under `models/`, which is the
normal launcher's MCP working directory.

The route budget began after grasp generation, so the download did not consume
that three-second interval. A later
[offline replay](evidence/local-rtx-validation-20261001/nv13-veto-replay.json)
reproduced all six recorded geometric vetoes: five at closure and one during
descent. The five closure points lie one pixel outside the recorded target mask;
this does not establish their object identity or justify expanding that mask.
The replay does not include the historical ESDF requests, and its local CPU
timings do not assign the original timeout to a particular component.
[Localization freshness](LOCALIZATION_FRESHNESS.md) describes the separate
correction that rejects images expiring during analysis. The
[independent review](evidence/local-rtx-validation-20261001/nv13-native-peer.json)
retains both the native failure and the absence of actuation. The
[preservation manifest](evidence/local-rtx-validation-20261001/nv13-retention.json)
binds 46 original files, independently rehashed against their retained copies.

A [final read-only capture](evidence/local-rtx-validation-20261001/nv13-final-capture.json)
confirmed the preserved initial home pose, open jaws, empty contacts and object
positions within 26 nanometres of their spawns. This observed state does not
represent a reset: none was requested or performed. The
[administrative closure](evidence/local-rtx-validation-20261001/nv13-close.json)
then signaled only the two owned bridge/mapper identities. All three owned
processes exited and both ports closed. The 287 frozen source files and protected
process identities remained intact, with no actuator commands. The
[root review](evidence/local-rtx-validation-20261001/nv13-close-root.json)
independently rehashed the final capture artifacts and retained closure result.
