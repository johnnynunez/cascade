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

The local run includes two software changes:

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
  confirms its tree exactly equals reviewed `ed29a59`.

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
kept perception active. The next diagnostic uses the normal complete launcher
and its single proof session. This removes the extra diagnostic session; its
effect on throughput and physical acceptance still requires measurement.

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
