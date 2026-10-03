# Manipulation and assembly integration

## 2026-10-03 — distance outcome after command completion

Plan: reproduce the PR87 review counterexample on cbe9873 (30 mm admitted
travel followed by 50 mm coasting and quiet rest), then retain post-completion
geometry as veto-only independent evidence. Positive credit remains bounded
by the admission/completion interval. Keep the existing geometry, support,
freshness, stop and cancellation limits; no new physics or GPU run. Validate
signed goals, late-only travel, terminal overshoot/retreat/drift, bad support,
cancelled reads and execution failure with synthetic CPU records. Changes
belong to the independent verifier and regressions, not the policy/driver.

Result: the exact inspected review probe reproduced the 30+50 mm false
confirmation on cbe9873 and refutes it after the fix. Its four other verdicts
are unchanged. Eight signed/rotated overshoot controls fail on the original
source; the final seven-file selection passes 566 cases in 48.88 seconds with
source and four protected stores unchanged. An intermediate 64-pass/1-fail
run retained an overly specific test reason assertion: the existing settle
veto legitimately won reason precedence. Only that assertion was corrected;
it now checks both refusal and the retained geometric veto. Ruff F/E9 and diff
checks pass. No SDK, simulator, GPU, service or shared environment mutation.
Positive credit and all limits are unchanged; the added O(n) outcome pass is
veto-only and limited to walk_distance. Receipt:
`benchmark/results/mobile_distance_outcome_20261003.json`.
## 2026-10-03: passive, calibrated RGB-D without an arm

The configured sensing domain could expose independent mobile state and JPEG
observations, but could not consume a registered RGB-D producer through the
ordinary robot runtime/MCP path. An exact passive catalog probe on `2b1d8e4`
rejects the new provider kind; the same probe and configuration succeed with
the implementation in this branch, without opening a socket or an actuator.

The opt-in `--camera-rgbd` producer now captures the existing static overview
camera's RGB8, metric optical-axis depth, actual USD calibration and rigid
optical-to-world transform. The new source/calibration become part of the
effective model identity. Independent reader-role TCP, a bounded registered
capture cache and `mobile_rgbd` sensing expose it through ordinary MCP. Legacy
RGB replies and RGB-D serialization without extrinsics retain their schemas.
No camera mounted on a robot, tactile hardware, mapping, or navigation is
added. See [the usable configuration and exact limits](docs/OBSERVED_RGBD.md).

Validation: 358 CPU tests passed, nine optional-dependency tests skipped; Ruff
F/E9 and diff checks passed. Tests include real TCP/MCP routing, full 640×480
lossless channels, identity/epoch/calibration mismatch, stale/replay/timeout,
bounded decompression, AOV-reference mismatch, a retained signal during depth
readback and legacy protocol compatibility. A separate OpenUSD-only SDK check
passed camera optics and coordinate transforms without loading Kit or physics.
Its first prototype failure exposed a single time sample overriding the default
despite `ValueMightBeTimeVarying()` returning false; the producer now rejects
any authored time sample. The original failure is retained in local evidence.

Independent read-only review found no remaining material blocker. Render-product
tokens around each AOV are software provenance, not a new native proof of pixel
alignment. The new recipe still needs GPU/native validation and admission;
existing MicroDuck model hashes and frozen voice/video candidates are untouched.
No simulation, provider, GPU or audiovisual process was launched for this work.

Base: `1271d52d09feaa0edb4652d39a9ab21d69e4a686`.

User priority (2026-10-02): implement OVRTX/cuMotion in manipulation and
physical screw manipulation first; address acceptance trial 12 afterward;
update documentation last. The initial documentation-first interpretation
was corrected by the user.

## Capability mapping

| Capability | Skill / implementation | Acceptance |
| --- | --- | --- |
| Ordered integration | isaac-sim-orchestrator | Isolated foundation checks, then combined runtime |
| Planned arm motion | manipulation-ik / motion-generation; cuMotion followup | Native SDK curve executed through normal SafeArm motion with live safety gates |
| Dynamic RGBD | isaac-camera; OVRTX followup | Physics-owned transforms, camera calibration, captured state and frame identity bound to returned pixels |
| Fastening physics | physics-simulation / manipulation-ik | Independently measured fastener rotation and axial advance; seating/torque claimed only when measured |
| Simulation QA | isaac-sim-validator | Focused regression, native runtime evidence, visible output and exact source binding |
| Lessons | skill-distillation | Record corrections and propose reusable updates; no unrequested skill writes |

## Ownership and environment

- Root owns this integration worktree and screw implementation.
- cuMotion and OVRTX agents work in separate `*-runtime-followup` worktrees.
- Original adapter worktrees are still changing in another session; snapshot
  their existing work, never modify or stop their owners.
- Two RTX PRO 6000 GPUs are present. Existing Isaac/GGX/occupancy processes
  belong to earlier sessions. New native checks must use separate owned
  processes and ports, without global termination or dependency changes.
- Trial 12 evidence has been audited read-only. Orange planning failed before
  actuation at the shared 3-second route preflight budget. Green cube passed;
  proof2/campaign were not completed. This is deferred until integration.

## Progress

OVRTX and cuMotion are integrated in the normal arm/skill path. The joint
green-object episode completed grasp, lift, transport, release, return home and
park with unchanged safety gates. Its runtime source was `f2b2186`; compact
receipts, native curve counts and corrected video frames are committed under
`benchmark/results/`. Pink-object failures remain visible, including the final
loaded joint-6 following error. The owned 8731/8732 simulators and OVRTX worker
were closed with process-identity receipts; no elevated payload was released.

The SO-101 Factory experiment completed physical threading, measured contact
at the spacer seat and two seconds of zero-motor retention. Positive, zero-drive,
half-timestep and failed bare-head seating results remain separate. Source and
asset hashes bind the saved solver-state video. Ordinary `turn_screw` remains
explicitly unverified for physical tightening.

Trial 12's historical green success/orange preflight failure and separate
terminal process-binding error are retained. The same-frame route replay saves
about one second by checking identical endpoint vetoes earlier. Both replay
versions pass on this host. The fresh 1280 native follow-up passed that route
stage and subsequently failed its 120-second physical streaming wall budget;
fresh cameras did not imply sufficient simulation throughput. Its owned
simulator was closed. A separate 640 variant retained all original motion,
perception and geometry limits. After read-only exact-label warmup, orange and
green both passed physical placement, home and reset with unchanged source.
The earlier cold-perception rejection is retained. Skill times of 483.72 and
438.83 seconds exceed the host's 300-second limit; this direct-skill diagnostic
does not establish real-host/MCP proof2 or full campaign acceptance.

The final production source `c07d922` passed 3,797 tests, 48 skipped and four
deselected. The later fixture-import lint cleanup passed 12 trajectory tests;
changed Python files pass Ruff F/E9. Documentation was reconciled after the
native manipulation and assembly work, as requested.

PR #65 publishes implementation and compact evidence. Its first three-platform
CI run passed the main tests and found two portable-bundle inventory failures.
The correction adds the OVRTX helper/identity files and the optional cuMotion
XRDF/provenance. All 62 packaging tests pass locally on Python 3.12, including
isolated profile resolution and missing-dependency rejection; no SDK is loaded.

Final review found that the new retained-payload shutdown guard also skipped
Feetech parking before its torque-off disconnect. The correction scopes that
guard per arm to explicit `disconnect_preserves_drive_state` capability, which
only Isaac currently declares. Hardware keeps its prior controlled parking,
and LazyArm does not connect when queried. All 119 focused shutdown/runtime
checks pass, including the actual Feetech driver against an in-memory register
bus, loaded Isaac, standby lazy arms and both mixed-rig orders.

## Reusable lessons and skill proposals (dry run)

This records the `skill-distillation` pass in project documentation. No installed
skill-library files were modified, and the proposals are not promoted as applied
skill guidance.

- User priority is a dependency order: when corrected from documentation to
  implementation, complete and measure the implementation before reconciling
  the status report. Update the canonical report around final outcomes.
- Freeze loaded production source during a native run and during tests that
  inspect live function source. A shared editable environment does not identify
  the intended worktree; bind imports through its absolute source directory.
- Configure every resolved arm node, not just the top-level profile, when a
  diagnostic retargets its endpoint or park pose. Record effective flags and
  actual gate callbacks; a config key alone did not enable the observed gate.
- A native curve is only a candidate. Preserve independent corridor checks,
  original cancellation generation, final measured start-drift admission and
  retained load state across planning, failure and shutdown.
- A second USD stage inside Kit can still emit global notices affecting the
  physical stage. Export without stage edits and author renderer-only opinions
  in an isolated process; test global notice absence and live tensors.
- Measure contact continuity per physics substep. A contact seen in one rendered
  frame does not prove contact throughout that frame. Final tightening requires
  an identified support contact and a bounded motor-off rest, not just stalled
  rotation and motor effort. Retain zero-drive controls and collision failures.
- Native readiness, prompt-free perception and exact-label perception are
  distinct startup checks. Warm read-only perception in the same runtime;
  preserve expired-frame failures and require the task's own newer captures.
- Record physical time, wall time and target acknowledgement cadence together.
  A live camera and advancing simulation can still be too slow for a motion
  wall deadline. Treat resolution changes as explicit configuration variants.

Proposed library additions, pending a separately requested skill update:

| Owning skill / section | Proposed addition | Canonical project reference |
| --- | --- | --- |
| `isaac-sim-orchestrator/SKILL.md`, Phase 2 — Incremental integration | Freeze source per episode; audit effective resolved configuration and ownership | `docs/MANIPULATION_ASSEMBLY_20261002.md` |
| `physics-simulation/SKILL.md`, Troubleshooting | Distinguish thread-friction stall, real seat contact and per-substep tool interference | `sim/seating_verification.py`, `docs/FACTORY_THREAD_CONTACT.md` |
| `isaac-sim-validator/SKILL.md`, Universal checklist | Bind raw evidence and video to loaded source; separate startup, task and shutdown verdicts | native component receipts and `docs/LOCAL_RTX_VALIDATION.md` |

These proposals contain no embedded implementation and require no change to the
completed project work. Existing specialist references remain authoritative.

## Composable robot foundation — 2 October 2026

Continued from merged MicroDuck and stop-lifecycle fixes (`0a65887`), in an
isolated worktree. Research and concrete boundaries are in
`docs/ROBOT_MODULARITY.md`: capability domains, command endpoint ownership,
passive typed sensing, bounded skill graphs, and optional VAB/Arena adapters.
The real composed MCP software route uses mock arm/base and synthetic IMU;
physical mobile manipulation remains refused pending shared-frame and
whole-body admission. No simulator/policy assets or heavy optional frameworks
were installed into the production environment.

Cross-review caught and fixed graph finalization after stop, undeclared motion
writes, blocked-domain stop fanout, external Arena controller stop/reset,
partial factory cleanup, pending sensor shutdown and graceful arm teardown.
The local full suite retained 5,178 passes, 252 skips, four deselections and
three existing mobile observer read-deadline failures. The five affected cases
passed unchanged in isolation; this is a separate rerun, not a replacement for
the failed receipt. Packaging: 62 passes. See
`docs/evidence/robot-modularity/software-validation.json` for source bindings.
Shared envelope bytes and the absence of active grasp memory were preserved.

Native Arena/VAB rollouts, tactile device calibration, whole-body humanoid
control and hosted speech deployment remain separate measured integrations.
MicroDuck's retained gait/turn findings are unchanged by software composition.

The requested HomeBody review is in `docs/HOMEBODY_COMPARISON.md`, with pinned
primary sources and separately inspected component contracts. It proposes
grounded memory and transform replay first, followed by local correction,
actuator health and later whole-body admission. The inspected HomeBody
repository has not released its robot implementation; this review does not
claim a port or measured physical integration.

Initial PR #69 CI passed Linux x86, minimal install and browser checks, but
failed one ARM freshness case and two macOS healthy late-read cases. A matching
macOS failure predates this refactor. A bounded injected-GC comparison motivated
isolation in two healthy-channel test functions; 192 focused cases pass without
relaxing any runtime gate. macOS gets a compact assertion diagnostic. Actual CI
root causes remain unproven; the initial failures are retained in
`docs/evidence/robot-modularity/ci-followup.json`.

CI on `3dae06a` passed Linux x86/ARM, minimal install and browser checks, but
macOS rejected two held late replies as stale. The fixture now yields to an
absolute target and retains its detached wire snapshot without a redundant
copy in that timed path. A controlled 150 ms timer delay reproduces the old
failure; it does not identify the actual CI scheduler/GC cause. Six additional
tests exercise the real sampler with module-local logical clocks and distinguish
late fresh, faulty and stale returns. The focused suite passed 198 cases;
production sources, limits and shared learned stores remain unchanged. Evidence:
`docs/evidence/robot-modularity/macos-late-read-fix.json`.

## Conversation admission foundation (2026-10-02)

The new conversation service needs delayed provider intents bound to the runtime
cancellation generation. `RobotRuntime.execute(expected_generation=...)` now
checks that token under its admission lock; stop followed by an explicit operator
reset cannot admit an older episode. Token-bound reset is refused; priority stop
remains available independently. Skill graphs use the same atomic boundary.
Deterministic regressions force stop/reset between graph validation and dispatch.
This is a software admission fix, with no new physical acceptance claim.

## Spatial providers and navigation replay — 2 October 2026

Feature-to-skill map before foundations: `isaac-sim-orchestrator` and
`isaac-sim-workflow` define evidence and admission boundaries;
`navigation-primitives` informs conservative footprints and grid planning;
`spatial-reasoning` supplies explicit transform conventions. This first increment
uses a pure Python read-only replay: no simulator stage is edited and no robot
is actuated. Unknown cells, transform epochs and capture/calibration provenance
must survive through real composed-runtime/MCP reads and plans. Native navigation
requires a separately admitted controller and fresh localization; a path is not
physical acceptance. Shared learned stores and other agents' processes remain
untouched.

The spatial increment implements a bounded source/clock/epoch frame tree,
source-bound landmark observations, immutable occupancy capture history, planar
range integration and conservative cardinal route planning. The synthetic
`spatial_replay` profile exercises these through normal runtime and real stdio
MCP without actuator resources. Independent review caught subcell-ray
 over-clearing and whole-map rejuvenation after one partial scan; both have
specific regressions. Relative transforms also no longer depend on an unused
common ancestor's age. Frozen source passed 53 focused cases with learned stores
unchanged. See `docs/SPATIAL_PROVIDERS.md` and its evidence receipt. This does not
admit SLAM, native navigation, world-frame grasps or whole-body manipulation.

CI at `735591e` passed both Linux architectures, minimal install and browser
checks; macOS retained one healthy-late failure. Its held response was already
139 ms old at capture; the 78 ms transport hold correctly exceeded the unchanged
200 ms age limit. The test now selects a genuinely recent published packet before
its deliberate hold, returning older replies normally without restamping them.
A 170 ms publisher-pause comparison reproduces old-fail/new-pass. An earlier
190 ms experiment made the channel itself stale and both variants failed; that
receipt is retained. The precise hosted producer scheduling cause remains unknown.
500 focused regressions passed, including stop, MCP, runtime and graph admission;
source and protected memory identities stayed unchanged. See
`docs/evidence/robot-modularity/macos-fresh-capture-fix.json`.

The same optional admission boundary now accepts a **local monotonic** deadline,
checked under the lock both at admission and immediately before domain dispatch.
Graph deadlines use this boundary. Tests deterministically expire the deadline
during validation and between admission and domain lookup; no domain call occurs.
Priority stop ignores an expired episode deadline. This coordinator check is not
a real-time actuator guarantee: domain owners still enforce backend leases and
last-moment cancellation. Provider/browser timestamps cannot supply this deadline.

Root review integrated the generation/deadline fix from PR #70, preserving both
passive embodiment metadata and atomic dispatch checks. The frozen integrated
source passed 100 focused runtime/graph/embodiment/sensing/MCP cases; protected
learned stores were unchanged. This follows the author's 193-case final suite.

Integration with PR #70 passed 83 focused spatial/runtime/graph/MCP cases on
frozen source, preserving all generation/deadline checks and learned stores.

## macOS TCP fixture phase correction (2026-10-02)

PR72/73 macOS logs exposed two test assumptions: a healthy transport might
never enter an 83 ms selection window with a <=20 ms old packet, and the
independent sampler might miss a published movement endpoint. The real TCP
regression now injects a fault at the first post-finish request and checks the
actual decoded packet. Six unchanged deterministic actual-sampler cases retain
precise late-fresh/fault/stale discrimination. Movement expectations use the
known scripted positions at the independently observed interval endpoints.

No production source, limits or ACK semantics changed. A controlled transport
delay reproduces four old failures; the corresponding revised six cases pass.
The complete three affected test files pass 342 cases. This identifies fixture
defects, not the exact scheduler/GC cause on the CI machine. Logs, old/new hashes
and retained results are indexed in
`docs/evidence/robot-modularity/macos-causal-fixtures.json`.

## Turn geometry fixture separation (2026-10-02)

PR77 Linux CI returned unverified for one scripted overshoot case; the original
assertion did not include its reason, so the exact hosted cause is unknown.
Eight known quaternion-path cases now exercise measurement and intent decisions
directly: signed rotation across the ±pi cut, no effect, wrong sign, matching
turn and overshoot with zero gyro. They no longer assume that a threaded sampler
will observe every scripted endpoint inside its wall-clock admission budget.
These are geometry unit cases, not additional full-window acceptance evidence.
Actual sampler, TCP, freshness and stop tests remain, and all production limits
are unchanged. The three affected/regression files passed 400 cases in 33.71s,
with source and protected stores unchanged. A preceding command used a missing
filename and ran no tests; it is retained separately. Receipt:
`docs/evidence/robot-modularity/yaw-geometry-fixture.json`.

## MicroDuck SDK callback signal consumption (2026-10-02)

A native locomotion campaign required forced termination after Kit caught the
`SignalRequest` raised by SIGTERM inside its callback. The recorded scalar
signal survived, but the native runner did not check it after callback return.
The CLI now uses persistent signal checkpoints after SDK initialization and
capture, at loop boundaries, and before the stepper commits inference or submits
new actuator/solver work. Existing interactive one-shot handling is unchanged;
cleanup still runs outside the handler. This cannot interrupt an uncooperative
native call, so the external owned-process supervisor remains necessary.

Feature-to-skill map: `isaac-sim-troubleshooting` for the observed native hang,
`isaac-sim-orchestrator` for owned process/source/closure evidence, and
`isaac-sim-validator` for regression checks before native follow-up.

Validation: 175 focused lifecycle/stepper/CLI tests passed. A retained old-bridge
control failed all six new SIGINT/SIGTERM callback-consumption cases, showing
additional solver work after the signal. The fixed six cases passed again with
an explicit assertion that the SDK fixture consumed the signal. All test runs
retained unchanged source snapshots and protected learned-store hashes. This
is software evidence only; native shutdown validation is a separate follow-up.
Receipt: `docs/evidence/robot-modularity/microduck-signal-checkpoint.json`.

## Inner Kit capture and initialization fences (2026-10-02)

Reviewed the user-supplied Hermes packet and verified the exact baseline, patch
candidate and regression-test hashes before integration. Its capture regression
found 15 additional cold updates or two warm updates plus RGB readback after
a callback consumed the signal. Capture now rechecks immediately between native
calls. The backend also checks camera-authoring synchronization, initialization
phase boundaries, and before/after `play(commit=True)` and model preparation.
No checkpoint is inserted into teardown; the SDK's already-running native call
remains outside Python's interruption guarantees.

The integrated software suite passed 202 cases; five USD cases skipped because
that interpreter has no pxr. The ten Hermes render cases passed against the
actual worktree. Eight additional OS-signal cases cover authoring synchronization
and actual `_initialize` orchestration through camera, scene export and play.
A retained old-source control reached stage acquisition after consuming signals;
all six initialization cases failed instead of unwinding. An earlier red-harness
import error is retained without regression credit. Source and protected memory
were unchanged. Native Kit closure is still a separate required replay.

Hashes and source-bound results: `docs/evidence/robot-modularity/microduck-inner-signal-fences.json`.

## Shutdown PR full-suite environment check (2026-10-02)

The isolated source at41307b8 ran the complete test directory: 5,073 passed,
252 skipped and four deselected. All132 failures were the same missing pinned
kitchen artwork prerequisite in the fresh worktree. Installed those eight
release files from the already verified local bundle, then normal offline
`kitchen_assets.py --check` passed. All139 tests in the five affected files then
passed. Production code was unchanged; only two native-closure documentation
files were added between runs. The initial failed full run remains preserved,
not labelled as one green full run. Frozen source and protected stores remained
unchanged in both executions. Detailed source/artifact records:
`docs/evidence/robot-modularity/native-shutdown-suite.json`.

## Robot-agnostic conversation implementation (2026-10-02)

Implemented an optional local HTTP/browser/PCM vertical with lazy `aiohttp`,
HF GA Realtime provider negotiation and `MediaIO`/provider boundaries. The
configured robot catalog supplies only explicitly allowed typed tools. Complete
provider responses produce session/robot/request/generation/deadline-bound
intents into RobotRuntime; no raw joint interface or Reachy identity is added.
Cancellation, tool budget, media backpressure, origin/authentication and process
ownership are explicit. Timeout preserves a pending worker instead of inventing
thread cancellation or physical success. CLI run stores are private.

Independent review by the embodiment agent found and fixed: stop depending on
media flush; a delayed `response.created` after barge-in before response birth;
and simultaneous session POST requests overwriting the sole provider owner.
Regression tests reproduce each interleaving with events/real local sockets.
The entire connection retains its original runtime generation, so a late new
response cannot refresh command authority after stop plus operator reset.

The upstream wire contract was read at speech-to-speech revision
411399d34555b2169823a6eaeb7f8ff192db89db, including the actual response handler's
response_id/output_index fields and session acknowledgement. The browser and
CASCADE code are original implementation. No provider allocator, paid service,
model download or audio device was contacted by the software validation.

Validation: 135 targeted conversation/runtime/graph/MCP/sensing tests passed;
three optional-stack cases skipped in the task-private test environment.
Conversation alone has 41 passing cases, including Node worklet execution.
Ruff, uv lock consistency, documentation links and git diff checks pass.

Root review on the integrated PR #70 source passed 97 focused conversation,
runtime, graph and MCP tests, with three optional-stack skips. Source and shared
learned stores were unchanged; all 41 conversation cases passed. The provider
inference experiment is separate from these loopback protocol results.

## MicroDuck conversation composition (2026-10-02)

Added explicit robot profiles for the existing MicroDuck mock and native base
domains. Both expose only base resources, preserving each backend's canonical
robot identifier. Robot identity validation now uses the same bounded identifier
contract as resource descriptors; profile filenames remain strict slugs. The
native profile refuses to construct its bridge client without an explicit model
identity pin. It does not mark a gait, provider, or physical episode admitted.

Validation: 71 profile/conversation/runtime/MCP/config tests passed, with three
optional-stack skips. The mock profile reads actual mock base state and exposes
no arm; the native profile rejects missing identity before bridge I/O. Frozen
source and protected learned stores stayed unchanged. An earlier failed identity
mismatch test and an incorrect test assumption about validation timing are
retained in the task's `microduck-profiles-02` results; the final run is
`microduck-profiles-03`.
## Owned native speech-provider validation (2026-10-02)

Connected the committed conversation implementation to a private local HF
speech-to-speech server on the authorized GPU1, with two CPU threads, a Torch
allocator ceiling below 12 GiB and a 900-second supervisor watchdog. The first
Whisper-tiny/SmolLM2 recipe produced real audio and a separate typed synthetic
IMU read, while retaining a failed spoken read and malformed model output.
Onset/sample-rate audit confirmed exact sent waveform bytes and preserved the
uncertainty: full uncropped tiny ASR also misrecognized the phrases.

The second Whisper-base/Qwen3-1.7B recipe completed three actual synthetic-audio
STT/LLM/TTS paths. Greeting and sensor-catalog intents succeeded; requested
sensor reading failed because the model only listed sensors and falsely narrated
data retrieval. Its authoritative tool receipt records the mismatch. Upstream
one-second streamer warmup failed first; the successful isolated provider uses a
recorded one-line 10-second transport-wait patch, with CASCADE sources unchanged.
Both owned servers closed, reaped and released their ports. No motion, audio
hardware, physical acceptance or public/paid deployment is claimed.

See `docs/CONVERSATION_NATIVE_20261002.md` and the hash-bound compact receipt
`benchmark/results/conversation_native_20261002.json`. MMS voice/audio artifacts
remain local under its model-card noncommercial license. No shared environment,
learned store or foreign service was modified.

Native audio timestamps exposed a frontend issue: ordinary provider bursts
exceeded the original two-second playback limit. Added a bounded 15-second,
512-object PCM queue with only two seconds scheduled ahead. Flush removes both
queued and scheduled audio, and local revisions suppress pending context-resume
audio after stop or reconnect. The actual stop button remains independent of
audio resume. CPU Node replay covers all three native timestamp/length traces
using generated samples; negative controls reproduce the original rejections.
No native process or audio device is needed for this follow-up.
Independent review found an adjacent pending-microphone-permission race. Capture
now checks session/socket/revision after each asynchronous setup boundary and
immediately stops tracks returned to a superseded session. Six Node controls
stop or disconnect during context resume, worklet loading and permission.
Root review additionally identified callbacks from superseded WebSockets and
pending session creation. Socket/session/revision binding now prevents stale
callbacks from muting or stopping a replacement connection. A stopped pending
POST is cleaned up before another connect can begin; a reset superseded by stop
cannot subsequently issue a reset. Three event-controlled Node cases cover
these interleavings without changing backend stop/admission behavior.
Final focused validation: 52 conversation tests passed, Ruff passed, and source
hashes were unchanged during the run. The playback receipt binds the exact
JavaScript, timing-only fixture, protocol tests and captured pytest output.

## Reset request generation admission (2026-10-02)

Root review found that frontend cancellation alone could not fence an already
sent reset POST arriving after a newer stop. `RobotRuntime.reset_stop` now accepts
an optional expected generation, checked atomically under its existing gate
before any domain reset. Legacy direct callers may omit it. The authenticated
conversation reset route requires an exact nonnegative JSON integer and passes
it to that boundary. Browser reset reads status after closing its old session,
rechecks its local revision, then sends that observed generation.

Real HTTP regressions hold a partial reset body across a newer stop (zero domain
resets) and hold a previously successful reply across a newer stop (the stop
remains latched). A runtime gate interleaving verifies the comparison happens
after acquiring the lock; stop during an admitted reset still relatches domains.
Browser controls cover stop while the generation read or reset reply is pending.
No speech provider, hardware or physical source evidence was modified.
Validation: 128 runtime/graph/MCP/conversation tests passed; three optional-stack
cases skipped. Conversation files pass full Ruff; runtime/test-runtime pass the
repository's F/E9 check (unrelated existing full-rule findings remain). Git diff
checks pass.

## Actual Chromium conversation path (2026-10-02)

Frozen source 4d8c350 passed an owned Chromium 153 AudioWorklet/AudioContext
exercise through real HTTP/WebSocket and composed synthetic sensor runtime.
Ten fake-microphone PCM chunks reached the provider; list_sensors returned the
actual synthetic IMU descriptor. A 6.784 s burst drained 68 buffers without queue
failure and stayed within the 2 s scheduling horizon. Stop flushed a second 4 s
burst, stopped scheduled sources and latched the runtime; reset/reconnect/
disconnect then passed. No page errors; both owned services closed and source/
protected stores stayed unchanged. A prior probe failed because its Playwright
string wait violated CSP; only the probe was corrected, preserving shipped CSP.
This is synthetic-provider browser integration, not native speech inference or
physical microphone/speaker acceptance. Root regression on the same source:
130 passed, 3 skipped in 2.86 s. Evidence: docs/evidence/robot-modularity/conversation-chromium.json.

## Native provider plus Chromium (2026-10-02)

The frozen 4d8c350 gateway completed an actual CPU Whisper/Qwen/Kokoro greeting
through Chromium fake microphone/AudioWorklet and AudioContext, then Stop.
ASR matched input, reply spoke a greeting, all 70,656 output samples drained at
24kHz, and owned processes closed with source/protected stores unchanged.
Response 68.923 s includes 61.580 s CPU LLM time, longer than 60 s tool admission; no
tool/actuator call occurred. This limitation and the earlier failed sensor-read
intent remain explicit. Native provider log audio 2.20 s counts input, not output.
Receipt: docs/evidence/robot-modularity/conversation-native-chromium.json.

## Inert RPC geometry fixture (2026-10-02)

The composed suite at d09489b retained 5,507 passes, one failure, 252 skips and
four deselections. The inert-actor case correctly returned `unverified` after
`missing_state` on observation attempt 169; its geometry assertion required a
healthy channel and `refuted`. The original run did not log reader errors or GC
activity, so it does not establish why that observation was lost.

Apply the existing `healthy_episode_gc` fixture to this test only. Automatic
cyclic collection is moved outside the bounded software TCP episode; original
GC mode is restored. Clocks, limits, production code and assertions are unchanged.
An external controlled automatic-GC pause reproduced `missing_state` in the
original test; the patched test kept zero measured post-admission displacement
and returned `refuted`. A separately injected TCP delay still returned
`unverified` with GC disabled. These controls establish fixture behavior, not
the unrecorded cause of the original suite failure or physical robot validation.


### 2026-10-02 — preserve utterance authority across delayed VAD and continuations

Native Kokoro greeting traces showed that the pinned provider emits
`speech_stopped` late, after much of model inference. The previous session
started its tool deadline there, admitted unsolicited `response.created`, and
renewed deadlines for tool continuations. Two deterministic replays of the exact
native event payloads against archived 80d0122 actually admitted an injected
expired readonly tool; both now reject it at the real domain/runtime boundary.
The original greetings requested no tool and remain speech-only evidence.

InputContext is immutable and bound at local text submission or receipt of a
unique `speech_started` item. Stopped events only confirm that item. Text and
tool-continuation requests carry a nonce echoed in response metadata; every
continuation retains its original turn, runtime generation and deadline.
Ambiguous automatic responses after cancellation cannot reacquire authority;
the UI mutes/disables input and asks for explicit disconnect/reconnect. This
revocation sends no robot stop in speech_only mode. Initial idle speech remains
benign. The source does not pretend the pinned server honors manual VAD
create_response=False, nor equate provider event receipt with microphone age.

104 conversation/hosting CPU tests pass, including real local WebSockets/HTTP,
old/stale/replayed IDs, nonce mismatch, two onsets before old stopped/created,
operator-stop/disconnect during flush, text-send cancellation, browser status
and unchanged earlier safety cases. An isolated actual pinned HF handler
(no models, network or inference) emitted exact nonce-echo JSON bytes which
were admitted by a local WebSocket regression. Ruff and git diff checks pass.
Compact source/log/fixture hashes are in
benchmark/results/conversation_input_origin_20261002.json. All changes remain
local while the integration coordinator owns GitHub publication.


### 2026-10-03 — Priority stop follow-up on the coordinator integration

Applied the reviewed aggregate stop correction from 4c7b4d7 to the existing
92aa1e5 + support-flight fixture + input-origin composition, without importing
unpublished gait or provider-hosting features. Declared stop tools retain their
authority checks and bypass an outstanding action; a reserved worker prevents
trace IO from starving stop delivery. Cancelled consumers retain one trace
obligation. Close reports pending delivery/action/records, and terminal close
is idempotent without presenting its old ACK as a new stop.

The production diff applied unchanged. The synthetic Owner in the new stop
fixture uses already-admitted walk_velocity instead of unpublished
walk_distance; all priority, cancellation, identity, deadline and closure
assertions are unchanged. Source-specific verification will be recorded in the
external LOCAL_MERGE_REVIEW packet. No GitHub write or native run was performed.

## Adjacent staged stop before provider output (2026-10-03)

When a completed model response staged an operation followed by a declared stop,
Session sent the operation's result to the provider before dispatching that stop.
A retained send could therefore delay a stop that the model had already requested.
Session now retains results only while the next staged tool is declared `effect=stop`.
It dispatches each original call in its existing order through ConversationDomain,
with the same input context, request ID, generation and deadline, then emits the
retained function outputs in their original order. It does not synthesize a stop,
move one ahead of an unfinished operation, or skip any admission gate. An ordinary
sequence without an adjacent stop keeps its prior dispatch/output ordering.

The exact final regression fails on isolated source 4c7b4d7 because the synthetic
owner's stop event is unset when a held provider send starts; it passes on this
change before that send is released. Seven further controls cover preceding
operation completion, expiry, catalog/generation changes, context cancellation,
ordinary sequencing and a stop after an intervening operation. The focused
conversation/priority/composed-runtime suite passes 158 tests in 3.85 s; Ruff and
whitespace checks pass. Source/store snapshots and causal logs are retained
outside the repo in CONVERSATION_STOP_ORDER/evidence. No model inference, browser,
simulator, physical actuation or publication was performed. Demo snapshots remain
unchanged; this is a software dispatch-order correction, not physical stop proof.

### 2026-10-03 — Bounded speech view of observed tool outcomes

A retained native movement result is 1,836,087 JSON bytes: the prior conversation
path replaced it with a 92-byte transport error. Added a bounded view for traced
results, omitting only known image/sample/evidence attachments with SHA256
references. Retained verdicts, metrics, IDs, bindings and ACKs are unchanged.
Small outputs keep exact serialization; absent trace or oversized remaining
fields keep the explicit failure. Encoding runs outside the audio event loop,
with cancellation rechecked before send and adjacent stop already dispatched.

115 CPU checks passed in 1.26 s; 1,611 source inputs and protected stores were
unchanged. Same-input causal replay through the actual session and synthetic
owner fails on the original source and passes with a 20,495-byte view; full
traces remain exact and the synthetic task stays unverified. Independent
read-only review and a separate formatter control passed; Ruff F/E9 and diff
checks are clean. No LLM, TTS, native motion or audio was executed. This does
not repair or explain the video's absent output speech; supervisor stop can
independently revoke that session. Publication/merge freeze remains active.

### 2026-10-03 — Planned passive generalized joint observations

Work is isolated from source `5f50b8b` in GENERALIZED_JOINTS. The bounded feature
is an opt-in version-2 structural declaration for internal planar, spherical
and floating joints, plus immutable observations with separate configuration
and velocity dimensions. Version-1 wire values and declaration digests remain
unchanged. New joints are passive declarations only; scalar transmissions,
legacy angular bindings and legacy physical arm control cannot adopt them.

Implementation order: explicit coordinate/frame/unit contracts; payload and
structural binding; a synthetic observation-only profile through ordinary
runtime/MCP; adversarial CPU regressions and documentation. No new movement
skill, driver, pose conversion, physical admission or GPU run is in scope.
The resulting contract includes mandatory child-relative-to-parent motion,
explicit Hamilton wxyz orientation, work-dual effort, and exact attachment/digest
binding. The fixed-root fixture still refuses a legacy physical arm for each of
planar, spherical and floating joints. New sensor reads retain the captured
envelope and never declare a physical producer or controller.

Validation after the concurrent kitchen episode: 177 CPU tests passed in 4.73 s
across the new contracts/actual stdio MCP and eight affected regression files;
Ruff F/E9 and diff checks passed. The two version-1 wire/digest fixtures were
captured from unmodified 5f50b8b, not regenerated by the new implementation.
Initial development failures are retained: malformed YAML indentation (43 failed,
6 passed), then two integration tests attempting a repeated synthetic capture
(2 failed, 47 passed). Fixes corrected YAML and waited one existing capture period;
freshness/replay gates were unchanged. Shared envelope SHA-256 still matches the
protected baseline and grasp memory remains absent. No GPU, native robot or
hardware admission was tested. See benchmark/results/generalized_joints_20261003.json.
# 2026-10-03 — Retained RGB-D to spatial observations

- In an isolated clone based on `fb750e9`, reproduced the missing composed
  sensor-to-spatial path: only synthetic replay profiles were accepted.
- Added explicit sensor-domain binding, exact immutable capture fan-out with
  age revalidation, and ordinary MCP pixel-to-surface annotations through
  `FrameTree` and `SpatialMemory`. Labels are caller annotations; confidence and
  geometric uncertainty remain unknown. No grid, localization or motion tool.
- Preserved producer replay watermarks, original clocks/model/calibration,
  legacy spatial replay, private stores and frozen native/voice candidates.
- Focused CPU suite: 193 passed, four optional skips (one missing OpenUSD and
  three SO-101/Pinocchio checks). New real TCP/MCP path executed successfully.
  No native/GPU/hardware validation or publication performed.

# 2026-10-03 — Explicit native RGB-D pixel centers

- A separate bounded native probe on source `5362c7a` passed capture, bounded
  decode, four exact TCP/Hub/MCP joins and natural closure. Offline geometry
  found a half-pixel contract error: K used raster coordinates while spatial
  projection used integer centers. No geometry admission is credited to that
  original episode; its packets, model and sources remain unchanged.
- Calibration v2 now declares `[0.5, 0.5]` in its content/model identity and
  sensor payload. Spatial annotations consume that field without shifting K,
  changing pixels, refreshing capture time or granting motion. Native v1 is
  rejected; generic legacy payload bytes and integer-center semantics remain.
- Identical CPU consumer probe: baseline `b6fd84b` fails the retained ground
  geometry check (maximum Z residual 0.5118 mm), candidate passes (7.162e-8 m).
  Legacy calibrated payload serialization has the same SHA in both checkouts.
  Focused suite: 122 passed, six OpenUSD-import skips. Separate installed
  OpenUSD 0.25.5 CPU probe passed with unchanged K/pose and explicit v2 metadata.
- Ruff F/E9 and diff checks pass; protected stores unchanged. Evidence:
  `benchmark/results/rgbd_pixel_centers_20261003.json`. No new native launch,
  new model admission or publication; a new native recipe still needs admission.
# 2026-10-03 — CPU independent planar RGB-D reference

- Added benchmark-only visual checkerboard authoring, image-only orientation/corner detection, fixed-split homography and exact-pixel geometry comparison. The oracle never consumes K, extrinsics, depth, producer pixel offsets or raycasts.
- Full annotation checks bind original RGB/capture/epoch/model/calibration/clock and recorded ages; no fresh acquisition, authority or runtime/native source changes. Corrected planning assumption: native02 explicitly used 2 s age/read limits.
- Normal conftest: 36 PASS, 1 SKIP (`pxr` absent), 10.34 s, 1596 source inputs and protected store unchanged. Separate OpenUSD 0.25.5 CPU authoring PASS on final source: 53 visual meshes, no physics schema additions, 239 existing prims unchanged. No Kit/GPU/model inference.
- Synthetic image/consumer controls detect focal, translation, units, offset and axis errors; stale/foreign/mutated data is refused. Actual RTX board visibility, native model/collider invariance and live geometry remain pending a reviewed future recipe. Evidence: `benchmark/results/rgbd_planar_reference_cpu_20261003.json`; external source-bound logs in sibling `RGBD_XY_VALIDATION/evidence/`.

# 2026-10-03 — Prepared native planar-reference wiring (no launch)

- Added a benchmark entrypoint through the existing bridge/backend factory. The visual-only board precedes the original camera/export/play sequence; source and actual scene enter the new canonical model identity. Strict existing-stage and native recipe comparisons refuse any body/shape/collider/configuration change.
- Added the 17-point consumer over the real in-process MCP handler, retaining one exact SensorHub capture and all projection receipts; image-only homography never reads camera K/T/depth. A complete fresh set is required, with the unchanged 2 s age/read bounds.
- CPU: 71 PASS, 2 OpenUSD-import SKIP, 15.49 s; 1601 inputs/protected store unchanged. Separate installed OpenUSD 0.25.5 CPU probe: 254 existing prims unchanged, 53 visual meshes, time-sample/collision/points mutation controls rejected. No Kit/GPU/native launch.
- Development caught non-content repr addresses for USD list operations and legitimate infinite joint bounds; content serialization now preserves both explicitly. Previous native01/02 sources/packets remain untouched. External campaign/owner/check-only recipe lives in sibling RGBD_XY_NATIVE; no execution approval or model admission follows from CPU preparation.
- Portable CPU receipt: benchmark/results/rgbd_planar_native_preparation_20261003.json.

# 2026-10-03 — Explicit binary planar reference, CPU candidate

- Added schema-3 Ground bitmap with fixed ArUco IDs and exact 0.5 mm texel
  geometry; colored/default recipes and PNG remain intact. Oracle consumes
  RGB plus declared board only, preserving 35 corners and the 18/17 split.
- Actual CPU detector controls reject missing/duplicate/foreign/reordered
  geometry, corrupted bits, occlusion and insufficient pixel/margin support.
  A local contrast gate rejects an occluded corner that OpenCV otherwise
  inferred despite passing the unchanged fit residuals. No calibrated
  confidence, semantic object identity or navigation authority is introduced.
- Normal conftest: 117 PASS / 5 explicit OpenUSD-import SKIP, 24.90 s.
  Separate installed OpenUSD CPU: 2 PASS / 1.21 s, preserved original Ground
  geometry and physics binding for both variants. 1617 source inputs and
  protected stores remained identical during the checks; Ruff F/E9 passes.
- Review follow-up changes only the corner test: retain OpenCV5.0's actual
  inferred grid and inject it to test the contrast guard; permit newer real
  detectors to reject earlier. Final binary focal: 42 PASS / 3.95 s, 1619
  inputs/stores unchanged. Runtime and detector bytes remained unchanged.
- Existing real MCP handler/Hub/spatial fan-out verified on declared synthetic
  pixels, including wrong X/Y calibration, stale capture and extra-RPC vetoes.
  Authoring uses retained K/T only to forecast placement (37.56 px margin,
  2.64 px minimum cell); it is separate from the independent RGB oracle.
- No native entrypoint, GPU launch, new physical/geometry admission or remote
  publication. Previous native Ground FAIL remains unchanged. See
  `docs/RGBD_BINARY_REFERENCE.md` and its compact receipt for scope and pins.


# 2026-10-03 — Binary native entrypoint preparation, CPU only

- Added explicit immutable/authenticated consumer descriptor and benchmark
  entrypoint, retaining the original guarded Ground factory and complete native
  comparator. Consumer5.0 and SDK author4.14 are separately identified; exact
  dictionary/PNG authoring does not claim cross-version detector execution.
- Consumer validates its own actual version/package code/parameters before
  detection. New source/bitmap/declaration and actual USD metadata enter the new
  model identity. Historical frames, model pins and colored entrypoint remain.
- CPU:152 PASS/6 missing-pxr SKIP in27.07s; installed OpenUSD CPU3 PASS/3.01s.
  1624 inputs/protected stores unchanged. External campaign five methods cover
  seven failure subcases plus module/codebook/import drift and exact authoring.
- Passive recording diagnostics preserve original two bootstrap solves and
  all80 observed rows, epoch/model/support-channel/fault alarms. Known-empty
  contacts never imply physical support/rest. No external actuation/reset client.
- No Kit/GPU/renderer/native campaign launched. External final plan/check-only
  remains preparation subject to root review and a coordinated resource window.
  See docs/RGBD_BINARY_REFERENCE.md and its new CPU preparation receipt.


# 2026-10-03 — Exact recorded-clock guard correction (offline only)

- Preserved binary native01 global FAIL and all original bytes. Actual dt is
  exactly0.004999999888241291, matching both baseline and the producer's existing
  float32(5ms) bootstrap guard. Python-double0.005 equality was the audit defect.
- Offline helper now requires the authenticated reference recipe and both exact
  comparisons; neighboring doubles, nominal0.005, foreign/missing baseline,
  nonfinite/bool values reject. No widened clock/state/verdict thresholds.
- 21 CPU PASS/.24s;1625inputs/stores intact. Same80 retained rows reproduce the
  old ValueError and pass the new component audit;8known-empty contact rows
  remain explicit, without measured-support/balance claim. Detector failure,
  overallFAIL, native model identity and all original captures remain unchanged.
- No Kit/GPU/service/native retry or publication. Receipt and raw causal outputs
  live in RGBD_NATIVE_CLOCK_AUDIT; source candidate is separate from52a4dc0.


# 2026-10-03 — Explicit binary layout A, CPU candidate

- Added schema4 with the reviewed84mm tags, local centers, world origin and
  yaw−44°. The new PNG uses exact local texel boundaries; actual USD float32 ST
  values are bound explicitly. Historical schema3 assets and entrypoint remain.
- Admission retains all original detector,18/17-fit and metric gates and adds
  a rational whole-tag singular-value lower bound≥4px/cell, computed from
  observed RGB tag corners only. K/T design predictions never enter the oracle.
- Final CPU217PASS/12OpenUSD SKIP in33.77s; separate real OpenUSD CPU5PASS in
  5.21s. Both1634-input inventories and protected stores match before/after.
  Ruff F/E9 and local links pass. An exploratory80% rescale fails unchanged
  held-out residuals and is retained; no universal blur/rescale guarantee.
- CPU authoring preserves physics material and checks postbootstrap appearance,
  ancestors and complete native-property comparisons via explicit doubles.
  No Kit/solver/GPU or native metric acceptance. New model/scene admission is
  pending. See docs/RGBD_BINARY_LAYOUT_A.md and the compact CPU result receipt.

## RGB-D publication boundary — 3 October 2026

Composed14 reviewed RGB-D checkpoints onto main075c08c, excluding unrelated
MuJoCo capture-cost891d6c7. Functional added/deleted lines unchanged; only append
WORKLOG conflicts resolved. General588PASS/18SKIP plus original-USD18PASS gives
606 distinct passing cases. Two optional environment failures remain retained;
no source/test changes, no shared environment edits, no native/GPU admission.
Source/store checks and47-file Ruff F/E9 pass. See docs/RGBD_PUBLICATION.md and
benchmark/results/rgbd_publication_20261003.json.


Composition completed using the reviewed publication equivalents through
15382f3. Only append conflicts in this worklog were resolved; control, verifier
and both mobile runtime files match that checkpoint exactly. Current RGB-D
producer changes merge with the optional solver/observation recipes. Signal,
identity, task/stop, teardown, sensor hub and conversation code remain intact.
No native identity or physical admission is inherited.

Source-bound CPU selections resolve 1,406 distinct cases to PASS: 243 focused
and 1,061 compatibility passes, plus the previously skipped optional cases.
Newton1.6/MJW3.12/Warp1.17 optional BAM tests initially gave 96P/2F because
subprocesses lacked the parent-only PyYAML preload; an external YAML-only path
resolved those two imports, and six OpenUSD cases passed in the same 8P run.
The initial integration fixture's two incorrect field expectations are retained
separately. No product or physical thresholds changed for either correction.
All source/protected-store snapshots match; Ruff F/E9, AST and diff checks pass.
No GPU, Kit, mobile native service or new locomotion experiment was run.
See docs/MICRODUCK_LOCOMOTION_COMPOSITION.md and its evidence receipt for the
commit map, raw logs, scope of the two 12-step CPU BAM fixtures and limitations.

### 2026-10-03 — Separate locomotion publication checkpoint

Create LOCOMOTION_PUBLISH_20261003/cascade from generalized observations base
7e15cfca; import only the fourteen reviewed locomotion checkpoints plus the
aggregate composition tests/documentation. No RGB-D dependency is needed: all
six changed production files match the reviewed 15382f3 checkpoint exactly.
Current conversation/generation/teardown files remain byte-identical to this
publication base. Keep the prior aggregate499 receipt explicitly historical.
CPU checks for this branch follow separately; no GPU or native episode is run.

Publication selection complete: 1,279 PASS/5 OpenUSD SKIP in 117.47 s;
the pinned converter interpreter passed all 112 optional BAM/camera checks,
including those five skipped cases, in 27.67 s. Combined unique JUnit cases:
1,382 PASS, zero remaining failures/skips (not a full-suite claim).
1,602 main inputs and protected stores remained unchanged; optional source and
stores also match. The zero-test invocation with a nonexistent generalized-joint
test name is retained as an invocation error; corrected selection used the two
existing generalized-joint files. No code or threshold change was needed.
Ruff F/E9, AST and diff checks pass. The separate publication receipt records
the branch base, eight exact reviewed source files, nine untouched base files,
raw logs, CPU numerical-fixture scope and pending native identity/admission.


## Explicit alternative walking checkpoint (2026-10-02)

Primary-source comparison found that the pinned official CPU inference recipe
also stalls VelStand below the geometric target during a three-second command;
stiff friction alone is not established as its cause. Keep those failures and
the official default. The alternative rough-walk-e checkpoint is selected only
by an explicit profile, exact byte count/SHA and a reviewed 61-observation,
14-action, 50 Hz contract. Its weight model card declares Apache-2.0 and
simulation-only experience; robot geometry terms remain separate.

The official CPU inference path demonstrated bounded 30 mm feedback episodes
with rough-e, while 50 mm episodes violate the existing heading bound. This is
diagnostic evidence, not native admission. No action deadline, pose/support
threshold, rest window or velocity-tracking claim changes. Native follow-up
must retain the same criteria, actual independent observations and closure.

Feature-to-skill map remains physics-simulation, isaac-sim-robot-navigation and
isaac-sim-validator. Validation: 208 policy/CLI/stepper/native BAM tests passed
with CPU-only Newton 1.6 and the pinned PR source, using a private Warp cache.
The new CLI and source inventory bind the selected checkpoint; weights are not
downloaded or executed by offline admission and are not committed.

## Explicit solver graph experiment (2026-10-02)

The rough-e native trial reached signed distance targets but failed the unchanged
three-second wall rest gate. A separate SDK solver-only CUDA-graph option now
addresses measured per-step compute cost. The reviewed NewtonStage source is
hash-pinned. Capture is enabled only after model preparation and initial HOME;
the SDK warms once, captures its solver work and launches one solve per step.
BAM computation, controller fencing, signal checkpoints, force extraction and
rendering remain outside capture. State/control/contact buffer replacement,
changed capture timestep, silent mode changes and graph replacement fail closed.

No physical/action/rest timeout or acceptance threshold changes. The default
remains uncaptured. Sixteen new buffer/mode regressions and the combined actual
Newton CPU suite passed (242 cases). GPU capture remains an experiment until
source-bound foundation and MCP receipts validate actual one-solve cadence,
force extraction, policy responsiveness and process closure.

## Same-solve read reuse (2026-10-02)

The actual graph profile reduced solver calls from 3.689 to 0.102 seconds over
240 steps; duplicate native state/support reads still consumed 1.512 seconds.
The explicit `--reuse-solved-read` option reuses a detached payload only for the
same completed solve and only with graph-bound state/control/contact buffers.
Every read still checks model, layout, timestep, mode and current clocks. Every
solve, initialization and containment invalidates the cache. Consumers receive
separate copies; no capture/receipt timestamp is refreshed and no new sample is
published by a cache hit. This backend owns its state on one thread and exposes
no pose/reset writes between solves. Actual BAM state checks remain per step.

Thirteen regressions cover changed clocks/support solve, buffer/model/layout/
timestep/mode drift, closure, failed solver invalidation and consumer mutation.
The combined actual Newton CPU suite passed 255 cases. Physical parameters and
all task/rest gates remain unchanged; the next native recipe separately binds
this compute option and must revalidate trace, force channel and gait outcomes.

## MicroDuck locomotion admission continuation — 2 October 2026

Task branch `fix/microduck-locomotion-admission`, based on `735591e`, in the
owned `LOCOMOTION_NEXT/cascade` worktree. The user authorizes continued
implementation and native validation. The target is commanded forward/reverse
walking, turning and independently measured stop through CASCADE, without
relaxing existing criteria or substituting animated/pose-written motion.
Standing, the old gait failures and prior native source recipes remain retained.

Feature-to-skill map (before foundation execution):

| Feature / stage | Skill / source | Planned evidence |
| --- | --- | --- |
| Deliverable and admission contract | `isaac-sim-workflow` | Explicit command matrix, unchanged verifier limits, scoped simulation claims |
| Ordered foundation and integration | `isaac-sim-orchestrator` | Source/asset/model hashes, one-variable comparisons, bounded owned processes |
| Native BAM, articulation and contact | `physics-simulation` | Effective actuator/solver identity and same-solve support; no state animation |
| Existing policy execution | `isaac-sim-robot-navigation` plus audited upstream policy/runtime | Observation/action/command/time parity; gait measured from native state |
| Existing camera and replay capture | `isaac-camera`, `isaac-sim-rendering` when capture changes are needed | Same-episode frames and inspected start/middle/end video |
| Final admission and packaging | `isaac-sim-validator` before delivery | Focused regressions, native task/verifier receipts, teardown and visual review |

Plan: (1) pin the requested awesome-microduck registry and inspect primary
official/benchmark sources; (2) compare current model/policy/actuator contracts
to measured upstream baselines; (3) implement only discrepancies supported by
that evidence, with regression coverage; (4) run isolated native foundations
and then actual MCP command admission, retaining failures; (5) record exact
scope, uncertainty and proposed skill lessons. No generic navigation/SLAM or
voice success is inferred from locomotion work.

Environment safeguards: read existing validated dependencies without modifying
them, or create a separate task environment. Never mutate the collaborator's
MICRODUCK checkout. Foreign GPU services were observed on both GPUs and must
not be stopped or reconfigured. Each new process gets an owned output directory,
private ports/caches, source binding, deadline and birth-bound termination.
Initial protected envelope SHA256 is
`56a2f090c7e34fa2a31467966da033b33c52db438f2ad5f91bf0d0a03e16b1a2`;
shared `grasp_memory.json` is absent. Both conditions must remain unchanged.



## Native distance checkpoint and publication (2026-10-02)

Frozen native source `d9f4766` confirmed two forward and two reverse 30 mm
fresh-start MCP episodes with the configured independent motion/support/rest
checks. A separate priority stop during observed motion cancelled the distance
call, fenced later policy input to zero travel intent and confirmed post-ACK
rest. Balancing targets continue; this is neither instantaneous motor-off nor
general gait admission. The combined reverse ankle-shell contact veto, turn
translation-path failures and earlier velocity/rest/closure failures remain.

Publication uses a separate worktree based on the reviewed native shutdown
branch. All 22 identity-bound native sources and mobile production files remain
byte-identical to the measured tree; newer composed-runtime admission safeguards
from the base are preserved. No native episode ran in the publication worktree.
Corrected contact-test fixtures now initialize the full uncaptured solver
contract and exercise production guards. The resulting 37-file CPU suite passed
1,508 cases with eight explicit dependency/source skips. Separate compatible
USD conversion passed 77 cases with three external-asset skips. F/E9 checks
passed, and successful suites kept source and protected shared stores unchanged.

The initial six fixture failures, 74 missing-OpenUSD setup errors and private
USD wheel overlap/import aborts are retained. `usd-exchange` supplies its own
OpenUSD libraries; the successful private conversion recipe avoids coinstalling
another wheel over those `pxr` paths. No shared dependency environment changed.
Report and raw artifact hashes: `docs/MICRODUCK_DISTANCE_CANDIDATE.md` and
`docs/evidence/robot-modularity/microduck-distance-candidate.json`.

## Locomotion publication: current main composition

Merged GitHub main075c08c through d9aa5a6 without rebase or conflict. Preserved
publication freeze01 and all44 artifacts. Runtime/MCP, distance, sensing and
kitchen-observer selection:332PASS/0SKIP in40.25s;1,612inputs and protected
stores unchanged. Factory is absent from this main; no Factory or native claim.
Receipt:docs/evidence/robot-modularity/microduck-locomotion-main-composition.json.


## 2026-10-03 — retain runtime teardown failures

Task-owned clone `RUNTIME_TEARDOWN_RECEIPT/cascade`, base125e248. A read-only real-host preflight found shutdown_runtime suppressed stage errors and ArmRig/CameraRig only logged member failures, while MCP could exit with a handled143. Implemented structured software-only receipts and per-stage continuation, preserved errors/pending workers and first-receipt history, propagated through composed manipulation and MCP. MCP persists teardown.json and refuses a clean exit on unsuccessful teardown or persistence failure. Existing parking/stop/driver torque policy remains unchanged.

Evidence: eight baseline causal tests fail in0.17s; final eight-file selection165PASS/45.11s,655source files and protected store identical before/after. Prior exploratory121PASS/1FAIL was a mock fixture returning Mock rather than the synchronous None contract; corrected fixture only. A mistyped test path produced a separate collection error/no tests and is retained externally. Ruff is not installed in the reused environment; no install was attempted. AST/diff validation performed separately. No services, models, simulator, GPU, publication or shared environment mutation. See docs/RUNTIME_TEARDOWN_RECEIPTS.md and benchmark/results/runtime_teardown_receipt_20261003.json.


## 2026-10-03 — finish pending teardown without erasing failure

Independent review of cfff2f2 reproduced a real composed owner whose read outlasted the unchanged five-second drain budget. After the reader returned, shutdown_runtime and MCP caches prevented its close from running. Explicit repeat calls now resume only incomplete delegated cleanup; legacy park/disconnect and the first MCP stop are not repeated. RobotRuntime propagates current complete separately from sticky historical ok, preserves direct and nested attempts, and never recasts a cached failed receipt as successful. Composed domains still require dictionary receipts; legacy synchronous drivers keep None.

Causal baseline: 3 failures/1 passing no-repeat control in10.16s; a separate source export adds2 causal failures/.19s for direct history and nested completion. Final six-file selection139PASS/45.74s,568source/test/config hashes and protected stores unchanged. Added explicit persistence-error history controls; intermediate74PASS/30.53s retained as earlier limited evidence. Ruff F/E9 and diff checks pass. No native model, GPU, service, host campaign, publication or shared environment writes. Receipt: benchmark/results/runtime_teardown_retry_20261003.json. Earlier lifecycle failure remains failed even after cleanup becomes complete.

## Publication follow-up: teardown merge from main70d22de

Normal merge b50229adeb40b5c0439f4b0285aba790a4f3bc99 preserves latest teardown and appends both
logs. Focused runtime/MCP/close composition: 96PASS/0SKIP, 26.92s;
1618 source inputs and protected stores unchanged. Prior freezes archived;
no physics, GPU, limits, or admission changed. Receipt: docs/evidence/robot-modularity/microduck-locomotion-teardown-composition.json.
### 2026-10-03 — isolate healthy support semantics from cyclic GC

The local post-merge check on dfa0af2 returned reader_timeout for a supported
rest episode. Its log does not include GC or read timing; its cause remains
unknown. A separate external injection of 80 ms automatic-GC callback work at
read four reproduces 11 failures among 12 semantic cases. Existing test-only
healthy_episode_gc isolation restores all 12 without changing the sampler,
40 ms read budget, clocks, support contract or required verdicts. Six semantic
functions opt in; the unknown-support control now also asserts its reason so a
transport timeout cannot satisfy it accidentally. A bounded delayed-reader
negative now covers both walking and stationary stop, alongside the unchanged
TCP-negative control. All three continue to require unverified/reader_timeout.
This is software fixture coverage, not native locomotion admission.

The final support/helper/effects selection on main 075c08c passes 395 tests in
40.12 s, with source and protected stores unchanged. No runtime diff exists.
The controlled original 12 verdicts all timed out; one test accidentally
accepted unverified without its reason, now strengthened. Raw source-bound
receipts are linked from benchmark/results/mobile_support_rest_fixture_20261003.json.
### 2026-10-03 — ordinary fastening postcondition

The ordinary `turn_screw` routine counts commanded wrist strokes. Its nested
`physical_verification: unverified` was absent from the standard postcondition
registry, so trace/memory and the reflex path could record success without
observing fastener motion. This isolated branch starts at coordinator
2067019b8d7397110d0f139f11e116ff5f1bd9b6. The planned correction now adds an explicit
unverified fastening postcondition, preserves execution/failure information, and
prevents unobserved fastening from receiving success credit. No pose-only, wrist-only or
self-reported result will prove threading or seating. The Factory contact
scene/controller is not connected by this correction.

Adversarial checks exercise ordinary dispatch, memory, trace and reflex learning,
alongside existing error/command regressions. Tests were deferred until the
coordinator closed the native voice window. Final validation: 154 passed in
20.88 s; the 15 new cases alone passed in 0.25 s, while the same cases on base206
gave 14 expected failures and one unchanged-convention pass. The initial extended
run's sole failure was a 900-character static source guard; shortening its nearby
comment preserved the guard and runtime semantics. Sources (649 files) and
protected memory stayed unchanged during every run. Ruff F/E9 and diff checks
passed. Details and hashes: benchmark/results/fastening_postcondition_20261003.json.
No simulator, model or service was started. The physical fixture integration
remains a separate explicit profile/controller/observer/lifecycle task, described
in docs/FASTENING_RUNTIME_GAP.md; this correction grants no physical admission.

## Legacy completion obligations — 3 October 2026

- Baseline `8a62a49`: real legacy dispatch marked `turn_screw` unverified, but both direct LLM and failed-reflex→LLM `task_done(success=true)` produced a successful TaskReport. The original three-control probe is retained outside the checkout; the two causal acceptance tests fail on that baseline and the read-only control passes.
- Added a task/arm/action ledger only for the existing `POSTCONDITIONS` registry. Local checker verdicts, rather than actor/LLM fields, settle obligations. Unknown/failed effects remain cumulative across later calls and model-accessible resets; only the trusted host opens a new task. Preparation failures and cancellation release active ownership without success credit.
- `task_done`, TaskReport and fast-path credit now consult that ledger. Individual actuator/checker contracts remain unchanged, including the limited historical commanded-home check. Passive sensor domains do not implement or need an effect boundary.
- Retained the first wiring run (265 passed, 8 failed): one adapter bug, four `__new__` fixture initialization omissions, and three obsolete static-mock success expectations. Fixed the adapter/fixtures; the mock E2E and memory checks preserve all execution assertions while reporting unverified, and the positive fast-path fixture uses independently read open-jaw feedback.
- No GPU, physical processes, protected learned-store mutation or voice/video changes. Final targeted tests and inventories are linked from `benchmark/results/legacy_task_completion_20261003.json`; no new full-suite claim.

### 2026-10-03 — refuse positive placement credit after failed release

The retained MuJoCo withdrawal experiment reported a failed placement while
still holding the blue cube, yet the proximity-only postcondition confirmed it.
Placement execution failure or explicit possession now vetoes positive credit,
preserving the independent measurements and any refutation. Annotation derives
`verified` from the current postcondition even for already-failed executions;
it preserves the actor's error and does not turn a home-only failure into a
placement refusal. No motion or verifier tolerance changes.

The original replay was nine expected failures and three passing controls;
the corrected production passes 199 focused checks. Final test-only naming
cleanup passes all 14 new regressions. Sources and protected stores remain
unchanged during each check; the rounded retained trace is not a new physical
run. See docs/PLACEMENT_REFUSAL_VERIFICATION.md and its source-bound receipt.
The separate two-cube physical failure remains open. No push or merge.

## Minimal-install task-effect smoke correction — 3 October 2026

The minimal-install job on 0249f718 correctly returns exit1: the SO-101 mock
exhausted its air-grasp attempts and the static camera refuted displacement.
The task ledger prevents a later mock-LLM greeting from concealing that debt.
This isolated fix retains the nine absent optional dependencies and exercises
the real CLI pipeline with explicit mock/analytic inputs and private stores.
The existing mock-jaw contact hook permits the whole SO-101 grasp/place/home
command path; the static image still cannot independently confirm relocation.
Assertions bind TaskReport, trace, task effects and teardown instead of accepting
an arbitrary nonzero exit. A separate read-only task must still succeed. No
runtime, verifier, motion limits, shared stores or GPU behavior changed.

A fresh private `uv sync --frozen --extra dev --extra kinematics` reproduced
the original CLI exit1 in 7.87 s with all nine optional imports absent. The
final selection passes 36 tests in 2.21 s (two CLI cases, task-effect ledger,
and teardown failure/pending/exit controls); 1,610 checkout inputs and the
protected learned store stayed identical. The retained initial test run also
records why the synthetic completed path may be unverified rather than refuted:
its own belief update is explicitly not independent evidence. Both verdicts
deny completion. External evidence: `MINIMAL_INSTALL_LEDGER_20261003/evidence/`
(`baseline/result.json`, `smoke-02/pytest.log`, `focused-01/result.json`).

Normal merge 43928e6c01b438c02ebb5d417c21a40713ce2fe3 preserves latest teardown and appends both
logs. Focused runtime/MCP/close composition: 103PASS/0SKIP, 13.11s;
1662 source inputs and protected stores unchanged. Prior freezes archived;
no physics, GPU, limits, or admission changed. Receipt: benchmark/results/rgbd_teardown_composition_20261003.json.
### 2026-10-03 — measured attachment frames in placement aiming (in progress)

Apply `manipulation-ik` to the optional MuJoCo region adapter: bind the measured
tool-to-object transform and use the destination orientation when aiming XY.
Share the resulting pose between geometric preview and ordinary `place_at`,
rejecting snapshot drift before consumption. Preserve the release height,
three-second planning budget, IK candidates, collision paths and physical
verifiers. The 7a52 physical episode and scratch diagnosis remain frozen outside
this worktree. CPU contract/causal controls precede review; no new dynamics,
renderer, GPU run, physical attachment or publication is authorized here.

The completed candidate passes 313 targeted checks in 38.35 s, including 44 new
frame/binding/ordinary-skill controls. Tracked inputs and protected stores stayed
unchanged during validation. The first combined check retained 308 PASS and one
incorrect test assertion: it compared the selector deadline against a timestamp
before function entry (17.8 microseconds difference). The corrected control
records the selector's actual first clock read; the three-second production
budget did not change. Existing no-attachment pose vectors remain covered.
The ordinary point path consumes the same measured pose as region preview,
and its private handoff is an in-call guard, not single-use actuation authority.
No physical episode, GPU, renderer or publication was launched for this change.
See docs/MUJOCO_PLACEMENT_ATTACHMENT.md and the source-bound benchmark receipt.

The 313-test checkpoint is retained in local commit 326720e7, before independent
review exposed a stop/reset race inside SafeArm's first start-state read.
The original external control reaches backend dispatch after cancellation;
the corrected consumer keeps the original token through all carry segments and
checks it around reads, stream start and waypoint approvals. The real release
owner must retain that token before opening. No SDK/callback runs under the
short stop lock, and no deadline, planning geometry or physical criterion was
relaxed. Fifteen new consumer/transfer controls pass with native writes and
integration intercepted, plus the two-case external control (only its caught
exception types changed to include SafetyViolation).

The final selection passes 539 tests in 61.16 s with all inputs/stores unchanged.
Its earlier 382 PASS / 5 FAIL selection is retained: the five incomplete harness
doubles also fail against the exact old harness. They now bind the real
withdrawal guard; grasp_evidence's two-line migration matches root 9b97b654.
The optional cancellation check is only invoked for an explicit token. A
separate 147-test callback/legacy selection passes. Ruff F/E9 adds no findings;
two pre-existing unused imports in the carry test remain. No new physical
episode, renderer or GPU was run; the original two-pick FAIL remains unchanged.

### 2026-10-03 — Preserve explicitly owned Spark model endpoints

Spark launch now honors an explicitly supplied CASCADE_QWEN_BASE_URL instead
of unconditionally redirecting it to port 8080. Unset retains the default; an
explicitly empty value fails the existing resolver. Required model identifier,
context, installer and MCP 300-second budget are unchanged. Added four actual
Bash/resolver checks, including an ephemeral HTTP model fixture and a pre-HTTP
regression guard to avoid contacting unrelated services.

83 focused launcher/proof checks passed before the final test-only HOME
preservation correction; all four affected cases passed afterwards. Both runs
kept 1602 input files and protected stores unchanged. The archived original
launcher fails both explicit endpoint controls while the candidate passes; the
unset-default control passes for both. Independent source review, Ruff F/E9,
Bash syntax and diff checks pass. No model, host, robot or GPU was launched, and
this change does not validate trial12 or authenticate model weights.

### 2026-10-03 — ordinary fastening postcondition

The ordinary `turn_screw` routine counts commanded wrist strokes. Its nested
`physical_verification: unverified` was absent from the standard postcondition
registry, so trace/memory and the reflex path could record success without
observing fastener motion. This isolated branch starts at coordinator
2067019b8d7397110d0f139f11e116ff5f1bd9b6. The planned correction now adds an explicit
unverified fastening postcondition, preserves execution/failure information, and
prevents unobserved fastening from receiving success credit. No pose-only, wrist-only or
self-reported result will prove threading or seating. The Factory contact
scene/controller is not connected by this correction.

Adversarial checks exercise ordinary dispatch, memory, trace and reflex learning,
alongside existing error/command regressions. Tests were deferred until the
coordinator closed the native voice window. Final validation: 154 passed in
20.88 s; the 15 new cases alone passed in 0.25 s, while the same cases on base206
gave 14 expected failures and one unchanged-convention pass. The initial extended
run's sole failure was a 900-character static source guard; shortening its nearby
comment preserved the guard and runtime semantics. Sources (649 files) and
protected memory stayed unchanged during every run. Ruff F/E9 and diff checks
passed. Details and hashes: benchmark/results/fastening_postcondition_20261003.json.
No simulator, model or service was started. The physical fixture integration
remains a separate explicit profile/controller/observer/lifecycle task, described
in docs/FASTENING_RUNTIME_GAP.md; this correction grants no physical admission.

### 2026-10-03 — mounted Factory runtime implementation, stage one

Authorized scope: optional pre-engaged fastening domain through RobotRuntime,
one measured tightening turn and independently observed rest. Apply the
physics-simulation and manipulation-ik skills: actual solved state/contact
readback, no pose assistance, no pickup or seating inference. First implement
immutable model/epoch/solve contracts, a per-write lease/generation guard, a
passive bounded solve journal and the independent domain verifier. The native
adapter must then provide joint velocities/efforts, tool poses and complete
contact coverage from each solve. RobotRuntime supplies coordination, not these
physical guards. No physical profile is enabled by the CPU stage.

Bounds are explicit and identity-bound: 0.8 rad/s SO-101 arm limit, imported
joint/effort limits, 0.05 Nm spindle cap, 10 rad/s measured nut/spindle bound,
Factory M20 pitch 2.5 mm and unchanged ThreadContract geometry/contact gates.
Synthetic adversarial fixtures will cover expired/stopped writes, stale/missing
or discontinuous observations, forbidden contact, wrist-only motion and failure
to rest. GPU0 and voice ports remain reserved for Hermes. No native launch or
publication is authorized by this stage; CPU suites are coordinated separately.

Stage one implemented in control/fastening.py and skills/fastening_runtime.py,
with direct ordinary RobotRuntime composition and a single arm/spindle command
resource. Initial 41 and 108 CPU checks passed. Independent review then added
deadline-after-read, pre-ACK capture and replay guards. Final 115 passed in
2.19 s (56 new synthetic cases); 658 source hashes and protected memory were
unchanged. One intermediate fixture failure is retained: its stop-prefence
timestamp accidentally reversed the capture clock; adding modeled ACK latency
isolated the intended exclusion without relaxing limits. Ruff F/E9 and diff
checks pass. See docs/FACTORY_FASTENING_RUNTIME.md and the source-bound receipt
benchmark/results/factory_runtime_stage1_20261003.json. Stage two will connect
native per-solve measurements and exact zero-spindle stop. This checkpoint has
no physical profile/launch/admission and has not modified the old native scene.

### 2026-10-03 — Factory per-solve native readback checkpoint

Implemented sim/factory_observation.py against inspected Newton1.6/MJWarp3.12
APIs: all collision/reducer capacity counters, actual solved vector contacts,
explicit joint/DOF/control maps, bounded actuator-model descriptors, separate
external/passive/constraint efforts, and CPU shadow FK with complete moving
collision AABBs. Force/contact interval and endpoint pose times are distinct.
This is instrumentation only; no profile or physics launch is enabled. The next
owner stage supplies binding, fenced uploads, zero-spindle stop and lifecycle.

Final 210 CPU tests passed in 0.96 s, with 660 source hashes/protected memory
unchanged, Ruff F/E9 and diff checks clean. Independent exact MuJoCo3.12 rotated
box FK control passed without dynamics or Newton/Warp/Kit import, confirming
geom_aabb half extents. A first control invocation failed on missing PyYAML in
the minimal converter interpreter; the successful read-only package overlay and
failure distinction are retained in benchmark/results/factory_observation_20261003.json.
No gains, physical thresholds, old native sources or shared environments changed.
Independent review added exact float32/shape/finite checks before any zero-wrench
claim: empty arrays previously passed np.any. The final count includes 39 new
missing/transposed/wrong-type channel adversaries; the 159-pass prior checkpoint
and original MuJoCo3.12 geometry receipt remain recorded separately.

### 2026-10-03 — Factory single-owner implementation (in progress)

Continue the authorized physical integration with a private per-subsolve owner,
bounded admission queue, guarded upload stamps, real native step/time checks,
and exact zero-spindle writes on stop/error/closure while retaining arm targets.
Bind the compiled model, collision arrays, mapping, imported limits, assets and
source files before exposing the controller. Force/contact values describe the
completed interval and poses its endpoint. No solver graph or extra forward
pass is implied. Synthetic CPU faults exercise scheduling, uncertain writes,
zero upload failure and observation loss; these do not establish native proof.
Ordinary profile/builder wiring follows this owner checkpoint. Native launches
remain separately coordinated with the RGB-D work; no GPU is used here.

Owner/readback checkpoint: 255 CPU checks passed in 1.16 seconds with 663
source hashes and protected stores unchanged. Forty-five owner cases include
native count/time reset, model/force-parameter drift, queued admission, uncertain
uploads, zero upload failure and stop during preparation versus an admitted
solve. Independent review found the first upload preceded collision preparation;
a source-bound synthetic control retains the old 0.03 Nm post-stop solve and
the corrected zero/new-generation result. The guarded upload now immediately
precedes the admitted solve. Observed effort tolerance was tightened from
1e-7 Nm to the exact float32 representation of each declared cap; requested
limits are unchanged and the next float32 value is refused. Raw is not clipped.
No native task or active profile is admitted. See the owner receipt and runtime
guide; ordinary config/MCP construction remains the next separate checkpoint.

### 2026-10-03 — configured Factory domain (in progress)

Add explicit kind=fastening to the existing composition builder and MCP route.
The shipped factory_m20_mounted profile has a null model pin: passive discovery
works, but construction refuses before any SDK/device import until a separately
owned preparation supplies the exact model digest. A pin is not a physical
success flag. Construction must observe a full quiet pre-engaged window while
the initial latch remains on, retain all solves, and preserve failed startup
and closure receipts. No fake ArmBase, alternate MCP server, automatic reset,
pickup, or seating capability. Native execution remains unperformed here.

Configured Factory checkpoint complete: 322 CPU checks passed in 4.40 s; after
correcting only the two shipped asset paths to official installer layout, all
30 configuration cases passed in 0.35 s. Each run preserved 666 sources and the
protected stores. Static independent review closed startup/closure exception
masking and sticky evidence-write failures; original execution outcomes remain
visible but cannot receive physical success credit after evidence loss. The
initial fixture failures and final source-bound receipts are retained in
benchmark/results/factory_config_20261003.json. Discovery uses existing MCP and
construction requires an exact model pin plus measured quiet readiness; the
shipped null pin remains inert. No native launch, threshold relaxation, SDK
installation, publication, or shared-environment change occurred. Next work is
an external reviewed preparation/readiness harness with private ownership/cache,
separate construction/readiness times and unchanged 10 s / 120 s budgets.

### 2026-10-03 — inert Factory profile leaves accelerator selection unresolved

The composed full suite found the shipped profile violated the existing global
rule against pinning an accelerator. Keep discovery passive with device:null
and model pin:null; explicit model preparation and configured construction must
reject an unresolved device before SDK access, without auto/CPU fallback.
Preserve the original global test and all native thresholds. Add direct no-IO
refusal controls and select CUDA explicitly only inside synthetic/native private
fixtures. The preparation harness remains unexecuted pending this source update.

Final control: 50 PASS in 0.44 s, including the unchanged global shipped-device
policy test and preparation/build refusals before SDK or output creation. All
666 source hashes and protected stores remained identical. The first edit's
YAML indentation error caused 23 parser failures (27 passes); it is retained,
and fixing indentation alone resolved it. No runtime physics limit changed.
See benchmark/results/factory_device_selection_20261003.json.

### 2026-10-03 — repair the pinned Factory SDK collision-mode interface

The first shared-GPU prepare-only process exited naturally with an AttributeError
before model binding: Newton 1.6 stores `_use_mujoco_contacts`, whereas the new
adapter accessed a nonexistent public attribute. Preserve that source-bound
failure and both original harnesses. Read the pinned SDK declarations and solve
branch, require its actual private mode and effective MJWarp collision option
to be explicitly false, and reject missing/contradictory values without fallback.
Add synthetic regression controls that expose only the real interface, plus a
static inventory of the remaining model/owner/observer APIs. No verifier limit,
SDK package, native process, or physical admission changes in this checkpoint.

Final controls: the old source failed 15/16 new adversarial cases (one existing
CPU-route refusal already passed); the corrected targeted suite passed 274 tests
in 1.36 s, and exact MuJoCo 3.12 CPU owner/readback controls passed 163 in 0.45 s.
Each preserved all 666 tracked test inputs and protected stores. The fingerprint
inventory and owner signatures match the pinned source declarations. Independent
read-only inspection found no other concrete observer API mismatch; neither that
review nor the tests establish native instantiation or contact coverage. The
failed prepare receipt and six unchanged foreign process identities are retained
in benchmark/results/factory_sdk_interface_20261003.json. A new external harness
variant will inspect the exact failed preparation traceback frame for original
scene counters without replacing the scene class or granting a valid model pin.

### 2026-10-03 — inspect optional native arrays after preparation V2

The second shared prepare-only attempt on af463859 failed when fingerprinting a
Newton model field that was None. The exception checkpoint measured scene/native
step and time zero; owned closure and unchanged input/peer records were retained.
No pin or task admission resulted. Before a correction, reproduce the array
layout with the exact SDK on CPU, inspect allocation conditions for every bound
array and distinguish an explicitly optional field from missing required data.
Preserve optional absence in the fingerprint and reject later presence, value,
shape or dtype changes. No None-to-empty conversion, omitted required arrays,
solver/verifier changes or additional GPU run are authorized by this work item.

CPU causal control with the pinned real Newton builder reproduced the old
fingerprint error and isolated shape_filter as the only None field. The corrected
code preserves its declared optional absence, rejects the other 88 required
arrays when unavailable and fingerprints all presence/layout/value changes.
375 targeted tests passed in 1.56 s; the real CPU builder fingerprint and all 43
MJWarp model/seven option layouts also passed inspection without constructing a
solver or taking a physics step. Sources/stores were stable within final checks;
SDK/assets/harness/stores still match the closed V2 baseline. Root reviewed the
code delta. No new GPU run, native pin, readiness or physical admission exists.

### 2026-10-03 — diagnose the initial mounted joint-margin rejection

Preparation V3 on 8f7f5b3 passed SDK and array binding but rejected the authored
initial joint state against the unchanged 0.02 rad margin. Its native counters
and time remained zero, closure was natural, and all inputs/foreign identities
were retained. Preserve that checkout and investigate in a separate worktree:
measure the exact initial IK, joint/control intersections and MuJoCo/Newton
reference offsets on CPU before changing the fixture recipe. Add diagnostic
joint values without changing the rejection. Any corrected authoring must retain
the legacy recipe, solve within the guard's margins without clipping, and carry
a new explicit identity. No GPU, readiness, motion, tolerance relaxation, shared
store modification or publication is part of this checkpoint.

Measured cause: legacy wrist_flex=1.64468204 rad leaves 13.377956 mrad below
its effective 1.65806 upper bound; all six MuJoCo/Newton references are zero.
With unchanged IK residual criterion, three declared CPU placement comparisons
rejected x=.25/.26 and accepted x=.23 across all 43 follower targets. The new
explicit margin-v2 profile moves the complete thread fixture 10 mm toward the
base and solves within 25 mrad of joint/control bounds; the 20 mrad guard and legacy
default/recipe/profile are unchanged. Model identity now binds/rechecks actual
authoring parameters and targets; diagnostic rejection names q/bounds/margin.
398 targeted tests passed in 2.27 s with 669 source inputs and protected stores
unchanged. Exact Newton 1.6 CPU import/FK passed all 43 positions with maximum
weighted residual 1.069e-7, without Factory SDF/collision/solver/dynamics. An
earlier CPU import lacked trimesh and is retained separately. Ruff F/E9 and
diffcheck passed. No new GPU attempt, model pin or physical admission exists;
independent code review and future native preparation remain separate gates.

### 2026-10-03 — preserve strict binding at the native scalar boundary

Prepare-only V4 on 96ecdb5 closed naturally with zero original scene/native
counters after rejecting NumPy float64 fixture coordinates in FasteningBinding.
Keep that source and receipt frozen. In this separate worktree, reproduce the
exact constructor binding assignment on CPU before converting the validated
native coordinate values explicitly to Python floats. Keep the public numeric
contract, recipe checks, physical limits and array values unchanged. No new
native run, readiness, owner or movement is part of this correction.

The same new test executes the constructor's exact binding AST after real
authoring validation: old production source fails both recipes with the native
float64 array; seven strict-contract/invalid-authoring controls pass. A one-line
explicit float conversion then passes all 362 focused tests in 1.48 s. The
public `_number` contract is unchanged; no coordinate mutation or normalization
occurs. Both runs preserve all 670 source inputs and protected stores. Ruff F/E9
and diffcheck pass. Native V4 remains FAIL with zero measured counters, no pin,
natural exit 1 and all owned processes absent; six original foreign births
were observed unchanged. No native retry or physical admission is claimed.

## 2026-10-03 — Finite Factory precompilation (in progress)

Base 7851e474, separate feat/factory-finite-precompile worktree. Readiness01
remains FAIL: its first cold solve returned after the unchanged 10 s readiness
deadline. This opt-in preparation loads a finite pinned Newton/MuJoCoWarp module
inventory without executing kernels or advancing physics. It will require a new
model identity and separate native review; CPU controls cannot establish quiet
readiness. No shared caches, model files, SDKs, or learned stores are modified.

The implementation is now ready for independent review. The seven-file focused
selection passed 433 tests with 4 existing SO101-asset skips in 1.75 s, using
normal conftest/private stores; 556 source/config/test inputs and protected stores
were unchanged. A separate exact SDK CPU probe traversed 1,463 arrays, built
20 lazy kernel definitions and loaded one generated CPU module without a kernel
launch, solver or physics step. Before/after physical digests match. It exposed
and corrected two host representation assumptions (sets and Warp runtime
infrastructure); all failed probes are retained outside the checkout. The
recipe now checks the actual nv_pad20 and native tile/CSR dimensions. Comparing
native01's seven-hex module prefixes gives27 direct matches and one match with
the separately retained native CSR width90 (minimal CPU fixture width97).
Full CUDA hashes/loads, Factory native buffers and readiness remain unvalidated.
No native launch is authorized by this checkpoint. See FACTORY_PRECOMPILATION.md
and benchmark/results/factory_precompile_cpu_20261003.json.

Independent review found scene-level operation/authoring state outside the seven
original snapshot roots. Snapshot now covers all scene fields (except its prior
evidence receipt) and effective class constants, including IK data, targets,
epoch and drive flags. New controls also found temporary-dictionary ID reuse;
retaining the referenced objects prevents a false alias. Python3.10-compatible
error notes preserve the primary failure if receipt persistence also fails.
Final selection: 448 PASS / 4 existing asset SKIP in1.62s, same556inputs/stores.
Real SDK CPU control now observes1,627arrays including IK data plus authoring
and class fields; both physical digests are identical. Earlier receipts and
byte-matching pre-review helper/runtime sources are retained under external
validation/review-source-01-retained. No new native process or solve.

## 2026-10-03 — Factory explicit-pair compiler admission

Preserved native preparation on a2c40c83 as FAIL/CLOSED: the constructor guard
assumed `nxn`, but the unchanged pinned CollisionPipeline defaults to explicit
pairs. No finite-plan load or solve completed. New optional selector/profile
v2 requires the exact BroadPhaseExplicit type and owned int32 pair inventory;
it uses the SDK's precomputed-pair kernel handle without changing physics.
The old selector/profile is withdrawn, not reinterpreted.

Admission now retains all 22 values/errors/verdicts before deciding, including
missing SDK attributes and serializable nonfinite clock rejection. Normal
preparation persists the report even when a guard rejects. Full scene/model
snapshot fencing and the 39 reviewed definitions remain; io.py layout source
is now pinned too. The predicate-by-predicate SDK audit and retained failure
are linked from benchmark/results/factory_precompile_explicit_cpu_20261003.json.

Validation: seven focused files 466 passed, 4 optional skips, 1.92 s; 556 inputs
and protected stores unchanged. Real pinned SDK CPU pipeline with exact source
kwargs reproduces the old rejection, accepts the explicit branch and builds
39 definitions with equal snapshots/zero solves/zero finite compiler loads.
CPU builder allocation kernels do run. Reduced CPU CSR width75 remains refused
against native90; it is not the full Factory geometry. No new GPU/readiness
run, limit change, SDK patch, push or physical-admission claim.


### 2026-10-03 — finite Factory snapshot uses actual native arena storage

The first explicit-route CUDA preparation completed39 loads/22 predicates but
failed on nine CPU descriptor hashes; all clocks/steps stayed zero and it closed
naturally. That failed result remains unchanged. A separate CPU/header audit
proved that NULL MuJoCo arena pointers can return owning, uninitialized Python
arrays. The snapshot now records presence/address/shape/dtype for all64 numeric
arena descriptors, hashes actual native views, and refuses inconsistent storage.
The generated offsetof layout is pinned with the binding binary, installed
headers, pointer ABI and actual mapped allocator library before pointer reads.
No runtime compilation or unsupported-ABI fallback is introduced.

Independent review also found inherited omission of nested contact/option/stat
structures. The same CPU mutation probe fails oldefd84ed2 and passes newd84c913a;
position/friction/gravity/warning/solver changes are detected, while repeated
snapshots and restored values match. Source01/failures remain archived.
Final486CPU PASS/4optionalSKIP (2.41s),587inputs/protected stores unchanged.
Real pinned SDK creates39 definitions with identical snapshots,0finite loads
and0solves; CPU builder kernels remain distinct from physical steps.
Receipt: benchmark/results/factory_native_snapshot_cpu_20261003.json.
No revised CUDA preparation, readiness or fastening task has yet been admitted.


## 2026-10-03 — Instantiate Factory contact-writer kernels before loading

- Retained ordinary readiness-v2 failure on16653c84:10s deadline, first advance9.3643s,20 completed steps plus1 drained at closure, no quiet startup or fastening action.
- Three v2 modules contained zero PTX entrypoints. Definition-only SDK probes reproduce the later compiled hashes by binding the exact `sim.collide.ContactWriterData` type.
- New opt-in writer-v3 profile explicitly binds those overloads, records concrete kernel keys/signatures and verifies each loaded forward symbol. Previous selectors are refused; finite39 inventory,22 guards, physical snapshots and deadlines remain unchanged.
- Native preparation/readiness for this new recipe are pending; no physical fastening success or reusable model pin is claimed.
- Validation:492 PASS/11 SKIP plus7 pinned-ABI PASS; two exact-old-function causal failures retained. Independent SDK definition/retained-PTX review found no blocker. No native v3 launch.

## 2026-10-03 — Preserve the exact Factory observation-age rejection

The separate v3 readiness run failed after 309 solves; its native exit1 and
natural closure remain intact. Existing timestamps cannot establish the exact
rejected age or attribute the delay. Added bounded stage/capture/check/age/limit
and identity diagnostics to the existing FasteningFault string paths. Capture,
0.2-second freshness, 10-second readiness, 0.5-second quiet interval, check order,
owner lifecycle and all verdicts remain unchanged. No native rerun or optimization.

Causal same-tests control: old5FAIL/3PASS, candidate8PASS. Final five-file CPU
selection370PASS in1.51s;588 inputs and protected stores unchanged. Source/model
identity changes are explicit, and no old native pin is transferred. Evidence:
benchmark/results/factory_observation_age_20261003.json and the retained v3
failure summary beside it. No SDK changes, GPU operation or publication.

## 2026-10-03 — Retain the threaded outcome through the rest endpoint

Verified the supplied CPU unwind counterexample on exact b7a0deee: the cached
pre-stop verdict incorrectly survived complete passive rollback. One-module
fix retains the same baseline and post-stop observations, runs the unchanged
threading verifier through the rest endpoint, and requires both original
achievement and final retention. No late positive credit, freshness change,
new control write, quiet-window relaxation or native success claim.

Baseline02:5FAIL/4PASS; final402PASS/2.26s across seven files,589 inputs and
protected stores unchanged. Baseline01 also preserves one incorrect test
expectation of the timeout wording (budget vs physical rest deadline), fixed
before baseline02. All original repro/data are retained. Receipt:
benchmark/results/factory_final_outcome_20261003.json. CPU only; no SDK/GPU run.
## 2026-10-03 — opt-in Isaac command evidence (in progress)

Independent checkout from e5211ae. Parent approved passive command evidence
only: queued ACK, first setter-returned target and repetitions, existing state
clocks and nested home stream boundaries. Defaults, commands, safety checks,
post-ACK pacing, waits and verdicts must remain unchanged. No SDK reads, RPCs,
physics updates, native runs or shared-store writes are authorized by this
change. The new optional evidence uses bounded memory and flushes after action.
Source campaigns and precompilation harnesses remain immutable.

Command-evidence CPU checkpoint: 409 PASS / 0 SKIP in 21.47 s across 16 files
(including portable bundle checks), 760 code/config inputs and all three
protected stores identical before/after. Ruff F/E9 and diff check passed.
The first focused run's five grasp-evidence failures are retained; a clean
base e5211ae reproduces the missing check_model_withdrawal method on its old
SimpleNamespace test harness. Only that fixture now binds the real guard with
no pending debt. Two initial new TCP tests omitted connect and were corrected.
No bridge pacing, action limits, physics or independent verdict changed. This
feature is optional and has no native validation/performance claim. Receipt:
benchmark/results/isaac_command_evidence_cpu_20261003.json. Source remains local
for review; no publication or simulator launch.

Review follow-up (freeze02): optional direct-script import now explicitly uses
this checkout's src, without depending on PYTHONPATH. The actual bootstrap AST
failed only when enabled in freeze01 and passes enabled/disabled now. The first
attempt to run that test had two temporary-directory setup errors, retained
separately; it was not evidence of the import defect. Supersession is named
`superseded_before_setter_receipt`: the earlier target may already be in flight
and still return successfully. Its immutable first write is retained. Every
update attempt clears the diagnostic boundary before the existing SDK update.
A controlled update that advances then raises previously reused the old clock
(1 FAIL / 2 PASS across update/capture/record failure cases); now all three
leave it unavailable. No SDK reads or physical clock/pacing changes were added.
Final freeze02 selection: 414 PASS / 0 SKIP, 21.09 s, 760 inputs and protected
stores identical; Ruff F/E9 and diff check pass. Freeze01 and all 17 files are
retained outside this checkout, as are its 409-PASS receipt and earlier failures.

## 2026-10-03 — Preserve the opt-in target receipt flag at the ordinary Isaac launcher

The diagnostic bridge flag introduced in 2bab2379 was filtered out by the
ordinary `isaac_launch.clean_environment` allowlist. Add only that key, keeping
the default absent and values literal; the MCP-only evidence directory remains
excluded from the bridge environment. Tests call the real sanitizer and the
real Python child launcher without Kit/GPU. The causal baseline and isolated
controls are retained outside this checkout under `../validation/flag-causal`.
No pacing, duration, safety gate, simulator recipe or runtime verdict changes.

### 2026-10-03 — Isolated Isaac command-evidence publication composition

Applied only the command-evidence and launcher opt-in changes on main `0fe33fbb`.
The motion-evidence tests now define their small real-dispatch fixture locally;
the unrelated, unpublished review-v5 module is not a dependency. The grasp
telemetry fixture retains the main-branch guard interface rather than importing
the unpublished MuJoCo withdrawal API. Product behavior is unchanged from the
two selected patches. The affected 16-file CPU selection passed 422 tests in
21.83 s, with 707 source/config/test inputs and protected stores unchanged.
Ruff F/E9 and diff checks passed. This adds no native or physical acceptance.

Publication composition follow-up: merged actual main `075c08c` without rebasing
the topic. Only the worklog append conflicted; all production merged unchanged.
The 17 topic files other than this log remain byte-identical to `b4ea52b`;
removing the three runtime instrumentation hunks reproduces main exactly.
The affected observer, generalized-joint MCP, packaging and command-evidence
selection passed 206 tests in 5.79 s, with 715 inputs and protected stores
unchanged. The earlier temporary composition against non-main `1275c294`
passed 247 tests and was aborted before commit after correcting the branch
identification; its evidence remains separate. No Factory implementation is
introduced by this topic. No native run or physical-admission claim.
### 2026-10-03 — Preserve TCP timeout coverage across caller scheduling

macOS CI on PR84 retained `unverified/reader_timeout` but failed a test-only
assumption that the independent reader makes exactly one TCP connection. A
controlled delayed caller reproduces that same assertion failure: the real
socket can expire and reconnect before `begin()` resumes to cancel its sampler.
Both ordinary and delayed schedules now require the original 40 ms timeout
verdict and allow only read-only hello/state messages. Runtime, clocks, limits,
measured-support gates and controller ownership are unchanged. The controlled
schedule does not claim to identify the historical OS scheduling event.

Retained causal baseline: 1 failed; corrected focal: 2 passed; affected support
and effect suite: 389 passed in 40.24 s. All 1,599 source inputs and four stores
are unchanged across validation. No native SDK, GPU, simulation or local macOS
execution. Receipt: benchmark/results/mobile_support_tcp_scheduling_20261003.json.

## 2026-10-03: Terminal reports retain unresolved effects and cancellation

All normal orchestrator reports now use a common ledger-aware finalizer, including step-budget exhaustion. Failed fast-path calls remain in the same task tool history with an explicit tier. Composed motion cancellation records uncertainty before releasing admission, then re-raises the original interruption; later successful motion cannot erase it. Reads and host-only task boundaries retain their prior meaning.

Exact PR90 head `381d65ff` reproduced three review counterexamples (two omitted-debt reports and one pre-existing composed cancellation false success), with two controls passing. Final eight-file CPU selection: **141 passed, 0 skipped in 19.82 s**; source and four protected stores unchanged. The initial two new-test failures were an incorrect assumption that `RobotRuntime.begin_task()` returns an ID; only that test assumption changed. No GPU, SDK, physical episode, shared environment modification or production limit change. Evidence: `benchmark/results/task_terminal_obligations_20261003.json` and external `TASK_TERMINAL_FIX_20261003/{baseline-01,focused-01,focused-02}`.
### 2026-10-03 — align pytest console and module imports

PR91's Ubuntu collection fails in eleven RGB-D modules importing repository-only
benchmark helpers. The unchanged source reproduces all eleven errors with the
actual pytest console script and no PYTHONPATH. Add pytest's built-in
pythonpath=["."] configuration; conftest continues selecting the checkout src.
Package discovery remains src-only and production dependencies are unchanged.

Console and module collection now select the same 6,110 node IDs in the same
order (6,114 total, four hardware deselections). The affected console selection
passes 325 tests with 13 optional OpenUSD skips. Its first execution retained
17 failures from missing jsonschema in the old shared interpreter; the same
console-script body with the existing complete read-only interpreter resolves
that environment issue. No shared environment install, verifier change, new
physics/GPU run or store mutation. Original failure logs and hashes remain in
benchmark/results/pytest_root_path_20261003.json. This validates local collection
and the affected tests; it does not assert a remote CI pass or full-suite run.

### 2026-10-03 — Compare RGB-D extraction within floating-point roundoff

CI passed collection after the import-path fix, then exposed exact-dictionary
equality on computed homography/residual floats. Retained corner/hash/board/ID
fields were identical; the largest reported residual difference was
2.5049e-13 pixels. The extraction test now keeps those noncomputed fields exact
and compares its three computed fields with absolute1e-12/relative0 tolerance.
The frozen fixture, native pixel gates and production checker are unchanged.
Affected checker/planar/live tests:81 passed,1 OpenUSD skip in17.85s. Original
CI failure and the initial zero-test filename error remain retained. Receipt:
benchmark/results/rgbd_reference_roundoff_20261003.json.
### 2026-10-03 — Current capability and evidence index

Added a documentation-only status index separating implementation, source-bound
measurements, retained failures and pending admission. Updated README and the
historical index/architecture entry points. Portable receipts retain original
bytes and hashes; GitHub publication state is a dated read-only API snapshot.
No current publication freeze or old video-resource reservation is introduced.
No code, unit test, benchmark, model, SDK, physical scene or learned store changed.
## 2026-10-03 — Optional kitchen cuMotion selection (CPU implementation)

Scope: add an explicit `isaac_kitchen_cumotion` profile and campaign backend
selection checks while preserving the existing kitchen profile, physical
verifiers, occupancy behavior and motion/planning budgets. Main base is
`67f0b61c490f1a83529ecd9143b0361737b3897b`. The `manipulation-ik` skill's
contact-only/object-state acceptance applies; this checkpoint runs no SDK,
GPU, simulator or native task. No physical admission is implied.

Plan: reuse the existing planner and exact-curve stream, inspect already
captured camera bindings, retain per-curve candidate and execution outcomes,
and reject selected-backend mismatches or missing evidence. CPU controls cover
profile inheritance, unchanged default, admission/refusal and terminal evidence.
No generic runtime instrumentation, new safety thresholds or planner fallback.

Validation: 206 CPU tests passed in 1.46 s across seven affected files; Ruff
F/E9 and diff checks pass. Source inventories and four protected store states
are unchanged during each run. The initial 73-pass/one-failure check caught
a `Cfg[key]` access in the selected-model manifest; its failure is retained,
and the correction uses the existing `Cfg.get` API. Camera selection reuses
the ordinary readiness packet validator, including native render tokens and
actual RGB-D presence. No SDK, GPU, simulator or physical task was run.
Evidence: `benchmark/results/kitchen-cumotion-selection-20261003.json`.
### 2026-10-03 — publishable MuJoCo placement extraction and physical closure

The original two-object memory test passes once on frozen d273dc4: contact-only
SO-101 MuJoCo3.14/CPU/Mesa, both ordinary placements independently confirmed
after withdrawal and home, first object retained after the second. Red/blue
settling windows are11/.600s and12/.660s; final original-point distances25.776
and30.657mm satisfy the unchanged60mm test. Exit0 is natural and every owned
process is absent. Sources, model assets and four protected stores are intact.
The previous physical failures remain retained, as does the cancelled handoff
red/green control. This is one explicit recipe, not general robot admission.

Extract reviewed manipulation hunks onto main70d22de, then merge main67f0b61;
do not copy the old runtime or carry independent Factory/sensor/task-ledger work.
575 affected tests pass, then106 reader/destination checks pass after correcting
a partial test fixture and materializing missing pinned kitchen assets. Initial
collection/preparation failures remain in the receipt. Model-only reconstruction
prohibits steps and produces both exact physical model digests at time0. The
main merge leaves tested src/config bytes intact; no repeated physical episode.
See docs/MUJOCO_MANIPULATION_VALIDATION_20261003.md and its two receipts.


## 2026-10-03: Gripper feedback fixture retains the model-withdrawal guard

The isolated jaw-feedback fixture now supplies a real idle `SafetyHarness`, as the production `SafeArm` does. The common withdrawal guard remains enabled; production code, jaw feedback, timeout assertions and limits are unchanged.

On PR93 base `f06837ce`, all eight feedback cases reproduced the missing-harness `AttributeError`. After the fixture correction, the existing feedback, model-withdrawal, postrelease, attachment-fence and safety checks passed: **117 passed in 32.62 s**, with 26 locally retained SO-101 asset files verified before and after. An earlier run without those fetched assets is retained separately as 22 passed / 95 skipped. Normal conftest, hidden CUDA, software GL and four private/protected store checks were used; no simulator service or GPU was launched. Evidence is retained outside git in `GRIPPER_FEEDBACK_FIX_20261003/{baseline-01,focused-01,focused-02,asset-materialization.json}`.


### PR93 retained-withdrawal cancellation correction — 2026-10-03

- Reproduced explicit-generation completion debt loss and a target crossing the consumer after stop/reset during approval (2 red / 4 controls on 1ac1923).
- Forward the original token for home/withdraw; register only successful explicit recovery context; fence reset reads/final observation and atomically validate token/latch/owner before debt removal.
- Independent review caught and closed explicit-original-context epoch bypass; legitimate registered reset may change the placement-history epoch.
- 151 CPU/static-geometry tests pass, 0 skip; one original physical reset test deselected. Twenty cancellation controls include the real SafeArm/ArmBase boundary with intercepted writes.
- Preserved the initial fixture errors and missing-LFS-mesh failure; 26 SO101 assets and two gripper meshes verified against hashes/OIDs, sources and protected stores unchanged. No GPU, new native physics episode or physical admission.
- Receipt: `benchmark/results/mujoco_withdrawal_cancellation_20261003.json`.

### 2026-10-03 — Retain instrumented trial 12 failure and partial command evidence

Documentation-only follow-up to a1977d3. Added the unchanged offline audit of
the ef773697 real-host episode: MCP timeout after a 303.910-second agent turn,
partial 16 MiB diagnostics with 1102 dropped events, and measured command/physics
clock costs without OS/GPU attribution or changed gates. The earlier f674fa64
failure and all prior portable receipts remain intact. Separate software
closure from missing task/home/rest acceptance. No code, tests, model inference,
GPU, physical action, shared environment or learned store changed; validation
is limited to relative links, source/evidence hashes and documentation diff.

### PR93 cancellation between reset callbacks — 2026-10-03

- Independent controls on fe0a49ee: 38 pass / 9 fail because verification, capture or depth returned after stop/reset and the next callback still ran; failure and pending debt already remained correct.
- Fence verification→memory→capture→depth→describe boundaries and preserve actual forgotten-belief count if clearing finished before cancellation. No replay or rollback of completed work.
- Exact supplied 47 controls pass; expanded seven-file selection 183 pass, 0 skip, one original physical reset test deselected. Source snapshots and protected stores unchanged; no native/GPU/physical acceptance.
- Preserve initial six new-test KeyErrors from assuming an optional earlier-failure counter; fixture assertion corrected without changing production.
- Receipt: `benchmark/results/mujoco_reset_callback_cancellation_20261003.json`.


### 2026-10-03 — Postrelease delivery fixture preserves cancellation token

The synthetic first-command substitute now accepts the explicit cancellation
token passed by the ordinary withdrawal caller and asserts that it remains the
original plan token. Exact target, state-drift, stop/reset and no-send assertions
remain intact; no runtime, physics or safety guard changes. On exact fe0a49ee,
the three parameter variants reproduced the old signature failure. The corrected
postrelease, cancellation, withdrawal and attachment-fence selection passes
122 CPU/static-geometry tests in 32.85 s, with 1,643 inputs, 26 SO101 assets and
four protected stores unchanged. No GPU or physical episode. Original red and
green evidence remains in POSTRELEASE_TOKEN_FIX_20261003 outside the checkout.

### 2026-10-03 — Bounded diagnostic lifecycle retention (plan)

The trial-12 detailed command prefix exhausted its 16 MiB budget before the
last stream/skill records. Preserve that failed receipt. This isolated follow-up
reserves at most 128 records/64 KiB inside the existing event/byte budgets for a
compact lifecycle tail, using only existing event kinds and the timestamp
already captured for each record. The original detailed prefix and submission
adjudicator remain the authority for diagnostic coverage; truncation, eviction,
drops and logging errors cannot restore completeness or physical acceptance.

Scope: `control/motion_evidence.py`, CPU regression tests and diagnostic docs.
No simulator, policy, SDK, pacing, deadline, verifier, command or tool changes.
No Isaac skill is needed for this pure Python logger change. First retain a
source-bound failing CPU control, then validate saturation/exception retention,
total bounds, malformed values and enabled/disabled control-sequence equality
with private stores. No native replay, GPU, push or PR creation in this task.

Implemented the reserved FIFO tail using the eight existing lifecycle kinds.
Fields have fixed bounds; serialization failures, nonfinite/invalid values,
evictions, oversize drops and truncation remain explicit. `Recording.error` and
action-exception formatting preserve the original exception, including
`BaseException`, when a diagnostic formatter itself fails. The detailed event
schema, full-precision values and submission matcher are unchanged. Overall
completeness stays false for any loss even when the retained prefix by itself
has matching request/ACK/setter records. No clock, command or read was added.

Validation: retained TDD baseline 24 FAIL; initial candidate 33 PASS; final
seven-file CPU selection 141 PASS / 0 SKIP in 2.56 s. Each run preserves its
source inventory and all four protected-store states. The final selection
includes saturated on/off sequence equality across normal, slow-ACK, jump,
cancel and frozen-clock cases. Ruff F/E9 and diff checks pass. Root and CM
reviewed the frozen module/tests without a remaining material blocker; no
native replay or physical closure is claimed. See
`benchmark/results/motion_lifecycle_journal_20261003.json` for hashes, retained
red sources, logs and limits. All simulator/driver/control/runtime files outside
`motion_evidence.py` remain byte-identical to base `789d1d5`.
### 2026-10-03 — Source-bound published software corrections

Added four concise status rows for PR85 final threading through rest, PR87
post-completion distance veto, PR90 terminal task obligations/cancellation and
PR93 withdrawal/reset callback fences. Copied five original CPU receipts and
the PR93 publication checkpoint byte-exactly, with source commits and hashes.
A separate composition receipt retains the 985/3 result and 264-pass follow-up;
its 1044-case union is not a single full-suite/final-head result.
The 151/183/122 selections overlap and are not summed. Historical physical
failures and all prior receipts remain unchanged; no CI-wide or new physical
admission is inferred. Documentation validation only: links, hashes, exact Git
receipt versions and diff checks; no test/runtime/SDK/GPU execution.

### 2026-10-03 — Isolate the mobile progress decision fixture from suite GC

PR86 Ubuntu job111194921868 on `b0723e0` reported `unverified` instead of
`confirmed` for the unit progress case. Its log lacks the verdict reason or
reader timing, so the remote cause remains unknown. A source-bound external
probe now preserves a controlled mechanism: adding 80 ms of automatic-GC
callback work inside read4 causes all four geometric decision cases to return
`reader_timeout`; without the intervention all four pass. An earlier probe
that triggered no automatic collection also passed and remains retained.

Apply the existing `healthy_episode_gc` fixture only to this four-case family;
retain the original geometry, thresholds and assertions, adding the verdict
reason to its failure message. Validate the same controlled probe afterward,
plus the existing independent delayed-reader veto and the affected test file.
No new timeout, runtime change, simulator or GPU use is authorized by this fix.

The controlled comparison is complete: baseline GC intervention gives four
`reader_timeout` failures (81–101 ms read4); the isolated candidate restores all
four original geometry verdicts with the same intervention. A real 80 ms reader
delay still gives four `unverified/reader_timeout` refusals with cyclic GC off.
The existing committed delayed-reader negative also passes without change.
The final affected selection is 368 PASS / 0 SKIP in 33.51 s, with source and
all four protected-store states unchanged. Ruff F/E9 and diff checks pass.
No production or shared helper changes; the original CI cause is not proven.
The later Mac86 failure is a separate exact-one-TCP-pair fixture assertion,
preserved for the existing independently reviewed TCP fix. Raw logs, all
controlled failures and source bindings are indexed in
`benchmark/results/mobile_effects_ci_fixture_20261003.json`.

### 2026-10-03 — Keep late exception debt on its admitted task

`RobotRuntime` could release a failed motion, let the trusted host start a new
task, then add the old motion's uncertainty to that new task from its ordinary
exception handler. Bind that final accounting to the existing admission task
ID. The inner cancellation fence still records uncertainty before ownership
release; pre-admission validation failures retain their existing behavior.

A controlled two-thread handoff reproduces one failure with two controls on
`3e696d4`; the fix passes 78 runtime, cancellation, conversation-stop and handoff
tests across four files (1.88 s). Preserve the first invalid file-selection
invocation as an execution error with zero tests. Sources and protected stores
remain unchanged during checks. No physical run or new physical admission.
Evidence: `benchmark/results/task_exception_epoch_20261003.json`.

### 2026-10-03 — Preserve TCP timeout coverage across caller scheduling

macOS CI on PR84 retained `unverified/reader_timeout` but failed a test-only
assumption that the independent reader makes exactly one TCP connection. A
controlled delayed caller reproduces that same assertion failure: the real
socket can expire and reconnect before `begin()` resumes to cancel its sampler.
Both ordinary and delayed schedules now require the original 40 ms timeout
verdict and allow only read-only hello/state messages. Runtime, clocks, limits,
measured-support gates and controller ownership are unchanged. The controlled
schedule does not claim to identify the historical OS scheduling event.

Retained causal baseline: 1 failed; corrected focal: 2 passed; affected support
and effect suite: 389 passed in 40.24 s. All 1,599 source inputs and four stores
are unchanged across validation. No native SDK, GPU, simulation or local macOS
execution. Receipt: benchmark/results/mobile_support_tcp_scheduling_20261003.json.

## 2026-10-03 — deterministic inert-distance geometry fixture

PR87 macOS job 111189738373 expected the measured-distance veto but reached the
unchanged wall watchdog first. The log does not identify producer cadence or
GC/OS causality. A controlled .013s automatic publisher with original .002s
simulation ticks reproduced that refusal (1 RED); explicit fixture stepping
completed the same .4s simulated command and retained measured distance zero
(1 PASS). Only this kinematic double changes: ten original ticks per read,
actual ACK/horizon assertions and no commanded-velocity travel credit.

All production sources, wall/freshness/lease/geometry bounds and original
negative verdicts remain unchanged. The walk-distance file and existing
stationary-clock, wall-stop, priority-stop/reset and interruption controls
passed 35 cases in 1.06s; 1621 sources and private-store checks remained unchanged.
See benchmark/results/walk_distance_inert_fixture_20261003.json. This CPU
control does not establish macOS rerun success or physical locomotion.
### 2026-10-03 — Preserve current Factory preparation PASS and readiness FAIL

Documentation-only follow-up on `06805d13`: source `cce8880`/model `69d9a46a`
passed zero-solve preparation, then failed ordinary readiness. Step303 was
rejected at controller acceptance with age 0.43165935698 s above 0.2 s; the
preceding raw quiet span 64–302 covers 0.3966666763 s below the required 0.5 s. Neither the raw
zero-spindle records nor a closed owner thread clears its fault or proves rest.
Retain natural scope exit 1, all three tracked own processes closed, six foreign
births unchanged and unchanged input inventories. No turn/reset/task requested;
no GC/GPU/scheduling cause inferred. Add only the three small byte-exact audits,
including the bound error extractor, and extend their copy manifest. Preserve
all 25 prior portable receipts and earlier failed episodes. Validation is limited
to evidence hashes, relative links and docs diff; no tests, SDK imports, model
inference, simulator, GPU query, action or new native run.


### 2026-10-03 — Factory owner fault containment

A backend exception whose `__str__` raises could bypass error retention in
`FactorySolveOwner._run`: four CPU probes incorrectly reported a clean close,
and two zero-upload probes abandoned finalization. Stop now latches before
formatting; a guarded fallback preserves the exception type, sticky failure and
pending admission rejection even when formatting raises `BaseException`.
Zero upload remains independently reported and never proves physical rest.

Baseline: 6 failures / 3 controls pass. The source-bound Factory/fastening
regression selection initially passed 389 tests. Independent review then
identified a string-subclass formatter escape (three retained red probes); the
final corrected source passed 393 tests, including primary-plus-zero failure
and stop-before-format ordering. These selections overlap. Sources and protected stores remained unchanged; an initial
invalid test-file invocation is retained separately. No native run or physical
admission follows. Receipt: `benchmark/results/factory_error_containment_20261003.json`.

## PR87 ARM balance fixture isolation — 2026-10-03

Scope: CPU software tests only. The ARM job on eb55cf91 reported one balance
classification as unverified instead of refuted; it did not record the reason.
An external source-bound control injects 80 ms of automatic cyclic-GC callback
work during scripted read four. Both zero-twist semantic cases then hit the
unchanged 40 ms reader budget. This demonstrates a fixture vulnerability, not
the cause of the original CI event. Reuse the existing `healthy_episode_gc`
fixture only for those cases, with direct late-reader negatives preserving the
original captured state and later healthy reads. No runtime, verifier threshold,
physics, hardware, native SDK, or GPU change. Validation receipt follows below.

Validation: the unmodified pair passed without injection, then both failed
`reader_timeout` under controlled automatic GC (95.035/81.414 ms read RTT).
The candidate passed both semantic cases with the existing fixture, while real
80 ms delayed reads still caused both expected refusals with GC disabled. The
normal affected-file plus fixture selection passed **336 tests in 33.60 s**,
including two new negatives checking retained capture timestamps and later
healthy observations. Ruff F/E9 and diff checks passed. All runs used private
stores, and source/protected-store hashes were unchanged. The retained CI failure
was **5699 passed / 1 failed / 269 skipped / 4 deselected**; no GC cause is inferred
from its missing verdict reason. See
`benchmark/results/mobile_balance_arm_fixture_20261003.json` for raw bindings.

## PR87 macOS measured-distance fixture schedule — 2026-10-03

CPU test-only follow-up from 1e9c8902. The macOS job rejected the positive 20 mm
kinematic fixture; its printed result omitted the internal error. A controlled
slow relative-wait producer reproduces refusal before sufficient toy travel,
without attributing this schedule to that uninstrumented CI run. The positive
geometry test now uses ten explicit original .002 steps per fixture read with
its automatic motion producer disabled, matching the existing inert geometry
control. ACK, post-admission baseline, stopping, real wall budget, and mock
unverified outcome remain unchanged. A held-reader negative exercises the real
wall watchdog independently of those explicit steps. No production code or
physical/native/GPU execution changes. Retained results and hashes follow.

Validation: the original pair passed normally, but with the controlled slow
producer both hit the unchanged 2 s wall deadline at ±14.8 mm measured travel.
The candidate pair passed with that same control, measuring ±20 mm from the
post-ACK baseline. Holding the original first admitted read for 2.1 s still
caused both wall-deadline refusals and no baseline/travel credit. The normal
selection passed **106 tests in 3.16 s** (distance control, mobile safety and
mobile base), including the event-driven blocked-read negative. Ruff F/E9 and
diff checks passed. Sources and all four protected stores were unchanged during
checks. The macOS failure remains retained: 5667 passed, 1 failed, 303 skipped,
4 deselected; its internal error/cadence were not recorded. See
`benchmark/results/walk_distance_positive_fixture_20261003.json` for bindings.
## Status addendum: instrumented Factory refusal and software follow-ups — 2026-10-03

Documentation-only follow-up from f3f357629e5c77fdcf4d2aef67f9eb0024673243.
Preserve all 28 existing portable receipts and their original bytes. Add the
source/model/epoch-bound Factory phase diagnostic: step 309 rejected at
441.3566 ms age under the unchanged 200 ms gate, quiet span below 0.5 simulated
seconds, no task. The same-thread generation-2 GC interval was measured inside
contact_records in this episode; this does not identify retained objects or
attribute prior failures. Natural administrative closure retains the owner fault
and does not verify readiness or physical stop. Copy the terminal audit and
bound error byte-for-byte, without raw trace/log expansion.

Also preserve the earlier software receipts and add PR90 task-epoch exception
accounting (92cc6c11, 78 CPU passes) and PR87 balance-fixture isolation (1e9c8902,
336 CPU passes). These overlapping selections are not summed or presented as
physical validation. This documentation task performs hash/link checks only;
no new tests, SDK/model imports, native execution, GPU queries, or publication.

Validation: all 28 previous copies are byte-identical; 32 portable receipt hashes,
70 referenced artifact hashes and 36 relative documentation links checked.
The same-thread GC interval was also compared directly with the retained raw
trace rows and shown nested inside contact_records. Only docs and WORKLOG change;
production, tests, scripts, configs and benchmark files remain byte-identical.


### 2026-10-03 — Verified owner and CI follow-ups

Document PR85 owner error containment at 5c21af4 (393 CPU passes), PR87 explicit
positive-distance fixtures at 40e36d90 (106 passes), and PR90 read-only TCP
reconnection fixture at 80035374 (68 passes). Selections overlap with earlier
receipts and are not combined. The existing native Factory failures remain
bound to cce88808; no new runtime physical result is inferred.

Preserve all 32 earlier portable receipts and add three byte-exact records.
Documentation-only source/hash/link review; no runtime, SDK, simulation or
shared-store changes. CI on newly published heads remains separately pending.
