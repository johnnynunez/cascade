# Manipulation and assembly integration

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
