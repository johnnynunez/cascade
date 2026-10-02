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
