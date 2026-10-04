# MicroDuck mobile runtime

MicroDuck is an opt-in mobile base with a bounded velocity interface. It does
not instantiate an arm, inverse kinematics, grasping, or the kitchen scene.
The implementation is a candidate: general walking and turning have not passed
physical admission. The optional [geometric distance candidate](MICRODUCK_DISTANCE_CANDIDATE.md)
retains four historically confirmed native fresh-start ±30 mm receipts. A later
[observation audit](evidence/voice-reply-recovery-20261004/guarded-continuation.json)
finds heading drift after completion in both forward cases, exceeding the
unchanged 0.08 rad bound; those receipts do not admit the current contract.
Separate evidence covers a stop during movement and one small negative turn;
composed reverse motion and larger or positive turns still have retained failures.
PhysX BAM, scene reset, hardware, and hosted conversation
are not delivered by this locomotion change.

The implementation continues the [approved design](superpowers/plans/2026-10-01-microduck.md)
and the [dated review](MICRODUCK_DESIGN_REVIEW_20261002.md). That review describes
an earlier source snapshot; the current continuation adds inference cancellation
fencing, explicit solved support, and complete effective-model identity.
The [continuation report](MICRODUCK_CONTINUATION_20261002.md) preserves its
original native MCP outcomes and failures. The [current capability index](PROJECT_STATUS_20261003.md)
links later distance and conversation results, with their source/model limits.
The [modular architecture](ROBOT_MODULARITY.md) separates single-robot control
from shared-scene ownership; the ordinary native bridge still owns one robot.

## Runtime and clocks

`MobileBase` supplies immutable physical state and bounded body-frame velocity
commands. `SafeBase` enforces limits, freshness, cancellation and progress.
`MobileRig` selects bases by name. `MobileSkillRuntime` records each execution
and its independently sampled postcondition in the normal CASCADE trace and
episodic memory. The MCP reader handles stop and cancellation while the worker
is inside a motion call.

The native bridge runs a floating-root articulation in a separate Isaac Sim
arena. The current executable backend is Newton / MJWarp, using the
pinned native BAM implementation. The official `velstand.onnx` consumes 61
observations and emits 14 actions. Physics advances nominally every 5 ms and
the policy runs every four completed solves; the actual native timestep is
recorded. Camera capture and LLM latency do not advance the policy clock.
Neck and head joints belong to the policy; another controller must not write
them. The mouth is outside its action vector.

Inference previews do not mutate policy history. The bridge rechecks command,
generation, episode, physical clocks and wall deadlines before committing an
inference result. Stop invalidates walking intent while allowing the policy
and actuators to maintain balance. It does not certify an instantaneous
physical stop or promise to undo a motor target already committed.

The policy API and single/shared launchers offer the explicit
`target_profile="robotd-targets-v1"` / `--target-profile robotd-targets-v1`
candidate. It reproduces the gait output transform in
[robotd at 9136aa4](https://github.com/pollen-robotics/microduck/blob/9136aa4ee88e81edf2bcaf3527e90b65da25f1eb/robotd/src/control.rs#L607):
float64 HOME plus `0.9 * raw_action`, then target EMA with alpha 0.5 for the
four head joints and 0.7 for the ten leg joints. The first target after
construction/reset is unfiltered; subsequent targets use the last committed
filtered target. Raw float32 actions remain the observation history. Preview
and discarded evaluations cannot move either history. After `infer()` or
`commit()`, `committed_targets` returns the accepted target; `targets(action)`
always previews the next slot. `target_contract` binds the profile and policy
digest. The default `direct-v1` retains scale 1.0 and no filtering.

This is a target-transform candidate, not the complete robotd pipeline or its
separate standing/skill overrides. It does not add command smoothing, voltage
adaptation, stale-sensor coasting or actuator delay. Launchers keep `direct-v1`
as the default and reject a missing, changed or silently ignored target contract.
The effective model binds this selection separately from the policy weights.
Its `target_upload` record distinguishes calculation precision from the existing
BAM upload: filtered targets retain float64 history, then `set_targets` converts
them to float32 before Warp assignment. No native result is transferred between
profiles. The current official VelStand LFS digest
is the same `1c659be5…` used by the retained eight-case negative corpus; its
published provenance names `protective_fall` and a September 14 export but
does not identify the exact training run. CPU formula replay cannot predict
the new closed-loop actions or establish improved gait or stopping.

The [native target-transform foundation](evidence/robot-modularity/microduck-robotd-target-foundation-20261004.json)
retains one robot at source `2c27f846`, with 800 completed solves, 200 policy
commits, nine camera/support samples and two reader events. The independent
auditor reproduced every raw-action history and float64 target, then checked
all 800 float32 BAM uploads. No preview was discarded. Maximum displacement
from the first solve was 2.6421 mm, maximum tilt 0.0206804 rad, and minimum
sole support over the final two simulated seconds was 7.2072 N. These zero
caller commands still caused real postural actuation; they do not admit walking,
turning, braking or a verified stop. Both weights and target transformation
differ from the prior rough/direct reader. Native and owned scope exited zero,
all seven observed process births disappeared, and inputs and protected stores
were unchanged. Owner-thread attempt duration was 11.580 ms median and 109.550 ms maximum;
no generation-2 GC occurred, so this is neither a sustained deadline guarantee
nor a causal speed comparison under the recorded foreign GPU occupancy.
The separate [Newton BAM proposal](https://github.com/newton-physics/newton/issues/4397)
motivates reviewing actuator contracts; this episode retains its pinned BAM
plant and does not test a new friction, backlash or delay implementation.

The subsequent [single-robot forward attempts](evidence/robot-modularity/microduck-robotd-forward-failures-20261004.json)
retain two failures with this same source and model. The first omitted the
composed-runtime tag and failed its initial MCP observation before reset or
motion. After a passive composition regression and a separately reviewed
harness correction, the second reached the real `walk_distance(0.025)` path.
It reported 10.6337 mm before its configured three-second simulation deadline,
below the original 25 mm target with 5 mm tolerance. The full-rate journal
never entered that tolerance, including the retained tail: maximum forward
travel was 10.6421 mm, with no heading/lateral or forbidden-support veto.
The client therefore withheld the scored stop/rest phase; containment ACKs
do not establish physical stopping.

Each episode retained 2,000 solves, 500 policy commits and 21 camera/support
samples, with ordinary native/scope exits and closed clients. In the second,
150 policy rows carried the observed forward command and every raw-history,
filtered-target and float32 upload was reproduced. The failure envelope and
both traces omitted the admission ACK; the original auditor failures remain
preserved. A separate supplement joins the actual command, controller and
physical records without reconstructing that missing receipt or granting
admission. Per-joint target/tracking/torque statistics are diagnostic only.
Neither episode exercised generation-2 GC, establishes causality for the
target transformation, or admits locomotion or a verified physical stop.

The separate [headless MuJoCo prefix](evidence/robot-modularity/microduck-mujoco-prefix-negative-20261004.json)
retains one negative with the same VelStand weights and target transformation.
It stopped at solve 10 (0.05 simulated seconds), after three policy commits,
before any forward command. The frozen preparation gate rejected constraint
contacts between the floor and both foot housings; each had zero normal force,
while the soles carried 23.4769 N. This gate applied throughout preparation,
whereas the historical CPU corpus checked support at its end and native
geometric control began later. The same housing pairs, also at zero normal
force, occur at the corresponding retained native step 12. This result therefore
identifies a difference in preparation criteria, not a demonstrated fall or
backend cause of the earlier walking failure.

The first 61-element observation, raw actions, float64 filtered targets and actual
float32 BAM uploads matched the native prefix exactly; the next two policy
commits differed. Source XML/meshes and primary physical fields were bound,
but compiled mesh equivalence and solver equivalence were not established.
The episode exited with code 1 and both observed process births absent,
unchanged inputs, and no signals or forced closure. The original negative and
all contacts remain retained. No active-motion, tail, rest or verified-stop
window was reached, and no retry or parameter adjustment followed.

The [bootstrap A/B diagnostic](evidence/robot-modularity/microduck-bootstrap-ab-20261004.json)
completed two solves using one pinned MuJoCo model,
two independent `MjData` states and the retained first policy target, without an
ONNX session or inference. Case A reproduced the original first actuator and
completed-state rows exactly. Case B changed only the external-force argument
of the first BAM friction calculation to zero; its original force buffers,
motor calculation, target, initial state, solver and timestep were preserved.

Within this CPU setup, that substitution changed the maximum joint velocity by
0.00537643258 rad/s and the friction budget by 0.00246892206 Nm. Against the
retained Newton first row, the maximum q/dq/friction residual fell from
2.68850674e-5 rad / 0.00537642489 rad/s / 0.00246892205 Nm in A to
1.37083754e-8 rad / 3.01011785e-8 rad/s / 1.56197433e-11 Nm in B.
The original CPU external input is now observed directly: right-hip-roll
received 0.030560578339 Nm before the substitution. This establishes the effect
of that input on this CPU first solve. The small residual against Newton does
not establish equivalence of integrators, compiled geometry or later dynamics.

The owned scope exited zero, without signals, forced termination or remaining
processes; both observed births disappeared. This diagnostic exercised no
DRIVE interval, sequential rollout, locomotion task or physical stop/rest
window. The earlier failed walk and preparation gate remain failed. No
production physics setting or locomotion controller changed.

The subsequent [effective-options capture](evidence/robot-modularity/microduck-effective-options-20261004.json)
observed `implicitfast` (3) in both the CPU MuJoCo and Warp models under the
original native recipe. Their other options differ: CPU timestep 0.002 s and
tolerance approximately 1e-8, versus Warp timestep 0.004999999888241291 s and
tolerance approximately 1e-6. Agreement applies only to the integrator. The
solver configuration had no integrator override; the imported Newton model
held 3. USD exposed `euler` as an unauthored fallback with no property stack,
which did not select the effective integrator.

Two identical snapshots retained the same seven native model/solver/state
objects and step-2 clock. The original startup performed two native bootstrap
solves, then initialized HOME/FK and bound/reset BAM; the snapshots followed
that initialization. No policy, command endpoint or episode solve ran. The
complete identity recipe matched foundation02 except for the absent episode
profiler label. Native and scope exits were zero, all eight observed births
were absent, and no signals or forced closure occurred. This measures the
current recipe, not the historical gait's effective options, and does not
establish an integrator cause, training alignment, locomotion or physical rest.

## Evidence and identity

A SHA-256 digest binds the complete admitted bundle, consumed USD layers,
normalized runtime scene, ONNX weights, native BAM source and effective
parameters, actual solver/device/timestep, runtime source, and contact
registry. A client must explicitly pin this digest in
`model_identity_sha256`; learning it from an arbitrary `hello` is not admission.
State, camera, and independent truth channels enforce the same binding.
Changing the recipe requires a new identity and physical evaluation.

Support uses versioned, complete post-solve contact records: exact shape
pairs, points, normals and reaction forces in world coordinates, bound to
the same epoch, step, time and model identity as pose. The independent checker
uses an explicit registry of sole and ground shapes. Empty contacts do not
prove standing; unavailable forces remain unknown. Stationary windows require
upward foot support and reject forbidden external body support. Walking phases
are evaluated separately from rest windows.

An emergency-stop acknowledgement always reports physical verification as
pending/unverified. A separate observer can later confirm or refute that stop,
with the same receipt ID. It cannot repair the outcome of an earlier failed
motion. Late, pending and duplicate observations cannot supply positive settling
evidence, but their faults, residual motion and support failures still veto an
earlier quiet window. See [stop verification and native finalization](MICRODUCK_STOP_VERIFICATION.md)
for this distinction, signal handling and provisional SDK-shutdown receipts.
`reset_stop` only restores command permission; it does not reset the
physics world or prove a new scene epoch.

## Running the candidate

Offline smoke with the kinematic mock:

```sh
PYTHONPATH="$PWD/src" CASCADE_BASE=microduck_mock CASCADE_LLM=mock \
  python -m cascade.apps.mcp_server
```

The mock exposes the mobile tool surface and explicitly cannot verify physical
motion. Mobile tools include `get_base_state`, `walk_velocity`, `turn`,
`stop_navigation`, `emergency_stop`, `reset_stop`, and `verify_last_action`.
`camera_snapshot` is available when a camera channel is configured.

For native execution, `scripts/isaac_microduck_bridge.py --help` lists required
paths, hashes, BAM profile, private port, limits and episode deadlines.
`--check-only` verifies offline inputs without opening Kit or a socket. Use
the selected Isaac release's `python.sh`, its matching native dependencies,
a new output directory and an outer process deadline. `BRIDGE_LISTENING`
means transport availability, not locomotion acceptance.

`configs/bases/microduck_isaac.yaml` is an incomplete candidate template.
A reviewed run profile must supply actual engine/device, bundle-root and policy
hashes, the independently inspected model digest, private endpoint, explicit
support registry and verifier limits. There is no default admitted recipe.
Do not change gains, checkpoints, speed limits or verifier tolerances merely
to turn a failed campaign green.

The bridge saves startup admission, the canonical model identity, runtime
receipt, every physical step, actual policy inputs/outputs and discarded
inferences, and images with episode/step timestamps. Check process exit and
owned-resource teardown separately from the receipt. Keep failed episodes.

Isaac Sim compatibility is selected explicitly with `--sdk-recipe`; the
supported dependency versions and source checks live in
[`sim/microduck_sdk.py`](../src/cascade/sim/microduck_sdk.py). Offline checks
read files only, and initialization rechecks the modules actually imported.
The selected dependencies and solver outputs enter the effective model
identity. A supported installation does not establish balance or locomotion:
each new combination still requires its own single-robot native validation.

`--private-rtx-cache` optionally redirects only the two writable RTX shader/PSO
caches to `rtx-cache/` inside the new output directory. It requires the exact
`isaac62_48b2d951` recipe and checks the cache-routing SDK sources. The default
keeps SDK cache selection. Bundled application caches, rendering, physics and
deadlines are unchanged. Kit receives the paths before initialization; effective
settings and loaded sources are checked before the bridge can publish.

An optional `--rtx-cache-seed DIR --rtx-cache-seed-sha256 SHA` requires a separately
reviewed manifest from a successfully closed warming probe. `DIR` contains only
`manifest.json` and `data/`; the manifest binds the exact cache policy and the
complete file/directory inventory returned by
[`private_rtx_cache.inventory`](../src/cascade/sim/private_rtx_cache.py).
Its schema is `cascade.private-rtx-cache-seed.v1`, with `policy` and `inventory`
fields. Admission rejects extra, missing or changed members, symlinks and special
files. Bootstrap rechecks and copies the seed into an exclusive writable directory,
then verifies the seed and copy again. The two absolute paths and input inventory
are retained in `rtx-cache.json`; measured settings go in `rtx-cache-effective.json`.
Only the relative cache policy and implementation hashes enter model identity,
allowing independently admitted probe/episode paths to differ. Seed bytes remain
separately bound deployment inputs.

The [native cache validation](evidence/microduck-private-rtx-20261004/native.json)
retains the failed SDK-origin import and the separate one-step readiness timeout.
A renderer-only preparation subsequently closed normally in 166 seconds. Its
verified cache copy admitted source `eac66e6` to an 800-step zero-command run:
200 policy commits, 41 captures, bridge readiness at 10.7 seconds and native
duration 18.4 seconds, within the original 180/240-second limits. Both writable
destinations and loaded SDK origins matched their admissions; native and scope
exits were zero, with no shutdown signals or changed inputs. This supplies a
new source-bound model identity for the next episode. Balanced rest, commanded
locomotion and the spoken reply still require their own verification; one run
does not establish general startup reliability or a causal speed improvement.
The subsequent [VOICE04 episode](CONVERSATION.md) obtained the real walk/stop
pair but correctly refuted the walk for heading drift after completion. Its
fallback stop acknowledgement and closed resources did not establish the
requested rest-and-spoken-reply cycle.

## Remaining admission work

The current real MCP campaign confirms supported standing and a separately
observed stop. Forward/reverse tracking is refuted and both turns expire.
Interruption and disconnect invalidate admitted commands; braking from an
established gait remains unproved. Small forward and reverse commands also
failed tracking in an earlier upstream reference. The
[startup research](research/microduck-policy-startup-2026-10-02.md) reports a
related upstream issue and its limits; the
[BAM follow-up](research/microduck-bam-2026-10-02.md) distinguishes the pinned
actuator from incompatible newer fits. Neither establishes a remedy.
The [full-mjlab diagnostic](research/microduck-fullenv-startup-2026-10-02.md)
also records negligible +0.1 m/s motion from rest and after a +0.3 phase,
with the same official policy and unchanged production limits.

Required acceptance includes current-source equilibrium, forward/reverse
tracking, turns, interruption, disconnect, reset, repeated starts, and video
from the same episode. Each advertised engine needs its own evidence.
PhysX currently lacks the separate previous-step external generalized load
required by the chosen BAM contract; tested friction and contact primitives
alone do not provide that contract. The native CLI rejects PhysX explicitly.

Models remain external: their notices declare BY-SA-NC without a version;
code, weights and the BAM implementation retain their separate notices.
This change does not vendor meshes, USD or ONNX files. The implemented
[conversation gateway](CONVERSATION.md) retains the ownership boundary from the
[earlier hosted conversation design](MICRODUCK_CONVERSATION_DESIGN.md).

### Shared-scene implementation boundary

`sim/microduck_shared.py` binds explicit robot namespaces to disjoint native
coordinates, body/shape identities and a single world. Its coordinator previews
all independent policies without changing their history or native targets, then
checks the common completed clock and each permission under a shared memory
fence. Only an admitted group commits its histories, applies the owned BAM groups
and requests **one** solve. A changed command before that fence discards the whole
preview group and retries once against the same completed state. Repeated
revocation withholds the tick without consuming a step or publishing a frame;
wall, signal and freshness guards still apply on each attempt. Attempt/evaluation
records include discarded work, while solve, history and BAM counts do not.
A stop after admission applies to the next regular control slot: the in-flight
solve and native delay cannot be rolled back. A partial commit or native fault
still contains the scene; stop acknowledgements do not prove physical rest. All solved contact
rows remain visible to each robot's existing support checker, so another robot
cannot count as ground. Model identities and command epochs remain per robot.

The shared owner captures and validates global state/contact channels once for
each completed scene, then detaches each robot's joint and body observations.
The capture binds the model, current state, contact buffer, solver data and
completed clock before reading; a changed binding rejects the entire capture.
No snapshot is reused across solves, and each robot still receives the complete
contact evidence. CPU tests check channel read counts and exact data equivalence;
they do not establish a native speedup.

Within that completed scene, support is decoded once into immutable contacts
and rebound to each robot's model and epoch without reconstructing every
contact. Only transitively immutable records can be shared; mutable legacy
values retain copy isolation. Public observations and physics logs preserve the
complete dictionary schema, while each reply remains detached. Layout, shape,
clock and source-admission checks still gate reuse; permission and age remain
live on every controller read. This reduces repeated CPU work and does not
establish a control deadline or physical-stop result for twelve robots.

The [retained support diagnostic](evidence/microduck-shared-20261004/support-profile.json)
completed 800 solves, 2,400 policy commits, nine overview/support-probe pairs and
24 reader calls with twelve robots; all owned processes and clients closed
naturally. Owner-attempt durations had a median of 114.911 ms, p95 of 130.478 ms
and maximum of 655.533 ms. **22 attempts exceeded 400 ms**, so the original
control deadline remains unresolved. This run sent no stop, reset or motion
commands and does not supersede the retained twelve-robot control failure.
These times describe this episode; they do not establish a causal speedup.
The earlier prelaunch refusal remains recorded alongside the completed run.

The subsequent [pinned zero-command trial](evidence/microduck-shared-20261004/support-control.json)
also completed 800 native solves, but **its twelve-robot control sequence failed**.
The first robot completed its zero-command execution; the second timed out while
requesting a fresh preflight state, before sending its command. Its stop was
acknowledged, the client closed, and the remaining ten controls were not attempted.
All physical outcomes remain unverified. The original 0.5-second RPC and
0.4-second no-progress limits were retained; a 691 ms publication gap crossed
the failure window. The timing evidence does not establish the cause.

The shared stepper also retains immutable copies of the full legacy contact
pairs and constraint addresses for that solve. Public native, robot-view and
controller reads still return fresh lists; the private recording path keeps
the same JSON arrays, order and bytes. Nonplain legacy values retain ordinary
deep-copy isolation. This removes repeated list construction within a solve,
without changing GC settings, validity checks or the public state contract.

CPU tests cover 1, 2 and 12 synthetic participants, including index permutations,
command isolation, cancellation and contact vetoes. The ordinary one-robot
entrypoint retains its control cadence.

`scripts/isaac_microduck_shared.py` now supplies a bounded native foundation:
one shared scene, separate ONNX/BAM histories, disjoint robot views, a common
completed-step overview and per-robot content identities. It takes the native
bridge's explicit input flags plus `--robots 1..12 --spacing 2`. Use `--port 0`,
`--solver-cuda-graph --reuse-solved-read` and the explicit SDK recipe; this
foundation opens no command sockets. Every run needs a fresh output directory
and an outer process deadline. Shared solver storage scales with robot count;
contact observations retain every solved row.

The [retained native evidence](evidence/microduck-shared-20261004/foundation.json)
records separate source-bound runs with **1, 2 and 12 robots**. Each completed
800 global physics steps (4 simulation seconds), 200 policy evaluations per
robot, and 9 overview/force-probe pairs. Commands stayed zero, no robot fell,
clocks and identities matched, and owned processes closed with native exit 0.
The twelve-robot run retained measured sole support for every robot, with
maximum XY drift below 2.8 mm and maximum tilt below 0.014 rad. These are
initialization/balance results, not walking, fleet task or real-time acceptance.
The earlier twelve-robot attempt failed closed on insufficient contact storage
and is retained alongside its corrected retry.

An opt-in candidate, `--serve-base-port 0`, exposes separate loopback endpoints
after the first completed state and matching overview; a positive value chooses
the first port in a consecutive range. `BRIDGE_LISTENING.json` maps robot IDs to
ports and bound identities. Each endpoint retains the existing reader, control,
renew and stop channels. Commands and resets wait in a bounded per-robot queue
until the simulation owner drains it before a tick, preserving their original
receipt deadline. Expired, cancelled or closed requests cannot run on a later
tick. A nonblocking control-channel guard rejects observed EOF or invalid
pipelining before admission, after the mutating callback and before the reply;
discarding an admitted reset restores its latch. Stop bypasses the queue and
withdraws pending commands immediately; a stop crossing pure preview invalidates
that group without faulting healthy peers. Stop ACKs do not
prove physical rest. CPU/TCP regressions cover these contracts, partial listener
startup rollback and shutdown with pending commands.

The [endpoint evidence](evidence/microduck-shared-20261004/endpoints.json) records
fresh identity probes with **1, 2 and 12 robots**, each completing 800 shared
solves, 200 policy commits per robot and 9 overview/force-probe pairs. Separate
1- and 2-robot episodes completed state/camera reads, stop, reset and a zero
velocity command through `FleetRuntime -> RobotRuntime -> locomotion -> SafeBase
-> IsaacBase`. The two-robot run recorded two naturally discarded previews at
one permission change; the retry committed against the same completed state.
Adversarial timing barriers were exercised only in CPU tests.

The twelve-robot control episode also completed all 800 native solves with known
support and no falls, but **its control sequence failed** on the first robot:
the admitted zero command received feedback through step 11, then exceeded the
unchanged 0.4 s progress deadline. The measured publication gap to step 12 was
0.705 s; the stop latch was acknowledged and the remaining eleven commands were
not attempted. Native completion does not make this a control pass. The evidence
retains this failure, the earlier preparation-race failure, the earlier
794-step wall timeout and all prelaunch refusals. Completed native scopes closed
naturally; input files and protected stores remained intact. No performance
comparison is claimed across changing GPU occupancy.

For a separately admitted diagnostic, `--profile-phases` records inclusive
owner-thread spans in `timing.jsonl`: completed reads, support/native capture,
policy preparation, BAM, solve, publication, recording and camera/probe work.
Rows retain the attempt's completed clocks and solved/withheld/error outcome,
plus the preceding profile write/flush cost; the footer retains the final cost.
Nested durations overlap and must not be summed. Reader calls are not wrapped.
The flag installs no wrappers when disabled and is bound in the effective model,
so enabling it requires a fresh identity probe and pin. Profiling keeps the
existing control gates, cadence, contact evidence and physical records.

The [retained twelve-robot phase diagnostic](
evidence/microduck-shared-20261004/phase-profile.json) completed 800 solves,
2,400 policy commits, 24 reader events and 9 overview/support-probe pairs, then
closed naturally with exit 0. Median attempt duration was 343 ms (p95 831 ms).
Inclusive per-attempt medians were 82 ms for preparation, 101 ms for completed
state validation, 79 ms for publication and 55 ms for constructing/writing the
physics record. These overlapping categories are not additive. Large pauses
occurred in several phases; their cause remains unresolved. The first 160
attempts overlapped the recorded scope of a separate RTX probe on GPU0. This
diagnostic neither admits twelve-robot control nor establishes a native speedup.

Physics rows share only private contact projections during one JSON encoding;
public controller responses remain detached. This moves projection work into
`write.physics.jsonl`, within the inclusive `record.physics` and attempt spans.
Compare those enclosing spans for total work: the write span includes encoding,
so its change alone does not measure file I/O or GC cost.

The phase profile also retains passive Python GC callbacks from all threads,
with absolute timestamps, event IDs, generation and collection counts. Each
attempt and the footer drain a persistent queue bounded to 4096 events, including
events outside owner attempts; starts and stops may cross rows. Cumulative ticket
counts expose overflow or callbacks still in flight, without waiting for them.
Missing/error events or a removed callback prevent complete correlation claims.
Draining callback metadata is outside the attempt and write/flush durations.
GC settings are only observed, and closing the profile removes only its own
callback. Callback intervals include scheduling
and other callbacks; they do not measure isolated CPU cost or establish a cause
for a control timeout. Profiling does not disable or force collections.

The trigger diagnostic also copies at most 16 Python frame locations on each
generation-2 start, with paths and function names limited to 512 characters.
It retains no frames, locals or source lines; absent Python callers and truncated
stacks are explicit. This identifies the observed trigger, not the heap scanned.
Collector counts/statistics and allocated-block counters bracket callback
registration; they are not a heap-object census or SDK-startup measurement.
Stack and counter capture durations are recorded, but exclude callback dispatch,
queue and encoding overhead. They do not measure the observer's entire cost or
justify subtracting that cost from GC intervals. The recipe uses a distinct
profile identity, and counter/capture errors prevent complete attribution.

The [retained trigger diagnostic](evidence/microduck-shared-20261004/gc-trigger.json)
completed 800 shared solves, 2,400 policy commits, 24 reader events and 9
overview/support-probe pairs, with normal closure and complete callback
accounting. All ten generation-2 starts had stacks: eight in deep-copy paths,
one in plain-record tuple construction and one in support validation. Eight
stacks reached the 16-frame limit. These are observed trigger sites, not evidence
of which objects were scanned or what caused the pause.

Stack extraction took 0.020 ms median and 0.024 ms maximum, excluding the other
observer costs described above. Generation-2 intervals remained about 475 ms
median; ten attempts exceeded 400 ms, with a maximum attempt of 615 ms. This
reader-only diagnostic did not test control recovery, change collector settings
or alter the three retained twelve-robot control failures.

The [legacy-contact sharing reader](evidence/microduck-shared-20261004/legacy-contacts.json)
completed the same 800 solves, 2,400 commits, 24 reader events and 9 capture/probe
pairs with normal closure. Attempt durations were 83.40 ms median, 91.54 ms p95
and 595.44 ms maximum. One attempt exceeded 400 ms, overlapping a 508.06 ms
generation-2 interval; the preceding trigger reader retained ten such attempts.
Its thirteen-frame stack points to `dataclasses.fields` during private physics
JSON projection. This observed trigger does not identify the heap scanned.
The integrated source includes the optional RTX-cache feature at its inactive
default. Both GPU admission cohorts were empty, but host load was uncontrolled;
these separate episodes establish neither causal speedup nor a control deadline
guarantee. The original control limits and earlier failures remain intact.

The [subsequent fixed twelve-robot zero episode](
evidence/microduck-shared-20261004/legacy-control.json) completed all 72 client
events: stop, reset and one zero command per robot, with `execution_ok=true`
and physical outcomes `unverified`. It completed 800 solves, 2,400 policy
commits and 9 capture/probe pairs, then closed normally. Two naturally
invalidated preview cohorts discarded 24 evaluations and retried against the
same completed state; there were no withheld solves or extra history commits.

Its only generation-2 interval started 31.84 s after the final command result
and 31.72 s after the final cleanup stop ACK. The longest attempt was still
575.06 ms, so this protocol pass did not exercise control through that pause or
establish worst-case deadlines, physical rest or gait. The episode kept the
original 0.5 s RPC, 0.4 s progress and 0.5 s state-age limits, with no favorable
retry. The three earlier twelve-robot control failures remain retained.

Two [reader-only GC profiles](evidence/microduck-shared-20261004/gc-serialization.json)
each completed 800 solves, 2,400 policy commits, 24 reader events and 9
overview/support-probe pairs, with normal native and client closure. The second
used the plain-record serializer; its effective recipe changed only that source
file. The recorded attempt durations were:

| Profile | Median | p95 | Maximum | Attempts over 400 ms |
| --- | ---: | ---: | ---: | ---: |
| GC baseline | 117.15 ms | 161.18 ms | 646.41 ms | 22 |
| GC with plain-record serialization | 110.75 ms | 127.60 ms | 655.07 ms | 22 |

Both traces had complete callback accounting. Generation-2 GC intervals
overlapped all 22 slow attempts in each run; this establishes temporal overlap,
not an isolated CPU cost or a causal speedup. The comparison started after the
separate voice scopes closed. Both launch admissions recorded the same foreign
GPU process identity.
The original 0.5 s RPC and 0.4 s progress limits remain unchanged and unresolved
for twelve-robot control. These profiles requested only observations; the
earlier control failures remain retained.

The subsequent [private-row encoding profile](
evidence/microduck-shared-20261004/row-serialization.json) also completed
800 solves, 2,400 policy commits, 24 reader events and 9 overview/support-probe
pairs, with normal closure. Attempt durations were 105.59 ms median, 116.03 ms
p95 and 662.65 ms maximum. Ten attempts exceeded 400 ms; each overlapped a
generation-2 GC interval, with complete callback accounting. The preceding
serializer profile retained 22 such attempts. GPU admission changed from a
foreign process to an empty cohort, so this comparison does not establish a
causal speedup. The projection work moved inside the write span as described
above. The remaining long attempts still exceed the original control limits;
this reader-only episode grants no control or physical-stop acceptance.

The [subsequent bounded control episode](
evidence/microduck-shared-20261004/row-control.json) **failed** during `duck02`'s
reset of the stop latch: its reply exceeded the unchanged 0.5 s RPC limit.
`duck00` and `duck01` completed zero commands with unverified physical outcomes;
no walk was dispatched for `duck02`, and the remaining nine control sequences
were not attempted. The failure window overlapped a 605 ms attempt and a
generation-2 GC interval; the measured publication gap was 601 ms. Timing overlap
does not establish isolated GC cost or a cause.

All 800 native solves, 2,400 policy commits and 9 overview/support-probe pairs
completed, and client resources and native processes closed normally. The
`duck02` cleanup stop lacked an ACK because its owner channel was disconnected,
with delivery marked uncertain. Completed states show its latch set from step
21 onward; they do not reconstruct whether reset ran between states or establish
physical rest. This remains a control failure despite native completion, and
both earlier twelve-robot control failures remain retained.

These endpoint outcomes remain physically unverified. Walking, fleet tasks,
shared-space interactions and measured individual/global motion stops still need
their own acceptance evidence. The [original twelve-robot overview](
evidence/microduck-shared-20261004/endpoints-twelve-zero.jpg) comes from the
read-only probe at completed step 802; every velocity command stayed zero.

The overview uses a widely separated grid; its small floor display
rectangle does not describe the extent of the physical infinite plane.
