# MicroDuck mobile runtime

MicroDuck is an opt-in mobile base with a bounded velocity interface. It does
not instantiate an arm, inverse kinematics, grasping, or the kitchen scene.
The implementation is a candidate: general walking and turning have not passed
physical admission. The optional [geometric distance candidate](MICRODUCK_DISTANCE_CANDIDATE.md)
has four confirmed native fresh-start ±30 mm cases and a verified stop during
movement, plus one small negative-turn case; composed reverse motion and larger
or positive turns still have retained failures.
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
Nested durations overlap and must not be summed. Reader threads are not profiled.
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

These endpoint outcomes remain physically unverified. Walking, fleet tasks,
shared-space interactions and measured individual/global motion stops still need
their own acceptance evidence. The [original twelve-robot overview](
evidence/microduck-shared-20261004/endpoints-twelve-zero.jpg) comes from the
read-only probe at completed step 802; every velocity command stayed zero.

The overview uses a widely separated grid; its small floor display
rectangle does not describe the extent of the physical infinite plane.
