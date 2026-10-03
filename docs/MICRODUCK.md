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
The [continuation report](MICRODUCK_CONTINUATION_20261002.md) records the current
original native MCP outcomes, software checks, retained failures and video provenance.

## Runtime and clocks

`MobileBase` supplies immutable physical state and bounded body-frame velocity
commands. `SafeBase` enforces limits, freshness, cancellation and progress.
`MobileRig` selects bases by name. `MobileSkillRuntime` records each execution
and its independently sampled postcondition in the normal CASCADE trace and
episodic memory. The MCP reader handles stop and cancellation while the worker
is inside a motion call.

The native bridge runs a floating-root articulation in a separate Isaac Sim
6.1 arena. The current executable backend is Newton 1.6 / MJWarp, using the
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
This change does not vendor meshes, USD or ONNX files. The later conversation
layer is specified in [the hosted conversation design](MICRODUCK_CONVERSATION_DESIGN.md).
