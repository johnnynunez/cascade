# Bounded grasp feasibility search

Native06 (`b28ebc3`) placed the green cube, then stopped before moving for the orange. The orange GGX response contained 400 diffusion poses: 167 passed the existing score filter, 10 passed the existing vertical-approach filter, and only one passed both. That pose and its symmetric twin intersected observed surfaces on the approach. A historical candidate still passed on the native06 frame; the observed-scene gate did not exclude every possible grasp.

Four predeclared offline samples (seeds 60601–60604) reused the exact 2,711-point native06 cloud, model checkpoints, sweep parameters and 400-pose diffusion request. They produced 4/2/8/2 eligible candidates. The unchanged selector, memory prior, IK, carry check and approach/closing gate accepted two batches and rejected two. All four results, including failures, are recorded in [model evidence](../benchmark/results/grasp-planning-resample06-model.json) and [selection evidence](../benchmark/results/grasp-planning-resample06-selection.json). The original response reproduces the native rejection exactly. These are offline samples, not a success-rate estimate or physical acceptance; production does not choose their seeds or reuse historical coordinates.

The [actual runtime replay](../benchmark/results/grasp-planning-runtime-replay06.json) executes the new SkillRuntime, backend conversion and all guards against the original rejected response followed by seed 60601. It rejects batch one, selects batch two, preserves the cloud/prior/deadline and stops at a local callback before the first open command; it has zero live transport or actuator calls.

The calibrated GPU Isaac profile now permits at most three planning batches under a single eight-second wall deadline, enabled only with the observed-finger gate. The deadline starts before the first backend call and is capped by any enclosing pick/persistence/task deadline. Each batch uses the same ObjectFix, frame, masks, transforms and frozen memory prior. The same model filters, nudge, floor adjustment and geometry checks apply. There are no jaw commands, re-homing, fresh perception or memory updates between batches.

Only explicit feasibility exhaustion permits another batch: no executable selector candidate, or a valid nonempty learned response removed by the unchanged filters. Empty, malformed, non-finite, non-rigid, wrong-provenance and transport/model responses terminate the attempt. Bounded mode validates numeric wire shapes before casting and all rigid poses before filtering, including low-scored poses. Its float32 rotation tolerance is 3e-6, matching the existing observed-camera transform validation; it does not change a collision margin.

Send and receive share one deadline; polling checks cancellation without robot reads. Cancellation or timeout discards the socket so a delayed response cannot become the next batch's reply. Explicit state checks between batches keep the original gate/epoch and use the smaller of remaining planning time and the existing motion RPC cap (one second by default). The original halt generation survives every batch. Checks before and after IK/validation reject late results, and the final pre-open state/geometry check remains inside the planning deadline. A native IK call cannot be preempted by these Python checks; it can overrun the wall deadline but its late result never authorizes an actuator.

The planning deadline ends immediately before the first authorized open command. Physical motion keeps its existing clock, duration, velocity, settling and deadline contracts. Exhaustion preserves the failure and the gate's prohibition on automatic recovery. Telemetry records batch numbers, filtered counts, every raw response, rejection reasons and the selected batch. Seeds are explicitly unrecorded for the production server; the retained raw responses support replay.

The [endpoint occlusion correction](observed-finger-occlusion.md) checks each
symmetric wrist orientation independently, including a twin whose original
orientation was rejected. Both run the complete IK, harness and observed-scene
checks. Ranked candidate order and the preference for shorter joint travel when
both orientations pass remain unchanged. This adds no planning batch or deadline
extension and performs no actuator command while alternatives are examined.

This contract is merged; source-bound physical results and current acceptance
are listed in [project status](PROJECT_STATUS_20261001.md). Rebinding feedback does not refresh the observed scene, certify occluded space or provide continuous braking during gripper closure.
