# MicroDuck / Hermes mobile architecture review — 2 October 2026

Continuation: the implementation has since been imported from a preserved
handoff into the repository. [MicroDuck runtime documentation](MICRODUCK.md)
describes the current contract. The findings and counts below retain their
original snapshot scope; they are not assertions that every finding remains
open in the later implementation.

**Assessment: the design direction is suitable for approval; complete physical integration remains pending.** The reviewed work already includes a mobile implementation and native Newton trials. The earlier description, “design only, not implemented,” no longer describes that external worktree. The evidence does not establish a complete PhysX and Newton delivery, accepted locomotion through Hermes/MCP, or a completed general humanoid controller.

This document reviews a collaborator's implementation separately from the code shipped in this repository. The external corpus root is `/home/johnny/Projects/demo/cascade-lab/MICRODUCK`; all source and receipt paths below are relative to that root unless linked explicitly. These paths identify reviewed external files and are not links to committed implementation in this repository.

The review read that worktree without modifying it, running simulations, or intervening in its processes. The snapshot was recorded at `2026-10-02T01:44:55.358251+00:00`: branch `feat/microduck-isaac`, HEAD `5359405a607a14e9d30182309c40d30c3b95f80b`, four modified tracked files, and numerous uncommitted modules. HEAD identifies the base revision, **not the implemented file contents**. The [source and receipt hash manifest](evidence/microduck-review/source-receipt-hashes.json) identifies the reviewed sources and receipts, their timestamps, and read stability. The collaborator continues working; these findings apply exclusively to the recorded hashes. Later corrections require new review and evidence.

## Implemented architecture and its scope

`DESIGN.md` separates Hermes/MCP, mobile capabilities, SafeBase, policy, and physics, with an independent reader judging physical effects. This composition exists: `cascade/src/cascade/apps/demo.py:299` selects the mobile path before constructing arms, and `cascade/src/cascade/apps/mobile_runtime.py` constructs MobileRig, SafeBase, memory, traces, and verifiers. A mobile runtime does not need an invented tool center point, gripper, or arm kinematics. Its MCP catalog derives from capabilities and rejects unavailable manipulation.

`cascade/src/cascade/skills/mobile_runtime.py:32` exposes `walk_velocity(vx, vy, wz, duration_s)` and `turn(angle_rad)`, with optional base selection, observation, memory, and stop operations. `turn` uses the profile's speed; it does not introduce another joint writer. The normal catalog contains eleven MCP tools; `task_done` belongs to the agent runtime, and camera exposure is conditional. The implemented transport is loopback TCP JSON with `hello`, `state`, `command_velocity`, `renew`, `stop`, `reset_stop`, and optional `frame` operations (`cascade/src/cascade/sim/mobile_bridge.py:456`). It is distinct from the upstream UDP body server.

The implementation checks identity and epoch, permission generations, limits, wall-clock expiration, physical duration, priority stop, and observation after acknowledgment. `stop_navigation` and `emergency_stop` invalidate commands and request zero velocity while preserving the balance controller. Neither operation means torque removal or proves physical rest. `reset_stop` only releases permission: it neither resumes the previous command nor resets the world, and it rejects latched faults. Fault recovery belongs to the simulator owner, with a repaired scene and a new epoch, outside the public tool (`cascade/src/cascade/sim/mobile_bridge.py:333`). Stepper containment pauses the backend and explicitly does not claim a verified physical stop.

The policy interface is exactly `obs float32[1,61] -> actions float32[1,14]`. The observation comprises gyro 3 + gravity 3 + relative joint positions 14 + joint velocities 14 + previous unscaled action 14 + command 13. Normalization lives in the ONNX graph; targets are `HOME + scale × action`. This format is specific to the profile. Matching dimensions and hashes does not establish command semantics or the historical training recipe. Publication provenance is documented, but the complete link between checkpoint and historical configuration remains unproven in `POLICY_PROVENANCE_REPORT.md`.

Newton uses native BAM from IsaacLab pin `28aa1fca5843208ff9a67935695a4d5376e44d50`, with friction represented as a solver budget and the previous step's load, rather than a substitute PD controller. The CPU numerical implementation is labeled as a reference. Explicit profiles cover no current limit, a 1.75 A limit, and a 3–6-step delay; there is no default BAM profile. Every campaign must record its effective profile (`cascade/assets/microduck/newton-bam.json`). PhysX representability still requires proof of the actuator's external-load and friction contract. Static, dynamic, and viscous friction plus armature do not alone establish BAM equivalence. See `PHYSX_REPRESENTABILITY_REPORT.md` and the [upstream BAM pull request](https://github.com/isaac-sim/IsaacLab/pull/8161).

## Physical evidence and its limits

| Local evidence | Observed result | What it establishes |
| --- | --- | --- |
| `parent-preflight/kit-dedicated-live2/receipt.json` | Newton CUDA, SolverMuJoCo, 40 steps, 10 inferences, approximately 0.2 physical seconds, up to 47 contacts and 14 DOF friction rows; three frames | Short native integration; `mcp_executed=false`, `locomotion_verified=false` |
| Four `kit-native-vx-*-trial1` runs, identified in the hash manifest | Each records 400 steps, 100 inferences, and approximately 2 physical seconds. Requested vx values 0 / +0.15 / +0.3 / −0.3 produce final x positions 0.001740 / 0.008445 / 0.164699 / −0.001890 m | Finite state and a real physical response; slow forward and backward motion do not follow the request. These are not accepted walking trials |
| `parent-preflight/kit-native-vx0.3-video1/video-verification.json` | 21 frames, 2 physical seconds, 4.2 seconds of video; final x 0.164473 m against a requested velocity integral of 0.600 m | Video tied to a physical diagnostic, without an exercised MCP path or validated locomotion |
| `parent-preflight/kit-physx-capability2.json` | PhysX CUDA, 14 DOF, clock 0.010 → 0.035 s, finite state, and friction roundtrip | Basic physical capability of an earlier asset; no BAM, locomotion, or equivalence to the later Newton asset |
| `production-live/newton-product1/receipt.json` | Bootstrap failure; zero steps, inferences, and frames | Historical timestep-representation failure, corrected in the reviewed source |
| `production-live/newton-product2/receipt.json` | `completed=true`, 40 steps, 10 inferences, three frames, 15.3384 wall-clock seconds, owned resources closed; `physical_acceptance=false` | The production runner passes bootstrap and completes a short trial. It does not establish walking, turning, physical stop, or complete Hermes/MCP execution |

The `newton-product1` failure compared decimal nominal timestep 0.005 with native float32 value 0.004999999888241291. The reviewed source preserves both identities and checks native progression. `newton-product2` records bootstrap at step 2 and time 0.009999999776482582. **The timestep error is corrected and is not an open finding.** The correction appears in `cascade/src/cascade/sim/microduck_newton.py:399` and its `dt` property at line 470. The receipt says `sdk_shutdown=requested_after_receipt; verify process exit externally`: internal resource closure and final process exit are separate claims.

Existing CPU checks provide separate evidence: JUnit report `parent-balance-status-green.xml` records 806 passed and one skipped; `parent-bridge-production.log` records 248 passed; and `parent-native-bam.log` records 98 passed. These counts overlap and must not be added. ONNX replays and the solver comparison campaign have separate manifests included in the evidence JSON. This review did not rerun those tests. Their logs neither certify the full suite or CI of a future commit nor replace physical acceptance.

## Three open findings at the recorded snapshot

### P1 — Stop during inference can allow targets from an invalidated command

In `cascade/src/cascade/sim/microduck_stepper.py:141`, the stepper reads the command and generation under a lock, releases the lock for ONNX inference, then applies targets and advances physics without rechecking permission, generation, or expiration. If the watchdog or stop invalidates the command during a slow inference, its acknowledgment can precede a new application of targets computed from the previous velocity.

Reviewed file SHA-256: `b164138b6513fa25db54ad81d03faa6e4833fd9b9d410dbe62e108447c395d49`.

Required correction: discard invalidated or expired inference results, define balance behavior after invalidation, and test stop, lease expiry, and epoch changes while inference is deliberately blocked. Holding the lock across ONNX would block stop and is not an acceptable fix. Tests that invalidate commands only between ticks do not cover this interleaving. This is a static finding; the review did not provoke a physical incident.

### P2 — Client admission does not fully bind physical and BAM configuration

`cascade/src/cascade/control/isaac_base.py:45` and `cascade/src/cascade/sim/mobile_bridge.py:70` bind robot, source, engine, device, asset and policy hashes, clocks, and epoch. They do not bind a digest of the BAM/configuration profile or the complete bundle and effective solver. The runner records `configuration_sha256` in receipts, but that value is absent from the client/reader admission contract. Two profiles with different current limits or delays can therefore present the same accepted identity.

Reviewed SHA-256 values: IsaacBase `8ae0408cc0bebfdde4d56a19e6ca8bb2267d61fb802ec9460c30c2310180596c`; bridge `948031ec823014fc11a193f5506d62d7f982a19b7049285613836c8bf2133c39`.

Required correction: add effective configuration identity checked by control, observation, and the verifier, with negative cases for mismatched configurations. Local layer loading already performs checks. The finding concerns the missing complete external binding; it does not claim that the runner accepts arbitrary USD files.

### P2 — The independent rest/balance verdict does not require foot support

`cascade/src/cascade/agent/base_effects.py:271` checks movement, time, height, tilt, faults, and rest velocities/drift. It records contacts but neither requires admitted physical support nor rejects forbidden body supports. A synthetic sequence with advancing timestamps, upright posture, no movement, and `contacts=[]` can satisfy these conditions.

Reviewed file SHA-256: `5e963e1af2a560225ec05122377cae34e59c2b45b2ef91827eb9752585589dbb`.

Required correction: define each profile's contact contract for balance/rest and add negative cases for missing ground support or improper support. This should not require continuous contact throughout every walking phase. The static gap is not evidence that the robot actually floated.

## Head, mouth, and voice ownership

The policy already owns `neck_pitch`, `head_pitch`, `head_yaw`, and `head_roll`, indices 5–8 of the 14 actions. The other ten joints belong to the legs. Its command block is `[vx, vy, wz, neck_pitch, head_pitch, head_yaw, head_roll, body_x, body_y, body_z, body_roll, body_pitch, body_yaw]`, documented in `cascade/src/cascade/control/microduck_policy.py:10`. The current stepper fills only the first three fields. Head and body commands receive zero, while the policy continues computing their joint targets. No public gesture/head-pose tool is implemented.

The mouth is outside those 14 actions; no local mouth controller is implemented or admitted. The physical name `jaw_soft` within the `head_roll` hierarchy does not establish an independent mouth servo. Voice can supply audio and semantic requests to the supervisor. Head gestures would require an explicit capability with limits, expiration, arbitration, and validated policy-command semantics. A second loop writing head targets, such as Reachy's MovementManager, would conflict with policy ownership. Mouth animation requires its own physical mapping, single owner, limits, feedback, and neutral return before it can be advertised as supported.

The [MicroDuck conversation design](MICRODUCK_CONVERSATION_DESIGN.md) uses these ownership boundaries. The proposed voice adaptation does not replace the locomotion controller or put LLM/audio latency into its control and stop paths.

## Licensing and PR readiness

The notices distinguish Apache-2.0 code, BSD-3-Clause native BAM, and 3D models declared Creative Commons BY-SA-NC without a version in the upstream text. The weights' model card separately declares Apache-2.0. The code license must not be extended to meshes or derived USD assets, and keeping assets outside git does not establish authorization for commercial use. Origin/hash manifests and external download/conversion are implemented. See `cascade/assets/microduck/NOTICE.md`, the [pinned model README](https://github.com/pollen-robotics/microduck_rl/blob/8d0db74916a4f833d1d9b95d6a1d7f4d13b9d5ec/README.md), and the [weights model card](https://huggingface.co/pollen-robotics/microduck-policies/blob/d5a8b55033e157f1af2ed6bd5c1e435b770a8ee0/README.md).

The candidate profile `cascade/configs/bases/microduck_isaac.yaml` retains unresolved hashes, port, and verifier, `admission: pending_physical_admission`, and PhysX as the default engine. It is an honest template, not a READY configuration. A complete implementation PR requires resolving the three findings, pinning and versioning implementation/configuration, completing the suite and CI, and demonstrating forward/backward motion, turning, stop/reset, and negative cases through the actual agent path for each declared engine. A clearly scoped documentation or foundations PR can proceed without treating diagnostics as physical acceptance.

Capability separation can extend to other robots. SLAM/localization, mapping, obstacle/terrain navigation, a physical driver, and whole-body coordination with arms remain outside the demonstrated scope. They are future extensions, not additional requirements for approving this first mobile design. Each humanoid will need its own model, policy, actuators, limits, and admission; 61 observations and 14 actions are a profile contract, not a universal architecture.
