# MicroDuck mobile runtime

**Next action:** run the offline smoke with the kinematic mock (no Isaac; seconds):

```sh
PYTHONPATH="$PWD/src" CASCADE_BASE=microduck_mock CASCADE_LLM=mock \
  uv run python -m cascade.apps.mcp_server
```

MicroDuck is an opt-in mobile base with a bounded velocity interface. It instantiates no arm,
IK, grasping or kitchen scene. **Status: candidate.** General walking and turning have not passed
physical admission.

1. **Works (simulation):** supported standing, a separately observed stop, one small negative
   turn, ±30 mm `walk_distance` episodes that track while commanded, twelve robots standing or
   following scripted twists in one world.
2. **Refuted or open:** post-completion heading drift and overrun (forward and reverse), larger
   or positive turns, composed reverse motion, interruption, disconnect, reset, repeated starts,
   braking from a steady gait, twelve-robot closed-loop control under the original limits.
3. **Not delivered:** PhysX BAM (the native CLI refuses PhysX), scene reset, hardware, hosted
   conversation.

The four historical ±30 mm fresh-start receipts of the [distance candidate](#implemented-contract)
do not admit the current contract: a later
[observation audit](evidence/voice-reply-recovery-20261004/guarded-continuation.json) finds
heading drift after completion beyond the unchanged 0.08 rad bound. Capability status:
[ROADMAP](ROADMAP.md). Single-robot control vs shared-scene ownership:
[ROBOT_MODULARITY](ROBOT_MODULARITY.md).

## Runtime and clocks

1. `MobileBase` supplies immutable physical state and bounded body-frame velocity commands;
   `SafeBase` enforces limits, freshness, cancellation and progress; `MobileRig` selects bases by
   name; `MobileSkillRuntime` records each execution and its independently sampled postcondition
   in the trace and episodic memory. The MCP reader handles stop and cancellation mid-motion.
2. The native bridge runs a floating-root articulation in its own Isaac Sim arena on Newton /
   MJWarp with the pinned native BAM actuator. The official `velstand.onnx` takes 61 observations
   and emits 14 actions. Physics steps nominally every 5 ms, the policy every four completed
   solves (the actual timestep is recorded); camera and LLM latency never advance the policy
   clock. Neck and head joints belong to the policy; the mouth is outside its action vector.
3. Inference previews never mutate policy history. Before committing a result the bridge
   rechecks command, generation, episode, physical clocks and wall deadlines.
4. **Stop** invalidates walking intent while the policy and actuators keep balance. It does not
   certify an instantaneous physical stop or undo a committed motor target. An e-stop ACK always
   reports physical verification as pending; a separate observer later confirms or refutes it
   under the same receipt ID (and cannot repair an earlier failed motion). `reset_stop` restores
   command permission only: it does not reset physics or prove a new scene epoch.
5. **Target profiles:** default `direct-v1` (scale 1.0, no filter). The opt-in
   `--target-profile robotd-targets-v1` (policy API `target_profile="robotd-targets-v1"`)
   reproduces [robotd at 9136aa4](https://github.com/pollen-robotics/microduck/blob/9136aa4ee88e81edf2bcaf3527e90b65da25f1eb/robotd/src/control.rs#L607):
   float64 HOME plus `0.9 * raw_action`, then EMA alpha 0.5 (four head joints) / 0.7 (ten leg
   joints), first target unfiltered; raw float32 actions stay the observation history;
   `committed_targets` / `targets(action)` / `target_contract` expose it; filtered float64 targets
   are converted to float32 at `set_targets` (`target_upload`). It is not the full robotd pipeline
   (no command smoothing, voltage adaptation, stale-sensor coasting or actuator delay).
   Launchers reject a missing, changed or ignored target contract.

## Implemented contract

`walk_distance(distance_m=...)` exists only in a configured mobile profile
(`configs/bases/microduck_distance_candidate.yaml`). SafeBase integrates
displacement and lateral/heading drift from the first completed post-ACK
sample; the independent checker compares the signed displacement with the
request, and progress after the physical deadline never completes it. The
opt-in pair `max_admission_lateral_m` / `max_admission_heading_rad` bounds the
change from the last pre-ACK sample to that baseline: set both in
`distance_control` and `verifier` (one-sided disables the verifier). No value
is measured yet.

## Evidence and identity

1. **Model identity.** One SHA-256 binds the admitted bundle, consumed USD layers, normalized
   runtime scene, ONNX weights, native BAM source and effective parameters, actual
   solver/device/timestep, runtime source and contact registry. Clients must pin it explicitly in
   `model_identity_sha256` (learning it from `hello` is not admission); state, camera and truth
   channels enforce the same binding. A changed recipe needs a new identity and a new physical
   evaluation.
2. **Collision hull limits.** Newton's USD importer (Isaac Sim 6.2 / Newton 1.6.1rc1) honours only
   an authored `newton:maxHullVertices`, `mjc:maxhullvert` or
   `physxConvexHullCollision:hullVertexLimit`; unauthored meshes get a 64-vertex partial hull
   whose vertex-order dependence left the mirrored soles 2.3 mm apart and curled a 20 s straight
   command by +3.3 rad. Two layers fix it: `scripts/convert_microduck.py` authors each mesh's MJCF
   limit (`-1` = complete hull; old bundles must be reconverted), and the owner's
   `author_collision_hull_limits` authors `-1` in the runtime layer where none exists
   (de-instancing Isaac Lab colliders; pinned bytes unchanged), then
   `read_native_collision_hulls` refuses a limit that did not reach the native model. Both records
   (`collision_hull_limits`, `collision_hulls`) enter the identity. Result: complete symmetric
   soles (4964/5029 vertices) and curl +3.5 → +1.3..+1.7 rad on the converted bundle
   ([comparison](evidence/microduck-hull-limits-20261004/live-gate.json)); on the Isaac Lab USD
   +1.27/+1.61 → +0.34/+0.21 rad and lateral drift 1.84/1.81 → 0.19/0.13 m over 2.3 m
   ([owner-side](evidence/microduck-hull-limits-20261005/live-gate-external-usd.json)). The
   remaining curl exceeds the official MuJoCo reference (+0.5 rad) and is not explained.
3. **Support** uses versioned, complete post-solve contact records (shape pairs, points, normals,
   world reaction forces) bound to the pose's epoch, step, time and identity, checked against an
   explicit sole/ground registry. Empty contacts never prove standing; unavailable forces stay
   unknown; stationary windows need upward foot support and reject forbidden body support.
4. **Settling evidence.** Late, pending and duplicate observations cannot supply positive
   settling, but their faults, residual motion and support failures still veto an earlier quiet
   window.

## Running the candidate

1. **Mock:** the command at the top exposes `get_base_state`, `walk_velocity`, `turn`,
   `stop_navigation`, `emergency_stop`, `reset_stop` and `verify_last_action` (plus
   `camera_snapshot` with a camera channel). It cannot verify physical motion.
2. **Native bridge:** `scripts/isaac_microduck_bridge.py --help` lists the required paths,
   hashes, BAM profile, private port, limits and deadlines; `--check-only` verifies offline
   inputs without Kit or a socket. Use the selected Isaac release's `python.sh`, a new output
   directory and an outer process deadline. `BRIDGE_LISTENING` means transport, not locomotion.
3. **Profile:** `configs/bases/microduck_isaac.yaml` is an incomplete template. A reviewed run
   profile supplies engine/device, bundle and policy hashes, the inspected model digest, private
   endpoint, support registry and verifier limits; there is no default admitted recipe. **Do not
   change gains, checkpoints, speed limits or verifier tolerances to turn a failed campaign
   green.** Keep failed episodes; check process exit and teardown separately from the receipt.
4. **SDK and solver:** `--sdk-recipe` selects Isaac Sim compatibility (versions and source checks
   in [`sim/microduck_sdk.py`](../src/cascade/sim/microduck_sdk.py); rechecked against the
   modules actually imported; part of the identity). Outside Kit, Newton 1.6.1rc1 is accepted
   only via the CPU recipe `newton161rc1_cpu`. `--integrator-profile euler-v1` (default
   `sdk-default`) authors `mjc:option:integrator` before bootstrap and refuses any disagreement
   between authored, imported and effective CPU/Warp integrators; CPU (0.002 s, ~1e-8) and Warp
   (0.005 s, ~1e-6) timesteps/tolerances stay distinct. Its preparation (model `a74083e7…`) ran
   no policy or episode ([record](evidence/robot-modularity/microduck-euler-preparation-20261005.json)).
5. **RTX caches:** `--private-rtx-cache` (requires recipe `isaac62_48b2d951`) routes the two
   writable shader/PSO caches into `rtx-cache/` of the output directory; optional
   `--rtx-cache-seed DIR --rtx-cache-seed-sha256 SHA` admits a reviewed seed (`manifest.json` +
   `data/`, schema `cascade.private-rtx-cache-seed.v1`, inventory from
   [`private_rtx_cache.inventory`](../src/cascade/sim/private_rtx_cache.py); extra, missing or
   changed members, symlinks and special files are refused). Paths go to `rtx-cache.json`,
   measured settings to `rtx-cache-effective.json`; only the relative policy enters the identity
   ([native validation](evidence/microduck-private-rtx-20261004/native.json): 800-step zero
   command run, bridge ready in 10.7 s).

The bridge saves startup admission, the canonical identity, runtime receipt, every physical step,
actual policy inputs/outputs and discarded inferences, and timestamped images. The
[VOICE04 episode](CONVERSATION.md) obtained a real walk/stop pair but correctly refuted the walk
for post-completion heading drift.

## Isaac Lab USD folder as the admitted asset (5 October 2026)

The admitted asset is the Isaac Lab MicroDuck USD folder (`Robots/PollenRobotics/MicroDuck`,
[#8265](https://github.com/isaac-sim/IsaacLab/pull/8265)/#8266 from `microduck_rl@d424a0c8`),
pinned by content in
[`assets/microduck/isaaclab-microduck-usd-manifest.json`](../assets/microduck/isaaclab-microduck-usd-manifest.json)
(eight files, listing digest `0a295b65…`):

1. `scripts/microduck_assets.py --check` admits the folder from a local directory only (no
   download path). `scripts/admit_microduck_usd.py --build` makes a schema-2 bundle
   (`usd/microduck_allcollisions.usd`, `LICENSE`, `ATTRIBUTION.txt`, provenance,
   `receipt.json`); `verify_bundle` accepts schema 1 and 2; kind and variant enter the identity.
   Launchers are unchanged: `--bundle <dir> --bundle-sha256 <receipt digest>`. The USD embeds the
   Pollen meshes (separate models notice); only the two text files are Apache-2.0.
2. Before the stage parse the owner loads the pinned BAM sources (registers
   `NewtonBamDriveAPI`), parses the fourteen `NewtonActuator` prims with
   `newton.actuators.parse_actuator_prim` (17 coefficients must equal the pinned Rhoban m6 fit),
   records their deployment attributes and deactivates them; removes the zero-gain
   `PhysicsDriveAPI:angular`; leaves the ground outside any collision group; uses the sole proxies
   `left_foot_collision/sole_left` / `right_foot_collision/sole_right`
   (`FOOT_SHAPES_BY_ASSET`); asserts the Isaac joint properties (`ASSET_SOURCE_PROPERTIES`)
   before the unchanged M6 replacements; and authors complete hulls on its ten colliders.
3. Physics differences vs the converted bundle (CPU probes, not an engine comparison): 10 vs 70
   enabled hulls; the asset authors Isaac Lab's deployment distribution (kp 200, 7.4 V, delay 3–6
   steps, 1.75 A, 1.0676 N·m) while CASCADE keeps `official_infer_nominal_no_delay` (no delay, sag
   0.1, no current limit, 0.9634 N·m); masses, limits, joint order, root height and the 0.96 N·m
   MJCF range are identical; MJCF revisions differ (`d424a0c8` vs `8d0db749`). VelStand was
   trained on neither USD.
4. Newton's own BAM drive ([newton#4504](https://github.com/newton-physics/newton/pull/4504),
   proposed in [#4397](https://github.com/newton-physics/newton/issues/4397)) uses a different
   token and parameters (`NewtonBAMControlAPI`), so the pinned component stays until that ships
   and is re-admitted. With `newton161rc1_cpu` the pinned Isaac Lab #8161 sources (`28aa1fca`)
   pass `tests/test_newton_bam.py` and `tests/test_microduck_sdk_recipe.py` (recorded in
   `assets/microduck/newton-bam.json`).
5. **Results:** twelve robots (reader-only, zero commands) settled 0.125 → 0.1172 m and held it
   with only sole-ground contacts ([receipts](evidence/microduck-isaaclab-usd-20261005/)). The
   ±30 mm MCP rerun with `rough_walk_e`: forward +27.04 mm then **refuted** (−0.28 rad yaw after
   completion vs 0.08 rad); reverse −25.68 mm **confirmed**; both stops confirmed
   ([audits](evidence/microduck-isaaclab-usd-20261005/locomotion/summary.json)). A lab-only
   [PhysX smoke](evidence/microduck-isaaclab-usd-20261005/physx-smoke-01.json) parsed one
   14-DOF articulation and stepped 400 updates while the unactuated robot collapsed; no PhysX
   locomotion is claimed.

### Isaac Lab Newton policies as opt-in candidates (5 October 2026)

Twelve operator-supplied Isaac Lab exports (rsl_rl 5.4.1, `obs float32[1,61]` → `actions
float32[1,14]`) are pinned by content; `velocity_flat` (iteration 49,999) and `velocity_rough` are
registered in `assets/microduck/policy-candidates.json` as `isaaclab_velocity_flat` /
`isaaclab_velocity_rough` with `rough_walk_e`'s contract (the Lab task's joint order, 61-wide
layout and `default_joint_pos + 1.0 × action` equal CASCADE's `POLICY_JOINTS`, `observation()` and
`direct-v1`; stand pose = `HOME_Q`). Their license is undeclared. `isaaclab_velocity_flat` on the
USD asset: forward +25.62 mm then +0.13 rad after completion (refuted), reverse −27.63 mm then
−48.55 mm (refuted, excess progress), no fall. Both policies track while commanded and keep
stepping at zero twist (training uses 2 % standing environments).

### Standing handoff at zero twist, measured (5 October 2026)

`cascade.control.microduck_handoff.StandingHandoff` (opt-in) selects the motion policy for a
nonzero twist and the official VelStand for exactly zero (no lease, completed distance, latched
stop), once per attempt; cadence, targets, limits, stop semantics and the verifier are untouched.
Enable with `--handoff-profile stand-on-zero-twist-v1 --standing-policy <onnx>
--standing-policy-sha256 <sha>` (the standing policy must be the pinned VelStand and differ from
the motion policy; identity gains `recipe.handoff`).
([audits](evidence/microduck-isaaclab-usd-20261005/locomotion/summary.json),
[braking analysis](evidence/microduck-isaaclab-usd-20261005/locomotion/post-completion-analysis.json))

| Walk | Executed mm | After completion mm | Yaw after completion rad | Verdict |
|---|---:|---:|---:|---|
| `rough-forward01` (`isaaclab_velocity_rough`) | +25.86 | +26.94 | +0.006 | confirmed |
| `rough-reverse01` (`isaaclab_velocity_rough`) | −26.65 | −55.39 | +0.032 | refuted (excess progress) |
| `roughho-forward01` (rough, handoff) | +25.68 | +28.14 | +0.053 | confirmed |
| `roughho-reverse01` (rough, handoff) | −25.64 | −46.81 | +0.008 net, 0.080 peak | refuted (wz drift; the first violation is kept) |
| `flatho-forward01` (flat, handoff) | +26.50 | +27.57 | +0.127 | refuted (wz drift) |
| `flatho-reverse01` (flat, handoff) | −27.33 | −41.76 | +0.107 | refuted (wz drift) |

Switches were clean, but verdicts match the runs without the handoff: 0.3 m/s for 30 mm completes
mid-stride (14–17 policy slots, body at 0.09–0.23 m/s) and any network then needs 0.7–0.9 s to
rest in reverse. The handoff only shortens the reverse overrun by 4–7 mm; it stays a software
candidate. Next candidates: a checkpoint trained with zero and decelerating commands, or a
distance-aware walk speed, judged by the unchanged verifier.

The same overrun reproduces in Isaac Lab's own environments outside CASCADE (36 walk episodes, no
fall; forward +4 to +31 mm, reverse −11 to −46 mm after the zero command; CASCADE's 61-value
observation matches Lab's term by term), so it is a property of the checkpoints and their training
commands, not of CASCADE's pipeline ([report](evidence/microduck-isaaclab-repro-20261005/REPORT.md),
[aggregates](evidence/microduck-isaaclab-repro-20261005/report.json),
[trajectories](evidence/microduck-isaaclab-repro-20261005/trajectories.csv.gz); driver offered as
[AntoineRichard/IsaacLab#21](https://github.com/AntoineRichard/IsaacLab/pull/21)).

**Keynote showcase (scripted, not admission):** `scripts/isaac_microduck_shared.py --layout
choreography --choreography configs/microduck/choreography-keynote-12.json` makes twelve robots
follow a presenter proxy (P-law to formation slots, clipped to 0.3 m/s / 1 rad/s, `hello`
announces `scripted_twist`; stage assets in `assets/showcase/keynote/`). Run `k1-keynote-12`:
12/12 no fall or fault over 68 s of simulation (830 s wall), 6.65–7.00 m each
([report](evidence/microduck-keynote-showcase-20261005/REPORT.md)). Twelve closed-loop clients
still fail the limits ([route fleet](evidence/microduck-route-fleet-20261005/REPORT.md)).

## Remaining admission work

1. Current MCP campaign: supported standing and a separately observed stop are confirmed;
   forward/reverse tracking is refuted and both turns expire; interruption and disconnect
   invalidate admitted commands; braking from an established gait is unproved.
2. Required acceptance: current-source equilibrium, forward/reverse tracking, turns,
   interruption, disconnect, reset, repeated starts and video from the same episode, per engine.
3. PhysX lacks the separate previous-step external generalized load the BAM contract needs; the
   native CLI rejects PhysX.
4. Earlier negatives stay retained: the robotd forward attempts (10.63 mm of 25 mm), the MuJoCo
   prefix rejected at preparation, the bootstrap A/B and effective-options diagnostics (see the
   evidence index).
5. Models stay external (notices declare BY-SA-NC without a version; code, weights and BAM keep
   separate notices); no meshes, USD or ONNX are vendored. The
   [conversation gateway](CONVERSATION.md) keeps the hosted-conversation ownership boundary.

### Shared-scene implementation boundary

`sim/microduck_shared.py` binds explicit robot namespaces to disjoint native coordinates and
body/shape identities in one world:

1. The coordinator previews all policies without touching history or targets, checks the common
   completed clock and each permission under a shared fence, then commits the admitted group and
   requests **one** solve. A command change before the fence discards the whole preview and
   retries once on the same state; repeated revocation withholds the tick (no step, no frame).
2. A stop after admission applies to the next control slot (the in-flight solve cannot be rolled
   back); partial commits and native faults still contain the scene; stop ACKs never prove rest.
   Every robot's support checker sees all solved contact rows, so another robot never counts as
   ground. Identities and command epochs stay per robot.
3. Global state/contact channels are captured and validated once per completed scene (bound to
   model, state, contact buffer, solver data and clock), support is decoded once into immutable
   contacts and rebound per robot; replies stay detached. CPU tests cover 1, 2 and 12
   participants.
4. `scripts/isaac_microduck_shared.py` takes the bridge's input flags plus
   `--robots 1..12 --spacing 2`; use `--port 0`, `--solver-cuda-graph --reuse-solved-read`, the
   explicit SDK recipe, a fresh output directory and an outer deadline. `--serve-base-port 0`
   (opt-in) exposes per-robot loopback endpoints (`BRIDGE_LISTENING.json` maps ids to ports):
   commands and resets queue per robot until the owner drains them before a tick, expired or
   closed requests never run later, stop bypasses the queue.
5. Opt-in diagnostics and candidates, each bound into the identity: `--profile-phases`
   (`timing.jsonl` inclusive spans plus passive GC callbacks and generation-2 trigger stacks;
   nested spans are not additive), `--profile-sync-solve`, `--profile-cprofile`,
   `--gc-policy freeze-startup-heap` (one full collection + `gc.freeze()` before
   `fleet.start()`, `src/cascade/sim/heap_freeze.py`; recorded in `runtime.json`
   `backend.gc_policy` and `receipt.json` `gc_policy.applied` / `.released`) and
   `--state-reader process` (below).

Native results: 1, 2 and 12 robots stand through 800 solves with matching identities (twelve:
XY drift < 2.8 mm, tilt < 0.014 rad). **Twelve-robot control still fails the original 0.5 s RPC
/ 0.4 s progress / 0.5 s state-age limits** in three of four retained episodes (one fixed zero
episode passed its 72 client events with every outcome `unverified`); long attempts overlap
~475 ms generation-2 collections. The startup-heap freeze removed every generation-2 collection
from three reader-only runs (attempt maxima 202.6–212.1 ms vs a 1344.8 ms baseline outlier), but
no command was exercised under it ([probes](evidence/heap-freeze-20261005/summary.json),
[CPU mechanism](../benchmark/results/heap_freeze_cpu_20261005.json): 200.3 ms → 0.001 ms full
collection on a synthetic 2.3 M-object heap). The
[twelve-robot overview](evidence/microduck-shared-20261004/endpoints-twelve-zero.jpg) is a
read-only probe (grid floor display only).

### Shared-owner per-step cost: one host snapshot for the BAM checks (7 October 2026)

`BamHostSnapshot` (`control/newton_bam.py`): one per-step host copy of the ~25 whole-model Warp
arrays the twelve `NewtonBamAdapter`s each re-read (~400 device syncs per 5 ms step), handed to
every adapter's `before_step(dt, snapshot=…)`; every check still runs, and a snapshot of another
stage or step is refused. `bam.before_step` 24.3 → 13.6 ms, step median 76.2 → 61.3 ms
([A/B](evidence/microduck-owner-host-snapshot-20261007/REPORT.md)). `--profile-sync-solve`
shows **7.6 ms of GPU physics per 5 ms step** for twelve robots, so the owner cannot reach real
time on physics alone.

### Shared-owner per-step cost: cheaper per-robot binds and reader polls (7 October 2026)

`cProfile` of the owner found per-robot recomputation of immutable data: the binding digest is now
a `cached_property`, `SupportObservation.rebound` validates only the two changed identifiers,
`BaseState.as_dict` memoizes the plain contact copy per shared tuple and `identifier()` uses a
compiled regex. No check removed; replies stay content-identical and isolated. Step median
61.3 → 48.55 ms, p95 147.6 → 126.3 ms ([A/B](evidence/microduck-owner-reader-rebind-20261007/REPORT.md)).

### Shared-owner per-step cost: one device-side finiteness check for the BAM outputs (7 October 2026)

`BamOutputCheck`: one int32 flag array per cohort (slot per adapter and output) filled by two Warp
kernels (`wp.isfinite`, `wp.atomic_max`), reset per admitted step and read ONCE by `verify()` right
before `owner.step()` (span `bam.verify`); a non-finite output raises with the original wording
plus robot and array, and the solver never steps. The single-robot path (`output_check=None`) is
byte-for-byte unchanged. `bam.before_step` + verify 13.91 → 11.87 ms per step
([A/B](evidence/microduck-owner-bam-check-20261007/REPORT.md)).

### Shared-owner per-step cost: cohort Warp launches for the twelve BAM adapters (7 October 2026)

`BamCohort` drives all adapters with ONE pinned `DriveBam` over 14·N DOFs (`env_dof_stride=14`) and
ONE `MjWarpActuatorBridge`, after running every adapter's host-side checks first (one refusal
refuses the fleet with nothing written): 72 → 6 launches and 108 → 9 copies per step, bitwise
identical rows for delay-free drives. `NewtonBamAdapter.cohort()` returns `None` (per-adapter path,
receipted as `bam_cohort`) for a stochastic-delay profile or differing shared settings, because the
pinned kernels seed the delay draw per 14-DOF block. `bam.before_step` 12.11 → 6.57 ms, step median
46.79 → 42.43 ms ([A/B](evidence/microduck-owner-bam-cohort-20261007/REPORT.md)). All A/B client
walks stay `unverified` on the unchanged deadlines.

**Off-GIL `state()` reader (opt-in, GPU A/B pending):** `--state-reader process` (requires
`--serve-base-port` and Unix descriptor passing; default `owner-gil` is byte-identical) hands each
reader channel after its first `state` reply (SCM_RIGHTS) to a stdlib-only server
(`src/cascade/sim/mobile_state_server.py`, `sys.executable -I`). `StatePublishingController`
mirrors `marshal.dumps(state())` into a shared-memory slot per robot under the controller lock
after every step and permission change (seqlock + CRC32, monotonic publication count, bound robot
id and identity); the server recomputes only `state_age_s` / `state.producer_age_s` from the
publish time, so an owner stall still ages the reply and the unchanged `max_state_age_s` check
refuses it. Failure is closed (`off-GIL state unavailable`, never an older reply). CPU only: ~6 ms
per-step publication cost with no clients; with polling clients the owner step stops depending on
the poll rate (`tests/test_mobile_state_server.py`). No real-owner speedup is claimed.

## Evidence index

| Topic | Records |
|---|---|
| robotd target foundation and forward failures, MuJoCo prefix negative, bootstrap A/B, effective options | [foundation](evidence/robot-modularity/microduck-robotd-target-foundation-20261004.json), [forward failures](evidence/robot-modularity/microduck-robotd-forward-failures-20261004.json), [MuJoCo prefix](evidence/robot-modularity/microduck-mujoco-prefix-negative-20261004.json), [bootstrap A/B](evidence/robot-modularity/microduck-bootstrap-ab-20261004.json), [effective options](evidence/robot-modularity/microduck-effective-options-20261004.json) |
| shared scene: foundation, endpoints, support | [foundation](evidence/microduck-shared-20261004/foundation.json), [endpoints](evidence/microduck-shared-20261004/endpoints.json), [support profile](evidence/microduck-shared-20261004/support-profile.json), [support control (failed)](evidence/microduck-shared-20261004/support-control.json) |
| shared scene: phases and GC | [phase profile](evidence/microduck-shared-20261004/phase-profile.json), [GC trigger](evidence/microduck-shared-20261004/gc-trigger.json), [GC serialization](evidence/microduck-shared-20261004/gc-serialization.json), [row serialization](evidence/microduck-shared-20261004/row-serialization.json) |
| shared scene: control episodes | [legacy contacts](evidence/microduck-shared-20261004/legacy-contacts.json), [legacy control (passed, unverified)](evidence/microduck-shared-20261004/legacy-control.json), [row control (failed)](evidence/microduck-shared-20261004/row-control.json) |
