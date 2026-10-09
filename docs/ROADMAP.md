# Roadmap

The active scope covers the capabilities below. The
[capability index](PROJECT_STATUS_20261003.md) records measured results and the
[architecture](ROBOT_MODULARITY.md) describes shared interfaces. Dated entries
below this table preserve historical decisions; they are not the current backlog.

## Remaining capabilities

A row closes when its implemented path passes the stated end-to-end checks.
An interface, mock episode or successful dependency import alone does not close it.
The numbered follow-ups below the dated sections are worked through the
[2026-10-07 backlog program ledger](BACKLOG_PROGRAM_20261007.md) (one worktree,
branch and PR per item; what landed and what stays open, with class S/G/H/U/D).
Robot-specific policies, drivers and limits implement shared contracts; the
architecture does not depend on a particular Isaac Sim build or robot shape.

| Capability | Remaining implementation and completion criteria |
| --- | --- |
| Manipulation | Extend the three retained two-object campaigns and the completed [five-object campaign](PROJECT_STATUS_20261001.md#native-five-object-kitchen-campaign--2026-10-04) to more scenes and repeated coverage. Demonstrate measured collision/carry protection through the agent route while retaining release, support, rest, reset and owned closure; validate each new robot and hardware setup. **Reach envelope (B45, opt-in, 2026-10-09; Seeed: "the object has to be really close to the arm"):** measured offline on the reBot URDF through the real selector and the runtime's harness vet (`scripts/reachability_study.py`, [REACH_ENVELOPE.md](REACH_ENVELOPE.md)): top-down grasps reach only r ≤ 0.45 m (0.38 m at 10 cm grasp height), leaning the approach 45° away from the base 0.69 m, horizontal 0.75 m, leaning sideways nothing; the `*_reach` arm profiles (`rebot_rs_reach`, `isaac_reach`, `mock_reach`) add tilted analytic candidates (`grasp.angled_approach_tilts_deg: [30, 45, 90]`, behind every top-down one) and the measured box x 0.10..0.55, y ±0.50 (2833 vs 953 reachable grid points), every other harness gate unchanged and every default profile golden. Still open: live Isaac picks in the new region (`isaac_reach` vs `isaac`), the real rig (table, mount, camera field of view), contact/gripper-housing collision of angled and side grasps, objects that only fit a jaw pointing at the robot (no reach gain). |
| Locomotion | Demonstrate visible longer routes, turns, balance-preserving stop, disconnect and reset; integrate and validate additional actuator/physics combinations and humanoid policies. Running requires its own controller and evidence. Rerun the standing, walking, braking and reset campaigns against the newly admitted [Isaac Lab USD folder asset](MICRODUCK.md#isaac-lab-usd-folder-as-the-admitted-asset-5-october-2026) (twelve-robot standing probe; with `rough_walk_e` reverse confirmed and forward refuted; with the pinned Isaac Lab Newton `velocity_flat` export both directions track while commanded but are refuted after completion) and keep the converted bundle as the retained baseline; the switch to the standing policy at zero twist is built as an opt-in and [measured](MICRODUCK.md#standing-handoff-at-zero-twist-measured-5-october-2026): clean switches, verdicts unchanged, the residue is braking from 0.09 to 0.23 m/s at completion; next candidates are a checkpoint trained with zero and decelerating commands or a distance-aware speed in the walk skill, under the unchanged verifier; re-admit the BAM implementation when Newton ships its native BAM drive (newton-physics/newton#4504) instead of the pinned Isaac Lab #8161 sources. |
| Fastening | Establish repeatability of the measured mounted turn-and-rest task; the opt-in `gc_policy: freeze-startup-heap` profile now has one native same-revision pair ([on/off](FACTORY_FASTENING_RUNTIME.md#opt-in-startup-heap-freeze-software-candidate-5-october-2026)) that passed the verifier without an automatic generation-2 pause in either run, so an episode where such a pause would overlap the command window is still needed, as is the cause of the 0.2–0.8 s observation-drain gaps at readiness return and after stop; add tool acquisition, engagement, calibrated torque/preload and withdrawal with independent contact and thread measurements. |
| Perception and spatial understanding | Extend the measured static planar RGB-D result to broader metric geometry and moving cameras; validate IMU/proprioception fusion and ~~time alignment while preserving missing, stale and uncertain observations~~ **time alignment preserving missing, stale and uncertain observations landed in software 2026-10-09 (B50, opt-in)**: a sensors-domain `alignment:` block adds `sensing.read_aligned`, which pairs one fresh reference capture (a camera) with each IMU/joint sensor's admitted capture nearest its capture time and reports `aligned` / `stale` / `missing` / `uncertain` with the measured skew — never interpolated, stale and missing values withheld, unmeasured channels listed as absent, another epoch or clock domain never aligned, saturation or rate × skew above a declared tolerance uncertain; B39's link self-mask now uses the same component ([contract](ROBOT_MODULARITY.md#capture-time-alignment-b50-opt-in)). Measured by 42 CPU tests, incl. the real loopback mobile bridge through MCP (a step-10 camera capture pairs with the step-11 IMU and joint samples at +5 ms; a step-14 capture finds only step 11, −15 ms, stale; a 3 rad/s yaw 5 ms from the capture is uncertain) and a golden digest of every shipped sensors catalog. Still open: the live skew distribution on the Isaac MicroDuck bridge, the fusion estimator that consumes the pairings, producer-rate sampling (the pairing sees only captures that reads admitted) and hardware clock synchronization. |
| Mapping and navigation | [Twelve live cuVSLAM tracks](CUVSLAM_NATIVE_VALIDATION.md#live-native-rgb-d-stream) join a 320-solve native producer to registered camera/base poses and verified stop invalidation. A separate [native synthetic fixture](CUVSLAM_NATIVE_VALIDATION.md#native-covariance-transport) retained all twelve odometry covariance matrices through IPC; physical pose uncertainty remains uncalibrated. Admit calibrated pose and complete collision mapping/volume, then validate native route execution, replanning and arrival/rest, including tracking loss and changed map epochs. |
| Speech and interaction | [Two bounded voice cases](CONVERSATION.md#two-complete-native-speech-cases--voice13) now join synthetic microphone input, actual model decisions, native simulated hand motion/rest and same-origin browser audio, including operator stop, reset and reconnect. Extend to repeated and varied tasks, camera context, microphone/speaker hardware and an independently deployable conversation service; validate each physical robot setup. |
| Whole-body humanoids | Bind dynamic frames and a whole-body controller to explicit embodiments; validate balance, self/environment collision and coordinated base/arm/head actions. ~~Separate domain controllers must not compete for the same command endpoint.~~ **landed 2026-10-08** (B30, mock-only): an opt-in `whole_body` profile contract composes a locomotion base and an arm `mounted_on` it with disjoint command endpoints (overlap refused naming both claimants), a capture-time `world ← base ← arm_base` frame chain (stale/missing/out-of-order base pose refuses arm motion, never extrapolated), `exclusive` coordination by default (`concurrent` opt-in), global stop with explicit per-domain `reset_stop(domain=...)`, and world-frame verdicts with the frame valid at command start ([contract](ROBOT_MODULARITY.md#multi-domain-embodiments-mounted-arms), `configs/robots/mobile_manipulator_mock.yaml`); measured by 43 software tests incl. golden digests of every earlier robot profile. Still open: physical admission of any mounted composition (measured mount calibration, independent base-pose source, moving-frame arm limits), the whole-body controller, balance, self/environment collision and H2 arm physics. Embodiment chosen on 7 October 2026: [Unitree H2, PhysX first](HUMANOID_H2.md) with NVIDIA's public H2 USD and `Velocity-H2-History-v0` policy (pinned in `configs/h2/bundle.json`, contract in `control/h2_policy_contract.py`); the reference loop walks 0.63 m in 2 s on the internal 6.2 build ([smoke](evidence/h2-reference-smoke-20261007/manifest.json)); `scripts/isaac_h2_bridge.py` owns the H2 on PhysX and CASCADE's SafeBase drove the first episodes with the independent verifier confirming 3 of 5 `walk_velocity` commands ([evidence](evidence/h2-owner-first-episodes-20261007/REPORT.md)); binding gate passed in simulation, no physical admission; Newton route tracked through newton-assets PR #53. Same day, [geometric skills + chat-host path](evidence/h2-geometric-candidate-20261007/REPORT.md): `walk_distance` ±0.5 m and `turn` 0.8 rad execute to within 5 cm / 0.03 rad through the harness and `walk_velocity` is confirmed through MCP, but every goal-stopped command is refuted by the settle check — measured cause: the policy's post-stop yaw oscillation outlasts the 4 s wall settle budget at ≈0.47× RT (≈1.9 s sim; the robot is still after ~3 s sim). Candidate revision 2 the same day (settle budget 8 s wall, rest thresholds unchanged): 6/7 confirmed on both paths incl. `walk_distance` ±0.5 m and MCP `turn` 0.6; `turn` 0.8 rad still refuted on measurement (yaw oscillation 0.69 → 0.32 rad/s over 3.75 s sim) — ~~next revision is control-side (ramp the turn rate before the goal), not a bigger budget~~ **landed 2026-10-07 in software (B29, candidate revision 3)**: `turn_control.goal_ramp` lowers the admitted 0.5 rad/s on the measured yaw (0.4 rad/s², floor 0.15 rad/s, then the unchanged zero-twist stop) inside ONE admission through the opt-in `scale_velocity` bridge primitive (`--velocity-scaling` on the H2 owner), so the unchanged verifier still binds one generation; CPU tests only, every other profile byte-for-byte unchanged. Owner episode 8 October ([live result](HUMANOID_H2.md#live-result-8-october-2026)): 0.8 rad 8/8 with the ramp vs 1/6 without, but 1.0 rad 1/5 vs 5/6 (the ramp's deceleration plus a mid-turn yaw-rate dip exhaust the unchanged 3 s command); not admitted, ~~next control revision open~~ **landed 2026-10-08 in software (B29b, candidate revision 4)**: `goal_ramp.time_budget` never lets the ramp decelerate below `(remaining − tol) / (tracking · (admitted command time left − reserve))` (0.93, 0.3 s), still inside the one admission; measured on a CPU toy with the owner's mid-turn dip (1.0 rad: timeout → 2.72 s; 0.6/0.8 rad: revision 3 unchanged) and a golden digest of every base profile (75/76 command sequences identical, only the candidate's dipped 1.0 rad turn changes). Still open: the owner A/B (revision 4 vs 3 vs 2), the 1.0 rad translation margin (0.166–0.227 m vs 0.20 m in both arms), physical admission. |
| Hands and touch | Extend the [fixed LEAP free-finger runtime](ARTICULATED_HAND.md) beyond its four repetitions of one native motion/rest task: broaden trajectories and disturbances, add other hand drivers/controllers, calibrated tactile observations, contact/slip estimation and grasp/force skills; validate dexterous object interaction for each supported hand and sensor. |
| Multiple robots | Resolve worst-case twelve-robot feedback latency without relaxing its limits; the later completed zero-command lifecycle did not overlap its long GC pause. The opt-in startup-heap freeze (`--gc-policy freeze-startup-heap`, [MICRODUCK.md](MICRODUCK.md#shared-scene-implementation-boundary)) is the current software candidate: [reader-only probes](MICRODUCK.md#shared-scene-implementation-boundary) on 5 October kept generation-2 collections out of the 800-attempt window, and a native twelve-robot command episode under the original limits is still required. Extend the completed 1/2/12-robot probes and endpoint zero sequences to independent native agent tasks, shared-space collision coordination and individual/global physical stop, disconnect and reset. The owner's per-step host cost is now [measured by phase](MICRODUCK.md#shared-owner-per-step-cost-one-host-snapshot-for-the-bam-checks-7-october-2026): one per-step host snapshot for the twelve BAM adapters cut the step median 76.2 → 61.3 ms with twelve closed-loop clients, and [cheaper per-robot binds and reader polls](MICRODUCK.md#shared-owner-per-step-cost-cheaper-per-robot-binds-and-reader-polls-7-october-2026) (cached binding digest, metadata-only support rebinding, memoized plain contacts for `state()` polls) 61.3 → 48.55 ms, and [one device-side finiteness check for the twelve BAM adapters' outputs](MICRODUCK.md#shared-owner-per-step-cost-one-device-side-finiteness-check-for-the-bam-outputs-7-october-2026) (60 device syncs → one read per step) 48.82 → 46.78 ms with `bam.before_step` 13.9 → 11.9 ms, and [one cohort of Warp launches for the twelve BAM adapters](evidence/microduck-owner-bam-cohort-20261007/REPORT.md) (one pinned `DriveBam` + bridge over 14·12 DOFs: 72 → 6 launches and 108 → 9 copies per step) 46.79 → 42.43 ms with `bam.before_step` 12.11 → 6.57 ms (all twelve walks still `unverified` on the unchanged deadlines), and an [opt-in off-GIL `state()` reader](MICRODUCK.md#shared-owner-per-step-cost-an-opt-in-off-gil-state-reader-7-october-2026) (`--state-reader process`: after each reader channel's first reply its polls are answered by a stdlib reader-server process from per-robot seqlocked shared-memory slots the owner rewrites under the controller lock; byte-identical replies, ages never refreshed, fail-closed on owner death) is implemented with CPU contract tests — on CPU it costs the owner ~6 ms per step for twelve robots and removes the poll-rate dependence of a fake owner loop (step median 25–28 → 16–18 ms under twelve polling client processes), while its GPU A/B against `owner-gil` on the route harness is pending — and the GPU physics step alone costs 7.6 ms per 5 ms step for twelve robots in one world, so real-time twelve-robot control needs cheaper physics (hull/solver budget per asset) as well as less host work. Measured so far: twelve robots in one world walk 68 s without a fall under in-process scripted twists ([keynote showcase](MICRODUCK.md#keynote-showcase-twelve-robots-follow-a-presenter-proxy-in-one-world-5-october-2026), no clients, no verifier), while twelve closed-loop clients on the same owner still trip the unchanged limits ([route fleet evidence](evidence/microduck-route-fleet-20261005/REPORT.md)). |
| Agent intelligence and evaluation | Validate perception-guided planning, bounded skill graphs, memory-assisted recovery and multimodal feedback across tasks; complete Arena/VAB task episodes and held-out failures. Learned policy adaptation needs separate implementation and evaluation. **Learned-policy grasp executor (B49, opt-in, 2026-10-09):** `grasp.executor: vla` serves `grasp_object` from an openpi/LingBot websocket policy server under the SafetyHarness and the unchanged verifier ([VLA_EXECUTOR.md](VLA_EXECUTOR.md)); measured against its protocol stub only. Still open: any run with real weights. |
| Hardware and deployment | Connect selected robots and sensors through explicit adapters; verify calibration, transport loss, controller ownership, stop/restart and task outcomes before each physical deployment. Keep installation and service configuration portable. **Sandboxed agent host (B35, opt-in, 2026-10-08):** an OpenClaw agent inside an NVIDIA OpenShell sandbox (NemoClaw) drives the robot through `mcp_server --http` (TLS + bearer, same serial worker and stop channel as stdio); on Isaac it ran a physics-confirmed pick and a verified reset in one session on 2 of 3 fresh stages ([NEMOCLAW.md](NEMOCLAW.md), [evidence](evidence/b35-nemoclaw-openshell-20261008/REPORT.md)). Still open: the kitchen proof cases and `demo_proof.py` on this route, the Spark, a firewall rule for the listener, sandbox GPU passthrough off. **B36 (new, found by B35):** pick reliability on the bare Isaac reBot scene, 3 of 6 fresh-stage picks failed with three signatures independent of the caller — a lifted cube counted as a failed attempt so every retry refuses "already holding" (cube kept at home height; the reset then fails "did not settle at home"), "did not settle above the place target", and a drop in carry after a verified grip; ~~plus the cold GraspGen-X first inference (15.5 s) exceeding the 8 s client timeout~~ **landed 2026-10-09 (B15a)**: the learned server runs one warm-up inference before it binds, so the first client call was 0.16 s ([evidence](evidence/b15a-graspgenx-warmup-20261009/REPORT.md)); the three pick signatures stay open. **Grip squeeze cap (B38, opt-in, landed 2026-10-09):** the real reBot pick close left the jaws pushing at a fixed fraction of travel, so the holding torque grew with object width (Seeed: a paper cup crushed); the arm-profile key `gripper.max_contact_squeeze_rad` now caps the squeeze past the first contact, steady torque = min(today's, kp·effort·cap), today's exact commands for objects already within the cap — measured on a simulated RobStride jaw over 3/5/7.5/8.5 cm × every grip profile ([REBOT_GRIP_SQUEEZE_CAP.md](REBOT_GRIP_SQUEEZE_CAP.md)). Still open: the user's supervised hardware protocol in that doc, then a value in `rebot_rs.yaml` (ships `null`). |

## Delivery plan recorded on 2026-10-02

[The source and acceptance index](PROJECT_STATUS_20261001.md) supersedes the
historical status lists below. Previous command baseline for physical validation `ff8d58b` includes the
retained NV carry guard, explicit MCP GPU registration, verifier startup
readiness, localization freshness checks and [detector model reuse](DETECTOR_MODEL_REUSE.md). Source-bound software results are
recorded in that index. Spark uses the
installed PhysX/Qwen/real-GGX profile, physical-time joint motion, render-bound
camera state, observed-finger approach/closing preflight, bounded planning,
preserved memory ranking, coherent held-object observations, release-opening
synchronization and per-camera pending publication. These are
implemented contracts, not a blanket physical-success claim.

The owner disconnected the Sparks. Immediate work is resolving the local
dual-RTX Pro native timeout and the separate nvblox route/carry failures,
then completing source-bound proof, the five-object campaign and normal
same-version restart. See [local validation](LOCAL_RTX_VALIDATION.md). Preserve failed runs and outcome memory. Newton,
new hardware, Cosmos evaluation, mobility, and ovrtx retain separate validation
or design scopes. The offline [Jev/Kev pilot](JEV_DECISIONS.md) has not demonstrated
a decision-quality advantage over the recorded-status rule baseline and is not
part of presenter startup.

Local profiles 08 and 09 retained real traces and fresh sampled cameras but
timed out during placement. Profile 09 on `3ccdc2e8` includes optional
[bridge Python spans](ISAAC_BRIDGE_PROFILING.md) and periodic task-clock
markers. Its 120.460-second technical capture and administrative closure
passed. The first offline name matcher failed on the SDK's Python suffix;
a strict adapter then qualified a closed 118.593-second CPU window with 968
updates. Execution-job code dominates its measured elapsed time. Measure that
job's subcomponents and possible observer overhead before selecting an
optimization; the current spans do not establish an internal cause, speedup
or completed physical case. Global/GPU absolute alignment remains unqualified,
and earlier failed runs remain unchanged.

Current main `f7a8823` includes the optional spans merged in PR #62 and the
profile 09 documentation from PR #63. Separately,
release-open stability candidate `6e1bc81a` passed 3,741 tests with the original
deadlines and geometry guards. Integration and physical validation are still
pending; it does not yet resolve the retained NV15 retreat failure.

Passive profile 10 measured the observer itself and found unused depth encoding
on RGB-only reads. [Independent component encoding](ISAAC_FRAME_ENCODING.md)
preserved complete frame bytes and reduced RGB-only time in a CPU replay of
one retained capture. Combined candidate `d1d54dc` integrates this change with
release-open stability; its full software suite passed 3,757 tests with 1,305
inputs unchanged. Passive profile 11 on that source passed 100 samples and
owned closure; new-capture observer median was 68.170 ms versus 272.239 ms in
profile 10, with the slow geometry sample retained. Obtain new native
placement/release/home, camera, campaign and restart evidence; these two passive
runs do not establish whole-task performance or physical acceptance.

## Optional cuMotion planning and assembly research (2026-10-01)

[PR #47](https://github.com/johnnynunez/cascade/pull/47) adds a standalone
[cuMotion planner](CUMOTION.md). Its factory and CLI export validated trajectory
candidates; real x86 and Spark aarch64 GPU smokes passed with the 1.1.0 SDK. The native execution
path is unchanged. Live scene/payload binding and trajectory-aware execution
need separate implementation and physical validation.

The [screw manipulation investigation](SCREW_MANIPULATION_RESEARCH.md) identifies
SimReady tools/fasteners and permissively licensed original Factory meshes.
The proposed first task is measured nut advancement on a fixed bolt, followed
by seating/torque verification. Existing `turn_screw` counts commanded wrist
travel; it does not establish physical tightening. No assembly task is implemented
by this research entry.

## PhysX finger envelope (2026-10-01)

The [observed-scene envelope](physx-finger-envelope.md) now combines the complete
nominal finger components with both retained x86 and ARM derived PhysX cooking
representations, giving forty components per finger.
It preserves the opening feedback bound and observed target/non-target rules;
it does not modify colliders or invent a contact margin. Candidate `26b6826`
passed 3,238 software tests before the cross-platform extension. The final
geometry passes both representation audits and 74 focused tests. The merged
source passed the [two-object Spark proof](SPARK_DELIVERY.md#earlier-two-object-proof-on-28061a6);
campaign and restart acceptance are separate. The earlier orange failures and failed ARM coverage
of the first artifact stay recorded. The combined renderer/geometry integration
passed 3,295 tests and five CI jobs before merging as
[PR #44](https://github.com/johnnynunez/cascade/pull/44); the
[source and acceptance index](PROJECT_STATUS_20261001.md) binds those results to
the tested and merged trees.

## Optional sensor backend: NVIDIA ovrtx (2026-10-01)

An opt-in static `CameraBase` profile and scene-snapshot API now render real
RGB-D through OVRTX 0.5 and `ovstage`. Real analytic smokes on Linux x86_64
and Spark aarch64 verify metric Z-depth, integer-index pinhole calibration,
local scene transforms, camera poses and whole-packet duplicate retention.
The public `make_camera(load_profile("cameras", "ovrtx"))` entry point also passed
on x86 with the static example profile. Square pixels are required;
unequal focal lengths are explicitly rejected after the measured SDK path
ignored that request. See [OVRTX_RENDERER.md](OVRTX_RENDERER.md) for exact
versions, platform evidence, installation and limits.

Connecting an automatic Isaac/Newton physics producer, supplying the masks
and physical identity required by safety consumers, and comparing the same
kitchen scene remain future work. No demo default changes. Rendering tests
do not establish grasping, mapping or physical acceptance.

## PAAI event follow-up (2026-09-15)

The initial **PAAI, Physical Agentic AI** demo for **Build a Claw** has
successful orange and tomato-can pick/place cases checked with camera images
and simulator physics. Guided scene inspection, delivery-zone grounding,
camera streaming, authenticated access and reset/recovery have also been
tested. These are bounded event checks; object coverage and generalization
remain work for later versions.

- Evaluate **Cosmos3-Edge** as a perception and reasoning backend. Structured
  grounding, generalization and OpenClaw integration need more validation.
  Retain the reliable event backend until Cosmos passes the same acceptance
  gates.

## Historical status at a glance (2026-09-10)

This section records the September design and open work at that time; its
Newton/Cosmos target and unavailable-Spark statement are not current defaults.

**Delivery priority at that date:** one command per Linux DGX Spark to install
Isaac Sim **6.1.0.0**, use its **Newton** experience, serve **Cosmos3-Edge**
and open an isolated OpenClaw chat. Implementation and acceptance boundary
are tracked in [SPARK_DELIVERY.md](SPARK_DELIVERY.md). No Spark was
available for this change: publication and the cold-start GPU rehearsal
remain pending. Newton standalone CPU is an additional test path, not a
substitute for Isaac/Cosmos on the target.

What a visitor gets today, in one command, on a laptop: `./run.sh` →
MuJoCo (or Isaac Sim when installed) + OpenClaw chat with 41 robot tools,
self-proven (runtime built, tools listed, brain answered, one pick
CONFIRMED by physics, scene reset). Every skill's effect is verified on an
independent channel, the planner sees its own history as images, and the
pre-delivery baseline was 737 passed / 0 skipped (not the final delivery
suite). Unverified on hardware: real-arm motion,
the SO-101 serial driver, the ROS2 and Unitree backends.

Open, in priority order (details in the sections below):

0. **Spark delivery acceptance** — finish integration tests, run the exact
   published build on a clean Spark, require the session-bound physical
   proof/reset and test every supported scene object before replicating it.
1. **Real rig first motions** -- onsite checklist (CAN up, gripper travel,
   hand-eye, table plane), `pytest -m hardware`, then `--arm rebot_rs` at
   low velocity. Everything above the driver has been exercised in two
   simulators; the drivers have not.
2. ~~**Persistence-loop leftovers** #2–#7 below (provisional held marker,
   per-task budget cap across tiers, handover/sort persistence, thin-object
   slip heuristic, fail-fast on over-width, the listed coverage gaps).~~
   **landed 2026-10-07** — see the struck items in "Persistence-loop review
   leftovers" below; `tests/test_persistence_leftovers.py` (25 tests) pins
   each one.
3. **Learned grasps for real**: run `serve_graspgenx.sh` (CUDA) instead of
   the protocol stub and calibrate `tip_offset_m` / the reBot sweep volume
   in Isaac; the stub only proves the wire.
4. ~~**Wrist camera** extrinsics validated mid-descent against physics truth.~~
   **landed 2026-10-07 (sim)** — every wrist frame's own `K` + `T_base_cam` projects the
   resting box's physics-truth corners onto its pixels during a slow descent/ascent:
   PhysX 0.69 px median moving (max 1.07), Newton 2.73 px (max 9.4), silhouette IoU
   ≥ 0.96, depth exact, moving = static on both engines (time-aligned); see
   [NEWTON_ENGINE.md](NEWTON_ENGINE.md#real-rebot-asset-on-the-internal-62-build-start-up-probe-battery-wrist-camera-7-october-2026).
   The real D435i hand-eye calibration stays with the rig (B25).
5. **Newton target validation** — the Isaac bridge already defaults to
   Newton. Validate the real reBot asset in Isaac Sim 6.1 on Spark; synthetic
   contacts or SO-101 standalone tests do not certify that different asset.
   **x86 part landed 2026-10-07:** `physics_probe.py` on the real asset passes
   8/8 on Newton and 8/8 on PhysX on the internal 6.2 build (and 8/8 on Newton on
   the 6.1 baseline) after fixing the 6.2 Newton start-up deadlock and the probe's
   silent unreachable-lift failure; still open: the Spark (GB10) run, and the 6.2
   Newton camera cadence that leaves 34–40 % of refreshes frameless during arm motion
   (2.8 % on 6.1 with the same three physics steps per update; cause open)
   ([details](NEWTON_ENGINE.md#real-rebot-asset-on-the-internal-62-build-start-up-probe-battery-wrist-camera-7-october-2026)).
6. ~~**Judge as a metric**: run `scripts/judge_run.py` over every launcher
   proof turn and keep the judge-vs-physics confusion matrix in the run
   summary, so a regression in the outcome pictures shows up as `fn`.~~
   **landed 2026-10-09 (opt-in, advisory, CPU-measured only; B44).**
   `scripts/launch.sh --judge fake|vlm|grm` (or `CASCADE_JUDGE`; default
   off) runs `judge_run.py` over the proof turn's `pick_and_place` rows via
   `scripts/judge_proof.py`, bounded by `CASCADE_JUDGE_TIMEOUT_S` (default
   180 s; the judge's process group is killed at the bound), and writes the
   confusion matrix into `<proof evidence>/run-summary.json` plus one banner
   line; `proof.json`, READY and the exit status never change, every failure
   reads `unavailable`. The previous default-on, unbounded pass (shipped
   `eval.judge` = a frontier model through the gateway) is gone: `--judge vlm`
   is the same call, opted into. Measured on CPU only:
   `tests/test_judge_proof_turn.py` (fake judge over recorded and mock-stack
   traces, a stub OpenAI-compatible endpoint answering, refusing to score, or
   hanging; 35 failed / 4 premise+golden passed on 4d0947b → 39 passed),
   all mutants killed. **Live, on recorded picks (B44-live, 2026-10-09):**
   the local Qwen3.8-27B (llama.cpp :8080) judged 72 recorded live Isaac
   picks (B36 bare-reBot A/B series + B35 NemoClaw runs; 40 physics-confirmed,
   27 refuted, 5 unverified): with the 1536-token budget Spark ships, 60
   scored and `tp=40 tn=1 fp=15 fn=0` (agreement 73 %); the 12 left unscored
   (11 refuted, 0 confirmed: Qwen reasons past the budget and never writes a
   score) re-judged at 8192 tokens give `tp=40 tn=7 fp=19 fn=0` over 66
   (71 %). Every confirmed pick scored +100 %, so `fn` -- the regression this
   metric exists for -- is trustworthy here; refuted picks scored −100…+90 %,
   so `hop > 0` counts partial progress as success (`hop ≥ 1.0` would separate
   all 66, chosen in-sample, not shipped)
   ([evidence](evidence/b44-live-qwen-judge-20261009/README.md)). STILL OPEN:
   a full launch whose proof turn is judged with `--judge vlm`; a success
   threshold validated out of sample (B65) and a score that fits the budget
   (B66); GRM (#14).
7. ~~**Visual embedder** for episodic recall (`embed_dim`), and action↔object
   consolidation on top of ExperienceMemory (keys on text today).~~
   **landed 2026-10-07 (opt-in, CPU-measured only).** `memory/embedder.py`:
   `memory.embedder.backend: none` (shipped, byte-identical default) | `hash`
   (deterministic colour/layout + hashed-word vectors, no dependency) |
   `siglip` / `clip` (`memory-embed` extra; a requested backend that cannot
   load fails `build_runtime` before any hardware, never a silent fallback).
   `EpisodicMemory` indexes motion frames and the crops of objects a call
   localized, pruned with a task-scale ring (the explicit-`embedding` index
   used to grow without bound); `recall_memory(query)` gains `looks_like`
   hits only with a joint image-text embedder. Skill-library notes rank by
   text embedding (floor OR guard overlap, promotion gate first).
   `memory/consolidation.py` (`memory.action_objects: true`) folds the tier-2
   outcome stream into (skill, object) wins/losses across wordings, one credit
   per executed call, as an advisory LLM-tier digest. Measured with the hash
   embedder and a deterministic joint stub on CPU: `tests/test_memory_embedder.py`,
   `tests/test_memory_visual_recall.py` (31 RED on c5012e7 → 35 GREEN with the
   golden pins of `tests/test_memory_default_path.py`), 7/7 mutants killed.
   STILL OPEN: real SigLIP/CLIP weights were never loaded here -- recall
   quality and the uncalibrated text floors need a GPU-host evaluation;
   ~~crops come from localizations, not every watcher detection; recall is
   in-process only.~~ **landed 2026-10-09 (B43, both opt-in, CPU-measured
   only):** `memory.visual_recall_detections` -- the WorldWatcher indexes a
   crop of every COMMITTED detection (one per belief per frame, at most one
   per belief per `visual_recall_interval_s` 30 s, ≤ 2 per tick, own ring of
   128; fusion is paused during motion skills, so no crop comes from a
   motion frame) -- and `memory.persist_episodic` (`CASCADE_EPISODIC`,
   `CASCADE_EPISODIC_PATH`) -- the visual index is saved on shutdown and
   restored at startup like the belief store (wall-clock ages, 6 h max age
   dropped before a 2 s minimum apparent age, atomic writes, another
   embedder's file refused, every restored hit `restored`/`remembered`).
   Measured with the hash embedder and the joint stub on CPU:
   `tests/test_memory_visual_recall_v2.py` (32 RED on 4d0947b → 38 GREEN with
   6 premise/golden pins), 90/90 mutants killed. Still open: no live Isaac
   run of either switch yet (watcher-crop cost per tick with a real
   detector, restart on the booth), and real-weight recall quality as above.
   Found while measuring (NOT fixed, separate item): tier-2
   crops come from localizations, not every watcher detection; recall is
   in-process only. ~~Found while measuring (NOT fixed, separate item): tier-2
   recall matches the compound "pick up the red cube and then pick up the
   blue cube" to the single habit "pick up the red cube" (cosine 0.901 ≥ 0.9)
   because experience is consulted before the curriculum split, so the fast
   tier can run half a command and report success.~~ **landed 2026-10-09
   (B37)** — `ExperienceMemory.recall` accepts a habit or recipe only with
   the task's clause structure: the same number of `split_subgoals` clauses
   and, for a sequence, every clause ≥ 0.9 against its counterpart in order
   (a single instruction keeps the pre-B37 rule exactly). Measured with the
   real recall + `FastPlanner` on 23 instruction pairs, main vs branch: the
   compound now runs both clauses through the curriculum, each clause still
   warm-started from its own habit; the same rule stops a clause or a single
   command from replaying a recorded compound (0.929: both cubes moved), a
   reversed sequence (0.994: wrong order) and a one-word clause difference
   (0.958: bowl for box). `tests/test_tier2_clause_structure.py` (20 RED on
   4e896c3 → 28 GREEN, through `run_task` with `MockLLM`), 10/10 mutants
   killed. Still open: a bare "and" is no clause boundary anywhere in the
   fast tier (two "move …" commands joined by "and" still replay the first
   one's habit, 0.951), and the opt-in programs tier offers programs by
   keyword overlap or text embedding (B42) and leaves whole-vs-part to the
   brain (ARCHITECTURE, Known limitations).
8. **Multi-arm on physics**: `so101_left`/`so101_right` are mock; render a
   two-arm MuJoCo scene so the inter-arm gate is measured, not simulated.
   **Half landed 2026-10-07** — `sim/demo_scene.multi_arm_scene_xml` attaches
   N prefixed copies of the robot MJCF at the profiles' `base_pose`s and
   `tests/test_multi_arm_physics.py` measures the gate against
   `mj_geomDistance` over the real collision geometry: the centreline gate
   approved overlapping meshes (12 of 2000 random pose pairs), so the gate
   now subtracts measured per-link radii (`safety.link_radii_m`) and its
   clearance is a proven lower bound on the physical one. **Second half
   landed the same day**: `so101_left_mujoco`/`so101_right_mujoco` run both
   arms in ONE generated MuJoCo world (`mj_prefix`, loader-planted
   `mj_rig`, each arm drives only its own actuators) and the gate stops a
   physics arm 0.13 m before the meshes touch. Still open: a finer sphere
   model to win back the workspace the single-radius envelope costs (the
   inward-yaw pose is 0.13 m clear in physics, 0.032 m to the gate), wrist
   cameras and the placement observers on a rig.
9. **Mobility + navigation (the next structural addition, approved
   2026-09-10 as design-first).** ~~cascade has no mobile base, navigation,
   mapping or robot self-localization; ROS2 and the humanoid profiles are
   arm-only.~~ **Partly landed since, with a different interface (re-derived
   from the code 2026-10-09, B48;
   [current state](ARCHITECTURE.md#ros2-humanoids-and-what-is-not-here-yet)):**
   a velocity-level `MobileBase` + `MobileRig` + `SafeBase` +
   `MobileSkillRuntime` (`walk_velocity` / `walk_distance` / `turn` /
   `stop_navigation`, independent verifier) with exactly two backends, `mock`
   and `isaac` (MicroDuck on Newton and the Unitree H2 on PhysX: simulation
   candidates, not admitted); an opt-in `go_to` route runner (CPU tests only,
   no shipped pose/clearance provider); optional cuVSLAM localization (native
   simulation stream measured, pose uncertainty uncalibrated); the mock-only
   `whole_body` mount contract (B30). ROS2 is still arm-only. Still open from
   this design: `go_to_pixel` / `go_to_object` / `where_am_i`, the ESDF
   costmap, Nav2 / `ros2_base`, `mujoco_base`, `unitree_base`, a G1, the
   two-room demo and any hardware run. Design in `docs/MOBILITY_AND_NAVIGATION_DESIGN.md`:
   `MobileBase` (twin of `ArmBase`) + `MobileRig` + `base=` in
   `execute()`; Vesta's three navigation verbs as skills (`go_to_pixel`,
   `turn`, `stop_navigation`, plus `go_to_object`, `where_am_i`) with the
   memory harness spanning the walk and `nav` postconditions on the pose
   channel; a 2D costmap sliced from the existing Warp ESDF; two nav
   backends behind one interface (Nav2 `NavigateToPose` when ROS2 is
   sourced, Warp planner otherwise -- the laptop one-click keeps working);
   backends `mock_base`, `mujoco_base` (Menagerie Go2/G1, floating base in
   the shared world), `isaac_base` (G1 + Isaac Lab velocity policy, bridge
   ops `set_velocity`/`base_pose`), `ros2_base` (`/cmd_vel` Twist +
   `/odom` + Nav2; reference target = Isaac ROS Deploy's G1 AGILE WBC
   bringup, which is exactly Twist-driven and has `hardware_type:=mujoco`),
   `unitree_base` (sdk2 `LocoClient.Move/StopMove/Damp`). First target:
   Unitree G1/H1 in Isaac Sim. Demo task where mobility is visible: two
   tables, find the cube, bring it to the tray -- after turning away only
   the history frames know table A was checked. Order of work and the
   test-per-step plan are in the design doc. What Vesta does NOT provide
   here: mapping, SLAM, odometry (it leaves motion to a "navigation
   backend"), weights or code.

## Landed 2026-07-31: the orchestration-gap upgrades

Six-paper synthesis (Pigey, Agentic-VLA, Harness-VLA/RPent, ASPIRE, VIA,
Waddle) + a Cosmos3-Edge backend. Full write-up in
`docs/AGENTIC_UPGRADES.md`; summary:

- **Effect verification (Pigey).** Motion primitives no longer self-report:
  each has a postcondition checked against a channel the actuator does not own
  (sim physics truth via `sim/truth.py`, else the belief store). A refuted
  postcondition *downgrades* a claimed success; `unverified` is a first-class
  outcome. Live-verified: cube displacement 30.8 cm, `channel: physics`.
- **Milestone progress (Agentic-VLA).** Decomposition is now a checked signal
  (symbolic tier from beliefs, rate-limited visual tier via `VERIFY_USER`),
  with stall detection and honest "could not confirm" reporting.
- **Operating envelopes (Harness-VLA).** `memory/envelope.py` learns where each
  primitive actually works + a normalised failure taxonomy, injected at task
  start. Advisory only — the harness stays the sole authority on motion.
- **Annotated interface (VIA).** `annotated_view` skill: numbered object
  badges, 5 cm metric grid, TCP, and an optional configured display band.
  MCP receives the image and key. These do not verify IK or grasp reachability.
- **Cosmos3-Edge.** `configs/llm/local_cosmos.yaml` + `agent/cosmos3.py`
  (parses its XML tool-call format — the plain `openai_compat` client silently
  never calls tools) + `scripts/serve_cosmos_vllm.sh`.

Open follow-ups from this work:
1. Run a full booth rehearsal against Cosmos3-Edge and compare tool-call
   reliability + latency with Qwen3-VL (needs the vLLM-Omni container pulled).
2. Feed `max_frames > 1` (short clip at ~4 fps) to the Cosmos3 reasoner and
   measure whether motion context improves failure diagnosis.
3. ~~Phantom beliefs~~ **fixed 2026-08-27.** `annotated_view` surfaced a stale
   4th "cube" mark — root cause was two bugs in `perception/visual_interface.py`,
   not the belief store (which correctly never forgets during a demo run —
   object permanence is deliberate, see `memory/beliefs.py`): (a) every mark
   was drawn with equal confidence regardless of how many times it had been
   re-observed, and (b) the `age_s` reported alongside each mark read
   `getattr(b, "age", 0.0)`, a field `ObjectBelief` never sets, so it was
   silently always `0.0` and hid staleness from anyone reading the tool
   result. A belief now renders/describes as `confirmed: false` /
   `UNCONFIRMED` unless it is currently visible or was re-observed at least
   once (`VisualInterface.min_observations`, default 2) — pinned in
   `tests/test_visual_interface.py`.
4. ~~Envelope features are currently raw skill args; add derived features
   (TCP z at grasp, object height) so the learned ranges capture the real
   B601-RS constraint rather than a proxy.~~ **landed 2026-10-07.**
   `memory/envelope.py` `DERIVED_FEATURES`: the runtime measures
   `tcp_z_at_grasp_m` (FK of the joint vector read back when the jaws
   closed), `object_height_m` (fix top above the support plane),
   `object_width_m` (narrower horizontal footprint extent) and
   `object_tcp_lateral_offset_m` inside `skill_grasp_object` and passes them
   through `record(..., measured=...)`; a feature the call could not measure
   is counted in `missing` (never defaulted), both ride in the trace context
   for `ingest_trace`, and `envelope_digest()`/`export_markdown()` show them
   as `measured`. Measured on the mock stack: a grasp records all four
   (height 0.050 m against the 5 cm synthetic box, lateral offset < 1 cm), a
   grasp that dies at localization records four `missing`, confidence tiers
   and `contradictions` unchanged (`tests/test_envelope_derived_features.py`,
   RED 14 failed on main → GREEN). Advisory only, as before.
5. **SGLang Omni as a second serving engine for Cosmos3-Edge**, landed
   2026-08-27: `scripts/serve_cosmos_sglang.sh` + `configs/llm/local_cosmos_sglang.yaml`
   (`local_cosmos_sglang`, :8083) alongside the existing vLLM path (`local_cosmos`,
   :8082). `agent/cosmos3.py` needed zero changes — the XML tool-call format
   is a property of the checkpoint's chat template, not the serving engine, so
   both profiles use `type: cosmos3` and differ only in `base_url`. Unlike the
   vLLM script (verified on GB10 2026-07-21), the SGLang script is
   **unverified** — it mirrors the vLLM script's known day-one pitfalls
   (transformers git-main, diffusers→HF re-export, the `get_rope_index`
   no-video guard) plus SGLang's own `--disable-cuda-graph` warmup flag, but
   has not had a rehearsal run on real hardware. `scripts/openclaw_demo.sh
   --brain cosmos-sglang` wires it into the OpenClaw front-end the same way
   `--brain cosmos` does. This also answers follow-up #1's engine half — the
   comparison there can now run vLLM vs. SGLang, not just Cosmos vs. Qwen.

## Landed 2026-08-27: four more sources, one landed mechanism

The user pointed at four more references beyond the 2026-07-31 synthesis:
Human-CLAW (2607.27180), LaMem-VLA (2607.07608), grasping.io (resolves to
HUG — Human Universal Grasping, NYU/Tsinghua/UMich), and a re-read of the
Waddle Labs blog post this repo had only cited secondhand before. Only one
of the four shipped code today (`memory/envelope.py`'s graduated
confidence) — the rest are scoped, concrete next steps, not vague
inspiration lifted from an abstract.

- **Graduated envelope confidence (RLinf/RPent, the real repo).** Checking
  the actual repo behind the already-cited Harness-VLA paper (not just its
  abstract) turned up two things `memory/envelope.py`'s port was missing: a
  three-tier confidence label (`single-shot` / `probable` / `verified`, by
  sample count — this module has no per-task grouping to match RPent's
  "distinct tasks" breadth signal; spans are deliberately scene-independent,
  see the module docstring) and a `contradicted_by`-style regression signal.
  Landed: `_Span.confidence()` + a `contradictions` counter, incremented
  when a later call's feature value falls INSIDE a "proven" range and still
  fails. Surfaces as a non-blocking `Verdict.notes` caution (booth rule
  preserved — advisory only, never a veto) and in `envelope_digest()` /
  `export_markdown()`. Pinned in `tests/test_envelope_confidence.py`.
- **Human-CLAW** (2607.27180) — a humanoid-control paper; wrong embodiment
  for a 6-DoF tabletop arm (its diffusion-motion / ControlNet / half-physics
  -sim machinery does not transfer). One idea does: a **pre-execution skill
  verifier** that interrogates a proposed call with skill-specific questions
  before it runs — distinct from this repo's existing *post-hoc* effect
  verification (`agent/effects.py`) and *static* operating envelopes
  (`memory/envelope.py`). Landed 2026-10-07 as an advisory-only critic —
  see follow-up #6 below.
- **LaMem-VLA** (2607.07608) — dual latent-memory architecture (Curator →
  Seeker → Condenser → Weaver) that splices condensed memory tokens directly
  into a VLA policy's embedding space. Requires a trainable VLA backbone
  this repo does not have — added to "Deliberately not built" in
  `docs/AGENTIC_UPGRADES.md`, next to Agentic-VLA's GRPO note for the same
  reason. cascade's belief store + envelope + skill library already cover
  the same short-term/long-term split symbolically (text woven into the
  LLM's system prompt, not latents woven into a policy).
- **grasping.io → HUG** (Human Universal Grasping, NYU/Tsinghua/UMich) — an
  open-source flow-matching grasp model trained on 1M egocentric human-grasp
  frames, cross-embodiment by design (no hand-specific retraining). No
  hosted API; would need self-hosting the same way `graspgenx` already is
  (ZMQ server, `scripts/serve_graspgenx.sh`). A real candidate for a second
  `grasp.backend` option, untested against the B601-RS's specific IK
  envelope. See open follow-up #7.
- **Waddle Labs, re-read in full.** The existing citation ("code-as-policy +
  a shared skill library") was accurate but incomplete: the actual post
  describes a three-tier hierarchy — `primitives` (fixed low-level platform
  functions) → `skills` (agent-authored, reusable, composed from
  primitives) → `programs` (full task-specific policies composed from
  skills, written fresh per instruction). cascade has the first two
  (`TOOL_SPECS` = primitives, `skills_library/*.md` = skills) ~~but no
  `programs` tier — see open follow-up #8~~ **landed**: the opt-in programs
  tier (follow-up #8, 2026-10-08; exposed to MCP chat hosts and ranked by
  embedding since 2026-10-09, B42). Also notable, as a *contrast*
  and not a pattern to adopt: Waddle's described safety layer is a single
  `verify(...)` check with no rate-limiting or rollback protocol — thinner
  than this repo's harness-as-sole-authority design, worth stating
  explicitly rather than citing Waddle as a safety precedent.

Open follow-ups from this work:
6. ~~**Pre-motion plausibility check (Human-CLAW).** Extend
   `agent/milestones.py`'s existing rate-limited `VERIFY_USER` critic
   pattern to run *before* dispatch for `_MOTION_SKILLS`
   (`skills/runtime.py:31`), not just post-hoc for milestone progress: ask
   "is this specific call, with these specific args, plausible given
   current beliefs/reachability?" and let it veto/substitute, the way
   Human-CLAW's verifier does. Reuses existing rate-limiting so it does not
   blow booth-clock budget.~~ **landed 2026-10-07** —
   `agent/milestones.py::PlausibilityChecker` (+ `VisualBudget`, the
   tracker's per-task limiter factored out so both critics share one
   mechanism; `prompts.PLAUSIBILITY_USER`), consulted by the orchestrator
   before every LLM-tier motion dispatch with the current frame, a
   skill-specific question and the belief/held/reach digest; the answer
   rides on the result and the trace row as
   `plausibility: {verdict, reasons, source}`, an `implausible` verdict is a
   caution the planner reads on its next turn, and no model / no frame /
   exhausted budget / verifier fault record `skipped` with the reason.
   **Deliberately NOT the veto/substitute half of Human-CLAW**: the call is
   dispatched unchanged whatever the verdict says — the harness stays the
   sole authority that refuses motion (booth rule, same as envelopes).
   Measured on the mock stack with the scripted planner + a scripted critic
   (`tests/test_premotion_plausibility.py`, 12 tests, RED against main):
   an "implausible" grasp still drives the arm and holds the cube, the
   budget caps critic turns at `agent.premotion_max_checks` (default 3) per
   task and resets per task, a raising verifier yields `skipped`, and
   `agent.premotion_check: false` reproduces main's orchestrator path
   write-for-write (runtime attribute writes, dispatches, planner messages
   pinned against a golden taken from main). Reflex/experience tiers stay
   LLM-free; `fleet.py` / `dashboard_runner.py` / `booth_rehearsal.py` do
   not opt in. No live-rig run; no claim about the verdicts' accuracy.
7. ~~**HUG as a second grasp backend.** Add `grasp.backend: hug` alongside
   `graspgenx`, self-hosted the same way (a serve script + client mirroring
   `grasping/graspgenx_backend.py`), re-ranked by the same
   `GraspOutcomeMemory`. Whether HUG's cross-embodiment grasps clear this
   arm's IK envelope is untested — the point of landing it is to find out,
   not to assume it is better.~~ **landed 2026-10-08 (CPU / stub only)** —
   [docs/HUG.md](HUG.md). `scripts/serve_hug.py` wraps HUG's documented
   inference path (code `8d1c52d`, weights `1415c9e`, sha256-pinned) behind
   a GraspGen-X-shaped REQ/REP protocol. CUDA is required unless
   `--device cpu` is explicit, the operator supplies MANO (never shipped),
   and `--stub` is an analytic double.

   `grasping/hug_backend.py` maps each human hand to a parallel-jaw pinch.
   That mapping is **our** assumption (thumb tip vs index tip, palm →
   pinch approach), and so is `quality`: HUG emits no score, so the value
   is CASCADE's geometric score. Then the same `GraspOutcomeMemory`
   re-rank, selector and harness apply. HUG is opt-in only, via
   `isaac_kitchen_hug` or `CASCADE_GRASP_BACKEND=hug`, under GraspGen-X's
   required/optional contract: never an OBB substitute when required, and
   bounded-search aware. The launcher refuses a `--graspgenx` override of
   that profile. Every existing profile's backend and call shape are
   pinned unchanged. `tests/test_hug_backend.py` and
   `tests/test_hug_runtime.py` (63 tests: 58 RED against main c5012e7, and
   the 5 golden pins of existing behaviour pass there by design) include a
   mock-stack `grasp_object` that executes a stub HUG pinch through the
   harness.

   Measured on the 5-DoF mock SO-101: the stub's ~8° tilted palm approach
   fails pregrasp IK for every pinch; `pinch_approach: vertical` reaches
   them. **Still open (parent):**
   - real weights + MANO on a CUDA host;
   - the live Isaac A/B `isaac_kitchen_hug` vs `isaac_kitchen_gpu`, which
     is the actual answer to "do HUG grasps clear this arm's IK
     envelope";
   - latency.

   No claim about grasp quality, success rate or GraspGen-X comparison.
8. ~~**A `programs` tier (Waddle).** cascade has primitives (`TOOL_SPECS`)
   and skills (`skills_library/*.md`, ASPIRE-distilled) but nothing above
   skills: an agent-composed, reusable, task-level script distinct from a
   one-off orchestrator run. Scoping question before landing: does a
   "program" get authored the same way ASPIRE distills a skill (diagnose a
   successful multi-skill run, persist it), or does the agent write one
   proactively? Needs a design pass, not a first draft in this file.~~
   **landed 2026-10-08** — design pass `docs/PROGRAMS_TIER.md` answers the
   scoping question with **both, under one admission rule**: a program is
   AUTHORED by the brain for one instruction (one text turn, tier 2.5:
   consulted only when no reflex/habit plan exists) or DISTILLED from a
   verified LLM-tier run, and either way authorship is never evidence — it
   is stored only from an execution whose every registered effect the
   task-effects ledger CONFIRMED, and offered for reuse only once verified in
   ≥2 distinct tasks (ASPIRE's promotion rule, the same constant) and more
   often than it failed. `agent/programs.py` (contract, runner, authoring,
   distillation) + `memory/programs.py` (`runs/programs.jsonl`): a program is
   a bounded list (≤12 steps) of REGISTERED tool calls with labels as
   parameters and positions only as `localize_object(label)+offset` queries
   (#9's recipes) re-grounded before the first motion; each step is a
   top-level `SkillRuntime.execute()` call with its own trace row and
   three-state postcondition, the harness stays the sole motion authority,
   and the first failed / refused / refuted / unverified step stops the
   program with a `next_action` for tier 3 (a `stuck` step ends the task,
   #13). Opt-in: `agent.programs: false` (default) / `CASCADE_PROGRAMS=1`.
   Measured on the mock stack with MockLLM-scripted brains
   (`tests/test_programs_tier.py`, 40 tests, RED against a c5012e7 export):
   an authored program runs as trace rows `localize_object, grasp_object,
   place_at` with `place_at` at the re-grounded cube + offset, both effects
   CONFIRMED (stand-in physics channel, as in the recipe tests), one LLM
   turn; an unverified grasp, an e-stop refusal and an unresolvable anchor
   each stop with zero later motion and hand tier 3 the reason; a candidate
   is never offered and a `use` of it is refused; a promoted program is
   reused with new bindings; an authored and a distilled program of two
   instructions fold into one promoted record; `programs=None` reproduces
   main's orchestrator path write-for-write (golden). **Still open:** no
   real brain has authored a program (the mock brain is never given the
   tier; whether Cosmos3/Gemma write valid programs is unmeasured), no
   physical or Isaac run, ~~not exposed to MCP chat hosts~~ **landed
   2026-10-09** (B42: with the tier on, an arm MCP server serves
   `list_programs` + `run_program` through the same `ProgramTier`/runner and
   admission rule, withheld by the capability matrix without the library or
   the verifier, a cancel mid-program latches the e-stop and dispatches no
   later step; `agent.programs: false` keeps catalog, calls and `world_state`
   byte-identical; `tests/test_programs_mcp.py`, 25 tests on the mock stack,
   23 RED against a 4e896c3 export, the 2 golden pins pass there; the same
   file found and pins a lost-update bug on main: two processes sharing
   `programs.jsonl` counted 40 of 80 admits, now 80 under an advisory lock), no
   loops/branches (skill graphs cover outcome routing on the composed
   runtime), motions without a registered postcondition keep a program out of
   the library, ~~retrieval is keyword overlap~~ **landed 2026-10-09** (B42:
   with `memory.embedder` set, programs are ranked by text embedding with the
   skill library's floor-or-guard rule; keyword overlap stays the default;
   measured with the dependency-free `hash` embedder only, SigLIP/CLIP text
   floors uncalibrated). Still open after B42: no real local brain (Qwen) has
   authored or reused a program over MCP, no Isaac run of `run_program`.

## Landed 2026-09-09: the demo verifies itself in MuJoCo, and one click brings it up

Third pass over the same source list (Agentic-VLA 2605.22896, ASPIRE,
RPent/Harness-VLA, VIA 2607.11119, Waddle, Pigey, Human-CLAW, LaMem-VLA,
HUG). What changed upstream since 08-27, checked against the actual pages:

- **Harness-VLA v4 (2026-09-02)** — RoboCasa365 margin 25.4→27.1 pp, no new
  mechanism; but the docs now spell out *Task-Specific Memory*: a successful
  run serialized as JSONL with concrete xyz REPLACED by symbolic perception
  queries, plus a semantic summary, re-grounded at replay. That is exactly
  the gap in tier-2 `ExperienceMemory` (keys on text, stores raw args) and
  the concrete shape for follow-up #8. Also `finish(status=stuck)` as a third
  outcome and `view_env_state(step=N)` step recall. RPent repo pushed 09-08
  (Franka + dual-Franka real-robot, non-reasoning mode −40 % runtime).
- **ASPIRE code is public** (github.com/NVlabs/ASPIRE, pushed 2026-09-01).
  `skills/library.py` promotes a distilled skill only when it recurs in ≥2
  *distinct tasks*; this repo's `agent/aspire.py` dedupes by (skill, signature)
  with no cross-task gate, so one lucky repair is retrieved as if proven
  (closed 2026-10-07 by follow-up #10 below).
  Its `launch_servers.py` (readiness waits, dependency order, refuse-if-
  session-exists) is the shape `scripts/launch.sh` follows.
- **Pigey code is public** (github.com/lianegalanti/Pigey, `real/agent-
  system.md`): occlusion-search protocol (lift the largest hollow occluder,
  park it +0.2 m, re-perceive, resume the ORIGINAL task) and memorize/
  restore (snapshot centroids → LookAway → diff → restore blocker-first).
- **Waddle** published `/developers/waddle-stack`: a capability matrix drives
  graceful degradation (no depth → RGB-only tools offered). cascade does this
  by hand via `CASCADE_HIDE_TOOLS`.
- Agentic-VLA, VIA, LaMem-VLA, Human-CLAW: unchanged; nothing new that is
  feasible without a trainable VLA backbone. HUG shipped code + weights
  (2026-09-04) — still a GPU sidecar, still follow-up #7.

What landed, all measured on this CUDA-less Mac (suite 623 → 639 passed,
0 skipped; each new guard shown to FAIL under a deliberate mutation):

- **Independent verification channel for MuJoCo.** Before: a MuJoCo pick
  ended `postcondition: unverified — "only in the belief channel it also
  wrote"`, because `sim/truth.py` knew Isaac only. Now `MujocoTruthReader`
  reads the prop's true pose (`data.xpos`) from the arm's own world, so the
  chat user sees `status: confirmed, channel: physics, moved 19.8 cm, 3.9 cm
  from the drop point` — or `refuted` when the cube did not move.
- **One shared physics world** (`sim/mujoco_world.py`): arm, rendered camera
  and truth channel hold the same `MjModel/MjData` by MJCF path (refcounted,
  one lock). Previously the arm stepped a private copy and the camera painted
  a synthetic cube — two worlds that could disagree without anyone noticing.
- **A rendered camera** (`perception/mujoco_camera.py`, `type: mujoco`,
  profile `configs/cameras/mujoco_scene.yaml`): RGB-D from the live world via
  `mujoco.Renderer`, extrinsics per frame. Measured against physics truth:
  bbox exact to the pixel, depth 0.550/0.600 m as designed, lateral error
  0.7 mm. The prop moved from base (0.20, 0) — which sat directly under the
  SO-101's HOME TCP, 100 % occluded from a top-down camera and only ever
  "visible" because the mock camera painted it — to (0.20, +0.10).
- **Lazy truth binding for MCP mode** (`LazyTruthPoseFn`): under OpenClaw the
  arm is a `LazyArm` built on the first motion, so a startup-time
  `make_truth_pose_fn` returned None and every chat-driven pick verified
  against beliefs only — the headline check silently off in exactly the mode
  the demo is shown in. The channel now binds on first use, and can bind
  through the rendered camera's world BEFORE the arm exists.
- **Displacements never mix channels** (`PostconditionChecker.
  _comparable_start`). Found by running the real chat path: a belief
  RESTORED FROM DISK (last episode's drop point) served as the pre-motion
  snapshot, physics served the post-motion pose, and "physics_after −
  belief_before = 4.8 cm" refuted a pick that visibly succeeded. A
  displacement is a difference of two readings of the SAME channel; on a
  mismatch the check reports the final pose and `start_channel_mismatch`
  instead of a false REFUTED.
- **`scripts/launch.sh` — one click.** `--sim auto|isaac|mujoco|none`: Isaac
  bridge under Isaac's python with a readiness wait on :8611, or MuJoCo (the
  MCP server owns the world and opens the viewer — under `mjpython` on
  macOS, where plain python refuses `launch_passive`), or nothing for real
  hardware (`--sim none` demands explicit `--arm/--cameras`; it never guesses
  which robot is plugged in). Then OpenClaw ≥ 2.0 guard (upgrade stops the
  gateway first — `doctor` fails while it runs), idempotent `openclaw mcp set`
  (`mcp add` errors on an existing name), gateway restart (stale tool list
  otherwise), `mcp probe --json` must list `pick_and_place` etc. (2.0's plain
  probe prints only a count), a trivial brain turn, then `openclaw dashboard`.
  `--brain auto` prefers an answering local server, else keeps OpenClaw's own
  auth. `--dry-run`, `--down`, no `ss` (python socket probe; macOS).
  Verified end to end on OpenClaw 2026.9.3: 39 tools listed; a chat turn
  called `get_observation` and answered "one red cube at (0.201, 0.100,
  0.050) m"; a chat-driven `pick_and_place` executed in physics.

Open follow-ups from this work:
9. ~~**Task-Specific Memory recipes (Harness-VLA v4).** Store successful runs
   with xyz replaced by `localize_object(label)+offset` queries and re-ground
   at replay; this is the shape for #8 and fixes tier-2's text keys. (M)~~
   **landed 2026-10-07.** `memory/recipes.py` + tier-2 `ExperienceMemory`:
   a VERIFIED LLM-tier run is stored as a recipe (`runs/recipes.jsonl`, one
   per line, beside `experience.json`) whose motion steps carry
   `{"$target": {"query": "localize_object", "label", "offset_m"}}` where the
   run had `place_at` coordinates — anchored on an object perceived before
   the first motion, preferring a non-held anchor (the bowl) over the
   manipulated object's own start pose; drop-zone names and labels stay
   symbolic; a coordinate with no anchor refuses the whole recipe rather
   than storing a raw value. At a tier-2 hit the orchestrator grounds every
   query through the runtime's `localize_object` BEFORE any motion; a query
   that fails aborts the replay to the LLM tier with a note and zero motion
   (there is no stored coordinate to fall back to), and a replay outcome
   never overwrites the stored queries with the grounded coordinates.
   Measured on the mock stack with the cube rendered 6 cm from where the
   recipe was learned: the replay re-grounds, places relative to the NEW
   position, never calls the LLM; an anchor missing from the table aborts
   with no `_MOTION_SKILLS` call executed; pre-recipe `experience.json`
   entries load and replay unchanged (`tests/test_task_recipes.py`, RED 15
   failed on main → GREEN). Still advisory: the harness vets every grounded
   motion; this is not a physical acceptance of any task.
10. ~~**ASPIRE cross-task promotion gate.** `agent/aspire.py`: promote a
    distilled skill only when seen in ≥2 distinct tasks (`occurrences`,
    `source_tasks`); the scoped retry-admission gate still permits retrieval
    after one confirmed retry, without cross-task validation. (S)~~
    **landed 2026-10-07** — `skills/library.py` front matter counts
    `occurrences` (distinct runs, idempotent re-harvest via `source_runs`) and
    `source_tasks` per `(skill, signature)` note; `aspire.retrieve()` injects
    only notes promoted by ≥ 2 distinct tasks (`PROMOTION_MIN_TASKS`), single-
    task and legacy notes stay stored candidates, `memory.skill_min_tasks: 1` /
    `--min-tasks 1` is the explicit relaxation. Measured by
    `tests/test_aspire_promotion.py` (11 tests RED against main's
    `aspire.py`/`library.py`, GREEN after) plus a two-harvest CLI run
    (`learned 1 → 0 → 1`, `status: candidate → promoted`); no physical trial,
    no claim that promoted notes improve success.
11. ~~**Pigey snapshot/restore + occlusion search** as composite skills over
    `BeliefStore` (`snapshot_scene`/`restore_scene`, `search_for_object`):
    the one demo beat visible from chat that no current skill covers. (M)~~
    **landed 2026-10-07.** `BeliefStore` keeps named ADVISORY
    `SceneSnapshot`s (confirmed objects' label/colour/centroid/extent;
    saved/loaded with the beliefs, dropped by `clear()`); `snapshot_scene`
    writes one with no motion; `restore_scene` diffs it against current
    beliefs colour-first, moves only objects displaced beyond `tolerance_m`,
    blocker-first (a swap cycle parks one object on free, IK-reachable table
    inside the workspace), each move `_grasp_with_persistence` +
    `skill_place_at` like `sort_by_color`, bounded by `max_moves` and the
    per-task persistence deadline; `search_for_object` lifts the largest
    hollow/large occluder, parks it +0.2 m (0.15/0.12 fallbacks) on free
    reachable table, re-perceives, and on a sighting returns
    `task_complete: false` + "resume the ORIGINAL task", else `ok: false,
    stuck: true`. Postconditions `restored`/`searched` (`agent/effects.py`)
    confirm/refute only on the physics channel and stay `unverified` on the
    belief the place itself wrote. Measured: 18 new tests in
    `tests/test_pigey_scene_memory.py` (RED 18 failed on b5477d8 → GREEN);
    on the rendered two-prop MuJoCo world: a physics-confirmed
    `pick_and_place` of the red cube, `snapshot_scene` of that layout, the
    props teleported back to spawn behind the robot's back
    (`MujocoWorld.reset_props`, belief store not told), then
    `restore_scene` ignored the remembered drop-zone belief, moved only the
    red cube back to within 5 cm of its memorized spot (physics truth, blue
    cube untouched < 1 cm) and its `restored` verdict came from the physics
    channel; on the static mock stack the same restore drove the arm through
    `SafeArm` and its verdict stayed `unverified`. Found on the way: the
    MuJoCo release-escape planner refuses a plain `place_at` next to a
    neighbour 7 cm away (the spawn layout) and at several free spots
    (`no collision-clear release escape`), so a restore to the spawn layout
    ends as an honest `ok: false, stage: place, holding: red cube` — the
    harness stays the authority; nothing was relaxed.
    Not claimed: physical acceptance on a real rig, any change to safety
    limits, planners or physics assets, occluder recognition beyond label
    words and footprint size.
12. ~~**Capability matrix → tool surface (Waddle).** Compute `_EXCLUDED_TOOLS`
    from what the rig can do (depth, sidecars, n_arms) instead of
    `CASCADE_HIDE_TOOLS` by hand. (S)~~ **landed 2026-10-07** —
    `apps/capabilities.py` derives the matrix from the BUILT runtime's probed
    state (per-camera depth chain via `DepthProvider.depth_source_for`, the
    sidecar probes behind `runtime.backends()`, `ArmRig` length, bases,
    verifier, memory; tri-state, unknown never withholds) and
    `TOOL_REQUIREMENTS` trims the MCP catalog by it: RGB-only rig → the 3D
    tools are withheld and rejected with the reason, single arm → `list_arms`
    and the injected `arm` parameter go, dead GraspGen-X/occupancy → reported
    fallback, nothing hidden; `CASCADE_HIDE_TOOLS` stays the operator
    override, `_EXCLUDED_TOOLS` keeps `task_done` out. Reported in the
    `[cascade] capabilities:` banner, `/state`, `world_state.tools_withheld`
    and the server log; a catalog listed before the build is refreshed via
    `notifications/tools/list_changed`. Measured on the mock stack over real
    JSON-RPC (`tests/test_mcp_server.py`, 10 new tests, each RED on main):
    `mock_rgb` withholds 13 tools and keeps 28 RGB/motion tools, a table
    plane keeps the grasp tools and withholds only `place_on_object` (and
    `list_arms`, one arm), the
    default rig lists 40 of 41, `CASCADE_ARMS=so101_left,so101_right` lists
    all 41 with `list_arms` naming both arms. No skill, limit or asset
    changed; nothing here was run on a physical rig.
13. ~~**`recall_step(n)` + a `stuck` outcome (RPent).** Trace keyframes already
    exist per step; expose them, and let motion skills return a human-
    actionable ask distinct from failure. (S)~~ **landed 2026-10-07** —
    `recall_step(n)` (34th skill, 42 MCP tools) reads the trace row +
    BEFORE/AFTER keyframes back for the planner (shown once on its next
    turn) and the chat host (image content items, one caption each, like
    `task_memory`); every result now carries `outcome: ok | failed | stuck`,
    where `stuck` is `ok: false` + a human-actionable `ask` from the
    persistence loops (`pick_and_place`, `_grasp_with_persistence` →
    `handover`/`sort_by_color`), the orchestrator ends the task on it without
    a retry and `summary.txt` records `outcome: stuck`. Measured on the mock
    stack (`tests/test_recall_step_stuck.py`, 12 tests): recalled bytes ==
    the recorded keyframe files; a 2-attempt budget yields exactly 2 grasp
    attempts, one pick, zero LLM re-plans; e-stop stays a plain failure. No
    physics, limit or asset change; no physical acceptance.

## Landed 2026-09-09 (second pass): sidecars that tell the truth, ROS2 arms, an outcome judge

Prompted by one question — *"is the demo really using nvblox + GraspGen-X?"*
— answered by listening on the ports the config names: **neither was
running**. GraspGen-X at :5556 got 54 connection attempts, the "nvblox"
bridge at :5557 got 226; every grasp waited out an 8 s timeout and fell back
to the analytic OBB planner, every clearance check read `cached=None` and
became a no-op, and nothing in the banner, the dashboard or `summary.txt`
said so. The bridge script that did exist was an Open3D/numpy voxel
aggregator, not nvblox. Fixes, all measured on this CUDA-less Mac (suite
664 → 708 passed, 2 deselected, 0 skipped):

- **Sidecars are probed at startup and reported, never assumed.**
  `OccupancyMap.probe()` / `GraspGenXPlanner.probe()` (300 ms each, `health`
  is what the real GraspGen-X server answers) decide once; the banner prints
  `backends: grasp_planner=… | occupancy=…`, `runtime.backends()` feeds the
  dashboard `/state`, and `TraceLogger.finish` appends the same line to
  every `summary.txt`. A dead server latches (`_graspgenx_down`) so later
  grasps go straight to OBB instead of paying 8 s each; the summary reads
  `grasp_planner=obb (graspgenx down)`.
- **One occupancy bridge, three backends** (`perception/occupancy_backends.py`,
  `scripts/serve_occupancy_bridge.py`): `nvblox` — real nvblox_torch TSDF +
  ESDF, P0 wherever the CUDA wheel installs (x86 Linux; no aarch64 wheel,
  confirmed against the v0.0.10 release); `warp` — the hardware-agnostic
  default: projective TSDF with free-space carving and an exact separable
  Euclidean distance transform in Warp kernels (CPU here, CUDA on Jetson/
  x86; 2.8 ms/frame at 1 cm voxels on 320×240 depth, EDT byte-equal to
  `scipy.ndimage.distance_transform_edt`); `voxel` — numpy, no distance
  field. The wire now ships a **distance grid** (`query` → `grid`, `origin`,
  `voxel`) so the harness reads clearance with one trilinear lookup instead
  of a brute-force nearest-point over a cloud; older bridges (cloud only)
  still work. Survey behind the choice: pywavemap (CPU occupancy, no ESDF),
  Open3D `VoxelBlockGrid` (no distance queries beyond an 8 cm band),
  vdbfusion (unmaintained), Bonxai (C++ only), cuRobo (CUDA-gated).
- **Robot-body masking** (`perception/robot_mask.py`). The first live run of
  the clearance gate refused the very first grasp: the camera sees the arm,
  the arm became an obstacle, and three of five link points at HOME read
  0.000–0.007 m clearance against a 0.03 m minimum. Every production ESDF
  stack masks the robot first; cascade now thickens the arm's own link
  polyline (FK, per-arm `body_mask_radius_m`, default 6 cm) and zeroes those
  depth pixels before integration — "no measurement", not free space. After:
  0.065–0.229 m at HOME, the pick runs, and the harness consulted the map
  **991 times with real data** during one `pick_and_place` (physics-confirmed,
  3.9 cm from the drop point).
- **ROS2 arms.** `control/unitree_arm.py` (Unitree Arm SDK: `rt/arm_sdk` +
  `rt/lowstate`, LowCmd CRC verified against `unitree_hg`/`unitree_go`),
  `control/ros2_arm.py` grew `GripperCommand` action support (what
  `franka_ros2` ships, not the Float64 topic hand-written examples use).
  Profiles derived from vendored URDFs and measured: FR3 (`down_open` 0.81
  strict solve rate over the workspace grid, home ±0.55 rad), G1 right arm
  (0.81, home ±0.92 rad), H1 / H1-2; FR3 + Franka Hand composed via Menagerie
  `<attach>` with 0 mm TCP disagreement between URDF and MJCF.
- **`scripts/launch.sh` starts the sidecars** (occupancy bridge; GraspGen-X
  protocol stub in sim modes, `--graspgenx external` for a CUDA box), waits
  on each port, probes the bridge and prints the backend in the READY banner;
  `--down` stops only what it started. Verified: `occupancy: :5557 warp
  TSDF+EDT on cpu (141x121x86 @ 1.0 cm)`, `graspgenx: :5556 (stub)`, 39 tools
  listed, a chat-driven `pick_and_place` confirmed in physics.
  **Follow-up (backlog B34):** ~~the launcher started the bridge and the
  occupancy sidecar on `CASCADE_BRIDGE_PORT` / `CASCADE_OCCUPANCY_PORT`, but
  the runtime still dialled the profile ports 8611 / 5557 (only GraspGen-X
  read its variable; LOCAL_RTX_VALIDATION.md, profiling attempt 07)~~
  **landed 2026-10-08** — `load_demo_config` applies the three
  `CASCADE_*_PORT` variables last, over demo.yaml, booth.yaml and every arm's
  `overrides:`, to the top level and each arm's `resolved` view (every
  `type: isaac` camera/arm `bridge_port`, `grasp.graspgenx.port`,
  `occupancy.port`); `load_profile` does the bridge port for the standalone
  viewer/recorder; `launch.sh` registers the bridge/occupancy variables with
  the MCP server. Empty = unset; anything but ASCII digits in 1..65535 raises
  naming the variable; base profiles keep `CASCADE_MICRODUCK_BRIDGE_PORT`.
  Measured on CPU (`tests/test_port_env_overrides.py`, spies on the bridge
  `create_connection` and the ZMQ `connect`): every client dials the override.
  With no variable set, 170 resolved configurations (every shipped arm,
  camera, base and robot profile) are byte-identical to 133876c; with all
  three set, only 705 `bridge_port`, 271 `graspgenx.port` and 271
  `occupancy.port` values differ. Left open then: ~~`CASCADE_HUG_PORT` and the
  `*_HOST` variables stay construction-time reads, `scripts/setup_agents.py`
  forwards none of the ports~~ **landed 2026-10-09** (B41) —
  `load_demo_config` also applies `CASCADE_HUG_PORT`, `CASCADE_GRASPGENX_HOST`
  and `CASCADE_HUG_HOST` last, to every view (a host is a hostname or IPv4
  address: 1–253 ASCII letters, digits, `.`, `-`, `_`, the first a letter or
  digit; anything else raises naming the variable; empty = unset). A planner
  dials what its resolved section says and falls back to a variable only for a
  key the section lacks, so an override written into `cfg._data` after loading
  is no longer undone at construction; `setup_agents.py` copies every
  `CASCADE_*_PORT` / `CASCADE_*_HOST` set in its shell into each host's entry
  (checked by the same rules); `serve_hug.py` reads `CASCADE_HUG_PORT` by the
  client's rule. The bridge and the occupancy sidecar get no host variable on
  purpose (launch.sh starts both on this machine; the bridge binds loopback by
  default). Measured on CPU
  (`tests/test_endpoint_env_overrides.py`, same spies): with no variable set,
  213 resolved configurations and the registrar's output for all four hosts
  are byte-identical to 4d0947b; with all six set, only 1395 `bridge_port` and
  396 each of `graspgenx.port`/`.host`, `hug.port`/`.host` and
  `occupancy.port` values differ. Still open: no live Isaac run on private
  ports yet, and `launch.sh --graspgenx external` still probes
  `127.0.0.1:$CASCADE_GRASPGENX_PORT` whatever `CASCADE_GRASPGENX_HOST` says.
- **Robo-Dopamine as the outcome judge** (`eval/progress_judge.py`,
  `scripts/judge_run.py`; https://robo-dopamine.github.io/). The GRM is a
  VLM prompted with the task, optional START/END references and BEFORE/AFTER
  images that answers `<score>+NN%</score>` — RoboChallenge's "PRM-as-a-
  Judge" for VLA policies. cascade uses it exactly that way: an EXTERNAL
  judge over the BEFORE/AFTER keyframes the runtime already records per
  skill (the first skill of a run used to log `keyframe_before: null`; it
  now grabs a frame first). Upstream prompt and fusion arithmetic verbatim
  (incremental / forward / backward), single-view + blank-goal are upstream's
  documented usages. Backends: `grm` (the released GRM-2.0 checkpoint behind
  vLLM's OpenAI server on a CUDA box), `vlm` (same prompt against any
  OpenAI-compatible vision model — default routes through the OpenClaw
  gateway's `/v1/chat/completions` so the judge shares the demo's
  credentials), `fake` (tests). The point is the **calibration**: every hop
  is cross-checked against the physics postcondition channel, so the
  judge-vs-physics confusion matrix (`tp/tn/fp/fn`, agreement) is computed
  per run before anyone quotes the judge's number — GRM was trained on
  multi-view real + LIBERO/RoboCasa footage, and a top-down rendered MuJoCo
  camera is a new distribution. `per_tier()` gives hop-per-second by
  dispatch tier (reflex / experience / LLM / mcp-host — now written on
  every trace row), the agentic-policy metric. First calibrated run on
  this rig: the judge scored **0 %** on a physics-confirmed pick (fn=1) —
  and was right: the runtime had saved byte-identical BEFORE/AFTER
  keyframes (the AFTER was the skill's last observed frame, taken before
  the place motion). After the fix (fresh frame after every motion
  skill, pinned by `tests/test_keyframes.py` with a per-grab-distinct
  camera and shown to fail with the fix reverted) the same pick scores
  **+0.45, tp=1, agreement 100 %** (GPT-5.6 via the OpenClaw gateway).
  Not a training signal: Dopamine-RL needs a gradient-trainable policy and
  cascade's Cosmos3 + skills tiers have none.

- **`./run.sh` — the shareable one click.** `scripts/launch.sh` assumed a
  venv, the extras and the OpenClaw CLI already existed; a colleague cloning
  the repo hit "no python found". Now `run.sh [isaac|mujoco|check|down]`
  wraps `launch.sh --setup`: uv venv (python 3.12), `cascade[<extras for
  the mode>]` (isaac needs no `sim` extra — physics lives in Isaac's python,
  cascade talks TCP to the bridge), robot assets, OpenClaw CLI install, the
  ONE interactive step (`openclaw onboard`, only when no model server and
  no provider auth exist), then the usual launch with a probed Isaac bridge
  (`ping` + `state` through the real client, not just an open port).
  `run.sh check isaac` is a preflight that lists every missing piece
  (Isaac python, GPU, USD, weights, CLI, extras) and starts nothing.
  Measured on a fresh copy (no venv): three real defects fixed on the way —
  (1) OpenClaw's tool probe allows 1.5 s but a fresh venv's first
  `tools/list` took **34 s** (bytecode compile of torch/ultralytics/cv2), so
  the launcher's own proof step failed with "MCP tool listing timed out";
  it now warms the imports once (0.1 s afterwards). (2) macOS ships bash
  3.2, where an empty array is "unbound" under `set -u` — `run.sh` died on
  its own setup flag; arrays replaced by strings in both scripts. (3) The
  occupancy bridge came up on the numpy `voxel` backend because nothing
  installed warp/scipy — new `occupancy` extra, installed by default, pinned
  by a test. The brain proof now reads the `agent exec` envelope's `final`
  and retries 3× (first run after `gateway restart` can race auth warm-up).
  Two more found by the new end-to-end proof: (4) the fresh venv lacked
  `pinocchio` — the wire extras were installed, the kinematics extra was
  not, and the visitor's first pick answered "hardware stack failed to
  initialize"; the launcher now builds the runtime once before READY, and
  each mode installs the extras it needs. (5) The chat model sends
  `arm=""` (fills every optional field) and then `arm="default"` (the name
  `list_arms` reports for one arm); both were refused as unknown, costing
  two failed tool calls per pick — `_select_arm` now maps ""/default/
  primary to the primary while a wrong name still fails loudly. Each MCP
  tool call is now logged to `runs/mcp_<pid>/server.log`, since OpenClaw
  surfaces only `failures: N`. Final fresh-copy run: `tools=2 failures=0`,
  pick confirmed by physics, from an empty venv in one command.

Open follow-ups from this pass:
14. **GRM on the GPU box.** Serve `Robo-Dopamine-GRM-2.0-8B-Preview` with
    vLLM (`--limit-mm-per-prompt image=8`) on the Spark/Jetson and re-run
    `judge_run.py --judge grm` over the same runs; compare its
    judge-vs-physics agreement with the API VLM's. (S once the box is up)
15. ~~**Wrist camera for the judge.** GRM's prompt reserves two wrist slots;
    the SO-101 has none, so both repeat the front view. A wrist `<camera>`
    in the MJCF scene (and `Frame` wrists in `build_images`) would exercise
    the model as trained. (S)~~ **landed 2026-10-07** —
    `configs/cameras/mujoco_wrist.yaml` (`type: mujoco`, `role: wrist`,
    `mj_attach: {body: gripper, T}`): `write_demo_scene` declares the
    `<camera>` inside the SO-101 gripper body through a verbatim ElementTree
    copy of the robot's include chain (asset untouched; compiled physics
    pinned exactly equal to the plain scene plus one camera), the runtime
    writes `keyframe_{before,after}_wrists` per motion skill, and `judge_run`
    fills the wrist slots from them (one stream → both slots; none → the
    documented front repeat), with every record naming the slot sources
    (`StepVerdict.wrist_slots`, `wrist=` in the summary line). Measured on
    the rendered world (`tests/test_wrist_camera.py`, EGL): gripper subtree
    = 33,973 px of the wrist frame at every pose; red-cube bbox in the wrist
    frame 46,410 px at TCP z = 0.08 m and 66,123 px at z = 0.05 m over the
    prop (absent at home), straddling the frame centre between the jaws,
    while the front camera loses the prop under the arm. **Not claimed:** any
    change in judge-vs-physics agreement (no GRM/VLM re-run; #14 still
    pending), a hand-eye calibration for the wrist view (no `extrinsics`, no
    fusion; the wrist-cam extrinsics follow-up stays open), or Isaac/real-rig
    wrist keyframes (eye-in-hand profiles qualify via `is_wrist_view`, but
    the extra per-skill grab was not measured on the bridge).
16. **nvblox on aarch64.** No wheel for Jetson (JetPack 7) as of v0.0.10;
    track the release and switch `auto` to prefer it there once it exists —
    the backend code already runs it. (blocked upstream)

## Landed 2026-09-10: the planner remembers what it did (Vesta memory harness)

Source: NVIDIA GEAR, *Vesta: A Generalist Embodied Reasoning Model*
(arXiv:2606.20905, June 2026). No weights or code are released ("when
releasing assets in the future"), so nothing of the MODEL is usable; the
paper's transferable result is about the **harness** around any planner:

- **Image+text history beats text-only by 26 points on their planner suite**
  (Table 5: text-only 49.7, image-only 63.1, image+text uniform 75.9). Their
  diagnosis of text-only: the planner "learns to be overly reliant on the
  history text shortcuts, leading to excessive 'continue the current task'
  predictions". cascade was exactly text-only: `_prune_images` kept one
  image in context, `EpisodicMemory.digest()` was text, and the thumbnails
  the memory already stored per event had **zero consumers**.
- **The harness is minimal**: memory tuple ⟨step, time, frame, action,
  goal⟩; up to K past frames, the first always kept (initial state), the
  rest sampled -- uniform and recency-biased "perform on par", so uniform.
  Four reasoning phases before each action (Observation, Progress,
  Reasoning, Action); only the action is written to memory.
- **Their demo tasks are chosen so a memory-less actor structurally fails**
  (Count Fruits, Find Object without re-opening a drawer, Memorize Candy):
  +38.3 % success over actor-only on the real robot.

Landed, each with a test proven by mutation:

- `memory/episodic.py`: `memory_frames(k)` sampler + `frame_caption()`;
  frames live in their own ring with a task-scale horizon (600 s: the 15 s
  text window would forget the initial state before one ~20 s pick
  finished) and `reset_frames()` per episode.
- `skills/runtime.py`: every MOTION skill's memory event carries its AFTER
  frame and the independent postcondition verdict; the first motion of an
  episode pins the scene before anything moved when no observation frame
  anchors it yet. Observation skills add no frame (near-duplicates).
- `agent/orchestrator.py`: `_with_memory_harness()` appends ONE trailing
  message per request with K captioned past frames + the current view;
  images enter the request there and nowhere else (never accumulate);
  `memory_frames_k=0` reproduces the old text-only path exactly.
  `prompts.SYSTEM_PROMPT` asks for the four phases and says a REFUTED step
  did not happen. `memory.frames_k` / `frames_horizon_s` in demo.yaml.
- `apps/mcp_server.py`: `task_memory` tool -- the same frames for a chat
  host as image content items with one caption each, `new_task: true` to
  start an episode. 40 tools now.
- **A task where memory is visible**: `configs/cameras/mujoco_scene_two.yaml`
  adds a blue prop (`extra_props:`); `sim/demo_scene.py` writes N props
  from one profile, the mock detector finds N colours, the physics channel
  already handled N free bodies. `tests/test_memory_task_mujoco.py` runs two
  chat-style `pick_and_place` calls on the rendered world: both
  `postcondition: confirmed / channel: physics`, three memory frames with
  verdicts ["", confirmed, confirmed], both props within 6 cm of the drop
  zone in physics truth, `count_objects` = 2.

Two bugs the two-prop scene exposed that one prop never could (the "second
engine" rule again -- a second OBJECT is also an independent channel):

1. **The mock detector returned every colour for any query** sharing the
   token "cube", so `localize("blue cube")` got the red prop first (equal
   confidence) and the blue belief sat one cube-width off truth (3.5 cm).
   Colour words in the vocabulary now select the colour.
2. **The belief store fused two props of different colours** because
   proximity matching (for label aliases of ONE object) ignores colour and
   3.5 cm cubes 5.8 cm apart are inside the 8 cm gate: `count_objects` said
   1. Two confirmed, different mask colours are now two objects however
   close; same-colour aliases still fuse; colour-less observations keep the
   old rule.
   **Follow-up (backlog B31):** ~~same-colour IDENTICAL props inside the
   8 cm gate still blurred into one belief (ARCHITECTURE "Known
   limitations")~~ **landed 2026-10-08** — a camera frame is fused as a
   whole (`BeliefStore.update_frame`): detections sharing image support are
   one instance, instances and beliefs are matched one-to-one by a min-cost
   assignment inside the unchanged gates. Measured with
   `scripts/measure_same_colour_sweep.py` (two rendered 3.5 cm red cubes,
   MuJoCo physics truth, the real WorldWatcher path,
   `benchmark/results/same_colour_separation_sweep.json`): every pair the
   detector returns as two detections is two beliefs at single-cube accuracy
   (top view from 3.75 cm centre distance; the old store merged every such
   pair below 8 cm, 1.6–3.6 cm off). Live on Isaac (6.2 PhysX, YOLOE
   prompt-free, both demo cameras, PhysX truth;
   `docs/evidence/b31-isaac-same-colour-20261008/`): identical pink twins
   5–9 cm apart were two beliefs in 9/10 runs (old store 0/10, although YOLOE
   gave 2–3 detections on the pair), and the 3-prop open-vocabulary scene
   scores the same with either store. Still open: pairs the DETECTOR returns
   as one detection (touching cubes; 5.5–7.0 cm in the oblique probe view;
   the default mock detector's one blob per colour at any distance), and two
   bare-scene errors seen with BOTH stores on this build. The first is a phantom
   on the arm's own upper link (side camera; fusion ignored the robot mask);
   **landed 2026-10-08 (B32a)**: `WorkspaceFilter` gates both fusion
   paths on the render self-mask. Live, the phantom rate went from
   17.9–19.6 % to 0 % in 3/3 runs, with precision 100 % and recall 100 %
   (`docs/evidence/b32-fusion-self-mask-20261008/`). ~~The second is still open
   (B32b): a duplicate bin belief, because the side camera names the bin
   "yellow" and the top camera "orange". Because of it the scene ends with 4
   beliefs for 3 props, so the STABLE count of the 0 % row of
   `SOTA_PERCEPTION_AND_EVALUATION.md` still does not reproduce.~~ The second
   (B32b, a duplicate bin belief: the bin is H 22 "orange" in the top camera
   and H 23 "yellow" in the side camera, 16/16 and 7/7 frames) **landed
   2026-10-08 (B32b + B32c, measured live)**: colour identity is per camera
   (`ObjectBelief.source_colors`, `BeliefStore._identity_ok`). A camera is held
   to the name it gave a belief; a camera that never named it may fuse a
   perceptual-neighbour name only at 3D box IoU >= 0.75 (2nd–98th percentile
   boxes of the two clouds; union, not the smaller box, so a prop inside the
   bin stays apart); red/blue never fuse. Measured on CPU only: the bin's two
   views ray-cast from the calibrated poses score 0.90–0.91 IoU (0.78–0.94
   with 2–4 px of mask bleed), a cube inside it ≤ 0.03 (a bleeding sliver of
   one ≤ 0.68); 23 tests (RED 21/23 on main 133876c, the 2 passing are the
   geometric premises), 20/20 mutants killed. Switch:
   `memory.per_camera_colour` (default true; false = the one-name A/B
   baseline), `memory.neighbour_colour_iou`. ~~Still open: the live Isaac A/B
   (target: bare scene 4 → 3 beliefs, precision unchanged)~~ **live A/B
   2026-10-08**: with the B32b box the bin was one belief in only 5/7 runs,
   because YOLOE's masks take in table pixels and the bin's two views scored
   IoU 0.708–0.926 (median 0.744, 86 logged decisions) against 0.75.
   **B32c landed 2026-10-08**: the overlap box drops each cloud's lowest
   centimetre (above its own 2nd-percentile z), so the same decisions score
   0.870–0.945. The threshold is unchanged. Live: the bin is one belief in
   **5/5 runs, all 60 frames**, 3 beliefs for 3 props, precision and recall
   1.0; the one-name arm keeps 4 in 5/5. A yellow 5 cm prop in or next to the
   bin was never merged (IoU ≤ 0.10)
   (`docs/evidence/b32b-colour-identity-live-20261008/`). Still open: the kitchen
   scene, and a small object named across a band boundary by two cameras
   (cube views overlap 0.66 at 320 × 180, 0.85 at 1280 × 720 in the ray-cast:
   it may stay two beliefs, as before). ~~Also found, pre-existing and not
   changed: with a same-coloured prop inside the bin, the camera that names
   both "yellow" fuses its view of the bin into the prop's belief (identical
   with the one-name rule). The follow-up is a size-consistency check in the
   fusion gate.~~ **Size-consistency check landed 2026-10-09 (B40, CPU;
   live A/B owed)**: a view more than 2× the largest view a belief has had
   (robust horizontal diameter of the real-mask cloud) AND more than 5 cm
   larger is refused (`BeliefStore._size_ok`, `memory.size_gate`, default
   true; false = the old store byte for byte, golden-tested). Measured on
   B32b's live clouds: one object's views within × 1.29 (bin 0.183–0.235 m,
   205 views), container view ≥ × 2.48 and +10.9 cm the prop's largest view
   (0.066–0.074 m); 36 tests, 26/26 mutants killed
   (`docs/evidence/b40-fusion-size-gate-20261009/`). Still open: the live A/B
   (yellow prop in the orange bin, ≥ 3 runs per arm), a belief born from a
   quarter of a container's view, ≥ 4 px of mask bleed onto a container (the
   view of a prop inside it then looks × 2.5 its size; not seen live), and a
   prop's view fusing into the container's belief. ~~Frames without
   a render self-mask (the real rig) need a link-geometry mask in fusion
   (follow-up).~~ **landed 2026-10-09 (B39)**: `perception/link_mask.py`
   draws the robot's pixels from the URDF collision geometry (5 cm link-frame
   cells, each cell's 3D hull) posed by FK at the joint sample nearest the
   frame's capture time. Both fusion paths' B32a gate consumes it like a
   render mask. It is opt-in (`workspace_filter.link_self_mask.enabled`,
   default false = unchanged); a render mask stays authoritative; no joint
   sample within 0.15 s = no mask. Measured on CPU only, against a
   per-triangle rasterisation of the reBot RS meshes (4 poses × 2 cameras,
   1280 × 720): coverage 1.0 and IoU 0.889–0.951 with the shipped 2 px, at
   3–5 ms per frame (`docs/evidence/b39-link-self-mask-20261009/`). Still
   open: the live comparison with the Isaac render mask
   (`scripts/compare_link_self_mask.py`, owed), a hardware measurement, and
   turning it on by default.

Then the REAL chat turn on the two-prop scene ("put both cubes in the drop
zone, one at a time, call task_memory before each action, tell me how many
you moved and how you know") found three more, none reachable by the unit
suite:

3. **`destination: "drop zone"` was localized as an OBJECT.** The planner
   echoed the literal string it had read in a previous result; `pick_and_
   place` tried to detect a thing called "drop zone", failed 8 place
   attempts holding the cube, and the planner spent four minutes inventing
   `place_at` coordinates outside the workspace. The drop zone is a
   configured point: its spellings (`drop zone`, `bin`, `default`, ...) now
   mean "use it", the tool description says so, test pinned.
4. **The host's 60 s per-call budget latched the e-stop.** A persistent
   pick legitimately runs up to `grasp.persist_seconds` (120 s); OpenClaw's
   default `requestTimeoutMs` is 60 s, its `notifications/cancelled`
   mid-motion is (correctly) treated as the operator walking away → e-stop,
   and every later motion failed "e-stop latched". Measured: the pick
   finished at 60.0 s, physics-confirmed, reported as cancelled.
   `launch.sh` now registers the server with `requestTimeoutMs: 300000`;
   the cancel log line names the cause.
5. **The tool log showed only the first caption of a multi-part result**,
   so `task_memory` looked stuck on "memory frame 1" while frames 2..k were
   present. Multi-part results log their LAST text part (the JSON summary)
   plus an image count.

**Between visitors: `reset_scene`.** The launcher's own proof turn moves the
red cube into the drop zone, so the first visitor of the day would start
"put both cubes in the drop zone" with one already there. New skill (tool +
reflex phrases: "reset the scene", "start over", "reinicia la escena",
"nueva demo" -- works with the LLM down): arm home first, every free body
back on its MJCF spawn pose at rest (`MujocoWorld.reset_props`, under the
world lock; Isaac best-effort via the bridge's `reset_props`), held-state
released, beliefs cleared, task frames cleared, one fresh observation. The
launcher calls it after the proof turn and the banner lists it. Physics is
the judge in the test: spawn pose within 2 mm after a confirmed pick moved
it 20 cm, still there after 50 sim steps, perception sees both cubes again.
Three bugs it exposed:

6. The mock detector kept the narrow vocabulary a grasp had asked for
   (`["red cube"]`) across later OPEN scans and hid the blue prop -- the
   real detector's contract is `classes=None` = open world; the mock now
   honours it.
7. `_update_beliefs_from_frame` (the `get_observation` path) did not tag
   the measured colour; only the WorldWatcher path did. Without the tag the
   colour rule from bug 2 cannot fire and a fresh scan fused both cubes.
8. A render already in flight on the camera thread when the props
   teleported completed "after the call" but showed the OLD world (red cube
   still at the drop zone, occluded by the home-pose gripper): the fresh
   scan reported one cube of two, on ~1 in 3 runs. The reset burns one frame
   before observing, so the observation is provably post-reset.

**Audit pass (2026-09-10, after the visitor runs).** Lint (`ruff --select
F,E9`) had never run on this repo: 22 findings, one of them real -- an
undefined name `ObjectFix` in a `runtime.py` annotation (`F821`); the rest
unused imports/variables, all fixed, lint now clean and cheap to keep clean.
Runtime findings, each from a measurement on this machine:

9. **Idle burn.** An MCP server with the runtime up and no tool calls used
   ~85 % of a core: the rendered camera pumped at the real-camera default of
   30 fps and each frame is a full offscreen render (1374/2000 samples in
   `_render`). `mujoco_scene` now declares `fps: 10` (nothing consumes
   faster than the 3 Hz watcher). And the gateway keeps one MCP server per
   chat SESSION forever -- two servers from finished sessions were still
   rendering at 60 % each hours later. `launch.sh --down` now kills ours.
10. **A dead MCP entry costs every turn.** `wrc-demo` pointed at a removed
    venv; every visitor turn paid a failed spawn + catalog retry
    ("[bundle-mcp] failed to start server ... Connection closed").
    `launch.sh` prunes OpenClaw MCP entries whose command -- or `-m`
    module, asked of that interpreter -- no longer exists. (First version
    piped `mcp show` into `python - <<EOF`: the heredoc IS stdin, the pipe
    is silently dropped, nothing was pruned. The listing goes via a file.)
11. **The promised MuJoCo window never opened.** The banner said "the MuJoCo
    window opens on the FIRST motion command"; the arm profile says
    `view: false` and nothing overrode it, and the failure path logged
    through an unconfigured `logging` tree, so there was no trace either.
    `CASCADE_MJ_VIEW=1` (set by the launcher for sim runs) opens it;
    success and failure both print to the server log.
11b. **Opening that window with the display asleep killed the server.**
    First live run after the fix above: the proof turn ran with the screen
    locked; `CGGetActiveDisplayList` returned 0 displays, GLFW had no
    monitor and `launch_passive` segfaulted in `_glfwGetVideoModeCocoa`
    inside the tool call ("MCP error -32000: Connection closed" for the
    visitor, failures=2). The engine now asks CoreGraphics first, skips
    with the reason in the run log, retries on the first motion after the
    screen wakes, and the READY banner reports what actually happened
    instead of promising a window. The launcher's verdict also only reads
    traces written DURING its own turn (it had printed "CONFIRMED by
    physics" from a previous session's trace over a crashed turn).
12. **The cv2 camera window threw an opaque C++ exception in every run**
    ("Unknown C++ exception from OpenCV code"): macOS Cocoa windows must be
    created on the main thread and the RigViewer paints from a worker (and
    the base dependency is `opencv-python-headless`, which has no highgui at
    all). It is now skipped up front with a one-line reason.
13. **`place_at` placed air.** On a visitor run the red cube slipped 4 cm
    into the carry; `place_at` lowered the empty gripper, reported ok, and
    only the `pick_and_place` postcondition refuted it 20 s later. The same
    look that measures the in-jaw offset sees the object far BELOW the TCP
    -- `place_at` now raises "slipped out of the gripper" (`grasp.slip_drop_m`,
    6 cm) and releases the held state, so the persistence loop re-grasps
    immediately instead of after a refuted place.
14. Tests now pin the reverse skill/spec mapping AGENTS.md warned about for
    months (a `skill_*` method without a `TOOL_SPECS` entry), that every
    `_MOTION_SKILLS` name is a real skill, and that README's headline skill
    and tool counts equal the derived numbers (they were 30/37 against
    33/41).
15. **The closed loop could switch itself off silently.** `execute()` wrapped
    the Pigey postcondition check in `except Exception: pass`: a verifier
    crash (camera hiccup, truth channel down, a checker bug) left the
    skill's self-reported `ok: True` as the final word with no `verified`
    flag at all. Now the crash becomes an UNVERIFIED postcondition naming
    the cause (`verified: false`, `verification_note`), logged to the run
    dir. Test drives the real `execute()` with a raising verifier; proven
    by mutation (restoring the swallow fails both tests). Of the other 53
    `except Exception: pass` sites, the rest guard best-effort side paths
    (gripper release on abort, keyframe grabs, memory writes); left as is.
16. **`--check` mutated the venv.** Documented as "report what is missing,
    exit", `--check --sim isaac` on a MuJoCo venv installed 121 MB of torch
    (+ torchvision) and would have fetched robot assets: the extras step
    had a CHECK gate, the mujoco/torch/asset steps did not. All four now
    warn under `--check`; verified with a package-list diff before/after
    (103 packages, unchanged).
17. Two `B023` late-binding closures fixed (`usd_model._parse_joints`
    bound `body` from the loop -- every joint's attribute lookup would
    read the LAST joint's block if the lambda were ever called after the
    loop; `record_demo._rs`), and four `raise ... from e` chains in the
    probe skills so a KeyError's origin survives into the SkillError.
18. Verified again on a fresh export of the committed tree
    (`git archive HEAD` → /tmp, no venv/runs/assets): `./run.sh mujoco
    --cameras mujoco_scene_two` created the venv, installed the extras,
    fetched the 24 SO-101 meshes, proved 41 tools, pick CONFIRMED by
    physics, reset OK, and -- with the display awake this time -- the
    banner read "the MuJoCo window is open (it follows every motion)".

Not adopted, with reasons: Vesta as the brain (no weights); navigation and
SFT mixture (training); GR00T actor (VLA as executor was ruled out at the
time; since 2026-10-09 it exists as the opt-in `grasp.executor: vla` route, B49);
the async planner–actor loop with max staleness (Appendix B) -- tool calls
are synchronous by design here, noted for long-horizon work.

## Near term (before the demo)

- **Booth experience.** The attendee-facing session is scripted in
  `docs/BOOTH_RUNBOOK.md` (hard 15-min format, typed-chat interaction — no
  voice on an expo floor, first visible result <3 min, scripted
  fail→learn→succeed arc, fallback ladders); `scripts/booth_up.sh` /
  `booth_reset.sh` are the ops entry points. Landed 2026-07-20: out-of-band
  MCP e-stop (incl. stop-during-startup latch) + cancellation→freeze +
  `CASCADE_HIDE_TOOLS` (reset_stop becomes staff-only; SIGUSR1 is the staff
  reset channel), dashboard STOP wired in MCP mode, `setup_agents.py
  --detect-classes/--hide-tools/--env` + offline env by default, stale-path
  fixes in `dashboard_runner.py`/`hermes_demo.sh`; adversarially reviewed
  same day, defects pinned in `tests/test_review_regressions_v4.py`.
  Second pass (also 2026-07-20) closed the remaining six: (1) booth tuning
  is a `CASCADE_BOOTH=1` overlay (`configs/booth.yaml`, deep-merged in
  `load_demo_config` — dev keeps dev values); (2) `scripts/booth_rehearsal.py`
  dry-runs the session prompts through the real orchestrator — first run on
  local Qwen3.6-27B: 6/6 prompts clean tool calls, 4/6 tasks succeeded (the
  2 failures are the mock air-grasp, handled with retries + honest report);
  (3) dispatch tier on the dashboard (`/state.last_path` + "via:" chip);
  (4) `/keyframes` before/after filmstrip route; (5) grasp-memory panel on
  the dashboard; (6) MCP-mode dashboard chat runs the reflex grammar
  LLM-free. Remaining (on-site): re-run the rehearsal with the final
  cheat-card nouns, and validate the point-at-under-cup beat on the real
  rig with `CASCADE_BOOTH=1`.

- **Persistence-loop review leftovers (2026-07-18, adversarial review run;
  fixed same-day: e-stop break, fail-fast on never-seen objects, place
  release/ascent desync, frozen belief epoch, descent-path vetting,
  place-stage re-home).** Still open, in priority order:
  1. ~~MCP server is single-threaded: a 150 s pick_and_place blocks
     emergency_stop~~ **fixed 2026-07-20 (booth prep):** the stdin reader
     now latches the e-stop out-of-band the moment the frame arrives,
     `notifications/cancelled` on an in-flight motion tool freezes the arm,
     SIGINT latches instead of free-falling, and the dashboard STOP button
     is wired in MCP mode (tests in `tests/test_mcp_server.py`).
  2. ~~Exception between gripper close and held_object assignment leaves a
     physically held object logically unheld (reconcile only clears the
     opposite desync); consider a provisional held marker before close.~~
     **landed 2026-10-07** (marker already in `skill_grasp_object` /
     `_reconcile_held`; finished today): `_held_provisional` is set before
     the first jaw command and promoted/refuted by the next skill; jaws AT
     the open position now refute it instead of promoting a phantom hold,
     and `open_gripper` discards it. Pinned by
     `tests/test_persistence_leftovers.py` (`..._reconcile_promotes`,
     `..._on_air_is_dropped`, `..._around_the_close`,
     `test_open_gripper_discards_a_provisional_marker`,
     `test_provisional_marker_with_jaws_at_the_open_position_is_dropped`).
  3. ~~Budget can multiply across tiers: fast-path burns persist_seconds,
     then a real-LLM tier can call pick_and_place again. Cap per task.~~
     **already landed** (`begin_task_budget` / `_task_deadline`,
     `grasp.task_persist_seconds` = 1.5x `persist_seconds` by default,
     opened/closed by `AgentOrchestrator.run_task`, capping pick_and_place
     AND `_grasp_with_persistence`); pinned by
     `test_task_budget_caps_a_second_tier_call`,
     `test_orchestrator_opens_and_closes_the_task_budget` and (2026-10-07)
     `test_handover_persistence_is_capped_by_the_task_budget`. The key is
     now documented in `configs/demo.yaml`.
  4. ~~handover / sort_by_color still single-attempt (inconsistent with
     pick_and_place persistence).~~ **already landed** (both grasp through
     `_grasp_with_persistence`: re-home, re-scan, fresh plan, same
     `max_pick_attempts` / `persist_seconds`, capped by the task budget;
     sort_by_color gives each object `sort_object_persist_seconds`);
     pinned by `test_handover_retries_a_grasp_like_pick_and_place`,
     `test_sort_by_color_does_not_give_up_on_the_first_miss`. Still
     single-attempt by design: sort_by_color's PLACE (a failed place while
     holding stops the sort instead of cascading).
  5. ~~_reconcile_held mistakes a legitimately-held VERY thin object
     (<4% jaw span ~ 3.6 mm) for a slip; booth objects are chunky.~~
     **landed 2026-10-07**: the held width is MEASURED when the object is
     taken (`_held_width_m` = jaw stall after the lift; planned width when
     feedback is unavailable; stall width at promotion/adoption). An object
     that measured thinner than `air_grasp_frac` x jaw span is never
     cleared on width alone (position feedback cannot tell that hold from
     air); a chunky known width keeps the 4 % rule unchanged. The older
     `gripper.min_object_m` profile key still works. Pinned by
     `test_grasp_records_the_measured_held_width`,
     `test_known_thin_object_is_never_read_as_a_slip_on_width_alone`,
     `test_thin_object_declared_in_profile_is_not_read_as_a_slip`.
  6. ~~Fail fast when every grasp candidate exceeds jaw width (currently
     retries perception on an object-property error).~~ **landed
     2026-10-07** (the string heuristic in `_grasp_retry_verdict` already
     stopped after one attempt; finished today): `select_grasp` raises an
     explicit refusal — "every candidate exceeds the jaw span (narrowest
     Xmm > gripper max Ymm; use push_object)" with `all_too_wide`,
     `narrowest_width_m`, `jaw_max_width_m` on the exception — and lists
     non-width reasons first, so a 4-reason truncation of a MIXED list
     (five too-wide ahead of one vetoed candidate) can no longer read as
     all-too-wide and end persistence after one attempt. Pinned by
     `test_pick_and_place_gives_up_early_when_every_grasp_is_too_wide`,
     `test_ik_failure_alongside_a_width_reason_is_still_retried`,
     `test_selector_refuses_explicitly_when_every_candidate_exceeds_the_jaw_span`,
     `test_a_truncated_mixed_reason_list_is_not_an_over_width_refusal`.
  7. ~~Test-coverage gaps flagged: place-stage loop, deadline expiry,
     epoch fallback, z-clamp, exemption z_min through _in_cylinder,
     McpClient timeout is dead code.~~ **landed 2026-10-07**: place-stage
     loop (`test_place_stage_retries_after_a_failed_place`,
     `test_place_stage_is_bounded_while_still_holding`,
     `test_mid_carry_slip_restarts_the_grasp_stage_within_the_budget`);
     deadline expiry, epoch fallback, z-clamp and `_in_cylinder` z_min were
     already pinned (`test_persistence_deadline_expiry_stops_the_loop_early`,
     `test_localize_epoch_fallback_uses_motion_start_when_no_rescan`,
     `test_place_at_caps_release_height_to_the_topdown_ceiling`,
     `test_exemption_cylinder_z_min_is_honoured_by_in_cylinder`);
     `McpClient.recv(timeout=)` is live (queue-pumped stdout) and is now
     exercised without a server
     (`test_mcp_client_recv_times_out_instead_of_hanging`).

- **Newton upstream issue (2026-07-19).** Manipulation contacts are broken
  at the PARSER level on the 6.0 develop build: identical failure under
  mjcwarp default, `use_mujoco_contacts=true` and XPBD (constant +3.7 cm
  float even box-vs-box, fingers pass through objects, boot NaN, "Triangle
  pair buffer overflowed"). Repro = `scripts/physics_probe.py --engine
  newton`. File against isaac-sim/IsaacSim with the probe reports in
  /tmp/probe_newton_*.json. Demo manipulation stays on PhysX (full battery
  green) until fixed.
  - **RE-TESTED 2026-08-31 on Newton 1.5.1 — the contact bug does NOT
    reproduce.** Correcting the earlier note in this file: the package is
    `newton` (PyPI, v1.5.1), not `newton-physics` (a stale 1.0.0 squatter),
    and `pip install "newton[examples]"` works on macOS arm64. `SolverMuJoCo`
    also runs WITHOUT CUDA — Warp's CPU device is enough — so this was
    testable on a laptop all along. Measured here (`mujoco` 3.11.0, Warp
    1.17.0, device `cpu:arm`):

    | symptom (2026-07-19) | Newton 1.5.1 result |
    |---|---|
    | constant +3.7 cm float, box-vs-box | **−0.03 mm** — rests exactly at contact |
    | fingers pass through objects | **0.3–0.8 mm** penetration at 2–10 N, 12 contacts registered |
    | boot NaN | none; all states finite |

    The finger test drives two prismatic-actuated fingers onto a 6 cm box
    commanding 0.09 m of travel where contact is at 0.06 m, so a pass-through
    would show as the full 0.09. It stops at 0.0603 m. Penetration grows to
    70 mm only when the drive is pushed to 50 N against a 50 g box, i.e. the
    solver is compliant under absurd force, which is not the reported bug.

    Two API traps that produced FALSE PASSES while writing that probe, worth
    knowing before anyone re-runs it: (1) `add_body()` already creates a FREE
    joint, so adding a prismatic joint on top makes a parallel LOOP joint that
    MuJoCo silently drops ("no supported equality constraint mapping") — the
    actuator then does nothing and the fingers never move, which reads as
    "stopped on the box". Articulated links need `add_link()` +
    `add_articulation(joints)`. (2) A joint with no `parent_xform` is anchored
    at the world origin, so both fingers start inside the box.

    STILL OPEN: this is a synthetic box-and-fingers scene, NOT the reBot Isaac
    asset with its custom fixed-joint stack, and not the "Triangle pair buffer
    overflowed" mesh path. Before flipping the demo default off PhysX, run
    `scripts/physics_probe.py --engine newton` against the real asset on the
    DGX. What is settled is that the blanket claim "manipulation contacts are
    broken at the PARSER level" no longer holds for current Newton.
    **2026-10-07:** run against the real asset on the x86 6.2 build — 8/8 on
    Newton (settle/drop/grasp/push, both boxes), same as PhysX; the DGX run is
    still pending.
- **Wrist cam follow-ups.** ~~Validate the eye-in-hand extrinsics during a
  real grasp (reproject wrist depth of the target object against the
  physics-truth pose mid-descent)~~ **landed 2026-10-07 in sim** (near-term #4); ~~consider serving the wrist stream a
  narration highlight ("what the gripper sees") on the dashboard~~ **landed
  2026-10-09 (B47, opt-in `stream.wrist_narration`)** -- during a motion skill
  the wrist tile is highlighted and captioned with one line from verifiable
  state only (skill + target, runtime held-state, three-state postcondition;
  unverified grasp reads unverified, no wrist camera = no panel), measured on
  the mock stack through the real runtime, HTTP dashboard and its script
  (`tests/test_wrist_narration.py`; flag off = dashboard bytes sha256-pinned
  to main); not yet shown on Isaac or the real rig, and it narrates runtime
  state, not the image. Still open: on the
  real rig map `isaac_wrist.yaml` to the physical D435i + hand-eye calib.
- **Sim perception flakiness** (separate campaign): YOLOE misses the YCB
  banana on some boots and label-flickers the soup can (bottle/toy);
  belief 3D positions themselves verified ±3 mm against physics truth.

- **Persistent spatial memory — DONE 2026-08-31.** `BeliefStore.save/load`
  (JSON, wall-clock timestamps, everything reloaded as "remembered"), wired
  into `build_runtime`/`shutdown_runtime` and gated by
  `memory.persist_beliefs` / `CASCADE_BELIEFS`. Verified across two real
  processes: run 2 prints `recalled 1 object(s)` and the observation count
  accumulates instead of resetting. The monotonic→wall-clock conversion is
  the load-bearing part (see AGENTS.md); `LOADED_MIN_AGE_S` guarantees a
  restored belief never reads as `visible`, and `_localize`'s existing
  3 s `belief_fallback_age_s` gate means it can never aim the jaws.
  STILL OPEN from this item: action↔object consolidation on top of
  ExperienceMemory (sub-goal-level credit now exists via
  `FastPlanner.note_subgoal_outcome`, but it keys on TEXT, not on objects).
  **Landed 2026-10-07 as an opt-in** (`memory.action_objects`, follow-up #7
  above): `memory/consolidation.py` keys on (skill, object label).
- **Straight-up spawn on the local tuned Isaac asset.** Blocked: drive
  travel from q=0 sweeps the props; joint-state authoring and tensor
  teleports NaN the solver on this asset (custom fixed-joint stack).
  Upstream asset gets it properly via Seeed-Projects/reBot-Isaacsim#9.
  Investigate the -plus asset's root joint / articulation root config.

- **GraspGen-X backend (integrated 2026-07-18, first-light verified;
  probed-at-startup + protocol stub since 2026-09-09).**
  `grasp.backend: graspgenx` sends the fix's base-frame object cloud to the
  GraspGen-X ZMQ server (`scripts/serve_graspgenx.sh`, own venv
  `~/Projects/demo/.graspgenx`, checkpoints in `GraspGenX/ext/`) and gets
  ranked 6-DoF grasps back (~1.2 s for 100 samples on the GB10); OBB stays
  as automatic fallback and additional IK candidates. The launcher starts
  `serve_graspgenx_stub.py` on hosts without CUDA so the client path is
  exercised everywhere -- the banner says `graspgenx-stub (analytic protocol
  double)`, and that is NOT the learned model. ~~(0) cold first inference~~
  **landed 2026-10-09 (B15a)**: a fresh server answered its first
  `infer_object` in 15.53 s and every later one in 0.09 s (x86 RTX PRO 6000,
  torch 2.7.0+cu128), longer than the client's 8 s `timeout_ms`, so the
  first pick of a session fell back to the analytic planner on every path
  that does not run `check_graspgenx.py` first (the booth runbook's
  `serve_graspgenx.sh` + `booth_up.sh`, a directly started MCP server).
  `scripts/graspgenx_server.py` now runs one synthetic inference before it
  binds its port (`--no-warmup` restores the old start-up) and reports
  `warmed_up` / `warmup_s` / `warmup_error` in `health`; a failed warm-up is
  logged and the server binds cold as before (no CUDA / no model still
  refuses first). Live ([evidence](evidence/b15a-graspgenx-warmup-20261009/REPORT.md)):
  warm-up 15.42 s before the port opened, then client inferences 0.16 /
  0.14 / 0.13 s; `tests/test_graspgenx_warmup.py`. Still open: the
  fail-open branch is CPU-tested only, GB10 not measured. TODO: (1) calibrate
  `tip_offset_m` in Isaac Sim (gripper-base -> reBot jaw center), (2) refine
  the URDF-derived reBot sweep-volume params in `configs/demo.yaml` with an
  Isaac Sim measurement (franka_panda remains only the no-sweep fallback),
  (3) use `infer_scene_pc` for collision-aware grasps in clutter.

- **Onsite bring-up checklist**
  1. `sudo ip link set can0 up type can bitrate 1000000`; kill any
     motorbridge-gateway/Studio.
  2. Re-verify RS gripper travel + stall torque; update
     `configs/arms/rebot_rs.yaml` (open/closed rad, kp, max_width_m). Then run
     the B38 squeeze-cap protocol ([REBOT_GRIP_SQUEEZE_CAP.md](REBOT_GRIP_SQUEEZE_CAP.md))
     before setting `gripper.max_contact_squeeze_rad`.
  3. Hand-eye calibration: run the baseline repo's `collect_handeye_eih.py`
     (eye-in-hand) or measure the static mount, point the camera profile's
     `extrinsics` at it.
  4. Fit the table plane once (`DepthProvider.fit_table_plane`) and set
     `table_z` / workspace AABB for the physical setup.
  5. `pytest -m hardware` (read-only), then first motions with
     `--arm rebot_rs` at low `max_joint_vel`.
- **Local LLM**: run `scripts/serve_qwen_llamacpp.sh` (Qwen3.6-27B GGUF, MTP
  speculative decoding) and rehearse with `--llm local_qwen` so the demo has
  a no-internet fallback. vLLM variant in `serve_qwen_vllm.sh`. The profile
  (`configs/llm/local_qwen.yaml`) pins `model: Qwen/Qwen3.6-27B` to match
  both scripts (reconciled 2026-07-20); if you change the script's
  `MODEL`/`HF_REPO`, update the profile — llama.cpp ignores the requested
  name but vLLM rejects a mismatch, and `supports_vision` gates the advisor
  and image context.
- ~~**Visual embedder for memory**: plug a CLIP/SigLIP image encoder into
  `EpisodicMemory(embed_dim=...)` + crops per detection, enabling
  "the thing that looked like X" recall through the TurboQuant index.~~
  **landed 2026-10-07 (opt-in)** as `memory.embedder` -- see open item #7 in
  the priority list above; ~~crops are per LOCALIZED detection~~ (**landed
  2026-10-09**, B43: `memory.visual_recall_detections` adds a crop per
  committed watcher detection, `memory.persist_episodic` keeps the index
  across restarts, both opt-in), and real weights remain to be evaluated on
  the GPU host.

## Mid term

- ~~**VLA policy backend**: LingBot-VLA-v2 exposes a websocket policy server
  (msgpack-numpy, `infer(obs) -> action chunks`). Add a `VLAExecutor` behind
  the skill API so `grasp_object` can be served either by the deterministic
  OBB pipeline or by a language-conditioned policy; the agent layer stays
  unchanged.~~ **landed 2026-10-09 (opt-in, B49)** as `grasp.executor: vla`
  (`grasping/vla_client.py` + `vla_executor.py`, `vla` extra, protocol double
  `scripts/serve_vla_stub.py`, [docs/VLA_EXECUTOR.md](VLA_EXECUTOR.md)): the
  openpi/LingBot websocket msgpack-numpy wire; every chunk admitted whole by
  the SafetyHarness and streamed waypoint by waypoint through `SafeArm`;
  stop latch, chunk and episode deadlines; the unchanged verifier judges.
  Measured on the mock stack against the protocol stub only
  (`tests/test_vla_executor.py`). Still open: any run with real weights (a
  policy post-trained on this arm does not exist yet; the public LingBot /
  openpi checkpoints are other embodiments), streamed chunk execution at the
  policy's native rate, multi-camera observations, and the Spark caveat:
  flash-attn must build for aarch64+Blackwell or the two hardcoded
  attention impls patched to SDPA.
- **GraspNet-class 6-DoF grasps** — ✅ superseded by the GraspGen-X backend
  (integrated 2026-07-18, see near term). The baseline's graspnet path
  (vendored sdk + checkpoint-rs.tar + THC-era CUDA patches) is no longer
  worth pursuing; remaining learned-grasp work (tip-offset calibration,
  reBot sweep params, collision-aware `infer_scene_pc`) is tracked in the
  near-term GraspGen-X item.
- **Skill-library growth loop** (ASPIRE) — ✅ **landed 2026-07-31** (and
  wired end to end: `retrieve()` runs in `orchestrator.run_task`; docs that
  called it store-only were corrected 2026-09-10). After each
  run, `agent/aspire.py` diagnoses the trace and distils scoped,
  measured-confirmed retry associations into `skills_library/*.md`, deduped
  by (skill, signature) within a harvest. Matching goals, resolved arms and
  pre-call possession are required; explicit reset/task-end rows stop matching.
  `retrieve()` loads guard-matched entries into the agent context at task
  start. Batch entry point: `scripts/learn_from_runs.py` (runs between
  sessions, never mid-demo). This is not causal or cross-task validation;
  see [admission rules and limits](DREAM_RSI_ADAPTATION.md).

## Long term: sim2real with NuRec / Isaac

- **Isaac Sim bridge (scaffolded 2026-07-18; live on PhysX since
  2026-07-19).** `scripts/isaac_bridge.py` (runs inside Isaac Sim's Python)
  serves RGB-D frames + articulation control over newline-JSON TCP;
  `--cameras isaac --arm isaac` runs the identical demo against the sim.
  Protocol + client backends are covered by fake-server tests, and the live
  path has run scripted picks (`dashboard_runner.py`, `night_runner.sh`)
  with the `physics_probe.py` battery green on PhysX — joint order/signs,
  gripper fraction mapping and camera extrinsics are wired in
  `configs/{arms,cameras}/isaac*.yaml`. Still open: wrist-cam extrinsics
  validation during a real grasp (near term), Newton engine (blocked
  upstream, near term), and the straight-up-spawn asset issue.
  The Downloads/isaac-companion-v1-franka pack's `isaac-sim-remote` skill
  (TCP Python-exec extension) is a good live-debugging companion for this.
- `reBot-Isaacsim` + `sim2real-rebot-devarm` already provide USD assets, a
  real→sim UDP mirror, and an HTTP control daemon for this arm.
- **NuRec (neural reconstruction)**: reconstruct the actual demo tabletop
  into a photoreal digital twin; rehearse perception + grasping against the
  twin (domain gap ≈ 0 for the camera), then replay on the real rig. The
  ASPIRE sim2real recipe applies directly: skills discovered in sim transfer
  as *in-context guidance* for the real-robot agent, not as weights.
- **Online adaptation** (Agentic-VLA proper): a VLA executor now exists
  (opt-in `grasp.executor: vla`, B49, 2026-10-09; it only RUNS a policy),
  so their GRPO + reward-synthesis loop is the open path to improving one
  from demo logs; the trace format already captures per-primitive evidence
  needed for progress rewards.
