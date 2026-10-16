# Roadmap: capability status

Next work per capability is in the table below; a row closes only when its
implemented path passes the stated end-to-end checks (an interface, mock episode or
dependency import alone does not close it). Measured so far (5 October index): one
five-object OpenClaw/Qwen Isaac campaign (source `d2b78586`), one native
[mounted turn-and-rest](FACTORY_FASTENING_RUNTIME.md#native-mounted-turn-and-rest-2026-10-04)
and twelve [live cuVSLAM tracks](SPATIAL_PROVIDERS.md#live-native-rgb-d-stream).
Shared interfaces: [architecture](ROBOT_MODULARITY.md). Each numbered item is
worked one worktree, branch and PR at a time; code comments cite the numbers below
(`ROADMAP #7`, `follow-up #8`, `near-term #2..#7`).

## Remaining capabilities

| Capability | Remaining implementation and completion criteria |
| --- | --- |
| Manipulation | Extend the three retained two-object campaigns and the completed five-object campaign to more scenes and repeated coverage; demonstrate measured collision/carry protection through the agent route while retaining release, support, rest, reset and owned closure; validate each new robot and hardware setup. **Reach envelope (B45, opt-in):** top-down grasps reach only r ≤ 0.45 m (0.38 m at 10 cm grasp height), leaning 45° away from the base 0.69 m, horizontal 0.75 m; the `*_reach` profiles add tilted analytic candidates and the measured box x 0.10..0.55, y ±0.50 ([REACH_ENVELOPE.md](REACH_ENVELOPE.md)). Still open: live Isaac picks in the new region (`isaac_reach` vs `isaac`), the real rig (table, mount, camera field of view), ~~contact/gripper-housing collision of angled and side grasps~~ **landed 2026-10-10** (B73, `grasp.angled_clearance_vet`, [measured offline](REACH_ENVELOPE.md#gripper-clearance-of-the-tilted-candidates-b73-measured-2026-10-10): of 5586 reachable tilted grasps on the B45 grid 5539 survive: the vet refuses 47 of 47 side (90°) grasps at 5 cm (wrist motor `link5` -3.1 mm from the table) and no 30°/45° grasp (their lowest point is the fingertips); still open there: contact physics and live picks, the home → pregrasp transit, learned candidates, other obstacles), objects that only fit a jaw pointing at the robot. |
| Locomotion | Demonstrate visible longer routes, turns, balance-preserving stop, disconnect and reset; validate more actuator/physics combinations and humanoid policies. Rerun the standing, walking, braking and reset campaigns against the admitted [Isaac Lab USD folder asset](MICRODUCK.md#isaac-lab-usd-folder-as-the-admitted-asset-5-october-2026) (`rough_walk_e` reverse confirmed, forward refuted; `velocity_flat` tracks while commanded but is refuted after completion). The opt-in [standing handoff at zero twist](MICRODUCK.md#standing-handoff-at-zero-twist-measured-5-october-2026) switches cleanly but leaves 0.09–0.23 m/s braking at completion; next candidates are a checkpoint trained with zero and decelerating commands or a distance-aware walk speed. Re-admit BAM when Newton ships its native BAM drive (newton-physics/newton#4504). The `walk_distance` admission-drift pair (B74, opt-in, 95 CPU tests) needs a value from a live A/B. |
| Fastening | Establish repeatability of the measured mounted turn-and-rest task; the opt-in `gc_policy: freeze-startup-heap` has one native same-revision [on/off pair](FACTORY_FASTENING_RUNTIME.md#opt-in-startup-heap-freeze-software-candidate-5-october-2026), so an episode where a generation-2 pause would overlap the command window is still needed, as is the cause of the 0.2–0.8 s observation-drain gaps; add tool acquisition, engagement, calibrated torque/preload and withdrawal with independent contact and thread measurements. |
| Perception and spatial understanding | Extend the measured static planar RGB-D result to broader metric geometry and moving cameras; validate IMU/proprioception fusion and ~~time alignment while preserving missing, stale and uncertain observations~~ **time alignment landed in software 2026-10-09 (B50, opt-in, 42 CPU tests)** and ~~producer-rate sampling~~ **landed in software 2026-10-10 (B72, opt-in, 60 CPU tests: 16/24 read-time pairings stale vs 24/24 aligned from the ring)** ([contracts](ROBOT_MODULARITY.md#capture-time-alignment-b50-opt-in)). Still open: the live skew distribution on the Isaac MicroDuck bridge, the fusion estimator that consumes the pairings, producer history through the off-GIL `--state-reader process` reader, and hardware clock synchronization. |
| Mapping and navigation | [Twelve live cuVSLAM tracks](SPATIAL_PROVIDERS.md#live-native-rgb-d-stream) join a 320-solve native producer to registered camera/base poses and verified stop invalidation; physical pose uncertainty remains uncalibrated. Admit calibrated pose and complete collision mapping/volume, then validate native route execution, replanning and arrival/rest, including tracking loss and changed map epochs. |
| Speech and interaction | [Two bounded voice cases](CONVERSATION.md#two-complete-native-speech-cases--voice13) join synthetic microphone input, model decisions, native simulated hand motion/rest and browser audio, including stop, reset and reconnect. Extend to repeated and varied tasks, camera context and microphone/speaker hardware. ~~An independently deployable conversation service~~ **landed 2026-10-09 (B51, opt-in)** with ~~cross-host transport~~ **TLS landed 2026-10-10 (B71)** ([deployment](CONVERSATION.md#deploy-the-conversation-service-separately), [across hosts](CONVERSATION.md#across-hosts-opt-in-tls-b71)); the extra hop costs p50 +0.5 to +0.7 ms per call in clear, about +1.6 ms with TLS (one host's loopback, N = 500). Still open: a run between two real hosts, the real local speech provider with a native robot, latency under load. |
| Whole-body humanoids | Bind dynamic frames and a whole-body controller to explicit embodiments; validate balance, self/environment collision and coordinated base/arm/head actions. ~~Separate domain controllers must not compete for the same command endpoint.~~ **landed 2026-10-08** (B30, mock-only): an opt-in `whole_body` contract composes a locomotion base and an arm `mounted_on` it with disjoint command endpoints, a capture-time `world ← base ← arm_base` frame chain, `exclusive` coordination by default (`concurrent` opt-in) and per-domain `reset_stop(domain=...)` ([contract](ROBOT_MODULARITY.md#multi-domain-embodiments-mounted-arms)); measured by 43 software tests incl. golden digests of every earlier robot profile. Still open: physical admission of any mounted composition (measured mount calibration, independent base-pose source, moving-frame arm limits), the whole-body controller, balance, self/environment collision and H2 arm physics. Embodiment chosen on 7 October 2026: [Unitree H2, PhysX first](HUMANOID_H2.md) with NVIDIA's public H2 USD and `Velocity-H2-History-v0` policy; the reference loop walks 0.63 m in 2 s on the internal 6.2 build; `scripts/isaac_h2_bridge.py` owns the H2 on PhysX, with the independent verifier confirming 3 of 5 `walk_velocity` commands in the first episodes; Newton route tracked through newton-assets PR #53. Candidate revision 2 the same day (settle budget 8 s wall, rest thresholds unchanged): 6/7 confirmed; `turn` 0.8 rad still refuted (yaw oscillation 0.69 → 0.32 rad/s over 3.75 s sim) — ~~next revision is control-side (ramp the turn rate before the goal), not a bigger budget~~ **landed 2026-10-07 in software (B29, candidate revision 3)**; [owner episode](HUMANOID_H2.md#live-result-8-october-2026): 0.8 rad 8/8 with the ramp vs 1/6 without, but 1.0 rad 1/5 vs 5/6; ~~next control revision open~~ **landed 2026-10-08 in software (B29b, candidate revision 4)**: a time-budgeted ramp, 75/76 command sequences identical across every base profile. Still open: the owner A/B (revision 4 vs 3 vs 2), the 1.0 rad translation margin (0.166–0.227 m vs 0.20 m in both arms), physical admission. |
| Hands and touch | Extend the [fixed LEAP free-finger runtime](ROBOT_MODULARITY.md#fixed-leap-hand-leap_hand_right_mujoco) beyond its four repetitions of one native motion/rest task: broader trajectories and disturbances, other hand drivers, calibrated tactile observations, contact/slip estimation and grasp/force skills. |
| Multiple robots | Resolve worst-case twelve-robot feedback latency without relaxing its limits; run a native twelve-robot command episode under the original limits with the opt-in startup-heap freeze ([MICRODUCK.md](MICRODUCK.md#shared-scene-implementation-boundary)). Host work per step is measured by phase and was cut in separate steps from a 76.2 ms to a 42.43 ms step median ([phases](MICRODUCK.md#shared-owner-per-step-cost-one-host-snapshot-for-the-bam-checks-7-october-2026)); the GPU physics step alone costs 7.6 ms per 5 ms step for twelve robots, so real-time control also needs cheaper physics. The opt-in off-GIL `state()` reader still needs its GPU A/B. Twelve robots walk 68 s without a fall under scripted twists, but twelve closed-loop clients still trip the unchanged limits ([route fleet evidence](evidence/microduck-route-fleet-20261005/REPORT.md)). |
| Agent intelligence and evaluation | Validate perception-guided planning, bounded skill graphs, memory-assisted recovery and multimodal feedback across tasks; complete Arena/VAB task episodes and held-out failures. The opt-in VLA executor (`grasp.executor: vla`, B49, [VLA_EXECUTOR.md](VLA_EXECUTOR.md)) is measured against its protocol stub only; with the registration env (B63) a live Isaac run drove it with a scripted stub policy (17 policy requests over 8 grasp episodes, every empty close refused by the jaw check). Still open: any run with real weights. |
| Hardware and deployment | Connect selected robots and sensors through explicit adapters; verify calibration, transport loss, controller ownership, stop/restart and task outcomes before each physical deployment. **Sandboxed host (B35):** an OpenClaw agent in an OpenShell sandbox ran a physics-confirmed pick and verified reset on 2 of 3 fresh Isaac stages ([NemoClaw route](OPENSHELL.md#nemoclaw-route-opt-in)); still open: the kitchen proof cases on this route, the Spark, a listener firewall rule. **Pick reliability (B36):** ~~a lifted cube counted as a failed attempt~~ **closed 2026-10-10** (0 of 39 live recovery-arm runs vs 7 of 37 on main); the opt-in hold after contact took live PhysX GraspGen-X picks from 9/16 to 15/16 confirmed, Newton keeps the plain close (14/14 without the hold, 1/4 with it) ([evidence](evidence/b36-pick-reliability-20261010/README.md)); still open: stalls chattering wider than the 0.002 band (1 of 16), OBB grasps (0/3 either way), kitchen profiles. **Grip squeeze cap (B38):** run the hardware protocol in [REBOT_GRIP_SQUEEZE_CAP.md](REBOT_GRIP_SQUEEZE_CAP.md), then set a value in `rebot_rs.yaml` (ships `null`). **Read-only MCP lane (B46):** measured live on Isaac over 4 lane-on / 4 lane-off picks: reads in 0.001–0.003 s (`camera_snapshot` 0.024–0.038 s) during the pick, picks confirmed 2/4 vs 1/4, `emergency_stop` in 0.032 s ([evidence](evidence/w6-live-20261010/README.md)); still open: a look tool that also detects. |

## Delivery status

1. Resolve the local dual-RTX Pro native timeout and the separate nvblox route/carry failures.
2. Complete source-bound proof, the five-object campaign and a normal same-version restart; preserve failed runs and outcome memory.
3. Before optimizing the profiled placement timeouts, measure the execution job's subcomponents and observer overhead ([bridge timing](ARCHITECTURE.md#optional-bridge-timing), [lazy frame encoding](ARCHITECTURE.md#lazy-frame-encoding)).

Newton, new hardware, Cosmos evaluation, mobility and ovrtx keep separate validation
scopes. Cosmos3-Edge replaces the Qwen event backend only after it passes the same
acceptance gates. The offline Jev/Kev pilot showed no decision-quality advantage over
its rule baseline and is not part of presenter startup. The
[observed-scene finger envelope](ARCHITECTURE.md#motion-safety-path) and the
[OVRTX renderer](OVRTX_RENDERER.md) are merged; OVRTX still needs an automatic
physics producer, safety masks and a kitchen comparison.

## Priority list (set 2026-09-10)

0. **Spark delivery acceptance:** run the exact published build on a clean Spark with the session-bound proof/reset and every supported scene object.
1. **Real rig first motions:** the [onsite checklist](#onsite-bring-up-checklist), `pytest -m hardware`, then `--arm rebot_rs` at low velocity.
2. ~~**Persistence-loop leftovers**~~ **landed 2026-10-07**: see [leftovers](#persistence-loop-leftovers) (`tests/test_persistence_leftovers.py`).
3. **Learned grasps for real:** run `serve_graspgenx.sh` (CUDA), calibrate `tip_offset_m` and the reBot sweep volume in Isaac, try `infer_scene_pc` for clutter; the stub only proves the wire.
4. ~~**Wrist camera** extrinsics validated mid-descent~~ **landed 2026-10-07 (sim)**: PhysX 0.69 px median reprojection, Newton 2.73 px ([details](NEWTON_ENGINE.md#real-rebot-asset-on-the-internal-62-build-start-up-probe-battery-wrist-camera-7-october-2026)); the real D435i hand-eye stays with the rig.
5. **Newton target validation:** `physics_probe.py` passes 8/8 on Newton and PhysX on the x86 6.2 build; still open: the Spark (GB10) run and the 6.2 Newton camera cadence (34–40 % frameless refreshes during motion).
6. ~~**Judge as a metric**~~ **landed 2026-10-09 (B44, opt-in, advisory)**: `launch.sh --judge fake|vlm|grm`. Live on recorded picks (B44-live), the local Qwen3.8-27B judged 72 recorded Isaac picks: at 1536 tokens `tp=40 tn=1 fp=15 fn=0`, 12 unscored; re-judged at 8192 tokens `tp=40 tn=7 fp=19 fn=0` ([evidence](evidence/b44-live-qwen-judge-20261009/README.md)). A live launcher-style proof turn gave `tp=1 tn=0 fp=0 fn=0`. ~~A score that fits the budget~~ **landed 2026-10-10 (B66, CPU only)**. Still open: a full `./run.sh` launch with the OpenClaw brain, an out-of-sample success threshold (B65), GRM (#14).
7. ~~**Visual embedder** and action↔object consolidation~~ **landed 2026-10-07 (opt-in)**, watcher crops and persistence **landed 2026-10-09 (B43)**, the clause rule (B37, B69) too. Still open: real SigLIP/CLIP weights on a GPU host, a live Isaac run of either switch.
8. **Multi-arm on physics:** the inter-arm gate is measured against `mj_geomDistance` and `so101_left_mujoco` / `so101_right_mujoco` run in one MuJoCo world (landed 2026-10-07). Still open: a finer sphere model, wrist cameras and placement observers on a rig.
9. **Mobility + navigation:** partly landed with a velocity interface ([current state](ARCHITECTURE.md#ros2-humanoids-and-what-is-not-here-yet)). Still open: `go_to_pixel` / `go_to_object` / `where_am_i`, the ESDF costmap, Nav2 / `ros2_base`, `mujoco_base`, `unitree_base`, a G1, the two-room demo and any hardware run.

## Numbered follow-ups

1. Booth rehearsal against Cosmos3-Edge vs Qwen (tool-call reliability, latency), now also vLLM vs SGLang.
2. Feed `max_frames > 1` (a short clip) to the Cosmos3 reasoner and measure failure diagnosis.
3. ~~Phantom beliefs~~ **fixed 2026-08-27**: `annotated_view` marks a belief UNCONFIRMED unless visible or re-observed (`VisualInterface.min_observations`, default 2).
4. ~~Derived envelope features~~ **landed 2026-10-07**: `DERIVED_FEATURES` in `memory/envelope.py` (TCP z at grasp, object height/width, lateral offset; unmeasured → `missing`).
5. SGLang Omni as a second Cosmos3-Edge engine (`scripts/serve_cosmos_sglang.sh`, :8083): landed, **unverified** on hardware.
6. ~~**Pre-motion plausibility check (Human-CLAW).**~~ **landed 2026-10-07**: `agent/milestones.py::PlausibilityChecker`, advisory only, capped by `agent.premotion_max_checks` (3) per task.
7. ~~**HUG as a second grasp backend.**~~ **landed 2026-10-08 (CPU / stub only)**, [HUG.md](HUG.md). Still open: real weights + MANO on CUDA, the live A/B `isaac_kitchen_hug` vs `isaac_kitchen_gpu`, latency.
8. ~~**A `programs` tier (Waddle).**~~ **landed 2026-10-08**: [PROGRAMS_TIER.md](PROGRAMS_TIER.md), authored or distilled programs under one admission rule (opt-in `agent.programs` / `CASCADE_PROGRAMS=1`); ~~not exposed to MCP chat hosts~~ **landed 2026-10-09** (B42: `list_programs` + `run_program`); ~~retrieval is keyword overlap~~ **landed 2026-10-09** (B42: text embedding with `memory.embedder`). Still open: no real local brain (Qwen) has authored or reused a program, no Isaac run of `run_program`, no loops/branches.
9. ~~**Task-Specific Memory recipes (Harness-VLA v4).**~~ **landed 2026-10-07**: `memory/recipes.py`, re-grounded before any motion.
10. ~~**ASPIRE cross-task promotion gate.**~~ **landed 2026-10-07**: notes promoted only by ≥ 2 distinct tasks.
11. ~~**Pigey snapshot/restore + occlusion search.**~~ **landed 2026-10-07**: `snapshot_scene`, `restore_scene`, `search_for_object`. Not claimed: physical acceptance on a real rig.
12. ~~**Capability matrix → tool surface (Waddle).**~~ **landed 2026-10-07**: `apps/capabilities.py`.
13. ~~**`recall_step(n)` + a `stuck` outcome (RPent).**~~ **landed 2026-10-07**: the step recall tool and `outcome: ok | failed | stuck`.
14. **GRM on the GPU box:** serve `Robo-Dopamine-GRM-2.0-8B-Preview` with vLLM and rerun `judge_run.py --judge grm`.
15. ~~**Wrist camera for the judge.**~~ **landed 2026-10-07**: `configs/cameras/mujoco_wrist.yaml` fills the GRM wrist slots. No agreement change claimed until #14 runs.
16. **nvblox on aarch64:** no Jetson wheel yet; switch `auto` to prefer it once it exists (blocked upstream).

## Persistence-loop leftovers

1. ~~MCP server single-threaded behind a long pick~~ **fixed 2026-07-20**: out-of-band e-stop, cancel freezes the arm.
2. ~~Held marker across an exception at close~~ **landed 2026-10-07**: `_held_provisional`.
3. ~~Budget multiplying across tiers~~ **landed**: `grasp.task_persist_seconds` (1.5 × `persist_seconds`) caps every tier.
4. ~~Single-attempt handover / sort_by_color~~ **landed**: both use `_grasp_with_persistence`; sort's PLACE stays single-attempt by design.
5. ~~Thin objects read as slips~~ **landed 2026-10-07**: the held width is measured at the grasp.
6. ~~Retrying an over-width object~~ **landed 2026-10-07**: `select_grasp` refuses explicitly ("use push_object").
7. ~~Coverage gaps~~ **landed 2026-10-07**: place-stage loop, deadlines, epoch fallback and `McpClient.recv(timeout=)` are pinned.

## Other open items

- **Wrist cam follow-ups.** ~~Validate eye-in-hand extrinsics mid-descent~~ **landed 2026-10-07 in sim**; ~~a narration highlight on the dashboard~~ **landed 2026-10-09 (B47, opt-in)**. Still open: map `isaac_wrist.yaml` to the physical D435i with a hand-eye calibration.
- **Booth.** Re-run `scripts/booth_rehearsal.py` with the final cheat-card nouns and validate the point-at-under-cup beat on the real rig with `CASCADE_BOOTH=1`.
- **Newton on the real asset.** The 2026-07-19 contact bug does not reproduce on Newton 1.5.1, and the real reBot asset passes 8/8 on x86; the DGX run is pending. Repro: `scripts/physics_probe.py --engine newton`.
- **Straight-up spawn on the local tuned Isaac asset:** blocked (joint-state authoring NaNs the solver); upstream tracks it in Seeed-Projects/reBot-Isaacsim#9.
- **GraspGen-X cold start (B15a):** the warm-up's fail-open branch is CPU-tested only and GB10 is not measured.

## Onsite bring-up checklist

1. `sudo ip link set can0 up type can bitrate 1000000`; kill any motorbridge-gateway / Studio.
2. Re-verify RS gripper travel and stall torque in `configs/arms/rebot_rs.yaml`, then run the B38 squeeze-cap protocol ([REBOT_GRIP_SQUEEZE_CAP.md](REBOT_GRIP_SQUEEZE_CAP.md)) before setting `gripper.max_contact_squeeze_rad`.
3. Hand-eye calibration ([HANDEYE_CALIBRATION.md](HANDEYE_CALIBRATION.md)); point the camera profile's `extrinsics` at the record.
4. Fit the table plane once (`DepthProvider.fit_table_plane`) and set `table_z` / the workspace box for the physical setup.
5. `pytest -m hardware` (read-only), then first motions with `--arm rebot_rs` at low `max_joint_vel`.

## Mid term

- ~~**VLA policy backend**: a `VLAExecutor` behind the skill API so `grasp_object` can be served by a language-conditioned policy~~ **landed 2026-10-09 (opt-in, B49)** as `grasp.executor: vla` ([VLA_EXECUTOR.md](VLA_EXECUTOR.md)). Still open: real weights (no policy post-trained on this arm exists), streamed chunk execution at the policy's native rate, multi-camera observations, and flash-attn on aarch64 + Blackwell (or SDPA patches) for the Spark.
- **Skill-library growth loop** (ASPIRE): landed; batch entry point `scripts/learn_from_runs.py`, between sessions only ([admission rules](DREAM_RSI_ADAPTATION.md)).

## Long term

- **NuRec:** reconstruct the demo tabletop into a photoreal twin, rehearse perception and grasping against it, then replay on the real rig; skills found in sim transfer as in-context guidance, not weights.
- **Online adaptation** (Agentic-VLA): with a VLA executor in place, a GRPO + reward-synthesis loop over demo logs is the open path to improving a policy; the trace already captures per-primitive evidence.
- **sim2real assets:** `reBot-Isaacsim` and `sim2real-rebot-devarm` provide USD assets, a real→sim UDP mirror and an HTTP control daemon for this arm.
