# Architecture

Synced to the code on 2026-09-10 (737 tests, 33 skills, 41 MCP tools;
re-derive before quoting -- see "Counts" at the end). Read this after the
README and before `CLAUDE.md`, which carries the invariants an editor must
not break.

## Design position

CASCADE is an *agentic* manipulation stack: an LLM (or a human in a chat
host) commands a **curated skill API**, every skill is safety-gated and
traced, and the physical effect of every skill is **verified by a channel
the actuator does not own**. Three published systems set the shape:

- **ASPIRE** (NVIDIA GEAR, 2026): a coding agent programs a robot through
  primitives whose every call records multimodal evidence (their ablation:
  the trace-exposing execution engine alone lifted success 14% → 62%). We
  replicate the API surface (`get_observation`, `localize_object`,
  `preview_grasp`, motion + gripper primitives), the per-call trace
  (`trace.jsonl` + before/after keyframes) and the post-run diagnosis →
  skill-note loop (`agent/aspire.py`, `scripts/learn_from_runs.py`).
- **Agentic-VLA** (ICML 2026): LLM sub-goal decomposition, a VLM
  "exploration critic" consulted on failure (prompt reused near-verbatim,
  `agent/advisor.py`), embedding-indexed experience memory (tier 2), and
  *adaptive reward synthesis* -- which we run at inference time as
  **checkable milestones** (`agent/milestones.py`: symbolic against the
  world model first, VLM only when the symbolic tier abstains).
- **Claude plays robotics** (Anthropic, 2026): control-interface level
  dominates model choice; one LLM turn costs 2–15 s so routine commands
  must not wait on the model; structured state beats extra image context;
  a *cursor the model can query* (`probe_point`) beats overlays it must
  read (their 6% → 32%).

Two later additions changed what the loop measures rather than how it acts:

- **Pigey / Harness-VLA** (2026): a self-reported `ok` is a claim, not a
  fact. `agent/effects.py` verifies every primitive's effect against an
  independent channel and **downgrades a reported success** when refuted;
  `memory/envelope.py` folds outcomes into a per-skill operating envelope
  the planner reads back.
- **Vesta** (arXiv:2606.20905, 2026): a VLM planner given its own history as
  *images + text* plans multi-step tasks far better than one given text
  alone (their Table 5: 49.7 → 75.9). `memory/episodic.py::memory_frames`
  + `orchestrator._with_memory_harness` + the `task_memory` MCP tool are
  that harness. No Vesta weights or code were released; only the harness
  and the evaluation design were adopted (ROADMAP "Landed 2026-09-10").

We deliberately did **not** build a VLA-policy-in-the-loop executor: the
deterministic skill stack is debuggable, safety-gateable and runs offline
(ROADMAP records the decision and the LIBERO layer-attribution numbers
that back it).

## The runtime, end to end

```
                 chat host (OpenClaw / Hermes / Claude Code / Codex)          CLI / REPL
                 host LLM picks tools over MCP stdio                         --task / --interactive
                          │                                                          │
                          ▼                                                          ▼
              apps/mcp_server.py  ── 41 tools ──┐                  agent/orchestrator.py
              (33 skills − task_done             │                  tier 1 REFLEX   regex grammar      ~µs
               + 8 host extras: camera_snapshot, │                  tier 2 HABIT    experience memory  ~ms
               world_state, task_memory, ...)    │                  tier 3 LLM      + memory harness   2–15 s/turn
                                                 ▼                            │
                              skills/runtime.py  SkillRuntime.execute()  ◀────┘
                              ONE choke point: arm selection, BEFORE keyframe, watcher pause,
                              skill body, postcondition VERIFY, envelope, AFTER keyframe,
                              trace row (with tier), memory tuple <frame, action, verdict>
                                                 │
        ┌──────────────┬──────────────┬──────────┼───────────────┬─────────────────┬──────────────┐
        ▼              ▼              ▼          ▼               ▼                 ▼              ▼
   perception/     grasping/       control/    safety/         memory/            sim/            eval/
   CameraRig →     GraspGen-X      Pinocchio   SafetyHarness   BeliefStore        MuJoCo world    Robo-Dopamine
   WorldWatcher    (ZMQ) + OBB     FK/IK,      per arm: every  (persisted),       registry,       progress judge
   → BeliefStore   fallback,       min-jerk    50 Hz waypoint  EpisodicMemory     rendered RGB-D  (GRM/VLM, off
   + occupancy     outcome memory  streaming   + occupancy     (frames K=4),      cameras, truth  the hot path)
   client          re-rank         to ANY arm  + neighbours    envelope, habits   channel
```

Everything above `SkillRuntime` decides *what*; everything below it is the
same for every brain, every tier and every robot. That is the property the
name refers to: behaviour, safety and tracing do not depend on which tier
(or which hardware) acted.

### Always-on perception, reflex-first dispatch (since 2026-07-18)

```
N cameras ──CameraStream (thread each, latest-frame slot, drop-stale; rendered
   │          cameras pump at profile `fps`, 10 for MuJoCo scenes)
   │            ├── WorldWatcher (thread, ~3 Hz): detector + HSV colour tag +
   │            │     robot-body mask ──▶ BeliefStore (label + colour + 3D +
   │            │     freshness) and occupancy integration
   │            └── StreamServer (lazy MJPEG dashboard: rgb | depth | agent view,
   │                  narration, object table, chat, STOP)
   │
chat command ("pick and place the red cube")
   ├─ tier 1 REFLEX   template grammar -> skill plan          agent/reflex.py
   ├─ tier 2 HABIT    hashed-BoW cosine ≥ 0.9, wins > losses  runs/experience.json
   └─ tier 3 LLM      decomposition + tool loop + advisor     agent/orchestrator.py
        all tiers execute through the same SkillRuntime; every trace row
        records `tier: reflex | experience | llm | mcp-host`
```

- **Latest-slot streaming, never queues** (`perception/stream.py`).
- **Warm world model.** Command resolution is a belief lookup, not an
  observe→detect round trip. Fusion pauses during `_MOTION_SKILLS` (the
  held object must not be re-fused mid-air). Two observations with
  DIFFERENT measured colours are two objects however close; proximity
  fusion (8 cm) is for label aliases of one object.
- **Colour without CLIP** (`perception/colors.py`): median mask HSV → colour
  word, stored on beliefs, matched against colour words in queries.
- **LazyArm** (`control/lazy_arm.py`): the MCP server pre-warms cameras,
  detector and world model at startup; motors are not touched until the
  first motion command. Its `profile_type` is readable without
  materializing, which is what lets the truth channel bind early.

### The execute() choke point

`SkillRuntime.execute(name, args)` is the only way a skill runs, from any
tier or host. In order:

1. `arm=` popped from the args and bound thread-locally for this call
   (multi-arm; `""`/`default` = primary; unknown name = `SkillError`).
2. BEFORE keyframe (a fresh frame if none exists -- the first skill of a
   run used to record `null`); a `PostconditionChecker.snapshot()` of the
   target object.
3. Watcher paused for motion skills; the skill body runs; every exception
   becomes `{"ok": false, "error": ...}` -- nothing escapes by design.
4. **Postcondition verification** (`agent/effects.py`): the effect is
   measured on the strongest available channel -- `physics` (sim truth),
   `belief` (perception), `gripper` (jaw width). A refuted claim
   *downgrades* `ok` and sets `self_reported_ok`. A displacement is two
   readings of the SAME channel (`_comparable_start`); a verifier that
   itself crashes yields an UNVERIFIED verdict naming the cause, never a
   silent pass.
5. Envelope update (`memory/envelope.py`), AFTER keyframe -- a FRESH frame
   for motion skills, taken after the arm stopped (the pre-motion
   `last_frame` graded the logger, not the robot, and an outcome judge
   scored 0% on a confirmed pick).
6. Trace row (`trace.jsonl`, with `tier`), and the Vesta memory tuple:
   AFTER frame + action text + independent verdict.

### Motion safety path

Skills only ever hold a `SafeArm`. `SafeArm.move_joints()` stretches the
duration so the min-jerk peak stays under the velocity cap, checks
perception freshness once at `begin_motion()`, then `ArmBase.stream_to()`
asks `SafetyHarness.approve()` for every 50 Hz waypoint: joint limits and
margins, workspace AABB, table-plane clearance (with an explicit exemption
cylinder around a grasp target), keep-out zones, perception watchdog,
e-stop latch, the **occupancy clearance gate** (`perception/occupancy.py`:
an ESDF/EDT distance query against the fused map, robot body masked out
before integration; stale or absent map = SKIP, never "blocked"), and the
**inter-arm gate** (segment-to-segment link-centreline distance in a shared
table frame, `safety/geometry.py`, when two arms declare `base_pose`).
`SafetyViolation` aborts mid-stream. `vet_pose()` is the static twin used
by grasp ranking so a doomed candidate loses before the arm moves. Gripper
commands bypass geometric gating (e-stop check only).

### Grasp pipeline

```
localize ─▶ ObjectFix (base-frame OBB; de-biased centre, verified on 2 engines)
   ├─▶ GraspGen-X candidates (ZMQ :5556, learned 6-DoF; gripper passed as a
   │    swept volume -- the arm profile owns `grasp.graspgenx.sweep`)
   └─▶ OBB candidates (analytic, always computed)     any server error → OBB only,
                                                      probed ONCE at startup, banner says which
   grasp-outcome memory re-rank + z-nudge (~/.cascade/grasp_memory.json)
   select_grasp: jaw-width filter ▸ IK pregrasp → grasp (seeded from home_q, on
   purpose) ▸ harness pre-vet incl. 7 samples along the descent
   re-home (forces the elbow-up branch) ▸ pregrasp ▸ exempted descent ▸ two-stage
   stall-aware close ▸ lift ▸ air-grasp check (jaw fraction) ▸ grip verified
   place_at: same look that aims the held object detects a SLIP (object far below
   the TCP) and raises instead of lowering an empty gripper; pick_and_place
   re-grasps until `grasp.persist_seconds` / `max_pick_attempts` run out
```

Single-hinge jaws (SO-101) close toward the fixed tip, so the profile
declares the jaw datum (`jaw_fixed_tip_m`, `jaw_close_dir`) and the selector
displaces the IK target accordingly -- without it every grasp straddled the
prop while perception was accurate to 1.4 mm.

### Sim as an instrument, not a stand-in

`sim/mujoco_world.py` keeps ONE `MjModel`/`MjData` per resolved MJCF path
(refcounted registry, one lock). The arm steps it; rendered cameras
(`perception/mujoco_camera.py`, `type: mujoco`) paint from it; the truth
channel (`sim/truth.py`) reads free-body poses from it. So the camera sees
the physics prop, not a painted one, and `postcondition: confirmed
(channel: physics)` is a measurement. `demo_scene.py` writes the scene
(arm MJCF + table + N props from the camera profile's `extra_props`)
deterministically, so either the arm or a camera can create it first.
Isaac Sim plays the same role over a TCP bridge (`scripts/isaac_bridge.py`,
`sim/bridge_client.py`) with `RigidPrim` poses as truth. `reset_scene`
puts free bodies back on `qpos0` under the world lock, then burns one frame
before observing (a render in flight when the state was teleported would
otherwise be served as fresh).

### Memory

| store | what | horizon | consumer |
|---|---|---|---|
| `BeliefStore` (`memory/beliefs.py`) | objects: label, colour, 3D, freshness; visible/remembered | persisted across runs (wall-clock stamps, `LOADED_MIN_AGE_S` floor, 6 h max age) | every skill; can inform the agent, can never aim the jaws (`belief_fallback_age_s`) |
| `EpisodicMemory` text ring | events, outcomes | ~15 s | `recall_memory`, narration |
| `EpisodicMemory` frame ring | AFTER frame + action + verdict per motion skill | task-scale (600 s), reset per task / by `reset_scene` | `memory_frames(k)`: first frame pinned, uniform sample, newest last → LLM turn (images) and `task_memory` tool |
| `ExperienceMemory` (`agent/reflex.py`) | command → plan habits, hashed BoW in a TurboQuant index | `runs/experience.json` | tier 2 |
| `GraspOutcomeMemory` | per-object grasp features, wins/losses | `~/.cascade/grasp_memory.json` | grasp re-rank + z-nudge |
| `OperatingEnvelope` (`memory/envelope.py`) | per-skill outcome statistics and failure classes | `runs/` | planner context, ROADMAP follow-ups |

`skills/library.py` holds markdown skill notes: `agent/aspire.py` distils a
*validated repair* (a failure followed by the same primitive succeeding)
from a finished run's trace (`scripts/learn_from_runs.py`, between sessions,
never mid-demo), and `retrieve()` loads guard-matched notes into the tier-3
context at task start (`orchestrator.run_task`, wired in `build_runtime`).

### Evaluation

`eval/progress_judge.py` is a Robo-Dopamine-style progress judge: BEFORE/
AFTER keyframes (plus optional goal image) → `<score>±NN%</score>` from a
GRM or any OpenAI-compatible VLM. It runs **off the hot path**
(`scripts/judge_run.py` over a finished run dir) and is calibrated against
the physics postcondition per step (confusion matrix in the run summary).
The first honest number on this rig: +0.45 on a physics-confirmed pick
after the AFTER-keyframe fix; 0.00 before it.

## Module map

```
src/cascade/
├── types.py            Frame / Detection / ObjectFix / Grasp / RobotState / SkillError
├── config.py           YAML profiles (cameras/, arms/, llm/) → one Cfg; `extends:`,
│                       arm `overrides:`, ${repo}/${assets}; CASCADE_BOOTH overlay
├── device.py           resolve_device(): auto CUDA/ROCm → MPS → CPU, degrade with a warning
├── perception/
│   ├── camera_base.py        CameraBase ABC + make_camera(); Frames carry METRIC depth
│   ├── realsense_camera.py   D4xx / L515          opencv_camera.py  RGB-only UVC
│   ├── mock_camera.py        synthetic tabletop / npz replay
│   ├── mujoco_camera.py      RGB-D RENDERED from the shared MuJoCo world (type: mujoco)
│   ├── isaac_camera.py       Isaac bridge frames (RGB-D + per-frame T_base_cam)
│   ├── depth_provider.py     sensor → mono plugin → table-plane ray-cast
│   ├── detector.py           YOLOE / YOLO-World + MockDetector (open world by default)
│   ├── vlm_detector.py       VLM as detector      vlm_ground.py  second-chance grounder
│   ├── segmenter.py          mask refinement      robot_mask.py  arm body out of depth
│   ├── grounding.py          Extrinsics + localize (colour/near-aware, de-biased OBB centre)
│   ├── calibration.py        Kabsch camera→base fit with RMSE + degeneracy refusal
│   ├── colors.py             mask HSV → colour word; colour-query parsing
│   ├── stream.py / world.py  CameraStream + CameraRig / WorldWatcher (always-on fusion)
│   ├── occupancy.py          OccupancyMap client + harness clearance gate
│   ├── occupancy_backends.py nvblox | warp | voxel (+ _warp_tsdf_kernels.py: TSDF carve + exact EDT)
│   ├── probe.py / pixel_target.py / visual_interface.py / visual_diff.py
│   │                         cursor, pixel→object, annotated agent view, before/after diff
│   ├── reference.py          goal/reference images      workspace.py  reachable-region filter
├── memory/
│   ├── beliefs.py      object permanence, colour-aware fusion, save/load (wall clock)
│   ├── episodic.py     text ring (15 s) + frame ring (task-scale) + memory_frames(k)
│   ├── envelope.py     Harness-VLA operating envelope (per-skill outcome stats)
│   ├── grasp_memory.py persisted grasp-outcome prior (re-rank + z-nudge)
│   └── turboquant.py / vector_index.py   4-bit rotation quantizer + asymmetric top-k
├── control/
│   ├── arm_base.py     six abstract methods + min-jerk stream_to() + make_arm()
│   ├── arm_rig.py      N named arms, first = manipulation arm (twin of CameraRig)
│   ├── kinematics.py   Pinocchio FK/IK (DLS + restarts, N joints, task weights)
│   ├── usd_model.py    USD-physics → URDF (no aarch64 usd-core)
│   ├── lazy_arm.py     motors untouched until the first motion command
│   ├── mock_arm.py     kinematic sim, any DoF
│   ├── mujoco_arm.py   any MJCF; engines mjc (C) | warp (MuJoCo Warp); viewer guarded
│   │                   by a display probe (a sleeping display segfaults GLFW)
│   ├── isaac_arm.py    Isaac articulation over the TCP bridge
│   ├── feetech.py / feetech_arm.py   SO-101 & co over Feetech serial (UNVERIFIED on hw)
│   ├── rebot_rs_arm.py / rebot_rs_mb_arm.py   reBot B601 over CAN / MotorBridge
│   ├── ros2_arm.py     ANY ros2_control robot (JointState in, JointTrajectory out)
│   └── unitree_arm.py  Unitree SDK arms (H1 / H1-2 / G1)
├── safety/
│   ├── harness.py      SafetyHarness (approve / vet_pose, escape rules) + SafeArm
│   └── geometry.py     segment-segment distances for the inter-arm gate
├── grasping/
│   ├── obb_grasp.py    base-frame OBB grasps      graspgenx_backend.py  ZMQ client + fallback
│   ├── selector.py     width ▸ IK walk ▸ harness pre-vet (jaw datum aware)
│   └── force.py        material → two-stage close profiles
├── agent/
│   ├── orchestrator.py reflex → habit → LLM loop; memory harness injection; TaskReport
│   ├── reflex.py       tier-1 grammar (incl. reset_scene) + tier-2 ExperienceMemory
│   ├── effects.py      PostconditionChecker + annotate_result (Pigey closed loop)
│   ├── milestones.py   checkable milestones: symbolic first, VLM second, UNKNOWN honest
│   ├── llm.py          OpenAI-compat (cloud/local) / Anthropic / Cosmos3 / Mock
│   ├── cosmos3.py      Cosmos3-Edge XML tool-call dialect
│   ├── prompts.py / advisor.py   persona, decomposition, VLM critic
│   ├── aspire.py       post-run diagnosis → skill-library note
│   └── trace.py        trace.jsonl + keyframes
├── skills/
│   ├── runtime.py      SkillRuntime: 33 skills + task_done, TOOL_SPECS, _MOTION_SKILLS
│   └── library.py      markdown repair notes; written by aspire.py, retrieved per task
├── sim/
│   ├── mujoco_world.py shared MjModel/MjData registry (arm + cameras + truth, one lock)
│   ├── demo_scene.py   deterministic scene writer: arm MJCF + table + N props
│   ├── truth.py        physics-truth channel (MuJoCo + Isaac), LazyTruthPoseFn
│   ├── mujoco_rgbd.py  offscreen RGB-D + data.xpos truth (perception verification)
│   └── bridge_client.py newline-JSON TCP client for scripts/isaac_bridge.py
├── eval/progress_judge.py   Robo-Dopamine progress judge (GRM / VLM), off the hot path
└── apps/
    ├── demo.py         build_runtime() = the composition root; CLI --task / --interactive
    ├── mcp_server.py   MCP stdio front-end: 41 tools, out-of-band stop, per-call log
    ├── stream_server.py lazy MJPEG dashboard (+ chat, STOP)     live_view.py  RigViewer
    ├── live_control.py viewer-driven control        record.py / viewer.py  capture / view
```

Sidecars (own process, own venv, ZMQ): `scripts/serve_graspgenx.sh`
(learned grasps, CUDA) / `serve_graspgenx_stub.py` (protocol double, any
host); `scripts/serve_occupancy.sh` → `serve_occupancy_bridge.py`
(`--backend auto`: nvblox > warp > voxel). Both are **probed at startup**
and named in the banner; a missing sidecar degrades loudly to its fallback,
never silently.

## ROS2, humanoids, and what is NOT here yet

**ROS2 today = arms.** `type: ros2` (`control/ros2_arm.py`) speaks the two
interfaces every `ros2_control` deployment has -- `sensor_msgs/JointState`
in, `trajectory_msgs/JointTrajectory` (or `Float64MultiArray` for a forward
position controller) out -- with joints addressed **by name**, so the
driver's `JointState` order can never shift the mapping. Adding a ROS2 robot
is copying `configs/arms/ros2_generic.yaml` (a `template: true` file the
factory refuses to run until its numbers are filled in) and pointing it at
the robot's URDF, joint names, keyframes, gripper travel and workspace; the
harness, IK, grasping, skills and MCP tools drive it unchanged. `rclpy` is
imported inside `connect()`, so the mock stack and CI without ROS2 still
collect the module; the unit tests inject stub `rclpy` modules. Shipped
profiles: `so101_ros2`, `piper`, `h1`, `h1_2`, `fr3`. Design rationale (QoS,
streaming vs. single trajectory, stop semantics, licence notes) in
`docs/ROS2_BACKEND_BRIEF.md`. **Unverified on hardware.**

**Humanoids today = one arm of a standing robot.** `type: unitree_arm`
(`control/unitree_arm.py`) drives an arm of a G1 / H1 / H1-2 over Unitree's
Arm-SDK channel (`rt/arm_sdk` LowCmd with the per-family motor index table
and the CRC the firmware validates; `rt/lowstate` in), ramping the SDK
"weight" so the locomotion controller hands the arm over without a jerk.
Balance, legs, waist and walking stay with Unitree's own controller; a
handless gen-1 H1 declares `max_width_m: 0` so grasps are refused, not mimed.
The humanoid profiles' `base_pose` places the shoulder in the shared table
frame, which is what the inter-arm and occupancy gates need. **Unverified on
hardware.**

**Not here: a mobile base, navigation, mapping, robot self-localization.**
Nothing publishes a Twist, consumes odometry or a map, or talks to Nav2;
"localization" in this codebase means object grounding. The design for that
layer -- a `MobileBase` twin of `ArmBase`, `MobileRig`, `base=` binding in
`execute()`, Vesta's three navigation verbs as skills (`go_to_pixel`,
`turn`, `stop_navigation`) with the memory harness spanning the walk, a
2D costmap sliced from the existing Warp ESDF, and two navigation backends
(Nav2 when ROS2 is sourced, the Warp planner otherwise), targeting a Unitree
G1/H1 in Isaac Sim first -- is written up in
`docs/MOBILITY_AND_NAVIGATION_DESIGN.md` and scheduled in the ROADMAP.

## Launch and hosts

`run.sh` → `scripts/launch.sh` is the one-click entry: `--sim auto|isaac|
mujoco|none`, `--setup` (venv, extras, assets, OpenClaw CLI, provider
onboarding), `--check` (report only, never mutates), `--dry-run`, `--down`
(stops the sidecars it started AND the per-session MCP servers the gateway
never reaps). Before READY it proves the stack: runtime built with the
server's exact env, tools listed, a trivial brain turn, one real
`pick and place` turn checked against the physics channel (verdict scoped
to traces written during that turn), then `reset_scene` so the first
visitor sees the spawn layout. The MCP server is registered with
`requestTimeoutMs: 300000` (a persistent pick runs 60–120 s; the host's
60 s default cancelled it and latched the e-stop), dead MCP entries are
pruned, and on macOS the server runs under `mjpython` so the MuJoCo window
can open -- when a display is active; a sleeping display is detected and
the window retried on the next motion instead of segfaulting the server.

Hosts: OpenClaw (native `mcp.servers`), Hermes (`~/.hermes/config.yaml`),
Claude Code (`.mcp.json`), Claude Desktop, Codex -- all via
`scripts/setup_agents.py`. Host and brain are different roles: as a host
the platform's LLM picks tools and cascade's tiers are bypassed
(`llm=mock` inside the server); as a brain (`--llm hermes|anthropic|
openai|local_*`) cascade runs its own loop with all three tiers.

## Key decisions (still load-bearing)

- **Metric depth in the Frame.** Sensor units differ per camera (L515
  0.25 mm/unit vs D4xx 1 mm); convert at the camera boundary once.
- **Grasps planned in the base frame.** Camera pose affects visibility,
  not grasp geometry -- that is what camera-agnostic means operationally.
- **The arm's shape is data.** `n_joints`, limits, keyframes, tool-frame
  order, gripper travel, jaw datum, reach come from the profile;
  `_profile_q` *requires* them (a 6-vector broadcast onto a 5-DoF arm is
  a bent link). Profiles inherit with `extends:` (same robot, different
  transport) and carry `overrides:` for rig geometry.
- **One backend class, pluggable physics engine.** `engine: mjc | warp`
  under one `MujocoArm`; the engine surface is batch-oriented (whole
  qpos/ctrl vectors) so a device runtime does not pay a host↔device
  round-trip per joint. Measured single-arm: C ~4.9 µs/step, Warp on CPU
  ~3.2 ms/step -- the C engine is the demo default, Warp is for developing
  the GPU path.
- **Feedback, not sleep.** Every backend reports real joint positions;
  settling is `max|q − q*| < tol` with a per-profile tolerance and timeout,
  and a stepped-on-demand sim steps inside its own wait.
- **Fail-closed safety, degrade-open knowledge.** Limits and the e-stop
  fail closed; *information* sources (occupancy map, neighbour arm,
  grasp server, truth channel, verifier) degrade to "no data, say so",
  never to "blocked" or to a silent pass. The banner and the run summary
  name what actually answered.
- **A claim is not a fact.** Every effect is verified on an independent
  channel when one exists, and the verdict travels with the result, into
  the trace, into memory and to the judge.
- **Memory is structured first, embeddings second.** Recall tools work on
  labels/time/positions; the TurboQuant index has one live consumer (tier
  2). Frames -- not text -- are what the planner is shown of its own past.
- **Device agnosticism is a resolution step.** `device.py` answers "where
  does this model run" once; torch is not a dependency (per-platform build).

## Verification status (2026-09-10, macOS, extras sim + sim-warp + grasping + occupancy + llm)

- `pytest tests/ -q`: **737 passed, 2 deselected (hardware), 0 skipped**,
  ~4 min. `ruff check src/ scripts/*.py tests/ --select F,E9,B023,B904` clean.
- Real chat path, one gateway session (the dashboard path): two-cube
  memory task -- 2 `pick_and_place` confirmed on the physics channel,
  0 tool failures, the brain's answer cites the memory frames' verdicts;
  `reset_scene` returns both props to their spawn pose (mm); a second
  `world_state` matches the first.
- One click on a fresh export of the committed tree (no venv, no assets):
  venv + extras + 24 fetched meshes + 41 tools + physics-confirmed pick +
  reset + MuJoCo window open, exit 0.
- Perception de-bias replicated on two engines against physics truth
  (Isaac 1.85 → 0.56 cm; MuJoCo 2.27 → 0.49 cm).
- Progress judge vs physics: tp on a confirmed pick after the keyframe
  fix (was fn).
- Live hardware: L515 streaming and RobStride mechPos reads were exercised
  on the reference rig (read-only); **real-arm motion, the SO-101 serial
  driver, the ROS2 and Unitree backends are unverified on hardware.**

## Known limitations

- Same-colour identical objects closer than 8 cm can blur into one belief
  (different colours never do).
- Grip force is a stiffness proxy (kp scaling + stall detection), not a
  calibrated force loop.
- `RebotRSArm.disconnect()` cuts torque: park (`move_home`) first.
- The MCP server executes one tool call at a time; stops are handled
  out-of-band by the stdin reader (never queued behind a motion), but a
  second *motion* request waits.
- The rendered-camera window (`RigViewer`) cannot open on macOS from the
  server (Cocoa needs the main thread; `opencv-python-headless` has no
  highgui); the MuJoCo physics window and the browser dashboard are the
  visuals there.
- Skill-library notes are retrieved by guard-word match on the task text
  (`aspire.retrieve`), not by embedding; a visual embedder for episodic
  recall is still on the ROADMAP.

## Counts

Numbers in these docs drift. Re-derive before quoting:

```bash
python -m pytest tests/ -q --collect-only | tail -1
python - <<'EOF'
import sys; sys.path.insert(0, "src")
from cascade.skills.runtime import TOOL_SPECS, _MOTION_SKILLS
from cascade.apps.mcp_server import _EXCLUDED_TOOLS, _EXTRA_TOOLS
n = {t["name"] for t in TOOL_SPECS}
print(len(n) - 1, "skills;", len(_MOTION_SKILLS), "motion;", len(n - _EXCLUDED_TOOLS) + len(_EXTRA_TOOLS), "MCP tools")
EOF
```

`tests/test_llm_and_library.py` pins the README's headline skill and tool
counts to these derived numbers, so a drift there fails the suite.
