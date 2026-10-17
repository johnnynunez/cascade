# Architecture

Read this after the [README](../README.md) and before `AGENTS.md`. Measured results
and open work per capability: [capability status](ROADMAP.md). Counts are
re-derived at the [end of this document](#counts).

CASCADE puts **curated domain tools** between an LLM (or a human) and robot
controllers. Motion is gated and traced; independent observations decide whether
a physical postcondition is confirmed, refuted or unverified, and missing evidence
stays unverified. Robot profiles select capabilities, policies, sensors and limits;
Isaac Sim, MuJoCo and hardware drivers are adapter implementations, and deployment
guides carry the concrete versions and source pins.

## Domains beyond the arm

The [architecture diagram and capability matrix](ROBOT_MODULARITY.md) show the
opt-in `RobotRuntime`: manipulation, locomotion, fastening, passive sensing and
read-only spatial domains behind explicit resource contracts. MCP and bounded
skill graphs dispatch registered tools; `FleetRuntime` coordinates separate robot
runtimes. The arm and mobile entrypoints stay available on their own.

- **Embodiment and frames.** [Embodiment declarations](ROBOT_MODULARITY.md#declared-embodiment-embodiment) describe links, joints, transmissions and sensor attachments; [measured frame trees](SPATIAL_PROVIDERS.md) bind transforms to capture time, map and epoch. Mixed physical actuation stays refused until dynamic frames, shared control and robot-specific limits are validated. The opt-in `whole_body` contract (mock arm on a mock base, [contract](ROBOT_MODULARITY.md#multi-domain-embodiments-mounted-arms)) is software evidence and refuses every physical mounted composition.
- **Spatial.** The [spatial domain](SPATIAL_PROVIDERS.md) resolves capture transforms and landmark memory and plans on immutable planar maps; the [RGB-D observation domain](SPATIAL_PROVIDERS.md#retained-rgb-d-surface-observations) adds retained surface annotations. Optional cuVSLAM runs in an isolated process ([current state](#ros2-humanoids-and-what-is-not-here-yet)).
- **Legged bases.** [MicroDuck](MICRODUCK.md) selects `MobileRig`, `SafeBase` and `MobileSkillRuntime` before any arm is built and shares MCP, traces and episodic memory. Its Newton bridge runs the policy on completed physics solves, with no LLM in the control loop; independently sampled pose and solved support decide outcomes. A native foundation has run 1, 2 and 12 robots with zero velocity commands in one scene; commanded fleet tasks are unvalidated. The Unitree H2 PhysX candidate ([HUMANOID_H2.md](HUMANOID_H2.md)) uses the same `MobileBase` / `SafeBase` / verifier path. Neither robot is admitted physically.
- **Planning, rendering, fastening.** [cuMotion](CUMOTION.md) adds `planning/runtime.py` at the arm composition boundary: `isaac_cumotion` binds model, joint order/signs, base and tool; `SafeArm` approves the native trajectory and the Isaac executor slows it uniformly while checking sampled edges (contact segments need a linear TCP corridor; the final 1 mrad start-drift veto is unchanged). The [OVRTX bridge producer](OVRTX_RENDERER.md) sends immutable physics snapshots to an isolated renderer, because editing another live `UsdStage` inside Kit can invalidate the simulator. The [Factory controller](FACTORY_FASTENING_RUNTIME.md#thread-contact-fixture) drives SO-101 joints and a hex socket in Newton with read-only seating verifiers; it does not turn `turn_screw` into a preload controller.
- **Conversation, split deployment (B51, opt-in).** `cascade-robot-service` owns the composed `RobotRuntime` and serves it over `cascade.robot-runtime/1` (`robotics/endpoint.py`, stdlib only); `cascade-conversation` with `robot_endpoint` uses a `RemoteRobotRuntime` client and builds no robot. Every check stays in the robot process. The protocol adds a bearer token, one supervision lease whose expiry latches the robot stop, a lease-free stop route that runs beside a motion, and reset only with the lease and the observed generation; `/healthz` and `/readyz` are unauthenticated. Across hosts it runs over opt-in TLS (B71): `--tls-cert/--tls-key` allow a non-loopback bind (refused without TLS), the client trusts only `robot_tls_ca` (hostname checked) and/or `robot_tls_fingerprint`, and handshakes run per connection so a stalled peer cannot delay a stop. The plaintext default is golden-pinned. See [separate deployment](CONVERSATION.md#deploy-the-conversation-service-separately) and [across hosts](CONVERSATION.md#across-hosts-opt-in-tls-b71).

## Design position

The manipulation stack adapts three published systems:

- **ASPIRE** (NVIDIA GEAR): the API surface (`get_observation`, `localize_object`, `preview_grasp`, motion + gripper primitives), the per-call trace (`trace.jsonl` + before/after keyframes) and the post-run diagnosis → skill-note loop (`agent/aspire.py`, `scripts/learn_from_runs.py`).
- **Agentic-VLA**: sub-goal decomposition, a VLM critic consulted on failure (`agent/advisor.py`), embedding-indexed experience memory (tier 2) and **checkable milestones** (`agent/milestones.py`: symbolic against the world model first, VLM only when it abstains). The same rate-limited critic runs *before* each LLM-tier motion dispatch (`PlausibilityChecker`, Human-CLAW's pre-execution verifier); its `plausibility` verdict is **advisory only**, the harness alone refuses motion.
- **Claude plays robotics** (Anthropic): one LLM turn costs 2–15 s, so routine commands must not wait on the model; structured state beats extra image context; a cursor the model can query (`probe_point`) beats overlays it must read.

Two later additions change what the loop measures. **Pigey / Harness-VLA**: a
self-reported `ok` is a claim; `agent/effects.py` verifies each primitive's effect on
an independent channel and downgrades a refuted success, and `memory/envelope.py`
folds outcomes into a per-skill operating envelope. **Vesta** (arXiv:2606.20905): a
planner shown its own history as images + text plans multi-step tasks far better
than with text alone; `memory/episodic.py::memory_frames`,
`orchestrator._with_memory_harness` and the `task_memory` MCP tool are that harness.

The deterministic skill stack is the default executor: debuggable, gateable and
offline. A language-conditioned policy can serve label grasps as an opt-in second
executor, `grasp.executor: vla` ([VLA executor](VLA_EXECUTOR.md), B49):
`grasp_object` keeps its name, schema, localization and verifier; only the motion
between "jaws open at home" and "lifted" comes from the policy's action chunks, and
every chunk passes the SafetyHarness before and during motion.

## Manipulation runtime detail

This is the arm path retained by the composed runtime; other domains share only
the `RobotRuntime.execute()` boundary ([modular diagram](ROBOT_MODULARITY.md)).

```
                 chat host (OpenClaw / Hermes / Claude Code / Codex)          CLI / REPL
                 host LLM picks tools over MCP stdio                         --task / --interactive
                          │                                                          │
                          ▼                                                          ▼
              apps/mcp_server.py  ── 52 tools ──┐                  agent/orchestrator.py
              (43 specs − task_done              │                  tier 1 REFLEX   regex grammar      ~µs
               + 8 host extras: camera_snapshot, │                  tier 2 HABIT    experience memory  ~ms
               world_state, task_memory, ...;    │                  tier 3 LLM      + memory harness   2–15 s/turn
               + list_programs/run_program only  │
               with the opt-in programs tier;    │
               minus what the rig's capability   │
               matrix withholds, with reasons)   ▼                            │
                              skills/runtime.py  SkillRuntime.execute()  ◀────┘
                              ONE choke point: arm selection, BEFORE keyframe, watcher pause,
                              skill body, postcondition VERIFY, envelope, AFTER keyframe,
                              trace row (with tier + advisory `plausibility` hand-off),
                              memory tuple <frame, action, verdict>,
                              outcome stamp ok | failed | stuck (+ human-facing ask)
                                                 │
        ┌──────────────┬──────────────┬──────────┼───────────────┬─────────────────┬──────────────┐
        ▼              ▼              ▼          ▼               ▼                 ▼              ▼
   perception/     grasping/       control/    safety/         memory/            sim/            eval/
   CameraRig →     GraspGen-X      Pinocchio   SafetyHarness   BeliefStore        MuJoCo world    Robo-Dopamine
   WorldWatcher    (ZMQ) + OBB     FK/IK,      per arm: every  (persisted),       registry,       progress judge
   → BeliefStore   fallback,       min-jerk    profiled edge   EpisodicMemory     rendered RGB-D  (GRM/VLM, off
   + occupancy     outcome memory  streaming   + occupancy     (frames K=4),      cameras, truth  the hot path)
   client          re-rank         to ANY arm  + neighbours    envelope, habits   channel
```

Everything above `SkillRuntime` decides *what*. Skills, safety and tracing share
one composition root across brains and tiers. Physical clocks, captured joint
state, collision geometry and payload recovery are validated per arm, never
assumed interchangeable.

### Always-on perception, reflex-first dispatch

```
N cameras ──CameraStream (thread each, latest-frame slot, drop-stale; rendered
   │          cameras pump at profile `fps`, 10 for MuJoCo scenes)
   │            ├── WorldWatcher (thread, ~3 Hz): detector + HSV colour tag +
   │            │     robot-body mask ──▶ BeliefStore (label + colour + 3D +
   │            │     freshness) and occupancy integration; never waits for
   │            │     the brain (detect_nowait)
   │            └── StreamServer (lazy MJPEG dashboard: rgb | depth | agent view,
   │                  narration, object table, chat, STOP; opt-in wrist-tile
   │                  highlight `stream.wrist_narration`, B47)
   │
chat command ("pick and place the red cube")
   ├─ tier 1 REFLEX   template grammar -> skill plan          agent/reflex.py
   ├─ tier 2 HABIT    hashed-BoW cosine ≥ 0.9, wins > losses  runs/experience.json
   │                  AND the task's clause structure (a
   │                    sequence matches clause by clause)
   │                  + recipes (xyz -> perception queries,   runs/recipes.jsonl
   │                    re-grounded before any motion)        memory/recipes.py
   ├─ tier 2.5 PROGRAM  OPT-IN (agent.programs: false):       agent/programs.py
   │                  one authoring turn when tiers 1-2 had   memory/programs.py
   │                    no plan; registered calls, run step   runs/programs.jsonl
   │                    by step; first unverified step stops
   └─ tier 3 LLM      decomposition + tool loop + advisor     agent/orchestrator.py
        all tiers execute through the same SkillRuntime; every trace row
        records `tier: reflex | experience | program | llm | mcp-host`
```

- **Tier 2 respects clause structure (B37, B69).** `ExperienceMemory.recall` (whole task and each curriculum clause in `_plan_one`) admits a habit or recipe only when its instruction has the task's clauses as `split_subgoals` cuts them: the same count and, for a sequence, every clause ≥ `min_sim` against its counterpart in order. `split_subgoals` cuts at sequence connectives and at a coordinated "and"/"y" (optionally "and also"/"y también") followed by a clause verb of the reflex/skill vocabulary (`_OBJECT_VERBS`, `_BARE_VERBS` in `agent/reflex.py`), except verb coordination ("pick and place the cube"), a back-reference ("… and put it in the box") and an object-taking verb with no object ("… and throw to the left"). `parse_command` refuses such a compound like a "then" one, so the curriculum plans it all-or-nothing. A wrong cut can only move a command to the curriculum or the LLM tier; a single instruction is decided by index similarity alone (`tests/test_tier2_clause_structure.py`, `tests/test_tier2_and_clause.py`).
- **Programs tier (opt-in; [design note](PROGRAMS_TIER.md)).** A program is a bounded (≤ 12 steps) list of REGISTERED `TOOL_SPECS` calls with labels as parameters and positions only as `localize_object(label)+offset` queries, re-grounded before the first motion (one unresolved query aborts with zero motion). Each step is a top-level `SkillRuntime.execute()` call (`tier: program`) with its own trace row, keyframes, harness vetting and postcondition; the verdict comes from the task-effects ledger, and the first failed / refused / refuted / unverified step stops the program with a `next_action` for tier 3. `task_done`, `reset_scene`, `halt_motion`, pixel tools, `recall_step` and live-view tools are not composable. Authored and distilled programs share one admission rule: stored only from a fully CONFIRMED execution, offered only once promoted across ≥ 2 distinct tasks. `programs=None` (default) is the pre-tier dispatch exactly; mock brains and mobile/composed runtimes never get the tier. **MCP chat hosts (B42):** with the tier on, an arm server serves `list_programs` (promoted only) and `run_program` (by name, or a host-written spec run once) through the same `ProgramTier` and runner; it also refuses a step or query the capability matrix withholds or the operator hid, and a cancel of `run_program` latches the e-stop. With `memory.embedder`, programs rank by text embedding (floor-or-guard rule), else keyword overlap. Tier off: both tools are absent and every result is byte-identical (golden-pinned).
- **Latest-slot streaming, never queues** (`perception/stream.py`).
- **Warm world model.** Beliefs resolve names and rank current matches; localization still analyses an image and bounds any memory fallback. Fusion pauses during `_MOTION_SKILLS`. A frame is fused as a whole (`BeliefStore.update_frame`): detections sharing image support are one instance, matched one-to-one to beliefs by min-cost assignment. Different measured colours are two objects; proximity fusion (8 cm) is for label aliases. Colour names are held per camera (B32b): a camera that never named a belief fuses a neighbour colour (orange~yellow) only at 3D box IoU ≥ 0.75, ignoring each cloud's lowest centimetre (B32c). Fusion is size-consistent (B40): a view more than 2× AND more than 5 cm larger than the largest view a belief has had (`ObjectBelief.diameter_m`) is not that object.
- **Robot self-mask in fusion** (`perception/workspace.py`, B32a; `perception/link_mask.py`, B39). A detection more than half robot pixels is dropped; below that its robot pixels never reach 3D. Robot pixels come from the render self-mask when the frame carries one (Isaac with `CASCADE_ISAAC_PIXEL_MASK=1`); otherwise opt-in `workspace_filter.link_self_mask` projects the URDF collision geometry (5 cm link-frame cells, FK at the joint sample nearest the capture time, dilated 2 px). No sample within `max_skew_s` (0.15 s) means no mask. Measured on CPU against the reBot RS meshes: coverage 1.0, IoU 0.889–0.951, 3–5 ms per frame (`docs/evidence/b39-link-self-mask-20261009/`).
- **Capture-time alignment** (`sensing/alignment.py`, B50, opt-in). Pairs a reference capture with the nearest sample of another stream as `aligned` (|skew| ≤ `max_skew_s`), `stale` (value withheld), `missing` or `uncertain`; never interpolates, ties keep the earlier sample, one clock instance only. The sensors-domain `alignment:` block adds `sensing.read_aligned` ([contract](ROBOT_MODULARITY.md#capture-time-alignment-b50-opt-in)). With `alignment.producer_history: N` (B72), producers record every published sample in a bounded ring (mobile bridge `--state-history N`, `BufferedSensorProvider(history=N)`) and pairing uses those too; measured on CPU: 16/24 read-time pairings stale vs 24/24 aligned from the ring ([contract](ROBOT_MODULARITY.md#producer-rate-sampling-b72-opt-in)).
- **Brain-driven perception** (`perception/brain_detector.py`, `detector: type: brain`). The local Qwen3.8-27B brain names and boxes every object (inventory), each word of a configured `detect_classes` vocabulary, or the one asked for ("find the X"); SAM 2.1-L (`segmenter.model: ${repo}/models/sam2.1_l.pt`) masks the boxes. Endpoint `CASCADE_PERCEPTION_URL` / `_MODEL` / `_KEY_FILE` (defaults :8080, `qwen3.8-27b`, `~/.cascade/nemoclaw/llama.key`). The watcher calls `detect_nowait`, which never calls the brain on its own thread: a per-camera cache lets SAM track the last boxes, an unchanged scene refreshes every `idle_refresh_s` (30 s) and a changed one at once, with at most one background brain request per detector (waiting cameras served oldest-first). `describe_scene`, `_reobserve` and localization's vocabulary pass read the same cache (background refresh after `refresh_s`, 4 s); with nothing usable cached they wait for a synchronous call inside the five-second image-age check. Exception: past `max_cache_age_s` (30 s) an unchanged scene keeps its track step while that entry's background refresh is in flight or waiting. "Find the X" is always a fresh synchronous refer. `LockedDetector` skips its lock for this concurrent-safe detector. After a failed brain call a camera waits `refresh_s`; meanwhile a same-scene entry is tracked, else the result is `[]`. Without a SAM mask a detection keeps its box (`mask` `None`), except in a track step with a working segmenter: there a box without a mask is dropped (`BrainDetector._detections`). Why Qwen + SAM: on 47 real reBot keyframes it identified 83 % of objects and found 94 % of referred ones ([bake-off](evidence/perception-bakeoff-20261015/REPORT.md)).
- **Colour without CLIP** (`perception/colors.py`): median mask HSV → colour word, matched against colour words in queries.
- **LazyArm** (`control/lazy_arm.py`): the MCP server pre-warms cameras, detector and world model; motors are untouched until the first motion command. `profile_type` is readable without materializing, so the truth channel binds early.

### The execute() choke point

`SkillRuntime.execute(name, args)` is the only way a skill runs, from any tier or host:

1. `arm=` is popped and bound thread-locally (`""`/`default` = primary; unknown = `SkillError`).
2. BEFORE keyframe (a fresh frame if none exists), one wrist keyframe per `role: wrist` / eye-in-hand stream for motion skills, and a `PostconditionChecker.snapshot()` of the target.
3. Watcher paused for motion skills; the body runs; every exception becomes `{"ok": false, "error": ...}`. Every result is stamped `outcome: ok | failed | stuck`. `stuck` (`SkillStuck`, RPent's third status) means the persistence loop spent its budget or hit a reason no retry cures; `ask` is the concrete human request, the result is forced to `ok: false`, never `verified`, and the orchestrator ends the task and relays the ask verbatim. E-stop and harness refusals stay plain failures.
4. **Postcondition verification** (`agent/effects.py`) on the strongest channel: `physics` (sim truth), `belief` (perception), `gripper` (jaw width). A refuted claim downgrades `ok` and sets `self_reported_ok`. A displacement compares two readings of the SAME channel (`_comparable_start`); a crashing verifier yields UNVERIFIED with the cause. A confirmed verdict on a stuck step becomes unverified.
5. Envelope update (`memory/envelope.py`), then a FRESH AFTER keyframe once the arm stopped (wrist AFTER frames too; a failing wrist camera is dropped from the row, never fails the skill).
6. Trace row (`trace.jsonl`, with `tier`) and the Vesta memory tuple (AFTER frame + action text + verdict). `recall_step(n)` reads this back (RPent `view_env_state(step=N)`): skill, args, outcome/ask, verdict, tier and BEFORE/AFTER keyframes, as MCP image items with one caption each. Negative `n` counts from the end; an invalid `n` is an explicit error, never an old frame.

**Pick persistence never re-grasps what it holds (B36).** The persistence loops
(`pick_and_place`, and `_grasp_with_persistence` behind `handover`,
`sort_by_color` and `rearrange`) retry with a fresh re-home, re-scan and plan. An
attempt can fail AFTER its close took the object, so `_held_after_failed_attempt`
reconciles the held flag with the jaws first (`_reconcile_held` promotes the
provisional marker only when the jaws are stalled on something): the requested
object in the jaws IS the grasp and the pick goes on to place it
(`grasp_recovered_after`, `grip_verified: null`); another object stops the loop as
`stuck`. Nothing is asked during an e-stop or an unfinished contact episode.

**Grasp hold after contact on Isaac (B36).** The Isaac bridge has no force bound,
so the stage-2 position target squeezes at full drive stiffness; on PhysX that left
wrist roll a median 0.045 rad off its lift target. `gripper.hold_squeeze_frac` (a
number, or a mapping keyed by the physics engine the bridge reports in `ping`)
makes the GRASP's close wait for the jaw stall (`hold_stall_timeout_s`, default 4 s)
and hold at the measured contact opening minus that fraction of travel
(`_grasp_close`); no stall, or one within 0.02 of the stage-2 target, keeps the
stage-2 command. `close_gripper`, the screw stroke and drivers with their own
two-stage close are unchanged; the post-lift air-grasp check judges the hold
opening. `configs/arms/isaac.yaml` (and `isaac_cumotion`, `isaac_reach`) ship
`{physx: 0.05}`; kitchen profiles opt out (`null`); unset = the two-stage close
byte for byte ([evidence](evidence/b36-pick-reliability-20261010/README.md)).

**Wrist narration on the dashboard (B47, opt-in).** With
`stream.wrist_narration: true` and a wrist-view camera (`role: wrist` or eye-in-hand
extrinsics), `build_runtime` attaches a `WristNarrator` (`apps/wrist_narration.py`)
to the FIRST such stream and `execute()` brackets every `_MOTION_SKILLS` call with
`begin`/`end` (in a `finally`). The tile is highlighted and captioned with one line
built only from verifiable state: skill and target, the runtime's held-state (an
unfinished close reads `closing on ... (grasp not complete, unverified)`) and the
three-state postcondition; a hold reads `grasp unverified` unless `grasp_object`'s
own verdict covers it (`CARRIES_GRASP_VERDICT`). It never reads pixels or the arm and
changes no result. `/state` and the MCP world state gain `wrist_view`; no wrist
stream = no panel. Flag off = the exact previous dashboard bytes (sha256 pinned in
`tests/test_wrist_narration.py`).

**Kitchen placement verdict.** For single-arm Isaac kitchen `pick_and_place` calls
that report a completed motion to the green square or open box, `sim/placement.py`
takes a fresh passive observation on a private bridge connection (`LazyTruthPoseFn`,
no arm materialization); the motion result cannot supply its own identity or
verdict. `demo/kitchen/physics/placement_verdict.py` requires full footprint
containment, measured support, actual jaw release and an advancing window of at
least 0.5 simulation seconds across four steps (gaps ≤ 0.25 s), with contact
observations from the same object, jaws, engine and step. Measured failures refute;
missing or inconsistent evidence stays unverified. Acquisition has a 20-second
budget on its own socket; JSON receipts land in `<run>/placement/` with their
SHA-256 in the result and trace. Without the checkout's audit modules and kitchen
assets the check returns `unverified`. Multi-arm runtimes skip this adapter.

### Optional rendering without a physics driver

`OvrtxCamera` implements `CameraBase` for an explicitly static USD profile
(`static_scene: true`, so its images never verify live robot effects).
`OvrtxRenderer` accepts complete caller-owned scene snapshots and calibrated camera
poses; it does not step physics, read joints or connect to Isaac/Newton. Each render
seals transforms and calibration before the SDK step: depth is metric Z, RGB uses
Cascade's BGR convention, and only centered square pixels are supported. No masks
or attachment authority are fabricated. See [OVRTX](OVRTX_RENDERER.md).

### Optional bridge timing

The Isaac bridge has disabled-by-default Python profiling zones around its target,
update, frame-history, camera and execution-queue operations; profiler output never
authorizes motion or renews camera freshness, and profiler errors leave the
original operation intact. Enable zones with `CASCADE_ISAAC_PYTHON_SPANS=1` and
timing summaries with `CASCADE_ISAAC_PYTHON_TIMINGS=1`.
`scripts/isaac_bridge.py --gc-policy freeze-startup-heap` (or
`CASCADE_ISAAC_GC_POLICY`) applies `src/cascade/sim/heap_freeze.py` after startup and
before the first main-loop step; `ping` reports it as `gc_policy`.

### Lazy frame encoding

The bridge's `_LazyFrame` (`scripts/isaac_bridge.py`) gives RGB JPEG and compressed
depth separate locks and caches, so an RGB-only reader never starts depth
compression. `wire()` still returns both with the original encoding and field order
(JPEG quality 90, depth zlib level 3). Codec completion never updates timestamps,
epochs or measured joints.

### Motion safety path

Skills hold a `SafeArm`. Its motion path enforces the configured peak velocity,
joint limits, workspace, table clearance, keep-out zones, perception watchdog, stop
generation and optional occupancy / inter-arm checks; a `SafetyViolation` ends the
stream. `vet_pose()` is the static counterpart for candidate rejection, and planned
routes vet their sampled segments before execution.

- **Camera on top.** Arms with a gripper camera (every reBot profile) set `safety.wrist_camera_up`: a TCP-frame axis that must not point below horizontal (`min_up_z`, -0.1 by default, which admits a straight-down dive). `vet_pose()` and every streamed step enforce it, so grasp selection takes the camera-up twin of a parallel-jaw grasp; an arm already camera-down may only hold or turn back up (`tests/test_wrist_camera_up.py`).
- **[Contact stop](CONTACT_STOP.md).** On profiles with `safety.contact_monitor`, every streamed motion compares measured joints with the commanded trajectory while streaming and settling (lag band + stall rule; on the reBot RS one or two single-motor mechPos reads per 20 ms). A blocked joint stops the stream with torque on and `ContactDetected(SafetyViolation)` in the skill result, MCP result and trace; a contact zone keeps selection, routes and the persistence loops from going straight back until `reset_scene`. `rebot_rs` enforces, `rebot_rs_mb` measures only; calibrate with `scripts/contact_calibrate.py`. Measured offline against a PD plant (CPU tests), unverified on hardware.
- **Isaac physical clock.** Isaac defaults to 30 Hz nominal targets paced by authoritative physical time, preserving every legacy 50 Hz edge and checking intervening subedges (hardware keeps 50 Hz); sampled coverage, not continuous collision certification. Trajectories and settling run on simulation seconds read with q/dq on the simulation thread (`physics_clock`) with no wall-clock fallback; `IsaacArm.validate_simulation_clock()` checks it read-only. `motion_wall_timeout_s` (120 s) bounds each stream/settle and `motion_rpc_timeout_s` (1 s) each RPC (`configs/arms/isaac.yaml`). Optional command evidence: `CASCADE_ISAAC_TARGET_RECEIPTS=1` in the bridge plus `CASCADE_MOTION_EVIDENCE_DIR=<private dir>` in the MCP process; none of it alone proves motion.
- **Occupancy.** An absent or stale optional map is not evidence of free space. Profiles with `occupancy.required` refuse startup when the bridge cannot be probed and refuse motion on missing, stale or failed map observations; ordinary samples also need observed support unless `occupancy.unobserved: free` (the reBot: the camera never sees its own masked volume). A known body-mask fault raises even on a stale map. Payload/recovery modes also need observed geometry, correct source/epoch/contact binding and fresh commits from all required cameras; the 30 mm clearance stays a rejection condition. Contact and release episodes keep narrowly scoped recovery authority after a failure; no other command can borrow that scope (`SafetyHarness.check_contact_episode`) ([nvblox](NVBLOX.md), [release recovery](NVBLOX.md#payload-and-release-contracts)). The reBot requires its map (`rebot_rs` `overrides.occupancy`): the watcher refreshes it with mesh-masked depth every tick, also during motion, and clearance samples include the gripper's own volume (`safety.tool_volume`: wrist camera box and open jaws), with the table band and the target's voxels attributed away ([always-on occupancy](NVBLOX.md#always-on-occupancy-on-the-rebot)).
- **Carry guard.** The [NV carry attachment guard](NVBLOX.md#payload-and-release-contracts) keeps the post-close attachment through lift and transport; loss or unknown evidence is terminal (no automatic open, retry, home or reset). It is armed only by the NV payload barrier and does not cure mechanical slip.

Gripper commands keep stop/generation and pending-episode checks. On the
observed-finger path (`CASCADE_OBSERVED_FINGER_GATE=1`, set by the Spark install),
each close stage has an actual-pose, full-stroke preflight against non-target
observed surfaces (event `observed_finger_close_preflight`); it does not brake an
already submitted command. The approach gate covers the two calibrated finger links,
not the palm, arm or unobserved space. On a validated PhysX clock the finger envelope
(`assets/grasp_geometry/rebot_rs_fingers_physx.json`; rebuild or check with
`scripts/build_physx_finger_geometry.py [--check]`) has forty components per finger
(eight nominal plus sixteen derived from each of the x86 and ARM cookings), bound by
source hashes; it changes the checked envelope, not the colliders, the 0.1 mm opening
interval or the occupancy clearance. An endpoint occlusion veto clips those envelopes
against captured pixel rays and rejects portions behind valid non-target depth at
pregrasp/grasp endpoints and measured pre-close poses; it does not cover invalid
depth, masked (robot or target) pixels, the space between rays or intermediate poses.

### Grasp pipeline

Localization freshness checks the analysed image's age before and after detector or
VLM work (`skills/runtime._analyzed_frame_age_s`: past 5 s it raises
`SlowPerceptionError`); an expired result cannot authorize a grasp or an automatic
home retry. The Isaac startup check prepares the read-only verifier and waits for
newer camera captures before proof starts; its `TruthPoseReader` probe
(`sim/truth.py`) builds `RigidPrim` views with `reset_xform_properties=False` and
`prepare_contact_sensors=False`.

```
localize ─▶ ObjectFix (base-frame OBB; de-biased centre, verified on 2 engines)
   ├─▶ GraspGen-X candidates (ZMQ :5556, learned 6-DoF; gripper passed as a
   │    swept volume -- the arm profile owns `grasp.graspgenx.sweep`)
   ├─▶ or HUG pinches (opt-in `grasp.backend: hug`, ZMQ :5558): human hands
   │    from the RGB-D frame + a query pixel, mapped to parallel-jaw pinches
   │    and scored by CASCADE geometry (HUG has no score; docs/HUG.md)
   └─▶ OBB candidates (optional profiles only)       optional server error → reported OBB fallback
                                                      learned inference retried after cooldown
   stable quality order ▸ outcome-memory re-rank + existing z-nudge
   select_grasp(preserve_order=True): width ▸ IK ▸ harness and observed-scene vetoes
   current→home ▸ home→pregrasp ▸ descent, with shared motion-profile preflight
   actual-pose full-stroke preflight before each close stage ▸ lift ▸ verification
   place_at: aiming compensation and fresh slip authority are separate results

opt-in `grasp.executor: vla` (B49, docs/VLA_EXECUTOR.md), label grasps only:
localize ─▶ (same fix) ─▶ refuse before motion: observed-finger gate | native planner |
   payload map | no policy server | no `vla` extra   (never an analytic fallback)
   open ▸ home (vetted route) ▸ per chunk: obs (fresh frame, measured joints/jaw, prompt)
   ▸ infer within chunk + episode deadline ▸ stop latch / halt check ▸ rows → joint
   targets ▸ harness vet_pose on EVERY target (no exemption) ▸ SafeArm.move_joints per
   waypoint (approve() per sample) + set_gripper ▸ end after close + lift chunk(s)
   ▸ same jaw check + execute()'s verifier; the server's own success flags are not read
```

- **Required learned grasps.** The Spark presenter profile requires real GraspGen-X candidates and checks diffusion inference at startup; a missing server, protocol stub or failed inference raises instead of substituting OBB. The five-second fallback cooldown applies to optional profiles only. `isaac_kitchen_hug` requires real HUG the same way (only HUG receives the RGB-D frame).
- **Bounded feasibility search.** With GGX and the observed-finger gate, infeasible planning may request at most three batches within eight seconds from the same saved scene and frozen prior, with no motion between them (`grasp.graspgenx.feasibility_max_batches: 3`, `feasibility_timeout_s: 8.0` in `configs/arms/isaac_kitchen_gpu.yaml`); transport, malformed data, cancellation and clock errors are terminal. The runtime orders by quality, applies memory, then calls `select_grasp(preserve_order=True)`.
- **Held objects.** A cached offset or unbound legacy position can aim a carried object but cannot prove slip or authorize opening; held-object observations need identified, temporally coherent evidence, the driver's joint convention and the TCP at capture time.
- **Single-hinge jaws** (SO-101) close toward the fixed tip: the profile declares `jaw_fixed_tip_m` and `jaw_close_dir` and the selector displaces the IK target.
- **Reach envelope (B45, opt-in).** On the reBot, top-down grasps reach only r ≤ 0.45 m (0.38 m at a 10 cm grasp height), measured through the real selector and harness by `scripts/reachability_study.py` ([REACH_ENVELOPE.md](REACH_ENVELOPE.md)); leaning away from the base reaches 0.69 m (45°), horizontal 0.75 m. `grasp.angled_approach_tilts_deg` (empty by default) appends tilted candidates behind every top-down one; the `*_reach` profiles (`rebot_rs_reach`, `isaac_reach`, `mock_reach`) set [30, 45, 90] and the measured box (x 0.10..0.55, y ±0.50); every other gate is the parent's. `tests/test_reach_envelope.py` pins profiles, JSON, URDF hash and every other profile golden.
- **Gripper clearance of angled candidates (B73, opt-in).** `grasp.angled_clearance_vet` (on in the `*_reach` profiles) places the convex hulls of the URDF collision meshes from `link3` onward (`assets/grasp_geometry/rebot_rs_clearance_hulls.json`, built by `scripts/build_gripper_clearance_hulls.py`) along each tilted candidate's q_pre → q_grasp approach (`grasping/gripper_clearance.py`, bisected to 1 mm) and refuses it when the hulls come within `width_pad_m / 2` of the table or the target's observed box; failures to load or bind refuse (fail closed). Measured on the B45 grid: all 47 side grasps at 5 cm refused, the other 5539 tilted grasps survive, for any margin 0–15 mm.

### Sim as an instrument, not a stand-in

`sim/mujoco_world.py` keeps ONE `MjModel`/`MjData` per resolved MJCF path
(refcounted registry, one lock). The arm steps it, rendered cameras
(`perception/mujoco_camera.py`, `type: mujoco`) paint from it and the truth channel
(`sim/truth.py`) reads free-body poses from it, so `postcondition: confirmed
(channel: physics)` is a measurement. `demo_scene.py` writes the scene (arm MJCF +
table + the camera profile's `extra_props`) deterministically. Isaac Sim plays the
same role over a TCP bridge (`scripts/isaac_bridge.py`, `sim/bridge_client.py`).
Truth readers disable mutating constructor defaults; slip-authorizing observations
need valid physical handles and same-step articulation/base/clock evidence. MuJoCo
resets restore `qpos0` under the world lock; Isaac resets restore backend state and
require the fresh-camera/map barrier before the next motion. The renderer publishes
only a token bound to recorded state, so an old render cannot become current. The
bridge reads cameras every `--cam-every` iterations (`CASCADE_ISAAC_CAM_EVERY`) and
after 0.5 wall seconds; each frame carries `render_reference` (kept in
`Frame.capture`), and `scripts/isaac_frame_history.py` binds render time to the
completed physics snapshot.

### Memory

| store | what | horizon | consumer |
|---|---|---|---|
| `BeliefStore` (`memory/beliefs.py`) | objects: label, colour, 3D, freshness; visible/remembered | persisted across runs (wall-clock stamps, `LOADED_MIN_AGE_S` floor, 6 h max age) | every skill; can inform the agent, can never aim the jaws (`belief_fallback_age_s`) |
| `BeliefStore` snapshots (`SceneSnapshot`) | ADVISORY layouts of confirmed objects (label, colour, centroid, extent) | saved with the beliefs; dropped by `clear()` (`reset_scene`) | `snapshot_scene` writes, `restore_scene` reads them as targets; only the `restored` postcondition is a claim |
| `EpisodicMemory` text ring | events, outcomes | ~15 s | `recall_memory`, narration |
| `EpisodicMemory` frame ring | AFTER frame + action + verdict per motion skill | task-scale (600 s), reset per task / by `reset_scene` | `memory_frames(k)` → LLM turn (images) and the `task_memory` tool |
| `EpisodicMemory` visual index (**opt-in**, `memory.embedder`) | one vector per motion frame and per localized object crop | task-scale, ≤ `max_visual` (512) | `recall_visual`; `recall_memory(query)` `looks_like` hits only with a joint image-text embedder; never a current observation |
| `EpisodicMemory` watcher crops (**opt-in**, `memory.visual_recall_detections`, B43) | one crop per committed WorldWatcher belief per frame, ≥ `visual_recall_interval_s` (30 s) apart, ≤ `visual_recall_max_per_tick` (2) | own ring ≤ `visual_recall_max_detections` (128) | ranked with the session index; no crop comes from a motion frame |
| `EpisodicMemory` restored entries (**opt-in**, `memory.persist_episodic` / `CASCADE_EPISODIC`, `CASCADE_EPISODIC_PATH`, B43) | the previous session's visual index (float16 vectors) | `runs/episodic.json`; `episodic_max_age_s` (6 h) applied before the 2 s floor; another embedder's file refused | hits carry `restored: true, state: "remembered"`; never aim motion |
| `ExperienceMemory` (`agent/reflex.py`) | command → plan habits (hashed BoW, TurboQuant index) + Task-Specific Memory **recipes** (`memory/recipes.py`: coordinates replaced by `localize_object(label)+offset`) | `runs/experience.json`, `runs/recipes.jsonl` | tier 2, only for the record's clause structure; a recipe is re-grounded before motion, a failed grounding goes to the LLM tier |
| `ActionObjectMemory` (`memory/consolidation.py`, **opt-in**, `memory.action_objects`) | wins/losses per (motion skill, normalized object label), one credit per executed call | `runs/action_objects.json` | advisory LLM-tier digest |
| `ProgramLibrary` (`memory/programs.py`, opt-in) | programs keyed by a structural sha256; `occurrences`, `source_tasks`, `origins`, `losses` | `runs/programs.jsonl` (`memory.programs_path`, `CASCADE_PROGRAMS_PATH`) | tier 2.5 and MCP `list_programs` / `run_program`, **promoted** records only (≥ `agent.program_min_tasks` = 2 tasks); writes re-read the store under an advisory lock |
| `GraspOutcomeMemory` | per-object grasp features, wins/losses | `~/.cascade/grasp_memory.json` | grasp re-rank + z-nudge |
| `OperatingEnvelope` (`memory/envelope.py`) | per-skill outcome statistics, failure classes and runtime-measured `DERIVED_FEATURES` (unmeasured → `missing`) | `~/.cascade/envelope.json` (`CASCADE_ENVELOPE_PATH`) | planner context; advisory |

**Skill library.** `skills/library.py` stores markdown guidance. Between sessions
`agent/aspire.py` admits a failed-then-successful retry only when skill, goal
parameters, resolved arm and pre-call held-object context match, the postcondition
is measured and confirmed, and no reset/task-end intervenes.
`scripts/learn_from_runs.py` harvests notes into `skills_library/*.md`, one per
`(skill, signature)` counting `occurrences` and `source_tasks`; `retrieve()` loads
keyword-matched notes into tier 3 **only once promoted** (≥ 2 distinct tasks,
ASPIRE's rule; `memory.skill_min_tasks: 1` relaxes it). Details:
[retry evidence admission](DREAM_RSI_ADAPTATION.md).

**Learned notes on the kitchen route.** A runtime reads its note library once
(`skills/notes.py`; the kitchen reads `${CASCADE_ASPIRE_LIBRARY:-~/.cascade/aspire/sim/library}`)
and records its `library_sha` and `block_sha` (the rendered prompt block's sha256) in
`run.json` beside `trace.jsonl`. A Spark launch first freezes the library: it copies the
note files into `<launch state>/aspire-library` and points `CASCADE_ASPIRE_LIBRARY` there,
so every robot server of the launch and `LEARNED.md` read the same bytes (a failed copy
points it at an empty directory: notes off, one warning line). Then
`skills/learned_block.py` writes `LEARNED.md` into the kitchen workspace: one header
line with both shas, then the top-2 note block (at most 2000 bytes). The kitchen prompt
plugin injects it right after `AGENTS.md`; no `AGENTS.md` is edited. After the proof the
launcher still compares it with the proof robot server's `run.json` and removes it on any
difference (a warning line). `robot_knowledge`, the seventh attendee tool, lists every
served note.

**Memory embedder (opt-in).** `memory.embedder.backend` is `none` in the shipped
profile. `hash` is deterministic and dependency-free (separate image and text
spaces; notes match at cosine ≥ `text_floor` OR a guard-word overlap). `siglip` /
`clip` load a real model through the `memory-embed` extra with `local_files_only`;
a requested backend that cannot load fails `build_runtime` with
`EmbedderUnavailable` before any hardware, never falling back to `hash`. Faults
during a run are counted (`visual_stats`). Nothing here confirms an outcome or gates
motion. Watcher crops run on the watcher thread after the commit, outside the pause
lock; `persist_episodic` saves in `shutdown_runtime` (its own `episodic` stage) and
restores in `build_runtime` with the belief store's rules (atomic writes, corrupt
file ignored). Bad limits fail the build; a switch without an embedder is reported
and ignored. A global `CASCADE_EPISODIC_PATH` is refused for two manipulation
robots; composed manipulation domains get `stores/episodic.json`.

### Lifecycle and task receipts

1. `shutdown_runtime(runtime, arm)` returns a software receipt (`scope: software_teardown`, `physical_rest_verified: false`); a failed stage or a live owned thread makes `ok` false. MCP saves it as `teardown.json` in its run directory.
2. `SkillRuntime` keeps a task ledger for every skill in `agent/effects.POSTCONDITIONS`: an unknown effect blocks `task_done(success=true)` and stays in `TaskReport.unverified`. `begin_task()` is a host API, not a tool.

### Evaluation

`eval/progress_judge.py` is a Robo-Dopamine-style progress judge: BEFORE/AFTER
keyframes (plus an optional goal image) → `<score>±NN%</score>` from a GRM or any
OpenAI-compatible VLM. It runs **off the hot path** (`scripts/judge_run.py` over a
finished run dir) and is calibrated against the physics postcondition per step
(confusion matrix in the run summary). The prompt's two WRIST slots take the rig's
wrist keyframes (`keyframe_{before,after}_wrists`) when the trace has them, else the
front image repeats; every record names which (`wrist_slots`).

The launcher can judge its own proof turn (B44, opt-in): `scripts/launch.sh --judge
fake|vlm|grm` (or `CASCADE_JUDGE`; default off) hands the receipt to
`scripts/judge_proof.py` (`eval/proof_judge.py`), which reads `proof.json` read-only,
skips an unverified receipt, runs `judge_run.py --skills pick_and_place` in its own
process group under `CASCADE_JUDGE_TIMEOUT_S` (default 180 s, max 1800) and merges the
confusion matrix into `<evidence_dir>/run-summary.json` plus ONE `judge:` banner line
(`fn>0` = the pictures missed physics-confirmed progress). READY, the exit status and
`proof.json` never depend on it; any failure reads `unavailable`. Opt-in
`eval.judge.score_followup_tokens: N` (B66; default 0) gives an answer without a
`<score>` ONE follow-up call within N tokens; `response_metadata` records `score_via`
(first / follow-up / none) and the `abstention` reason.

## Module map

```
src/cascade/
├── types.py / config.py / device.py   Frame, Detection, ObjectFix, Grasp, SkillError; YAML profiles →
│                       one Cfg (`extends:`, arm `overrides:`, ${repo}/${assets}, CASCADE_BOOTH overlay,
│                       endpoint env vars applied last); resolve_device(): CUDA/ROCm → MPS → CPU
├── perception/
│   ├── camera_base.py        CameraBase + make_camera(); Frames carry METRIC depth
│   ├── realsense / opencv / mock / mujoco / isaac / ovrtx _camera.py   one backend per camera type
│   ├── depth_provider.py     sensor → mono plugin → table-plane ray-cast
│   ├── detector.py / brain_detector.py / vlm_detector.py / vlm_ground.py / segmenter.py
│   │                         MockDetector; brain names + boxes, SAM masks (default); VLM grounder; SAM
│   ├── workspace.py / link_mask.py / robot_mask.py   base cylinder + self-mask gate; link-geometry
│   │                         self-mask; arm body out of depth
│   ├── grounding.py / calibration.py / colors.py   localize (de-biased OBB centre); Kabsch fit; HSV colour
│   ├── stream.py / world.py  CameraStream + CameraRig / WorldWatcher (always-on fusion)
│   ├── drift_monitor.py      opt-in extrinsic drift monitor (per-camera fusion off on drift)
│   ├── occupancy.py / occupancy_backends.py   map client + clearance gate; nvblox | warp | voxel
│   └── probe.py / pixel_target.py / visual_interface.py / visual_diff.py / reference.py
│                             cursor, pixel→object, annotated view, before/after diff, goal images
├── sensing/            typed passive sensors, SensorHub, providers; alignment.py (B50, B72)
├── memory/             beliefs.py (object permanence, save/load) · episodic.py (text + frame rings,
│                       opt-in visual index) · embedder.py · consolidation.py · envelope.py · recipes.py ·
│                       programs.py (ProgramLibrary, tier 2.5) · grasp_memory.py · turboquant.py / vector_index.py
├── control/
│   ├── arm_base.py / arm_rig.py / lazy_arm.py   six abstract methods + min-jerk stream_to(); N named
│   │                   arms; motors untouched until the first motion
│   ├── kinematics.py / usd_model.py   Pinocchio FK/IK (DLS + restarts); USD-physics → URDF
│   ├── mock_arm.py / mujoco_arm.py / isaac_arm.py   kinematic sim; any MJCF (mjc | warp, viewer
│   │                   guarded by a display probe); Isaac articulation over the TCP bridge
│   ├── simulation_motion.py / motion_profile.py   physical clock + shared safety edges
│   ├── feetech_arm.py / rebot_rs_arm.py / rebot_rs_mb_arm.py / robstride.py   SO-101 serial
│   │                   (UNVERIFIED on hw); reBot over CAN / MotorBridge; RS helpers, squeeze cap (B38)
│   ├── ros2_arm.py / unitree_arm.py   ANY ros2_control robot; Unitree SDK arms
│   └── mobile_base.py / mobile_rig.py / mock_base.py / isaac_base.py / h2_policy_contract.py
│                       MobileBase ABC, named bases, simulator-owned controllers (microduck | h2)
├── safety/             harness.py (SafetyHarness + SafeArm) · trajectory.py (route validation) ·
│                       geometry.py (inter-arm distances) · base_harness.py (SafeBase)
├── calibration/        hand-eye: handeye.py (A X B = Z), markerless.py (ICP of the arm's meshes),
│                       session.py (sweep via SafeArm.move_planned), cli.py -- docs/HANDEYE_CALIBRATION.md
├── grasping/           obb_grasp.py · graspgenx_backend.py (ZMQ + fallback) · hug_backend.py ·
│                       selector.py (order ▸ width ▸ IK ▸ harness) · observed_scene.py (finger veto) ·
│                       gripper_clearance.py (B73) · vla_client.py / vla_executor.py (B49) · force.py
├── agent/
│   ├── orchestrator.py reflex → habit → (opt-in) program → LLM loop; memory harness; TaskReport
│   ├── reflex.py       tier-1 grammar + tier-2 ExperienceMemory
│   ├── programs.py     tier 2.5 (opt-in): contract, runner, authoring, distillation
│   ├── effects.py / base_effects.py   postconditions for arm and mobile skills
│   ├── milestones.py   checkable milestones + advisory pre-motion plausibility critic
│   ├── llm.py / cosmos3.py   OpenAI-compat, Anthropic, Codex CLI, Cosmos3 XML dialect, Mock
│   └── prompts.py / advisor.py / aspire.py / trace.py   persona, VLM critic, skill notes, trace
├── skills/
│   ├── runtime.py      SkillRuntime: 42 skills + task_done, TOOL_SPECS, _MOTION_SKILLS (24)
│   ├── contact_episode.py / release_episode.py / held_observation.py   scoped recovery authority
│   ├── rest_pose.py / mobile_runtime.py / library.py   move_to_rest; MobileSkillRuntime; notes
├── spatial/            frames.py / memory.py / grid.py · navigation.py (`go_to`, CPU tests only) · cuvslam.py
├── robotics/           runtime.py / contracts.py / embodiment.py (RobotRuntime) · whole_body.py (B30) · endpoint.py
├── sim/                mujoco_world.py · demo_scene.py · truth.py · mujoco_rgbd.py · isaac_reset.py ·
│                       mobile_bridge.py / h2_physx.py / h2_stepper.py · ovrtx_renderer.py · bridge_client.py
├── eval/progress_judge.py   Robo-Dopamine progress judge (GRM / VLM), off the hot path
└── apps/
    ├── demo.py         build_runtime() = the composition root; CLI --task / --interactive
    ├── mcp_server.py   MCP front-end: 52 tools (2 only with the programs tier), out-of-band stop,
    │                   read-only lane (B46); stdio, or Streamable HTTP (`--http`, bearer + TLS)
    ├── capabilities.py / mcp_env.py   capability matrix; the one registration-env list
    ├── mobile_runtime.py / robot_runtime.py / process_owner.py   `--base` / `--robot` runtimes
    └── stream_server.py / wrist_narration.py / live_view.py / live_control.py / record.py / viewer.py
```

Sidecars run in their own process and venv over ZMQ: `scripts/serve_graspgenx.sh`
(learned grasps, CUDA) or `serve_graspgenx_stub.py` (protocol double, any host), and
`scripts/serve_occupancy.sh` → `serve_occupancy_bridge.py` (`--backend auto`: nvblox >
warp > voxel). Both are **probed at startup** and named in the banner; a missing
sidecar degrades loudly to its fallback. The learned GraspGen-X server runs one
synthetic inference before binding its port (the first CUDA inference takes ~15 s,
the client waits 8 s), so an open port means a warm model unless `health` says
`warmed_up: false`; a failed warm-up is logged and the server binds cold. Both
scripts take `--port 0` (announced in one `CASCADE_SERVER_READY {"endpoint", "port",
"pid", "instance"}` stdout line) and `--instance-id TOKEN` (echoed in `health` /
`probe`); `tests/owned_server.py` accepts a test server only when that line and one
round trip name the started process and token, dead-server tests hold their port
(`held_dead_port()`), and a fake that needs its port up front is handed the bound
socket (`handed_over_port`, SCM_RIGHTS), so parallel suites never answer each other
(B64, B70).

## ROS2, humanoids, and what is NOT here yet

Every claim below names its evidence: *measured* (a run with numbers, linked), *CPU
tests* / *mock-only* (software evidence), *simulation only* (Isaac Sim / Newton /
PhysX episodes, no robot), or *unverified on hardware* (written against a real
interface, never run on a robot). `tests/test_docs_architecture_current.py` fails
when the "Still not here" list names a capability whose module exists, when a
present capability's module is not cited here, or when a cited path is gone. Open
work is in the [capability status](ROADMAP.md) rows "Locomotion", "Mapping and
navigation" and "Whole-body humanoids".

**ROS2 today = arms.** `type: ros2` (`control/ros2_arm.py`) speaks the two
interfaces every `ros2_control` deployment has -- `sensor_msgs/JointState` in,
`trajectory_msgs/JointTrajectory` (or `Float64MultiArray` for a forward position
controller) out -- with joints addressed **by name**, so the driver's `JointState`
order can never shift the mapping. Adding a ROS2 robot is copying
`configs/arms/ros2_generic.yaml` (a `template: true` file the factory refuses to run
until its numbers are filled in) and pointing it at the robot's URDF, joint names,
keyframes, gripper travel and workspace. `rclpy` is imported inside `connect()`, so
the mock stack and CI without ROS2 still collect the module. Shipped profiles:
`so101_ros2`, `piper`, `h1`, `h1_2`, `fr3`. **Unverified on hardware.**

**Humanoid arms on a standing robot.** `type: unitree_arm`
(`control/unitree_arm.py`) drives an arm of a G1 / H1 / H1-2 over Unitree's Arm-SDK
channel (`rt/arm_sdk` LowCmd with the per-family motor index table and CRC;
`rt/lowstate` in), ramping the SDK "weight" so the locomotion controller hands the
arm over without a jerk. Balance, legs, waist and walking stay with Unitree's
controller; a handless gen-1 H1 declares `max_width_m: 0` so grasps are refused, not
mimed. `base_pose` places the shoulder in the shared table frame. **Unverified on
hardware.**

**Mobile bases (velocity level).** `control/mobile_base.py` defines `MobileBase`:
immutable `BaseState` snapshots of completed physics steps, finite body-frame
`VelocityCommand`s admitted against a generation fence, and separate stop /
`reset_stop` channels. `MobileRig` (`control/mobile_rig.py`) names the bases;
`SafeBase` (`safety/base_harness.py`) enforces limits, freshness, cancellation and
progress; `MobileSkillRuntime` (`skills/mobile_runtime.py`) serves `walk_velocity`,
`walk_distance`, `turn`, `stop_navigation`, `get_base_state` and the stop tools; and
`agent/base_effects.py` judges every command on a state channel the actuator does
not own (the mock base has none, so its motions stay `unverified`).
`apps/mobile_runtime.py::make_base` builds exactly two backends: `mock`
(`control/mock_base.py`) and `isaac` (`control/isaac_base.py`, a bridge client of a
simulator-owned controller, kinds `microduck` and `h2`). A base runs alone with
`--base <profile>` / `CASCADE_BASE` or as the `locomotion` domain of a `--robot`
profile; profiles live in `configs/bases/`. Evidence: CPU tests
(`tests/test_mobile_*.py`) plus the simulation episodes below; no base has moved a
robot.

**Distance baselines (B67, B74).** `walk_distance` credits travel and integrates
lateral/heading drift only from a baseline: SafeBase's first completed post-ACK
sample and the checker's first admitted independent sample (at most
`max_sample_gap_s` after the ACK's admission clock). The opt-in pair
`max_admission_lateral_m` / `max_admission_heading_rad`, set together in a base
profile's `distance_control` and `verifier`, also bounds the change that baseline
would absorb: SafeBase vetoes and latches on it, and the checker refutes a
confirmation on it ([contract](MICRODUCK.md#implemented-contract)). CPU tests only;
no shipped profile sets the pair.

**Legged robots on that layer: MicroDuck and the Unitree H2 (simulation only).**
MicroDuck ([MICRODUCK.md](MICRODUCK.md)) runs its official ONNX policy with the
pinned native BAM on Newton (`scripts/isaac_microduck_bridge.py`); measured native
episodes (standing and stop checks, one small negative turn) sit beside retained
failures (composed reverse motion, larger and positive turns), so its locomotion is
a candidate, not admitted. The Unitree H2 candidate ([HUMANOID_H2.md](HUMANOID_H2.md))
binds NVIDIA's public H2 USD and `Velocity-H2-History-v0` policy through
`control/h2_policy_contract.py` (31 joints, 14 commanded; arms, head and waist yaw
held) and a PhysX owner (`sim/h2_physx.py`, `sim/h2_stepper.py`,
`scripts/isaac_h2_bridge.py`) on the internal Isaac Sim 6.2 build. Measured there
through `SafeBase` and the independent verifier: first episodes 3 of 5
`walk_velocity` commands confirmed; candidate revision 2 confirmed 6 of 7; the
revision-3 turn ramp took 0.8 rad turns to 8/8 (1/6 without) but 1.0 rad turns to
1/5 (5/6 without); revision 4 has CPU tests only. Binding gate passed in simulation;
no physical admission and no hardware binding.

**Arm mounted on a base (B30, mock-only).** `robotics/whole_body.py` and an opt-in
`whole_body:` profile block (`configs/robots/mobile_manipulator_mock.yaml`) compose a
locomotion base and an arm `mounted_on` it: disjoint command endpoints, a
capture-time `world ← base ← arm_base` frame chain (a stale or missing base pose
refuses arm motion), `exclusive` coordination by default and per-domain
`reset_stop(domain=...)`
([contract](ROBOT_MODULARITY.md#multi-domain-embodiments-mounted-arms)). Measured by
43 CPU tests (`tests/test_whole_body_domains.py`) on the kinematic mock arm and base
only; a composition with any physical actuating resource is refused. It is a
composition contract, not a whole-body controller.

**Localization: cuVSLAM (optional, uncalibrated).** `spatial/cuvslam.py`
(`CuVslamSpatialDomain`; the SDK runs in an owned worker process,
`spatial/cuvslam_worker.py`) estimates the optical-camera pose from retained RGB-D
captures through `warmup_localization` / `track_capture` / `get_localization`; it
never acquires a frame or moves anything, and cuVSLAM is not installed by default.
`spatial/cuvslam_uncertainty.py` keeps the SDK's odometry covariance apart from the
SLAM pose. Measured natively ([record](SPATIAL_PROVIDERS.md#live-native-rgb-d-stream)):
a 12-frame synthetic replay (final translation error 0.01178 m), a live stream that
tracked 12 real captures from a 320-solve Isaac/Newton MicroDuck producer under zero
policy commands with `emergency_stop` invalidating the map epoch, and 12 of 12
odometry covariance matrices through the worker IPC. Pose uncertainty has no
calibrated bound, tracking loss and map-epoch changes are untested natively, and no
physical camera has fed it.

**Route execution: `go_to` (opt-in, CPU tests only).** `spatial/navigation.py`
(`NavigationDomain`) replaces the free-motion tools with `go_to(goal_xy_m, map_epoch,
map_sha256)`: a conservative route on an immutable planar grid (`spatial/grid.py`,
flat ground only), executed as `turn` / `walk_distance` segments through the same
`SafeBase` and verifier, with the whole-robot swept volume
(`spatial/robot_volume.py`) cleared against fresh registered base poses. It is
reachable only through the Python API (`build_mobile_runtime(...,
navigation_source=...)` or `build_robot_runtime(navigation_bindings=...)`) and
refused for whole-body compositions. No shipped provider satisfies its source
contract ([SPATIAL_PROVIDERS.md](SPATIAL_PROVIDERS.md#opt-in-observed-route-execution));
only the fixture in `tests/test_spatial_navigation.py` does, over the mock
controller. Grids come only from the synthetic `spatial_replay` profile
(`spatial/domain.py`).

**Still not here** (checked against `src/cascade`, `scripts/` and `configs/`):

- ROS2 for a base: nothing publishes `geometry_msgs/Twist` on `/cmd_vel`,
  subscribes to `nav_msgs/Odometry` or an `OccupancyGrid`, or calls Nav2;
  there is no `ros2_base`, `unitree_base` (Unitree `LocoClient`) or
  `mujoco_base` backend (`make_base` accepts `mock` and `isaac` only).
- A collision map built from the robot's own sensors for a base: the
  arm-local occupancy service is not sliced into a costmap, RGB-D surface
  annotations are not a collision map, and so no shipped source can feed
  the route runner above.
- A calibrated pose: no localization source has a measured error bound, and
  tracking loss and map-epoch changes are not validated natively.
- The 2026-09-10 design's planner verbs `go_to_pixel`, `go_to_object` and
  `where_am_i`, and a `base=` binding inside the arm runtime's
  `SkillRuntime.execute()`.
- A whole-body controller: balance-aware arm motion, self/environment
  collision for an arm on a moving base, arm physics on the H2 (its arms
  are held) and any physical mobile manipulation.
- Hardware: no base or legged robot has been driven by CASCADE.

## Launch and hosts

**Spark.** Distribution starts at `scripts/bootstrap.sh` → `scripts/install.sh`
(explicit EULA acceptance, Isaac Sim, PhysX CUDA, GraspGen-X, a local
vision-language model and OpenClaw; versions in the
[deployment guide](DGX_SPARK_SETUP.md)). `.venv`, `.isaacsim` and `.graspgenx`
isolate incompatible Python dependencies, `.llama.cpp` runs the local brain and
`cascade-demo` isolates the attendee host profile. The installer replaces no
drivers. Occupancy/nvblox is disabled; JEv, Jev/Kev and Cosmos are not active
presenter backends. The installed Spark defaults to `dt=1/120`, six bridge
iterations per nominal camera poll and the observed-finger gate on; explicit
overrides keep precedence. `--prepare-only` stops after dependencies/assets and
cannot print READY.

**One click.** `run.sh` → `scripts/launch.sh`: `--sim auto|isaac|mujoco|none`,
`--setup` (venv, extras, assets, OpenClaw CLI, provider onboarding), `--check`
(report only, never mutates), `--dry-run`, `--down` (stops its sidecars AND the
per-session MCP servers the gateway never reaps). READY needs the runtime built with
the server's exact env, the tools listed, a brain turn, and a real pick and reset in
the SAME chat session; the proof binds model/session, exact MCP tool names and the
owned live process to its trace, and reset must include the manipulated prop and a
`world_state` read-back. `--no-robot-turn` is STARTED / UNVERIFIED. The MCP server
is registered with `requestTimeoutMs: 300000`. Native visitor turns and proof pick
turns share a 360-second whole-turn budget (60-second host reserve around the
per-tool limit) plus a 30-second CLI return margin; an expired motion still latches
the e-stop (`scripts/native_turn_budget.py`: `NATIVE_TURN_TIMEOUT_S = 360`,
`AGENT_EXIT_GRACE_S = 30`). On macOS the server runs under `mjpython`; a sleeping
display defers the MuJoCo window to the next motion instead of crashing the server.

**Hosts.** OpenClaw (native `mcp.servers`), Hermes (`~/.hermes/config.yaml`),
Claude Code (`.mcp.json`), Claude Desktop and Codex, all via
`scripts/setup_agents.py`. As a host the platform's LLM picks tools and cascade's
tiers are bypassed (`llm=mock` in the server); as a brain (`--llm codex_astra|hermes|
anthropic|openai|local_*`) cascade runs its own loop. Codex can be either:
`--codex-profile NAME` registers the tool server in a `codex -p NAME` layer, while
`codex_astra` runs GPT-6-Astra through one `codex exec --ignore-user-config`
subprocess per step, so that session never sees the user's own MCP servers.

**Registration environment (B63, B68).** A host starts `mcp_server` with the `env`
of its entry, not the registering shell. Every registrar (`launch.sh`,
`setup_agents.py`, `scripts/hermes_demo.sh`, `scripts/openclaw_demo.sh`) reads ONE
list, `cascade.apps.mcp_env`: each `CASCADE_*` variable `src/cascade` reads is
`FORWARDED` (copied verbatim when set) or `NOT_FORWARDED` with a category and reason
(registration-written rig names, rig selectors such as `CASCADE_ROBOT` /
`CASCADE_ARMS`, the per-process `CASCADE_RUN_DIR`, demo-CLI brain settings, launcher
judge settings, HTTP-transport settings, the secret `CASCADE_MCP_TOKEN`). Values the
registrar derives from its own flags win (`--occupancy none`, `--graspgenx none`,
`--headless`, `--env`, `--detect-classes`, `--hide-tools`). The shell registrars run
`python -m cascade.apps.mcp_env KEY=VALUE ...` with `PYTHONPATH=<checkout>/src` before
any host call, so a failure registers nothing. With nothing extra set every host
call is byte-identical (`tests/test_mcp_env_forwarding.py`,
`tests/test_legacy_mcp_registrars.py`).

**Endpoints (B34, B41, B70).** `CASCADE_BRIDGE_PORT`, `CASCADE_GRASPGENX_PORT`,
`CASCADE_OCCUPANCY_PORT`, `CASCADE_HUG_PORT`, `CASCADE_GRASPGENX_HOST` and
`CASCADE_HUG_HOST` are applied by `load_demo_config` after every config layer, to
the top level and each arm's `resolved` view (`config.ENDPOINT_ENV_VARS`; empty =
unset; a port is ASCII digits in 1..65535, a host a hostname or IPv4 address,
anything else raises naming the variable). A GraspGen-X / HUG planner dials its
resolved section (`config.sidecar_endpoint`). The bridge and the occupancy sidecar
have no host variable. `launch.sh --graspgenx local|external` parses
`CASCADE_GRASPGENX_HOST` by the same rule, prints `GraspGen-X endpoint: HOST:PORT`,
probes that host (`external`) and passes `--host` to `check_graspgenx.py`.

| variable | effect |
|---|---|
| `CASCADE_CAMERAS` / `CASCADE_CAMERA` | comma list (first = manipulation camera) / single-camera fallback |
| `CASCADE_ARMS` / `CASCADE_ARM` | comma list (first = manipulation arm, builds the `ArmRig`) / single-arm fallback |
| `CASCADE_PERCEPTION_URL` / `_MODEL` / `_KEY_FILE`, `CASCADE_DETECT_CLASSES` | the brain that names objects; the default detection vocabulary |
| `CASCADE_HIDE_TOOLS`, `CASCADE_MCP_READONLY_LANE` | operator tool override on top of the capability matrix; the read-only lane (1/0) |
| `CASCADE_VIEW`, `CASCADE_MJ_VIEW`, `DISPLAY` | cv2 camera window; MuJoCo physics window (the launcher sets it in sim modes) |
| `CASCADE_PREWARM`, `CASCADE_STREAM`, `CASCADE_STREAM_PORT`, `CASCADE_RUN_DIR` | perception pre-warm; dashboard mode / port (default 8090); trace dir |
| `CASCADE_OCCUPANCY` | `0` skips the bridge probe; a startup error for a profile that requires its map (`rebot_rs`) |
| `CASCADE_BRIDGE_PORT` / `CASCADE_GRASPGENX_PORT` / `CASCADE_OCCUPANCY_PORT` | 8611 / 5556 / 5557; `CASCADE_HUG_PORT` (5558), `CASCADE_GRASPGENX_HOST`, `CASCADE_HUG_HOST` likewise |
| `CASCADE_USD`, `CASCADE_PHYSICS_DEVICE`, `CASCADE_BRIDGE_BIND`, `CASCADE_BRIDGE_NO_TARGETS`, `CASCADE_COMPANION_EXTS` | Isaac bridge side (`scripts/isaac_bridge.py`); `CASCADE_PHYSICS_DEVICE=cpu` is the escape hatch for GPU-PhysX boot NaNs |

**Sandboxed host (B35, opt-in).** An agent inside an NVIDIA OpenShell sandbox
managed by NemoClaw reaches the robot through `mcp_server --http` (Streamable HTTP,
TLS from a private CA, bearer token in OpenShell's provider store); the transport
shares the stdio server's serial worker and receive-side stop channel (`_admit`) and
namespaces JSON-RPC ids per session. `scripts/nemoclaw_mcp.py` issues certificate,
token and registration ([OPENSHELL.md](OPENSHELL.md#nemoclaw-route-opt-in)). The
standalone OpenShell route (`scripts/openshell_rig.py`, [OPENSHELL.md](OPENSHELL.md))
serves the same server on host loopback and is where `launch.sh` always runs the
agent on Linux; macOS keeps stdio. Provider profiles bind the token to the agent's
binary and endpoint and refuse `reset_stop` (or, observe-only, every motion tool) at
L7 inside the sandbox.

**Read-only lane (B46, opt-in).** `mcp.readonly_lane: true` (or
`CASCADE_MCP_READONLY_LANE=1`) lets a chat host look while the robot moves. After
`_admit` (stops, cancels and pings keep first claim), `_enqueue` hands a `tools/call`
for one of `READONLY_LANE_TOOLS` (`world_state`, `robot_knowledge`,
`verify_last_action`, `camera_snapshot`) to a single lane thread when the in-flight
call is a motion (`_MOTION_SKILLS`, or `run_program`); everything else queues for the
serial worker, so motions stay serialized and a host cancel still latches the
e-stop. Lane tools never enter `SkillRuntime.execute()`, never take `_exec_lock`,
never command hardware and touch only state already guarded for concurrent readers
(`camera_snapshot` is a passive stream read with no `observe_fresh`).
`describe_scene` and `get_observation` stay out: they fuse beliefs, run the detector
and read the arm. Results carry `served_during_motion`.
`tests/test_mcp_readonly_lane.py` pins both modes through the real server.

## Key decisions (still load-bearing)

### Data and frames

- **Metric depth in the Frame.** Sensor units differ per camera (L515 0.25 mm/unit vs D4xx 1 mm); convert once at the camera boundary.
- **Grasps planned in the base frame.** Camera pose affects visibility, not grasp geometry; that is what camera-agnostic means operationally.
- **The arm's shape is data.** `n_joints`, limits, keyframes, tool-frame order, gripper travel, jaw datum and reach come from the profile; `_profile_q` *requires* them. Profiles inherit with `extends:` and carry `overrides:` for rig geometry.
- **Memory is structured first, embeddings second.** Recall works on labels, time and positions; frames, not text, show the planner its own past.
- **Device agnosticism is a resolution step.** `device.py` answers "where does this model run" once; torch is not a dependency.

### Execution and authority

- **One backend class, pluggable physics engine.** `engine: mjc | warp` under one `MujocoArm` with a batch-oriented engine surface (C ~4.9 µs/step, Warp on CPU ~3.2 ms/step, so C is the laptop default). The Spark presenter uses Isaac PhysX CUDA; Isaac Newton is an explicit, separately validated option.
- **Feedback, not sleep.** Every backend reports real joint positions; settling checks target error with a per-profile tolerance and timeout, and Isaac also requires stable q over advancing physical steps.
- **Authority depends on evidence.** Limits, stop generation and required geometry fail closed. Optional observations can be unavailable; they never become free-space proof, a required-GGX fallback or authority to open a held object. An unavailable postcondition stays unverified, and results name the channel that answered.
- **The tool surface is derived, not declared.** `apps/capabilities.py` reads the BUILT runtime (each camera's real depth chain via `DepthProvider.depth_source_for`, the sidecar probes behind `runtime.backends()`, the `ArmRig`, verifier and memory) and `TOOL_REQUIREMENTS` names what each tool cannot run without. The MCP server withholds (and rejects) only what probed state shows unmet; unknown is not unavailable, a fallback is reported, the stop path is never a capability, and reasons go to the banner, `/state` and `world_state`.

## Known limitations

### Hardware and grasping

- **Hardware coverage:** L515 streaming and RobStride mechPos reads were exercised read-only on the reference rig; real-arm motion, the SO-101 serial driver, the ROS2 and Unitree backends are unverified on hardware.
- Grip force is a stiffness proxy (kp scaling + stall detection), not a calibrated force loop. On the real reBot RS the pick close (`close_gripper_two_stage`) pushes at a FIXED fraction of travel, so holding torque grows with object width. The arm profile's opt-in `gripper.max_contact_squeeze_rad` (B38) caps every jaw target after the first mechPos-detected contact (`robstride.close_two_stage_capped`, [REBOT_GRIP_SQUEEZE_CAP.md](REBOT_GRIP_SQUEEZE_CAP.md)); it ships `null`, is measured only on a simulated jaw, does not cap the stage-1 scout squeeze, and the hardware protocol in that doc must run before a value is set.
- Isaac pick reliability (B36): the hold after contact applies only when the jaw stall detector (0.5 s within a 0.002 band) sees the stall; PhysX contact chatter of ±0.003 hid it in 1 of 16 live runs, which then failed "did not settle above the place target". It helps only learned GraspGen-X grasps (OBB fallback: 0/3 either way); Newton keeps the plain close; the kitchen profiles are unmeasured and opt out.
- Reach: the default reBot profiles keep the top-down workspace box (x 0.10..0.50, y ±0.30), part of which top-down grasps cannot reach (r > 0.45 m). The `*_reach` envelope (B45) is kinematic only; live picks there are unmeasured. The B73 clearance vet covers the forearm-to-finger hulls against the table and the target's observed box along pregrasp → grasp only: not the home → pregrasp transit, other objects, learned or top-down candidates, or physics.
- `RebotRSArm.disconnect()` cuts torque: park first. `move_to_rest` (`skills/rest_pose.py`) folds onto `park_q` and verifies the measured pose before returning the LazyArm to standby; `move_home` keeps the motors on.

### Perception and fusion

- Same-colour identical objects: a frame is fused as a whole (`BeliefStore.update_frame`), so twins the DETECTOR returns as two detections stay two beliefs (rendered MuJoCo twins resolved from 3.75 cm apart; live Isaac twins 5–9 cm apart were two beliefs in 9/10 runs, `docs/evidence/b31-isaac-same-colour-20261008/`). Still one belief: props returned as ONE detection (touching, overlapping in the image, the mock detector's one blob per colour unless `instances: true`). Two disjoint detections of ONE object with no whole detection become two beliefs. `memory.instance_association: false` is the A/B switch.
- Open-vocabulary detections of the robot: fusion uses the render self-mask when the frame has one (live phantom rate 17.9–19.6 % → 0 %, `docs/evidence/b32-fusion-self-mask-20261008/`); real-rig frames rely on the base cylinder unless `workspace_filter.link_self_mask` is on, which is measured only on CPU against the URDF meshes and depends on hand-eye calibration and joint offsets. A stale joint state means no mask.
- Colour names differ between cameras (B32b, B40): a camera that flips its own name still makes a second belief; small objects' views may overlap below the 0.75 IoU gate; the size gate only refuses with real-mask clouds on both sides, and a prop's view fusing into the CONTAINER's belief is not handled. `memory.size_gate: false` and `memory.per_camera_colour: false` restore the earlier stores byte for byte (`docs/evidence/b40-fusion-size-gate-20261009/`).
- IMU / proprioception alignment (B50, B72) is software only and pairs, never fuses; without `producer_history` the hub's admitted captures (at most 32 / 16 MB shared, a large RGB-D capture can evict IMU samples) bound the skew. The rate × skew test is a first-order estimate; no hardware clock sync; the off-GIL `--state-reader process` refuses `state_history`; no estimator consumes the pairings yet.
- Wrist narration (B47) is measured only on the mock stack; the physical D435i wrist camera has no profile mapping or hand-eye calibration (`isaac_wrist.yaml` is the simulated one). The dashboard polls `/state` every 0.5 s, so a shorter motion may never show as `now:`; only the first wrist stream is captioned.

### Memory, learning and judging

- Skill-library notes are retrieved by guard-word match unless `memory.embedder` is set; the SigLIP/CLIP text floors (`text_floor` 0.85, CLIP `text_image_floor` 0.25) are NOT calibrated with real weights. The same caveat applies to program ranking (B42).
- Visual recall indexes motion frames and localized crops by default; watcher crops and persistence (B43) are opt-in and CPU-measured only. No semantic recall quality is measured with real weights. Action-object consolidation keys on the normalized label, so a label flicker (bottle/toy) stays two objects.
- The VLA executor (`grasp.executor: vla`, B49) is measured only against its protocol stub: no real policy has run, chunks execute as quasi-static harness-gated moves, the observation is the manipulation camera only, there is no grasp exemption, and the route refuses rigs with the observed-finger gate, a native motion planner or a payload map. A policy server started after a failed probe needs a runtime restart.
- Tier-2 clauses (B37, B69): still one clause when the second clause has no verb of its own, an adverb other than "also"/"también" precedes the verb, the verb is outside the vocabulary, or "that"/"this" follows the verb as a determiner. A noun that is also a clause verb can over-cut prose (sending the task to the curriculum or the LLM tier). The programs tier is outside the clause rule: a one-clause program reused for a sequence would report the same half-command success the LLM tier's `task_done` could.
- The launcher judge (B44) ran once live (`docs/evidence/w6-live-20261010/`): `--judge vlm` with the local Qwen over a launcher-style Isaac proof turn gave `tp=1 tn=0 fp=0 fn=0` with `proof.json` unchanged; no full `./run.sh` launch with the OpenClaw brain has judged its proof turn. Offline the local Qwen3.8-27B judged 72 recorded live Isaac picks (`docs/evidence/b44-live-qwen-judge-20261009/`): at 1536 tokens `tp=40 tn=1 fp=15 fn=0` with 12 unscored, re-judged at 8192 tokens `tp=40 tn=7 fp=19 fn=0`. So `fn` is trustworthy here but `hop > 0` is not a failure detector. The B66 follow-up is CPU-tested only; a local judge needs `CASCADE_JUDGE_CONFIG`.

### Servers, endpoints and deployment

- The MCP server executes one call at a time; stops, cancels and pings are handled out of band. With the opt-in read-only lane (B46) the four lane tools answer during a motion; a `run_program` call is one motion for its whole program. Measured once live on Isaac over 4 lane-on / 4 lane-off pink-cube picks (`docs/evidence/w6-live-20261010/`): lane reads 0.001–0.003 s (`camera_snapshot` 0.024–0.038 s) during the pick, picks confirmed 2/4 vs 1/4, the stop answered in 0.032 s. No real-rig latency yet.
- Endpoint overrides are CPU-tested only (no live Isaac run on private ports); the remote GraspGen-X host check is measured only against a second loopback address (those tests skip on macOS). `local` and `stub` still start their server here whatever `CASCADE_GRASPGENX_HOST` says. `scripts/demo_proof.py` reads `CASCADE_BRIDGE_PORT` with its own `int()`.
- Test-server ownership (B64, B70) covers the GraspGen-X stub and occupancy fixtures; `tests/test_endpoint_port_followups.py` fails on any new bind-0 / release / use site and lists four older ones (three conversation-provider tests by design, and `tests/test_spark_install.py::launch_fixture`). A held "dead" port is refused at once on Linux but times out on macOS runners.
- Registration environment (B63) ran once live (a server started from launch.sh's registration env ran the VLA executor against the live Isaac bridge); no real MCP host has started such an entry yet. The guard sees only `CASCADE_*` names written as whole string constants (plus the `CASCADE_MICRODUCK_*` family). The Brev container (`deploy/runtime/runtime.py`) builds its own pinned MCP env. The launcher's runtime check inherits the whole shell, so an exported selector changes the check, not the registered server.
- The split conversation deployment (B51, B71) is measured on one host only: no two-host run, no mutual TLS, no token rotation or certificate revocation, one full handshake per TLS call. The robot's deadline is late by the one-way request latency; with the robot service unreachable `/api/status` fails (HTTP 500), `/readyz` says why, and the lease expiry (default 3 s) is the stop. A lost lease is terminal for that conversation process.

### Platform and mobility

- macOS: the server cannot open the cv2 camera window (`RigViewer`: Cocoa needs the main thread); use the MuJoCo window and the dashboard.
- The `walk_distance` admission-drift pair (B74) is CPU-tested only and unset in every shipped base profile; forward delivery motion is excluded from progress, never bounded.

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

Today: 42 skills + task_done (43 specs), 24 in `_MOTION_SKILLS`, 52 tools in the
full MCP catalog (50 with the programs tier off). `tests/test_llm_and_library.py`
pins the README's headline counts to these numbers. A running server lists the
catalog minus what the rig's capability matrix withholds
(`world_state.tools_withheld`), so compare `mcp probe --json` against that, not
against the full catalog.
