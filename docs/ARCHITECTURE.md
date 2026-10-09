# Architecture

The [current architecture diagram and capability matrix](ROBOT_MODULARITY.md)
show the opt-in `RobotRuntime`: manipulation, locomotion, fastening, passive
sensing and read-only spatial domains behind explicit resource contracts.
MCP and bounded skill graphs dispatch registered tools; the optional
[conversation supervisor](CONVERSATION.md) adds speech without owning joints.
The existing arm/mobile entrypoints remain available. `FleetRuntime` coordinates
separate robot runtimes. A [native MicroDuck foundation](MICRODUCK.md) has run
1, 2 and 12 robots with zero velocity commands in one scene; commanded fleet
tasks and a shared whole-body controller remain unvalidated.

The architecture targets robot and backend interfaces, not a particular build.
Isaac Sim, MuJoCo and hardware drivers are adapter implementations. Robot
profiles select capabilities, policies, sensors and limits; deployment guides
and validation records carry the concrete dependency versions and source pins.

[Embodiment declarations](EMBODIMENT.md) describe links, joints, transmissions
and sensor attachments. [Measured frame trees](SPATIAL_PROVIDERS.md) separately
bind transforms to capture time, map and epoch. Neither substitutes for the
other. Mixed physical actuation remains refused until dynamic frames, shared
control and robot-specific limits are validated. An opt-in `whole_body`
profile contract composes a mock arm mounted on a mock base (disjoint command
endpoints, capture-time mount frames, coordination policy, per-domain reset;
[contract](ROBOT_MODULARITY.md#multi-domain-embodiments-mounted-arms)); it is
software evidence and refuses every physical mounted composition.

The optional [spatial domain](SPATIAL_PROVIDERS.md) resolves source-bound
capture transforms and landmark memory and plans on immutable planar maps.
Its synthetic replay uses the same composed MCP route without actuator resources;
the [RGB-D observation domain](RGBD_SPATIAL_OBSERVATIONS.md) adds retained surface
annotations. The optional cuVSLAM provider estimates local RGB-D poses in an
isolated process with capture, calibration and map-epoch checks. Native
localization validation, collision-map construction and route execution remain
pending.

The optional [MicroDuck path](MICRODUCK.md) selects `MobileRig`, `SafeBase` and
`MobileSkillRuntime` before arm construction. It shares CASCADE's MCP, traces
and episodic memory while exposing mobile capabilities. Its Newton bridge
runs the policy on physical steps; independently sampled pose and solved
support determine outcomes. Canonical model identity binds state, frames and
verifier profiles to the effective recipe. Historical short-distance episodes
have bounded positive results; general gait, longer paths and later source/model
compositions require their own validation. The policy advances on completed
physics solves, with no LLM call in the control loop.

See the [capability and acceptance index](PROJECT_STATUS_20261003.md) for merged
changes, software validation and physical runs, and the
[integration report](MANIPULATION_ASSEMBLY_20261002.md) for earlier experiments.
Counts are derived at the end of this document; dated benchmark measurements
retain their original scope.
Read this after the README and before `AGENTS.md`.

The optional [cuMotion planner](CUMOTION.md) retains its candidate API/CLI and
adds `planning/runtime.py` at the arm composition boundary. `isaac_cumotion`
binds the model, joint ordering/signs, base and tool to the selected arm.
`SafeArm` approves the native trajectory; the Isaac executor preserves its
shape while uniformly slowing it and checking its sampled edges. Contact
segments require a linear TCP corridor. A bounded post-close observation
window precedes lift planning, while the final 1 mrad start-drift veto remains
unchanged. Shutdown skips park/open commands when load or contact state remains.

The [OVRTX bridge producer](OVRTX_RENDERER.md) reads completed physics tensors
and sends immutable snapshots to an isolated renderer. Only that worker edits
its private USD stage: editing another live `UsdStage` inside Kit can emit global
PhysX notices and invalidate the simulator. RGB-D, semantic masks and captured
joint/contact metadata share an epoch, physical step and snapshot digest.

The separate [Factory assembly controller](FACTORY_THREAD_CONTACT.md) drives
SO-101 joints and a motorized hex socket in Newton; the nut moves through solved
thread/tool contacts. Read-only verifiers measure advancement, actual seating
contact and zero-motor retention. This experiment does not change the ordinary
`turn_screw` skill into an autonomous tool-acquisition or preload controller.

## Design position

CASCADE uses **curated domain tools** between an LLM or human and robot
controllers. Motion is gated and traced; independent observations determine
whether its physical postcondition is confirmed, refuted or unverified.
Missing evidence remains unverified. The manipulation stack's research
background includes three published systems:

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
  world model first, VLM only when the symbolic tier abstains). Since
  2026-10-07 the same rate-limited critic also runs *before* each LLM-tier
  motion dispatch (Human-CLAW's pre-execution verifier, ROADMAP #6):
  `PlausibilityChecker` asks the VLM whether this call with these args is
  plausible given the frame, beliefs and held state; the answer is attached
  to the result/trace as `plausibility` and is **advisory only** -- the
  harness alone refuses motion.
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

The deterministic skill stack is the default executor: it is debuggable,
safety-gateable and runs offline (ROADMAP records the original decision not
to put a VLA policy in the loop, and the LIBERO layer-attribution numbers
that back it). Since 2026-10-09 (B49) a language-conditioned policy can
serve label grasps as an OPT-IN second executor, `grasp.executor: vla`
([VLA executor](VLA_EXECUTOR.md)): `grasp_object` keeps its name, schema,
localization and verifier; only the motion between "jaws open at home" and
"lifted" comes from the policy's action chunks, and every chunk passes the
SafetyHarness before and during motion.

## Manipulation runtime detail

This is the arm-specific path retained by the composed runtime. Mobile,
fastening, sensing and spatial domains use their own implementations; their
shared boundary is `RobotRuntime.execute()`, as shown in the
[modular diagram](ROBOT_MODULARITY.md).

```
                 chat host (OpenClaw / Hermes / Claude Code / Codex)          CLI / REPL
                 host LLM picks tools over MCP stdio                         --task / --interactive
                          │                                                          │
                          ▼                                                          ▼
              apps/mcp_server.py  ── 47 tools ──┐                  agent/orchestrator.py
              (38 specs − task_done              │                  tier 1 REFLEX   regex grammar      ~µs
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
one composition root across brains and tiers. Backend capabilities remain
explicit: physical clocks, captured joint state, collision geometry and payload
recovery are validated for the selected arm rather than assumed interchangeable.

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

- **Tier 2 respects the clause structure (B37, 2026-10-09).** The hashed
  bag-of-words vector cannot see clause boundaries or their order: "pick up
  the red cube and then pick up the blue cube" scored 0.901 against the habit
  "pick up the red cube", and because `FastPlanner` consults experience on the
  whole task before the curriculum split, the fast tier ran half the command
  and reported success. `ExperienceMemory.recall` (both call sites: the whole
  task and each curriculum clause in `_plan_one`) now admits a habit or recipe
  only when its instruction has the task's clauses as `split_subgoals` cuts
  them: the same count, and for a sequence every clause ≥ `min_sim` against
  its counterpart IN ORDER. A refused candidate is skipped like a demoted one.
  So a sequence falls through to the curriculum (each clause still
  warm-started from its own habit) or to the LLM tier; a clause or a single
  command never replays a recorded compound (main: 0.929, both cubes moved);
  a reversed sequence (0.994) and a one-word difference inside one clause of a
  long command (0.958) no longer pass; a compound recorded as that compound
  still replays. A single instruction is decided by the index similarity
  alone, exactly as before (`tests/test_tier2_clause_structure.py` pins it
  differentially against the pre-B37 rule).

- **Programs tier (opt-in, ROADMAP #8; [design note](PROGRAMS_TIER.md)).**
  Waddle's level above skills: a program is a bounded (≤ 12 steps), declarative
  list of REGISTERED `TOOL_SPECS` calls, object labels as parameters and
  positions only as `localize_object(label)+offset` queries (the recipe
  mechanism), re-grounded before the first motion — one unresolved query aborts
  with zero motion. The orchestrator runs each step as a TOP-LEVEL
  `SkillRuntime.execute()` call (`tier: program`), so every step keeps its own
  trace row, keyframes, harness vetting and three-state postcondition; the
  per-step verdict is read from the task-effects ledger, never from the step's
  own result, and the first failed / refused / refuted / unverified step stops
  the program with a `next_action` for tier 3. `task_done`, `reset_scene`,
  `halt_motion`, pixel tools, `recall_step` and live-view tools are not
  composable. Authored and distilled programs share one admission rule (stored
  only from a fully CONFIRMED execution, offered only once promoted across ≥ 2
  distinct tasks). `programs=None` (the default) is the pre-change dispatch
  exactly; the mock brain and mobile/composed runtimes never get the tier.
  **MCP chat hosts (B42, opt-in with the same switch):** an arm MCP server
  with the tier on serves `list_programs` (promoted programs only) and
  `run_program` (a promoted program by name, or a host-written spec run
  once). The host is the brain, so its `run_program` call replaces the
  authoring turn and goes through the SAME `ProgramTier` + runner (top-level
  `execute()` per step, ledger verdicts, re-grounding, first-unverified stop,
  CONFIRMED-only admission, ≥ 2-task promotion); the server additionally
  refuses a program whose step or grounding query the capability matrix
  withholds or the operator hid, and hands the runner a halt check so a
  latched stop dispatches no further step. A cancel of an in-flight
  `run_program` latches the e-stop like any motion tool. With a
  `memory.embedder`, programs are ranked by text embedding with the skill
  library's floor-or-guard rule (keyword overlap otherwise). Tier off: both
  tools are absent and the catalog, every call result and `world_state` are
  the pre-tier ones (golden-pinned).

- **Latest-slot streaming, never queues** (`perception/stream.py`).
- **Warm world model.** Beliefs help resolve names and rank current visual
  matches; localization still analyzes an image and bounds any permitted
  memory fallback. Fusion pauses during `_MOTION_SKILLS` (the
  held object must not be re-fused mid-air). Two observations with
  DIFFERENT measured colours are two objects however close; proximity
  fusion (8 cm) is for label aliases of one object. A camera frame is
  fused as a whole (`BeliefStore.update_frame`, 2026-10-08): detections
  that share image support (an open-vocabulary second name, a part inside
  its whole) are one instance, and instances are matched to beliefs
  one-to-one by a min-cost assignment inside the same gates, so two
  identical props the detector sees as two stay two beliefs. The colour
  rule is per camera (B32b, 2026-10-08): a belief keeps the name each
  camera gave it, an observation is held to its own camera's name, and a
  camera that never named a belief fuses a perceptual-neighbour name
  (orange~yellow) only at 3D box IoU >= 0.75, with each cloud's lowest
  centimetre left out of the box (B32c: masks take in table pixels). Fusion
  is also size-consistent (B40, 2026-10-09): a view more than 2× the largest
  view a belief has had (robust horizontal diameter of the real-mask cloud,
  `ObjectBelief.diameter_m`) AND more than 5 cm larger is not that object, so
  a camera that names a container and the prop inside it alike cannot fuse
  the container into the prop. Smaller views always pass; no cloud, no veto.
- **Robot self-mask in fusion** (`perception/workspace.py`, B32a;
  `perception/link_mask.py`, B39). A detection more than half robot pixels is
  the robot and is dropped; below that, its robot pixels never reach 3D. The
  robot pixels are the render self-mask when a frame carries one (Isaac with
  `CASCADE_ISAAC_PIXEL_MASK=1`). For frames without one (the real rig),
  `workspace_filter.link_self_mask` (opt-in, default off) draws them from the
  arm's URDF collision geometry: each mesh is split into 5 cm link-frame cells,
  each cell's 3D hull contains its triangles, the links are posed by FK at the
  joint sample nearest the frame's capture time (sampled before inference), and
  the projected hulls are filled and dilated 2 px. No sample within
  `max_skew_s` (0.15 s) means no mask (counted, never interpolated). A render
  mask is never overwritten, and the arm is not even read then. The mask goes into a
  fusion-local copy of the frame; occupancy keeps its own body masking.
  Measured on CPU against a per-triangle rasterisation of the reBot RS meshes
  (4 poses × 2 cameras, 1280 × 720): coverage 1.0, IoU 0.889–0.951,
  3–5 ms per frame (`docs/evidence/b39-link-self-mask-20261009/`).
- **Detector preparation.** The open-world and prompted YOLO models remain
  resident, with up to eight successful text-embedding vocabularies retained
  in LRU order. This adds model residency while avoiding checkpoint and text
  encoder reloads when the watcher and a query alternate. Localization prepares
  its vocabulary under the detector lock before selecting an image; this step
  produces no detections or geometry. Every inference uses its own actual
  image and the existing five-second age check. Slow inference and preparation
  failures remain terminal for the attempted localization, with no automatic
  movement recovery. Watcher detection and safety heartbeats continue normally.
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
   run used to record `null`); for motion skills also one wrist keyframe
   per `role: wrist` / eye-in-hand stream in the rig; a
   `PostconditionChecker.snapshot()` of the target object.
3. Watcher paused for motion skills; the skill body runs; every exception
   becomes `{"ok": false, "error": ...}` -- nothing escapes by design.
   Every result is then stamped with one of three outcomes, `outcome: ok |
   failed | stuck`. `stuck` (`SkillStuck`, or a result that says so) is
   RPent's third finish status: the persistence loop spent its budget, or
   stopped for a reason no retry cures (object never seen after re-scans,
   wider than the jaws, destination not placeable) and `ask` is the
   concrete human-actionable request. A stuck result is forced to
   `ok: false` and never `verified`; the e-stop and every harness refusal
   stay plain failures. The orchestrator ends the task on it (fast path and
   LLM tier alike), relays the ask verbatim and writes `outcome: stuck` to
   `summary.txt`.
4. **Postcondition verification** (`agent/effects.py`): the effect is
   measured on the strongest available channel -- `physics` (sim truth),
   `belief` (perception), `gripper` (jaw width). A refuted claim
   *downgrades* `ok` and sets `self_reported_ok`. A displacement is two
   readings of the SAME channel (`_comparable_start`); a verifier that
   itself crashes yields an UNVERIFIED verdict naming the cause, never a
   silent pass. A confirmed verdict on a stuck step is downgraded to
   unverified (a stuck step claims no effect); refuted stays refuted.
5. Envelope update (`memory/envelope.py`), AFTER keyframe -- a FRESH frame
   for motion skills, taken after the arm stopped (the pre-motion
   `last_frame` graded the logger, not the robot, and an outcome judge
   scored 0% on a confirmed pick); the wrist AFTER frames follow the same
   fresh-grab rule, and a wrist camera that fails is dropped from the row,
   never a reason to fail the skill.
6. Trace row (`trace.jsonl`, with `tier`), and the Vesta memory tuple:
   AFTER frame + action text + independent verdict. `recall_step(n)` reads
   this evidence back for the planner/chat host (RPent
   `view_env_state(step=N)`): skill, args, outcome/ask, verdict, tier and
   the BEFORE/AFTER keyframes -- served over MCP as image content items
   with one caption each, and shown to the LLM tier once on its next turn.
   Negative `n` counts from the end; recall rows are not steps; an invalid
   `n` is an explicit error and never an old frame.

For single-arm Isaac kitchen `pick_and_place` calls that report a completed
motion to the configured green square or open box, `sim/placement.py` adds a fresh
passive observation on a private bridge connection. `LazyTruthPoseFn` binds
that reader to the configured endpoint and robot without materializing the
arm. Runtime aliases resolve the requested destination; the motion result
cannot supply its own object identity or placement verdict.
Multi-arm runtimes skip this adapter: the existing truth reader is bound to
the primary arm and cannot verify another arm's world.

`demo/kitchen/physics/placement_verdict.py` reuses the external proof's scene,
collider and settling auditors. It requires full footprint containment,
measured support, actual jaw release and an advancing window of at least
0.5 simulation seconds across four distinct steps, with sample gaps at most
0.25 simulation seconds. Contact observations must belong to the same object,
jaws, engine and physics step. A confirmation covers this final placement;
lift, transport, camera and reset evidence remain outside its scope.
Valid measured failures refute the placement; unavailable or inconsistent
evidence leaves it unverified and never repairs an original motion failure.

Acquisition has a 20-second observation budget and a separate socket so a
timeout cannot corrupt the motion or camera streams. Per-call JSON receipts
in `<run>/placement/` retain the raw observations and expected geometry;
their paths and SHA-256 hashes accompany the verdict in the tool result and
trace. Checkout-local audit modules and verified kitchen assets are optional
resources: if absent, including in a package-only installation, this check
returns `unverified`. Synthetic integration tests cover this contract; live
delivery acceptance remains the separate boundary documented above.

### Optional rendering without a physics driver

`OvrtxCamera` implements `CameraBase` for an explicitly static USD profile;
`static_scene: true` prevents its images from serving as live robot-effect
verification. The separate `OvrtxRenderer` API accepts complete caller-owned
scene snapshots and calibrated camera poses. It does not step physics, read
current joints or connect Isaac/Newton automatically. SDK imports and GPU
resources are acquired only on the optional path, and closed with its owner.

Each render transaction seals local prim transforms and camera calibration
before the SDK step. RGB and image-plane depth belong to that fixed snapshot;
whole-packet repeats retain the original timestamp. Depth is metric Z, RGB uses
Cascade's BGR convention, and calibration supports centered square pixels only.
Capture-start monotonic time, producer/snapshot identity and SDK render time
are distinct metadata. No robot/target masks or physical attachment authority
are fabricated. The USD source hash covers the root file, not its dependency
closure. See [OVRTX rendering and platform evidence](OVRTX_RENDERER.md).

### Optional bridge timing

The Isaac bridge has disabled-by-default [Python profiling zones](ISAAC_BRIDGE_PROFILING.md)
around its existing target, update, frame-history, camera and execution-queue
operations. The standard-library helper imports Carbonite only when enabled
after application creation. Its initial clock anchor and periodic main-thread
samples bind captured CPU event order to bounded monotonic intervals;
profiler logs and timings never authorize motion or renew camera freshness.
Profiler errors invalidate diagnostics while the
original operation and safety behavior remain intact.

### Lazy frame encoding

The [component-encoding candidate](ISAAC_FRAME_ENCODING.md) gives RGB JPEG and
compressed depth separate locks and caches. An RGB-only reader does not start
depth compression or acquire its lock. `wire()` still returns both components
with their original encoding and field order. Camera buffers retain their
existing render/history identity; codec completion never updates timestamps,
epochs or measured joints. Unused raw components remain owned by the retained
frame until requested or released. Live performance and native acceptance
remain separate from the retained CPU comparison.

### Motion safety path

Skills hold a `SafeArm`. Its motion path enforces the configured peak velocity,
joint limits, workspace, table clearance, keep-out zones, perception watchdog,
stop generation and optional occupancy/inter-arm checks. A `SafetyViolation`
ends the stream. `vet_pose()` is the static counterpart for candidate rejection;
planned routes also vet their sampled segments before execution.

Isaac defaults to 30 Hz nominal targets paced by authoritative physical time.
The common profile preserves every complete legacy 50 Hz edge with its original
dt, adds actual command edges and checks intervening subedges. Hardware retains
its existing 50 Hz default. This is sampled coverage, not continuous collision
certification. Settling and bounded state/command RPCs are described in
[Isaac motion clock](ISAAC_MOTION_CLOCK.md).

Optional non-payload occupancy may report no data for an absent/stale cache;
that is not evidence of free space. Profiles opting into `occupancy.required`
refuse startup if the bridge cannot be probed and refuse motion on missing,
stale or failed map observations. Their ordinary arm samples also need observed
interpolation support outside the existing local contact exemption. A known body-mask fault raises even when
the map is stale. Payload/recovery modes additionally require observed geometry,
correct source/epoch/contact binding and successful fresh commits from all
required cameras. Unknown payload samples and the configured 30 mm clearance
remain rejection conditions. Contact/release episodes retain narrowly scoped
recovery authority after failure; another command cannot borrow their cylinder.
See [nvblox](NVBLOX.md) and [release recovery](NVBLOX_RELEASE_RETREAT.md).

The [NV carry attachment guard](NVBLOX_CARRY_ATTACHMENT.md) retains the original
post-close attachment through lift and placement transport. Isaac state reuses
the same completed-update contact history; existing feedback reads validate
source, epoch, step, joints and jaws before further targets. Loss or unknown
evidence is terminal, with no automatic open, retry, home or reset. This
capability is armed only by the NV payload barrier, not by occupancy-off or
legacy backends, and it does not correct mechanical slip.

Gripper commands retain stop/generation and pending-episode checks. On the
observed-finger path, each close stage also has an actual-pose, full-stroke
preflight against non-target observed surfaces. That check does not continuously
brake an already submitted gripper command. The [approach gate](observed-finger-gate.md)
keeps target points in the scene and covers the two calibrated finger links,
not the palm, whole arm or unobserved space. For a validated PhysX clock,
[the finger envelope](physx-finger-envelope.md) retains eight nominal components
and adds sixteen derived cooking components from each of the x86 and ARM
exports, giving forty components per finger. Source hashes and
kinematic metadata bind that choice. This changes the checked geometric envelope,
not the physical colliders, 0.1 mm opening interval or occupancy clearance.
Local representation admission and physical acceptance are separate evidence.

The additional [endpoint occlusion veto](observed-finger-occlusion.md) clips
those convex envelopes against captured pixel rays and rejects portions behind
valid non-target, non-robot depth. It covers pregrasp/grasp endpoints and measured
pre-close poses; it does not certify invalid depth, masked regions, inter-ray
space or intermediate poses. Symmetric wrist alternatives pass all checks
independently, within the existing planning deadlines and preserved ranking.

### Grasp pipeline

[Localization freshness](LOCALIZATION_FRESHNESS.md) checks the analyzed image's
client-local age before and after detector or VLM work, including accumulated
fallback time. An expired result cannot authorize a grasp or an automatic home
retry. A secondary-camera VLM fix keeps its actual image and calibration.
The separate [Isaac startup check](ISAAC_STARTUP_READINESS.md) prepares the native
read-only verifier and waits for newer camera captures before proof starts.

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

The Spark presenter profile requires real GraspGen-X candidates and checks
diffusion inference during startup. A missing server, protocol stub or failed
required inference raises an error instead of substituting OBB. The five-second
fallback cooldown applies to optional profiles; required profiles retry on the
next request.

The opt-in `isaac_kitchen_hug` profile requires real HUG in the same way. A
missing server, the HUG protocol stub or a failed request is an error, never
an OBB or GraspGen-X substitute. Only HUG receives the RGB-D frame; other
backends are called exactly as before. HUG candidates are proposals: the
same outcome-memory re-rank, selector and harness vetoes apply.

With GGX and the observed-finger gate, infeasible planning may request at most
three batches within eight seconds, using the same saved scene and frozen prior.
No motion separates these batches. Transport, malformed data, cancellation and
clock errors are terminal, rather than reasons to regenerate candidates.
[Planning search](grasp-planning-search.md) and [memory ranking](grasp-memory-ranking.md)
define the selector contract.

A cached offset or an unbound legacy position can aim a carried object but cannot
prove slip or authorize opening. [Held-object observations](HELD_OBJECT_OBSERVATION.md)
require identified, temporally coherent evidence and the driver's actual joint
convention. Camera geometry uses the TCP at capture time, not a later TCP.
Identity/epoch/stop failures remain terminal. Optional retry behavior elsewhere
does not override those guards or the observed-gate failure boundary.

Single-hinge jaws (SO-101) close toward the fixed tip, so the profile
declares the jaw datum (`jaw_fixed_tip_m`, `jaw_close_dir`) and the selector
displaces the IK target accordingly -- without it every grasp straddled the
prop while perception was accurate to 1.4 mm.

**Reach envelope (B45, opt-in, 2026-10-09).** The analytic planner is
top-down by default, and on the reBot top-down grasps reach only r ≤ 0.45 m
from the base (0.38 m at a 10 cm grasp height), measured through the real
selector and the runtime's harness vet by `scripts/reachability_study.py`
([REACH_ENVELOPE.md](REACH_ENVELOPE.md)); leaning the approach away from the
base reaches 0.69 m (45°), a horizontal approach 0.75 m, leaning sideways
nothing. `grasp.angled_approach_tilts_deg` (empty by default) makes
`plan_grasps_from_fix` append tilted versions of every footprint candidate,
same horizontal jaw axis, approach leaning perpendicular to it away from the
base, ranked behind every top-down and rim candidate. The `*_reach` arm
profiles (`rebot_rs_reach`, `isaac_reach`, `mock_reach`) set the tilts to
[30, 45, 90] and replace the workspace box with the measured envelope
(x 0.10..0.55, y ±0.50: the shipped box grown only over grid cells where one
of those approaches reaches at every studied grasp height). That box is the
new limit of those profiles; every other harness gate is the parent's.
`tests/test_reach_envelope.py` pins the profiles to the study's JSON, the JSON
to the URDF hash and settings, and every other profile golden.

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
`sim/bridge_client.py`). Truth readers disable mutating constructor defaults;
slip-authorizing observations additionally require valid physical handles and
same-step articulation/base/clock evidence. MuJoCo resets restore `qpos0` under
the world lock. Isaac resets restore backend physical state and require the
appropriate fresh-camera/map barrier before subsequent motion. Its renderer
publishes only a token bound to recorded state, so reading an old render cannot
make that image current by assigning a new timestamp. See
[frame history](isaac-render-frame-history.md).

### Memory

| store | what | horizon | consumer |
|---|---|---|---|
| `BeliefStore` (`memory/beliefs.py`) | objects: label, colour, 3D, freshness; visible/remembered | persisted across runs (wall-clock stamps, `LOADED_MIN_AGE_S` floor, 6 h max age) | every skill; can inform the agent, can never aim the jaws (`belief_fallback_age_s`) |
| `BeliefStore` named snapshots (`SceneSnapshot`) | ADVISORY layouts: the confirmed (visible) objects' label, colour, centroid, extent at memorize time | saved/loaded with the beliefs; dropped by `clear()` (`reset_scene`) | `snapshot_scene` writes, `restore_scene` reads them as TARGETS; never a claim about the present — only the `restored` postcondition is |
| `EpisodicMemory` text ring | events, outcomes | ~15 s | `recall_memory`, narration |
| `EpisodicMemory` frame ring | AFTER frame + action + verdict per motion skill | task-scale (600 s), reset per task / by `reset_scene` | `memory_frames(k)`: first frame pinned, uniform sample, newest last → LLM turn (images) and `task_memory` tool |
| `EpisodicMemory` visual index (**opt-in**, `memory.embedder`) | one TurboQuant vector per motion frame and per object crop localized during a call (`_localize` detections, keyed by detector label), from `memory/embedder.py` | task-scale (`frames_horizon_s`), ≤ `max_visual` (512) entries, pruned with the index; survives the per-task frame reset | `recall_visual(image \| text \| vector)`; `recall_memory(query)` adds `looks_like` hits only with a joint image-text embedder (siglip/clip). A hit is a remembered appearance, never a current observation |
| `EpisodicMemory` watcher crops (**opt-in**, `memory.visual_recall_detections` + an embedder; B43) | one `detection` vector per crop of a COMMITTED WorldWatcher fusion, via `DetectionCropRecorder`: one per belief per frame, none for a belief cropped < `visual_recall_interval_s` (30 s) ago, ≤ `visual_recall_max_per_tick` (2) per tick; the recall line says what, where and which camera | task-scale, its own ring ≤ `visual_recall_max_detections` (128), so it never evicts the session entries above; events stay out of the text/frame rings | ranked together with the session index by `recall_visual` / `looks_like`. Fusion is paused during `_MOTION_SKILLS`, so no crop comes from a motion frame |
| `EpisodicMemory` restored entries (**opt-in**, `memory.persist_episodic` / `CASCADE_EPISODIC`, `CASCADE_EPISODIC_PATH`; B43) | the visual index of the previous session (`save_visual` on shutdown, `load_visual` at build): kind, label, recall line, verdict, the decoded vector (float16) | `runs/episodic.json`; wall-clock stamps, `episodic_max_age_s` (6 h) dropped BEFORE the 2 s `LOADED_MIN_AGE_S` floor, own ring ≤ `max_visual`, pruned at that max age (not the task horizon); another embedder's file is refused | every restored hit carries `restored: true, state: "remembered"` and `recall_memory` says so; never a current observation, never aims motion |
| `ExperienceMemory` (`agent/reflex.py`) | command → plan habits, hashed BoW in a TurboQuant index; plus Task-Specific Memory **recipes** (verified LLM-tier runs, coordinates replaced by `localize_object(label)+offset` queries + a summary, `memory/recipes.py`) | `runs/experience.json` (habits), `runs/recipes.jsonl` (recipes) | tier 2; a recipe is re-grounded through perception before any motion, a failed grounding aborts to the LLM tier |
| `ExperienceMemory` (`agent/reflex.py`) | command → plan habits, hashed BoW in a TurboQuant index; plus Task-Specific Memory **recipes** (verified LLM-tier runs, coordinates replaced by `localize_object(label)+offset` queries + a summary, `memory/recipes.py`) | `runs/experience.json` (habits), `runs/recipes.jsonl` (recipes) | tier 2, only for an instruction with the record's clause structure (same `split_subgoals` count; a sequence clause by clause, in order); a recipe is re-grounded through perception before any motion, a failed grounding aborts to the LLM tier |
| `ActionObjectMemory` (`memory/consolidation.py`, **opt-in**, `memory.action_objects`) | the tier-2 outcome stream consolidated per (motion skill, normalized object label) across instruction wordings: wins / losses / wordings; one credit per EXECUTED call (never again per curriculum sub-goal); deliberately not merged by embedding (red cube ≠ blue cube) | `runs/action_objects.json` | LLM-tier intro: advisory digest for the objects the task names |
| `ProgramLibrary` (`memory/programs.py`, opt-in) | programs: parameterized registered-call lists (labels as params, positions as perception queries), keyed by a structural sha256; `occurrences`, `source_tasks`, `origins` (authored / distilled / reused), `losses` | `runs/programs.jsonl` (`memory.programs_path`, `CASCADE_PROGRAMS_PATH`) | tier 2.5 authoring prompt, **only promoted** records (≥ `agent.program_min_tasks` = 2 distinct tasks, verified more often than failed); admitted only from a CONFIRMED execution; also the MCP `list_programs` / `run_program` when the tier is on; keyword overlap, or text embedding with `memory.embedder` (floor-or-guard); every write re-reads the store under an advisory lock, so per-session MCP servers sharing it never lose each other's evidence |
| `GraspOutcomeMemory` | per-object grasp features, wins/losses | `~/.cascade/grasp_memory.json` | grasp re-rank + z-nudge |
| `OperatingEnvelope` (`memory/envelope.py`) | per-skill outcome statistics and failure classes, raw args plus runtime-measured derived features (`DERIVED_FEATURES`: TCP z at close, object height/width, lateral offset; unmeasured → `missing`, never defaulted) | `~/.cascade/envelope.json` (`CASCADE_ENVELOPE_PATH`) | planner context; advisory |

`skills/library.py` stores markdown guidance. Between sessions,
`agent/aspire.py` admits a failed-then-successful retry only when its skill,
goal parameters, resolved arm and pre-call held-object context match, its
postcondition is measured and confirmed, and no explicit reset/task-end
record intervenes. `SkillRuntime.execute()` records that context separately
from tool arguments, without probing a lazy backend. Legacy traces lacking
context cannot produce new notes.

`scripts/learn_from_runs.py` harvests eligible associations into
`skills_library/*.md`, one note per `(skill, signature)` whose front matter
counts `occurrences` (distinct runs) and `source_tasks` (distinct tasks);
`retrieve()` loads keyword-matched notes into the tier-3 context at task start
(`orchestrator.run_task`, with the library supplied by `demo.main`) **only once
they are promoted** — recurred in ≥ 2 distinct tasks, upstream ASPIRE's rule.
A note seen in one task stays a stored candidate; `memory.skill_min_tasks: 1`
is the explicit relaxation. A note
preserves the recorded evidence, not proof of a causal repair or transfer to
another rig.
This gate does not change experience-memory or operating-envelope admission.
See [retry evidence admission](DREAM_RSI_ADAPTATION.md) for the exact checks,
CLI workflow and remaining state/episode-lineage limits.

**Memory embedder (ROADMAP #7, opt-in).** `memory.embedder.backend` is `none`
in the shipped profile, and then nothing above changes. `hash` is a
deterministic, dependency-free embedder (colour histogram + coarse layout for
images, stemmed hashed words for text, in SEPARATE spaces): image→image recall
of frames and localized object crops, and embedding-ranked skill notes
(`relevant_entries(..., embedder=)`: cosine ≥ the embedder's `text_floor` OR a
guard-word overlap, so an inflected paraphrase is found and no guard match is
lost; the promotion gate runs first, unchanged). `siglip`/`clip` load a real
image-text model through the `memory-embed` extra (torch from the host) with
`local_files_only` by default; a requested backend that cannot load fails
`build_runtime` with `EmbedderUnavailable` before any arm or camera exists --
never a silent fallback to `hash`, whose vectors would answer a semantic query
with noise. Embedding faults during a run are counted (`visual_stats`), never
raised into a skill. Everything here is advisory: no recall confirms an
outcome or gates motion.

**Visual recall v2 (B43, both opt-in, both need the embedder).**
`memory.visual_recall_detections` hands the detections of every COMMITTED
WorldWatcher fusion to a `DetectionCropRecorder` (`memory/episodic.py`), which
uses `BeliefStore.update_frame`'s per-observation belief to keep one crop per
object per frame, at most one per object per `visual_recall_interval_s`, at
most `visual_recall_max_per_tick` per tick; the crops land in a `detection`
ring of their own, so a static table is not re-embedded at 3 Hz and the
always-on watcher cannot evict motion frames or localized crops. The work
runs on the watcher thread after the commit and outside the pause lock, and
never for a frame taken while fusion was paused (motion skills), nor once a
pause began after the commit; a recorder fault is counted
(`WorldWatcher.visual_recall_errors`), logged once per distinct message, and
costs only the crops. `memory.persist_episodic` (`CASCADE_EPISODIC`,
`CASCADE_EPISODIC_PATH`) saves the visual index in `shutdown_runtime` (its
own `episodic` teardown stage, right after `beliefs`, only when enabled) and
restores it in `build_runtime`, with the belief store's rules: wall-clock
stamps, the max age (`episodic_max_age_s`, 6 h) applied before the 2 s
`LOADED_MIN_AGE_S` floor, atomic temp + `os.replace`, a corrupt file ignored
and a file from another embedder (name or dim) refused. Vectors are stored
decoded (float16) in the embedder's own space, so the file does not depend
on the quantizer's rotation; restored entries keep their own ring, pruned at
that max age rather than the task horizon, and every restored hit carries
`restored: true, state: "remembered"`. Bad limits fail the build before any
hardware; a switch without an embedder is reported and ignored. Off (the
shipped default), the watcher, the memory and the teardown receipt are
exactly B21's. Fleet: a global `CASCADE_EPISODIC_PATH` is refused for two
manipulation robots, `episodic_path` joins the store-file exclusivity check,
and composed manipulation domains get `stores/episodic.json`.

### Evaluation

`eval/progress_judge.py` is a Robo-Dopamine-style progress judge: BEFORE/
AFTER keyframes (plus optional goal image) → `<score>±NN%</score>` from a
GRM or any OpenAI-compatible VLM. It runs **off the hot path**
(`scripts/judge_run.py` over a finished run dir) and is calibrated against
the physics postcondition per step (confusion matrix in the run summary).
Since 2026-10-09 (B44) the launcher can run it over its own proof turn:
opt-in `scripts/launch.sh --judge fake|vlm|grm` (or `CASCADE_JUDGE`; default
off, no judge process) hands the receipt to `scripts/judge_proof.py`
(`eval/proof_judge.py`), which reads `proof.json` read-only, skips an
unverified receipt, runs `judge_run.py --skills pick_and_place` in its own
process group under a hard bound (`CASCADE_JUDGE_TIMEOUT_S`, default 180 s,
max 1800; the group is killed at the bound) and merges the judge-vs-physics
confusion matrix into `<evidence_dir>/run-summary.json` next to a copy of
the receipt's verdict and sha256, plus ONE `judge:` banner line (`fn>0` =
the pictures missed physics-confirmed progress). Advisory: READY, the exit
status and `proof.json` never depend on it, and every failure (bad config,
refused or hung endpoint, nothing scored, crash) reads `unavailable`.
The first honest number on this rig: +0.45 on a physics-confirmed pick
after the AFTER-keyframe fix; 0.00 before it. The prompt's two WRIST slots
are filled from the rig's wrist keyframes when the trace has them
(`keyframe_{before,after}_wrists`, written for motion skills on any rig
with a `role: wrist` / eye-in-hand camera -- `mujoco_wrist` renders one
from the SO-101 gripper body); otherwise the front image is repeated, and
every record names which it was (`wrist_slots`, `wrist=` in the summary).
No agreement change is claimed from the wrist view until the judge is
re-run with it.

## Module map

```
src/cascade/
├── types.py            Frame / Detection / ObjectFix / Grasp / RobotState / SkillError
├── config.py           YAML profiles (cameras/, arms/, llm/) → one Cfg; `extends:`,
│                       arm `overrides:`, ${repo}/${assets}; CASCADE_BOOTH overlay;
│                       CASCADE_{BRIDGE,GRASPGENX,OCCUPANCY,HUG}_PORT and
│                       CASCADE_{GRASPGENX,HUG}_HOST applied last (B34, B41)
├── device.py           resolve_device(): auto CUDA/ROCm → MPS → CPU, degrade with a warning
├── perception/
│   ├── camera_base.py        CameraBase ABC + make_camera(); Frames carry METRIC depth
│   ├── realsense_camera.py   D4xx / L515          opencv_camera.py  RGB-only UVC
│   ├── mock_camera.py        synthetic tabletop / npz replay
│   ├── mujoco_camera.py      RGB-D RENDERED from the shared MuJoCo world (type: mujoco)
│   ├── isaac_camera.py       Isaac bridge frames (RGB-D + per-frame T_base_cam)
│   ├── ovrtx_camera.py       optional static USD RGB-D; no live robot-effect authority
│   ├── depth_provider.py     sensor → mono plugin → table-plane ray-cast
│   ├── detector.py           YOLOE / YOLO-World + MockDetector (open world by default)
│   ├── vlm_detector.py       VLM as detector      vlm_ground.py  second-chance grounder
│   ├── segmenter.py          mask refinement      robot_mask.py  arm body out of depth
│   ├── workspace.py          WorkspaceFilter (base cylinder, self-mask gate)  link_mask.py  self-mask from link geometry
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
│   ├── episodic.py     text ring (15 s) + frame ring (task-scale) + memory_frames(k); opt-in visual index (recall_visual), watcher crops (DetectionCropRecorder) and save_visual/load_visual (B43)
│   ├── embedder.py     opt-in memory embedders: hash (deterministic, no deps) | siglip/clip (`memory-embed` extra)
│   ├── consolidation.py opt-in action<->object outcome counts over tier-2 plans (advisory digest)
│   ├── envelope.py     Harness-VLA operating envelope (per-skill outcome stats + runtime-measured derived features)
│   ├── recipes.py      Task-Specific Memory: xyz ⇄ localize_object(label)+offset queries (symbolize / ground)
│   ├── programs.py     ProgramLibrary: stored programs, CONFIRMED-only admission, ASPIRE-style ≥2-task promotion (opt-in tier 2.5)
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
│   ├── simulation_motion.py / motion_profile.py  physical clock + shared safety edges
│   ├── feetech.py / feetech_arm.py   SO-101 & co over Feetech serial (UNVERIFIED on hw)
│   ├── rebot_rs_arm.py / rebot_rs_mb_arm.py   reBot B601 over CAN / MotorBridge
│   ├── robstride.py    shared RS helpers: travel clamp, fault clear, opt-in squeeze-capped close (B38)
│   ├── ros2_arm.py     ANY ros2_control robot (JointState in, JointTrajectory out)
│   └── unitree_arm.py  Unitree SDK arms (H1 / H1-2 / G1)
├── safety/
│   ├── harness.py      SafetyHarness (approve / vet_pose, escape rules) + SafeArm
│   ├── trajectory.py   sampled route validation and bounded planning
│   └── geometry.py     segment-segment distances for the inter-arm gate
├── calibration/        hand-eye calibration (ArUco; ported from Seeed's WRC fork) -- docs/HANDEYE_CALIBRATION.md
│   ├── handeye.py      joint SE(3) solve of A X B = Z (eye-to-hand + eye-in-hand), robust LM + quality gate
│   ├── session.py      collection sweep: presets vetted twice, every move via SafeArm.move_planned
│   ├── dataset.py      schema-v1 record (gated on load)    aruco.py  detection + IPPE-square PnP
│   └── cli.py          --list/--bind/--dry-run/--verify    synthetic.py  rendered-marker camera
├── grasping/
│   ├── obb_grasp.py    base-frame OBB grasps      graspgenx_backend.py  ZMQ client + fallback
│   ├── selector.py     supplied/quality order ▸ width ▸ IK ▸ harness pre-vet
│   ├── observed_scene.py calibrated observed-finger approach and closing veto
│   ├── vla_client.py   openpi/LingBot websocket msgpack-numpy policy client (`vla` extra, lazy)
│   ├── vla_executor.py opt-in `grasp.executor: vla`: chunk → joint targets, harness-gated episode
│   └── force.py        material → two-stage close profiles
├── agent/
│   ├── orchestrator.py reflex → habit → (opt-in) program → LLM loop; memory harness injection; TaskReport
│   ├── reflex.py       tier-1 grammar (incl. reset_scene, memorize/restore/find) + tier-2 ExperienceMemory
│   ├── programs.py     tier 2.5 (opt-in): program contract/validation, runner (ledger verdicts, stop + next_action),
│   │                   authoring prompt/parse, distillation of verified runs (docs/PROGRAMS_TIER.md)
│   ├── effects.py      PostconditionChecker + annotate_result (Pigey closed loop; `restored`/`searched` for the composites)
│   ├── milestones.py   checkable milestones: symbolic first, VLM second, UNKNOWN honest;
│   │                   + advisory pre-motion plausibility critic (never a veto)
│   ├── llm.py          OpenAI-compat (cloud/local) / Anthropic / Codex CLI (GPT-6-Astra via `codex exec`, no API key) / Cosmos3 / Mock
│   ├── cosmos3.py      Cosmos3-Edge XML tool-call dialect
│   ├── prompts.py / advisor.py   persona, decomposition, VLM critic
│   ├── aspire.py       post-run diagnosis → skill-library note
│   └── trace.py        trace.jsonl + keyframes
├── skills/
│   ├── runtime.py      SkillRuntime: 37 skills + task_done, TOOL_SPECS, _MOTION_SKILLS (19)
│   │                   incl. the Pigey composites snapshot_scene / restore_scene / search_for_object
│   ├── contact_episode.py / release_episode.py  scoped retained recovery
│   ├── held_observation.py aiming estimates vs coherent release authority
│   └── library.py      markdown repair notes; written by aspire.py, retrieved per task
├── sim/
│   ├── ovrtx_renderer.py     optional owned RTX renderer for explicit scene snapshots
│   ├── mujoco_world.py shared MjModel/MjData registry (arm + cameras + truth, one lock)
│   ├── demo_scene.py   deterministic scene writer: arm MJCF + table + N props
│   ├── truth.py        physics-truth channel (MuJoCo + Isaac), LazyTruthPoseFn
│   ├── mujoco_rgbd.py  offscreen RGB-D + data.xpos truth (perception verification)
│   ├── isaac_reset.py  validates measured per-prop reset replies, never fabricates poses
│   └── bridge_client.py newline-JSON TCP client for scripts/isaac_bridge.py
├── eval/progress_judge.py   Robo-Dopamine progress judge (GRM / VLM), off the hot path
└── apps/
    ├── demo.py         build_runtime() = the composition root; CLI --task / --interactive
    ├── mcp_server.py   MCP front-end: 47 tools (2 only with the opt-in programs tier), out-of-band stop, per-call log, opt-in read-only lane (B46); stdio by default, Streamable HTTP (`--http`, bearer + TLS) for NemoClaw/OpenShell
    ├── capabilities.py capability matrix from the built runtime; TOOL_REQUIREMENTS trims the MCP catalog
    ├── process_owner.py profile-owned process identity for shutdown and proof binding
    ├── stream_server.py lazy MJPEG dashboard (+ chat, STOP)     live_view.py  RigViewer
    ├── live_control.py viewer-driven control        record.py / viewer.py  capture / view
```

Sidecars (own process, own venv, ZMQ): `scripts/serve_graspgenx.sh`
(learned grasps, CUDA) / `serve_graspgenx_stub.py` (protocol double, any
host); `scripts/serve_occupancy.sh` → `serve_occupancy_bridge.py`
(`--backend auto`: nvblox > warp > voxel). Both are **probed at startup**
and named in the banner; a missing sidecar degrades loudly to its fallback,
never silently. The learned GraspGen-X server runs one synthetic inference
before it binds its port (the first CUDA inference takes ~15 s, the client
waits 8 s), so an open port means a warm model unless its `health` says
`warmed_up: false`: the warm-up is advisory, and a failed one is logged and
the server binds cold as before (no CUDA or no loadable model still refuses
first; the required-profile inference check above is unchanged).

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

Spark distribution starts at `scripts/bootstrap.sh` → `scripts/install.sh`:
Linux DGX Spark is the reference deployment, with explicit EULA acceptance,
Isaac Sim, PhysX CUDA, GraspGen-X, a local vision-language model and OpenClaw.
Concrete versions are specified in the [deployment guide](DGX_SPARK_SETUP.md).
`.venv`, `.isaacsim` and `.graspgenx` isolate incompatible Python dependencies;
`.llama.cpp` runs the local brain, and `cascade-demo` isolates the attendee
host profile. The installer does not replace drivers. Required learned grasps
fail visibly if inference is unavailable. Occupancy/nvblox is disabled; JEv
and Cosmos are not active presenter backends. Jev/Kev evaluation code is merged
research; it is not selected by normal launch. The installed Spark defaults to
`dt=1/120`, six bridge iterations per nominal camera poll and the observed-finger
gate enabled. Explicit supported overrides retain precedence.
`--prepare-only` stops after dependencies/assets; it cannot print READY.
The package/model resolution and CPU contract tests are not GPU rehearsal:
see `docs/DGX_SPARK_SETUP.md` for the release pin and its acceptance status.

`run.sh` → `scripts/launch.sh` is the one-click entry: `--sim auto|isaac|
mujoco|none`, `--setup` (venv, extras, assets, OpenClaw CLI, provider
onboarding), `--check` (report only, never mutates), `--dry-run`, `--down`
(stops the sidecars it started AND the per-session MCP servers the gateway
never reaps). Before READY it proves the stack: runtime built with the
server's exact env, tools listed, a trivial brain turn, a real pick and reset in the SAME chat session (the installed Spark profile
requires its green and orange cases). Proof must bind model/session, exact MCP
tool names and the owned live process to its trace; a recent timestamp
alone cannot establish identity. Reset must include the manipulated prop
and a subsequent `world_state` read-back. `--no-robot-turn` is STARTED /
UNVERIFIED, not READY. The MCP server is registered with
`requestTimeoutMs: 300000`. Native visitor turns and proof pick turns share a 360-second
whole-turn budget, including a 60-second host reserve around the per-tool limit,
plus a 30-second CLI return margin; an expired motion still
latches e-stop. See [turn budgets](native-visitor-turn-budget.md). Dead MCP entries are
pruned, and on macOS the server runs under `mjpython` so the MuJoCo window
can open -- when a display is active; a sleeping display is detected and
the window retried on the next motion instead of segfaulting the server.

Hosts: OpenClaw (native `mcp.servers`), Hermes (`~/.hermes/config.yaml`),
Claude Code (`.mcp.json`), Claude Desktop, Codex -- all via
`scripts/setup_agents.py`. Host and brain are different roles: as a host
the platform's LLM picks tools and cascade's tiers are bypassed
(`llm=mock` inside the server); as a brain (`--llm codex_astra|hermes|
anthropic|openai|local_*`) cascade runs its own loop with all three tiers.
Codex can be either: `--codex-profile NAME` registers the tool server in a
`codex -p NAME` layer (host), while `codex_astra` runs GPT-6-Astra as the
brain through a `codex exec` subprocess per step -- `--ignore-user-config`,
so that session never sees the user's own MCP servers and cannot reach the
robot except through cascade's harness; `--llm auto` picks it first when
the CLI is logged in.

Endpoints (B34, B41): `CASCADE_BRIDGE_PORT`, `CASCADE_GRASPGENX_PORT`,
`CASCADE_OCCUPANCY_PORT`, `CASCADE_HUG_PORT`, `CASCADE_GRASPGENX_HOST` and
`CASCADE_HUG_HOST` are applied by `load_demo_config` after every config layer,
to the top level and to each arm's `resolved` view (`config.ENDPOINT_ENV_VARS`;
empty = unset; a port is ASCII digits in 1..65535, a host a hostname or IPv4
address, anything else raises naming the variable). That is the only
application: a GraspGen-X / HUG planner dials what its resolved section says
(`config.sidecar_endpoint`) and falls back to a variable only for a key the
section lacks, i.e. when it is built from a hand-made config, so an override
written into `cfg._data` after loading stands. The bridge and the occupancy
sidecar have no host variable: the launcher starts both on this machine.
`launch.sh` registers the six with the MCP server; `setup_agents.py` copies
every `CASCADE_*_PORT` / `CASCADE_*_HOST` set in its shell into each host's
entry, checked by the same rules before anything is written.

Sandboxed host (opt-in, B35): an agent inside an NVIDIA OpenShell sandbox
managed by NemoClaw reaches the robot through `mcp_server --http`
(Streamable HTTP, TLS from a private CA, bearer token in OpenShell's provider
store), because NemoClaw registers only authenticated HTTP MCP servers. The
robot runtime stays on the host; the transport shares the stdio server's
serial worker and receive-side stop channel (`_admit`), and namespaces
JSON-RPC ids per session. `scripts/nemoclaw_mcp.py` issues the certificate,
token and registration. Stdio through `launch.sh` remains the default; see
[NEMOCLAW.md](NEMOCLAW.md).

Read-only lane (opt-in, B46): `mcp.readonly_lane: true` (or
`CASCADE_MCP_READONLY_LANE=1`) lets a chat host look while the robot moves.
After `_admit` (stops, cancels, pings keep first claim on every frame), the
receive side's `_enqueue` hands a `tools/call` for one of
`READONLY_LANE_TOOLS` -- `world_state`, `robot_knowledge`,
`verify_last_action`, `camera_snapshot` -- to a single lane thread when the
worker's in-flight call is a motion (`_MOTION_SKILLS`, or `run_program` with
the programs tier on); every other frame is queued for the serial worker as
before, so motions stay strictly serialized and a second motion still waits.
The receive side only checks and enqueues, so a held lane call cannot delay a
stop, and the lane never writes `_inflight`, so a host cancel of the motion
still latches the e-stop. Membership rule, per tool: it never enters
`SkillRuntime.execute()` (no trace row, memory event, envelope record,
watcher pause, per-call scratchpad or `last_frame` write), never takes
`_exec_lock`, never commands an arm, gripper or base, and touches only state
already guarded for concurrent readers. `world_state` is the dashboard
`/state` body, which runs on HTTP threads during every motion (its only
writes are the belief store's and episodic memory's own age-based expiry,
under their locks, which every reader performs); the envelope
and grasp-memory digests take their stores' own short locks, which the 50 Hz
control loop (`harness.approve` + arm streaming) never takes; the verdict
history is append-only; on the lane `camera_snapshot` is a passive stream
read like the dashboard's MJPEG reads (the stream's condition is held only to
take a frame reference; waiting releases it) -- no `observe_fresh`, so no
`last_frame` write, no in-place depth filling, no harness heartbeat.
`describe_scene` and `get_observation` stay
out: they fuse beliefs mid-motion (exactly what the watcher pause prevents),
run the shared detector and read the arm. A lane result carries
`served_during_motion` (`motion`, `note`: state may be in flux, the motion's
verdict is not recorded yet). Off (the default) the routing is the plain
queue put; `tests/test_mcp_readonly_lane.py` pins both modes through the real
server process (stdio and HTTP), holding a motion mid-stream with file
barriers.

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
  ~3.2 ms/step -- the C engine is the laptop MuJoCo default. The Spark
  presenter delivery uses Isaac PhysX CUDA; Isaac Newton is an explicit,
  separately validated option. Standalone Newton CPU tests
  are a separate validation path, not an additional CASCADE arm backend.
- **Feedback, not sleep.** Every backend reports real joint positions;
  settling checks target error with a per-profile tolerance and timeout. Isaac
  additionally requires stable q over advancing physical steps; a stepped-on-demand
  simulator advances through its own backend contract.
- **Authority depends on evidence.** Limits, stop generation and required
  geometry fail closed. Optional observations can remain unavailable; they do
  not become free-space proof, required-GGX fallback or authority to open a
  held object. Known mask faults and payload barriers reject. An unavailable
  postcondition remains unverified. Results name the channel that answered.
- **A claim is not a fact.** Every effect is verified on an independent
  channel when one exists, and the verdict travels with the result, into
  the trace, into memory and to the judge.
- **The tool surface is derived, not declared.** `apps/capabilities.py`
  reads the BUILT runtime -- the depth chain each camera stream really
  produces (`DepthProvider.depth_source_for`), the sidecar probes behind
  `runtime.backends()`, the `ArmRig`, the verifier, the memory -- and
  `TOOL_REQUIREMENTS` names what each tool cannot run without. The MCP
  server withholds (and rejects if called) only what probed state shows
  unmet; unknown is not unavailable, a fallback (OBB for a dead GraspGen-X)
  is reported, never hidden, and the stop path is never a capability. The
  matrix is printed next to `backends:`, served by `/state` and by
  `world_state` with every withheld tool's reason. `CASCADE_HIDE_TOOLS`
  remains the explicit operator override on top.
- **Memory is structured first, embeddings second.** Recall tools work on
  labels/time/positions; the TurboQuant index has one live consumer by
  default (tier 2), plus the opt-in episodic visual index (ROADMAP #7).
  Frames -- not text -- are what the planner is shown of its own past.
- **Device agnosticism is a resolution step.** `device.py` answers "where
  does this model run" once; torch is not a dependency (per-platform build).

## Verification status (2026-09-10, macOS, extras sim + sim-warp + grasping + occupancy + llm)

- Historical pre-delivery baseline: **737 passed, 2 deselected (hardware),
  0 skipped**, ~4 min. This is not the final count after installer changes;
  current delivery checks are recorded in `docs/SPARK_DELIVERY.md`.
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

- Same-colour identical objects: since 2026-10-08 a camera frame is fused as
  a whole (`BeliefStore.update_frame`, instance-level association), so two
  identical props the DETECTOR returns as two detections stay two beliefs
  inside the old 8 cm gate, each as accurate as that prop alone (rendered
  MuJoCo twins against physics truth, `scripts/measure_same_colour_sweep.py`:
  top view resolved from 3.75 cm centre distance, the per-detection store
  merged every pair below 8 cm). Still one belief: props the detector returns
  as ONE detection — touching props, props that overlap in the image (the
  oblique probe view up to 5.5–7.0 cm), the default mock detector's one blob
  per colour at any distance (`instances: true` is opt-in), one detection
  drawn around both. New failure mode to watch: two disjoint detections of
  ONE object with no whole detection (a lid and a handle alone) are now two
  beliefs. Live on Isaac (6.2 PhysX, YOLOE, both demo cameras; see
  `docs/evidence/b31-isaac-same-colour-20261008/`): identical twins 5–9 cm
  apart were two beliefs in 9/10 runs (old store: 0/10), and the
  open-vocabulary 3-prop scene scores the same with either store
  (`memory.instance_association: false` stays the A/B switch). Different
  colours never fuse.
- Open-vocabulary detections of the robot itself: the workspace filter's base
  cylinder covers only the links near the base. Since 2026-10-08 fusion also
  consults the render self-mask when a frame carries one (Isaac with
  `CASCADE_ISAAC_PIXEL_MASK=1`). A detection more than half robot pixels is
  dropped, and the robot's pixels never reach 3D. Live, the bare scene's
  phantom rate went from 17.9–19.6 % to 0 %
  (`docs/evidence/b32-fusion-self-mask-20261008/`). Frames without that mask
  (the real rig) rely on the cylinder alone unless
  `workspace_filter.link_self_mask` is on (B39, 2026-10-09). That mask is drawn
  from the URDF collision geometry at the nearest joint sample, and it is
  measured only on CPU against the URDF meshes themselves. Its comparison with
  the Isaac render mask (USD visual geometry) and any hardware measurement are
  still open. Its accuracy on hardware also depends on the hand-eye calibration
  and the joint offsets. A stale or unreadable joint state means no mask, i.e.
  the cylinder alone.
- Colour names differ between cameras: one object can sit on a hue band
  boundary (the Isaac bin is H 22 "orange" in the top camera, H 23 "yellow"
  in the side camera). Since 2026-10-08 (B32b) a belief keeps each camera's
  name (`source_colors`) and an observation is held to its own camera's name;
  a camera that never named a belief fuses a NEIGHBOUR name only at 3D box
  IoU >= `memory.neighbour_colour_iou` (0.75), and needs real-mask clouds on
  both sides. The box ignores each cloud's lowest centimetre (B32c): YOLOE's
  masks take in table pixels, and with them the bin's two live views scored a
  median of 0.744. Live on Isaac 6.2 PhysX the bare scene is now 3 beliefs for
  3 props in 5/5 runs (the one-name rule: 4 in 5/5), and a yellow prop in or
  next to the orange bin was never merged
  (`docs/evidence/b32b-colour-identity-live-20261008/`). Remaining failure
  modes:
  - a camera that flips its own name for one object still makes a second
    belief (held to its first name, as before);
  - two views of a SMALL object overlap less (a 3.5 cm cube: 0.66–0.85 IoU in
    the ray-cast) and may stay two beliefs;
  - a bleeding sliver of a short prop wholly inside the bin, seen only by a
    camera that never named the bin, reaches 0.63 in the ray-cast;
  - ~~pre-existing and unchanged: a camera that names a container and the prop
    inside it with the SAME colour fuses its view of the container into the
    prop's belief (the side camera's "yellow" bin into the yellow cube, 3 frames
    per run, identical with the one-name rule)~~ **landed 2026-10-09 (B40,
    measured on CPU, live A/B owed)**: fusion is size-consistent
    (`BeliefStore._size_ok`). A view's size is the robust horizontal diameter
    of its real-mask cloud; a belief remembers the largest one fused into it
    (`ObjectBelief.diameter_m`); a view more than 2× that AND more than 5 cm
    larger is not that object. On B32b's live clouds one object's views differ
    by ≤ × 1.29 and a container view is ≥ × 2.48 (+10.9 cm) the prop's largest
    view; 2.0 and 5 cm are estimates between those
    (`docs/evidence/b40-fusion-size-gate-20261009/`). It only refuses, and only
    with a real-mask cloud on both sides. What it cannot do, from the ray-cast:
    a belief BORN from a quarter of a container's view (3/4 hidden from its
    first frame) refuses the container's full view (up to × 2.64) and the bin
    becomes two beliefs; ≥ 4 px of mask bleed onto a container makes the view
    of a prop inside it × 2.5 its size, so it is refused from the prop's own
    belief (live YOLOE views of that prop: 0.066–0.074 m, no such bleed); and a
    prop's view fusing into the CONTAINER's belief is not handled (a smaller
    view always passes: it looks like an occluded view of the container).
    `memory.size_gate: false` restores the store before it, byte for byte.

  `memory.per_camera_colour: false` restores the one-name rule.
- Grip force is a stiffness proxy (kp scaling + stall detection), not a
  calibrated force loop. On the real reBot RS the pick close
  (`close_gripper_two_stage`) leaves the jaws pushing at a FIXED fraction of
  travel, so under MIT the holding torque kp·effort·(contact − target) grows
  with object width (Seeed: a paper cup was crushed). Since 2026-10-09 (B38)
  the arm profile's opt-in `gripper.max_contact_squeeze_rad` caps every jaw
  target after the first mechPos-detected contact at that many radians past
  it, so the steady torque is min(today's, kp·effort·cap). Objects whose
  squeeze already fits the cap keep today's exact commands; a jaw that reaches
  a capped target in free air gets today's target back
  (`robstride.close_two_stage_capped`, [REBOT_GRIP_SQUEEZE_CAP.md](REBOT_GRIP_SQUEEZE_CAP.md)).
  It ships `null` (today's close, byte-identical) and is measured only on a
  simulated jaw: the suggested 2.52 rad and the per-object torques depend on
  the unverified width map, the stage-1 scout squeeze before the first stall
  is not capped, and the user's hardware protocol in that doc must run before
  a value is set.
- Reach: the default reBot profiles keep the top-down-only workspace box
  (x 0.10..0.50, y ±0.30), part of which top-down grasps cannot reach
  (r > 0.45 m). The measured larger envelope and the tilted analytic
  candidates are opt-in (`*_reach` profiles, B45) and kinematic only: no
  contact physics or gripper-housing collision was modelled, the planner's
  lean is perpendicular to the jaw (an object that only fits a jaw pointing at
  the robot gains no reach), and live Isaac / real-rig picks in the new region
  are not yet measured ([REACH_ENVELOPE.md](REACH_ENVELOPE.md)).
- `RebotRSArm.disconnect()` cuts torque: park (`move_home`) first.
- The MCP server executes one tool call at a time by default; stops, cancels
  and pings are handled out-of-band by the receive side (never queued behind
  a motion), but every other request waits for the motion in flight. With the
  opt-in read-only lane (`mcp.readonly_lane`, B46) `world_state`,
  `robot_knowledge`, `verify_last_action` and `camera_snapshot` answer during
  a motion instead (marked `served_during_motion`); a second *motion*
  request, `describe_scene`, `get_observation` and every other tool still
  wait. A `run_program` call is one motion for its whole program (seconds per
  step): a stop or cancel interrupts it, a second request waits. The lane
  ships off and is measured on the mock stack only (no live Isaac or real-rig
  latency yet).
- The rendered-camera window (`RigViewer`) cannot open on macOS from the
  server (Cocoa needs the main thread; `opencv-python-headless` has no
  highgui); the MuJoCo physics window and the browser dashboard are the
  visuals there.
- Skill-library notes are retrieved by guard-word match on the task text
  (`aspire.retrieve`) by default, and only once promoted (recurred in ≥ 2
  distinct tasks). Embedding retrieval exists since 2026-10-07 but is opt-in
  (`memory.embedder`); with the dependency-free `hash` embedder it only adds
  inflections (grasping ~ grasp), and the SigLIP/CLIP text floors
  (`text_floor` 0.85, CLIP `text_image_floor` 0.25) are NOT calibrated with
  real weights -- measure them on the GPU host before relying on them. The
  same opt-in embedder ranks stored programs (B42, `ProgramLibrary.ranked`)
  under the same floor-or-guard rule and the same caveat; keyword overlap
  stays the default.
- The opt-in VLA executor (`grasp.executor: vla`, B49) is measured only
  against its protocol stub on the mock stack: no real policy has run
  through it, and none post-trained on this arm exists (public LingBot /
  openpi checkpoints are other embodiments, so their actions here are
  expected to be refused by the harness or to miss). Chunks execute as
  harness-gated per-action moves that settle at every action
  (quasi-static), not at the policy's native control rate; the observation
  is the manipulation camera only (no wrist/side views); there is no grasp
  exemption, so a policy grasp of a very low object can be refused where
  the analytic descent is allowed; the route refuses rigs with the
  observed-finger gate, a native motion planner (cuMotion profiles) or a
  payload-tracking map. The `vla_policy` capability cell reflects the
  startup probe and the last episode's connect, and nothing re-probes in the
  background: a policy server started after a failed probe needs a runtime
  restart (MCP hosts see the label-grasp tools withheld until then).
- Visual recall indexes motion frames and the crops of objects a call
  LOCALIZED by default; crops of every committed watcher detection
  (`memory.visual_recall_detections`) and persistence across restarts
  (`memory.persist_episodic`) exist since 2026-10-09 (B43) but are opt-in
  and CPU-measured only -- the per-tick cost with a real detector and
  embedder, and a restart on the booth, are not measured yet. Text queries
  ("looks like X") need a joint embedder; no semantic recall quality has
  been measured with real weights. Action-object consolidation keys on the
  normalized label: a detector label flicker (bottle/toy) stays two
  objects, by design.
  LOCALIZED, not every detection the watcher sees; it is in-process
  (task-scale horizon, lost on restart). Text queries ("looks like X") need
  a joint embedder; no semantic recall quality has been measured with real
  weights. Action-object consolidation keys on the normalized label: a
  detector label flicker (bottle/toy) stays two objects, by design.
- Endpoint overrides are CPU-tested only (spies on the bridge
  `create_connection` and the ZMQ `connect`): no live Isaac run on private
  ports yet. `launch.sh --graspgenx external` checks the server at
  `127.0.0.1:$CASCADE_GRASPGENX_PORT` even when `CASCADE_GRASPGENX_HOST`
  points the runtime elsewhere, and `scripts/demo_proof.py` still reads
  `CASCADE_BRIDGE_PORT` with its own `int()`.
- Tier-2 clauses are cut only at sequence connectives (`split_subgoals`,
  B37). A bare "and" is no clause boundary anywhere in the fast tier, so
  "move the red cube to the front-left of the table and move the blue cube to
  the front-left of the table" still replays the first command's habit
  (measured 0.951) and runs half of it; within one clause the hashed
  bag-of-words stays blind to word order. The opt-in programs tier (2.5) is
  not covered by the clause rule: it OFFERS promoted programs by keyword
  overlap (one shared content word) or, with `memory.embedder`, text
  similarity (B42 floor-or-guard), the brain (or the MCP chat host calling
  `run_program`) sees the whole task and picks, and a reused program's
  success means every step it ran was confirmed, not that it covered every
  clause -- a one-clause program reused for a sequence would report the same
  half-command success the LLM tier's `task_done` could.
- The launcher judge pass (B44) is opt-in and has only been exercised on
  CPU (fake judge, recorded traces, a stub OpenAI-compatible endpoint): no
  live Isaac proof turn has been judged by a real VLM or GRM yet, so no
  judge-vs-physics agreement is claimed for this rig. It judges only the
  proof turn's `pick_and_place` rows, and the shipped `eval.judge` targets a
  frontier model through the OpenClaw gateway -- a local judge needs
  `CASCADE_JUDGE_CONFIG`.

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
counts to these derived numbers, so a drift there fails the suite. The
MCP count is the FULL catalog; a running server lists it minus what the
rig's capability matrix withholds (`world_state.tools_withheld` names them,
e.g. 40 on a single-arm rig), so compare `mcp probe --json` against
`41 - len(tools_withheld)`, not against 41.
