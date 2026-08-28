# Architecture

## Design position

Three reference systems informed this design:

- **ASPIRE** (NVIDIA GEAR, 2026): a coding agent programs a robot through a
  *curated primitive API* whose every call records multimodal evidence
  (their ablation: the trace-exposing execution engine alone lifted success
  14% → 62%). We replicate the API surface (`get_observation`, `localize`,
  `plan_grasp`-equivalent, `solve_ik`, motion + gripper primitives), the
  per-call trace format (`trace.jsonl` + before/after keyframes), the
  debugging heuristics (gripper-width-after-close as the grasp-success
  signal), and the markdown skill library (store implemented in
  `skills/library.py`; the load-into-context loop is still on the ROADMAP).
- **Agentic-VLA** (ICML 2026): LLM sub-goal decomposition, a zero-shot VLM
  "exploration critic" that emits ONE spatial suggestion per consultation
  (we reuse their prompt near-verbatim), and embedding-indexed experience
  memory. Their loop is training-time; we repurposed the components into an
  inference-time execute→verify→recover loop.
- **Claude plays robotics** (Anthropic, 2026): control-interface level
  dominates model choice — LLMs commanding high-level primitives massively
  outperform LLMs near the metal; one LLM turn costs 2–15 s, so anything
  routine must not wait on the model; only the newest frames matter; and
  structured text state beats extra image context. This drove the
  livestreaming redesign below.

## Livestreaming + the reflex fast path (2026-07-18)

The demo is now an always-on system rather than a per-task pipeline,
modelled on how humans act: perception never stops, routine commands are
reflexes, and deliberation is reserved for the novel.

```
N cameras ──CameraStream (thread each, latest-frame slot, drop-stale)
   │            │
   │            ├── WorldWatcher (thread, ~3 Hz): detector + HSV color tag
   │            │     └─> BeliefStore (thread-safe): label+color+3D+freshness
   │            └── StreamServer (MJPEG dashboard): camera grid + narration
   │                  feed (episodic events) + object table   -> browser
   │
chat command ("pick and place pink object")
   ├─ tier 1 REFLEX   template grammar -> skill calls          (~µs)
   ├─ tier 2 HABIT    experience memory (hashed-BoW cosine)    (~ms)
   └─ tier 3 LLM      the original orchestrator loop           (2-15 s/turn)
        all tiers execute through the same safety-gated SkillRuntime
```

Key mechanisms:

- **Latest-slot streaming, never queues.** A consumer always gets the
  newest frame; nothing falls behind the sensor (`perception/stream.py`).
- **Warm world model.** The WorldWatcher fuses every rig camera with depth
  + extrinsics into the BeliefStore continuously, so command resolution is
  a dictionary lookup, not an observe→detect round trip. It pauses during
  arm motion (the held object must not be re-fused mid-air) and heartbeats
  the safety watchdog (`perception/world.py`).
- **Color without CLIP.** Detections are color-named from median mask HSV
  (`perception/colors.py`), stored on beliefs, and matched against color
  words in queries — "pink object" works on a closed-set COCO detector.
  When the ultralytics CLIP fork lands, open-vocab prompts slot in and the
  HSV tag becomes redundant metadata. The VLM can still be consulted (tier
  3 / advisor), but never on the hot path.
- **One-call pick-and-place.** `skill_pick_and_place` = resolve → grasp →
  place (named object or drop zone) → home, with stage timings in the
  result; both grasp and place stages keep retrying on fresh perception
  (re-home, re-scan, re-plan) until success or the persistence budget runs
  out (`grasp.persist_seconds: 120` / `grasp.max_pick_attempts: 8` in
  configs/demo.yaml). Hermes executes the whole command as a single MCP
  tool call; compute overhead is <100 ms and total time is arm motion time
  (`skills/runtime.py`).
- **LazyArm.** The MCP gateway pre-warms cameras/detector/world model at
  startup, but motors are not touched until the first motion command
  (`control/lazy_arm.py`) — starting a chat server must not power a robot.
- **Social skills.** wave / point_at / handover / sort_by_color /
  describe_scene / count_objects / move_relative give booth visitors a
  vocabulary beyond pick-and-place; all deterministic, all harness-gated.

We deliberately did NOT build a VLA-policy-in-the-loop baseline first: the
deterministic skill stack is debuggable, safety-gateable, and runs offline.
The LingBot-VLA-style websocket policy server slots in later as an
alternative *executor* behind the same skill API (see ROADMAP).

## Module map

```
src/cascade/
├── types.py            Frame / Detection / ObjectFix / Grasp / RobotState
├── config.py           YAML profiles (cameras/, arms/, llm/) merged into one Cfg
├── perception/
│   ├── camera_base.py  CameraBase ABC + factory; Frames carry METRIC depth
│   ├── realsense_camera.py   D4xx + L515 (local fork's pyrealsense2)
│   ├── opencv_camera.py      RGB-only UVC
│   ├── mock_camera.py        synthetic tabletop + npz replay (cascade-record)
│   ├── isaac_camera.py       Isaac Sim bridge frames (RGB-D + per-frame T_base_cam)
│   ├── depth_provider.py     sensor → mono plugin → table-plane ray-cast
│   ├── detector.py           open-vocab YOLOE/YOLO-World + MockDetector
│   ├── grounding.py          Extrinsics + localize_object (color/near-aware)
│   ├── vlm_ground.py         second-chance VLM grounder (detector-miss path)
│   ├── colors.py             mask HSV → color name; color-query parsing
│   ├── stream.py             CameraStream (pump thread) + CameraRig (N cams)
│   └── world.py              WorldWatcher: always-on detection → beliefs
├── memory/
│   ├── turboquant.py   TurboQuant-style rotation+scalar quantizer (numpy)
│   ├── vector_index.py asymmetric top-k cosine over 4-bit codes
│   ├── episodic.py     10-15 s ring: events, thumbnails (embeddings opt-in)
│   ├── beliefs.py      object permanence (visible/remembered states)
│   └── grasp_memory.py persisted per-object grasp-outcome prior
│                       (re-ranks candidates + nudges grasp z)
├── device.py           compute-device resolution: auto CUDA/ROCm → MPS → CPU,
│                       explicit-but-absent degrades with a warning
├── control/
│   ├── kinematics.py   Pinocchio FK/IK, explicit URDF/USD (assets/), DLS+restarts,
│   │                   N controlled joints, optional task weighting
│   ├── usd_model.py    USD-physics → URDF translation (no aarch64 usd-core)
│   ├── arm_base.py     min-jerk streaming, feedback-based settling + make_arm
│   ├── mock_arm.py     kinematic sim + gripper object-stop emulation (any DoF)
│   ├── lazy_arm.py     defer motor bring-up until the first motion command
│   ├── isaac_arm.py    Isaac Sim articulation over the TCP bridge
│   ├── mujoco_arm.py   any MJCF in MuJoCo physics, joints addressed BY NAME
│   ├── feetech.py      Feetech SCS/STS servo-bus protocol (pure framing)
│   ├── feetech_arm.py  SO-101 & co: Feetech servos over USB serial, SYNC_WRITE
│   │                   streaming (UNVERIFIED on hardware)
│   └── rebot_rs_arm.py real RS arm: motorbridge CAN, mechPos param reads,
│                       stall-aware two-stage gripper close
├── safety/harness.py   fail-closed gate for every waypoint + SafeArm wrapper
├── grasping/
│   ├── obb_grasp.py    base-frame OBB grasps (short-axis yaw, height frac)
│   ├── graspgenx_backend.py  learned 6-DoF grasps via ZMQ (default backend,
│   │                   silent OBB fallback when the server is down)
│   ├── selector.py     width + IK walk + harness pre-vet callback
│   │                   (dense sampling along the descent segment)
│   └── force.py        material → two-stage close profiles
├── agent/
│   ├── llm.py          OpenAI-compat (cloud + local Qwen) / Anthropic / Mock
│   ├── prompts.py      system persona, decomposition, advisor (paper prompt)
│   ├── advisor.py      failure-triggered VLM consultation
│   ├── reflex.py       tier-1 command grammar + tier-2 experience memory
│   ├── trace.py        ASPIRE trace logger
│   └── orchestrator.py reflex-first agent loop + TaskReport
├── skills/
│   ├── runtime.py      the curated tool surface + JSON schemas (31 specs:
│   │                   30 skills incl. pick_and_place + social skills,
│   │                   plus the loop-terminator task_done -- re-derive via
│   │                   `TOOL_SPECS` in this file, counts drift fast)
│   └── library.py      learned-skill markdown store (loading loop: ROADMAP)
├── sim/bridge_client.py  newline-JSON TCP client for scripts/isaac_bridge.py
└── apps/               demo.py CLI (build_runtime = the composition root),
                        record.py capture, viewer.py (cascade-view live RGB+D),
                        live_view.py (RigViewer window: one row per camera,
                        RGB beside its depth colormap, + draw helpers),
                        stream_server.py (MJPEG dashboard + narration +
                        POST /task), mcp_server.py (Hermes/Claude Code
                        front-end: skill runtime over MCP stdio,
                        perception pre-warm, lazy arm)
```

## Key decisions

**Metric depth in the Frame.** The baseline passed uint16 depth in sensor
units; the L515 (0.25 mm/unit) silently breaks any code written against
D4xx's 1 mm/unit. Converting at the camera boundary makes every downstream
consumer unit-safe.

**Grasps planned in the base frame, not the camera frame.** The baseline
derived approach direction from the camera ray, so grasp quality depended on
camera mounting. Here mask points are lifted to 3D, transformed to base, and
the OBB there gives yaw + width + height — the camera pose only affects
visibility, not grasp geometry. That is what "camera-agnostic" means
operationally.

**Learned grasps by default, analytic always available.** Since 2026-07-18
`grasp.backend: graspgenx` (configs/demo.yaml) sends the object's base-frame
cloud to a GraspGen-X ZMQ server (`scripts/serve_graspgenx.sh`, :5556,
~1.2 s for 100 samples on the GB10; reBot jaws passed as a swept volume for
cross-embodiment conditioning, `tip_offset_m` 0.098 empirical) and prepends
the ranked 6-DoF grasps to the analytic OBB candidates, which are always
computed. Any server error degrades silently to OBB — the booth must never
stall on a dead model server. The full pipeline:

```
localize ─▶ ObjectFix (base-frame OBB)
   ├─▶ GraspGen-X candidates (ZMQ, learned 6-DoF) ──┐ prepended; any server
   └─▶ OBB candidates (analytic, always computed) ──┤ error → OBB only
                                                    ▼
   grasp-outcome memory re-rank + z-nudge (~/.cascade/grasp_memory.json)
                                                    ▼
   select_grasp: jaw-width filter ▸ IK (pregrasp, then grasp seeded from
   it) ▸ harness pre-vet (pregrasp WITHOUT the exemption cylinder, then 7
   samples along q_pre→q_grasp with the exemption floored at table_z−0.06)
                                                    ▼
   re-home (forces elbow-up IK branch) ▸ pregrasp ▸ exempted descent ▸
   two-stage stall-aware close ▸ lift ▸ air-grasp check (jaw width fraction)
```

**Own kinematics wrapper.** reBotArm_control_py's kinematics silently loads
the URDF named in its *global* config file, ignoring the hardware YAML you
pass (DM URDF loaded for the RS arm = wrong tool frame). We load whichever
URDF the arm profile names explicitly. IK: damped least squares in the LOCAL
frame with joint-limit clamping and random restarts (matches the SDK's math,
minus the config hazard).

**The arm's shape is data, not code.** `n_joints`, joint limits, `home_q`,
`handover_q`, the tool-frame column order (`tool_axis_order`), gripper travel
and the reachable envelope all come from the profile; `SkillRuntime` fetches
joint-space keyframes through `_profile_q`, which *requires* them rather than
falling back to a default — a 6-element reBot home pose broadcast onto a
5-DoF arm is exactly the kind of silent error that ends in a bent link. The
shipped arms span 5, 6 and 7 DoF with no branching in the skill layer.

Under-actuated chains get `ik_task_weights` (a per-task-DOF weight vector, in
world axes) if they need it, but *no shipped profile does* — including the
5-DoF SO-101, whose wrist-roll axis is collinear with the tool approach, so
for a vertical approach it behaves like a full-pose arm and plain 6-DoF IK
solves its workspace at a 0.94 rate. The DOF it lacks only shows up for a
tilted approach; leaving IK unweighted keeps failure as the honest gate
rather than accepting poses the wrist cannot hold.

**Feedback, not sleep.** The baseline's motions were `sleep(duration + 0.6)`.
Every backend reports real positions instead: RS motors via `mechPos` (0x7019)
param reads (motorbridge's `get_state()` never decodes the type-0x18 report
frames — verified on this rig), Feetech servos via `Present_Position`, MuJoCo
via `qpos`. Settling is `max|q - q_target| < tol` with a timeout, and the
tolerance is per profile because a finite-gain position actuator (MuJoCo,
Isaac) genuinely arrives late where the kinematic mock arrives exactly.

**Device agnosticism is a resolution step, not a policy.** `device.py` is the
single place that answers "where does this model run": `auto` probes the host
(CUDA/ROCm → Apple MPS → CPU), and an explicit device the machine lacks
degrades to the best available with a warning. That last rule matters because
configs travel — a rig config opened on a laptop should cost a slow run, not a
dead session, which is the same booth rule the grasp and occupancy backends
follow. torch is not a dependency: its correct build is per-platform.

**Fail-closed safety.** The SDK enforces nothing outside its IK. Our harness
gates every waypoint of every streamed motion; grasp descents happen inside
an explicit exemption cylinder around the target so "don't touch the table"
and "grasp the object on the table" coexist. Auto-scaled durations keep
planned min-jerk peaks under the velocity cap; the harness remains the
backstop.

**Memory is structured first, embeddings second.** The agent's recall tools
work on labels/time/positions (BeliefStore) — deterministic and testable.
The TurboQuant index (dim-agnostic, 4-bit codes, asymmetric search) has one
live consumer today: the tier-2 ExperienceMemory (hashed bag-of-words habit
recall). Episodic-memory embeddings are opt-in via `embed_dim` and stay OFF
in the shipped demo (`recall_similar` returns `[]`) until a visual embedder
is added (ROADMAP). A third store, GraspOutcomeMemory
(`memory/grasp_memory.py`), persists dimensionless per-object grasp features
across sessions and re-ranks planner candidates on the grasp hot path. The
quantizer is pure numpy so the demo carries no exotic dependency (pip
`turbovec` can slot in behind the same interface).

**Gripper force ≈ commanded stiffness.** The RS gripper is one MIT-mode
motor; commanded kp bounds stall torque, so material profiles scale effort
and close depth, and stall detection (mechVel) doubles as the grasp-success
signal. Current-loop force calibration is an onsite task (see rebot_rs.yaml
warnings).

## Verification status (2026-07-20, this rig)

- 161 unit/integration tests collected, 159 selected by default (`pytest
  -q`; 2 hardware-marked), including a full mock-stack grasp-and-place e2e
  that exercises config → perception → beliefs → grasp planning → IK →
  safety-gated streaming → gripper verification → memory → trace files.
- Two adversarial multi-lens review passes, all critical/major findings
  fixed: 2026-07-16 (4 reviewers × skeptic verification, 34 agents, 29
  confirmed defects) and 2026-07-18 on the livestreaming redesign (45-agent
  workflow, 33 confirmed findings). Regression-pinned in
  `tests/test_review_regressions.py`: inverted spatial hints, baseline
  hand-eye npz loading, watchdog tripping mid-grasp, below-clearance
  recovery deadlock, OpenAI/Anthropic tool-protocol violations, unbounded
  image-context growth (air-grasp detection is pinned in
  `tests/test_orchestrator_e2e.py`); v2 findings in
  `tests/test_review_regressions_v2.py`; the remaining v1 fixes
  (IK-vs-harness margin invariant, stale-CAN-feedback masking bounds,
  soft-stop torque hold) were pinned late in
  `tests/test_review_regressions_v3.py` (2026-07-20) against fakes — no
  hardware needed. A third pass (2026-07-20, booth prep: 40-agent workflow,
  34 confirmed findings) reviewed the MCP stop channel and booth tooling;
  its stop-channel defects (stop lost during runtime build, cancellation
  TOCTOU, reader-thread crash on non-object frames, no staff reset path)
  are pinned in `tests/test_review_regressions_v4.py`. A fourth pass (same
  day, 18 agents, 13 confirmed) reviewed the booth-roadmap batch (CASCADE_BOOTH
  overlay, dashboard routes, MCP-mode reflex chat); its critical —
  dashboard-chat motion could run concurrently with an MCP tool call — is
  fixed by `_exec_lock` and pinned in the same v4 file.
- Live L515 streaming through the full camera stack (hardware-marked test).
- Live RobStride mechPos param reads for all 7 motors over can0 (read-only).
- Live YOLO inference (CUDA, GB10) on L515 frames with metric depth lookup.
- Isaac Sim bridge exercised live on PhysX (scripted picks via
  `scripts/dashboard_runner.py` / `night_runner.sh`; `physics_probe.py`
  battery green). Newton engine blocked upstream — see ROADMAP.
- GraspGen-X backend integrated with first-light verification in sim;
  tip-offset / sweep-volume calibration still open (see ROADMAP).
- NOT yet exercised: real-arm motion (needs onsite gripper re-verification
  and hand-eye calibration), local Qwen serving (scripts provided; the
  `local_qwen.yaml` profile was reconciled with the serve scripts on
  2026-07-20 — keep `model:` in sync per the README note).

## Known limitations (accepted for the baseline)

- BeliefStore merges same-label observations within 8 cm (EMA) — two
  identical objects closer than that can blur into one belief.
- `RebotRSArm.disconnect()` goes through the SDK's disable_all: park the arm
  (move_home) before shutting down or it will fall under gravity.
- Grip force is a stiffness proxy (MIT kp scaling + stall detection), not a
  calibrated force loop; current-based calibration is an onsite task.
- The advisor/decompose prompts are single-frame; no video context yet.
- The MCP server is single-threaded: a long `pick_and_place` blocks
  `emergency_stop` and every other tool until it returns (SIGINT e-stop
  works; a bypass stop channel is on the ROADMAP).
