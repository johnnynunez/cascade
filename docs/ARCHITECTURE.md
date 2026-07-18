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
  signal), and the markdown skill library.
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
- **One-call pick-and-place.** `skill_pick_and_place` = resolve → grasp
  (one deterministic retry) → place (named object or drop zone) → home,
  with stage timings in the result. Hermes executes the whole command as a
  single MCP tool call; compute overhead is <100 ms and total time is arm
  motion time (`skills/runtime.py`).
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
src/wrc_demo/
├── types.py            Frame / Detection / ObjectFix / Grasp / RobotState
├── config.py           YAML profiles (cameras/, arms/, llm/) merged into one Cfg
├── perception/
│   ├── camera_base.py  CameraBase ABC + factory; Frames carry METRIC depth
│   ├── realsense_camera.py   D4xx + L515 (local fork's pyrealsense2)
│   ├── opencv_camera.py      RGB-only UVC
│   ├── mock_camera.py        synthetic tabletop + npz replay (wrc-record)
│   ├── depth_provider.py     sensor → mono plugin → table-plane ray-cast
│   ├── detector.py           open-vocab YOLOE/YOLO-World + MockDetector
│   ├── grounding.py          Extrinsics + localize_object (color/near-aware)
│   ├── colors.py             mask HSV → color name; color-query parsing
│   ├── stream.py             CameraStream (pump thread) + CameraRig (N cams)
│   └── world.py              WorldWatcher: always-on detection → beliefs
├── memory/
│   ├── turboquant.py   TurboQuant-style rotation+scalar quantizer (numpy)
│   ├── vector_index.py asymmetric top-k cosine over 4-bit codes
│   ├── episodic.py     10-15 s ring: events, thumbnails, embeddings
│   └── beliefs.py      object permanence (visible/remembered states)
├── control/
│   ├── kinematics.py   Pinocchio FK/IK, explicit URDF (assets/), DLS+restarts
│   ├── arm_base.py     min-jerk streaming, feedback-based settling
│   ├── mock_arm.py     kinematic sim + gripper object-stop emulation
│   ├── lazy_arm.py     defer motor bring-up until the first motion command
│   └── rebot_rs_arm.py real RS arm: motorbridge CAN, mechPos param reads,
│                       stall-aware two-stage gripper close
├── safety/harness.py   fail-closed gate for every waypoint + SafeArm wrapper
├── grasping/
│   ├── obb_grasp.py    base-frame OBB grasps (short-axis yaw, height frac)
│   ├── selector.py     width + IK feasibility walk (pregrasp AND grasp)
│   └── force.py        material → two-stage close profiles
├── agent/
│   ├── llm.py          OpenAI-compat (cloud + local Qwen3.6) / Anthropic / Mock
│   ├── prompts.py      system persona, decomposition, advisor (paper prompt)
│   ├── advisor.py      failure-escalated VLM consultation
│   ├── reflex.py       tier-1 command grammar + tier-2 experience memory
│   ├── trace.py        ASPIRE trace logger
│   └── orchestrator.py reflex-first agent loop + TaskReport
├── skills/
│   ├── runtime.py      the curated tool surface (20 skills incl.
│   │                   pick_and_place + social skills) + JSON schemas
│   └── library.py      learned-skill markdown store
└── apps/               demo.py CLI, record.py capture, stream_server.py
                        (MJPEG dashboard + narration), mcp_server.py (Hermes/
                        Claude Code front-end: skill runtime over MCP stdio,
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

**Own kinematics wrapper.** reBotArm_control_py's kinematics silently loads
the URDF named in its *global* config file, ignoring the hardware YAML you
pass (DM URDF loaded for the RS arm = wrong tool frame). We load the RS URDF
shipped in `assets/` explicitly. IK: damped least squares in the LOCAL frame
with joint-limit clamping and random restarts (matches the SDK's math, minus
the config hazard).

**Feedback, not sleep.** The baseline's motions were `sleep(duration + 0.6)`.
On the RS motors, motorbridge's `get_state()` never decodes the type-0x18
report frames, so live positions come from `mechPos` (0x7019) param reads —
verified on this rig. Settling is `max|q - q_target| < tol` with a timeout.

**Fail-closed safety.** The SDK enforces nothing outside its IK. Our harness
gates every waypoint of every streamed motion; grasp descents happen inside
an explicit exemption cylinder around the target so "don't touch the table"
and "grasp the object on the table" coexist. Auto-scaled durations keep
planned min-jerk peaks under the velocity cap; the harness remains the
backstop.

**Memory is structured first, embeddings second.** The agent's recall tools
work on labels/time/positions (BeliefStore) — deterministic and testable.
The TurboQuant index (dim-agnostic, 4-bit codes, asymmetric search) is wired
for CLIP-style embeddings when a visual embedder is added; the quantizer is
pure numpy so the demo carries no exotic dependency (pip `turbovec` can slot
in behind the same interface).

**Gripper force ≈ commanded stiffness.** The RS gripper is one MIT-mode
motor; commanded kp bounds stall torque, so material profiles scale effort
and close depth, and stall detection (mechVel) doubles as the grasp-success
signal. Current-loop force calibration is an onsite task (see rebot_rs.yaml
warnings).

## Verification status (2026-07-16, this rig)

- 57 unit/integration tests green (`pytest -q`), including a full
  mock-stack grasp-and-place e2e that exercises config → perception →
  beliefs → OBB grasp → IK → safety-gated streaming → gripper verification →
  memory → trace files.
- An adversarial multi-lens review (4 reviewers × skeptic verification per
  finding, 34 agents) surfaced 29 confirmed defects — all critical/major
  ones fixed and regression-tested (`tests/test_review_regressions.py`):
  inverted spatial hints, baseline hand-eye npz loading, watchdog tripping
  mid-grasp, below-clearance recovery deadlock, IK-vs-harness margin
  mismatch, stale-CAN-feedback masking, gripper stall/air-grasp logic,
  free-fall on soft stop, OpenAI/Anthropic tool-protocol violations,
  unbounded image context growth.
- Live L515 streaming through the full camera stack (hardware-marked test).
- Live RobStride mechPos param reads for all 7 motors over can0 (read-only).
- Live YOLO inference (CUDA, GB10) on L515 frames with metric depth lookup.
- NOT yet exercised: real-arm motion (needs onsite gripper re-verification
  and hand-eye calibration), local Qwen3.6 serving (scripts provided).

## Known limitations (accepted for the baseline)

- BeliefStore merges same-label observations within 8 cm (EMA) — two
  identical objects closer than that can blur into one belief.
- `RebotRSArm.disconnect()` goes through the SDK's disable_all: park the arm
  (move_home) before shutting down or it will fall under gravity.
- Grip force is a stiffness proxy (MIT kp scaling + stall detection), not a
  calibrated force loop; current-based calibration is an onsite task.
- The advisor/decompose prompts are single-frame; no video context yet.
