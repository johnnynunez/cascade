# Roadmap

## Near term (before the demo)

- **Newton upstream issue (2026-07-19).** Manipulation contacts are broken
  at the PARSER level on the 6.0 develop build: identical failure under
  mjcwarp default, `use_mujoco_contacts=true` and XPBD (constant +3.7 cm
  float even box-vs-box, fingers pass through objects, boot NaN, "Triangle
  pair buffer overflowed"). Repro = `scripts/physics_probe.py --engine
  newton`. File against isaac-sim/IsaacSim with the probe reports in
  /tmp/probe_newton_*.json. Demo manipulation stays on PhysX (full battery
  green) until fixed.
- **Wrist cam follow-ups.** Validate the eye-in-hand extrinsics during a
  real grasp (reproject wrist depth of the target object against the
  physics-truth pose mid-descent); consider serving the wrist stream a
  narration highlight ("what the gripper sees") on the dashboard; on the
  real rig map `isaac_wrist.yaml` to the physical D435i + hand-eye calib.
- **Sim perception flakiness** (separate campaign): YOLOE misses the YCB
  banana on some boots and label-flickers the soup can (bottle/toy);
  belief 3D positions themselves verified ±3 mm against physics truth.

- **Persistent spatial memory (Johnny's idea, 2026-07-18).** BeliefStore
  already gives in-session object permanence (visible→remembered, EMA
  fusion, grasp-from-memory fallback); add save/load (JSON with wall-clock
  timestamps, loaded as "remembered") so the world model survives restarts,
  plus action↔object consolidation on top of ExperienceMemory. "Even if I
  don't see it, I roughly know where it is — like a human."
- **Straight-up spawn on the local tuned Isaac asset.** Blocked: drive
  travel from q=0 sweeps the props; joint-state authoring and tensor
  teleports NaN the solver on this asset (custom fixed-joint stack).
  Upstream asset gets it properly via Seeed-Projects/reBot-Isaacsim#9.
  Investigate the -plus asset's root joint / articulation root config.

- **GraspGen-X backend (integrated 2026-07-18, first-light verified).**
  `grasp.backend: graspgenx` sends the fix's base-frame object cloud to the
  GraspGen-X ZMQ server (`scripts/serve_graspgenx.sh`, own venv
  `~/Projects/demo/.graspgenx`, checkpoints in `GraspGenX/ext/`) and gets
  ranked 6-DoF grasps back (~1.2 s for 100 samples on the GB10); OBB stays
  as automatic fallback and additional IK candidates. TODO: (1) calibrate
  `tip_offset_m` in Isaac Sim (gripper-base -> reBot jaw center), (2) author
  reBot sweep-volume params (12 numbers, see GraspGenX "Integrating a New
  Gripper") instead of borrowing franka_panda, (3) use `infer_scene_pc` for
  collision-aware grasps in clutter.

- **Onsite bring-up checklist**
  1. `sudo ip link set can0 up type can bitrate 1000000`; kill any
     motorbridge-gateway/Studio.
  2. Re-verify RS gripper travel + stall torque; update
     `configs/arms/rebot_rs.yaml` (open/closed rad, kp, max_width_m).
  3. Hand-eye calibration: run the baseline repo's `collect_handeye_eih.py`
     (eye-in-hand) or measure the static mount, point the camera profile's
     `extrinsics` at it.
  4. Fit the table plane once (`DepthProvider.fit_table_plane`) and set
     `table_z` / workspace AABB for the physical setup.
  5. `pytest -m hardware` (read-only), then first motions with
     `--arm rebot_rs` at low `max_joint_vel`.
- **Local LLM**: run `scripts/serve_qwen_llamacpp.sh` (Qwen3.6-27B GGUF, MTP
  speculative decoding) and rehearse with `--llm local_qwen` so the demo has
  a no-internet fallback. vLLM variant in `serve_qwen_vllm.sh`.
- **Visual embedder for memory**: plug a CLIP/SigLIP image encoder into
  `EpisodicMemory(embed_dim=...)` + crops per detection, enabling
  "the thing that looked like X" recall through the TurboQuant index.

## Mid term

- **VLA policy backend**: LingBot-VLA-v2 exposes a websocket policy server
  (msgpack-numpy, `infer(obs) -> action chunks`). Add a `VLAExecutor` behind
  the skill API so `grasp_object` can be served either by the deterministic
  OBB pipeline or by a language-conditioned policy; the agent layer stays
  unchanged. Spark caveats: flash-attn must build for aarch64+Blackwell or
  the two hardcoded attention impls patched to SDPA.
- **GraspNet-class 6-DoF grasps**: the baseline's graspnet path needs the
  vendored sdk + checkpoint-rs.tar and THC-era CUDA patches for torch 2.x.
  Alternative: a modern 6-DoF grasp head served as a `plan_grasp` skill.
- **Skill-library growth loop** (ASPIRE): after each failed→repaired run,
  distill the fix into `skills_library/*.md` (schema already implemented);
  load `relevant(task)` entries into the agent context.

## Long term: sim2real with NuRec / Isaac

- **Isaac Sim bridge (scaffolded 2026-07-18, live validation PENDING).**
  `scripts/isaac_bridge.py` (runs inside Isaac Sim's Python) serves RGB-D
  frames + articulation control over newline-JSON TCP; `--cameras isaac
  --arm isaac` runs the identical demo against the sim. Protocol + client
  backends are fully covered by fake-server tests; the sim side is written
  against the Isaac Sim 5.x core API and must be validated on first launch:
  (1) reBot USD path + articulation joint ORDER vs the RS URDF, (2) gripper
  DOF index and units (config assumes RS export open=-6.8), (3) camera
  intrinsics/orientation vs the cam0 extrinsics in configs/cameras/isaac.yaml,
  (4) position-target gains (use the tuned values from the gain-tuner work).
  The Downloads/isaac-companion-v1-franka pack's `isaac-sim-remote` skill
  (TCP Python-exec extension) is a good live-debugging companion for this.
- `reBot-Isaacsim` + `sim2real-rebot-devarm` already provide USD assets, a
  real→sim UDP mirror, and an HTTP control daemon for this arm.
- **NuRec (neural reconstruction)**: reconstruct the actual demo tabletop
  into a photoreal digital twin; rehearse perception + grasping against the
  twin (domain gap ≈ 0 for the camera), then replay on the real rig. The
  ASPIRE sim2real recipe applies directly: skills discovered in sim transfer
  as *in-context guidance* for the real-robot agent, not as weights.
- **Online adaptation** (Agentic-VLA proper): once a VLA executor exists,
  their GRPO + reward-synthesis loop is the path to improving it from demo
  logs; the trace format already captures per-primitive evidence needed for
  progress rewards.
