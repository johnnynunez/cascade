# Roadmap

This file is a dated log: each "Landed <date>" section records what shipped,
the SOTA delta that motivated it, and what was deliberately NOT adopted.
Read top-down for status; the dated sections are history.

## Status at a glance (2026-09-10)

What a visitor gets today, in one command, on a laptop: `./run.sh` →
MuJoCo (or Isaac Sim when installed) + OpenClaw chat with 41 robot tools,
self-proven (runtime built, tools listed, brain answered, one pick
CONFIRMED by physics, scene reset). Every skill's effect is verified on an
independent channel, the planner sees its own history as images, and the
suite is 737 passed / 0 skipped. Unverified on hardware: real-arm motion,
the SO-101 serial driver, the ROS2 and Unitree backends.

Open, in priority order (details in the sections below):

1. **Real rig first motions** -- onsite checklist (CAN up, gripper travel,
   hand-eye, table plane), `pytest -m hardware`, then `--arm rebot_rs` at
   low velocity. Everything above the driver has been exercised in two
   simulators; the drivers have not.
2. **Persistence-loop leftovers** #2–#7 below (provisional held marker,
   per-task budget cap across tiers, handover/sort persistence, thin-object
   slip heuristic, fail-fast on over-width, the listed coverage gaps).
3. **Learned grasps for real**: run `serve_graspgenx.sh` (CUDA) instead of
   the protocol stub and calibrate `tip_offset_m` / the reBot sweep volume
   in Isaac; the stub only proves the wire.
4. **Wrist camera** extrinsics validated mid-descent against physics truth.
5. **Newton as the Isaac default** once `physics_probe.py --engine newton`
   passes on the real reBot asset (the synthetic-scene blocker is gone).
6. **Judge as a metric**: run `scripts/judge_run.py` over every launcher
   proof turn and keep the judge-vs-physics confusion matrix in the run
   summary, so a regression in the outcome pictures shows up as `fn`.
7. **Visual embedder** for episodic recall (`embed_dim`), and action↔object
   consolidation on top of ExperienceMemory (keys on text today).
8. **Multi-arm on physics**: `so101_left`/`so101_right` are mock; render a
   two-arm MuJoCo scene so the inter-arm gate is measured, not simulated.

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
  badges, 5 cm metric grid, TCP, and the top-down IK band drawn on the frame.
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
4. Envelope features are currently raw skill args; add derived features
   (TCP z at grasp, object height) so the learned ranges capture the real
   B601-RS constraint rather than a proxy.
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
  (`memory/envelope.py`). Not landed yet — see open follow-up #6 below.
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
  (`TOOL_SPECS` = primitives, `skills_library/*.md` = skills) but no
  `programs` tier — see open follow-up #8. Also notable, as a *contrast*
  and not a pattern to adopt: Waddle's described safety layer is a single
  `verify(...)` check with no rate-limiting or rollback protocol — thinner
  than this repo's harness-as-sole-authority design, worth stating
  explicitly rather than citing Waddle as a safety precedent.

Open follow-ups from this work:
6. **Pre-motion plausibility check (Human-CLAW).** Extend
   `agent/milestones.py`'s existing rate-limited `VERIFY_USER` critic
   pattern to run *before* dispatch for `_MOTION_SKILLS`
   (`skills/runtime.py:31`), not just post-hoc for milestone progress: ask
   "is this specific call, with these specific args, plausible given
   current beliefs/reachability?" and let it veto/substitute, the way
   Human-CLAW's verifier does. Reuses existing rate-limiting so it does not
   blow booth-clock budget. Deliberately not landed today: this touches the
   safety-critical motion-dispatch choke point in `SkillRuntime.execute()`,
   and per CLAUDE.md the harness must remain the sole authority that
   refuses motion — a verifier here has to be advisory-only (same booth
   rule as envelopes), and that needs a live-rig or at minimum a
   MockLLM-scripted test pass before landing, not a speculative edit to the
   motion path from a machine that cannot run the rig.
7. **HUG as a second grasp backend.** Add `grasp.backend: hug` alongside
   `graspgenx`, self-hosted the same way (a serve script + client mirroring
   `grasping/graspgenx_backend.py`), re-ranked by the same
   `GraspOutcomeMemory`. Whether HUG's cross-embodiment grasps clear this
   arm's IK envelope is untested — the point of landing it is to find out,
   not to assume it is better.
8. **A `programs` tier (Waddle).** cascade has primitives (`TOOL_SPECS`)
   and skills (`skills_library/*.md`, ASPIRE-distilled) but nothing above
   skills: an agent-composed, reusable, task-level script distinct from a
   one-off orchestrator run. Scoping question before landing: does a
   "program" get authored the same way ASPIRE distills a skill (diagnose a
   successful multi-skill run, persist it), or does the agent write one
   proactively? Needs a design pass, not a first draft in this file.

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
  with no cross-task gate, so one lucky repair is retrieved as if proven.
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
9. **Task-Specific Memory recipes (Harness-VLA v4).** Store successful runs
   with xyz replaced by `localize_object(label)+offset` queries and re-ground
   at replay; this is the shape for #8 and fixes tier-2's text keys. (M)
10. **ASPIRE cross-task promotion gate.** `agent/aspire.py`: promote a
    distilled skill only when seen in ≥2 distinct tasks (`occurrences`,
    `source_tasks`); today one lucky repair is retrievable as proven. (S)
11. **Pigey snapshot/restore + occlusion search** as composite skills over
    `BeliefStore` (`snapshot_scene`/`restore_scene`, `search_for_object`):
    the one demo beat visible from chat that no current skill covers. (M)
12. **Capability matrix → tool surface (Waddle).** Compute `_EXCLUDED_TOOLS`
    from what the rig can do (depth, sidecars, n_arms) instead of
    `CASCADE_HIDE_TOOLS` by hand. (S)
13. **`recall_step(n)` + a `stuck` outcome (RPent).** Trace keyframes already
    exist per step; expose them, and let motion skills return a human-
    actionable ask distinct from failure. (S)

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
15. **Wrist camera for the judge.** GRM's prompt reserves two wrist slots;
    the SO-101 has none, so both repeat the front view. A wrist `<camera>`
    in the MJCF scene (and `Frame` wrists in `build_images`) would exercise
    the model as trained. (S)
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
14. Tests now pin the reverse skill/spec mapping CLAUDE.md warned about for
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
SFT mixture (training); GR00T actor (VLA as executor was ruled out earlier);
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
  2. Exception between gripper close and held_object assignment leaves a
     physically held object logically unheld (reconcile only clears the
     opposite desync); consider a provisional held marker before close.
  3. Budget can multiply across tiers: fast-path burns persist_seconds,
     then a real-LLM tier can call pick_and_place again. Cap per task.
  4. handover / sort_by_color still single-attempt (inconsistent with
     pick_and_place persistence).
  5. _reconcile_held mistakes a legitimately-held VERY thin object
     (<4% jaw span ~ 3.6 mm) for a slip; booth objects are chunky.
  6. Fail fast when every grasp candidate exceeds jaw width (currently
     retries perception on an object-property error).
  7. Test-coverage gaps flagged: place-stage loop, deadline expiry,
     epoch fallback, z-clamp, exemption z_min through _in_cylinder,
     McpClient timeout is dead code.

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
- **Wrist cam follow-ups.** Validate the eye-in-hand extrinsics during a
  real grasp (reproject wrist depth of the target object against the
  physics-truth pose mid-descent); consider serving the wrist stream a
  narration highlight ("what the gripper sees") on the dashboard; on the
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
  the load-bearing part (see CLAUDE.md); `LOADED_MIN_AGE_S` guarantees a
  restored belief never reads as `visible`, and `_localize`'s existing
  3 s `belief_fallback_age_s` gate means it can never aim the jaws.
  STILL OPEN from this item: action↔object consolidation on top of
  ExperienceMemory (sub-goal-level credit now exists via
  `FastPlanner.note_subgoal_outcome`, but it keys on TEXT, not on objects).
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
  double)`, and that is NOT the learned model. TODO: (1) calibrate
  `tip_offset_m` in Isaac Sim (gripper-base -> reBot jaw center), (2) refine
  the URDF-derived reBot sweep-volume params in `configs/demo.yaml` with an
  Isaac Sim measurement (franka_panda remains only the no-sweep fallback),
  (3) use `infer_scene_pc` for collision-aware grasps in clutter.

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
  a no-internet fallback. vLLM variant in `serve_qwen_vllm.sh`. The profile
  (`configs/llm/local_qwen.yaml`) pins `model: Qwen/Qwen3.6-27B` to match
  both scripts (reconciled 2026-07-20); if you change the script's
  `MODEL`/`HF_REPO`, update the profile — llama.cpp ignores the requested
  name but vLLM rejects a mismatch, and `supports_vision` gates the advisor
  and image context.
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
- **GraspNet-class 6-DoF grasps** — ✅ superseded by the GraspGen-X backend
  (integrated 2026-07-18, see near term). The baseline's graspnet path
  (vendored sdk + checkpoint-rs.tar + THC-era CUDA patches) is no longer
  worth pursuing; remaining learned-grasp work (tip-offset calibration,
  reBot sweep params, collision-aware `infer_scene_pc`) is tracked in the
  near-term GraspGen-X item.
- **Skill-library growth loop** (ASPIRE) — ✅ **landed 2026-07-31** (and
  wired end to end: `retrieve()` runs in `orchestrator.run_task`; docs that
  called it store-only were corrected 2026-09-10). After each
  run, `agent/aspire.py` diagnoses the trace, localizes the salient failure,
  and distils *validated repairs* (a failure followed by the same primitive
  succeeding) into `skills_library/*.md`, deduped by (skill, signature);
  `retrieve()` loads guard-matched entries into the agent context at task
  start. Batch entry point: `scripts/learn_from_runs.py` (runs between
  sessions, never mid-demo). See `docs/AGENTIC_UPGRADES.md`.

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
- **Online adaptation** (Agentic-VLA proper): once a VLA executor exists,
  their GRPO + reward-synthesis loop is the path to improving it from demo
  logs; the trace format already captures per-primitive evidence needed for
  progress rewards.
