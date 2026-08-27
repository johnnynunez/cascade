# Roadmap

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

## Near term (before the demo)

- **Booth experience (WRC).** The attendee-facing session is scripted in
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
- **Wrist cam follow-ups.** Validate the eye-in-hand extrinsics during a
  real grasp (reproject wrist depth of the target object against the
  physics-truth pose mid-descent); consider serving the wrist stream a
  narration highlight ("what the gripper sees") on the dashboard; on the
  real rig map `isaac_wrist.yaml` to the physical D435i + hand-eye calib.
- **Sim perception flakiness** (separate campaign): YOLOE misses the YCB
  banana on some boots and label-flickers the soup can (bottle/toy);
  belief 3D positions themselves verified ±3 mm against physics truth.

- **Persistent spatial memory (proposed 2026-07-18).** BeliefStore
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
- **Skill-library growth loop** (ASPIRE) — ✅ **landed 2026-07-31.** After each
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
