# CASCADE backlog program — ledger (2026-10-07)

Dated record of the 7 October 2026 backlog pass: every numbered `docs/ROADMAP.md` follow-up and
open item was classified (S software-only CPU · G needs Isaac/GPU · H needs hardware · U blocked
upstream · D docs) and the S items were implemented in parallel worktrees, one branch and PR each,
then landed as a stacked merge train (#230 → #235 → #231 → #229 → #232 → #233 → #234 → #236 → #228;
each branch carried the previous one so all CI matrices ran at once; every head was green on all five
jobs before `gh pr merge --merge --match-head-commit`). Rules that bind every child agent working the
backlog are in the project `AGENTS.md`; the per-agent operational rules (worktree paths, ports,
interpreter) of the original coordination file are omitted here. The G/H/U rows are the open work.

## Ledger (class: S software-only CPU · G needs Isaac/GPU · H needs hardware · U blocked upstream · D docs)

| id | item (ROADMAP ref) | class | owner | worktree / branch | ports | status |
| --- | --- | --- | --- | --- | --- | --- |
| B01 | #10 ASPIRE cross-task promotion gate (`agent/aspire.py`: promote only when seen in ≥2 distinct tasks; `occurrences`, `source_tasks`) | S | child | `aspire-cross-task` | 42000-42099 | **merged 2026-10-07 (#229)** |
| B02 | #12 capability matrix → tool surface (`_EXCLUDED_TOOLS` computed from rig capabilities; `CASCADE_HIDE_TOOLS` stays as explicit override) | S | child | `capability-matrix` | 42100-42199 | **merged 2026-10-07 (#231)** |
| B03 | #13 `recall_step(n)` tool + `stuck` outcome for motion skills (human-actionable ask distinct from failure) | S | child | `recall-step-stuck` | 42200-42299 | **merged 2026-10-07 (#233)** |
| B04 | Persistence-loop leftovers #2–#7 (provisional held marker; per-task budget cap across tiers; handover/sort_by_color persistence; thin-object slip heuristic; fail fast on over-width; the listed coverage gaps) | S | child | `persistence-leftovers` | 42300-42399 | **merged 2026-10-07 (#230)** |
| B05 | #4 envelope derived features (TCP z at grasp, object height) + #9 Task-Specific Memory recipes (xyz → `localize_object(label)+offset` queries, re-grounded at replay; fixes tier-2 text keys) | S | child | `memory-recipes` | 42400-42499 | **merged 2026-10-07 (#232)** |
| B06 | #6 pre-motion plausibility check for `_MOTION_SKILLS`, advisory-only, rate-limited like `VERIFY_USER`, MockLLM-scripted tests; never a veto | S | child | `premotion-plausibility` | 42500-42599 | **merged 2026-10-07 (#235)** |
| B07 | #11 Pigey `snapshot_scene` / `restore_scene` / `search_for_object` composite skills over `BeliefStore` | S | child | `pigey-scene-memory` | 42600-42699 | **merged 2026-10-07 (#236)** |
| B08 | #15 wrist camera for the judge (MJCF wrist `<camera>`, `Frame` wrists in `build_images`) | S | child | `judge-wrist-camera` | 42700-42799 | **merged 2026-10-07 (#234)** |
| B09 | ROADMAP: merge the duplicated "Fastening" rows; add this ledger to docs | D | Hermes | `HERMES_AUDIT_20261007/cascade` | — | this document + the duplicated row removed |
| B10 | H2 owner `scripts/isaac_h2_bridge.py` (PhysX 200 Hz, policy every 4th solve, `MobileBridgeController`, fall criteria) → gate 1; then `walk_velocity`/`walk_distance` episodes + video (gates 3/5 partial) | G | Hermes | `HERMES_AUDIT_20261007/cascade` (GPU 0) | 18400-18499 | **owner merged 2026-10-07 (#228)**: first episodes 3/5 confirmed by the verifier (the two refutations are the post-command settle window: yaw wobble 0.27–0.30 rad/s against the 0.20 rad/s candidate stop limit); next: standing-settle measurement, `distance_control`/`turn_control`, MCP-driven episodes |
| B11 | Multiple robots: batch the 60 BAM output syncs per step (device-side finiteness, one read) — next owner slice | G | Hermes | `HERMES_AUDIT_20261007/route` (GPU 1) | 18300-18399 | pending |
| B12 | Multiple robots: RPC serving off the owner's GIL; physics budget per asset (hull/solver) tied to the yaw-drift gate | G | Hermes | — | — | pending (design) |
| B13 | Locomotion: rerun standing/walking/braking/reset campaigns on the Isaac Lab USD asset; standing-handoff residue (zero/decelerating-command checkpoint or distance-aware speed) | G | Hermes | — | GPU 1 | pending |
| B14 | #14 GRM judge on the GPU box (vLLM, `Robo-Dopamine-GRM-2.0-8B-Preview`, `--limit-mm-per-prompt image=8`), re-run `judge_run.py --judge grm`, agreement vs API VLM | G | Hermes | — | GPU 1 (beside Qwen) | pending |
| B15 | GraspGen-X TODOs: `tip_offset_m` calibration in Isaac; reBot sweep-volume params; `infer_scene_pc` collision-aware grasps | G | Hermes | — | GPU 0 | pending |
| B16 | Newton: `physics_probe.py --engine newton` against the real reBot asset on this build (closes the "parser-level" claim either way) | G | Hermes | — | GPU 0 | pending |
| B17 | #7 HUG second grasp backend (serve script + client mirroring graspgenx, re-ranked by `GraspOutcomeMemory`) | G (weights) | later | — | — | pending |
| B18 | Wrist-cam extrinsics validated mid-descent against physics truth; `isaac_wrist.yaml` → D435i hand-eye on the rig | G/H | later | — | — | pending |
| B19 | Sim perception flakiness campaign (YOLOE banana misses, soup-can label flicker) | G | later | — | — | pending |
| B20 | #8 `programs` tier (Waddle) — design pass first (how a program is authored vs ASPIRE distillation) | D→S | later | — | — | pending |
| B21 | Visual embedder for episodic recall (CLIP/SigLIP into `EpisodicMemory(embed_dim)`); action↔object consolidation over `ExperienceMemory` | S (weights) | later | — | — | pending |
| B22 | #1/#2/#5 Cosmos3-Edge rehearsal, `max_frames>1`, SGLang Omni rehearsal | G | — | — | — | **superseded by the user's "the brain is Qwen3.8-27B, nothing else" rule; keep as comparison ceiling only, no default change** |
| B23 | #16 nvblox on aarch64 | U | — | — | — | blocked upstream (no JetPack 7 wheel as of v0.0.10) |
| B24 | Straight-up spawn on the tuned Isaac asset (root joint / articulation root) | G | later | — | — | pending; upstream reBot-Isaacsim#9 |
| B25 | Onsite bring-up checklist, real-rig first motions, `pytest -m hardware`, booth on-site re-rehearsal | H | user + Hermes | — | — | needs the rig |
| B26 | Mid/long term: VLA executor behind the skill API; NuRec twin; online adaptation | research | — | — | — | design scopes only |
| B27 | CI timing flakes on slow runners (5 tests so far: `test_mobile_stop_temporal`, `test_hand_postures`, `test_factory_process_adapter` on macOS; `test_mobile_runtime::test_late_stop_verifier_error_is_preserved_in_receipt` on arm; `test_turn_control[delivery--0.2]` on ubuntu) — fixture deadlines only, never production limits | S | child | `macos-flakes` / `fix/backlog-macos-timing-flakes` | 42900-42999 | PR #237 (tests only) |
| B11b | B11 implementation slice: cohort `BamOutputCheck` (device-side finiteness, one read/step) | G | child (CPU impl) + Hermes (A/B on GPU 1 with `HERMES_AUDIT_20261007/harness/route_walk.py`, only while the host is quiet) | `bam-output-batch` / `feat/backlog-bam-output-batch` | 43000-43099 | in progress (CPU implementation + A/B on the x86 rig) |

## Baseline

`origin/main` b5477d8: full suite 8754 passed / 476 skipped / 4 deselected
(with `feat/h2-humanoid-admission`), CI green on all five jobs for every
merged PR of 7 Oct. `cascade-qwen.service` active on GPU 1 (:8080).
