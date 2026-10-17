# Why the sim kitchen pick on main is refused by the 5 s perception age limit

Read-only investigation, 2026-10-16. Code: worktree `/home/spark/Projects/cascade-wt/demo-video` (origin/main 2a37cca,
`git status` clean before and after). Evidence: `/home/spark/.hermes/cache/scratch/demo-video/` (REPORT.md,
evidence/run/{trace.jsonl,server.log}, logs/{brain,graspgenx}.log, rec1/). Bake-off: `~/Projects/perception-bakeoff/out/`.
`[m]` = measured in a file named next to it, `[est]` = my estimate from those numbers, `[?]` = unknown.
All file:line references are to 2a37cca unless they name a log.

## 0. On one screen

Root cause chain (sim, one GB10):
1. Every foreground localization (pick_and_place, grasp_object, preview_grasp) is ONE brain refer + ONE SAM2.1-L pass +
   depth lift on the same frame, and all of it counts against 5 s from frame receipt (`runtime.py:2258-2274`).
   10 of 11 took 5.3-8.3 s, median 6.85 s [m trace.jsonl]; one preview_grasp passed.
2. The brain refer alone takes 3.5 s with Isaac running [m probe, REPORT.md:85-87] (1.87 s median without Isaac
   [m bake-off ref_qwen n=182]), so the brain alone uses 70 % of the budget before SAM starts.
3. SAM + lift alone take 2.2-3.4 s in the same run [m: get_observation `frame_age_s` 2.19-3.37 s, a cached track step
   with no brain call] vs 0.25 s median uncontended [m bake-off qwen_boxes_sam2 n=47].
4. Both are slow because the GPU and the vLLM batch are never free: the background inventory never finishes
   (25.1 s needed [m probe] vs `timeout_s: 20`, `demo.yaml:555`) and restarts at once, so vLLM ran >=1 request in 72 of
   75 ten-second windows [m brain.log]. With 2 requests running each gets ~8.4 tok/s instead of 14.7 tok/s.
5. The never-ending refresh is made permanent by a cache-key collision: `isaac` and `isaac_side` have the same image
   size and K, so they share one cache entry (`brain_detector.py:174-177`); their views differ by 33.6 mean grey
   levels vs the 6.0 scene-change threshold [m probe_camkey.py], so each camera's tick declares the other's entry
   a changed scene and asks for a new inventory.
6. The watcher also runs a SAM2.1-L track step on every new frame of every camera (`brain_detector.py:521-524`,
   `590-612`, `world.py:454-457`), holding the one shared SAM lock (`segmenter.py:86-98,140-144`) and the GPU.
7. Second wall behind it: the grasp planner. GraspGen-X 400-pose diffusion took 12.7-13.6 s [m graspgenx.log, 4 calls]
   (0.21 s warm offline [m benchmark/results/grasp-planning-resample06-model.json]) vs the kitchen profile's 8 s
   planning budget (`isaac_kitchen_gpu.yaml:17-18`) -> grasp_at_pixel's PlanningBudgetExceeded. A pick that passes
   perception hits this next.
8. Confounder: an unidentified client with default GraspGen-X settings (100 poses, `graspmoe`) sent 45 requests
   (173 s of inference) to the same :5556 server during the recording until 15:24:23 [m graspgenx.log]. The failures
   continue without it (turn 4: 5.3-7.7 s), so it worsened but did not cause the wall.

Rig risk (rebot_rs, brain on :8081, no Isaac): **unknown, probably not systematic**. Estimated median ~2.5-3 s of
5 s, but the tail (refer max 4.29 s alone + a concurrent inventory + SAM) can cross 5 s, and the wrist camera's
inventory restarts after every arm motion. Not measured under the production load shape (section 4).

Options (section 5): (a) background never competes, est. foreground 6.85 -> ~3.5-4.5 s in the sim (thin margin);
(b) brain names + box on frame t0, SAM re-anchor + geometry on a fresh frame t1, est. t1 analysis 0.5-1.5 s with (a)
(<= 3.4 s even under today's contention); (c) serving tweaks, at most ~0.3-1 s; (d) last, owner's explicit yes only:
the limit itself (not recommended).

**Recommendation: (a) + (b) in one PR. The 5 s limit stays as it is and is applied to the frame whose geometry is
used.** Then check the GraspGen-X planning time in the sim (the 8 s budget is a separate decision).
Fix spec: section 7.

## 1. What the 5 s limit measures (Q1)

- `_frame_age_s(frame)` = `time.monotonic() - frame.t`, where `Frame.t` is the CLIENT-local RECEIPT time, never the
  remote capture clock (`runtime.py:147-157`; `types.py:27,33-37`). `_analyzed_frame_age_s` is the same age with
  max 5.0 s, checked AFTER analysis; it raises `SlowPerceptionError` ("Perception took X s on a frame that was fresh
  at capture", `runtime.py:160-175`). So: **frame received by the robot server -> end of the analysis** (the fix
  returned by `localize_object`).
- Callers of `_analyzed_frame_age_s`:
  - `runtime.py:2274` `_localize.analyze` (primary camera and the other-camera pass `2299-2325`) - the one that fired.
  - `runtime.py:2361`, `2371` belief-memory fallback / before VLM grounding, measured from `localization_started`
    (`2231`), i.e. the whole localization.
  - `runtime.py:2428`, `2432`, `2462`, `2476` `_vlm_ground_fix`, per view.
  - `runtime.py:2696` `_describe_observation` (get_observation, describe_scene, reset_scene's closing look;
    `skill_get_observation` retries once on a newer frame, `2631-2645`; reset does not, `6218`, `6232-6234`).
  - Pre-analysis freshness (`_frame_age_s`, also 5 s): `runtime.py:187`, `2262`, `2397`, `2649`.
- Inside the window for `pick_and_place`/`grasp_object`/`preview_grasp` (all `_localize`, `runtime.py:2851`, `2776`,
  `4592`): `observe()` returns the frame (`1315-1325`); `_frame_age_s` pre-check (`2262`); `localize_object`
  (`grounding.py:410-552`): for "green cube" `_resolve_query` gives prompts `["green cube", "cube"]`
  (`runtime.py:1856-1858`), round 1 = `detector.detect(frame, ["green cube"])` (`grounding.py:457-458`) =
  `BrainDetector._refer` (`brain_detector.py:319-326`, `424-438`): JPEG encode (`196-202`) + one synchronous
  chat completion (timeout 20 s, max_retries 0, `384-420`) + SAM2.1-L box masks on the same frame
  (`590-612` -> `segmenter.py:200-224`, waiting on the process-wide SAM lock, `140-144`), then colour check, depth
  points, transform, OBB, recentre, sphere/cylinder refinement, workspace check (`grounding.py:464-521`).
- Not inside: `_resolve_query`, workspace bounds, `prepare()` (before `observe`, `runtime.py:2231-2247`; the brain
  detector has no `prepare`); cache lookups (an agent query never uses the cache, `brain_detector.py:208-210`);
  the detector lock (`LockedDetector.reserve()` is a no-op for the concurrent-safe brain detector,
  `world.py:100-102`, `110-119`); grasp planning (after `_localize` returns; own 8 s budget, section 3).
- Timing breakdown of the failed calls. The trace records only totals (`trace.jsonl` has `duration_ms` and the
  error text; the server log prints no per-stage timing; `LockedDetector` timing is INFO and not in the log):

| call (trace step) | total | perception (error text) |
|---|---|---|
| pick_and_place 1, 2, 4, 9, 12 (turn 2) | 8.2 / 6.2 / 6.5 / 7.0 / 6.5 s | 8.1 / 6.2 / 6.4 / 7.0 / 6.4 s |
| grasp_object 6 | 8.4 s | 8.3 s |
| reset_scene 13 (closing observation) | 83.6 s | 6.5 s |
| pick_and_place 14, 16, 20 (turn 4) | 6.8 / 7.8 / 7.3 s | 6.7 / 7.7 / 7.3 s |
| preview_grasp 17 | 5.6 s | 5.3 s |
| preview_grasp 5 (passed) | 21.8 s | < 5 s, then 12.7 s GraspGen-X (graspgenx.log 15:22:02-15:22:14) |

  The error's "took X s" (analysis time) equals the age within 0.1 s, so frame receipt -> analysis start is ~0.
  Reconstruction [est]: brain refer ~3.5 s alone with Isaac [m probe] + ~1 s for sharing the decode with the
  inventory (30-token reply at ~8.4 vs 14.7 tok/s, brain.log) + SAM/lift 2.2-3.4 s [m get_observation] = 6.7-7.9 s,
  consistent with the measured median 6.85 s.

## 2. Why the background inventory competes with foreground refers (Q2)

- Cadence: the watcher ticks every camera at `rate_hz: 3` (`demo.yaml:462`, `world.py:382-400`); every NEW frame
  (`world.py:406-420`) calls `detect_nowait` (`world.py:454-457`), BEFORE the fuse check (`461-462`), so the
  non-fusing `isaac_proof` camera (`isaac_proof.yaml:6`) gets its own inventory too (all cameras are watched,
  `apps/demo.py:623-631`). `detect_nowait` starts a refresh when the entry is missing, its scene changed, or it is
  older than `idle_refresh_s: 30` (`brain_detector.py:343-367`, `demo.yaml:565`).
- One background request per detector (`brain_detector.py:526-562`); timeout 20 s, `max_retries=0`
  (`demo.yaml:555`, `brain_detector.py:389-393`). The probe needed 25.1 s (1045 prompt / 365 completion tokens)
  [m REPORT.md:85-86], so every refresh times out. ">= 18 APITimeoutError" in server.log is a lower bound: the
  warning is rate-limited to one per 30 s (`brain_detector.py:146`, `616-621`).
- Backoff: only the failed KEY backs off `refresh_s: 4` (`brain_detector.py:564-568`, `576-580`); the slot goes at
  once to the next waiting key (`552-553`). Result: vLLM saw one 1045-token prompt every 20 s while idle (run
  374-524 s: 104.5 prompt tok/s every other 10 s window) and >= 1 running request in 72/75 windows [m brain.log].
- Camera-key collision (new finding): K comes from the bridge (`bridge_client.py:163`), computed from focal 18 mm /
  aperture 20.955 / 1280x720 for both `cam0` and `side` (`isaac_bridge.py:638`, `676-678`, `716-717`; `proof` uses
  16 mm, `728-731`). Key = (720, 1280, 1099, 640) for both [m probe_camkey.py]. cam0-vs-side thumbnail diff 33.5-34.1
  vs `scene_change: 6.0`; same camera 3 frames apart 0.06-0.08 [m probe_camkey.py on rec1/]. So the side tick finds
  no valid entry (`_known_entry`, `510-519`) and requests a refresh; even a fast brain would never be idle in the
  kitchen. Isaac frames carry `capture["camera"]` (`bridge_client.py:208`), so a fix is local.
- Foreground does not pause or cancel it: `watcher.paused()` holds belief FUSION only, detection keeps running
  (`world.py:286-299`; entered for `_MOTION_SKILLS`, `runtime.py:1019-1023`); `preview_grasp` is not a motion skill
  (`runtime.py:115-123`); `brain_detector.py` has no pause or cancel; an in-flight request ends only by answer or
  timeout.
- vLLM: `max_num_seqs 4`, `max_num_batched_tokens 8192`, DFlash 7 (brain.log:7; same in `serve_qwen_vllm.sh:156-159`,
  rig GPU_UTIL 0.55, `~/cascade-demo.sh:104-106`). `Waiting: 0` in all 75 windows: no queueing; both requests decode
  in one batch. Aggregate decode median 16.7 tok/s with 2 running vs 14.7 with 1 [m brain.log] -> ~8.4 tok/s each.
- GPU: the watcher's track step runs SAM2.1-L on every new frame with a valid entry (`brain_detector.py:521-524`), on
  the one shared SAM model and lock (`segmenter.py:86-98`, `140-144`; demo.yaml:588-590 uses SAM2.1-L for both
  segmenters), so the foreground SAM call queues behind up to one watcher forward and competes with it for the GPU.

## 3. Why grasp_at_pixel hit PlanningBudgetExceeded (Q3)

- `grasp_at_pixel` does not use `_localize`: `observe()` -> SAM point mask (`fix_at_pixel`, `segmenter.py:227-273`)
  -> `skill_grasp_object(_fix=, _frame=)` (`runtime.py:6645-6669`). With `CASCADE_OBSERVED_FINGER_GATE=1` and
  `feasibility_max_batches` set, planning runs under `PlanningBudget(3 batches, 8.0 s)` (`runtime.py:2884-2899`;
  `isaac_kitchen_gpu.yaml:17-18`; raise in `planning_budget.py:32-42`).
- The stage that consumed it: GraspGen-X diffusion with `num_grasps: 400` (`isaac_kitchen_gpu.yaml:9-10`):
  12.7, 12.7, 13.3, 13.6 s per inference in this run [m graspgenx.log 15:22:14, 15:23:05, 15:24:23, 15:30:29] vs
  0.21 s warm offline for the same 400-pose request [m resample06-model.json rows 1-3].
  - Step 7 (15:22:42-15:22:52) and step 10 (15:24:02-15:24:12): the request also queued behind the unidentified
    client's 100-pose request (served 15:22:47-15:22:52, 15:24:04-15:24:09) and started after the budget expired.
  - Step 18 (15:30:09-15:30:18): the request reached the server at 15:30:15.9 (nothing ahead of it), needed 13.6 s.
    What used the 6.3 s before it is not recorded (`CASCADE_GRASP_EVIDENCE_DIR` unset, `evidence.py:284`) [?].
- The server keeps computing after the client gives up (3 x ~13 s of wasted GPU in this run).
- The 8 s is a deliberate pre-actuation bound for the observed-finger scene (commit 12534a4,
  `docs/grasp-planning-search.md`): changing it needs the owner, like the 5 s limit.

## 4. Would the real rig hit the same wall after the pull? (Q4)

Rig settings: `CASCADE_ARM=rebot_rs CASCADE_CAMERAS=d455f_scene,d455f_wrist CASCADE_BOOTH=1` (`~/cascade-demo.sh:112-113`),
brain vLLM :8081 GPU_UTIL 0.55, `max-num-seqs 4` (`~/cascade-demo.sh:104-106`, `serve_qwen_vllm.sh:158`); perception
URL :8081 (`openshell_rig.py:107`, `503`). No detector override in `rebot_rs.yaml` or `booth.yaml`, so
`demo.yaml:551-574` applies: `detect_classes: []` (open-world inventory), `timeout_s 20`, `refresh_s 4`,
`idle_refresh_s 30`, `max_cache_age_s 30`, `scene_change 6`. Two cameras watched: scene (fuses) and wrist
(`d455f_wrist.yaml:25` `fuse_beliefs: false`, still gets inventories).

| stage, rig | alone [m bake-off, rig frames, no Isaac] | with one concurrent inventory [est] |
|---|---|---|
| refer (brain) | median 1.87 / p90 2.36 / max 4.29 s (n=182) | x1.2-1.8 -> ~2.3-3.4 / ~2.8-4.2 / ~5.1-7.7 s |
| SAM2.1-L + lift | 0.25 s median (n=47) | +0-0.25 s lock wait behind a watcher track step |
| inventory | median 5.12 / p90 8.21 / max 12.81 s (n=47) | fits `timeout_s 20`: refreshes succeed |

Multiplier source: vLLM 4-concurrent aggregate 68 vs 20.7 tok/s single (x1.2 per stream, RESULTS_flashnext.md:65,68)
to the sim's measured halving with 2 running (x1.8, brain.log).

- Verdict: **unknown; probably not systematic**. Median localization ~2.5-3.5 s (1.5-2.5 s margin); the tail can
  cross 5 s when an inventory decodes at the same time (est. a few % of picks [?]). Nothing like the sim's
  every-call wall: no Isaac on the GPU, and inventories finish inside 20 s.
- Rig-specific load the pull adds [est]: the wrist view changes with every arm motion -> a wrist inventory starts
  at once after each motion (`brain_detector.py:362-363`), i.e. one is likely decoding when the next pick
  localizes; SAM2.1-L track steps on both 30 fps cameras keep the watcher thread busy with SAM nearly continuously
  (2 x 0.25 s per 0.33 s tick), where YOLOE cost ~0.01 s/frame; the booth's GraspGen-X `timeout_ms: 2000`
  (`booth.yaml:12-13`) shares that GPU [?].
- Wrist/scene cache-key collision: scene key (720, 1280, 643, 637) from `configs/calib/d455f_scene_rebot_rs.handeye.json`;
  the wrist K is unknown [?]. RealSense frames carry no `capture` identity, so a collision would ping-pong like the sim.
- Not measured: the rig load shape (2 inventories + refer + agent turns on :8081, SAM watcher, nvblox, GraspGen-X).

## 5. Options, escalating (Q5)

| | change | touches | expected effect |
|---|---|---|---|
| a1 | foreground scope: while a localization or motion skill runs, start no background refresh, cancel the one in flight (streamed request, closed -> vLLM aborts), watcher `detect_nowait` returns [] without SAM | `brain_detector.py` (`_start_refresh`, `_refresh`, `_ask`, `detect_nowait`), `world.py` LockedDetector, `runtime.py:1019-1023` | refer decode 8.4 -> 14.7 tok/s (~-1 s); SAM/lift no longer shares GPU with decode or with watcher SAM (2.2-3.4 s -> est. 0.5-1.5 s). Sim total est. 6.85 -> ~3.5-4.5 s: under 5 s, thin margin. Rig: removes the tail risk |
| a2 | cache key includes `frame.capture` source+camera when present; startup warning when two watched cameras still share a key | `brain_detector.py:174-177`, `apps/demo.py` | ends the cam0/side ping-pong: background requests only on real change / idle cadence |
| a3 | no background inventory for `fuse=False` cameras (sim proof, rig wrist) | `world.py:454-457` | sim 3 keys -> 2; rig: no refresh after every arm motion. Cost: no boxes on those cameras' overlays |
| a4 | background timeout >= measured, slower idle cadence in the static sim scene: kitchen `inventory_timeout_s: 45`, `idle_refresh_s: 120` | `demo.yaml:555-567`, `isaac_kitchen.yaml` | inventories complete (25.1 s); sim duty 100 % -> ~2 x 25 s per 120 s while idle; 0 while foreground |
| b | two-phase localization: brain names + boxes on t0 (semantic, capped at 20 s), then SAM box re-anchor + geometry on a FRESH frame t1 with scene-unchanged checks; the 5 s check applies to t1 | `runtime.py:2258-2325`, `grounding.py:464-521` (factor out), new `BrainDetector.reanchor` | t1 analysis = SAM + lift only: 0.25 s uncontended, <= 2.2-3.4 s under today's full contention [m], est. 0.5-1.5 s with (a). Robust to brain speed (refer up to 20 s) |
| c | serving: downscale the image sent to the brain (1280x720 -> 640x360, ~4x fewer vision tokens); compact inventory schema (fewer than ~45 tokens/object); MPS/SM share with Isaac | `brain_detector.py:196-202`, prompts, serve script, host | est. -0.3-0.5 s per refer, inventory 25 -> ~18 s; both need a bake-off re-score (accuracy on small objects). `max-num-seqs` already 4 and Waiting=0: no gain there; lowering it makes the refer wait behind a 25 s inventory |
| d | LAST, needs the owner's explicit yes, not recommended: a per-profile analysed-frame limit for the simulator | `runtime.py:166` callers | removes the refusal without making anything faster; the geometry acted on would be up to 7-8 s old |

Recommendation: a1+a2+a3+a4 and b together. (a) alone leaves the sim at ~4 s of 5 (any extra load fails it again);
(b) alone still works under today's contention (<= 3.4 s) but leaves the GPU so busy that GraspGen-X planning stays
at 12.7-13.6 s vs its 8 s budget. Together they move the 5 s check onto geometry measured < ~1.5 s before, and
free the GPU for planning.

## 6. Unknowns and confounders

- Per-stage split of a failed localization (section 1) is reconstructed, not logged. The fix adds per-call timing.
- The unidentified GraspGen-X client: `planner graspmoe, num_grasps 100` = code defaults (`graspgenx_backend.py:191,196`,
  `demo.yaml:75`), not the kitchen profile's `diffusion/400`. Likely a default-config process on the shared :5556
  (another agent's tests?). Verify runs should give the sim GraspGen-X a private port.
- GraspGen-X 400 poses with Isaac but no vLLM/SAM contention: not measured. Pre-#332 kitchen runs planned within the
  same 8 s (native06 placed the green cube, commit 12534a4), which suggests (a) is enough [?].
- Recorders (3 fps frame reads, 2 fps screenshots) added CPU load; the timeouts appeared before they ran (REPORT.md:89-90).

## 7. Fix spec (recommended combination: a1 + a2 + a3 + a4 + b)

Goal: the sim kitchen `pick_and_place(green cube -> green square)` passes perception on main's Spark sim stack,
with the 5 s analysed-frame limit unchanged and applied to the frame whose geometry is used.

### 7.1 Unchanged (reviewers must check)
- `_frame_age_s` / `_analyzed_frame_age_s` (`runtime.py:147-175`): same signature, 5.0 s default, every caller in
  section 1 keeps its check. No profile, env or config knob changes it.
- `SlowPerceptionError` semantics: never retried inside `_localize`, no memory fallback (`2285-2286`, `2320-2321`),
  pick_and_place returns without motion recovery (`4599-4603`).
- An agent's query is never served from a cache: the t0 refer stays a fresh `detect(frame, [q])`.
- PlanningBudget 8 s, observed-finger gate, GraspGen-X settings, watcher heartbeat (`world.py:448-449`),
  `LockedDetector` lock rules for non-brain detectors, `concurrent_safe`.
- Detectors without the new class-defined methods (mock, others) take today's single-frame path, byte for byte.

### 7.2 New behaviour

B1. Two-phase localization (`runtime.py` `_localize.analyze`, `2258-2274`; same for the other-camera pass
`2299-2325`). Used only when the detector CLASS defines `reanchor` (forwarded by `LockedDetector`, class-only like
`detect_vocabulary`, `world.py:139-145`) and a segmenter is configured:
1. `frame0 = observe()`; `_frame_age_s(frame0)` (unchanged pre-check).
2. Semantic step on frame0: today's `localize_object(frame0, ...)` (brain refer, SAM, colour/reference/workspace
   rules) picks the object (`fix0`, its `detection.bbox`). No 5 s check here; instead
   `semantic_age = monotonic() - frame0.t <= perception_loop.semantic_max_age_s` (new, default 20.0 s, the brain
   timeout), else `SlowPerceptionError` (same type and handling as today).
3. Fresh frame: `frame1 = _fresh_camera_frame(stream, previous=frame0)` (`runtime.py:178-188`: receipt strictly after
   frame0, its own 5 s pre-check); `started1 = monotonic()`.
4. Scene-unchanged checks frame0 -> frame1, all required:
   - same camera identity and K; for frames with `T_base_cam` (eye-in-hand) translation <= 2 mm and rotation <= 0.5 deg;
   - global: `_scene_diff(_thumbnail(frame0), _thumbnail(frame1))` is measured and LOGGED only (`scene_diff_global`;
     P11): a visitor walking behind the table changes it, the object's box does not. The cache keeps its own
     `scene_change` (6.0) rule;
   - exposure gain: per-channel (B, G, R) median of frame1/frame0 outside the box grown by 10 % (unclipped pixels);
     a gain outside 0.7-1.4 (0.0005 slack) is a scene change (lights, not exposure);
   - local: mean absolute difference of the most-changed colour channel inside `fix0`'s box grown by 10 %, frame1 vs
     frame0 scaled by that gain and clipped at 255, <= `reanchor_local_diff` (new detector key, default 8.0); a frame0
     pixel >= 250 is a lower bound of its true value (only a shortfall of frame1 under the prediction counts).
5. Re-anchor: `det1 = detector.reanchor(frame1, fix0.detection)` = SAM box prompt with fix0's box on frame1 (the existing
   track step, `brain_detector.py:590-612`, no brain call). Required: a non-empty mask (opaque, CUDA-safe checks),
   mask bbox IoU with the prompt box >= 0.5.
6. Geometry on frame1 only: the per-detection block of `localize_object` (`grounding.py:464-521`, factored into
   `fix_from_detection(frame, det, extrinsics, *, color, min_points, workspace_bounds)`; `localize_object` calls it
   too, so the single-frame path is unchanged): colour strict match on frame1's mask, depth points, OBB, refinements,
   workspace. Consistency: `|fix1.position - fix0.position| <= 0.02 m`.
7. `_analyzed_frame_age_s(frame1, monotonic() - started1)`: the 5 s rule on the geometry actually returned.
8. Return `(frame1, fix1)`; `_note_localized`, beliefs, observed-finger gate, grasp planning, keyframes all use frame1.
9. Add `perception_timing` to the skill result and the trace: `semantic_s`, `frame0_age_at_reanchor_s`,
   `reanchor_s`, `frame1_age_s`, `scene_diff_global`, `scene_diff_local`, `moved_m` (follow-ups, evidence only:
   `scene_gain`, `naming_overhang`, `naming_foreign_share`; P11).

B2. Foreground scope (a1). `BrainDetector.foreground()` re-entrant context (counter under `_lock`) and
`LockedDetector.foreground()` forwarding to a CLASS-defined `foreground` (else `nullcontext`), never taking the RLock.
`SkillRuntime.execute` (`runtime.py:1019-1023`) enters it for `_MOTION_SKILLS` plus `preview_grasp` and
`localize_object` (`runtime.py:2776`, `2747`: the `_localize` callers that are not motion skills). While the count > 0:
- `_start_refresh` (`526-562`) records/updates `_waiting` as today but starts no thread; on exit (count -> 0) the oldest
  non-backing-off waiter starts (the same rule as `552-553`).
- The background request in flight is cancelled: background calls use `stream=True` with a per-request cancel `Event`
  checked between chunks; on cancel the stream is closed (vLLM aborts the request) and the key returns to `_waiting`
  with its frame. A cancelled refresh is NOT a failure: no `_failed_at`, no warning. Foreground `_refer` stays
  non-streaming, `max_retries=0`.
- `detect_nowait` (`343-367`) returns [] without any SAM call (the request is still queued). Watcher fusion of [] is a
  no-op (`world.py:514-515`), the heartbeat is unaffected (`448-449`); the live overlay shows no boxes meanwhile.
- `close()` cancels the in-flight stream before its bounded join; the refresh thread stays owned as today.

B3. Cache key (a2). `_camera_key` (`brain_detector.py:174-177`) appends `(capture["source"], capture["camera"])` when
`frame.capture` is a dict with a `camera` string (Isaac `bridge_client.py:208`, OVRTX); unchanged otherwise.
`apps/demo.build_runtime` logs one WARNING when two watched cameras' first frames yield the same key (rig D455f pair).

B4. Fusing cameras only (a3). `WorldWatcher._tick` (`world.py:454-457`): for a camera with `fuse=False` and a detector
whose CLASS sets `background_fusing_cameras_only = True` (BrainDetector), `dets = []` without calling the detector.

B5. Config (a4). `demo.yaml` detector: `inventory_timeout_s` (default = `timeout_s` 20, used for background calls via
`client.with_options(timeout=...)`), `reanchor_local_diff: 8.0`; `perception_loop.semantic_max_age_s: 20.0`.
`isaac_kitchen.yaml` overrides: `detector.inventory_timeout_s: 45`, `detector.idle_refresh_s: 120`. `apps/demo._make_detector`
passes the new keys.

### 7.3 Failure behaviour

| situation | behaviour |
|---|---|
| scene changed frame0 -> frame1 (local colour diff over threshold after the exposure gain, a gain outside 0.7-1.4, camera moved, mask newly at the prompt border or mask agreement < 0.95, `moved_m > 0.02`, colour mismatch on frame1; the whole-view thumbnail diff is logged only, P11) | t0 box never used for geometry; ONE retry of the whole two-phase on a new frame inside the `localize_frames` loop; second change -> new `SceneChangedError(SkillError)` "scene changed during perception", treated like SlowPerceptionError (re-raised at `2285`/`2320`, no memory fallback); arm never moves |
| semantic step older than `semantic_max_age_s` at re-anchor | `SlowPerceptionError` (today's message shape, names the semantic step) |
| re-anchor + geometry > 5 s | `SlowPerceptionError` exactly as today |
| brain down / timeout / garbled | unchanged: `_refer` returns [] (`brain_detector.py:319-326`) = a miss; today's frame retries, other cameras, belief fallback, VLM fallback; no re-anchor |
| SAM returns no mask / empty / IoU < 0.5 / < `min_points` depth points on frame1; the naming mask's box and the brain's box disagree (box IoU < 0.5, M5) | no box fallback (a box cloud includes table pixels); one retry with a fresh refer; then `SkillError("could not re-anchor <query> on a fresh image")` |
| no segmenter configured / detector without `reanchor` | today's single-frame path and 5 s rule |
| background refresh cancelled by foreground | requeued, no backoff, no warning; starts when the scope ends |
| foreground scope exits by exception | counter decremented in `finally`; queued refresh starts |

### 7.4 Tests (scaled real time, `SCALE = 0.1`, fixture pattern of `tests/test_brain_vocabulary_runtime.py:33-91`)

New file `tests/test_brain_two_phase_localize.py` (fake OpenAI client that sleeps, mock stack, `perception_loop.enabled:
false` except where a watcher is needed, `monkeypatch.setattr(runtime_mod, "_analyzed_frame_age_s",
functools.partial(orig, max_age_s=0.5))`, assert `len(arm.commands) == n0` for "refused, arm stayed put"):

1. RED on main: refer sleeps 0.65 s (6.5 s live, the run's median) -> `skill_grasp_object("red cube")` raises
   SlowPerceptionError, arm commands 0 -> 0. Verified on 2a37cca by `probe_red.py` (case A: both localize_object
   and grasp_object refused after 0.65-0.70 s; control case B at 0.30 s: OK, 410 arm commands). GREEN after the fix:
   succeeds, returned `frame.frame_id > ` the brain's frame id, `perception_timing.frame1_age_s < 0.5`.
2. Limit still applies to geometry: fake segmenter sleeps 0.6 s on the re-anchor -> SlowPerceptionError, no motion.
   Mutation: delete the frame1 `_analyzed_frame_age_s` -> must be KILLED.
3. Semantic cap: refer sleeps 2.5 s (> `semantic_max_age_s` 2.0 scaled) -> SlowPerceptionError, no motion.
4. Scene changed: mock camera moves the cube between refer and frame1 (global change) -> retry finds the NEW position,
   never the old; moved twice -> SceneChangedError, no motion. Variant: small local change only (global <= 6.0) ->
   caught by the local check. Variant: `moved_m` 3 cm with identical pixels around the old box -> refused.
5. Brain down: fake raises `APITimeoutError` / returns prose -> same result and exception type as on main.
6. SAM no mask on frame1 (fake segmenter returns `[None]`, then an empty mask, then a mask outside the box) ->
   SkillError, no box fallback, no motion.
7. RED on main for contention (a1): real watcher + mock camera + fake brain whose inventory blocks on an Event and whose
   refer takes 0.30 s alone, +0.35 s while an inventory is in flight. Main: SlowPerceptionError (probe_red.py case C,
   both refers recorded `busy=True`). Fix: the in-flight stream is closed when the scope opens (fake records it), no
   refresh starts during the scope, the refer records `busy=False`, heartbeat count GROWS during the scope, the queued
   refresh starts within one tick after it, `_failed_at` stays empty.
8. Cache key: two frames, same K, different `capture["camera"]` -> two entries; a watcher alternating cam0/side frames
   (thumb diff ~34) for 20 ticks issues exactly 2 inventory requests (main: one after every completion -> RED).
9. `fuse=False` camera with the brain detector -> zero brain requests and zero SAM calls for it.
10. `detect_nowait` inside a foreground scope -> [] and zero segmenter calls (counting fake).
Keep green: `tests/test_brain_detector.py`, `test_brain_vocabulary_runtime.py`, `test_make_detector_brain.py`,
`test_failed_build_teardown.py`, `test_memory_visual_recall_v2.py`; full-suite failing-ID diff vs base (4 shards).

### 7.5 End-to-end on the Spark sim after the merge (Isaac + Qwen)

1. Preconditions: `free -g` available >= 60 GB; no foreign listener on the chosen ports; back up
   `~/.cascade/{grasp_memory,envelope}.json` (cmp after); live ports untouched (8081, 18741, 5556, 5557, 8611 of
   the rig).
2. Detached worktree of the merged main; the units of the demo-video run (REPORT.md "Commands used" 2-6): brain
   `serve_qwen_vllm.sh` PORT=18481 GPU_UTIL=0.45 with its own key; Isaac `isaac_launch.py ... isaac_bridge.py --engine
   physx --scene-config demo/scene/kitchen_config.json` with the Spark runtime env; GraspGen-X on a PRIVATE port
   (e.g. 5566, `CASCADE_GRASPGENX_PORT=5566` for the launch) so no default-config client shares it; `launch.sh --sim
   isaac ... --graspgenx external --no-open` in the OpenShell `sim` instance.
3. Run 10 x `pick_and_place(object="green cube", destination="green square")` with `reset_scene` between, then the
   same order once through the Control UI.
4. Pass: 0 SlowPerceptionError; `perception_timing.frame1_age_s` p95 < 2.5 s (half the limit); 0
   PlanningBudgetExceeded; physics postcondition `relocated` verified on >= 8/10; brain.log `Running` <= 1 during each
   pick; server.log: no inventory APITimeoutError; graspgenx.log: only 400-pose diffusion requests.
5. If PlanningBudgetExceeded remains: record GraspGen-X inference times; it is the 8 s budget question, an owner
   decision, not part of this PR.
6. Rig, before its pull + restart: same check of the rig load shape on a scratch vLLM (not :8081): 2 camera
   inventories + refer + watcher SAM; confirm the two D455f keys differ (log line from B3).

## 8. Files

- `findings.md` (this file).
- `probe_camkey.py`: cache-key collision + thumbnail diffs from rec1/ (CPU only).
- `probe_red.py`: RED reproduction on 2a37cca, mock stack, fake brain, scaled time; output in section 7.4; its
  throwaway run dir `pl-probe-*/` (private beliefs/grasp-memory/envelope stores).

## 9. Controller rulings (2026-10-16, before implementation)

- P1. Implement section 7 as written (B1-B5), in two tasks: T1 = B1 two-phase localization + tests 1-6
  (branch `fix/perception-two-phase`); T2 = B2-B5 + tests 7-10 + B6 below (branch `fix/perception-background`, merged
  into `fix/perception-two-phase` after its review). One PR.
- P2. B6 (added): the test suite never reaches a live GraspGen-X server. `tests/conftest.py` pins the GraspGen-X endpoint
  (env / config key the mock profile reads, default port 5556) to a dead port (`tests/owned_server.held_dead_port`) unless a
  test sets its own; tests that need a server start a fake one. Evidence: during the recorded sim run, 45 GraspGen-X
  requests with default settings came from another process on :5556 while agents ran unit tests that build the mock
  profile; a re-reviewer confirmed its probe reached the host's GraspGen-X. On the rig that load lands on the robot's GPU.
- P3. `semantic_max_age_s: 20.0` is accepted: identity from frame0 is used only when frame0 -> frame1 passes the
  local colour diff (after the exposure gain), the component filter, border and mask agreement checks, the camera-pose
  and the <= 2 cm position checks (the global thumbnail diff is logged only inside the re-anchor: P11); every geometry
  the arm uses is from frame1 under the unchanged 5 s rule.
- P4. Unchanged: the 5 s analysed-frame limit, the 8 s planning budget, GraspGen-X settings. If PlanningBudgetExceeded
  remains in the end-to-end run, it is reported to the owner with measured inference times (separate decision).
- P5 (after T1 review). Re-anchor checks: local diff on colour (max over B,G,R) inside fix0's box grown 10 %; frame0/frame1
  mask IoU >= `detector.reanchor_mask_iou` (0.85); a frame1 mask touching the prompt-box border where frame0's did not = scene
  change; SAM on frame1 is prompted with fix0's box grown 20 % (with the plain box a 1 cm iso-luminant slide passes every
  check; 10 % growth refuses the 0.25 cm slide that must pass). A named detection without a mask is refused (fail closed).
  The two-phase path stops retrying once `semantic_max_age_s` has elapsed (absent object, 6 s brain, 2 other cameras:
  24.9 s instead of 61.9 s). The end-to-end run logs `mask_iou` per pick to check 0.85 against real SAM.
- P6 (after T1 rounds 2-3). Still scenes must pass, moved objects must be caught. The comparison is like with like: frame0 is
  re-masked by SAM with the same grown prompt as frame1 (`BrainDetector.reanchor_reference`, before frame1 is taken, so
  frame1's age is unchanged; `reanchor(frame1, det, *, since=frame0, reference=...)`; LockedDetector forwards both, class-only).
  Frame1's mask keeps only components overlapping the frame0 re-mask (dilated 2 px); the re-mask keeps only components
  overlapping the naming mask and must cover >= 0.95 of it. Agreement replaces raw IoU: with tol_px = max(2, 0.03 x diagonal),
  >= `reanchor_mask_agreement` (0.95) of each mask lies within tol_px of the other. The 2 cm move check compares medians of
  depth points inside 2 px-eroded masks on both frames. SAM prompt margin >= 6 px a side. 3 SAM calls per attempt (naming,
  re-mask, frame1). Mock probes: still scenes 76/76 accepted; moved slides 108/108 end at the new position (<= 0.17 cm).
- P7 (after T1 round 4). Two paths. Order: name the object on frame0 (brain + SAM), take frame1, measure the global and colour
  local diffs (no SAM call). STILL path (global <= scene_change and colour local diff <= `detector.reanchor_still_diff`, 2.0):
  reuse the naming mask on frame1's depth (main's single-frame mask; no extra SAM call); the eroded-median 2 cm move check and
  the 5 s rule on frame1 still apply. SAM path (still_diff < local diff <= reanchor_local_diff): re-mask frame0 with the grown
  prompt (reference clipped to the naming mask dilated by tol_px, so a touching neighbour never enters it), re-check the
  semantic cap, take a NEW frame1, SAM on it, component filter, border and agreement checks. Mock probe (420 cases): 0 accepted
  with XY error > 0.5 cm vs main's single-frame fix; still scenes 132/132 accepted via the still path; 156 moved slides end at
  the new position (<= 0.18 cm). Known (same as main): when SAM merges a touching neighbour into the naming mask itself, the
  still path inherits main's ~1 cm error (follow-up: refuse a naming mask whose box disagrees with the brain's box).
- P8 (supersedes P7; after the round-4 re-review). The still path is WITHDRAWN: a mean colour diff over the brain box cannot see a
  thin diagonal or low-contrast object sliding (mock probe: 47/166 moved cases accepted, up to 5.8 cm off), and real camera noise
  (still-frame diff ~1.13 sigma) disables it anyway. The flow is P6's (same-prompt re-anchor: re-mask frame0 before frame1, SAM on
  frame1, component filter, border + agreement checks), plus: the GEOMETRY mask on frame1 is the re-anchored mask intersected with
  the naming mask dilated by tol_px (comparisons unchanged), so a neighbour merged under the grown prompt never enters the geometry
  and a tight brain box gives main's single-frame geometry; the semantic cap is re-checked after the re-mask.
- P9 (after T1 round 5). The geometry clip removes only frame1 pixels farther than tol_px from the naming mask that frame0's
  same-prompt re-mask ALSO contains (a neighbour merged under the grown prompt, the ring a tight box cut off); pixels new on
  frame1 stay (the agreement admits 5 % past tol_px, so a long object can slide past it; the literal clip measured it up to
  0.64 cm back toward the old place). Acceptance probe (698 cases): moved objects 0/378 accepted > 0.5 cm XY vs main (max 0.13);
  still scenes incl. a touching neighbour within main + 0.12 cm; tight boxes 52/52; noise sigma 1-4 70/70. The E2E logs
  `mask_clipped_px`.
- P10 (integration). Every shipped profile keeps `max_cache_age_s >= idle_refresh_s + 5` (demo.yaml 35 / 30; kitchen 125 / 120),
  so the watcher's idle refresh is always pending before an unchanged scene's entry ages out (no synchronous inventory on an
  unchanged scene). A SceneChangedError ends handover / sort_by_color grasps like a SlowPerceptionError (no re-home loop).
  "Released" is reported only after a measured jaw widening of >= 0.02 over the pre-call width of a jaw that held something.
- P11 (follow-ups, PR #343). Inside `reanchor` the whole-view thumbnail diff is evidence only (`scene_diff_global`; the
  cache keeps `scene_change`): a visitor behind the table changes it, the object's box does not. The camera's exposure
  gain (per B, G, R: the median of frame1/frame0 outside the box grown 10 %, unclipped pixels; < 256 samples -> 1) is
  divided out of the local colour diff, and a gain outside 0.7-1.4 (0.0005 slack, printed to 3 decimals) is a scene
  change (lights, not exposure). A frame0 pixel >= 250 is a lower bound of its true value: only a shortfall of frame1
  under min(a x g, 255) counts, so a still white object the camera clips is not refused under a darkening step (x0.85
  read 21-39 > 8 before); clipped pixels are never dropped (the strip a sliding white object leaves behind counts). The
  identity is gated by the local colour diff, the component filter, the border and mask agreement checks and the 2 cm
  move check. M5 stays the coarse fail-closed guard: a naming mask whose box overlaps the brain's by IoU < 0.5 ("SAM's
  mask and the brain's box disagree") is a failed re-anchor; it alone catches a naming mask on part of the object (the
  re-mask covers it and agrees with frame1; accepted 0.6-2.6 cm off before). `perception_timing` logs the naming mask's
  per-side overhang past the brain's box and its foreign-colour share, never refusing on them: the rig's data decides
  between an overhang + colour refusal and clipping (the ~1 cm touching-neighbour cases are main's own error and stay).
  Measured on the final head (mock, vs main's single-frame fix of the returned frame): 0 moved objects accepted > 0.5 cm
  XY on the round-5 probe (698 cases, 378 moved, max 0.13 cm, outcomes identical to before), the gain / visitor probe
  (322, 204 moved, max 0.13) and a clipped-white-object probe under light steps x0.75-x1.15, white balance and shadows
  (288, 228 moved, max 0.00 vs main, 0.11 vs truth; still scenes accepted at once 21 -> 57 of 60); the reviewer's 500
  adversarial lights / shadows cases on the previous head: 0 as well. Real keyframe (s05.jpg): gain steps, AE hunting
  and a visitor leaving are accepted at once; slides are re-measured at the new position (<= 0.11 cm).
