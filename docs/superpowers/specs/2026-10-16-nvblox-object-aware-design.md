# Object-aware nvblox clearance and grasp-corridor margin

Date: 2026-10-16 · Status: option C approved in chat by the owner; spec for review
Branch: `feat/nvblox-object-aware` (base `origin/main` c45e456, includes PR #332)
Evidence: `.superpowers/sdd/2026-10-16-nvblox-object-aware/investigation/` (local, not committed:
`findings.md`, `rows.json`, `occupancy_refusals.json`, and the replay scripts named in §6)

## 0. Decisions on one screen

1. **PR-A "unstick + measure"** lowers no margin. It adds an escape rule for TCP/joint clearance,
   a target-exempt exit after an air grasp, a hover height taken from the map's top, a map
   snapshot on every clearance refusal, and (blocked on calipers) a tool volume built from
   measured parts.
2. **PR-B "object-aware"** keeps the bridge unchanged. Cascade labels map voxels from the scene
   camera's same-frame SAM masks. Exactly the target's voxels are exempt, which makes the descent
   stricter than today. Fingers/jaws get a measured margin (floor 1.0 cm) against known
   neighbours, in the grasp corridor only. Every stale or doubtful label falls back to "unknown" (3 cm).
3. **Replay estimate over the 84 recorded refusals:**
   - PR-A: 14 likely allowed, 13 possible, 57 still refused.
   - PR-A + PR-B: 14 likely allowed, 23 possible, 47 still refused.
   - All 7 mid-motion halts and all 28 calls with a camera hit at z ≥ 0.20 m stay refused. PR-A's
     snapshots must explain those before anything else changes.
4. **Two code facts change the margin discussion** (§1.3):
   - The 9 cm "jaws" sample box is the gap between the fingers. The open fingers sit outside it.
   - The body mask hides everything within 2 cm of the fingers in new frames. A corridor margin
     below 2 cm therefore holds only against geometry frozen before the descent. The owner
     decides this (§11, Q1).

## 1. Problem and evidence

### 1.1 The "3 cm"

1. It is `safety.min_clearance_m: 0.03` (`configs/demo.yaml:22`; default `harness.py:53,106`).
2. One number guards the TCP and joint origins (`harness.py:872-877`, `1005-1030`). It also guards
   the wrist-camera slab and the open-jaw box (`harness.py:883-887`, `1061-1096`) against every
   occupied 1 cm nvblox voxel.
3. Escape rules: the tool volume has a retreat rule, "no sample closer" (`harness.py:1082-1088`).
   TCP/joint points have **none** (`harness.py:1026-1029`).

### 1.2 Rig numbers (2026-10-11 → 10-13, 29 runs; `findings.md` §C)

| Measure | Value |
|---|---|
| Failed motion-skill calls | 148 |
| Ended on map clearance | **84 (57 %)**: 35 route preflight, 32 nudge vet, 9 grasp-candidate vet, 1 rest fold, **7 mid-motion halts** |
| Gate (calls, by each call's top reason) | wrist-camera slab 60, TCP point 20, jaws 1, jaws + camera 3; joint origins 0. `findings.md` counts 64 camera calls because it reads every reason string |
| Gap vs 0.030 m | camera: 22 hits in [2, 3) cm, 23 in [1, 2), 24 < 1 cm; TCP: 22 of 23 between 2.1 and 2.8 cm |
| Camera hits at z ≥ 0.20 m | 30 of 69 strings, above every tabletop object |
| Contact monitor | 0 trips in 221 monitored motions; worst error 0.045 rad (limit 0.10) |
| Worst stranding | `mcp_132730`: 15 refusals in 3 min 47 s, all `point 0 clearance 0.028 m` |

### 1.3 Facts verified in code for this spec (new, or corrections to `findings.md`)

1. **Most camera hits are on the slab's front face.** For 22 of the 64 tool-volume calls (camera
   and/or jaws) the start pose is known: the last measured `q_deg` before the call, with no motion in between. FK
   reproduces the logged TCP to ≤ 0.9 mm. In those 22, the offending sample lies:
   - for 19, on the fingertip plane (TCP-frame x ≈ 0). This is the slab's front, ahead of the
     real camera;
   - for 2, inside the URDF gripper housing, at ≤ 1 mm from a voxel (`mcp_121191` #2/#5). A voxel
     inside the robot's own housing is self-residue or stale;
   - for 1, at x = −2 cm beside the finger.

   Script: `local_hits.py`.
2. **The jaws box does not contain the fingers.**
   - The sampled jaws box is the GraspGen-X sweep (`demo.yaml:93-98`): ±4.5 cm across, ±1 cm
     thick, 4.5 cm deep. That is the gap *between* the fingers.
   - The URDF fingers, fully open: the pads span |y| 5.0-8.15 cm and z ±1.92 cm; the finger roots
     reach |y| 8.48 cm and 8.9 cm behind the tip. Sources:
     `assets/grasp_geometry/rebot_rs_clearance_hulls.json`, URDF `joint_left/right` (prismatic
     0-0.05 m).
   - The camera slab covers only z 0..0.08 m, |y| ≤ 7.5 cm (`relative_motion.py:657-680`,
     `demo.yaml:306-314`). So the open fingers below the finger axis (z < 0) outside the jaw gap,
     and everything beyond |y| 7.5 cm, are sampled by nothing. Today only the 3 cm margin around
     the neighbouring samples covers them.
3. **Neighbours along the jaw axis.** With the jaws fully open (`gripper.pregrasp_open_margin_m:
   null`), a neighbour along the jaw axis whose gap to the target is below about
   (8.15 cm − target half-width) is inside the finger's path. For an 8 cm cup that is about 4.2 cm. No margin can
   admit it. Neighbours *across* the jaw axis, and on the camera side, are what PR-B can admit.
4. **The body mask hides the corridor band.**
   - `body_mask.inflate_m` is 0.02 (`rebot_rs.yaml:150`). The fingers are swept over their full
     travel while a stream runs (`rebot_rs.yaml:142-144`).
   - So new frames never integrate a surface within ~2 cm of the fingers.
   - `docs/NVBLOX.md:269-275` keeps inflate (2 cm) < margin (3 cm) for exactly this reason.
5. **Code version of the recorded runs.**
   - The rig checkout pulled the lift-first routes (`05467ef`) at 12:57 on 10-13. Only
     `mcp_150541` started after that (13:01).
   - So 83 of the 84 refusals ran without the +3/+6 cm lift candidates (`trajectory.py:110-131`).
   - `mcp_150541` shows that lift candidates alone do not free TCP stranding, because TCP points
     have no escape rule.
   - The TCP reason format changed in `321f2a2` (`.9f` plus shortfall, `harness.py:1027-1029`)
     after the runs. The replay regexes match both formats.
6. **What started the 132730 stranding.** It was not a place failure alone. In `pick_and_place`
   step 24 the grasp attempt failed *after the close* on the **camera** gate:
   `grasp_recovered_after: SafetyViolation: wrist camera clearance 0.018 m … [0.2270, -0.1872,
   0.1077]`, about 18 cm from the doll, outside the 15 cm cylinder. B36 then treated the doll as
   held (`runtime.py:1730-1749`). The arm stayed near grasp height (`tcp_z_at_grasp_m` 0.032;
   step 26 refused at TCP z 0.031), the place failed as unreachable, and the TCP refusals followed.

## 2. Goals / non-goals

Goals:
1. No stranding: an arm already inside the margin can always take a strictly retreating, fully
   vetted route out.
2. Every clearance refusal or halt is diagnosable from the trace: the geometry, the gap, and the
   voxels near it.
3. Perception and map agree on what each voxel near the gripper is: target, known neighbour
   `<name>`, robot, or unknown.
4. Grasps beside known neighbours succeed where the fingers physically fit, with a margin set by
   measurement.

Non-goals:
1. Lowering any margin for the arm body (TCP, joint origins, links), the wrist camera, the
   housing, held payloads or unknown voxels. These stay at 3 cm everywhere.
2. Touching the contact monitor (`safety/contact.py`, `rebot_rs.yaml:92-103`) or contact zones.
3. Changing the occupancy bridge protocol or backends (`scripts/serve_occupancy_bridge.py`,
   `perception/occupancy_backends.py`).
4. Making payload/recovery map rules a general skip-on-failure policy (AGENTS.md:61). The
   attached-payload check (`harness.py:992-1003`) gets no escape rule.
5. Adaptive pre-grasp opening (`gripper.pregrasp_open_margin_m`). It is a separate follow-up and
   needs the width map measured first (`rebot_rs.yaml:194-201`).

## 3. PR-A "unstick + measure" (no margin lowered)

### 3.1 Lever 1: escape rule for TCP and joint/link clearance

Rule:
1. **Per step** (`approve`, `vet_step`): if any TCP/joint point is below `min_clearance_m` at
   `q_next`, the step is admitted only when **both** hold:
   - (a) every such point was already below the margin at `q_prev` (no new entrant);
   - (b) its clearance **strictly** increases, `d_next > d_prev + 1e-9` (1e-6 refused every escape at its first tick: a 3 s min-jerk escape from rest gains ~9e-8 m there; the route-end rule bounds a slow creep), or the step is an exact
     hold (`q_next == q_prev` within 1e-9).
   Both distances come from the same grid in the same call.
2. **Route end** (planner rule, `vet_route` and the spatial `plan_route`): a route is accepted
   only if its final waypoint passes the static occupancy check with no escape. A route that
   leaves the arm still inside the margin is never chosen.
3. A refusal keeps the original reason text. It appends "(inside the margin: only a strictly
   retreating step is allowed)". `test_empty_tool_at_19mm_fails_before_any_home_command` still
   matches `0\.019000000 m below 0\.030000000 m`.
4. Static vets (`vet_pose`, grasp candidates) never apply the escape, as today
   (`harness.py:923-924`).

Code:
- `harness.py:872-877`: pass `q_prev` to `_occupancy_violation`.
- `harness.py:1005-1030`: add the rule. Payload calls (`attached=True`, `:992-1003`) stay strict.
- `safety/trajectory.py:76-83`: `vet_route` checks the end pose. `plan_route` (`:86`, candidate
  loop `:133-143`) inherits it.
- `skills/spatial_moves.py:394-453`: the same end check after the last goal (`:445`).
- No config key. The tolerance is a module constant, like the other escape rules
  (`harness.py:771-773`, `:811`, `:829`).

Failure behaviour:
1. A uniform field (no gradient) refuses, as today.
2. A step that brings any point closer, or lets a point enter from outside, refuses at 50 Hz.
   The arm holds, as today.
3. A new obstacle mid-escape is caught by the per-step rule. The map refreshes during motion
   (`world.py:428-430`).

### 3.2 Lever 2: target-exempt exit and a hover height from the map

Rule:
1. **Exit withdrawal.** When `grasp_object` ends with the jaws verified empty (air grasp,
   `runtime.py:3506-3521`), the arm sits at `q_pre`. Before returning, the runtime re-enters
   **the same target column it approached with** (`_column_exemption`, `runtime.py:110-113`) and
   moves up through a vetted route to a TCP lift of +3 cm, else +6 cm. The lift poses are built
   as in `trajectory.py:121-131`. Every other gate runs at full strength, and the column is
   released afterwards.
2. **No automatic exit with a possible payload** (provisional hold, `_held_provisional`,
   `runtime.py:3376-3379`). This follows `runtime.py:3462-3466` ("do not invent an empty-tool
   recovery") and AGENTS.md:61. Lever 1 handles the TCP part once the operator opens the jaws.
3. **Hover from the map's top.**
   - `pregrasp_min_z = max(fix_top, map_top) + pregrasp_clear_top_m + voxel/2`.
   - `map_top` is the top face of the highest occupied voxel inside the target column, read under
     `_refresh_lock`.
   - It is never `view_snapshot()`, which is "display only, never a safety input"
     (`occupancy.py:1243-1244`).
   - The raise over `fix_top` is capped at 2 cm, so one residue voxel cannot push the hover out of
     reach. The recorded gaps were 0.2-0.9 cm; `NVBLOX.md:556` measured ~1.5 cm on a cup. Beyond
     the cap, the exit withdrawal (rule 1) still frees the arm.
4. The hover only ever goes up compared with today, and the harness still vets the result.

Code:
- `runtime.py:3176-3184`: `pregrasp_min_z`.
- New `OccupancyMap.column_top(center_xy, radius, z_min)` next to `clearance_excluding`
  (`occupancy.py:1367-1411`).
- `runtime.py:3493-3530`: the air-grasp branch runs the exit.
- Config: `grasp.pregrasp_map_top: true` on `rebot_rs.yaml` (opt-in, next to `:114`); default
  `false` in `demo.yaml`.

Failure behaviour:
1. Map stale or the column empty → `map_top` is unavailable → the old fix-top hover. The harness
   still vets it.
2. Exit refused → the arm stays put. The reason is appended to the air-grasp error, and the column
   is released as today.
3. Halt or e-stop during the exit → nothing more happens, as with every motion.

### 3.3 Lever 3: a map snapshot on every clearance refusal or halt

Rule:
1. Whenever `_occupancy_violation`, `_payload_occupancy_violation` or `_tool_volume_violation`
   returns a reason, the harness files one report in a bounded deque. This mirrors
   `_contact_reports` (`harness.py:196-198`, `385-388`).
2. Each report records:
   - when and where: `phase` (approve / vet_step / vet_pose), `gate` (tcp, link i, wrist camera,
     jaws, attached), `q`, the TCP pose;
   - the shortfall: the offending sample (base frame and TCP frame), clearance, margin;
   - the context: the exemption in force, and the map (refresh count, age, voxel size, occupied
     total);
   - the evidence: up to 64 occupied voxel centres within 5 cm of the sample, with distances,
     rounded to 1 mm.
3. Bounds: dedupe per call by (gate, sample rounded to 1 cm, map refresh); at most 32 reports per
   call; work happens only on the refusal path.
4. The runtime drains the deque into `trace_context["clearance"]`, the way contact reports drain
   (`runtime.py:891-896`, `1275-1277`).

Code:
- `harness.py`: the three methods above, plus `drain_clearance_reports()`.
- New `OccupancyMap.nearby_occupied(point, radius_m, limit)` under `_refresh_lock`, so the
  snapshot is the same map the gate read.
- `runtime.py:891-896`, `1275-1277`.
- Config: `occupancy.refusal_snapshot: {radius_m: 0.05, max_voxels: 64, max_reports: 32}`.

Failure behaviour: a snapshot error never changes the gate outcome. A refusal stays a refusal, and
the report carries `snapshot_error`.

### 3.4 Lever 4: tool volume from measured parts (blocked on owner input, §8)

Rule:
1. The rebuilt volume is the union of four parts:
   - (a) the calipered camera + padding + bracket box, every face moved outward to the next 5 mm;
   - (b) the URDF `gripper_end` housing hulls' box, rounded up: x −0.160..−0.070, y ±0.095,
     z ±0.045 m (hull AABB −0.1572..−0.0732, ±0.0920, ±0.0410);
   - (c) both URDF fingers over the full stroke, rounded up: x −0.090..0, y ±0.085, z ±0.020 m
     (hull AABB −0.0886..0, ±0.0848, ±0.0192);
   - (d) the existing jaw-gap box (`demo.yaml:93-98`).
2. Every part keeps the 3 cm margin.
3. Until the calipers arrive, nothing changes: the 14 × 15 × 8 cm slab stays (`demo.yaml:306-314`).
   Parts (b)-(c) ship together with (a), so the refusal count is measured on one change.
4. **Fact for the reviewer.** Part (c) is required. Dropping the slab's front without the finger
   samples would leave the open fingers unsampled (§1.3.2). In the 4 calls of `mcp_150541`
   #13-16, the 1.2 cm hit is on the open finger.

Code:
- `safety/tool_volume.py:30` (`_KEYS` += `housing`, `fingers`) and `:84-145`.
- `skills/relative_motion.py:657-680` (`camera_box` takes `front_m`, `bottom_m`; both default to
  0, which is today's slab).
- `skills/spatial_moves.py:115-124` (one parser, two layers; `test_harness_tool_volume.py:118`).
- `rebot_rs.yaml:77-81`.

Failure behaviour: a malformed block raises at build time, as today (`tool_volume.py:66-81`).

## 4. PR-B "object-aware nvblox + corridor margin"

### 4.1 Where the association lives: cascade side, bridge unchanged

Decision: label voxels in cascade, from the scene camera's same-frame SAM masks, right after
detection in the watcher tick. Freeze the result per grasp episode.

Why not in the bridge or in nvblox:
1. The wire carries no semantics: probe, clear, integrate*, query (`serve_occupancy_bridge.py:18-35`,
   `110-149`). Adding them changes the protocol for three backends (nvblox, warp, voxel).
   `auto` picks warp or voxel on hosts without nvblox/CUDA (`occupancy_backends.py:463-472`).
2. nvblox_torch can fuse float16 feature vectors per voxel (`add_feature_frame`) and run several
   mappers (`mapper_id`). Source: `/home/spark/Projects/cascade/.nvblox/source/nvblox_torch/nvblox_torch/mapper.py:51-55`,
   `148-175`.
   - Fused features are weighted averages over time (a feature voxel is `[f0 … fF-2, weight]`,
     `mapper.py:351`), so an old label persists in a voxel and cannot be expired per frame. The freshness rules (§4.2) need per-frame labels.
   - Per-object mappers need the masks *before* integration. The watcher integrates before
     detection on purpose: the robot mask must match the image's pose (`world.py:423-430`).
3. Cascade already holds everything needed:
   - occupied voxel centres per query (`occupancy.py:963`, `981`);
   - the exact re-measure path (`clearance_excluding`, `occupancy.py:1367-1411`);
   - SAM masks on the same frame the map just integrated (`world.py:430` then `:454`;
     `brain_detector.py:590-612`);
   - K and `T_base_cam` (`world.py:426-427`).

Why not world-model boxes alone: beliefs are fused over time, with up to 384 points from the last
real-mask view (`memory/beliefs.py:53-78`). A box alone cannot tell a target voxel from a
neighbour that slid into the box. The design requires same-frame mask **and** depth band **and**
fitted prism. Beliefs feed only the reconciliation report (§4.7).

### 4.2 Label snapshot and freshness rules (fail closed)

The snapshot is built per tick, for map cameras only (`maps_depth`; today `d455f_scene`):
1. **One frame.** Build it only if the map's last commit is this frame. Required maps stamp
   `_last_refresh = frame.t` (`occupancy.py:783`, `983`). Optional maps stamp commit time, so
   `_commit_grid` also records the committed frame's `t`. Otherwise skip the tick.
2. **Per occupied voxel:**
   - project into the full-res frame;
   - assign object *i* only if *i*'s SAM mask covers the pixel;
   - the voxel depth must lie in [measured − 1.5 cm, measured + 4 cm]. 4 cm is nvblox's default
     TSDF truncation band, 4 voxels
     (`/home/spark/Projects/cascade/.nvblox/source/nvblox/include/nvblox/integrators/projective_integrator_params.h:37-40`;
     the backend sets only the integration distance, `occupancy_backends.py:333-336`);
   - the voxel must lie inside *i*'s fitted prism dilated by 1 cm (§4.4).
3. **Conflicts → unknown:** two masks; mask plus prism of another instance; mask and robot body.
   Voxels not visible in this frame → unknown. Labels never carry over between frames.
4. **Robot label.** Voxels within 5 cm of the posed URDF hulls at the frame's q are labelled
   `robot`, for **report only**. They are never exempt (§11, Q4).
5. **Masks** may be CUDA tensors (`brain_detector.py:593-594`). Convert them on the watcher
   thread, vectorised; no per-voxel Python loop (watcher shares the GIL with the 50 Hz stream).

A label may exempt a voxel or shrink a margin only when all of these hold:
- F1. The snapshot is younger than `label_max_age_s` (default = `occupancy.max_age_s`, 1.5 s on
  the rig).
- F2. The detection has a real SAM mask. Box-only detections never label.
- F3. The detection's brain box is younger than `box_max_age_s`, default = `detector.max_cache_age_s`
  (30 s, `demo.yaml:566`). This needs a new optional `Detection.box_t`, set from the cache
  entry's `t` (`types.py:59-66`; `brain_detector.py:162`, `:611`). Otherwise the watcher's
  `detect_nowait` tracks cached boxes "at any age" while the frame shows the entry's scene
  (`brain_detector.py:348-350`), and `detect` keeps an over-age entry tracked while its refresh
  is pending (Ruling 35, `:218-220`).
- F4. The grid identity (origin, voxel size) and `scene_reset_generation` match those of the
  snapshot.

Grasp episode:
1. At grasp start (candidate selection) the runtime freezes:
   - `E`, the target's voxel keys;
   - the known-neighbour prisms.
2. The frozen data come from the newest snapshot that passes F1-F4.
3. **Target identity** comes from the fix's own mask, so inventory recall does not matter. The
   localize frame must come from the map camera, be no older than 1.5 s, and show the map frame's
   scene (thumbnail diff ≤ `scene_change`).
4. **Neighbours** come from the watcher's inventory instances in the same snapshot.
5. During the episode only voxel keys in `E` are exempt; new keys are unknown. `E` may only shrink:
   a later snapshot that labels a key in `E` as something else removes it.
6. `E` expires at the earliest of:
   - the end of the corridor (lift back at `q_pre`) or the end of the exit withdrawal;
   - a halt or e-stop;
   - a map clear or scene reset.

### 4.3 Exemption rule (replaces the column and the 15 cm cylinder for the map)

1. In the approach leg, pregrasp vet, descent, close, lift and exit withdrawal, voxels with keys
   in `E` are ignored by:
   - TCP/link points: an attributed re-measure like `clearance_excluding`;
   - the tool volume: `clearance_excluding` with `exclude_keys` (`occupancy.py:1367-1411`).
2. **Every other voxel counts, at full margin, in every phase.** Neighbours within 15 cm are now
   checked during the descent. Today they are invisible there (`harness.py:1105-1109` via
   `runtime.py:3312-3316`, r = 0.15 m from `demo.yaml:163`).
3. **The cylinder stays for the table rules only**: TCP ≥ table + 2 cm and joint origins
   ≥ table + 1 cm (`harness.py:833-847`, `866-870`). Those are unchanged. The map checks stop
   reading `_grasp_exempt` in object-aware mode.
4. In the corridor, TCP/link points also exclude table-band voxels (≤ table_z + `table_band_m`,
   as the tool volume does at `harness.py:1047`). This is no reduction: today the cylinder hides
   those voxels from the points too, and the table rule (TCP ≥ table + 2 cm) governs them.
   Outside the corridor, points keep today's rule.
5. Opt-in: `grasp.object_aware.enabled` (default `false`; `rebot_rs` turns it on after §7). With
   it off, the column and cylinder behave exactly as today.

### 4.4 Corridor margin rule (owner decision: corridor only)

The reduced margin `m_corr` applies **only** when all four hold:
1. **Phase:** after the pregrasp pose, meaning descent, close, and the lift that retraces the
   descent back to `q_pre`.
   - The pregrasp pose keeps 3 cm, because it is where transport ends.
   - "Lift start" is read as "until the lift is back at `q_pre`": a narrower corridor would strand
     the arm at grasp height (§11, Q2).
2. **Part:** fingers and jaws only. That means the URDF finger hulls at the commanded opening
   (`gripper_clearance.py`, `rebot_rs_clearance_hulls.json`) plus the jaw-gap box. Excluded: the
   camera, the housing, the TCP/link points.
3. **Obstacle:** a voxel explained by a **known neighbour**, i.e. inside that neighbour's frozen
   prism dilated by `prism_dilate_m` (1 cm).
4. **Distance:** the exact finger-hull ↔ prism distance, the separating-axis lower bound already
   used by B73 (`gripper_clearance.py:1-43`), must be ≥ `m_corr`.

Everything else stays at 3 cm: finger samples against unexplained voxels, the camera, housing,
TCP and links against anything, and any part outside the corridor.

Known neighbour:
- not the target;
- a SAM mask with ≥ 50 lifted points;
- passes the workspace filter (`demo.yaml:604-620`);
- prism height ≥ 1 cm;
- passes F1-F4.

**Fitted prism:** the minimum-area rectangle of the mask points projected on the table, extruded
from `table_z` to the highest point. A neighbour that fails any test, or whose later voxels fall
outside its frozen prism, is **unknown** (3 cm).

**Margin value:**
- `m_corr = grasp.object_aware.corridor_margin_m = max(0.010, ceil_5mm(e_fit_total))`, capped at
  `min_clearance_m`;
- code default when the key is absent: 0.03 (no reduction);
- pinned by a test against the measurement (§4.6).

### 4.5 Fallback to unknown

1. When no fresh target association exists at grasp start, the object-aware grasp **refuses**:
   "target voxels unknown: <reason>", e.g. "labels 2.3 s old" or "no SAM mask". Nothing moves.
2. No silent fallback to the legacy column/cylinder in object-aware mode. That would exempt
   voxels without a fresh label. The per-profile switch `enabled: false` is the explicit rollback.
3. Transport, home and nudges are unaffected by labels: 3 cm everywhere, plus PR-A's escape rule.

### 4.6 How the fit error is measured

Quantity: the one-sided outward error `e`. For each object placement, `e` is the largest distance
by which the true surface sticks out of the fitted prism. Points inside count 0, and hidden faces
are included.

Data that exists:
- **(a) MuJoCo RGB-D with ground truth on CPU** (`sim/mujoco_rgbd.py`, used by
  `tests/test_perception_truth_mujoco.py`).
- **(b) Isaac kitchen truth props** (`tests/test_truth_*`). It needs the GPU, so it is a later
  cross-check.
- **There is no recorded rig RGB-D of objects.** Traces keep JPEG keyframes only
  (`runs/*/keyframes`). The only rig depth is the hand-eye sweep with an empty table
  (`~/.cascade/calibration-evidence/20261012_192623_d455f_scene`).

Procedure:
1. **Sim term.** At least 40 placements of 6 prop shapes (5 cm cube, 8 × 10 cm cup cylinder,
   12 × 8 × 6 box, spool, doll proxy, 2 cm flat box).
   - Camera at the rig's `d455f_scene` hand-eye pose, 1280 × 720 with depth stride 2.
   - Masks from the render (exact), then from SAM 2.1-L when the GPU is free.
   - The production lift → prism code. Truth surface sampled at 2 mm.
   - `e_sim = max` over placements.
2. **Rig size term.** New read-only capture, operator-gated, frames from the robot server's
   published stream (no second camera owner): 5 calipered props × 4 placements.
   `e_size = max(caliper half-extent − prism half-extent, 0)`.
3. **Registration term.** `e_calib` = the accepted `d455f_scene` record's residual. Its
   `validation.json` lists `rmse_m` 3.1-3.3 mm across entries; take the max, read from the record
   at test time.
4. `e_fit_total = e_sim + e_size + e_calib`. Commit `docs/evidence/corridor-margin-<date>/measurement.json`.
5. Test (`tests/test_corridor_margin_measured.py`):
   - every profile with `object_aware.enabled` has `corridor_margin_m ≥ max(0.010,
     ceil_5mm(e_fit_total))`;
   - the code default is 0.03;
   - the test fails against a lowered config and against a larger measurement.

### 4.7 Reconciliation report (world model vs map, report only)

Per snapshot, add three lists to `occupancy_state.reconcile` in `backends()`
(`runtime.py:1386`, `:1406-1407`, the map's `report()`) and to lever-3 snapshots:
1. **map without world model**: 26-connected clusters of ≥ 20 unknown voxels above the table
   band. The query grid is the workspace AABB, 40 × 60 × 56 cells at 1 cm (`demo.yaml:60-61`,
   `apps/demo.py:505-507`).
2. **world model without map**: beliefs seen ≤ 3 s ago whose prism holds < 5 occupied voxels
   (flat or occluded).
3. **robot residue candidates**: unknown voxels within 5 cm of the posed URDF hulls, or at
   z ≥ 0.20 m.

It never changes a gate. Mock rig: no map, no key, so the `_runtime_state` golden is unchanged.

## 5. Failure-behaviour table (what the safety layer does)

| Situation | PR-A | PR-B (object-aware on) |
|---|---|---|
| Masks stale (snapshot > 1.5 s, or box > 30 s) | n/a | no label → unknown (3 cm). Grasp start: refuse "target voxels unknown: labels N s old". Mid-episode: frozen `E` keys only; new voxels unknown |
| Brain down | n/a | Cached boxes keep tracking while the scene is unchanged, until F3 expires them at 30 s; then unknown. `localize` fails anyway: no grasp |
| SAM missing or failing | n/a | Box-only detections never label (F2): all voxels unknown, object-aware grasps refused, transport unchanged |
| Map stale or failed | Required map refuses, as today (`occupancy.py:1437-1446`); no snapshot voxels, report says why | No snapshot (F1, F4); grasp start refused; mid-episode the stale map already refuses every step |
| Wrong label (mask leaks onto a neighbour) | n/a | Bounded by depth band ∧ target prism ∧ conflict rule (a voxel in two prisms → unknown). Residual: a leak that also inflates the target prism. Backstop: the contact monitor |
| Neighbour moves after the fit | n/a | Its new voxels leave the frozen prism → unknown at 3 cm → refuse or halt. A move inside the 2 cm mask band during the corridor is invisible to the map (§1.3.4); the contact monitor is the only check there (§11, Q1) |
| Held object | Held-payload mask, as today (`rebot_rs.yaml:173-175`); its old voxels at the pick spot stay obstacles; the held body is not a clearance sample on the rig (unchanged gap) | Old voxels exempt only while `E` lives (until `q_pre`), then unknown. Grasp from a 3 s belief (no mask) → no target association → refuse |
| Target pushed during descent | Halt if new voxels are within 3 cm (descent cylinder today: none) | New voxel keys are unknown → halt; contact monitor |
| Escape step closes in on anything | Refused per step; arm holds | Same |
| Snapshot code raises | Gate outcome unchanged, report carries `snapshot_error` | Same |

## 6. Testing

### 6.1 Unit tests (offline, mock arm / CartesianKin / GridBridge, no hardware)

1. **L1.**
   - A 2.8 cm hover with a voxel below: move_home, +3 cm and up 2.5 cm are admitted.
   - A lateral 1 cm move over a flat top, and a hold-then-close step, are refused.
   - Keep `test_empty_tool_at_19mm…`. Add "first lift safe, final leg blocked"
     (`test_clearance_lift_route.py:94` pattern).
   - Mutation-check `>` vs `>=` and the end rule.
2. **L2.**
   - An air grasp leaves the arm at map top + clear top + voxel/2, outside the margin.
   - The exit runs under the column only, and is refused with a neighbour above.
   - A provisional hold never triggers the exit.
   - The map top is capped at +2 cm.
3. **L3.** Every gate kind files exactly one report per call. Reports are bounded, deduped and
   drained into the trace. An exception in the snapshot does not admit motion.
4. **L4.** The shipped config covers the caliper JSON, rounded up. It is red against the old slab.
   The harness box equals the spatial-tools box. Fingers and housing are sampled.
5. **L5/L6.**
   - Same-frame binding; F1-F4; conflict → unknown; `E` only shrinks; key identity on a grid
     change.
   - Descent with a neighbour at 10 cm is now checked.
   - Finger margin applies only after `q_pre`, only to fingers/jaws, only to explained voxels.
   - Camera stays at 3 cm.
   - Use `tests/occupancy_grid_bridge.py` `GridBridge`/`DepthBridge` with rendered URDF meshes.

### 6.2 Replay estimate of the 84 recorded refusals (done now, rule-based)

- Scripts: `poses.py`, `local_hits.py`, `lever4.py`, `replay_estimate.py`, with output in
  `replay_estimate.json`, in the investigation directory.
- Inputs: `occupancy_refusals.json` and `rows.json`, plus the last measured `q` before each call
  from the read-only traces, and FK on the worktree's signed model.
- The traces hold no map, so verdicts are *likely* or *possible*, never certain.

Rules:
- **L1** applies only to TCP-only refusals at the start pose with a gap ≥ 2 cm.
  - move_home, reset_scene, move_to_rest and up-moves ≥ 2 cm → likely (the lift leg moves
    strictly away and ends ≥ 3 cm).
  - Tiny up-moves → possible.
  - Lateral moves, gaps below 2 cm, and static candidate vets → refused.
- **L2** covers calls made while the arm sat where an air grasp left it (`mcp_150541` #6-16 →
  likely; #5 and `mcp_70337` #16 → possible).
- **L4** uses an assumed camera box (x −0.14..−0.06, |y| ≤ 0.075, z 0.035..0.09) plus the URDF
  housing and fingers. The voxel lies within the gap of the reported sample:
  - voxel-to-volume < 3 cm for sure → refused;
  - otherwise → possible.
- **L5** → possible for grasp-family candidate vets and route refusals at z < 0.15 m (the
  offending voxel may be the target's own, outside the footprint column).
- **L6** → only refusals in descent, close or lift.

| Lever | Likely allowed | Possible | Still refused | Note |
|---|---|---|---|---|
| L1 TCP escape | 8 | 4 | 8 of 20 TCP | 132730 #28/31/33/36/38/39, 150541 #6/#9 |
| L2 exit + map-top hover | 8 | 2 | — | all in `mcp_150541` after the #1 air grasp |
| L3 snapshots | 0 | 0 | — | diagnostic for all 84 |
| L4 rebuilt volume (assumed) | 0 | 8 | 14 of 22 placeable | 42 tool-volume calls not placeable (failing sample not at the start pose) |
| **PR-A union** | **14** | **13** | **57** | |
| L5 target voxels | 0 | 13 (10 new) | — | needs snapshots to confirm the voxel is the target's |
| L6 corridor margin | 0 | 0 | — | no recorded refusal in descent/close/lift: the cylinder hid them |
| **PR-A + PR-B union** | **14** | **23** | **47** | |

Reading the table:
1. PR-A ends both recorded strandings early.
   - `mcp_132730`: the first refusal was at 11:41:25. The move_home at 11:41:43 is likely
     admitted, versus 3 min 47 s of refusals today.
   - `mcp_150541`: avoided. The 8 refused moves from the air-grasp hover never happen.
2. Unchanged by any lever: all 7 halts, all 28 calls with a camera hit at z ≥ 0.20 m, and 14 of the
   22 placeable camera hits (on real gripper parts, or self-voxels inside the housing). That class
   (self/stale voxels) needs the lever-3 snapshots, then a mask or attachment fix, never an
   exemption.
3. L6 converts nothing recorded. Its job is to stop L5's stricter descent from refusing grasps
   beside known neighbours. The live run measures it (§7, B2).

### 6.3 Exact replay after PR-A

`scripts/replay_clearance.py <trace.jsonl>`:
1. Rebuild a local `GridBridge` from each `context.clearance` voxel set.
2. Re-run `vet_pose`/`vet_step` at the recorded `q` with the old and new rules.
3. Print allowed vs refused per lever.
4. Run it on every rig session between PR-A and PR-B. Its numbers replace §6.2's estimate.

## 7. Live acceptance (operator-gated, real-robot-bringup rules)

Before every motion run: the motion handshake. That means a plan and dry run, then "OK, e-stop in
hand", a ~50 cm clear table except the props, nobody at the arm, the arm resting, and a final "go".
Read the live state first (service PID, deployed revision, motion lock).

PR-A, one 30-minute session, standard props (cup, cube, spool, doll), 20 grasp attempts:
1. **A1 no stranding.** Zero sequences of ≥ 3 consecutive clearance refusals at one pose. Every air
   grasp ends with the exit withdrawal admitted or with a reported reason.
2. **A2 diagnosable.** 100 % of refusals and halts carry a `context.clearance` record with voxels.
   The operator reviews the five worst z ≥ 0.20 m hits on the `nvblox_map` tile.
3. **A3 no new contact.** 0 contact-monitor trips. No admitted escape step is followed by a halt
   or contact.

PR-B, after the lever-3 review, opt-in on `rebot_rs` only:
1. **B1 neighbours.** A target cup with a block at measured gaps of 1.5, 2.0 and 3.0 cm *across*
   the jaw axis, and at 3 and 5 cm on the camera side; 5 attempts each.
   - Pass: admitted wherever finger-to-prism ≥ `m_corr` and the camera ≥ 3 cm.
   - No contact trip, and no touch seen on slow-motion video.
2. **B2 descent strictness.** A neighbour at 5-10 cm (hidden by the 15 cm cylinder today). Over
   20 picks, "descent unsafe" refusals per pick must be no worse than a PR-A-only baseline taken
   the same day, after any residue fixes from A2.
3. **B3 fail closed (no motion needed).**
   - The operator stops the perception brain: the grasp is refused with the stale-label reason,
     and move_home still works.
   - Repeat with the segmenter disabled.
4. **B4 reconciliation.** 3 named props plus 1 unnamed object: the unnamed cluster is listed.
   Remove a prop: "world model without map" appears, then clears.

Success: B1 ≥ 80 % admitted at a 3 cm gap across the jaw axis, against a PR-A-only baseline
run with the same layout the same day, with A3's zero-contact criterion held for the whole
session.

## 8. Inputs needed from the owner (calipers, mm, ±1 mm, one photo per view)

Reference: the fingertip plane (`gripper_end` origin) and the finger axis, with the jaws closed.
x is back along the jaws, y across the jaws (camera side up), z up.

1. **Camera + padding + bracket envelope:**
   - front face distance behind the fingertip plane;
   - rear face, padding included;
   - −y and +y extents, measured separately;
   - bottom above the finger axis (is there air between the housing top and the camera?);
   - top.
2. **Bracket:** where it clamps to the housing, and any part wider or lower than the camera.
3. **Cable:** exit point, the route along the wrist and forearm, and any loop. `NVBLOX.md:325`
   found cable points at the elbow.
4. **Housing cross-check:** the outer box (URDF says x 7.3-15.7 cm behind the tip, ±9.2 cm across,
   ±4.1 cm in z).
5. **Fingers cross-check:** the outer-face distance fully open and closed, finger thickness and
   length (URDF says ±8.15 cm open at the pads, 8.48 cm at the roots, 3.84 cm thick, 8.9 cm long).

Rule: every face moves outward to the next 5 mm. A missing number keeps today's slab for that face.

## 9. Pinned tests and contracts to update

1. **Keep passing unchanged:**
   - `test_home_routes.py:113` (19 mm refuses home);
   - `:99` (payload never escapes);
   - `test_rebot_occupancy_live.py:108/149/172/194`;
   - the 0.03 pins at `test_home_routes.py:109`, `test_clearance_lift_route.py:106/371/390`,
     `test_place_path_safety.py:83` and `test_kitchen_acceptance_configuration.py:29`.
2. **Update with L4:** `test_harness_tool_volume.py:94/118/163-222/258` (box shape, two layers
   one box, retreat), and `test_clearance_lift_route.py:345` (its obstacle offset is defined
   against today's slab; derive it from the config).
3. **Update with L2:** `test_pregrasp_clears_object_top.py:58/121/137` (hover = max(fix, map) top).
   Add tests for `pregrasp_map_top`.
4. **Update with L5/L6:**
   - `test_approach_target_column.py` keeps the legacy mode and adds object-aware cases;
   - `:168-178` keeps the profile pins;
   - `test_harness_tool_volume.py:236` gains `exclude_keys`;
   - new `test_corridor_margin_measured.py` (§4.6).
5. **Docs:**
   - `docs/ARCHITECTURE.md:640-652` ("the configured 30 mm clearance remain[s] rejection
     conditions"): add the corridor exception;
   - `docs/NVBLOX.md:269-275` (inflate < margin) and `:551+` (target approach);
   - `rebot_rs.yaml:67-81` and `:108-114` comments: extend, never delete (AGENTS.md:103);
   - `tests/test_wrist_narration.py` golden: no change expected (mock has no map).

## 10. AGENTS.md proposed text (protected file: for the PR body, not edited here)

Append to **Motion safety path** (AGENTS.md:61):

> TCP/joint occupancy clearance has an escape rule: a step from inside `min_clearance_m` is
> admitted only if every point already inside strictly gains clearance (or the step is an exact
> hold) and none enters, and planners accept a route only if its final pose is outside the margin
> (`vet_route`, spatial `plan_route`). Attached-payload samples never escape. Every clearance
> refusal/halt files a bounded map snapshot into the trace (`context.clearance`). With
> `grasp.object_aware.enabled`, map voxels are labelled from the scene camera's same-frame SAM
> masks (depth band + fitted prism, F1-F4 freshness), frozen per grasp episode; exactly the
> target's voxels are exempt (the descent cylinder governs only the table rules), and fingers/jaws
> may use `grasp.object_aware.corridor_margin_m` against known-neighbour prisms between the
> pregrasp and the lift's return to it. That value is pinned to a committed measurement
> (`test_corridor_margin_measured.py`), never below 1.0 cm; arm body, wrist camera, housing,
> payload and unknown voxels keep `min_clearance_m` everywhere. Stale or doubtful labels are
> "unknown"; never fall back to the legacy column silently.

## 11. Owner decisions (2026-10-16: "Ok all" on the recommended answers)

- **Q1: (a).** Corridor floor 1.0 cm, enabled only after the fit-error measurement (§4.6) passes; until
  then the corridor margin equals `min_clearance_m` (3 cm), so PR-B ships with no margin change.
- **Q2: yes.** The corridor ends when the lift has retraced the descent back to `q_pre`; the
  pregrasp pose itself keeps 3 cm.
- **Q3: yes, as the next PR** (adaptive pre-grasp opening), out of scope here.
- **Q4: report-only.** `robot`-labelled voxels are reported, never exempt, in this spec.
- **Q5:** calipers and the 20-minute no-motion capture are pending owner input; lever 4 and the
  corridor-margin value wait for them, nothing else does.

The original questions, for the record:


1. **Q1, corridor margin below the 2 cm body mask.** New frames hide anything within ~2 cm of the
   fingers (§1.3.4). Below 2 cm, `m_corr` holds against the frozen prism only, so a neighbour
   that moves during the 2-4 s corridor is seen only by the contact monitor. Pick one:
   - (a) accept it, floor 1.0 cm (as approved);
   - (b) floor at 2.0 cm;
   - (c) measure the finger mask error and reduce the finger links' inflate first.

   Recommendation: (a) for the corridor only, after B1 passes. Otherwise (b).
2. **Q2, corridor end.** Confirm "lift start" means the lift retracing the descent back to `q_pre`,
   and that the pregrasp pose itself keeps 3 cm.
3. **Q3, neighbours along the jaw axis.** These are not admissible with the jaws fully open
   (§1.3.3). Should the adaptive pre-grasp opening be the next PR, after measuring the width map?
4. **Q4, self-voxels.** 2 placeable camera hits sit inside the URDF housing at ≤ 1 mm. Lever 3
   will show how many of the 28 high-z calls are self or stale. Keep `robot`-labelled voxels
   report-only (this spec), or decide later to exempt voxels inside the unpadded URDF body?
5. **Q5, calipers (§8)** and the rig capture for the size term (§4.6.2, about 20 min,
   operator-gated, no motion).
