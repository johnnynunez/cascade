# base_pose arms: the skill runtime works in the running arm's base frame (Option 2)

Status: design approved by the owner on 2026-10-17 (Q1 a, Q2 a, Q3 a, Q4 a).
Branch `feat/base-pose-runtime-frame`. Plan: `docs/superpowers/plans/2026-10-17-base-pose-runtime-frame.md`.
Sources: the Step 0 design report (`cascade-wt/base-pose-guard/.superpowers/sdd/base-pose-guard/design-report.md`,
§1.1-1.4, §2 Option 2, "Fix round 1" addendum), the #352 review (Minors 1-4) and the PR-A final review (M6).
Line numbers are for main `ec2d95c`. Re-anchor them by symbol after PR-A merges (§8).

## 0. Decisions on one screen

1. **One frame inside a skill.** Inside a skill call, everything is in the RUNNING arm's base frame. The call
   boundary converts:
   - in: camera transforms at target sites, belief reads used as targets, agent coordinates;
   - out: results, belief writes, configured zones, viewer overlays.
2. **The agent uses the table frame (Q1 a).** Agent-facing coordinates are in the shared TABLE frame, in both
   directions. Without a base_pose the table frame is the base frame, so nothing an agent sees changes there.
3. **`execute(arm=X)` binds X's own frame (Q4 a).** For that one call it binds X's Kinematics, X's resolved
   profile (the `arm`, `grasp`, `safety` and `relative_motion` sections) and X's base_pose. The binding is
   thread-local, beside `_arm_override`. This also fixes so101_right using so101_left's `drop_zone`.
4. **Fail closed.** These calls are refused before any work:
   - a selected arm with no bound kinematics or profile;
   - an arm whose base_pose is unreadable or not rigid.

   A tilted base_pose (any roll or pitch) still refuses the 15 world-target skills: this is #352's guard,
   narrowed to tilt. No shipped profile is tilted.
5. **`T_world_base is None` means no transform anywhere.** "None" covers no base_pose and the exact identity.
   Every helper then returns its input object.
   - Commands, results and tool specs stay byte-identical on the reBot, the kitchen arm, mock and so101_mock.
   - The one exception: the `list_arms` sentence that #352 added is removed.
6. **The guard shrinks one skill at a time.** A skill leaves `_WORLD_TARGET_SKILLS` in the same commit whose
   tests prove that it reaches the world point.
7. **Merge order.** PR-A (`feat/nvblox-object-aware`, option C) merges first. This branch then rebases onto that
   main and also fixes PR-A's review finding M6.
8. **Acceptance.** The so101 pair places a picked object at the world target:
   - mock pair in CI: the FK of the commanded joints, lifted by base_pose, lands within 5 mm;
   - MuJoCo pair in a scratch venv: the physics position lands within 3 cm (`DEST_TOLERANCE_M`).

## 1. Problem (verified on main ec2d95c)

- **Two frames, no conversion.** Perception, beliefs, the occupancy map and the MuJoCo world use the table frame
  (camera `extrinsics.T`; the MuJoCo rig attaches each arm at its base_pose). The runtime solves IK in the running
  arm's base frame. `git grep base_pose -- src/cascade/skills src/cascade/grasping` finds only #352's guard.
- **Measured before #352 (report §1.2):**
  - so101_right's lowest TCP was at world (0.201, -0.198) for an object at (0.198, 0.000): 22 cm off;
  - place_at(0.25, -0.10) landed at (0.250, -0.320);
  - the verifier still said ok, and place_at wrote a belief where the object was not.
- **Exemption columns sit in the wrong place.** They are built from the same numbers. The map gate therefore
  exempts whatever stands where the arm really descends (report §1.4).
- **`execute(arm=)` swaps only the arm.** It rebinds only `self.arm`. `self.kin` and `self.cfg` stay the
  primary's (`demo.py:688` `if i == 0: kin = k`). So so101_right uses so101_left's `drop_zone` [0.20, 0.12], and a
  mixed-model rig would solve IK with the wrong robot.
- **Today the pair cannot pick at all.** #352 (merged in f6d3880) refuses the 15 world-target skills on any
  non-identity base_pose.

## 2. Owner decisions

| Q | Answer | Meaning |
|---|---|---|
| Q1 | a | Agent coordinates, sent and read, are in the shared table frame. |
| Q2 | a | The Option 1 guard ships first as its own PR. Done: #352. |
| Q3 | a | Option 2: the runtime works in the running arm's base frame. |
| Q4 | a | `execute(arm=)` binds that arm's own Kinematics and resolved profile for each call. |

The owner said the same day: "stop asking; decide the rest with the recommended option". The decisions D1-D10
below were made that way. The owner can override any of them.

- D1: the frame boundary is the skill call. Composites share the bound frame.
- D2: `T_world_base` comes from the running arm's harness `base_pose`, read through instance dicts. The identity
  counts as None.
- D3: the primary keeps the runtime's own `kin` and `cfg` objects. Each other arm gets its pre-pass Kinematics
  and a Cfg built once, with the arm-scoped sections taken from its resolved view.
- D4: fail closed when the binding or the pose is missing or broken. A tilted mount refuses only the world-target
  skills.
- D5: an eye-in-hand camera on a rig with any base_pose refuses to build.
- D6: WorkspaceFilter stays in the table frame and excludes every arm's base. Reach counts from any base.
- D7: agent text changes only on rigs that declare a base_pose. The #352 `list_arms` sentence is removed.
- D8: directions stay relative to the running arm's base: push_object, throw, bare move_relative. Only the
  viewer frame is camera-relative.
- D9: the derived envelope scalars and the held offset stay in the arm frame. A release on the arm that does not
  hold the object is refused (`_held_arm`).
- D10: the guard shrinks one skill at a time, then becomes a tilt-only refusal.

## 3. Design

### 3.1 Frames

- **Table frame W:** camera extrinsics, the occupancy map, beliefs, the MuJoCo world, agent coordinates.
- **Arm base frame B:** FK and IK, the harness workspace, `table_z` and keep-outs, per-arm config positions,
  the held offset and the derived envelope scalars.
- **`T_world_base`:** the base_pose as a 4x4 matrix (W <- B), so p_W = T p_B. It is level when the z row of R
  is [0, 0, 1] (atol 1e-9, the same tolerance #349 uses).

### 3.2 The per-call arm frame (D1-D5)

New module `skills/arm_frame.py`:
- `ArmBinding(kin, cfg, grip)`: built once per arm by `build_runtime`;
- `ArmFrame(name, T_world_base, binding)`: resolved for each `execute()` call.

**D1: the boundary is the skill call.** Composites call `skill_*` methods directly and share the frame. Nothing
inside a skill converts twice.

**D2: one source for `T_world_base`.** It is the running arm's harness `base_pose`, read through instance dicts
only (the #352 `_mounted_off_table_origin` pattern). This never materializes a LazyArm, and a Mock never "has" a
pose.
- None or the exact identity -> None.
- Not a finite rigid 4x4 (|RᵀR - I| <= 1e-9, det R = 1) -> `ArmFrameError`.
- Because the harness lifts its own map queries with the same matrix, the runtime and the harness cannot
  disagree.

**D3: bindings.**
- The primary uses the runtime's own `kin` and `cfg` objects, uncopied, so its path is byte-identical.
- Every other arm gets two things:
  - its pre-pass Kinematics (`_ArmPrepass.kin`, the one its harness already uses);
  - a Cfg built once: the top-level data, with the sections in `ARM_SCOPED_SECTIONS = ("arm", "grasp",
    "safety", "relative_motion")` taken from `cfg.arms[i].resolved`. Building it once keeps the
    `held_observation.binding()` identities stable across calls.
- Overrides that belong to the whole rig stay top-level: `workspace_filter`, `memory`, `perception_loop`,
  `detector`, `object_descriptions`, `occupancy`.
- A test walks the `overrides:` keys of every shipped profile, so a new key forces a choice.

**Properties.** `SkillRuntime.kin`, `.cfg`, `.T_world_base`, `._grip_open`, `._grip_closed` and `._max_width`
become properties.
- They read the bound frame (`_arm_override.frame`), or the primary's when nothing is bound.
- Their setters write the primary.
- The frame is bound and restored exactly where `_arm_override.arm` is today (runtime.py:1511/1560).
- Other threads see the primary: the MCP stop reader, the stream server and the watcher.

**D4: fail closed.** At `_execute_skill`'s guard site (runtime.py:1353-1357), before any perception, keyframe,
snapshot or command, these calls are refused:
- a selected non-primary arm with no registered binding;
- an `ArmFrameError`.

The error reads `"<skill> refused on arm 'X': <why>; no motion was sent"`. The result carries no move_home hint.

**D5: eye-in-hand cameras.** A rig with an eye-in-hand camera and any base_pose arm refuses to build, in the
pre-pass, before any arm is made. The reason: `Extrinsics.cam_to_base()` composes the PRIMARY's FK, which is a
base frame, not the table frame (demo.py:718-727, 748). No shipped combination exists: the MuJoCo loader already
refuses wrist cameras on a rig.

### 3.3 Inbound: table frame to arm frame (only at target sites)

| Input | Sites (ec2d95c) | Rule |
|---|---|---|
| Camera transform of a frame analysed for a target | `_localize` (2768: primary, two-phase, side cameras), `_vlm_ground_fix` (3133), `skill_grasp_at_pixel` (7616), `_destination_fix` (5254), `held_observation.measure` image path (211-228) | `frame_to_arm`: `T_base_cam := T⁻¹ (frame.T_base_cam or ext.cam_to_base())` |
| Map refresh and belief fusion | `_update_beliefs_from_frame` (1938), `_reobserve` (3248), `_reset_camera_frames` (6981), WorldWatcher | unchanged (table frame) |
| Belief positions used as targets | `_resolve_query` near_xyz (2393), `_localize` belief fallback (3088-3118, `_fix_from_belief`), `_restore_plan` (6471), `sort_by_color` (6204), `search_for_object` (6704), `observed_scene.for_runtime` tracked objects (799-801), held observation's legacy pose channel (194-201) | `to_arm` / `fix_to_arm` |
| Agent coordinates | `place_at(x, y, z)` (4916); z omitted -> the arm's own release height, computed in B after converting (x, y) | `to_arm` |
| Viewer directions | `spatial_moves.viewer_axes` (242) | camera rotation through `cam_to_arm` |
| Observed-finger gate | `observed_scene.for_runtime` (789-797): the arm-frame frame's T, and its static-T fallback through `cam_to_arm` | arm frame, matching the base-frame FK finger hulls |

Once every perceived point is in B, several things hold with no further code:
- the base-origin heuristics (top-down yaw `atan2(y, x)`, reach annuli, the 5-DoF approach plane);
- every exemption column (`_target_column`, `allow_grasp_descent`, the place, contact and release columns);
- the latent `table_z` comparisons of §3.7.

### 3.4 Outbound: arm frame to table frame

- **Results.** Every position, TCP or xy field goes through `to_world`, directions through R, and extents stay
  as they are. This covers localize_object, point_at, get_observation `tcp_xyz`, the grasp, place and push
  reports, the move_near_object target and the configured zones. The equivariance test (§6.1) classifies every
  numeric 3- or 2-vector key in every result.
- **Belief writes** go through `fix_to_world` / `to_world`, with `top_z` taken from the world points:
  localize_object, place_at, `_destination_fix`, place_on_object, push_object, and `_promote_held`'s
  `mark_removed(near=)`.
- **Traces** keep the args (agent input, frame W) and the results (now W). `grasp_evidence.localized` adds
  `T_world_base`, so an offline replay can lift. Motion evidence is joint space and needs nothing.
- **Viewer overlays** (`visual_interface`, the stream server) lift the TCP and the workspace box by the primary's
  T, because viewers run on unbound threads.
- **D9: what stays in B.** The derived envelope scalars (`tcp_z_at_grasp_m`, `object_height_m`, ...) and the held
  offset. The held offset belongs to the grasping arm. `_promote_held` records that arm as `_held_arm`, and
  place_at, place_on_object, throw and handover on any other arm are refused before any command. The
  held-observation binding, which already invalidates the offset on another arm or Cfg, stays as a second layer.

### 3.5 Agent contract text (D7; #352 Minors 1-3)

- **`tool_specs_for_config`** changes the coordinate tools when any arm declares a base_pose, including
  so101_right alone (Minor 1: `list_arms` is withheld on a single-arm rig, so text placed there is never seen).
  - "base-frame" and "robot base frame" read "table frame".
  - One sentence is added: "the same numbers for every arm on this rig".
  - Rigs with no base_pose keep byte-identical specs.
- **`list_arms`:** #352's sentence is removed.
- **Tilted-mount refusal (Minor 2).** Its `next_action` names only skills that complete on so101: move_home,
  move_to_rest, rotate_joint, open_gripper, close_gripper. move_relative and rotate_gripper refuse at the so101
  home pose.
- **README and so101 headers (Minor 3).** The README multi-arm line (README.md:193) and the so101_left / _right /
  _left_mujoco / _right_mujoco headers state the contract:
  - base_pose is where the base stands in the table frame;
  - agent coordinates are table-frame;
  - `drop_zone` is in this arm's base frame;
  - the advertised pick-and-place works.

### 3.6 Per-arm config positions

Per-arm config positions stay in the arm's base frame and are read from the running arm's profile (§3.2):
- so101_right's `drop_zone` [0.20, -0.12] is world (0.20, -0.34);
- so101_left's [0.20, 0.12] is world (0.20, 0.34).

Reported zones go out in W.

### 3.7 Latent mixed-frame `table_z` sites (report, fix round 1 addendum)

`table_z` is a base-frame height. Most sites below compared it with perceived points, which are table-frame
today. Every shipped base_pose has z = 0, which hides the mismatch.

| Site (ec2d95c) | After Option 2 |
|---|---|
| `_angled_clearance_check` support_z, runtime.py 2695 | consistent (the fix is in B) |
| VLM bbox shave above `table_z`, 3196-3197 | consistent |
| `object_height_m` feature, 3622-3628 | consistent |
| grasp `z_floor`, 3683-3684 | consistent (its inputs were already in B, and the candidates now are too); pinned |
| `_grasp_support_height`, 4651 | consistent |
| push contact height, 5386-5391 | consistent |
| place release height (`skill_place_at` 4929-4931), `_reachable_place_xy` (6328) | consistent once x and y are in B |
| obb_grasp `object_vertical_span` and grasp floor, 308-431 | consistent |
| gripper_clearance `object_box` / `target_box`, 122-153 | consistent |
| mujoco_withdrawal: MuJoCo plane z vs `table_z` (243), released body vs `table_z` (483) | world vs base: compare with `table_z_in_world(T, table_z)` |

Each site gets a RED test on a base 3 cm above the table plane (`base_pose` z = 0.03, `safety.table_z: -0.03`).

### 3.8 WorkspaceFilter (D6)

- The filter stays in W. It takes `bases`: each arm's base origin in W, with that arm's resolved `base_radius_m`
  and `reach_m`.
- It rejects as "robot" any point inside any base's cylinder, and as "out_of_reach" only a point beyond every
  base's reach.
- `bases=None` (no arm declares a base_pose) keeps today's origin checks.
- The watcher and the runtime share one filter. Arm-frame callers (grasp_at_pixel) pass `to_world(center)`.

### 3.9 PR-A review M6 (lands on this branch)

- **The pregrasp map read.** `_pregrasp_min_z` (with `grasp.pregrasp_map_top`) queries
  `occupancy.column_top(center, r, z_min)` with base-frame numbers. The fix:
  - lift the centre and `z_min` into the map frame with the harness's own `_to_map_frame`;
  - bring the returned top back with `_to_base_frame`.
- **The air-grasp exit.** The exit column (`_column_geometry` built from the fix) and its ceiling (hover FK +
  5 mm) are both in B once the fix is. A test pins that the harness compares arm points and map voxels with the
  4-tuple cylinder in B (`_in_cylinder` after `_to_base_frame`).
- No shipped profile reaches either today: `pregrasp_map_top` is true only on rebot_rs and rebot_rs_reach, and
  neither has a base_pose.

### 3.10 The guard (D10)

- `_WORLD_TARGET_SKILLS` loses each skill in the task that converts it.
- The last conversion task replaces `_mounted_off_table_origin(arm)` with a tilt check on the bound frame.
- The call-graph derivation widens its sources (Minor 4):
  - `_localize`, `_belief_pose`, `fix_at_pixel`;
  - any `beliefs.` read;
  - `T_base_cam`, `cam_to_base`, `extrinsics`, `viewer_axes`;
  - x/y parameters.
- Every skill the derivation finds is either in the equivariance test or listed as frame-free with a reason.

## 4. Guarantees

- **No base_pose, or the identity:**
  - `arm_frame._rigid_apply` is never called, and every helper returns the same object;
  - the command, jaw, result, belief and tool-spec diff against main is 0 rows on rebot_rs (MockArm),
    isaac_kitchen (MockArm, mock camera), mock, so101_mock and the identity pair (calls on its primary).
  - `execute(arm=<non-primary>)` uses that arm's own profile (Q4 a), so a non-primary call can differ from main
    by design. Example: pick_and_place's default drop zone.
- **A level base_pose arm:**
  - sends the same commands as the matching no-base_pose arm whose scene moved with it (equivariance);
  - its results and beliefs are those of the no-base_pose arm, transformed by T.
- **Margins unchanged:** `min_clearance_m` 0.03, table band 0.02, `neighbor_clearance_m` 0.02,
  `DEST_TOLERANCE_M` 0.03.

## 5. Failure behaviour

| Condition | Result | Commands | Perception |
|---|---|---|---|
| Non-primary arm, no ArmBinding | refused: "has no kinematics/profile bound to this runtime" | 0 | none |
| base_pose unreadable or not rigid | refused: "base_pose is not a rigid transform" | 0 | none |
| Tilted base_pose + world-target skill | refused: tilt text with roll and pitch in degrees, plus the §3.5 `next_action` | 0 | none |
| Tilted base_pose + joint or relative skill | runs as today | as today | as today |
| Eye-in-hand camera + any base_pose arm | the build refuses in the pre-pass | 0 make / 0 connect | none |
| Release skill on the arm that does not hold the object | refused by `_check_holding_arm`: "<obj> is held by arm 'Y', not 'X'" (the held-observation binding stays a second layer) | 0 | none |
| `place_at` with a MuJoCo region plan on a base_pose arm | refused: "region placement is not supported on a base_pose arm" | 0 | none |
| Anything else on a level base_pose | the same as a no-base_pose arm with the scene moved | equivariant | equivariant |

## 6. Tests

1. **Equivariance oracle (the core RED set).**
   - Rig A: so101_right without its base_pose.
   - Rig B: the same arm with base_pose T, for T in {shipped [0, -0.22, 0, 0, 0, 0], yawed [0.03, -0.22, 0, 0,
     0, 0.4]}. Every camera's `extrinsics.T` becomes T · T_A, so the scene moves with the arm.
   - Both rigs run the same seeded calls; agent coordinates are mapped by T.
   - Arm commands must be equal within 1e-6 rad, and jaw commands equal.
   - Every numeric key in the results must be classified:
     - a point maps to T·a within 1.1 mm (after rounding to the mm);
     - a direction maps to R·a;
     - a frame-free value stays equal.
   - Beliefs must map to T·a within 1e-6 m.
   - On main, B is refused by #352, or, with the guard lifted, it sends different commands.
2. **World point.** The setup:
   - the shipped so101_right with the camera fixed in W;
   - the camera shifted 10 cm in -y, which puts the object at W (0.198, -0.10) (right-arm r = 0.231, inside the
     verified 0.12-0.28 annulus);
   - also the pair, with `arm=so101_right`.

   Three FK points, lifted by base_pose, must lie within 5 mm of rig A's matching point mapped by T: the grasp's
   lowest TCP, the place release TCP, and the push segment. They must also lie within 2.5 cm of the world object
   (22 cm on main).
3. **Columns and the gate.**
   - `allow_grasp_descent` exempt_xy and the `target_column_exemption` centre, lifted, lie within 5 mm of the
     lifted grasp point.
   - `observed_scene.for_runtime` builds its ObservedScene with T⁻¹ T_table_cam and tracked objects in B.
4. **`table_z` sites:** on a base 3 cm above the table (§3.7).
5. **Byte-identical diff and no-transform pin (§4).** `_rigid_apply` is patched to raise.
6. **Agent text.**
   - Rigs with no base_pose keep byte-identical specs.
   - On so101_right alone and on the pair, every coordinate tool says "table frame".
7. **Mutations** for each task (listed in the plan).

## 7. Acceptance

- **Mock pair (CI).** Setup: `--arms so101_left,so101_right --camera mock_small`, object at W (0.198, -0.10).
  - `pick_and_place(object="red cube", arm="so101_right")`: the lifted release TCP lies within 5 mm of the
    right drop zone in W, (0.20, -0.34), corrected by the held offset.
  - `grasp_object`, then `place_at(x=0.22, y=-0.30, arm="so101_right")`: the same check against (0.22, -0.30).
  - The beliefs end within 5 mm of each target. so101_left sends no command.
- **MuJoCo pair** (scratch venv layered on the rig venv, mujoco 3.12.0 and websockets 17.2 from uv.lock;
  `scripts/fetch_robot_assets.py so101`).
  - Setup: `--arms so101_left_mujoco,so101_right_mujoco --camera mock_small`, the same two sequences.
  - The prop's physics position (MuJoCo truth channel, W) lies within 0.03 m of each target.
  - The postcondition is CONFIRMED by physics, and the left arm's joints do not move.
  - Control run first: a single-arm `so101_mujoco` + `mock_small` pick_and_place passes on main.
- **Full suite.** The failing-ID diff against main shows no new IDs.

## 8. Merge order

1. PR-A (`feat/nvblox-object-aware`, head 73bd9eb) merges. It rewrites the runtime's halt, pregrasp and air-grasp
   exit code, which this plan edits.
2. Rebase this branch onto that main. It holds only docs so far, so this is cheap; keep a backup ref. Then
   re-anchor the plan's line numbers by symbol.
3. Run Tasks 1-8. If another runtime PR merges meanwhile, run `git merge --no-ff origin/main` before the
   pre-final round.
4. Open a draft PR. Mark it ready after the final review, both acceptance runs and green CI.

## 9. Non-goals and known limitations

- `held_object` stays runtime-global: one held object per rig.
- Directions stay relative to the running arm's base: push "left", throw "forward", bare move_relative.
- A tilted base_pose keeps refusing the world-target skills.
- Isaac multi-arm rigs are out of scope; no profile ships one.
- The MuJoCo placement and withdrawal observers (`mj_delivery_area`, `mj_release_*`) stay single-owner and are
  not declared on rig arms.
- One envelope file still serves all arms; each arm's scalars are in its own frame.

## 10. Open questions

None block the work. D1-D10 are the recommended options under the owner's "decide with the recommended option"
rule. Any of them can be overridden before Task 1 starts.
