# base_pose Runtime Frame (Option 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for
> tracking.

**Goal:** A world-target skill on an arm mounted with a `base_pose` reaches the world point. The skill runtime
works in the running arm's base frame and converts at the skill-call boundary. Arms without a base_pose stay
byte-identical.

**Architecture:**
- `execute()` binds a per-call `ArmFrame` beside `_arm_override`, thread-local. It holds `T_world_base` and the
  running arm's Kinematics and resolved profile.
- `SkillRuntime.kin`, `.cfg` and the gripper constants become properties that read that binding.
- Inbound, frames, belief targets and agent coordinates go through one helper module `skills/arm_frame.py`.
- Outbound, results and belief writes go back through the same module.
- #352's guard shrinks one skill at a time and ends as a refusal for tilted mounts only.

**Tech Stack:** Python 3.12, numpy, pytest on the mock stack (`build_runtime`, MockArm). Pinocchio comes from the
rig venv. MuJoCo 3.12.0 runs only in a scratch venv for acceptance.

**Spec:** `docs/superpowers/specs/2026-10-17-base-pose-runtime-frame-design.md` (read §0, §3 and §5 before any
task).

## Global Constraints

- **Order.** Start only after PR-A (`feat/nvblox-object-aware`) has merged into main and pre-flight P0 is done.
- **Where to work.** Work only in `/home/spark/Projects/cascade-wt/base-pose-frame2`.
  - Never touch `/home/spark/Projects/cascade`. It serves the live robot.
  - No ports, no GPU, no hardware.
  - Pass an explicit `workdir=` on every terminal call.
- **Tests.** Run them as `cd /home/spark/Projects/cascade-wt/base-pose-frame2 && PYTHONDONTWRITEBYTECODE=1
  PYTHONPATH=$PWD/src /home/spark/Projects/cascade/.venv/bin/python -m pytest <files> -q -p no:cacheprovider`,
  with `CASCADE_GRASP_BACKEND=obb` exported by every new fixture.
- **No transform without a base_pose.** When `T_world_base is None` (no base_pose, or the exact identity), every
  `arm_frame` helper returns its input object unchanged and `_rigid_apply` is never called.
- **Byte-identical.** rebot_rs, isaac_kitchen, mock and so101_mock must keep identical commands, jaw commands,
  results, beliefs and tool specs. The one allowed change is removing #352's `list_arms` sentence.
- **Instance dicts only.** Read arm and harness state as `arm.__dict__.get("harness")` and
  `harness.__dict__.get("base_pose")`. Never use `getattr` on an arm: it can materialize a LazyArm.
- **Frames.**
  - Agent coordinates are in the TABLE frame, in and out (Q1 a).
  - Directions (push, throw, bare move_relative) are relative to the arm's base (D8).
  - Per-arm config positions (`drop_zone`, `table_z`, `source_support_top_z_m`, workspace) are in that arm's
    base frame.
- **Arm-scoped sections:** `ARM_SCOPED_SECTIONS = ("arm", "grasp", "safety", "relative_motion")`.
  **Rig-scoped overrides:** `{"workspace_filter", "memory", "perception_loop", "detector",
  "object_descriptions", "occupancy"}`.
- **Margins stay as they are:** `min_clearance_m` 0.03, table band 0.02, `neighbor_clearance_m` 0.02,
  `DEST_TOLERANCE_M` 0.03, tilt tolerance 1e-9.
- **Tolerances:**
  - world point: 5 mm (FK of the commanded joints, lifted by base_pose);
  - equivariant commands: 1e-6 rad;
  - rounded result points: 1.1e-3 m;
  - belief positions: 1e-6 m;
  - absolute sanity: within 2.5 cm of the world object (main is off by 22 cm).
- **Poses** (in `tests/base_pose_oracle.py`):
  - `SHIPPED_RIGHT = [0.0, -0.22, 0.0, 0.0, 0.0, 0.0]`
  - `YAWED = [0.03, -0.22, 0.0, 0.0, 0.0, 0.4]`
  - `RAISED = [0.0, -0.22, 0.03, 0.0, 0.0, 0.0]`, used with `safety.table_z: -0.03` and workspace min z -0.05
  - `TILTED = [0.0, -0.22, 0.0, 0.1, 0.0, 0.0]`
- **Agent text.** Every failed result keeps exactly one FAILED line (`_with_failure_instruction`). A refusal
  before any work carries no move_home hint and no `arm_left_in_place`.
- **Mutations.** Run them with `bash ~/.hermes/skills/robotics/cascade-feature-worktree/scripts/mutate.sh
  <file> '<old>' '<new>' <tests>`. Every listed mutant must come back KILLED by a behavioural assertion, not by an
  import error.
- **Commits.** One commit per task, with the message given. Never push.

## Review Focus

1. **A thread started inside a bound skill** reads `kin`, `cfg` or `arm` and sees the primary instead. Pinned in
   Task 1 by an AST census: there is no `threading.Thread(` or `.submit(` under `skills/` and `grasping/` except
   in `skills/mobile_runtime.py`.
2. **A release on the arm that does not hold the object** must be refused through the held-observation binding,
   with 0 jaw commands. Pinned in Task 5 (place_at) and Task 6 (throw, handover).
3. **Resolving the frame of a standby LazyArm** must never materialize it, refusal paths included. Pinned in
   Task 1.
4. **Viewer and MCP threads during a bound `arm=` call** see the primary's frame. Pinned in Task 7.
5. **An identity base_pose behaves exactly like no base_pose** (results, commands, tool specs). Pinned in Task 1
   (unit) and Task 8 (the diff rig `pair_identity`).

## File structure

| File | Role |
|---|---|
| `src/cascade/skills/arm_frame.py` (new) | `ArmBinding`, `ArmFrame`, `ArmFrameError`, `world_from_base`, `is_level`, `tilt_deg`, point/direction/frame/fix converters, `table_z_in_world`, `arm_view_cfg`, `arm_binding`, `_rigid_apply` (the only matrix multiply) |
| `src/cascade/skills/runtime.py` | properties, binding in `_execute_skill`, conversion at the sites in §3.3-3.4 of the spec, the guard -> tilt refusal |
| `src/cascade/apps/demo.py` | registers the bindings, refuses eye-in-hand on a base_pose rig, builds the WorkspaceFilter bases |
| `src/cascade/perception/workspace.py` | `WorkspaceBase`, `bases=` |
| `src/cascade/grasping/observed_scene.py`, `grasping/evidence.py`, `skills/held_observation.py`, `skills/spatial_moves.py`, `skills/mujoco_withdrawal.py`, `perception/visual_interface.py`, `apps/stream_server.py` | site conversions |
| `tests/base_pose_oracle.py` (new helper) | moved-rig configs, argument mapping, result-key classification, equivariance assertions |
| new tests | `test_arm_frame_context.py`, `test_workspace_filter_bases.py`, `test_base_pose_runtime_frame.py`, `test_base_pose_table_z_sites.py`, `test_base_pose_tilted_mount.py` (replaces `test_base_pose_world_target_guard.py`), `test_base_pose_agent_text.py`, `test_base_pose_no_transform.py`, `test_base_pose_pair_pick_place.py`, `test_base_pose_mujoco_pair_pick.py` |

## Anchor table (main ec2d95c; P0 re-anchors these by symbol)

| Symbol | File:line |
|---|---|
| `_WORLD_TARGET_SKILLS`, `_mounted_off_table_origin`, `_world_target_refusal` | runtime.py:362-410 |
| `SkillRuntime.__init__` (kin/cfg 638/653, grip 789-792, `_workspace` 798) | runtime.py:620-812 |
| class defaults, `arm` property | runtime.py:815-863 |
| `_execute_skill`: guard 1353-1357, bind 1511-1512, restore 1560 | runtime.py:1329-1830 |
| `_update_beliefs_from_frame` / `_tcp` / `_tcp_if_awake` | runtime.py:1938 / 2016 / 2019 |
| `_resolve_query` / `_plan_grasps` / `_hug_candidates` / `_camera_frame_candidates` | runtime.py:2380 / 2418 / 2554 / 2621 |
| `_angled_clearance_check` / `_localization_workspace_bounds` / `_localize` | runtime.py:2670 / 2725 / 2768 |
| `_vlm_ground_fix` / `_reobserve` / `_fix_from_belief` | runtime.py:3133 / 3248 / 3284 |
| `_describe_observation` (tcp_xyz 3418) / `_configured_zones` / `skill_localize_object` / `skill_preview_grasp` | runtime.py:3397 / 3452 / 3486 / 3516 |
| `skill_grasp_object` (object height 3618-3628, z_floor 3683, `_vet` 3819) / `_promote_held` | runtime.py:3571 / 4523 |
| `_grasp_support_height` / `skill_place_at` / `_destination_fix` / `skill_place_on_object` / `skill_push_object` / `skill_pick_and_place` | runtime.py:4644 / 4916 / 5254 / 5320 / 5376 / 5425 |
| `skill_point_at` / `skill_handover` / `skill_throw` / `skill_turn_screw` / `skill_sort_by_color` / `_reachable_place_xy` / `_move_object_to` | runtime.py:5870 / 5925 / 5956 / 6042 / 6204 / 6314 / 6393 |
| `_restore_plan` / `skill_restore_scene` / `skill_search_for_object` / `skill_grasp_at_pixel` / `skill_annotated_view` | runtime.py:6471 / 6566 / 6704 / 7616 / 7463 |
| `TOOL_SPECS` `list_arms` / `tool_specs_for_config` | runtime.py:8633-8645 / 8683 |
| `_target_column` (PR-A: `_column_geometry`, `_pregrasp_min_z`, `_air_grasp_exit`) | runtime.py:78 (73bd9eb: 97 / 130 / 4729) |
| `_arm_safety` / `_build_runtime` arm loop / `SkillRuntime(...)` / watcher | demo.py:247 / 674-709 / 861-880 / 890-905 |
| `observed_scene.for_runtime` | observed_scene.py:789-806 |
| `held_observation.binding` / `measure` | held_observation.py:29 / 113-290 |
| `spatial_moves.viewer_axes` / `_target_geometry` / `move_near_object` | spatial_moves.py:214 / 869 / 892 |
| `mujoco_withdrawal` table sites | mujoco_withdrawal.py:243, 483 |
| `WorkspaceFilter.reject` / `from_config` | workspace.py:145 / 185 |

## Pre-flight P0: after PR-A merges (no product code)

- [ ] Confirm PR-A merged: `git -C /home/spark/Projects/cascade fetch -q origin` (read-only), then check that
      `git -C /home/spark/Projects/cascade-wt/base-pose-frame2 merge-base --is-ancestor 73bd9eb origin/main` exits 0.
- [ ] `git branch backup/base-pose-runtime-frame-pre-rebase && git rebase origin/main` (the branch holds only
      these two docs). Record the base sha as `BASE` in the ledger.
- [ ] Append `.superpowers/` to `$(git rev-parse --git-common-dir)/info/exclude`, then check
      `git check-ignore -q .superpowers`.
- [ ] Copy the evidence into `.superpowers/sdd/base-pose-runtime-frame/`:
  - `diff_probe.py`, `compare.py` and `probe_targets.py` from
    `/home/spark/Projects/cascade-wt/base-pose-guard/.superpowers/sdd/base-pose-guard/`;
  - this plan's Global Constraints, as `global-constraints.md`.
- [ ] Re-anchor the table above by symbol (`grep -n "def <symbol>"`). PR-A renamed or added `_column_geometry`,
      `_pregrasp_min_z`, `_air_grasp_exit` and `_column_exemption(harness, column, z_max=None)`. Write the moved
      anchors into the ledger.

---

### Task 1: The per-call arm frame (Q4 a, D1-D5)

**Files:**
- Create: `src/cascade/skills/arm_frame.py`
- Modify: `src/cascade/skills/runtime.py` (class defaults ~815-837, `__init__` 638/653/789-792, `_execute_skill`
  1340-1357 + 1511-1560)
- Modify: `src/cascade/apps/demo.py`. Refuse eye-in-hand in the pre-pass (~669-681); register the bindings after
  `SkillRuntime(...)` (~876). The binding uses the `kins` list from the arm loop (683-689).
- Modify: `tests/test_aspire_admission.py:91-124`. The hand-built rig registers `ArmBinding`s for "left" and
  "right".
- Test: `tests/test_arm_frame_context.py`

**Interfaces:**
- Produces (`skills/arm_frame.py`):
  - `ARM_SCOPED_SECTIONS: tuple[str, ...]`, `RIG_SCOPED_OVERRIDES: frozenset[str]` (values in Global
    Constraints)
  - `class ArmFrameError(SkillError)`
  - `@dataclass(frozen=True) class ArmBinding: kin: Any; cfg: Any; grip: tuple[float, float, float] | None`.
    `grip` is (open_pos, closed_pos, max_width_m); None means "use the runtime's own".
  - `@dataclass(frozen=True) class ArmFrame: name: str | None; T_world_base: np.ndarray | None; binding:
    ArmBinding`
  - `world_from_base(arm) -> np.ndarray | None` (raises ArmFrameError)
  - `is_level(T) -> bool`, `tilt_deg(T) -> tuple[float, float]` (roll, pitch)
  - `to_arm(T, p)`, `to_world(T, p)` for (3,) or (N,3); `dir_to_arm(T, v)`, `dir_to_world(T, v)`
  - `cam_to_arm(T, T_table_cam) -> np.ndarray`
  - `frame_to_arm(T, frame, extrinsics) -> Frame`
  - `fix_to_arm(T, fix)`, `fix_to_world(T, fix) -> ObjectFix`. These transform position, points and axes; extent,
    detection, label and t are kept.
  - `table_z_in_world(T, table_z: float) -> float`
  - `arm_view_cfg(top_cfg, acfg) -> Cfg`, `arm_binding(top_cfg, acfg, kin) -> ArmBinding`
  - `_rigid_apply(T, points) -> np.ndarray`: the ONLY place that multiplies by a pose.
- Produces (`SkillRuntime`):
  - properties with setters: `kin`, `cfg`, `_grip_open`, `_grip_closed`, `_max_width`; read-only property
    `T_world_base`
  - `register_arm_bindings(bindings: dict[str, ArmBinding]) -> None`
  - `_resolve_arm_frame(effective_arm, resolved_arm) -> ArmFrame`
  - `_to_arm(p)`, `_to_world(p)`, `_dir_to_world(v)`, `_cam_to_arm(T_table_cam)`
  - `_frame_in_arm(frame, extrinsics=None)`; None means `self.extrinsics`
  - `_fix_to_arm(fix)`, `_fix_to_world(fix)`, `_table_z_world() -> float`
  - the thread-local `_arm_override.frame`

- [ ] **Step 1: Write the failing tests** in `tests/test_arm_frame_context.py`. The file gets a local
  `make_runtime` / `_cfg` fixture copied from `tests/test_base_pose_world_target_guard.py:84-116`; Task 3 moves it
  into the oracle module.

```python
def test_helpers_return_their_input_without_a_base_pose():
    # to_arm(None, p) is p; to_world(None, p) is p; cam_to_arm(None, M) is M
    # frame_to_arm(None, frame, ext) is frame; fix_to_world(None, fix) is fix; fix_to_arm(None, fix) is fix
    # table_z_in_world(None, -0.03) == -0.03
def test_helpers_invert_each_other_on_a_yawed_pose():
    # T = pose_to_transform(YAWED): to_world(T, to_arm(T, P)) ~ P (atol 1e-12) for 50 random points
    # to_world(T, zeros(3)) ~ [0.03, -0.22, 0.0]
    # frame_to_arm(T, f(T_base_cam=C)).T_base_cam ~ inv(T) @ C
    # with T_base_cam None it equals inv(T) @ ext.cam_to_base()
    # fix_to_world(T, fix).axes ~ R @ fix.axes; .extent is fix.extent
    # table_z_in_world(pose_to_transform(RAISED), -0.03) == pytest.approx(0.0, abs=1e-12)
@pytest.mark.parametrize("case", ["no harness", "no pose", "identity", "Mock arm", "Mock harness",
                                  "shipped", "string", "3x3", "nan", "scaled R"])
def test_world_from_base_reads_only_a_real_rigid_pose(case):
    # the first 5 give None; "shipped" gives pose_to_transform(SHIPPED_RIGHT); the last 4 raise ArmFrameError
def test_world_from_base_never_materializes_a_lazy_arm():
    # LazyArm(factory that appends) alone and wrapped in SafeArm(lazy, harness(pose)); factory never called
def test_arm_selection_binds_that_arms_kinematics_profile_and_pose(make_runtime, monkeypatch):
    # pair so101_left + so101_right; a patched rt.skill_list_objects records what it sees.
    # arm="so101_right": rt.kin is right.harness.kin; rt.cfg.grasp.drop_zone == [0.20, -0.12];
    #                    rt.T_world_base ~ pose_to_transform(SHIPPED_RIGHT); rt._max_width == 0.055
    # no arm:            rt.kin is the build's primary kin; rt.cfg.grasp.drop_zone == [0.20, 0.12]
    # after both calls:  rt.kin is the primary's; rt.T_world_base ~ left's pose (unbound read = primary)
def test_a_mixed_model_rig_solves_with_the_selected_arms_model(make_runtime, monkeypatch):
    # arms ["so101_mock", "mock"]: inside execute(..., arm="mock")
    #   rt.cfg.arm["name"] == "mock"; rt.kin is not the primary's kin;
    #   rt._max_width == the mock profile's gripper.max_width_m (so101_mock: 0.055)
def test_the_frame_is_thread_local_and_restored_after_a_raise(make_runtime, monkeypatch):
    # inside a bound call, a threading.Thread reads rt.kin -> the primary's;
    # a skill raising SkillError leaves rt._arm_override.frame unset afterwards
def test_the_held_binding_is_stable_per_arm(make_runtime, monkeypatch):
    # held_observation.binding(rt) taken in two calls on so101_right: element-wise `is`;
    # left vs right: some element differs
def test_a_selected_arm_without_a_binding_is_refused_before_any_work():
    # hand-built ArmRig (the test_aspire_admission pattern) with no register_arm_bindings
    # -> ok False, "has no kinematics/profile bound" in error, FAILED line once, skill fn never called
def test_an_unreadable_base_pose_refuses_every_skill_on_that_arm(make_runtime):
    # right harness.__dict__["base_pose"] = np.full((4, 4), np.nan); move_home(arm=so101_right) -> ok False,
    # "base_pose is not a rigid transform", 0 arm commands on both arms
def test_an_eye_in_hand_camera_on_a_base_pose_rig_refuses_to_build(tmp_path, monkeypatch):
    # pair + mock_small with every camera dict's extrinsics.mode = "eye_in_hand"; recording demo.make_arm
    # -> ValueError naming the camera and "base_pose"; 0 make, 0 connect
def test_arm_scoped_sections_cover_every_shipped_override_key():
    # yaml walk of configs/arms/*.yaml: overrides keys <= ARM_SCOPED_SECTIONS + RIG_SCOPED_OVERRIDES
def test_no_skill_or_grasp_module_starts_a_thread():            # Review Focus 1
    # AST: no `threading.Thread(` / `.submit(` under src/cascade/skills and src/cascade/grasping,
    # except skills/mobile_runtime.py
```

- [ ] **Step 2: Run the tests and see them fail.**
  `... -m pytest tests/test_arm_frame_context.py` -> FAIL (`ModuleNotFoundError: cascade.skills.arm_frame`).
- [ ] **Step 3: Implement `skills/arm_frame.py`.**
  - `world_from_base` follows `_mounted_off_table_origin`, reading instance dicts only.
  - Rigid means: finite, shape (4,4), last row [0,0,0,1], `allclose(R.T @ R, I, atol=1e-9)`, and det R > 0.
  - `arm_view_cfg` = `Cfg({**top._data, **{k: resolved[k] for k in ARM_SCOPED_SECTIONS if k in resolved}})`.
- [ ] **Step 4: Change the runtime.**
  - Add class defaults `_kin = _cfg = None` and `_arm_bindings = None`.
  - Add the properties. Setters write the primary; the gripper setters write instance values.
  - `__init__` assigns through them, unchanged in meaning.
  - In `_execute_skill`, right after `resolved_arm` (1349-1352): `frame = self._resolve_arm_frame(...)`, or return
    the refusal on `ArmFrameError`.
  - Bind `self._arm_override.frame = frame` next to `.arm` (1511-1512), and restore it in the same `finally`
    (1560).
  - The primary's binding is `ArmBinding(self._kin, self._cfg, None)`, made on the fly, so `__new__` runtimes
    work.
- [ ] **Step 5: Change `demo._build_runtime`.**
  - Add the eye-in-hand refusal to the pre-pass (before `_build_arm`).
  - Call `runtime.register_arm_bindings({name: arm_binding(cfg, acfg, k) for name, acfg, k in arms[1:]})`.
- [ ] **Step 6: Run the tests** -> PASS. Also run `tests/test_aspire_admission.py tests/test_arm_rig.py
  tests/test_base_pose_world_target_guard.py tests/test_arm_build_validates_first.py` -> PASS. The guard is
  unchanged in this task.
- [ ] **Step 7: Mutations. Each must be KILLED by the named test.**
  - bind the primary's kin under `arm=` -> `binds_that_arms_kinematics`
  - store the frame on a plain attribute instead of the thread-local -> `thread_local`
  - rebuild `arm_view_cfg` in every call -> `held_binding_is_stable`
  - drop the missing-binding refusal -> `without_a_binding`
  - make `world_from_base` use `getattr(arm, "harness")` -> `never_materializes`
  - let `world_from_base` map the identity to a matrix -> `reads_only_a_real_rigid_pose[identity]`
- [ ] **Step 8: Commit.**
  `git add src/cascade/skills/arm_frame.py src/cascade/skills/runtime.py src/cascade/apps/demo.py
  tests/test_arm_frame_context.py tests/test_aspire_admission.py`;
  `git commit -m "feat(skills): per-call arm frame: execute(arm=) binds the arm's kinematics, profile and base_pose"`

### Task 2: WorkspaceFilter excludes every arm's base (D6)

**Files:**
- Modify: `src/cascade/perception/workspace.py` (`WorkspaceFilter` fields, `reject` 169-183, `from_config` 185)
- Modify: `src/cascade/apps/demo.py`. Add `_workspace_bases(arm_cfgs) -> tuple[WorkspaceBase, ...] | None`; the
  watcher's `workspace=` (~898) and `runtime._workspace` both use it.
- Test: `tests/test_workspace_filter_bases.py`

**Interfaces:**
- Consumes: `arm_frame.world_from_base` is not used here. The bases come from config through
  `demo._base_transform`.
- Produces:
  - `@dataclass(frozen=True) class WorkspaceBase: x: float; y: float; z: float; radius_m: float; height_m: float;
    reach_m: float`
  - `WorkspaceFilter.from_config(cfg, bases=None)`; the filter has a `bases` attribute.
  - When no arm declares a base_pose, `_workspace_bases` returns None. An arm without a base_pose on a mixed rig
    sits at the origin.

- [ ] **Step 1: Write the failing tests.**

```python
def test_no_bases_keeps_every_decision():
    # 20_000 seeded (center, extents, mask_frac): from_config(c).reject == _main_reject(...)
    # _main_reject is main's rule copied verbatim into the test
def test_each_arm_base_is_excluded_and_the_phantom_origin_is_not():
    # pair bases: (0, +0.22, 0.05) -> "robot"; (0, -0.22, 0.05) -> "robot"; (0, 0, 0.05) -> None (main: "robot")
def test_out_of_reach_only_beyond_every_arm():
    # (0.0, 0.62, 0.05) -> None (0.40 from the left base); (0.0, 0.75, 0.05) -> "out_of_reach"
def test_the_watcher_and_the_runtime_share_the_bases(make_runtime):
    # pair: rt._workspace.bases == rt.watcher's filter .bases, two entries at y = +-0.22
    # so101_mock: rt._workspace.bases is None
```

- [ ] **Step 2: Run the tests** -> FAIL (`TypeError: from_config() got an unexpected keyword argument 'bases'`).
- [ ] **Step 3: Implement.**
  - With `bases is None`, `reject` keeps today's two expressions, unchanged.
  - Otherwise: "robot" if any base has `hypot(c - base_xy) < radius_m` and `c[2] < base.z + height_m`;
    "out_of_reach" only if `hypot(c - base_xy) > reach_m` for every base.
  - Each base's radius, height and reach come from that arm's resolved `workspace_filter`.
- [ ] **Step 4: Run the tests** -> PASS. Also run `tests/test_workspace_filter*.py tests/test_world*.py`.
- [ ] **Step 5: Mutations.**
  - "robot" checked against the first base only -> `each_arm_base`
  - reach counted "any" where it should be "every" -> `out_of_reach_only`
  - bases built on a rig without a base_pose -> `share_the_bases`
- [ ] **Step 6: Commit.**
  `git commit -m "feat(perception): the workspace filter excludes each arm's base at its base_pose"`

### Task 3: Perception into the arm frame, reports back out; the equivariance oracle

**Files:**
- Create: `tests/base_pose_oracle.py`, `tests/test_base_pose_runtime_frame.py`
- Modify `src/cascade/skills/runtime.py`:
  - `_resolve_query` 2393: `near` goes through `_to_arm`;
  - `_localize` 2768-3131: every analysed frame goes through `_frame_in_arm` (primary_frame0/1, `observe()`,
    side_frame0/1). `ext_of` returns arm-frame `Extrinsics(T=...)`. The belief fallback goes through
    `_fix_to_arm(_fix_from_belief(...))`. `in_workspace` sees B.
  - `_vlm_ground_fix` 3133-3247: each view's T goes through `_cam_to_arm`;
  - `_configured_zones` 3452-3474 (report-only) returns `position_xy_m` in W;
  - `skill_localize_object` 3486-3514: belief update and result via `_fix_to_world`;
  - `_describe_observation` `tcp_xyz` (3418) goes through `_to_world`;
  - `skill_point_at` 5870-5907 and `skill_preview_grasp` 3516-3570: reported positions in W;
  - remove `"point_at"` and `"preview_grasp"` from `_WORLD_TARGET_SKILLS`.
- Modify: `src/cascade/grasping/evidence.py` `localized` (~230-245). Record `T_world_base`, None or a 4x4 list.
- Modify: `tests/test_base_pose_world_target_guard.py`. Drop the two skills from `WORLD_TARGET_CALLS`; the
  derivation test then asserts `_WORLD_TARGET_SKILLS <= derived`.

**Interfaces:**
- Consumes: Task 1's runtime helpers.
- Produces (`tests/base_pose_oracle.py`, used by Tasks 4-6 and 8):
  - `SHIPPED_RIGHT`, `YAWED`, `RAISED`, `TILTED`
  - pytest fixtures, imported by name into each test module:
    - `make_runtime(tmp_path, monkeypatch)` -> `make(cfg, *, lazy=False) -> rt`; shuts every runtime down;
      sets `CASCADE_GRASP_BACKEND=obb`; sets `object_stop_frac = 0.5` on every raw MockArm;
    - `make_pair_rigs(tmp_path, monkeypatch)` -> `make(pose, *, arm="so101_right") -> (rt_a, rt_b, T)`.
      Rig A is `moved_cfg(arm, None)`; rig B is `moved_cfg(arm, pose)` with its cameras moved.
  - `ORACLE_SKILLS: frozenset[str]`: every skill the oracle tests run. Each task adds its own; Task 6's
    derivation test reads the set.
  - `moved_cfg(arm, pose, *, camera="mock_small", shift_xy=(0.0, 0.0), move_cameras=True) -> Cfg`. It sets or
    removes `base_pose` on every arm entry, and sets every camera dict's `extrinsics.T` to
    `T @ T_profile` (+ shift). The camera dicts are: `camera`, `cameras[*]`, each arm's
    `resolved.camera(s)`, `mj_prop_from_camera`, `mj_cameras[*]`.
  - `map_args(skill, args, T) -> dict`: place_at (x, y, z) -> T·p; other skills unchanged.
  - `POINT_KEYS`, `XY_KEYS`, `DIRECTION_KEYS`, `FRAME_FREE_KEYS: frozenset[str]`
  - `run_pair(rt_a, rt_b, T, skill, args) -> dict` with keys cmds_a/b, jaws_a/b, result_a/b, beliefs_a/b
  - `assert_equivariant(run, T) -> None`. An unclassified numeric 2- or 3-vector key fails with
    `"classify result key <path>"`.
  - `lifted_tcps(rt, safe, n0, T) -> np.ndarray`: FK of each command after n0, lifted.

- [ ] **Step 1: Write the failing tests.**

```python
ORACLE_T3 = {"localize_object": {"label": "red cube"}, "point_at": {"label": "red cube"},
             "preview_grasp": {"label": "red cube"}, "get_observation": {}, "list_objects": {},
             "describe_scene": {}, "count_objects": {}}
@pytest.mark.parametrize("pose", [SHIPPED_RIGHT, YAWED], ids=["so101_right", "yawed"])
@pytest.mark.parametrize("skill", sorted(ORACLE_T3))
def test_perception_skills_are_equivariant(make_pair_rigs, pose, skill): ...
def test_localize_reports_the_world_position_on_so101_right(make_runtime):
    # moved_cfg("so101_right", SHIPPED_RIGHT, move_cameras=False, shift_xy=(0, -0.10)):
    # result position[:2] within 5 mm of (0.198, -0.10); the "red cube" belief too
def test_point_at_hovers_over_the_world_object(make_runtime):
    # same rig: lifted TCP of the point pose, xy within 5 mm of (0.198, -0.10)   (main: refused)
def test_the_belief_fallback_aims_in_the_arm_frame(make_pair_rigs, monkeypatch):
    # detector.detect -> [] during the call, fresh belief: point_at is equivariant (YAWED)
def test_the_near_hint_is_in_the_arm_frame(make_runtime, monkeypatch):
    # spy _localize's localize_object(near_xyz=...): equals inv(T) @ belief.position (1e-9)
def test_configured_zones_are_reported_in_the_table_frame(make_runtime):
    # pair: localize_object("drop zone", arm=so101_right) -> position_xy_m (0.20, -0.34);
    #       arm=so101_left -> (0.20, 0.34); so101_mock alone -> (0.20, -0.12)
def test_grasp_evidence_records_the_arm_pose(make_runtime):
    # the tests/test_grasp_evidence.py capture pattern: the `localized` event of a preview_grasp carries
    # T_world_base == pose_to_transform(YAWED).tolist() on YAWED, None on so101_mock
```

- [ ] **Step 2: Run the tests** -> FAIL.
  - point_at and preview_grasp rows: B refused by the guard, so the commands differ.
  - The zones test fails: so101_right reports (0.20, 0.12), which is the left zone, unconverted.
  - The localize_object, list_objects and describe_scene oracle rows already PASS on main, because they report the
    table frame today. They must stay green.
- [ ] **Step 3: Implement** the listed sites. No other site changes in this task.
- [ ] **Step 4: Run the tests** -> PASS. Also run `tests/test_base_pose_world_target_guard.py`, the perception
  two-phase tests (`grep -l _localize tests/*.py`) and `tests/test_agent_results_compose.py` -> PASS.
- [ ] **Step 5: Mutations.**
  - `_frame_in_arm` returns the frame unchanged -> `equivariant[yawed-point_at]`
  - localize_object reports the B position -> `localize_reports_the_world_position`
  - belief fallback left unconverted -> `belief_fallback`
  - `near` left unconverted -> `near_hint`
  - zones left unconverted -> `configured_zones`
- [ ] **Step 6: Commit.**
  `git commit -m "feat(skills): localization aims in the running arm's frame and reports in the table frame"`

### Task 4: The grasp path, its columns, the observed gate, the held offset, the grasp `table_z` sites, PR-A M6

**Files:**
- Modify `src/cascade/skills/runtime.py`:
  - `skill_grasp_object` 3571: verify the planners receive the arm-frame frame from `_localize`. `_hug_candidates`
    2587 and `_camera_frame_candidates` 2643 read `frame.T_base_cam`, so no change is expected there.
  - `_promote_held` 4552: `mark_removed(near=self._to_world(fix.position))`
  - `skill_grasp_at_pixel` 7616-7677: `_frame_in_arm`; `self._workspace.reject(self._to_world(center), ...)`
  - PR-A's `_pregrasp_min_z` and `_air_grasp_exit` (M6):
    - query `column_top` with `harness._to_map_frame` (centre and z_min);
    - bring the top back with `harness._to_base_frame`;
    - leave the exit column and ceiling as they are; the tests pin their frame.
  - remove `"grasp_object"` and `"grasp_at_pixel"` from `_WORLD_TARGET_SKILLS`.
- Modify: `src/cascade/grasping/observed_scene.py` `for_runtime` 789-806. The static-T fallback goes through
  `runtime._cam_to_arm(ext.T)`. `tracked_objects` receives beliefs whose positions went through
  `runtime._to_arm`; use a list of `(label, to_arm(position), extent)` shims, so `tracked_objects` is unchanged.
- Modify: `src/cascade/skills/held_observation.py` `measure`:
  - the legacy channel (198-201): `runtime._to_arm(p)`;
  - the image path (212): `frame = runtime._frame_in_arm(runtime.observe())`.
- Test: `tests/test_base_pose_runtime_frame.py` (grasp section); `tests/test_base_pose_table_z_sites.py` (new).

**Interfaces:**
- Consumes: Task 3's oracle and `_frame_in_arm`.
- Produces: nothing new. Task 5 relies on `_held_offset` being in B and on `_held_observation_binding` per arm.

- [ ] **Step 1: Write the failing tests.** Every grasp test sets `object_stop_frac = 0.5` on the raw MockArm.

```python
@pytest.mark.parametrize("pose", [SHIPPED_RIGHT, YAWED], ids=["so101_right", "yawed"])
@pytest.mark.parametrize("skill,args", [("grasp_object", {"label": "red cube"}),
                                        ("grasp_at_pixel", {"u": 320, "v": 327, "normalized": False})])
                                        # (320, 327) = centre of mock_small's box_px [301, 308, 339, 346]
def test_grasps_are_equivariant(make_pair_rigs, pose, skill, args): ...
@pytest.mark.parametrize("rig", ["so101_right", "pair, arm=so101_right"])
def test_the_grasp_descends_on_the_world_object(make_runtime, rig):
    # camera fixed in W, shift_xy=(0, -0.10): lowest lifted TCP within 5 mm of T @ (rig A's lowest TCP)
    # and xy within 0.025 m of (0.198, -0.10)      (main: refused; before #352: 0.22 m off)
def test_the_descent_and_target_columns_sit_on_the_target(make_runtime, monkeypatch):
    # grasp.approach_target_exemption = True on so101_right; spy harness.allow_grasp_descent(exempt_xy=...)
    # and target_column_exemption(cylinder): both centres, lifted, within 5 mm of the lifted grasp point
    # and within 0.02 m of (0.198, -0.10)
def test_the_observed_gate_reads_the_scene_in_the_arm_frame(make_runtime, monkeypatch):
    # gate on (the _GateDouble/_gate_on pattern of tests/test_agent_results_compose.py), YAWED:
    # observed_scene.for_runtime receives a frame whose T_base_cam ~ inv(T) @ T_table and a fix in B
def test_for_runtime_converts_tracked_beliefs(monkeypatch):
    # unit: stub runtime (T = YAWED), ObservedScene patched to record `tracked`;
    # tracked positions ~ inv(T) @ belief positions
def test_the_held_offset_is_measured_in_the_arm_frame(make_pair_rigs, monkeypatch):
    # after grasp_object: rt_b._held_offset ~ rt_a._held_offset (1e-3);
    # legacy channel rt._object_pose = lambda _: world_point -> offset == to_arm(world_point) - tcp_B
def test_grasp_at_pixel_filters_in_the_table_frame(make_runtime):
    # a pixel whose world point is within 0.08 m of the right base -> refused with the "robot" reason
def test_pregrasp_map_top_reads_the_map_in_the_map_frame():        # M6
    # harness double, base_pose RAISED and YAWED: occupancy.column_top records its args and returns
    # (world_top, 0.01); recorded centre ~ lifted fix xy, z_min ~ lifted; result ladder hover
    # == (world_top - 0.03) + caps
def test_the_air_grasp_exit_column_and_ceiling_share_the_arm_frame():   # M6
    # recording harness: the target_column_exemption cylinder centre, lifted, is on the world object (5 mm);
    # z_max == fk(hover)[2,3] + 0.005; harness unit: a voxel 1 cm above the lifted ceiling is not exempt,
    # one 1 cm below is (base z 0.03)
# tests/test_base_pose_table_z_sites.py
#   RAISED, table_z -0.03, workspace min z -0.05, camera fixed in W, shift (0, -0.10); world table at z = 0
def test_object_height_is_measured_from_the_arms_table():
    # envelope feature object_height_m == 0.05 +- 0.003 (mock_small box_height_m 0.05)   (main: 0.08)
def test_grasp_support_height_is_the_world_support_in_the_arm_frame():
    # spy _grasp_support_height: its return == (min z of the fix's WORLD points) - 0.03 (+-1e-6)
    # (main: 0.03 higher)
def test_a_low_object_is_grasped_at_the_floor_above_the_world_table():
    # camera box_height_m 0.01: lowest lifted TCP z == grasp.min_grasp_z_m (default 0.008) +- 0.002 above z = 0
    # (main: about 3 cm higher)
def test_the_clearance_box_sits_on_the_world_table():
    # spy gripper_clearance.target_box(fix, support_z) on the grasp path (approach_target_exemption on):
    # box bottom, lifted, == 0.0 +- 1e-3. If the profile never calls it, unit-call it with the RAISED fix and
    # `_grasp_support_height`.
def test_vlm_bbox_shave_drops_the_table_plane():
    # _vlm_ground_fix on RAISED (grounder double returning the object bbox): every returned fix point
    # z > -0.03 + 0.005, and the fix z-extent <= 0.05 + 0.003
```

- [ ] **Step 2: Run the tests** -> FAIL. The oracle and world-point rows are refused by the guard. On main the
  `table_z` site values are the stated main numbers. M6: `column_top` receives the B centre.
- [ ] **Step 3: Implement** the listed sites.
- [ ] **Step 4: Run the tests** -> PASS. Also run `tests/test_agent_results_compose.py
  tests/test_agent_honest_results.py tests/test_held_observation.py tests/test_held_object_offset.py`, the
  observed-gate tests (`grep -l observed_scene tests/*.py`) and PR-A's pregrasp and exit tests
  (`grep -l "_air_grasp_exit\|pregrasp_map_top" tests/*.py`), each with the gate flag OFF and ON
  (`CASCADE_OBSERVED_FINGER_GATE=0/1`).
- [ ] **Step 5: Mutations.**
  - grasp_at_pixel frame left unconverted -> `equivariant[yawed-grasp_at_pixel]`
  - `reject` on the B centre -> `filters_in_the_table_frame`
  - tracked beliefs left unconverted -> `converts_tracked_beliefs`
  - legacy pose channel left unconverted -> `held_offset`
  - `column_top` queried unlifted -> `pregrasp_map_top`
  - returned top not converted back -> `pregrasp_map_top`
  - `mark_removed(near=` the B position) -> oracle belief rows
- [ ] **Step 6: Commit.**
  `git commit -m "feat(skills): grasps aim in the running arm's frame; columns, observed gate and held offset follow (incl. PR-A M6)"`

### Task 5: Place, push and pick_and_place; per-arm drop zones; the MuJoCo `table_z` sites

**Files:**
- Modify `src/cascade/skills/runtime.py`:
  - `skill_place_at` 4916-5253:
    - `_model_plan is not None and self.T_world_base is not None` -> SkillError("region placement is not
      supported on a base_pose arm");
    - (x, y[, z]) go through `_to_arm` first. With z None, convert (x, y, 0.0) for xy only; the release z stays
      the B default.
    - the belief update goes through `_to_world`; result fields are in W.
  - `_destination_fix` 5254-5314: belief writes go through `_fix_to_world`.
  - `skill_place_on_object` 5320-5375 and `skill_push_object` 5376-5424: `beliefs.update(label, end)` takes
    `_to_world(end)`; results are in W.
  - `skill_pick_and_place` 5425-5833: reports are in W. The default destination is the bound `cfg.grasp.drop_zone`.
  - holding arm: `_promote_held` (4538, beside `_held_observation_binding`) records `self._held_arm =
    <bound frame name>`. Every path that clears `held_object` sets it back to None. New
    `_check_holding_arm()` raises SkillError(f"{held!r} is held by arm {held_arm!r}, not {arm!r}; no motion was
    sent") when `_held_arm` is set and differs from the bound frame's name. place_at and place_on_object call it
    first.
  - remove place_at, place_on_object, push_object and pick_and_place from `_WORLD_TARGET_SKILLS`.
- Modify: `src/cascade/skills/mujoco_withdrawal.py` 243 and 483. Compare against
  `table_z_in_world(runtime.T_world_base, table_z)`.
- Verify only (expect no change; say so in the report): `skills/place_geometry.py`,
  `skills/release_episode.py:517`, `skills/contact_episode.py:167`.
- Test: `tests/test_base_pose_runtime_frame.py` (place section), `tests/test_base_pose_table_z_sites.py`.

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: nothing new.

- [ ] **Step 1: Write the failing tests.**

```python
@pytest.mark.parametrize("pose", [SHIPPED_RIGHT, YAWED], ids=["so101_right", "yawed"])
@pytest.mark.parametrize("seq", ["grasp+place_at", "grasp+place_on_object", "push_object", "pick_and_place"])
def test_place_and_push_are_equivariant(make_pair_rigs, pose, seq): ...
    # place_at args: A (0.20, -0.08, 0.06) -> B map_args; push_object {"label": "red cube",
    #   "direction": "left", "distance_m": 0.05}
@pytest.mark.parametrize("rig", ["so101_right", "pair, arm=so101_right"])
def test_place_at_releases_at_the_world_point(make_runtime, rig):
    # grasp, then place_at(x=0.22, y=-0.30): lifted release TCP xy within 5 mm of T @ (rig A's release TCP)
    # and within 0.025 of (0.22, -0.30); belief "red cube" xy within 5 mm of (0.22, -0.30)
def test_push_passes_through_the_world_object(make_runtime):
    # lifted push segment: perpendicular xy distance to (0.198, -0.10) <= 0.005
def test_pick_and_place_uses_the_running_arms_drop_zone(make_runtime):
    # pair, arm=so101_right: release xy within 0.025 of W (0.20, -0.34)   (main: refused; before: left zone)
def test_a_place_on_the_arm_that_does_not_hold_is_refused(make_runtime):     # Review Focus 2
    # pair: grasp(arm=so101_right) ok; place_at(..., arm=so101_left) -> ok False,
    # "is held by arm 'so101_right'" in error, 0 jaw and 0 arm commands on either arm during the place
def test_a_region_plan_on_a_base_pose_arm_is_refused(): ...
# tests/test_base_pose_table_z_sites.py  (RAISED)
def test_place_at_z_is_a_table_frame_height(make_runtime):
    # place_at(0.22, -0.30, z=0.06): lifted release TCP z == 0.06 +- 0.002
    # (main: 0.06 read as B, clamped to the 0.045 B ceiling = world 0.075)
    # z omitted: lifted z == 0.0 + release_height_m (0.05) +- 0.002
def test_push_contact_height_is_measured_from_the_arms_table(make_runtime):
    # lifted push TCP z == min(max(0.015, 0.35 * 0.05), 0.06) +- 0.003 above the world table (z = 0)
def test_mujoco_withdrawal_compares_the_world_table(monkeypatch):
    # unit on the two sites with base z 0.03, table_z -0.03: a MuJoCo plane at world z 0.0 passes 243;
    # the release envelope floor at 483 is 0.0
```

- [ ] **Step 2: Run the tests** -> FAIL. The guard refuses. On main, z = 0.06 is released at a world z of 0.075.
  The withdrawal plane check refuses the world table.
- [ ] **Step 3: Implement** the listed sites.
- [ ] **Step 4: Run the tests** -> PASS. Also run `tests/test_place*.py tests/test_mujoco_withdrawal.py
  tests/test_mujoco_region*.py tests/test_release_episode.py tests/test_contact_episode*.py` (MuJoCo files skip
  in the rig venv; Task 8 runs them in the scratch venv).
- [ ] **Step 5: Mutations.**
  - place_at args left unconverted -> `equivariant[yawed-grasp+place_at]`
  - place_at converts z when it was omitted -> `z_is_a_table_frame_height`
  - push belief `end` left unconverted -> oracle belief rows
  - withdrawal compares the B `table_z` -> `withdrawal_compares_the_world_table`
  - pick_and_place reads the primary's drop zone -> `running_arms_drop_zone`
  - `_check_holding_arm()` dropped from place_at -> `place_on_the_arm_that_does_not_hold`
- [ ] **Step 6: Commit.**
  `git commit -m "feat(skills): place, push and pick_and_place aim in the running arm's frame"`

### Task 6: The remaining world-target skills, the viewer frame, and the guard narrowed to tilt

**Files:**
- Modify `src/cascade/skills/runtime.py`:
  - beliefs used as targets go through `_to_arm`, and results go to W, in: `skill_handover` 5925,
    `skill_throw` 5956, `skill_turn_screw` 6042, `skill_sort_by_color` 6204 (incl. `_reachable_place_xy` 6314,
    `_move_object_to` 6393), `_restore_plan` 6471 / `skill_restore_scene` 6566, `skill_search_for_object` 6704;
  - `skill_throw` and `skill_handover` call Task 5's `_check_holding_arm()` first;
  - the guard 362-410 + 1353-1357: replace `_mounted_off_table_origin` with
    `frame.T_world_base is not None and not is_level(frame.T_world_base)`. Keep `_WORLD_TARGET_SKILLS` (the
    15) as the tilt scope.
  - `_world_target_refusal(name, arm_name, roll_deg, pitch_deg)`:
    - error: `f"{name} refused on arm {arm!r}: its base_pose is tilted (roll {roll:.2f} deg, pitch
      {pitch:.2f} deg); table-frame targets need a level mount; no motion was sent"`;
    - next_action: `"Do not retry this skill on this arm. Level its base_pose in the profile, or use move_home,
      move_to_rest, rotate_joint, open_gripper or close_gripper on it."` (Minor 2).
- Modify: `src/cascade/skills/spatial_moves.py` `viewer_axes` 242 (`rt._cam_to_arm(ext.cam_to_base())`),
  `_target_geometry` 869 (verify it is in B), `move_near_object` 929-946 (the reported `center_xy_m` in W).
- Delete: `tests/test_base_pose_world_target_guard.py`. Create: `tests/test_base_pose_tilted_mount.py`. Port
  these pins from the deleted file to TILTED:
  - refusal before any work: Spy, 0 commands, 0 perception;
  - the MCP one-FAILED-line relay;
  - LazyArm never materialized;
  - the AST early-return pin;
  - joint/relative skills still run;
  - running arm, not primary.
- Test: `tests/test_base_pose_runtime_frame.py` (remaining section).

**Interfaces:**
- Consumes: Tasks 1-5. Produces: `_derive_world_target_skills()` (moves into the new test file), wider sources.

- [ ] **Step 1: Write the failing tests.**

```python
@pytest.mark.parametrize("pose", [SHIPPED_RIGHT, YAWED], ids=["so101_right", "yawed"])
@pytest.mark.parametrize("skill,args", [("sort_by_color", {}), ("search_for_object", {"label": "red cube"}),
    ("restore_scene", {"name": "scene"}),            # after snapshot_scene + a push in the same rigs
    ("turn_screw", {"label": "red cube"}), ("throw", {"label": "red cube"}),
    ("handover", {"label": "red cube"}), ("move_near_object", {"label": "red cube", "relation": "above"})])
def test_remaining_skills_are_equivariant(make_pair_rigs, pose, skill, args): ...
def test_viewer_directions_follow_the_camera_on_a_yawed_base(make_pair_rigs):
    # arm profile "mock" (move_relative completes at its home pose), YAWED, cameras moved with the base:
    # move_relative(direction="right", frame="viewer", distance_m=0.02) commands equal A's
    # (main: rotated by 0.4 rad); same for rotate_gripper(direction="left", angle_deg=10, frame="viewer")
@pytest.mark.parametrize("skill", sorted(WORLD_TARGET_CALLS))      # the 15
def test_a_tilted_mount_refuses_world_target_skills_before_any_work(make_runtime, monkeypatch, skill):
    # TILTED: ok False; "its base_pose is tilted (roll 5.73 deg, pitch 0.00 deg)" in error; FAILED line once;
    # Spy 0 commands / 0 jaw / 0 perception; next_action has no "move_relative" and no "rotate_gripper"
def test_a_level_yaw_is_never_refused():   # YAWED: none of the 15 returns the tilt error
def test_throw_and_handover_on_the_arm_that_does_not_hold_are_refused(make_runtime):   # Review Focus 2
    # pair: grasp(arm=so101_right); throw(arm=so101_left) and handover(arm=so101_left) -> ok False,
    # "is held by arm 'so101_right'", 0 commands on either arm; right still holds (rt.held_object unchanged)
def test_every_skill_reaching_a_world_source_is_covered():       # Minor 4
    # widened derivation: sources {_localize, _belief_pose, fix_at_pixel, beliefs.<any>, T_base_cam,
    # cam_to_base, extrinsics, viewer_axes} or x/y params; derived == ORACLE_SKILLS | set(FRAME_FREE)
    # FRAME_FREE: {skill: reason}, each reason non-empty
def test_the_derivation_sees_a_new_belief_or_camera_reader(monkeypatch):
    # synthetic skill_* reading self.beliefs.find(...) or frame.T_base_cam -> derived
```

- [ ] **Step 2: Run the tests** -> FAIL. The oracle rows are refused by the guard on YAWED. The viewer commands
  differ. The tilt text is missing.
- [ ] **Step 3: Implement** the listed sites and the narrowed guard.
- [ ] **Step 4: Run the tests** -> PASS. Also run `tests/test_spatial*.py tests/test_relative_motion*.py
  tests/test_sort*.py tests/test_restore*.py tests/test_mcp_server.py tests/test_robot_mcp.py`.
- [ ] **Step 5: Mutations.**
  - viewer axes left unconverted -> `viewer_directions`
  - tilt check reads roll only (pitch case: add `[0, -0.22, 0, 0, 0.1, 0]` to the tilt test ids) -> `tilted_mount`
  - tilt check fires on any non-identity pose -> `level_yaw_is_never_refused`
  - `_restore_plan` left unconverted -> `equivariant[yawed-restore_scene]`
  - next_action lists move_relative -> `tilted_mount`
  - a skill dropped from ORACLE_SKILLS / FRAME_FREE -> `every_skill_reaching_a_world_source`
- [ ] **Step 6: Commit.**
  `git commit -m "feat(skills): every world-target skill converts; the base_pose guard now refuses only tilted mounts"`

### Task 7: Agent contract text, viewers, docs (D7; #352 Minors 1 and 3)

**Files:**
- Modify `src/cascade/skills/runtime.py`:
  - `TOOL_SPECS` `list_arms` (8633-8645): restore the text of `git show 42eaddd:src/cascade/skills/runtime.py`;
  - `tool_specs_for_config` (8683): when any arm profile has `base_pose`, apply the table-frame wording to the
    coordinate tools. Those are the tools whose description contains "base-frame", "robot base frame" or
    "robot's base frame": `localize_object`, `place_at`, `annotated_view`, `analyze_scene`, and the others
    found by that search. Append one sentence: "Coordinates are in the shared table frame: the same numbers
    for every arm on this rig."
  - `skill_annotated_view` 7483-7496: the result text says "table frame" on base_pose rigs.
  - add `_tcp_world_if_awake()`: the primary's TCP, lifted by the primary's T, for viewers.
- Modify: `src/cascade/perception/visual_interface.py`. `render(..., T_world_base=None)` lifts the TCP and the 8
  workspace-box corners. `apps/stream_server.py:750-763` uses `_tcp_world_if_awake` and passes the primary's T.
- Modify: README.md:193 (one sentence on the frame contract);
  `configs/arms/so101_left.yaml`, `so101_right.yaml`, `so101_left_mujoco.yaml`, `so101_right_mujoco.yaml` headers
  (base_pose, table-frame agent coordinates, `drop_zone` in this arm's base frame).
- Test: `tests/test_base_pose_agent_text.py`

- [ ] **Step 1: Write the failing tests.**

```python
@pytest.mark.parametrize("rig", ["rebot_rs", "mock", "so101_mock", "isaac_kitchen"])
def test_tool_specs_are_unchanged_on_rigs_without_a_base_pose(rig):
    # sha256(json.dumps(tool_specs_for_config(cfg), sort_keys=True)) == GOLDEN[rig]
    # GOLDEN = values computed once on BASE with the list_arms sentence stripped (Step 2 records them)
@pytest.mark.parametrize("rig", ["so101_right alone", "pair"])
def test_coordinate_tools_say_table_frame_on_a_base_pose_rig(rig):              # Minor 1 (single arm)
    # each COORDINATE_TOOLS description: "table frame" in it, none of "base-frame" / "robot base frame"
def test_list_arms_no_longer_mentions_base_pose(): ...
def test_the_viewer_draws_the_tcp_where_it_is():
    # VisualInterface with a known camera: the TCP pixel == project(T @ tcp_B)
def test_viewers_read_the_primary_frame_during_a_bound_call(make_runtime, monkeypatch):   # Review Focus 4
    # while execute(..., arm="so101_right") runs, another thread's rt._tcp_world_if_awake()
    # == left FK lifted by left's pose
```

- [ ] **Step 2: Record the GOLDEN values.** Use a scratch probe on a `git archive BASE` copy (cascade skill:
  read-only review recipe) that prints the four digests after stripping #352's sentence from `list_arms`. Paste
  them into the test. Run the tests -> FAIL: no table-frame text, `list_arms` still mentions base_pose, the
  viewer TCP is unlifted.
- [ ] **Step 3: Implement.** Update the README line and the four headers in the same commit.
- [ ] **Step 4: Run the tests** -> PASS. Also run `tests/test_llm_and_library.py tests/test_mcp_server.py
  tests/test_decision_replay.py tests/test_decision_probe.py tests/test_wrist_narration.py` (the dashboard golden
  stays the same) and `tests/test_openclaw_natural.py tests/test_attendee_english.py` (the kitchen catalog stays
  the same).
- [ ] **Step 5: Mutations.**
  - text changed on every rig -> `unchanged_on_rigs_without`
  - condition `len(arms) > 1` instead of "any base_pose" -> `[so101_right alone]`
  - viewer TCP unlifted -> `draws_the_tcp`
  - viewer uses the bound frame -> `primary_frame_during_a_bound_call`
- [ ] **Step 6: Commit.**
  `git commit -m "docs(agent): table-frame coordinates on base_pose rigs; viewers draw the lifted TCP"`

### Task 8: Invariance gate and acceptance

**Files:**
- Create: `tests/test_base_pose_no_transform.py`, `tests/test_base_pose_pair_pick_place.py`,
  `tests/test_base_pose_mujoco_pair_pick.py`
- Modify (ledger only, never committed): `.superpowers/sdd/base-pose-runtime-frame/diff_probe.py`. Keep its
  rigs: rebot_rs, mock, so101_mock, pair_identity_primary, so101_right_joint_only, pair_right_joint_only. Add:
  - `isaac_kitchen`: the kitchen arm profile on a MockArm with camera `mock`. If its `detector` override needs a
    server, give both trees the mock detector: the same input on both sides.
  - `pair_identity`: both poses identity, world-target calls on the primary (no `arm=`). Calls with
    `arm=<non-primary>` change by design (Q4 a: that arm's own profile), so they are not in the byte-identical
    set. `pair_right_joint_only` is expected to stay identical, because both so101 profiles carry the same
    gripper and relative_motion values. Any diff there must be explained by Q4 a in the ledger.
  - `so101_right_tilted_joint`: TILTED, joint-only calls.

- [ ] **Step 1: Write the no-transform test.**

```python
@pytest.mark.parametrize("rig", ["rebot_rs", "isaac_kitchen", "mock", "so101_mock", "pair_identity"])
def test_no_transform_runs_without_a_base_pose(rig, monkeypatch, tmp_path):
    # demo.make_arm -> MockArm (diff_probe's _mock_only); occupancy.required False for rebot_rs;
    # arm_frame._rigid_apply patched to raise AssertionError;
    # 3 seeds x 8 calls of diff_probe's random_call: no AssertionError,
    # and the bound frame's T_world_base is None in every call (spy on _resolve_arm_frame)
```
  It should PASS immediately. Prove that it discriminates: the mutant "identity -> matrix" in `world_from_base`
  must be KILLED by `[pair_identity]`.
- [ ] **Step 2: Run the byte-identical diff.** Do it on `git archive` copies of BASE (run twice: the noise floor)
  and HEAD:
  `PYTHONDONTWRITEBYTECODE=1 python diff_probe.py <tree> out_<x>.json`, then `compare.py`.
  - Pass: 0 rows whose cmd_sha, jaw, `held` or normalised result differ beyond the base-vs-base rows.
  - Scrub wall-clock text. pick_and_place `measured.final` is a known race: rerun a lone differing row.
  - Paste the counts into the ledger.
- [ ] **Step 3: Write the mock pair acceptance test** (`tests/test_base_pose_pair_pick_place.py`, runs in CI).
  - Setup: `moved_cfg` pair, camera fixed in W, `shift_xy=(0, -0.10)`.
  - `pick_and_place(object="red cube", arm="so101_right")`:
    - lifted release TCP within 5 mm of (0.20, -0.34) minus the held offset (rotated to W);
    - the belief within 5 mm of the placed point;
    - `so101_left` sends 0 commands.
  - `grasp_object` + `place_at(x=0.22, y=-0.30, arm="so101_right")`: the same checks against (0.22, -0.30).
  - RED on BASE: refused by #352. Copy the test into a BASE archive, run it, record the output.
- [ ] **Step 4: Write the MuJoCo pair acceptance test**
  (`tests/test_base_pose_mujoco_pair_pick.py`, `needs_rig` skip as in `tests/test_multi_arm_physics.py:46-49`).
  - Setup: `--arms so101_left_mujoco,so101_right_mujoco --camera mock_small` through `moved_cfg(...,
    move_cameras=False, shift_xy=(0, -0.10))`, so the planted prop is at W (0.198, -0.10). Use the same two
    sequences.
  - The truth channel (`rt.effects` object pose = MuJoCo body position, W) puts the prop within
    `DEST_TOLERANCE_M` (0.03) of each target.
  - The `postcondition` status is CONFIRMED, with channel physics.
  - `so101_left_mujoco` joints change by less than 1e-3 rad.
  - The scratch venv:
    - `uv venv --python /home/spark/Projects/cascade/.venv/bin/python <S>/venv`;
    - `<S>/venv/lib/python3.12/site-packages/zz_rig.pth` containing `import site;
      site.addsitedir('/home/spark/Projects/cascade/.venv/lib/python3.12/site-packages')` (addsitedir, so
      pinocchio's cmeel .pth loads);
    - `uv pip install --python <S>/venv/bin/python mujoco==3.12.0 websockets==17.2`;
    - the MJCF: `python scripts/fetch_robot_assets.py so101`, or copy the gitignored
      `assets/mjcf/so101/` from `/home/spark/Projects/cascade-wt/openshell-everywhere`.
  - Control first, on BASE: single-arm `so101_mujoco` + `mock_small` `pick_and_place(object="red cube")` is
    CONFIRMED. If it fails, record that and accept on the mock pair; do not tune physics.
  - Also run `tests/test_multi_arm_physics.py tests/test_mujoco_withdrawal*.py tests/test_mujoco_region*.py` in
    that venv, on BASE and HEAD. The failing IDs must be identical.
- [ ] **Step 5: Run the full-suite failing-ID diff.** Use 4 shards per tree on detached worktrees of BASE and
  HEAD, under `systemd-run --user --collect -p MemoryMax=48G` (cascade skill "Full-suite baseline diff"). NEW
  failing IDs: none. Also check ruff: `uvx ruff check` counts on the touched files, base vs head.
- [ ] **Step 6: Copy the evidence.** RESULTS (diff counts, acceptance numbers, MuJoCo log, suite diff) go into
  `.superpowers/sdd/base-pose-runtime-frame/RESULTS.md`.
- [ ] **Step 7: Commit.**
  `git add tests/test_base_pose_no_transform.py tests/test_base_pose_pair_pick_place.py
  tests/test_base_pose_mujoco_pair_pick.py`;
  `git commit -m "test(base_pose): no-transform pin and so101 pair pick-and-place at the world target"`

## Final checks before the PR (controller)

- **Pre-final round:**
  - `git merge --no-ff origin/main` if main moved, then carry any new frame-reading site through the Task 6
    derivation test;
  - fold the ledger's fix-now minors.
- **Then, in parallel:**
  - the final whole-branch review on `git diff origin/main...HEAD`, with spec §5 and the five Review Focus items
    as named risks;
  - the MuJoCo acceptance on the exact head.
- **PR body:**
  - why: report §1.2 numbers;
  - what changed: Tasks 1-8;
  - verification: oracle rows, diff counts, acceptance numbers, suite diff;
  - deploy notes: no weights or env changes; the live reBot is byte-identical;
  - known limitations: spec §9;
  - closes #352's Minors 1-4 and PR-A's M6.
