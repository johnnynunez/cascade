# Object-aware nvblox clearance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the reBot stranding inside the 3 cm map margin and make every clearance stop diagnosable (PR-A), then label map voxels from same-frame SAM masks so exactly the grasp target is exempt and fingers/jaws can later use a measured margin against known neighbours inside the grasp corridor (PR-B, shipped at 3 cm).

**Architecture:** PR-A changes admission logic only: a strict-retreat escape plus a route-end rule for TCP/link points, a map-top hover and a target-column exit after an air grasp, a refusal-path snapshot deque drained into the trace, and a config-driven tool-volume builder whose shipped values equal today's slab. PR-B adds a cascade-side voxel labeller on the watcher thread (bridge unchanged), a frozen per-grasp episode whose exact target voxel keys replace the column/cylinder for map checks, a dormant corridor rule (active only when `corridor_margin_m < min_clearance_m`), a report-only reconciliation, and the fit-error measurement tool.

**Tech Stack:** Python 3.12, numpy, OpenCV (`minAreaRect`), pinocchio `Kinematics` (tests marked `needs_pin`), pytest. scipy only for reconcile clustering (optional import). MuJoCo (`uv run --extra sim`) only for the offline sim term of the fit error.

**Spec:** `docs/superpowers/specs/2026-10-16-nvblox-object-aware-design.md` (owner decisions in §11). Evidence: `.superpowers/sdd/2026-10-16-nvblox-object-aware/investigation/` (local only).

**Branches (stacked):**
- PR-A: `feat/nvblox-object-aware` in `/home/spark/Projects/cascade-wt/nvblox-object-aware` (spec, plan, A1-A6).
- PR-B: `feat/nvblox-object-aware-b` in a new worktree `/home/spark/Projects/cascade-wt/nvblox-object-aware-b`, created from PR-A's head once PR-A is in review. Pick up PR-A fix rounds with `git merge origin/feat/nvblox-object-aware` (no force-push). Open PR-B only after PR-A merges.

## Global Constraints

- Never edit, run services in, or write to `/home/spark/Projects/cascade` (live rig). Its `.venv/bin/python` is the read-only interpreter. No cameras, CAN, bridge (:5557), brain, GPU or `systemctl` from any step.
- Test command (`$T` below), PR-A: `cd /home/spark/Projects/cascade-wt/nvblox-object-aware && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src /home/spark/Projects/cascade/.venv/bin/python -m pytest -q -p no:cacheprovider <tests>`. PR-B: the same with `cd /home/spark/Projects/cascade-wt/nvblox-object-aware-b`. Pass an explicit working directory on every shell call.
- Tests are offline: mock arm, `CartesianKin`/`case()` (`tests/test_home_routes.py`), `PointObstacles` stand-ins, `GridBridge`/`DepthBridge`/`make_map`/`render_depth`/`box_surface` from `tests/occupancy_grid_bridge.py`. `tests/conftest.py` pins `CASCADE_OCCUPANCY=0`.
- `min_clearance_m` stays 0.03 for the arm body (TCP, joint origins, links), wrist camera, housing, held payloads and unknown voxels, in every phase. The only reduced margin is PR-B's `corridor_margin_m` for fingers/jaws against known-neighbour prisms: floor 0.010 m, capped at `min_clearance_m`, shipped = `min_clearance_m` (Q1).
- Attached-payload checks (`attached=True`) never get an escape rule. Static vets (`vet_pose`, grasp candidates) never apply an escape.
- No change to the contact monitor (`safety/contact.py`, `rebot_rs.yaml:92-103`), contact zones, the bridge protocol (`scripts/serve_occupancy_bridge.py`) or backends (`perception/occupancy_backends.py`).
- Fail closed: a stale, doubtful or conflicting label is "unknown" (3 cm). A snapshot or report error never changes a gate outcome. In object-aware mode there is no silent fallback to the legacy column/cylinder.
- Safety map reads go through `_refresh_lock` methods (`clearance`, `clearance_excluding`, new `column_top`/`nearby_occupied`/`label_inputs`), never `view_snapshot()`.
- Watcher-thread work is vectorised (no per-voxel Python loop) and never blocks; trace bookkeeping reads `arm.__dict__.get("harness")`, never `getattr` on a backend.
- Pinned tests (spec §9) keep passing unchanged: `test_home_routes.py:99,109,113`; `test_rebot_occupancy_live.py:108,149,172,194`; the 0.03 pins at `test_clearance_lift_route.py:106,371,390`, `test_place_path_safety.py:83`, `test_kitchen_acceptance_configuration.py:29`; `test_wrist_narration.py::test_golden_flag_off_is_byte_identical`. Only the tests named in a task's Files may change.
- Safety volumes from measurements: every face moves outward to the next 5 mm; a missing caliper number keeps today's face; pinned by a test on the shipped config.
- New config keys default to today's behaviour on every profile unless the task says otherwise. Config comments are extended, never deleted (AGENTS.md:103).
- `AGENTS.md` is protected: never edited. Its text (spec §10) goes into the PR bodies (see each PR's exit).
- English everywhere (code, comments, docs, commit messages).
- Every safety guard gets a mutation check: `M=~/.hermes/skills/robotics/cascade-feature-worktree/scripts/mutate.sh; bash $M <file> '<exact old>' '<new>' <tests> [-k ...]` must print KILLED by the named test. Run mutations with no other reader of the tree; afterwards `find src tests -name __pycache__ -prune -exec rm -rf {} +`.
- Lint: `/home/spark/.local/bin/uvx ruff check <changed .py> --select F,E9` clean.

## Review Focus

1. **A stale label exempting a moved neighbour:** a neighbour that slides into a target voxel key after the freeze must count again at 3 cm. Pinned in Task B5 (`test_neighbour_moved_into_target_keys_is_not_exempt`, `test_new_voxels_during_the_episode_are_unknown`).
2. **The escape rule letting a move get closer:** a lateral slide over a flat top, a point entering from outside while another retreats, and a hold followed by a closing step must all refuse. Pinned in Task A1 (`test_lateral_1cm_over_a_flat_top_is_refused`, `test_a_point_entering_the_margin_refuses_although_another_retreats`, `test_hold_is_admitted_then_closing_in_is_refused`).
3. **The target exemption leaking to neighbours:** voxels of a neighbour touching the target (1 cm gap) must still block jaws, camera and TCP. Pinned in Task B4 (`test_target_exemption_never_leaks_to_a_touching_neighbour`).
4. **Map snapshot cost on the motion path:** admitted 50 Hz steps must never build a snapshot, and a failing snapshot must leave the refusal unchanged. Pinned in Task A4 (`test_admitted_stream_never_builds_a_snapshot`, `test_snapshot_failure_keeps_the_refusal`).
5. **The corridor margin active outside the corridor:** in transport, at the pregrasp pose itself, after the lift, for camera/housing/TCP, and for unexplained voxels, the margin stays 3 cm. Pinned in Task B6 (`test_corridor_margin_never_applies_outside_the_corridor`).

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `src/cascade/safety/harness.py` | Escape rule, route-end check, clearance reports, object-aware scope, corridor rule | A1, A4, B4, B6, B7 |
| `src/cascade/safety/trajectory.py` | `vet_route` end check; `lift_ik` shared by `plan_route` and the exit | A1, A3 |
| `src/cascade/skills/spatial_moves.py` | Spatial route end check; camera dims incl. `front_m`/`bottom_m` | A1, A5 |
| `src/cascade/skills/relative_motion.py` | `camera_box` with `front_m`/`bottom_m` (default 0 = today's slab) | A5 |
| `src/cascade/safety/tool_volume.py` | `housing`/`fingers` parts; caliper rounding helpers | A5 |
| `src/cascade/perception/occupancy.py` | `column_top`, `nearby_occupied`, refusal-snapshot config, label inputs, `exclude_keys`, reconcile in `report()` | A2, A4, B1, B4, B7 |
| `src/cascade/skills/runtime.py` | Map-top hover, air-grasp exit, trace drain, grasp episode and corridor wiring | A2, A3, A4, B5, B6 |
| `scripts/replay_clearance.py` | Offline exact replay of `context.clearance` records (spec §6.3) | A6 |
| `src/cascade/types.py`, `perception/brain_detector.py`, `perception/detector.py` | `Detection.box_t` | B1 |
| `src/cascade/perception/voxel_labels.py` (new) | Prism fit, per-frame label snapshot, F1-F4, key encoding, reconcile | B2, B7 |
| `src/cascade/safety/object_aware.py` (new) | `ObjectAwareConfig`, `GraspEpisode` (freeze, shrink, expiry), `CorridorState` | B3, B4, B5, B6 |
| `src/cascade/perception/world.py`, `src/cascade/apps/demo.py:725` | Watcher builds snapshots, shrinks the active episode, stores reconcile | B3, B5, B7 |
| `src/cascade/grasping/gripper_clearance.py` | Public finger-hull/box SAT distance | B6 |
| `src/cascade/perception/fit_error.py` (new), `scripts/measure_fit_error.py` (new) | Fit-error math and the operator-gated measurement tool | B8 |
| `configs/demo.yaml`, `configs/arms/rebot_rs.yaml` | New keys (defaults = today) | A2, A4, A5, B3, B6 |
| `docs/CLEARANCE_LIFT_ROUTES.md`, `docs/NVBLOX.md`, `docs/ARCHITECTURE.md` | Rule text next to the code that changes | A1, A4, B5, B6 |

---

## PR-A "unstick + measure" (no margin lowered)

### Task A1: Escape rule for TCP/link map clearance + route-end rule (spec §3.1)

**Files:**
- Modify: `src/cascade/safety/harness.py:872-877` (pass previous points), `:1005-1030` (`_occupancy_violation`); add `route_end_violation` after `vet_pose` (`:907-990`); constants next to `_unknown_blocks` (`:133`)
- Modify: `src/cascade/safety/trajectory.py:76-83` (`vet_route`)
- Modify: `src/cascade/skills/spatial_moves.py:445-453` (before `return Route(segments, cart)`)
- Modify: `docs/CLEARANCE_LIFT_ROUTES.md` (one paragraph: the escape rule, the end rule, what it does not do)
- Test: `tests/test_clearance_escape.py` (new); `tests/test_spatial_commands.py` (one test after `:175`)

**Interfaces:**
- Produces: `CLEARANCE_ESCAPE_TOL_M = 1e-6`, `ESCAPE_NOTE = " (inside the margin: only a strictly retreating step is allowed)"` in `harness.py`.
- Produces: `SafetyHarness._occupancy_violation(points, exempt, *, attached=False, prev_points=None, hold=False) -> str | None`. With `prev_points` (never with `attached`), ONE `clearance(np.vstack([points, prev_points]))` call. A step with points below the margin is admitted only if every such point was already below the margin at `q_prev` and either `d_next > d_prev + CLEARANCE_ESCAPE_TOL_M` or `hold`. NaN/unobserved points never escape. The refusal text is today's text, plus `ESCAPE_NOTE` when any TCP/link point was inside the margin at `q_prev`.
- Produces: `SafetyHarness.route_end_violation(q) -> str | None`. This is the static TCP + `links[1:]` map check under the active `_grasp_exempt`, with no escape. It returns None when `kin` or `occupancy` is None.
- Produces: `vet_route` checks `end = goals[-1] if goals else start` through `getattr(harness, "route_end_violation", None)`. Harnesses without it (GeometryHarness, SimpleNamespace doubles) have no escape and need no end check. Reason: `"route ends inside the margin: " + reason`. The spatial `plan_route` raises `Refused("safety_harness", <same text>, len(goals) - 1)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_clearance_escape.py  (CartesianKin + case() from test_home_routes; start TCP (.3, 0, .3))
VOXEL = np.array([.3, 0., .3 - .028])                    # 2.8 cm under the hovering TCP

def test_hover_at_28mm_admits_home_a_3cm_lift_and_up_25mm():
    for goal in ([.3, 0., .33], [.3, 0., .325], [.4, 0., .4]):
        safe, raw, h = case(occupancy=points_map(VOXEL))
        assert safe.move_planned(goal) and raw.commands and h.violations == []

def test_lateral_1cm_over_a_flat_top_is_refused():        # clearance = z - 0.272 everywhere
    safe, raw, _ = case(occupancy=flat_top(.272))
    with pytest.raises(SkillError, match=r"0\.028000000 m below 0\.030000000 m.*strictly retreating"):
        safe.move_planned([.31, 0., .3])
    assert raw.commands == []

def test_hold_is_admitted_then_closing_in_is_refused():
    _, _, h = case(occupancy=points_map(VOXEL)); q = np.array([.3, 0., .3])
    assert h.vet_step(q, q, .02) is None
    assert "strictly retreating" in h.vet_step(q, q - [0., 0., .001], .02)

def test_a_point_entering_the_margin_refuses_although_another_retreats():
    # kin whose link point is tcp + [.05, 0, 0]; voxels: VOXEL and one 3.1 cm beside the link
    # step up-and-toward: TCP 2.8 -> 2.9 cm, link 3.1 -> 2.9 cm
    assert "point 1 clearance" in h.vet_step(q0, q1, .02)

def test_route_ending_inside_the_margin_is_refused():
    safe, raw, h = case(occupancy=points_map(VOXEL)); s = np.array([.3, 0., .3])
    assert h.vet_step(s, s + [0., 0., .001], .02) is None          # the step itself retreats
    assert vet_route(h, s, [s + [0., 0., .001]], 3.).startswith(
        "route ends inside the margin: point 0 clearance 0.029")
    with pytest.raises(SkillError, match="no safe joint route"):
        safe.move_planned(s + [0., 0., .001])
    assert raw.commands == []

def test_first_lift_safe_final_leg_blocked():             # test_clearance_lift_route.py:94 pattern
    # lift_case(wall_top=.40) + VOXEL: both +3/+6 cm lifts retreat, the second leg hits the keep-out
    with pytest.raises(SkillError, match="no safe joint route"):
        safe.move_planned(goal)
    assert raw.commands == [] and harness.violations == []

def test_payload_samples_never_escape():                  # payload .019 at start, .025 at goal
    with pytest.raises(SkillError, match="no safe joint route"):
        safe.move_planned([.4, 0., .4])
    assert raw.commands == []

def test_static_vets_never_escape():
    assert "0.028000000 m below" in h.vet_pose(np.array([.3, 0., .3]))
```

```python
# tests/test_spatial_commands.py
@needs_pin
def test_a_nudge_that_ends_inside_the_map_margin_is_refused(rig):
    runtime, arm = rig
    runtime.arm.harness.occupancy = _PointObstacle(_tcp(runtime)[:3, 3] + [0., 0., -.020])
    n0 = len(arm.commands)
    r = runtime.execute("move_relative", {"direction": "up", "distance_m": .005})   # ends at 2.5 cm
    assert r["ok"] is False and r["refused_by"] == "safety_harness"
    assert "route ends inside the margin" in r["error"] and len(arm.commands) == n0
    assert runtime.execute("move_relative", {"direction": "up", "distance_m": .02})["ok"] is True
```

- [ ] **Step 2: Run, expect FAIL.** `$T tests/test_clearance_escape.py tests/test_spatial_commands.py -k "margin or escape or hover or lateral or hold or entering or ending or lift or payload or static"`. Today every step inside the margin refuses, so the hover/home test fails, and the end-rule texts are absent.
- [ ] **Step 3: Implement** the interfaces above. In `_approve_step` pass `prev_points = [fk(q_prev) tcp, link_positions(q_prev)[1:]]` and `hold = np.allclose(q_prev, q_next, atol=1e-9)`. `vet_pose` and the payload path pass neither.
- [ ] **Step 4: Run, expect PASS.** Run Step 2's tests, then `$T tests/test_home_routes.py tests/test_clearance_lift_route.py tests/test_approach_target_column.py tests/test_harness_tool_volume.py tests/test_spatial_commands.py tests/test_rebot_occupancy_live.py tests/test_place_path_safety.py`. In particular, `test_empty_tool_at_19mm_fails_before_any_home_command` still matches `0\.019000000 m below 0\.030000000 m`.
- [ ] **Step 5: Mutation checks (all KILLED).**
  - `>` → `>=` in the strict-improvement comparison: killed by `test_lateral_1cm…`.
  - Drop the "already below at q_prev" condition: killed by `test_a_point_entering…`.
  - `hold` forced True: killed by `test_hold_is_admitted…`.
  - `vet_route` end check returning None: killed by `test_route_ending…`.
  - Spatial end check removed: killed by `test_a_nudge_that_ends…`.
  - `prev_points` passed on the attached path: killed by `test_payload_samples_never_escape`.
- [ ] **Step 6: Commit** `feat(safety): strict-retreat escape for TCP/link map clearance; routes must end outside the margin`

### Task A2: Pregrasp hover from the map's top (spec §3.2 rules 3-4)

**Files:**
- Modify: `src/cascade/perception/occupancy.py` (new `column_top` after `clearance_excluding`, `:1411`)
- Modify: `src/cascade/skills/runtime.py:74-101` (extract `_column_geometry`; `_target_column` keeps its opt-in checks and calls it), `:3176-3184` (call `_pregrasp_min_z`)
- Modify: `configs/demo.yaml:158` (`pregrasp_map_top: false` + comment), `configs/arms/rebot_rs.yaml:111-114` (`pregrasp_map_top: true`, extend the comment)
- Test: `tests/test_pregrasp_map_top.py` (new); `tests/test_pregrasp_clears_object_top.py:137` (pin the new key)

**Interfaces:**
- Produces: `OccupancyMap.column_top(center_xy, radius_m: float, z_min: float) -> tuple[float, float] | None`. It returns (top face z = highest occupied centre z + voxel/2, voxel_m) over occupied centres within `radius_m` horizontally and with centre z ≥ `z_min`. It reads under `_refresh_lock`. It returns None (never raises) when the map is stale, failed (`last_error`/`_body_error`), empty, or the column is empty.
- Produces: `_column_geometry(gcfg, harness, center_xy, points) -> tuple | None`. It is today's formula: radius = 95th-percentile spread + 1 cm, clipped to [0.02, `exempt_radius_m`]; floor = max(top − 0.02, table + table_clearance).
- Produces: `MAP_TOP_RAISE_CAP_M = 0.02`; `_pregrasp_min_z(gcfg, harness, fix) -> float | None`.
  - `pregrasp_clear_top_m` null → None (today).
  - `pregrasp_map_top` false, no map, or `column_top` None → `fix_top + clear_top` (today).
  - Otherwise `fix_top + min(max(map_top - fix_top, 0), 0.02) + clear_top + voxel/2`. The column is `_column_geometry` centred on `fix.position[:2]`.

- [ ] **Step 1: Write the failing tests**

```python
FIX_TOP, CLEAR, VOXEL = .093, .03, .01
@pytest.mark.parametrize("map_top, expected", [
    (None,           FIX_TOP + CLEAR),                     # stale map / empty column: today's hover
    (FIX_TOP + .009, FIX_TOP + .009 + CLEAR + VOXEL / 2),  # the recorded 0.2-0.9 cm gaps
    (FIX_TOP + .05,  FIX_TOP + .02 + CLEAR + VOXEL / 2),   # one residue voxel: raise capped at 2 cm
    (FIX_TOP - .01,  FIX_TOP + CLEAR + VOXEL / 2),         # never below the fix top
])
def test_hover_from_the_map_top(map_top, expected):
    assert _pregrasp_min_z(GCFG_ON, harness_with(column_top=map_top), cup_fix()) == pytest.approx(expected)

def test_flag_off_or_clear_top_null_keeps_todays_hover(): ...   # FIX_TOP + CLEAR, and None

def test_column_top_reads_only_the_target_column():     # GridBridge(voxel=.01): cup box top .095, neighbour top .15 at 8 cm
    top, voxel = occ.column_top(CUP_XY, .05, .07)
    assert voxel == .01 and .095 <= top < .095 + .01 + 1e-9

def test_column_top_is_none_when_stale_failed_or_empty(): ...
```

Also add to `test_only_the_rebot_profile_turns_it_on` (`:137`): rebot_rs `pregrasp_map_top is True`, mock `False`.

- [ ] **Step 2: Run, expect FAIL** (`ImportError: _pregrasp_min_z`; `AttributeError: column_top`). Command: `$T tests/test_pregrasp_map_top.py tests/test_pregrasp_clears_object_top.py`.
- [ ] **Step 3: Implement** the interfaces. `skill_grasp_object` replaces the inline block at `:3176-3184` with `pregrasp_min_z = _pregrasp_min_z(gcfg, harness, fix)`.
- [ ] **Step 4: Run, expect PASS.** Run Step 2 plus `tests/test_approach_target_column.py`.
- [ ] **Step 5: Mutation checks.**
  - Cap `0.02` → `0.2`: killed by the residue case.
  - Remove `max(…, 0)`: killed by the "never below" case.
  - Radius computed without the 95th percentile (whole workspace): killed by `test_column_top_reads_only_the_target_column`.
- [ ] **Step 6: Commit** `feat(grasp): pregrasp hover from the map top (capped +2 cm, opt-in per profile)`

### Task A3: Target-column exit after an air grasp (spec §3.2 rules 1-2)

**Files:**
- Modify: `src/cascade/safety/trajectory.py:121-131` (extract `lift_ik`; `plan_route` keeps its deadline checks around it)
- Modify: `src/cascade/skills/runtime.py`:
  - keep the approach column in `approach_column` (`:3301-3305`; None on the scene-gate path);
  - new `SkillRuntime._air_grasp_exit`;
  - air-grasp branch `:3507-3522`.
- Test: `tests/test_air_grasp_exit.py` (new; an `rt` fixture like `tests/test_b36_pick_recovery.py:39-48` but with `arm.object_stop_frac = None` (nothing held), and its `_fail_first_lift`)

**Interfaces:**
- Produces: `trajectory.lift_ik(kin, start, height_m: float) -> np.ndarray | None`. It takes FK of `start`, adds `height_m` in base z, and calls `kin.ik(target, q_init=start.copy(), max_iter=80, retries=0)`. It returns None on failure or a non-finite result.
- Produces: `SkillRuntime._air_grasp_exit(self, column, move: Callable[[np.ndarray], bool]) -> str`. Return values: `"completed (+3 cm)"`, `"completed (+6 cm)"`, `"refused: <reason>"`, `"did not settle"`, `"skipped: <why>"`.
  - It runs inside `_column_exemption(harness, column)` and tries heights `(0.03, 0.06)` from the measured q.
  - A preflight refusal (`SkillError`) moves on to the next height. `MotionHalted`, a latched e-stop or a halt end it at once.
  - It is skipped when `_held_provisional` or `carry_attachment.active(self)` is set, or when the harness is e-stopped/halted.
  - `move` matches the pre-close retreat (`:3468-3477`): `_move_joints(q, …, _linear_tool_path=True, **_scene_motion(q))` with a scene gate, else `_move_planned(q, duration_s=descend_duration_s)`.
- Produces: the air-grasp result gains `"exit_withdrawal": <str>`. `error` gets `f"; exit withdrawal {recovery}"` appended only for `refused…` / `did not settle`.

- [ ] **Step 1: Write the failing tests**

```python
@needs_pin
def test_air_grasp_withdraws_from_the_pregrasp(rt):           # MockArm object_stop_frac None = air grasp
    res = rt.skill_grasp_object("red cube")
    assert res["error"].startswith("air grasp: gripper closed fully, object not held")
    assert res["exit_withdrawal"] == "completed (+3 cm)"
    assert tcp_z(rt) >= tcp_z_at(q_pre_spy) + .03 - 2e-3

def test_exit_runs_under_the_approach_column_only(rt, monkeypatch):
    # spy harness.target_column_exemption: entered once with the approach column; _grasp_exempt None after

def test_exit_refused_by_a_neighbour_above_keeps_the_arm_put(rt):
    # make_map(GridBridge([TABLE, box 2 cm above the +3/+6 cm lift TCP, outside the column]), required=False)
    out = rt._air_grasp_exit(column, move)
    assert out.startswith("refused:") and len(arm.commands) == n0

def test_a_provisional_hold_never_runs_the_exit(rt, monkeypatch):   # _fail_first_lift: lift raises after close
    # spy _air_grasp_exit: never called; result error is the lift failure

def test_halt_during_the_first_lift_never_tries_6cm(rt, monkeypatch): ...
```

- [ ] **Step 2: Run, expect FAIL** (`KeyError: 'exit_withdrawal'`, `AttributeError: _air_grasp_exit`). Command: `$T tests/test_air_grasp_exit.py`.
- [ ] **Step 3: Implement** the interfaces. Keep the air-grasp memory/record lines exactly as they are. The exit runs after them, before `return`.
- [ ] **Step 4: Run, expect PASS.** Run Step 2 plus `tests/test_b36_pick_recovery.py tests/test_clearance_lift_route.py tests/test_home_routes.py`.
- [ ] **Step 5: Mutation checks.**
  - Drop the `_held_provisional` skip: killed by `test_a_provisional_hold…`.
  - Continue to +6 cm after `MotionHalted`: killed by `test_halt_during…`.
  - Exit outside `_column_exemption`: killed by `test_exit_runs_under_the_approach_column_only`.
- [ ] **Step 6: Commit** `feat(grasp): withdraw through the approach column after an air grasp`

### Task A4: Map snapshot on every clearance refusal or halt (spec §3.3)

**Files:**
- Modify: `src/cascade/perception/occupancy.py`:
  - `__init__` (`:181-283`): `refusal_snapshot` kwarg;
  - `from_config` (`:625-695`): pass `cfg.get("refusal_snapshot")`;
  - new `nearby_occupied`.
- Modify: `src/cascade/safety/harness.py`:
  - deque next to `_contact_reports` (`:196-198`);
  - `drain_clearance_reports` next to `:385-388`;
  - a `ctx` keyword on `_occupancy_violation`, `_payload_occupancy_violation`, `_tool_volume_violation`, passed from `_approve_step` (`:872-887`) and `vet_pose` (`:977-986`).
- Modify: `src/cascade/skills/runtime.py:891-896` (drain stale reports with the contact reports), `:1275-1277` (`trace_context["clearance"]`)
- Modify: `configs/demo.yaml:45-62` (`refusal_snapshot: {radius_m: 0.05, max_voxels: 64, max_reports: 32}` + comment); `docs/NVBLOX.md` (one paragraph on `context.clearance`)
- Test: `tests/test_clearance_snapshots.py` (new)

**Interfaces:**
- Produces: `OccupancyMap.refusal_snapshot: RefusalSnapshot` (NamedTuple `radius_m=0.05, max_voxels=64, max_reports=32`). Bad values raise `OccupancyError`.
- Produces: `OccupancyMap.nearby_occupied(point, radius_m: float, limit: int) -> np.ndarray`. It returns rows (≤ limit, 4) of `[x, y, z, d]`, sorted by d, read under `_refresh_lock` from the cache the gate read. It is empty when there is no cache.
- Produces: `GateCtx(phase: str, q, q_prev, T_tcp)` (NamedTuple in harness.py). `phase` is `approve` | `vet_step` | `vet_pose`.
- Produces: `SafetyHarness.drain_clearance_reports() -> list[dict]`. Report keys (exact):
  - `phase`, `gate` (`tcp`, `link {i+1}`, `attached`, or the tool part name), `reason`;
  - `q`, `q_prev` (None for `vet_pose`), `tcp`;
  - `sample` (base), `sample_tcp`, `clearance_m` (None if non-finite), `margin_m`;
  - `exemption` (`{"cylinder": [cx, cy, r, z_min]}` or None);
  - `map` (`{"refreshes", "age_s", "voxel_m", "occupied"}`);
  - `voxels` (`[[x, y, z, d], …]`, rounded to 1 mm);
  - `snapshot_error` (only on failure).
- Bounds: dedupe per drain by `(gate, sample rounded to 0.01 m, map refreshes)`. The deque keeps the newest `max_reports`, so the refusal that ended the call is always kept. Reports are built only after a reason exists.
- Produces: `trace_context["clearance"]` (list of reports) when non-empty. Stale reports are drained at the start of each call.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("gate", ["tcp", "link 2", "attached", "wrist camera", "jaws"])
def test_each_gate_files_exactly_one_report_per_refusal(gate): ...   # rebot + GridBridge per gate

def test_report_fields_and_bounds():
    (r,) = h.drain_clearance_reports()
    assert set(r) >= {"phase", "gate", "q", "q_prev", "tcp", "sample", "sample_tcp", "clearance_m",
                      "margin_m", "exemption", "map", "voxels"}
    assert r["margin_m"] == .03 and 0 < len(r["voxels"]) <= 64
    assert all(row[3] <= .05 for row in r["voxels"]) and r["voxels"] == sorted(r["voxels"], key=lambda v: v[3])
    assert all(round(v, 3) == v for row in r["voxels"] for v in row)

def test_reports_dedupe_and_keep_the_newest_32(): ...  # 10x same -> 1; 40 distinct samples -> 32 newest

def test_admitted_stream_never_builds_a_snapshot(monkeypatch):    # Review Focus 4
    calls = spy(occ, "nearby_occupied")
    assert safe.move_planned(clear_goal) and calls == []
    with pytest.raises(SkillError):
        safe.move_planned(blocked_goal)
    assert len(calls) == 1

def test_snapshot_failure_keeps_the_refusal(monkeypatch):         # nearby_occupied raises
    assert h.vet_step(a, b, .02) == reason_without_snapshot_failure
    assert "snapshot_error" in h.drain_clearance_reports()[0]

@needs_pin
def test_runtime_drains_reports_into_the_trace(rig):              # refused move_relative
    row = last_trace_row(runtime)                                   # runtime.trace.run_dir / "trace.jsonl"
    assert row["context"]["clearance"][0]["gate"] == "tcp"
    # the next (clear) call carries no "clearance" key
```

- [ ] **Step 2: Run, expect FAIL** (`AttributeError: drain_clearance_reports`). Command: `$T tests/test_clearance_snapshots.py`.
- [ ] **Step 3: Implement** the interfaces. Snapshot building is wrapped in `try/except Exception` that stores `snapshot_error` and re-raises nothing. A report never changes the returned reason.
- [ ] **Step 4: Run, expect PASS.** Run Step 2 plus `tests/test_clearance_escape.py tests/test_harness_tool_volume.py tests/test_home_routes.py tests/test_aspire_admission.py tests/test_wrist_narration.py`.
- [ ] **Step 5: Mutation checks.**
  - Build the snapshot before the reason check: killed by `test_admitted_stream…`.
  - Let the snapshot exception propagate: killed by `test_snapshot_failure…`.
  - Dedupe key without `refreshes`: killed by a dedupe case with two refreshes.
- [ ] **Step 6: Commit** `feat(safety): file a bounded map snapshot with every clearance refusal (context.clearance)`

### Task A5: Tool volume from measured parts; shipped values stay today's slab (spec §3.4, §8)

**Files:**
- Modify: `src/cascade/skills/relative_motion.py:657-680` (`camera_box` gains `front_m=0.0`, `bottom_m=0.0`)
- Modify: `src/cascade/skills/spatial_moves.py`:
  - `:115-124`, `:155`, `:158-181`: `CameraDims`, `DEFAULT_WRIST_CAMERA` += `front_m: 0.0, bottom_m: 0.0`;
  - `:330-332`: use the fields by name.
- Modify: `src/cascade/safety/tool_volume.py:30` (`_KEYS` += `housing`, `fingers`), `:84-145` (parts, guard, caliper helpers)
- Modify: `configs/arms/rebot_rs.yaml:67-81` (extend the comment with the URDF part boxes and "ship with the calipers"; no new keys)
- Create: `docs/evidence/wrist-camera-calipers/measurement.json` (schema `cascade.wrist_camera_calipers/1`; mm faces `front_behind_tip`, `rear_behind_tip`, `minus_y`, `plus_y`, `bottom_above_axis`, `top_above_axis`, all `null`; `photos: []`; reference frame text from spec §8)
- Test: `tests/test_tool_volume_parts.py` (new); `tests/test_harness_tool_volume.py:118-135` (pass `front_m`/`bottom_m`); `tests/test_clearance_lift_route.py:364` (derive the obstacle from the config: `[-wc.behind_tip_m, wc.half_width_m - .01, -.024]`)

**Interfaces:**
- Produces: `CameraDims(protrusion_m, behind_m, half_width_m, front_m=0.0, bottom_m=0.0)` NamedTuple, returned by `wrist_camera_dims()` and held in `Settings.wrist_camera`. `camera_box` validates `0 ≤ front_m < behind_m` and `0 ≤ bottom_m < protrusion_m`. The box spans approach −behind..−front and up bottom..protrusion.
- Produces: `safety.tool_volume.housing` / `.fingers`: absent or null = off. Otherwise `{min: [a, j, u], max: [a, j, u]}` in (approach, jaw, camera-up) coordinates, mapped with `_TOOL_COLUMNS[tool_axis_order]`, finite, min < max, |v| ≤ 0.30. Part names are `"housing"` and `"fingers"`.
- Produces: `build_tool_volume` raises `ValueError` when `front_m > 0` without `fingers` ("dropping the slab front leaves the open fingers unsampled", spec §3.4.4).
- Produces: `TODAY_SLAB = CameraDims(0.08, 0.14, 0.075, 0.0, 0.0)`.
- Produces: `camera_faces_from_calipers(measured_mm: dict, today=TODAY_SLAB) -> CameraDims`.
  - rear, top and `max(minus_y, plus_y)` round UP to 5 mm.
  - front and bottom round DOWN to 5 mm (outward for those faces), never below 0.
  - A null face keeps `today`.
- Produces: `uncovered_camera_faces(shipped: CameraDims, required: CameraDims) -> list[str]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_faces_round_outward_to_5mm():
    got = camera_faces_from_calipers({"front_behind_tip": 81, "rear_behind_tip": 136, "minus_y": 74,
                                      "plus_y": 71, "bottom_above_axis": 37, "top_above_axis": 77})
    assert got == CameraDims(.08, .14, .075, .08, .035)

def test_a_missing_number_keeps_todays_face():
    assert camera_faces_from_calipers({"rear_behind_tip": 146}) == TODAY_SLAB._replace(behind_m=.15)

def test_shipped_config_covers_the_committed_calipers():      # all null today -> today's slab
    assert uncovered_camera_faces(shipped_rebot_dims(), required_from_json()) == []

def test_the_coverage_check_is_red_against_the_old_slab():
    assert uncovered_camera_faces(TODAY_SLAB, camera_faces_from_calipers({"rear_behind_tip": 146})) == ["rear"]

@needs_pin
def test_shipped_rebot_volume_is_unchanged(rebot):            # same samples as before the change
    assert set(rebot.volume.names) == {"wrist camera", "jaws"}

@needs_pin
def test_urdf_part_boxes_cover_the_hulls_closed_and_fully_open():
    # ApproachClearance(load_geometry(model), kin).placed(q, gap) for gap in (0, 0.09), into the TCP frame
    HOUSING = ((-.160, -.095, -.045), (-.070, .095, .045)); FINGERS = ((-.090, -.085, -.020), (0., .085, .020))

def test_housing_and_fingers_are_sampled_when_configured(): ...      # names + extents + spacing
def test_front_without_fingers_is_refused(): ...                     # ValueError
def test_calipers_and_parts_ship_together():   # any non-null caliper face => rebot_rs has housing and fingers
```

- [ ] **Step 2: Run, expect FAIL** (`ImportError: CameraDims`). Command: `$T tests/test_tool_volume_parts.py`.
- [ ] **Step 3: Implement** the interfaces. Both layers keep calling one parser (`wrist_camera_dims`) and one geometry function (`camera_box`).
- [ ] **Step 4: Run, expect PASS.** Run Step 2 plus `tests/test_harness_tool_volume.py tests/test_clearance_lift_route.py tests/test_spatial_commands.py tests/test_relative_motion.py tests/test_franka_spatial_profile.py tests/test_rebot_occupancy_live.py`.
- [ ] **Step 5: Mutation checks.**
  - ceil → floor on the rear face: killed by `test_faces_round_outward…`.
  - floor → ceil on the front face: same test.
  - Drop the front-without-fingers guard: killed by `test_front_without_fingers…`.
  - Test mutation: set `rear_behind_tip: 146` in the committed JSON → `test_shipped_config_covers…` KILLED; restore.
- [ ] **Step 6: Commit** `feat(safety): tool volume parts from measurements (housing, fingers, calipered camera); shipped slab unchanged`

### Task A6: Offline exact replay of recorded refusals (spec §6.3)

**Files:**
- Create: `scripts/replay_clearance.py` (stdlib + cascade; docstring: dev interpreter `uv run --extra kinematics python scripts/replay_clearance.py <trace.jsonl>…`; never connects to anything)
- Test: `tests/test_replay_clearance.py` (new)

**Interfaces:**
- Consumes: A4 report keys; `SafetyHarness`, `build_tool_volume`, `load_demo_config(arm=…, camera="mock", llm="mock")`.
- Produces:
  - `load_reports(paths) -> list[dict]`: every `context.clearance` entry, plus `run`, `step`, `skill`.
  - `ReplayMap(voxels: np.ndarray, voxel_m: float)`: occupancy stand-in with `clearance`, `clearance_excluding`, `nearby_occupied`, `_occupied`, `_refresh_lock`, `required=False`, `unknown_blocks=False`; exact cube distances.
  - `replay(report, *, arm, tool_volume_override=None) -> dict`, keys `old`, `L1`, `L4` (each `"allowed"`/`"refused: …"`), `partial_map: True`.
    - `old` is the static strict check at `q`.
    - `L1` is `vet_step(q_prev, q)` (static when `q_prev` is None).
    - `L4` is `old` with the override tool volume.
  - CLI prints one JSON line per report and a summary `{lever: {allowed, refused}}`.

- [ ] **Step 1: Write the failing test.** Produce a real refusal on a rebot harness + GridBridge, drain its report, and write it as a trace row with `json.dumps`. Assert the following:

```python
assert out["old"].startswith("refused") and f"{rep['clearance_m']:.3f}" in out["old"]
assert out["L1"] == "allowed"               # the recorded step strictly retreated (built that way)
assert summary == {"old": {"allowed": 0, "refused": 1}, "L1": {"allowed": 1, "refused": 0}, ...}
```

- [ ] **Step 2: Run, expect FAIL** (`ModuleNotFoundError`; load the script with `importlib.util.spec_from_file_location`). Command: `$T tests/test_replay_clearance.py`.
- [ ] **Step 3: Implement** the interfaces.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat(tools): replay_clearance.py re-vets recorded refusals from their map snapshots`

### PR-A exit (controller, not a task)

1. Run a full-suite failing-ID diff against `origin/main` (4 shards from detached worktrees of both shas, same flags), run ruff, and confirm every mutation above was KILLED.
2. PR body, evidence table: spec §1.2 and the §6.2 replay rows for PR-A.
3. Deploy notes:
   - no weights and no env changes;
   - `rebot_rs` turns on `pregrasp_map_top`;
   - lever-4 parts stay off until the calipers arrive.
4. Live acceptance A1-A3 (spec §7) is operator-gated after merge, under `real-robot-bringup`. Run the replay on every rig session before PR-B ships.
5. AGENTS.md paste text, Motion safety path, AGENTS.md:61 (spec §10, sentences 1-3 verbatim):
   > TCP/joint occupancy clearance has an escape rule: a step from inside `min_clearance_m` is admitted only if every point already inside strictly gains clearance (or the step is an exact hold) and none enters, and planners accept a route only if its final pose is outside the margin (`vet_route`, spatial `plan_route`). Attached-payload samples never escape. Every clearance refusal/halt files a bounded map snapshot into the trace (`context.clearance`).
6. Follow-ups:
   - calipers → fill `measurement.json`, then configure camera faces + housing + fingers in one commit (the pinned tests force it);
   - Q3 adaptive pre-grasp opening, as its own PR.

---

## PR-B "object-aware nvblox + corridor margin" (ships with no margin change)

### Task B1: Label inputs: box age and the map's committed frame

**Files:**
- Modify: `src/cascade/types.py:58-66` (`Detection.box_t: float | None = None`)
- Modify: `src/cascade/perception/brain_detector.py:590-612` (`_detections(…, box_t)`; the track step passes the entry's `t` (`:163`), a fresh brain answer the frame's `t`); `src/cascade/perception/detector.py:70` (`MockDetector` sets `box_t=frame.t`)
- Modify: `src/cascade/perception/occupancy.py`:
  - `__init__`: `committed_frame_t = None`, `clears = 0`;
  - `_commit_grid` (`:978-987`) records `committed_frame_t = float(frame.t)`;
  - `clear_bridge_map` (`:697-716`) increments `clears` on success, and so does `begin_scene_reset` (`:365-390`);
  - new `label_inputs`.
- Test: `tests/test_label_inputs.py` (new)

**Interfaces:**
- Produces: `Detection.box_t`. None means "unknown box time" and never labels (F3).
- Produces: `LabelInputs(occupied, origin, voxel_m, frame_t, epoch, refreshes)` NamedTuple. `epoch = (scene_reset_generation, clears)`; `occupied` is a read-only view.
- Produces: `OccupancyMap.label_inputs() -> LabelInputs | None`, read under `_refresh_lock`. It is None when stale, failed, or without a grid.

- [ ] **Step 1: Failing tests.**
  - A brain track step carries the cached entry's `t`; a fresh refer carries `frame.t` (use the fake-brain doubles of `tests/test_brain_detector*.py`).
  - `MockDetector` sets `frame.t`.
  - `label_inputs().frame_t == frame.t` after `refresh(frame)` on a required and an optional `make_map`.
  - `clears` grows on `clear_bridge_map`.
  - The result is None when stale.
- [ ] **Step 2: Run, expect FAIL.** Command: `$T tests/test_label_inputs.py`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS.** Run Step 2 plus `tests/test_brain_detector*.py tests/test_occupancy*.py`.
- [ ] **Step 5: Commit** `feat(perception): detection box time and the map's committed frame for voxel labels`

### Task B2: Same-frame voxel label snapshot (spec §4.1-4.2)

**Files:**
- Create: `src/cascade/perception/voxel_labels.py`
- Test: `tests/test_voxel_labels.py` (new; a synthetic camera, `render_depth` + `box_surface` for depth and per-object masks, occupied 1 cm voxel centres of the boxes)

**Interfaces:**
- Produces: constants `UNKNOWN = -1`, `ROBOT = -2`, `CONFLICT = -3`; `DEPTH_BAND_M = (-0.015, 0.04)`; `PRISM_DILATE_M = 0.01`; `MIN_POINTS = 50`; `MIN_HEIGHT_M = 0.01`; `ROBOT_BAND_M = 0.05`.
- Produces: `encode_keys(keys: (N, 3) int) -> (N,) int64`, offset 2**20 per axis; `voxel_keys(points, origin, voxel_m) -> (N, 3) int` (round).
- Produces: `Prism(center_xy, axes_xy (2, 2), half_xy (2,), z_lo, z_hi)` with:
  - `contains(points, dilate_m=0.0) -> bool[N]`;
  - `as_box() -> gripper_clearance.Box`;
  - `fit_prism(points_base, table_z) -> Prism | None`: `cv2.minAreaRect` on xy, extruded table_z..max z; None below `MIN_POINTS` or `MIN_HEIGHT_M`.
- Produces: `Instance(label, det_index, prism, n_points, box_t)` and `LabelSnapshot(frame_t, built_t, K, T_base_cam, thumb, origin, voxel_m, epoch, keys (N,3) int32, labels (N,) int16, instances: tuple[Instance, …], dropped: dict[int, str])`.
- Produces: `build_snapshot(inputs, frame, dets, T_base_cam, *, table_z, accept_points: Callable[[np.ndarray], bool], robot_points: np.ndarray | None, now, box_max_age_s) -> tuple[LabelSnapshot | None, str | None]`.
  - It returns None with reason `"map commit is another frame"` when `inputs.frame_t != frame.t`.
  - Per instance: F2 (a real mask; masks with `.cpu()` are converted once, vectorised), F3 (`box_t` not None and `now - box_t <= box_max_age_s`), ≥ 50 lifted points, the workspace filter (`accept_points`), and prism height ≥ 1 cm.
  - Per voxel: in image, valid depth, cam-z within the depth band, inside mask i, inside prism i dilated 1 cm → i.
  - Two candidates, mask i inside another instance's dilated prism, or a candidate within `ROBOT_BAND_M` of `robot_points` → `CONFLICT`. Otherwise within the band → `ROBOT`; else `UNKNOWN`. Labels never carry over between frames.
- Produces: `snapshot_usable(snap, inputs_now, *, now, label_max_age_s) -> str | None`.
  - F1: `"labels {age:.1f} s old"`, with age measured from `snap.frame_t`.
  - F4: `"map changed since the labels"` when origin, voxel or epoch differs.
- Produces: `exemptable(snap, i) -> keys`: voxels labelled exactly `i`. `ROBOT`/`CONFLICT`/`UNKNOWN` are never exemptable (Q4).

- [ ] **Step 1: Write the failing tests**

```python
def test_labels_bind_to_the_map_frame_only():
    assert build_snapshot(inputs._replace(frame_t=frame.t - .3), frame, dets, ...) == (None, "map commit is another frame")

def test_target_and_neighbour_voxels_get_their_own_labels(): ...    # no voxel swapped
def test_box_only_detection_never_labels(): ...                    # F2: mask None -> UNKNOWN
@pytest.mark.parametrize("age, labelled", [(None, False), (29., True), (31., False)])
def test_box_age(age, labelled): ...                               # F3, box_max_age_s 30
def test_freshness():                                              # F1 / F4
    assert snapshot_usable(snap, inputs, now=snap.frame_t + 1.6, label_max_age_s=1.5) == "labels 1.6 s old"
    assert snapshot_usable(snap, inputs._replace(epoch=(0, 1)), now=snap.frame_t, label_max_age_s=1.5) \
        == "map changed since the labels"
@pytest.mark.parametrize("offset_m, labelled", [(.039, True), (.041, False), (-.014, True), (-.016, False)])
def test_depth_band(offset_m, labelled): ...                       # voxel cam-z = measured + offset
def test_conflicts_are_never_labelled(): ...  # two masks; mask i in prism j + 1 cm; mask + robot band
def test_robot_voxels_are_reported_never_exemptable(): ...         # Q4
def test_prism_fit_thresholds(): ...                               # 49 points / 9 mm tall -> None
def test_tensor_like_masks_are_converted(): ...                    # .cpu().numpy() object
def test_build_stays_vectorised():                                 # 20 000 voxels, 5 masks, 1280x720
    assert median_ms(build, 5) < 50
```

- [ ] **Step 2: Run, expect FAIL** (`ModuleNotFoundError`). Command: `$T tests/test_voxel_labels.py`.
- [ ] **Step 3: Implement** the interfaces with numpy + cv2 only.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Mutation checks.**
  - Depth band `+0.04` → `+0.06`: killed by `test_depth_band`.
  - Drop the two-mask conflict: killed by `test_conflicts…`.
  - F3 `<=` → no check: killed by `test_box_age`.
  - Frame-binding check removed: killed by `test_labels_bind…`.
- [ ] **Step 6: Commit** `feat(perception): same-frame voxel labels from SAM masks, depth band and fitted prisms`

### Task B3: Watcher builds snapshots; `grasp.object_aware` config

**Files:**
- Create: `src/cascade/safety/object_aware.py` (`ObjectAwareConfig` only in this task)
- Modify: `src/cascade/perception/world.py:200-243` (`object_aware` kwarg) and `_tick` after `:458`; `src/cascade/apps/demo.py:725` (pass the config); `src/cascade/perception/occupancy.py` (`label_snapshot`, `label_status` attributes)
- Modify: `configs/demo.yaml` after `:192`: `object_aware: {enabled: false, labels: false, label_max_age_s: null, box_max_age_s: null, prism_dilate_m: 0.01, corridor_margin_m: 0.03}` + comment. `configs/arms/rebot_rs.yaml:104-114`: `object_aware: {labels: true}`, report-only, with a comment that `enabled` waits for §7.
- Test: `tests/test_object_aware_config.py`, `tests/test_watcher_labels.py` (new; watcher pattern of `tests/test_occupancy_capture_order.py`)

**Interfaces:**
- Produces: `ObjectAwareConfig.from_config(cfg) -> ObjectAwareConfig`, frozen dataclass with:
  - `enabled`, `labels` (`enabled` implies labels);
  - `label_max_age_s` (null → `occupancy.max_age_s`) and `box_max_age_s` (null → `detector.max_cache_age_s`, 30);
  - `prism_dilate_m`;
  - `corridor_margin_m`: key absent → 0.03; < 0.010 → `ValueError("corridor_margin_m below the 0.010 m floor")`; capped at `safety.min_clearance_m`.
- Produces: on each map-camera tick (`maps_depth`, `T` set, labels on), the watcher calls `build_snapshot` after `last_dets` is set.
  - It stores `occupancy.label_snapshot` (newest non-None) and `occupancy.label_status` (None or the reason).
  - Any exception → `label_status = f"label snapshot failed: {e}"`; the tick continues.
  - `robot_points` are the posed vertices that masked this frame: `MeshBodyMask.last_hull_points` (new attribute recorded in `__call__`), None for other masks.

- [ ] **Step 1: Failing tests.**
  - Config: defaults; rebot_rs `labels True` / `enabled False`; mock both False; floor error; the cap; the code default when the key is absent.
  - Watcher: a map-camera tick stores a snapshot with `frame_t == frame.t`; labels off → `build_snapshot` never called (spy).
  - Watcher: a raising labeller → `label_status` set, `harness.heartbeat` count and `watcher.ticks` still grow.
- [ ] **Step 2: Run, expect FAIL.** Command: `$T tests/test_object_aware_config.py tests/test_watcher_labels.py`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS.** Run Step 2 plus `tests/test_occupancy_capture_order.py tests/test_occupancy_during_motion.py tests/test_link_self_mask.py tests/test_wrist_narration.py`.
- [ ] **Step 5: Mutation:** remove the `try/except` around the build → KILLED by the raising-labeller test.
- [ ] **Step 6: Commit** `feat(perception): watcher builds per-frame voxel label snapshots (report-only on rebot_rs)`

### Task B4: Exact target exemption in the harness (spec §4.3)

**Files:**
- Modify: `src/cascade/perception/occupancy.py:1367-1411` (`clearance_excluding`)
- Modify: `src/cascade/safety/object_aware.py` (`GraspEpisode` data + expiry)
- Modify: `src/cascade/safety/harness.py`:
  - scope context;
  - `_occupancy_violation`, `_payload_occupancy_violation`, `_tool_volume_clearance` (`:1032-1059`);
  - `vet_pose` map block (`:977-986`).
- Test: `tests/test_object_aware_exemption.py` (new; `side_grasp_geometry`, `harness`, `neighbour`, `descent` from `tests/test_harness_tool_volume.py`); `tests/test_harness_tool_volume.py:236` (one `exclude_keys` case)

**Interfaces:**
- Produces: `clearance_excluding(points, *, below_z=None, cylinder=None, exclude_keys=None, key_grid=None, exclude_where=None, within=None)`.
  - `exclude_keys` (sorted int64 from `encode_keys`) drops voxels whose key is listed, but only when `key_grid == (tuple(origin), voxel_m, epoch)` of the current grid, compared under the lock. On a mismatch it excludes nothing.
  - `exclude_where(points) -> bool[M]` is a hook for Task B6.
- Produces: `GraspEpisode(keys: np.ndarray, key_grid: tuple, neighbours: tuple[Instance, …], target_prism: Prism, halt_generation: int)` with `expired_reason: str | None` (latched) and `armed: bool`.
- Produces: `SafetyHarness.object_aware(episode)`, a context manager that arms it and sets `occupancy.active_episode`; on exit it disarms and clears it. `SafetyHarness._episode_exclusion() -> tuple | None` expires the episode on e-stop or a `_halt_generation` change.
- Inside an armed, unexpired scope:
  - map checks for TCP/link points, the tool volume and payload samples use `clearance_excluding(…, exclude_keys=episode.keys, key_grid=…, within=min_clearance_m)`;
  - map checks ignore `_grasp_exempt` and `exempt_xy`;
  - TCP/link points ignore table-band voxels (≤ `table_z + tool_volume.table_band_m`) only while they lie inside the descent cylinder in force (`_grasp_exempt` or `vet_pose`'s `exempt_xy`), which is exactly where today's cylinder hid them (spec §4.3.4: no reduction). They use one `clearance_excluding` call with a per-sample `band_mask: bool[N]` (new kwarg), so the escape rule's prev+next points still come from one grid read;
  - table rules keep the cylinder.
- Outside a scope everything is byte-identical to today (place, push, release keep the cylinder).

- [ ] **Step 1: Write the failing tests**

```python
@needs_pin
def test_exact_target_keys_admit_the_wide_cup_descent_without_a_map_cylinder(rebot): ...  # :205 geometry
@needs_pin
def test_a_neighbour_10cm_away_is_now_checked_during_the_descent(rebot):
    assert legacy_descent_reason(rebot, neighbour(rebot, .10)) is None                  # premise: hidden today
    assert "clearance" in object_aware_descent_reason(rebot, neighbour(rebot, .10))
@needs_pin
def test_target_exemption_never_leaks_to_a_touching_neighbour(rebot):                 # Review Focus 3
    # neighbour box 1 cm from TARGET; E = target keys only; jaws/camera/TCP sample next to it -> refused
def test_grid_change_or_map_clear_excludes_nothing(rebot): ...                         # clears += 1
def test_halt_or_estop_expires_the_episode(rebot): ...
def test_table_rules_still_use_the_descent_cylinder(rebot): ...
def test_table_band_is_ignored_only_inside_the_descent_cylinder(rebot): ...   # a link 1.5 cm over a
                                                       # table voxel OUTSIDE the cylinder still refuses
def test_payload_samples_use_the_episode_keys(rebot): ...
def test_no_scope_is_todays_behaviour(rebot): ...    # :172/:192/:205 reasons identical to legacy
```

- [ ] **Step 2: Run, expect FAIL.** Command: `$T tests/test_object_aware_exemption.py tests/test_harness_tool_volume.py`.
- [ ] **Step 3: Implement** the interfaces.
- [ ] **Step 4: Run, expect PASS.** Run Step 2 plus `tests/test_approach_target_column.py tests/test_clearance_escape.py tests/test_rebot_occupancy_live.py tests/test_occupancy_held_payload.py`.
- [ ] **Step 5: Mutation checks.**
  - `key_grid` mismatch → exclude anyway: killed by `test_grid_change…`.
  - Expiry ignored: killed by `test_halt_or_estop…`.
  - In scope, still skip map points inside `_grasp_exempt`: killed by `test_a_neighbour_10cm…`.
  - `band_mask` all True: killed by `test_table_band_is_ignored_only…`.
  - Payload without `exclude_keys`: killed by `test_payload_samples…`.
- [ ] **Step 6: Commit** `feat(safety): exempt exactly the frozen target voxels in object-aware grasps`

### Task B5: Grasp episode: freeze, refuse, shrink, release (spec §4.2 "Grasp episode", §4.5)

**Files:**
- Modify: `src/cascade/safety/object_aware.py` (`freeze_episode`, `GraspEpisode.shrink`)
- Modify: `src/cascade/skills/runtime.py`:
  - `skill_grasp_object`: freeze after `frame, fix` are known (`:2849-2851`) and before `grasps = generate_batch(0)` (`:2952`);
  - `_target_column` returns None in object-aware mode (`:74-101`);
  - the approach (`:3295-3308`) through the lift (`:3448`) runs inside `harness.object_aware(episode)`;
  - the A3 exit runs in a second `object_aware(episode)` scope.
- Modify: `src/cascade/perception/world.py` (after storing a snapshot: `occupancy.active_episode.shrink(snap)` when set)
- Modify: `docs/NVBLOX.md:551+` (target approach), `docs/ARCHITECTURE.md:640-652` (exact-target exemption), `configs/arms/rebot_rs.yaml:108-110` comment (extend)
- Test: `tests/test_object_aware_grasp.py` (new); `tests/test_approach_target_column.py` (add object-aware cases; `:168-178` profile pins unchanged)

**Interfaces:**
- Consumes: `LabelSnapshot`, `snapshot_usable`, `fit_prism`, `exemptable` (B2); `ObjectAwareConfig` (B3); `GraspEpisode` (B4); `_thumbnail`, `_scene_diff` from `perception/brain_detector.py` (imported, not copied).
- Produces: `freeze_episode(occ, frame, fix, cfg, *, now, scene_change, table_z, halt_generation) -> GraspEpisode`. It raises `SkillError("target voxels unknown: <reason>")`. Reasons, checked in this order (exact text):
  1. `no occupancy map`
  2. `no label snapshot`
  3. `labels {age:.1f} s old`
  4. `map changed since the labels`
  5. `no SAM mask`
  6. `no box time`
  7. `box {age:.0f} s old`
  8. `localize frame is not from the map camera` (frame `K`/`T_base_cam` differ from the snapshot's, atol 1e-6)
  9. `localize frame {age:.1f} s old` (> `label_max_age_s`)
  10. `scene changed since the map frame` (thumbnail diff > `detector.scene_change`, 6.0)
  11. `no target voxels`
- `E` = snapshot keys visible in the localize frame (depth band), inside `fix.detection.mask`, inside `fit_prism(fix.points, table_z)` dilated 1 cm, and not labelled another instance, `ROBOT` or `CONFLICT`.
- Neighbours = snapshot instances not ≥ 50 % inside E, passing B2's instance tests.
- Produces: `GraspEpisode.shrink(snap)`, effective only when `snapshot_usable` is None. It REPLACES `keys` (never mutates in place) without the keys the snapshot labels as another instance. Keys that are invisible, robot or conflict stay in E (the fingers sit next to the target in the corridor). It never adds keys.
- Runtime, `enabled`:
  - a freeze refusal returns `{"ok": False, "error": "target voxels unknown: …"}` before any command (jaws included);
  - E is armed from the approach to the lift's settle at q_pre, and during the A3 exit only; afterwards held-object voxels are unknown.
- Runtime, `enabled: false`: today's column/cylinder path, unchanged.

- [ ] **Step 1: Write the failing tests**

```python
def test_freeze_takes_exactly_the_visible_target_voxels(): ...
@pytest.mark.parametrize("break_it, reason", [
    ("snapshot 2.3 s old", "target voxels unknown: labels 2.3 s old"),
    ("fix mask None",      "target voxels unknown: no SAM mask"),
    ("frame K differs",    "target voxels unknown: localize frame is not from the map camera"),
    ("frame 1.6 s old",    "target voxels unknown: localize frame 1.6 s old"),
    ("thumbnail diff 7",   "target voxels unknown: scene changed since the map frame"),
])
def test_freeze_refuses_with_the_reason(break_it, reason): ...
def test_neighbour_rules(): ...   # >= 50 % in E -> target; 49 points / 9 mm / workspace reject -> not known
def test_e_only_shrinks(): ...    # other-instance label removes; new target keys never added; stale snapshot no-op;
                                  # robot/conflict/invisible keys stay
@needs_pin
def test_neighbour_moved_into_target_keys_is_not_exempt(rebot):         # Review Focus 1
    episode.shrink(snapshot_where_neighbour_owns(key)); assert "clearance" in h.vet_pose(q_near(key))
@needs_pin
def test_new_voxels_during_the_episode_are_unknown(rebot): ...           # Review Focus 1
# rig_object_aware: build_runtime(demo_cfg) with grasp.object_aware.enabled = True and
# make_map(GridBridge([TABLE, box at the mock red cube]), required=False) on the harness and watcher
@needs_pin
def test_object_aware_grasp_refuses_before_any_command(rig_object_aware):
    res = rt.skill_grasp_object("red cube")                                # stale snapshot on the map
    assert res["error"].startswith("target voxels unknown: labels") and len(arm.commands) == n0
@needs_pin
def test_object_aware_grasp_holds_e_from_approach_to_lift_then_releases(rig_object_aware): ...
    # spy clearance_excluding kwargs: exclude_keys == E on approach/descent/lift; none after; no _target_column call
```

- [ ] **Step 2: Run, expect FAIL.** Command: `$T tests/test_object_aware_grasp.py tests/test_approach_target_column.py`.
- [ ] **Step 3: Implement** the interfaces.
- [ ] **Step 4: Run, expect PASS.** Run Step 2 plus `tests/test_object_aware_exemption.py tests/test_air_grasp_exit.py tests/test_b36_pick_recovery.py tests/test_pregrasp_clears_object_top.py tests/test_wrist_narration.py`.
- [ ] **Step 5: Mutation checks.**
  - `shrink` keeps keys labelled as another instance: killed by `test_neighbour_moved…`.
  - `shrink` adds new target keys: killed by `test_e_only_shrinks`.
  - Silent fallback to `_target_column` on a freeze failure: killed by `test_object_aware_grasp_refuses…`.
  - Robot keys admitted into E: killed by `test_freeze_takes_exactly…`.
- [ ] **Step 6: Commit** `feat(grasp): freeze the target's voxels per grasp episode; refuse when labels are stale`

### Task B6: Corridor margin for fingers/jaws vs known neighbours (spec §4.4; Q1, Q2)

**Files:**
- Modify: `src/cascade/grasping/gripper_clearance.py`:
  - `ApproachClearance.box_clearance(q, jaw_gap, box: Box, links=("gripper_left", "gripper_right")) -> float` (SAT lower bound via `_box_clearance`, `:307`);
  - `box_box_clearance(a: Box, b: Box) -> float`.
- Modify: `src/cascade/safety/tool_volume.py` (record each part's TCP-frame box at build time; `ToolVolume.part_box(name, T_tcp) -> gripper_clearance.Box`)
- Modify: `src/cascade/safety/object_aware.py` (`CorridorState`), `src/cascade/safety/harness.py` (`corridor()` context; rule inside `_tool_volume_violation`, `:1061-1096`)
- Modify: `src/cascade/skills/runtime.py`: wrap the selection-time descent vets in `_vet` (`:3096-3115`) and the executed descent → close → lift (`:3326-3448`) in `harness.corridor(...)`.
- Modify: `docs/ARCHITECTURE.md:640-652` (the corridor exception), `docs/NVBLOX.md:269-275` (inflate 2 cm vs a corridor margin below 2 cm: frozen prisms only, contact monitor otherwise)
- Test: `tests/test_corridor_margin.py` (new)

**Interfaces:**
- Produces: `CorridorState(q_pre, margin_m, neighbours, jaw_gap, prism_dilate_m)`. `SafetyHarness.corridor(q_pre, neighbours, jaw_gap)` is a no-op unless inside an armed `object_aware` scope AND `cfg.corridor_margin_m < limits.min_clearance_m`. The shipped 0.03 makes the whole rule dormant.
- Rule, for `q_next` ≠ `q_pre` (atol 1e-9; the pregrasp pose itself keeps 3 cm), in `approve`, `vet_step` and `vet_pose`:
  1. Fingers/jaws parts only: voxels inside a neighbour prism dilated `prism_dilate_m` are excluded via `exclude_where`. Every other voxel still counts at 3 cm.
  2. `min(box_clearance(q, jaw_gap, prism.as_box()), box_box_clearance(tool_volume.part_box("jaws", fk(q)), prism.as_box())) >= margin_m` for each neighbour. Otherwise refuse `f"fingers clearance {d:.3f} m below corridor margin {m:.3f} m to known neighbour {label!r} (object-aware corridor)"`.
  3. Camera, housing and TCP/link points are unchanged.

- [ ] **Step 1: Write the failing tests** (rebot + GridBridge + a frozen episode; `corridor_margin_m=0.01` is set by the test on the config object, never in `configs/`)

```python
@needs_pin
def test_shipped_margin_makes_the_corridor_dormant(rebot):          # Q1: PR-B changes no margin
    assert reasons_with_corridor(rebot, cfg_shipped) == reasons_without_corridor(rebot)
@needs_pin
def test_neighbour_15mm_across_the_jaw_axis_is_admitted_in_the_corridor(rebot): ...   # margin 0.01
@needs_pin
def test_corridor_margin_never_applies_outside_the_corridor(rebot):  # Review Focus 5, margin 0.01
    assert refused(transport_step_near(prism))                      # no corridor context
    assert refused(step_onto(q_pre))                                # the pregrasp pose keeps 3 cm (Q2)
    assert refused(step_after_lift())                               # corridor exited
    assert refused(camera_sample_2cm_from(prism)) and refused(tcp_2cm_from(prism))
    assert refused(unexplained_voxel_2cm_from_finger())             # outside every prism + 1 cm
@needs_pin
def test_a_neighbour_outside_its_frozen_prism_is_unknown(rebot): ...     # moved: 3 cm again
def test_finger_hull_distance_is_a_lower_bound(): ...   # box_clearance <= dense-sampled true distance
```

- [ ] **Step 2: Run, expect FAIL.** Command: `$T tests/test_corridor_margin.py`.
- [ ] **Step 3: Implement** the interfaces.
- [ ] **Step 4: Run, expect PASS.** Run Step 2 plus `tests/test_object_aware_exemption.py tests/test_object_aware_grasp.py tests/test_harness_tool_volume.py tests/test_angled_grasp_clearance.py`.
- [ ] **Step 5: Mutation checks.**
  - `<` → `<=` in the dormancy guard: killed by `test_shipped_margin…`.
  - Drop the `q_next == q_pre` exception: killed by `…outside_the_corridor` (pregrasp case).
  - Apply the exclusion to all parts: killed by the camera case.
  - Drop the dilated-prism test (exclude all voxels): killed by the unexplained-voxel case.
- [ ] **Step 6: Commit** `feat(safety): dormant corridor margin for fingers/jaws vs known-neighbour prisms`

### Task B7: Reconciliation report and labels in clearance snapshots (spec §4.7)

**Files:**
- Modify: `src/cascade/perception/voxel_labels.py` (`reconcile`)
- Modify: `src/cascade/perception/world.py` (compute per snapshot; store `occupancy.reconcile_report`)
- Modify: `src/cascade/perception/occupancy.py:1289-1319` (`report()` adds `"reconcile"` only when `reconcile_report` is set)
- Modify: `src/cascade/safety/harness.py` (A4 reports gain `labels` and `reconcile_counts`)
- Test: `tests/test_reconcile_report.py` (new)

**Interfaces:**
- Produces: `reconcile(snap, inputs, beliefs, *, robot_points, now, table_z, table_band_m) -> dict` with keys:
  - `map_without_world_model`: 26-connected clusters of ≥ 20 `UNKNOWN` voxels above the table band (`scipy.ndimage.label` if importable, else `"clusters unavailable: scipy missing"`);
  - `world_model_without_map`: beliefs seen ≤ 3 s ago whose `fit_prism(points)` holds < 5 occupied voxels;
  - `robot_residue_candidates`: unknown voxels within 5 cm of the robot points or at z ≥ 0.20 m;
  - `instances`: `[{label, center_xy, half_xy, yaw, z_lo, z_hi, n_points, box_age_s}]`, which the Task B8 capture reads.
- Produces: report `labels`: `"target"` / `<neighbour label>` / `"robot"` / `"unknown"` per voxel row when the newest snapshot is usable, else `labels: None` + `labels_reason`.

- [ ] **Step 1: Failing tests.**
  - An unnamed 20-voxel cluster is listed; 19 is not.
  - A belief seen 1 s ago with 4 voxels in its prism is listed, with 5 it is not, and seen 3.1 s ago it is not; removing the prop makes it appear and then clear.
  - High-z and near-robot unknown voxels are listed.
  - `report()` has no `"reconcile"` key without a snapshot.
  - Clearance reports carry labels when fresh, and `labels_reason` when stale.
- [ ] **Step 2: Run, expect FAIL.** Command: `$T tests/test_reconcile_report.py`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS.** Run Step 2 plus `tests/test_wrist_narration.py tests/test_clearance_snapshots.py tests/test_watcher_labels.py`.
- [ ] **Step 5: Commit** `feat(perception): world-model vs map reconciliation report (report-only)`

### Task B8: Fit-error measurement tool and the measured-margin pin (spec §4.6)

**Files:**
- Create: `src/cascade/perception/fit_error.py`, `scripts/measure_fit_error.py`
- Test: `tests/test_fit_error.py`, `tests/test_corridor_margin_measured.py` (new)

**Interfaces:**
- Produces:
  - `outward_error(truth_points, prism) -> float`: the largest distance of a truth point outside the prism; 0 inside.
  - `size_term(caliper_half_extents_m, prism) -> float`: max over axes of `max(caliper − prism half extent, 0)`, xy sorted, height = z_hi − z_lo.
  - `ceil_5mm(x) -> float`.
  - `calib_term(record_path) -> float`: max of `metrics.rmse_m` and every `samples[*].rmse_m` of the committed record `configs/calib/d455f_scene_rebot_rs.handeye.json` (named by `configs/cameras/d455f_scene.yaml:37`). Today that is 0.0035154 m.
  - `required_corridor_margin(e_fit_total) -> float = max(0.010, ceil_5mm(e_fit_total))`.
  - `latest_measurement() -> dict | None`: the newest `docs/evidence/corridor-margin-*/measurement.json`.
- Produces: `scripts/measure_fit_error.py`:
  - `sim --placements 40 --out DIR` (`uv run --extra sim`): 6 shapes, namely a 5 cm cube, an 8 × 10 cm cup cylinder, a 12 × 8 × 6 box, a spool, a doll proxy and a 2 cm flat box; the camera at the record's pose, 1280 × 720, depth stride 2; render masks; production `mask_to_points_cam` → `fit_prism`; truth sampled at 2 mm; `e_sim = max`.
  - `capture --state-url http://127.0.0.1:8090/state --prop NAME --caliper-mm L,W,H --placement K --out DIR`: operator-gated (TTY prompt "place NAME, placement K, hands clear, press Enter"; refuses without a TTY). Read-only GETs of `/state` → `backends.occupancy_state.reconcile.instances`, median of 10 polls. Never calls a motion or MCP tool. 5 props × 4 placements.
  - `analyze --dir DIR --out docs/evidence/corridor-margin-<date>/measurement.json`, schema `cascade.corridor_margin_measurement/1`: `e_sim`, `e_size`, `e_calib`, record path + sha256, `e_fit_total`, `required_margin_m`.

- [ ] **Step 1: Write the failing tests**

```python
def test_outward_error(): assert outward_error(box_points_with_7mm_bulge, prism) == pytest.approx(.007, abs=1e-6)
def test_ceil_5mm_and_floor(): assert ceil_5mm(.0101) == .015 and required_corridor_margin(.004) == .010
def test_calib_term_reads_the_committed_record(): assert calib_term(RECORD) == pytest.approx(.0035154, abs=1e-6)
def test_capture_refuses_without_an_operator_tty(): ...         # exit code 2, no HTTP request made

# tests/test_corridor_margin_measured.py
def test_every_profile_keeps_the_measured_corridor_margin():   # all configs/arms/*.yaml
    m = latest_measurement()
    need = min_clearance if m is None else required_corridor_margin(m["e_fit_total"])   # Q1: none yet -> 0.03
    assert all(effective_corridor_margin(p) >= need for p in profiles())
def test_code_default_is_3cm(): assert ObjectAwareConfig.from_config(cfg_without_key).corridor_margin_m == .03
def test_red_against_a_lowered_config(): ...     # synthetic profile 0.02, no measurement -> fails the check
def test_red_against_a_larger_measurement(): ... # e_fit_total 0.021 -> need 0.025 > 0.02
```

- [ ] **Step 2: Run, expect FAIL.** Command: `$T tests/test_fit_error.py tests/test_corridor_margin_measured.py`.
- [ ] **Step 3: Implement.** The sim command is not run by tests (MuJoCo is absent in the rig venv). The script's import guard prints the `uv run --extra sim` command.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat(tools): fit-error measurement (sim, operator-gated capture, analysis) and the measured-margin pin`

### PR-B exit (controller, not a task)

1. Run a full-suite failing-ID diff against the PR-A merge base, run ruff, confirm every mutation was KILLED, and check that the `test_wrist_narration` golden is unchanged.
2. Deploy notes:
   - `rebot_rs` builds label snapshots (report-only, `labels: true`); `object_aware.enabled` stays false;
   - the corridor margin is 0.03 (dormant);
   - no weights and no env changes.
3. Live acceptance, operator-gated after merge:
   - lever-3 review;
   - B3 fail-closed and B4 reconciliation (no motion);
   - the 20-minute capture (`measure_fit_error.py capture`);
   - only then B1/B2 with `enabled: true` on `rebot_rs`, and a separate commit for any margin from `measurement.json`.
4. AGENTS.md paste text, Motion safety path, AGENTS.md:61 (spec §10, sentences 4-7 verbatim, appended after PR-A's):
   > With `grasp.object_aware.enabled`, map voxels are labelled from the scene camera's same-frame SAM masks (depth band + fitted prism, F1-F4 freshness), frozen per grasp episode; exactly the target's voxels are exempt (the descent cylinder governs only the table rules), and fingers/jaws may use `grasp.object_aware.corridor_margin_m` against known-neighbour prisms between the pregrasp and the lift's return to it. That value is pinned to a committed measurement (`test_corridor_margin_measured.py`), never below 1.0 cm; arm body, wrist camera, housing, payload and unknown voxels keep `min_clearance_m` everywhere. Stale or doubtful labels are "unknown"; never fall back to the legacy column silently.

---

## Spec coverage

| Spec | Task |
|---|---|
| §3.1 escape + route end + reason text | A1 |
| §3.2 hover from map top (cap 2 cm, voxel/2) | A2 |
| §3.2 exit withdrawal, no exit with a possible payload | A3 |
| §3.3 snapshots, bounds, drain, error isolation | A4 |
| §3.4 / §8 measured parts, 5 mm outward, missing face = today | A5 |
| §6.3 exact replay | A6 |
| §4.1-4.2 same-frame labels, F1-F4, conflicts, robot report-only (Q4), CUDA masks | B1, B2, B3 |
| §4.2 grasp episode (freeze, target identity, neighbours, shrink, expiry) | B4, B5 |
| §4.3 exact exemption, cylinder for table rules only, opt-in | B4, B5 |
| §4.4 corridor (phase Q2, parts, known neighbours, SAT, value Q1) | B6 |
| §4.5 fallback to unknown, no silent legacy fallback | B5 |
| §4.6 fit-error measurement + pinned test | B8 |
| §4.7 reconciliation | B7 |
| §5 failure table | A1-A4, B2-B6 tests |
| §7 live acceptance | PR exits (operator-gated, not run here) |
| §9 pins and docs | Global Constraints; A1, A4, A5, B4-B6 |
| §10 AGENTS.md text | PR exits |
