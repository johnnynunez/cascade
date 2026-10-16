# Natural-language spatial commands

**Next action:** run the mock-stack tests for the four tools and the rest pose (about 1 min, no
robot):

```bash
uv run --extra dev --extra kinematics pytest tests/test_relative_motion.py tests/test_spatial_commands.py \
  tests/test_spatial_commands_registration.py tests/test_move_to_rest.py \
  tests/test_move_to_rest_registration.py -q
```

Status: tested on the mock stack (MockArm, mock camera, the RS kinematics) only. **Not yet verified
on the real robot**; do the [hardware checks](#not-yet-verified-on-the-real-robot) before relying
on it with visitors.

Vague, open-ended instructions that are not pick-and-place ("lift it up", "a bit more", "bring it
toward me", "hold it above the green spool", "go back", "tilt it down"), also while holding a
grasped object, map to four MCP tools:

| tool | what it does |
|---|---|
| `move_relative` | Short straight move of the gripper and its load: a named direction or a free vector, in the robot, screen or gripper frame; `direction=again` repeats the last one |
| `rotate_gripper` | Bounded orientation nudge about the gripper tip: roll (clockwise / counterclockwise) or tilt toward a direction, at most 30 degrees, never turning the wrist camera under |
| `undo_move` | "Go back": retrace the last 1-5 moves of these tools, through their own recorded waypoints, re-vetted |
| `move_near_object` | Position relative to a named object or the table: `above`, `just_above`, `next_to`, `closer` |

Code: `src/cascade/skills/relative_motion.py` (interpretation: frames, sizes, rotations, history,
wrist-camera check) and `src/cascade/skills/spatial_moves.py` (plan, vet, execute, record).

## Phrase to call

The tool descriptions carry this mapping so a mid-size model (Qwen3.8-27B on the reference rig)
does not invent numbers or frames.

| the person says | call |
|---|---|
| "lift it up" | `move_relative(direction="up")` (5 cm) |
| "a bit to the left" | `move_relative(direction="left", amount="small")` (2 cm, the ROBOT's left) |
| "move it right on the screen" / "to my right" | `move_relative(direction="right", frame="viewer")` |
| "bring it toward me" | `move_relative(direction="toward_viewer")` |
| "a bit more", "again", "keep going" | `move_relative(direction="again", amount="small")` |
| "3 cm left and 2 cm up" | `move_relative(vector=[0, 0.03, 0.02])` |
| "push it in a little" (along the jaws) | `move_relative(direction="in", amount="small")` |
| "go back", "undo that" | `undo_move()` |
| "hold it above the green spool" | `move_near_object(label="green spool", relation="above")` |
| "lower it until it's just above the table" | `move_near_object(label="table", relation="just_above")` |
| "hold it next to the cup, on the left" | `move_near_object(label="cup", relation="next_to", side="left")` |
| "bring it closer to the box" | `move_near_object(label="box", relation="closer")` |
| "rotate it a bit" | `rotate_gripper(direction="clockwise", amount="small")` |
| "tilt it down a little" | `rotate_gripper(direction="down", amount="small")` |
| "turn it upside down" | refused: the wrist camera must stay on top |
| "go to the initial pose" / "ready pose" / "pose inicial" | `move_home()` (raised, motors on) |
| "go home" / "pose 0" / "rest" / "park" / "fold it" / "vuelve a home" | `move_to_rest()` (folded, then motors off) -- see [below](#initial-pose-vs-rest-pose-home-pose-0-and-powering-off) |

## Sizes

| `amount` | translation | rotation | typical words |
|---|---|---|---|
| `tiny` | 1 cm | 5 deg | "a hair", "a tiny bit" |
| `small` | 2 cm | 10 deg | "a bit", "slightly", "a little" |
| `medium` (default) | 5 cm | 15 deg | no size given |
| `large` | 10 cm | 30 deg | "a lot", "much" |

`distance_m` / `angle_deg` give an exact number and win over `amount` (the result says the word was
ignored). Per call: 0.005-0.15 m, 1-30 degrees. Values outside are **refused, not clamped** (a
silent 20 -> 15 cm change is a different motion), and the refusal tells the agent to split the
move.

## Frames, and why the default is the robot

| frame | axes | free `vector` |
|---|---|---|
| `base` (default) | +x forward (away from the robot base), +y the robot's left, +z up | `[x, y, z]` |
| `viewer` | the fixed scene camera the demo screen shows: right = image right projected on the table, away = up x right ("up on the screen"), up = vertical | `[right, away, up]` |
| `tool` | along the gripper: `in` = where the jaws point, `out` = back off | `[approach, jaw axis, third]` |

`up`/`down` are world vertical in every frame; `toward_viewer` / `away_from_viewer` imply the
viewer frame and `in` / `out` the tool frame; a contradictory explicit frame (`in` with
`frame=viewer`) is refused.

**Bare left/right/forward/back are in the robot base frame.** On the reference rig the two frames
disagree by 90 degrees (`configs/calib/d455f_scene_rebot_rs.handeye.json`, camera on the robot's
left looking back across the table):

| robot direction | as seen on the screen |
|---|---|
| right (-y) | away from the camera (up on the screen) |
| left (+y) | toward the camera (down on the screen) |
| forward (+x) | left on the screen |
| back (-x) | right on the screen |

The robot frame is the default because it is always available (the viewer frame depends on a
hand-eye calibration that the extrinsic drift monitor can invalidate mid-session), it is the frame
of every coordinate the agent sees (`world_state`, `localize_object`, `place_at`) and of the reflex
grammar, and forward/back are only unambiguous for the robot. The descriptions tell the agent to
use `frame="viewer"` when the person refers to themselves or the screen.

Every result states the interpretation (`interpretation`, `frame`, `camera`,
`displacement_base_m`, `as_seen_on_screen`) so the person can correct it. An uncalibrated scene
camera refuses viewer-frame requests ("viewer frame unavailable ... use frame='base'"). The viewer
camera is `relative_motion.viewer_camera`, else the first fixed (eye-to-hand) camera; a wrist
camera is never the viewer.

## Safety invariants

Each is a test in the files above (the important ones mutation-checked).

1. **The harness is never bypassed:** every motion is a `SafeArm` call (`carry_attachment.move` ->
   `move_joints`, or `move_cartesian`), so every streamed tick is `approve()`d.
2. **A refused move leaves the arm where it is:** the whole route is vetted first with the
   harness's static API (`safety.trajectory.vet_segment` -> `SafetyHarness.vet_step`): workspace,
   table clearance, keep-outs, link-table check, occupancy, joint limits, neighbour arms. A refusal
   returns `ok: false`, `refused_by`, `arm_moved: false` and a reason. Offers come from vetting,
   never motion: a bisection reports `feasible_distance_m` / `feasible_angle_deg` (1/64
   resolution); only a new call moves.
3. **Caps and absurd input:** 0.15 m per relative move, 30 degrees per rotation, 0.5 m
   (`relative_motion.max_object_move_m`) per object-relative move. NaN, infinity, negative, out of
   cap, bool, string, wrong vector length and unknown or ambiguous words are refused with zero
   commands ("go back" as a direction points to `undo_move`).
4. **Motion lock:** all four are in `_MOTION_SKILLS`, so the OpenShell `motion lock` denies them;
   `emergency_stop` and `halt_motion` are never deniable. With `cartesian_relative_moves` there is
   no silent cartesian -> joint fallback.
5. **History:** a refused move does not change it; an undo is vetted like any move and a refused
   undo keeps its record.

Holding an object: the tools never open, close or re-grip (a static test pins it); the held flag is
reconciled read-only with the jaws and every result reports `holding`; speed is halved
(`relative_motion.held_speed_scale`); the object's **bottom** keeps the harness's TCP table
clearance, using the grasp's measured support offset or `relative_motion.held_drop_default_m` (6 cm);
a load already under that plane may only climb; `move_near_object` clears the target's **observed**
top (never an OBB extent).

## The wrist camera stays on top

The reBot's wrist camera sits on top of the gripper; rolled under, it hits the table. Every planned
pose (not only the end pose) must keep `(R_tcp @ relative_motion.tool_up_axis).z >=
relative_motion.tool_up_min_z` (defaults `[0, 0, 1]` and `0.0`; on the reBot camera-up is TCP +z).
"Upside down", "flip" and "turn over" are refused by name; from a pose already below the limit, no
motion may lower the camera further.

### Its volume: it sticks out 6-8 cm

The harness checks only the TCP and joint origins, so every planned pose also checks the camera as
an oriented box in the TCP frame: back `behind_tip_m` (0.14 m) from the fingertip plane,
±`half_width_m` (0.075 m) across the jaws, 0 to `protrusion_m` (0.08 m) up. Figures come from
tape-measure photos (2026-10-11, ±1-2 cm; a padded D455F, 124 x 29 x 26 mm) rounded UP; re-measure
if the padding or mount changes and set them in the arm profile's `overrides:`. At every sample of
the route:

1. **Table:** the box's lowest corner keeps `safety.table_clearance` over `safety.table_z` (a
   camera already under that plane may only climb).
2. **Keep-out zones:** exact box-against-box test against every `safety.keep_out`, no escape rule.
3. **Occupancy map** (opt-in `wrist_camera.check_occupancy: true`): points ≤ 2 cm apart keep
   `safety.min_clearance_m`; turn it on only once a masked depth refresh is known to remove the
   camera, or every move is refused.
4. The harness runs first (its refusals say `refused_by: safety_harness`), so `refused_by:
   wrist_camera` means only the camera is in the way.
5. A misspelled or out-of-range `wrist_camera` key refuses every move until fixed;
   `wrist_camera: null` disables the volume check; a missing key gets the reBot defaults.

Two layers: the harness (PR #318, `safety.wrist_camera_up` in the reBot profiles) refuses a
camera-down pose for EVERY motion in `approve()`, `vet_step()` and `vet_pose()` down to `min_up_z`
(-0.1); these four tools add the stricter `tool_up_min_z`, the wording refusal and the camera
volume (grasps, places and `move_home` do not model the volume).
`tests/test_spatial_commands_registration.py` pins that `relative_motion.tool_up_axis` names the
same axis and is at least as strict; an arm without a gripper camera sets `tool_up_axis: null`.

## "Again" and "go back"

1. `direction="again"` re-resolves the last request of that tool (words and frame, not the old
   vector), optionally with a new size.
2. `undo_move(steps=1..5)` retraces recorded moves newest first through their measured waypoints
   in reverse (a carry-shaped route climbs back before travelling).
3. It runs only from where the recorded move ended (1.5 cm / 5 degrees); otherwise the history is
   stale, cleared, and the undo refused. History is per arm (`relative_motion.history_len`).

## Object-relative positioning

`move_near_object` localizes the target fresh (the grasps' `_localize`; refuses an unseen label and
the object in the gripper) and keeps the current gripper orientation (if unreachable, the refusal
suggests pointing the jaws down first with `rotate_gripper`).

1. `above` / `just_above`: over the target's centre, load bottom `clearance_m` (5 cm / 1.5 cm) over
   its observed top; `label="table"` uses the current XY and the table plane.
2. `next_to`: on a named side (robot or screen frame) or the nearest side, at footprint (95th
   percentile of its cloud) + jaw/held radius + gap.
3. `closer`: horizontal approach, stops at that standoff.
4. Route: climb first if the goal is higher, travel at the higher height, descend last; vetted end
   to end before the first command.
5. Withheld on rigs without measured depth heights (`capabilities.TOOL_REQUIREMENTS`).

## Initial pose vs rest pose ("home", "pose 0"), and powering off

| the person says | pose | tool | motors after |
|---|---|---|---|
| "initial pose", "ready pose", "start pose" ("pose inicial", "vuelve al inicio") | `home_q` = `[0, 1.2, 1.2, 0, 0, 0]`: raised, gripper forward, ready for work | `move_home` | on |
| "home", "go home", "pose 0", "zero", "rest", "park", "fold it", "so I can turn it off" ("vuelve a home", "a casa", "pose 0 / pose cero", "reposo", "aparca", "pliega el brazo", "vuelve al origen") | `park_q` = all joints 0: folded back over the base, links on their stops, wrist camera on top | `move_to_rest` | **off** (unless `keep_torque=true`) |

Tool descriptions and the reflex grammar (`agent/reflex.py`, intents `rest` and `initial_pose`)
route this way (`tests/test_move_to_rest_registration.py`). `move_home` keeps its name and
behaviour (`reset_scene`, re-home, calibration and tests use it). `move_to_rest`
(`src/cascade/skills/rest_pose.py`):

1. **Refuses before any command** while the e-stop is latched or the gripper holds (or may hold)
   something: the person is asked to put it down first.
2. **Folds through SafeArm** on the rig-proven route (`scripts/home_rebot_rs.py`): to the initial
   pose first unless already there, then 4 joint-space steps each **measured** against its target
   (0.08 rad); only the last relaxes the joint margin to 0. The whole route is vetted first; the
   occupancy map is never bypassed (a refused fold stays put, motors on).
3. **Releases torque only on a measured rest** (every joint within 0.05 rad of `park_q`): the
   gripper closes, the LazyArm returns to standby, and the result says `{"at": "rest", "torque":
   "released", "safe_to_power_off": true}`. Only then may the agent tell the person to switch the
   robot off; any other outcome keeps the motors ON (`"at": "not_rest"`): a raised arm without
   torque falls.
4. **The next motion re-energizes by itself** from the rest pose (on the RS, opening clears latched
   faults with a torque-off stop frame). Rest requested again in standby moves and reads nothing
   (`"at": "standby"`).
5. Profiles without `park_q` (all but the reBot RS and the cuMotion Isaac twins) go to the initial
   pose, motors on, `safe_to_power_off: false`. A directly opened backend or a simulator keeps its
   connection at rest and says so.

## Configuration

`configs/demo.yaml`, overridable per arm through `overrides:`:

```yaml
relative_motion:
  duration_s: 1.0
  held_speed_scale: 0.5
  held_drop_default_m: 0.06
  history_len: 10
  viewer_camera: null
  tool_up_axis: [0.0, 0.0, 1.0]
  tool_up_min_z: 0.0
  wrist_camera:
    protrusion_m: 0.08
    behind_tip_m: 0.14
    half_width_m: 0.075
    check_occupancy: false
  max_object_move_m: 0.50
```

## Not yet verified on the real robot

1. Watch each tool at low speed on the reBot with the real scene camera: "right on the screen" must
   move right on the demo screen (the calibration says robot -x) and `as_seen_on_screen` must match.
2. Confirm the wrist-camera axis (TCP +z = camera up) and that joint 6 rolls never pass the limit,
   including from top-down grasp poses where the camera axis is exactly horizontal.
3. Tighten the camera box with a tape measurement (padded length, fingertip to back of padding);
   measure `held_drop_default_m` against real props (with `grasp.place_support_clearance_m` unset
   the 6 cm default applies); check the 1.5 cm / 5 degree undo tolerance against real tracking.
4. 5-DoF arms (SO-101): translations keep the full orientation, which a 5-DoF wrist often cannot
   hold off its azimuth plane; expect kinematic refusals.
5. `move_to_rest`: watch the first fold from the initial pose at the profile's speed cap with a hand
   near the arm; check that the occupancy map does not refuse the last step on the arm's own
   unmasked parts near the base, that the stops stay within 0.05 rad, that the gripper closes, that
   the dashboard shows standby, and that the next motion re-opens from rest without a jolt.
