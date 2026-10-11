# Natural-language spatial commands

The arm can be steered by vague, open-ended instructions that are not
pick-and-place: "lift it up", "a bit more", "move it to the right", "bring it
toward me", "hold it above the green spool", "go back", "lower it until it's
just above the table", "rotate the gripper a little", "tilt it down" -- also
while it holds a grasped object. Four MCP tools serve them:

| tool | what it does |
|---|---|
| `move_relative` | Short straight move of the gripper and its load: a named direction or a free vector, in the robot, screen or gripper frame; `direction=again` repeats the last one |
| `rotate_gripper` | Bounded orientation nudge about the gripper tip: roll (clockwise / counterclockwise) or tilt toward a direction, at most 30 degrees, never turning the wrist camera under |
| `undo_move` | "Go back": retrace the last 1-5 moves of these tools, through their own recorded waypoints, re-vetted |
| `move_near_object` | Position relative to a named object or the table: `above`, `just_above`, `next_to`, `closer` |

Code: `src/cascade/skills/relative_motion.py` (pure interpretation: frames,
sizes, rotations, history, the wrist-camera check) and
`src/cascade/skills/spatial_moves.py` (plan, vet, execute, record). Tests:
`tests/test_relative_motion.py`, `tests/test_spatial_commands.py` (real
`build_runtime` on the mock stack) and
`tests/test_spatial_commands_registration.py`.

## Phrase to call

The tool descriptions carry this mapping so a mid-size model (Qwen3.8-27B on
the reference rig) does not have to invent numbers or frames.

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

`distance_m` / `angle_deg` give an exact number and win over `amount` (the
result says the word was ignored). Per call: 0.005-0.15 m, 1-30 degrees.
Values outside are **refused, not clamped**: a silent 20 -> 15 cm change is a
different motion from the one asked for, and the refusal tells the agent to
split the move. Non-finite numbers, negative values, booleans and strings are
refused before anything else happens.

## Frames, and why the default is the robot

| frame | axes | free `vector` |
|---|---|---|
| `base` (default) | +x forward (away from the robot base), +y the robot's left, +z up | `[x, y, z]` |
| `viewer` | the fixed scene camera the demo screen shows: right = image right projected on the table, away = up x right ("up on the screen"), up = vertical | `[right, away, up]` |
| `tool` | along the gripper: `in` = where the jaws point, `out` = back off | `[approach, jaw axis, third]` |

`up`/`down` are world vertical in every frame. `toward_viewer` /
`away_from_viewer` imply the viewer frame; `in` / `out` imply the tool frame.
A contradictory explicit frame (`in` with `frame=viewer`) is refused, never
re-read.

**Bare left/right/forward/back are in the robot base frame.** Ambiguity is a
safety issue here, because on the reference rig the two frames disagree by 90
degrees. From the calibrated scene camera record
(`configs/calib/d455f_scene_rebot_rs.handeye.json`, the camera sits on the
robot's left looking back across the table):

| robot direction | as seen on the screen |
|---|---|
| right (-y) | away from the camera (up on the screen) |
| left (+y) | toward the camera (down on the screen) |
| forward (+x) | left on the screen |
| back (-x) | right on the screen |

The robot frame is the default because (1) it is always available: the viewer
frame depends on a hand-eye calibration that can be missing, rejected or
invalidated mid-session by the extrinsic drift monitor, and a default that can
disappear would make the same sentence work at 10:00 and fail at 10:05; (2) it
is the frame of every coordinate the agent already sees (`world_state`,
`localize_object`, `place_at`) and of the reflex grammar ("move a bit left"),
so one word never means two things in two tiers; (3) forward/back are only
unambiguous for the robot. The descriptions tell the agent to switch to
`frame="viewer"` whenever the person refers to themselves or the screen.

**Every result states the interpretation**: `interpretation` (for example
"2.0 cm to the robot's right (-y) -- the robot's own directions"), `frame`,
`camera`, `displacement_base_m`, and `as_seen_on_screen` ("away from the
camera (up on the screen)"), so the agent can tell the person what happened
in their terms and the person can correct it. An uncalibrated scene camera
refuses viewer-frame requests ("viewer frame unavailable ... use
frame='base'") and leaves robot-frame requests working.

The viewer camera is `relative_motion.viewer_camera` (a stream name), else
the first fixed (eye-to-hand) camera of the rig. A wrist camera is never the
viewer: its "right" turns with every move.

## Safety invariants

Each of these is a test in the files above; the important ones were
mutation-checked (the implementation broken on purpose, a test seen failing,
the code restored).

1. **The harness is never bypassed.** Every motion is a `SafeArm` call
   (`carry_attachment.move` -> `move_joints`, or `move_cartesian`), so every
   streamed tick is `approve()`d like any other skill's.
2. **A refused move leaves the arm where it is.** The whole route is vetted
   before the first command with the harness's own static API
   (`safety.trajectory.vet_segment` -> `SafetyHarness.vet_step`; `vet_step`
   over the dense path for straight lines): workspace box, table clearance,
   keep-outs, link-table check, occupancy-map clearance, joint limits,
   neighbour arms. Without that preflight a doomed nudge would start and be
   cut off mid-way. A refusal returns `ok: false`, `refused_by`,
   `arm_moved: false` and a reason the agent can relay; vetting records no
   harness violation.
3. **Offers come from vetting, never from motion.** When the full move is
   refused, a bisection over the fraction of the displacement (or angle)
   re-plans and re-vets candidates without commanding anything and reports
   `feasible_distance_m` / `feasible_angle_deg` (1/64 resolution). Only a new
   call moves.
4. **Per-call caps**: 0.15 m per relative move, 30 degrees per rotation,
   0.5 m (`relative_motion.max_object_move_m`) per object-relative move.
5. **Absurd input is refused** with zero commands (NaN, infinity, negative,
   out of cap, bool, string, wrong vector length, unknown or ambiguous words
   -- "go back" as a *direction* is refused with a pointer to `undo_move`).
6. **Every new motion tool is a motion skill**: in `_MOTION_SKILLS`, so the
   OpenShell `motion lock` (which reads that set) denies all four;
   `emergency_stop` and `halt_motion` are never deniable.
7. **No silent cartesian -> joint fallback**: with
   `cartesian_relative_moves`, an unplannable straight line, a
   planner-owned arm or an armed carry attachment is a refusal.
8. **Table clearance and occupancy** are the harness's, enforced in the
   preflight; a held object and the wrist camera's volume add their own
   checks (below), which run after the harness's.
9. **A refused move does not change the history**; an undo is vetted like
   any move and a refused undo keeps its record.

## Holding an object

- The tools never open, close or re-grip (a static test pins that neither
  module contains a gripper call). The held flag is reconciled read-only
  with the jaws first, and every result reports `holding`.
- Speed is halved (`relative_motion.held_speed_scale`).
- The object hangs below the TCP, so its **bottom** must keep the same table
  clearance the harness keeps for the TCP. The drop is the grasp's measured
  support offset when known, else `relative_motion.held_drop_default_m`
  (6 cm, conservative). From a pose whose load already sits under that plane,
  only climbing moves are allowed (the harness's escape rule).
- `move_near_object` keeps the held object's bottom `clearance_m` over the
  target's **observed** top (never an OBB extent).

## The wrist camera stays on top

The reBot's wrist camera is mounted on top of the gripper; rolled under, it
faces the table and hits it. On the reBot the camera-up axis is TCP +z
(at home the TCP frame equals the base frame: x = approach, z = up; joint 6
rolls about the approach). Every pose these tools plan -- not only the end
pose, because a roll whose ends are fine can pass through upside down -- must
keep `(R_tcp @ relative_motion.tool_up_axis).z >= relative_motion.tool_up_min_z`
(defaults `[0, 0, 1]` and `0.0`). "Upside down", "flip" and "turn over" are
refused by name, an over-cap request such as 180 degrees is answered with the
camera reason, and the tool description tells the model never to offer an
upside-down roll. From a pose that is already below the limit, no motion may
lower the camera further (moves that raise it, or keep it, are allowed).

### Its volume: it sticks out 6-8 cm

Facing up is not enough: the camera **sticks out 6-8 cm above the gripper**
(user, 2026-10-11), and the harness only checks the TCP and the joint origins.
Rolled 80 degrees, the camera-up axis still points up, but the camera's side
hangs about 7 cm below the TCP; a descent the harness accepts for the TCP would
put that side into the table. So every planned pose also checks the camera as
an oriented box in the TCP frame:

- along the gripper: from the fingertip plane back `behind_tip_m` (0.14 m);
- across the jaws: +-`half_width_m` (0.075 m);
- up along the camera-up axis: 0 to `protrusion_m` (0.08 m, the upper end of
  the user's 6-8 cm).

Measured from tape-measure photos of the rig (2026-10-11, +-1-2 cm): the
camera is a D455F (124 x 29 x 26 mm) padded with bubble wrap and tape, about
15 cm long across the jaws with the padding; its front sits about 8 cm behind
the fingertips and the padding reaches 13-14 cm back; its top is 5-7 cm above
the finger axis. Every figure is rounded UP (a larger box only refuses more).
Re-measure if the padding or the mount changes, and set the numbers in the
arm profile's `overrides:`. Checks, at every sample of the route (a joint-space
arc can dip in the middle while both ends are fine):

1. **Table**: the box's lowest corner keeps `safety.table_clearance` over
   `safety.table_z`, the same clearance the harness keeps for the TCP. Escape
   rule as for the TCP: a camera already under that plane may only climb.
   Pointing straight down, the camera's lowest point is level with the TCP, so
   this never makes a top-down move stricter than the harness already is.
2. **Keep-out zones**: an exact box-against-box test (separating axes) against
   every `safety.keep_out` zone; no escape rule, as for the TCP.
3. **Occupancy map** (opt-in, `wrist_camera.check_occupancy: true`): a grid of
   points over the box at most 2 cm apart keeps `safety.min_clearance_m` from
   observed obstacles. Off by default because the map's robot-body mask is
   built from link radii; if it does not cover the camera, the camera reads as
   an obstacle at zero distance and every move is refused. Turn it on once a
   masked depth refresh is known to remove the camera.

The harness runs first: when it objects, `refused_by` is `safety_harness`;
`wrist_camera` therefore means only the camera is in the way. A misspelled or
out-of-range key under `wrist_camera` refuses every move until fixed (a typo
must not silently keep a default); `wrist_camera: null` disables the volume
check, and a config without the key gets the reBot defaults.

### Two layers

- **The harness** (PR #318, `safety.wrist_camera_up` in the reBot profiles)
  refuses a camera-down pose for EVERY motion -- grasps, places, `move_home`
  and these tools -- in `approve()`, `vet_step()` and `vet_pose()`, down to
  `min_up_z` (-0.1, about 6 degrees below horizontal, so a straight-down dive
  passes). It checks the orientation only.
- **These four tools** add, on top: a stricter orientation limit
  (`tool_up_min_z` 0.0), the refusal of upside-down wording before any
  planning, the camera's volume (table, keep-outs, optional occupancy) and the
  model guidance in the tool descriptions. Grasps, places and `move_home` do
  not model the camera's volume.

`tests/test_spatial_commands_registration.py` pins that, on every arm profile
that declares `safety.wrist_camera_up`, `relative_motion.tool_up_axis` names the
same TCP axis and is at least as strict. An arm without a camera on the
gripper sets `tool_up_axis: null` in its profile `overrides:`.

## "Again" and "go back"

- `direction="again"` re-resolves the last request of that tool (words and
  frame, not the old vector), optionally with a new size: "a bit more" after
  "move it left" is `again` + `amount=small`.
- `undo_move(steps=1..5)` retraces recorded moves newest first, through their
  measured waypoints in reverse (a carry-shaped `move_near_object` route climbs
  back before travelling, never a diagonal across the target). It only runs
  from where the recorded move ended (1.5 cm / 5 degrees): if anything else
  moved the arm since, the history is stale, it is cleared and the undo is
  refused. History is per arm and bounded (`relative_motion.history_len`).

## Object-relative positioning

`move_near_object` localizes the target fresh (the same `_localize` grasps
use; it refuses an unseen label and the object in the gripper) and keeps the
current gripper orientation (a silent re-orientation would swing a held
object; if the position is unreachable that way, the refusal suggests
pointing the jaws down first with `rotate_gripper`).

- `above` / `just_above`: over the target's centre, load bottom `clearance_m`
  (5 cm / 1.5 cm) over its observed top. `label="table"` uses the current XY
  and the table plane.
- `next_to`: on a named side (robot or screen frame) or the side nearest the
  gripper, at footprint (95th percentile of its cloud) + jaw/held radius + gap
  from its centre, with the load still clearing the target's top.
- `closer`: horizontal approach, stops at that standoff.
- Route: climb vertically first if the goal is higher, travel horizontally at
  the higher height, descend vertically last; vetted end to end before the
  first command. The tool is withheld on rigs without measured depth heights
  (`capabilities.TOOL_REQUIREMENTS`): plane-cast depth reads every top as the
  table.

## Initial pose vs rest pose ("home", "pose 0"), and powering off

The reBot has two named poses, and the words for them follow its owner:

| the person says | pose | tool | motors after |
|---|---|---|---|
| "initial pose", "ready pose", "start pose" ("pose inicial", "vuelve al inicio") | `home_q` = `[0, 1.2, 1.2, 0, 0, 0]`: raised, gripper forward, ready for work | `move_home` | on |
| "home", "go home", "pose 0", "zero", "rest", "park", "fold it", "so I can turn it off" ("vuelve a home", "a casa", "pose 0 / pose cero", "reposo", "aparca", "pliega el brazo", "vuelve al origen") | `park_q` = all joints 0: folded back over the base, links on their stops, wrist camera on top | `move_to_rest` | **off** (unless `keep_torque=true`) |

Both the tool descriptions (LLM tier) and the reflex grammar
(`agent/reflex.py`: dashboard chat, offline brain; intents `rest` and
`initial_pose`) route this way, pinned by
`tests/test_move_to_rest_registration.py`. `move_home` keeps its name and
behaviour: `reset_scene`, the grasp pipeline's re-home, calibration and the
tests use it.

`move_to_rest` (`src/cascade/skills/rest_pose.py`, tests
`tests/test_move_to_rest.py`):

1. **Refuses before any command** while the e-stop is latched, and while the
   gripper holds -- or may hold -- something (the shutdown park's retained-load
   markers): the person is asked to put it down first.
2. **Folds through SafeArm** on the route the rig has proven
   (`scripts/home_rebot_rs.py`): up to the initial pose first unless the arm
   is already there (`move_planned`, `move_home`'s route), then home -> rest
   in 4 joint-space steps, each **measured** against its target (0.08 rad).
   Only the last step relaxes the joint margin to 0 (joints 2/3 have their
   lower limit at the rest pose). Every step is vetted by the harness (speed
   cap, table, workspace, occupancy, wrist camera up); the whole route is
   vetted before the first command, so a refused fold never starts. The
   occupancy map is never bypassed (only the shutdown park may do that: it
   would otherwise drop a raised arm; this tool simply stays put, motors on).
3. **Releases torque only on a measured rest**: every joint within 0.05 rad
   of `park_q`. Then the gripper closes (the RS's torque-detected close, as at
   shutdown) and the LazyArm returns to standby -- the backend's disconnect
   cuts torque. The result says `{"at": "rest", "torque": "released",
   "safe_to_power_off": true}`; only then may the agent tell the person to
   switch the robot off. Any other outcome keeps the motors ON and says why
   (`"at": "not_rest"` with the measured pose): a raised arm without torque
   falls.
4. **The next motion re-energizes by itself**: any motion tool re-opens the
   backend from the rest pose. (On the RS, opening clears latched faults with
   a torque-off stop frame -- harmless at rest, the only place this tool ever
   releases.) Asking for rest again while in standby moves nothing and reads
   nothing (`"at": "standby"`): reading the pose would switch the motors on.

Profiles without `park_q` (every arm except the reBot RS and the cuMotion
Isaac twins) have no known folded pose and none is guessed: the request goes
to the initial pose, motors on, `safe_to_power_off: false`. A backend opened
directly (the demo CLI, not a LazyArm) or a simulator keeps its connection
at rest and says so.

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

Everything above is tested on the mock stack (MockArm, mock camera, the RS
kinematics). Before relying on it with visitors:

- Watch each tool at low speed on the reBot with the real scene camera:
  confirm that "right on the screen" moves right on the demo screen (the
  calibration record says robot -x) and that `as_seen_on_screen` matches what
  the screen shows.
- Confirm the wrist-camera axis on hardware (TCP +z = camera up) and that
  joint 6 rolls never pass the limit, including from top-down grasp poses
  where the camera axis sits exactly horizontal.
- The camera box was read from photos (+-1-2 cm), not measured with the arm
  in a known pose; a straight tape measurement of the padded length and of
  fingertip-to-back-of-padding would tighten it.
- Measure `held_drop_default_m` against real props; with
  `grasp.place_support_clearance_m` unset the measured support offset is
  never recorded and the 6 cm default applies (conservative: low moves are
  refused early).
- The undo staleness tolerance (1.5 cm / 5 degrees) against real tracking
  error after a move.
- 5-DoF arms (SO-101): translations keep the full orientation, which a 5-DoF
  wrist often cannot hold off its azimuth plane; expect kinematic refusals.
- `move_to_rest`: watch the first fold from the initial pose at the profile's
  speed cap with a hand near the arm, and check that the occupancy map does not
  refuse the last step because of the arm's own unmasked parts near the base
  (the shutdown park has seen this). Confirm the measured pose at the stops
  stays within the 0.05 rad release tolerance, that the gripper closes, that
  the dashboard shows the arm in standby afterwards, and that the next motion
  re-opens it from rest without a jolt.
