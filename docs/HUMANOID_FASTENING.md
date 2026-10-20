# Humanoid fastening: Unitree G1 tightens the Factory nut with a three-finger screwdriver grip

**Next action:** run the episodes in a standalone Newton environment (newton 1.6.0, warp-lang
1.17.0, mujoco and mujoco-warp 3.12.0, scipy, trimesh, plus cascade's base dependencies
pyyaml/jsonschema/opencv-python-headless), not cascade's `.venv`:

```bash
<newton-env>/bin/python scripts/validate_humanoid_fastening.py \
  --robot-asset <newton-assets>/unitree_g1/mjcf/g1_29dof_with_hand_rev_1_0.xml \
  --output runs/humanoid-fastening-01 --device cuda:0 --episodes 1,1,1
```

`<newton-assets>` is `newton.utils.download_asset("unitree_g1")` (newton-assets commit
`f8fb7abc`, needs GitPython); the shipped profile expects that `unitree_g1` folder at
`assets/mjcf/unitree_g1` (gitignored). The Factory meshes come from `scripts/fetch_factory_assets.py`.
`--output` must be a new directory. A two-turn request uses the standalone fixture:
`scripts/validate_humanoid_screw_contact.py --turns 2 ... --cache <dir>`. `--fingers-open` is the
grasp negative control.

## What it is

A Unitree G1 humanoid stands at a work table and holds a vertical hex socket driver the way a
person holds a screwdriver from above: the right Dex3 hand is on top of the handle, the **thumb**
on one side and the **index and middle** fingers wrapped around the opposite side. The hand holds
the driver **only through solved finger/handle contacts with friction**: there is no weld,
attachment, equality constraint or driver pose write. The arm turns the hand about the driver
axis, the driver turns the pre-engaged Factory M20 nut through the hex socket, and the hand
re-grips between strokes. Newton (MJWarp solver, SDF mesh contact) solves the nut on the threaded
bolt; the unchanged CASCADE fastening verifier judges each command.

## Grasp design

| Choice | Value and reason |
| --- | --- |
| Grip | Palm on top of the handle, thumb opposing index + middle (the Dex3 thumb sits at z = 0 on the palm, the index/middle bases at z = +-28.5 mm on its far edge). |
| Handle | Collision cylinder, **40 mm diameter, 100 mm long**, top 3.7 mm below the palm, axis 102 mm from the wrist along the hand x axis; then an 80 mm shaft and the six-wall socket (0.3 mm clearance). The index and middle inner faces are 31.2 mm apart (asset meshes): a 32 mm handle overlaps them by 0.4 mm, so both fingers can only bear load on a handle of about 36 mm or more. |
| Finger collision | The Dex3 asset collision meshes of the palm, thumb (3 links), index and middle (2 links each); hand links never collide with each other (self-collisions on, hand pairs excluded). |
| Friction | **mu = 0.8** on finger pads and handle (MuJoCo takes the max of the pair): elastomer pads on a rubberised screwdriver grip, the low end of rubber-on-rubber values; a bare metal handle would be about 0.3-0.5. |
| Grasp synthesis | CPU MuJoCo, convex-hull distances: each finger curls both joints together until a link touches the handle (0.3 mm), then the distal joint alone until the distal link touches. Contact posture: thumb (0, -0.076, -0.106) rad, index and middle about 1.14 rad on both joints. Open posture: contact backed off by 0.45 rad, at least 4 mm clearance. |
| Grip force | Closing targets are clamped **0.07 rad past the measured finger angle** (force-limited closing): kp 16 N m/rad on the thumb (1.12 N m), 10 on index and middle (0.7 N m each), kv 0.7, actuator cap 1.4 N m. A blocked finger presses with kp x 0.07 and its following error stays below the guard's 0.1 rad. |
| Turn | The handle axis is the palm normal through the grasp centre, 102 mm from the wrist, so no single G1 wrist joint is coaxial with it (the wrist-roll axis runs through the palm plate between the index and middle bases). The hand rotates about the tool axis with a **whole-arm motion**: an IK table q(phi) over waist, shoulder, elbow and wrist (phi = hand rotation about the tool axis, at most 0.74 rad of any joint per rad of phi), 3.42 rad per stroke at 1.0 rad/s, open-hand return at 1.5 rad/s. The wrist roll is the one torque-controlled joint (host PD to its table value, capped at 1 N m) and stays the domain's spindle channel: a stop is an exact-zero wrist-roll torque. |
| Re-grip | Settle, open the fingers (the driver stands in the socket on the nut), rotate the open hand back around the handle, close thumb/index/middle until contact, stroke. Every command starts with a re-grip: re-applying a squeeze from a loaded contact snapped the thumb at 13-14 rad/s, closing from open stayed below 1.6 rad/s. |
| Grasp check | Per stroke, defined before the runs (`stroke_grip_stats`): thumb, index and middle each carry **>= 15 % of the summed grip normal force** and are in loaded contact on **>= 95 %** of the stroke's solves; otherwise the stroke is a failed grasp. |

Code: `sim/newton_humanoid_screw.py` (scene, grasp synthesis, IK turn table, `WristDriverProgram`),
`sim/humanoid_fastening.py` (binding, limits, geometry bounds, backend, per-stroke grip statistics),
`apps/humanoid_fastening_runtime.py` (profile route), `configs/robots/g1_wrist_driver_fastening.yaml`.

## Real vs simplified

Real (solved by Newton, never written kinematically):

1. The driver is a free body. It is held only by solved finger/handle contact forces and friction
   and rests axially on the nut through a frictionless socket floor.
2. The nut is a free body on the fixed bolt; it turns and advances only through solved SDF thread
   contact and hex-wall contact. No nut drive, helical joint, external wrench or pose write.
3. The verdict comes from the unchanged `FasteningDomain` (rotation before stop >= 0.95 turn, axial
   advance and pitch, axis alignment, fixed fixture, >= 80 % contact-supported travel, 0.5 s observed
   rest with exactly zero spindle effort, final threading over the whole interval), fed by the
   unchanged `FactorySolveOwner`, `FasteningController`/`FasteningWriteGuard` and `SolveJournal`.
   Finger/palm-to-handle pairs are allowed contacts, never tool witnesses; a hand or handle contact
   with the nut, bolt or table is a forbidden pair.

Simplified (also in `SIMPLIFICATIONS` and burned into the video):

1. **Pelvis fixed to the world**: standing balance is not simulated (legs held by PD).
2. Only the right hand keeps its collision geometry; the rest of the robot does not collide.
3. Gravity compensation is a feed-forward through the joint actuators (MuJoCo `actuatorgravcomp`),
   passive on the torque-controlled wrist roll.
4. The driver starts engaged on the pre-engaged nut with the fingers 0.015 rad open of the
   synthesized contact posture (a light initial grip that seats the socket on a hex wall); pickup,
   initial engagement, seating and preload are not part of this task. Thread friction is 0.1.
5. The arm turn follows a precomputed IK table (position actuators), not a learned policy.

Task budgets (`humanoid_limits`) and the verifier are unchanged from the merged weld version.

## Results, 20 October 2026

Source `bc875b77c10eff9571f6488178891439dfbdc945`, one RTX PRO 6000 (shared with other agents'
GR00T jobs), Newton 1.6.0 / Warp 1.17.0 / MuJoCo and MJWarp 3.12.0. Each episode: fresh runtime
from the pinned profile, readiness 0.5 s quiet, `reset_stop`,
`fastening.turn_screw(turns=1, direction="tighten")` via `RobotRuntime.execute` (the call the MCP
server makes; no stdio or LLM). The runs went one after another in one job.

| Episode | Verdict | Rotation before stop (turn) | Final rotation (turn) | Axial advance (mm) | Max pitch error (mm) | Max radial offset (mm) | Contact-supported travel | Rest | Command wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **confirmed** | 0.9502 | 0.9544 | 2.214 | 0.253 | 0.422 | 99.0 % | 0.5 s, zero wrist-roll effort | 117.0 |
| 2 | **confirmed** | 0.9501 | 0.9554 | 2.230 | 0.256 | 0.419 | 99.5 % | 0.5 s, zero wrist-roll effort | 113.4 |
| 3 | **confirmed** | 0.9502 | 0.9560 | 2.201 | 0.262 | 0.426 | 98.7 % | 0.5 s, zero wrist-roll effort | 111.6 |
| Two-turn request (standalone, `verify_threading`, requested 2) | **confirmed** | - | 2.0051 | 4.84 | 0.26 | 0.42 | 99.4 % | wrist-roll command 0 | 114.6 |
| Negative control, fingers open (same turn, hand never closes) | **refuted** (requested rotation, advance, thread/tool contact) | - | - | - | - | - | - | lease expired after 3 strokes, nut rotation 0.001 turn | 121.7 |

Grip per stroke (mean normal force over the stroke, share of the summed grip force; every finger
was in loaded contact on 100 % of the solves of every stroke, the palm never touched):

| Run | Stroke | Thumb | Index | Middle | Hand rotation (rad) | Nut rotation (turn) | Handle rotation in the hand (rad) | Peak finger-to-driver torque (N m) | Grasp check |
|---|---|---|---|---|---|---|---|---|---|
| Episode 1 | 1 | 21.9 N (26 %) | 29.6 N (35 %) | 32.7 N (39 %) | 3.418 | 0.579 | 0.222 ahead | 0.041 | pass |
| Episode 1 | 2 | 23.2 N (31 %) | 27.8 N (37 %) | 23.5 N (32 %) | 2.492 | 0.361 | 0.224 behind | 0.039 | pass |
| Episode 2 | 1 | 22.0 N (27 %) | 28.3 N (35 %) | 31.6 N (39 %) | 3.418 | 0.582 | 0.237 ahead | 0.035 | pass |
| Episode 2 | 2 | 23.2 N (32 %) | 26.5 N (37 %) | 22.4 N (31 %) | 2.492 | 0.361 | 0.219 behind | 0.038 | pass |
| Episode 3 | 1 | 22.0 N (26 %) | 29.8 N (35 %) | 33.0 N (39 %) | 3.417 | 0.579 | 0.220 ahead | 0.037 | pass |
| Episode 3 | 2 | 21.0 N (29 %) | 27.5 N (38 %) | 23.4 N (33 %) | 2.457 | 0.353 | 0.238 behind | 0.025 | pass |
| Two-turn | 1-4 | 21.8-22.4 N (22-31 %) | 25.3-39.2 N (34-40 %) | 22.1-37.1 N (32-38 %) | 2.82-3.42 | 0.41-0.58 | 0.24 ahead to 0.27 behind | 0.032-0.040 | 4/4 pass |

Friction margin: the peak torque carried from the fingers to the driver (0.041 N m) needs
0.041 / (0.8 x 0.020 m) = 2.6 N of summed normal force at mu = 0.8; the grip held 72-85 N, a
capacity of about 1.2-1.4 N m (about 30 times the peak). The handle still rotated 6-9 % of the
hand rotation relative to the hand in each stroke (ahead in the first stroke, behind in the
second). That relative rotation is measured, not decomposed: at this margin it is not gross
sliding; finger-joint compliance under the force-limited grip (the fingers roll the cylinder) and
the soft-contact friction of MuJoCo (impratio 1) both allow it. Other per-episode peaks: wrist-roll
torque 0.33-0.36 N m (cap 1 N m), joint speed 1.51-1.58 rad/s (limit 2), nut speed 3.6-4.1 rad/s
(limit 10), about 8,200 solves.

History: the merged version (#420) welded the hand to the driver. A first contact grasp kept the
driver on the wrist-roll axis, which forces the handle between the index and middle fingers
(cradle grip); it was replaced by the grip above after review ("it squeezes without using the
middle finger"), with the per-stroke middle-finger load requirement. An earlier run of the
committed grip (`fb7f1b89`, three jobs in parallel on the shared GPU) confirmed episodes 1-2; its
episode 3 reached 0.9501 turn before the stop and was then rejected by the domain's 0.2 s
observation-age guard (a 0.214 s host stall); the table above is the serial re-run. Command wall
times of 111-117 s are close to the 120 s budget on this shared machine.

## Video

`scripts/render_humanoid_fastening.py` replays the saved solver body poses (no solver steps)
through Newton's headless viewer: main view, a close-up camera that follows the hand and looks at
the grip level with the fingers from the index/middle side, live nut rotation/advance,
thumb/index/middle normal forces, finger-to-driver torque, handle angle in the hand, solved contact
counts, each command's verdict and the per-stroke grasp check. A green tab under the nut is a
render-only marker (the socket hides the nut); it is not part of the simulated model.

## Not shown

Standing balance or a locomotion policy, tool pickup and initial engagement, seating/preload,
in-hand slip decomposition, repeatability beyond these episodes, more than one command per runtime
past about 10,000 solves, MCP over stdio or an LLM in the loop, and hardware.
