# Humanoid fastening: Unitree G1 tightens the Factory nut with a hand-held driver

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
`scripts/validate_humanoid_screw_contact.py --turns 2 ... --cache <dir>`.

## What it is

A Unitree G1 humanoid stands at a low work table, holds a hex socket driver in its right hand
and tightens the pre-engaged Factory M20 nut by **rolling its right wrist**, re-gripping between
strokes. Newton (MJWarp solver, SDF mesh contact) solves the nut on the threaded bolt; the
unchanged CASCADE fastening verifier judges each command.

| Choice | Why |
| --- | --- |
| **G1 29-DoF + Dex3 hands**, newton-assets MJCF `g1_29dof_with_hand_rev_1_0.xml` | Newton-native asset with hands and a forearm wrist roll. The H2 Newton route is not ready ([HUMANOID_H2.md](HUMANOID_H2.md): newton-assets PR #53 open, PhysX-trained policy) and the pinned H2 stage's 31 joints are legs, waist, arms and head (no fingers). |
| **Hex socket driver** on the forearm axis | `right_wrist_roll_joint` turns about the forearm; with wrist pitch/yaw held at 0 the driver axis *is* that axis, so the roll spins the socket without orbiting it. Same six-wall socket and 0.3 mm clearance as the SO-101 fixture. |
| **Vertical forearm over the bolt** | Sampled IK search (waist + right shoulder/elbow): the forearm can only be vertical with the elbow beside the body, so the bolt sits at the robot's right-front on a 0.42 m table. |
| **Re-grip** | The wrist roll range is +-1.97 rad; each stroke is 3.24 rad (0.52 turn), so one turn needs two strokes and one re-grip: settle, open the hand (grasp released, the driver rests on the nut), roll back, settle, close the hand (grasp re-made where the hand is), next stroke. |

Code: `sim/newton_humanoid_screw.py` (scene, IK, wrist PD, `WristDriverProgram`),
`sim/humanoid_fastening.py` (binding, limits, geometry bounds, backend),
`apps/humanoid_fastening_runtime.py` (profile route), `configs/robots/g1_wrist_driver_fastening.yaml`.

## Real vs simplified

Real (solved by Newton, never written kinematically):

1. The nut is a free body on the fixed bolt; it turns and advances only through solved SDF
   thread contact and hex-wall contact (same Factory meshes, SDF resolution, contact material and
   MJWarp settings as `ThreadingScene`). No nut drive, helical joint, external wrench or pose write.
2. The turning torque comes from the G1 `right_wrist_roll_joint` actuator (host PD, capped at
   1 N m; measured peak 0.10 N m) through the grasp into the driver. Every other joint is a
   position actuator; the arm follows the measured nut height by IK interpolation.
3. The verdict comes from the unchanged `FasteningDomain` (rotation before stop >= 0.95 turn,
   axial advance and pitch, axis alignment, fixed fixture, >= 80 % contact-supported travel,
   0.5 s observed rest with exactly zero wrist effort, final threading over the whole interval),
   fed by the unchanged `FactorySolveOwner`, `FasteningController`/`FasteningWriteGuard` and
   `SolveJournal` through a backend with the `FactoryNewtonBackend` surface.

Simplified (also in `SIMPLIFICATIONS` and burned into the video):

1. **Pelvis fixed to the world**: standing balance is not simulated (legs held by PD).
2. **Grasp = weld equality** between `right_wrist_yaw_link` and the driver. It is on exactly when
   the commanded fingers are at the closed posture; a re-grip re-enables it at the *current*
   relative pose (no correction). Finger motion is cosmetic; no physical finger grasp was tried.
3. Only the right hand keeps its asset collision geometry (excluded against its own driver), so a
   hand contact with the nut, bolt or table would be a forbidden pair; the rest of the robot does
   not collide.
4. Gravity compensation is a feed-forward through the joint actuators (MuJoCo
   `actuatorgravcomp`) on every joint except the turning wrist roll (vertical axis: no gravity
   moment).
5. The driver starts mounted on the pre-engaged nut (as the SO-101 fixture); pickup, initial
   engagement, seating and preload are not part of this task. Thread friction is 0.1 (the
   seating recipe's declared value), not the turn recipe's 0.01: 0.01 is below the M20 thread's
   self-locking threshold (tan of the 2.5 mm lead angle, about 0.044) while the released driver
   rests on the nut; that choice was made from this estimate, not from a measured 0.01 run. The socket floor that carries the released driver is frictionless (MJWarp
   minimum), so torque reaches the nut only through the hex walls.

Task budgets that differ from the SO-101 candidate (`humanoid_limits`): joint speed 2 rad/s
(finger re-grip), wrist effort cap 1 N m, one-turn command 15 s sim / 120 s wall (a re-grip sits
inside the turn). Rest window and thresholds, observation age, tracking error, margin, the
10 rad/s bound and `ThreadContract` are the shared values. The profile selects the existing
opt-in startup-heap freeze: without it, measured generation-2 collections of 0.25-0.28 s inside
the owner exceeded the 0.2 s observation age during readiness.

## Results, 20 October 2026

Source `54994b311a9748d934dd6cd53619331e71d16e18`, model identity `3d73562c…`, one RTX PRO 6000
(GPU 0, shared with another agent's Isaac Lab-Arena jobs), Newton 1.6.0 / Warp 1.17.0 / MuJoCo
and MJWarp 3.12.0. Each episode: fresh runtime from the pinned profile, readiness 0.5 s quiet,
`reset_stop`, `fastening.turn_screw(turns=1, direction="tighten")` via `RobotRuntime.execute` (the
call the MCP server makes; no stdio or LLM). Wrist stroke 1.0 rad/s, return 3.0 rad/s.

| Episode | Verdict | Rotation before stop (turn) | Final rotation (turn) | Axial advance (mm) | Max pitch error (mm) | Max radial offset (mm) | Contact-supported travel | Rest | Command wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **confirmed** | 0.9502 | 0.9578 | 2.393 | 0.050 | 0.296 | 88.0 % | 0.50 s, zero wrist effort | 58.2 |
| 2 | **confirmed** | 0.9502 | 0.9577 | 2.393 | 0.017 | 0.295 | 87.4 % | 0.50 s, zero wrist effort | 58.0 |
| 3 | **confirmed** | 0.9501 | 0.9577 | 2.392 | 0.023 | 0.290 | 86.0 % | 0.50 s, zero wrist effort | 65.1 |
| 4, command 1 | **confirmed** | 0.9500 | 0.9576 | 2.392 | 0.025 | 0.291 | 83.8 % | 0.50 s, zero wrist effort | 58.8 |
| 4, command 2 | **unverified**: observation age 0.2002 s > 0.2 s at solve 10154 | - | - | - | - | - | - | - | 53.0 |
| Two-turn request (standalone, `verify_threading`, requested 2) | **confirmed** | - | 2.0085 | 5.02 | 0.03 | 0.29 | 96.0 % | wrist command 0, wrist speed < 0.001 rad/s | 60.5 |
| Negative control (episode 1 with wrist torque forced to 0) | pre-stop **refuted** (0.0010 turn, no advance); overall unverified | 0.0010 | - | - | - | - | - | lease expired | 90.0 |

Contact/torque statistics per one-turn episode: peak wrist torque 0.102 N m, peak solved thread
load 4.9-8.9 N, peak hex-wall load 0.85-1.56 N, peak nut speed 2.0-2.9 rad/s (limit 10), one
re-grip, about 5,900 solves. Episode 4's second command was rejected by the domain's 0.2 s
observation-age guard (host side): the generic solve journal keeps every typed row (dozens of
tracked objects each), and past about 10,000 solves an automatic full collection, even with the
startup heap frozen, approaches 0.2 s. The two-turn request is therefore shown with the standalone
fixture. The negative control's lease expiry ends in the domain's existing "stop receipt"
fault after the guard's own expiry stop; it is never reported as success.

Stroke speed is the contact-margin lever: one-turn standalone runs measured 0.92/0.94
contact-supported travel at 1.0 rad/s, 0.89/0.93 at 1.5 and 0.80-0.85 at 2.0 rad/s, and one
2.0 rad/s runtime episode was refuted at 0.7998 pre-stop. The hand is about 50 times the nut's
inertia, so a faster wall makes the nut chatter inside the 0.3 mm socket clearance.

## Video

`scripts/render_humanoid_fastening.py` replays the saved solver body poses (no solver steps)
through Newton's headless viewer: main view, socket close-up, live nut rotation/advance, wrist
angle/torque, grasp state, solved contact counts and each command's verdict. A green tab under
the nut is a render-only marker (the socket hides the nut); it is not part of the simulated model.

## Not shown

Standing balance or a locomotion policy, a physical finger grasp of the handle, tool pickup and
initial engagement, seating/preload, repeatability beyond these episodes, more than one command
per runtime past about 10,000 solves, MCP over stdio or an LLM in the loop, and hardware.
