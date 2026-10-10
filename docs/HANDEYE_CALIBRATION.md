# Hand-eye calibration (ArUco, eye-to-hand and eye-in-hand)

`scripts/calibrate_handeye.py` (also `cascade-calib-handeye`) finds where a
camera sits relative to the arm, so 3D beliefs from that camera land where the
gripper will actually go. It is a port of Seeed's real-rig calibration from
their WRC fork (`scripts/calib_top_orbbec.py`, `scripts/calib_wrist.py`,
`src/wrc_demo/calibration/*`). It is camera-agnostic (intrinsics come from
every `Frame.K`), arm-agnostic (any arm profile with registered presets) and
runs every motion through cascade's `SafetyHarness` / `SafeArm`.

Two mountings, chosen by the camera profile's `extrinsics.mode`:

- **eye_to_hand**: a fixed camera (the overhead `d455f_scene`). The marker is
  fixed flat on top of the gripper, face up. The solve gives `T_cam2base`
  (and, as a by-product, `T_marker2gripper`).
- **eye_in_hand**: a camera on the wrist (`d455f_wrist`). The marker is taped
  flat on the table about 0.45 m in front of the base. The solve gives
  `T_cam2gripper` (and `T_marker2base`).

Both are the same problem, `A_i X B_i = Z`, solved jointly for X and Z (the
camera AND the marker pose) from every (FK, PnP) pair. Solving for both
jointly avoids the classic AX = XB pairwise formulation's amplification of
noise in small relative motions.

## Before you start

1. **Print the marker.** The default is ArUco `4x4_50`, id 0, 100 mm.
   ```
   python -c "import cv2; d=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50); \
   cv2.imwrite('aruco_4x4_50_id0.png', cv2.aruco.generateImageMarker(d, 0, 1000, borderBits=1))"
   ```
   Print at 100 % on stiff card. **Measure the printed black square with a
   ruler** and pass it as `--marker-size` (metres). Printers scale, and a 2 %
   size error is a 2 % depth error in every sample.
2. **Bind the camera serial.** The rig has two identical D455Fs, and only the
   serial tells them apart:
   ```
   python scripts/calibrate_handeye.py --list
   python scripts/calibrate_handeye.py --bind d455f_scene --serial 261522301814
   ```
   `--bind` refuses a serial that is not connected and keeps the profile's
   comments. A real run refuses a RealSense/Orbbec profile with no serial.
3. **Clear the workspace, keep the e-stop in reach.** The sweep covers the
   whole preset set at half the arm profile's joint-velocity cap.
4. **Rehearse without hardware** (optional, 5-20 s):
   ```
   python scripts/calibrate_handeye.py --dry-run
   python scripts/calibrate_handeye.py --dry-run --camera d455f_wrist
   ```
   This builds the arm profile's real harness on a `MockArm`, renders the
   marker where FK and a known transform put it (with pixel noise and one
   bumped-marker outlier), and reports how far the solve landed from the truth
   (about 2 mm / 0.1 deg eye-to-hand, 0.6 mm eye-in-hand). It cannot write to
   the real output path.

## Running it

```
# fixed overhead camera, marker on the gripper
python scripts/calibrate_handeye.py --camera d455f_scene --arm rebot_rs --marker-size 0.0995

# wrist camera, marker on the table
python scripts/calibrate_handeye.py --camera d455f_wrist --arm rebot_rs --marker-size 0.0995
```

The command prints the mounting instructions and asks you to type `yes`.
Then it:

1. Moves to the profile's `home_q` (vetted, speed-limited) inside the traced
   session.
2. Solves IK for every preset in `configs/calibration/handeye_poses.yaml`,
   seeded from `home_q`, and vets each solution with `harness.vet_pose`.
   Unreachable or vetoed presets are skipped and logged, never forced.
3. Orders the sweep greedily in joint space. On the reBot top-camera set this
   cuts the largest single move from 2.7 to 1.6 rad. Moves over 2 rad are
   skipped.
4. For each pose, it vets the pose **again** immediately before moving, then
   moves with `SafeArm.move_planned` (route vetting plus per-waypoint
   approval). The duration keeps the min-jerk peak at `--speed-frac`
   (default 0.5, maximum 1.0) of the harness `max_joint_vel`.
5. It settles (`--settle-time`, default 1 s), then waits for
   `--stable-frames` consecutive marker readings that agree within 2 mm.
6. It records FK of the **measured** joints, not the commanded ones.

`--manual` asks for ENTER before every vetted preset (`s` skips, `q`
finishes). `--poses FILE` replaces the preset set with your own
`[x, y, z, roll, pitch, yaw]` list (base frame, URDF rpy).

Ctrl+C or a safety stop follows the runtime's shutdown: halt the motion, park
through `SafeArm` to the profile's `park_q`, then torque off.

Each run writes `runs/handeye/<timestamp>_<camera>/`:

- `trace.jsonl`: every planned, skipped, recorded and home event.
- `captures/*.png`: the annotated detection per sample.
- A copy of the record. A rejected fit is kept there as
  `<camera>.REJECTED.json` for diagnosis and is **never** written to `--out`.

Fewer than 6 samples: nothing is solved or saved.

## The record and its quality gate

The output (default `configs/calib/<camera>_<arm>.handeye.json`) is a
schema-v1 JSON record (`cascade.calibration.dataset`). It holds:

- the transform and the marker pose;
- the intrinsics used;
- every sample with its measured joints and inlier flag;
- the metrics and the gate verdict.

The gate (`calibration.handeye.assess_hand_eye`) refuses a fit when any of
these holds:

- translation RMSE > 20 mm (the same bar as the Kabsch extrinsic fit);
- rotation RMSE > 3 deg;
- rotation spread < 5 deg. The poses rotated about at most one axis, so the
  transform is unconstrained. Such sets score a near-zero residual on a
  **wrong** answer, so the residual alone can never be trusted.
- fewer than 8 inliers, or inliers under 50 %.

Outliers are rejected robustly (median + 3 x scaled MAD of the residual
norms), and the fit is re-solved without them. The gate is recomputed from
the stored metrics on every load, so a hand-edited `"acceptable": true` does
not pass.

## Using it at runtime

Point the camera profile at the record. The CLI prints the snippet:

```yaml
extrinsics:
  mode: eye_to_hand
  hand_eye_json: ${repo}/configs/calib/d455f_scene_rebot_rs.handeye.json
```

Use `${repo}/...` or an absolute path. A relative path resolves against
whatever directory the runtime was started in.

A configured record that is missing, malformed, rejected by the gate, or made
for a **different camera serial** leaves that camera UNCALIBRATED. It keeps
streaming (evidence, keyframes) but never fuses 3D beliefs or depth, and
`Extrinsics.cam_to_base()` raises with the reason. There is no silent
fallback to the placeholder `T`. A profile `mode` that contradicts the record
is a config error.

`extrinsics.hand_eye_compensation_m: {x, y, z}` (ADR-0009, from WRC) is a
per-camera base-frame translation for a residual offset measured after
calibration:

- eye_to_hand: `T_comp @ T_cam2base`.
- eye_in_hand: `T_comp @ FK @ T_cam2gripper`.

It is identity by default and never inherited from another camera. Prefer
recalibrating; use it only for a measured, stable offset.

The legacy `hand_eye_npz` (key `T_result`) still loads.

## Verifying

The solve's own residual is not an independent check. Verify against
positions you measured:

```
# eye_to_hand: lay the marker flat at known base-frame points, ENTER at each
python scripts/calibrate_handeye.py --verify configs/calib/d455f_scene_rebot_rs.handeye.json \
    --camera d455f_scene --known-points "0.30,0,0;0.25,-0.05,0;0.35,0.05,0"

# eye_in_hand: the measured centre of the table marker used in the sweep
python scripts/calibrate_handeye.py --verify configs/calib/d455f_wrist_rebot_rs.handeye.json \
    --known-points "0.45,0,0"
```

The pass bar is RMSE < 10 mm (Seeed wiki §5.3). `--verify` refuses a rejected
record and a record whose serial differs from the profile's. Use at least 3
points, spread over the working area. Measure from the base origin defined by
the arm's URDF, not from the base plate's edge.

## Differences from WRC

Bugs fixed while porting (each pinned by a test):

- **Solver.**
  - The "Huber" weight was quartic.
  - MAD was measured from zero, which on the real top-camera data rejected 14
    of 37 good samples.
  - Random restarts were scored on synthetic data rather than the real
    samples, so one saved WRC record reports a 5e-6 mm residual for a
    transform more than 100 mm off.
  - Eye-in-hand was solved with the eye-to-hand equation.
  - `so3_log` was unstable near pi, `se3_log` / `se3_exp` did not round-trip,
    and `is_se3` accepted NaN.
- **Records.** For eye-in-hand, the runtime read the marker transform as the
  camera transform. A rejected or other-camera calibration only warned and
  was used anyway.
- **ArUco.** The corner model was left-handed. cascade uses OpenCV's marker
  convention (centre origin, x right, y up, z out of the face) and
  `SOLVEPNP_IPPE_SQUARE` with an ambiguity ratio.
- **Motion.** The wrist script drove the SDK's `move_to_traj` with no
  harness, and the top-camera script only pre-checked joint limits. Here
  every pose is vetted twice and every move goes through `SafeArm`.

Not ported:

- `--gravity-comp` hand-guided capture. Free-drive is motion the harness
  cannot vet; the CLI refuses it with that reason. Use `--manual` or
  `--poses`.
- `flip_hand_eye.py`. Flipping a solved transform by hand is how a wrong
  convention becomes a confident wrong calibration. The record carries its
  own `frame_convention`.
- `_calib_cleanup.py`. Shutdown is the runtime's halt, park, torque-off.

## Files

- `src/cascade/calibration/`:
  - `frames.py`: SE(3) helpers.
  - `handeye.py`: the solver and the gate.
  - `dataset.py`: the record.
  - `aruco.py`: detection and PnP.
  - `session.py`: the collection loop and the preset registry.
  - `synthetic.py`: the rendered marker camera.
  - `devices.py`: `--list` / `--bind`.
  - `cli.py`: the command.
- `configs/calibration/handeye_poses.yaml`: per-arm preset sets. A new arm
  gets an entry only after its reachability is measured;
  `tests/test_handeye_presets.py` pins the reBot sets.
- `perception/grounding.py` `Extrinsics.from_config`: the runtime loader.
- Tests: `tests/test_handeye_*.py`, `tests/test_extrinsics_handeye.py`.
- The older point-correspondence fit (`scripts/calibrate_camera.py`, Kabsch
  on the gripper tip seen in depth) still works for eye-to-hand cameras with
  depth and no marker.
