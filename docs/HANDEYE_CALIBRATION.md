# Hand-eye calibration (ArUco or markerless; eye-to-hand and eye-in-hand)

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

An eye-to-hand RGB-D camera can also be calibrated **without a marker**
(`--method markerless`, [below](#markerless-eye-to-hand-depth-no-marker)):
the same vetted sweep, but the arm's own meshes are fitted to depth. The
same machinery runs at runtime as an opt-in
[extrinsic drift monitor](#extrinsic-drift-monitor-runtime) that notices a
knocked camera and stops fusing it.

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

The gate is per method (`dataset.assess_record`): a marker record is judged
by `calibration.handeye.assess_hand_eye`, a markerless one by
`calibration.markerless.assess_markerless` (its own section below), and an
unknown `method` never loads. The marker gate refuses a fit when any of
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

## Markerless eye-to-hand (depth, no marker)

```
python scripts/calibrate_handeye.py --camera d455f_scene --arm rebot_rs --method markerless
python scripts/calibrate_handeye.py --dry-run --method markerless   # synthetic, MockArm
```

No marker is mounted: the arm is the target. Hydra-style (Marker-Free RGB-D
Hand-Eye Calibration, arXiv 2504.20584), pure numpy, in
`calibration/markerless.py`:

- **The arm's surface** (`calibration/robot_surface.py`): the arm profile's
  URDF collision meshes (binary or ASCII STL, metres), area-weighted samples
  with normals at a fixed seed, only the exterior skin (about 60 % of the
  reBot CAD area is internal and would pull the fit behind the surface),
  posed with the profile's own signed `Kinematics` (`joint_signs`). A second,
  unsigned model would pose every link mirrored and converge confidently onto
  the wrong arm. Links driven beyond `kin.n` (the reBot finger slides, at an
  unmeasured opening) are left out.
- **Collection**: the marker method's sweep, unchanged (presets IK'd from
  `home_q`, vetted when planned and again right before each move,
  `SafeArm.move_planned`, the same halt -> park -> torque-off shutdown). After
  settling, it takes the temporal median of `--depth-frames` (default 5)
  frames of SENSOR depth, at joints verified unchanged across the grabs, and
  records FK of those measured joints. The depth is saved as 16-bit mm PNGs
  in `captures/`, referenced from the record.
- **Solve**: robust point-to-plane ICP over all poses jointly on SE(3).
  Visible model points per pose come from back-face culling plus a coarse
  z-buffer at the current estimate. Association is projective: model point ->
  depth pixel (small window in a block-median pyramid), back-projected point,
  and normal from depth gradients. A pair is used only if the model normal
  and the depth normal agree within 60 deg, which stops a link's side from
  snapping onto the table. Tukey weights with a MAD scale; the rejection
  distance anneals 20 cm -> 1 cm coarse-to-fine. The start is the profile's
  accepted record, else its inline `extrinsics.T` (a profile with neither is
  refused), plus 12 camera-centred perturbations scored by fitness.
- **Refused before anything moves**: an eye-in-hand profile (the wrist camera
  does not see the arm), an RGB-only camera type or a camera that delivers no
  depth, an arm without a URDF surface model, no starting T.

### Markerless gate

`assess_markerless` refuses a fit when any of these holds (thresholds in
`markerless.py`):

- fewer than 6 usable poses (a pose needs >= 150 inlier points), or fewer
  than 3000 inlier points in total (the arm is barely visible);
- under 50 % of the arm's visible model points explained by the depth (wrong
  basin, occluded arm);
- point-to-plane RMSE > 15 mm;
- poses disagree: one pose alone moves its arm surface > 10 mm (RMS) or turns
  the camera > 1 deg from the joint solution;
- degenerate: the smallest eigenvalue of the normalised 6x6 normal matrix
  < 0.01 or its condition number > 200 (some direction is unconstrained,
  whatever the residual);
- pose diversity: TCP positions span < 10 mm along their 2nd principal axis
  (one pose or a line), or the tool rotation spread is < 5 deg.

Measured on the synthetic rig (camera ~0.95 m above the base, looking down,
D455-like noise: 0.75 % of range plus correlated low-frequency noise, edge
and random dropout, 1 mm quantisation, a table plane):

| case | result |
|---|---|
| all 37 reBot eye-to-hand presets, start 10 cm / 15 deg off, 5 noise seeds, 1 frame per pose | 0.87-1.35 mm, 0.147-0.161 deg; RMSE 2.2 mm, inliers 71 %, min eig 0.051, cond 14 |
| same, temporal median of 5 frames | 0.97-1.41 mm, 0.066-0.132 deg; RMSE 1.9 mm |
| solve time (37 poses) | 2.7-3.0 s on the Spark, under heavy shared load |
| `--dry-run --method markerless` (start 8 cm / 10 deg off, 37 poses, 5 frames) | 0.94 mm / 0.103 deg, accepted; 13.7 s wall including the simulated sweep |
| basin, 13 poses (every 3rd preset), 10 random start directions each | 5 cm/8 deg, 10 cm/15 deg, 15 cm/22 deg: 10/10. 20 cm/30 deg: 2/10. 30 cm/45 deg: 2/10. Accepted-but-wrong: 0 of 50 |
| one pose | refused: 1 usable pose, 659 inlier points, no diversity (its fit was 5.3 mm / 1.08 deg off) |
| 8 poses translating the TCP along a line, no rotation | refused on diversity (its fit was 1.4 mm / 0.13 deg off) |
| a 15 cm plate that only translates (controlled geometry) | refused as degenerate (min eig < 0.01); the same plate tilted passes |

Two consequences. The basin is wide enough for the shipped placeholders,
which are within a few cm and about 15 deg; outside it the fit is refused,
not accepted wrong. The reBot's own geometry is rich enough that even one
view is not numerically degenerate (min eig 0.026 for one pose, 0.034 for
the translation-only sweep, against the 0.01 bar). Those are refused on pose
count, inlier count and diversity instead, because a single view cannot show
that the model or the FK is wrong. The degeneracy check is for genuinely
under-constrained geometry (planar or symmetric parts, an arm mostly out of
view).

All of this is synthetic: the real D455F's depth on the arm (dark anodised
links, specular highlights, edge flying pixels) has not been measured yet.
Verify a real markerless record with `--verify` (method-independent; the
marker spec comes from the command line, since the record has none).

## Extrinsic drift monitor (runtime)

A knocked fixed camera keeps streaming self-consistent images, while every
belief and occupancy voxel fused through its old extrinsic lands centimetres
off. `perception/drift_monitor.py` watches for that with the arm as the
target, per eye-to-hand camera, opt-in:

```yaml
extrinsics:
  mode: eye_to_hand
  hand_eye_json: ${repo}/configs/calib/d455f_scene_rebot_rs.handeye.json
  drift_monitor:
    enabled: false        # opt-in
    period_s: 5.0         # one check per period while the arm is static
    max_offset_m: 0.010   # RMS displacement of the visible arm surface
    max_rot_deg: 3.0      # rotation of the correction (backstop)
    consecutive: 3        # drift checks in a row before acting
    auto_apply: false     # adopt a passive re-calibration automatically
```

Unknown keys, non-bools, and non-finite or non-positive values are refused at
startup. So is `enabled: true` on an eye-in-hand or RGB-only camera. An
uncalibrated camera has nothing to monitor; the monitor says so on stderr and
does not start.

- **Static only.** A check runs when no harness motion is in progress, the
  WorldWatcher is not paused for a motion skill, the measured joints have not
  changed for 1 s, the frame was captured after that, and the joints are
  unchanged after the grab. A moving arm is never checked. The monitor's
  only arm interface is a joint reader. That reader returns None while the
  arm is in standby (it never materialises a LazyArm). **The monitor never
  commands the arm.**
- **The check**: a short single-view ICP of the arm's surface at FK through
  the current extrinsic. It reports the RMS displacement of the visible
  surface and the rotation of the correction. A view that cannot decide is
  inconclusive: the arm occluded, out of view, too few points, or depth that
  does not explain it. An inconclusive view neither counts nor resets the
  count; an ok view resets it.
- **Flagged** after `consecutive` drift checks:
  - `Extrinsics.invalidate()` marks the camera uncalibrated (the same
    semantics as a rejected record: `cam_to_base()` raises with the reason,
    and every holder sees it).
  - Fusion and depth mapping turn off for that ONE stream through
    `WorldWatcher.set_camera_fusion`. Its generation bump makes a tick
    already in flight drop its commit.
  - Other cameras are untouched.
  - The flag is surfaced on stderr, in `<run>/extrinsics_drift.jsonl`, as a
    memory note (the dashboard activity feed), and as `extrinsics_drift` in
    the dashboard `/state` and MCP `world_state`.
- **Passive re-calibration.** The static views the checks already take are
  kept: distinct joint configurations only, at most 24. Views from before
  the knock are dropped when the camera is flagged. Once the views are
  diverse enough, they are solved into a CANDIDATE record at
  `<run>/extrinsics_candidates/<camera>_<stamp>.handeye.json`. The arm is
  never moved for it.
  - With `auto_apply: false` (the default), the monitor prints how to adopt
    the candidate: point `hand_eye_json` at it and restart, or re-run the
    markerless calibration.
  - With `auto_apply: true`, it is applied only if it passes the markerless
    gate AND explains more of the visible arm in recent depth than the active
    extrinsic (by 5 points or more). Fusion then comes back on.

Thresholds are measured on the synthetic D455, single view, 111 checks over
37 presets x 3 noise seeds:

| situation | surface displacement | rotation | checks over threshold |
|---|---|---|---|
| noise only | 2.0 mm median, 4.7 p99, 5.7 max | 0.52 deg median, 1.53 p99, 1.65 max | 0 / 111 |
| 0.5 deg knock | 5.5 mm median | 0.49 deg | 0 % (below a single view's noise floor) |
| 1 deg knock | 11.7 mm median | 1.02 deg | 95 % |
| 2 deg knock | 24.4 mm median (22.8 min) | 1.97 deg | 100 % |
| 5 mm shift | 5.8 mm median | 0.56 deg | 0 % |
| 10 mm shift | 10.6 mm median | 0.77 deg | 84 % |
| 20 mm shift | 20.1 mm median | 0.88 deg | 100 % |

One check takes about 0.05 s. `max_offset_m` 10 mm sits 1.75x above the worst
noise-only reading. `max_rot_deg` is only a backstop: a single view
constrains rotation about the arm weakly (1.65 deg of noise), and the surface
displacement is what flags real knocks. In the tests, a 2 deg knock is
flagged on exactly the 3rd check after it. Noise alone, a moving arm and an
occluded arm never flag. A passive candidate recovers the knocked camera
within 5 mm / 0.5 deg. A knock under about 1 deg / 10 mm is not detected; for
that, re-run the calibration.

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
  - `robot_surface.py`: the arm's exterior surface from its URDF meshes,
    posed by the signed kinematics.
  - `markerless.py`: the markerless solver, its gate, `measure_offset`
    (the drift check) and the record.
  - `synthetic_depth.py`: the D455-like depth camera that renders the arm.
  - `devices.py`: `--list` / `--bind`.
  - `cli.py`: the command.
- `configs/calibration/handeye_poses.yaml`: per-arm preset sets. A new arm
  gets an entry only after its reachability is measured;
  `tests/test_handeye_presets.py` pins the reBot sets.
- `perception/grounding.py` `Extrinsics.from_config`: the runtime loader.
- `perception/drift_monitor.py`: the runtime extrinsic drift monitor,
  wired in `apps/demo.py` (`_drift_monitors`).
- Tests: `tests/test_handeye_*.py`, `tests/test_extrinsics_handeye.py`,
  `tests/test_robot_surface.py`, `tests/test_synthetic_depth.py`,
  `tests/test_markerless_{solver,record,cli}.py`,
  `tests/test_extrinsic_drift.py`.
- The older point-correspondence fit (`scripts/calibrate_camera.py`, Kabsch
  on the gripper tip seen in depth) still works for eye-to-hand cameras with
  depth and no marker.
