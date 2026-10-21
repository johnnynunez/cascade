# Hand-eye calibration (ArUco or markerless; eye-to-hand and eye-in-hand)

**Next action:** rehearse without hardware (5–20 s), from the repo root (`--extra kinematics`
brings pinocchio for the arm model):

```bash
uv run --extra kinematics python scripts/calibrate_handeye.py --dry-run
```

Hardware runs (`--list`, `--bind`, the sweep, `--verify`) talk to the rig's cameras and arm: run
them from the repo root with the rig venv's `.venv/bin/python` (the rig has no bare `python`, and
`uv run` would re-sync the live `.venv`).

`scripts/calibrate_handeye.py` (also `cascade-calib-handeye`) finds where a camera sits
relative to the arm, so 3D beliefs from that camera land where the gripper will go. Ported from
Seeed's WRC fork (`scripts/calib_top_orbbec.py`, `scripts/calib_wrist.py`,
`src/wrc_demo/calibration/*`); camera-agnostic (intrinsics from every `Frame.K`), arm-agnostic
(any arm profile with registered presets), and every motion goes through `SafetyHarness` /
`SafeArm`.

The camera profile's `extrinsics.mode` picks the mounting:

| mode | camera | marker | solve gives |
| --- | --- | --- | --- |
| `eye_to_hand` | fixed (overhead `d455f_scene`) | flat on top of the gripper, face up | `T_cam2base` (+ `T_marker2gripper`) |
| `eye_in_hand` | wrist (`d455f_wrist`) | taped flat on the table ~0.45 m in front of the base | `T_cam2gripper` (+ `T_marker2base`) |

Both solve `A_i X B_i = Z` jointly for camera and marker from every (FK, PnP) pair, which
avoids the noise amplification of pairwise AX = XB. An eye-to-hand RGB-D camera can also be
calibrated [without a marker](#markerless-eye-to-hand-depth-no-marker), and the same machinery
runs at runtime as the opt-in [extrinsic drift monitor](#extrinsic-drift-monitor-runtime).

## Before you start

1. **Print the marker** (default ArUco `4x4_50`, id 0, 100 mm) at 100 % on stiff card:
   `uv run python -c "import cv2; d=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50); cv2.imwrite('aruco_4x4_50_id0.png', cv2.aruco.generateImageMarker(d, 0, 1000, borderBits=1))"`.
   **Measure the black square with a ruler** and pass it as `--marker-size` (metres): a 2 %
   size error is a 2 % depth error in every sample.
2. **Bind the camera serial.** The rig's two D455Fs differ only by serial:
   `.venv/bin/python scripts/calibrate_handeye.py --list`, then
   `.venv/bin/python scripts/calibrate_handeye.py --bind d455f_scene --serial 261522301814`.
   `--bind` refuses a serial that is not connected and keeps the profile's comments; a real
   run refuses a RealSense/Orbbec profile with no serial.
3. **Clear the workspace and keep the e-stop in reach.** The sweep covers the whole preset set
   at half the arm profile's joint-velocity cap.
4. **Rehearse** (`--dry-run`, add `--camera d455f_wrist` for the wrist). It builds the arm
   profile's real harness on a `MockArm`, renders the marker with pixel noise and one bumped
   outlier, and reports the error to the truth (about 2 mm / 0.1 deg eye-to-hand, 0.6 mm
   eye-in-hand). It cannot write the real output path.

## Running it

```bash
# fixed overhead camera, marker on the gripper
.venv/bin/python scripts/calibrate_handeye.py --camera d455f_scene --arm rebot_rs --marker-size 0.0995
# wrist camera, marker on the table
.venv/bin/python scripts/calibrate_handeye.py --camera d455f_wrist --arm rebot_rs --marker-size 0.0995
```

It prints the mounting instructions and waits for `yes`. Then, per run (a few minutes):

1. Moves to the profile's `home_q` (vetted, speed-limited) and solves IK for every preset in
   `configs/calibration/handeye_poses.yaml` seeded from `home_q`; `harness.vet_pose` vets each.
   Unreachable or vetoed presets are skipped and logged, never forced.
2. Orders the sweep greedily in joint space (reBot top camera: largest move 2.7 -> 1.6 rad);
   moves over 2 rad are skipped.
3. Vets each pose **again** right before moving, then `SafeArm.move_planned` (route vetting plus
   per-waypoint approval) with the min-jerk peak at `--speed-frac` (default 0.5, max 1.0) of
   the harness `max_joint_vel`.
4. Settles (`--settle-time`, default 1 s), waits for `--stable-frames` (default 4) consecutive
   marker readings within 2 mm, and records FK of the **measured** joints.
5. Ctrl+C or a safety stop: halt, park through `SafeArm` to `park_q`, torque off. A refused park
   (dead camera) keeps the torque on and says so: support the arm before you power it off. The
   hold is remembered: the next calibrate, arm-check or start refuses to connect that arm until
   you hold it and run `python scripts/arm_hold.py confirm` at a terminal on the robot host (your
   sudo password, then the phrase). The calibrate that follows connects at its first motion and
   only HOLDS: it asks you to let go and type `go` at that terminal (never through a pipe or in an
   agent's shell) before the sweep moves the arm. Over an arm
   that measures raised with no hold recorded it stops with "the arm was NOT connected" (lower the
   arm to rest, or support it and confirm).

`--manual` asks for ENTER before every vetted preset (`s` skips, `q` finishes). `--poses FILE`
replaces the presets with your own `[x, y, z, roll, pitch, yaw]` list (base frame, URDF rpy).

Each run writes `runs/handeye/<timestamp>_<camera>/`: `trace.jsonl` (every planned, skipped,
recorded and home event), `captures/*.png` and a copy of the record. A rejected fit is kept
there as `<camera>.REJECTED.json` and is **never** written to `--out`. Fewer than 6 samples:
nothing is solved or saved.

## The record and its quality gate

Output (default `configs/calib/<camera>_<arm>.handeye.json`) is a schema-v1 record
(`cascade.calibration.dataset`): transform, marker pose, intrinsics, every sample with measured
joints and inlier flag, metrics and gate verdict. The gate is per method (`dataset.assess_record`:
`calibration.handeye.assess_hand_eye` or `calibration.markerless.assess_markerless`); an unknown
`method` never loads. The marker gate refuses when:

1. translation RMSE > 20 mm (the Kabsch extrinsic fit's bar);
2. rotation RMSE > 3 deg;
3. rotation spread < 5 deg: poses rotated about at most one axis leave the transform
   unconstrained and score a near-zero residual on a **wrong** answer;
4. fewer than 8 inliers, or inliers under 50 %.

Outliers (median + 3 x scaled MAD of residual norms) are dropped and the fit re-solved. The
gate is recomputed from the stored metrics on every load: a hand-edited `"acceptable": true`
does not pass.

## Using it at runtime

Point the camera profile at the record (the CLI prints the snippet):

```yaml
extrinsics:
  mode: eye_to_hand
  hand_eye_json: ${repo}/configs/calib/d455f_scene_rebot_rs.handeye.json
```

Use `${repo}/...` or an absolute path; a relative path resolves against the runtime's start
directory.

1. A record that is missing, malformed, gate-rejected or made for a **different camera serial**
   leaves that camera UNCALIBRATED: it keeps streaming (evidence, keyframes) but never fuses 3D
   beliefs or depth, and `Extrinsics.cam_to_base()` raises with the reason. No silent fallback
   to the placeholder `T`.
2. A profile `mode` that contradicts the record is a config error.
3. `extrinsics.hand_eye_compensation_m: {x, y, z}` (ADR-0009, from WRC) adds a measured residual
   base-frame offset: `T_comp @ T_cam2base` (eye_to_hand) or `T_comp @ FK @ T_cam2gripper`
   (eye_in_hand). Identity by default, never inherited; prefer recalibrating.
4. The legacy `hand_eye_npz` (key `T_result`) still loads.

## Verifying

The solve's residual is not an independent check. Verify against measured positions
(pass: RMSE < 10 mm, Seeed wiki §5.3):

```bash
# eye_to_hand: lay the marker flat at known base-frame points, ENTER at each
.venv/bin/python scripts/calibrate_handeye.py --verify configs/calib/d455f_scene_rebot_rs.handeye.json \
    --camera d455f_scene --known-points "0.30,0,0;0.25,-0.05,0;0.35,0.05,0"
# eye_in_hand: the measured centre of the table marker used in the sweep
.venv/bin/python scripts/calibrate_handeye.py --verify configs/calib/d455f_wrist_rebot_rs.handeye.json \
    --known-points "0.45,0,0"
```

Use at least 3 points spread over the working area, measured from the URDF base origin (not the
base plate's edge). `--verify` refuses a rejected record and a serial that differs from the
profile's.

## Markerless eye-to-hand (depth, no marker)

```bash
.venv/bin/python scripts/calibrate_handeye.py --camera d455f_scene --arm rebot_rs --method markerless
uv run --extra kinematics python scripts/calibrate_handeye.py --dry-run --method markerless   # synthetic, MockArm
```

The arm is the target (Hydra-style, arXiv 2504.20584; pure numpy in `calibration/markerless.py`):

1. **Surface** (`calibration/robot_surface.py`): exterior skin of the URDF collision meshes
   (~60 % of the reBot CAD area is internal), posed by the profile's signed `Kinematics`
   (`joint_signs`; an unsigned model converges confidently onto a mirrored arm). Links beyond
   `kin.n` (reBot finger slides) are left out.
2. **Collection**: the marker sweep unchanged (vetted twice, `SafeArm.move_planned`, same
   shutdown). After settling, the temporal median of `--depth-frames` (default 5) sensor depth
   frames at joints verified unchanged; depth saved as 16-bit mm PNGs in `captures/`.
3. **Solve**: robust point-to-plane ICP over all poses jointly on SE(3), back-face culling plus
   a coarse z-buffer, projective association with a 60 deg normal-agreement test (stops a link
   side snapping onto the table), Tukey weights, rejection distance annealed 20 cm -> 1 cm.
   Start: the accepted record, else inline `extrinsics.T`, plus 12 perturbations.
4. **Refused before anything moves**: eye-in-hand, RGB-only or depth-less camera, an arm without
   a URDF surface model, no starting T.

### Markerless gate

`assess_markerless` refuses (thresholds in `markerless.py`) when:

1. fewer than 6 usable poses (>= 150 inlier points each) or fewer than 3000 inliers in total;
2. under 50 % of the visible model points explained by depth (wrong basin, occluded arm);
3. point-to-plane RMSE > 15 mm, or one pose alone moves the surface > 10 mm RMS / the camera
   > 1 deg;
4. degenerate: smallest eigenvalue of the normalised 6x6 normal matrix < 0.01 or condition
   number > 200;
5. pose diversity: TCP span < 10 mm along the 2nd principal axis, or tool rotation spread < 5 deg.

Synthetic rig only (camera ~0.95 m above the base, D455-like noise, a table plane): 37 reBot
presets from 10 cm / 15 deg off land 0.87–1.41 mm, 0.07–0.16 deg (RMSE ~2 mm), solve 2.7–3.0 s
on the Spark; starts up to 15 cm / 22 deg converge 10/10, 20 cm / 30 deg 2/10, and 0 of 50
wrong fits were accepted. One pose, a translation-only line and a translating flat plate are
all refused. The real D455F's depth on the arm (dark anodised links, specular highlights,
flying pixels) is **not measured yet**: verify a real record with `--verify` (method-independent).

## Extrinsic drift monitor (runtime)

A knocked fixed camera keeps streaming self-consistent images while every belief and occupancy
voxel fused through its old extrinsic lands centimetres off. `perception/drift_monitor.py`
watches each eye-to-hand camera with the arm as the target. Opt-in:

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

Unknown keys, non-bools and non-finite or non-positive values are refused at startup, as is
`enabled: true` on an eye-in-hand or RGB-only camera; an uncalibrated camera is not monitored.

1. **Static only, never commands the arm.** A check runs only with no harness motion, the
   WorldWatcher not paused, joints unchanged for 1 s before and after the grab. Its only arm
   interface is a joint reader that returns None in standby (never materialises a LazyArm).
2. **The check**: a short single-view ICP of the arm surface (~0.05 s). Occluded, out of view,
   too few points or unexplained depth = inconclusive (neither counts nor resets); an ok view
   resets the count.
3. **Flagged** after `consecutive` drift checks: `Extrinsics.invalidate()` marks the camera
   uncalibrated, `WorldWatcher.set_camera_fusion` turns off fusion and depth for that ONE stream
   (a tick in flight drops its commit), other cameras untouched. Surfaced on stderr, in
   `<run>/extrinsics_drift.jsonl`, as a memory note, and as `extrinsics_drift` in dashboard
   `/state` and MCP `world_state`.
4. **Passive re-calibration**: up to 24 distinct static views (pre-knock views dropped) are
   solved into a CANDIDATE `<run>/extrinsics_candidates/<camera>_<stamp>.handeye.json`; the arm
   never moves for it. `auto_apply: false` prints how to adopt it (point `hand_eye_json` at it
   and restart, or re-run markerless); `auto_apply: true` applies it only if it passes the
   markerless gate AND explains at least 5 points more of the visible arm than the active
   extrinsic, then fusion comes back on.

Thresholds (synthetic D455, single view, 111 checks): noise alone reads 2.0 mm median / 4.7 mm
p99 / 5.7 mm max and 0.52 deg median / 1.65 deg max, 0/111 over threshold, so `max_offset_m`
10 mm sits 1.75x above the worst noise. Detection: 1 deg knock 95 %, 2 deg 100 % (flagged on
exactly the 3rd check), 10 mm shift 84 %, 20 mm 100 %; 0.5 deg and 5 mm are below a single
view's noise floor. `max_rot_deg` is only a backstop (rotation is weakly constrained). A passive
candidate recovers within 5 mm / 0.5 deg. A knock under ~1 deg / 10 mm is not detected:
re-run the calibration.

## Differences from WRC

Fixed while porting, each pinned by a test:

1. **Solver**: the "Huber" weight was quartic; MAD was measured from zero (rejected 14 of 37 good
   real samples); restarts were scored on synthetic data (one WRC record reports 5e-6 mm for a
   transform >100 mm off); eye-in-hand used the eye-to-hand equation; `so3_log` was unstable
   near pi, `se3_log` / `se3_exp` did not round-trip and `is_se3` accepted NaN.
2. **Records**: eye-in-hand read the marker transform as the camera transform; a rejected or
   other-camera record only warned and was used.
3. **ArUco**: the corner model was left-handed. cascade uses OpenCV's marker convention (centre
   origin, x right, y up, z out) and `SOLVEPNP_IPPE_SQUARE` with an ambiguity ratio.
4. **Motion**: WRC's wrist script drove `move_to_traj` with no harness; here every pose is
   vetted twice and every move goes through `SafeArm`.
5. **Not ported**: `--gravity-comp` hand-guided capture (free-drive cannot be vetted; the CLI
   refuses it, use `--manual` or `--poses`), `flip_hand_eye.py` (the record carries its own
   `frame_convention`) and `_calib_cleanup.py` (shutdown is halt, park, torque-off).

## Files

- `src/cascade/calibration/`: `frames.py` (SE(3)), `handeye.py` (solver, gate), `dataset.py`
  (record), `aruco.py` (detection, PnP), `session.py` (collection loop, presets), `synthetic.py`
  / `synthetic_depth.py` (rendered cameras), `robot_surface.py`, `markerless.py` (solver, gate,
  `measure_offset` drift check), `devices.py` (`--list` / `--bind`), `cli.py`.
- `configs/calibration/handeye_poses.yaml`: per-arm presets; a new arm gets an entry only after
  its reachability is measured (`tests/test_handeye_presets.py` pins the reBot sets).
- Runtime: `perception/grounding.py` `Extrinsics.from_config`; `perception/drift_monitor.py`,
  wired in `apps/demo.py` (`_drift_monitors`).
- Tests: `tests/test_handeye_*.py`, `tests/test_extrinsics_handeye.py`,
  `tests/test_robot_surface.py`, `tests/test_synthetic_depth.py`,
  `tests/test_markerless_{solver,record,cli}.py`, `tests/test_extrinsic_drift.py`.
- The older point-correspondence fit (`scripts/calibrate_camera.py`, Kabsch on the gripper tip
  in depth) still works for eye-to-hand depth cameras without a marker.
