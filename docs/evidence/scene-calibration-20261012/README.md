# Fixed scene-camera calibration: closer, re-aimed mounting

## Scope

This is a **new supervised physical calibration** of the fixed D455F, serial
`261522301814`, in `eye_to_hand` mode on `rebot_rs`. It supersedes the record
for the previous camera placement. It does not calibrate the wrist camera or
certify a pick-and-place.

- Native capture run: `20261012_192623_d455f_scene`.
- Collection source base: `3dd5a10b4f43984b0db1e90e6fff597bee59119c`.
- Record: [`d455f_scene_rebot_rs.handeye.json`](../../../configs/calib/d455f_scene_rebot_rs.handeye.json).
- Pose list: [`d455f_scene_current.yaml`](../../../configs/calibration/d455f_scene_current.yaml).
- Numeric validation and depth-image checksums: [`validation.json`](validation.json).

## How the sweep was prepared

A new RGB-D capture and read-only measured joint snapshot supplied the rough
starting pose. Four single-rest-pose refinements agreed on approximately
`(0.311, -0.567, 0.465)` metres in the robot base frame. This estimate was only a
starting guess, not an accepted calibration.

The stock sweep contains poses whose arm geometry is cropped or too close to
the depth camera at this mounting. **Before acquisition**, 21 stock poses were
selected using these criteria:

1. The existing inverse-kinematics and safety-harness checks pass.
2. At least 97% of the sampled arm model projects inside the RGB-D image.
3. The fifth percentile of visible model points' optical depth is at least
   0.42 m, above the configured 0.4 m depth cutoff.

The source preset indices and selection criteria are retained in the pose file
and validation receipt. The selection preserves translational and rotational
diversity; the exact 21-pose list passed the native synthetic dry run. No pose
was selected or removed based on the new sweep's residuals.

The calibration CLI prioritizes an existing accepted `hand_eye_json` over the
inline `extrinsics.T`. To avoid silently seeding from the previous mounting,
collection used a new, initially absent per-mount output path in the worktree.
After acceptance and validation, that native record was copied byte-for-byte to
the canonical profile path. Keeping the canonical path preserves the normal
calibration CLI's default output contract.

## Physical run

The operator explicitly confirmed presence with the e-stop in hand, empty jaws
and a clear sweep area. The guarded entry point refused concurrent robot/Studio
owners and verified the measured rest pose before the CLI confirmation.

All 21 planned poses produced a saved sample. The arm then completed its parking
sequence, and the gripper referenced its closed stop. A separate read-only check
afterwards found the arm within **0.0157 rad** of the configured rest pose.

The actual stream negotiation was **848 x 480 native depth** and **1280 x 720
color**, both at 30 fps; the backend aligns depth to color. The configured color
resolution must not be mistaken for the native depth resolution.

## Native acceptance result

The unmodified markerless solver and acceptance gate returned `[OK]`:

- Samples: **21/21**, with **no post-fit exclusions**.
- Inlier fraction: **70.6%**.
- Point-to-plane fit RMSE: **3.1 mm**.
- Maximum per-pose surface offset: **2.3 mm**.
- Maximum per-pose rotation disagreement: **0.81 degrees**.
- TCP spread: **21.2 mm**; rotation spread: **18.6 degrees**.
- Fitted camera position: approximately **(0.308, -0.568, 0.470) m**, in the
  robot base frame (+x forward, +y left, +z up).

The profile's `Extrinsics.from_config` loads this exact record without a
calibration error. A different camera serial is refused. The run copy and
profile output have identical calibration data; the serializer's `saved_at`
field differs between those two native saves. The committed canonical record
is byte-identical to the native profile output, with SHA-256:

`47d94e43f0ecc2a03680751b9b3d09babe344159b2e1ff87ed44041ce453f5c9`

## Checks outside the fit

Separate rest-pose RGB-D captures from before and after the sweep were not fit
samples. Both gave conclusive model-surface offset checks. The after-sweep view
explained **72.8%** of the model and required an approximately **3.0 mm RMS
visible-surface correction** in the single-view drift check. The table-plane
check put the surface about **1.3 mm above base z=0** at the base, with about a
one-degree lateral slope; this is a consistency check, not a claim that the
slope's physical cause has been identified.

These checks use the same robot model and depth sensor. They do **not** establish
independent fiducial accuracy, guarantee absolute millimetre positioning, or
prove that the red object can be grasped. No physical pick was attempted here.

The 15 relevant camera/profile/hand-eye/markerless/regression test files passed:
**278 passed, zero failures, zero errors, zero skips**.

## Raw evidence and deployment boundary

The full 21 depth images, trace, native record and before/after holdouts are
retained privately on the rig under
`~/.cascade/calibration-evidence/20261012_192623_d455f_scene/`.
Raw room images are deliberately not committed. The validation receipt contains
the depth-file hashes and numerical checks.

The live checkout was not updated or restarted as part of this calibration.
Before a supervised pick, land the PR with required CI green, safely deploy the
record with the arm parked, and confirm that the restarted runtime loads it.
Old scene beliefs must not be treated as current coordinates after a camera
move. The wrist camera remains a visual feed, without a hand-eye record for 3D
fusion.

**Moving the fixed camera mount or robot base invalidates this calibration.**
For a repeat sweep at this unchanged mounting, pass the committed pose list:

```bash
python scripts/calibrate_handeye.py --camera d455f_scene --arm rebot_rs \
  --method markerless --poses configs/calibration/d455f_scene_current.yaml
```

This command moves the real arm and requires an operator with the e-stop, a
clear sweep area, empty jaws, and exclusive camera/CAN ownership. Never run it
alongside the robot server or bypass its confirmation.
