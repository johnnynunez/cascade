# SO-101 (Standard Open Arm 101) — vendored model provenance

`so101.urdf` is an unmodified copy of the upstream follower-arm URDF, renamed
only so the profile path is stable if upstream re-runs its calibration.

| | |
|---|---|
| Upstream | https://github.com/TheRobotStudio/SO-ARM100 |
| Path | `Simulation/SO101/so101_new_calib.urdf` |
| Commit | `385e8d7c68e24945df6c60d9bd68837a4b7411ae` (2025-07-02, "fix(urdf): fixing multiple issues related to the last URDF update (#117)") |
| sha256 | `3a65d2d35e68a8d2f0c2cc176d19b884506543c93ba72980145b80abe276022c` |
| License | Apache-2.0 — see `LICENSE` beside this file |
| Modifications | none (byte-identical to upstream; only the filename differs) |

`_new_calib` is the current upstream calibration; `so101_old_calib.urdf` also
exists upstream and is **not** vendored. They differ in link offsets, so mixing
one arm's calibration with the other's model puts FK off by millimetres in a
way nothing downstream can detect.

## Why only the URDF

Pinocchio builds the kinematic model from URDF **text**; it never opens the
meshes. `buildModelFromXML` on this file yields nq=6 with `gripper_frame_link`
present, so FK/IK/limits/safety are all fully functional with a 16 KB file and
no LFS payload. The ~20 MB of STL meshes referenced by `<mesh filename=...>`
are needed only for rendering (MuJoCo viewer, Isaac) — see below.

Keeping them out matters beyond disk: CI's `actions/checkout` does not fetch
LFS content by default, so an LFS-backed model file breaks every kinematics
test with a pointer file (this repo already ate that incident once — see the
`.usda` note in `.gitattributes`).

## Getting the meshes and the MuJoCo model

```bash
python scripts/fetch_robot_assets.py so101          # meshes + MJCF
python scripts/fetch_robot_assets.py --list         # what's available
```

That fetches into `assets/urdf/so101/assets/` (meshes, matching the relative
paths already inside the URDF) and `assets/mjcf/so101/` (the MuJoCo Menagerie
MJCF, needed only by `--arm so101_mujoco`). Both directories are gitignored.

The MJCF comes from a different upstream with its own provenance:

| | |
|---|---|
| Upstream | https://github.com/google-deepmind/mujoco_menagerie |
| Path | `robotstudio_so101/` |
| Commit pinned by the fetch script | `ac6b2b09983786f3036cab1000221017fa2193b4` |
| License | Apache-2.0 |

The two models are **not** interchangeable: Menagerie added primitive collision
geometry, a camera mount and tuned gripper solver parameters, and it is derived
from the same `so101_new_calib` calibration. `tests/test_kinematics_so101.py`
pins the URDF facts the arm profile depends on so an upstream re-vendor that
moves a joint cannot pass silently.
