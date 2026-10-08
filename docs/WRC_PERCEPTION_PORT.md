# WRC perception + grasp port (Orbbec Gemini 2, camera-frame planner)

Seeed Studio's WRC repository (`wrc_demo`) is a fork of the July 2026 cascade
baseline at commit `389a18d`. This document records which of Seeed's
real-rig perception and grasp changes (`git log 389a18d..8a0c9e3` in WRC)
were ported to cascade's current architecture, where they went, what was not
ported and why, how to enable them, and what still has to be checked on
hardware. Branch: `feat/wrc-perception`.

Nothing in this port was run against a device. Every test drives fakes:
`tests/fake_orbbec_sdk.py` stands in for `pyorbbecsdk`, and the mock stack
stands in for the arm. **Every hardware statement below is UNVERIFIED in
cascade.**

## Ported

| WRC source | cascade location | Notes |
|---|---|---|
| `src/wrc_demo/perception/orbbec_camera.py` (+460), `camera_base.py` diff | `src/cascade/perception/orbbec_camera.py`, `make_camera` branch `type: orbbec` | Lazy `pyorbbecsdk` import in `open()`. The CameraBase contract holds: `_grab` raises `CameraError` and never sets `frame_id`, colour is BGR, depth is float32 metres aligned to colour. Includes `list_orbbec_devices()`. |
| `configs/cameras/orbbec_overhead.yaml`, `orbbec_wrist.yaml` | `configs/cameras/orbbec_overhead.yaml`, `configs/cameras/orbbec_wrist.yaml` | The serials are unpinned (WRC pinned Seeed's units). The overhead profile carries Seeed's measured `T_cam2base` (2026-09-02 solve) as a labelled reference. The wrist profile is `role: wrist` and never fuses. |
| `tests/test_orbbec_camera.py`, `tests/test_orbbec_color_decode.py`, the `conftest.py` fake SDK | the same test names plus `tests/fake_orbbec_sdk.py` | The fake uses real OBFormat numbering and pybind11-strict enums, and it encodes MJPG/YUYV payloads for real. |
| `37285b0` "USB stability": detector init moved after `rig.open()` | `apps/demo.py::build_runtime`, `tests/test_detector_after_camera_open.py` | The detector is now built inside the existing partial-build guard, so a failed model load closes the open rig. |
| `src/wrc_demo/grasping/camera_grasp.py` (+525) | `src/cascade/grasping/camera_grasp.py` (`plan_grasp_from_mask`, `camera_frame_grasps`) | The output follows the arm's `tool_axis_order`. `finger_drop_m` is an argument instead of an env var. There is a frame/extrinsic consistency check. |
| `tests/test_camera_grasp.py`, `test_camera_grasp_mask_cleaning.py` | `tests/test_camera_grasp.py` | The finger-cleaning test was strengthened. In WRC's version the finger sat exactly on the threshold, was kept, and the test passed anyway. |
| `tests/test_camera_grasp_replay.py` (the parts that still bind to the recording) | `tests/test_camera_grasp_replay.py` | Two real rebot_grasp banana runs. The tests check the recorded camera-frame position (1 mm print precision), the jaw width, and the line of sight on the rig's 32° mount. |
| WRC `_plan_grasps` layer 1 (the camera-frame planner as primary) | `grasp.backend: camera_frame` and `SkillRuntime._camera_frame_candidates` (opt-in, see below) | It does not replace GraspGen-X, and it is not silent. |

Cascade changes against WRC's Orbbec code, each pinned by a test:

- **Format matching.** Formats are matched by name or int value. pybind11
  enums are strict (`OBFormat.MJPG == 5` is False), so WRC's int comparisons
  only matched its own fake. That fake numbered the formats differently from
  the constants the camera compared against.
- **YUYV/YUY2.** These are decoded as the lowest-preference uncompressed
  fallback. WRC blocked them.
- **Profile choice.** Exact w/h/fps comes first, then the same w/h, then any
  resolution. Inside a bucket the order is BGR/RGB, then MJPG, then YUYV.
  Codecs the backend cannot decode (H.264/5, NV12, ...) are never chosen.
- **Depth scale.** The scale is read per frame
  (`DepthFrame.get_depth_scale()`, mm per unit). The device-level accessor
  WRC used does not exist in v2, so WRC always fell back to 1 mm. An
  implausible scale raises `CameraError`.
- **Misaligned depth.** If the depth shape does not match the colour shape
  after alignment, that is a `CameraError` and no frame is returned.
- **Intrinsics.** They are never synthesized. WRC's 70° fallback misplaces
  every grasp. Pin `intrinsics:` in the profile, or rely on the SDK.
- **Hardware alignment refused.** If hardware D2C is refused at `start()`,
  the pipeline restarts once with software `AlignFilter` alignment only. The
  `AlignFilter.process()` result is converted with `as_frame_set()`.
- **Partial open.** A failure after `start()` stops the pipeline.

## Enabling

### Orbbec camera

1. Install the Orbbec SDK v2 Python bindings (`pyorbbecsdk2` or
   github.com/orbbec/pyorbbecsdk) and the udev rules **into the rig venv
   only**. Cascade never installs them, and CI does not need them.
2. List the attached units. This is read-only and opens no stream:
   `python -c 'from cascade.perception.orbbec_camera import list_orbbec_devices as l; print(l())'`
3. Put the serial in `configs/cameras/orbbec_overhead.yaml`, or in a copy of
   it. With two Orbbec units attached this is required.
4. Calibrate the extrinsic for your mount (`scripts/calibrate_camera.py`) and
   replace the reference `T`.
5. Run `python -m cascade.apps.demo --cameras orbbec_overhead[,orbbec_wrist] --arm ...`.

### Camera-frame planner

Set `grasp.backend: camera_frame` in a profile `overrides:` block, or export
`CASCADE_GRASP_BACKEND=camera_frame`. The knobs are in `configs/demo.yaml` under
`grasp.camera_frame`:

| key | default | meaning |
|---|---|---|
| `required` | `false` | `true`: if there is no mask grasp, raise a visible `SkillError` and never substitute OBB. `false`: fall back to OBB and report it as `grasp_planner_used = "obb (camera_frame: <reason>)"` plus a memory note. |
| `include_obb` | `true` | Analytic OBB alternates are ranked behind the mask grasp (WRC layer 3). |
| `insertion_depth_m` | `0.015` | The TCP is pushed this far into the object along the approach. WRC's live config ran `0.0` because its hand-eye JSON had a 2 cm compensation baked in. Tune this together with the extrinsics (calibration workstream: `hand_eye_compensation_m`). |
| `depth_quantile` | `0.5` | Which depth statistic inside the mask is used (the median). |
| `finger_drop_m` | `0.030` | Mask pixels this much nearer the camera than the mask median are dropped. These are gripper fingers segmented into the object mask. |
| `max_fix_offset_m` | `0.03` | The planned surface point must lie inside the localized object's box plus this margin, or it is refused as a frame/extrinsic mismatch. |

The planner needs a segmentation mask (YOLOE-seg or the mock detector) and
measured depth (`sensor` or `mono`). It refuses plane-cast depth, because
that puts every pixel on the table. Candidates are ordinary `Grasp`s. They
go through the `GraspOutcomeMemory` re-rank, the selector's jaw-width check,
pregrasp and grasp IK seeded from `home_q`, the flip twin, and harness
`vet_pose` plus descent-segment vetting, the same as every other backend. No
threshold, margin or timeout was changed. The Spark presenter still requires
real GraspGen-X: no shipped profile selects `camera_frame`.

## Not ported, and why

- **`skills/rebot_grasp_bridge.py` (+227) and the bridge branches of
  `skill_grasp_object` / `skill_pick_and_place`.** These are pinned by
  `test_align_to_rebot_grasp.py`. The bridge drives
  `RebotArmEndPose.move_to_traj` + `GraspDriver` directly. Its own docstring
  says per-waypoint validation is left to the SDK controller, so
  `harness.approve()` never sees the streamed joints. Cascade's motion
  safety path (SafeArm per-waypoint approval, observed-finger gating, segment
  vetting) must stay the only path. The bridge also hard-codes the place pose
  (z = 0.08 m, rpy (0, 1.2, 0)), the durations and the gripper force. It also
  needs the hardware-only SDK and WRC's `WrcGripper` port, which belongs to
  the control workstream.
- **`test_place_rotation_setpy_parity.py`.** It asserts pinocchio RPY
  arithmetic, not cascade code. Changing `place_at`'s orientation to set.py's
  tilted pose would be a motion-behaviour change with no rig evidence in
  cascade.
- **`grasping/obb_grasp.py` (+111), `approach_pitch_rad: 1.2`.** Measured on
  WRC's own `_approach_rotation(1.2, yaw=0)`: the approach comes out
  **68.8° off vertical**, close to horizontal. set.py's rpy pitch 1.2 is
  21° off vertical. The jaw-opening axis is also turned 90°: it opens along
  the major axis while `width_m` still reports the minor axis. Cascade's OBB
  planner has moved on independently (`tool_axis_order`, widest section,
  rim grasps), and the tilted-approach use case is what `camera_frame` covers.
  The motivating IK basin problem is handled in cascade by IK with random
  restarts seeded from `home_q` and by the flip twins.
- **`grasping/force.py` (+77).** This is width-aware close targets with effort
  raised to 1.0, the MIT torque cap (WRC ADR-0002, tuned for a 3D-printed
  banana). It is gripper actuation and force, which belongs to the control
  workstream. It interacts with the air-grasp heuristic, which reads
  `close_frac_stage2`. Cascade's RS gripper instead closes at `kp` and relaxes
  to `hold_kp` on contact so objects are held lightly. Raising force without
  rig evidence in cascade is not a port.
- **`grasping/selector.py` `ik_retries=3`.** Superseded: cascade's
  `Kinematics.ik` already defaults to 4 random restarts inside the limit
  margins.
- **`perception/grounding.py` (+167).** These are extrinsics loaders (WRC JSON,
  cuRobo/legacy NPZ, serial cross-check) and `hand_eye_compensation_m`. They
  belong to the calibration workstream, as do `requires_calibration` and
  `data/calibration/cameras.json` and `calib_top_orbbec --bind`.
- **`37285b0` detector conf 0.25 → 0.10 and extra `detect_classes`.** This is
  detection policy tuned for Seeed's props, not a robustness fix. The
  extrinsics `${repo}` path part is already how cascade resolves paths.
- **RealSense backend.** WRC's `realsense_camera.py` is byte-identical to
  cascade's. The only USB-stability change was the detector ordering above.
- **WRC replay tests for base-frame position, pregrasp, approach and IK.**
  WRC marks them xfail itself. The recorded poses predate a 30 mm finger
  change and the 2026-09-02 recalibration, so no `T` on disk reproduces them.

## Hardware verification still owed (phased, see the real-robot bring-up rules)

Nothing here moves the arm until a human is at the e-stop. Steps 1–3 are
read-only.

1. **SDK and enumeration.** Install pyorbbecsdk v2 in the rig venv. Run
   `list_orbbec_devices()` and expect one entry per Gemini 2 with the right
   serials. Confirm the link speed is USB 3. On USB 2 the 1280x720 colour
   stream may not be offered at all.
2. **Stream contract.** Open `orbbec_overhead` (and `orbbec_wrist`) with
   their serials pinned. Check four things. (a) The chosen colour format is
   logged or inspectable: MJPG is expected at 1280x720. (b) The colours are
   correct, i.e. a red object reads red in BGR. (c) `depth_m` has the colour
   image's shape and reads the tape-measured distance at 1 mm/unit: verify
   that `DepthFrame.get_depth_scale()` reports 1.0 at default precision. (d)
   `K` matches the factory intrinsics. If HW D2C is refused, the log names
   the software-alignment fallback. The backend opens depth only as Y16. If
   the firmware offers only packed formats (Y11/Y12/RLE), `open()` fails with
   "no depth stream profile". Enable the SDK's format conversion, or extend
   `_pick_video_profile`'s depth preference with a decoder and a test.
3. **Serial binding.** With both units attached, swap the profile serials and
   confirm the views swap. Unplug one unit and confirm the profile pinned to
   it fails with "not found" rather than streaming the other.
4. **USB stability.** Start the full demo with YOLOE on CUDA and confirm no
   USB reset or re-enumeration (`dmesg -w`) during model load and warm-up.
5. **Extrinsic.** Calibrate the overhead mount (`scripts/calibrate_camera.py`)
   and replace the reference `T`. Coordinate `hand_eye_compensation_m` with
   the calibration workstream.
6. **Camera-frame planner, dry.** Use `preview_grasp` with
   `grasp.backend: camera_frame` on real objects. Check that the approach
   tilt matches the camera ray (the memory note reports degrees off
   vertical), that the jaw width matches calipers plus 15 mm, and that the
   finger cleaning works: put the open gripper inside the view and confirm
   the grasp centre does not move toward it.
7. **Camera-frame planner, supervised motion.** Run at reduced velocity.
   Execute `grasp_object` on one object and record the result's
   `grip_verified`, `gripper_open_frac` and the grasp evidence trace. Only
   then tune `insertion_depth_m` together with the extrinsic. Never relax
   `max_fix_offset_m` to make a grasp pass.
