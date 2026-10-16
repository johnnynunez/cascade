# WRC perception + grasp port (Orbbec Gemini 2, camera-frame planner)

**Next action:** on the rig venv, list the attached Orbbec units (read-only, opens no stream;
seconds):

```bash
python -c 'from cascade.perception.orbbec_camera import list_orbbec_devices as l; print(l())'
```

Seeed's WRC repo (`wrc_demo`) forked cascade at `389a18d`; this records which of its real-rig
perception and grasp changes (`git log 389a18d..8a0c9e3` in WRC) cascade ported, how to enable
them and what hardware checks are owed. **Every hardware statement here is UNVERIFIED in
cascade**: tests drive fakes only (`tests/fake_orbbec_sdk.py` for `pyorbbecsdk`, the mock arm).

## Ported

| WRC source | cascade location | Notes |
|---|---|---|
| `src/wrc_demo/perception/orbbec_camera.py`, `camera_base.py` diff | `src/cascade/perception/orbbec_camera.py`, `make_camera` `type: orbbec`, `list_orbbec_devices()` | Lazy `pyorbbecsdk` import in `open()`; CameraBase contract (`_grab` raises `CameraError`, never sets `frame_id`; BGR colour; float32 metre depth aligned to colour). |
| `configs/cameras/orbbec_overhead.yaml`, `orbbec_wrist.yaml` | same paths | Serials unpinned. Overhead carries Seeed's measured `T_cam2base` (2026-09-02) as a labelled reference; wrist is `role: wrist` and never fuses. |
| Orbbec tests + fake SDK | `tests/test_orbbec_camera.py`, `tests/test_orbbec_color_decode.py`, `tests/fake_orbbec_sdk.py` | Real OBFormat numbering, pybind11-strict enums, real MJPG/YUYV payloads. |
| `37285b0` detector init after `rig.open()` | `apps/demo.py::build_runtime`, `tests/test_detector_after_camera_open.py` | Built inside the partial-build guard: a failed model load closes the rig. |
| `src/wrc_demo/grasping/camera_grasp.py` | `src/cascade/grasping/camera_grasp.py` (`plan_grasp_from_mask`, `camera_frame_grasps`), `tests/test_camera_grasp.py`, `tests/test_camera_grasp_replay.py` | Follows `tool_axis_order`; `finger_drop_m` is an argument; frame/extrinsic consistency check. Finger-cleaning test strengthened (WRC's sat on the threshold). Replay: two real banana runs (camera-frame position, jaw width, line of sight on the 32° mount). |
| WRC `_plan_grasps` layer 1 | `grasp.backend: camera_frame`, `SkillRuntime._camera_frame_candidates` | Opt-in, visible, does not replace GraspGen-X. |

Changes against WRC's Orbbec code, each pinned by a test:

1. **Formats** matched by name or int value (pybind11 enums are strict: `OBFormat.MJPG == 5` is
   False). YUYV/YUY2 decode as the last uncompressed fallback. Profile choice: exact w/h/fps,
   then same w/h, then any; inside a bucket BGR/RGB > MJPG > YUYV; undecodable codecs (H.264/5,
   NV12) never chosen.
2. **Depth scale** read per frame (`DepthFrame.get_depth_scale()`, mm per unit; WRC's device
   accessor does not exist in v2); an implausible scale raises `CameraError`.
3. **Misaligned depth** after alignment is a `CameraError`; no frame is returned.
4. **Intrinsics are never synthesized** (WRC's 70° fallback misplaces every grasp): pin
   `intrinsics:` or rely on the SDK.
5. **Refused hardware D2C** restarts the pipeline once with software `AlignFilter` (result via
   `as_frame_set()`); a failure after `start()` stops the pipeline.

## Enabling

### Orbbec camera

1. Install the Orbbec SDK v2 Python bindings (`pyorbbecsdk2` or github.com/orbbec/pyorbbecsdk)
   and udev rules **into the rig venv only** (cascade and CI never install them).
2. List the units (command at the top).
3. Put the serial in `configs/cameras/orbbec_overhead.yaml` (or a copy); required with two units.
4. Calibrate the mount (`scripts/calibrate_camera.py` or [hand-eye](HANDEYE_CALIBRATION.md)) and
   replace the reference `T`.
5. Run `python -m cascade.apps.demo --cameras orbbec_overhead[,orbbec_wrist] --arm ...`.

### Camera-frame planner

Set `grasp.backend: camera_frame` in a profile `overrides:` block or export
`CASCADE_GRASP_BACKEND=camera_frame`. Knobs (`configs/demo.yaml`, `grasp.camera_frame`):

| key | default | meaning |
|---|---|---|
| `required` | `false` | `true`: no mask grasp raises a visible `SkillError`, never OBB. `false`: falls back to OBB, reported as `grasp_planner_used = "obb (camera_frame: <reason>)"` plus a memory note. |
| `include_obb` | `true` | Analytic OBB alternates ranked behind the mask grasp (WRC layer 3). |
| `insertion_depth_m` | `0.015` | TCP pushed this far into the object along the approach (WRC ran `0.0` with 2 cm baked into its hand-eye JSON). Tune with the extrinsics (`hand_eye_compensation_m`). |
| `depth_quantile` | `0.5` | Depth statistic inside the mask (median). |
| `finger_drop_m` | `0.030` | Mask pixels this much nearer than the median (segmented fingers) are dropped. |
| `max_fix_offset_m` | `0.03` | Planned surface point must lie inside the localized box plus this margin, else refused as a frame/extrinsic mismatch. |

It needs a segmentation mask (brain detector's SAM masks or the mock detector) and measured depth
(`sensor` or `mono`); plane-cast depth is refused. Candidates are ordinary `Grasp`s and go
through the same memory re-rank, jaw-width check, IK from `home_q`, flip twin and harness
`vet_pose` plus descent-segment vetting as every backend; no threshold, margin or timeout
changed. No shipped profile selects it (the Spark presenter requires GraspGen-X).

## Not ported, and why

1. **`skills/rebot_grasp_bridge.py` and its `skill_grasp_object` / `skill_pick_and_place`
   branches**: it drives `RebotArmEndPose.move_to_traj` + `GraspDriver` directly, so
   `harness.approve()` never sees the streamed joints. SafeArm per-waypoint approval,
   observed-finger gating and segment vetting stay the only motion path. It also hard-codes the
   place pose (z = 0.08 m, rpy (0, 1.2, 0)), durations and force.
2. **`grasping/obb_grasp.py` `approach_pitch_rad: 1.2`**: WRC's own `_approach_rotation(1.2,
   yaw=0)` comes out **68.8° off vertical** (set.py's pitch is 21°), with the jaw axis turned
   90° while `width_m` reports the minor axis. `camera_frame` covers tilted approaches; IK
   restarts from `home_q` and flip twins cover the IK basin issue. `test_place_rotation_setpy_parity.py`
   tests pinocchio arithmetic, not cascade.
3. **`grasping/force.py`** (effort 1.0, WRC ADR-0002): gripper force belongs to the control port
   ([WRC_CONTROL_PORT.md](WRC_CONTROL_PORT.md)); cascade's pick close holds at kp·effort and the
   opt-in `gripper.max_contact_squeeze_rad` bounds the squeeze past contact.
4. **Superseded or out of scope**: selector `ik_retries=3` (`Kinematics.ik` already does 4
   restarts); `perception/grounding.py` loaders and `hand_eye_compensation_m` (calibration
   work); `37285b0` detector conf 0.25 → 0.10 and extra `detect_classes` (prop-specific policy);
   RealSense backend (byte-identical).
5. **WRC replay tests for base-frame position, pregrasp, approach and IK**: xfail in WRC itself;
   the poses predate a 30 mm finger change and the 2026-09-02 recalibration.

## Hardware verification still owed

Phased per the real-robot bring-up rules. Nothing moves the arm until a human is at the e-stop;
steps 1–3 are read-only.

1. **SDK, enumeration, serials.** One entry per Gemini 2 with the right serials, USB 3 link (USB 2
   may not offer 1280x720 colour). With both units attached, swap the profile serials and
   confirm the views swap; unplug one and confirm its profile fails with "not found".
2. **Stream contract.** Open `orbbec_overhead` (and `orbbec_wrist`): colour format inspectable
   (MJPG expected at 1280x720); a red object reads red in BGR; `depth_m` has the colour shape
   and the tape-measured distance (`get_depth_scale()` 1.0 at default precision); `K` matches
   the factory intrinsics; a refused HW D2C logs the software fallback. Depth opens only as Y16:
   packed-only firmware (Y11/Y12/RLE) fails with "no depth stream profile" (enable SDK
   conversion or extend `_pick_video_profile` with a decoder and a test).
3. **USB stability.** Full demo with SAM on CUDA: no USB reset or re-enumeration (`dmesg -w`)
   during model load and warm-up. Then calibrate the overhead extrinsic.
4. **Camera-frame planner, dry.** `preview_grasp` with `grasp.backend: camera_frame` on real
   objects: approach tilt matches the camera ray (memory note reports degrees off vertical), jaw
   width = calipers + 15 mm, and an open gripper in view does not pull the grasp centre.
5. **Supervised motion** at reduced velocity: one `grasp_object`, record `grip_verified`,
   `gripper_open_frac` and the evidence trace; only then tune `insertion_depth_m` with the
   extrinsic. Never relax `max_fix_offset_m` to make a grasp pass.
