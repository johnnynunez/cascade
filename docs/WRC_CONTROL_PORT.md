# WRC control port (reBot B601-RS)

Seeed Studio's WRC fork (`wrc_demo`, forked from the July-2026 baseline at
`389a18d`) ran the reBot B601-RS on a real rig and found and fixed control
problems. This document records which of those fixes cascade now has, where
they live, what was deliberately left out and why, how WRC's constants differ
from cascade's, and the onsite checks still needed. Everything below was
developed and tested offline: fakes stand in for the SDK, motorbridge and the
motors. **No change here has run on hardware.**

Sources: `git -C wrc log 389a18d..HEAD`, WRC
`docs/HARDWARE_VERIFICATION_HANDOVER.md`, `docs/adr/0001..0005`,
`docs/specs/teach_record_replay.md`.

## Ported

| WRC source | What | cascade location | Tests |
|---|---|---|---|
| Handover Finding #1; `control/__init__.py::make_arm` | RobStride motors keep fault bits (`fault_raw 0x4`) across sessions, and the SDK's `enable_all()` does not clear them. A faulted motor ignores MIT commands without reporting an error. The fix clears every motor (joints and gripper) with bounded retries on a late `comm_type=4` ack. Any other failure raises `RuntimeError` before torque is enabled and releases the bus. | `control/robstride.py::clear_motor_faults`; `RebotRSArm.connect`, `RebotRSMotorBridgeArm.connect`, `scripts/jog_rebot_mb.py`, `scripts/sign_check_rebot_mb.py` | `test_rebot_clear_error.py`, `test_rebot_bringup_scripts_clear.py` |
| Handover Finding #2 | The SDK's `RebotArmEndPose` 500 Hz `_loop_cb` overwrites any direct `send_mit` within 2 ms. **This does not apply to cascade.** Neither RS driver starts that loop, and each sends one MIT frame per harness-approved 50 Hz waypoint. A test pins this so a later refactor cannot add a second writer. | `rebot_rs_arm.py` docstring | `test_rebot_rs_sdk_loop.py` (AST check plus fake SDK; verified by mutation) |
| Handover Finding #3 | At mechanical zero, joint 2 reads +0.004 rad, below its limit (0) plus the 0.02 margin. WRC's workaround skipped the harness for the first waypoint. **That bypass is not ported.** cascade's directional escape rule already handles this case: a joint outside its margin may move strictly back toward the valid band, while holding or moving deeper is still refused. Tests on the real RS kinematics and harness show this is sufficient. | none needed (`SafetyHarness.approve` escape rule) | `test_rebot_rest_and_park.py` |
| Found while testing #3 | `rebot_rs.yaml` set `park_q` joint 3 to −0.05, below the joint's URDF lower limit of 0. The park only relaxes the margin, so the harness refused it at joint 3 = −0.001 while joint 2 was still about 0.05 rad up. The gripper park then never ran, and teardown reported `ParkIncomplete`. `park_q` is now exactly `[0]*6`, and a test keeps every profile's `park_q` inside its URDF limits. | `configs/arms/rebot_rs.yaml` | `test_rebot_rest_and_park.py` |
| `control/gripper.py` (`WrcGripper._send_gripper_mit` clip) | Gripper targets are clamped to the profile's travel `[min(open,closed), max(open,closed)]` on both RS transports, including the light-hold path. | `robstride.clamp_to_travel`; `rebot_rs_arm.py`, `rebot_rs_mb_arm.py` | `test_rebot_gripper_travel.py` |
| e3b0b2a (gripper sign convention) | `RebotRSArm` still fell back to the DM build's `open_pos=-6.8` when a profile had no gripper block. It now uses the measured RS default (6.2/0), which `rebot_rs_mb` already used. The stale RS polarity note in AGENTS.md is also fixed. | `rebot_rs_arm.py`, `AGENTS.md` | `test_rebot_gripper_travel.py` |
| a2d5950 (adaptive opening) | Opt-in pre-grasp opening to the grasp width plus a margin. It goes through `SafeArm.set_gripper` and is never narrower than the grasp width. A negative or non-finite margin is a config error. An unknown width means fully open. | `grasping/force.py::pregrasp_open_position`; `skill_grasp_object`; profile key `gripper.pregrasp_open_margin_m` (rebot_rs: `null`) | `test_pregrasp_adaptive_open.py` |
| e97998c, ba4e110 (settle window) | `RebotRSArm` now reads `settle_timeout_s` from the profile, as the motorbridge backend already did. Defaults and shipped values are unchanged. | `rebot_rs_arm.py` | `test_rebot_settle_config.py` |
| e97998c (`planning/cartesian_planner.py`, `SafeArm.move_joints_path/move_cartesian`) | Straight-line TCP motion; details below the table. | `planning/cartesian.py`, `ArmBase.stream_path`, `SafeArm.move_cartesian/move_joint_path`, `skill_move_relative` (profile key `cartesian_relative_moves`, rebot_rs: `false`) | `test_cartesian_path.py` |
| b70f3a0 (reflex) | "return (to the) home", "back home" and "go back home" are now `move_home` reflexes. | `agent/reflex.py` | `test_reflex_return_home.py` |

How the Cartesian port works:

- It does not depend on the SDK. It uses cascade's own `Kinematics` with numpy SE(3) interpolation.
- It is stricter than WRC's version:
  - WRC accepted up to 5 % unconverged IK samples. Here every 5 mm sample must solve, seeded from the previous one with no restarts.
  - A joint jump above 0.1 rad means the IK changed branch, and the motion is refused.
- Execution:
  - The path streams as one continuous motion, min-jerk in arc length, with the same velocity stretch as `move_joints`.
  - The exact ticks are preflighted with `vet_step` before the first command, and every tick then goes through `approve()`.
- It refuses to run on arms bound to a motion planner and on Isaac, which has its own clock.
- If a Cartesian move is refused, the arm does not silently fall back to a joint-space move.

## Not ported, and why

- **WRC's first-waypoint harness bypass** (`SafeArm._make_hook` `first_seen`). This is a safety bypass, and cascade's escape rule makes it unnecessary (Finding #3 above).
- **WRC e-stop auto-reset** (b70f3a0 calls `harness.reset_estop()` before every interactive task). It would silently clear a latched e-stop. cascade keeps `reset_stop` explicit and staff-only. The problem it worked around does not occur here either: Ctrl+C in the cascade CLI is a halt followed by a park, not an e-stop latch, and `StopSignals` defers signals while teardown runs. A *repeated* Ctrl+C still force-exits a stuck teardown; that is deliberate in cascade. WRC's other b70f3a0 change (`suppress(KeyboardInterrupt)` around disconnect) would swallow that escape, so it is not ported. A forced exit skips `disconnect()`, which leaves the motors holding their last MIT command rather than cutting torque.
- **"back to zero", "zero position" and "reset" mapped to home** (b70f3a0, 6b926fe, f916acc).
  - "Zero" is the park pose, not home, so mapping it to home would move the arm somewhere the user did not ask for.
  - "reset" is already `reset_scene` in cascade.
  - WRC itself deleted `move_to_joint_zero` on 2026-08-13.
  - cascade's shutdown park already drives the arm to `park_q` through the harness.
- **`scripts/emergency_stow.py`.** Its docstring says "no harness", although the code calls `SafeArm.move_joints(zeros)`; in cascade that call would be refused at joints 2 and 3 without the park's margin relaxation. cascade's teardown (`shutdown_runtime` → `_park_arm`) already parks through the harness, with only the margin relaxed, before torque off. A standalone stow script would also have to decide how to treat the perception watchdog (feeding a fake heartbeat would be a bypass). That decision is left for later.
- **`_calib_cleanup.safe_park_and_disconnect`.** This is calibration-script plumbing tied to WRC's calibration CLIs. cascade's teardown receipt (`lifecycle.teardown_step`) already enforces "never raise, always disconnect".
- **WRC's 10 ms gripper re-send loop** (a2d5950).
  - Its `set_gripper` ignores `effort` and always sends full `kp`, which would override cascade's `hold_kp` light hold after contact.
  - It competes for the bus lock with the 50 Hz arm stream and the mechPos reads.
  - It re-sends the same target, so it does not make the motion smoother.
  - RobStride MIT already holds the last command.
- **`WrcGripper` state machine** (IDLE→POSITION→CLOSING→HOLDING, torque-mode close at 1.5 N·m, stall when `|vel| < 0.12`, hard-stop = empty grasp).
  - It depends on a 500 Hz loop plus `get_state()` velocity and torque feedback. cascade measured that RS `get_state()` is not decoded on this firmware and that mechVel (0x701A) is not in rad/s, so the stall detector would read noise.
  - cascade already covers what it does: `close_gripper_torque` (stall on mechPos, then `hold_kp` light hold) and the skill-level `air_grasp_frac` empty-grasp check.
- **Width-aware close** (`grasping/force.py::compute_close_target`, `tests/test_gripper_width_aware_close.py`). WRC iterated v2 0.75 → v5 0.30 with `effort` 1.0, so the constants are not settled. They are also tuned against WRC's 5.0 rad ≙ 95 mm map, which disagrees with cascade's, and under MIT the squeeze force still scales with object width. This is not demonstrably better than cascade's fixed fractions, which since B38 can be bounded by the opt-in contact-relative squeeze cap `gripper.max_contact_squeeze_rad` ([REBOT_GRIP_SQUEEZE_CAP.md](REBOT_GRIP_SQUEEZE_CAP.md)); the light hold (`hold_kp`) runs only in the park close. Revisit after onsite force measurements.
- **ADR-0002 `force=1.0`.** This is a torque-mode hold value for `GraspDriver.grasp()`. cascade closes in position mode with kp-scaled effort, so the number has no equivalent here.
- **home_q changes** (e3b0b2a, ba4e110, f6fca0d, ADR-0001).
  - WRC tried three different home poses and settled on `[0, 0.1684, 0.6226, -0.4543, 0, 0]` (TCP 0.25/0/0.35, identity rotation).
  - cascade's `home_q` `[0, 1.2, 1.2, 0, 0, 0]` (TCP 0.452/0/0.438, facing forward) is pinned by `test_rebot_initial_pose.py` and matches the Isaac ready pose. It also seeds grasp IK, so changing it changes which IK branch grasps use.
  - It stays as it is. WRC's pose is available as an onsite option.
- **`kinematics.py` and `arm_base.py` diffs.**
  - WRC removed USD support and inlined `apply_joint_signs`. cascade keeps USD support.
  - WRC raised `settle_timeout_s` from 2.0 to 6.0 s in code. cascade instead takes it from the profile (see Ported) and does not change the default.
- **`make_arm` SDK stack** (`RebotArmEndPose` + `WrcGripper` + `RebotGraspBridge`). cascade's drivers own the MIT stream so the harness sees every waypoint (Finding #2).
- **Teach record/replay** (`skills/teach.py`, `skills/master_arm.py`, `docs/specs/teach_record_replay.md`). **Remaining.**
  - WRC's version drives the follower arm with raw SDK `send_pos_vel` at `vlim=15 rad/s`, and replays through `RebotArmEndPose` and `safe_home()`, all outside any harness.
  - It needs `fashionstar-uart-sdk` (the PiPER Mate leader arm) and `pynput`, neither of which is installed here.
  - A cascade port would need these parts:
    - Record: every follower command goes through `harness.approve()` at the stream rate. The leader arm is a lazy, optional backend.
    - Replay: through `SafeArm.move_joint_path`, which this port added.
    - Skills: both added under the five-step checklist in AGENTS.md (TOOL_SPECS in both directions, `_MOTION_SKILLS`, a plausibility question, `TOOL_REQUIREMENTS` for the leader arm, README counts).
    - Rig data: the leader→follower joint map (WRC hard-codes joints 1, 4 and 6 negated and does no calibration) must be measured on the rig.

## Constant discrepancies (WRC vs cascade `configs/arms/rebot_rs.yaml`)

| Quantity | WRC | cascade | Note |
|---|---|---|---|
| Gripper open angle | `_RS_ANGLE_OPEN = 5.0` rad (`open_pos: 5.0`) | `open_pos: 6.2` (travel 0 → +6.39 measured) | Same polarity, different scale |
| Gripper width at open | 0.09 m (`GRIPPER_MAX_DISTANCE_M`); `max_width_m: 0.095` in the profile | `max_width_m: 0.09` | With the linear map, 5.0 rad ≙ 0.09–0.095 m in WRC versus 6.2 rad ≙ 0.09 m in cascade. At least one of these maps is wrong, so measure it before enabling `pregrasp_open_margin_m` |
| Gripper close | torque mode, `close_torque 1.5`, `tau_max 1.5` N·m | position mode, `kp 2.0` × effort | Different force models |
| Hold | `default_force 0.30` (ADR-0002: 1.0) torque | `hold_kp 0.01` position stiffness | Not comparable |
| Stall | `|vel| < 0.12` rad/s, startup travel 0.30 rad | mechPos change < 0.03 rad per 150 ms | cascade avoids the velocity feedback that is not decoded |
| Arm settle tolerance | 0.05 (final; 0.08–0.15 tried) | `settle_tol` default 0.05 | Same |
| Settle timeout | 6.0 s (code default) | 2.0 s (ArmBase), profile-overridable | Tune onsite only if large moves report "did not settle" |
| home_q | `[0, .1684, .6226, -.4543, 0, 0]` | `[0, 1.2, 1.2, 0, 0, 0]` | See above |
| Zero / park | `[0]*6` | `park_q [0]*6` (was −0.05 on joint 3) | Now the same |
| Joint frame | asset URDF with axes flipped, no `joint_signs` | URDF + `joint_signs [-1]*6` (local) | Equivalent (FK of WRC home = 0.25/0/0.35 in cascade's local frame) |
| `max_joint_vel` | 1.2 (demo.yaml) | 0.8 (RS override) | cascade's lower cap is kept |

## Onsite hardware verification still required

Follow the phases in the `real-robot-bringup` checklist. **Nothing below may
run until a person is at the arm with the e-stop or power cut within reach.**
Support the arm before anything enables it: after a clean shutdown it is limp.

1. **Read-only probe (no torque).** Stop motorbridge-gateway and MotorBridge
   Studio (they use the same host id, 0xFD). Run `python scripts/diag_rebot_mb.py`
   then `--snapshot rest`. Record joints 2 and 3 at rest (WRC saw +0.004;
   cascade's rig saw −0.0009) and confirm every joint is inside the URDF limits.
   **Incident 2026-10-10.** `sign_check_rebot_mb.py --joints 2,3` drove the
   shoulder into its end at full stiffness, and the operator cut power. The
   cause was a mechPos read of +2.3e18 on joint 2's way back: the stall guard
   commanded the motor to that value. In addition, every motor had been
   registered as `rs-00` although joints 1-3 are `rs-06`, which put the MIT
   gains on the wrong scale. Both bring-up scripts now:
   - vet every reading (`robstride.plausible_position`);
   - clamp every target to the probe envelope;
   - ramp back from the last commanded pose, never from a reading;
   - stop the run, holding that pose, on an implausible reading;
   - refuse without the SDK's per-motor models.

   See `tests/test_rebot_probe_reading_guard.py`. Do not loosen any of these.
2. **Fault clear.** Use `python scripts/jog_rebot_mb.py --joint 1 --hold-only`
   (it now clears faults after the mode write and before enable). Confirm that
   no `clear_error failed` is raised and that the hold drift is under 50 mrad.
   If a clear fails, stop: power-cycle the arm and probe again.
3. **First low-velocity home move, e-stop in hand.** Temporarily lower
   `overrides.safety.max_joint_vel` in `rebot_rs.yaml` (for example to 0.3;
   only ever lower it). Start
   `python -m cascade.apps.demo --arm rebot_rs --camera mock --llm mock --interactive --no-serve`
   and type `go home` (a reflex: no LLM involved) from the rest pose. Report
   the measured joint state.
   This is the real test of Finding #3: the move must start without any
   harness rejection. Then exit with an empty line (or one Ctrl+C) and confirm
   the teardown receipt shows the park `complete` at `park_q = 0`.
4. **Gripper travel and stall re-verification.**
   - Hand-sweep the jaw with `diag_rebot_mb.py` and confirm 0 → +6.39 rad.
   - Measure the jaw opening in mm at about 1.5, 3.0, 4.5 and 6.2 rad to settle
     the 5.0-versus-6.2 rad ≙ 90 mm disagreement, and fix `max_width_m`/`open_pos`.
   - Run the park close (`close_gripper_torque`) on air and on a ~30 mm object,
     and confirm the `[rebot] grip contact` line appears only for the object.
   - Only after the width map is confirmed, consider
     `gripper.pregrasp_open_margin_m: 0.01`.
   - Then run the pick-close squeeze protocol in
     [REBOT_GRIP_SQUEEZE_CAP.md](REBOT_GRIP_SQUEEZE_CAP.md) (B38) before
     setting `gripper.max_contact_squeeze_rad`.
5. **Cartesian nudge (optional).** At the lowered velocity cap, set
   `cartesian_relative_moves: true` and issue `move down a bit` and `move up a bit`.
   Watch that the TCP travels straight. Revert the setting if anything looks wrong.
6. **Settle window.** If large moves report "did not settle" while the arm is
   visibly on its way, raise `settle_timeout_s` in the profile (WRC needed 4–6 s).
   Never touch `settle_tol` or the velocity cap to compensate.
