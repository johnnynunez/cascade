# WRC control port (reBot B601-RS)

**Next action:** before any onsite step, read [Onsite hardware verification](#onsite-hardware-verification-still-required)
and put a person at the arm with the e-stop or power cut within reach.

Seeed's WRC fork (`wrc_demo`, forked at `389a18d`) ran the reBot B601-RS on a real rig and fixed
control problems. This records which fixes cascade has, what was left out and why, the constant
differences and the owed onsite checks. Sources: `git -C wrc log 389a18d..HEAD`, WRC
`docs/HARDWARE_VERIFICATION_HANDOVER.md`, `docs/adr/0001..0005`,
`docs/specs/teach_record_replay.md`. The port was developed offline against fakes (SDK,
motorbridge, motors); except the scripted home move below, it is **unverified on hardware**.

## Ported

| WRC source | What | cascade location | Tests |
|---|---|---|---|
| Handover Finding #1 | RobStride motors keep fault bits (`fault_raw 0x4`) across sessions and `enable_all()` does not clear them; a faulted motor silently ignores MIT. Every motor (joints + gripper) is cleared with bounded retries on a late `comm_type=4` ack; any other failure raises `RuntimeError` before torque and releases the bus. | `control/robstride.py::clear_motor_faults`; `RebotRSArm.connect`, `RebotRSMotorBridgeArm.connect`, `scripts/jog_rebot_mb.py`, `scripts/sign_check_rebot_mb.py` | `test_rebot_clear_error.py`, `test_rebot_bringup_scripts_clear.py` |
| Handover Finding #2 | The SDK's 500 Hz `RebotArmEndPose._loop_cb` overwrites a direct `send_mit` within 2 ms. **Does not apply**: neither RS driver starts it; each sends one MIT frame per harness-approved 50 Hz waypoint. Pinned so no refactor adds a second writer. | `rebot_rs_arm.py` docstring | `test_rebot_rs_sdk_loop.py` (AST + fake SDK, mutation-verified) |
| Handover Finding #3 | At mechanical zero joint 2 reads +0.004 rad, below limit 0 + 0.02 margin. WRC's first-waypoint harness bypass is **not ported**: the harness escape rule lets a joint outside its margin move strictly back toward the band, while holding or going deeper is refused. | `SafetyHarness.approve` escape rule | `test_rebot_rest_and_park.py` |
| Found with #3 | `park_q` joint 3 was −0.05, below its URDF limit 0, so park stalled and teardown reported `ParkIncomplete`. `park_q` is now `[0]*6`; a test keeps every profile's `park_q` inside URDF limits. | `configs/arms/rebot_rs.yaml` | `test_rebot_rest_and_park.py` |
| `WrcGripper._send_gripper_mit` clip; e3b0b2a | Gripper targets clamped to `[min(open,closed), max(open,closed)]` on both RS transports incl. light hold; `RebotRSArm` without a gripper block uses the measured RS default (6.2/0), not the DM `open_pos=-6.8`. | `robstride.clamp_to_travel`; `rebot_rs_arm.py`, `rebot_rs_mb_arm.py` | `test_rebot_gripper_travel.py` |
| a2d5950 | Opt-in pre-grasp opening to grasp width + margin through `SafeArm.set_gripper`, never narrower than the width; negative/non-finite margin = config error; unknown width = fully open. | `grasping/force.py::pregrasp_open_position`, `gripper.pregrasp_open_margin_m` (rebot_rs: `null`) | `test_pregrasp_adaptive_open.py` |
| e97998c, ba4e110 | `RebotRSArm` reads `settle_timeout_s` from the profile (defaults unchanged). | `rebot_rs_arm.py` | `test_rebot_settle_config.py` |
| e97998c (`cartesian_planner.py`) | Straight-line TCP motion (below). | `planning/cartesian.py`, `ArmBase.stream_path`, `SafeArm.move_cartesian/move_joint_path`, `skill_move_relative` (`cartesian_relative_moves`, rebot_rs: `false`) | `test_cartesian_path.py` |
| b70f3a0 | "return (to the) home", "back home", "go back home" are reflexes for `move_to_rest` (folded rest); `move_home` is the raised "initial pose" ([SPATIAL_COMMANDS.md](SPATIAL_COMMANDS.md)). | `agent/reflex.py` | `test_reflex_return_home.py` |

Cartesian moves use cascade's `Kinematics` (no SDK) and are stricter than WRC's:

1. Every 5 mm sample must solve, seeded from the previous one with no restarts (WRC accepted 5 %
   unconverged); a joint jump above 0.1 rad (branch change) refuses the motion.
2. One continuous min-jerk stream in arc length with `move_joints`' velocity stretch; the exact
   ticks are preflighted with `vet_step`, then every tick goes through `approve()`.
3. Refused on arms bound to a motion planner and on Isaac (own clock); a refused Cartesian move
   never falls back to joint space.

## Not ported, and why

1. **Safety bypasses.** WRC's first-waypoint harness bypass (`SafeArm._make_hook` `first_seen`;
   the escape rule makes it unnecessary) and its e-stop auto-reset (b70f3a0 calls
   `harness.reset_estop()` before every task): `reset_stop` stays explicit and staff-only. In
   cascade Ctrl+C is halt + park, `StopSignals` defers signals during teardown, and a *repeated*
   Ctrl+C deliberately force-exits a stuck teardown (WRC's `suppress(KeyboardInterrupt)` would
   swallow that). A forced exit skips `disconnect()`: motors hold their last MIT command.
2. **"back to zero" / "zero position" / "reset" → home** (b70f3a0, 6b926fe, f916acc): zero is the
   park pose, "reset" is `reset_scene`, WRC deleted `move_to_joint_zero` itself, and shutdown
   already parks through the harness. **`scripts/emergency_stow.py`** and
   **`_calib_cleanup.safe_park_and_disconnect`**: teardown (`shutdown_runtime` → `_park_arm`,
   margin relaxed only; `lifecycle.teardown_step` never raises, always disconnects) already does
   this; a stow script would also need a watchdog decision (a fake heartbeat is a bypass).
3. **Gripper control.** The 10 ms re-send loop (ignores `effort`, overrides `hold_kp`, fights the
   50 Hz stream for the bus lock; MIT already holds the last command). The `WrcGripper` state
   machine (torque close at 1.5 N·m, stall on `|vel| < 0.12`): RS `get_state()` is not decoded on
   this firmware and mechVel (0x701A) is not rad/s; cascade has `close_gripper_torque` (stall on
   mechPos, then `hold_kp`) and the `air_grasp_frac` empty-grasp check. Width-aware close
   (`compute_close_target`, unsettled v2 0.75 → v5 0.30 at effort 1.0, a disagreeing width map):
   cascade keeps fixed fractions, bounded by the opt-in `gripper.max_contact_squeeze_rad`
   ([REBOT_GRIP_SQUEEZE_CAP.md](REBOT_GRIP_SQUEEZE_CAP.md)); ADR-0002 `force=1.0` is a torque-mode
   value with no position-mode equivalent.
4. **Poses and drivers.** WRC's `home_q` `[0, 0.1684, 0.6226, -0.4543, 0, 0]` (TCP 0.25/0/0.35):
   cascade keeps `[0, 1.2, 1.2, 0, 0, 0]` (TCP 0.452/0/0.438, pinned by
   `test_rebot_initial_pose.py`, matches Isaac, seeds grasp IK); WRC's is an onsite option.
   `kinematics.py` / `arm_base.py` diffs (USD removal; `settle_timeout_s` 2.0 → 6.0 in code)
   and the `make_arm` SDK stack (`RebotArmEndPose` + `WrcGripper` + `RebotGraspBridge`): cascade's
   drivers own the MIT stream so the harness sees every waypoint.
5. **Teach record/replay — remaining.** WRC drives the follower with raw `send_pos_vel` at
   `vlim=15 rad/s` and replays outside any harness; it needs `fashionstar-uart-sdk` and `pynput`.
   A port needs: recording through `harness.approve()` at stream rate with a lazy optional leader
   backend; replay through `SafeArm.move_joint_path`; skills added per the AGENTS.md checklist;
   a measured leader→follower joint map (WRC negates joints 1, 4, 6 uncalibrated).

## Constant discrepancies (WRC vs cascade `configs/arms/rebot_rs.yaml`)

| Quantity | WRC | cascade | Note |
|---|---|---|---|
| Gripper open angle | `_RS_ANGLE_OPEN = 5.0` rad (`open_pos: 5.0`) | `open_pos: 6.2` (travel 0 → +6.39 measured) | Same polarity, different scale |
| Gripper width at open | 0.09 m (`GRIPPER_MAX_DISTANCE_M`); `max_width_m: 0.095` | `max_width_m: 0.09` | 5.0 rad ≙ 0.09–0.095 m vs 6.2 rad ≙ 0.09 m: one map is wrong; measure before enabling `pregrasp_open_margin_m` |
| Gripper close | torque mode, `close_torque 1.5`, `tau_max 1.5` N·m | position mode, `kp 2.0` × effort | Different force models |
| Hold | `default_force 0.30` (ADR-0002: 1.0) torque | `hold_kp 0.01` position stiffness | Not comparable |
| Stall | `|vel| < 0.12` rad/s, startup travel 0.30 rad | mechPos change < 0.03 rad per 150 ms | cascade avoids undecoded velocity |
| Arm settle tolerance | 0.05 (final; 0.08–0.15 tried) | `settle_tol` default 0.05 | Same |
| Settle timeout | 6.0 s (code default) | 2.0 s (ArmBase), profile-overridable | Tune onsite only on "did not settle" |
| home_q | `[0, .1684, .6226, -.4543, 0, 0]` | `[0, 1.2, 1.2, 0, 0, 0]` | See above |
| Zero / park | `[0]*6` | `park_q [0]*6` (was −0.05 on joint 3) | Now the same |
| Joint frame | asset URDF with axes flipped, no `joint_signs` | URDF + `joint_signs [-1]*6` (local) | Equivalent (WRC home FK = 0.25/0/0.35 in cascade's local frame) |
| `max_joint_vel` | 1.2 (demo.yaml) | 0.8 (RS override) | cascade's lower cap is kept |

## Onsite hardware verification still required

Follow the `real-robot-bringup` phases. **Nothing below may run until a person is at the arm
with the e-stop or power cut within reach.** Support the arm before anything enables it: after
a clean shutdown it is limp. Never reconnect anything while the arm is raised: `connect()`
clears latched faults with a stop frame that cuts torque (support it, e-stop, lower by hand).

1. **Read-only probe (no torque).** Stop motorbridge-gateway and MotorBridge Studio (same host id
   0xFD). Run `python scripts/diag_rebot_mb.py`, then `--snapshot rest`; record joints 2 and 3
   (WRC +0.004, cascade's rig −0.0009) and confirm every joint is inside its URDF limits.
   **Incident 2026-10-10**: `sign_check_rebot_mb.py --joints 2,3` drove the shoulder into its
   end at full stiffness (a mechPos read of +2.3e18 became the stall-guard target, and joints
   1-3 were registered `rs-00` instead of `rs-06`). Both bring-up scripts now vet every reading
   (`robstride.plausible_position`), clamp targets to the probe envelope, ramp back from the
   last commanded pose (never a reading), stop holding on an implausible reading and refuse
   without the SDK's per-motor models (`tests/test_rebot_probe_reading_guard.py`). Do not loosen
   any of these.
2. **Fault clear.** `python scripts/jog_rebot_mb.py --joint 1 --hold-only` (clears faults after
   the mode write, before enable): no `clear_error failed`, hold drift under 50 mrad. A failed
   clear: stop, power-cycle, probe again.
3. **First low-velocity home move, e-stop in hand.** Either lower
   `overrides.safety.max_joint_vel` in `rebot_rs.yaml` (e.g. 0.3; only ever lower), run
   `python -m cascade.apps.demo --arm rebot_rs --camera mock --llm mock --interactive --no-serve`
   and type `go to the initial pose` (a reflex; `go home` is `move_to_rest`), or run the scripted
   `python scripts/home_rebot_rs.py --yes` (rest -> home -> rest at 0.4 rad/s, 4 measured steps
   per leg, soft-stop and hold with torque ON on a failed check; rehearse with `--dry-run
   --camera mock`). The move must start without a harness rejection (Finding #3); exit with an
   empty line or one Ctrl+C and confirm the receipt shows park `complete` at `park_q = 0`.
   Measured 2026-10-11: every step within 0.013 rad, home (0.452, -0.001, 0.432) m vs nominal
   (0.452, 0, 0.438). The 5 s perception watchdog refuses motion without a running fixed camera
   (it stopped half-way up). Joint 3 cannot be sign-checked from rest (~6.7 N·m of gravity vs
   the probe's 3.6 N·m abort): with motors limp, unfold the forearm by hand and check
   `diag_rebot_mb.py --snapshot` rises.
4. **Gripper travel and stall.** Hand-sweep with `diag_rebot_mb.py` (0 → +6.39 rad); measure the
   opening in mm at ~1.5, 3.0, 4.5, 6.2 rad to settle the 5.0-vs-6.2 rad ≙ 90 mm disagreement and
   fix `max_width_m` / `open_pos`; run the park close (`close_gripper_torque`) on air and on a
   ~30 mm object (`[rebot] grip contact` only for the object). Only then consider
   `gripper.pregrasp_open_margin_m: 0.01`, and run the squeeze protocol in
   [REBOT_GRIP_SQUEEZE_CAP.md](REBOT_GRIP_SQUEEZE_CAP.md) before setting
   `gripper.max_contact_squeeze_rad`.
5. **Cartesian nudge and settle (optional).** At the lowered cap, set
   `cartesian_relative_moves: true`, issue `move down a bit` / `move up a bit` and watch the TCP
   travel straight (revert on doubt). If large moves report "did not settle" while visibly
   moving, raise `settle_timeout_s` in the profile (WRC needed 4–6 s); never touch `settle_tol`
   or the velocity cap to compensate.
