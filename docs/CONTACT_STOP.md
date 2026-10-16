# Contact stop: motor-feedback collision detection

**Next action:** run the contact-stop tests offline (about 1 min, no robot):

```bash
uv run --extra dev --extra kinematics pytest tests/test_contact_*.py tests/test_home_rebot_rs.py -q
```

The arm stops, holds where it is and reports a `SafetyViolation` when something nobody modelled
blocks it mid-motion (an occluded or thin object, anything in the wrist camera's blind zone). It is
the second safety layer under the camera-based obstacle checks (occupancy / nvblox) and uses only
the joint feedback the motors already give. Origin: on 2026-10-11 a side-approach `grasp_object`
pressed the wrist camera into clutter for ~8 s until `SkillError: did not settle at grasp pose;
pre-close retreat completed`, because nothing compared measured joints with the command while the
motion streamed.

Status: enforced on `rebot_rs` / `rebot_rs_reach`, measure-only on `rebot_rs_mb`, off everywhere
else. The latency figures below are offline simulation, not rig measurements: **do the
[on-rig calibration](#on-rig-calibration) before trusting the shipped numbers.**

## What it does

`cascade.safety.contact.ContactMonitor` checks every streamed motion, per joint, against the
commanded trajectory:

1. **Tracking band:** the measured position must lie inside the range the command swept over the
   last `lag_s` (the RS MIT loop has no velocity feedforward, so a joint trails by about kd/kp,
   60-100 ms with the SDK gains); the distance outside the band must stay below `track_tol_rad`.
2. **Stall:** over `stall_window_s`, a joint whose command advanced at least
   `stall_min_travel_rad` in the same direction under every lag in `[0, lag_s]` must have advanced
   at least `stall_ratio` of it (every lag must agree, so a mid-path reversal is not a stall).
3. **Debounce:** a trip needs `debounce_samples` consecutive bad samples of one joint.
4. **Fail safe:** no good sample of a joint for `max_feedback_gap_s` stops the stream with a plain
   `SafetyViolation` ("lost joint J feedback"); an impossible reading (not a position, or farther
   than 6 rad/s + 0.15 rad from the last) is a read failure, never a contact.
5. On a trip the stream stops before the next target goes out and `ContactDetected` (a
   `SafetyViolation` subclass) propagates:

```
contact detected at joint 4: tracking error 0.081 rad for 60 ms -- something is blocking the arm (stalled: moved +0.000 rad while commanded +0.062 rad in 0.12 s)
```

### Hold semantics

As `scripts/home_rebot_rs.py`'s soft stop: torque stays on, nothing is cut. After a contact the arm
holds its **current measured pose** (a fresh full read checked against the monitor's samples),
never the target it could not reach; if that read fails, is not finite or disagrees, no hold is sent
and the motors keep the last streamed target. A later soft `stop()` re-asserts the hold. When
feedback is lost the last command stands. The gripper is not monitored, so jaws closing on an object
never trip it.

### No re-planning into the same obstacle

1. **Contact zone:** the harness fences `contact_zone_m` (default 5 cm) around the TCP at the hold
   pose. `approve()` and `vet_step()` refuse steps that *enter* a zone (leaving is always allowed,
   so the pre-close retreat and `move_home` still work); `vet_pose()` refuses candidates inside.
   `reset_scene` clears the zones (the operator's "I removed the obstacle"); `contact_zone_m: 0`
   disables zones but keeps the stop.
2. `pick_and_place`, `handover` and `sort_by_color` make no further attempt after a contact, in the
   grasp and the place stage (`_grasp_retry_verdict` and the place loop): `outcome: stuck` with an
   `ask` to clear the obstacle and say "reset".
3. A skill failing on the contact returns `error: "ContactDetected: contact detected at joint J:
   ..."`, a `contact` object (joint, error, phase, hold pose, zone) and a `hint` not to retry the
   same motion (`grasp_object`, `move_home`, `move_relative`, `place_at`; MCP results carry the same
   JSON with `isError`).
4. The grasp descent keeps its recovery: it withdraws to the vetted pregrasp (`...; pre-close
   retreat completed`), still monitored.
5. **Grasp descent contact:** the descent must still settle within `settle_tol` (0.05 rad on the RS),
   and the monitor's tolerance must exceed it (`tests/test_contact_profiles.py`), so a light press
   the settle check accepts never trips; a hard press now stops within a fraction of a second.

## Where it runs

1. `SafeArm.move_joints` (so `move_planned`, `move_home`, grasp approach/descent/lift/retreat,
   place, nudges) and `SafeArm.move_joint_path` (`move_cartesian`) build one monitor per streamed
   motion and pass it as `contact_monitor`. Once enabled it is mandatory: a backend that cannot take
   it, or a planner-bound (cuMotion) arm, refuses before the first command.
2. `ArmBase._stream_ticks` (the 50 Hz loop of `stream_to` / `stream_path`) runs `ContactFeedback`
   after every target, reading joints until the next target is due.
3. The **RS drivers** (`rebot_rs`, `rebot_rs_mb`) read one motor's mechPos per call with a 50 ms
   timeout (motorbridge's default is 1000 ms), round-robin, as many as fit before the next target
   and at least `min_reads_per_cycle`: at ~8 ms per read each joint is sampled every 60-120 ms
   (8-17 Hz). Mock and sim backends do one full `get_state()` per cycle.
4. **Settle phase:** every 50 ms poll after the last tick uses a fresh full read
   (`read_positions_for_monitor()`, which raises instead of serving a cached pose) through the
   monitor.
5. **Reports:** each motion's summary goes into that call's `trace.jsonl` row as
   `context.contact_monitor`; a `[contact] ...` stderr line is printed in measure mode, with
   `log_motions: true`, and whenever a motion tripped, would have tripped or lost feedback.

## Configuration

`safety.contact_monitor` in the arm profile (under `overrides:`). A malformed block, unknown key or
bad value is a startup config error.

| key | default | meaning |
|---|---|---|
| `enabled` | `false` | read feedback mid-stream at all |
| `mode` | `enforce` | `enforce` stops; `measure` only logs and never stops |
| `track_tol_rad` | `0.10` | lag-compensated tolerance, scalar or one per joint |
| `lag_s` | `0.25` | how far the measured joint may trail the command |
| `debounce_samples` | `2` | consecutive bad samples of one joint before tripping |
| `stall_window_s` | `0.12` | stall look-back |
| `stall_min_travel_rad` | `0.03` | commanded advance needed before stall applies (`null` = band only) |
| `stall_ratio` | `0.2` | stalled when it advanced at most this fraction |
| `max_feedback_gap_s` | `0.35` | no good sample of a joint this long = stop |
| `read_timeout_s` | `0.05` | per single-motor read (RS) |
| `min_reads_per_cycle` | `1` | joint reads per cycle even without slack |
| `contact_zone_m` | `0.05` | fence radius around the TCP after a contact (`0` = off) |
| `log_motions` | `false` | enforce mode: log every motion, not only trips |

Shipped: `rebot_rs` (and `rebot_rs_reach`) `enforce` with every key written out; `rebot_rs_mb`
`measure` only (MIT gains unvalidated away from zero, no gravity compensation); every mock, sim and
other profile off (`configs/demo.yaml` documents it disabled). `tests/test_contact_profiles.py`
pins these and, for every enabled profile, a tolerance ≥ `settle_tol` + 0.02 and ≥ 7× the rig's
0.013 rad, `lag_s` ≥ 0.2 s, debounce ≥ 2 and a feedback gap ≥ 0.3 s.

## Expected detection latency (offline simulation)

From `tests/test_contact_stream.py` (PD plant with the RS MIT gains, guessed inertias, 8 ms reads,
rigid obstacle):

1. **Moving joint blocked** (stall rule, ≥ 0.25 rad/s): the stream stops 0.11-0.20 s after impact for
   every joint (blocked at 20, 50 and 80 % of a 0.8 rad move; the test asserts ≤ 0.25 s).
2. **Slow joint or settle phase** (band rule): `lag_s` + `track_tol_rad` / commanded speed + 1-2
   samples; held short of a stopped command, roughly 0.4 s.
3. **Then the hold:** one full read (~48 ms on the RS) and one MIT command.

## False-trip risks

1. The lag model is a guess until measured: droop, a gravity-compensation mismatch or more lag than
   `lag_s` trips the band on free motion. The default band (0.10 rad) is wide against the 0.013 rad
   settled error measured on 2026-10-11; simulated free motion reached 0.002 rad (0.014 rad with 3×
   inertia).
2. Underdamped joints: the base joint (kd 3 on kp 50) tripped 1 of 30 simulated moves at 10× inertia
   (0.111 rad); watch joint 1's `max_track_err_rad` in measure mode.
3. Stiction or backlash at motion start: the stall rule needs ≥ 0.03 rad of command in 0.12 s and a
   5× shortfall (normal motion's lowest ratio was 1.0 vs 0.2).
4. Deliberate contact skills (pushing, screwing) that deflect a joint more than `track_tol_rad` will
   stop; no shipped RS skill presses that hard.
5. Bus trouble: two consecutive failed reads of one joint (~0.35 s blind) stop the motion (safe, but
   shows as aborted moves on a flaky CAN bus).

## On-rig calibration

The operator stays at the e-stop throughout (about 30 min). Rig commands run from the repo root
with the rig venv's `.venv/bin/python` (the rig has no bare `python`, and `uv run` would re-sync the
live `.venv`).

1. Set `mode: measure` (keep `enabled: true`) in `configs/arms/rebot_rs.yaml` under
   `overrides.safety.contact_monitor`; restart the MCP server (`~/cascade-demo.sh reset`).
2. Run normal moves in a clear workspace: `.venv/bin/python scripts/home_rebot_rs.py --yes
   --contact-log rig.jsonl` (rest -> home -> rest, 8 slow segments), `move_home`, a few
   `move_relative` nudges, two or three `grasp_object` / `pick_and_place` runs (top-down and
   angled) and at least one move at the 0.8 rad/s cap.
3. Summarise: `.venv/bin/python scripts/contact_calibrate.py rig.jsonl runs/*/trace.jsonl`. Per joint
   it prints the worst lag-compensated and raw errors, the longest lag and the lowest stall advance
   ratio, plus read rate, feedback gaps and every motion that WOULD have tripped; it suggests
   `lag_s` (1.5× the longest lag, 0.10-0.40 s) and per-joint `track_tol_rad` (3× the worst error,
   never below `settle_tol` + 0.02). If it suggests a shorter `lag_s`, repeat step 2 with it. Every
   WOULD_TRIP needs an explanation; joints within 2× of `stall_ratio` are flagged.
4. Put the suggested numbers in the profile, set `mode: enforce`, restart.
5. Check a trip deliberately: a slow `move_relative` toward a foam block held by hand must stop, hold
   and report `contact detected at joint J`; then `reset_scene`.

Tighter numbers mean less force at the obstacle: about kp × error, 150 × 0.10 rad = 15 Nm on the
shoulder and elbow and 5 Nm on the wrist with the SDK gains; a calibrated 0.04 rad cuts that 2.5×.

## Tests

`tests/test_contact_monitor.py` (detection rules on a simulated clock), `test_contact_stream.py`
(executors, RS PD plant, hold pose, settle phase, read budget), `test_contact_safearm.py` (wiring,
refusals, contact zones), `test_contact_rebot_rs.py` (both RS transports on fake motors),
`test_contact_runtime.py` (skill/MCP results, trace, grasp descent, no retries, `reset_scene`),
`test_contact_profiles.py` (shipped profiles), `test_contact_calibrate.py` and
`test_home_rebot_rs.py`. The guards are mutation-checked: disabling any comparison fails a test.
