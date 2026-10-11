# Contact stop: motor-feedback collision detection

The arm stops, holds where it is, and reports a `SafetyViolation` when it is
blocked mid-motion by something nobody modelled: an occluded object, a thin
one, anything in the wrist camera's blind zone. It is the second safety layer
under the camera-based obstacle checks (occupancy / nvblox), and it uses
nothing but the joint feedback the motors already give.

## Why

Rig incident, 2026-10-11: a `grasp_object` with a side approach drove the
wrist and the wrist camera into clutter beside the target. The arm pressed
against it for about 8 s until the end-of-move settle check reported
`SkillError: did not settle at grasp pose; pre-close retreat completed`.
Nothing compared the measured joints with the command while the motion
streamed.

## What it does

`cascade.safety.contact.ContactMonitor` checks every streamed motion, per
joint, against the commanded trajectory:

- **Tracking band.** The measured position must lie inside the range the
  command swept over the last `lag_s` seconds. The RS MIT loop runs without
  velocity feedforward, so a joint trails a moving setpoint by about kd/kp
  (60-100 ms with the SDK gains). The distance outside that band is the
  lag-compensated tracking error, and it must stay below `track_tol_rad`.
- **Stall.** Over `stall_window_s`, a joint whose command advanced at least
  `stall_min_travel_rad`, in the same direction under every lag in
  `[0, lag_s]`, must have advanced at least `stall_ratio` of that distance. A
  blocked joint stops while its command keeps moving, well before the band
  opens. Requiring every lag to agree keeps a joint that reverses mid-path
  (Cartesian lines) from reading as stalled.
- **Debounce.** A trip needs `debounce_samples` consecutive bad samples of
  one joint.
- **Fail safe.** If a joint has no good sample for `max_feedback_gap_s`, the
  stream stops with a plain `SafetyViolation` ("lost joint J feedback"). A
  reading that cannot be a position, or that is farther from the last one
  than the arm can travel (6 rad/s + 0.15 rad), counts as a read failure,
  never as a contact.

On a trip the stream stops before the next target goes out, and
`ContactDetected` (a `SafetyViolation` subclass) propagates:

```
contact detected at joint 4: tracking error 0.081 rad for 60 ms -- something is blocking the arm (stalled: moved +0.000 rad while commanded +0.062 rad in 0.12 s)
```

### Hold semantics

The same as `scripts/home_rebot_rs.py`'s soft stop: torque stays on and
nothing is ever cut. After a contact, the arm is commanded to hold its
**current measured pose**, never the target it could not reach. That pose
comes from a fresh full read, checked against the monitor's own samples. If
that read fails, is not finite, or disagrees with the samples, no hold is
sent and the motors keep the last streamed target, as on every other abort.
Because the hold goes out as a normal joint target, a later soft `stop()`
re-asserts the hold rather than the command the arm was pressing toward.
When feedback is lost the last command stands, because there is no
measurement to hold. The gripper is not monitored: it is not one of the arm
joints, so jaws closing on an object can never trip the monitor.

### No re-planning into the same obstacle

- A contact fences the spot. The harness keeps a **contact zone** of
  `contact_zone_m` (default 5 cm) around the TCP at the measured hold pose.
  `approve()` and `vet_step()` refuse any step whose TCP *enters* a zone.
  Leaving is always allowed: an arm already inside may move anywhere, so the
  grasp's pre-close retreat and `move_home` still work. `vet_pose()` refuses
  candidates inside a zone, so grasp selection and planned routes do not head
  back there. `reset_scene` clears the zones; it is the operator's "I removed
  the obstacle". `contact_zone_m: 0` turns the zones off but keeps the stop.
- `pick_and_place`, `handover` and `sort_by_color` make no further attempt
  after a contact: no re-scan, no re-plan, no second grasp. This applies to
  the grasp stage and the place stage (`_grasp_retry_verdict` and the place
  loop). The result is `outcome: stuck`, with an `ask` telling the human
  that something blocked the arm, to clear it, and to say "reset".
- A skill that fails on the contact itself returns
  `error: "ContactDetected: contact detected at joint J: ..."`, plus
  `contact` (joint, error, phase, hold pose, zone) and a `hint` telling the
  planner not to retry the same motion. This covers `grasp_object`,
  `move_home`, `move_relative` and `place_at`, all tested. The MCP tool
  result carries the same JSON (`isError`).
- The grasp descent keeps its existing recovery: after a contact it withdraws
  to the vetted pregrasp (`...; pre-close retreat completed`), still under
  the monitor.

### Grasp descent contact

The descent must still settle within the arm's `settle_tol` (0.05 rad on the
RS), as before. Pressing lightly into the table or the object, within that
tolerance, passes the settle check. The monitor's tolerance has to be larger
than `settle_tol` (`tests/test_contact_profiles.py` pins this for every
profile that enables it), so the monitor never trips on a final error the
settle check would accept. What changes is a hard press: before, the arm
pushed until the settle timeout; now the stream stops within a fraction of a
second.

## Where it runs

- `SafeArm.move_joints` (and through it `move_planned`, `move_home`, grasp
  approach, descent, lift and retreat, place, nudges) and
  `SafeArm.move_joint_path` (`move_cartesian`) build one monitor per streamed
  motion and pass it to the backend as `contact_monitor`. Once enabled it is
  mandatory: a backend that cannot take it refuses before the first command,
  and so does a planner-bound (cuMotion) arm.
- `ArmBase._stream_ticks` (the 50 Hz loop shared by `stream_to` and
  `stream_path`) runs `ContactFeedback` after every target it sends. It
  records the target, then reads joints until the next target is due.
- The **RS drivers** (`rebot_rs`, `rebot_rs_mb`) read one motor's mechPos per
  call, with a 50 ms timeout instead of motorbridge's 1000 ms default, because
  a full pose is six reads of about 8 ms each with the driver lock held. The
  loop reads as many joints as fit before the next target (an EMA of the read
  cost plus a 1 ms guard), round-robin, and at least `min_reads_per_cycle`.
  At 8 ms per read that is 1 to 2 joints per 20 ms cycle, so each joint is
  sampled every 60-120 ms (8-17 Hz per joint). Backends without a per-joint
  reader (mock, sim) do one full `get_state()` per cycle.
- **Settle phase:** after the last tick, every 50 ms poll uses a fresh full
  read (`read_positions_for_monitor()`, which raises instead of serving a
  cached pose) and goes through the monitor. This covers the incident's last
  stretch, where the stream had ended and the arm pressed until settle
  timeout.
- **Reports:** every motion's summary is filed on the harness. The skill
  runtime adds them to that call's `trace.jsonl` row as
  `context.contact_monitor`. One `[contact] ...` line goes to stderr in
  measure mode, with `log_motions: true`, and whenever a motion tripped,
  would have tripped, or lost feedback.

## Configuration

`safety.contact_monitor` in the arm profile (under `overrides:` for arm
profiles). A malformed block, an unknown key or a bad value is a config error
at startup.

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

Shipped:

- `rebot_rs`, and `rebot_rs_reach`, which extends it: `enforce` mode, with
  every key written out in the YAML.
- `rebot_rs_mb`: `measure` only. Its MIT gains are unvalidated away from the
  zero pose, and that backend has no gravity compensation, so it logs until
  someone calibrates it.
- Every mock, sim and other profile leaves the monitor off. The demo's
  `safety` block in `configs/demo.yaml` documents it, disabled.

`tests/test_contact_profiles.py` pins these choices, and pins the invariants
for every enabled profile: a tolerance of at least `settle_tol` + 0.02 and at
least 7 times the rig's 0.013 rad, `lag_s` of at least 0.2 s, a debounce of at
least 2, and a feedback gap of at least 0.3 s.

## Expected detection latency

These figures come from the offline simulation in
`tests/test_contact_stream.py`: a PD plant with the RS MIT gains, guessed link
inertias, 8 ms reads and a rigid obstacle. They are not rig measurements.

- **Moving joint blocked** (the stall rule, joint commanded at 0.25 rad/s or
  faster): the stream stops 0.11-0.20 s after the joint hits the obstacle,
  for every joint of the arm. The joint was blocked at 20, 50 and 80% of a
  0.8 rad move; the test asserts at most 0.25 s.
- **Slow joint or settle phase** (band rule): `lag_s` plus `track_tol_rad`
  divided by the joint's commanded speed, plus 1-2 samples. For a joint held
  short of a stopped command, that is about 0.25 s plus 2 settle polls,
  roughly 0.4 s.
- **Then the hold:** one full read, about 48 ms on the RS, and one MIT
  command.

## False-trip risks

- The lag model is a guess until measured. If the real arm trails by more
  than `lag_s`, or has a larger static error than expected (droop under a
  heavy payload, a gravity-compensation mismatch), the band trips on free
  motion. The default band is wide: 0.10 rad, against the 0.013 rad settled
  error measured on 2026-10-11. In simulation, over 30 random multi-joint
  moves at the speed cap, the free-motion error was:

  - 0.002 rad with the guessed link inertias;
  - 0.014 rad with three times those inertias.

  The raw error against the instantaneous command reached 0.065 rad; that is
  the lag the band absorbs.
- Underdamped joints. The SDK gains damp the base joint lightly (kd 3 on kp
  50). In the same simulation with ten times the guessed inertias, its
  overshoot at the end of a move tripped 1 of the 30 moves (0.111 rad).
  Measure mode on the rig shows whether the real arm comes close: watch
  joint 1's `max_track_err_rad`.
- Stiction or backlash at motion start could look like a stall. The stall
  rule only considers commands that moved at least 0.03 rad in 0.12 s, and
  requires a 5x shortfall. In simulation, the lowest advance ratio of normal
  motion was 1.0 against the 0.2 threshold.
- Deliberate contact skills (pushing, screwing) see real reaction torque. A
  push that has to deflect a joint by more than `track_tol_rad` will stop. No
  shipped RS skill presses that hard.
- Bus trouble: two consecutive failed reads of one joint (~0.35 s blind)
  stop the motion. That is safe, but it would show up as aborted moves on a
  flaky CAN bus.

## On-rig calibration

Do this before trusting the shipped numbers. The operator stays at the
e-stop.

1. Set `mode: measure` in `configs/arms/rebot_rs.yaml` under
   `overrides.safety.contact_monitor`, and keep `enabled: true`. The new
   config takes effect when the MCP server is restarted (`~/cascade-demo.sh
   reset`).
2. Run normal moves in a clear workspace:
   - `scripts/home_rebot_rs.py --yes --contact-log rig.jsonl`
     (rest -> home -> rest, 8 slow segments);
   - `move_home`, a few `move_relative` nudges, two or three `grasp_object`
     / `pick_and_place` runs (top-down and angled);
   - at least one move that runs at the 0.8 rad/s cap.
3. Summarise:

   ```
   python scripts/contact_calibrate.py rig.jsonl runs/*/trace.jsonl
   ```

   For each joint it prints:

   - the worst lag-compensated error;
   - the worst raw error against the instantaneous command;
   - the longest lag behind the command;
   - the lowest stall advance ratio.

   It also prints the read rate, the feedback gaps, and every motion that
   WOULD have tripped. It then suggests values:

   - `lag_s`: 1.5 times the longest lag seen, between 0.10 and 0.40 s.
   - `track_tol_rad`, per joint: 3 times the worst error, never below
     `settle_tol` + 0.02.

   If it suggests a shorter `lag_s` than the one the moves were measured
   at, repeat step 2 with the shorter lag before trusting the tolerances.
   Every WOULD_TRIP must have an explanation. A joint whose advance ratio
   came within 2 times of `stall_ratio` is flagged. The stderr
   `[contact] measure ...` lines carry the same maxima for a quick look.
4. Put the suggested numbers in the profile, set `mode: enforce`, restart.
5. Check a trip, deliberately: a slow `move_relative` toward a soft obstacle
   held by hand (a foam block). The arm must stop, hold and report
   `contact detected at joint J`. Then `reset_scene`.

Tighter numbers mean less force at the obstacle. The force at a trip is about
kp x error: 150 x 0.10 rad = 15 Nm on the shoulder and elbow, 5 Nm on the
wrist with the SDK gains. A calibrated 0.04 rad cuts that by 2.5x.

## Tests

- `tests/test_contact_monitor.py`: the detection rules on a simulated clock
  (trips, debounce, lag, reversals, drift, lost feedback, implausible
  readings, measure mode).
- `tests/test_contact_stream.py`: the executors. MockArm with
  `block_joint`; ArmBase's paced loop on a fake clock with the RS PD plant
  (latency for every joint, hold pose, settle phase, read budget, untrusted
  hold reads).
- `tests/test_contact_safearm.py`: SafeArm/harness wiring, the backend-hint
  retry, refusal paths, contact zones.
- `tests/test_contact_rebot_rs.py`: both RS transports on fake SDK motors.
- `tests/test_contact_runtime.py`: skill results, the MCP tool result, the
  trace, the grasp descent, no retries in `pick_and_place`, `reset_scene`.
- `tests/test_contact_profiles.py`: the shipped profiles.
- `tests/test_contact_calibrate.py` and `tests/test_home_rebot_rs.py`: the
  calibration tool, and the home script's contact stop and `--contact-log`.

The guards are mutation-checked: disabling any comparison fails a test.
