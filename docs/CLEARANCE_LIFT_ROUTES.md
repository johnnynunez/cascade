# Bounded lift-first joint routes

`SafetyHarness` already permits non-worsening escape from an existing
clearance-margin violation. A route generator can still miss that escape:
a direct joint interpolation and one-joint-first/last corners do not explore
all coupled joint movements.

`cascade.safety.trajectory.plan_route` now keeps those original candidates
in their original order, then uses any remaining candidate slots for up to
two model-only TCP targets: **+3 cm and +6 cm in robot-base Z**, preserving
the start orientation and XY. IK uses the measured start as its only seed,
80 iterations and no restart seeds. No actuator is queried or commanded by
this planner.

A successful IK endpoint is only a proposal. The complete start-to-lift and
lift-to-goal joint routes must pass the existing dense `vet_step` preflight.
A raised endpoint does **not** guarantee that the intervening joint path is
vertical or collision-free. Execution retains its own preflight and
per-streamed-step checks, including newly appearing obstacles.

## Bounds and unchanged behavior

- Existing successful routes are returned unchanged, without calling lift IK.
- The existing 15-candidate cap includes failed lift attempts. Seven changed
  joints can consume every slot before a lift is considered.
- The same three-second planning deadline covers model work and preflight.
  It is checked before and after IK; an individual solver call is not
  preemptible, but a result arriving after the deadline cannot authorize motion.
- The existing geometry lock covers candidate generation and whole-route
  preflight. Missing model FK/IK preserves the original refusal.
- No workspace, clearance, velocity, joint, contact, occupancy-freshness or
  stop limits change. No new obstacle exemption or unconditional home command
  is introduced. Robot profiles, camera masks and calibration are unchanged.

## Evidence and limitations

`tests/test_clearance_lift_route.py` exercises synthetic obstacles through the
real harness, joint profiles and mock executor. It also uses the signed
`rebot_rs` model and shipped wrist-camera volume with an **invented obstacle**:
the initial model clearance rounds to 26 mm against the unchanged 30 mm
limit. One direction admits a fully vetted lift-first route; the opposite
direction remains refused because its second leg is unsafe. This fixture is
not a reconstruction of the physical cup/basket incident.

Regression coverage includes unsafe intermediate paths despite safe endpoints,
invalid and failed IK, preserved orientation and seed, exhausted work/time
budgets, the geometry lock, and obstacles appearing before/during streaming.

This is an offline-tested search improvement, not physical acceptance. It does
not fix unreachable placement targets, robot self-mask leakage, stale maps,
unreadable joints or calibration errors. Residual robot points require a
separate masking diagnosis; deleting nearby obstacles or lowering the margin
is not part of this change. Deployment and supervised hardware validation are
separate steps.
