# Bounded lift-first joint routes

`SafetyHarness` lets an arm that already sits inside a clearance margin move
only away from the obstacle: the TCP and joint origins by a strictly
retreating step, the tool volume by a step that brings no sample closer (see
"Escape and route end" below). A route generator can still miss that escape:
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

## Escape and route end (TCP and joint origins)

1. **Escape, per step** (`approve`, `vet_step`): a TCP or joint-origin point
   below `min_clearance_m` (0.03 m) at the next pose passes only if it was
   already below it at the previous pose and its map clearance grows by more
   than `CLEARANCE_ESCAPE_TOL_M` (1e-9 m), or the step is an exact hold
   (joints within 1e-9 rad) that lowers no such point's clearance. Such a
   step is also sampled between its ends (FK, at most half a voxel apart, at
   most 16 samples): no escaping point may read less there than at the
   previous pose, so one long step cannot jump a voxel. The ends and the
   interior come from one map read. A lateral slide, a point entering the
   margin, or an unobserved (NaN) clearance at either pose refuses. The
   refusal keeps its text and adds "(inside the margin: only a strictly
   retreating step is allowed)".
2. **Route end**: `vet_route` (so `plan_route`, `move_planned`, and the up-leg
   of the rest move, `plan_route` to the initial pose) and the spatial
   `plan_route` refuse a route whose last pose fails the static check,
   `route ends inside the margin: ...`. The rest move's fold (`vet_fold`)
   gets the per-step escape only.
3. **Not changed**: the margin; static vets (`vet_pose`, grasp candidates) and
   attached-payload samples never escape; the tool volume keeps its own
   non-worsening rule; no planner searches for an escape beyond the lifts.

Tests: `tests/test_clearance_escape.py`.

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
