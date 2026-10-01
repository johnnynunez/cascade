# Retained attachment during NV transport

The normal-five run in environment 12 failed on its first object. Orange had
bilateral support through witness sequence 1529 and lost it at sequence 1530
(monotonic 113973.297737, physics step 177098), during horizontal transport.
The release guard later rejected opening. Its producer epoch remained unchanged.
This is distinct from the earlier successful release-recovery diagnostic in the
same environment. Four objects were not attempted; the campaign's planned reset
completed. Neither result establishes five-object acceptance.

The offline correction stops subsequent transport targets when the retained
attachment is lost or unavailable. It does **not** explain or prevent mechanical
slip, increase grip force, or change friction, clearance, trajectories, speed,
camera cadence, RPC deadlines or motion budgets.

## Scope and state contract

Only Isaac's explicit attachment-feedback capability together with the original
NV post-close payload barrier arms this guard. Expected object paths, producer
epoch, source, robot, map, scene-reset generation and cancellation generation
remain bound to that episode. Isaac with `occupancy=none`, mock, MuJoCo and legacy
hardware do not acquire a new contact-sensor requirement. This is NV coverage,
not a claim that the occupancy-off presenter or every backend now detects slips.

The bridge reuses contacts already captured after each existing completed update
with `CASCADE_ISAAC_CONTACT_MASK=1`. A `state` response resolves that history by
the SDK rational clock and checks exact physics step, simulation time, epoch,
asset joint positions and individual jaws against the current state read.
Ambiguous history, tracking disabled, sensor error or mismatched state yields
unavailable attachment evidence, never an empty usable attachment. Sensor channel
and step are explicit. No sensor constructor, extra tensor read, render, physics
update or camera request is added by the state accessor. The client binds source
locally and checks asset joints before applying its existing sign conversion.

The guard checks existing feedback before the initial lift, pre-carry lift,
horizontal transfer and lowering, before subsequent targets, and during settling.
It also checks SafeArm's existing duration-stretch read and release preparation's
existing state read. No extra RPC is issued for the guard. State acquisition and
the executor retain their existing deadlines. Frozen physics cannot authorize a
burst of commands; same-step contradictory joints or jaws fail closed.

An observed empty or changed bilateral attachment is terminal. Missing or invalid
evidence is also terminal, with status unknown rather than a physical-loss claim.
Both preserve the internal held/provisional marker as a precaution, report
`grip_verified: false`, and suppress automatic opening, retry, homing and reset.
The latch is not removed by a later good sample or convenience retry. Recovery
after this new failure requires separate evidence and explicit authority; this
change does not implement that recovery. A fresh empty contact sample proves
absence of bilateral support, not the absence of all unilateral contact or the
mechanical cause of loss.

Native intentional release ends carry scope after its own checks and opening
acknowledgement. The existing release episode then owns opening feedback, map
barriers and the exact withdrawal/recovery. Its valid explicit reset contract
is unchanged. A successful existing scene reset clears a still-valid carry
token; a latched carry failure blocks that path before movement.

## Validation and evidence limits

CPU tests cover poisoned history, exact state/clock/jaw binding, source and epoch
changes, asset signs, absent tracking, loss before and between targets, settling,
the initial lift and all three placement legs. They also cover preserved failure
across calls, malformed initial authority, callback cancellation/no retry, and
legacy/non-payload compatibility. Existing release/opening and observed-finger
regressions are included in focused validation. Full-suite and physical results
must be attributed to their frozen source receipts; this document makes no new
physical-success claim.

The saved environment-12 witness supports abrupt loss of support during carry.
It does not identify a unique mechanical cause. Recorded neighboring motion
begins after loss of bilateral attachment, so it does not by itself establish
that a neighbor caused the initial slip.
