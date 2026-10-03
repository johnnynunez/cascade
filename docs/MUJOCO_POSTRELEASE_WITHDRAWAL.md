# Measured withdrawal after release

The region2 episode on e2d89c8 remains failed. Opening changed the red cube's
position and orientation: the pre-opening `joint_0_first` escape no longer
cleared the moving jaw. Its actual-state check stopped before sending any
retreat waypoint, skipped home and left the task unverified. The original
two-pick test did not attempt blue. No tolerance, region or contact parameter
was changed to reinterpret that outcome.

The optional MuJoCo adapter now plans again after measured opening and the
held-object handoff, using copied final joint/object coordinates and measured
jaw position. The existing finite full-pose and joint-corner candidates must
pass their entire escape and subsequent home path, under the original3s
computation budget. All geometry runs on detached MjData; no integration or
object pose write is part of planning. The ordinary driver still controls
every executed joint waypoint.

The retained SafetyHarness fence remains the same object and same generation.
A copied planning context cannot clear it or authorize unrelated commands.
Model, coordinate mapping, data object, optional region-history epoch and
cancellation token remain bound. A stop followed by reset cannot renew this
context. The digest of `time`, `qpos`, `qvel` and `ctrl` must stay unchanged through planning and
up to stream preflight; drift refuses without a retry. Any opening, planning,
transport or verification failure retains the pending withdrawal. Explicit
reset recovery keeps its separate observed-reset contract.

The coordinate binding includes the driver's joint count; joint names and
qpos/qvel addresses; actuator names and control indices; jaw joint/actuator
names and addresses; open/closed positions; jaw control scale/offset; and the
runtime's opening/width units. Initial indices must resolve the declared model
names. Engine, model, data, world lock and control-buffer identity/layout stay
bound to that driver. The cheap coordinate/channel check also runs through
the common actuation fence, including an already-entered transmission scope;
it adds no geometry calculation or solver call per waypoint.

The first review checkpoint omitted velocity/actuator mappings and jaw affine
control. Its source and 78-test receipt remain preserved, together with three
guard-only adversaries that exposed the omission. They are not evidence of
admission. The repaired tests explicitly model matching driver/runtime jaw
units rather than supplying an unrelated synthetic open position.

The receipt includes the previous plan and the measured replanning context;
its target is the attempted geometric goal, not a claim that the robot reached
it. `physical_task_verdict` remains false. Subsequent ordinary placement,
prefix and support/rest verification are still required.

The captured causal fixture binds phases716/718 from the failed source:
at43.5s the old route passes native collider replay; at44.1s it crosses from
0.847675mm clearance into0.169248mm penetration between `moving_jaw_box2` and
`red_cube`. The same planner from the measured later state finds
`joint_3_last`. This is static feasibility only. The replay prohibits
integration and does not reclassify the failed physical episode. Phase-observer
contact/site data are cached solve outputs; the scratch geometry uses the
separately saved final integrated qpos.

The failed episode has no persisted region journal: the failure path returned
unverified before requesting a placement proof. Its last cached solve reports
support contacts, but does not establish a supported-rest window. Persisting
diagnostic owner-produced journal rows on failure is a separate follow-up;
such records must carry no successful verdict or extra SDK read/solve.
