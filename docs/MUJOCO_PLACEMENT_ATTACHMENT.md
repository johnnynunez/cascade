# Measured attachment frames for MuJoCo placement

The physical two-pick episode on `7a52d956cae5ad14e2c7b717fabf1f692e3755a8`
remains FAIL+CLOSED. Red was independently confirmed after home; blue was held
and rejected before carry/release. A later, separate episode on `d273dc4`
passed both objects after the corrections and consumer fences below. See
[the final validation](MUJOCO_MANIPULATION_VALIDATION_20261003.md) for its source,
physical evidence and limits. The earlier failure remains unchanged.

The source-bound CPU replay found a frame mismatch: region selection and
ordinary `place_at` subtracted a world-frame object−TCP translation, then chose
a different placement orientation. The existing scratch carry correctly rotated
the held body, so the first orientation predicted an object error of
(+18.431, +16.905) mm. Changing yaw alone or the packing cross-row position alone
did not pass the unchanged geometry gates.

For an explicitly configured native MuJoCo delivery region, placement now reads
both body pose and arm joints from one bound final-state snapshot. Detached FK
expresses the measured body origin in the declared kinematic tool frame:

```
T_tool_body = inverse(T_base_tool_now) @ T_base_body_now
TCP_target.xy = requested_object.xy - (R_tool_destination @ T_tool_body.translation).xy
```

`place_geometry.plan` applies this translation for each existing orientation
candidate. Release z retains its existing TCP-height meaning and ceiling. The
original orientation list, IK limits, carry/escape/home candidate families,
region, margins, separation, capacity checks and verifier thresholds are
unchanged. This is an observed rigid-transport prediction in scratch; no weld,
physical joint, pose assistance, force or solver change is added. It does not
certify contact, a secure grasp or freedom from future slip.

`mujoco_placement_aim.capture` is optional and refuses a configured region
without its materialized native driver and bound placement observer. It never
activates a lazy arm. It binds the runtime/arm/backend, engine/model/data/lock/
control channel, kinematics, full driver coordinate mapping, labels, model
digest, epoch, halt generation and cancellation token. Its state digest names
exactly time/qpos/qvel/ctrl. The transform is derived from detached final-qpos
FK, not cached last-solve contact geometry or a renderer. No observer solve or
live forward is used.

A private `RegionPlan` carries the same proposed pose from region preview into
the ordinary `place_at` path. Consumption rechecks the original snapshot and
three-second planning deadline; it does not recapture or silently recalibrate
an old plan. The repeated `consume` calls are in-call validation, not a claim
of a single-use permission. The safety harness and existing release/withdrawal
protocol retain actuation authority. Changed state, model, identity, target,
prepared geometry or a stop/reset interval invalidates the handoff before a
command. The same calculation is used for an explicit point request on a
profile with this observation capability; the requested object point does not
move. Existing held-observation/slip checks still run. The reported object
point and effective TCP aim remain separate.

The original cancellation token also reaches every actual carry segment, not
just its geometric preview. `SafeArm` checks it after reading the start state,
before/after the existing waypoint approval and stream-start callback, and
after the stream returns. The first stream rechecks the measured snapshot and
original planning deadline; subsequent streams preserve identity, epoch and
cancellation while allowing legitimate joint/time progress. The short stop
lock only compares the token/latch; it runs no callback, SDK operation or I/O.
An already admitted backend call is not claimed to be interruptible.

Before opening, the runtime requires a real `Withdrawal` with the same arm,
world, model, observer epoch, halt generation and cancellation token. Its debt
is retained before opening, and the withdrawal checks that original token at
its command boundary. A stop followed by reset cannot renew the old carry or
opening permission. Calls without this optional token retain the legacy
streamer path; no planner or backend can silently drop its required callback.

Profiles without `mj_delivery_area` retain the legacy world-offset aiming path;
this patch supplies no new observation authority for Isaac, hardware, Warp or
an unconfigured MuJoCo scene. Existing static no-attachment pose vectors remain
covered by their original regression fixture.

In the retained scratch comparison, the same first orientation and requested
body point pass all existing geometric checks after rotation-aware compensation:
TCP `(0.230669309, -0.137642114, 0.045)`, escape `joint_1_first`, 20.886 mm
separation from red, and the full predicted footprint within the original
20 mm planning margins. The copied state remains unchanged. The fixture and
negative tests cover frame/offset variants, stale state/model/epoch/mapping,
stop followed by reset, original deadline, altered target/geometry, consumption
through `skill_place_at`, and an unsafe target overlapping the placed prefix.
At that scratch checkpoint, actual carry, release, rest and prefix preservation
remained untested; the separately authorized final episode is linked above.

The earlier 313-test checkpoint is preserved at `326720e7`; subsequent review
found that stop/reset inside the first `SafeArm` read could still reach backend
dispatch. The external control retains its original red result. With the
consumer fence, the same cancelled case produces no backend dispatch; the
healthy control still reaches the intercepted boundary. The control only adds
`SafetyViolation` to its caught public failure types.

Final validation: 539 directed tests passed in 61.16 s, including 59 new
attachment/fence cases. An expanded intermediate selection retained five
failures from incomplete legacy harness doubles; all five also failed with the
exact old harness. The doubles now bind the real withdrawal guard. The earlier
clock-assertion failure is retained separately. Sources and protected stores
were unchanged during validation; Ruff found no new F/E9 findings (two old
unused imports remain in the carry test). Test elapsed time is diagnostic
under shared host use, not a causal performance benchmark. See
`benchmark/results/mujoco_attachment_aim_20261003.json` for pins and raw hashes.
