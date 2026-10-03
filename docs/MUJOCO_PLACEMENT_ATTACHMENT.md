# Measured attachment frames for MuJoCo placement

The physical two-pick episode on `7a52d956cae5ad14e2c7b717fabf1f692e3755a8`
remains FAIL+CLOSED. Red was independently confirmed after home; blue was held
and rejected before carry/release. No subsequent two-object physical success
is claimed by this correction.

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
Actual carry, release, rest and prefix preservation still require a separately
authorized physical episode.

Validation: 313 directed tests passed in 38.35 s, including 44 new cases and the
existing no-attachment, placement, withdrawal, refusal and held-observation
regressions. The first combined check's single failed clock assertion and its
correction remain in the receipt. Test elapsed time is diagnostic under shared
host use, not a causal performance benchmark. See
`benchmark/results/mujoco_attachment_aim_20261003.json` for pins and raw hashes.
